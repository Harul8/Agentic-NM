"""
Nyaymalaw MCP Server — production-ready Model Context Protocol server.

Exposes 8 tools covering the full Nyaymalaw capability surface:
  Legal research  : search_bare_acts, search_case_laws, lookup_section, lookup_case
  Legal opinion   : start_intake, continue_intake, get_intake_state, draft_opinion

────────────────────────────────────────────────────────────────────────────────
TRANSPORTS
────────────────────────────────────────────────────────────────────────────────

  stdio  (VS Code / Cursor / Claude Desktop — subprocess, zero network):
      python mcp_server.py

  SSE    (Cowork / browser / any HTTP client — persistent HTTP stream):
      python mcp_server.py --transport sse --port 8010
      Then add  http://localhost:8010/sse  as an MCP server in the client.

────────────────────────────────────────────────────────────────────────────────
SETUP
────────────────────────────────────────────────────────────────────────────────

  pip install "mcp[cli]"          # one-time, in your project venv
  export OPENAI_API_KEY=sk-...    # required

  VS Code   → .vscode/mcp.json already configured (see that file)
  Cowork    → add the entry from claude_desktop_config.json.example to
              %APPDATA%\\Claude\\claude_desktop_config.json  (Windows)
              ~/Library/Application Support/Claude/claude_desktop_config.json (Mac)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Optional

# ── Project root on path (needed when run as __main__) ───────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nyaymalaw.mcp")

# ── In-memory session store for intake tools ─────────────────────────────────
# Keyed by session_id (UUID string). Cleared on server restart (intentional —
# intake sessions are short-lived; persistence can be added via SQLite later).
_SESSIONS: dict[str, dict] = {}
_SESSION_TTL_SEC = 3600  # auto-expire after 1 hour of inactivity


def _gc_sessions() -> None:
    """Evict sessions idle longer than TTL."""
    now = time.time()
    stale = [sid for sid, s in _SESSIONS.items()
             if now - s.get("updated_at", 0) > _SESSION_TTL_SEC]
    for sid in stale:
        del _SESSIONS[sid]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# Each is a thin adapter: validate → call domain function → return JSON string.
# Never let an exception propagate out — always return {"error": ...}.
# ─────────────────────────────────────────────────────────────────────────────

def _tool_search_bare_acts(query: str, top_k: int = 5) -> str:
    """
    search_bare_acts implementation.
    Returns sections formatted for model consumption:
      [{act_name, section_number, section_title, text, score}]
    """
    try:
        from core.retriever import search_bare_acts_auto
        top_k = max(1, min(int(top_k), 20))
        raw = search_bare_acts_auto(query, top_k=top_k)
        results = []
        for r in raw:
            results.append({
                "act_name":       r.get("act_name", ""),
                "section_number": r.get("section_number", ""),
                "section_title":  r.get("section_title", ""),
                "text":           (r.get("text") or r.get("full_text") or "")[:2000],
                "score":          round(float(r.get("rerank_score", 0) or 0), 4),
            })
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("search_bare_acts failed")
        return json.dumps({"error": str(exc), "query": query})


def _tool_search_case_laws(query: str, jurisdiction: str = "", top_k: int = 5) -> str:
    """
    search_case_laws implementation.
    Returns paragraphs formatted for model consumption:
      [{case_name, citation, court, year, paragraph_num, paragraph_type, text, score}]
    """
    try:
        from core.retriever import search_case_laws_auto
        top_k = max(1, min(int(top_k), 20))
        search_q = f"{query} {jurisdiction}".strip() if jurisdiction else query
        raw = search_case_laws_auto(search_q, top_k=top_k)
        results = []
        for r in raw:
            results.append({
                "case_name":      r.get("case_name", ""),
                "citation":       r.get("citation", ""),
                "court":          r.get("court", ""),
                "year":           r.get("year", ""),
                "paragraph_num":  r.get("paragraph_num", ""),
                "paragraph_type": r.get("paragraph_type", ""),
                "text":           (r.get("text") or r.get("full_text") or "")[:2000],
                "score":          round(float(r.get("rerank_score", 0) or 0), 4),
            })
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("search_case_laws failed")
        return json.dumps({"error": str(exc), "query": query})


def _tool_lookup_section(act_name: str, section_number: str) -> str:
    """
    lookup_section implementation.
    Searches for an exact act + section combination.
    Returns the full verbatim text of every matching sub-chunk.
    """
    try:
        from core.retriever import search_bare_acts_auto
        query = f"{act_name} section {section_number}"
        raw = search_bare_acts_auto(query, top_k=10)
        matches = [
            r for r in raw
            if str(r.get("section_number", "")).strip() == str(section_number).strip()
        ]
        if not matches:
            matches = raw[:3]  # fallback: top results

        chunks = []
        for r in matches:
            chunks.append({
                "act_name":       r.get("act_name", ""),
                "section_number": r.get("section_number", ""),
                "section_title":  r.get("section_title", ""),
                "sub_section":    r.get("sub_section", ""),
                "text":           (r.get("text") or r.get("full_text") or ""),
            })
        return json.dumps({
            "act_name":       act_name,
            "section_number": section_number,
            "chunks":         chunks,
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("lookup_section failed")
        return json.dumps({"error": str(exc), "act_name": act_name, "section_number": section_number})


def _tool_lookup_case(case_name: str, para_num: Optional[str] = None) -> str:
    """
    lookup_case implementation.
    Returns matching paragraphs for a known case name.
    If para_num is given, filters to that paragraph only.
    """
    try:
        from core.retriever import search_case_laws_auto
        raw = search_case_laws_auto(case_name, top_k=20)

        # Filter to case name match (partial, case-insensitive)
        name_lower = case_name.lower()
        matches = [
            r for r in raw
            if name_lower in (r.get("case_name") or "").lower()
        ]
        if not matches:
            matches = raw[:5]

        # Further filter by paragraph number if requested
        if para_num:
            para_matches = [r for r in matches
                            if str(r.get("paragraph_num", "")).strip() == str(para_num).strip()]
            if para_matches:
                matches = para_matches

        paragraphs = []
        for r in matches[:10]:
            paragraphs.append({
                "case_name":      r.get("case_name", ""),
                "citation":       r.get("citation", ""),
                "court":          r.get("court", ""),
                "year":           r.get("year", ""),
                "paragraph_num":  r.get("paragraph_num", ""),
                "paragraph_type": r.get("paragraph_type", ""),
                "text":           (r.get("text") or r.get("full_text") or ""),
            })
        return json.dumps({
            "case_name":  case_name,
            "paragraphs": paragraphs,
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("lookup_case failed")
        return json.dumps({"error": str(exc), "case_name": case_name})


def _tool_start_intake(first_message: str = "") -> str:
    """
    start_intake implementation.
    Creates a new legal opinion intake session.
    If first_message is provided, processes it as the client's first turn.
    Returns session_id + AI reply + current intake_state.
    """
    try:
        _gc_sessions()
        from intake.session import new_session, append_turn
        from intake.stage1_opening import generate_opening, process_turn

        session = new_session()
        session_id = session["session_id"]

        if first_message and first_message.strip():
            # Client sent a message with start — process it immediately
            append_turn(session, "user", first_message.strip())
            result = process_turn(session, first_message.strip())
            reply = result["reply"]
            append_turn(session, "assistant", reply)
            session["intake_state"] = result["intake_state"]
        else:
            # No message yet — send the warm opening
            reply = generate_opening()
            append_turn(session, "assistant", reply)

        _SESSIONS[session_id] = session
        return json.dumps({
            "session_id":    session_id,
            "reply":         reply,
            "stage":         session.get("stage", "stage1"),
            "intake_state":  session.get("intake_state", {}),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("start_intake failed")
        return json.dumps({"error": str(exc)})


def _tool_continue_intake(session_id: str, client_message: str) -> str:
    """
    continue_intake implementation.
    Passes the client's next message to the active intake session.
    Returns the AI's next reply, updated intake_state, and whether to advance stage.
    """
    try:
        if session_id not in _SESSIONS:
            return json.dumps({
                "error": f"Session '{session_id}' not found. Call start_intake first.",
                "session_id": session_id,
            })

        session = _SESSIONS[session_id]
        msg = (client_message or "").strip()
        if not msg:
            return json.dumps({"error": "client_message cannot be empty", "session_id": session_id})

        from intake.session import append_turn
        from intake.stage1_opening import process_turn

        append_turn(session, "user", msg)
        result = process_turn(session, msg)

        reply = result["reply"]
        append_turn(session, "assistant", reply)
        session["intake_state"] = result["intake_state"]

        if result.get("advance_to_stage2"):
            session["stage"] = "stage2"

        session["updated_at"] = time.time()
        _SESSIONS[session_id] = session

        return json.dumps({
            "session_id":        session_id,
            "reply":             reply,
            "stage":             session.get("stage", "stage1"),
            "advance_to_stage2": result.get("advance_to_stage2", False),
            "urgency_signal":    result.get("urgency_signal", "unknown"),
            "intake_state":      session.get("intake_state", {}),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("continue_intake failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_get_intake_state(session_id: str) -> str:
    """
    get_intake_state implementation.
    Returns the current structured intake state for a session.
    Useful for the advocate review panel to inspect collected facts.
    """
    try:
        if session_id not in _SESSIONS:
            return json.dumps({
                "error": f"Session '{session_id}' not found.",
                "session_id": session_id,
            })
        session = _SESSIONS[session_id]
        return json.dumps({
            "session_id":   session_id,
            "stage":        session.get("stage", "stage1"),
            "intake_state": session.get("intake_state", {}),
            "turn_count":   len([t for t in session.get("history", []) if t.get("role") == "user"]),
            "category":     (session.get("intake_state") or {}).get("category"),
            "urgency":      (session.get("intake_state") or {}).get("urgency_signal"),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("get_intake_state failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_draft_opinion(session_id: str) -> str:
    """
    draft_opinion implementation.
    Triggers Stage 5 draft generation from a completed intake session.
    Returns formatted_draft (markdown) + advocate_review (structured JSON brief).
    The advocate_review is the brief shown in the senior advocate review panel.
    """
    try:
        if session_id not in _SESSIONS:
            return json.dumps({
                "error": f"Session '{session_id}' not found. Run intake first.",
                "session_id": session_id,
            })

        session = _SESSIONS[session_id]
        intake_state = session.get("intake_state") or {}

        if not intake_state.get("known_facts"):
            return json.dumps({
                "error": "Intake not complete — no facts collected yet. Continue the intake conversation.",
                "session_id": session_id,
                "stage": session.get("stage", "stage1"),
            })

        from intake.stage5_draft import build_draft
        result = build_draft(intake_state, session.get("history", []))

        return json.dumps({
            "session_id":      session_id,
            "formatted_draft": result.get("formatted_draft", ""),
            "advocate_review": result.get("advocate_review", {}),
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("draft_opinion failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


# ─────────────────────────────────────────────────────────────────────────────
# MCP server wiring
# ─────────────────────────────────────────────────────────────────────────────

def _build_server(port: int = 8010) -> "FastMCP":
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        name="Nyaymalaw",
        instructions=(
            "Nyaymalaw — Indian legal AI. "
            "Use search_bare_acts / search_case_laws for research. "
            "Use lookup_section / lookup_case for exact verbatim text from the database. "
            "Use start_intake → continue_intake (repeat) → draft_opinion for the full "
            "autonomous legal opinion workflow. "
            "Use get_intake_state at any point to inspect the collected facts."
        ),
        port=port,
    )

    # ── 1. search_bare_acts ───────────────────────────────────────────────────
    @app.tool(
        description=(
            "Search the Nyaymalaw vector store for Bare Act sections relevant to a legal issue. "
            "Returns verbatim statutory text with act name, section number, and relevance score. "
            "Use this when you need the LAW — statutes, sections, sub-sections. "
            "Do NOT use this for case law judgments — use search_case_laws for that. "
            "Do NOT use this when you know the exact act + section — use lookup_section instead."
        )
    )
    def search_bare_acts(query: str, top_k: int = 5) -> str:
        """Search Bare Act sections for a legal issue. Returns verbatim statutory text."""
        return _tool_search_bare_acts(query, top_k)

    # ── 2. search_case_laws ───────────────────────────────────────────────────
    @app.tool(
        description=(
            "Search the Nyaymalaw vector store for case law paragraphs relevant to a legal issue. "
            "Returns verbatim judgment text with case name, citation, court, year, and paragraph number. "
            "Use this when you need PRECEDENT — court judgments, holdings, reasoning. "
            "Do NOT use this for statutory text — use search_bare_acts for that. "
            "Do NOT use this when you know the exact case name — use lookup_case instead."
        )
    )
    def search_case_laws(query: str, jurisdiction: str = "", top_k: int = 5) -> str:
        """Search case law paragraphs for a legal issue. Returns verbatim judgment text."""
        return _tool_search_case_laws(query, jurisdiction, top_k)

    # ── 3. lookup_section ─────────────────────────────────────────────────────
    @app.tool(
        description=(
            "Fetch the exact verbatim text of a specific section from a specific Act in the database. "
            "Use this when you already know WHICH act and WHICH section number you need. "
            "Returns every sub-section chunk stored for that section. "
            "Do NOT use this for open-ended research — use search_bare_acts for discovery."
        )
    )
    def lookup_section(act_name: str, section_number: str) -> str:
        """Fetch verbatim text of a known act section (e.g. PWDVA s.18, BNS s.85)."""
        return _tool_lookup_section(act_name, section_number)

    # ── 4. lookup_case ────────────────────────────────────────────────────────
    @app.tool(
        description=(
            "Fetch verbatim paragraphs from a specific case judgment stored in the database. "
            "Use this when you already know the CASE NAME and optionally a paragraph number. "
            "Returns the exact judgment text at paragraph level — cite para_num in your output. "
            "Do NOT use this for open-ended research — use search_case_laws for discovery."
        )
    )
    def lookup_case(case_name: str, para_num: str = "") -> str:
        """Fetch verbatim paragraphs from a known case judgment by case name."""
        return _tool_lookup_case(case_name, para_num or None)

    # ── 5. start_intake ───────────────────────────────────────────────────────
    @app.tool(
        description=(
            "Start a new autonomous legal opinion intake session. "
            "Creates a session and returns an opening message for the client. "
            "Optionally pass the client's first message to process it immediately. "
            "Returns session_id — pass this to continue_intake for every subsequent turn. "
            "Use this ONLY to begin a fresh legal opinion workflow, not for research queries."
        )
    )
    def start_intake(first_message: str = "") -> str:
        """Begin a new legal opinion intake. Returns session_id + opening reply."""
        return _tool_start_intake(first_message)

    # ── 6. continue_intake ────────────────────────────────────────────────────
    @app.tool(
        description=(
            "Pass the client's next message in an active intake session and get the AI's reply. "
            "Call this repeatedly — once per client turn — until advance_to_stage2 is true. "
            "The AI will ask one targeted question per turn, gather facts, detect legal category, "
            "and signal when enough information has been collected. "
            "Do NOT call this if session_id is unknown — call start_intake first."
        )
    )
    def continue_intake(session_id: str, client_message: str) -> str:
        """Send client's message to an active intake session. Returns next AI reply + state."""
        return _tool_continue_intake(session_id, client_message)

    # ── 7. get_intake_state ───────────────────────────────────────────────────
    @app.tool(
        description=(
            "Inspect the current structured state of an intake session without advancing it. "
            "Returns collected facts, detected legal category, urgency signal, and turn count. "
            "Use this to check readiness before calling draft_opinion, "
            "or to show the advocate review panel the structured brief."
        )
    )
    def get_intake_state(session_id: str) -> str:
        """Return structured intake state for a session without advancing the conversation."""
        return _tool_get_intake_state(session_id)

    # ── 8. draft_opinion ─────────────────────────────────────────────────────
    @app.tool(
        description=(
            "Generate the full structured legal opinion draft from a completed intake session. "
            "Returns two outputs: formatted_draft (markdown document with Statement of Facts, "
            "Legal Framework with verbatim statutory text, Case Law Support with exact paragraph "
            "numbers, Prayer/Relief, and Documents Checklist) and advocate_review (structured JSON "
            "brief for the senior advocate review panel). "
            "Call this only after the intake is sufficiently complete (advance_to_stage2 was true "
            "or get_intake_state shows meaningful known_facts). "
            "Do NOT call this on a fresh or near-empty session."
        )
    )
    def draft_opinion(session_id: str) -> str:
        """Generate structured legal opinion draft from a completed intake session."""
        return _tool_draft_opinion(session_id)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Nyaymalaw MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="stdio = VS Code / Cursor / Claude Desktop  |  sse = Cowork / HTTP clients",
    )
    parser.add_argument(
        "--port", type=int, default=8010,
        help="Port for SSE transport (default 8010)",
    )
    args = parser.parse_args()

    logger.info(
        "Starting Nyaymalaw MCP Server | transport=%s%s",
        args.transport,
        f" port={args.port}" if args.transport == "sse" else "",
    )

    app = _build_server(port=args.port)

    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="sse")


if __name__ == "__main__":
    main()
