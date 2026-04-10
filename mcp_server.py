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

    Entity-aware routing:
      1. extract_legal_entities() scans the query for explicit act/section refs.
      2. For each entity found, lookup_section() fetches the verbatim text directly
         from the structured DB — faster and more precise than FAISS.
      3. If no entities are found (or the entity lookup returns nothing), fall through
         to the FAISS + BM25 hybrid search as before.

    Returns sections formatted for model consumption:
      [{act_name, section_number, section_title, text, score, source}]
    """
    try:
        from retrieval.retriever import search_bare_acts_auto
        top_k = max(1, min(int(top_k), 20))

        # ── Stage 1: entity-aware direct lookup ──────────────────────────────
        entity_results: list[dict] = []
        try:
            from retrieval.legal_graph import extract_legal_entities, lookup_section as _lookup_sec
            entities = extract_legal_entities(query)
            for ent in entities:
                sec_rows = _lookup_sec(
                    ent.get("section_number", ""),
                    ent.get("act_hint", ""),
                )
                for row in (sec_rows or []):
                    entity_results.append({
                        "act_name":       row.get("act_name", ent.get("act_hint", "")),
                        "section_number": row.get("section_number", ent.get("section_number", "")),
                        "section_title":  row.get("section_title", ""),
                        "text":           (row.get("text") or row.get("full_text") or "")[:2000],
                        "score":          1.0,       # direct-lookup — treat as top score
                        "source":         "entity_lookup",
                    })
        except Exception:
            logger.debug("entity routing unavailable — falling back to FAISS", exc_info=True)

        if entity_results:
            return json.dumps(
                {"query": query, "results": entity_results, "routing": "entity_lookup"},
                ensure_ascii=False, indent=2,
            )

        # ── Stage 2: FAISS + BM25 hybrid fallback ────────────────────────────
        raw = search_bare_acts_auto(query, top_k=top_k)
        results = []
        for r in raw:
            results.append({
                "act_name":       r.get("act_name", ""),
                "section_number": r.get("section_number", ""),
                "section_title":  r.get("section_title", ""),
                "text":           (r.get("text") or r.get("full_text") or "")[:2000],
                "score":          round(float(r.get("rerank_score", 0) or 0), 4),
                "source":         "faiss_hybrid",
            })
        return json.dumps({"query": query, "results": results, "routing": "faiss_hybrid"}, ensure_ascii=False, indent=2)
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
        from retrieval.retriever import search_case_laws_auto
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
        from retrieval.retriever import search_bare_acts_auto
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
        from retrieval.retriever import search_case_laws_auto
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


def _tool_start_intake(first_message: str = "", user_id: str = "") -> str:
    """
    start_intake implementation — backed by LangGraph intake subgraph.

    Creates a new SQLite-checkpointed intake session.  If first_message is
    provided it is processed immediately (category detection + fact extraction
    run in parallel).  Returns session_id + AI reply + current intake_state.
    user_id (optional) — links this session to the cross-session memory store.
    """
    try:
        from agents.intake.graph import start_intake_session
        result = start_intake_session(first_message or "", user_id=user_id or "")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("start_intake failed")
        return json.dumps({"error": str(exc)})


def _tool_continue_intake(session_id: str, client_message: str) -> str:
    """
    continue_intake implementation — backed by LangGraph intake subgraph.

    Resumes the checkpointed intake session with the client's next message.
    Category detection + fact extraction run in parallel inside the graph.
    Returns the AI's next reply, updated intake_state, and stage-advance flag.
    """
    try:
        msg = (client_message or "").strip()
        if not msg:
            return json.dumps({"error": "client_message cannot be empty", "session_id": session_id})

        from agents.intake.graph import continue_intake_session
        result = continue_intake_session(session_id, msg)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("continue_intake failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_get_intake_state(session_id: str) -> str:
    """
    get_intake_state implementation — reads directly from the LangGraph checkpointer.
    No LLM call.  Returns structured intake state for the advocate review panel.
    """
    try:
        from agents.intake.graph import get_intake_session_state
        result = get_intake_session_state(session_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("get_intake_state failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_draft_opinion(session_id: str) -> str:
    """
    draft_opinion implementation — reads from the LangGraph checkpointer.
    Returns formatted_draft (client-facing markdown) + advocate_review
    (structured JSON brief with chamber-note, research packets, and action plan).
    """
    try:
        from agents.intake.graph import get_draft_from_session
        result = get_draft_from_session(session_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("draft_opinion failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_advocate_review(session_id: str, approved: bool, notes: str = "") -> str:
    """
    advocate_review implementation — resumes the intake graph's advocate_review node.

    Submits the advocate's decision on a generated draft:
      approved=True  → draft accepted; session summary saved to user memory.
      approved=False → draft rejected; revision notes injected into next draft.
    """
    try:
        from agents.intake.graph import advocate_review_session
        result = advocate_review_session(session_id, approved=bool(approved), notes=notes or "")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("advocate_review failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_get_session_history(session_id: str) -> str:
    """
    get_session_history implementation — time-travel via LangGraph checkpoints.

    Returns a list of checkpoint summaries for the session (most recent first).
    Each entry includes: checkpoint_index, turn_count, category, urgency,
    facts_count, advocate_approved, and a reply_preview.
    """
    try:
        from agents.intake.graph import get_session_history
        history = get_session_history(session_id)
        return json.dumps(
            {"session_id": session_id, "checkpoints": history},
            ensure_ascii=False, indent=2,
        )
    except Exception as exc:
        logger.exception("get_session_history failed")
        return json.dumps({"error": str(exc), "session_id": session_id})


def _tool_expand_precedents(case_names: list[str], direction: str = "both") -> str:
    """
    expand_precedents implementation.

    Walks the citation graph to find cases that are closely related to the
    provided seed case names:
      - "cited_by"  : cases that the seeds cite (authorities they relied on)
      - "citing"    : cases that cite the seeds (how they've been followed/distinguished)
      - "both"      : union of the above two sets  (default)

    Also calls expand_case_names_by_precedent() to surface the highest-authority
    precedents connected to the seeds via PageRank.

    Returns:
      {
        seeds:            [...],
        cited_by:         [{case_id, case_name}],   # authorities relied on by seeds
        citing:           [{case_id, case_name}],   # cases that cite the seeds
        high_authority:   [{case_name}],             # top PageRank neighbours
        sections_touched: [{section_number, act}],  # sections interpreted by precedents
      }
    """
    try:
        from retrieval.citations import (
            get_cases_cited_by,
            get_cases_citing,
            expand_case_names_by_precedent,
        )

        direction = direction.lower()
        cited_by_raw: list[tuple] = []
        citing_raw:   list[tuple] = []

        if direction in ("cited_by", "both"):
            cited_by_raw = get_cases_cited_by(case_names) or []
        if direction in ("citing", "both"):
            citing_raw = get_cases_citing(case_names) or []

        extra_names, sections_touched = expand_case_names_by_precedent(case_names)

        def _pair_to_dict(pair: tuple) -> dict:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                return {"case_id": pair[0], "case_name": pair[1]}
            return {"case_name": str(pair)}

        return json.dumps(
            {
                "seeds":            case_names,
                "cited_by":         [_pair_to_dict(p) for p in cited_by_raw],
                "citing":           [_pair_to_dict(p) for p in citing_raw],
                "high_authority":   [{"case_name": n} for n in (extra_names or [])],
                "sections_touched": sections_touched or [],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        logger.exception("expand_precedents failed")
        return json.dumps({"error": str(exc), "case_names": case_names})


def _tool_get_cases_for_section(section_number: str, act_hint: str = "") -> str:
    """
    get_cases_for_section implementation.

    Queries the citation graph for all case law that has been recorded as
    interpreting or applying the given statutory section.

    Returns:
      {
        section_number: "...",
        act_hint:       "...",
        cases:          [{case_id, case_name, court, year}],
      }
    """
    try:
        from retrieval.citations import get_cases_interpreting_section

        cases_raw = get_cases_interpreting_section(section_number, act_hint) or []

        cases = []
        for row in cases_raw:
            if isinstance(row, dict):
                cases.append({
                    "case_id":   row.get("case_id", ""),
                    "case_name": row.get("case_name", ""),
                    "court":     row.get("court", ""),
                    "year":      row.get("year", ""),
                })
            else:
                cases.append({"case_name": str(row)})

        return json.dumps(
            {
                "section_number": section_number,
                "act_hint":       act_hint,
                "cases":          cases,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        logger.exception("get_cases_for_section failed")
        return json.dumps({"error": str(exc), "section_number": section_number, "act_hint": act_hint})


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
            "Pass user_id to link this session to the cross-session memory store "
            "so prior context is surfaced in future sessions. "
            "Use this ONLY to begin a fresh legal opinion workflow, not for research queries."
        )
    )
    def start_intake(first_message: str = "", user_id: str = "") -> str:
        """Begin a new legal opinion intake. Returns session_id + opening reply."""
        return _tool_start_intake(first_message, user_id=user_id)

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
            "Returns two outputs: formatted_draft (client-facing markdown opinion) and "
            "advocate_review (structured JSON brief with the chamber-note view, issue-wise "
            "research packets, and action-drafting readiness plan for the senior advocate panel). "
            "Call this only after the intake is sufficiently complete (advance_to_stage2 was true "
            "or get_intake_state shows meaningful known_facts). "
            "Do NOT call this on a fresh or near-empty session."
        )
    )
    def draft_opinion(session_id: str) -> str:
        """Generate structured legal opinion draft from a completed intake session."""
        return _tool_draft_opinion(session_id)

    # ── 9. expand_precedents ─────────────────────────────────────────────────
    @app.tool(
        description=(
            "Walk the citation graph to find cases related to the given seed case names. "
            "Returns cases the seeds cite (authorities relied on), cases that cite the seeds "
            "(how they've been followed or distinguished), high-authority PageRank neighbours, "
            "and the statutory sections those precedents interpret. "
            "Use after lookup_case or search_case_laws to deepen precedent research. "
            "direction: 'cited_by' | 'citing' | 'both' (default)."
        )
    )
    def expand_precedents(case_names: list[str], direction: str = "both") -> str:
        """Expand a set of case names into a web of related precedents via the citation graph."""
        return _tool_expand_precedents(case_names, direction)

    # ── 10. get_cases_for_section ─────────────────────────────────────────────
    @app.tool(
        description=(
            "Query the citation graph for all cases recorded as interpreting or applying "
            "a specific statutory section. Returns case_id, case_name, court, and year. "
            "Use when you know the section and want the case law that has applied it. "
            "Complement with lookup_section to get the verbatim text of the section itself."
        )
    )
    def get_cases_for_section(section_number: str, act_hint: str = "") -> str:
        """Return all cases in the citation graph that interpret a given statutory section."""
        return _tool_get_cases_for_section(section_number, act_hint)

    # ── 11. advocate_review ───────────────────────────────────────────────────
    @app.tool(
        description=(
            "Submit the advocate's review decision for a generated draft. "
            "This is the Advocate-in-the-Loop gate: the draft waits for approval before "
            "being delivered to the client. "
            "approved=True accepts the draft and saves the session to user memory. "
            "approved=False with revision notes triggers a re-draft incorporating the feedback. "
            "Call this after draft_opinion returns a draft and before surfacing it to the client."
        )
    )
    def advocate_review(session_id: str, approved: bool, notes: str = "") -> str:
        """Submit advocate approval or revision request for a generated draft."""
        return _tool_advocate_review(session_id, approved, notes)

    # ── 12. get_session_history ───────────────────────────────────────────────
    @app.tool(
        description=(
            "Time-travel: return a checkpoint-by-checkpoint history of an intake session. "
            "Each checkpoint captures the state at a specific point in the conversation: "
            "turn count, legal category, urgency level, facts collected, and advocacy status. "
            "Use for the advocate review panel to audit the intake progression, "
            "or to identify the exact point where a key fact was collected. "
            "Does not advance the session or call any LLM."
        )
    )
    def get_session_history(session_id: str) -> str:
        """Return checkpoint summaries for all saved states of an intake session."""
        return _tool_get_session_history(session_id)

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
