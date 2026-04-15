"""
agents/intake/graph.py — LangGraph intake subgraph for Nyaymalaw.

Graph structure
---------------

    START
      │
      ▼
  [process_turn]  ◄─────────────────────┐
      │                                  │
      │  advance_to_stage2=False         │
      ▼                                  │
  [wait_for_user]  — interrupt() ───────┘
      │  (resumes with next client message)
      │  advance_to_stage2=True
      ▼
   [draft]
      │
      ▼
  [advocate_review]  — interrupt() ─────┐
      │                                  │  (resume with approved=False + notes)
      │  approved=True                   │
      ▼                                  │
     END ◄────── [draft] ◄──────────────┘

Key improvements over the original _SESSIONS approach
------------------------------------------------------

1. Checkpointing (SQLite)
   Every node return is checkpointed.  Sessions survive server restarts.
   Any worker can resume any session — no shared memory required.

2. Human-in-the-Loop: client turns (wait_for_user)
   interrupt() in wait_for_user pauses execution; continue_intake() resumes
   via Command(resume=client_message).

3. Human-in-the-Loop: advocate review (advocate_review)
   interrupt() in advocate_review presents the draft to the advocate.
   advocate_review_session(session_id, approved, notes) resumes.
   If rejected, notes are injected into the next draft pass.
   If approved, the session summary is saved to cross-session memory.

4. Parallel LLM calls in process_turn
   Round A (parallel): category detection  +  fact extraction
   Round B (parallel): urgency recheck     +  remedy assessment
   Round C (sequential): reply generation  (depends on A+B)

5. Time-travel via get_session_history()
   Wraps intake_graph.get_state_history() into a structured list of
   per-checkpoint summaries for the advocate review panel.
"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from agents.intake.casefile import make_case_file_template
from agents.intake.state import IntakeState

logger = logging.getLogger("nyaymalaw.intake.graph")


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------

def _get_checkpointer():
    """
    Return a SQLite checkpointer backed by the app's existing DB_PATH.

    Uses SqliteSaver(conn) directly (not from_conn_string which returns a
    context manager in newer langgraph-checkpoint-sqlite versions).
    Falls back to MemorySaver if the sqlite package is not installed.
    """
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        logger.warning("SqliteSaver not available — falling back to MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_human_message(state: IntakeState) -> str:
    """Extract the content of the most recent HumanMessage in state."""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _fresh_intake_state() -> dict:
    """Return a blank Stage-1 intake state dict."""
    return {
        "turn_count":               0,
        "primary_issue_cluster":    None,
        "secondary_issue_clusters": [],
        "category_confidence":      None,
        "issue_summary":            "",
        "jurisdiction":             "unknown",
        "urgency_signal":           "unknown",
        "risk_flags":               [],
        "client_role":              "unknown",
        "other_party":              "unknown",
        "relationship_to_other_party": "unknown",
        "timeframe_status":         "unknown",
        "client_goal_initial":      None,
        "immediate_need":           "unknown",
        "emotional_ask":            "unknown",
        "assessed_remedy":          None,
        "known_facts":              [],
        "open_questions":           [],
        "detail_request_issued":    False,
        "detail_groups_requested":  [],
        "missing_detail_groups":    [],
        "followup_questions":       [],
        "analysis_ready":           False,
        "latest_intake_summary":    None,
        "case_file":                make_case_file_template(),
        "ready_for_stage2":         False,
    }


def _session_dict_from_state(state: IntakeState) -> dict:
    """
    Build the legacy session dict expected by stage1_opening.process_turn().
    Maps IntakeState messages → history list.
    """
    history = []
    for msg in (state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content or ""})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content or ""})
    return {
        "session_id":   state.get("session_id", ""),
        "stage":        "stage1",
        "history":      history,
        "intake_state": dict(state.get("intake_state") or {}),
    }


# ---------------------------------------------------------------------------
# Parallel Stage-1 processing
# ---------------------------------------------------------------------------

def _run_stage1_parallel(session: dict, msg: str, intake_state: dict) -> dict:
    """
    Run Stage-1 per-turn LLM calls with internal parallelism.

    Round A (parallel):
        - _detect_category  (sets primary cluster, urgency seed, anchor fields)
        - _update_known_facts  (appends structured fact objects)

    Round B (parallel, after A):
        - _recheck_urgency   (escalate / de-escalate urgency_signal)
        - _assess_remedy     (if client_goal_initial is now known)

    Round C (sequential):
        - _generate_reply    (depends on all A+B results)

    Returns
    -------
    dict with: reply, advance_to_stage2, urgency_signal
    """
    return _run_stage1_parallel_v2(session, msg, intake_state)

    from agents.intake.stage1_opening import (
        _detect_category,
        _update_known_facts,
        _recheck_urgency,
        _assess_remedy,
        _check_readiness,
        _build_context,
        _generate_followup,
        _generate_safety_first_response,
        _get_vetting_question,
    )

    intake_state["turn_count"] = intake_state.get("turn_count", 0) + 1
    context = _build_context(session)

    needs_category = (
        intake_state.get("primary_issue_cluster") is None
        or intake_state.get("category_confidence") in (None, "low")
    )

    # ── Round A: category detection + fact extraction (parallel) ─────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if needs_category:
            futures["category"] = pool.submit(_detect_category, msg, context)
        futures["facts"] = pool.submit(_update_known_facts, intake_state, msg)

        category_result = None
        for key, fut in futures.items():
            try:
                if key == "category":
                    category_result = fut.result(timeout=60)
                else:
                    fut.result(timeout=60)
            except Exception as exc:
                logger.warning("Round A parallel call %r failed: %s", key, exc)

    if needs_category and category_result:
        d = category_result
        intake_state["primary_issue_cluster"]       = d.get("primary_category")
        intake_state["secondary_issue_clusters"]    = d.get("secondary_categories") or []
        intake_state["category_confidence"]         = d.get("confidence")
        intake_state["issue_summary"]               = d.get("issue_summary") or intake_state.get("issue_summary") or ""
        intake_state["jurisdiction"]                = d.get("jurisdiction") or intake_state.get("jurisdiction") or "unknown"
        if not intake_state.get("urgency_signal") or intake_state.get("urgency_signal") == "unknown":
            intake_state["urgency_signal"]          = d.get("urgency_signal") or "unknown"
        intake_state["risk_flags"]                  = d.get("risk_flags") or intake_state.get("risk_flags") or []
        intake_state["client_role"]                 = d.get("client_role") or intake_state.get("client_role") or "unknown"
        intake_state["other_party"]                 = d.get("other_party") or intake_state.get("other_party") or "unknown"
        intake_state["relationship_to_other_party"] = d.get("relationship_to_other_party") or intake_state.get("relationship_to_other_party") or "unknown"
        intake_state["timeframe_status"]            = d.get("timeframe_status") or intake_state.get("timeframe_status") or "unknown"
        intake_state["client_goal_initial"]         = d.get("client_goal_initial") or intake_state.get("client_goal_initial")
        intake_state["immediate_need"]              = d.get("immediate_need") or intake_state.get("immediate_need") or "unknown"
        intake_state["emotional_ask"]               = d.get("emotional_ask") or intake_state.get("emotional_ask") or "unknown"

    # ── Round B: urgency recheck + remedy assessment (parallel) ──────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures_b = {
            "urgency": pool.submit(_recheck_urgency, intake_state, msg),
        }
        if intake_state.get("client_goal_initial") and not intake_state.get("assessed_remedy"):
            futures_b["remedy"] = pool.submit(_assess_remedy, intake_state)

        for key, fut in futures_b.items():
            try:
                fut.result(timeout=60)
            except Exception as exc:
                logger.warning("Round B parallel call %r failed: %s", key, exc)

    # ── Round C: generate reply ────────────────────────────────────────────────
    if intake_state.get("urgency_signal") == "immediate":
        reply = _generate_safety_first_response(intake_state)
    elif intake_state.get("turn_count", 0) >= 2:
        vetting_q = _get_vetting_question(intake_state, context)
        reply = vetting_q if vetting_q else _generate_followup(intake_state, msg, context)
    else:
        reply = _generate_followup(intake_state, msg, context)

    # ── Readiness check (deterministic — no LLM) ─────────────────────────────
    if intake_state.get("turn_count", 0) < 2:
        ready, missing = False, []
    else:
        ready, missing = _check_readiness(intake_state, context)

    intake_state["ready_for_stage2"] = ready
    if missing:
        existing_open = set(str(x).lower() for x in intake_state.get("open_questions") or [])
        for item in missing:
            if item.lower() not in existing_open:
                intake_state.setdefault("open_questions", []).append(item)
                existing_open.add(item.lower())

    logger.info(
        "Intake turn %d | category=%s | facts=%d | ready=%s | urgency=%s",
        intake_state["turn_count"],
        intake_state.get("primary_issue_cluster"),
        len(intake_state.get("known_facts") or []),
        ready,
        intake_state.get("urgency_signal"),
    )

    return {
        "reply":             reply,
        "advance_to_stage2": ready,
        "urgency_signal":    intake_state.get("urgency_signal", "unknown"),
    }


def _run_stage1_parallel_v2(session: dict, msg: str, intake_state: dict) -> dict:
    """
    Compact intake flow:
    1. Detect category and collect facts.
    2. Recheck urgency and assess remedy when possible.
    3. If immediate danger is active, ask the safety question first.
    4. Otherwise send one grouped detail request.
    5. On the next substantive turn, run one gap review and either ask only
       for missing grouped details or mark the intake ready for analysis.
    """
    from agents.intake.stage1_opening import (
        _detect_category,
        _update_known_facts,
        _recheck_urgency,
        _assess_remedy,
        _build_context,
        _generate_safety_first_response,
        _generate_initial_detail_request,
        _generate_gap_review,
        _check_analysis_readiness,
        _refresh_case_file,
    )

    intake_state["turn_count"] = intake_state.get("turn_count", 0) + 1
    context = _build_context(session)

    needs_category = (
        intake_state.get("primary_issue_cluster") is None
        or intake_state.get("category_confidence") in (None, "low")
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if needs_category:
            futures["category"] = pool.submit(_detect_category, msg, context)
        futures["facts"] = pool.submit(_update_known_facts, intake_state, msg)

        category_result = None
        for key, fut in futures.items():
            try:
                if key == "category":
                    category_result = fut.result(timeout=60)
                else:
                    fut.result(timeout=60)
            except Exception as exc:
                logger.warning("Stage1 v2 round A %r failed: %s", key, exc)

    if needs_category and category_result:
        d = category_result
        intake_state["primary_issue_cluster"] = d.get("primary_category")
        intake_state["secondary_issue_clusters"] = d.get("secondary_categories") or []
        intake_state["category_confidence"] = d.get("confidence")
        intake_state["issue_summary"] = d.get("issue_summary") or intake_state.get("issue_summary") or ""
        intake_state["jurisdiction"] = d.get("jurisdiction") or intake_state.get("jurisdiction") or "unknown"
        if not intake_state.get("urgency_signal") or intake_state.get("urgency_signal") == "unknown":
            intake_state["urgency_signal"] = d.get("urgency_signal") or "unknown"
        intake_state["risk_flags"] = d.get("risk_flags") or intake_state.get("risk_flags") or []
        intake_state["client_role"] = d.get("client_role") or intake_state.get("client_role") or "unknown"
        intake_state["other_party"] = d.get("other_party") or intake_state.get("other_party") or "unknown"
        intake_state["relationship_to_other_party"] = (
            d.get("relationship_to_other_party") or intake_state.get("relationship_to_other_party") or "unknown"
        )
        intake_state["timeframe_status"] = d.get("timeframe_status") or intake_state.get("timeframe_status") or "unknown"
        intake_state["client_goal_initial"] = d.get("client_goal_initial") or intake_state.get("client_goal_initial")
        intake_state["immediate_need"] = d.get("immediate_need") or intake_state.get("immediate_need") or "unknown"
        intake_state["emotional_ask"] = d.get("emotional_ask") or intake_state.get("emotional_ask") or "unknown"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures_b = {
            "urgency": pool.submit(_recheck_urgency, intake_state, msg),
        }
        if intake_state.get("client_goal_initial") and not intake_state.get("assessed_remedy"):
            futures_b["remedy"] = pool.submit(_assess_remedy, intake_state)

        for key, fut in futures_b.items():
            try:
                fut.result(timeout=60)
            except Exception as exc:
                logger.warning("Stage1 v2 round B %r failed: %s", key, exc)

    if intake_state.get("urgency_signal") == "immediate":
        reply = _generate_safety_first_response(intake_state)
        ready, missing = False, []
    elif not intake_state.get("detail_request_issued"):
        reply = _generate_initial_detail_request(intake_state, msg, context)
        # Honour the LLM's judgment: if the opening message already contained
        # enough facts, skip further intake and go straight to analysis.
        if intake_state.get("analysis_ready"):
            ready, missing = True, []
        else:
            ready, missing = False, []
    else:
        reply, _ = _generate_gap_review(intake_state, msg, context)
        ready, missing = _check_analysis_readiness(intake_state)

    intake_state["ready_for_stage2"] = ready
    if missing:
        intake_state["open_questions"] = list(missing)

    _refresh_case_file(intake_state, context)

    logger.info(
        "Intake v2 turn %d | category=%s | facts=%d | ready=%s | urgency=%s | detail_request=%s",
        intake_state["turn_count"],
        intake_state.get("primary_issue_cluster"),
        len(intake_state.get("known_facts") or []),
        ready,
        intake_state.get("urgency_signal"),
        intake_state.get("detail_request_issued"),
    )

    return {
        "reply": reply,
        "advance_to_stage2": ready,
        "urgency_signal": intake_state.get("urgency_signal", "unknown"),
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _process_turn_node(state: IntakeState) -> dict:
    """
    Core intake processing node.  Runs Stage-1 with parallel LLM calls.
    First turn with empty message → generate opening greeting.
    """
    msg = _last_human_message(state)
    intake_state = dict(state.get("intake_state") or _fresh_intake_state())

    if not msg:
        from agents.intake.stage1_opening import generate_opening
        reply = generate_opening()
        return {
            "messages":          [AIMessage(content=reply)],
            "intake_state":      intake_state,
            "reply":             reply,
            "advance_to_stage2": False,
            "urgency_signal":    "unknown",
        }

    session = _session_dict_from_state(state)
    result = _run_stage1_parallel_v2(session, msg, intake_state)

    return {
        "messages":          [AIMessage(content=result["reply"])],
        "intake_state":      intake_state,
        "reply":             result["reply"],
        "advance_to_stage2": result["advance_to_stage2"],
        "urgency_signal":    result["urgency_signal"],
    }


def _wait_for_user_node(state: IntakeState) -> dict:
    """
    Human-in-the-Loop: client turn pause.

    interrupt() suspends execution and returns the reply to the tool caller
    via the checkpointed snapshot.  Resumed by continue_intake() via
    Command(resume=client_message).
    """
    next_msg: str = interrupt(state.get("reply", ""))
    return {
        "messages": [HumanMessage(content=next_msg)],
    }


def _draft_node(state: IntakeState) -> dict:
    """
    Stage-2 draft generation node.

    If advocate_notes are present (revision requested), they are passed to
    build_draft so the LLM can address the feedback.
    """
    intake_state = state.get("intake_state") or {}
    session = _session_dict_from_state(state)
    advocate_notes = (state.get("advocate_notes") or "").strip()

    try:
        from agents.intake.stage5_draft import build_draft
        # Pass revision notes if available; fall back gracefully if not accepted
        try:
            result = build_draft(
                intake_state,
                session.get("history", []),
                revision_notes=advocate_notes,
            )
        except TypeError:
            result = build_draft(intake_state, session.get("history", []))

        draft_text = result.get("formatted_draft", "")
        advocate_review = result.get("advocate_review", {})

        if advocate_notes:
            logger.info("Draft re-generated with advocate revision notes.")
    except Exception as exc:
        logger.exception("Draft generation failed: %s", exc)
        draft_text = "Draft generation failed. Please try again."
        advocate_review = {}

    payload = json.dumps(
        {"formatted_draft": draft_text, "advocate_review": advocate_review},
        ensure_ascii=False,
    )

    return {
        "messages":          [AIMessage(content=draft_text)],
        "reply":             payload,
        "advocate_approved": False,   # reset for this draft pass
        "advocate_notes":    "",
    }


def _advocate_review_node(state: IntakeState) -> dict:
    """
    Advocate-in-the-Loop gate before the draft is delivered to the client.

    interrupt() fires with the full draft payload:
      {"draft": "<formatted_draft>", "advocate_review": {...}, "action": "advocate_review"}

    The advocate (or API caller) resumes via Command(resume=...) with:
      {"approved": True}                           — accept the draft as-is
      {"approved": False, "notes": "revision..."}  — request revisions

    If approved:  session summary saved to cross-session memory → END.
    If rejected:  advocate_notes injected into state → draft node re-runs.
    """
    # Parse draft payload from state
    raw_reply = state.get("reply", "")
    try:
        draft_payload = json.loads(raw_reply)
    except (json.JSONDecodeError, TypeError):
        draft_payload = {"formatted_draft": raw_reply, "advocate_review": {}}

    # Pause — advocate reads the draft here
    resume_data = interrupt({
        "action":          "advocate_review",
        "draft":           draft_payload.get("formatted_draft", ""),
        "advocate_review": draft_payload.get("advocate_review", {}),
        "session_id":      state.get("session_id", ""),
        "urgency":         state.get("urgency_signal", "unknown"),
    })

    # Process advocate's decision
    if isinstance(resume_data, dict):
        approved = bool(resume_data.get("approved", True))
        notes    = str(resume_data.get("notes") or "").strip()
    else:
        approved = bool(resume_data)
        notes    = ""

    if approved:
        # Save session summary to cross-session memory
        user_id    = state.get("user_id")
        session_id = state.get("session_id")
        if user_id and session_id:
            try:
                from agents.memory import save_session_summary
                save_session_summary(user_id, session_id, state.get("intake_state") or {})
            except Exception as exc:
                logger.warning("save_session_summary failed: %s", exc)

        logger.info("Advocate approved draft for session=%s", state.get("session_id"))
        return {
            "advocate_approved": True,
            "advocate_notes":    "",
        }
    else:
        logger.info(
            "Advocate requested revisions for session=%s: %s",
            state.get("session_id"), notes[:100],
        )
        return {
            "advocate_approved": False,
            "advocate_notes":    notes,
        }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_process(state: IntakeState) -> Literal["wait_for_user", "draft"]:
    if state.get("advance_to_stage2"):
        return "draft"
    return "wait_for_user"


def _route_after_advocate(state: IntakeState) -> Literal["draft", "__end__"]:
    """If advocate approved → END. If rejected → re-run draft with revision notes."""
    if state.get("advocate_approved"):
        return END
    return "draft"


# ---------------------------------------------------------------------------
# Build and compile the graph (once at module import)
# ---------------------------------------------------------------------------

def _build_intake_graph():
    builder = StateGraph(IntakeState)

    builder.add_node("process_turn",    _process_turn_node)
    builder.add_node("wait_for_user",   _wait_for_user_node)
    builder.add_node("draft",           _draft_node)
    builder.add_node("advocate_review", _advocate_review_node)

    builder.add_edge(START, "process_turn")
    builder.add_conditional_edges(
        "process_turn",
        _route_after_process,
        {"wait_for_user": "wait_for_user", "draft": "draft"},
    )
    builder.add_edge("wait_for_user", "process_turn")
    builder.add_edge("draft", "advocate_review")
    builder.add_conditional_edges(
        "advocate_review",
        _route_after_advocate,
        {"draft": "draft", END: END},
    )

    checkpointer = _get_checkpointer()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_user", "advocate_review"],
    )


intake_graph = _build_intake_graph()


# ---------------------------------------------------------------------------
# Public helpers used by mcp_server tool implementations
# ---------------------------------------------------------------------------

def start_intake_session(first_message: str = "", user_id: str = "") -> dict:
    """
    Create a new intake session and process the first client message.

    Parameters
    ----------
    first_message : First message from the client (optional).
    user_id       : Optional user identifier for cross-session memory.
    """
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial_state: IntakeState = {
        "messages":          [HumanMessage(content=first_message)] if first_message else [],
        "session_id":        session_id,
        "intake_state":      _fresh_intake_state(),
        "advance_to_stage2": False,
        "reply":             "",
        "urgency_signal":    "unknown",
        "advocate_approved": False,
        "advocate_notes":    "",
        "user_id":           user_id or None,
    }

    state = intake_graph.invoke(initial_state, config=config)

    return {
        "session_id":        session_id,
        "reply":             state.get("reply", ""),
        "intake_state":      state.get("intake_state", {}),
        "advance_to_stage2": state.get("advance_to_stage2", False),
        "urgency_signal":    state.get("urgency_signal", "unknown"),
        "stage":             "stage2" if state.get("advance_to_stage2") else "stage1",
    }


def continue_intake_session(session_id: str, client_message: str) -> dict:
    """
    Resume an existing intake session with the client's next message.
    """
    config = {"configurable": {"thread_id": session_id}}
    state = intake_graph.invoke(Command(resume=client_message), config=config)

    return {
        "session_id":        session_id,
        "reply":             state.get("reply", ""),
        "intake_state":      state.get("intake_state", {}),
        "advance_to_stage2": state.get("advance_to_stage2", False),
        "urgency_signal":    state.get("urgency_signal", "unknown"),
        "stage":             "stage2" if state.get("advance_to_stage2") else "stage1",
    }


def advocate_review_session(
    session_id: str,
    approved: bool,
    notes: str = "",
) -> dict:
    """
    Submit the advocate's review decision for a draft.

    Parameters
    ----------
    session_id : Active intake session.
    approved   : True to accept the draft; False to request revisions.
    notes      : Revision instructions (required when approved=False).
    """
    config = {"configurable": {"thread_id": session_id}}
    resume_data = {"approved": approved, "notes": notes}
    state = intake_graph.invoke(Command(resume=resume_data), config=config)

    return {
        "session_id":        session_id,
        "advocate_approved": state.get("advocate_approved", False),
        "advocate_notes":    state.get("advocate_notes", ""),
        "reply":             state.get("reply", ""),
        "stage":             "approved" if state.get("advocate_approved") else "revision_requested",
    }


def get_intake_session_state(session_id: str) -> dict:
    """
    Inspect the current state of an intake session without advancing it.
    Reads directly from the checkpointer — no LLM call.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = intake_graph.get_state(config)
        values = snapshot.values
        intake_state = values.get("intake_state") or {}
        turn_count = sum(
            1 for m in (values.get("messages") or [])
            if isinstance(m, HumanMessage)
        )
        return {
            "session_id":        session_id,
            "stage":             "stage2" if values.get("advance_to_stage2") else "stage1",
            "intake_state":      intake_state,
            "turn_count":        turn_count,
            "category":          intake_state.get("primary_issue_cluster"),
            "urgency":           intake_state.get("urgency_signal"),
            "advocate_approved": values.get("advocate_approved", False),
            "next_nodes":        list(snapshot.next),
        }
    except Exception as exc:
        logger.error("get_intake_session_state failed for %s: %s", session_id, exc)
        return {"error": str(exc), "session_id": session_id}


def get_draft_from_session(session_id: str) -> dict:
    """
    Read the draft from a completed intake session.
    Reads directly from the checkpointer — no re-generation.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = intake_graph.get_state(config)
        values = snapshot.values
        intake_state = values.get("intake_state") or {}

        if not intake_state.get("known_facts"):
            return {
                "error": "Intake not complete — no facts collected yet.",
                "session_id": session_id,
            }

        raw_reply = values.get("reply", "")
        try:
            parsed = json.loads(raw_reply)
            return {
                "session_id":      session_id,
                "formatted_draft": parsed.get("formatted_draft", ""),
                "advocate_review": parsed.get("advocate_review", {}),
                "advocate_approved": values.get("advocate_approved", False),
            }
        except (json.JSONDecodeError, TypeError):
            pass

        # Draft not yet generated — run it now
        from agents.intake.stage5_draft import build_draft
        session = {
            "history": [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant",
                 "content": m.content or ""}
                for m in (values.get("messages") or [])
            ]
        }
        result = build_draft(intake_state, session.get("history", []))
        return {
            "session_id":      session_id,
            "formatted_draft": result.get("formatted_draft", ""),
            "advocate_review": result.get("advocate_review", {}),
            "advocate_approved": False,
        }

    except Exception as exc:
        logger.exception("get_draft_from_session failed for %s: %s", session_id, exc)
        return {"error": str(exc), "session_id": session_id}


def get_session_history(session_id: str) -> list[dict]:
    """
    Time-travel: return a summary of every checkpoint stored for this session.

    Each entry describes the state AT that checkpoint — useful for the
    advocate review panel to replay the intake conversation step by step.

    Returns
    -------
    list[dict] — most recent checkpoint first.
    Each entry:
      checkpoint_index, next_nodes, turn_count, category, urgency,
      facts_count, advance_to_stage2, advocate_approved
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshots = list(intake_graph.get_state_history(config))
        history = []
        for i, snap in enumerate(snapshots):
            values      = snap.values
            is_obj      = values if isinstance(values, dict) else {}
            istate      = is_obj.get("intake_state") or {}
            history.append({
                "checkpoint_index": i,
                "created_at":       str(getattr(snap, "created_at", "")),
                "next_nodes":       list(snap.next),
                "turn_count":       istate.get("turn_count", 0),
                "category":         istate.get("primary_issue_cluster"),
                "urgency":          is_obj.get("urgency_signal", "unknown"),
                "facts_count":      len(istate.get("known_facts") or []),
                "advance_to_stage2": is_obj.get("advance_to_stage2", False),
                "advocate_approved": is_obj.get("advocate_approved", False),
                "reply_preview":    (is_obj.get("reply") or "")[:120],
            })
        return history
    except Exception as exc:
        logger.error("get_session_history failed for %s: %s", session_id, exc)
        return [{"error": str(exc), "session_id": session_id}]
