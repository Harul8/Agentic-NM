"""
Legal Opinion Intake — Stage 2: Structured Deep-Dive

Handles the second phase of the autonomous intake workflow.

Stage 2 receives the locked category + known facts from Stage 1 and
systematically gathers the specific facts that a court requires — one
question per turn, dynamically selected based on which legal elements
are still uncovered.

Flow per turn
-------------
  1. Extract new facts from the client's message + map them to legal elements
  2. Update the element-coverage map (covered / partial / missing)
  3. Identify the highest-priority uncovered element
  4. Generate one warm, plain-language question targeting that gap
  5. Run the completeness check — advance to Stage 3 when all critical
     elements have at least partial coverage

Public API
----------
  generate_stage2_opening(stage1_state, model_override)  → str
  process_turn(session, user_message, model_override)     → dict
      Returns: {reply, stage2_state, advance_to_stage3, urgency_signal}
  new_stage2_state(stage1_state)                          → dict
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM helpers
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "fast", model_override: str | None = None) -> str:
    from llm.ollama_client import ask_llm
    return ask_llm(prompt, task_hint=task_hint, model=model_override) or ""


def _ask_llm_quality(prompt: str, model_override: str | None = None) -> str:
    return _ask_llm(prompt, task_hint="quality", model_override=model_override)


# ---------------------------------------------------------------------------
# Prompt imports
# ---------------------------------------------------------------------------

from prompts.advocate_prompts import (
    LEGAL_ELEMENT_CHECKLISTS,
    LEGAL_ISSUE_CATEGORIES,
    STAGE2_OPENING_SYSTEM,
    STAGE2_QUESTION_SYSTEM,
    STAGE2_FACT_EXTRACTOR_SYSTEM,
    STAGE2_COMPLETENESS_CHECK_SYSTEM,
    STAGE2_STATE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Debug timing (optional)
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage2.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# JSON extraction helper (shared pattern with Stage 1)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if "{" in part and "}" in part:
                text = part
                break
    start = text.find("{")
    end   = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Conversation context builder
# ---------------------------------------------------------------------------

def _build_context(session: dict, max_turns: int = 8, max_chars: int = 280) -> str:
    history = session.get("history", [])
    lines: list[str] = []
    for turn in history[-max_turns:]:
        role    = "Client" if turn.get("role") == "user" else "Counsel"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "…"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------

def _get_elements(category: str) -> list[dict]:
    return list(LEGAL_ELEMENT_CHECKLISTS.get(category, LEGAL_ELEMENT_CHECKLISTS["general"]))


def _elements_as_text(elements: list[dict]) -> str:
    lines = []
    for el in elements:
        lines.append(f"- [{el['priority'].upper()}] {el['name']}: {el['description']}")
    return "\n".join(lines)


def _coverage_as_text(coverage: dict, elements: list[dict]) -> str:
    lines = []
    for el in elements:
        status = coverage.get(el["name"], "missing")
        lines.append(f"  {el['name']}: {status}")
    return "\n".join(lines)


def _next_gap(coverage: dict, elements: list[dict]) -> str:
    """Return the description of the highest-priority uncovered element."""
    for priority in ("critical", "important", "supporting"):
        for el in elements:
            if el["priority"] == priority and coverage.get(el["name"], "missing") == "missing":
                return f"{el['name']} — {el['description']}"
    # All missing elements are partial — find the first partial critical
    for el in elements:
        if el["priority"] == "critical" and coverage.get(el["name"]) == "partial":
            return f"{el['name']} (needs more detail) — {el['description']}"
    return "Confirm any final details the client wants to add"


def _initialise_coverage(elements: list[dict]) -> dict:
    return {el["name"]: "missing" for el in elements}


# ---------------------------------------------------------------------------
# Urgency check (reused pattern from Stage 1)
# ---------------------------------------------------------------------------

_IMMEDIATE_SIGNALS = (
    "beaten", "hit", "assault", "arrested", "arrest", "locked out",
    "thrown out", "evict", "evicted", "threatened", "threat", "danger",
    "hospital", "injury", "injured", "court today", "hearing today",
    "deadline today", "auction today", "sale today",
)


def _recheck_urgency(state: dict, client_message: str) -> None:
    if state.get("urgency_signal") == "immediate":
        return
    low = (client_message or "").lower()
    if any(sig in low for sig in _IMMEDIATE_SIGNALS):
        state["urgency_signal"] = "immediate"


# ===========================================================================
# Public: new_stage2_state — create from Stage 1 handoff
# ===========================================================================

def new_stage2_state(stage1_state: dict) -> dict:
    """
    Create a fresh Stage 2 state dict seeded from the Stage 1 intake state.
    Call this once when Stage 1 reports advance_to_stage2=True.
    """
    category      = (stage1_state.get("category") or "general").strip()
    cat_cfg       = LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"])
    elements      = _get_elements(category)
    coverage      = _initialise_coverage(elements)

    state = copy.deepcopy(STAGE2_STATE_SCHEMA)
    state.update({
        "category":        category,
        "category_label":  cat_cfg.get("label", category),
        "secondary_issue_clusters": list(stage1_state.get("secondary_issue_clusters") or []),
        "jurisdiction":    stage1_state.get("jurisdiction") or "unknown",
        "urgency_signal":  stage1_state.get("urgency_signal") or "unknown",
        "risk_flags":      list(stage1_state.get("risk_flags") or []),
        "client_role":     stage1_state.get("client_role") or "unknown",
        "other_party":     stage1_state.get("other_party") or "unknown",
        "relationship_to_other_party": stage1_state.get("relationship_to_other_party") or "unknown",
        "timeframe_status": stage1_state.get("timeframe_status") or "unknown",
        "client_goal_initial": stage1_state.get("client_goal_initial"),
        "emotional_ask": stage1_state.get("emotional_ask"),
        "immediate_need": stage1_state.get("immediate_need"),
        # Seed known_facts from Stage 1 — Stage 2 appends more
        "known_facts":     list(stage1_state.get("known_facts") or []),
        "fact_records":    list(stage1_state.get("fact_records") or []),
        "legal_element_coverage": coverage,
        "open_elements":   [el["name"] for el in elements if el["priority"] == "critical"],
    })
    return state


# ===========================================================================
# Public: generate_stage2_opening
# ===========================================================================

def generate_stage2_opening(
    stage1_state: dict,
    model_override: str | None = None,
) -> str:
    """
    Generate the brief transition message from Stage 1 to Stage 2.
    Should feel like a natural deepening, not the start of a new form.
    """
    t0 = time.perf_counter()
    issue_summary = (stage1_state.get("issue_summary") or "").strip() or "the situation"
    urgency       = stage1_state.get("urgency_signal") or "unknown"

    prompt = (
        STAGE2_OPENING_SYSTEM
        .replace("{issue_summary}", issue_summary)
        .replace("{urgency_signal}", urgency)
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("generate_opening", t0)
        if reply and len(reply) > 20:
            return reply
    except Exception as exc:
        logger.warning("Stage2 opening LLM failed: %s", exc)

    # Static fallback — warm, not robotic
    return (
        "Thank you for sharing that. To make sure we cover everything that matters, "
        "I'd like to ask you a few more specific questions about what happened."
    )


# ===========================================================================
# Internal: extract facts + map to legal elements
# ===========================================================================

def _extract_and_map_facts(
    client_message: str,
    category: str,
    conversation_context: str,
    model_override: str | None = None,
) -> dict:
    """
    Extract atomic facts from the client's message and map each to a
    legal element. Returns the raw extraction JSON dict.
    """
    t0 = time.perf_counter()
    elements = _get_elements(category)
    elements_text = _elements_as_text(elements)

    prompt = (
        STAGE2_FACT_EXTRACTOR_SYSTEM
        .replace("{category}", category)
        .replace("{elements_list}", elements_text)
        .replace("{client_message}", (client_message or "").strip())
        .replace("{conversation_context}", conversation_context or "(none)")
    )

    _DEFAULT = {
        "extracted_facts": [],
        "new_evidence_items": [],
        "remedy_stated": None,
        "urgency_escalation": False,
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("extract_facts", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage2 fact extraction failed: %s", exc)

    return _DEFAULT


# ===========================================================================
# Internal: update coverage from extracted facts
# ===========================================================================

def _update_coverage(stage2_state: dict, extraction: dict) -> None:
    """
    Merge newly extracted facts and evidence into the Stage 2 state.
    Promote element coverage from 'missing' → 'partial' → 'covered'.
    """
    elements      = _get_elements(stage2_state["category"])
    valid_names   = {el["name"] for el in elements}
    coverage      = stage2_state.setdefault("legal_element_coverage", {})
    known_facts   = stage2_state.setdefault("known_facts", [])
    evidence_items = stage2_state.setdefault("evidence_items", [])

    existing_facts_lower = {f.lower() for f in known_facts}

    for item in (extraction.get("extracted_facts") or []):
        fact      = str(item.get("fact") or "").strip()
        el_name   = str(item.get("legal_element") or "general_background").strip()
        confidence = str(item.get("confidence") or "stated").strip()
        ev_mentioned = bool(item.get("evidence_mentioned"))
        ev_type   = item.get("evidence_type")

        # Add fact to known_facts if not a near-duplicate
        if fact and fact.lower() not in existing_facts_lower and len(fact) > 8:
            existing_facts_lower.add(fact.lower())
            known_facts.append(fact)

        # Update coverage
        if el_name in valid_names:
            current = coverage.get(el_name, "missing")
            if current == "missing":
                coverage[el_name] = "partial" if confidence == "uncertain" else "covered"
            elif current == "partial" and confidence in ("stated", "implied"):
                coverage[el_name] = "covered"

        # Collect evidence
        if ev_mentioned and ev_type:
            ev_str = str(ev_type).strip()
            if ev_str and ev_str not in evidence_items:
                evidence_items.append(ev_str)

    # Collect any explicitly listed evidence items
    for ev in (extraction.get("new_evidence_items") or []):
        ev_str = str(ev).strip()
        if ev_str and ev_str not in evidence_items:
            evidence_items.append(ev_str)

    # Capture stated remedy
    remedy = extraction.get("remedy_stated")
    if remedy and not stage2_state.get("remedy_stated"):
        stage2_state["remedy_stated"] = str(remedy).strip()[:300]

    # Urgency escalation
    if extraction.get("urgency_escalation"):
        stage2_state["urgency_signal"] = "immediate"

    # Keep known_facts capped
    stage2_state["known_facts"] = known_facts[:30]

    # Recompute open_elements
    elements_list = _get_elements(stage2_state["category"])
    stage2_state["open_elements"] = [
        el["name"]
        for el in elements_list
        if el["priority"] in ("critical", "important") and coverage.get(el["name"], "missing") != "covered"
    ]


# ===========================================================================
# Internal: generate the question for this turn
# ===========================================================================

def _generate_question(
    stage2_state: dict,
    client_message: str,
    conversation_context: str,
    model_override: str | None = None,
) -> str:
    t0   = time.perf_counter()
    cat  = stage2_state["category"]
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(cat, LEGAL_ISSUE_CATEGORIES["general"])
    elements = _get_elements(cat)

    primary_acts = "\n".join(f"- {a}" for a in cat_cfg.get("primary_acts", []))
    elements_list_text = _elements_as_text(elements)
    coverage_text = _coverage_as_text(stage2_state.get("legal_element_coverage", {}), elements)
    gap = _next_gap(stage2_state.get("legal_element_coverage", {}), elements)

    known_facts = stage2_state.get("known_facts", [])
    facts_summary = "\n".join(f"- {f}" for f in known_facts[-15:]) or "(none yet)"

    prompt = (
        STAGE2_QUESTION_SYSTEM
        .replace("{category_label}", stage2_state.get("category_label", cat))
        .replace("{primary_acts_list}", primary_acts or "(general law applies)")
        .replace("{legal_elements_list}", elements_list_text)
        .replace("{known_facts_summary}", facts_summary)
        .replace("{coverage_status}", coverage_text)
        .replace("{next_gap}", gap)
        .replace("{conversation_context}", conversation_context or "(first turn of Stage 2)")
        .replace("{client_message}", (client_message or "").strip())
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("generate_question", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage2 question generation failed: %s", exc)

    # Generic fallback — no internal element names exposed to client
    return "Thank you for sharing that. Could you help me understand a bit more about what happened?"


# ===========================================================================
# Internal: completeness check
# ===========================================================================

def _check_completeness(
    stage2_state: dict,
    conversation_context: str,
    model_override: str | None = None,
) -> tuple[bool, list[str]]:
    """Returns (ready_for_stage3: bool, remaining_critical_gaps: list[str])."""
    t0 = time.perf_counter()

    # Fast heuristic: skip LLM if obvious
    if stage2_state.get("turn_count", 0) < 4:
        return False, ["need at least 4 substantive turns before advancing"]

    elements = _get_elements(stage2_state["category"])
    elements_text = _elements_as_text(elements)
    state_json = json.dumps(stage2_state, ensure_ascii=False, indent=2)

    prompt = (
        STAGE2_COMPLETENESS_CHECK_SYSTEM
        .replace("{category_label}", stage2_state.get("category_label", stage2_state["category"]))
        .replace("{elements_list}", elements_text)
        .replace("{stage2_state_json}", state_json)
    )

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("check_completeness", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            ready   = bool(out.get("ready_for_stage3", False))
            gaps    = [str(g).strip() for g in (out.get("remaining_critical_gaps") or []) if str(g).strip()]

            # Sync coverage from LLM output if provided
            if isinstance(out.get("coverage_summary"), dict):
                for el_name, status in out["coverage_summary"].items():
                    if status in ("covered", "partial", "missing"):
                        stage2_state["legal_element_coverage"][el_name] = status

            return ready, gaps
    except Exception as exc:
        logger.warning("Stage2 completeness check failed: %s", exc)

    return False, []


# ===========================================================================
# Public: process_turn
# ===========================================================================

def process_turn(
    session: dict,
    user_message: str,
    model_override: str | None = None,
) -> dict:
    """
    Process one client turn in the Stage 2 structured deep-dive.

    Parameters
    ----------
    session : dict
        Must contain:
          - "history"      : list of {role, content} turns
          - "stage2_state" : the running Stage 2 state dict (or None for first turn)
    user_message : str
        The client's latest message.
    model_override : str | None
        LLM model override (e.g. "provider:openai").

    Returns
    -------
    dict:
      - "reply"              : str   — AI's reply to send to client
      - "stage2_state"       : dict  — updated Stage 2 state
      - "advance_to_stage3"  : bool
      - "urgency_signal"     : str
    """
    msg = (user_message or "").strip()

    # ── Load state ───────────────────────────────────────────────────────────
    stage2_state: dict = session.get("stage2_state") or {}
    if not stage2_state.get("category"):
        # Fallback: should not happen if caller is correct
        logger.warning("Stage2 process_turn called without initialised stage2_state")
        stage2_state = copy.deepcopy(STAGE2_STATE_SCHEMA)
        stage2_state["category"] = "general"
        stage2_state["category_label"] = "General Matter"
        stage2_state["legal_element_coverage"] = _initialise_coverage(_get_elements("general"))

    stage2_state["turn_count"] = stage2_state.get("turn_count", 0) + 1

    # ── Build conversation context ────────────────────────────────────────────
    context = _build_context(session, max_turns=8)

    # ── Urgency recheck ───────────────────────────────────────────────────────
    _recheck_urgency(stage2_state, msg)

    # ── Extract facts and update element coverage ─────────────────────────────
    extraction = _extract_and_map_facts(
        msg,
        stage2_state["category"],
        context,
        model_override=model_override,
    )
    _update_coverage(stage2_state, extraction)

    # ── Generate the question ─────────────────────────────────────────────────
    reply = _generate_question(stage2_state, msg, context, model_override=model_override)

    # ── Completeness check ────────────────────────────────────────────────────
    ready, gaps = _check_completeness(stage2_state, context, model_override=model_override)
    stage2_state["ready_for_stage3"] = ready

    logger.info(
        "Stage2 turn %d | category=%s | covered=%d/%d | ready=%s | urgency=%s",
        stage2_state["turn_count"],
        stage2_state.get("category"),
        sum(1 for v in stage2_state.get("legal_element_coverage", {}).values() if v in ("covered", "partial")),
        len(stage2_state.get("legal_element_coverage", {})),
        ready,
        stage2_state.get("urgency_signal"),
    )

    return {
        "reply":             reply,
        "stage2_state":      stage2_state,
        "advance_to_stage3": ready,
        "urgency_signal":    stage2_state.get("urgency_signal", "unknown"),
    }
