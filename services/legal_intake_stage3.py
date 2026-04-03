"""
Legal Opinion Intake — Stage 3: Indirect Vetting

Handles the third phase of the autonomous intake workflow.

Stage 3 validates the client's account without ever making them feel
interrogated. Every question is framed as helping them build a stronger
case. The AI internally builds a confidence map that determines how the
draft will be written (assertive vs. hedged language) and flags evidence
gaps that need to be filled before filing.

Four vetting techniques (one per turn, chosen strategically):
  triangulation  — "who else was present / witnessed this?"
  timeline_probe — "help me get the sequence right — what happened next?"
  doc_anchoring  — "do you have any messages / medical reports from that time?"
  restatement    — "just to make sure I have this right — [key facts]"

Flow per turn
-------------
  1. Choose the next vetting technique and target fact
  2. Generate one warm, non-accusatory question
  3. After the client answers: update confidence map, detect timeline issues
  4. Check completeness → advance to Stage 4 when all critical facts probed

Output
------
  stage3_state["vetting_report"] feeds Stage 4 (remedy assessment) and
  Stage 6 (draft) — it tells the draft which facts are "high confidence"
  (state assertively) vs. "uncertain" (use "as alleged by client" hedging).

Public API
----------
  generate_stage3_opening(stage2_state, model_override)  → str
  process_turn(session, user_message, model_override)     → dict
      Returns: {reply, stage3_state, advance_to_stage4, urgency_signal}
  new_stage3_state(stage2_state)                          → dict
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time

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
    STAGE3_OPENING_SYSTEM,
    STAGE3_VETTING_QUESTION_SYSTEM,
    STAGE3_CONFIDENCE_ASSESSOR_SYSTEM,
    STAGE3_COMPLETENESS_CHECK_SYSTEM,
    STAGE3_STATE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Debug timing
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage3.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# JSON helpers
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

def _build_context(session: dict, max_turns: int = 10, max_chars: int = 300) -> str:
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
# Urgency recheck
# ---------------------------------------------------------------------------

_IMMEDIATE_SIGNALS = (
    "beaten", "hit", "assault", "arrested", "arrest", "locked out",
    "thrown out", "evict", "evicted", "threatened", "threat", "danger",
    "hospital", "injury", "injured", "court today", "hearing today",
    "deadline today",
)


def _recheck_urgency(state: dict, msg: str) -> None:
    if state.get("urgency_signal") == "immediate":
        return
    if any(sig in (msg or "").lower() for sig in _IMMEDIATE_SIGNALS):
        state["urgency_signal"] = "immediate"


# ---------------------------------------------------------------------------
# Vetting technique sequencing
# ---------------------------------------------------------------------------

# Ordered priority: cover all 4 techniques once, then repeat as needed
_TECHNIQUE_ORDER = ["triangulation", "doc_anchoring", "timeline_probe", "restatement"]


def _choose_technique(stage3_state: dict) -> str:
    used = stage3_state.get("techniques_used") or []
    for t in _TECHNIQUE_ORDER:
        if t not in used:
            return t
    # All used at least once — pick the one most needed given confidence state
    confidence = stage3_state.get("fact_confidence") or {}
    has_uncertain = any(v == "uncertain" for v in confidence.values())
    return "restatement" if has_uncertain else "doc_anchoring"


def _choose_target_fact(stage3_state: dict) -> str:
    """Return the most critical fact that still needs probing."""
    known_facts    = stage3_state.get("known_facts") or []
    confidence     = stage3_state.get("fact_confidence") or {}
    evidence_items = stage3_state.get("evidence_items") or []

    # Build a short key for each fact
    def _key(f: str) -> str:
        return f[:60].strip()

    # Priority order: uncertain → medium with no evidence → medium → high
    uncertain_facts = [f for f in known_facts if confidence.get(_key(f)) == "uncertain"]
    if uncertain_facts:
        return uncertain_facts[0]

    medium_no_ev = [
        f for f in known_facts
        if confidence.get(_key(f)) == "medium"
        and not any(ev.lower() in f.lower() for ev in evidence_items)
    ]
    if medium_no_ev:
        return medium_no_ev[0]

    medium_facts = [f for f in known_facts if confidence.get(_key(f)) == "medium"]
    if medium_facts:
        return medium_facts[0]

    # All are high confidence or unassessed — pick the first unassessed
    unassessed = [f for f in known_facts if _key(f) not in confidence]
    if unassessed:
        return unassessed[0]

    # Fallback: first critical element
    category = stage3_state.get("category", "general")
    elements = LEGAL_ELEMENT_CHECKLISTS.get(category, LEGAL_ELEMENT_CHECKLISTS.get("general", []))
    for el in elements:
        if el.get("priority") == "critical":
            return el["description"]

    return known_facts[0] if known_facts else "the core incident"


# ---------------------------------------------------------------------------
# Format helpers for prompts
# ---------------------------------------------------------------------------

def _facts_summary(known_facts: list[str], max_items: int = 20) -> str:
    if not known_facts:
        return "(none)"
    return "\n".join(f"- {f}" for f in known_facts[:max_items])


def _confidence_status_text(fact_confidence: dict) -> str:
    if not fact_confidence:
        return "(no assessments yet)"
    lines = []
    for fact_key, level in list(fact_confidence.items())[:15]:
        lines.append(f"  [{level.upper()}] {fact_key}")
    return "\n".join(lines)


def _evidence_summary(evidence_items: list[str]) -> str:
    if not evidence_items:
        return "(none mentioned)"
    return "\n".join(f"- {e}" for e in evidence_items[:15])


# ===========================================================================
# Public: new_stage3_state — seeded from Stage 2
# ===========================================================================

def new_stage3_state(stage2_state: dict) -> dict:
    """
    Create a fresh Stage 3 state seeded from Stage 2's final state.
    Inherits all facts, evidence, coverage, and remedy.
    """
    state = copy.deepcopy(STAGE3_STATE_SCHEMA)
    state.update({
        "category":              stage2_state.get("category") or "general",
        "category_label":        stage2_state.get("category_label") or "General Matter",
        "jurisdiction":          stage2_state.get("jurisdiction") or "unknown",
        "urgency_signal":        stage2_state.get("urgency_signal") or "unknown",
        "client_role":           stage2_state.get("client_role") or "unknown",
        "other_party":           stage2_state.get("other_party") or "unknown",
        "known_facts":           list(stage2_state.get("known_facts") or []),
        "legal_element_coverage":dict(stage2_state.get("legal_element_coverage") or {}),
        "evidence_items":        list(stage2_state.get("evidence_items") or []),
        "remedy_stated":         stage2_state.get("remedy_stated"),
        # Seed confidence map: all facts start at "medium" until probed
        "fact_confidence": {
            f[:60].strip(): "medium"
            for f in (stage2_state.get("known_facts") or [])
        },
    })
    return state


# ===========================================================================
# Public: generate_stage3_opening
# ===========================================================================

def generate_stage3_opening(
    stage2_state: dict,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()
    prompt = (
        STAGE3_OPENING_SYSTEM
        .replace("{issue_summary}", (stage2_state.get("known_facts") or ["the situation"])[0][:200])
        .replace("{category_label}", stage2_state.get("category_label") or "your matter")
    )
    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("opening", t0)
        if reply and len(reply) > 20:
            return reply
    except Exception as exc:
        logger.warning("Stage3 opening LLM failed: %s", exc)
    return (
        "I now have a solid picture of what happened. Before I tell you what can be done, "
        "I want to make sure we have the kind of specific detail that will hold up — "
        "courts focus on particular things, and I want your account to be as strong as possible."
    )


# ===========================================================================
# Internal: run confidence assessor after client response
# ===========================================================================

def _assess_confidence(
    stage3_state: dict,
    client_message: str,
    conversation_context: str,
    model_override: str | None = None,
) -> dict:
    t0 = time.perf_counter()

    known_facts_text   = _facts_summary(stage3_state.get("known_facts") or [])
    existing_conf_text = _confidence_status_text(stage3_state.get("fact_confidence") or {})

    prompt = (
        STAGE3_CONFIDENCE_ASSESSOR_SYSTEM
        .replace("{category}", stage3_state.get("category") or "general")
        .replace("{client_role}", stage3_state.get("client_role") or "unknown")
        .replace("{known_facts_summary}", known_facts_text)
        .replace("{existing_confidence}", existing_conf_text)
        .replace("{client_message}", (client_message or "").strip())
        .replace("{conversation_context}", conversation_context or "(none)")
    )

    _DEFAULT = {
        "confidence_updates": {},
        "new_evidence_items": [],
        "timeline_issues": [],
        "remedy_confirmation": None,
        "account_strength": "moderate",
        "risk_factors": [],
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("assess_confidence", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage3 confidence assessor failed: %s", exc)

    return _DEFAULT


# ===========================================================================
# Internal: update state from confidence assessor output
# ===========================================================================

def _apply_confidence_update(stage3_state: dict, assessment: dict) -> None:
    """Merge confidence assessor output into stage3_state."""
    confidence = stage3_state.setdefault("fact_confidence", {})
    evidence   = stage3_state.setdefault("evidence_items", [])
    timeline   = stage3_state.setdefault("timeline_issues", [])
    risks      = stage3_state.setdefault("risk_factors", [])

    # Update confidence per fact
    for fact_key, level in (assessment.get("confidence_updates") or {}).items():
        if level in ("high", "medium", "uncertain"):
            confidence[str(fact_key)[:60].strip()] = level

    # Accumulate new evidence
    for ev in (assessment.get("new_evidence_items") or []):
        ev_str = str(ev).strip()
        if ev_str and ev_str not in evidence:
            evidence.append(ev_str)

    # Accumulate timeline issues
    for issue in (assessment.get("timeline_issues") or []):
        issue_str = str(issue).strip()
        if issue_str and issue_str not in timeline:
            timeline.append(issue_str)

    # Update overall strength
    strength = str(assessment.get("account_strength") or "moderate").strip().lower()
    if strength in ("strong", "moderate", "weak"):
        stage3_state["account_strength"] = strength

    # Accumulate risk factors
    for rf in (assessment.get("risk_factors") or []):
        rf_str = str(rf).strip()
        if rf_str and rf_str not in risks:
            risks.append(rf_str)

    # Update remedy
    remedy = assessment.get("remedy_confirmation")
    if remedy and not stage3_state.get("remedy_stated"):
        stage3_state["remedy_stated"] = str(remedy).strip()[:300]


# ===========================================================================
# Internal: build vetting report
# ===========================================================================

def _build_vetting_report(stage3_state: dict) -> dict:
    """
    Compile the vetting report from the current state.
    This is the artefact that Stage 4 (remedy) and Stage 6 (draft) consume.
    """
    confidence     = stage3_state.get("fact_confidence") or {}
    known_facts    = stage3_state.get("known_facts") or []
    evidence_items = stage3_state.get("evidence_items") or []
    coverage       = stage3_state.get("legal_element_coverage") or {}
    category       = stage3_state.get("category", "general")
    elements       = LEGAL_ELEMENT_CHECKLISTS.get(category, [])

    high    = [f for f in known_facts if confidence.get(f[:60].strip()) == "high"]
    medium  = [f for f in known_facts if confidence.get(f[:60].strip()) == "medium"]
    uncertain = [f for f in known_facts if confidence.get(f[:60].strip()) == "uncertain"]
    unassessed = [f for f in known_facts if f[:60].strip() not in confidence]

    # Evidence gaps: critical elements with no evidence mentioned
    evidence_gaps = []
    for el in elements:
        if el.get("priority") in ("critical", "important"):
            el_status = coverage.get(el["name"], "missing")
            if el_status in ("covered", "partial"):
                # Check if evidence was mentioned for this element
                el_desc_lower = el["description"].lower()
                has_ev = any(
                    any(word in ev.lower() for word in el_desc_lower.split()[:3])
                    for ev in evidence_items
                )
                if not has_ev:
                    note = el.get("evidentiary_note", "")
                    evidence_gaps.append(f"{el['name']}: {note or 'obtain supporting document'}")

    # Corroboration gaps: facts in uncertain bucket
    corroboration_gaps = [f[:80] for f in uncertain]

    # Draft language flags: uncertain + medium-unassessed facts get hedged
    draft_flags = [f[:80] for f in uncertain] + [f[:80] for f in unassessed[:3]]

    return {
        "high_confidence_facts":   [f[:120] for f in high],
        "medium_confidence_facts":  [f[:120] for f in medium],
        "uncertain_facts":          [f[:120] for f in uncertain],
        "evidence_gaps":            evidence_gaps[:10],
        "corroboration_gaps":       corroboration_gaps[:8],
        "draft_language_flags":     draft_flags[:10],
    }


# ===========================================================================
# Internal: generate vetting question
# ===========================================================================

def _generate_vetting_question(
    stage3_state: dict,
    client_message: str,
    conversation_context: str,
    technique: str,
    target_fact: str,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()

    known_facts_text   = _facts_summary(stage3_state.get("known_facts") or [])
    evidence_text      = _evidence_summary(stage3_state.get("evidence_items") or [])
    confidence_text    = _confidence_status_text(stage3_state.get("fact_confidence") or {})

    prompt = (
        STAGE3_VETTING_QUESTION_SYSTEM
        .replace("{category_label}", stage3_state.get("category_label") or "the matter")
        .replace("{client_role}", stage3_state.get("client_role") or "unknown")
        .replace("{known_facts_summary}", known_facts_text)
        .replace("{evidence_items_summary}", evidence_text)
        .replace("{confidence_status}", confidence_text)
        .replace("{target_fact}", target_fact)
        .replace("{technique}", technique)
        .replace("{conversation_context}", conversation_context or "(first vetting turn)")
        .replace("{client_message}", (client_message or "").strip())
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("generate_question", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage3 question generation failed: %s", exc)

    # Fallback questions per technique
    fallbacks = {
        "triangulation":  "Was anyone else there at the time — family, neighbours, anyone who saw or heard what happened?",
        "doc_anchoring":  "Do you have any messages, photographs, or medical records from around that time? Even partial records can help.",
        "timeline_probe": "Help me get the sequence right — what happened immediately after that?",
        "restatement":    "Just to make sure I have this right — can you confirm the key events as you described them?",
    }
    return fallbacks.get(technique, "Can you tell me a little more about what happened?")


# ===========================================================================
# Internal: completeness check
# ===========================================================================

def _check_completeness(
    stage3_state: dict,
    conversation_context: str,
    model_override: str | None = None,
) -> tuple[bool, list[str]]:
    t0 = time.perf_counter()

    # Fast heuristic
    if stage3_state.get("turn_count", 0) < 2:
        return False, ["need at least 2 vetting turns before advancing"]

    state_json = json.dumps(stage3_state, ensure_ascii=False, indent=2)
    prompt = (
        STAGE3_COMPLETENESS_CHECK_SYSTEM
        .replace("{category_label}", stage3_state.get("category_label") or "the matter")
        .replace("{stage3_state_json}", state_json)
    )

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("check_completeness", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            ready  = bool(out.get("ready_for_stage4", False))
            gaps   = [str(g).strip() for g in (out.get("unchecked_critical_facts") or []) if str(g).strip()]
            return ready, gaps
    except Exception as exc:
        logger.warning("Stage3 completeness check failed: %s", exc)

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
    Process one client turn in Stage 3 indirect vetting.

    Parameters
    ----------
    session : dict
        Must contain:
          - "history"       : list of {role, content} turns
          - "stage3_state"  : running Stage 3 state (or None for first turn)
    user_message : str
    model_override : str | None

    Returns
    -------
    dict:
      - "reply"              : str
      - "stage3_state"       : dict  — updated
      - "advance_to_stage4"  : bool
      - "urgency_signal"     : str
    """
    msg = (user_message or "").strip()

    # ── Load state ───────────────────────────────────────────────────────────
    stage3_state: dict = session.get("stage3_state") or {}
    if not stage3_state.get("category"):
        logger.warning("Stage3 process_turn called without initialised stage3_state")
        stage3_state = copy.deepcopy(STAGE3_STATE_SCHEMA)
        stage3_state["category"] = "general"
        stage3_state["category_label"] = "General Matter"

    stage3_state["turn_count"] = stage3_state.get("turn_count", 0) + 1

    # ── Build conversation context ────────────────────────────────────────────
    context = _build_context(session, max_turns=10)

    # ── Urgency recheck ───────────────────────────────────────────────────────
    _recheck_urgency(stage3_state, msg)

    # ── Assess confidence from this response ──────────────────────────────────
    assessment = _assess_confidence(stage3_state, msg, context, model_override=model_override)
    _apply_confidence_update(stage3_state, assessment)

    # ── Update vetting report ─────────────────────────────────────────────────
    stage3_state["vetting_report"] = _build_vetting_report(stage3_state)

    # ── Choose technique + target for next question ───────────────────────────
    technique   = _choose_technique(stage3_state)
    target_fact = _choose_target_fact(stage3_state)

    # ── Record technique as used ──────────────────────────────────────────────
    if technique not in (stage3_state.get("techniques_used") or []):
        stage3_state.setdefault("techniques_used", []).append(technique)

    # ── Generate the question ─────────────────────────────────────────────────
    reply = _generate_vetting_question(
        stage3_state, msg, context, technique, target_fact,
        model_override=model_override,
    )

    # ── Completeness check ────────────────────────────────────────────────────
    ready, gaps = _check_completeness(stage3_state, context, model_override=model_override)
    stage3_state["ready_for_stage4"] = ready

    logger.info(
        "Stage3 turn %d | category=%s | strength=%s | technique=%s | ready=%s | urgency=%s",
        stage3_state["turn_count"],
        stage3_state.get("category"),
        stage3_state.get("account_strength"),
        technique,
        ready,
        stage3_state.get("urgency_signal"),
    )

    return {
        "reply":             reply,
        "stage3_state":      stage3_state,
        "advance_to_stage4": ready,
        "urgency_signal":    stage3_state.get("urgency_signal", "unknown"),
    }
