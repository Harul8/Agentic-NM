"""
Legal Opinion Intake — Stage 4: Remedy Understanding + Practical Assessment

Handles the fourth phase of the autonomous intake workflow.

Stage 4 is the transition from information-gathering to advice-giving.
The AI now tells the client what the law can actually do for them —
honestly, empathetically, practically.

Two sub-phases:
  4a (opening)  — AI runs a full remedy analysis using all gathered data,
                   then presents the honest picture to the client in plain
                   language.  Asks: "Is this the direction you want to go?"
  4b (confirm)  — Client responds. AI extracts their confirmed remedy
                   choices, addresses any concerns, finalises the remedy
                   plan, and advances to Stage 5 (draft).

Key output: stage4_state["remedy_plan"]
  Consumed by Stage 5 to structure and draft the legal document.

Public API
----------
  generate_stage4_opening(stage3_state, model_override)  → str
      Runs analysis + returns the presentation message.
  process_turn(session, user_message, model_override)     → dict
      Returns: {reply, stage4_state, advance_to_stage5, urgency_signal}
  new_stage4_state(stage3_state)                          → dict
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
    LEGAL_REMEDY_MAPS,
    LEGAL_ISSUE_CATEGORIES,
    STAGE4_REMEDY_ANALYSIS_SYSTEM,
    STAGE4_PRESENTATION_SYSTEM,
    STAGE4_CONFIRMATION_EXTRACTOR_SYSTEM,
    STAGE4_STATE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Debug timing
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage4.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


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
    "beaten", "hit", "assault", "arrested", "thrown out", "evict",
    "threatened", "danger", "hospital", "injury", "court today", "deadline today",
)


def _recheck_urgency(state: dict, msg: str) -> None:
    if state.get("urgency_signal") == "immediate":
        return
    if any(sig in (msg or "").lower() for sig in _IMMEDIATE_SIGNALS):
        state["urgency_signal"] = "immediate"


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _coverage_summary_text(coverage: dict) -> str:
    if not coverage:
        return "(no element coverage data)"
    lines = []
    for name, status in coverage.items():
        lines.append(f"  {name}: {status}")
    return "\n".join(lines)


def _vetting_highlights_text(vetting_report: dict) -> str:
    if not vetting_report:
        return "(no vetting data)"
    parts = []
    high  = vetting_report.get("high_confidence_facts") or []
    med   = vetting_report.get("medium_confidence_facts") or []
    unc   = vetting_report.get("uncertain_facts") or []
    gaps  = vetting_report.get("evidence_gaps") or []
    if high:
        parts.append("High-confidence facts:\n" + "\n".join(f"  - {f}" for f in high[:5]))
    if unc:
        parts.append("Uncertain facts (need hedging in draft):\n" + "\n".join(f"  - {f}" for f in unc[:5]))
    if gaps:
        parts.append("Evidence gaps:\n" + "\n".join(f"  - {g}" for g in gaps[:5]))
    return "\n".join(parts) or "(no highlights)"


def _remedy_options_text(category: str) -> str:
    options = LEGAL_REMEDY_MAPS.get(category, LEGAL_REMEDY_MAPS.get("general", []))
    lines = []
    for r in options:
        lines.append(
            f"- {r['name']} | {r['basis']} | {r['forum']} | {r['timeline']} | "
            f"priority:{r['priority']} | requires:{r['requires']}"
        )
    return "\n".join(lines) or "(see applicable law)"


# ===========================================================================
# Public: new_stage4_state
# ===========================================================================

def new_stage4_state(stage3_state: dict) -> dict:
    """Create a fresh Stage 4 state seeded from Stage 3's final state."""
    state = copy.deepcopy(STAGE4_STATE_SCHEMA)
    state.update({
        "category":              stage3_state.get("category") or "general",
        "category_label":        stage3_state.get("category_label") or "General Matter",
        "jurisdiction":          stage3_state.get("jurisdiction") or "unknown",
        "urgency_signal":        stage3_state.get("urgency_signal") or "unknown",
        "client_role":           stage3_state.get("client_role") or "unknown",
        "other_party":           stage3_state.get("other_party") or "unknown",
        "known_facts":           list(stage3_state.get("known_facts") or []),
        "legal_element_coverage":dict(stage3_state.get("legal_element_coverage") or {}),
        "evidence_items":        list(stage3_state.get("evidence_items") or []),
        "vetting_report":        dict(stage3_state.get("vetting_report") or {}),
        "account_strength":      stage3_state.get("account_strength") or "moderate",
        "risk_factors":          list(stage3_state.get("risk_factors") or []),
        "remedy_plan": {
            **state["remedy_plan"],
            # Seed remedy from Stage 3 if confirmed there
            "lead_remedy": stage3_state.get("remedy_stated"),
        },
    })
    return state


# ===========================================================================
# Internal: run remedy analysis
# ===========================================================================

def _run_remedy_analysis(
    stage4_state: dict,
    model_override: str | None = None,
) -> dict:
    """
    Run the full remedy analysis LLM call.
    Returns the structured analysis dict.
    """
    t0 = time.perf_counter()

    category  = stage4_state.get("category", "general")
    vr        = stage4_state.get("vetting_report") or {}
    known_facts = stage4_state.get("known_facts") or []

    # Build a compact issue summary from the first few known facts
    issue_summary = "; ".join(known_facts[:3]) or "(see context)"

    prompt = (
        STAGE4_REMEDY_ANALYSIS_SYSTEM
        .replace("{category_label}",  stage4_state.get("category_label") or category)
        .replace("{client_role}",     stage4_state.get("client_role") or "unknown")
        .replace("{jurisdiction}",    stage4_state.get("jurisdiction") or "unknown")
        .replace("{urgency_signal}",  stage4_state.get("urgency_signal") or "unknown")
        .replace("{coverage_summary}", _coverage_summary_text(stage4_state.get("legal_element_coverage") or {}))
        .replace("{account_strength}", stage4_state.get("account_strength") or "moderate")
        .replace("{vetting_highlights}", _vetting_highlights_text(vr))
        .replace("{remedy_stated}",   str(stage4_state.get("remedy_plan", {}).get("lead_remedy") or "not yet stated"))
        .replace("{remedy_options}",  _remedy_options_text(category))
    )

    _DEFAULT = {
        "legal_basis_strength": "moderate",
        "viable_remedies": [],
        "gap_with_client_ask": None,
        "recommended_strategy": "Further analysis is needed.",
        "immediate_actions": [],
        "honest_caveat": None,
        "timeline_summary": "Timeline depends on the remedies pursued.",
    }

    try:
        raw = _ask_llm(prompt, task_hint="quality", model_override=model_override)
        _t("remedy_analysis", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage4 remedy analysis failed: %s", exc)

    return _DEFAULT


# ===========================================================================
# Internal: generate the client-facing presentation from the analysis
# ===========================================================================

def _generate_presentation(
    stage4_state: dict,
    analysis: dict,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()

    known_facts = stage4_state.get("known_facts") or []
    issue_summary = "; ".join(known_facts[:3]) or "(situation described)"
    remedy_stated = stage4_state.get("remedy_plan", {}).get("lead_remedy") or "not specified"

    prompt = (
        STAGE4_PRESENTATION_SYSTEM
        .replace("{analysis_json}",   json.dumps(analysis, ensure_ascii=False, indent=2))
        .replace("{issue_summary}",   issue_summary[:300])
        .replace("{remedy_stated}",   remedy_stated)
        .replace("{urgency_signal}",  stage4_state.get("urgency_signal") or "unknown")
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("generate_presentation", t0)
        if reply and len(reply) > 40:
            return reply
    except Exception as exc:
        logger.warning("Stage4 presentation generation failed: %s", exc)

    # Structured fallback using the analysis data
    viable = analysis.get("viable_remedies") or []
    lead = viable[0]["name"] if viable else "the available legal remedy"
    timeline = viable[0].get("timeline", "some months") if viable else "some time"
    strategy = analysis.get("recommended_strategy") or ""
    return (
        f"Based on everything you've shared, {strategy or 'here is what I recommend'}. "
        f"The strongest immediate step is {lead}, which typically takes {timeline}. "
        "Does this direction make sense to you, and is it what you'd like to pursue?"
    )


# ===========================================================================
# Internal: build the remedy plan from analysis + confirmation
# ===========================================================================

def _build_remedy_plan(stage4_state: dict, analysis: dict, confirmed_remedies: list[str]) -> dict:
    viable = analysis.get("viable_remedies") or []
    lead_candidates = [r for r in viable if r.get("priority") == "lead"]
    supporting = [r["name"] for r in viable if r.get("priority") == "supporting"]

    lead = None
    if confirmed_remedies:
        # Match confirmed choice to a viable remedy
        for r in viable:
            if any(cr.lower() in r["name"].lower() or r["name"].lower() in cr.lower() for cr in confirmed_remedies):
                lead = r["name"]
                break
    if not lead and lead_candidates:
        lead = lead_candidates[0]["name"]
    elif not lead and viable:
        lead = viable[0]["name"]

    return {
        "lead_remedy":         lead,
        "supporting_remedies": supporting,
        "immediate_actions":   analysis.get("immediate_actions") or [],
        "strategy_note":       analysis.get("recommended_strategy"),
        "timeline_summary":    analysis.get("timeline_summary"),
        "honest_caveat":       analysis.get("honest_caveat"),
        "client_confirmed":    bool(confirmed_remedies),
    }


# ===========================================================================
# Internal: extract client confirmation
# ===========================================================================

def _extract_confirmation(
    stage4_state: dict,
    client_message: str,
    conversation_context: str,
    model_override: str | None = None,
) -> dict:
    t0 = time.perf_counter()

    remedies_presented = stage4_state.get("remedies_presented") or []
    remedies_text = "\n".join(f"- {r}" for r in remedies_presented) or "(remedies presented in conversation)"

    prompt = (
        STAGE4_CONFIRMATION_EXTRACTOR_SYSTEM
        .replace("{remedies_presented}",  remedies_text)
        .replace("{client_message}",      (client_message or "").strip())
        .replace("{conversation_context}", conversation_context or "(none)")
    )

    _DEFAULT = {
        "confirmed_remedies": [],
        "additional_asks": [],
        "concerns_raised": [],
        "ready_to_proceed": False,
        "clarification_needed": None,
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("extract_confirmation", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage4 confirmation extraction failed: %s", exc)

    # Heuristic fallback: positive words → ready
    positive = ("yes", "okay", "ok", "agree", "proceed", "go ahead", "sounds right",
                 "that's right", "correct", "confirm", "please proceed", "yes please")
    low = (client_message or "").lower().strip()
    if any(p in low for p in positive):
        _DEFAULT["ready_to_proceed"] = True
        _DEFAULT["confirmed_remedies"] = list(remedies_presented[:2])

    return _DEFAULT


# ===========================================================================
# Internal: generate follow-up if clarification is needed
# ===========================================================================

def _generate_followup_or_close(
    stage4_state: dict,
    confirmation: dict,
    model_override: str | None = None,
) -> str:
    """Generate either a clarification question or a closing acknowledgement."""
    clarification = confirmation.get("clarification_needed")
    concerns = confirmation.get("concerns_raised") or []
    additional = confirmation.get("additional_asks") or []

    if clarification:
        prompt = (
            f"A legal advice client needs a brief clarification on one point before we proceed to drafting. "
            f"Ask this clearly and warmly in one sentence: {clarification}\n"
            f"Output ONLY the question."
        )
        try:
            q = _ask_llm_quality(prompt, model_override=model_override).strip()
            if q and "?" in q:
                return q
        except Exception:
            pass
        return f"Before we move forward — {clarification}"

    if concerns:
        concern_text = concerns[0]
        prompt = (
            f"A legal advice client raised a concern: '{concern_text}'. "
            f"Respond in 2–3 sentences: acknowledge the concern honestly, briefly address it, "
            f"then confirm you're ready to prepare their legal documents. "
            f"No jargon, no Act names. Output ONLY the response."
        )
        try:
            r = _ask_llm_quality(prompt, model_override=model_override).strip()
            if r and len(r) > 20:
                return r
        except Exception:
            pass

    if additional:
        extra = additional[0]
        prompt = (
            f"A legal advice client added a new ask: '{extra}'. "
            f"Acknowledge it briefly and say you'll include it in the documents you're preparing. "
            f"Under 40 words. No jargon. Output ONLY the response."
        )
        try:
            r = _ask_llm_quality(prompt, model_override=model_override).strip()
            if r and len(r) > 10:
                return r
        except Exception:
            pass

    # Default closing
    return (
        "Understood. I'll now prepare the documents for you — "
        "pulling together the relevant legal protections and the case law that applies to your situation."
    )


# ===========================================================================
# Public: generate_stage4_opening
# ===========================================================================

def generate_stage4_opening(
    stage3_state: dict,
    model_override: str | None = None,
) -> tuple[str, dict]:
    """
    Run the full remedy analysis and generate the opening presentation.

    Returns
    -------
    (opening_message: str, analysis: dict)
      The analysis dict is stored in stage4_state so it isn't rerun later.
    """
    # Build a temporary stage4_state to drive the analysis
    s4 = new_stage4_state(stage3_state)
    analysis = _run_remedy_analysis(s4, model_override=model_override)
    presentation = _generate_presentation(s4, analysis, model_override=model_override)
    return presentation, analysis


# ===========================================================================
# Public: process_turn
# ===========================================================================

def process_turn(
    session: dict,
    user_message: str,
    model_override: str | None = None,
) -> dict:
    """
    Process one client turn in Stage 4 remedy confirmation.

    Parameters
    ----------
    session : dict
        Must contain:
          - "history"       : list of {role, content}
          - "stage4_state"  : running Stage 4 state (or None for first turn)
    user_message : str
    model_override : str | None

    Returns
    -------
    dict:
      - "reply"              : str
      - "stage4_state"       : dict
      - "advance_to_stage5"  : bool
      - "urgency_signal"     : str
    """
    msg = (user_message or "").strip()

    stage4_state: dict = session.get("stage4_state") or {}
    if not stage4_state.get("category"):
        logger.warning("Stage4 process_turn called without initialised stage4_state")
        stage4_state = copy.deepcopy(STAGE4_STATE_SCHEMA)
        stage4_state["category"] = "general"

    stage4_state["turn_count"] = stage4_state.get("turn_count", 0) + 1
    context = _build_context(session, max_turns=10)
    _recheck_urgency(stage4_state, msg)

    # ── Extract client confirmation ───────────────────────────────────────────
    confirmation = _extract_confirmation(stage4_state, msg, context, model_override=model_override)

    # Update state with confirmation data
    confirmed = [str(r).strip() for r in (confirmation.get("confirmed_remedies") or []) if str(r).strip()]
    for c in confirmed:
        if c not in stage4_state.get("confirmed_remedies", []):
            stage4_state.setdefault("confirmed_remedies", []).append(c)

    additional = [str(a).strip() for a in (confirmation.get("additional_asks") or []) if str(a).strip()]
    for a in additional:
        if a not in stage4_state.get("additional_asks", []):
            stage4_state.setdefault("additional_asks", []).append(a)

    concerns = [str(c).strip() for c in (confirmation.get("concerns_raised") or []) if str(c).strip()]
    for c in concerns:
        if c not in stage4_state.get("concerns_raised", []):
            stage4_state.setdefault("concerns_raised", []).append(c)

    # ── Decide if ready to advance ────────────────────────────────────────────
    ready = bool(confirmation.get("ready_to_proceed")) and not confirmation.get("clarification_needed")

    # Also advance if we've had 2+ turns with a confirmed remedy even without explicit "ready"
    if stage4_state.get("turn_count", 0) >= 2 and stage4_state.get("confirmed_remedies"):
        ready = True

    # ── Build or finalise the remedy plan ─────────────────────────────────────
    analysis = stage4_state.get("remedy_analysis") or {}
    if confirmed or ready:
        remedy_plan = _build_remedy_plan(
            stage4_state,
            analysis,
            stage4_state.get("confirmed_remedies", []),
        )
        stage4_state["remedy_plan"] = remedy_plan

    stage4_state["ready_for_stage5"] = ready

    # ── Generate reply ────────────────────────────────────────────────────────
    if ready:
        reply = _generate_followup_or_close(stage4_state, confirmation, model_override=model_override)
    else:
        reply = _generate_followup_or_close(stage4_state, confirmation, model_override=model_override)

    logger.info(
        "Stage4 turn %d | category=%s | confirmed=%s | ready=%s | urgency=%s",
        stage4_state["turn_count"],
        stage4_state.get("category"),
        stage4_state.get("confirmed_remedies"),
        ready,
        stage4_state.get("urgency_signal"),
    )

    return {
        "reply":             reply,
        "stage4_state":      stage4_state,
        "advance_to_stage5": ready,
        "urgency_signal":    stage4_state.get("urgency_signal", "unknown"),
    }
