"""
intake/stage1_opening.py — Stage 1: Opening + issue identification.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — avoid circular imports at module load time
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "fast") -> str:
    from core.llm import ask_llm
    return ask_llm(prompt, task_hint=task_hint) or ""


def _ask_llm_quality(prompt: str) -> str:
    return _ask_llm(prompt, task_hint="quality")


# ---------------------------------------------------------------------------
# Prompt imports
# ---------------------------------------------------------------------------

from prompts.intake import (
    LEGAL_OPINION_OPENING_SYSTEM,
    ISSUE_CATEGORY_DETECT_SYSTEM,
    STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM,
    STAGE1_READINESS_CHECK_SYSTEM,
    STAGE1_SAFETY_FIRST_SYSTEM,
    STAGE1_URGENCY_RECHECK_SYSTEM,
    STAGE1_URGENCY_FROM_HISTORY_SYSTEM,
    STAGE1_VETTING_QUESTION_SYSTEM,
    STAGE4_REMEDY_ASSESSMENT_SYSTEM,
    PRE_DRAFT_SUMMARY_SYSTEM,
    STAGE1_INTAKE_STATE_SCHEMA,
    LEGAL_ISSUE_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Pipeline timing (optional debug)
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage1.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text that may contain markdown / reasoning."""
    text = (text or "").strip()
    # Strip markdown fences
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
# Conversation context builder (compact, for prompt injection)
# ---------------------------------------------------------------------------

# Characters above which a turn is treated as a document upload rather than
# a typed message.  800 chars covers the longest realistic typed message;
# anything above this threshold is almost certainly pasted/extracted document text.
_DOC_TURN_THRESHOLD = 800

# Max chars to keep from a normal conversational turn
_CONV_TURN_MAX_CHARS = 800

# Max chars of the *legal summary* carried forward for document turns.
# The summary is generated once and cached; raw document text is never
# truncated mid-sentence into the conversation context.
_DOC_SUMMARY_MAX_CHARS = 500


def _summarize_doc_for_intake(raw_text: str) -> str:
    """
    Extract legally relevant facts from an uploaded document (FIR, legal notice,
    medical certificate, WhatsApp export, property deed, etc.) into a compact
    summary suitable for the intake conversation context.

    Summaries are cached by content hash in session["_doc_summaries"] so the
    LLM call only happens once per unique document, regardless of how many
    subsequent turns are built from that session.
    """
    prompt = (
        "A client uploaded a document during a legal intake. "
        "Extract ONLY the legally relevant facts in a single compact paragraph:\n"
        "- Key dates, events, parties named\n"
        "- Amounts, penalties, legal claims or charges mentioned\n"
        "- Any prior actions taken (complaints filed, notices served, orders issued, bail/custody details)\n"
        "- Signatures, stamps, or official references that establish authenticity\n\n"
        "Ignore boilerplate, headers, footers, and irrelevant text.\n"
        "Output: one paragraph, plain English, under 120 words.\n\n"
        "DOCUMENT:\n"
        f"{raw_text[:6000]}"          # cap raw input to avoid runaway cost on huge docs
    )
    try:
        summary = _ask_llm(prompt, task_hint="fast").strip()
        return summary[:_DOC_SUMMARY_MAX_CHARS] if len(summary) > _DOC_SUMMARY_MAX_CHARS else summary
    except Exception as exc:
        logger.warning("Doc summarisation failed: %s", exc)
        # Graceful fallback: return the first _DOC_SUMMARY_MAX_CHARS of the raw text
        return raw_text[:_DOC_SUMMARY_MAX_CHARS].rstrip() + "…"


def _build_context(session: dict, max_turns: int = 30) -> str:
    """Build a conversation history block for prompt injection.

    Two separate content limits apply:
    - Normal conversational turns  → up to _CONV_TURN_MAX_CHARS (800)
    - Document turns (content > _DOC_TURN_THRESHOLD) → replaced by a
      legally-focused summary (_DOC_SUMMARY_MAX_CHARS = 500 chars),
      generated once and cached in session["_doc_summaries"]

    30 turns covers any realistic intake (typical intakes are 8–15 turns).
    Document summaries are cached by content hash to avoid re-summarising
    the same file on every subsequent turn.
    """
    history = session.get("history", [])
    doc_cache: dict = session.setdefault("_doc_summaries", {})  # hash → summary

    lines: list[str] = []
    for turn in history[-max_turns:]:
        role    = "Client" if turn.get("role") == "user" else "Counsel"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue

        if len(content) > _DOC_TURN_THRESHOLD:
            # Document turn — use cached summary or generate one
            import hashlib
            key = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
            if key not in doc_cache:
                doc_cache[key] = _summarize_doc_for_intake(content)
            summary = doc_cache[key]
            lines.append(f"{role} [uploaded document]: {summary}")
        else:
            if len(content) > _CONV_TURN_MAX_CHARS:
                content = content[:_CONV_TURN_MAX_CHARS].rstrip() + "…"
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fresh intake state factory
# ---------------------------------------------------------------------------

def _fresh_state() -> dict:
    return copy.deepcopy(STAGE1_INTAKE_STATE_SCHEMA)


# ===========================================================================
# Public: generate_opening
# ===========================================================================

def generate_opening() -> str:
    """
    Generate the AI's very first message to a new client.

    Always LLM-generated for warmth and variation.
    Falls back to a safe static message if the LLM fails.
    """
    t0 = time.perf_counter()
    try:
        reply = _ask_llm(LEGAL_OPINION_OPENING_SYSTEM, task_hint="quality").strip()
        _t("generate_opening", t0)
        if reply and len(reply) > 20:
            return reply
    except Exception as exc:
        logger.warning("Stage1 opening LLM failed: %s", exc)

    # Static fallback — warm but not robotic
    return (
        "Whatever you're going through, I'm here and I'm listening. "
        "Take your time and share what's on your mind — there's no wrong way to start."
    )


# ===========================================================================
# Internal: category detection
# ===========================================================================

def _detect_category(client_message: str, conversation_context: str) -> dict:
    """
    Run the LLM category classifier on the client's message.

    Returns a dict with primary_category, secondary_categories, and all
    new anchor / risk fields.  Falls back to safe defaults on failure.
    """
    t0 = time.perf_counter()

    prompt = (
        ISSUE_CATEGORY_DETECT_SYSTEM
        + "\n\nCONVERSATION SO FAR:\n"
        + (conversation_context or "(first message)")
        + "\n\nCLIENT MESSAGE:\n"
        + (client_message or "").strip()
    )

    _DEFAULT = {
        "primary_category":          "general",
        "secondary_categories":      [],
        "confidence":                "low",
        "issue_summary":             "",
        "jurisdiction":              "unknown",
        "urgency_signal":            "unknown",
        "risk_flags":                [],
        "client_role":               "unknown",
        "other_party":               "unknown",
        "relationship_to_other_party": "unknown",
        "timeframe_status":          "unknown",
        "client_goal_initial":       None,
        "immediate_need":            "unknown",
        "emotional_ask":             "unknown",
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("detect_category", t0)
        out = _extract_json(raw)
        if not out or not isinstance(out, dict):
            return _DEFAULT

        valid_cats = set(LEGAL_ISSUE_CATEGORIES.keys())

        # Primary category
        primary = str(out.get("primary_category") or "general").strip().lower()
        if primary not in valid_cats:
            primary = "general"

        # Secondary categories (up to 2, valid only)
        raw_secondary = out.get("secondary_categories") or []
        secondary = [
            c.strip().lower() for c in raw_secondary
            if isinstance(c, str) and c.strip().lower() in valid_cats and c.strip().lower() != primary
        ][:2]

        confidence = str(out.get("confidence") or "low").strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        urgency = str(out.get("urgency_signal") or "unknown").strip().lower()
        if urgency not in ("immediate", "near_term", "no_urgency", "unknown"):
            urgency = "unknown"

        risk_flags = [str(f).strip() for f in (out.get("risk_flags") or []) if str(f).strip()]

        timeframe = str(out.get("timeframe_status") or "unknown").strip().lower()
        if timeframe not in ("ongoing", "recent", "historical", "unknown"):
            timeframe = "unknown"

        immediate_need = str(out.get("immediate_need") or "unknown").strip().lower()
        if immediate_need not in ("safety", "shelter", "protection_order", "bail", "stay_order", "money", "none", "unknown"):
            immediate_need = "unknown"

        client_goal = str(out.get("client_goal_initial") or "").strip()[:200] or None

        return {
            "primary_category":          primary,
            "secondary_categories":      secondary,
            "confidence":                confidence,
            "issue_summary":             str(out.get("issue_summary") or "").strip()[:300],
            "jurisdiction":              str(out.get("jurisdiction_hint") or "unknown").strip(),
            "urgency_signal":            urgency,
            "risk_flags":                risk_flags,
            "client_role":               str(out.get("client_role") or "unknown").strip(),
            "other_party":               str(out.get("other_party") or "unknown").strip()[:120],
            "relationship_to_other_party": str(out.get("relationship_to_other_party") or "unknown").strip(),
            "timeframe_status":          timeframe,
            "client_goal_initial":       client_goal,
            "immediate_need":            immediate_need,
            "emotional_ask":             str(out.get("emotional_ask") or "unknown").strip(),
        }

    except Exception as exc:
        logger.warning("Stage1 category detection failed: %s", exc)
        return _DEFAULT


# ===========================================================================
# Internal: determine next question hint
# ===========================================================================



# ===========================================================================
# Internal: remedy assessment (Stage 4)
# ===========================================================================

def _assess_remedy(intake_state: dict) -> None:
    """
    Populate stated_remedy and assessed_remedy when client_goal_initial is known
    and assessed_remedy has not yet been set.  Runs silently in the background —
    does not block the reply generation path.
    """
    if intake_state.get("assessed_remedy"):
        return  # Already done
    goal = intake_state.get("client_goal_initial") or ""
    if not goal:
        return

    # stated_remedy = what the client said
    intake_state["stated_remedy"] = goal

    facts_list = intake_state.get("known_facts") or []
    facts_blob = "; ".join([
        (f.get("fact", "") if isinstance(f, dict) else str(f))
        for f in facts_list[:8]
    ])

    prompt = (
        STAGE4_REMEDY_ASSESSMENT_SYSTEM
        .replace("{primary_category}", intake_state.get("primary_issue_cluster") or "general")
        .replace("{facts_summary}", facts_blob[:600] or "(facts not yet gathered)")
        .replace("{stated_remedy}", goal[:200])
        .replace("{urgency_signal}", intake_state.get("urgency_signal") or "unknown")
    )

    t0 = time.perf_counter()
    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("assess_remedy", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            intake_state["stated_remedy"]  = str(out.get("stated_remedy") or goal).strip()[:300]
            intake_state["assessed_remedy"] = str(out.get("assessed_remedy") or "").strip()[:400] or None
            # Store extra remedy signals in a sub-dict for Stage 5 / advocate review
            intake_state.setdefault("remedy_detail", {}).update({
                "faster_alternative":  out.get("faster_alternative"),
                "remedy_gap":          out.get("remedy_gap"),
                "recommended_lead":    out.get("recommended_lead"),
            })
    except Exception as exc:
        logger.warning("Stage1 remedy assessment failed: %s", exc)


# ===========================================================================
# Internal: indirect vetting question for uncertain facts
# ===========================================================================

def _get_vetting_question(intake_state: dict, conversation_context: str) -> str | None:
    """
    If any known_facts entry has confidence_seed=='uncertain', generate one
    indirect triangulating question to strengthen that fact.

    Returns the question string, or None if no uncertain facts exist.
    Only asks about the FIRST uncertain fact found (one vetting question per turn).
    """
    uncertain = [
        f for f in (intake_state.get("known_facts") or [])
        if isinstance(f, dict) and f.get("confidence_seed") == "uncertain"
    ]
    if not uncertain:
        return None

    target = uncertain[0]
    t0 = time.perf_counter()
    prompt = (
        STAGE1_VETTING_QUESTION_SYSTEM
        .replace("{uncertain_fact}", str(target.get("fact", "")).strip())
        .replace("{fact_type}", str(target.get("fact_type", "other")).strip())
        .replace("{conversation_context}", conversation_context or "(no prior context)")
    )
    try:
        reply = _ask_llm_quality(prompt).strip()
        _t("get_vetting_question", t0)
        if reply and len(reply) > 15 and "?" in reply:
            # Mark the fact as vetting_scheduled so we don't re-ask next turn
            target["confidence_seed"] = "medium_vetting_asked"
            return reply
    except Exception as exc:
        logger.warning("Stage1 vetting question LLM failed: %s", exc)
    return None


# ===========================================================================
# Internal: safety-first response (urgency_signal=immediate)
# ===========================================================================

def _generate_safety_first_response(intake_state: dict) -> str:
    """Generate an immediate safety-focused reply when risk_flags are present."""
    t0 = time.perf_counter()
    risk_flags  = intake_state.get("risk_flags") or []
    immediate   = intake_state.get("immediate_need") or "unknown"
    summary     = intake_state.get("issue_summary") or "the client's urgent situation"

    prompt = (
        STAGE1_SAFETY_FIRST_SYSTEM
        .replace("{risk_flags}", ", ".join(risk_flags) if risk_flags else "immediate distress")
        .replace("{immediate_need}", immediate)
        .replace("{issue_summary}", summary)
    )

    try:
        reply = _ask_llm_quality(prompt).strip()
        _t("generate_safety_first", t0)
        if reply and len(reply) > 20:
            return reply
    except Exception as exc:
        logger.warning("Stage1 safety-first LLM failed: %s", exc)

    # Deterministic fallback for physical danger
    return (
        "I hear you — this sounds serious and I want to make sure you are safe right now. "
        "If you are in immediate physical danger, please call 112. "
        "Can you tell me where you are and whether you have somewhere safe to go tonight?"
    )


# ===========================================================================
# Internal: confirm and follow up
# ===========================================================================

def _generate_followup(
    intake_state: dict,
    client_message: str,
    conversation_context: str,
) -> str:
    """
    Generate the AI's reply after the category is detected.

    Passes the full conversation history, the matter category, and a compact
    summary of anchor facts already established.  The model uses its own
    professional judgment to decide what to ask next — no hint injection,
    no topic checklist override.
    """
    t0 = time.perf_counter()

    # Human-readable category label for the prompt
    primary = intake_state.get("primary_issue_cluster") or "general"
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(primary, {})
    category_label = cat_cfg.get("label", primary.replace("_", " ").title())

    # Compact summary of what's already been established — prevents re-asking
    # anchor facts the model has already noted, without prescribing what to ask next
    anchors = {
        "issue":        intake_state.get("issue_summary"),
        "relationship": intake_state.get("relationship_to_other_party"),
        "timeframe":    intake_state.get("timeframe_status"),
        "goal":         intake_state.get("client_goal_initial"),
        "urgency":      intake_state.get("urgency_signal"),
    }
    established_parts = [
        f"{k}: {v}" for k, v in anchors.items()
        if v and v not in ("unknown", "")
    ]
    established_facts = "; ".join(established_parts) if established_parts else "none yet — first exchange"

    prompt = (
        STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM
        .replace("{category}", category_label)
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
        .replace("{established_facts}", established_facts)
    )

    try:
        reply = _ask_llm_quality(prompt).strip()
        _t("generate_followup", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage1 followup LLM failed: %s", exc)

    return "Could you tell me a bit more about what's been happening?"


# ===========================================================================
# Internal: Stage 1 readiness check
# ===========================================================================

def _check_readiness(intake_state: dict, conversation_context: str) -> tuple[bool, list[str]]:
    """
    Deterministic readiness gate: Stage 2 opens only when all four anchor
    fields are populated (non-null, non-'unknown').

    Falls back to an LLM call only to populate missing_critical labels for
    the caller to surface as the next question — the ready/not-ready decision
    itself is always deterministic.

    Returns (ready: bool, missing_critical: list[str]).
    """
    t0 = time.perf_counter()

    # Deterministic gate — check all four anchor fields
    missing: list[str] = []
    if not intake_state.get("issue_summary"):
        missing.append("what happened — client has not described the core events yet")
    if not intake_state.get("relationship_to_other_party") or intake_state.get("relationship_to_other_party") == "unknown":
        missing.append("relationship between client and the other party")
    if not intake_state.get("client_goal_initial"):
        missing.append("what the client is seeking or hoping to achieve")
    if not intake_state.get("timeframe_status") or intake_state.get("timeframe_status") == "unknown":
        missing.append("when this happened — recent, ongoing, or historical")

    if missing:
        _t("check_readiness_deterministic", t0)
        return False, missing

    # All four anchors present — ready
    _t("check_readiness_deterministic", t0)
    return True, []


# ===========================================================================
# Internal: update known_facts from latest client message
# ===========================================================================

def _update_known_facts(intake_state: dict, client_message: str) -> None:
    """
    Extract structured fact objects from the client's latest message and append
    to known_facts.  Each fact is stored as:
      {fact, source_turn, fact_type, time_reference, evidence_hook, witness_hook, confidence_seed}

    confidence_seed: "high" (specific, verifiable), "medium" (plausible, no corroboration),
                     "uncertain" (vague, inconsistent, or unverifiable)

    Capped at 20 entries (up from 12 — structured objects carry more signal per slot).
    """
    t0 = time.perf_counter()
    msg = (client_message or "").strip()
    if not msg:
        return

    turn_num = intake_state.get("turn_count", 1)

    prompt = (
        "Extract 1–4 specific facts from the client's message below.\n"
        "For each fact output a JSON object with these exact keys:\n"
        "  fact            : one clear sentence — only what the client said, no inference\n"
        "  fact_type       : event|relationship|evidence|action_taken|relief_sought|timeline|other\n"
        "  time_reference  : specific date/period the client mentioned, or null\n"
        "  evidence_hook   : type of document/evidence implied (e.g. 'medical report', 'WhatsApp messages', 'FIR'), or null\n"
        "  witness_hook    : witness implied (e.g. 'neighbour', 'sister', 'co-worker'), or null\n"
        "  confidence_seed : high|medium|uncertain  (high=specific+verifiable, uncertain=vague/no detail)\n\n"
        "Output ONLY a JSON array of these objects. No preamble.\n\n"
        f"CLIENT MESSAGE:\n{msg}"
    )

    # Deduplicate by fact text
    existing_facts = set(
        (x.get("fact") if isinstance(x, dict) else str(x)).lower().strip()
        for x in (intake_state.get("known_facts") or [])
    )

    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("update_known_facts", t0)
        text = (raw or "").strip()
        start = text.find("[")
        end   = text.rfind("]")
        if start >= 0 and end > start:
            extracted = json.loads(text[start:end + 1])
            if isinstance(extracted, list):
                for item in extracted:
                    if not isinstance(item, dict):
                        continue
                    fact_text = str(item.get("fact") or "").strip()
                    if not fact_text or len(fact_text) < 10:
                        continue
                    if fact_text.lower() in existing_facts:
                        continue
                    existing_facts.add(fact_text.lower())
                    intake_state["known_facts"].append({
                        "fact":            fact_text,
                        "source_turn":     turn_num,
                        "fact_type":       str(item.get("fact_type") or "other").strip(),
                        "time_reference":  item.get("time_reference") or None,
                        "evidence_hook":   item.get("evidence_hook") or None,
                        "witness_hook":    item.get("witness_hook") or None,
                        "confidence_seed": str(item.get("confidence_seed") or "medium").strip().lower(),
                    })
                intake_state["known_facts"] = intake_state["known_facts"][:20]
    except Exception as exc:
        logger.warning("Stage1 structured fact extraction failed: %s", exc)
        # Fallback: store as minimal structured entry
        short = msg[:200].rstrip()
        if short.lower() not in existing_facts:
            intake_state["known_facts"].append({
                "fact":            short,
                "source_turn":     turn_num,
                "fact_type":       "other",
                "time_reference":  None,
                "evidence_hook":   None,
                "witness_hook":    None,
                "confidence_seed": "uncertain",
            })
            intake_state["known_facts"] = intake_state["known_facts"][:20]





# ===========================================================================
# Internal: urgency recheck (model-driven — no keyword lists)
# ===========================================================================

def _recheck_urgency(intake_state: dict, client_message: str) -> None:
    """
    Ask the model whether the current message changes the urgency state.
    Returns one of: "immediate", "near_term", "unchanged".
    Updates intake_state["urgency_signal"] in-place.

    This replaces brittle keyword matching with semantic reasoning so that
    any phrasing — across any legal domain or language — is handled correctly.
    """
    msg = (client_message or "").strip()
    if not msg:
        return

    current = intake_state.get("urgency_signal", "unknown")
    prompt = STAGE1_URGENCY_RECHECK_SYSTEM.format(
        current_urgency=current,
        client_message=msg,
    )
    raw = _ask_llm(prompt, task_hint="fast")
    data = _extract_json(raw)
    if not isinstance(data, dict):
        logger.debug("_recheck_urgency: failed to parse model response")
        return

    update = data.get("urgency_update", "unchanged")
    if update in ("immediate", "near_term"):
        intake_state["urgency_signal"] = update
        logger.debug(
            "_recheck_urgency: %s → %s (%s)",
            current, update, data.get("reason", "")
        )


def _infer_urgency_from_history(session: dict) -> str:
    """
    When no persisted intake_state is available, infer the current urgency
    from the full conversation history in one model call.
    Returns one of: "immediate", "near_term", "unknown".
    """
    history = session.get("history") or []
    if not history:
        return "unknown"

    lines = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "Client" if role == "user" else "Advocate"
        lines.append(f"{label}: {content}")

    conversation_text = "\n".join(lines)
    prompt = STAGE1_URGENCY_FROM_HISTORY_SYSTEM.format(
        conversation_history=conversation_text
    )
    raw = _ask_llm(prompt, task_hint="fast")
    data = _extract_json(raw)
    if not isinstance(data, dict):
        logger.debug("_infer_urgency_from_history: failed to parse model response")
        return "unknown"

    signal = data.get("urgency_signal", "unknown")
    logger.debug(
        "_infer_urgency_from_history: inferred=%s (%s)",
        signal, data.get("reason", "")
    )
    return signal if signal in ("immediate", "near_term") else "unknown"


# ===========================================================================
# Public: process_turn
# ===========================================================================

def process_turn(session: dict, user_message: str) -> dict:
    """
    Process one client turn in the Stage 1 intake flow.

    Parameters
    ----------
    session : dict
        Caller-owned session dict. Must contain:
          - "history": list of {role, content} turns (may be empty)
          - "intake_state": the running Stage 1 state dict (or None for first turn)
    user_message : str
        The client's latest message.

    Returns
    -------
    dict with keys:
      - "reply"            : str   — the AI's reply to send to the client
      - "intake_state"     : dict  — updated intake state
      - "advance_to_stage2": bool  — True when ready to hand off to Stage 2
      - "urgency_signal"   : str   — current urgency level
    """
    msg = (user_message or "").strip()
    history: list[dict] = session.get("history", [])

    # ── Initialise or load intake state ─────────────────────────────────────
    had_persisted_state = bool(session.get("intake_state"))
    intake_state: dict = session.get("intake_state") or _fresh_state()

    # When there is no persisted intake_state (e.g. frontend hasn't been
    # rebuilt yet, or state was lost), seed turn_count from the conversation
    # history so the turn_count < 2 readiness gate and urgency de-escalation
    # work correctly regardless of persistence gaps.
    if not had_persisted_state:
        history_user_turns = sum(
            1 for m in (session.get("history") or []) if m.get("role") == "user"
        )
        intake_state["turn_count"] = max(history_user_turns, 0)

        # When history exists, ask the model to infer the urgency state from the
        # full conversation so far — avoids re-escalating on a later message that
        # describes past events after the client already confirmed they are safe.
        if history_user_turns > 0:
            inferred = _infer_urgency_from_history(session)
            if inferred != "unknown":
                intake_state["urgency_signal"] = inferred

    # Increment turn counter
    intake_state["turn_count"] = intake_state.get("turn_count", 0) + 1

    # ── Build conversation context for prompts ───────────────────────────────
    context = _build_context(session)

    # ── Category detection (soft lock — re-runs until primary confidence=high) ──
    if (
        intake_state.get("primary_issue_cluster") is None
        or intake_state.get("category_confidence") in (None, "low")
    ):
        detected = _detect_category(msg, context)
        intake_state["primary_issue_cluster"]    = detected["primary_category"]
        intake_state["secondary_issue_clusters"] = detected.get("secondary_categories") or []
        intake_state["category_confidence"]      = detected["confidence"]
        # Anchor fields — only overwrite if the new value is richer
        intake_state["issue_summary"]            = detected.get("issue_summary") or intake_state.get("issue_summary") or ""
        intake_state["jurisdiction"]             = detected.get("jurisdiction") or intake_state.get("jurisdiction") or "unknown"
        # Only update urgency from detection when we don't already have a
        # concrete signal (i.e. it hasn't been set from persisted state or
        # inferred from earlier turns in the history scan above).
        # _recheck_urgency() handles escalation/de-escalation from the current
        # message immediately after this block, so we don't lose safety signals.
        if not intake_state.get("urgency_signal") or intake_state.get("urgency_signal") == "unknown":
            intake_state["urgency_signal"] = detected.get("urgency_signal") or "unknown"
        intake_state["risk_flags"]               = detected.get("risk_flags") or intake_state.get("risk_flags") or []
        intake_state["client_role"]              = detected.get("client_role") or intake_state.get("client_role") or "unknown"
        intake_state["other_party"]              = detected.get("other_party") or intake_state.get("other_party") or "unknown"
        intake_state["relationship_to_other_party"] = detected.get("relationship_to_other_party") or intake_state.get("relationship_to_other_party") or "unknown"
        intake_state["timeframe_status"]         = detected.get("timeframe_status") or intake_state.get("timeframe_status") or "unknown"
        intake_state["client_goal_initial"]      = detected.get("client_goal_initial") or intake_state.get("client_goal_initial")
        intake_state["immediate_need"]           = detected.get("immediate_need") or intake_state.get("immediate_need") or "unknown"
        intake_state["emotional_ask"]            = detected.get("emotional_ask") or intake_state.get("emotional_ask") or "unknown"

        # Merge open_questions from primary + secondary categories (deduped)
        primary_cfg = LEGAL_ISSUE_CATEGORIES.get(intake_state["primary_issue_cluster"], LEGAL_ISSUE_CATEGORIES["general"])
        merged_questions = list(primary_cfg.get("key_facts_needed", []))
        for sec_cat in intake_state["secondary_issue_clusters"]:
            sec_cfg = LEGAL_ISSUE_CATEGORIES.get(sec_cat, {})
            for kf in sec_cfg.get("key_facts_needed", []):
                if kf not in merged_questions:
                    merged_questions.append(kf)
        intake_state["open_questions"] = merged_questions

        logger.info(
            "Stage1 category (soft lock): primary=%s secondary=%s confidence=%s urgency=%s",
            intake_state["primary_issue_cluster"],
            intake_state["secondary_issue_clusters"],
            intake_state["category_confidence"],
            intake_state["urgency_signal"],
        )

    # ── Urgency recheck (safety net for escalating distress signals) ──────────
    _recheck_urgency(intake_state, msg)

    # ── Extract and accumulate facts from client message ─────────────────────
    # known_facts is still built for Stage 2 handoff and vetting logic;
    # it is no longer used to retire open_questions (the model reads history directly).
    _update_known_facts(intake_state, msg)

    # ── Remedy assessment (runs once client_goal_initial is known) ────────────
    if intake_state.get("client_goal_initial") and not intake_state.get("assessed_remedy"):
        _assess_remedy(intake_state)

    # ── Generate the AI's reply ──────────────────────────────────────────────
    # Priority order:
    #   1. Emergency triage (risk flags / immediate urgency)
    #   2. Indirect vetting (uncertain facts need corroboration — only after turn 2)
    #   3. Normal follow-up (next open question from checklist)
    # Safety-first response only fires when urgency is still "immediate".
    # Once de-escalated to "near_term" (client confirmed safe), proceed to normal follow-up.
    if intake_state.get("urgency_signal") == "immediate":
        reply = _generate_safety_first_response(intake_state)
    elif intake_state.get("turn_count", 0) >= 2:
        vetting_q = _get_vetting_question(intake_state, context)
        reply = vetting_q if vetting_q else _generate_followup(intake_state, msg, context)
    else:
        reply = _generate_followup(intake_state, msg, context)

    # ── Readiness check — decide whether to advance to Stage 2 ───────────────
    # Minimum two substantive exchanges: a single opening message, however
    # detailed, never gives enough context for grounded legal analysis.
    # Note: urgency/risk_flags affect the *reply* (safety-first response above)
    # but do NOT permanently block advancement — after enough turns the client
    # should still be able to reach analysis even for urgent matters.
    if intake_state.get("turn_count", 0) < 2:
        ready, missing = False, []
    else:
        ready, missing = _check_readiness(intake_state, context)
    intake_state["ready_for_stage2"] = ready
    if missing:
        # Persist the missing items as open questions for Stage 2 handoff
        existing_open = set(str(x).lower() for x in intake_state.get("open_questions") or [])
        for item in missing:
            if item.lower() not in existing_open:
                intake_state.setdefault("open_questions", []).append(item)
                existing_open.add(item.lower())

    logger.info(
        "Stage1 turn %d complete | primary=%s | secondary=%s | facts=%d | ready=%s | urgency=%s",
        intake_state["turn_count"],
        intake_state.get("primary_issue_cluster"),
        intake_state.get("secondary_issue_clusters"),
        len(intake_state.get("known_facts") or []),
        ready,
        intake_state.get("urgency_signal"),
    )

    return {
        "reply":             reply,
        "intake_state":      intake_state,
        "advance_to_stage2": ready,
        "urgency_signal":    intake_state.get("urgency_signal", "unknown"),
    }


# ===========================================================================
# Public: get_category_framework
# ===========================================================================

def get_category_framework(category: str) -> dict:
    """
    Return the bare-act + procedural framework for a detected category.
    Used by Stage 2 to seed its structured deep-dive.
    Pass primary_issue_cluster; call once per secondary cluster and merge if needed.
    """
    return dict(LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"]))


# ===========================================================================
# Public: new_session
# ===========================================================================

def new_session() -> dict:
    """Create a fresh session dict for a new legal opinion intake."""
    return {
        "history":      [],
        "intake_state": _fresh_state(),
    }


# ===========================================================================
# Public: generate_pre_draft_summary
# ===========================================================================

def generate_pre_draft_summary(
    intake_state: Optional[dict] = None,
    facts_summary: str = "",
) -> str:
    """
    Generate a 3–5 sentence pre-draft summary to show the client immediately
    before the full legal draft is produced.

    Works with or without a full intake_state: falls back gracefully to the
    plain facts_summary string when Stage 1 state is not available.

    Returns a plain-text summary ending with
    "I'll now prepare your full legal analysis and draft."
    """
    state = intake_state or {}
    primary = state.get("primary_issue_cluster") or "your legal matter"

    # Build facts string from known_facts structured objects when not supplied
    if not facts_summary:
        known = state.get("known_facts") or []
        facts_summary = "; ".join(
            (f.get("fact") if isinstance(f, dict) else str(f))
            for f in known[:8]
        ).strip() or state.get("issue_summary") or "situation as described"

    # Evidence hooks — deduplicated list of implied evidence types
    evidence_hooks = [
        f.get("evidence_hook")
        for f in (state.get("known_facts") or [])
        if isinstance(f, dict) and f.get("evidence_hook")
    ]
    evidence_summary = ", ".join(dict.fromkeys(filter(None, evidence_hooks))) or "none noted"

    # Remedy fields — from _assess_remedy output stored in intake_state
    remedy_block = state.get("assessed_remedy") or {}
    stated_remedy    = state.get("stated_remedy") or "not stated"
    assessed_remedy  = remedy_block.get("assessed_remedy") or state.get("client_goal_initial") or "under review"
    recommended_lead = remedy_block.get("recommended_lead") or "to be determined"
    faster_alt       = remedy_block.get("faster_alternative") or "none identified"
    urgency          = state.get("urgency_signal") or "normal"

    prompt = (
        PRE_DRAFT_SUMMARY_SYSTEM
        .replace("{primary_category}",  str(primary))
        .replace("{facts_summary}",     str(facts_summary)[:800])
        .replace("{evidence_summary}",  str(evidence_summary))
        .replace("{stated_remedy}",     str(stated_remedy))
        .replace("{assessed_remedy}",   str(assessed_remedy))
        .replace("{recommended_lead}",  str(recommended_lead))
        .replace("{faster_alternative}", str(faster_alt))
        .replace("{urgency_signal}",    str(urgency))
    )

    try:
        t0 = time.perf_counter()
        reply = _ask_llm_quality(prompt).strip()
        _t("generate_pre_draft_summary", t0)
        if reply and len(reply) > 30:
            return reply
    except Exception as exc:
        logger.warning("Pre-draft summary LLM failed: %s", exc)

    # Deterministic fallback — always ends with the required closing line
    return (
        f"I've carefully noted everything you've shared about {primary}. "
        "The facts and evidence on record give us a solid foundation to proceed. "
        "I'll now prepare your full legal analysis and draft."
    )
