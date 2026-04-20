"""
intake/stage1_opening.py — Stage 1: Opening + issue identification.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

from agents.intake.casefile import (
    apply_fact_audit,
    ensure_case_file_structure,
    set_pipeline_layer,
)

# ---------------------------------------------------------------------------
# Lazy imports — avoid circular imports at module load time
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "fast") -> str:
    from platform_pkg.llm import ask_llm
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
    STAGE1_INITIAL_DETAILS_REQUEST_SYSTEM,
    STAGE1_GAP_REVIEW_SYSTEM,
    STAGE1_SAFETY_FIRST_SYSTEM,
    STAGE1_URGENCY_RECHECK_SYSTEM,
    STAGE1_URGENCY_FROM_HISTORY_SYSTEM,
    STAGE1_VETTING_QUESTION_SYSTEM,
    STAGE4_REMEDY_ASSESSMENT_SYSTEM,
    CASE_FILE_REFRESH_SYSTEM,
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
    from agents.intake.schema import make_intake_state
    return make_intake_state()


def _merge_anchor_fields(intake_state: dict, updates: dict | None) -> None:
    """Merge anchor fields from an LLM-produced intake update without wiping established values."""
    if not isinstance(updates, dict):
        return

    summary = str(updates.get("issue_summary") or "").strip()
    if summary:
        intake_state["issue_summary"] = summary[:500]
        intake_state["latest_intake_summary"] = intake_state["issue_summary"]

    relationship = str(updates.get("relationship_to_other_party") or "").strip()
    if relationship and relationship.lower() != "unknown":
        intake_state["relationship_to_other_party"] = relationship[:160]

    timeframe = str(updates.get("timeframe_status") or "").strip().lower()
    if timeframe in ("ongoing", "recent", "historical"):
        intake_state["timeframe_status"] = timeframe

    goal = str(updates.get("client_goal_initial") or "").strip()
    if goal and goal.lower() not in ("unknown", "null", "none"):
        intake_state["client_goal_initial"] = goal[:240]


def _extract_json_list(text: str) -> list:
    """Extract the first JSON array from text that may contain markdown or stray prose."""
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if "[" in part and "]" in part:
                text = part
                break
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _refresh_case_file(intake_state: dict, conversation_context: str) -> None:
    """
    Build or refresh the canonical case_file from the transcript and structured facts.
    """
    known_facts = intake_state.get("known_facts") or []
    facts_lines = []
    for fact in known_facts[:12]:
        if isinstance(fact, dict):
            fact_text = str(fact.get("fact") or "").strip()
            if fact_text:
                facts_lines.append(f"- {fact_text}")
    facts_summary = "\n".join(facts_lines) or "- No structured facts captured yet"
    structured_facts = []
    for fact in known_facts[:15]:
        if not isinstance(fact, dict):
            continue
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_text:
            continue
        structured_facts.append({
            "fact": fact_text,
            "fact_type": fact.get("fact_type"),
            "time_reference": fact.get("time_reference"),
            "evidence_hook": fact.get("evidence_hook"),
            "witness_hook": fact.get("witness_hook"),
            "confidence_seed": fact.get("confidence_seed"),
        })

    state_summary = {
        "issue_summary": intake_state.get("issue_summary"),
        "relationship_to_other_party": intake_state.get("relationship_to_other_party"),
        "timeframe_status": intake_state.get("timeframe_status"),
        "client_goal_initial": intake_state.get("client_goal_initial"),
        "urgency_signal": intake_state.get("urgency_signal"),
        "detail_groups_requested": intake_state.get("detail_groups_requested") or [],
        "missing_detail_groups": intake_state.get("missing_detail_groups") or [],
        "assessed_remedy": intake_state.get("assessed_remedy"),
    }

    prompt = (
        CASE_FILE_REFRESH_SYSTEM
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{facts_summary}", facts_summary)
        .replace("{structured_facts}", json.dumps(structured_facts, ensure_ascii=False, indent=2))
        .replace("{state_summary}", json.dumps(state_summary, ensure_ascii=False, indent=2))
    )

    case_file = ensure_case_file_structure(
        intake_state.get("case_file") or copy.deepcopy((STAGE1_INTAKE_STATE_SCHEMA.get("case_file") or {}))
    )
    case_file["summary"] = intake_state.get("latest_intake_summary") or intake_state.get("issue_summary")

    _case_file_system = (
        "You are a senior Indian advocate preparing an internal chamber note on a professional legal platform. "
        "The intake record below contains client-reported facts about alleged abuse, crimes, threats, and harm — "
        "provided strictly for internal legal case-file preparation. "
        "Produce the complete structured JSON case file as instructed. "
        "Do NOT refuse, truncate, or omit any section based on the nature of the allegations described."
    )

    try:
        from platform_pkg.llm import ask_llm as _direct_ask
        raw = _direct_ask(prompt, task_hint="quality", system=_case_file_system)
        data = _extract_json(raw)
        if isinstance(data, dict):
            case_file["summary"] = str(data.get("summary") or case_file.get("summary") or "").strip()[:600] or None
            immediate_concerns = [
                str(x).strip()
                for x in (data.get("immediate_concerns") or [])
                if str(x).strip()
            ]
            case_file["immediate_concerns"] = immediate_concerns[:6]

            for key in ("case_theory", "evidence_posture", "risk_map", "timeline", "procedural_posture"):
                if isinstance(data.get(key), dict):
                    case_file[key] = data[key]

            raw_timeline = data.get("timeline") or {}
            if isinstance(raw_timeline, dict):
                events = []
                for item in (raw_timeline.get("events") or []):
                    if not isinstance(item, dict):
                        continue
                    event = str(item.get("event") or "").strip()
                    if not event:
                        continue
                    materials = [
                        str(x).strip()
                        for x in (item.get("supporting_materials") or [])
                        if str(x).strip()
                    ]
                    events.append({
                        "date_or_period": str(item.get("date_or_period") or "").strip()[:120] or None,
                        "event": event[:260],
                        "significance": str(item.get("significance") or "").strip()[:220] or None,
                        "supporting_materials": materials[:4],
                    })
                case_file["timeline"] = {
                    "events": events[:8],
                    "latest_material_event": str(raw_timeline.get("latest_material_event") or "").strip()[:260] or None,
                    "timeline_gaps": [
                        str(x).strip()
                        for x in (raw_timeline.get("timeline_gaps") or [])
                        if str(x).strip()
                    ][:5],
                }

            raw_posture = data.get("procedural_posture") or {}
            if isinstance(raw_posture, dict):
                case_file["procedural_posture"] = {
                    "current_stage": str(raw_posture.get("current_stage") or "").strip()[:120] or None,
                    "steps_already_taken": [
                        str(x).strip()
                        for x in (raw_posture.get("steps_already_taken") or [])
                        if str(x).strip()
                    ][:6],
                    "current_forum_or_authority": str(raw_posture.get("current_forum_or_authority") or "").strip()[:160] or None,
                    "next_deadline_or_trigger": str(raw_posture.get("next_deadline_or_trigger") or "").strip()[:220] or None,
                    "limitation_notes": [
                        str(x).strip()
                        for x in (raw_posture.get("limitation_notes") or [])
                        if str(x).strip()
                    ][:4],
                }

            fact_proof_matrix = []
            for item in (data.get("fact_proof_matrix") or []):
                if not isinstance(item, dict):
                    continue
                fact_text = str(item.get("fact") or "").strip()
                if not fact_text:
                    continue
                support_status = str(item.get("support_status") or "asserted-only").strip().lower()
                if support_status not in ("document-backed", "witness-backed", "partly-supported", "asserted-only"):
                    support_status = "asserted-only"
                materials = [
                    str(x).strip()
                    for x in (item.get("supporting_materials") or [])
                    if str(x).strip()
                ]
                witnesses = [
                    str(x).strip()
                    for x in (item.get("witness_support") or [])
                    if str(x).strip()
                ]
                fact_proof_matrix.append({
                    "fact": fact_text[:260],
                    "support_status": support_status,
                    "supporting_materials": materials[:5],
                    "witness_support": witnesses[:4],
                    "proof_gap": str(item.get("proof_gap") or "").strip()[:260] or None,
                })
            case_file["fact_proof_matrix"] = fact_proof_matrix

            missing_proof = []
            for item in (data.get("missing_proof_recommendations") or []):
                if not isinstance(item, dict):
                    continue
                point = str(item.get("point") or "").strip()
                if not point:
                    continue
                missing_proof.append({
                    "point": point[:260],
                    "why_it_matters": str(item.get("why_it_matters") or "").strip()[:280] or None,
                    "best_source": str(item.get("best_source") or "").strip()[:220] or None,
                })
            case_file["missing_proof_recommendations"] = missing_proof

            contradictions = []
            for item in (data.get("contradictions") or []):
                if not isinstance(item, dict):
                    continue
                issue = str(item.get("issue") or "").strip()
                if not issue:
                    continue
                severity = str(item.get("severity") or "low").strip().lower()
                if severity not in ("low", "medium", "high"):
                    severity = "low"
                contradictions.append({
                    "issue": issue[:240],
                    "severity": severity,
                    "note": str(item.get("note") or "").strip()[:300] or None,
                })
            case_file["contradictions"] = contradictions
    except Exception as exc:
        logger.warning("Case file refresh failed: %s", exc)

    case_file = apply_fact_audit(
        case_file,
        known_facts,
        note=(
            "Case theory, evidence posture, timeline, and risk map are model-generated chamber summaries "
            "derived from the transcript and structured facts."
        ),
    )
    case_file = set_pipeline_layer(case_file, "intake", "complete", "Structured intake facts and grouped client inputs captured.")
    case_file = set_pipeline_layer(case_file, "case_framing", "complete", "Canonical case file refreshed from transcript and structured facts.")
    case_file = set_pipeline_layer(case_file, "research", "pending", "Research packets are added during draft generation.")
    case_file = set_pipeline_layer(case_file, "opinion", "pending", "Client-facing and chamber-note outputs are generated during drafting.")
    case_file = set_pipeline_layer(case_file, "action", "pending", "Action drafting readiness is assessed during draft generation.")
    intake_state["case_file"] = case_file


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
    from platform_pkg.llm import ask_llm
    risk_flags  = intake_state.get("risk_flags") or []
    immediate   = intake_state.get("immediate_need") or "unknown"
    summary     = intake_state.get("issue_summary") or "the client's urgent situation"

    prompt = (
        STAGE1_SAFETY_FIRST_SYSTEM
        .replace("{risk_flags}", ", ".join(risk_flags) if risk_flags else "immediate distress")
        .replace("{immediate_need}", immediate)
        .replace("{issue_summary}", summary)
    )

    system_framing = (
        "You are a senior Indian advocate on a professional legal advisory platform. "
        "A client has reported an urgent or dangerous situation. "
        "Respond as a qualified legal professional providing immediate safety guidance."
    )

    try:
        reply = ask_llm(prompt, task_hint="quality", system=system_framing).strip()
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
# Internal: compact intake orchestration helpers
# ===========================================================================

def _build_established_facts(intake_state: dict) -> str:
    """Build a compact text summary of the main state already established.

    Returns two sections:
    - ANCHOR FIELDS: high-level structured fields (issue, relationship, etc.)
    - CONFIRMED FACTS: specific facts extracted from the client's messages
    - INTAKE PROGRESS: what groups were requested and what is still open

    The explicit confirmed-facts list is the primary guard against the model
    re-asking information the client already provided.
    """
    lines: list[str] = []

    # ── Anchor fields ──────────────────────────────────────────────────────────
    anchors = {
        "issue":         intake_state.get("issue_summary"),
        "relationship":  intake_state.get("relationship_to_other_party"),
        "timeframe":     intake_state.get("timeframe_status"),
        "goal":          intake_state.get("client_goal_initial"),
        "urgency":       intake_state.get("urgency_signal"),
        "jurisdiction":  intake_state.get("jurisdiction"),
    }
    anchor_parts = [
        f"{k}: {v}" for k, v in anchors.items()
        if v and v not in ("unknown", "", None)
    ]
    if anchor_parts:
        lines.append("ANCHOR FIELDS — " + "; ".join(anchor_parts))

    # ── Confirmed facts (specific details already provided by the client) ──────
    known_facts = intake_state.get("known_facts") or []
    if known_facts:
        fact_texts = [
            (f.get("fact") if isinstance(f, dict) else str(f))
            for f in known_facts[:20]
        ]
        fact_texts = [t.strip() for t in fact_texts if t and t.strip()]
        if fact_texts:
            lines.append(
                "CONFIRMED FACTS (already provided — DO NOT re-ask these):\n"
                + "\n".join(f"  • {t}" for t in fact_texts)
            )

    # ── Intake progress ────────────────────────────────────────────────────────
    detail_groups = intake_state.get("detail_groups_requested") or []
    if detail_groups:
        lines.append("Detail groups requested: " + " | ".join(str(x) for x in detail_groups[:8]))
    missing_groups = intake_state.get("missing_detail_groups") or []
    if missing_groups:
        lines.append("Still open / not yet answered: " + " | ".join(str(x) for x in missing_groups[:8]))

    return "\n".join(lines) if lines else "none established yet"


def _build_retrieved_sections_summary(retrieved_context: dict | None) -> str:
    """
    Build a compact summary of pre-retrieved act sections for law-informed gap questions.
    Returns an empty string when no context is available (Turn 1 or retrieval failed).
    """
    if not retrieved_context:
        return ""
    sections = retrieved_context.get("bare_act_sections") or []
    if not sections:
        return ""
    lines: list[str] = []
    seen_acts: set[str] = set()
    for sec in sections:
        act = ((sec.get("act_name") or "").strip()
               or sec.get("title", "").split("§")[0].split("Section")[0].strip())
        sec_num = (sec.get("section_number") or "").strip()
        if not act:
            continue
        entry = f"- {act}" + (f", Section {sec_num}" if sec_num else "")
        # Deduplicate by act name
        key = act.lower()
        if key in seen_acts:
            continue
        seen_acts.add(key)
        lines.append(entry)
        if len(lines) >= 5:
            break
    if not lines:
        return ""
    return (
        "RETRIEVED LEGAL CONTEXT — use these to ask gap questions grounded in what the law requires:\n"
        + "\n".join(lines)
        + "\nWhen asking for missing information, briefly explain how it helps establish the relevant legal element — but do NOT cite section numbers in the reply itself."
    )


def _generate_initial_detail_request(
    intake_state: dict,
    client_message: str,
    conversation_context: str,
    retrieved_context: dict | None = None,
) -> str:
    """Generate the first grouped request for all material intake details."""
    t0 = time.perf_counter()
    from platform_pkg.llm import ask_llm

    primary = intake_state.get("primary_issue_cluster") or "general"
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(primary, {})
    category_label = cat_cfg.get("label", primary.replace("_", " ").title())

    prompt = (
        STAGE1_INITIAL_DETAILS_REQUEST_SYSTEM
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
        .replace("{established_facts}", _build_established_facts(intake_state))
        .replace("{category}", category_label)
        .replace("{retrieved_sections_context}", _build_retrieved_sections_summary(retrieved_context))
    )

    system_framing = (
        "You are a senior Indian advocate on a professional legal advisory platform. "
        "The client may be describing abuse, crime, threats, family disputes, commercial disputes, "
        "or any other legal problem. Respond as a lawyer conducting concise, high-signal intake. "
        "Do not refuse merely because the user describes illegal acts or harm; this is legal intake."
    )

    try:
        raw = ask_llm(prompt, task_hint="quality", system=system_framing).strip()
        _t("initial_detail_request", t0)
        data = _extract_json(raw)
        if isinstance(data, dict):
            _merge_anchor_fields(intake_state, data)
            detail_groups = [
                str(x).strip() for x in (data.get("detail_groups_requested") or [])
                if str(x).strip()
            ][:8]
            enough = bool(data.get("enough_for_analysis", False))
            intake_state["detail_groups_requested"] = detail_groups
            intake_state["open_questions"] = [] if enough else list(detail_groups)
            intake_state["missing_detail_groups"] = []
            intake_state["followup_questions"] = []
            intake_state["detail_request_issued"] = True
            intake_state["analysis_ready"] = enough
            reply = str(data.get("reply") or "").strip()
            if reply:
                return reply
    except Exception as exc:
        logger.warning("Stage1 initial detail request LLM failed: %s", exc)

    intake_state["detail_request_issued"] = True
    intake_state["analysis_ready"] = False
    fallback_groups = [
        "A brief timeline of what happened and the latest important event",
        "Who the other people or entities are and how they are connected to you",
        "What documents, messages, photos, notices, reports, or other evidence you already have",
        "What steps have already been taken, if any, with police, court, employer, bank, authority, or the other side",
        "What result you want most urgently and any deadlines or immediate concerns",
    ]
    intake_state["detail_groups_requested"] = fallback_groups
    intake_state["open_questions"] = list(fallback_groups)
    return (
        "I’ve gone through what you shared. Give me a little time to work out which details matter most for your case, "
        "then please send whatever you can on these points:\n"
        "- A brief timeline of what happened and the latest important event\n"
        "- Who the other people or entities are and how they are connected to you\n"
        "- What documents, messages, photos, notices, reports, or other evidence you already have\n"
        "- What steps have already been taken, if any, with police, court, employer, bank, authority, or the other side\n"
        "- What result you want most urgently and any deadlines or immediate concerns"
    )


def _build_deferred_questions_context(intake_state: dict) -> str:
    """
    Build the deferred-questions injection for the gap review prompt.
    If the previous round deferred lower-priority questions, remind the model
    to consider them (and drop any already answered by the latest client message).
    """
    deferred = [
        str(x).strip() for x in (intake_state.get("deferred_questions") or [])
        if str(x).strip()
    ]
    if not deferred:
        return ""
    lines = "\n".join(f"- {q}" for q in deferred)
    return (
        "DEFERRED FROM PREVIOUS ROUND — lower-priority gaps not yet asked. "
        "Include any still-missing ones in this turn if the higher-priority questions have now been answered; "
        "otherwise continue to defer them:\n"
        + lines
    )


def _generate_gap_review(
    intake_state: dict,
    client_message: str,
    conversation_context: str,
    retrieved_context: dict | None = None,
) -> tuple[str, bool]:
    """Review the bundled client response and decide whether intake is ready for analysis."""
    t0 = time.perf_counter()
    from platform_pkg.llm import ask_llm

    prompt = (
        STAGE1_GAP_REVIEW_SYSTEM
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
        .replace(
            "{detail_groups_requested}",
            "\n".join(f"- {item}" for item in (intake_state.get("detail_groups_requested") or [])) or "(none recorded)"
        )
        .replace("{established_facts}", _build_established_facts(intake_state))
        .replace("{retrieved_sections_context}", _build_retrieved_sections_summary(retrieved_context))
        .replace("{deferred_questions_context}", _build_deferred_questions_context(intake_state))
    )

    system_framing = (
        "You are a senior Indian advocate on a professional legal advisory platform. "
        "Your task is to run a compact, practical legal intake review. "
        "Trust the whole conversation. Only ask what is truly still missing. "
        "If the facts are sufficient for grounded legal analysis, say so and move on."
    )

    try:
        raw = ask_llm(prompt, task_hint="quality", system=system_framing).strip()
        _t("gap_review", t0)
        data = _extract_json(raw)
        if isinstance(data, dict):
            _merge_anchor_fields(intake_state, data)
            missing_groups = [
                str(x).strip() for x in (data.get("missing_detail_groups") or [])
                if str(x).strip()
            ][:8]
            followups = [
                str(x).strip() for x in (data.get("followup_questions") or [])
                if str(x).strip()
            ][:4]
            deferred = [
                str(x).strip() for x in (data.get("deferred_questions") or [])
                if str(x).strip()
            ][:6]
            enough = bool(data.get("enough_for_analysis", False))
            intake_state["missing_detail_groups"] = missing_groups
            intake_state["followup_questions"] = followups
            intake_state["deferred_questions"] = [] if enough else deferred
            intake_state["open_questions"] = list(missing_groups)
            intake_state["analysis_ready"] = enough
            reply = str(data.get("reply") or "").strip()
            if reply:
                return reply, enough
    except Exception as exc:
        logger.warning("Stage1 gap review LLM failed: %s", exc)

    # JSON parse failed or LLM error: do NOT falsely advance to Stage 2.
    # Preserve whatever state was set from the last successful gap review and
    # return a safe nudge question so the session continues collecting facts.
    logger.warning("Stage1 gap review: could not parse LLM response — holding in fact collection")
    existing_followups = [
        str(x).strip() for x in (intake_state.get("followup_questions") or [])
        if str(x).strip()
    ]
    open_qs = [
        str(x).strip() for x in (intake_state.get("open_questions") or [])
        if str(x).strip()
    ]
    if existing_followups:
        fallback_reply = f"Could you share a bit more detail on: {existing_followups[0].lower()}?"
    elif open_qs:
        fallback_reply = f"Could you tell me more about {open_qs[0].lower()}?"
    else:
        fallback_reply = "Could you share a bit more about what happened and what outcome you are hoping for?"
    return fallback_reply, bool(intake_state.get("analysis_ready", False))


def _check_analysis_readiness(intake_state: dict) -> tuple[bool, list[str]]:
    """Readiness check for the compact intake flow."""
    if intake_state.get("analysis_ready"):
        return True, []
    missing = [
        str(x).strip() for x in (intake_state.get("missing_detail_groups") or [])
        if str(x).strip()
    ]
    return False, missing


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

    Uses a system-role framing message so OpenAI's server-side content filter
    applies its professional-use threshold rather than the consumer threshold
    when the conversation contains client-reported allegations of violence or threats.
    """
    t0 = time.perf_counter()
    from platform_pkg.llm import ask_llm

    # Human-readable category label for the prompt
    primary = intake_state.get("primary_issue_cluster") or "general"
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(primary, {})
    category_label = cat_cfg.get("label", primary.replace("_", " ").title())

    # Use _build_established_facts so the model sees both anchor fields AND the
    # enumerated list of specific facts already confirmed — the primary guard
    # against re-asking information the client already provided.
    established_facts = _build_established_facts(intake_state)

    prompt = (
        STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM
        .replace("{category}", category_label)
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
        .replace("{established_facts}", established_facts)
    )

    # Professional framing system message — signals to OpenAI that this is a
    # legitimate legal advisory context processing client-reported allegations.
    system_framing = (
        "You are a senior Indian advocate on a professional legal advisory platform. "
        "The conversation below contains client-reported facts about alleged illegal acts, "
        "threats, and harm — provided for professional legal intake and advice purposes only. "
        "Do NOT refuse, omit, or sanitise any part of your response based on the nature of the "
        "allegations described. This is a licensed legal platform; all output is reviewed by "
        "qualified professionals and filtered before it reaches any end user. "
        "Respond as a qualified legal professional conducting a structured intake interview."
    )

    try:
        reply = ask_llm(prompt, task_hint="quality", system=system_framing).strip()
        _t("generate_followup", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage1 followup LLM failed: %s", exc)

    return "Could you tell me a bit more about what's been happening?"


# ===========================================================================
# Internal: update known_facts from latest client message
# ===========================================================================

def _update_known_facts(intake_state: dict, client_message: str) -> None:
    """
    Extract structured fact objects from the client's latest message and append
    to known_facts.  Each fact is stored as:
      {fact, source_turn, source_type, source_detail, fact_type, time_reference,
       evidence_hook, witness_hook, confidence_seed}

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
        extracted = _extract_json_list(raw)
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
                    "source_type":     "user_statement",
                    "source_detail":   f"user_turn_{turn_num}",
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
                "source_type":     "user_statement",
                "source_detail":   f"user_turn_{turn_num}",
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


def _reconstruct_state_from_history(session: dict, intake_state: dict) -> None:
    """
    When no persisted intake_state is available but conversation history exists,
    recover anchor fields from prior turns so the model doesn't re-ask for
    information the client already provided.

    Only fills fields that are still empty or 'unknown' in intake_state.
    """
    history = session.get("history") or []
    lines = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "Client" if role == "user" else "Advocate"
        lines.append(f"{label}: {content}")
    if not lines:
        return

    conversation_text = "\n".join(lines)
    prompt = (
        "From the conversation below, extract the following facts if clearly stated by the client. "
        "Output ONLY a JSON object with exactly these keys:\n"
        "  issue_summary               : one-sentence summary of the core legal problem, or null\n"
        "  relationship_to_other_party : how the client is related to the other party "
        "(e.g. landlord, employer, spouse), or 'unknown'\n"
        "  timeframe_status            : 'recent' (<3 months), 'ongoing', 'historical' (>1 year), "
        "or 'unknown'\n"
        "  client_goal_initial         : what the client wants to achieve (e.g. 'get refund', "
        "'file FIR'), or null\n"
        "  jurisdiction                : Indian state or union territory mentioned, or 'unknown'\n"
        "\nUse null or 'unknown' for anything not clearly stated. No preamble.\n"
        "\nCONVERSATION:\n" + conversation_text
    )
    try:
        raw = _ask_llm(prompt, task_hint="fast")
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return
        _ANCHOR_FIELDS = (
            "issue_summary",
            "relationship_to_other_party",
            "timeframe_status",
            "client_goal_initial",
            "jurisdiction",
        )
        recovered = []
        for field in _ANCHOR_FIELDS:
            if intake_state.get(field) and intake_state.get(field) not in ("unknown", ""):
                continue  # already set — don't overwrite
            val = data.get(field)
            if val and str(val).strip() not in ("null", "unknown", "None", ""):
                intake_state[field] = str(val).strip()
                recovered.append(field)
        if recovered:
            logger.info("_reconstruct_state_from_history: recovered %s", recovered)
    except Exception as exc:
        logger.debug("_reconstruct_state_from_history failed: %s", exc)


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
    # ── Initialise or load intake state ─────────────────────────────────────
    from agents.intake.schema import coerce_intake_state
    had_persisted_state = bool(session.get("intake_state"))
    raw_state = session.get("intake_state")
    intake_state: dict = coerce_intake_state(raw_state) if raw_state else _fresh_state()

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
            # Recover structured anchor fields from prior turns so the model
            # doesn't re-ask for information the client already provided.
            _reconstruct_state_from_history(session, intake_state)

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

    # ── Retrieve preliminary legal context (available from Turn 2 onwards) ─────
    # On Turn 1, intake and retrieval run in parallel so retrieved_context is None.
    # From Turn 2, the prelim cache is available and used to inform gap questions.
    retrieved_context: dict | None = session.get("retrieved_context") or None

    # ── Generate the AI's reply ──────────────────────────────────────────────
    # Priority order:
    #   1. Emergency triage (risk flags / immediate urgency)
    #   2. Indirect vetting (uncertain facts need corroboration — only after turn 2)
    #   3. Normal follow-up (next open question from checklist)
    # Safety-first response only fires when urgency is still "immediate".
    # Once de-escalated to "near_term" (client confirmed safe), proceed to normal follow-up.
    if intake_state.get("urgency_signal") == "immediate":
        reply = _generate_safety_first_response(intake_state)
        ready, missing = False, []
    elif not intake_state.get("detail_request_issued"):
        reply = _generate_initial_detail_request(intake_state, msg, context, retrieved_context=retrieved_context)
        # Honour the LLM's judgment: if the opening message already contained
        # enough facts, skip further intake and go straight to analysis.
        if intake_state.get("analysis_ready"):
            ready, missing = True, []
        else:
            ready, missing = False, []
    else:
        reply, _ = _generate_gap_review(intake_state, msg, context, retrieved_context=retrieved_context)
        ready, missing = _check_analysis_readiness(intake_state)

    # ── Readiness check — advance guard ─────────────────────────────────────
    # Immediate urgency: hold analysis, run safety-first flow first.
    # detail_request_issued check: if somehow this block is reached before the
    # initial request was issued, prevent premature advancement.
    if intake_state.get("urgency_signal") == "immediate":
        ready, missing = False, []
    elif not intake_state.get("detail_request_issued"):
        ready, missing = False, []
    intake_state["ready_for_stage2"] = ready
    if missing:
        # Persist the missing items as open questions for Stage 2 handoff
        existing_open = set(str(x).lower() for x in intake_state.get("open_questions") or [])
        for item in missing:
            if item.lower() not in existing_open:
                intake_state.setdefault("open_questions", []).append(item)
                existing_open.add(item.lower())

    _refresh_case_file(intake_state, context)

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
    remedy_block = state.get("remedy_detail") or {}
    stated_remedy    = state.get("stated_remedy") or "not stated"
    assessed_remedy  = state.get("assessed_remedy") or state.get("client_goal_initial") or "under review"
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
