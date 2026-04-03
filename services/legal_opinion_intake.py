"""
Legal Opinion Intake - Stage 1: Opening + Issue Identification.

Stage 1 keeps the client-facing experience warm and natural while building
the hidden structured intake state that later stages consume.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


def _ask_llm(prompt: str, task_hint: str = "fast", model_override: str | None = None) -> str:
    from llm.ollama_client import ask_llm

    return ask_llm(prompt, task_hint=task_hint, model=model_override) or ""


def _ask_llm_quality(prompt: str, model_override: str | None = None) -> str:
    return _ask_llm(prompt, task_hint="quality", model_override=model_override)


from prompts.advocate_prompts import (  # noqa: E402
    ISSUE_CATEGORY_DETECT_SYSTEM,
    LEGAL_ISSUE_CATEGORIES,
    LEGAL_OPINION_OPENING_SYSTEM,
    STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM,
    STAGE1_INTAKE_STATE_SCHEMA,
    STAGE1_READINESS_CHECK_SYSTEM,
)

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage1.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


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
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _build_context(session: dict, max_turns: int = 6, max_chars: int = 260) -> str:
    history = session.get("history", [])
    lines: list[str] = []
    for turn in history[-max_turns:]:
        role = "Client" if turn.get("role") == "user" else "Counsel"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _fresh_state() -> dict:
    return copy.deepcopy(STAGE1_INTAKE_STATE_SCHEMA)


def _normalize_timeframe_label(text: str) -> str:
    low = (text or "").strip().lower()
    if low in {"recent", "ongoing", "historical", "unknown"}:
        return low
    if any(w in low for w in ("today", "now", "currently", "ongoing", "still", "continuing")):
        return "ongoing"
    if any(w in low for w in ("yesterday", "last week", "recent", "few days", "this month")):
        return "recent"
    if any(w in low for w in ("years", "months ago", "long back", "earlier", "historical", "old")):
        return "historical"
    return "unknown"


def _dedupe_keep_order(values: list[str], cap: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if cap and len(out) >= cap:
            break
    return out


def generate_opening(model_override: str | None = None) -> str:
    t0 = time.perf_counter()
    try:
        reply = _ask_llm(
            LEGAL_OPINION_OPENING_SYSTEM, task_hint="quality", model_override=model_override
        ).strip()
        _t("generate_opening", t0)
        if reply and len(reply) > 20:
            return reply
    except Exception as exc:
        logger.warning("Stage1 opening LLM failed: %s", exc)

    return (
        "Whatever you're going through, I'm here and I'm listening. "
        "Take your time and share what's on your mind - there's no wrong way to start."
    )


def _detect_category(client_message: str, conversation_context: str, model_override: str | None = None) -> dict:
    t0 = time.perf_counter()
    prompt = (
        ISSUE_CATEGORY_DETECT_SYSTEM
        + "\n\nCONVERSATION SO FAR:\n"
        + (conversation_context or "(first message)")
        + "\n\nCLIENT MESSAGE:\n"
        + (client_message or "").strip()
    )

    default = {
        "category": "general",
        "secondary_categories": [],
        "confidence": "low",
        "issue_summary": "",
        "jurisdiction": "unknown",
        "urgency_signal": "unknown",
        "client_role": "unknown",
        "other_party": "unknown",
        "relationship_to_other_party": "unknown",
        "risk_flags": [],
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("detect_category", t0)
        out = _extract_json(raw)
        if not out or not isinstance(out, dict):
            return default

        valid_cats = set(LEGAL_ISSUE_CATEGORIES.keys())
        cat = str(out.get("category") or "general").strip().lower()
        if cat not in valid_cats:
            cat = "general"

        confidence = str(out.get("confidence") or "low").strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        urgency = str(out.get("urgency_signal") or "unknown").strip().lower()
        if urgency not in ("immediate", "near_term", "no_urgency", "unknown"):
            urgency = "unknown"

        secondary = []
        for item in (out.get("secondary_categories") or []):
            candidate = str(item).strip().lower()
            if candidate and candidate in valid_cats and candidate != cat:
                secondary.append(candidate)

        valid_risks = {
            "immediate_physical_danger",
            "child_safety_risk",
            "medical_emergency",
            "arrest_or_custody_risk",
            "same_day_deadline",
            "housing_lockout",
            "ongoing_contact_risk",
        }
        risk_flags = [
            str(flag).strip().lower()
            for flag in (out.get("risk_flags") or [])
            if str(flag).strip().lower() in valid_risks
        ]

        return {
            "category": cat,
            "secondary_categories": _dedupe_keep_order(secondary, cap=2),
            "confidence": confidence,
            "issue_summary": str(out.get("issue_summary") or "").strip()[:300],
            "jurisdiction": str(out.get("jurisdiction_hint") or "unknown").strip(),
            "urgency_signal": urgency,
            "client_role": str(out.get("client_role") or "unknown").strip(),
            "other_party": str(out.get("other_party") or "unknown").strip()[:120],
            "relationship_to_other_party": str(out.get("relationship_to_other_party") or "unknown").strip()[:80],
            "risk_flags": _dedupe_keep_order(risk_flags, cap=8),
        }
    except Exception as exc:
        logger.warning("Stage1 category detection failed: %s", exc)
        return default


def _extract_stage1_updates(client_message: str, intake_state: dict, model_override: str | None = None) -> dict:
    msg = (client_message or "").strip()
    if not msg:
        return {}

    prompt = f"""You are extracting hidden structured intake data from a client's latest message.

Existing Stage 1 state:
{json.dumps({
    "issue_summary": intake_state.get("issue_summary"),
    "relationship_to_other_party": intake_state.get("relationship_to_other_party"),
    "timeframe_status": intake_state.get("timeframe_status"),
    "client_goal_initial": intake_state.get("client_goal_initial"),
    "emotional_ask": intake_state.get("emotional_ask"),
    "immediate_need": intake_state.get("immediate_need"),
}, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "facts": [
    {{
      "fact": "<short fact stated by the client>",
      "fact_type": "event|party|relationship|timeframe|goal|emotional_ask|immediate_need|evidence|risk|other",
      "time_reference": "<exact or rough time if mentioned, else null>",
      "evidence_hook": "<document/message/medical/witness hook, else null>",
      "witness_hook": "<who may confirm this, else null>",
      "confidence_seed": "stated"
    }}
  ],
  "relationship_to_other_party": "<updated relationship label or null>",
  "timeframe_status": "recent|ongoing|historical|unknown|null",
  "client_goal_initial": "<practical outcome the client appears to want, else null>",
  "emotional_ask": "<emotional or expressive ask if different from practical goal, else null>",
  "immediate_need": "<what they need right now, else null>",
  "risk_flags": ["<any of: immediate_physical_danger, child_safety_risk, medical_emergency, arrest_or_custody_risk, same_day_deadline, housing_lockout, ongoing_contact_risk>"]
}}

Rules:
- Use only what the client actually said.
- Keep facts atomic and short.
- If a field is not newly supported by the message, return null for that field.
- If the client states what they want emotionally and practically, separate them.
- Output valid JSON only.

Client message:
{msg}"""

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        out = _extract_json(raw)
        if isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage1 structured extraction failed: %s", exc)
    return {}


def _generate_followup(
    intake_state: dict,
    client_message: str,
    conversation_context: str,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()
    category = intake_state.get("category") or "general"
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"])
    key_facts = cat_cfg.get("key_facts_needed", [])
    known_facts = intake_state.get("known_facts") or []

    known_facts_summary = "\n".join(f"- {f}" for f in known_facts[-10:]) if known_facts else "(nothing specific established yet)"
    category_key_facts = "\n".join(f"- {f}" for f in key_facts) if key_facts else "(general facts needed)"

    prompt = (
        STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM.replace("{known_facts_summary}", known_facts_summary)
        .replace("{category_key_facts}", category_key_facts)
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("generate_followup", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage1 followup LLM failed: %s", exc)

    return "Thank you for sharing that. Could you tell me a little more so I can fully understand what happened?"


def _highest_priority_risk(intake_state: dict) -> str | None:
    priority = [
        "immediate_physical_danger",
        "child_safety_risk",
        "medical_emergency",
        "arrest_or_custody_risk",
        "same_day_deadline",
        "housing_lockout",
        "ongoing_contact_risk",
    ]
    flags = set(intake_state.get("risk_flags") or [])
    for item in priority:
        if item in flags:
            return item
    return None


def _generate_triage_reply(intake_state: dict) -> str | None:
    risk = _highest_priority_risk(intake_state)
    if not risk and intake_state.get("urgency_signal") != "immediate":
        return None

    templates = {
        "immediate_physical_danger": "Your immediate safety matters most right now. Before anything else, are you safe at this moment?",
        "child_safety_risk": "I’m most concerned about immediate safety right now. Are any children with you currently in danger or without a safe place to stay?",
        "medical_emergency": "Your wellbeing comes first here. Do you need urgent medical care or help documenting injuries right now?",
        "arrest_or_custody_risk": "This may need urgent action before we go further. Has the police process already started, or is arrest being threatened right now?",
        "same_day_deadline": "This sounds time-sensitive, so I want to protect your position first. What is the exact hearing, filing, or deadline you are facing today?",
        "housing_lockout": "Housing safety comes first here. Are you currently locked out or without a safe place to stay tonight?",
        "ongoing_contact_risk": "I want to make sure immediate protection is not being missed. Are you still in direct contact with the person causing this problem right now?",
    }
    return templates.get(risk) or (
        "Before we continue with the details, I want to make sure no urgent risk is being missed. What needs immediate attention right now?"
    )


def _check_readiness(intake_state: dict, conversation_context: str, model_override: str | None = None) -> tuple[bool, list[str]]:
    t0 = time.perf_counter()
    turn_count = intake_state.get("turn_count", 0)
    if turn_count < 2:
        return False, ["need at least 2 substantive client turns before advancing"]

    missing: list[str] = []
    if not (intake_state.get("known_facts") or intake_state.get("issue_summary")):
        missing.append("core events")
    if not intake_state.get("relationship_to_other_party") or str(intake_state.get("relationship_to_other_party")).strip().lower() == "unknown":
        missing.append("parties and relationship")
    if not intake_state.get("client_goal_initial"):
        missing.append("what the client wants")
    if _normalize_timeframe_label(str(intake_state.get("timeframe_status") or "")) == "unknown":
        missing.append("general timeframe")
    if missing:
        return False, missing

    prompt = (
        STAGE1_READINESS_CHECK_SYSTEM.replace(
            "{intake_state_json}", json.dumps(intake_state, ensure_ascii=False, indent=2)
        ).replace("{conversation_context}", conversation_context or "")
    )

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("check_readiness", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            ready = bool(out.get("ready_for_stage2", False))
            llm_missing = [str(x).strip() for x in (out.get("missing_critical") or []) if str(x).strip()]
            return ready, llm_missing
    except Exception as exc:
        logger.warning("Stage1 readiness check LLM failed: %s", exc)

    return True, []


def _update_known_facts(intake_state: dict, client_message: str, model_override: str | None = None) -> None:
    t0 = time.perf_counter()
    msg = (client_message or "").strip()
    if not msg:
        return

    existing = set(str(f).lower().strip() for f in (intake_state.get("known_facts") or []))
    fact_records = intake_state.setdefault("fact_records", [])

    try:
        extraction = _extract_stage1_updates(msg, intake_state, model_override=model_override)
        _t("update_known_facts", t0)
        for item in (extraction.get("facts") or []):
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact") or "").strip()
            if not fact or len(fact) <= 8:
                continue
            fact_key = fact.lower()
            if fact_key not in existing:
                existing.add(fact_key)
                intake_state.setdefault("known_facts", []).append(fact)
            if not any(str(r.get("fact") or "").strip().lower() == fact_key for r in fact_records):
                fact_records.append(
                    {
                        "fact": fact,
                        "source_turn": intake_state.get("turn_count", 0),
                        "fact_type": str(item.get("fact_type") or "other").strip()[:40],
                        "time_reference": item.get("time_reference"),
                        "evidence_hook": item.get("evidence_hook"),
                        "witness_hook": item.get("witness_hook"),
                        "confidence_seed": str(item.get("confidence_seed") or "stated").strip()[:20],
                    }
                )

        rel = extraction.get("relationship_to_other_party")
        if rel and str(rel).strip().lower() != "unknown":
            intake_state["relationship_to_other_party"] = str(rel).strip()[:80]

        timeframe = extraction.get("timeframe_status")
        if timeframe:
            intake_state["timeframe_status"] = _normalize_timeframe_label(str(timeframe))

        client_goal = extraction.get("client_goal_initial")
        if client_goal and not intake_state.get("client_goal_initial"):
            intake_state["client_goal_initial"] = str(client_goal).strip()[:300]

        emotional_ask = extraction.get("emotional_ask")
        if emotional_ask and not intake_state.get("emotional_ask"):
            intake_state["emotional_ask"] = str(emotional_ask).strip()[:300]

        immediate_need = extraction.get("immediate_need")
        if immediate_need and not intake_state.get("immediate_need"):
            intake_state["immediate_need"] = str(immediate_need).strip()[:300]

        intake_state["risk_flags"] = _dedupe_keep_order(
            list(intake_state.get("risk_flags") or []) + list(extraction.get("risk_flags") or []),
            cap=8,
        )
        intake_state["known_facts"] = (intake_state.get("known_facts") or [])[:20]
        intake_state["fact_records"] = fact_records[:25]
    except Exception as exc:
        logger.warning("Stage1 fact extraction failed: %s", exc)
        short = msg[:200].rstrip()
        if short.lower() not in existing:
            intake_state.setdefault("known_facts", []).append(short)
            intake_state["known_facts"] = intake_state["known_facts"][:20]


_IMMEDIATE_SIGNALS = (
    "beaten",
    "hit",
    "assault",
    "arrested",
    "arrest",
    "locked out",
    "lockout",
    "thrown out",
    "evict",
    "evicted",
    "threatened",
    "threat",
    "danger",
    "hospital",
    "injury",
    "injured",
    "abuse right now",
    "happening now",
    "court today",
    "hearing today",
    "deadline today",
    "auction today",
    "sale today",
    "running out of time",
)

_RISK_KEYWORD_MAP = {
    "immediate_physical_danger": ("beaten", "hit", "assault", "threatened", "danger", "weapon"),
    "child_safety_risk": ("child", "children", "minor", "baby"),
    "medical_emergency": ("hospital", "injury", "injured", "bleeding", "medical"),
    "arrest_or_custody_risk": ("arrested", "arrest", "custody", "police coming", "police picked"),
    "same_day_deadline": ("court today", "hearing today", "deadline today", "auction today", "sale today"),
    "housing_lockout": ("locked out", "lockout", "thrown out", "evict", "evicted", "no place to stay"),
    "ongoing_contact_risk": ("happening now", "right now", "still with", "still living with"),
}


def _recheck_urgency(intake_state: dict, client_message: str) -> None:
    low = (client_message or "").lower()
    if any(sig in low for sig in _IMMEDIATE_SIGNALS):
        intake_state["urgency_signal"] = "immediate"
    risk_flags = list(intake_state.get("risk_flags") or [])
    for label, keywords in _RISK_KEYWORD_MAP.items():
        if any(word in low for word in keywords) and label not in risk_flags:
            risk_flags.append(label)
    intake_state["risk_flags"] = risk_flags[:8]


def _backfill_stage1_fields(intake_state: dict) -> None:
    facts_text = " ".join(intake_state.get("known_facts") or []).lower()
    if not intake_state.get("timeframe_status"):
        intake_state["timeframe_status"] = _normalize_timeframe_label(facts_text)
    if not intake_state.get("client_goal_initial"):
        for marker in ("want ", "need ", "seeking ", "looking for "):
            idx = facts_text.find(marker)
            if idx >= 0:
                intake_state["client_goal_initial"] = facts_text[idx : idx + 180].strip()
                break


def _compute_stage1_open_questions(intake_state: dict) -> list[str]:
    gaps: list[str] = []
    if not (intake_state.get("known_facts") or intake_state.get("issue_summary")):
        gaps.append("what happened")
    if not intake_state.get("relationship_to_other_party") or str(intake_state.get("relationship_to_other_party")).lower() == "unknown":
        gaps.append("relationship to the other side")
    if _normalize_timeframe_label(str(intake_state.get("timeframe_status") or "")) == "unknown":
        gaps.append("rough timeframe")
    if not intake_state.get("client_goal_initial"):
        gaps.append("what outcome matters most to the client")
    if intake_state.get("urgency_signal") == "immediate" and not intake_state.get("immediate_need"):
        gaps.append("what needs immediate attention")

    category = intake_state.get("category") or "general"
    cat_cfg = LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"])
    facts_text = " ".join(intake_state.get("known_facts") or []).lower()
    for item in cat_cfg.get("key_facts_needed", [])[:6]:
        tokens = [tok for tok in re.findall(r"[a-zA-Z]{4,}", item.lower()) if tok not in {"what", "their", "from", "with"}]
        if tokens and not any(tok in facts_text for tok in tokens[:2]):
            gaps.append(item)

    return _dedupe_keep_order(gaps, cap=10)


def process_turn(session: dict, user_message: str, model_override: str | None = None) -> dict:
    msg = (user_message or "").strip()
    intake_state: dict = session.get("intake_state") or _fresh_state()
    intake_state["turn_count"] = intake_state.get("turn_count", 0) + 1
    context = _build_context(session, max_turns=6)

    if intake_state.get("category") is None or intake_state.get("category_confidence") in (None, "low", "medium"):
        detected = _detect_category(msg, context, model_override=model_override)
        intake_state["category"] = detected["category"]
        intake_state["secondary_issue_clusters"] = _dedupe_keep_order(
            list(detected.get("secondary_categories") or []) + list(intake_state.get("secondary_issue_clusters") or []),
            cap=2,
        )
        intake_state["category_confidence"] = detected["confidence"]
        intake_state["issue_summary"] = detected.get("issue_summary") or intake_state.get("issue_summary") or ""
        intake_state["jurisdiction"] = detected.get("jurisdiction") or intake_state.get("jurisdiction") or "unknown"
        intake_state["urgency_signal"] = detected.get("urgency_signal") or intake_state.get("urgency_signal") or "unknown"
        intake_state["client_role"] = detected.get("client_role") or intake_state.get("client_role") or "unknown"
        intake_state["other_party"] = detected.get("other_party") or intake_state.get("other_party") or "unknown"
        if detected.get("relationship_to_other_party") and detected.get("relationship_to_other_party") != "unknown":
            intake_state["relationship_to_other_party"] = detected.get("relationship_to_other_party")
        intake_state["risk_flags"] = _dedupe_keep_order(
            list(intake_state.get("risk_flags") or []) + list(detected.get("risk_flags") or []),
            cap=8,
        )

        logger.info(
            "Stage1 category locked: %s (confidence=%s, urgency=%s)",
            intake_state["category"],
            intake_state["category_confidence"],
            intake_state["urgency_signal"],
        )

    _recheck_urgency(intake_state, msg)
    _update_known_facts(intake_state, msg, model_override=model_override)
    _backfill_stage1_fields(intake_state)
    intake_state["open_questions"] = _compute_stage1_open_questions(intake_state)

    reply = _generate_triage_reply(intake_state) or _generate_followup(
        intake_state, msg, context, model_override=model_override
    )

    ready, missing = _check_readiness(intake_state, context, model_override=model_override)
    intake_state["ready_for_stage2"] = ready
    if missing:
        intake_state["open_questions"] = _dedupe_keep_order(
            list(intake_state.get("open_questions") or []) + list(missing),
            cap=10,
        )

    logger.info(
        "Stage1 turn %d complete | category=%s | facts=%d | ready=%s | urgency=%s",
        intake_state["turn_count"],
        intake_state.get("category"),
        len(intake_state.get("known_facts") or []),
        ready,
        intake_state.get("urgency_signal"),
    )

    return {
        "reply": reply,
        "intake_state": intake_state,
        "advance_to_stage2": ready,
        "urgency_signal": intake_state.get("urgency_signal", "unknown"),
    }


def get_category_framework(category: str) -> dict:
    return dict(LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"]))


def new_session() -> dict:
    return {
        "history": [],
        "intake_state": _fresh_state(),
    }
