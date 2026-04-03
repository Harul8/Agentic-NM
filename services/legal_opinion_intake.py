"""
Legal Opinion Intake — Stage 1: Opening + Issue Identification

Handles the very first phase of the fully autonomous legal opinion workflow:

  Turn 0  : AI sends a warm opening — no question, just an invitation to speak.
  Turn 1+ : Client shares their situation. AI:
              a) Detects the legal category (domestic_violence / property / etc.)
              b) Locks the relevant bare acts + procedural checklist
              c) Confirms the category subtly (no legal labels) and asks the
                 single most important missing fact.
  Readiness: After each turn, checks whether enough is known to advance to
              Stage 2 (structured category-specific deep-dive).

Public API
----------
  generate_opening()                     → str   (first message to client)
  process_turn(session, user_message)    → dict  {reply, intake_state, advance_to_stage2}

Session state is a plain dict (caller owns persistence).
Import this module from the API endpoint / chat handler that deals with
legal opinion requests — do NOT import from fact_collector.py for this path.
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
    from llm.ollama_client import ask_llm
    return ask_llm(prompt, task_hint=task_hint) or ""


def _ask_llm_quality(prompt: str) -> str:
    return _ask_llm(prompt, task_hint="quality")


# ---------------------------------------------------------------------------
# Prompt imports
# ---------------------------------------------------------------------------

from prompts.advocate_prompts import (
    LEGAL_OPINION_OPENING_SYSTEM,
    ISSUE_CATEGORY_DETECT_SYSTEM,
    STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM,
    STAGE1_READINESS_CHECK_SYSTEM,
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

def _build_context(session: dict, max_turns: int = 6, max_chars: int = 260) -> str:
    """Build a compact conversation history block for prompt injection."""
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

    Returns a dict with keys: category, confidence, issue_summary,
    jurisdiction_hint, urgency_signal, client_role, other_party.
    Falls back to 'general' on failure.
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
        "category": "general",
        "confidence": "low",
        "issue_summary": "",
        "jurisdiction_hint": "unknown",
        "urgency_signal": "unknown",
        "client_role": "unknown",
        "other_party": "unknown",
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("detect_category", t0)
        out = _extract_json(raw)
        if not out or not isinstance(out, dict):
            return _DEFAULT

        # Validate category
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

        return {
            "category":        cat,
            "confidence":      confidence,
            "issue_summary":   str(out.get("issue_summary") or "").strip()[:300],
            "jurisdiction":    str(out.get("jurisdiction_hint") or "unknown").strip(),
            "urgency_signal":  urgency,
            "client_role":     str(out.get("client_role") or "unknown").strip(),
            "other_party":     str(out.get("other_party") or "unknown").strip()[:120],
        }

    except Exception as exc:
        logger.warning("Stage1 category detection failed: %s", exc)
        return _DEFAULT


# ===========================================================================
# Internal: determine next question hint
# ===========================================================================

def _next_question_hint(intake_state: dict) -> str:
    """
    Given the current intake state, return a plain-language hint for what
    the AI should ask next. Picks the first still-open critical fact from
    the category's key_facts_needed list.
    """
    category = intake_state.get("category") or "general"
    cat_cfg  = LEGAL_ISSUE_CATEGORIES.get(category, LEGAL_ISSUE_CATEGORIES["general"])
    key_facts = cat_cfg.get("key_facts_needed", [])
    known_blob = " ".join([
        str(x).lower()
        for x in (intake_state.get("known_facts") or [])
    ])

    # Simple signal map: which key_facts topics are already covered
    _COVERAGE_SIGNALS: dict[str, tuple[str, ...]] = {
        "nature and timeline of abuse":          ("abuse", "hit", "beat", "slap", "shout", "threat", "years", "months"),
        "shared household status":               ("house", "home", "living", "flat", "reside", "stay"),
        "children":                              ("child", "children", "son", "daughter", "kid"),
        "evidence":                              ("photo", "message", "whatsapp", "medical", "witness", "record", "fir"),
        "prior complaints":                      ("fir", "complaint", "police", "report"),
        "income / financial":                    ("income", "salary", "earning", "money", "rupee", "financial"),
        "immediate safety":                      ("safe", "afraid", "fear", "danger", "risk", "shelter"),
        "date and type of marriage":             ("married", "marriage", "wedding", "court marriage"),
        "grounds":                               ("cruelty", "desertion", "divorce", "separation"),
        "income of both parties":                ("income", "salary", "earning", "job"),
        "stridhan":                              ("jewellery", "gold", "stridhan", "dowry"),
        "nature of property":                    ("land", "flat", "house", "ancestral", "inherited", "property"),
        "title documents":                       ("document", "deed", "registered", "patta", "title"),
        "current possession":                    ("possession", "living", "locked", "access"),
        "offence alleged":                       ("fir", "arrest", "accused", "charge", "offence", "crime"),
        "fir number":                            ("fir number", "fir no", "case number"),
        "custody / bail":                        ("bail", "custody", "jail", "remand", "arrested"),
        "nature of employment":                  ("permanent", "contract", "probation", "confirmed", "job type"),
        "employer type":                         ("government", "private", "psu", "public sector"),
        "notice / termination order":            ("notice", "termination", "dismissal", "show cause", "chargesheet"),
        "service duration":                      ("years", "months", "joined", "since", "service period"),
        "product or service":                    ("product", "service", "builder", "bank", "insurance", "telecom"),
        "deficiency":                            ("defective", "wrong", "not delivered", "damaged", "fraud"),
        "amount paid":                           ("paid", "amount", "rupee", "lakh", "cost"),
        "date of accident":                      ("accident", "date", "when"),
        "injuries":                              ("injured", "injury", "fracture", "surgery", "hospital", "dead", "died"),
        "vehicle / insurance":                   ("vehicle", "car", "bike", "truck", "insurance"),
        "cheque amount and date":                ("cheque", "amount", "date", "issued"),
        "dishonour":                             ("bounced", "dishonoured", "returned", "insufficiency"),
        "demand notice":                         ("notice", "demand notice", "sent notice"),
        "survey / khasra":                       ("survey", "khasra", "extent", "acres", "area"),
        "purpose of acquisition":               ("government", "road", "highway", "dam", "project"),
        "compensation offered":                  ("compensation", "amount", "offered", "market value"),
    }

    for fact_desc in key_facts:
        # Find the first fact not yet covered
        fact_low = fact_desc.lower()
        covered = False
        for signal_key, signals in _COVERAGE_SIGNALS.items():
            if signal_key in fact_low or fact_low in signal_key:
                if any(sig in known_blob for sig in signals):
                    covered = True
                    break
        if not covered:
            return f"Ask about: {fact_desc}"

    # All key facts seem covered — ask for general confirmation
    return "Ask whether there is anything else important about the situation they haven't mentioned yet."


# ===========================================================================
# Internal: confirm and follow up
# ===========================================================================

def _generate_followup(
    intake_state: dict,
    client_message: str,
    conversation_context: str,
) -> str:
    """
    Generate the AI's reply after the category is detected:
    - Reflect back warmly (no legal labels)
    - Ask the single most important missing fact
    """
    t0 = time.perf_counter()
    next_hint = _next_question_hint(intake_state)

    prompt = (
        STAGE1_CONFIRM_AND_FOLLOWUP_SYSTEM
        .replace("{next_question_hint}", next_hint)
        .replace("{conversation_context}", conversation_context or "(first message)")
        .replace("{client_message}", (client_message or "").strip())
    )

    try:
        reply = _ask_llm_quality(prompt).strip()
        _t("generate_followup", t0)
        if reply and len(reply) > 20 and "?" in reply:
            return reply
    except Exception as exc:
        logger.warning("Stage1 followup LLM failed: %s", exc)

    # Minimal safe fallback — always ends with a question
    hint = next_hint.replace("Ask about: ", "").replace("Ask whether ", "")
    return f"Thank you for sharing that. Could you tell me more about {hint}?"


# ===========================================================================
# Internal: Stage 1 readiness check
# ===========================================================================

def _check_readiness(intake_state: dict, conversation_context: str) -> tuple[bool, list[str]]:
    """
    Ask the LLM whether enough is known to advance to Stage 2.

    Returns (ready: bool, missing_critical: list[str]).
    """
    t0 = time.perf_counter()

    # Fast heuristic first — skip LLM call if obviously not ready
    turn_count = intake_state.get("turn_count", 0)
    if turn_count < 2:
        return False, ["need at least 2 substantive client turns before advancing"]

    prompt = (
        STAGE1_READINESS_CHECK_SYSTEM
        .replace("{intake_state_json}", json.dumps(intake_state, ensure_ascii=False, indent=2))
        .replace("{conversation_context}", conversation_context or "")
    )

    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("check_readiness", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            ready   = bool(out.get("ready_for_stage2", False))
            missing = [str(x).strip() for x in (out.get("missing_critical") or []) if str(x).strip()]
            return ready, missing
    except Exception as exc:
        logger.warning("Stage1 readiness check LLM failed: %s", exc)

    return False, []


# ===========================================================================
# Internal: update known_facts from latest client message
# ===========================================================================

def _update_known_facts(intake_state: dict, client_message: str) -> None:
    """
    Append meaningful facts from the client's latest message into known_facts.
    Keeps the list deduplicated and capped at 12 entries.

    Uses a lightweight LLM call to extract atomic facts.
    """
    t0 = time.perf_counter()
    msg = (client_message or "").strip()
    if not msg:
        return

    prompt = (
        "Extract 1–4 short, specific facts from the client's message below.\n"
        "Each fact should be one sentence. Use only what the client said — no inference.\n"
        "Output ONLY a JSON array of strings: [\"fact 1\", \"fact 2\"]\n\n"
        f"CLIENT MESSAGE:\n{msg}"
    )

    existing = set(str(f).lower().strip() for f in (intake_state.get("known_facts") or []))

    try:
        raw = _ask_llm(prompt, task_hint="fast")
        _t("update_known_facts", t0)
        # Extract JSON array
        text = (raw or "").strip()
        start = text.find("[")
        end   = text.rfind("]")
        if start >= 0 and end > start:
            facts = json.loads(text[start:end + 1])
            if isinstance(facts, list):
                for fact in facts:
                    fact = str(fact).strip()
                    if fact and fact.lower() not in existing and len(fact) > 10:
                        existing.add(fact.lower())
                        intake_state["known_facts"].append(fact)
                intake_state["known_facts"] = intake_state["known_facts"][:12]
    except Exception as exc:
        logger.warning("Stage1 fact extraction failed: %s", exc)
        # Fallback: just store the raw message as a single fact entry
        short = msg[:200].rstrip()
        if short.lower() not in existing:
            intake_state["known_facts"].append(short)
            intake_state["known_facts"] = intake_state["known_facts"][:12]


# ===========================================================================
# Internal: urgency escalation check
# ===========================================================================

_IMMEDIATE_SIGNALS = (
    "beaten", "hit", "assault", "arrested", "arrest", "locked out", "lockout",
    "thrown out", "evict", "evicted", "threatened", "threat", "danger",
    "hospital", "injury", "injured", "abuse right now", "happening now",
    "court today", "hearing today", "deadline today", "auction today",
    "sale today", "running out of time",
)


def _recheck_urgency(intake_state: dict, client_message: str) -> None:
    """Escalate urgency_signal to 'immediate' if the message contains strong distress signals."""
    if intake_state.get("urgency_signal") == "immediate":
        return
    low = (client_message or "").lower()
    if any(sig in low for sig in _IMMEDIATE_SIGNALS):
        intake_state["urgency_signal"] = "immediate"


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
    intake_state: dict = session.get("intake_state") or _fresh_state()

    # Increment turn counter
    intake_state["turn_count"] = intake_state.get("turn_count", 0) + 1

    # ── Build conversation context for prompts ───────────────────────────────
    context = _build_context(session, max_turns=6)

    # ── Category detection (run on every turn until confidence=high) ─────────
    if (
        intake_state.get("category") is None
        or intake_state.get("category_confidence") in (None, "low")
    ):
        detected = _detect_category(msg, context)
        intake_state["category"]            = detected["category"]
        intake_state["category_confidence"] = detected["confidence"]
        intake_state["issue_summary"]       = detected.get("issue_summary") or intake_state.get("issue_summary") or ""
        intake_state["jurisdiction"]        = detected.get("jurisdiction") or intake_state.get("jurisdiction") or "unknown"
        intake_state["urgency_signal"]      = detected.get("urgency_signal") or intake_state.get("urgency_signal") or "unknown"
        intake_state["client_role"]         = detected.get("client_role") or intake_state.get("client_role") or "unknown"
        intake_state["other_party"]         = detected.get("other_party") or intake_state.get("other_party") or "unknown"

        # Lock the key_facts_needed checklist for this category
        cat_cfg = LEGAL_ISSUE_CATEGORIES.get(intake_state["category"], LEGAL_ISSUE_CATEGORIES["general"])
        intake_state["open_questions"] = list(cat_cfg.get("key_facts_needed", []))

        logger.info(
            "Stage1 category locked: %s (confidence=%s, urgency=%s)",
            intake_state["category"],
            intake_state["category_confidence"],
            intake_state["urgency_signal"],
        )

    # ── Urgency recheck (safety net for escalating distress signals) ──────────
    _recheck_urgency(intake_state, msg)

    # ── Extract and accumulate facts from client message ─────────────────────
    _update_known_facts(intake_state, msg)

    # ── Generate the AI's reply ───────────────────────────────────────────────
    reply = _generate_followup(intake_state, msg, context)

    # ── Readiness check — decide whether to advance to Stage 2 ───────────────
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
        "Stage1 turn %d complete | category=%s | facts=%d | ready=%s | urgency=%s",
        intake_state["turn_count"],
        intake_state.get("category"),
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
    Return the locked bare-act + procedural framework for a detected category.
    Used by Stage 2 to seed its structured deep-dive.
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
