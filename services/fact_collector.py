"""
Fact Collection Service — Adaptive advocate-style intake.

Phase 2 rewrite:
- Uses GREETING_PHRASES from prompts (Indian language support)
- Adaptive: skips questions when user already provided enough detail
- All user-facing messages from LLM — no hardcoded responses
- Falls back to research (not dead-end) if LLM parsing fails
"""

import json
import logging
import os
import re
import time

from llm.config import OLLAMA_MODEL, OLLAMA_MODEL_FAST
from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    GREETING_PHRASES,
    INTAKE_STATE_UPDATE_SYSTEM,
    NEXT_QUESTION_FROM_STATE_SYSTEM,
    STOP_PHRASES,
)

logger = logging.getLogger(__name__)

# Set PIPELINE_TIMING=1 to log elapsed ms for each step (debug slow follow-ups)
_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")
_ENABLE_INTAKE_FEWSHOT = os.environ.get("ENABLE_INTAKE_FEWSHOT", "1").lower() in ("1", "true", "yes")

# Static greeting for exact match (target <100 ms; no LLM call)
GREETING_STATIC_TEMPLATE = (
    "Hi! Tell me about your legal issue and I'll help you find relevant laws and cases."
)


def _log_fc_step(step_name: str, elapsed_ms: float, extra: str = "") -> None:
    if _PIPELINE_TIMING:
        msg = f"PIPELINE_TIMING fact_collector.{step_name}: {elapsed_ms:.0f} ms"
        if extra:
            msg += f" | {extra}"
        logger.info(msg)


# Legal keywords used to distinguish greetings from legal queries
_LEGAL_KEYWORDS = (
    "law", "act", "section", "case", "court", "judgment", "judgement",
    "legal", "advice", "sue", "file", "right", "compensation", "land",
    "property", "contract", "agreement", "bail", "fir", "police",
    "divorce", "custody", "maintenance", "tenant", "landlord", "eviction",
    "cheque", "bounce", "fraud", "theft", "murder", "ipc", "crpc", "cpc",
    "bnss", "bns", "bsa",  # new criminal codes
    "petition", "writ", "appeal", "tribunal", "arbitration",
    # Plain-language legal distress terms that often appear before formal legal words
    "harass", "harassment", "beat", "beating", "abuse", "assault", "violence",
    "threat", "threaten", "terrorise", "terrorize", "injury", "injured",
    "husband", "wife", "children", "child", "daughter", "son", "dowry",
    "separate", "separation", "protection",
)


def is_stop_signal(user_message: str) -> bool:
    """Check if user is signaling they have no more information."""
    msg = user_message.strip().lower()
    return any(phrase in msg for phrase in STOP_PHRASES)


def is_greeting(msg: str, conversation_history: list = None) -> bool:
    """
    True if the message is a greeting / small talk with no legal content.
    Uses the expanded GREETING_PHRASES list (English + Indian languages).
    
    If conversation_history exists and has legal content, short answers are NOT greetings
    (they're follow-up answers to questions).
    """
    m = (msg or "").strip().lower()
    if len(m) > 100:
        return False  # Long messages are substantive

    # Exact match → static template (no LLM); target <100 ms
    cleaned = m.rstrip("!?.,;:")
    if cleaned in GREETING_PHRASES:
        return True

    # If there's conversation history with legal keywords, short answers are likely follow-ups, not greetings
    if conversation_history:
        has_legal_context = any(
            any(kw in (m.get("content", "") or "").lower() for kw in _LEGAL_KEYWORDS)
            for m in conversation_history
            if m.get("role") == "user"
        )
        if has_legal_context:
            return False  # Short answer in legal conversation = follow-up, not greeting

    # Short message with no legal keywords (only if no prior legal context)
    if len(m) < 30 and not any(kw in m for kw in _LEGAL_KEYWORDS):
        return True

    return False


def generate_greeting_response(user_message: str) -> str:
    """Return a deterministic greeting so simple hellos never need an LLM call."""
    msg = (user_message or "").strip().lower()
    if any(word in msg for word in ("good morning",)):
        greeting = "Good morning!"
    elif any(word in msg for word in ("good afternoon",)):
        greeting = "Good afternoon!"
    elif any(word in msg for word in ("good evening", "good night")):
        greeting = "Good evening!"
    elif any(word in msg for word in ("thanks", "thank you", "dhanyavaad", "shukriya", "nandri", "dhonnobad", "aabhar")):
        greeting = "You're welcome."
    elif any(word in msg for word in ("bye", "goodbye", "see you", "alvida")):
        greeting = "Take care."
    else:
        greeting = "Hello!"
    return f"{greeting} I'm here to help with legal research, relevant laws, and next-step legal guidance. Tell me what happened."


# ---------------------------------------------------------------------------
# JSON extraction from LLM output (handles reasoning text, markdown, etc.)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text that may contain reasoning, markdown, etc."""
    text = (text or "").strip()
    # Try extracting from markdown code blocks first
    if "```" in text:
        parts = text.split("```")
        for p in parts[1:]:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if "{" in p and "}" in p:
                text = p
                break
    # Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Try raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: find a line that looks like JSON with "action"
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if line.startswith("{") and "action" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Intake deduplication helpers — prevent the model from repeating questions
# ---------------------------------------------------------------------------

# Keywords whose co-occurrence in two questions signals they are about the same topic.
_DEDUP_TOPIC_KEYWORDS: frozenset[str] = frozenset({
    "prayer", "relief", "outcome", "want", "seeking", "hope", "wish",
    "maintenance", "custody", "protection order", "residence order",
    "injunction", "compensation", "damages", "fir", "police report", "complaint",
    "arrest", "bail", "charge",
    "injury", "injured", "hurt", "wound", "hospital", "doctor", "medical", "treatment",
    "weapon", "knife", "rod", "stick", "object", "used",
    "evidence", "witness", "proof", "document", "certificate",
    "income", "salary", "earning", "financial", "money", "rupee", "lakh", "amount",
    "location", "state", "city", "district", "place", "where",
    "when", "date", "time", "how long", "since when", "duration", "frequency",
    "regularly", "often", "pattern", "specific time", "specific times",
    "property", "house", "land", "flat", "deed", "ownership", "possession",
    "agreement", "contract", "written", "registered",
    "children", "child", "son", "daughter", "minor",
    "dowry", "jewellery", "gold", "stridhan",
    "employer", "employment", "job", "termination", "notice",
    "cheque", "dishonour", "bounce", "payment",
    # Acquisition / land-value terms added to close gaps
    "claim", "claiming", "basis", "reason", "purpose", "total", "specific",
    "market", "value", "per acre", "acre", "area", "extent",
    "request", "requesting", "clarify", "share", "tell", "provide",
})


def _extract_asked_questions(conversation_history: list) -> list[str]:
    """
    Pull every sentence that ends with '?' from all assistant turns in the conversation.
    Returns a flat list of question strings.
    """
    questions: list[str] = []
    for msg in conversation_history:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Split on sentence boundaries; keep sentences that contain '?'
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            s = sentence.strip()
            if "?" in s and len(s) > 12:
                questions.append(s)
    return questions


def _compact_conversation_context(conversation_history: list, max_messages: int = 6, max_chars: int = 240) -> str:
    """Build a compact recent-history block for fast intake routing."""
    recent = conversation_history[-max_messages:] if conversation_history else []
    lines: list[str] = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_compact_intake_state_context(conversation_history: list, user_message: str) -> str:
    """Build a compact, high-signal context block for intake state extraction."""
    recent = _compact_conversation_context(conversation_history, max_messages=5, max_chars=200)
    asked_questions = _extract_asked_questions(conversation_history)[-5:]
    user_points: list[str] = []
    seen: set[str] = set()
    for msg in conversation_history:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content or content in seen:
            continue
        seen.add(content)
        user_points.append(content[:220])
    current = (user_message or "").strip().replace("\n", " ")
    if current and current not in seen:
        user_points.append(current[:220])

    lines = []
    if recent:
        lines.append("RECENT CONVERSATION:")
        lines.append(recent)
    if user_points:
        lines.append("")
        lines.append("USER STATEMENTS SO FAR:")
        for i, item in enumerate(user_points[-6:], 1):
            lines.append(f"{i}. {item}")
    if asked_questions:
        lines.append("")
        lines.append("QUESTIONS ALREADY ASKED:")
        for i, q in enumerate(asked_questions, 1):
            lines.append(f"{i}. {q[:180]}")
    lines.append("")
    lines.append(f"CURRENT USER MESSAGE: {current}")
    return "\n".join(lines).strip()


def _parse_intake_state(response: str) -> dict | None:
    out = _extract_json(response)
    if not out or not isinstance(out, dict):
        return None
    route = (out.get("route") or "").strip().lower()
    if route not in ("greeting", "generic_chat", "search", "lookup", "legal_opinion"):
        return None
    out["route"] = route
    out["known_facts"] = [str(x).strip() for x in (out.get("known_facts") or []) if str(x).strip()][:6]
    out["prior_actions_taken"] = [str(x).strip() for x in (out.get("prior_actions_taken") or []) if str(x).strip()][:5]
    out["open_points"] = [str(x).strip() for x in (out.get("open_points") or []) if str(x).strip()][:5]
    out["facts_summary"] = (out.get("facts_summary") or "").strip()
    out["enough_to_proceed"] = bool(out.get("enough_to_proceed"))
    return out


def _count_distinct_user_turns(conversation_history: list, user_message: str) -> int:
    """Count distinct substantive user turns seen so far, including the current one."""
    seen: set[str] = set()
    count = 0
    for msg in conversation_history or []:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip().lower()
        if not content or content in seen:
            continue
        seen.add(content)
        count += 1
    current = (user_message or "").strip().lower()
    if current and current not in seen:
        count += 1
    return count


_EVIDENCE_HINTS = (
    "document", "documents", "message", "messages", "whatsapp", "email", "notice",
    "order", "agreement", "contract", "record", "records", "receipt", "payment",
    "bank transfer", "photo", "photos", "video", "audio", "medical", "report",
    "certificate", "witness", "witnesses", "screenshot", "slip", "fir", "complaint",
)

_PRIOR_ACTION_HINTS = (
    "complaint", "fir", "police", "lawyer", "advocate", "notice sent", "replied",
    "representation", "appeal", "application", "petition", "emailed", "wrote",
    "requested", "asked", "called", "visited", "reported", "medical treatment",
    "clinic", "hospital",
)

_STAGE_HINTS = (
    "today", "yesterday", "notice", "order", "hearing", "auction", "termination",
    "dismissal", "chargesheet", "charge sheet", "sale notice", "deadline", "summons",
    "proceeding", "case filed", "complaint filed", "fir", "medical examination",
    "lock changed", "eviction notice", "show cause", "suspension", "arrest",
)

_RELIEF_HINTS = (
    "want", "need", "seeking", "relief", "protection", "custody", "maintenance",
    "compensation", "injunction", "stay", "access", "release", "bail", "quash",
    "set aside", "reinstatement", "refund", "possession", "stop", "restrain",
)

_LOW_SIGNAL_OPEN_POINT_HINTS = (
    "current situation", "next steps", "more details", "further details", "what happened",
    "background", "context", "general information", "immediate legal action",
    "lawyer consultation", "legal strategy",
)

_CANONICAL_OPEN_POINTS = (
    "client objective / relief sought",
    "present urgency / current position",
    "prior actions already taken",
    "documents / messages / witnesses currently available",
    "current stage / notice / immediate trigger",
)

_OPEN_POINT_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "client objective / relief sought": (
        "relief", "want", "need", "seeking", "outcome", "objective", "protection",
        "compensation", "refund", "custody", "maintenance", "injunction", "release",
    ),
    "present urgency / current position": (
        "urgent", "urgency", "right now", "currently", "immediate", "safe", "safety",
        "position", "risk", "access", "possession", "custody", "belongings",
    ),
    "prior actions already taken": (
        "already taken", "steps", "approached", "complaint", "police", "lawyer",
        "advocate", "notice sent", "replied", "representation", "appeal", "application",
    ),
    "documents / messages / witnesses currently available": (
        "document", "documents", "message", "messages", "whatsapp", "email", "record",
        "records", "witness", "witnesses", "photo", "photos", "video", "audio",
        "medical", "papers", "receipt", "payment", "proof",
    ),
    "current stage / notice / immediate trigger": (
        "notice", "order", "hearing", "deadline", "filing", "complaint filed",
        "trigger", "stage", "started", "served", "court order", "summons",
    ),
}

_QUESTION_STOPWORDS = {
    "what", "which", "when", "where", "there", "their", "about", "would", "could",
    "should", "right", "immediate", "present", "current", "already", "taken", "detail",
    "details", "provide", "share", "please", "help", "legal", "issue", "matter",
    "client", "steps", "support", "position", "available", "relief", "outcome",
}

_LEGAL_PROFESSIONAL_STYLE_HINTS = (
    "petition", "writ", "respondent", "petitioner", "tribunal", "article ", "section ",
    "client", "our client", "relief sought", "prayer", "departmental inquiry",
    "assessment order", "show cause", "specific performance", "injunction",
)

_LOW_INFORMATION_REPLY_HINTS = (
    "i don't know", "dont know", "do not know", "not sure", "no idea",
    "unclear", "unknown", "cannot say", "can't say",
)


def _combine_user_messages(conversation_history: list, user_message: str) -> str:
    seen: set[str] = set()
    user_msgs: list[str] = []
    for msg in conversation_history or []:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if content and content not in seen:
            seen.add(content)
            user_msgs.append(content)
    current = (user_message or "").strip()
    if current and current not in seen:
        user_msgs.append(current)
    return "\n".join(user_msgs).strip()


def _has_legal_context(conversation_history: list, user_message: str, force_legal: bool = False) -> bool:
    if force_legal:
        return True
    blob_parts = []
    for msg in conversation_history or []:
        if msg.get("role") in ("user", "assistant"):
            blob_parts.append((msg.get("content") or "").strip().lower())
    blob_parts.append((user_message or "").strip().lower())
    blob = " ".join(part for part in blob_parts if part)
    return any(kw in blob for kw in _LEGAL_KEYWORDS)


def _normalize_intake_route(route: str | None, conversation_history: list, user_message: str, force_legal: bool = False) -> str:
    route = (route or "").strip().lower()
    explicit_intent = _detect_intent_from_keywords(user_message)
    legal_context = _has_legal_context(conversation_history, user_message, force_legal=force_legal)
    if explicit_intent in ("search", "lookup"):
        return explicit_intent
    if route in ("search", "lookup") and not explicit_intent and legal_context:
        return "legal_opinion"
    if route in ("greeting", "generic_chat") and legal_context:
        return "legal_opinion"
    if route in ("greeting", "generic_chat", "search", "lookup", "legal_opinion"):
        return route
    return "legal_opinion" if legal_context else "generic_chat"


def _state_text_blob(intake_state: dict) -> str:
    parts = [
        intake_state.get("client_objective", ""),
        intake_state.get("facts_summary", ""),
        " ".join(intake_state.get("known_facts", []) or []),
        " ".join(intake_state.get("prior_actions_taken", []) or []),
        " ".join(intake_state.get("open_points", []) or []),
    ]
    return " ".join(str(part).strip().lower() for part in parts if str(part).strip())


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    for term in terms:
        pattern = rf"\b{re.escape(term.lower())}\b"
        if re.search(pattern, low):
            return True
    return False


def _append_unique_point(points: list[str], candidate: str) -> None:
    cand = (candidate or "").strip()
    if not cand:
        return
    low = cand.lower()
    for existing in points:
        ex = existing.lower()
        if low == ex or low in ex or ex in low:
            return
    points.append(cand)


def _canonicalize_open_point(
    point: str,
    *,
    objective: str,
    urgency: str,
    prior_actions: list[str],
    evidence_present: bool,
    stage_present: bool,
) -> str:
    cleaned = str(point or "").strip()
    low = cleaned.lower().strip(" .,:;!?")
    if not low:
        return ""

    if any(hint in low for hint in _LOW_SIGNAL_OPEN_POINT_HINTS):
        return ""

    if objective and any(token in low for token in ("objective", "relief", "outcome", "prayer", "remedy")):
        return ""
    if urgency not in ("", "unknown") and any(token in low for token in ("urgency", "current position", "immediate position", "safety")):
        return ""
    if prior_actions and any(token in low for token in ("prior action", "already taken", "steps taken", "complaint", "police", "lawyer", "advocate")):
        return ""
    if evidence_present and any(token in low for token in ("document", "communication", "message", "material", "supporting", "evidence", "witness", "proof", "photo", "record")):
        return ""
    if stage_present and any(token in low for token in ("stage", "notice", "trigger", "process", "hearing", "order", "court order")):
        return ""

    if any(token in low for token in ("want", "need", "seeking", "relief", "remedy", "outcome", "prayer")):
        return "client objective / relief sought"
    if any(token in low for token in ("police", "lawyer", "advocate", "complaint", "fir", "emailed", "wrote", "visited", "called", "approached", "representation", "appeal", "petition", "notice sent", "replied")):
        return "prior actions already taken"
    if any(token in low for token in ("document", "documents", "message", "messages", "whatsapp", "email", "record", "records", "payment", "bank transfer", "receipt", "photo", "photos", "video", "audio", "medical", "witness", "witnesses", "screenshot", "proof", "agreement", "contract")):
        return "documents / messages / witnesses currently available"
    if any(token in low for token in ("notice", "order", "court order", "hearing", "proceeding", "case filed", "complaint filed", "lock changed", "lock", "trigger", "deadline", "termination", "dismissal", "summons", "stage")):
        return "current stage / notice / immediate trigger"
    if any(token in low for token in ("access", "entry", "locked out", "lockout", "belongings", "possession", "safety", "urgent", "urgency", "immediate", "right now", "current position", "still inside")):
        return "present urgency / current position"

    # Question-shaped or legal-conclusion-shaped open points should be mapped to
    # concrete intake buckets instead of being asked back verbatim.
    if low.startswith(("is ", "can ", "whether ", "do we know", "what do we know")) or " illegal" in low or " lawful" in low:
        if any(token in low for token in ("access", "locked out", "lockout", "belongings", "possession")):
            return "present urgency / current position"
        if any(token in low for token in ("notice", "order", "court order", "eviction", "formal step")):
            return "current stage / notice / immediate trigger"
        if any(token in low for token in ("complaint", "police", "lawyer", "advocate", "step")):
            return "prior actions already taken"
        return ""

    return cleaned


def _derive_core_open_points(intake_state: dict, conversation_history: list, user_message: str) -> list[str]:
    points: list[str] = []
    state_blob = _state_text_blob(intake_state)
    objective = (intake_state.get("client_objective") or "").strip()
    urgency = (intake_state.get("urgency_level") or "unknown").strip().lower()
    prior_actions = [str(x).strip() for x in (intake_state.get("prior_actions_taken") or []) if str(x).strip()]
    evidence_present = _contains_any_term(state_blob, _EVIDENCE_HINTS)
    stage_present = _contains_any_term(state_blob, _STAGE_HINTS)

    if not objective and not _contains_any_term(state_blob, _RELIEF_HINTS):
        _append_unique_point(points, "client objective / relief sought")
    if urgency in ("", "unknown"):
        _append_unique_point(points, "present urgency / current position")
    if not prior_actions and not _contains_any_term(state_blob, _PRIOR_ACTION_HINTS):
        _append_unique_point(points, "prior actions already taken")
    if not evidence_present:
        _append_unique_point(points, "documents / messages / witnesses currently available")
    if _count_distinct_user_turns(conversation_history, user_message) <= 2 and not stage_present:
        _append_unique_point(points, "current stage / notice / immediate trigger")

    for point in intake_state.get("open_points", []) or []:
        cleaned = _canonicalize_open_point(
            point,
            objective=objective,
            urgency=urgency,
            prior_actions=prior_actions,
            evidence_present=evidence_present,
            stage_present=stage_present,
        )
        low = cleaned.lower()
        if not cleaned:
            continue
        if any(token in low for token in ("time", "timing", "date", "duration", "frequency", "when exactly")):
            continue
        _append_unique_point(points, cleaned)
        if len(points) >= 5:
            break
    return points[:5]


def _normalize_legal_intake_state(intake_state: dict | None, conversation_history: list, user_message: str) -> dict:
    state = dict(intake_state or {})
    facts_summary = _combine_user_messages(conversation_history, user_message) or (state.get("facts_summary") or "").strip()
    known_facts = [str(x).strip() for x in (state.get("known_facts") or []) if str(x).strip()]
    prior_actions = [
        str(x).strip()
        for x in (state.get("prior_actions_taken") or [])
        if str(x).strip() and str(x).strip().lower() not in {"none", "nil", "nothing", "not yet", "no action", "no action taken"}
    ]
    normalized = {
        "route": "legal_opinion",
        "client_objective": (state.get("client_objective") or "").strip(),
        "urgency_level": (state.get("urgency_level") or "unknown").strip().lower() or "unknown",
        "known_facts": list(dict.fromkeys(known_facts))[:6],
        "prior_actions_taken": list(dict.fromkeys(prior_actions))[:5],
        "facts_summary": facts_summary,
        "enough_to_proceed": False,
        "open_points": [],
    }
    normalized["open_points"] = _derive_core_open_points(normalized | {"open_points": state.get("open_points") or []}, conversation_history, user_message)
    return normalized


def _topic_already_asked(point: str, asked_questions: list[str]) -> bool:
    low = (point or "").lower()
    mapping = {
        "client objective / relief sought": ("relief", "want", "outcome", "seeking", "what do you want"),
        "present urgency / current position": ("urgent", "urgency", "current position", "right now", "immediate risk", "safe"),
        "prior actions already taken": ("already taken", "steps have you already taken", "complaint", "notice sent", "police", "lawyer"),
        "documents / messages / witnesses currently available": ("document", "message", "photo", "witness", "evidence", "records"),
        "current stage / notice / immediate trigger": ("notice", "order", "hearing", "deadline", "stage", "what happened today", "trigger"),
    }
    signals = mapping.get(low, tuple(token for token in re.split(r"[\s/]+", low) if len(token) > 3))
    for question in asked_questions or []:
        q_low = (question or "").lower()
        if any(signal in q_low for signal in signals):
            return True
    return False


def _question_content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
        if tok not in _QUESTION_STOPWORDS
    }


def _infer_question_buckets(text: str) -> set[str]:
    low = (text or "").lower()
    buckets: set[str] = set()
    for bucket, signals in _OPEN_POINT_SIGNAL_MAP.items():
        if any(signal in low for signal in signals):
            buckets.add(bucket)
    return buckets


def _critical_open_points(intake_state: dict) -> list[str]:
    critical: list[str] = []
    seen: set[str] = set()
    for point in intake_state.get("open_points", []) or []:
        if not str(point).strip():
            continue
        buckets = _infer_question_buckets(str(point))
        if buckets:
            for bucket in buckets:
                if bucket not in seen:
                    seen.add(bucket)
                    critical.append(bucket)
            continue
        low = str(point).strip().lower()
        if low not in seen:
            seen.add(low)
            critical.append(low)
    return critical


def _has_analysis_ready_record(intake_state: dict, conversation_history: list, user_message: str) -> bool:
    user_turns = _count_distinct_user_turns(conversation_history, user_message)
    facts_summary = (intake_state.get("facts_summary") or "").strip()
    known_facts = [str(x).strip() for x in (intake_state.get("known_facts") or []) if str(x).strip()]
    objective = (intake_state.get("client_objective") or "").strip()
    urgency = (intake_state.get("urgency_level") or "unknown").strip().lower()
    prior_actions = [str(x).strip() for x in (intake_state.get("prior_actions_taken") or []) if str(x).strip()]
    state_blob = _state_text_blob(intake_state)
    evidence_present = _contains_any_term(state_blob, _EVIDENCE_HINTS)
    stage_present = _contains_any_term(state_blob, _STAGE_HINTS)
    objective_present = bool(objective) or _contains_any_term(state_blob, _RELIEF_HINTS)
    readiness_score = sum((
        user_turns >= 3,
        len(facts_summary) >= 140,
        len(known_facts) >= 3,
        objective_present,
        evidence_present,
        bool(prior_actions) or stage_present,
        urgency not in ("", "unknown"),
    ))
    if readiness_score < 5:
        return False
    critical_remaining = _critical_open_points(intake_state)
    if len(critical_remaining) == 0:
        return True
    if len(critical_remaining) <= 1 and readiness_score >= 6:
        return True
    if len(critical_remaining) <= 2 and readiness_score >= 6 and evidence_present and objective_present and user_turns >= 3:
        return True
    return False




def _infer_tone_profile(facts_blob: str) -> str:
    if _contains_any_term(facts_blob, ("assault", "abuse", "injury", "violence", "beat", "harassment", "threat", "medical", "hospital")):
        return "sensitive"
    if _contains_any_term(facts_blob, ("lock", "locked out", "lockout", "belongings", "possession", "entry", "access", "evict", "auction", "sale notice", "deadline", "termination", "dismissal", "suspension", "arrest", "detention", "custody", "jail", "demolition", "freeze", "seizure")):
        return "urgent"
    return "standard"


def _looks_like_legal_professional(*texts: str) -> bool:
    blob = " ".join((t or "").lower() for t in texts if t)
    return any(hint in blob for hint in _LEGAL_PROFESSIONAL_STYLE_HINTS)


def _conversation_variation_index(conversation_history: list, tone_profile: str, missing_points: list[str], professional: bool) -> int:
    asked_count = len(_extract_asked_questions(conversation_history))
    signature = f"{tone_profile}|{'|'.join(missing_points[:3])}|{asked_count}|{1 if professional else 0}"
    return sum(ord(ch) for ch in signature) % 3


def _pick_variant(options: tuple[str, ...], index: int) -> str:
    if not options:
        return ""
    return options[index % len(options)]


def _infer_position_anchor(facts_blob: str) -> str:
    if _contains_any_term(facts_blob, ("assault", "abuse", "injury", "violence", "threat", "harassment", "medical", "hospital")):
        return "your immediate safety and risk position"
    if _contains_any_term(facts_blob, ("lock", "locked out", "lockout", "entry", "access", "belongings", "possession", "flat", "house", "property")):
        return "your present access or possession position"
    if _contains_any_term(facts_blob, ("arrest", "detention", "custody", "jail", "remand")):
        return "the present custody position"
    if _contains_any_term(facts_blob, ("employer", "employment", "termination", "dismissal", "suspension", "service", "salary")):
        return "your current employment position"
    if _contains_any_term(facts_blob, ("account", "bank", "freeze", "seizure", "attachment")):
        return "the present control over the account or property"
    return "the immediate practical position"


def _infer_action_channels(facts_blob: str) -> str:
    if _contains_any_term(facts_blob, ("assault", "abuse", "violence", "police", "fir", "arrest", "detention", "custody", "jail")):
        return "the police, the other side, any relevant authority, or any lawyer"
    if _contains_any_term(facts_blob, ("employer", "employment", "termination", "dismissal", "suspension", "department", "service")):
        return "the employer, department, any authority, or any lawyer"
    if _contains_any_term(facts_blob, ("landlord", "tenant", "rent", "society", "building")):
        return "the other side, building or society management, any authority, or any lawyer"
    return "the other side, any relevant authority, or any lawyer"


def _infer_evidence_phrase(facts_blob: str) -> str:
    if _contains_any_term(facts_blob, ("payment", "rent", "salary", "invoice", "receipt", "bank", "cheque")):
        return "documents, notices, messages, payment records, photos, recordings, or witness support"
    if _contains_any_term(facts_blob, ("assault", "abuse", "injury", "violence", "medical", "hospital")):
        return "documents, messages, photos, medical papers, recordings, or witness support"
    return "documents, messages, notices, records, photos, recordings, or witness support"


def _infer_stage_fragment(facts_blob: str) -> str:
    if _contains_any_term(facts_blob, ("auction", "sale", "tender", "bid")):
        return "has any notice, order, sale step, filing, hearing, or deadline already started"
    if _contains_any_term(facts_blob, ("employer", "termination", "dismissal", "suspension", "department", "service")):
        return "has any notice, inquiry, order, filing, hearing, or deadline already started"
    return "has any notice, order, complaint, filing, hearing, or deadline already started"


def _build_intake_opening(tone_profile: str, professional: bool, conversation_history: list, missing_points: list[str]) -> str:
    index = _conversation_variation_index(conversation_history, tone_profile, missing_points, professional)
    if tone_profile == "sensitive":
        options = (
            "I am sorry you are dealing with this.",
            "I am sorry this has happened.",
            "I can see this is serious, and I want to understand it carefully.",
        ) if not professional else (
            "I understand the seriousness of the situation.",
            "I can see the matter is serious.",
            "I understand why this needs careful handling.",
        )
        return _pick_variant(options, index)
    if tone_profile == "urgent":
        return _pick_variant((
            "I understand why this feels urgent.",
            "I can see why this needs immediate clarity.",
            "I understand why you need a quick and careful assessment here.",
        ), index)
    return _pick_variant((
        "I understand why this is concerning.",
        "I can see why this needs careful assessment.",
        "I understand why you want clarity on this.",
    ), index)


def _build_intake_reason(tone_profile: str, missing_points: list[str], professional: bool, conversation_history: list) -> str:
    low_points = {str(p).strip().lower() for p in (missing_points or []) if str(p).strip()}
    index = _conversation_variation_index(conversation_history, tone_profile, missing_points, professional)
    if "documents / messages / witnesses currently available" in low_points and "prior actions already taken" in low_points:
        options = (
            "The next details will help me assess the present record and what can be pursued responsibly from here.",
            "The next details will help me understand what has already been done and what material presently supports the matter.",
            "The next details will help me judge the current record and what step is realistically supportable now.",
        )
        return _pick_variant(options, index)
    if "present urgency / current position" in low_points and tone_profile == "urgent":
        return _pick_variant((
            "The next details will help me assess the immediate position, the urgency, and what can realistically be supported right now.",
            "The next details will help me understand the present risk and what immediate step may actually be supportable.",
            "The next details will help me judge the current practical position and what can responsibly be pursued right away.",
        ), index)
    if tone_profile == "sensitive":
        return _pick_variant((
            "The next details will help me assess immediate risk, available support, and what can be properly supported on the record.",
            "The next details will help me understand the immediate safety position, the available support, and the record that already exists.",
            "The next details will help me assess immediate protection needs and what can be responsibly supported on the material available right now.",
        ), index)
    if "current stage / notice / immediate trigger" in low_points:
        return _pick_variant((
            "The next details will help me assess where the matter presently stands and what legal path is realistically open.",
            "The next details will help me understand the current stage and what route is genuinely open from here.",
            "The next details will help me assess the present procedural position and what can responsibly be pursued next.",
        ), index)
    if "client objective / relief sought" in low_points:
        return _pick_variant((
            "The next details will help me assess what relief is realistically supportable on the facts shared so far.",
            "The next details will help me pin down what outcome is realistically supportable on the present record.",
            "The next details will help me assess what relief can be responsibly pursued on the facts you have shared.",
        ), index)
    if professional:
        return _pick_variant((
            "The next details will help me assess urgency, evidentiary support, and what is presently supportable.",
            "The next details will help me assess the present record, the urgency, and what is realistically supportable from here.",
            "The next details will help me judge urgency, supportability, and the present evidentiary position.",
        ), index)
    return _pick_variant((
        "The next details will help me understand the situation properly and assess what can be supported from here.",
        "The next details will help me understand the position more clearly and assess what can realistically be supported.",
        "The next details will help me see the situation more clearly and judge what can responsibly be pursued from here.",
    ), index)


def _is_low_information_reply(user_message: str) -> bool:

    low = (user_message or "").strip().lower()
    if not low:
        return True
    if any(hint in low for hint in _LOW_INFORMATION_REPLY_HINTS):
        return True
    return low in {"idk", "unknown", "not sure"}


def _should_complete_legal_intake(intake_state: dict, conversation_history: list, user_message: str, stop_requested: bool = False) -> bool:
    user_turns = _count_distinct_user_turns(conversation_history, user_message)
    facts_present = bool((intake_state.get("facts_summary") or "").strip())
    if stop_requested:
        return facts_present and (user_turns >= 2 or _has_analysis_ready_record(intake_state, conversation_history, user_message))
    if user_turns < 2:
        return False
    if _has_analysis_ready_record(intake_state, conversation_history, user_message):
        return True
    return len(_critical_open_points(intake_state)) == 0 and facts_present


def _can_proceed_with_partial_record(intake_state: dict, conversation_history: list, user_message: str) -> bool:
    """
    Allow analysis to begin when the record is already strong but the user cannot
    add more on a final narrow point. This prevents dead loops late in intake.
    """
    if not _is_low_information_reply(user_message):
        return False
    user_turns = _count_distinct_user_turns(conversation_history, user_message)
    if user_turns < 3:
        return False
    open_points = [str(x).strip() for x in (intake_state.get("open_points") or []) if str(x).strip()]
    if len(open_points) > 1:
        return False
    return _has_analysis_ready_record(intake_state, conversation_history, user_message)


def _is_low_quality_next_question(reply: str) -> bool:
    low = (reply or "").strip().lower()
    if len(low) < 20:
        return True
    if "?" not in low:
        return True
    banned_fragments = (
        "hello!", "hi!", "tell me what happened", "tell me more", "start from the beginning",
        "what happened", "share your facts", "landlord and tenant act", "section ", "under the act",
        "supreme court", "high court", "article ",
    )
    return any(fragment in low for fragment in banned_fragments)


def _question_is_grounded_in_state(reply: str, intake_state: dict) -> bool:
    state_blob = _state_text_blob(intake_state)
    if not state_blob:
        return True
    reply_low = (reply or "").lower()
    state_buckets = set(_critical_open_points(intake_state))
    question_buckets = _infer_question_buckets(reply_low)
    if question_buckets and state_buckets and question_buckets & state_buckets:
        return True
    reply_tokens = _question_content_tokens(reply_low)
    if not reply_tokens:
        return False
    overlap = {token for token in reply_tokens if token in state_blob}
    if overlap:
        return True
    if len(reply_tokens) <= 2:
        return False
    return False


def _join_question_fragments(fragments: list[str]) -> str:
    if not fragments:
        return ""
    if len(fragments) == 1:
        return f"{fragments[0]}?"
    if len(fragments) == 2:
        return f"{fragments[0]}, and {fragments[1]}?"
    return f"{fragments[0]}, {fragments[1]}, and {fragments[2]}?"


def _custom_open_point_to_fragment(point: str) -> str:
    low = (point or "").strip().lower().rstrip(".")
    if not low:
        return ""
    if low.startswith(("is ", "can ", "whether ", "do we know", "what do we know")) or " illegal" in low or " lawful" in low:
        return ""
    if low.startswith("possibility of ") and low.endswith(" claim"):
        target = low[len("possibility of "):]
        return f"what facts presently support {target}"
    if "intent behind" in low:
        tail = low.split("intent behind", 1)[1].strip()
        if tail:
            return f"what explanation, if any, the other side gave about {tail}"
    if low.startswith("immediacy of "):
        tail = low[len("immediacy of "):].strip()
        if tail:
            return f"whether {tail} is possible right now"
    if low.startswith("protections against "):
        tail = low[len("protections against "):].strip()
        if tail:
            return f"what immediate protection is needed against {tail}"
    if low.startswith("available support"):
        return "what family, local, or institutional support is available to you right now"
    return ""




def _build_fallback_next_question(intake_state: dict, conversation_history: list) -> str:
    asked_questions = _extract_asked_questions(conversation_history)
    open_points = list(intake_state.get("open_points") or [])
    missing_points = []
    for point in open_points:
        if point in _CANONICAL_OPEN_POINTS:
            missing_points.append(point)
            continue
        if not _topic_already_asked(point, asked_questions):
            missing_points.append(point)
    if not missing_points:
        missing_points = open_points

    facts_blob = _state_text_blob(intake_state)
    tone_profile = _infer_tone_profile(facts_blob)
    professional = _looks_like_legal_professional(
        facts_blob,
        " ".join(asked_questions[-3:]),
        intake_state.get("facts_summary", ""),
        intake_state.get("client_objective", ""),
    )
    position_anchor = _infer_position_anchor(facts_blob)
    action_channels = _infer_action_channels(facts_blob)
    evidence_phrase = _infer_evidence_phrase(facts_blob)
    stage_fragment = _infer_stage_fragment(facts_blob)

    fragment_map = {
        "client objective / relief sought": "what immediate result or protection do you want me to focus on first",
        "present urgency / current position": f"right now, what is the current position on {position_anchor}",
        "prior actions already taken": f"what steps have you already taken with {action_channels}",
        "documents / messages / witnesses currently available": f"what {evidence_phrase} can you presently rely on",
        "current stage / notice / immediate trigger": stage_fragment,
    }
    repeat_fragment_map = {
        "client objective / relief sought": "just to pin this down, what immediate result should I focus on first",
        "present urgency / current position": f"right now, what is the immediate practical position on {position_anchor}",
        "prior actions already taken": f"before I assess the next step, what have you already done with {action_channels}",
        "documents / messages / witnesses currently available": f"which specific parts of that material can you actually rely on right now from the {evidence_phrase}",
        "current stage / notice / immediate trigger": f"apart from what you have already mentioned, {stage_fragment}",
    }

    fragments: list[str] = []
    for point in missing_points:
        already_asked = _topic_already_asked(point, asked_questions)
        fragment = repeat_fragment_map.get(point) if already_asked else fragment_map.get(point)
        if not fragment:
            fragment = _custom_open_point_to_fragment(point)
        if fragment and fragment not in fragments:
            fragments.append(fragment)
        if len(fragments) >= 3:
            break
    if not fragments:
        fragments = [
            "what immediate result do you want me to focus on first",
            "what supporting material can you presently rely on",
            "what step has already been taken, if any",
        ]
    if fragments:
        fragments[0] = fragments[0][:1].upper() + fragments[0][1:]
    opening = _build_intake_opening(tone_profile, professional, conversation_history, missing_points)
    reason = _build_intake_reason(tone_profile, missing_points, professional, conversation_history)
    return (
        f"{opening} "
        f"{reason} "
        f"{_join_question_fragments(fragments)}"
    )



def _needs_more_intake_clarification(intake_state: dict, conversation_history: list, user_message: str) -> bool:
    """
    Generic guardrail against premature intake completion.

    The model still chooses the next question, but we do not allow `enough_to_proceed`
    to fire too early when the decision state is still thin.
    """
    if not intake_state or intake_state.get("route") != "legal_opinion":
        return False
    if not intake_state.get("enough_to_proceed"):
        return False

    objective = (intake_state.get("client_objective") or "").strip()
    urgency = (intake_state.get("urgency_level") or "unknown").strip().lower()
    prior_actions = [x for x in (intake_state.get("prior_actions_taken") or []) if str(x).strip()]
    open_points = [x for x in (intake_state.get("open_points") or []) if str(x).strip()]
    user_turns = _count_distinct_user_turns(conversation_history, user_message)

    # Do not stop on the very first substantive legal turn.
    # Even a strong opening narrative usually still needs one decision-critical follow-up.
    if user_turns <= 1:
        return True

    # More generally, if multiple decision-critical uncertainties remain, keep intake open.
    if len(open_points) >= 2:
        return True
    if open_points and (not objective or urgency in ("", "unknown") or not prior_actions):
        return True

    return False


def _run_compact_intake_state(conversation_history: list, user_message: str) -> dict | None:
    """Small model call: extract route + compact legal intake state."""
    context_block = _build_compact_intake_state_context(conversation_history, user_message)
    few_shot_block = ""
    if _ENABLE_INTAKE_FEWSHOT:
        try:
            from training.few_shot_retriever import get_intake_state_example_pack
            # Use only user turns for example retrieval — assistant intake messages
            # ("do you have any documents?", "have you filed a complaint?") contain generic
            # legal terms that create false lexical overlap with unrelated domain examples.
            full_query = " ".join(
                [(m.get("content") or "").strip() for m in conversation_history if m.get("role") == "user" and (m.get("content") or "").strip()]
                + [(user_message or "").strip()]
            ).strip()
            packed = get_intake_state_example_pack(full_query, max_examples=1)
            if packed:
                few_shot_block = f"\n\n{packed}\n"
        except Exception:
            few_shot_block = ""
    prompt = (
        f"{INTAKE_STATE_UPDATE_SYSTEM}"
        f"{few_shot_block}\n\n"
        f"{context_block}\n\n"
        f"Output one line of valid JSON only."
    )
    attempt_specs = [
        ("fast", ""),
        (
            "fast",
            "\n\nREPAIR FEEDBACK:\n- Return valid JSON only.\n- Do not add prose, markdown, code fences, or commentary.\n",
        ),
    ]
    for hint, suffix in attempt_specs:
        try:
            response = ask_llm(f"{prompt}{suffix}", task_hint=hint).strip()
            parsed = _parse_intake_state(response)
            if parsed:
                return parsed
        except Exception:
            continue
    return None


def _run_next_question_from_state(
    intake_state: dict,
    conversation_history: list,
    *,
    task_hint: str | None = None,
    feedback: str = "",
) -> dict | None:
    """Small model call: choose one next question or complete from compact state."""
    asked_questions = _extract_asked_questions(conversation_history)[-5:]
    few_shot_block = ""
    if _ENABLE_INTAKE_FEWSHOT:
        try:
            from training.few_shot_retriever import get_intake_reply_example_pack
            # Use only case facts (facts_summary + known_facts) for retrieval — NOT open_points.
            # Canonical open_points strings ("documents / messages / witnesses currently available",
            # "prior actions already taken", etc.) contain generic legal tokens that create false
            # lexical overlap with unrelated domain examples (e.g. cheque-bounce examples for DV queries).
            query_parts = [
                intake_state.get("facts_summary", ""),
                " ".join(intake_state.get("known_facts", []) or []),
            ]
            packed = get_intake_reply_example_pack(" ".join([p for p in query_parts if p]).strip(), max_examples=2)
            if packed:
                few_shot_block = f"\n\n{packed}\n"
        except Exception:
            few_shot_block = ""
    state_json = json.dumps({
        "client_objective": intake_state.get("client_objective", ""),
        "urgency_level": intake_state.get("urgency_level", "unknown"),
        "known_facts": intake_state.get("known_facts", []),
        "prior_actions_taken": intake_state.get("prior_actions_taken", []),
        "open_points": intake_state.get("open_points", []),
        "enough_to_proceed": intake_state.get("enough_to_proceed", False),
        "facts_summary": intake_state.get("facts_summary", ""),
    }, ensure_ascii=False)
    asked_json = json.dumps(asked_questions, ensure_ascii=False)
    prompt = (
        f"{NEXT_QUESTION_FROM_STATE_SYSTEM}"
        f"{few_shot_block}\n\n"
        f"COMPACT CASE STATE:\n{state_json}\n\n"
        f"QUESTIONS ALREADY ASKED:\n{asked_json}\n\n"
    )
    if feedback.strip():
        prompt += f"{feedback.strip()}\n\nReturn a fresh, case-specific intake move that fixes the issues above.\n\n"
    prompt += (
        f"Output one line of valid JSON only."
    )
    try:
        response = ask_llm(prompt, task_hint=task_hint).strip()
        return _parse_llm_response(response, intake_state.get("facts_summary", ""))
    except Exception:
        return None




def _is_duplicate_question(proposed: str, asked_questions: list[str]) -> bool:
    """
    Return True if the proposed question substantially overlaps (same topic keywords)
    with any question already in asked_questions.

    We allow a compact grouped follow-up when it stays in the same general area
    but adds a genuinely new related fact. Example: "Any eyewitnesses?" can be
    followed by "Would they testify or file an affidavit?" without being treated
    as a duplicate.
    """
    if not asked_questions or not proposed:
        return False
    p_lower = re.sub(r"\s+", " ", proposed.lower()).strip()
    p_topics = _infer_question_buckets(p_lower)
    p_tokens = _question_content_tokens(p_lower)
    for asked in asked_questions:
        a_lower = re.sub(r"\s+", " ", asked.lower()).strip()
        if p_lower == a_lower:
            return True
        a_topics = _infer_question_buckets(a_lower)
        a_tokens = _question_content_tokens(a_lower)
        if not p_tokens or not a_tokens:
            continue
        shared_topics = p_topics & a_topics if p_topics and a_topics else set()
        shared_tokens = p_tokens & a_tokens
        new_tokens = p_tokens - a_tokens
        union_tokens = p_tokens | a_tokens
        similarity = len(shared_tokens) / max(1, len(union_tokens))
        if p_tokens == a_tokens:
            return True
        if shared_topics and similarity >= 0.85 and len(new_tokens) <= 1:
            return True
        if not shared_topics and similarity >= 0.92 and len(new_tokens) == 0:
            return True
    return False


def _assess_model_next_reply(reply: str, intake_state: dict, asked_questions: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not reply:
        reasons.append("No usable reply was produced.")
        return False, reasons
    if _is_low_quality_next_question(reply):
        reasons.append("The reply was too generic, malformed, or not phrased as a concrete intake question.")
    # Extract question sentences from the reply before duplicate-checking.
    # Passing the full reply (with empathy preamble) into _is_duplicate_question
    # dilutes token similarity: the preamble tokens lower the Jaccard score below
    # the 0.85 threshold, so an identical question slips through as "not a duplicate".
    # Splitting on sentence boundaries and checking only the question sentences
    # avoids this false negative.
    _reply_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]
    _question_sentences = [s for s in _reply_sentences if "?" in s and len(s) > 12]
    _check_texts = _question_sentences if _question_sentences else [reply]
    _is_dup = any(_is_duplicate_question(t, asked_questions) for t in _check_texts)
    if _is_dup:
        reasons.append("The reply substantially repeated a question that was already asked.")
    if not _question_is_grounded_in_state(reply, intake_state):
        reasons.append("The reply was not tightly grounded in the current intake state or open points.")
    return len(reasons) == 0, reasons


def _build_next_question_feedback(reasons: list[str], intake_state: dict) -> str:
    critical = ", ".join(_critical_open_points(intake_state)[:3]) or "the strongest unresolved point"
    bullets = "\n".join(f"- {reason}" for reason in reasons[:4])
    return (
        "REPAIR FEEDBACK:\n"
        f"{bullets}\n"
        f"Focus the next move on: {critical}.\n"
        "Do not repeat earlier phrasing. Ask a fresh, case-specific follow-up in a calm advocate voice."
    )


def _is_low_value_timing_followup(proposed: str, intake_state: dict, asked_questions: list[str]) -> bool:
    """
    Reject repeated low-value timing questions when an ongoing pattern is already clear
    and higher-value decision gaps still exist.
    """
    if not proposed:
        return False
    p = proposed.lower()
    timing_signals = (
        "specific time", "specific times", "what time", "which time", "frequency",
        "how often", "how many times", "duration", "late evening", "regularly",
    )
    if not any(sig in p for sig in timing_signals):
        return False

    state_blob = " ".join(
        [intake_state.get("facts_summary", "")]
        + list(intake_state.get("known_facts", []) or [])
        + list(intake_state.get("open_points", []) or [])
    ).lower()
    ongoing_signals = ("last", "months", "regular", "regularly", "ongoing", "contin", "every", "often")
    if not any(sig in state_blob for sig in ongoing_signals):
        return False

    higher_value_gaps = (
        "objective", "relief", "protection", "complaint", "police", "fir", "medical",
        "evidence", "children", "child", "safety", "residence", "maintenance",
    )
    if not any(sig in state_blob for sig in higher_value_gaps):
        return False

    return _is_duplicate_question(proposed, asked_questions) or any(
        any(sig in (q or "").lower() for sig in timing_signals)
        for q in asked_questions
    )



# ---------------------------------------------------------------------------
# Intent detection (keyword safety net when LLM gets it wrong)
# ---------------------------------------------------------------------------

def _detect_intent_from_keywords(msg: str) -> str | None:
    """Keyword-based intent detection as a safety net."""
    m = msg.lower()
    search_signals = [
        "case law", "case laws", "caselaws", "judgment", "judgement",
        "judgments", "judgements", "ruling", "rulings", "verdict",
        "pull", "find me", "search for", "get me", "show me",
        "pull three", "pull 3", "find three", "get three",  # explicit count requests
    ]
    if any(s in m for s in search_signals):
        return "search"
    lookup_signals = [
        "bare act", "section of", "sections of", "provisions of",
        "which section", "ipc section", "crpc section", "cpc section",
        "bnss section", "bns section",
    ]
    if any(s in m for s in lookup_signals):
        return "lookup"
    # If message contains both "case" and "bare act" or "section", prefer search
    if ("case" in m or "judgment" in m) and ("bare act" in m or "section" in m):
        return "search"  # "pull case laws and bare act sections" → search
    return None


def _detect_search_strategy_from_keywords(msg: str) -> str | None:
    """Detect explicit user intent to disable the Indiankanoon fallback and stay local-only."""
    m = msg.lower()
    local_only_phrases = [
        "only local", "no web", "don't search internet", "do not search internet",
        "skip web", "local database only", "local only", "no internet",
    ]
    if any(p in m for p in local_only_phrases):
        return "local_only"
    return None


def _default_search_strategy_for_intent(
    intent: str | None,
    user_message: str,
    requested_strategy: str | None = None,
) -> str:
    """
    Keep legal-opinion flows on the fast local-only path unless the user explicitly
    asked for something else. Search/lookup flows retain the broader mixed strategy.
    """
    keyword_strategy = _detect_search_strategy_from_keywords(user_message)
    if keyword_strategy:
        return keyword_strategy
    normalized_intent = (intent or "").strip().lower()
    requested = (requested_strategy or "").strip().lower()
    if normalized_intent == "legal_opinion":
        return "local_only"
    if requested in ("local_only", "local_then_web"):
        return requested
    return "local_then_web"


def _is_all_acts_style_request(msg: str) -> bool:
    """True if user is asking for 'all acts' / 'pull all' / 'list all' (broad discovery, not a small count)."""
    m = (msg or "").lower()
    phrases = [
        "all acts", "all the acts", "all laws", "pull all", "list all",
        "list all acts", "list all laws", "every act", "every law",
        "all acts and laws", "all acts and laws made by", "acts made by government",
        "all acts enacted by", "all acts by government",
    ]
    return any(p in m for p in phrases)


def _extract_result_count(msg: str) -> int | None:
    """Extract a number from the user's message like 'find 3 case laws'."""
    import re
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(r"(?:top\s+)?(\d+)\s+(?:case|judgment|judgement|ruling|bare|section)", msg.lower())
    if m:
        return min(int(m.group(1)), 30) or 5
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\b\s+(?:case|judgment|judgement|ruling|bare|section)", msg.lower()):
            return num
    return None


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(response: str, user_message: str) -> dict | None:
    """Parse LLM output. Returns None if invalid."""
    out = _extract_json(response)
    if not out or not isinstance(out, dict):
        return None

    # Pass greeting action through as-is � the caller handles it before routing the rest.
    # This keeps greeting handling stable even when the router returns only {action:"greeting"}.
    if out.get("action") == "greeting":
        return out

    if out.get("action") not in ("ask", "complete"):
        return None

    reply = (out.get("reply_to_client") or out.get("question") or "").strip()
    if reply:
        normalized_reply = re.sub(r"[\s\W_]+", "", reply)
        if len(normalized_reply) < 6:
            reply = ""

    if out["action"] == "complete":
        # LLM-proposed intent; we will treat legal_opinion as the default and
        # only honour search/lookup when the user has clearly used retrieval
        # language (pull/find/get/show/search for acts/case laws).
        intent = out.get("intent", "legal_opinion")
        if intent not in ("search", "lookup", "legal_opinion", "chat", "greeting", "generic_chat"):
            intent = "legal_opinion"

        # Keyword-based intent from explicit retrieval phrases in the user's message.
        # This is the ONLY signal that may safely upgrade legal_opinion → search/lookup.
        keyword_intent = _detect_intent_from_keywords(user_message)

        # If the LLM suggested search/lookup but the user did NOT use any explicit
        # search/lookup phrasing, fall back to legal_opinion (default).
        if intent in ("search", "lookup") and not keyword_intent:
            intent = "legal_opinion"

        # Greeting/chat must never run research
        if intent in ("chat", "greeting"):
            if reply:
                return {"action": "ask", "question": reply}
            cleaned = user_message.strip().lower().rstrip("!?.,;:")
            question = (
                GREETING_STATIC_TEMPLATE
                if cleaned in GREETING_PHRASES
                else generate_greeting_response(user_message)
            )
            return {"action": "ask", "question": question}

        # Safety net: if user message is clearly greeting (and no prior legal context), don't run research
        # Don't check conversation_history here since we're inside _parse_llm_response which doesn't have it
        # The fast-path check at the top of get_next_question_or_complete already handles this
        if len(user_message.strip()) < 30 and not any(kw in user_message.lower() for kw in _LEGAL_KEYWORDS) and intent == "legal_opinion":
            # Only treat as greeting if it's an exact match to greeting phrases
            cleaned = user_message.strip().lower().rstrip("!?.,;:")
            if cleaned in GREETING_PHRASES:
                if reply:
                    return {"action": "ask", "question": reply}
                return {"action": "ask", "question": GREETING_STATIC_TEMPLATE}

        # Safety net: override intent based on keywords when routing LLM said legal_opinion
        if keyword_intent and intent == "legal_opinion":
            intent = keyword_intent

        # Model-driven intent: single source of truth for document_types, search_strategy, result_count
        research_intent = None
        try:
            from services.intent_extractor import extract_research_intent
            research_intent = extract_research_intent(user_message)
        except Exception as e:
            logger.debug("Intent extraction failed, using fallbacks: %s", e)

        # result_count: model-driven when user specified a number; None = flexible limit (no rigid cap)
        result_count = None
        if research_intent and research_intent.get("result_count") is not None:
            result_count = research_intent.get("result_count")
        else:
            text_count = _extract_result_count(user_message)
            if text_count:
                result_count = text_count
            else:
                try:
                    rc = out.get("result_count")
                    if rc is not None:
                        result_count = max(1, min(int(rc), 30))
                except (TypeError, ValueError):
                    pass
        # "Pull all" / broad: no cap
        if _is_all_acts_style_request(user_message):
            result_count = None

        # document_types: from intent first; fallback from routing intent + keywords.
        # IMPORTANT: document_types must NEVER override intent. Only keyword_intent
        # (based on explicit user phrasing) is allowed to upgrade legal_opinion →
        # search/lookup. Here we only decide which materials to prioritise.
        if research_intent and research_intent.get("document_types") in ("acts_only", "case_laws_only", "both"):
            document_types = research_intent["document_types"]
        else:
            if intent == "lookup":
                document_types = "acts_only"
            elif intent == "search":
                document_types = "case_laws_only"
            else:
                document_types = "both"
            msg_lower = (user_message or "").lower()
            acts_only_signals = [
                "acts enacted by", "enacted by the government", "only acts", "only laws",
                "state acts", "bare acts only", "no case laws", "don't want any case laws",
                "don't want case laws", "without case laws", "acts by telangana", "acts by the state",
                "government of telangana", "indiacode", "all the acts",
            ]
            case_laws_only_signals = [
                "only case laws", "only judgments", "only judgements", "only court",
                "no acts", "don't want acts", "case laws only", "judgments only",
            ]
            if any(s in msg_lower for s in acts_only_signals):
                document_types = "acts_only"
            elif any(s in msg_lower for s in case_laws_only_signals):
                document_types = "case_laws_only"

        # search_strategy: from intent first; fallback from routing LLM + keywords.
        # Explicit user phrases ("avoid local", "directly go to web", "web only") always override so we never ignore them.
        requested_search_strategy = None
        if research_intent and research_intent.get("search_strategy") in ("local_only", "local_then_web"):
            requested_search_strategy = research_intent["search_strategy"]
        else:
            candidate = (out.get("search_strategy") or "").strip().lower()
            if candidate in ("local_only", "local_then_web"):
                requested_search_strategy = candidate
        search_strategy = _default_search_strategy_for_intent(intent, user_message, requested_search_strategy)

        payload = {
            "action": "complete",
            "intent": intent,
            "result_count": result_count,
            "facts_summary": out.get("facts_summary") or user_message,
            "message": reply,
            "document_types": document_types,
            "search_strategy": search_strategy,
        }
        return payload

    # action == "ask"
    if reply:
        return {"action": "ask", "question": reply}
    return None


# ---------------------------------------------------------------------------
# Facts enrichment — always build facts_summary from ALL user messages
# ---------------------------------------------------------------------------

def _enrich_facts_summary(parsed: dict, conversation_history: list, user_message: str) -> dict:
    """
    Replace or augment the LLM-generated facts_summary with the full concatenation
    of every user message in the conversation.  This guarantees that dispute
    decomposition, bare-act retrieval and the final legal opinion all see the
    complete picture — not just whatever the LLM happened to digest.

    Only applied when action=complete so it never interferes with ask responses.
    """
    if parsed.get("action") != "complete":
        return parsed

    seen: set[str] = set()
    user_msgs: list[str] = []
    for m in conversation_history:
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content and content not in seen:
                seen.add(content)
                user_msgs.append(content)
    current = (user_message or "").strip()
    if current and current not in seen:
        user_msgs.append(current)

    full_text = "\n".join(user_msgs).strip()
    if not full_text:
        return parsed

    llm_summary = (parsed.get("facts_summary") or "").strip()
    # Append LLM summary only when it adds info not already in the raw messages
    if llm_summary and llm_summary not in full_text and len(llm_summary) > 50:
        parsed["facts_summary"] = f"{full_text}\n\n{llm_summary}"
    else:
        parsed["facts_summary"] = full_text
    return parsed


def _normalize_completion_payload(parsed: dict, user_message: str) -> dict:
    """Normalize completion routing so legal-opinion flows stay on the intended path."""
    normalized = dict(parsed or {})
    intent = normalized.get("intent") or "legal_opinion"
    normalized["intent"] = intent
    normalized.setdefault("document_types", "both")
    normalized.setdefault("result_count", None)
    if not (normalized.get("facts_summary") or "").strip():
        normalized["facts_summary"] = user_message
    normalized["search_strategy"] = _default_search_strategy_for_intent(
        intent,
        user_message,
        normalized.get("search_strategy"),
    )
    return normalized


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_next_question_or_complete(conversation_history: list, user_message: str, force_legal: bool = False, token_callback=None) -> dict:
    """
    Returns:
    - {"action": "ask", "question": "..."} — follow-up for the user
    - {"action": "complete", "facts_summary": "...", "message": "...", "intent": "...", "result_count": int}

    If LLM fails entirely, defaults to action="complete" with user's message as facts_summary,
    ensuring their request is never lost.
    """
    t_start = time.perf_counter()

    # --- Fast path: greetings don't need the full LLM pipeline ---
    # Pass conversation_history so short follow-ups (e.g. "telangana") aren't misclassified as greetings
    if is_greeting(user_message, conversation_history):
        t_before = time.perf_counter()
        cleaned = user_message.strip().lower().rstrip("!?.,;:")
        question = (
            GREETING_STATIC_TEMPLATE
            if cleaned in GREETING_PHRASES
            else generate_greeting_response(user_message)
        )
        out = {"action": "ask", "question": question}
        _log_fc_step("greeting_response", (time.perf_counter() - t_before) * 1000)
        return out

    stop_requested = is_stop_signal(user_message)
    legal_context = _has_legal_context(conversation_history, user_message, force_legal=force_legal)

    # --- Compact intake state: the single routing and state brain ---
    t_compact = time.perf_counter()
    raw_intake_state = _run_compact_intake_state(conversation_history, user_message)
    _log_fc_step(
        "compact_intake_state",
        (time.perf_counter() - t_compact) * 1000,
        f"route={raw_intake_state.get('route') if raw_intake_state else 'None'}",
    )

    intake_state = dict(raw_intake_state or {})
    route = _normalize_intake_route(intake_state.get("route"), conversation_history, user_message, force_legal=force_legal)

    if route == "greeting" and not legal_context:
        cleaned = user_message.strip().lower().rstrip("!?.,;:")
        question = (
            GREETING_STATIC_TEMPLATE
            if cleaned in GREETING_PHRASES
            else generate_greeting_response(user_message)
        )
        return {"action": "ask", "question": question}

    if route in ("search", "lookup"):
        parsed = {
            "action": "complete",
            "intent": route,
            "result_count": _extract_result_count(user_message),
            "facts_summary": (intake_state.get("facts_summary") or _combine_user_messages(conversation_history, user_message) or user_message),
            "message": "",
            "document_types": "acts_only" if route == "lookup" else "case_laws_only",
            "search_strategy": _default_search_strategy_for_intent(route, user_message),
        }
        return _normalize_completion_payload(parsed, user_message)

    if route == "generic_chat" and not legal_context and not force_legal:
        return _normalize_completion_payload({
            "action": "complete",
            "intent": "generic_chat",
            "facts_summary": intake_state.get("facts_summary") or user_message,
            "message": "",
            "document_types": "both",
            "search_strategy": _default_search_strategy_for_intent("generic_chat", user_message),
        }, user_message)

    # Legal opinion is the single structured intake path.
    legal_state = _normalize_legal_intake_state(intake_state, conversation_history, user_message)
    legal_state["route"] = "legal_opinion"
    legal_state["enough_to_proceed"] = _should_complete_legal_intake(
        legal_state,
        conversation_history,
        user_message,
        stop_requested=stop_requested,
    )
    if not legal_state["enough_to_proceed"] and _can_proceed_with_partial_record(
        legal_state,
        conversation_history,
        user_message,
    ):
        legal_state["enough_to_proceed"] = True

    if legal_state["enough_to_proceed"]:
        parsed = {
            "action": "complete",
            "intent": "legal_opinion",
            "result_count": None,
            "facts_summary": legal_state.get("facts_summary") or user_message,
            "message": "",
            "document_types": "both",
            "search_strategy": _default_search_strategy_for_intent("legal_opinion", user_message),
        }
        _log_fc_step("legal_intake_complete", (time.perf_counter() - t_start) * 1000)
        return _enrich_facts_summary(_normalize_completion_payload(parsed, user_message), conversation_history, user_message)

    asked_questions = _extract_asked_questions(conversation_history)
    t_next = time.perf_counter()
    model_next = _run_next_question_from_state(legal_state, conversation_history, task_hint="fast")
    _log_fc_step(
        "legal_intake_model_next",
        (time.perf_counter() - t_next) * 1000,
        f"action={model_next.get('action') if model_next else 'None'}",
    )
    retry_feedback = ""
    if model_next:
        if model_next.get("action") == "ask":
            reply = (model_next.get("question") or model_next.get("reply_to_client") or "").strip()
            acceptable, reasons = _assess_model_next_reply(reply, legal_state, asked_questions)
            if acceptable:
                _log_fc_step("legal_intake_ask_model", (time.perf_counter() - t_start) * 1000)
                return {"action": "ask", "question": reply}
            if reasons:
                retry_feedback = _build_next_question_feedback(reasons, legal_state)
        elif model_next.get("action") == "complete":
            if _should_complete_legal_intake(
                legal_state,
                conversation_history,
                user_message,
                stop_requested=stop_requested,
            ):
                model_next.setdefault("intent", "legal_opinion")
                model_next.setdefault("document_types", "both")
                model_next.setdefault("search_strategy", _default_search_strategy_for_intent("legal_opinion", user_message))
                model_next.setdefault("result_count", None)
                if not (model_next.get("facts_summary") or "").strip():
                    model_next["facts_summary"] = legal_state.get("facts_summary") or user_message
                _log_fc_step("legal_intake_complete_model", (time.perf_counter() - t_start) * 1000)
                return _enrich_facts_summary(_normalize_completion_payload(model_next, user_message), conversation_history, user_message)
            retry_feedback = (
                "REPAIR FEEDBACK:\n"
                "- Do not complete yet.\n"
                "- Ask the strongest remaining factual follow-up needed before analysis.\n"
                "- Return valid JSON only.\n"
            )
    else:
        retry_feedback = (
            "REPAIR FEEDBACK:\n"
            "- Return valid JSON only.\n"
            "- Ask one fresh, case-specific follow-up grounded in the current intake state.\n"
            "- Do not repeat earlier phrasing.\n"
        )

    if retry_feedback:
        retry_next = _run_next_question_from_state(
            legal_state,
            conversation_history,
            task_hint="fast",
            feedback=retry_feedback,
        )
        if retry_next and retry_next.get("action") == "ask":
            retry_reply = (retry_next.get("question") or retry_next.get("reply_to_client") or "").strip()
            retry_ok, _ = _assess_model_next_reply(retry_reply, legal_state, asked_questions)
            if retry_ok:
                _log_fc_step("legal_intake_ask_model_retry", (time.perf_counter() - t_start) * 1000)
                return {"action": "ask", "question": retry_reply}
        elif retry_next and retry_next.get("action") == "complete" and _should_complete_legal_intake(
            legal_state,
            conversation_history,
            user_message,
            stop_requested=stop_requested,
        ):
            retry_next.setdefault("intent", "legal_opinion")
            retry_next.setdefault("document_types", "both")
            retry_next.setdefault("search_strategy", _default_search_strategy_for_intent("legal_opinion", user_message))
            retry_next.setdefault("result_count", None)
            if not (retry_next.get("facts_summary") or "").strip():
                retry_next["facts_summary"] = legal_state.get("facts_summary") or user_message
            _log_fc_step("legal_intake_complete_model_retry", (time.perf_counter() - t_start) * 1000)
            return _enrich_facts_summary(_normalize_completion_payload(retry_next, user_message), conversation_history, user_message)

    fallback_question = _build_fallback_next_question(legal_state, conversation_history)
    if fallback_question:
        _log_fc_step("legal_intake_ask_fallback", (time.perf_counter() - t_start) * 1000)
        return {"action": "ask", "question": fallback_question}

    parsed = {
        "action": "complete",
        "intent": "legal_opinion",
        "result_count": None,
        "facts_summary": legal_state.get("facts_summary") or _combine_user_messages(conversation_history, user_message) or user_message,
        "message": "",
        "document_types": "both",
        "search_strategy": _default_search_strategy_for_intent("legal_opinion", user_message),
    }
    _log_fc_step("legal_intake_complete_fallback", (time.perf_counter() - t_start) * 1000)
    return _enrich_facts_summary(_normalize_completion_payload(parsed, user_message), conversation_history, user_message)
