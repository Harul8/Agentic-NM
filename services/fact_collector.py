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

from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    GREETING_PHRASES,
    ROUTING_SINGLE_GATE_SYSTEM,
    FACT_COLLECTION_SYSTEM,
    FACT_COLLECTION_RETRY_PROMPT,
    SENIOR_ADVOCATE_INTAKE_SYSTEM,
    STOP_PHRASES,
)

# Feature flag: set USE_SENIOR_ADVOCATE_INTAKE=1 to activate the new
# senior-advocate intake prompt (replaces ROUTING_SINGLE_GATE_SYSTEM).
import os as _os
_USE_SENIOR_INTAKE = _os.environ.get("USE_SENIOR_ADVOCATE_INTAKE", "1").strip() in ("1", "true", "yes")
_ACTIVE_INTAKE_SYSTEM = SENIOR_ADVOCATE_INTAKE_SYSTEM if _USE_SENIOR_INTAKE else ROUTING_SINGLE_GATE_SYSTEM

logger = logging.getLogger(__name__)

# Set PIPELINE_TIMING=1 to log elapsed ms for each step (debug slow follow-ups)
_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")

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
    "when", "date", "time", "how long", "since when", "duration",
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


def _build_intake_context_block(conversation_history: list) -> str:
    """
    Build a structured ALREADY-ASKED / ALREADY-KNOWN block to inject into Gate 2 prompts.
    Makes it impossible for the model to miss what has already been covered.
    """
    if not conversation_history:
        return ""

    asked_questions = _extract_asked_questions(conversation_history)
    user_facts: list[str] = []
    for msg in conversation_history:
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if len(content) > 5:
                # Truncate very long user messages for prompt brevity
                user_facts.append(content[:350] if len(content) > 350 else content)

    if not asked_questions and not user_facts:
        return ""

    lines = [
        "",
        "══════════════════════════════════════════════════════",
        "INTAKE MEMORY — READ BEFORE DECIDING WHAT TO ASK NEXT",
        "══════════════════════════════════════════════════════",
    ]

    if user_facts:
        lines.append("FACTS ALREADY STATED BY CLIENT — treat as fully known, do NOT ask about these:")
        for i, fact in enumerate(user_facts, 1):
            lines.append(f"  [{i}] {fact}")

    if asked_questions:
        lines.append("")
        lines.append("QUESTIONS ALREADY ASKED — NEVER repeat these or anything substantially similar:")
        for i, q in enumerate(asked_questions, 1):
            lines.append(f"  [{i}] {q}")
        lines.append("")
        lines.append("⛔  Your next question MUST be on a COMPLETELY DIFFERENT topic not listed above.")
        lines.append("⛔  If no genuinely new material fact is still missing, set action=complete NOW.")

    lines.append("══════════════════════════════════════════════════════")
    lines.append("")
    return "\n".join(lines)


def _is_duplicate_question(proposed: str, asked_questions: list[str]) -> bool:
    """
    Return True if the proposed question substantially overlaps (same topic keywords)
    with any question already in asked_questions. Two or more shared topic keywords = duplicate.
    """
    if not asked_questions or not proposed:
        return False
    p_lower = proposed.lower()
    p_topics = {kw for kw in _DEDUP_TOPIC_KEYWORDS if kw in p_lower}
    if not p_topics:
        return False
    for asked in asked_questions:
        a_topics = {kw for kw in _DEDUP_TOPIC_KEYWORDS if kw in asked.lower()}
        if len(p_topics & a_topics) >= 1:  # Any 1 shared topic = duplicate
            return True
    return False


# ---------------------------------------------------------------------------
# Single-gate routing — replaces the old two-gate (Gate1 + Gate2) design.
# One LLM call handles greeting / generalist / legal-search / legal-opinion.
# ---------------------------------------------------------------------------

def _run_single_gate(conversation_history: list, user_message: str, token_callback=None) -> dict | None:
    """
    Unified router: one LLM call that classifies AND decides action.
    Returns a parsed result dict ready for the caller, or None on failure.

    Handles:
      greeting    → action="greeting"   (caller converts to ask-style response)
      generic_chat → action="complete", intent="generic_chat"
      search/lookup → action="complete", intent="search"|"lookup"
      legal_opinion → action="ask" or action="complete"
    """
    context = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content') or ''}"
        for m in conversation_history
    )

    # Few-shot injection only on the first 2 user turns — after that the conversation
    # history itself demonstrates the desired style; injecting it is pure token waste.
    user_turn_count = sum(1 for m in conversation_history if m.get("role") == "user")
    few_shot_block = ""
    if user_turn_count <= 2:
        try:
            from training.few_shot_retriever import get_intake_example
            all_user_text = " ".join(
                m.get("content", "") for m in conversation_history if m.get("role") == "user"
            ) + " " + user_message
            example = get_intake_example(all_user_text)
            if example:
                few_shot_block = f"\n\n{example}\n"
        except Exception:
            pass

    # Structured INTAKE MEMORY block — explicit list of everything already asked / known.
    intake_context_block = _build_intake_context_block(conversation_history)

    prompt = (
        f"{_ACTIVE_INTAKE_SYSTEM}"
        f"{few_shot_block}"
        f"{intake_context_block}"
        f"\nConversation so far:\n{context}"
        f"\n\nUser: {user_message}"
        f"\n\nOutput one line of valid JSON only."
    )

    try:
        if token_callback:
            from llm.ollama_client import ask_llm_stream
            _parts: list[str] = []
            for _tok in ask_llm_stream(prompt):
                token_callback(_tok)
                _parts.append(_tok)
            response = "".join(_parts).strip()
        else:
            response = ask_llm(prompt).strip()
        parsed = _parse_llm_response(response, user_message)

        # Code-level deduplication guard: if the model still proposes a duplicate
        # question despite the INTAKE MEMORY block, force completion.
        # NOTE: _parse_llm_response normalises the reply into "question" key;
        # "reply_to_client" is only present in the raw LLM JSON before parsing.
        if parsed and parsed.get("action") == "ask":
            asked_questions = _extract_asked_questions(conversation_history)
            reply = parsed.get("question") or parsed.get("reply_to_client") or ""
            if asked_questions and _is_duplicate_question(reply, asked_questions):
                logger.info(
                    "[SingleGate] Duplicate question blocked — forcing complete. Proposed: %s",
                    reply[:120],
                )
                all_user_text = " | ".join(
                    (m.get("content") or "")
                    for m in conversation_history
                    if m.get("role") == "user"
                )
                return {
                    "action": "complete",
                    "intent": "legal_opinion",
                    "result_count": None,
                    "facts_summary": all_user_text.strip() or user_message,
                    "message": "",
                    "document_types": "both",
                    "search_strategy": "local_then_web",
                }

        return parsed
    except Exception:
        return None


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
    """Detect explicit user intent to use only web or only local. Returns 'web_only' | 'local_only' | None."""
    m = msg.lower()
    web_only_phrases = [
        "avoid local", "skip local", "don't search local", "do not search local",
        "only web search", "only web", "directly go to web", "go to web",
        "no local search", "search the web only", "use internet only", "web only",
        "internet only", "don't use local", "without local",
    ]
    if any(p in m for p in web_only_phrases):
        return "web_only"
    local_only_phrases = [
        "only local", "no web", "don't search internet", "do not search internet",
        "skip web", "local database only", "local only", "no internet",
    ]
    if any(p in m for p in local_only_phrases):
        return "local_only"
    return None


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

    # Pass greeting action through as-is — the caller (_run_single_gate / get_next_question_or_complete)
    # handles it before doing anything with the rest of the shape.
    if out.get("action") == "greeting":
        return out

    if out.get("action") not in ("ask", "complete"):
        return None

    reply = (out.get("reply_to_client") or out.get("question") or "").strip()

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
        if research_intent and research_intent.get("search_strategy") in ("web_only", "local_only", "local_then_web"):
            search_strategy = research_intent["search_strategy"]
        else:
            search_strategy = (out.get("search_strategy") or "local_then_web").strip().lower()
            if search_strategy not in ("local_only", "web_only", "local_then_web"):
                search_strategy = "local_then_web"
        keyword_strategy = _detect_search_strategy_from_keywords(user_message)
        if keyword_strategy:
            search_strategy = keyword_strategy

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

    # --- Fast path: stop signals ---
    if is_stop_signal(user_message):
        prompt = (
            f"The client said they have no more information. Summarise what they shared for research.\n"
            f"Output valid JSON only (one line): {{\"action\": \"complete\", \"facts_summary\": \"<brief summary>\", \"reply_to_client\": \"<your short sentence to the client>\"}}\n\n"
            f"Conversation:\n{json.dumps(conversation_history + [{'role': 'user', 'content': user_message}], indent=2)}"
        )
        try:
            response = ask_llm(prompt)
            parsed = _parse_llm_response(response, user_message)
            if parsed:
                return parsed
        except Exception:
            pass
        # Fallback: combine all user text; no result_count = flexible limit
        all_text = "\n".join(m["content"] for m in conversation_history if m.get("role") == "user")
        return {
            "action": "complete",
            "intent": "legal_opinion",
            "result_count": None,
            "facts_summary": f"{all_text}\n{user_message}".strip(),
            "message": "",
            "document_types": "both",
            "search_strategy": _detect_search_strategy_from_keywords(user_message) or "local_then_web",
        }

    # --- Single-gate routing: one LLM call handles everything ---
    t_gate = time.perf_counter()
    parsed = _run_single_gate(conversation_history, user_message, token_callback=token_callback)
    _log_fc_step("single_gate", (time.perf_counter() - t_gate) * 1000,
                 f"action={parsed.get('action') if parsed else 'None'}, "
                 f"intent={parsed.get('intent', '-') if parsed else '-'}")

    if parsed:
        action = parsed.get("action", "")
        reply  = (parsed.get("reply_to_client") or "").strip()

        # Greeting response from the LLM (covers edge-cases the keyword check missed)
        if action == "greeting":
            cleaned = user_message.strip().lower().rstrip("!?.,;:")
            question = (
                GREETING_STATIC_TEMPLATE
                if cleaned in GREETING_PHRASES
                else (reply or generate_greeting_response(user_message))
            )
            return {"action": "ask", "question": question}

        # force_legal overrides a generic_chat classification
        if parsed.get("intent") == "generic_chat" and force_legal:
            parsed["intent"] = "legal_opinion"

        # Hard cap: if 2+ assistant questions already asked, force complete so
        # the LLM can't keep asking even when it hasn't learned anything new.
        if parsed.get("action") == "ask":
            n_q = sum(
                1 for m in conversation_history
                if m.get("role") == "assistant" and "?" in (m.get("content") or "")
            )
            if n_q >= 2:
                all_user = "\n".join(
                    m["content"] for m in conversation_history if m.get("role") == "user"
                )
                logger.info("[FactCollector] Hard cap hit (%d questions asked) — forcing complete", n_q)
                return {
                    "action": "complete",
                    "intent": "legal_opinion",
                    "result_count": None,
                    "facts_summary": f"{all_user}\n{user_message}".strip(),
                    "message": "",
                    "document_types": "both",
                    "search_strategy": _detect_search_strategy_from_keywords(user_message) or "local_then_web",
                }

        # Enrich facts_summary with ALL user messages so downstream phases see
        # the full conversation, not just the LLM's digest of the latest turn.
        if parsed.get("action") == "complete":
            parsed = _enrich_facts_summary(parsed, conversation_history, user_message)

        return parsed

    # --- Fallback: single gate failed — try orchestrator then bare LLM ---
    try:
        t_orch = time.perf_counter()
        from agents.orchestrator_agent import run_orchestrator
        orchestrator_output = run_orchestrator(conversation_history, user_message)
        _log_fc_step("orchestrator_fallback", (time.perf_counter() - t_orch) * 1000)
        if orchestrator_output:
            parsed = _parse_llm_response(orchestrator_output, user_message)
            if parsed:
                return parsed
    except Exception:
        pass

    # --- Fallback: build prompt and call LLM ---
    conv_text = "\n".join(
        f"{'Client' if m['role'] == 'user' else 'Lawyer'}: {m['content']}"
        for m in conversation_history
    )
    conv_text += f"\nClient: {user_message}"

    # --- Few-shot injection for fallback path too ---
    fallback_few_shot = ""
    try:
        from training.few_shot_retriever import get_intake_example
        all_user_text = " ".join(
            m["content"] for m in conversation_history if m.get("role") == "user"
        ) + " " + user_message
        example = get_intake_example(all_user_text)
        if example:
            fallback_few_shot = f"\n\n{example}\n"
    except Exception:
        pass

    prompt = f"""{FACT_COLLECTION_SYSTEM}{fallback_few_shot}
Conversation so far:
{conv_text}

Now output only one JSON object with reply_to_client."""

    # Attempt 1
    try:
        t_llm = time.perf_counter()
        response = ask_llm(prompt)
        _log_fc_step("fallback_llm_attempt1", (time.perf_counter() - t_llm) * 1000)
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed
    except Exception:
        pass

    # Attempt 2: simpler retry prompt
    try:
        t_llm = time.perf_counter()
        retry_prompt = FACT_COLLECTION_RETRY_PROMPT.format(user_message=user_message[:500])
        response = ask_llm(retry_prompt)
        _log_fc_step("fallback_llm_attempt2", (time.perf_counter() - t_llm) * 1000)
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed
    except Exception:
        pass

    # --- Both attempts failed: default to research with what we have (flexible limit when no count) ---
    _log_fc_step("get_next_question_or_complete TOTAL", (time.perf_counter() - t_start) * 1000, "defaulting to complete")
    logger.warning("Fact collection LLM failed twice, proceeding to research with user's message")
    all_user_text = "\n".join(m["content"] for m in conversation_history if m.get("role") == "user")
    combined = f"{all_user_text}\n{user_message}".strip() or user_message
    return {
        "action": "complete",
        "intent": _detect_intent_from_keywords(user_message) or "legal_opinion",
        "result_count": _extract_result_count(user_message),
        "facts_summary": combined,
        "message": "",
        "document_types": "both",
        "search_strategy": _detect_search_strategy_from_keywords(user_message) or "local_then_web",
    }
