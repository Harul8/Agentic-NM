"""
Fact Collection Service — Adaptive advocate-style intake.

Phase 2 rewrite:
- Uses GREETING_PHRASES from prompts (Indian language support)
- Dedicated GREETING_RESPONSE_PROMPT for natural greeting handling
- Adaptive: skips questions when user already provided enough detail
- All user-facing messages from LLM — no hardcoded responses
- Falls back to research (not dead-end) if LLM parsing fails
"""

import json
import logging
from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    GREETING_PHRASES,
    GREETING_RESPONSE_PROMPT,
    ROUTING_GATE1_SYSTEM,
    ROUTING_GATE2_SYSTEM,
    FACT_COLLECTION_SYSTEM,
    FACT_COLLECTION_RETRY_PROMPT,
    STOP_PHRASES,
)

logger = logging.getLogger(__name__)

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

    # Exact match (with punctuation stripped)
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
    """Generate a warm greeting using the dedicated greeting prompt."""
    try:
        prompt = f"{GREETING_RESPONSE_PROMPT}\n\nUser said: \"{user_message}\""
        reply = ask_llm(prompt).strip()
        if reply and len(reply) > 10:
            return reply
    except Exception:
        pass
    return "Hello! I'm here to help with legal research — case laws, bare act provisions, or legal advice. What would you like to explore?"


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
# Two-gate routing: Gate 1 (legal vs generalist vs greeting), Gate 2 (legal intent)
# ---------------------------------------------------------------------------

def _run_gate1(conversation_history: list, user_message: str) -> dict | None:
    """
    Gate 1: Classify as GREETING | LEGAL | GENERALIST.
    Returns {"gate1": "GREETING"|"LEGAL"|"GENERALIST", "reply_to_client": "..."} or None.
    """
    context = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {(m.get('content') or '')[:200]}"
        for m in conversation_history[-4:]
    )
    prompt = f"""{ROUTING_GATE1_SYSTEM}

Conversation (recent):
{context or '(none)'}

User: {user_message}

Reply with ONLY one line of JSON (gate1 and reply_to_client when applicable)."""
    try:
        response = ask_llm(prompt).strip()
        out = _extract_json(response)
        if not out or not isinstance(out, dict):
            return None
        gate1 = (out.get("gate1") or "").strip().upper()
        if gate1 not in ("GREETING", "LEGAL", "GENERALIST"):
            return None
        reply = (out.get("reply_to_client") or "").strip()
        return {"gate1": gate1, "reply_to_client": reply}
    except Exception:
        return None


def _run_gate2_legal(conversation_history: list, user_message: str) -> dict | None:
    """
    Gate 2: Only for LEGAL requests. Classify intent (search, lookup, legal_opinion)
    and return full fact-collector shape for _parse_llm_response.
    """
    context = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content') or ''}"
        for m in conversation_history
    )
    prompt = f"""{ROUTING_GATE2_SYSTEM}

Conversation so far:
{context}

User: {user_message}

Output one line of valid JSON only (action, intent, facts_summary, reply_to_client; add result_count for search/lookup; add search_strategy only when user clearly wants web_only or local_only; use action "ask" only if you need one more question for legal_opinion)."""
    try:
        response = ask_llm(prompt).strip()
        return _parse_llm_response(response, user_message)
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
    if not out or not isinstance(out, dict) or out.get("action") not in ("ask", "complete"):
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
            return {"action": "ask", "question": generate_greeting_response(user_message)}

        # Safety net: if user message is clearly greeting (and no prior legal context), don't run research
        # Don't check conversation_history here since we're inside _parse_llm_response which doesn't have it
        # The fast-path check at the top of get_next_question_or_complete already handles this
        if len(user_message.strip()) < 30 and not any(kw in user_message.lower() for kw in _LEGAL_KEYWORDS) and intent == "legal_opinion":
            # Only treat as greeting if it's an exact match to greeting phrases
            cleaned = user_message.strip().lower().rstrip("!?.,;:")
            if cleaned in GREETING_PHRASES:
                if reply:
                    return {"action": "ask", "question": reply}
                return {"action": "ask", "question": generate_greeting_response(user_message)}

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
# Main entry point
# ---------------------------------------------------------------------------

def get_next_question_or_complete(conversation_history: list, user_message: str, force_legal: bool = False) -> dict:
    """
    Returns:
    - {"action": "ask", "question": "..."} — follow-up for the user
    - {"action": "complete", "facts_summary": "...", "message": "...", "intent": "...", "result_count": int}

    If LLM fails entirely, defaults to action="complete" with user's message as facts_summary,
    ensuring their request is never lost.
    """

    # --- Fast path: greetings don't need the full LLM pipeline ---
    # Pass conversation_history so short follow-ups (e.g. "telangana") aren't misclassified as greetings
    if is_greeting(user_message, conversation_history):
        return {"action": "ask", "question": generate_greeting_response(user_message)}

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

    # --- Two-gate routing: Gate 1 (legal vs generalist vs greeting) ---
    gate1_result = _run_gate1(conversation_history, user_message)
    if gate1_result:
        g1 = gate1_result.get("gate1", "")
        reply = (gate1_result.get("reply_to_client") or "").strip()
        if g1 == "GREETING":
            return {"action": "ask", "question": reply or generate_greeting_response(user_message)}
        if g1 == "GENERALIST" and not force_legal:
            return {
                "action": "complete",
                "intent": "generic_chat",
                "result_count": None,
                "facts_summary": user_message,
                "message": reply or "I'll answer that as a general question.",
                "document_types": "both",
                "search_strategy": "local_then_web",
            }
        if g1 in ("LEGAL", "GENERALIST") or force_legal:
            # Treat as LEGAL when gate1 says LEGAL, or when caller forces legal mode
            # even if gate1 returned GENERALIST.
            parsed = _run_gate2_legal(conversation_history, user_message)
            if parsed:
                return parsed
            # Gate 2 failed: fall back to full orchestrator
            try:
                from agents.orchestrator_agent import run_orchestrator
                orchestrator_output = run_orchestrator(conversation_history, user_message)
                if orchestrator_output:
                    parsed = _parse_llm_response(orchestrator_output, user_message)
                    if parsed:
                        return parsed
            except Exception:
                pass

    # --- Fallback: try full orchestrator if two-gate was skipped or Gate 2 failed ---
    try:
        from agents.orchestrator_agent import run_orchestrator
        orchestrator_output = run_orchestrator(conversation_history, user_message)
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

    prompt = f"""{FACT_COLLECTION_SYSTEM}

Conversation so far:
{conv_text}

Now output REASONING: then on the next line your JSON with reply_to_client."""

    # Attempt 1
    try:
        response = ask_llm(prompt)
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed
    except Exception:
        pass

    # Attempt 2: simpler retry prompt
    try:
        retry_prompt = FACT_COLLECTION_RETRY_PROMPT.format(user_message=user_message[:500])
        response = ask_llm(retry_prompt)
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed
    except Exception:
        pass

    # --- Both attempts failed: default to research with what we have (flexible limit when no count) ---
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
