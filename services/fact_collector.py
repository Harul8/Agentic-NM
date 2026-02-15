"""
Fact Collection Service - Dynamic advocate-style intake.
All user-facing messages come from the LLM (reply_to_client). No hardcoded responses.

KEY DESIGN: If the LLM fails or returns garbage, we do NOT show a hardcoded
question. Instead we treat the user's message as a complete research request
and proceed to research. This avoids the dead-end "Say proceed" → empty research loop.
"""

import json
from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    FACT_COLLECTION_SYSTEM,
    FACT_COLLECTION_RETRY_PROMPT,
    STOP_PHRASES,
)


def is_stop_signal(user_message: str) -> bool:
    """Check if user is signaling they have no more information."""
    msg = user_message.strip().lower()
    return any(phrase in msg for phrase in STOP_PHRASES)


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text that may contain reasoning, markdown, etc."""
    text = (text or "").strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts[1:]:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if "{" in p and "}" in p:
                text = p
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
        pass
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if line.startswith("{") and "action" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _is_greeting_or_small_talk(msg: str) -> bool:
    """True if the message is clearly not a legal request (greeting, thanks, very short)."""
    m = (msg or "").strip().lower()
    if len(m) > 80:
        return False  # Substantive messages are not greetings
    greetings = (
        "hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "ok", "okay", "yes", "no", "bye", "goodbye",
    )
    if m in greetings or m.rstrip("!?.") in greetings:
        return True
    # Very short and no legal/research keywords
    if len(m) < 25:
        legal_keywords = ("law", "act", "section", "case", "court", "judgment", "legal", "advice", "sue", "file", "right", "compensation", "land", "property", "contract", "agreement")
        if not any(kw in m for kw in legal_keywords):
            return True
    return False


def _detect_intent_from_keywords(msg: str) -> str | None:
    """Keyword-based intent detection as a safety net when the LLM gets it wrong."""
    m = msg.lower()
    # Search intent: user explicitly asks to find/pull/get case laws or judgments
    search_signals = [
        "case law", "case laws", "caselaws", "judgment", "judgement",
        "judgments", "judgements", "ruling", "rulings", "verdict",
        "pull", "find me", "search for", "get me", "show me",
    ]
    if any(s in m for s in search_signals):
        return "search"
    # Lookup intent: user explicitly asks for bare act sections
    lookup_signals = [
        "bare act", "section of", "sections of", "provisions of",
        "which section", "ipc section", "crpc section", "cpc section",
    ]
    if any(s in m for s in lookup_signals):
        return "lookup"
    return None


def _extract_result_count_from_text(msg: str) -> int | None:
    """Extract a number from the user's message like 'find 3 case laws'."""
    import re
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    # Match "3 case laws", "top 5 judgments", etc.
    m = re.search(r"(?:top\s+)?(\d+)\s+(?:case|judgment|judgement|ruling|bare|section)", msg.lower())
    if m:
        return min(int(m.group(1)), 20) or 5
    # Match word numbers: "three case laws"
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\b\s+(?:case|judgment|judgement|ruling|bare|section)", msg.lower()):
            return num
    return None


def _parse_llm_response(response: str, user_message: str) -> dict | None:
    """Parse LLM output. Returns None if invalid."""
    out = _extract_json(response)
    if not out or not isinstance(out, dict) or out.get("action") not in ("ask", "complete"):
        return None
    reply = (out.get("reply_to_client") or out.get("question") or "").strip()
    if out["action"] == "complete":
        intent = out.get("intent", "legal_opinion")
        if intent not in ("search", "lookup", "legal_opinion", "chat", "greeting"):
            intent = "legal_opinion"

        # Greeting/chat must never run research: treat as "ask" with the reply
        if intent in ("chat", "greeting"):
            if reply:
                return {"action": "ask", "question": reply}
            return {"action": "ask", "question": "Hello! How can I help you today? Please share your legal query or what you'd like to look up—case laws, bare act sections, or a situation you need advice on."}

        # Safety net: if user message is clearly greeting/small talk, do not run research
        if _is_greeting_or_small_talk(user_message) and intent == "legal_opinion":
            if reply:
                return {"action": "ask", "question": reply}
            return {"action": "ask", "question": "Hello! I'm here to help with legal research—case laws, bare act provisions, or a legal opinion. What would you like to explore?"}

        # Safety net: override intent based on keywords in user message
        keyword_intent = _detect_intent_from_keywords(user_message)
        if keyword_intent and intent == "legal_opinion":
            intent = keyword_intent

        result_count = 5
        try:
            result_count = int(out.get("result_count", 5))
            if result_count < 1:
                result_count = 5
            if result_count > 20:
                result_count = 20
        except (TypeError, ValueError):
            result_count = 5

        # Safety net: override result_count from user text if LLM missed it
        text_count = _extract_result_count_from_text(user_message)
        if text_count:
            result_count = text_count

        return {
            "action": "complete",
            "intent": intent,
            "result_count": result_count,
            "facts_summary": out.get("facts_summary") or user_message,
            "message": reply,  # may be empty — caller handles it
        }
    if reply:
        return {"action": "ask", "question": reply}
    return None


def get_next_question_or_complete(conversation_history: list, user_message: str) -> dict:
    """
    Returns:
    - {"action": "ask", "question": "..."} — LLM-generated follow-up
    - {"action": "complete", "facts_summary": "...", "message": "..."} — ready for research

    IMPORTANT: If the LLM fails entirely, we default to action="complete" with
    the user's original message as the facts_summary. This ensures their request
    is never lost and research always runs on what they actually said.
    """

    # --- Build the prompt ---
    if is_stop_signal(user_message):
        # User wants to proceed: summarise everything they said so far
        prompt = (
            f"The client said they have no more information. Summarise what they shared for research.\n"
            f"Output valid JSON only (one line): {{\"action\": \"complete\", \"facts_summary\": \"<brief summary>\", \"reply_to_client\": \"<your short sentence to the client>\"}}\n\n"
            f"Conversation:\n{json.dumps(conversation_history + [{'role': 'user', 'content': user_message}], indent=2)}"
        )
    else:
        conv_text = "\n".join(
            f"{'Client' if m['role']=='user' else 'Lawyer'}: {m['content']}"
            for m in conversation_history
        )
        conv_text += f"\nClient: {user_message}"
        prompt = f"""{FACT_COLLECTION_SYSTEM}

Conversation so far:
{conv_text}

Now output REASONING: then on the next line your JSON with reply_to_client."""

    # --- Call LLM (attempt 1) ---
    response = None
    try:
        response = ask_llm(prompt)
    except Exception:
        pass

    if response:
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed

    # --- Retry with simpler prompt (attempt 2) ---
    try:
        retry_prompt = FACT_COLLECTION_RETRY_PROMPT.format(user_message=user_message[:500])
        response = ask_llm(retry_prompt)
        parsed = _parse_llm_response(response, user_message)
        if parsed:
            return parsed
    except Exception:
        pass

    # --- Both attempts failed: default to COMPLETE with user's own message ---
    # This is the critical fix: we never return a hardcoded question or empty
    # question. We treat the user's input as a valid research request and proceed.
    all_user_text = "\n".join(
        m["content"] for m in conversation_history if m.get("role") == "user"
    )
    combined = f"{all_user_text}\n{user_message}".strip() or user_message
    fallback_intent = _detect_intent_from_keywords(user_message) or "legal_opinion"
    fallback_count = _extract_result_count_from_text(user_message) or 5
    return {
        "action": "complete",
        "intent": fallback_intent,
        "result_count": fallback_count,
        "facts_summary": combined,
        "message": "",  # empty = caller will generate dynamically
    }
