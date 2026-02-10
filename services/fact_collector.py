"""
Fact Collection Service - Professional advocate-style client intake.
Uses LLM to ask relevant, structured questions and stops when user has no more information.
"""

import json
from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    FACT_COLLECTION_SYSTEM,
    STOP_PHRASES,
)


def is_stop_signal(user_message: str) -> bool:
    """Check if user is signaling they have no more information."""
    msg = user_message.strip().lower()
    return any(phrase in msg for phrase in STOP_PHRASES)


def get_next_question_or_complete(conversation_history: list, user_message: str) -> dict:
    """
    Returns either:
    - {"action": "ask", "question": "..."} - next question to ask
    - {"action": "complete", "facts_summary": "..."} - fact collection done
    """
    if is_stop_signal(user_message):
        # Build facts summary from conversation
        facts_parts = []
        for msg in conversation_history:
            if msg.get("role") == "user" and msg.get("content"):
                facts_parts.append(msg["content"])
        facts_parts.append(user_message)
        facts_summary = "\n".join(facts_parts)

        prompt = f"""Based on this conversation, provide a brief structured summary of all facts gathered.
Output ONLY valid JSON: {{"action": "complete", "facts_summary": "<summary>"}}

Conversation:
{json.dumps(conversation_history + [{"role": "user", "content": user_message}], indent=2)}
"""
    else:
        # Build conversation context
        conv_text = "\n".join(
            f"{'Client' if m['role']=='user' else 'Lawyer'}: {m['content']}"
            for m in conversation_history
        )
        conv_text += f"\nClient: {user_message}"

        prompt = f"""{FACT_COLLECTION_SYSTEM}

Conversation so far:
{conv_text}

What is your next question? Output valid JSON only."""

    response = ask_llm(prompt)

    # Parse JSON from response (handle markdown code blocks)
    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: if user said stop, complete; else ask a generic follow-up
        if is_stop_signal(user_message):
            facts = "\n".join(m["content"] for m in conversation_history if m.get("role") == "user")
            return {"action": "complete", "facts_summary": facts or user_message}
        return {
            "action": "ask",
            "question": "Please share any other relevant details—parties, dates, documents, or relief sought. If you have nothing further to add, say 'that's all' or 'proceed' and I shall move to legal research."
        }
