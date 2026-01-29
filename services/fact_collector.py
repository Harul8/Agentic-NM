"""
Fact Collection Service - Lawyer-style questioning to gather case facts.
Uses LLM to ask relevant, logical questions and stops when user has no more information.
"""

import json
from llm.ollama_client import ask_llm

FACT_COLLECTION_SYSTEM = """You are an experienced Indian lawyer conducting an initial client intake.
Your role is to gather all relevant facts about the client's legal matter through plain-language questions.

RULES:
1. Ask ONE clear, relevant question at a time in plain English.
2. Ask logical follow-up questions based on what the client has shared.
3. Cover: parties involved, dates, key events, documents, jurisdiction, relief sought.
4. Do NOT give legal advice or conclusions - only gather facts.
5. Keep questions conversational and easy to understand.
6. If the client says they don't have more information, or "that's all", or "no more", or "nothing else" - STOP asking and output exactly: {"action": "complete", "facts_summary": "<brief summary of all facts gathered>"}
7. If you need to ask another question, output: {"action": "ask", "question": "<your next question>"}
8. Always respond with valid JSON only, no other text."""

STOP_PHRASES = [
    "i don't have more",
    "i don't have any more",
    "that's all",
    "that is all",
    "no more",
    "nothing else",
    "nothing more",
    "i've told you everything",
    "that's everything",
    "no further",
    "can't provide more",
    "don't know more",
    "not sure",
    "proceed",
    "generate",
    "go ahead",
    "that's it",
]


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
            "question": "Could you share any other relevant details about your situation? If you don't have more information, just say 'that's all' and I'll proceed with the legal research."
        }
