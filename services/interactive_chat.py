"""
Interactive Chat Orchestrator - Handles multi-phase legal chat flow:
1. Fact collection (professional advocate intake) with intent detection
2. Response generation (bare acts + case laws + structured opinion)
3. Case law confirmation & indexing

Intents (detected during fact collection):
- "search" — find case laws / judgments → skip to research immediately
- "lookup" — find bare act sections → skip to research immediately
- "legal_opinion" — user has a problem → interactive fact collection, then full opinion
"""

from services.fact_collector import get_next_question_or_complete, is_stop_signal
from services.response_generator import generate_response
from services.case_law_indexer_incremental import index_new_case_laws


def _run_search_or_lookup(facts_summary: str, intent: str, top_k: int, msg: str) -> dict:
    """Handle search/lookup intents: go straight to generate_response and return results."""
    try:
        resp = generate_response(facts_summary, top_k=top_k, intent=intent)
    except Exception as e:
        resp = {
            "bare_act_sections": [],
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": f"I couldn't complete your search right now. Please try again or rephrase your query. ({str(e)[:100]})",
        }

    # Map intent to response_type
    response_type = "search_results" if intent == "search" else "lookup_results"
    explanation = (resp.get("explanation") or "").strip()
    if not explanation:
        explanation = "Here’s what I found for your query. Below are any relevant materials."

    return {
        "phase": "done",
        "message": msg,
        "facts_summary": facts_summary,
        "response": {
            "bare_act_sections": resp.get("bare_act_sections", []),
            "case_laws": resp.get("case_laws", []),
            "internet_case_laws": resp.get("internet_case_laws", []),
            "explanation": explanation,
        },
        "response_type": response_type,
        "materials_to_confirm": None,
        "indexed": False,
    }


def process_chat(conversation: list, current_message: str, phase: str, facts_summary: str = None) -> dict:
    """
    Process a chat message and return the appropriate response.

    Args:
        conversation: List of {role, content} messages
        current_message: User's current message
        phase: "fact_collection" | "response_generation" | "confirm_index"
        facts_summary: Collected facts (when phase is response_generation)

    Returns:
        {
            phase: str,
            message: str (assistant message to display),
            facts_summary: str (if fact collection complete),
            response: dict (if response generated - bare_act_sections, case_laws, internet_case_laws, explanation),
            response_type: str ("search_results" | "lookup_results" | "legal_opinion"),
            case_laws_to_confirm: list (internet case laws for user to confirm),
            indexed: bool (if case laws were indexed)
        }
    """
    if phase == "fact_collection":
        result = get_next_question_or_complete(conversation, current_message)

        if result.get("action") == "complete":
            intent = result.get("intent", "legal_opinion")
            result_count = result.get("result_count", 5)
            facts = result.get("facts_summary", current_message)

            # Use LLM-generated message; generate one dynamically if empty or just symbol/emoji
            msg = result.get("message", "").strip()

            # For search/lookup intents: ensure we have a proper greeting before results
            if intent in ("search", "lookup"):
                # Treat as "too short" if empty, or only symbols/emoji (e.g. "⚖"), or < 15 chars
                def _is_proper_greeting(m):
                    if not m or len(m) < 15:
                        return False
                    # Reject if it's mostly non-word chars (emoji/symbols)
                    words = [w for w in m.split() if any(c.isalnum() for c in w)]
                    return len(words) >= 3
                if not _is_proper_greeting(msg):
                    try:
                        from llm.ollama_client import ask_llm
                        msg = ask_llm(
                            f"You are a friendly legal research assistant. The user asked: \"{facts[:400]}\"\n"
                            "Write 2-3 short sentences: greet them, acknowledge what they asked for, "
                            "and say you understand their request and what you are going to provide. "
                            "Be warm and specific. Output only those sentences, no emoji."
                        ).strip()
                    except Exception:
                        msg = ""
                    if not _is_proper_greeting(msg):
                        msg = f"I've looked up Supreme Court case laws and related materials for your query. Here's what I found."
                return _run_search_or_lookup(facts, intent, result_count, msg)

            # For legal_opinion: move to response_generation phase (traditional flow)
            return {
                "phase": "response_generation",
                "message": msg,
                "facts_summary": facts,
                "response": None,
                "response_type": None,
                "case_laws_to_confirm": None,
                "indexed": False,
            }
        else:
            # Use LLM-generated question (reply_to_client)
            msg = (result.get("question") or "").strip()
            return {
                "phase": "fact_collection",
                "message": msg,
                "facts_summary": None,
                "response": None,
                "response_type": None,
                "case_laws_to_confirm": None,
                "indexed": False,
            }

    elif phase == "response_generation":
        facts = facts_summary or current_message
        resp = generate_response(facts)

        if resp.get("needs_confirmation"):
            materials = resp.get("materials_to_confirm", {})
            bare_to_confirm = materials.get("bare_acts", [])
            case_to_confirm = materials.get("case_laws", [])
            return {
                "phase": "confirm_materials",
                "message": resp.get("summary", resp.get("explanation", "")),
                "facts_summary": facts,
                "response": {
                    "bare_act_sections": resp.get("bare_act_sections", []),
                    "case_laws": resp.get("case_laws", []),
                    "internet_case_laws": [],
                    "explanation": resp.get("summary", resp.get("explanation", "")),
                },
                "response_type": "legal_opinion",
                "materials_to_confirm": {
                    "bare_acts": bare_to_confirm,
                    "case_laws": case_to_confirm,
                },
                "indexed": False,
            }

        return {
            "phase": "done",
            "message": resp.get("explanation", ""),
            "facts_summary": facts,
            "response": {
                "bare_act_sections": resp.get("bare_act_sections", []),
                "case_laws": resp.get("case_laws", []),
                "internet_case_laws": resp.get("internet_case_laws", []),
                "explanation": resp.get("explanation", ""),
            },
            "response_type": "legal_opinion",
            "materials_to_confirm": None,
            "indexed": False,
        }

    elif phase == "confirm_index":
        return {
            "phase": "done",
            "message": "",
            "facts_summary": facts_summary,
            "response": None,
            "response_type": None,
            "case_laws_to_confirm": None,
            "indexed": False,
        }

    return {
        "phase": "done",
        "message": "",
        "facts_summary": None,
        "response": None,
        "response_type": None,
        "case_laws_to_confirm": None,
        "indexed": False,
    }
