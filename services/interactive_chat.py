"""
Interactive Chat Orchestrator — Handles multi-phase legal chat flow:
1. Fact collection (professional advocate intake) with intent detection
2. Response generation (bare acts + case laws + structured opinion)
3. Material confirmation & indexing

Phase 2 cleanup:
- Greeting logic moved to fact_collector (is_greeting + generate_greeting_response)
- No more inline _is_proper_greeting() or ad-hoc LLM calls for greetings
- Direct v2 pipeline import, no fallbacks
- Disclaimer appended to legal opinions
"""

import logging

from services.fact_collector import get_next_question_or_complete
from services.response_generator_v2 import generate_response_v2 as generate_response
from services.content_guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger(__name__)


def _ensure_message(msg: str, facts: str, intent: str) -> str:
    """
    Ensure the transition message to the user is substantive (not empty/emoji-only).
    If the LLM's reply_to_client was empty or too short, generate a sensible default.
    """
    if msg and len(msg) >= 15:
        # Check it has at least 3 real words (not just symbols/emoji)
        words = [w for w in msg.split() if any(c.isalnum() for c in w)]
        if len(words) >= 3:
            return msg

    # Generate contextual default based on intent
    if intent == "search":
        return "I've searched for relevant case laws and judgments for your query. Here's what I found."
    elif intent == "lookup":
        return "I've looked up the relevant bare act provisions for your query. Here's what I found."
    else:
        return "Thank you for sharing the details. I've researched the applicable bare acts and case laws. Here's my analysis."


def _run_search_or_lookup(facts_summary: str, intent: str, msg: str, result_count: int = 5) -> dict:
    """Handle search/lookup intents: go straight to research and return results."""
    try:
        resp = generate_response(facts_summary, jurisdiction_state="", intent=intent, result_count=result_count)
    except Exception as e:
        logger.error("Research generation failed: %s", e, exc_info=True)
        resp = {
            "bare_act_sections": [],
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": f"I couldn't complete your search right now. Please try again or rephrase your query.",
        }

    response_type = "search_results" if intent == "search" else "lookup_results"
    explanation = (resp.get("explanation") or "").strip()
    if not explanation:
        explanation = "Here's what I found for your query."

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


def _empty_result(phase: str = "done", facts_summary: str = None) -> dict:
    """Return a safe empty result dict."""
    return {
        "phase": phase,
        "message": "",
        "facts_summary": facts_summary,
        "response": None,
        "response_type": None,
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

    Returns dict with: phase, message, facts_summary, response, response_type,
    materials_to_confirm, indexed
    """

    # ---- Safety gate: check input before any processing ----
    current_message = sanitize_input(current_message)
    safety = check_query_safety(current_message)

    if not safety.get("safe"):
        logger.warning("Blocked unsafe query (risk=%s): %s", safety.get("risk_level"), current_message[:80])
        return {
            "phase": "fact_collection",
            "message": safety.get("reason", "I cannot process this request. Please rephrase your legal question."),
            "facts_summary": None,
            "response": None,
            "response_type": None,
            "materials_to_confirm": None,
            "indexed": False,
        }

    # ---- Phase: Fact Collection ----
    if phase == "fact_collection":
        result = get_next_question_or_complete(conversation, current_message)

        if result.get("action") == "complete":
            intent = result.get("intent", "legal_opinion")
            facts = result.get("facts_summary", current_message)
            msg = _ensure_message(result.get("message", ""), facts, intent)

            if intent in ("search", "lookup"):
                count = result.get("result_count", 5)
                return _run_search_or_lookup(facts, intent, msg, result_count=count)

            # Legal opinion: move to response_generation phase
            return {
                "phase": "response_generation",
                "message": msg,
                "facts_summary": facts,
                "response": None,
                "response_type": None,
                "materials_to_confirm": None,
                "indexed": False,
            }

        # Still collecting facts — return the question
        return {
            "phase": "fact_collection",
            "message": (result.get("question") or "").strip(),
            "facts_summary": None,
            "response": None,
            "response_type": None,
            "materials_to_confirm": None,
            "indexed": False,
        }

    # ---- Phase: Response Generation ----
    elif phase == "response_generation":
        facts = facts_summary or current_message
        try:
            resp = generate_response(facts, jurisdiction_state="", intent="legal_opinion")
        except Exception as e:
            logger.error("Response generation failed: %s", e, exc_info=True)
            return {
                "phase": "done",
                "message": "I encountered an issue while preparing your analysis. Please try again.",
                "facts_summary": facts,
                "response": None,
                "response_type": "legal_opinion",
                "materials_to_confirm": None,
                "indexed": False,
            }

        # Check if materials need user confirmation first
        if resp.get("needs_confirmation"):
            materials = resp.get("materials_to_confirm", {})
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
                    "bare_acts": materials.get("bare_acts", []),
                    "case_laws": materials.get("case_laws", []),
                },
                "indexed": False,
            }

        # Full response ready — check output safety
        explanation = (resp.get("explanation") or "").strip()
        if explanation:
            resp_safety = check_response_safety(explanation)
            if not resp_safety.get("safe"):
                logger.warning("Unsafe LLM output blocked")
                explanation = "I was unable to generate a safe response for this query. Please try rephrasing."

        return {
            "phase": "done",
            "message": "",  # transition text only — explanation goes in response.explanation
            "facts_summary": facts,
            "response": {
                "bare_act_sections": resp.get("bare_act_sections", []),
                "case_laws": resp.get("case_laws", []),
                "internet_case_laws": resp.get("internet_case_laws", []),
                "explanation": explanation,
            },
            "response_type": "legal_opinion",
            "materials_to_confirm": None,
            "indexed": False,
        }

    # ---- Phase: Confirm Index ----
    elif phase == "confirm_index":
        return _empty_result("done", facts_summary)

    return _empty_result()
