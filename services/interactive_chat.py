"""
Interactive Chat Orchestrator - Handles multi-phase legal chat flow:
1. Fact collection (lawyer-style questions)
2. Response generation (bare acts + case laws + explanations)
3. Case law confirmation & indexing
"""

from services.fact_collector import get_next_question_or_complete, is_stop_signal
from services.response_generator import generate_response
from services.case_law_indexer_incremental import index_new_case_laws


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
            case_laws_to_confirm: list (internet case laws for user to confirm),
            indexed: bool (if case laws were indexed)
        }
    """
    if phase == "fact_collection":
        result = get_next_question_or_complete(conversation, current_message)

        if result.get("action") == "complete":
            return {
                "phase": "response_generation",
                "message": "Thank you. I have enough information. Let me research the relevant bare acts and case laws for you.",
                "facts_summary": result.get("facts_summary", current_message),
                "response": None,
                "case_laws_to_confirm": None,
                "indexed": False,
            }
        else:
            return {
                "phase": "fact_collection",
                "message": result.get("question", "Could you share more details?"),
                "facts_summary": None,
                "response": None,
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
            "materials_to_confirm": None,
            "indexed": False,
        }

    elif phase == "confirm_index":
        # Handled by separate index endpoint - not here
        return {
            "phase": "done",
            "message": "Please use the confirm button to index case laws.",
            "facts_summary": facts_summary,
            "response": None,
            "case_laws_to_confirm": None,
            "indexed": False,
        }

    return {
        "phase": "done",
        "message": "How else can I help?",
        "facts_summary": None,
        "response": None,
        "case_laws_to_confirm": None,
        "indexed": False,
    }
