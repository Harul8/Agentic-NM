"""
agents/tool_registry.py — LangChain @tool definitions for all Nyaymalaw tools.

Replaces the hand-written TOOL_DEFINITIONS JSON schemas + dispatch lambdas.
LangChain generates OpenAI-compatible schemas automatically from the function
signatures and docstrings, so there is nothing to maintain manually.

Tool groups
-----------
Research  : search_bare_acts, search_case_laws, lookup_section, lookup_case
Document  : extract_document_facts, cross_reference_document
Forum     : identify_forum, check_limitation
Intake    : start_intake, continue_intake, get_intake_state, draft_opinion
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger("nyaymalaw.tool_registry")


# ---------------------------------------------------------------------------
# Research tools
# ---------------------------------------------------------------------------

@tool
def search_bare_acts(query: str, top_k: int = 5) -> str:
    """
    Search the local Nyaymalaw vector store for Bare Act sections relevant
    to a legal issue. Returns verbatim statutory text with act name, section
    number, and relevance score.

    Use when you need the LAW — statutes, sections, sub-sections.
    Do NOT use for case law judgments.
    Do NOT use when you already know the exact act + section — use lookup_section.

    Args:
        query: Natural-language description of the legal issue to search for.
        top_k: Number of results to return (1-20). Default 5.
    """
    from mcp_server import _tool_search_bare_acts
    return _tool_search_bare_acts(query, top_k)


@tool
def search_case_laws(query: str, jurisdiction: str = "", top_k: int = 5) -> str:
    """
    Search the local Nyaymalaw vector store for case law paragraphs relevant
    to a legal issue. Returns verbatim judgment text with case name, citation,
    court, year, and paragraph number.

    Use when you need PRECEDENT — court judgments, holdings, reasoning.
    Do NOT use for statutory text.
    Do NOT use when you already know the exact case — use lookup_case.

    Args:
        query:        Natural-language description of the legal issue or fact pattern.
        jurisdiction: Optional — narrow to a state/court (e.g. 'Delhi High Court').
                      Leave blank for all.
        top_k:        Number of results to return (1-20). Default 5.
    """
    from mcp_server import _tool_search_case_laws
    return _tool_search_case_laws(query, jurisdiction, top_k)


@tool
def lookup_section(act_name: str, section_number: str) -> str:
    """
    Fetch the exact verbatim text of a specific section from a specific Act.

    Use this when you already know WHICH act and WHICH section number you need.
    Returns every sub-section chunk stored for that section.
    Do NOT use for open-ended research — use search_bare_acts for discovery.

    Args:
        act_name:       Full or abbreviated act name
                        (e.g. 'Protection of Women from Domestic Violence Act', 'BNS').
        section_number: Section number as a string (e.g. '18', '85', '138').
    """
    from mcp_server import _tool_lookup_section
    return _tool_lookup_section(act_name, section_number)


@tool
def lookup_case(case_name: str, para_num: str = "") -> str:
    """
    Fetch verbatim paragraphs from a specific case judgment stored in the database.

    Use this when you already know the CASE NAME and optionally a paragraph number.
    Returns exact judgment text at paragraph level — cite para_num in your output.
    Do NOT use for open-ended research — use search_case_laws for discovery.

    Args:
        case_name: Full or partial case name (e.g. 'Indra Sarma v. V.K.V. Sarma').
        para_num:  Optional specific paragraph number to fetch (e.g. '14').
                   Leave blank for all paragraphs.
    """
    from mcp_server import _tool_lookup_case
    return _tool_lookup_case(case_name, para_num)


# ---------------------------------------------------------------------------
# Document agent tools
# ---------------------------------------------------------------------------

@tool
def extract_document_facts(file_text: str, dispute_context: str = "") -> str:
    """
    Extract structured facts from an uploaded legal document (rent agreement,
    legal notice, court order, FIR, employment letter, cheque, etc.).

    Pass the plain text of the document (already extracted by /upload-document).
    Returns parties, dates, amounts, legal references, obligations, and a summary.
    Use this when the user has uploaded a document to enrich the intake session.

    Args:
        file_text:        Plain text of the uploaded document (from OCR or text extraction).
        dispute_context:  Brief description of the dispute from the intake session
                          (optional but improves accuracy).
    """
    from agents.document.agent import extract_document_facts as _impl
    return _impl(file_text, dispute_context)


@tool
def cross_reference_document(document_facts_json: str, intake_state: dict) -> str:
    """
    Compare facts extracted from an uploaded document against the client's verbal
    intake account. Returns alignments (consistent facts), discrepancies (conflicts),
    and gaps (things in the document not yet mentioned in intake).

    Use after extract_document_facts to identify what the document adds or contradicts.

    Args:
        document_facts_json: JSON string from extract_document_facts.
        intake_state:        Current intake state dict from get_intake_state.
    """
    from agents.document.agent import cross_reference_document as _impl
    return _impl(document_facts_json, intake_state)


# ---------------------------------------------------------------------------
# Forum agent tools
# ---------------------------------------------------------------------------

@tool
def identify_forum(intake_state_json: str) -> str:
    """
    Identify the correct Indian legal forum for the dispute based on the intake state.

    Returns primary and alternative forum recommendations with jurisdiction notes,
    typical relief available, pecuniary limits, and urgency assessment.
    Use this after intake is complete and before or during draft_opinion.

    Args:
        intake_state_json: JSON string of the intake state (from get_intake_state).
    """
    from agents.forum.agent import identify_forum as _impl
    return _impl(intake_state_json)


@tool
def check_limitation(dispute_type: str, key_dates: list[str]) -> str:
    """
    Check whether the limitation period for a dispute is still open given the
    key event dates. Returns status (open/expiring_soon/expired), days remaining,
    and urgency flag.

    Use this when dates emerge in intake that suggest time may be running.
    Dispute types: cheque_bounce, consumer, contract, property, employment,
    domestic_violence, tort.

    Args:
        dispute_type: Type of dispute
                      (e.g. 'cheque_bounce', 'consumer', 'employment', 'contract').
        key_dates:    List of key event dates as strings (YYYY-MM-DD or DD/MM/YYYY).
    """
    from agents.forum.agent import check_limitation as _impl
    return _impl(dispute_type, key_dates)


# ---------------------------------------------------------------------------
# Intake tools
# ---------------------------------------------------------------------------

@tool
def start_intake(first_message: str = "") -> str:
    """
    Start a new autonomous legal opinion intake session.

    Creates a session and optionally processes the client's first message.
    Returns session_id — store this and pass to continue_intake on every
    subsequent turn.
    Use ONLY to begin a fresh legal opinion workflow, not for research queries.

    Args:
        first_message: The client's first message. Pass it here to process
                       immediately rather than in a separate continue_intake call.
    """
    from mcp_server import _tool_start_intake
    return _tool_start_intake(first_message)


@tool
def continue_intake(session_id: str, client_message: str) -> str:
    """
    Pass the client's next message in an active intake session and get the AI reply.

    Call this once per client turn until advance_to_stage2 is true.
    The AI asks targeted questions, gathers facts, detects legal category,
    and signals when enough information has been collected.
    Do NOT call this without a session_id — call start_intake first.

    Args:
        session_id:     The session_id returned by start_intake.
        client_message: The client's latest message.
    """
    from mcp_server import _tool_continue_intake
    return _tool_continue_intake(session_id, client_message)


@tool
def get_intake_state(session_id: str) -> str:
    """
    Inspect the current structured state of an intake session without advancing it.

    Returns collected facts, detected legal category, urgency signal, and turn count.
    Use this to check readiness before calling draft_opinion, or to show the
    advocate review panel the structured brief.

    Args:
        session_id: The session_id returned by start_intake.
    """
    from mcp_server import _tool_get_intake_state
    return _tool_get_intake_state(session_id)


@tool
def draft_opinion(session_id: str) -> str:
    """
    Generate the full structured legal opinion draft from a completed intake session.

    Returns formatted_draft (markdown with Statement of Facts, Legal Framework,
    Case Law Support, Prayer/Relief, Documents Checklist) and advocate_review
    (structured JSON brief).
    Call this only after advance_to_stage2 was true or get_intake_state shows
    meaningful known_facts. Do NOT call on a fresh or near-empty session.

    Args:
        session_id: The session_id returned by start_intake.
    """
    from mcp_server import _tool_draft_opinion
    return _tool_draft_opinion(session_id)


# ---------------------------------------------------------------------------
# Tool list — passed to LangGraph ToolNode and LLM.bind_tools()
# ---------------------------------------------------------------------------

TOOLS: list = [
    search_bare_acts,
    search_case_laws,
    lookup_section,
    lookup_case,
    extract_document_facts,
    cross_reference_document,
    identify_forum,
    check_limitation,
    start_intake,
    continue_intake,
    get_intake_state,
    draft_opinion,
]
