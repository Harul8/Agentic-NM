"""
agents/tool_registry.py — OpenAI function-calling definitions for all 8 Nyaymalaw tools.

Each entry maps to an implementation in mcp_server.py.
The orchestrator uses TOOL_DEFINITIONS as the `tools` parameter in every
OpenAI API call. TOOL_DISPATCH maps name → callable so the orchestrator
can execute whatever the model requests without a long if/elif chain.

Tool groups:
    Research  : search_bare_acts, search_case_laws, lookup_section, lookup_case
    Intake    : start_intake, continue_intake, get_intake_state, draft_opinion
"""

from __future__ import annotations

import logging

logger = logging.getLogger("nyaymalaw.tool_registry")

# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling format
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [

    # ── Research ─────────────────────────────────────────────────────────────

    {
        "type": "function",
        "function": {
            "name": "search_bare_acts",
            "description": (
                "Search the local Nyaymalaw vector store for Bare Act sections relevant "
                "to a legal issue. Returns verbatim statutory text with act name, section "
                "number, and relevance score. "
                "Use when you need the LAW — statutes, sections, sub-sections. "
                "Do NOT use for case law judgments. "
                "Do NOT use when you already know the exact act + section — use lookup_section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the legal issue to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1–20). Default 5.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_case_laws",
            "description": (
                "Search the local Nyaymalaw vector store for case law paragraphs relevant "
                "to a legal issue. Returns verbatim judgment text with case name, citation, "
                "court, year, and paragraph number. "
                "Use when you need PRECEDENT — court judgments, holdings, reasoning. "
                "Do NOT use for statutory text. "
                "Do NOT use when you already know the exact case — use lookup_case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the legal issue or fact pattern."
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Optional: narrow to a state/court (e.g. 'Delhi High Court', 'Supreme Court'). Leave blank for all."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1–20). Default 5.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "lookup_section",
            "description": (
                "Fetch the exact verbatim text of a specific section from a specific Act. "
                "Use this when you already know WHICH act and WHICH section number you need. "
                "Returns every sub-section chunk stored for that section. "
                "Do NOT use for open-ended research — use search_bare_acts for discovery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "act_name": {
                        "type": "string",
                        "description": "Full or abbreviated act name (e.g. 'Protection of Women from Domestic Violence Act', 'PWDVA', 'BNS')."
                    },
                    "section_number": {
                        "type": "string",
                        "description": "Section number as a string (e.g. '18', '85', '138')."
                    }
                },
                "required": ["act_name", "section_number"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "lookup_case",
            "description": (
                "Fetch verbatim paragraphs from a specific case judgment stored in the database. "
                "Use this when you already know the CASE NAME and optionally a paragraph number. "
                "Returns exact judgment text at paragraph level — cite para_num in your output. "
                "Do NOT use for open-ended research — use search_case_laws for discovery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_name": {
                        "type": "string",
                        "description": "Full or partial case name (e.g. 'Indra Sarma v. V.K.V. Sarma')."
                    },
                    "para_num": {
                        "type": "string",
                        "description": "Optional: specific paragraph number to fetch (e.g. '14'). Leave blank for all paragraphs."
                    }
                },
                "required": ["case_name"]
            }
        }
    },

    # ── Document Agent ────────────────────────────────────────────────────────

    {
        "type": "function",
        "function": {
            "name": "extract_document_facts",
            "description": (
                "Extract structured facts from an uploaded legal document (rent agreement, "
                "legal notice, court order, FIR, employment letter, cheque, etc.). "
                "Pass the plain text of the document (already extracted by /upload-document). "
                "Returns parties, dates, amounts, legal references, obligations, and a summary. "
                "Use this when the user has uploaded a document to enrich the intake session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_text": {
                        "type": "string",
                        "description": "Plain text of the uploaded document (from OCR or text extraction)."
                    },
                    "dispute_context": {
                        "type": "string",
                        "description": "Brief description of the dispute from the intake session (optional but improves accuracy)."
                    }
                },
                "required": ["file_text"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "cross_reference_document",
            "description": (
                "Compare facts extracted from an uploaded document against the client's verbal "
                "intake account. Returns alignments (consistent facts), discrepancies (conflicts), "
                "and gaps (things in the document not yet mentioned in intake). "
                "Use after extract_document_facts to identify what the document adds or contradicts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_facts_json": {
                        "type": "string",
                        "description": "JSON string from extract_document_facts."
                    },
                    "intake_state": {
                        "type": "object",
                        "description": "Current intake state dict from get_intake_state."
                    }
                },
                "required": ["document_facts_json", "intake_state"]
            }
        }
    },

    # ── Forum Agent ───────────────────────────────────────────────────────────

    {
        "type": "function",
        "function": {
            "name": "identify_forum",
            "description": (
                "Identify the correct Indian legal forum for the dispute based on the intake state. "
                "Returns primary and alternative forum recommendations with jurisdiction notes, "
                "typical relief available, pecuniary limits, and urgency assessment. "
                "Use this after intake is complete and before or during draft_opinion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intake_state_json": {
                        "type": "string",
                        "description": "JSON string of the intake state (from get_intake_state)."
                    }
                },
                "required": ["intake_state_json"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "check_limitation",
            "description": (
                "Check whether the limitation period for a dispute is still open given the "
                "key event dates. Returns status (open/expiring_soon/expired), days remaining, "
                "and urgency flag. "
                "Use this when dates emerge in intake that suggest time may be running. "
                "Dispute types: cheque_bounce, consumer, contract, property, employment, "
                "domestic_violence, tort."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dispute_type": {
                        "type": "string",
                        "description": "Type of dispute (e.g. 'cheque_bounce', 'consumer', 'employment', 'contract')."
                    },
                    "key_dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key event dates as strings (YYYY-MM-DD or DD/MM/YYYY format)."
                    }
                },
                "required": ["dispute_type", "key_dates"]
            }
        }
    },

    # ── Intake ────────────────────────────────────────────────────────────────

    {
        "type": "function",
        "function": {
            "name": "start_intake",
            "description": (
                "Start a new autonomous legal opinion intake session. "
                "Creates a session and optionally processes the client's first message. "
                "Returns session_id — store this and pass to continue_intake on every "
                "subsequent turn. "
                "Use ONLY to begin a fresh legal opinion workflow, not for research queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "first_message": {
                        "type": "string",
                        "description": "The client's first message. Pass it here to process immediately rather than in a separate continue_intake call."
                    }
                },
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "continue_intake",
            "description": (
                "Pass the client's next message in an active intake session and get the AI reply. "
                "Call this once per client turn until advance_to_stage2 is true. "
                "The AI asks targeted questions, gathers facts, detects legal category, "
                "and signals when enough information has been collected. "
                "Do NOT call this without a session_id — call start_intake first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id returned by start_intake."
                    },
                    "client_message": {
                        "type": "string",
                        "description": "The client's latest message."
                    }
                },
                "required": ["session_id", "client_message"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_intake_state",
            "description": (
                "Inspect the current structured state of an intake session without advancing it. "
                "Returns collected facts, detected legal category, urgency signal, and turn count. "
                "Use this to check readiness before calling draft_opinion, "
                "or to show the advocate review panel the structured brief."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id returned by start_intake."
                    }
                },
                "required": ["session_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "draft_opinion",
            "description": (
                "Generate the full structured legal opinion draft from a completed intake session. "
                "Returns formatted_draft (markdown with Statement of Facts, Legal Framework, "
                "Case Law Support, Prayer/Relief, Documents Checklist) and advocate_review "
                "(structured JSON brief). "
                "Call this only after advance_to_stage2 was true or get_intake_state shows "
                "meaningful known_facts. Do NOT call on a fresh or near-empty session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id returned by start_intake."
                    }
                },
                "required": ["session_id"]
            }
        }
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch — maps function name → implementation
# ---------------------------------------------------------------------------

def _build_dispatch() -> dict:
    """
    Lazily build the dispatch map.
    Imports are deferred so the registry can be imported without triggering
    the full mcp_server module init (which warms FAISS/BM25 indexes).
    """
    from mcp_server import (
        _tool_search_bare_acts,
        _tool_search_case_laws,
        _tool_lookup_section,
        _tool_lookup_case,
        _tool_start_intake,
        _tool_continue_intake,
        _tool_get_intake_state,
        _tool_draft_opinion,
    )
    from agents.document.agent import extract_document_facts, cross_reference_document
    from agents.forum.agent import identify_forum, check_limitation

    return {
        # Research
        "search_bare_acts":  lambda args: _tool_search_bare_acts(
            args["query"], args.get("top_k", 5)
        ),
        "search_case_laws":  lambda args: _tool_search_case_laws(
            args["query"], args.get("jurisdiction", ""), args.get("top_k", 5)
        ),
        "lookup_section":    lambda args: _tool_lookup_section(
            args["act_name"], args["section_number"]
        ),
        "lookup_case":       lambda args: _tool_lookup_case(
            args["case_name"], args.get("para_num", "")
        ),
        # Intake
        "start_intake":      lambda args: _tool_start_intake(
            args.get("first_message", "")
        ),
        "continue_intake":   lambda args: _tool_continue_intake(
            args["session_id"], args["client_message"]
        ),
        "get_intake_state":  lambda args: _tool_get_intake_state(
            args["session_id"]
        ),
        "draft_opinion":     lambda args: _tool_draft_opinion(
            args["session_id"]
        ),
        # Document agent
        "extract_document_facts":   lambda args: extract_document_facts(
            args["file_text"], args.get("dispute_context", "")
        ),
        "cross_reference_document": lambda args: cross_reference_document(
            args["document_facts_json"], args.get("intake_state", {})
        ),
        # Forum agent
        "identify_forum":    lambda args: identify_forum(
            args["intake_state_json"]
        ),
        "check_limitation":  lambda args: check_limitation(
            args["dispute_type"], args.get("key_dates", [])
        ),
    }


_DISPATCH: dict | None = None


def dispatch_tool(name: str, args: dict) -> str:
    """
    Execute a named tool with the given argument dict.
    Returns the tool's JSON string result.
    Raises KeyError if the tool name is unknown.
    """
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()

    fn = _DISPATCH.get(name)
    if fn is None:
        raise KeyError(f"Unknown tool: {name!r}")

    logger.debug("Dispatching tool %r with args %s", name, list(args.keys()))
    return fn(args)
