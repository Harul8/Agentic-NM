"""
agents/document/graph.py — Document extraction and cross-reference as a
LangGraph subgraph.

Wraps the existing agents/document/agent.py functions in a proper StateGraph
so that each processing stage appears as a named node in LangSmith traces.
Each stage can be interrupted independently for human review in future.

Graph structure
---------------

    START
      │
      ▼
  [classify]      (heuristic, fast — no LLM)
      │
      ▼
  [extract]       (LLM call — summary + obligations)
      │
      ├── intake_state present → [cross_reference] → END
      │
      └── no intake_state ───────────────────────► END

Public entry point
------------------
    run_document_analysis(file_text, dispute_context, intake_state) → dict
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger("nyaymalaw.document.graph")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DocumentState(TypedDict):
    file_text:               str
    dispute_context:         str
    intake_state:            Optional[dict]   # present when cross-ref is needed
    doc_type:                Optional[str]    # set by classify node
    extracted_facts:         Optional[str]    # JSON from extract node
    cross_reference_result:  Optional[str]    # JSON from cross_ref node


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _classify_node(state: DocumentState) -> dict:
    """
    Heuristic document type classification — no LLM, instant.
    Sets doc_type in state for use by downstream nodes and logging.
    """
    from agents.document.agent import _classify_document_type
    try:
        doc_type = _classify_document_type(state.get("file_text") or "")
        logger.info("Document classified as: %s", doc_type)
        return {"doc_type": doc_type}
    except Exception as exc:
        logger.warning("classify node failed: %s", exc)
        return {"doc_type": "general_legal_document"}


def _extract_node(state: DocumentState) -> dict:
    """
    Structured fact extraction — one LLM call for summary and obligations.
    Heuristic extraction (parties, dates, amounts) happens inside this node too.
    """
    from agents.document.agent import extract_document_facts
    try:
        result = extract_document_facts(
            state.get("file_text") or "",
            state.get("dispute_context") or "",
        )
        return {"extracted_facts": result}
    except Exception as exc:
        logger.exception("extract node failed: %s", exc)
        return {"extracted_facts": json.dumps({"error": str(exc)})}


def _cross_reference_node(state: DocumentState) -> dict:
    """
    Cross-reference extracted document facts against the verbal intake account.
    Only runs if intake_state is present in state.
    """
    from agents.document.agent import cross_reference_document
    try:
        result = cross_reference_document(
            state.get("extracted_facts") or "{}",
            state.get("intake_state") or {},
        )
        return {"cross_reference_result": result}
    except Exception as exc:
        logger.exception("cross_reference node failed: %s", exc)
        return {"cross_reference_result": json.dumps({"error": str(exc)})}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_extract(state: DocumentState):
    """Run cross-reference only when intake state is available."""
    if state.get("intake_state"):
        return "cross_reference"
    return END


# ---------------------------------------------------------------------------
# Graph compilation (once at module import)
# ---------------------------------------------------------------------------

def _build_document_graph():
    builder = StateGraph(DocumentState)

    builder.add_node("classify",         _classify_node)
    builder.add_node("extract",          _extract_node)
    builder.add_node("cross_reference",  _cross_reference_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "extract")
    builder.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"cross_reference": "cross_reference", END: END},
    )
    builder.add_edge("cross_reference", END)

    # No checkpointer — document analysis is a one-shot pipeline
    return builder.compile()


document_subgraph = _build_document_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_document_analysis(
    file_text: str,
    dispute_context: str = "",
    intake_state: Optional[dict] = None,
) -> dict:
    """
    Run document classification → extraction → cross-reference (if applicable).

    Parameters
    ----------
    file_text       : Plain text of the uploaded document (OCR output).
    dispute_context : Brief verbal description from the intake session.
    intake_state    : Current intake state dict. When provided, triggers the
                      cross-reference node to flag discrepancies.

    Returns
    -------
    dict: {doc_type, extracted_facts (JSON str), cross_reference_result (JSON str|None)}
    """
    if not (file_text or "").strip():
        return {"error": "No document text provided"}

    try:
        result = document_subgraph.invoke({
            "file_text":              file_text,
            "dispute_context":        dispute_context or "",
            "intake_state":           intake_state,
            "doc_type":               None,
            "extracted_facts":        None,
            "cross_reference_result": None,
        })
        return {
            "doc_type":               result.get("doc_type"),
            "extracted_facts":        result.get("extracted_facts"),
            "cross_reference_result": result.get("cross_reference_result"),
        }
    except Exception as exc:
        logger.exception("run_document_analysis failed: %s", exc)
        return {"error": str(exc)}
