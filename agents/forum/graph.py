"""
agents/forum/graph.py — Forum identification and limitation check as a
LangGraph subgraph.

Wraps the existing agents/forum/agent.py functions in a proper StateGraph
so that each step appears as a named node in LangSmith traces, and the
subgraph can be interrupted mid-execution for human review in future.

Graph structure
---------------

    START
      │
      ▼
  [identify_forum]
      │
      ├── key_dates present → [check_limitation] → END
      │
      └── no key_dates ──────────────────────────► END

Public entry point
------------------
    run_forum_analysis(intake_state_json, dispute_type, key_dates) → dict
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger("nyaymalaw.forum.graph")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ForumState(TypedDict):
    intake_state_json:  str
    dispute_type:       Optional[str]
    key_dates:          list
    forum_result:       Optional[str]      # JSON string from identify_forum
    limitation_result:  Optional[str]      # JSON string from check_limitation


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _identify_forum_node(state: ForumState) -> dict:
    """Run forum identification — pure rule-based, no LLM."""
    from agents.forum.agent import identify_forum
    try:
        result = identify_forum(state.get("intake_state_json") or "{}")
        return {"forum_result": result}
    except Exception as exc:
        logger.warning("identify_forum node failed: %s", exc)
        return {"forum_result": json.dumps({"error": str(exc)})}


def _check_limitation_node(state: ForumState) -> dict:
    """Run limitation period check — pure date arithmetic, no LLM."""
    from agents.forum.agent import check_limitation
    try:
        dispute_type = state.get("dispute_type") or "general"
        key_dates    = state.get("key_dates") or []
        result = check_limitation(dispute_type, key_dates)
        return {"limitation_result": result}
    except Exception as exc:
        logger.warning("check_limitation node failed: %s", exc)
        return {"limitation_result": json.dumps({"error": str(exc)})}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_forum(state: ForumState):
    """Run limitation check only if key dates are available."""
    if state.get("key_dates"):
        return "check_limitation"
    return END


# ---------------------------------------------------------------------------
# Graph compilation (once at module import)
# ---------------------------------------------------------------------------

def _build_forum_graph():
    builder = StateGraph(ForumState)

    builder.add_node("identify_forum",    _identify_forum_node)
    builder.add_node("check_limitation",  _check_limitation_node)

    builder.add_edge(START, "identify_forum")
    builder.add_conditional_edges(
        "identify_forum",
        _route_after_forum,
        {"check_limitation": "check_limitation", END: END},
    )
    builder.add_edge("check_limitation", END)

    # No checkpointer needed — forum analysis is stateless
    return builder.compile()


forum_subgraph = _build_forum_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_forum_analysis(
    intake_state_json: str,
    dispute_type: str = "",
    key_dates: Optional[list] = None,
) -> dict:
    """
    Run forum identification and (if dates are present) limitation check.

    Parameters
    ----------
    intake_state_json : JSON string of the intake state dict.
    dispute_type      : Dispute type key for limitation lookup
                        (e.g. 'cheque_bounce', 'consumer', 'employment').
    key_dates         : List of key event date strings for limitation calc.

    Returns
    -------
    dict: {forum_result: str (JSON), limitation_result: str|None (JSON)}
    """
    try:
        result = forum_subgraph.invoke({
            "intake_state_json": intake_state_json or "{}",
            "dispute_type":      dispute_type or "",
            "key_dates":         key_dates or [],
            "forum_result":      None,
            "limitation_result": None,
        })
        return {
            "forum_result":      result.get("forum_result"),
            "limitation_result": result.get("limitation_result"),
        }
    except Exception as exc:
        logger.exception("run_forum_analysis failed: %s", exc)
        return {
            "forum_result":      json.dumps({"error": str(exc)}),
            "limitation_result": None,
        }
