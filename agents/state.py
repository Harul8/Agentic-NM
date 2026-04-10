"""
agents/state.py — Shared LangGraph state for the Nyaymalaw orchestrator.

NyaymalaState flows through every node in the graph. LangGraph automatically
passes it between nodes and merges partial updates returned by each node.

Replaces the loose `workflow_state` dict that was manually threaded through
OrchestratorAgent.run() / run_stream() and the pipeline/chat.py layer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class NyaymalaState(TypedDict):
    """
    Full state carried through the Nyaymalaw LangGraph.

    Fields
    ------
    messages:
        Full conversation including system, human, AI, and tool messages.
        `add_messages` is LangGraph's built-in reducer — it appends new
        messages rather than overwriting the list, so each node only needs
        to return the messages it wants to add.

    safe:
        Set by the safety_check node. False causes the graph to route
        straight to END without calling the LLM.

    pii_warning:
        Non-blocking PII detection note forwarded to the client.

    session_id:
        Active intake session identifier. Persisted across turns so
        continue_intake always receives the correct session.

    intake_state:
        Structured facts collected by the intake agent so far.
        Forwarded to the frontend advocate-review panel.

    final_reply:
        Final text answer after the grounding guard has approved it.
        Populated by the grounding_guard node and read by the API layer.

    user_id:
        Identifies the user across sessions. Used by the cross-session
        memory store to inject prior case summaries into the system prompt.

    token_count:
        Total character count of all messages this session. Tracked by
        the rate_limit node to enforce a per-session context budget.

    retry_count:
        Number of tool-retry cycles consumed this turn. Prevents the
        tool_retry node from looping indefinitely on persistent failures.

    research_results:
        Pre-populated retrieval results from the Send() parallel fan-out.
        Uses operator.add so each research_worker branch appends its results
        rather than replacing the whole list.
    """

    messages:          Annotated[list[BaseMessage], add_messages]
    safe:              bool
    pii_warning:       Optional[str]
    session_id:        Optional[str]
    intake_state:      Optional[dict]
    final_reply:       Optional[str]
    user_id:           Optional[str]
    token_count:       int
    retry_count:       int
    research_results:  Annotated[list[dict], operator.add]
