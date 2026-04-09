"""
agents/state.py — Shared LangGraph state for the Nyaymalaw orchestrator.

NyaymalaState flows through every node in the graph. LangGraph automatically
passes it between nodes and merges partial updates returned by each node.

Replaces the loose `workflow_state` dict that was manually threaded through
OrchestratorAgent.run() / run_stream() and the pipeline/chat.py layer.
"""

from __future__ import annotations

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
    """

    messages:      Annotated[list[BaseMessage], add_messages]
    safe:          bool
    pii_warning:   Optional[str]
    session_id:    Optional[str]
    intake_state:  Optional[dict]
    final_reply:   Optional[str]
