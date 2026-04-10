"""
agents/intake/state.py — LangGraph state for the Nyaymalaw intake subgraph.

IntakeState flows through every node in the intake graph. LangGraph checkpoints
it to SQLite after every node so intake sessions survive server restarts and can
be resumed by any worker process without the in-memory _SESSIONS dict.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IntakeState(TypedDict):
    """
    State carried through the Nyaymalaw intake LangGraph subgraph.

    Fields
    ------
    messages:
        Full conversation including HumanMessages and AIMessages.
        `add_messages` appends rather than overwrites on each node return.

    session_id:
        Unique intake session identifier.  Also used as the LangGraph
        thread_id so the checkpointer can locate the right snapshot.

    intake_state:
        Running structured fact collection state built up by Stage 1.
        Contains known_facts, primary_issue_cluster, urgency_signal, and the
        canonical case_file with schema versions, pipeline-layer status, and
        audit metadata.

    advance_to_stage2:
        Set to True by the process_turn node when the compact intake flow has
        enough material for grounded legal analysis. The router uses this to
        break the intake loop and move to draft generation.

    reply:
        Latest AI reply text.  Persisted in state (before the interrupt)
        so the tool caller can read it from the snapshot without needing
        to parse the messages list.

    urgency_signal:
        Current urgency level: "immediate" | "near_term" | "unknown".
        Surfaced in the tool JSON so the orchestrator can flag urgent cases.

    advocate_approved:
        Set to True by the advocate_review node once a human advocate has
        approved the generated draft.  False means either not yet reviewed
        or the advocate requested revisions.

    advocate_notes:
        Revision instructions from the advocate when advocate_approved is
        False.  Injected into the next draft generation pass so the LLM
        can act on the feedback rather than regenerating identically.

    user_id:
        Optional user identifier passed from the orchestrator.  Used to
        save the session summary to the cross-session memory store when
        the draft is approved.
    """

    messages:          Annotated[list[BaseMessage], add_messages]
    session_id:        str
    intake_state:      dict
    advance_to_stage2: bool
    reply:             str
    urgency_signal:    str
    advocate_approved: bool
    advocate_notes:    str
    user_id:           Optional[str]
