"""
intake/session.py — Shared session state helpers for the legal opinion intake workflow.

A session is a plain dict the caller owns (stored in DB or in-memory).
This module provides factories, serialisation helpers, and stage-transition logic.

Session schema
--------------
{
  "session_id": str,
  "stage": "stage1" | "stage2" | "stage3" | "stage4" | "stage5" | "complete",
  "history": [{"role": "user"|"assistant", "content": str}],
  "intake_state": dict,    # stage-specific state (see each stage module)
  "category": str | None,  # locked after Stage 1
  "created_at": float,     # unix timestamp
  "updated_at": float,
}
"""
from __future__ import annotations
import time
import uuid
import copy


def new_session(session_id: str | None = None) -> dict:
    """Create a fresh session dict for a new legal opinion intake."""
    now = time.time()
    return {
        "session_id":   session_id or str(uuid.uuid4()),
        "stage":        "stage1",
        "history":      [],
        "intake_state": {},
        "category":     None,
        "created_at":   now,
        "updated_at":   now,
    }


def append_turn(session: dict, role: str, content: str) -> None:
    """Append a conversation turn and update the timestamp."""
    session.setdefault("history", []).append({"role": role, "content": content})
    session["updated_at"] = time.time()


def advance_stage(session: dict, next_stage: str) -> None:
    """Move the session to the next intake stage."""
    session["stage"] = next_stage
    session["updated_at"] = time.time()


def get_history_context(session: dict, max_turns: int = 8, max_chars: int = 280) -> str:
    """Return a compact conversation history string for prompt injection."""
    history = session.get("history", [])
    lines = []
    for turn in history[-max_turns:]:
        role    = "Client" if turn.get("role") == "user" else "Counsel"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "\u2026"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def clone_session(session: dict) -> dict:
    """Deep copy a session (for testing / branching)."""
    return copy.deepcopy(session)
