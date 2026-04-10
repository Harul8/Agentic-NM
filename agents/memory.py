"""
agents/memory.py — Cross-session user memory backed by SQLite.

Stores structured case summaries per user so the intake agent can skip
questions the user has already answered in prior sessions and surface
relevant prior context.

Design
------
- Keyed by (user_id, session_id).
- Written once per session, when the advocate approves the draft.
- Read at the start of every orchestrator turn to inject prior context
  into the system prompt — zero LLM cost, pure DB lookup.
- The last 5 sessions per user are surfaced; older sessions pruned after
  30 days to keep the context window lean.
- Fully optional — all code degrades gracefully if DB_PATH is unavailable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger("nyaymalaw.memory")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_memory (
    user_id     TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    category    TEXT,
    summary     TEXT,
    urgency     TEXT,
    jurisdiction TEXT,
    facts_count INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, session_id)
);
"""

_PRUNE_SQL = """
DELETE FROM user_memory
WHERE user_id = ?
  AND created_at < datetime('now', '-30 days');
"""


# ---------------------------------------------------------------------------
# Internal connection helper
# ---------------------------------------------------------------------------

def _get_db_path() -> Optional[str]:
    try:
        from config import DB_PATH
        return DB_PATH
    except Exception:
        return None


def _get_conn() -> Optional[sqlite3.Connection]:
    path = _get_db_path()
    if not path:
        return None
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute(_CREATE_TABLE)
        conn.commit()
        return conn
    except Exception as exc:
        logger.warning("user_memory DB init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_user_memory(user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the last `limit` session summaries for this user, most recent first.

    Returns an empty list if user_id is blank, DB is unavailable, or the
    user has no prior sessions.
    """
    if not user_id:
        return []

    conn = _get_conn()
    if conn is None:
        return []

    try:
        rows = conn.execute(
            """
            SELECT session_id, created_at, category, summary, urgency,
                   jurisdiction, facts_count
            FROM user_memory
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "session_id":   r[0],
                "date":         r[1],
                "category":     r[2] or "unknown",
                "summary":      r[3] or "",
                "urgency":      r[4] or "unknown",
                "jurisdiction": r[5] or "unknown",
                "facts_count":  r[6] or 0,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("get_user_memory failed for user=%s: %s", user_id, exc)
        return []
    finally:
        conn.close()


def save_session_summary(user_id: str, session_id: str, intake_state: dict) -> None:
    """
    Persist a session summary once the advocate has approved the draft.

    Safe to call multiple times — uses INSERT OR REPLACE so re-approval
    just overwrites the same row.
    """
    if not user_id or not session_id:
        return

    conn = _get_conn()
    if conn is None:
        return

    try:
        category     = intake_state.get("primary_issue_cluster") or intake_state.get("category") or ""
        summary      = (intake_state.get("issue_summary") or "")[:500]
        urgency      = intake_state.get("urgency_signal") or "unknown"
        jurisdiction = intake_state.get("jurisdiction") or "unknown"
        facts_count  = len(intake_state.get("known_facts") or [])

        conn.execute(
            """
            INSERT OR REPLACE INTO user_memory
                (user_id, session_id, category, summary, urgency, jurisdiction, facts_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, category, summary, urgency, jurisdiction, facts_count),
        )
        conn.execute(_PRUNE_SQL, (user_id,))
        conn.commit()
        logger.info(
            "user_memory saved: user=%s session=%s category=%s facts=%d",
            user_id, session_id, category, facts_count,
        )
    except Exception as exc:
        logger.warning("save_session_summary failed: %s", exc)
    finally:
        conn.close()


def format_memory_for_prompt(prior_sessions: list[dict]) -> str:
    """
    Convert the list returned by get_user_memory() into a compact prompt
    block that can be appended to the system prompt.

    Returns an empty string if the list is empty.
    """
    if not prior_sessions:
        return ""

    lines = ["PRIOR CLIENT SESSIONS (most recent first):"]
    for s in prior_sessions:
        line = (
            f"  [{s['date'][:10]}] {s['category'].replace('_', ' ').title()}"
            f" | urgency={s['urgency']} | {s['facts_count']} facts collected"
        )
        if s.get("summary"):
            line += f"\n    Summary: {s['summary'][:200]}"
        lines.append(line)

    lines.append(
        "\nUse this context to skip questions the client has already answered "
        "and to identify patterns across sessions. Do NOT reveal session IDs to the client."
    )
    return "\n".join(lines)
