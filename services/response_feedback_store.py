"""
Append-only storage for qualitative feedback on individual assistant responses.

This is intentionally separate from the older interaction-level Excel feedback log.
The goal here is lightweight, per-response learning signals that can later be
converted into eval cases, few-shot corrections, or fine-tuning data.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LEGAL_DB_CHAT_HISTORY

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

RESPONSE_FEEDBACK_TAGS: tuple[str, ...] = (
    "wrong_followup",
    "repeated_question",
    "premature_proceed",
    "missed_urgency",
    "missed_prior_actions",
    "missed_client_objective",
    "poor_empathy",
    "poor_clarity",
    "unsupported_legal_reference",
    "poor_grounding",
    "hallucinated_query_expansion",
    "bad_stop_continue_judgment",
    "too_verbose",
    "too_slow",
    "strong_reasoning",
    "strong_empathy",
    "strong_grounding",
)

ALLOWED_RATINGS: tuple[str, ...] = ("good", "okay", "bad")


def feedback_store_path() -> Path:
    base = Path(LEGAL_DB_CHAT_HISTORY)
    base.mkdir(parents=True, exist_ok=True)
    return base / "response_feedback.jsonl"


def append_response_feedback(payload: dict[str, Any]) -> None:
    """
    Append one response-feedback record as JSONL.
    """
    record = dict(payload or {})
    record["logged_at"] = datetime.now(timezone.utc).isoformat()

    tags = [str(t).strip() for t in (record.get("reason_tags") or []) if str(t).strip()]
    record["reason_tags"] = [t for t in tags if t in RESPONSE_FEEDBACK_TAGS]

    rating = str(record.get("rating") or "").strip().lower()
    if rating not in ALLOWED_RATINGS:
        raise ValueError(f"Invalid rating: {rating!r}")
    record["rating"] = rating

    path = feedback_store_path()
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    logger.info(
        "Stored response feedback | message_id=%s rating=%s tags=%s",
        record.get("message_id", ""),
        rating,
        ",".join(record["reason_tags"]),
    )

