"""
Persistent store for case law discovery: documents presented for indexing
and summary index (by act name). Persists across UI refresh and backend restart.
Cleared only when user completes indexing or clicks Clear.
"""

import json
import os
import time
import logging

logger = logging.getLogger(__name__)


def _load_json(path: str, default: dict | list):
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
    return default


def _save_json(path: str, data: dict | list) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not save %s: %s", path, e)


def load_pending():
    """Load documents presented for indexing (survives refresh/restart)."""
    from config import CASE_LAW_DISCOVERY_PENDING_PATH
    raw = _load_json(CASE_LAW_DISCOVERY_PENDING_PATH, {"items": [], "updated_at": None})
    items = raw.get("items", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    return list(items)


def save_pending(items: list) -> None:
    """Persist documents presented for indexing."""
    from config import CASE_LAW_DISCOVERY_PENDING_PATH
    data = {"items": items, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _save_json(CASE_LAW_DISCOVERY_PENDING_PATH, data)


def clear_pending() -> None:
    """Clear persisted list (e.g. when user clicks Clear or after indexing)."""
    save_pending([])


def load_summary_index():
    """Load summary index: act_name -> list of case law signatures (court + parties)."""
    from config import CASE_LAW_DISCOVERY_SUMMARY_INDEX_PATH
    raw = _load_json(CASE_LAW_DISCOVERY_SUMMARY_INDEX_PATH, {})
    return raw if isinstance(raw, dict) else {}


def save_summary_index(index: dict) -> None:
    """Persist summary index (by act name)."""
    from config import CASE_LAW_DISCOVERY_SUMMARY_INDEX_PATH
    _save_json(CASE_LAW_DISCOVERY_SUMMARY_INDEX_PATH, index)


def add_signature_to_summary_index(act_name: str, signature: str) -> None:
    """
    Add a case law signature to the act-case-law map for the given act.
    Call this when a document is successfully stored (e.g. on confirm-index).
    """
    if not act_name or not signature:
        return
    index = load_summary_index()
    lst = index.get(act_name)
    if not isinstance(lst, list):
        lst = []
    if signature not in lst:
        lst.append(signature)
    index[act_name] = lst
    save_summary_index(index)


# ---------------------------------------------------------------------------
# Bare act summary index (act_name -> summary text)
# ---------------------------------------------------------------------------

def load_bare_act_summary_index():
    """Load bare act summary index: act_name -> summary text."""
    from config import BARE_ACT_SUMMARY_INDEX_PATH
    raw = _load_json(BARE_ACT_SUMMARY_INDEX_PATH, {})
    return raw if isinstance(raw, dict) else {}


def save_bare_act_summary_index(index: dict) -> None:
    """Persist bare act summary index."""
    from config import BARE_ACT_SUMMARY_INDEX_PATH
    _save_json(BARE_ACT_SUMMARY_INDEX_PATH, index)


def get_bare_act_summary(act_name: str) -> str:
    """Return summary for act, or empty string if not found."""
    index = load_bare_act_summary_index()
    return (index.get(act_name) or "").strip()


def set_bare_act_summary(act_name: str, summary: str) -> None:
    """Store summary for act."""
    if not act_name:
        return
    index = load_bare_act_summary_index()
    index[act_name] = (summary or "").strip()
    save_bare_act_summary_index(index)


# ---------------------------------------------------------------------------
# Case law summary index (signature -> summary text)
# ---------------------------------------------------------------------------

def load_case_law_summary_index():
    """Load case law summary index: signature -> summary text."""
    from config import CASE_LAW_SUMMARY_INDEX_PATH
    raw = _load_json(CASE_LAW_SUMMARY_INDEX_PATH, {})
    return raw if isinstance(raw, dict) else {}


def save_case_law_summary_index(index: dict) -> None:
    """Persist case law summary index."""
    from config import CASE_LAW_SUMMARY_INDEX_PATH
    _save_json(CASE_LAW_SUMMARY_INDEX_PATH, index)


def get_case_law_summary(signature: str) -> str:
    """Return summary for case law (by signature), or empty string if not found."""
    index = load_case_law_summary_index()
    return (index.get(signature) or "").strip()


def add_case_law_summary(signature: str, summary: str) -> None:
    """Store summary for a case law (by signature). Call after indexing a case law."""
    if not signature:
        return
    index = load_case_law_summary_index()
    index[signature] = (summary or "").strip()
    save_case_law_summary_index(index)
