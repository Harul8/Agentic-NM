"""
core/legal_graph.py — Query interface for the Legal Knowledge Graph (legal.db).

Public API:
    get_db()                                → lazy-load singleton connection
    lookup_section(act_hint, section_number) → section node + chunk_ids (direct lookup)
    get_cross_referenced_sections(section_id) → list of related section nodes
    get_cases_interpreting_section(section_id) → list of case info dicts
    extract_legal_entities(query)           → parsed act/section references from query text
    find_sections_by_query(query)           → direct lookup if query has explicit entity refs
"""

import json
import logging
import os
import re
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy singleton
_db_conn: Optional[sqlite3.Connection] = None


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_db() -> Optional[sqlite3.Connection]:
    """Lazy-load the legal.db connection. Returns None if DB not yet built."""
    global _db_conn
    if _db_conn is not None:
        return _db_conn
    try:
        from config import LEGAL_GRAPH_DB
        if not os.path.isfile(LEGAL_GRAPH_DB):
            logger.debug("legal.db not found at %s (run scripts/build_legal_graph.py)", LEGAL_GRAPH_DB)
            return None
        _db_conn = sqlite3.connect(LEGAL_GRAPH_DB, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        logger.info("Legal graph DB loaded from %s", LEGAL_GRAPH_DB)
    except Exception as e:
        logger.debug("Legal graph DB unavailable: %s", e)
        _db_conn = None
    return _db_conn


def invalidate_db() -> None:
    """Close and clear the singleton so the next call reloads from disk."""
    global _db_conn
    if _db_conn:
        try:
            _db_conn.close()
        except Exception:
            pass
    _db_conn = None


# ---------------------------------------------------------------------------
# Entity extraction from query text
# ---------------------------------------------------------------------------

# Alias → canonical act_id fragments for resolution
_ALIAS_MAP: dict[str, list[str]] = {
    "ipc":          ["indian_penal_code"],
    "bns":          ["bharatiya_nyaya_sanhita"],
    "bnss":         ["bharatiya_nagarik_suraksha_sanhita"],
    "bsa":          ["bharatiya_sakshya_adhiniyam"],
    "crpc":         ["code_of_criminal_procedure"],
    "cpc":          ["code_of_civil_procedure"],
    "iea":          ["indian_evidence_act"],
    "hma":          ["hindu_marriage_act"],
    "ni act":       ["negotiable_instruments_act"],
    "tp act":       ["transfer_of_property_act"],
    "sra":          ["specific_relief_act"],
    "hsa":          ["hindu_succession_act"],
    "pwdva":        ["protection_of_women"],
    "mv act":       ["motor_vehicles_act"],
    "it act":       ["income_tax_act"],
    "constitution": ["constitution_of_india"],
}

# Patterns to detect explicit legal entity references in query text
# Matches: "IPC Section 302", "Section 302 IPC", "IPC 302", "Article 21 of Constitution"
_ENTITY_PATTERNS = [
    # "IPC Section 302" / "BNS Section 4"
    re.compile(
        r"\b(IPC|BNS|BNSS|BSA|CrPC|CPC|IEA|HMA|NI Act|TP Act|SRA|HSA|PWDVA|MV Act|IT Act|Constitution)\b"
        r"[\s,]*(?:Section|Sec\.?|S\.?|Article|Art\.?)?\s*(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\b",
        re.IGNORECASE,
    ),
    # "Section 302 IPC" / "Section 302 of the IPC"
    re.compile(
        r"\b(?:Section|Sec\.?|Article|Art\.?)\s+(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\b"
        r"(?:\s+of\s+(?:the\s+)?)?"
        r"\b(IPC|BNS|BNSS|BSA|CrPC|CPC|IEA|HMA|NI Act|TP Act|SRA|HSA|PWDVA|MV Act|IT Act|Constitution)\b",
        re.IGNORECASE,
    ),
    # "Section 302" alone (no act hint — try all acts)
    re.compile(
        r"\b(?:Section|Sec\.?|Article|Art\.?)\s+(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\b",
        re.IGNORECASE,
    ),
]


def extract_legal_entities(query: str) -> list[dict]:
    """
    Parse explicit section references from query text.
    Returns list of dicts: [{section_number, act_hint_raw, pattern_type}, ...]
    """
    results = []
    seen = set()
    query = query or ""

    # Pattern 0: "ALIAS Section N" or "ALIAS N"
    for m in _ENTITY_PATTERNS[0].finditer(query):
        alias_raw  = m.group(1)
        sec_num    = m.group(2)
        key = (alias_raw.upper(), sec_num)
        if key not in seen:
            seen.add(key)
            results.append({"section_number": sec_num, "act_hint": alias_raw.upper(), "pattern": 0})

    # Pattern 1: "Section N of ALIAS"
    for m in _ENTITY_PATTERNS[1].finditer(query):
        sec_num   = m.group(1)
        alias_raw = m.group(2)
        key = (alias_raw.upper(), sec_num)
        if key not in seen:
            seen.add(key)
            results.append({"section_number": sec_num, "act_hint": alias_raw.upper(), "pattern": 1})

    # Pattern 2: "Section N" alone — only if no richer matches found
    if not results:
        for m in _ENTITY_PATTERNS[2].finditer(query):
            sec_num = m.group(1)
            key = ("ANY", sec_num)
            if key not in seen:
                seen.add(key)
                results.append({"section_number": sec_num, "act_hint": None, "pattern": 2})

    return results


# ---------------------------------------------------------------------------
# Core lookup functions
# ---------------------------------------------------------------------------

def lookup_section(
    section_number: str,
    act_hint: Optional[str] = None,
) -> list[dict]:
    """
    Direct lookup of a section by number (and optional act alias/name fragment).
    Returns list of section row dicts (may be >1 if same section number appears in
    multiple acts and act_hint is ambiguous).

    Each returned dict has:
        id, act_id, act_name, section_number, section_title, chapter, chunk_ids (list)
    """
    db = get_db()
    if db is None:
        return []

    try:
        if act_hint:
            # Resolve alias to act_id fragment
            hint_lower = act_hint.lower().replace(" ", "_")
            fragments  = _ALIAS_MAP.get(act_hint.lower(), [hint_lower])

            rows = []
            for frag in fragments:
                rows += db.execute(
                    "SELECT * FROM sections WHERE section_number = ? AND act_id LIKE ?",
                    (section_number, f"%{frag}%"),
                ).fetchall()
            # Deduplicate
            seen_ids = set()
            unique_rows = []
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_rows.append(r)
            rows = unique_rows
        else:
            rows = db.execute(
                "SELECT * FROM sections WHERE section_number = ?",
                (section_number,),
            ).fetchall()

        return [_section_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("lookup_section error: %s", e)
        return []


def get_cross_referenced_sections(section_id: str, max_results: int = 10) -> list[dict]:
    """
    Return sections that the given section explicitly references in its text.
    (section_cross_refs.src_section_id = section_id)
    """
    db = get_db()
    if db is None:
        return []
    try:
        rows = db.execute(
            """SELECT s.* FROM sections s
               JOIN section_cross_refs xr ON xr.dst_section_id = s.id
               WHERE xr.src_section_id = ?
               LIMIT ?""",
            (section_id, max_results),
        ).fetchall()
        return [_section_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_cross_referenced_sections error: %s", e)
        return []


def get_cases_interpreting_section(section_id: str, max_results: int = 20) -> list[dict]:
    """
    Return cases (from citation graph) that have interpreted the given section.
    Each result dict: {case_id, case_name, court, year}
    """
    db = get_db()
    if db is None:
        return []
    try:
        rows = db.execute(
            """SELECT case_id, case_name, court, year
               FROM case_section_links
               WHERE section_id = ?
               LIMIT ?""",
            (section_id, max_results),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_cases_interpreting_section error: %s", e)
        return []


def get_sections_in_chapter(act_id: str, chapter: str) -> list[dict]:
    """Return all sections belonging to a given act+chapter."""
    db = get_db()
    if db is None:
        return []
    try:
        rows = db.execute(
            "SELECT * FROM sections WHERE act_id = ? AND chapter = ? ORDER BY section_number",
            (act_id, chapter),
        ).fetchall()
        return [_section_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_sections_in_chapter error: %s", e)
        return []


# ---------------------------------------------------------------------------
# High-level: query → direct chunk_ids (used by retriever pre-filter)
# ---------------------------------------------------------------------------

def find_sections_by_query(query: str) -> list[dict]:
    """
    If query contains explicit section references, look them up directly in legal.db.
    Returns list of section dicts (each with chunk_ids list) or [] if no explicit refs.

    This is the pre-filter entry point called from retriever.py.
    """
    entities = extract_legal_entities(query)
    if not entities:
        return []

    results: list[dict] = []
    seen_ids: set = set()

    for entity in entities:
        sec_num  = entity["section_number"]
        act_hint = entity.get("act_hint")
        sections = lookup_section(sec_num, act_hint)
        for sec in sections:
            if sec["id"] not in seen_ids:
                seen_ids.add(sec["id"])
                results.append(sec)

    if results:
        logger.debug(
            "Legal graph direct lookup: query matched %d section(s) for entities %s",
            len(results), [e["section_number"] for e in entities],
        )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Deserialize chunk_ids from JSON string to list
    try:
        d["chunk_ids"] = json.loads(d.get("chunk_ids") or "[]")
    except Exception:
        d["chunk_ids"] = []
    return d
