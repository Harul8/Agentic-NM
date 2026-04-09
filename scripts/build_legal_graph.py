"""
Build the Legal Knowledge Graph (legal.db) from existing chunk stores.

Graph contents:
  Nodes:
    - acts     — one row per unique act  (act_name, alias, year)
    - sections — one row per unique section, with list of chunk_ids that cover it

  Edges:
    - section_cross_refs  — Section A references Section B inside its own text
    - case_section_links  — Case X (from citation graph) interprets Section Y (reverse index)

Run after the v2 indexes + citation graph already exist.
Writes legal.db to LEGAL_GRAPH_DB (legal_database/vector_store/legal.db).

Usage:
    python scripts/build_legal_graph.py
"""

import json
import logging
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BARE_CHUNKS_V2,
    CASE_CHUNKS_V2,
    CITATION_GRAPH_PATH,
    LEGAL_GRAPH_DB,
)
from core.retriever import load_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS acts (
    id            TEXT PRIMARY KEY,   -- safe_id: e.g. "indian_penal_code_1860"
    act_name      TEXT NOT NULL,
    alias         TEXT DEFAULT '',    -- short form: "IPC", "BNS", etc.
    year          TEXT DEFAULT '',
    section_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sections (
    id              TEXT PRIMARY KEY,   -- "{act_id}::{section_number}"
    act_id          TEXT NOT NULL,
    act_name        TEXT NOT NULL,
    section_number  TEXT NOT NULL,
    section_title   TEXT DEFAULT '',
    chapter         TEXT DEFAULT '',
    chunk_ids       TEXT NOT NULL,      -- JSON array  ["chunk1", "chunk2", ...]
    FOREIGN KEY (act_id) REFERENCES acts(id)
);

CREATE TABLE IF NOT EXISTS section_cross_refs (
    src_section_id  TEXT NOT NULL,   -- section containing the reference
    dst_section_id  TEXT NOT NULL,   -- section being referenced
    ref_text        TEXT DEFAULT '',  -- snippet of text that contained the reference
    PRIMARY KEY (src_section_id, dst_section_id)
);

CREATE TABLE IF NOT EXISTS case_section_links (
    section_id  TEXT NOT NULL,   -- section node id in this graph
    case_id     TEXT NOT NULL,   -- stable case_id from citation graph
    case_name   TEXT DEFAULT '',
    court       TEXT DEFAULT '',
    year        TEXT DEFAULT '',
    PRIMARY KEY (section_id, case_id)
);

-- Indexes for fast query-time lookups
CREATE INDEX IF NOT EXISTS idx_sections_act      ON sections(act_id);
CREATE INDEX IF NOT EXISTS idx_sections_number   ON sections(section_number);
CREATE INDEX IF NOT EXISTS idx_sections_act_num  ON sections(act_id, section_number);
CREATE INDEX IF NOT EXISTS idx_case_sec_section  ON case_section_links(section_id);
CREATE INDEX IF NOT EXISTS idx_cross_refs_src    ON section_cross_refs(src_section_id);
CREATE INDEX IF NOT EXISTS idx_cross_refs_dst    ON section_cross_refs(dst_section_id);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Same alias map as chunker.py (for consistent short names)
_ACT_ALIASES: dict[str, str] = {
    "bharatiya nyaya sanhita": "BNS",
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "bharatiya sakshya adhiniyam": "BSA",
    "indian penal code": "IPC",
    "code of criminal procedure": "CrPC",
    "indian evidence act": "IEA",
    "transfer of property act": "TP Act",
    "specific relief act": "SRA",
    "negotiable instruments act": "NI Act",
    "code of civil procedure": "CPC",
    "hindu marriage act": "HMA",
    "registration act": "Registration Act",
    "indian contract act": "Contract Act",
    "consumer protection act": "Consumer Protection Act",
    "limitation act": "Limitation Act",
    "arbitration and conciliation act": "Arbitration Act",
    "hindu succession act": "HSA",
    "protection of women from domestic violence act": "PWDVA",
    "motor vehicles act": "MV Act",
    "income tax act": "IT Act",
    "constitution of india": "Constitution",
    "land acquisition act": "LA Act",
    "right to fair compensation": "RFCTLARR Act",
}


def _act_alias(act_name: str) -> str:
    name_lower = (act_name or "").lower()
    for key, alias in _ACT_ALIASES.items():
        if key in name_lower:
            return alias
    return ""


def _safe_id(text: str) -> str:
    """Normalise to a compact safe identifier."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:80]


def _extract_year(act_name: str) -> str:
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", act_name or "")
    return m.group(1) if m else ""


def _make_act_id(act_name: str) -> str:
    return _safe_id(act_name)


def _make_section_id(act_id: str, section_number: str) -> str:
    return f"{act_id}::{section_number.strip()}"


# Pattern to find explicit section/article references in body text.
# Captures: "section 302", "Section 302A", "Article 21", "s. 498A", "sub-section (1) of section 304"
_CROSS_REF_PATTERN = re.compile(
    r"\b(?:section|sec\.|s\.|article|art\.)\s*(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\b",
    re.IGNORECASE,
)


def _extract_cross_refs(full_text: str) -> list[str]:
    """Return list of referenced section numbers found in full_text."""
    return list({m.group(1).strip() for m in _CROSS_REF_PATTERN.finditer(full_text or "")})


# ---------------------------------------------------------------------------
# Phase 1 — Build acts + sections tables from BARE_CHUNKS_V2
# ---------------------------------------------------------------------------

def _build_acts_and_sections(conn: sqlite3.Connection, bare_chunks: dict) -> dict[str, str]:
    """
    Walk bare-act chunks; upsert into acts + sections tables.
    Returns section_number_to_id: {(act_id, section_number) -> section_id}
    used later for cross-ref edge resolution.
    """
    # Aggregate: section_id -> {act_id, act_name, section_number, section_title,
    #                            chapter, chunk_ids: set}
    sections: dict[str, dict] = {}
    acts: dict[str, dict] = {}  # act_id -> {act_name, alias, year}

    for _idx, chunk in bare_chunks.items():
        if not isinstance(chunk, dict):
            continue
        if chunk.get("doc_type") != "bare_act":
            continue
        act_name = (chunk.get("act_name") or "").strip()
        sec_num  = (chunk.get("section_number") or "").strip()
        if not act_name or not sec_num:
            continue

        act_id  = _make_act_id(act_name)
        sec_id  = _make_section_id(act_id, sec_num)
        chunk_id = (chunk.get("chunk_id") or _idx)

        # Acts table
        if act_id not in acts:
            acts[act_id] = {
                "id":       act_id,
                "act_name": act_name,
                "alias":    _act_alias(act_name),
                "year":     _extract_year(act_name),
            }

        # Sections table — merge sub-section chunks into one section node
        if sec_id not in sections:
            sections[sec_id] = {
                "id":             sec_id,
                "act_id":         act_id,
                "act_name":       act_name,
                "section_number": sec_num,
                "section_title":  (chunk.get("section_title") or "").strip(),
                "chapter":        (chunk.get("chapter") or "").strip(),
                "chunk_ids":      set(),
                "full_text":      "",   # accumulate for cross-ref extraction
            }
        sections[sec_id]["chunk_ids"].add(str(chunk_id))
        # Append full text for cross-ref extraction (we only need enough to find refs)
        sections[sec_id]["full_text"] += " " + (chunk.get("full_text") or "")

    logger.info("Acts found: %d | Sections found: %d", len(acts), len(sections))

    # Write acts
    conn.executemany(
        "INSERT OR REPLACE INTO acts (id, act_name, alias, year, section_count) VALUES (?,?,?,?,?)",
        [
            (a["id"], a["act_name"], a["alias"], a["year"], 0)
            for a in acts.values()
        ],
    )

    # Write sections
    conn.executemany(
        """INSERT OR REPLACE INTO sections
           (id, act_id, act_name, section_number, section_title, chapter, chunk_ids)
           VALUES (?,?,?,?,?,?,?)""",
        [
            (
                s["id"],
                s["act_id"],
                s["act_name"],
                s["section_number"],
                s["section_title"],
                s["chapter"],
                json.dumps(sorted(s["chunk_ids"])),
            )
            for s in sections.values()
        ],
    )

    # Update section_count on acts table
    conn.execute(
        """UPDATE acts SET section_count = (
               SELECT COUNT(*) FROM sections WHERE sections.act_id = acts.id
           )"""
    )

    conn.commit()
    logger.info("Wrote %d acts and %d sections to graph", len(acts), len(sections))

    # Return lookup: (act_id, section_number) -> section_id
    return {(s["act_id"], s["section_number"]): s["id"] for s in sections.values()}


# ---------------------------------------------------------------------------
# Phase 2 — Section cross-reference edges
# ---------------------------------------------------------------------------

def _build_cross_ref_edges(
    conn: sqlite3.Connection,
    bare_chunks: dict,
    section_lookup: dict,
) -> None:
    """
    For each section, extract all referenced section numbers from its text,
    resolve them to section_ids in the same act (or across acts if ambiguous),
    and write section_cross_refs edges.
    """
    # Rebuild per-section full_text and act_id from bare_chunks
    # (same aggregation as above — could be cached, but kept simple for clarity)
    section_texts: dict[str, tuple[str, str]] = {}  # sec_id -> (act_id, full_text)
    for _idx, chunk in bare_chunks.items():
        if not isinstance(chunk, dict) or chunk.get("doc_type") != "bare_act":
            continue
        act_name = (chunk.get("act_name") or "").strip()
        sec_num  = (chunk.get("section_number") or "").strip()
        if not act_name or not sec_num:
            continue
        act_id = _make_act_id(act_name)
        sec_id = _make_section_id(act_id, sec_num)
        if sec_id not in section_texts:
            section_texts[sec_id] = (act_id, "")
        section_texts[sec_id] = (act_id, section_texts[sec_id][1] + " " + (chunk.get("full_text") or ""))

    edges: list[tuple] = []  # (src, dst, ref_text)

    for sec_id, (act_id, full_text) in section_texts.items():
        referenced_nums = _extract_cross_refs(full_text)
        for ref_num in referenced_nums:
            # Resolve to section_id in same act first
            dst_id = section_lookup.get((act_id, ref_num))
            if dst_id and dst_id != sec_id:
                snippet = f"section {ref_num}"
                edges.append((sec_id, dst_id, snippet))

    if edges:
        conn.executemany(
            """INSERT OR IGNORE INTO section_cross_refs
               (src_section_id, dst_section_id, ref_text) VALUES (?,?,?)""",
            edges,
        )
        conn.commit()
    logger.info("Wrote %d section cross-reference edges", len(edges))


# ---------------------------------------------------------------------------
# Phase 3 — Case → Section reverse links from citation graph
# ---------------------------------------------------------------------------

def _build_case_section_links(conn: sqlite3.Connection, section_lookup: dict) -> None:
    """
    Load the citation graph (case → interprets → section_id strings like "IPC 302").
    Map each section_id string to a section node in legal.db and write case_section_links.
    """
    if not os.path.isfile(CITATION_GRAPH_PATH):
        logger.warning("Citation graph not found at %s — skipping case-section links", CITATION_GRAPH_PATH)
        return

    with open(CITATION_GRAPH_PATH, encoding="utf-8") as f:
        graph = json.load(f)

    case_info  = graph.get("case_info") or {}
    interprets = graph.get("interprets") or []  # [{case_id, section_id}, ...]

    # Build lookup: section_id string from citation graph → section node id in legal.db
    # Citation graph section_ids look like: "IPC 302", "Section 302 IPC", "302", etc.
    # Strategy: extract the section number portion and try to match by number across acts.
    _sec_num_re = re.compile(r"(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)")

    # Build reverse map: section_number -> [section_ids in legal.db]
    # (used when we can't determine the act from the citation graph entry)
    number_to_section_ids: dict[str, list[str]] = {}
    for (act_id, sec_num), sec_id in section_lookup.items():
        number_to_section_ids.setdefault(sec_num, []).append(sec_id)

    # Also build alias → act_id lookup for smarter resolution
    alias_to_act_ids: dict[str, list[str]] = {}
    rows = conn.execute("SELECT id, alias, act_name FROM acts").fetchall()
    for act_id, alias, act_name in rows:
        if alias:
            alias_to_act_ids.setdefault(alias.upper(), []).append(act_id)
        # Also index by first meaningful word of act name
        words = [w for w in act_name.split() if len(w) > 3]
        if words:
            alias_to_act_ids.setdefault(words[0].upper(), []).append(act_id)

    links: list[tuple] = []
    unresolved = 0

    for edge in interprets:
        case_id    = (edge.get("case_id") or "").strip()
        section_str = (edge.get("section_id") or "").strip()
        if not case_id or not section_str:
            continue

        info = case_info.get(case_id, {})

        # Try to extract section number and optional act hint from the section string
        # e.g. "IPC 302" → act_hint="IPC", sec_num="302"
        #      "Section 302 of the Indian Penal Code" → sec_num="302"
        #      "302" → sec_num="302"
        sec_num_match = _sec_num_re.search(section_str)
        if not sec_num_match:
            unresolved += 1
            continue
        sec_num = sec_num_match.group(1)

        # Try to find act hint (word before or after the number)
        candidates: list[str] = []
        upper_str = section_str.upper()
        for alias, act_ids in alias_to_act_ids.items():
            if alias in upper_str:
                for act_id in act_ids:
                    sid = section_lookup.get((act_id, sec_num))
                    if sid:
                        candidates.append(sid)

        # Fall back to all sections with this number (may be multiple acts)
        if not candidates:
            candidates = number_to_section_ids.get(sec_num, [])

        for sec_node_id in candidates:
            links.append((
                sec_node_id,
                case_id,
                info.get("case_name", ""),
                info.get("court", ""),
                str(info.get("year", "")),
            ))

    if links:
        conn.executemany(
            """INSERT OR IGNORE INTO case_section_links
               (section_id, case_id, case_name, court, year) VALUES (?,?,?,?,?)""",
            links,
        )
        conn.commit()
    logger.info(
        "Wrote %d case-section links (%d interprets edges unresolved)",
        len(links), unresolved,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_legal_graph() -> int:
    # Validate inputs
    if not os.path.isfile(BARE_CHUNKS_V2):
        logger.error("Bare act chunks not found: %s — run rebuild_vector_store.py first", BARE_CHUNKS_V2)
        return 1

    logger.info("Loading bare act chunks from %s ...", BARE_CHUNKS_V2)
    bare_chunks = load_chunks(BARE_CHUNKS_V2)
    if not bare_chunks:
        logger.error("No bare act chunks loaded from %s", BARE_CHUNKS_V2)
        return 1
    logger.info("Loaded %d bare act chunks", len(bare_chunks))

    # Create / overwrite the database
    os.makedirs(os.path.dirname(LEGAL_GRAPH_DB), exist_ok=True)
    if os.path.exists(LEGAL_GRAPH_DB):
        os.remove(LEGAL_GRAPH_DB)
        logger.info("Removed existing legal.db — building fresh")

    conn = sqlite3.connect(LEGAL_GRAPH_DB)
    conn.executescript(_SCHEMA)

    try:
        logger.info("Phase 1 — Building acts + sections hierarchy ...")
        section_lookup = _build_acts_and_sections(conn, bare_chunks)

        logger.info("Phase 2 — Extracting section cross-reference edges ...")
        _build_cross_ref_edges(conn, bare_chunks, section_lookup)

        logger.info("Phase 3 — Building case-section reverse links ...")
        _build_case_section_links(conn, section_lookup)

        # Summary
        act_count     = conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0]
        section_count = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        xref_count    = conn.execute("SELECT COUNT(*) FROM section_cross_refs").fetchone()[0]
        link_count    = conn.execute("SELECT COUNT(*) FROM case_section_links").fetchone()[0]

        logger.info(
            "Legal graph built successfully → %s\n"
            "  Acts:                 %d\n"
            "  Sections:             %d\n"
            "  Cross-ref edges:      %d\n"
            "  Case-section links:   %d",
            LEGAL_GRAPH_DB,
            act_count, section_count, xref_count, link_count,
        )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(build_legal_graph())
