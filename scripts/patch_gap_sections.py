"""
patch_gap_sections.py — Fix swallowed sections in bareacts_v2_chunks.json

Scans the existing chunks store for gaps in section numbering within each act.
When a gap > 1 is found (e.g. section 82 → section 86), the preceding chunk is
searched for missing section headings (83, 84, 85) and split in place.

The same logic is applied to subsection numbers within a section cluster.

After patching the JSON, rebuilds FAISS + BM25 from the corrected chunks so
that all vectors align with the updated text.

Usage:
    cd NM_API-1.0
    python scripts/patch_gap_sections.py [--dry-run]

    --dry-run   Print what would be split without writing anything.
"""

from __future__ import annotations

import re
import os
import sys
import json
import logging
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BARE_CHUNKS_V2,
    BARE_INDEX_V2,
    BARE_BM25_INDEX,
    VECTOR_STORE,
)
from core.chunker import (
    _SECTION_PATTERNS,
    _strip_bare_act_editorial_noise,
    _extract_keywords,
    _act_alias,
    _safe_id,
    _maybe_split_chunk,
    _filter_min_gap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section-number helpers
# ---------------------------------------------------------------------------

def _parse_sec_num(s: str):
    """
    Parse a section number string into a sortable tuple (int_part, alpha_suffix).

    Examples:
        "82"  → (82,  "")
        "82A" → (82,  "A")
        "82B" → (82,  "B")
        "3"   → (3,   "")
    Returns (inf, "") for unparseable strings so they sort last.
    """
    m = re.match(r"^(\d+)([A-Za-z]{0,3})$", str(s).strip())
    if not m:
        return (float("inf"), "")
    return (int(m.group(1)), m.group(2).upper())


def _int_part(s: str) -> int:
    """Return the integer portion of a section number, or -1 on failure."""
    tup = _parse_sec_num(s)
    if tup[0] == float("inf"):
        return -1
    return tup[0]


def _numeric_gap(a: str, b: str) -> int:
    """
    Return the gap between the integer parts of two section numbers.
    E.g. gap("82", "86") = 4.  Returns 0 when either is unparseable.
    """
    ia = _int_part(a)
    ib = _int_part(b)
    if ia < 0 or ib < 0:
        return 0
    return ib - ia


# ---------------------------------------------------------------------------
# Section-start finder (reuses smart_chunker patterns)
# ---------------------------------------------------------------------------

# Extra pattern for the "number on its own line, title on next line" format:
#   "83.\nMarriage ceremony fraudulently gone through without lawful marriage."
# Common in Indian bare acts that don't use the em-dash anchor.
_SPLIT_LINE_SECTION_PATTERN = re.compile(
    r"^[ \t]*(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\.\s*\n[ \t]*([A-Z][^\n\u2014\u2013]{1,150})\.",
    re.MULTILINE,
)


def _find_section_starts(text: str) -> list[dict]:
    """
    Find all section-heading positions in *text* using:
      - Pattern 1A: em-dash anchor "NNN. Title.—"
      - Pattern 1B: "Section NNN." keyword
      - Pattern SL: split-line "NNN.\\nTitle." (number on own line)
      - Pattern 2:  plain "NNN. Title" fallback

    Returns a list of dicts: {pos, section_number, section_title}.
    """
    def _title_from(m) -> str:
        raw = (m.group(2) or "").strip()
        first = raw.split("\n")[0].strip()
        first = re.sub(r"[\s\u2014\u2013.]+$", "", first)
        first = re.sub(r"^\d+[A-Za-z]?\.\s+", "", first)
        return first[:120]

    # Pattern 1A: em-dash anchor (highest confidence)
    p1a = []
    for m in _SECTION_PATTERNS[0].finditer(text):
        p1a.append({
            "pos": m.start(),
            "section_number": m.group(1).strip(),
            "section_title": _title_from(m),
        })

    if p1a:
        return p1a

    # Pattern 1B: "Section NNN."
    p1b = []
    for m in _SECTION_PATTERNS[1].finditer(text):
        p1b.append({
            "pos": m.start(),
            "section_number": m.group(1).strip(),
            "section_title": _title_from(m),
        })

    if p1b:
        return p1b

    # Pattern SL: split-line format — "83.\nMarriage ceremony..."
    # Checked before Pattern 2 because it's higher confidence (requires title on next line).
    p_sl = []
    for m in _SPLIT_LINE_SECTION_PATTERN.finditer(text):
        p_sl.append({
            "pos": m.start(),
            "section_number": m.group(1).strip(),
            "section_title": m.group(2).strip()[:120],
        })
    if p_sl:
        return _filter_min_gap(p_sl, min_gap=100)

    # Pattern 2 fallback: plain "NNN. Title" with gap filter
    p2 = []
    for m in _SECTION_PATTERNS[2].finditer(text):
        p2.append({
            "pos": m.start(),
            "section_number": m.group(1).strip(),
            "section_title": _title_from(m),
        })
    return _filter_min_gap(p2, min_gap=300)


# ---------------------------------------------------------------------------
# Subsection-gap helpers
# ---------------------------------------------------------------------------

_SUBSEC_RE = re.compile(r"(?m)^[ \t]*\((\d+)\)[ \t]+")


def _find_subsec_starts(text: str) -> list[dict]:
    """Find (N) sub-section markers at line starts."""
    results = []
    for m in _SUBSEC_RE.finditer(text):
        results.append({"pos": m.start(), "sub_num": int(m.group(1))})
    return results


# ---------------------------------------------------------------------------
# Core gap-split logic
# ---------------------------------------------------------------------------

def _find_missing_by_number(text: str, missing_nums: list[int]) -> list[dict]:
    """
    Find positions of specific section numbers inside *text* using a loose,
    number-first search — no format assumptions.

    This is the correct approach for gap rescue: the sections were swallowed
    precisely because they didn't match the strict regex patterns used during
    initial chunking. We must not reuse those patterns here.

    Strategy: look for each missing integer at the start of a line, optionally
    followed by an alpha suffix (e.g. 83A), then any separator (., space,
    em-dash, en-dash, hyphen, colon).  The `^` MULTILINE anchor ensures we
    only match at real line starts, not inside body text cross-references
    like "…as defined in section 83 of this Act…".

    Returns a list of {pos, section_number, section_title} dicts sorted by pos.
    """
    found = []
    seen_positions: set = set()
    for num in sorted(missing_nums):
        # Match: line-start whitespace, then the number (+ optional alpha suffix),
        # then any of: period, em-dash, en-dash, hyphen, colon, or a space.
        pat = re.compile(
            rf"(?m)^[ \t]*({num}[A-Za-z]{{0,3}})[ \t]*[.\u2014\u2013\-:\s]",
        )
        for m in pat.finditer(text):
            pos = m.start()
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            # Try to extract a title: look at the remainder of the line or next line
            after = text[m.end():].lstrip(" \t")
            first_line = after.split("\n")[0].strip()
            # If first_line looks like a title (starts with capital, not a number), use it
            title = ""
            if first_line and first_line[0].isupper() and not first_line[0].isdigit():
                title = re.sub(r"[\u2014\u2013.]+$", "", first_line).strip()[:120]
            found.append({
                "pos": pos,
                "section_number": m.group(1).strip(),
                "section_title": title,
            })
    found.sort(key=lambda x: x["pos"])
    return found


def _split_chunk_at_section_boundaries(
    chunk: dict,
    missing_nums: list[int],
) -> list[dict]:
    """
    Try to split *chunk*'s full_text at section boundaries for any of the
    *missing_nums* that appear inside the text.

    Uses a loose number-first search (_find_missing_by_number) rather than
    the strict regex patterns — because those patterns are exactly what failed
    to detect these sections during initial chunking (that is why they were
    swallowed in the first place).

    Returns a list of replacement chunks (always at least [chunk] so callers
    can unconditionally replace the original).
    """
    text = chunk.get("full_text", "")
    if not text:
        return [chunk]

    # Search for the missing section numbers as raw line-start occurrences
    missing_starts = _find_missing_by_number(text, missing_nums)
    if not missing_starts:
        return [chunk]

    # The chunk itself starts at position 0 — prepend that as the anchor
    # so the text before the first missing section stays with the parent chunk.
    own_num    = chunk.get("section_number", "")
    own_title  = chunk.get("section_title", "")
    relevant = [{"pos": 0, "section_number": own_num, "section_title": own_title}] + missing_starts

    # Sort by position and deduplicate
    relevant.sort(key=lambda x: x["pos"])
    seen_pos: set = set()
    relevant = [s for s in relevant if not (s["pos"] in seen_pos or seen_pos.add(s["pos"]))]

    if len(relevant) < 2:
        # Could not find any of the missing sections as headings
        return [chunk]

    act_name   = chunk["act_name"]
    act_alias_ = _act_alias(act_name)
    alias_part = f" ({act_alias_})" if act_alias_ else ""
    chapter    = chunk.get("chapter", "")
    source     = chunk.get("source_file", "")

    results = []
    for i, sec in enumerate(relevant):
        seg_start = sec["pos"]
        seg_end   = relevant[i + 1]["pos"] if i + 1 < len(relevant) else len(text)
        seg_text  = text[seg_start:seg_end].strip()
        seg_text  = _strip_bare_act_editorial_noise(seg_text)

        if len(seg_text) < 30:
            continue

        sec_num   = sec["section_number"]
        sec_title = sec["section_title"]
        search_text = (
            f"{act_name}{alias_part} Section {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{seg_text}"
        )
        base_chunk = {
            "chunk_id":       f"{_safe_id(act_name)}_section_{sec_num}",
            "act_name":       act_name,
            "section_number": sec_num,
            "section_title":  sec_title,
            "sub_section":    "",
            "chapter":        chapter,
            "full_text":      seg_text,
            "search_text":    search_text,
            "keywords":       _extract_keywords(seg_text),
            "source_file":    source,
            "doc_type":       "bare_act",
        }
        # Apply existing sub-section splitter in case a recovered section is large
        results.extend(_maybe_split_chunk(base_chunk))

    return results if results else [chunk]


def _split_chunk_at_subsec_boundaries(
    chunk: dict,
    missing_sub_nums: list[int],
) -> list[dict]:
    """
    Try to split a chunk (which is itself a subsection) by looking for
    (N) markers for *missing_sub_nums* inside its full_text.

    Uses a loose number-first search: looks for (N) at line starts regardless
    of what follows — same reasoning as _find_missing_by_number: these
    subsections were swallowed because the stricter pattern missed them.
    """
    text = chunk.get("full_text", "")
    if not text:
        return [chunk]

    relevant = []
    seen_pos: set = set()
    for num in sorted(missing_sub_nums):
        # Loose match: line-start, then (N) optionally followed by anything
        pat = re.compile(rf"(?m)^[ \t]*\({num}\)[ \t]*")
        for m in pat.finditer(text):
            if m.start() not in seen_pos:
                seen_pos.add(m.start())
                relevant.append({"pos": m.start(), "sub_num": num})

    if not relevant:
        return [chunk]

    # Add position 0 as the start of the existing subsection content
    split_points = sorted([{"pos": 0, "sub_num": _int_part(chunk.get("sub_section", "0") or "0")}]
                          + relevant, key=lambda x: x["pos"])

    act_name   = chunk["act_name"]
    act_alias_ = _act_alias(act_name)
    alias_part = f" ({act_alias_})" if act_alias_ else ""
    sec_num    = chunk.get("section_number", "")
    sec_title  = chunk.get("section_title", "")
    chapter    = chunk.get("chapter", "")
    source     = chunk.get("source_file", "")

    results = []
    for i, sp in enumerate(split_points):
        seg_start = sp["pos"]
        seg_end   = split_points[i + 1]["pos"] if i + 1 < len(split_points) else len(text)
        seg_text  = text[seg_start:seg_end].strip()
        if len(seg_text) < 30:
            continue
        sub_label = f"({sp['sub_num']})"
        search_text = (
            f"{act_name}{alias_part} Section {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + f" {sub_label}"
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{seg_text}"
        )
        sub_safe = re.sub(r"[^a-z0-9]+", "_", sub_label.lower()).strip("_")
        results.append({
            "chunk_id":       f"{_safe_id(act_name)}_section_{sec_num}_{sub_safe}",
            "act_name":       act_name,
            "section_number": sec_num,
            "section_title":  sec_title,
            "sub_section":    sub_label,
            "chapter":        chapter,
            "full_text":      seg_text,
            "search_text":    search_text,
            "keywords":       _extract_keywords(seg_text),
            "source_file":    source,
            "doc_type":       "bare_act",
        })

    return results if len(results) > 1 else [chunk]


# ---------------------------------------------------------------------------
# Main scan + patch routine
# ---------------------------------------------------------------------------

def patch_chunks(dry_run: bool = False) -> dict:
    """
    Load bareacts_v2_chunks.json, detect and fix section-number gaps,
    return the patched chunk store dict.
    """
    logger.info("Loading chunks from %s", BARE_CHUNKS_V2)
    with open(BARE_CHUNKS_V2, encoding="utf-8") as f:
        store: dict = json.load(f)

    logger.info("Loaded %d chunks", len(store))

    # Group chunk keys by act_name, preserving original keys
    act_index: dict[str, list[str]] = defaultdict(list)
    for key, chunk in store.items():
        act_name = chunk.get("act_name", "") or ""
        if act_name:
            act_index[act_name].append(key)

    total_splits = 0
    total_new_chunks = 0
    replacements: dict[str, list[dict]] = {}  # old_key → [new chunk dicts]
    unresolved: dict[str, list[str]] = {}    # act_name → [missing section numbers]

    # ── Section-level gap scan ───────────────────────────────────────────────
    for act_name, keys in act_index.items():
        # Sort keys by section number, then sub_section
        def _sort_key(k):
            c = store[k]
            return (_parse_sec_num(c.get("section_number", "")),
                    c.get("sub_section", "") or "")

        sorted_keys = sorted(keys, key=_sort_key)

        # Build a list of (section_number, [all_keys_for_this_section])
        # so we can search ALL sub-chunks when a gap is found — not just the first.
        seen_sections: list[tuple[str, list[str]]] = []  # (sec_num, [key, ...])

        for k in sorted_keys:
            sec_num = store[k].get("section_number", "") or ""
            if not sec_num:
                continue
            if not seen_sections or seen_sections[-1][0] != sec_num:
                seen_sections.append((sec_num, [k]))
            else:
                seen_sections[-1][1].append(k)

        for i in range(len(seen_sections) - 1):
            sec_a, keys_a = seen_sections[i]
            sec_b, _      = seen_sections[i + 1]

            gap = _numeric_gap(sec_a, sec_b)
            if gap <= 1:
                continue

            # Gap detected: sections between sec_a and sec_b are missing
            missing_int = list(range(_int_part(sec_a) + 1, _int_part(sec_b)))
            if not missing_int:
                continue

            logger.info(
                "[%s] Gap detected: section %s → %s (missing %s)",
                act_name, sec_a, sec_b, missing_int,
            )

            # The LAST sub-chunk of section sec_a is most likely to contain the
            # swallowed sections (they appear at the tail of that chunk's text).
            # Try each sub-chunk from last to first until we find a split.
            split_result = None
            winning_key  = None
            for candidate_key in reversed(keys_a):
                parent_chunk = store[candidate_key]
                result = _split_chunk_at_section_boundaries(parent_chunk, missing_int)
                if len(result) > 1:
                    split_result = result
                    winning_key  = candidate_key
                    break

            if split_result is None:
                logger.info(
                    "  → Could not find missing sections in any sub-chunk (may be repealed). Skipping."
                )
                # Record missing section numbers grouped by act
                unresolved.setdefault(act_name, []).extend(str(mn) for mn in missing_int)
                continue

            logger.info(
                "  → Split chunk (sub='%s') into %d sections: %s",
                store[winning_key].get("sub_section", ""),
                len(split_result),
                [c["section_number"] for c in split_result],
            )

            if not dry_run:
                replacements[winning_key] = split_result
            total_splits += 1
            total_new_chunks += len(split_result) - 1  # net new chunks

    # ── Apply section-level replacements ─────────────────────────────────────
    if not dry_run:
        next_key = max(int(k) for k in store.keys()) + 1
        for old_key, new_chunks in replacements.items():
            # Update the original key with the first split chunk (section N stays at same key)
            first = new_chunks[0]
            store[old_key] = {k: v for k, v in first.items() if k != "search_text"}
            # Append remaining split chunks as new keys
            for extra in new_chunks[1:]:
                store[str(next_key)] = {k: v for k, v in extra.items() if k != "search_text"}
                next_key += 1

    # ── Subsection-level gap scan ────────────────────────────────────────────
    # Reload act_index after section fixes
    if not dry_run:
        act_index_2: dict[str, list[str]] = defaultdict(list)
        for key, chunk in store.items():
            act_name = chunk.get("act_name", "") or ""
            sub = chunk.get("sub_section", "") or ""
            if act_name and re.match(r"^\(\d+\)$", sub):
                act_index_2[(act_name, chunk.get("section_number", ""))].append(key)

        sub_splits = 0
        sub_replacements: dict[str, list[dict]] = {}

        for (act_name, sec_num), keys in act_index_2.items():
            def _sub_sort(k):
                sub = store[k].get("sub_section", "") or ""
                m = re.match(r"^\((\d+)\)$", sub)
                return int(m.group(1)) if m else 9999

            sorted_sub_keys = sorted(keys, key=_sub_sort)
            sub_nums = []
            for k in sorted_sub_keys:
                sub = store[k].get("sub_section", "") or ""
                m = re.match(r"^\((\d+)\)$", sub)
                if m:
                    sub_nums.append((int(m.group(1)), k))

            for j in range(len(sub_nums) - 1):
                num_a, key_a = sub_nums[j]
                num_b, _     = sub_nums[j + 1]
                if num_b - num_a <= 1:
                    continue

                missing_sub = list(range(num_a + 1, num_b))
                logger.info(
                    "[%s §%s] Sub-section gap: (%d) → (%d), missing %s",
                    act_name, sec_num, num_a, num_b, missing_sub,
                )
                split_result = _split_chunk_at_subsec_boundaries(store[key_a], missing_sub)
                if len(split_result) > 1:
                    sub_replacements[key_a] = split_result
                    sub_splits += 1

        # Apply subsection replacements
        next_key2 = max(int(k) for k in store.keys()) + 1
        for old_key, new_chunks in sub_replacements.items():
            store[old_key] = {k: v for k, v in new_chunks[0].items() if k != "search_text"}
            for extra in new_chunks[1:]:
                store[str(next_key2)] = {k: v for k, v in extra.items() if k != "search_text"}
                next_key2 += 1
        logger.info("Subsection gaps fixed: %d splits, %d new sub-chunks", sub_splits,
                    sum(len(v) - 1 for v in sub_replacements.values()))

    total_unresolved = sum(len(v) for v in unresolved.values())
    logger.info(
        "Section gap scan complete. Splits: %d, Net new chunks: %d, Unresolved gaps: %d sections across %d acts",
        total_splits, total_new_chunks, total_unresolved, len(unresolved),
    )
    return store, unresolved


# ---------------------------------------------------------------------------
# Index rebuild from patched chunks
# ---------------------------------------------------------------------------

def rebuild_indexes(store: dict) -> None:
    """Re-embed all chunks from the patched store and rebuild FAISS + BM25."""
    from core.indexer import _get_embedder, build_index
    import numpy as np

    logger.info("Loading embedder for rebuild...")
    embedder = _get_embedder()

    # Reconstruct search_text for each chunk (stripped at store time, needed for embedding)
    chunks_with_search = []
    for chunk in store.values():
        act_name   = chunk.get("act_name", "")
        act_alias_ = _act_alias(act_name)
        alias_part = f" ({act_alias_})" if act_alias_ else ""
        sec_num    = chunk.get("section_number", "")
        sec_title  = chunk.get("section_title", "")
        chapter    = chunk.get("chapter", "")
        sub        = chunk.get("sub_section", "")
        full_text  = chunk.get("full_text", "")

        search_text = (
            f"{act_name}{alias_part} Section {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + (f" {sub}" if sub else "")
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{full_text}"
        )
        chunks_with_search.append({**chunk, "search_text": search_text})

    logger.info("Rebuilding FAISS + BM25 for %d chunks...", len(chunks_with_search))
    build_index(
        chunks        = chunks_with_search,
        faiss_path    = BARE_INDEX_V2,
        chunks_path   = BARE_CHUNKS_V2,
        bm25_path     = BARE_BM25_INDEX,
        embedder      = embedder,
        batch_size    = int(os.getenv("INDEX_EMBED_BATCH_GPU", "192")),
        resume_embeddings = True,
        append        = False,   # full rebuild — vectors must align with new chunk order
    )
    logger.info("Index rebuild complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Patch swallowed sections in bare-act chunks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect gaps and report without modifying any files")
    args = parser.parse_args()

    patched_store, unresolved = patch_chunks(dry_run=args.dry_run)

    # Write unresolved-gaps report — one entry per act, format: "Act Name - x, y, z"
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "legal_database", "vector_store", "gap_patch_unresolved.json",
    )
    lines = []
    total_unresolved_secs = 0
    for act in sorted(unresolved):
        secs_sorted = sorted(
            set(unresolved[act]),
            key=lambda s: (int(s) if s.isdigit() else float("inf"), s),
        )
        lines.append(f"{act} - {', '.join(secs_sorted)}")
        total_unresolved_secs += len(secs_sorted)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    logger.info(
        "Unresolved gaps report saved — %d acts, %d sections: %s",
        len(lines), total_unresolved_secs, report_path,
    )

    if args.dry_run:
        logger.info("Dry-run complete. No chunk files modified.")
        return

    # Save patched chunks JSON
    logger.info("Writing patched chunks JSON (%d chunks)...", len(patched_store))
    with open(BARE_CHUNKS_V2, "w", encoding="utf-8") as f:
        json.dump(patched_store, f, ensure_ascii=False, separators=(",", ":"))
    logger.info("Saved: %s", BARE_CHUNKS_V2)

    # Rebuild indexes from patched chunks
    rebuild_indexes(patched_store)
    logger.info("All done.")


if __name__ == "__main__":
    main()
