"""
patch_gap_caselaws.py — Fix swallowed paragraphs in caselaws_v2_chunks.json

For case law documents that use numbered paragraphs ("1.", "2.", …), scans
the existing chunks store for gaps in paragraph numbering within each case.
When a gap > 1 is found (e.g. paragraph 7 → paragraph 11), the preceding
chunk is searched for the missing paragraph headings (8, 9, 10) and split
in place.

This mirrors the logic in patch_gap_sections.py but targets:
  - caselaws_v2_chunks.json  (not bareacts_v2_chunks.json)
  - paragraph_num field       (not section_number)
  - Paragraph-level patterns  (not section patterns)
  - Only cases that use numbered paragraphs (Path A in chunk_case_law)

After patching the JSON, rebuilds FAISS + BM25 from the corrected chunks so
that all vectors align with the updated text.

Usage:
    cd NM_API-1.0
    python scripts/patch_gap_caselaws.py [--dry-run] [--case CASE_NAME]

    --dry-run          Print what would be split without writing anything.
    --case CASE_NAME   Restrict scan to a single case (partial match, case-insensitive).
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
    CASE_CHUNKS_V2,
    CASE_INDEX_V2,
    CASE_BM25_INDEX,
    VECTOR_STORE,
)
from Ingestion.smart_chunker import (
    _extract_keywords,
    _safe_id,
    _classify_paragraph_type,
    _extract_sections_cited,
    extract_cited_cases,
    _split_para_to_limit,
    _make_case_chunk,
    _detect_case_metadata,
    _determine_binding_authority,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paragraph-number helpers
# ---------------------------------------------------------------------------

# Matches a standalone numbered paragraph heading at the start of a line:
#   "7. Some text…"  or  "7.\nSome text"  or  "  7.  "
# We require at least one whitespace or newline after the dot so we don't
# collide with legal citations like "s.7" or "Art.7".
_PARA_NUM_RE = re.compile(r"(?m)^[ \t]*(\d+)\.[ \t]+")


def _parse_para_num(s: str):
    """Parse a paragraph_num string.  Returns int or float('inf') on failure."""
    try:
        # Strip sub-chunk suffixes like "3_1", "3_2"
        base = str(s).strip().split("_")[0]
        return int(base)
    except (ValueError, AttributeError):
        return float("inf")


def _int_para(s: str) -> int:
    v = _parse_para_num(s)
    if v == float("inf"):
        return -1
    return v


def _numeric_para_gap(a: str, b: str) -> int:
    """Return gap between integer parts of two paragraph numbers."""
    ia = _int_para(a)
    ib = _int_para(b)
    if ia < 0 or ib < 0:
        return 0
    return ib - ia


# ---------------------------------------------------------------------------
# Case-identity key
# ---------------------------------------------------------------------------

def _case_key(chunk: dict) -> str:
    """Stable identity string for a case — case_name + citation."""
    name = (chunk.get("case_name") or "").strip()
    cit  = (chunk.get("citation")  or "").strip()
    return f"{name}||{cit}"


# ---------------------------------------------------------------------------
# Missing-paragraph finder (number-first, no format assumptions)
# ---------------------------------------------------------------------------

def _find_missing_paras_by_number(text: str, missing_nums: list[int]) -> list[dict]:
    """
    Find positions of specific paragraph numbers inside *text*.

    Uses a loose line-start anchor — no assumption about what follows the
    number and dot — for the same reason as _find_missing_by_number in
    patch_gap_sections.py: these paragraphs were swallowed exactly because
    the strict regex missed their formatting variant.

    Returns [{pos, para_num, para_title}, …] sorted by pos.
    """
    found = []
    seen: set = set()

    for num in sorted(missing_nums):
        # Match: optional whitespace, then the digit(s), a dot, then whitespace.
        pat = re.compile(rf"(?m)^[ \t]*({num})\.[ \t]+")
        for m in pat.finditer(text):
            pos = m.start()
            if pos in seen:
                continue
            seen.add(pos)
            # Grab the first line after the match as a short title hint
            after     = text[m.end():]
            first_ln  = after.split("\n")[0].strip()
            short_ttl = first_ln[:100] if first_ln else ""
            found.append({"pos": pos, "para_num": str(num), "hint": short_ttl})

    found.sort(key=lambda x: x["pos"])
    return found


# ---------------------------------------------------------------------------
# Split a chunk at paragraph boundaries
# ---------------------------------------------------------------------------

def _split_chunk_at_para_boundaries(
    chunk: dict,
    missing_nums: list[int],
) -> list[dict]:
    """
    Try to split *chunk*'s full_text at paragraph boundaries for *missing_nums*.

    Returns a list of replacement chunks (≥ 1 element; [chunk] if no split found).
    """
    text = chunk.get("full_text", "")
    if not text:
        return [chunk]

    missing_starts = _find_missing_paras_by_number(text, missing_nums)
    if not missing_starts:
        return [chunk]

    own_num = chunk.get("paragraph_num", "")

    # Anchor: position 0 belongs to the parent paragraph
    split_points = [{"pos": 0, "para_num": own_num}] + [
        {"pos": s["pos"], "para_num": s["para_num"]} for s in missing_starts
    ]
    split_points.sort(key=lambda x: x["pos"])

    # Deduplicate positions
    seen_pos: set = set()
    split_points = [
        sp for sp in split_points
        if not (sp["pos"] in seen_pos or seen_pos.add(sp["pos"]))
    ]

    if len(split_points) < 2:
        return [chunk]

    # Re-use metadata from the parent chunk
    metadata = {
        "case_name": chunk.get("case_name", ""),
        "citation":  chunk.get("citation", ""),
        "court":     chunk.get("court", ""),
        "bench":     chunk.get("bench", ""),
        "year":      chunk.get("year", ""),
    }
    binding   = chunk.get("binding_authority", "")
    source    = chunk.get("source_file", "")
    total_sps = len(split_points)

    results = []
    for i, sp in enumerate(split_points):
        seg_start = sp["pos"]
        seg_end   = split_points[i + 1]["pos"] if i + 1 < total_sps else len(text)
        seg_text  = text[seg_start:seg_end].strip()

        if len(seg_text) < 50:
            continue

        para_label = sp["para_num"]
        ptype = _classify_paragraph_type(
            seg_text,
            para_label,
            i,
            total_sps,
        )

        # _split_para_to_limit handles oversized paragraphs
        for label, chunk_text in _split_para_to_limit(seg_text, para_label):
            if len(chunk_text) < 50:
                continue
            sections_cited = _extract_sections_cited(chunk_text)
            cited_cases    = extract_cited_cases(chunk_text)
            results.append(_make_case_chunk(
                chunk_text, metadata, binding, label, source,
                paragraph_type=ptype,
                sections_cited=sections_cited,
                cited_cases=cited_cases,
            ))

    return results if len(results) > 1 else [chunk]


# ---------------------------------------------------------------------------
# Main scan + patch routine
# ---------------------------------------------------------------------------

def patch_chunks(dry_run: bool = False, case_filter: str = "") -> tuple[dict, dict]:
    """
    Load caselaws_v2_chunks.json, detect and fix paragraph-number gaps in
    cases that use numbered paragraphs, and return the patched store + a
    dict of unresolved gaps.
    """
    logger.info("Loading case law chunks from %s", CASE_CHUNKS_V2)
    with open(CASE_CHUNKS_V2, encoding="utf-8") as f:
        store: dict = json.load(f)
    logger.info("Loaded %d chunks", len(store))

    # Group chunk keys by case identity
    case_index: dict[str, list[str]] = defaultdict(list)
    for key, chunk in store.items():
        ck = _case_key(chunk)
        if ck and ck != "||":
            case_index[ck].append(key)

    total_splits    = 0
    total_new_chunks = 0
    replacements: dict[str, list[dict]] = {}   # old_key → [new chunk dicts]
    unresolved:   dict[str, list[str]]  = {}   # case_key → [missing para numbers]

    for case_key_str, keys in case_index.items():
        # Optional filter for debugging a single case
        if case_filter and case_filter.lower() not in case_key_str.lower():
            continue

        # Only process cases that actually use numbered paragraphs.
        # Heuristic: at least 50 % of chunks must have a purely numeric paragraph_num.
        numeric_keys = [
            k for k in keys
            if re.match(r"^\d+$", str(store[k].get("paragraph_num", "") or "").strip())
        ]
        if len(numeric_keys) < max(2, len(keys) * 0.5):
            continue  # Mixed or un-numbered case — skip

        # Sort by paragraph number
        sorted_keys = sorted(
            numeric_keys,
            key=lambda k: _int_para(store[k].get("paragraph_num", "0") or "0"),
        )

        # Build ordered list of (para_num, key) skipping sub-chunks (N_1, N_2, …)
        seen_paras: list[tuple[str, str]] = []
        for k in sorted_keys:
            pn = str(store[k].get("paragraph_num", "") or "").strip()
            if not pn or not pn.isdigit():
                continue
            if not seen_paras or seen_paras[-1][0] != pn:
                seen_paras.append((pn, k))
            # else: same paragraph number (multiple sub-chunks) — keep first key

        for i in range(len(seen_paras) - 1):
            para_a, key_a = seen_paras[i]
            para_b, _     = seen_paras[i + 1]

            gap = _numeric_para_gap(para_a, para_b)
            if gap <= 1:
                continue

            missing_ints = list(range(_int_para(para_a) + 1, _int_para(para_b)))
            if not missing_ints:
                continue

            case_display = case_key_str.split("||")[0]
            logger.info(
                "[%s] Para gap: %s → %s (missing %s)",
                case_display, para_a, para_b, missing_ints,
            )

            result = _split_chunk_at_para_boundaries(store[key_a], missing_ints)

            if len(result) <= 1:
                logger.info("  → Missing paragraphs not found in chunk text (may be preamble/cover page). Skipping.")
                unresolved.setdefault(case_key_str, []).extend(str(m) for m in missing_ints)
                continue

            logger.info(
                "  → Split para %s into %d chunks: %s",
                para_a,
                len(result),
                [c["paragraph_num"] for c in result],
            )

            if not dry_run:
                replacements[key_a] = result
            total_splits    += 1
            total_new_chunks += len(result) - 1

    # Apply replacements
    if not dry_run:
        next_key = max(int(k) for k in store.keys()) + 1
        for old_key, new_chunks in replacements.items():
            # Update original key with first split chunk, append the rest
            store[old_key] = {kk: vv for kk, vv in new_chunks[0].items() if kk != "search_text"}
            for extra in new_chunks[1:]:
                store[str(next_key)] = {kk: vv for kk, vv in extra.items() if kk != "search_text"}
                next_key += 1

    total_unresolved = sum(len(v) for v in unresolved.values())
    logger.info(
        "Paragraph gap scan complete. Splits: %d, Net new chunks: %d, "
        "Unresolved gaps: %d paragraphs across %d cases",
        total_splits, total_new_chunks, total_unresolved, len(unresolved),
    )
    return store, unresolved


# ---------------------------------------------------------------------------
# Index rebuild from patched chunks
# ---------------------------------------------------------------------------

def rebuild_indexes(store: dict) -> None:
    """Re-embed all case law chunks and rebuild FAISS + BM25."""
    from Ingestion.build_v2_index import _get_embedder, build_index

    logger.info("Loading embedder for rebuild...")
    embedder = _get_embedder()

    # Reconstruct search_text (stripped at store time)
    chunks_with_search = []
    for chunk in store.values():
        case_name = chunk.get("case_name", "")
        citation  = chunk.get("citation", "")
        court     = chunk.get("court", "")
        year      = chunk.get("year", "")
        para_num  = chunk.get("paragraph_num", "")
        full_text = chunk.get("full_text", "")

        header = case_name
        if citation:
            header += f" ({citation})"
        if court:
            header += f" - {court}"
        if year and year not in header:
            header += f", {year}"

        search_text = f"{header}\n\n{full_text}" if header else full_text
        chunks_with_search.append({**chunk, "search_text": search_text})

    logger.info("Rebuilding FAISS + BM25 for %d case law chunks...", len(chunks_with_search))
    build_index(
        chunks            = chunks_with_search,
        faiss_path        = CASE_INDEX_V2,
        chunks_path       = CASE_CHUNKS_V2,
        bm25_path         = CASE_BM25_INDEX,
        embedder          = embedder,
        batch_size        = int(os.getenv("INDEX_EMBED_BATCH_GPU", "192")),
        resume_embeddings = True,
        append            = False,   # full rebuild — vectors must align with new chunk order
    )
    logger.info("Case law index rebuild complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Patch swallowed paragraphs in numbered case law chunks"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect gaps and report without writing any files",
    )
    parser.add_argument(
        "--case", default="",
        help="Restrict scan to cases whose name/citation contains this string (case-insensitive)",
    )
    args = parser.parse_args()

    patched_store, unresolved = patch_chunks(dry_run=args.dry_run, case_filter=args.case)

    # Write unresolved-gaps report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "legal_database", "vector_store", "gap_patch_caselaws_unresolved.json",
    )
    lines = []
    total_unresolved_paras = 0
    for ck in sorted(unresolved):
        paras_sorted = sorted(set(unresolved[ck]), key=lambda s: int(s) if s.isdigit() else 9999)
        case_display = ck.split("||")[0]
        lines.append(f"{case_display} - {', '.join(paras_sorted)}")
        total_unresolved_paras += len(paras_sorted)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    logger.info(
        "Unresolved gaps report saved — %d cases, %d paragraphs: %s",
        len(lines), total_unresolved_paras, report_path,
    )

    if args.dry_run:
        logger.info("Dry-run complete. No chunk files modified.")
        return

    # Save patched chunks JSON
    logger.info("Writing patched case law chunks JSON (%d chunks)...", len(patched_store))
    with open(CASE_CHUNKS_V2, "w", encoding="utf-8") as f:
        json.dump(patched_store, f, ensure_ascii=False, separators=(",", ":"))
    logger.info("Saved: %s", CASE_CHUNKS_V2)

    # Rebuild indexes
    rebuild_indexes(patched_store)
    logger.info("All done.")


if __name__ == "__main__":
    main()
