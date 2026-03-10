"""
Bare act folder deduplication by title + "As on" date.

Scans DATA_ROOT/BareActs: from the first 2 pages of each PDF, extracts
- act title (same logic as smart_chunker: ACT/CODE/... or "Name, YYYY")
- "As on the DD Month, YYYY" if present

Groups by normalized title. Among duplicates, keeps the file with the latest
"As on" date and removes the others. Used by rebuild_vector_store before rebuilding.
"""

import os
import re
import sys
from datetime import date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from Ingestion.smart_chunker import _detect_act_name_from_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# "As on the 1st January, 2025" or "[As on the 1th January, 2025]"
_AS_ON_PATTERN = re.compile(
    r"(?:\[?\s*)?As\s+on\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+),\s*(\d{4})\s*\]?",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Use first ~2 pages worth of text for title and As on
_TITLE_PAGE_CHARS = 3000


def _parse_as_on(text: str) -> Optional[date]:
    """Parse first 'As on the DD Month, YYYY' in text. Returns date or None."""
    m = _AS_ON_PATTERN.search(text)
    if not m:
        return None
    day_s, month_s, year_s = m.group(1), m.group(2), m.group(3)
    try:
        day = int(day_s)
        month = _MONTH_NAMES.get(month_s.lower())
        year = int(year_s)
        if month and 1 <= day <= 31 and 1900 <= year <= 2100:
            return date(year, month, day)
    except (ValueError, TypeError):
        pass
    return None


def _normalize_title_for_grouping(title: str) -> str:
    """Single line, lower, single spaces — for grouping duplicates."""
    if not title:
        return ""
    return " ".join(title.split()).strip().lower()


def scan_bare_acts_for_rename(bare_acts_dir: str) -> list[tuple[str, str]]:
    """
    Scan BareActs and return (filepath, text_from_first_2_pages) for each PDF/txt.
    Used by rename_bare_acts.py to derive target filename from ACT/CODE/... or first year.
    """
    results = []
    if not os.path.isdir(bare_acts_dir):
        return results
    for name in os.listdir(bare_acts_dir):
        if name.startswith("."):
            continue
        path = os.path.join(bare_acts_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not low.endswith(".txt"):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text_sample = (text or "")[:_TITLE_PAGE_CHARS]
            results.append((path, text_sample))
        except Exception as e:
            logger.warning("Skip %s: %s", name, e)
    return results


def scan_bare_acts_dir(bare_acts_dir: str) -> list[tuple[str, str, Optional[date]]]:
    """
    Scan BareActs directory. Return list of (filepath, normalized_title, as_on_date).
    Only PDF and .txt; skips non-files. Uses first 2 pages for PDFs.
    """
    results = []
    if not os.path.isdir(bare_acts_dir):
        return results
    for name in os.listdir(bare_acts_dir):
        if name.startswith("."):
            continue
        path = os.path.join(bare_acts_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not low.endswith(".txt"):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text_sample = (text or "")[:_TITLE_PAGE_CHARS]
            title = _detect_act_name_from_text(text_sample, name)
            title_norm = _normalize_title_for_grouping(title)
            as_on = _parse_as_on(text_sample)
            results.append((path, title_norm, as_on))
        except Exception as e:
            logger.warning("Skip %s: %s", name, e)
    return results


def decide_duplicates(
    scan_results: list[tuple[str, str, Optional[date]]],
) -> tuple[set[str], list[str]]:
    """
    Given list of (filepath, title_norm, as_on_date), return (keep_set, remove_list).
    Group by title_norm; within each group keep the one with latest as_on (None = oldest).
    """
    from collections import defaultdict
    by_title: dict[str, list[tuple[str, Optional[date]]]] = defaultdict(list)
    for path, title_norm, as_on in scan_results:
        if not title_norm:
            continue
        by_title[title_norm].append((path, as_on))

    keep: set[str] = set()
    remove: list[str] = []
    for title_norm, candidates in by_title.items():
        # Sort: latest date first, None last
        def sort_key(item: tuple[str, Optional[date]]) -> tuple[bool, date]:
            _, d = item
            if d is None:
                return (True, date.min)  # no date = treat as oldest
            return (False, d)

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        # Keep first (latest), remove the rest
        keep.add(sorted_candidates[0][0])
        for p, _ in sorted_candidates[1:]:
            remove.append(p)
    # Any file whose title_norm was empty: keep it (no dedup)
    for path, title_norm, _ in scan_results:
        if not title_norm and path not in keep:
            keep.add(path)
    return (keep, remove)


def run_dedup(bare_acts_dir: str) -> int:
    """
    Run duplicate check on BareActs folder: remove duplicate files (keep latest by "As on").
    Returns number of files removed.
    """
    logger.info("Bare act dedup: scanning %s", bare_acts_dir)
    scan_results = scan_bare_acts_dir(bare_acts_dir)
    if not scan_results:
        logger.info("Bare act dedup: no files to scan")
        return 0
    _keep_set, remove_list = decide_duplicates(scan_results)
    removed = 0
    for p in remove_list:
        try:
            os.remove(p)
            removed += 1
            logger.info("Removed duplicate: %s", os.path.basename(p))
        except OSError as e:
            logger.warning("Could not remove %s: %s", p, e)
    if removed:
        logger.info("Bare act dedup: removed %d duplicate(s)", removed)
    else:
        logger.info("Bare act dedup: no duplicates to remove")
    return removed


if __name__ == "__main__":
    from config import BARE_ACTS_DIR
    run_dedup(BARE_ACTS_DIR)
