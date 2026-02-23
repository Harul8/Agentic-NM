"""
Rename bare act files in BareActs folder.

Logic (first 2 pages of each file):
  1. Look for the word ACT (or CODE/ORDINANCE/REGULATION) followed by a year
     -> rename as "Name ACT YYYY" (name = text before the word ACT/CODE/...).
  2. If that fails, look for first appearance of year (19xx/20xx) in first 2 pages
     -> name = text before that year -> rename as "Name YYYY".

Collisions get _2, _3 etc. before the extension.

Run from project root:
    python scripts/rename_bare_acts.py           # dry run
    python scripts/rename_bare_acts.py --apply  # perform renames
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import BARE_ACTS_DIR

# Import from same scripts dir (works when run from project root)
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from bare_act_dedup import scan_bare_acts_for_rename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Chars unsafe in filenames (Windows / common FS)
_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Rule 1: First appearance of the word "ACT" -> "Name ACT YYYY". Name can contain (), -, comma, etc.
_ACT_WORD_YEAR = re.compile(
    r"\b(THE\s+)?([A-Za-z0-9\s,\'\-()&]+?)\s+ACT\s*,?\s*(19|20)(\d{2})\b",
    re.IGNORECASE,
)
# Rule 2: first year (19xx or 20xx) in text
_FIRST_YEAR = re.compile(r"(19|20)(\d{2})\b")


def _strip_leading_number(name: str) -> str:
    """Act names generally don't start with a number; strip leading digits and optional dot/space."""
    if not name:
        return name
    s = name.strip()
    # Remove leading digits and optional following dot, space, or parenthesis
    s = re.sub(r"^\d+[\.\)\s]*", "", s)
    return s.strip()


def _extract_name_and_year_from_text(text: str, filename: str) -> tuple[str, str]:
    """
    Apply rename logic on first-2-pages text.
    Returns (base_name_for_filename, "") where base_name is "Name ACT YYYY" or "Name YYYY".
    """
    if not text or not text.strip():
        fallback = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ")
        return (_safe_filename(fallback), "")
    sample = " ".join(text.split()).strip()[:3000]

    # Rule 1: first appearance of the word ACT followed by year -> "Name ACT YYYY"
    m = _ACT_WORD_YEAR.search(sample)
    if m:
        pre = ((m.group(1) or "") + m.group(2)).strip()
        year = m.group(3) + m.group(4)
        name_part = " ".join(pre.split()).strip()
        name_part = _strip_leading_number(name_part)  # act name generally doesn't start with a number
        if len(name_part) > 3:
            base = f"{name_part} ACT {year}"
            return (_safe_filename(base), year)

    # Rule 2: first appearance of year; name = text before it (title usually immediately before year)
    m = _FIRST_YEAR.search(sample)
    if m:
        year = m.group(1) + m.group(2)
        before = sample[: m.start()].strip()
        name_part = " ".join(before.split()).strip()
        # Use last ~150 chars before year (act title is usually at end of intro)
        if len(name_part) > 150:
            name_part = name_part[-150:].strip()
        name_part = _strip_leading_number(name_part)  # act name generally doesn't start with a number
        if name_part:
            base = f"{name_part} {year}"
            return (_safe_filename(base), year)

    fallback = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ")
    return (_safe_filename(fallback), "")


def _safe_filename(name: str) -> str:
    """Replace unsafe filesystem chars with space; collapse spaces."""
    s = _UNSAFE_FS.sub(" ", name)
    return " ".join(s.split()).strip() or "Untitled"


def _target_filename(path: str, text_sample: str) -> str:
    """Derive target basename: 'Name ACT YYYY.ext' or 'Name YYYY.ext'."""
    base_name, _ = _extract_name_and_year_from_text(text_sample, path)
    ext = os.path.splitext(path)[1].lower() or ".pdf"
    return base_name + ext


def main():
    parser = argparse.ArgumentParser(
        description="Rename bare act files to 'Name ACT YYYY' or 'Name YYYY' from first 2 pages"
    )
    parser.add_argument("--apply", action="store_true", help="Perform renames (default is dry run)")
    args = parser.parse_args()
    apply_renames = args.apply

    if not os.path.isdir(BARE_ACTS_DIR):
        logger.error("Bare Acts directory not found: %s", BARE_ACTS_DIR)
        sys.exit(1)

    logger.info("Scanning %s (first 2 pages per file)...", BARE_ACTS_DIR)
    scan = scan_bare_acts_for_rename(BARE_ACTS_DIR)
    if not scan:
        logger.info("No PDF/txt files to rename.")
        return

    # Build (path, target_basename); resolve collisions
    dir_path = BARE_ACTS_DIR
    renames: list[tuple[str, str]] = []
    used: dict[str, int] = {}
    for path, text_sample in scan:
        target_base = _target_filename(path, text_sample)
        current_base = os.path.basename(path)
        if current_base == target_base:
            continue
        # Collision: same target from different source
        key = target_base.lower()
        if key in used:
            used[key] += 1
            base, ext = os.path.splitext(target_base)
            target_base = f"{base}_{used[key]}{ext}"
        else:
            used[key] = 1
        renames.append((path, target_base))

    if not renames:
        logger.info("All files already have target names. Nothing to rename.")
        return

    if not apply_renames:
        logger.info("DRY RUN: would rename %d file(s). Run with --apply to rename.", len(renames))
        for path, target_base in renames:
            logger.info("  %s  ->  %s", os.path.basename(path), target_base)
        return

    ok = 0
    err = 0
    for path, target_base in renames:
        target_path = os.path.join(dir_path, target_base)
        try:
            if os.path.abspath(path) == os.path.abspath(target_path):
                continue
            if os.path.isfile(target_path) and target_path != path:
                logger.warning("Skip (target exists): %s", target_base)
                err += 1
                continue
            os.rename(path, target_path)
            ok += 1
            logger.info("Renamed: %s -> %s", os.path.basename(path), target_base)
        except OSError as e:
            logger.warning("Failed %s: %s", path, e)
            err += 1
    logger.info("Done: %d renamed, %d failed/skipped.", ok, err)


if __name__ == "__main__":
    main()
