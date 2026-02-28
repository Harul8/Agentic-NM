"""
Rename bare act files to clean, human-readable titles.

Extraction rules (applied in order, first 2 pages of each file):

  Rule 1: "Name ACT/CODE/ORDINANCE/REGULATION YYYY"
          Classic English act format — extracts text before the word ACT.

  Rule 2: "Name SANHITA/ADHINIYAM/NIYAMAWALI YYYY"
          New Indian-language act names (BNS, BNSS, BSA, etc.) —
          extracts text before the keyword.

  Rule 3: First appearance of year (19xx/20xx) in first 2 pages —
          uses last ~150 chars before that year.

  Rule 4: LLM fallback (when --llm flag is passed) — sends first 400
          chars of the document to an LLM and asks it to name the act.
          Used when rules 1-3 produce a result that looks wrong (too
          short, starts with a number, contains junk text, etc.).

  Rule 5: Original filename stripped of underscores/dashes —
          absolute last resort (no text extracted at all).

Usage:
  # Dry run — show what would be renamed (NO changes made):
  python scripts/rename_bare_acts.py

  # Dry run against a specific folder:
  python scripts/rename_bare_acts.py --dir "/path/to/Google Drive/BareActs"

  # Apply renames:
  python scripts/rename_bare_acts.py --apply
  python scripts/rename_bare_acts.py --dir "/path/to/BareActs" --apply

  # Apply with LLM fallback for unclear titles:
  python scripts/rename_bare_acts.py --dir "/path/to/BareActs" --apply --llm
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import BARE_ACTS_DIR

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from bare_act_dedup import scan_bare_acts_for_rename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Filesystem safety ────────────────────────────────────────────────────────
_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# ── Rule 1: English act keyword + Gregorian year ────────────────────────────
# Matches: "The Indian Penal Code, 1860", "The Negotiable Instruments Act 1881"
# Covers 18xx (colonial era), 19xx and 20xx.
# IMPORTANT: The "name part" (group 2) must be ≤ 12 words — this prevents the
# adaptation clause false-match where the regex greedily captures a long sentence
# ending in "...Andhra Pradesh Reorganisation ACT 2014".
_ACT_KEYWORD_YEAR = re.compile(
    r"\b(THE\s+)?([A-Za-z0-9\s,\'\-()&]{2,120}?)\s+"
    r"(ACT|CODE|ORDINANCE|REGULATION|RULES|SCHEDULE)\s*,?\s*(18|19|20)(\d{2})\b",
    re.IGNORECASE,
)

# ── Rule 1B: English act keyword + Fasli / Hijri year ───────────────────────
# Matches: "THE TELANGANA EUNUCHS ACT, 1329 F." or "... ACT 1317 Fasli"
# Fasli/Hijri years used in old Hyderabad/Telangana state acts are 3–4 digit
# numbers followed by "F", "F.", or "Fasli". We keep the Fasli year as-is in
# the output filename since converting to Gregorian is error-prone.
_ACT_FASLI_YEAR = re.compile(
    r"\b(THE\s+)?([A-Za-z0-9\s,\'\-()&]{2,120}?)\s+"
    r"(ACT|CODE|ORDINANCE|REGULATION|RULES|SCHEDULE)\s*,?\s*(\d{3,4})\s*(F\.?|Fasli\b)",
    re.IGNORECASE,
)

# ── Rule 2: Indian-language act keyword + Gregorian year ────────────────────
# Matches: "Bharatiya Nyaya Sanhita, 2023", "Bharatiya Sakshya Adhiniyam 2023"
_INDIAN_KEYWORD_YEAR = re.compile(
    r"\b(THE\s+)?([A-Za-z\s,\'\-()&]{2,100}?)\s+"
    r"(SANHITA|ADHINIYAM|NIYAMAWALI|NIYAM|SAMHITA|VIDHAN)\s*,?\s*(18|19|20)(\d{2})\b",
    re.IGNORECASE,
)

# ── Rule 3: First Gregorian year in text (18xx / 19xx / 20xx) ───────────────
_FIRST_YEAR = re.compile(r"(18|19|20)(\d{2})\b")

# ── Filename "clean" year: any 3–4 digit number (Gregorian or Fasli) ────────
_FILENAME_YEAR_LIKE = re.compile(r"\b\d{3,4}\b")

# ── Adaptation-clause context detector ───────────────────────────────────────
# Many Telangana PDFs open with a notice like:
#   "… has been adapted to the State of Telangana under section 101 of the
#    Andhra Pradesh Reorganisation Act, 2014 …"
# BEFORE the actual act title. Rule 1 then (incorrectly) picks up
# "Andhra Pradesh Reorganisation ACT 2014" as the act name.
#
# Fix: when iterating Rule-1 matches we look at the 400 chars *before* each
# match. If they contain adaptation-clause language we skip that match and
# continue to the next one (which is usually the real title).
_ADAPTATION_CONTEXT = re.compile(
    r"(adapted to|reorganisation act|reorganization act|extended to the state|"
    r"under section \d+ of the|as notified by|as modified by|"
    r"in its application to|with modifications|with adaptation)",
    re.IGNORECASE,
)

# Known administrative acts that ONLY appear as cross-references.
# If the extracted name part matches one of these, reject immediately.
_ADMIN_XREF_NAMES = re.compile(
    r"^(The\s+)?(Andhra Pradesh Reorganisation|States Reorganisation|"
    r"Government of Union Territories|Adaptation of Laws|"
    r"Andhra State Act|Central Acts Extension)\b",
    re.IGNORECASE,
)

# ── Quality check heuristics ─────────────────────────────────────────────────
_MIN_NAME_LEN = 5

# Max words allowed in the "name" portion before the keyword.
# "Andhra Pradesh Reorganisation" = 3 words → fine.
# "...words and expressions used and not defined in this Act but defined in the
#  Factories" = 18 words → reject (cross-reference sentence, not a title).
_MAX_NAME_WORDS = 12

# Phrases that appear in cross-references / adaptation boilerplate rather than
# as actual act titles. If the extracted name STARTS with one of these, reject it.
_CROSSREF_PREFIXES = re.compile(
    r"^("
    r"has been adapted|"
    r"words and expressions|"
    r"defined in (this|the)|"
    r"it has also been|"
    r"as amended|"
    r"inserted by|"
    r"substituted by|"
    r"under section \d|"
    r"subject to the|"
    r"in pursuance of"
    r")",
    re.IGNORECASE,
)

# If the filename (without extension) already looks like an act title, skip it.
# "THE TELANGANA LAND REVENUE ACT 1317" → already clean, don't touch.
# "a202345" → cryptic, needs renaming.
_FILENAME_ACT_PATTERN = re.compile(
    r"\b(ACT|CODE|ORDINANCE|REGULATION|SANHITA|ADHINIYAM|NIYAMAWALI|NIYAM)\b",
    re.IGNORECASE,
)


def _safe_filename(name: str) -> str:
    """Replace unsafe filesystem chars and collapse whitespace."""
    s = _UNSAFE_FS.sub(" ", name)
    return " ".join(s.split()).strip() or "Untitled"


def _strip_leading_number(name: str) -> str:
    """Act names don't start with a number — strip leading digits/dots."""
    s = (name or "").strip()
    return re.sub(r"^\d+[\.\)\s]*", "", s).strip()


def _is_adaptation_match(full_match_text: str) -> bool:
    """
    Return True if the full match text contains adaptation-clause language.

    Telangana adaptation notices look like:
      "The following Central Act has been adapted to the State of Telangana,
       under section 101 of the Andhra Pradesh Reorganisation Act, 2014"

    The regex greedily captures from "The following..." all the way to
    "Reorganisation Act, 2014", so the adaptation language is INSIDE the
    match — not in the context before it.  We check the match text itself.
    """
    return bool(_ADAPTATION_CONTEXT.search(full_match_text))


def _looks_poor(name: str) -> bool:
    """
    Return True if the extracted name seems wrong and should trigger fallback:
    - Too short (< _MIN_NAME_LEN chars)
    - Starts with a digit after stripping
    - Too many words (> _MAX_NAME_WORDS) → probably captured a sentence, not a title
    - Starts with a cross-reference/boilerplate phrase
    - Single word that is all-caps (section heading, not title)
    """
    n = name.strip()
    if len(n) < _MIN_NAME_LEN:
        return True
    if re.match(r"^\d", n):
        return True
    tokens = n.split()
    if len(tokens) > _MAX_NAME_WORDS:
        return True
    if _CROSSREF_PREFIXES.search(n):
        return True
    if len(tokens) == 1 and n.isupper():
        return True
    # Known administrative/reorganisation acts that only appear as xrefs
    if _ADMIN_XREF_NAMES.match(n):
        return True
    return False


def _filename_already_clean(basename_no_ext: str) -> bool:
    """
    Return True ONLY when the filename already contains BOTH an act keyword
    AND a year (Gregorian or Fasli).  If either is missing the file is still
    processed so we can extract a fully-qualified title from the PDF body.

    Examples that return True  → skip (already complete):
        "THE TELANGANA LAND REVENUE ACT 1317"     ← ACT + Fasli year
        "THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023"
        "The Contract Labour (Regulation And Abolition) ACT 1970"
        "The Guardians And Wards ACT 1890"         ← ACT + 1800s year

    Examples that return False → needs processing:
        "THE GUARDIANS AND WARDS ACT"   ← keyword present but NO year
        "Indian Succession Act"          ← no year
        "A1961-25", "a202345"            ← no keyword
    """
    has_keyword = bool(_FILENAME_ACT_PATTERN.search(basename_no_ext))
    has_year    = bool(_FILENAME_YEAR_LIKE.search(basename_no_ext))
    return has_keyword and has_year


def _extract_name_from_text(text: str, filename: str) -> tuple[str, str]:
    """
    Apply rename rules on first-2-pages text.
    Returns (base_name, year_str).
    base_name is a clean string like "Bharatiya Nyaya Sanhita ACT 2023".
    year_str is the 4-digit year as string, or "" if not found.
    """
    if not text or not text.strip():
        fallback = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ")
        return _safe_filename(fallback), ""

    # Normalise whitespace but keep enough text for the patterns to match
    sample = " ".join(text.split()).strip()[:4000]

    # ── Rule 1: English keyword + Gregorian year (ACT / CODE / ...) ────────
    # Iterate ALL matches so we can skip adaptation-clause hits.
    # Telangana PDFs often begin with "... adapted under the Andhra Pradesh
    # Reorganisation Act, 2014 ..." BEFORE the real act title — using only
    # search() would return that cross-reference as the name.
    for m in _ACT_KEYWORD_YEAR.finditer(sample):
        pre  = ((m.group(1) or "") + m.group(2)).strip()
        kw   = m.group(3).upper()
        year = m.group(4) + m.group(5)
        pre  = _strip_leading_number(" ".join(pre.split()))
        if _looks_poor(pre):
            continue
        if _is_adaptation_match(m.group(0)):
            logger.debug("Rule 1: skipping adaptation match '%s'", m.group(0)[:80])
            continue
        base = f"{pre} {kw} {year}"
        logger.debug("Rule 1 (English keyword, Gregorian year): %s", base)
        return _safe_filename(base), year

    # ── Rule 1B: English keyword + Fasli / Hijri year ────────────────────────
    # Handles old Hyderabad / Telangana acts like "THE TELANGANA EUNUCHS ACT, 1329 F."
    # Also iterated so adaptation-context hits are skipped.
    for m in _ACT_FASLI_YEAR.finditer(sample):
        pre      = ((m.group(1) or "") + m.group(2)).strip()
        kw       = m.group(3).upper()
        fasli_yr = m.group(4)
        fasli_kw = m.group(5).strip().rstrip(".")
        pre      = _strip_leading_number(" ".join(pre.split()))
        if _looks_poor(pre):
            continue
        if _is_adaptation_match(m.group(0)):
            logger.debug("Rule 1B: skipping adaptation Fasli match '%s'", m.group(0)[:80])
            continue
        base = f"{pre} {kw} {fasli_yr} {fasli_kw}."
        logger.debug("Rule 1B (Fasli year): %s", base)
        return _safe_filename(base), fasli_yr

    # ── Rule 2: Indian keyword + Gregorian year (SANHITA / ADHINIYAM / ...) ──
    m = _INDIAN_KEYWORD_YEAR.search(sample)
    if m:
        pre  = ((m.group(1) or "") + m.group(2)).strip()
        kw   = m.group(3).capitalize()     # Sanhita / Adhiniyam / ...
        year = m.group(4) + m.group(5)
        pre  = _strip_leading_number(" ".join(pre.split()))
        if not _looks_poor(pre):
            base = f"{pre} {kw} {year}"
            logger.debug("Rule 2 (Indian keyword): %s", base)
            return _safe_filename(base), year

    # ── Rule 3: First year; name = text immediately before it ────────────────
    m = _FIRST_YEAR.search(sample)
    if m:
        year   = m.group(1) + m.group(2)
        before = sample[: m.start()].strip()
        # Act title is usually at the tail of the intro text
        name_part = before[-200:].strip() if len(before) > 200 else before
        name_part = _strip_leading_number(" ".join(name_part.split()))
        if not _looks_poor(name_part):
            base = f"{name_part} {year}"
            logger.debug("Rule 3 (first year): %s", base)
            return _safe_filename(base), year

    # ── Rule 5 (final fallback): use filename ────────────────────────────────
    fallback = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ")
    logger.debug("Rule 5 (filename fallback): %s", fallback)
    return _safe_filename(fallback), ""


def _llm_extract_name(text_sample: str, filename: str) -> str:
    """
    Rule 4: Ask the LLM to identify the act name from the document opening.
    Returns a clean act name string, or "" if the LLM also fails.
    Only called when rules 1-3 produce a poor result AND --llm flag is used.
    """
    try:
        from llm.llm_client import call_llm

        snippet = " ".join(text_sample.split())[:600]
        prompt = (
            "You are a legal document classifier. The following is the opening text of an "
            "Indian legal document (bare act, code, ordinance, or rules).\n\n"
            f"DOCUMENT OPENING:\n{snippet}\n\n"
            "Task: Identify the OFFICIAL SHORT TITLE of this act. "
            "Return ONLY the act name including year (e.g. 'Bharatiya Nyaya Sanhita 2023', "
            "'Transfer of Property Act 1882', 'Telangana Land Encroachment Act 1905'). "
            "Do NOT add any explanation, prefix, or suffix. If you cannot identify the act, "
            "return UNKNOWN."
        )
        result = (call_llm(prompt) or "").strip()
        if result and result.upper() != "UNKNOWN" and len(result) > 4:
            return _safe_filename(result)
    except Exception as e:
        logger.debug("LLM fallback failed: %s", e)
    return ""


def _target_basename(path: str, text_sample: str, use_llm: bool = False) -> str:
    """
    Derive the target filename (basename with extension) for a given file.
    Applies rules 1-4 in sequence.
    """
    ext = os.path.splitext(path)[1].lower() or ".pdf"
    base_name, _ = _extract_name_from_text(text_sample, path)

    # If regex rules gave a poor result and LLM is enabled, try Rule 4
    if use_llm and _looks_poor(base_name):
        llm_name = _llm_extract_name(text_sample, path)
        if llm_name:
            logger.debug("Rule 4 (LLM fallback): %s", llm_name)
            base_name = llm_name

    return base_name + ext


def _resolve_collisions(renames: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    If two source files map to the same target basename, append _2, _3, etc.
    to the second and subsequent ones.
    """
    used: dict[str, int] = {}
    resolved = []
    for src, tgt in renames:
        key = tgt.lower()
        if key in used:
            used[key] += 1
            stem, ext = os.path.splitext(tgt)
            tgt = f"{stem}_{used[key]}{ext}"
        else:
            used[key] = 1
        resolved.append((src, tgt))
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename bare act PDFs/txt files to clean 'Name ACT YYYY' titles.\n"
            "Default mode is a DRY RUN — no files are changed.\n"
            "Pass --apply to actually rename."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        metavar="FOLDER",
        default=None,
        help=(
            "Folder containing the bare act files to rename. "
            "Defaults to BARE_ACTS_DIR from config.py "
            f"(currently: {BARE_ACTS_DIR})."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Without this flag only a dry run is shown.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use the LLM as a fallback when regex rules fail to produce a clean "
            "title. Requires the LLM service to be running. Slower but more accurate "
            "for Indian-language acts or non-standard formatting."
        ),
    )
    args = parser.parse_args()

    target_dir = os.path.normpath(args.dir) if args.dir else BARE_ACTS_DIR

    if not os.path.isdir(target_dir):
        logger.error(
            "Directory not found: %s\n"
            "Tip: if your files are in a Google Drive folder, pass the full local "
            "sync path with --dir, e.g.:\n"
            '  python scripts/rename_bare_acts.py --dir "C:/Users/you/Google Drive/BareActs" --apply',
            target_dir,
        )
        sys.exit(1)

    logger.info("Scanning: %s", target_dir)
    if args.llm:
        logger.info("LLM fallback enabled (Rule 4) — slow but accurate for ambiguous files.")
    if not args.apply:
        logger.info("DRY RUN — no files will be changed. Pass --apply to rename.")

    scan = scan_bare_acts_for_rename(target_dir)
    if not scan:
        logger.info("No PDF/txt files found in %s.", target_dir)
        return

    # Build (src_path, proposed_basename) pairs
    raw_renames: list[tuple[str, str]] = []
    skipped_clean = 0
    for path, text_sample in scan:
        current      = os.path.basename(path)
        current_stem = os.path.splitext(current)[0]

        # Skip files whose filename already looks like an act title.
        # Body-text extraction is unreliable for these (cross-ref risk).
        if _filename_already_clean(current_stem):
            logger.debug("SKIP (already clean): %s", current)
            skipped_clean += 1
            continue

        proposed = _target_basename(path, text_sample, use_llm=args.llm)
        if current != proposed:
            raw_renames.append((path, proposed))

    if skipped_clean:
        logger.info("Skipped %d file(s) that already have clean act-title names.", skipped_clean)

    if not raw_renames:
        logger.info("All %d file(s) already have clean names. Nothing to rename.", len(scan))
        return

    renames = _resolve_collisions(raw_renames)

    # Always print the plan
    logger.info("\n%s RENAMES PLANNED (%d files):", "DRY RUN —" if not args.apply else "APPLYING", len(renames))
    for src, tgt in renames:
        logger.info("  %-55s  →  %s", os.path.basename(src)[:55], tgt)

    if not args.apply:
        logger.info(
            "\nTo apply these renames, run:\n"
            "  python scripts/rename_bare_acts.py%s --apply",
            f' --dir "{target_dir}"' if args.dir else "",
        )
        return

    ok = err = 0
    for src, tgt_base in renames:
        tgt_path = os.path.join(target_dir, tgt_base)
        try:
            if os.path.abspath(src) == os.path.abspath(tgt_path):
                continue
            if os.path.isfile(tgt_path):
                logger.warning("  SKIP (target already exists): %s", tgt_base)
                err += 1
                continue
            os.rename(src, tgt_path)
            ok += 1
        except OSError as e:
            logger.warning("  FAILED %s: %s", os.path.basename(src), e)
            err += 1

    logger.info("\nDone: %d renamed, %d failed/skipped.", ok, err)


if __name__ == "__main__":
    main()
