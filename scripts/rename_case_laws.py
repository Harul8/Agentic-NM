"""
Case law filename generator — LLM-primary, regex fallback.

Target format:
  SC_YYYY_Appellant Name versus Respondent Name.pdf   (Supreme Court of India)
  HC_YYYY_Appellant Name versus Respondent Name.pdf   (any High Court)
  YYYY_Appellant Name versus Respondent Name.pdf      (court unknown)

The LLM reads the first ~3 000 chars (roughly the cover page) and extracts:
  court   — "SC" | "HC" | ""
  year    — 4-digit judgment year (from case number or date line), or ""
  first   — first/appellant/petitioner party name (main name only, no "& Ors.")
  second  — respondent/opposite party name (main name only)

A regex fast-path handles the most common well-structured formats without an LLM
call.  If it returns a complete result the LLM is skipped entirely.

Usage:
  # Dry run (no changes):
  python scripts/rename_case_laws.py
  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws"

  # Apply renames:
  python scripts/rename_case_laws.py --apply
  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws" --apply

  # Force LLM for every file (skip regex fast-path):
  python scripts/rename_case_laws.py --dir "..." --apply --llm-always

  # Skip LLM entirely (regex only):
  python scripts/rename_case_laws.py --dir "..." --apply --no-llm
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import CASELAW_DIR
from Ingestion.smart_chunker import extract_text_from_pdf_first_n_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Filesystem safety ─────────────────────────────────────────────────────────
_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# ── Max lengths ───────────────────────────────────────────────────────────────
_MAX_PARTY_CHARS  = 50    # per party name in filename
_MAX_STEM_CHARS   = 130   # total stem (before extension)

# ── Court detection ───────────────────────────────────────────────────────────
_COURT_SC = re.compile(r"supreme\s+court\s+of\s+india|supreme\s+court", re.IGNORECASE)
_COURT_HC = re.compile(r"high\s+court", re.IGNORECASE)

# ── Year patterns ─────────────────────────────────────────────────────────────
# From case numbers: "CIVIL APPEAL NO. 1234 OF 2023" → 2023
_YEAR_OF  = re.compile(r"\bof\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
# From dates: "decided on 15 January 2022" / "dated 15.01.2022"
_YEAR_GEN = re.compile(r"\b(20\d{2}|19\d{2})\b")

# ── Citation patterns (also give year) ───────────────────────────────────────
_SCC = re.compile(r"\((\d{4})\)\s*\d+\s*SCC\s*\d+", re.IGNORECASE)
_AIR = re.compile(r"AIR\s+(19\d{2}|20\d{2})\s+SC", re.IGNORECASE)

# ── VERSUS line (SC/HC multi-line format) ─────────────────────────────────────
# Matches "VERSUS" or "V S" or "VS" on its own line
_VERSUS_LINE = re.compile(
    r"^\s*(?:V\s*E\s*R\s*S\s*U\s*S|V\s*S\.?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── Inline "X v. Y" or "X versus Y" ─────────────────────────────────────────
_INLINE_VS = re.compile(
    r"((?:(?:M/s|Smt|Sri|Dr|Mr|Mrs|The|Union|State)\.?\s+)?[A-Z][A-Za-z0-9\s\.\,\'\-\(\)&]{4,55}?)"
    r"\s+(?:v\.?s?\.?|versus)\s+"
    r"((?:(?:M/s|Smt|Sri|Dr|Mr|Mrs|The|Union|State)\.?\s+)?[A-Z][A-Za-z0-9\s\.\,\'\-\(\)&]{4,55})",
    re.IGNORECASE,
)

# ── Annotation lines ("...Appellant", "...Respondent") ───────────────────────
_ANN_PET = re.compile(
    r"^([\w][^\n]{3,70}?)\s*\.{2,}\s*(?:Appellant|Petitioner|Plaintiff|Applicant)s?\b",
    re.IGNORECASE | re.MULTILINE,
)
_ANN_RES = re.compile(
    r"^([\w][^\n]{3,70}?)\s*\.{2,}\s*(?:Respondent|Defendant|Accused|Opposite\s+Party)s?\b",
    re.IGNORECASE | re.MULTILINE,
)

# ── Lines to reject as party names ───────────────────────────────────────────
_NOISE_LINE = re.compile(
    r"^\s*("
    r"judgment|order|coram|bench|hon.?ble|before|decided|dated|"
    r"civil\s+(appeal|appellate)|criminal\s+(appeal|appellate)|"
    r"writ\s+petition|letters\s+patent|slp|transfer\s+petition|"
    r"with\b|and\b|in\s+the\s+(matter|case)|"
    r"\d+[\s\.\-]"          # starts with digit (case number fragment)
    r")",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    """Strip filesystem-unsafe chars and normalise whitespace."""
    # M/s → Ms anywhere (/ not valid in filenames).
    # Use lookahead/lookbehind instead of \b because \b fails after "_" (word char).
    s = re.sub(r"(?<![A-Za-z0-9])M/[Ss](?![A-Za-z0-9])", "Ms", text)
    s = _UNSAFE_FS.sub(" ", s)
    return " ".join(s.split()).strip()


def _title(name: str) -> str:
    """Title-case a party name, keeping common lowercase connectives."""
    name = re.sub(r"\bM\s*/\s*[Ss]\.?\b", "M/s", " ".join(name.split()))
    lowers = {"of", "and", "the", "in", "at", "by", "for", "a", "an", "v", "vs"}
    words = name.split()
    out = []
    for i, w in enumerate(words):
        if re.match(r"^M/s\.?$", w, re.IGNORECASE):
            out.append("M/s")
        elif re.match(r"^(smt|sri|dr|mr|mrs)\.?$", w, re.IGNORECASE):
            out.append(w.capitalize().rstrip(".") + ".")
        elif i > 0 and w.lower() in lowers:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def _shorten(name: str, max_chars: int = _MAX_PARTY_CHARS) -> str:
    """Trim to max_chars at a word boundary, stripping trailing punctuation."""
    name = name.strip()
    # Drop "& Ors", "& Others", "and Others" suffixes
    name = re.sub(r"\s*[&,]\s*(Ors\.?|Others?|Anr\.?|Another)\s*$", "", name, flags=re.IGNORECASE).strip()
    if len(name) <= max_chars:
        return name
    return name[:max_chars].rsplit(" ", 1)[0].rstrip(",.&")


def _clean_party(raw: str) -> str:
    """Title-case + shorten, returning "" if result looks poor."""
    raw = raw.strip()
    # Strip annotation tails that slipped through ("… Appellant")
    raw = re.sub(
        r"\s*\.{2,}\s*(?:Appellant|Respondent|Petitioner|Plaintiff|Defendant|Applicant)s?\b.*",
        "", raw, flags=re.IGNORECASE,
    ).strip()
    # Strip citation suffixes that greedy _INLINE_VS group 2 may have captured.
    # e.g. "State of Maharashtra (2022) 5 SCC 123" → "State of Maharashtra"
    # e.g. "State of Telangana AIR 2019 SC 450"   → "State of Telangana"
    # e.g. "State of Telangana Civil Appeal No. 1234 of 2021" → "State of Telangana"
    raw = re.sub(r"\s*\((?:19|20)\d{2}\)\s*.*$", "", raw)                               # "(YYYY) …"
    raw = re.sub(r"\s+(?:AIR|SCC|SCR|MANU|SCJ)\s+.*$", "", raw, flags=re.IGNORECASE)    # "AIR YYYY …"
    raw = re.sub(r"\s+In\s+the\s+(?:Supreme|High)\s+.*$", "", raw, flags=re.IGNORECASE) # "In the Supreme Court …"
    raw = re.sub(                                                                         # "Civil/Criminal Appeal No./Writ Petition …"
        r"\s+(?:Civil|Criminal|Writ|Transfer|Special|Original)\s+(?:Appeal|Petition|Suit|Leave|Application).*$",
        "", raw, flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s+No\.\s+\d+.*$", "", raw, flags=re.IGNORECASE)                     # "No. 1234 …"
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw or len(raw) < 3:
        return ""
    if _NOISE_LINE.match(raw):
        return ""
    # Reject if more than 10 words (captured a sentence, not a name)
    if len(raw.split()) > 10:
        return ""
    return _shorten(_title(raw))


def _detect_court(text: str) -> str:
    sample = text[:2000]
    if _COURT_SC.search(sample):
        return "SC"
    if _COURT_HC.search(sample):
        return "HC"
    return ""


def _detect_year(text: str) -> str:
    """Best-effort year extraction — prefers citation year, then case-number year."""
    sample = text[:3000]
    # 1) SCC / AIR citation year
    for pat in (_SCC, _AIR):
        m = pat.search(sample)
        if m:
            return m.group(1)
    # 2) "... of 20XX" (case number)
    m = _YEAR_OF.search(sample)
    if m:
        return m.group(1)
    # 3) First 4-digit year in first 2000 chars
    m = _YEAR_GEN.search(text[:2000])
    if m:
        return m.group(1)
    return ""


def _filename_already_clean(stem: str) -> bool:
    """
    Return True if the filename already follows our target convention, i.e.
    has "versus" (or "v." / "vs.") AND a 4-digit year.
    These are skipped to avoid renaming files that were already fixed.
    """
    has_vs   = bool(re.search(r"\bversus\b|\bv\.?\s+[A-Z]|\bvs\.?\s+[A-Z]", stem, re.IGNORECASE))
    # Use lookahead/lookbehind instead of \b for the year: \b fails inside "SC_2022_…"
    # because "_" is a \w character (no word boundary between "_" and "2").
    has_year = bool(re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", stem))
    return has_vs and has_year


# ─────────────────────────────────────────────────────────────────────────────
# Regex fast-path extractor
# ─────────────────────────────────────────────────────────────────────────────

def _regex_extract(text: str) -> dict:
    """
    Try to extract {court, year, first, second} using pure regex.
    Returns a dict; any missing field is an empty string.
    'first' = petitioner/appellant, 'second' = respondent.
    """
    raw = text[:6000]
    flat = " ".join(text.split())[:5000]

    court = _detect_court(raw)
    year  = _detect_year(raw)

    first = second = ""

    # ── Try VERSUS on its own line first ────────────────────────────────────
    vm = _VERSUS_LINE.search(raw)
    if vm:
        before = raw[max(0, vm.start() - 400): vm.start()].strip()
        after  = raw[vm.end(): vm.end() + 400].strip()

        def _best_party_from_block(block: str, take_last: bool) -> str:
            """Strip annotation suffixes, filter noise lines, return best candidate."""
            block = re.sub(
                r"\s*\.{2,}\s*(?:Appellant|Petitioner|Plaintiff|Complainant|Applicant|"
                r"Respondent|Defendant|Accused|Opposite\s+Party)s?\b.*",
                "", block, flags=re.IGNORECASE,
            )
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            lines = [l for l in lines if not _NOISE_LINE.match(l)
                     and not re.match(r"^\(?\d{4}\)?", l)]
            if not lines:
                return ""
            candidate = lines[-1] if take_last else lines[0]
            return _clean_party(candidate)

        first  = _best_party_from_block(before, take_last=True)
        second = _best_party_from_block(after,  take_last=False)

    # ── Inline "X v. Y" (citation-header or SCC Online style) ──────────────
    if not first or not second:
        for m in _INLINE_VS.finditer(flat):
            p = _clean_party(m.group(1))
            r = _clean_party(m.group(2))
            if p and r:
                first, second = p, r
                break

    # ── Annotation scan ("...Appellant" / "...Respondent") ──────────────────
    if not first or not second:
        pm = _ANN_PET.search(raw)
        rm = _ANN_RES.search(raw)
        if pm:
            first  = _clean_party(pm.group(1))
        if rm:
            second = _clean_party(rm.group(1))

    return {"court": court, "year": year, "first": first, "second": second}


# ─────────────────────────────────────────────────────────────────────────────
# LLM extractor
# ─────────────────────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
You are a legal document parser for Indian courts.
Read the cover page of this judgment and return EXACTLY four lines:

COURT: SC          ← "SC" for Supreme Court of India, "HC" for any High Court, else blank
YEAR: 2022         ← 4-digit year from case number or decision date, else blank
FIRST: Vijay Kumar ← appellant / petitioner main name ONLY — no "& Ors.", no designation
SECOND: State of Maharashtra  ← respondent main name ONLY — no "& Ors."

Rules:
- Output nothing except those four lines.
- If a field cannot be determined, leave it blank after the colon.
- Names: title-case, main party only, max 8 words.
- For "State of X" use "State of X" not abbreviations.

COVER PAGE TEXT:
{snippet}
"""


def _llm_extract(text: str) -> dict:
    """
    Call the LLM on the first ~3 000 chars and parse its structured response.
    Returns {court, year, first, second}; any missing field is "".
    """
    result = {"court": "", "year": "", "first": "", "second": ""}
    try:
        from llm.llm_client import call_llm
        snippet = " ".join(text.split())[:3000]
        prompt  = _LLM_PROMPT.format(snippet=snippet)
        raw_out = (call_llm(prompt) or "").strip()

        for line in raw_out.splitlines():
            line = line.strip()
            if line.upper().startswith("COURT:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("SC", "HC"):
                    result["court"] = val
            elif line.upper().startswith("YEAR:"):
                val = line.split(":", 1)[1].strip()
                if re.match(r"^(19|20)\d{2}$", val):
                    result["year"] = val
            elif line.upper().startswith("FIRST:"):
                result["first"] = _clean_party(line.split(":", 1)[1].strip())
            elif line.upper().startswith("SECOND:"):
                result["second"] = _clean_party(line.split(":", 1)[1].strip())

    except Exception as e:
        logger.debug("LLM extract failed: %s", e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build target filename
# ─────────────────────────────────────────────────────────────────────────────

def _build_stem(court: str, year: str, first: str, second: str) -> str:
    """Assemble the filename stem from the four extracted fields."""
    parts = []
    if court:
        parts.append(court)
    if year:
        parts.append(year)
    prefix = "_".join(parts)   # "SC_2022" | "HC_2022" | "SC" | "2022" | ""

    if first and second:
        name_part = f"{first} versus {second}"
    elif first:
        name_part = first
    elif second:
        name_part = second
    else:
        return ""

    stem = f"{prefix}_{name_part}" if prefix else name_part
    if len(stem) > _MAX_STEM_CHARS:
        stem = stem[:_MAX_STEM_CHARS].rsplit(" ", 1)[0]
    return _safe(stem)


def _target_basename(
    path: str,
    text: str,
    use_llm: bool = True,
    llm_always: bool = False,
) -> str:
    """
    Derive the target filename (stem + extension) for one case law file.

    Strategy:
      1. Try regex fast-path.
      2. If fast-path gives a complete result (first + second + year) → use it.
      3. Otherwise (or if llm_always=True) call the LLM and merge results:
         LLM fills in any field the regex left blank.
      4. Build stem from merged fields.
      5. Fall back to current filename stem if nothing worked.
    """
    ext = os.path.splitext(path)[1].lower() or ".pdf"

    # Step 1: regex
    rx = _regex_extract(text)
    complete = bool(rx["first"] and rx["second"] and rx["year"])

    # Step 2: maybe call LLM
    lm: dict = {}
    if use_llm and (not complete or llm_always):
        lm = _llm_extract(text)

    # Step 3: merge — LLM fills gaps left by regex
    court  = rx["court"]  or lm.get("court",  "")
    year   = rx["year"]   or lm.get("year",   "")
    first  = rx["first"]  or lm.get("first",  "")
    second = rx["second"] or lm.get("second", "")

    source = "regex" if complete else ("regex+llm" if lm else "regex-only")
    logger.debug("(%s) court=%s year=%s first=%r second=%r", source, court, year, first, second)

    # Step 4: build stem
    stem = _build_stem(court, year, first, second)

    # Step 5: fallback
    if not stem:
        stem = _safe(os.path.splitext(os.path.basename(path))[0]
                     .replace("_", " ").replace("-", " "))
        logger.debug("Fallback to filename stem: %s", stem)

    return stem + ext


# ─────────────────────────────────────────────────────────────────────────────
# Directory scanning
# ─────────────────────────────────────────────────────────────────────────────

def _scan(directory: str) -> list[tuple[str, str]]:
    """Return (filepath, first_3_pages_text) for every PDF/txt in directory."""
    results = []
    if not os.path.isdir(directory):
        return results
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not (low.endswith(".pdf") or low.endswith(".txt")):
            continue
        try:
            if low.endswith(".pdf"):
                text = extract_text_from_pdf_first_n_pages(path, n=3)
            else:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()[:6000]
            results.append((path, text or ""))
        except Exception as e:
            logger.warning("Skip %s: %s", name, e)
    return results


def _resolve_collisions(renames: list[tuple[str, str]]) -> list[tuple[str, str]]:
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename case law PDFs to 'SC_YYYY_Petitioner versus Respondent.pdf'.\n"
            "Default mode is a DRY RUN — no files are changed.\n"
            "Pass --apply to actually rename."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir", metavar="FOLDER", default=None,
        help=(
            "Folder containing the case law files. "
            f"Defaults to CASELAW_DIR from config.py (currently: {CASELAW_DIR})."
        ),
    )
    parser.add_argument("--apply", action="store_true",
                        help="Apply the renames (default: dry run).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Regex only — never call the LLM.")
    parser.add_argument("--llm-always", action="store_true",
                        help="Always call the LLM even when regex gives a full result.")
    args = parser.parse_args()

    target_dir = os.path.normpath(args.dir) if args.dir else CASELAW_DIR

    if not os.path.isdir(target_dir):
        logger.error(
            "Directory not found: %s\n"
            "Pass the full path with --dir, e.g.:\n"
            '  python scripts/rename_case_laws.py --dir "G:/My Drive/Nyaymalaw/CaseLaws" --apply',
            target_dir,
        )
        sys.exit(1)

    use_llm    = not args.no_llm
    llm_always = args.llm_always

    logger.info("Scanning: %s", target_dir)
    logger.info(
        "LLM mode: %s",
        "disabled" if not use_llm else ("always" if llm_always else "fallback only"),
    )
    if not args.apply:
        logger.info("DRY RUN — no files will be changed. Pass --apply to rename.")

    scan = _scan(target_dir)
    if not scan:
        logger.info("No PDF/txt files found in %s.", target_dir)
        return

    raw_renames: list[tuple[str, str]] = []
    skipped_clean = 0

    for path, text in scan:
        current      = os.path.basename(path)
        current_stem = os.path.splitext(current)[0]

        if _filename_already_clean(current_stem):
            logger.debug("SKIP (already clean): %s", current)
            skipped_clean += 1
            continue

        proposed = _target_basename(path, text, use_llm=use_llm, llm_always=llm_always)
        if current != proposed:
            raw_renames.append((path, proposed))

    if skipped_clean:
        logger.info("Skipped %d file(s) that already have clean names.", skipped_clean)

    if not raw_renames:
        logger.info("All %d file(s) already have clean names. Nothing to rename.", len(scan))
        return

    renames = _resolve_collisions(raw_renames)

    logger.info(
        "\n%s RENAMES PLANNED (%d files):",
        "DRY RUN —" if not args.apply else "APPLYING",
        len(renames),
    )
    for src, tgt in renames:
        logger.info("  %-60s  →  %s", os.path.basename(src)[:60], tgt)

    if not args.apply:
        logger.info(
            "\nTo apply:\n  python scripts/rename_case_laws.py%s --apply",
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
                logger.warning("  SKIP (exists): %s", tgt_base)
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
