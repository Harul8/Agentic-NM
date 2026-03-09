import argparse
import os
import sys
import logging

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CASELAW_DIR
import scripts.rename_case_laws as rcl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from openpyxl import Workbook, load_workbook
except ImportError:
    Workbook = load_workbook = None


def _extract_fields(text: str, use_llm: bool, llm_always: bool):
    """
    Mirror rename_case_laws._target_basename's extraction:
    merge regex + LLM, then apply all final Between:/AND corrections.
    Returns (court, year, first, second).
    """
    rx = rcl._regex_extract(text)
    complete = bool(rx["first"] and rx["second"] and rx["year"])

    lm = {}
    if use_llm and (not complete or llm_always):
        lm = rcl._llm_extract(text)

    court = rx["court"] or lm.get("court", "")
    year = rx["year"] or lm.get("year", "")
    first = rx["first"] or lm.get("first", "")
    second = rx["second"] or lm.get("second", "")

    return court, year, first, second


def _proposed_basename(path: str, text: str, use_llm: bool, llm_always: bool):
    """Return (proposed_filename, first, second) using the same logic as rename_case_laws.

    - first/second are normalized with _first_party_only (so they match Party A/B used in filenames)
    - if we fall back to petition-number-only, first/second are returned as empty strings (we did not
      trust any party extraction enough to name the file from it).
    """
    ext = os.path.splitext(path)[1].lower() or ".pdf"
    court, year, first_raw, second_raw = _extract_fields(text, use_llm, llm_always)

    # Normalize parties exactly as the rename logic does
    first = rcl._first_party_only(first_raw) if first_raw else ""
    second = rcl._first_party_only(second_raw) if second_raw else ""

    stem = rcl._build_stem(court, year, first, second)

    # Petition/case-number fallback (text, then filename) – same as _target_basename
    if not stem:
        petition = rcl._extract_petition_number(text)
        if not petition:
            current_stem = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
            petition = rcl._extract_petition_number(current_stem)
        if petition:
            stem = petition
            logger.debug("Using petition/case number as filename for export: %s", stem)
            # If we are naming by petition number only, do not report parties (they were not trusted)
            first = ""
            second = ""

    if not stem:
        stem = rcl._safe(os.path.splitext(os.path.basename(path))[0]
                         .replace("_", " ").replace("-", " "))
        logger.debug("Fallback to filename stem for export: %s", stem)

    return stem + ext, first, second


def _export_parties(directory: str, use_llm: bool, llm_always: bool):
    if Workbook is None or load_workbook is None:
        logger.error("openpyxl is required for Excel export. Install with: pip install openpyxl")
        sys.exit(1)

    logger.info("Scanning %s for PDF/txt case-law files to export parties...", directory)
    scan = rcl._scan(directory)
    if not scan:
        logger.info("No PDF/txt files found in %s.", directory)
        return

    xlsx_path = os.path.join(directory, "1. case lasw party names.xlsx")

    if os.path.isfile(xlsx_path):
        wb = load_workbook(xlsx_path)
        ws = wb.active
        ws.title = "Parties"
        # Clear existing rows
        ws.delete_rows(1, ws.max_row)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Parties"

    # Header
    ws.append(["Original filename", "Party A (first)", "Party B (second)", "Proposed filename"])

    for path, text in scan:
        filename = os.path.basename(path)
        proposed, first, second = _proposed_basename(path, text or "", use_llm, llm_always)
        ws.append([filename, first or "", second or "", proposed])

    wb.save(xlsx_path)
    logger.info("Wrote party names for %d file(s) to %s", len(scan), xlsx_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export Party A / Party B / proposed filename for case-law PDFs to Excel.",
    )
    parser.add_argument(
        "--dir", metavar="FOLDER", default=None,
        help=f"Folder containing case law PDFs (default: CASELAW_DIR = {CASELAW_DIR})",
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Regex only — never call the LLM.")
    parser.add_argument("--llm-always", action="store_true",
                        help="Always call the LLM even when regex gives a full result.")
    args = parser.parse_args()

    target_dir = os.path.normpath(args.dir) if args.dir else CASELAW_DIR
    if not os.path.isdir(target_dir):
        logger.error("Directory not found: %s", target_dir)
        sys.exit(1)

    use_llm = not args.no_llm
    llm_always = args.llm_always

    _export_parties(target_dir, use_llm, llm_always)


if __name__ == "__main__":
    main()

