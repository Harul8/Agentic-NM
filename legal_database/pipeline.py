import os
import re
import sys
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz      # PyMuPDF — used if available for BareActs PDF extraction
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
import pdfplumber    # used for BareActs PDF extraction
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── RUN CONFIGURATION  ────────────────────────────────────────────────────────
#
# HOW TO USE WHEN YOU ADD NEW CASE LAW FILES
# ------------------------------------------
# Option A — you added specific month folders (most common incremental use):
#
#   NEW_FOLDERS = ["2025/MAR", "2025/APR"]   ← list the folders you just added
#
#   Then run:  python pipeline.py
#   Existing JSON files are NEVER touched.  Only new .txt files get processed.
#
# Option B — you want to process a date range (initial bulk load / backfill):
#
#   NEW_FOLDERS = []                          ← empty = use date range below
#   BATCH_START_YEAR  = 2020
#   BATCH_START_MONTH = "JAN"
#   BATCH_END_YEAR    = 2020
#   BATCH_END_MONTH   = "DEC"
#
# ── INDEX BUILD CONFIGURATION ─────────────────────────────────────────────────
#
# RUN_INDEX_APPEND  — controls how FAISS + chunks JSON are updated when you
#                     call run_build_indexes_only() for a new year/batch.
#
#   False (default) — build a fresh index from scratch for the configured date
#                     range.  Use this when doing your first ever index build,
#                     or when you want to rebuild everything cleanly.
#
#   True            — EXTEND the existing FAISS index + chunks JSON with the
#                     new batch.  Use this AFTER the first build when adding a
#                     new year without touching previous years' vectors.
#                     BM25 is always rebuilt from the full accumulated corpus
#                     so IDF values remain correct across batches.
#
#   Typical batched workflow:
#     1. Set RUN_INDEX_APPEND = False, date range = 2020/JAN–2020/DEC → run
#     2. Set RUN_INDEX_APPEND = True,  date range = 2021/JAN–2021/DEC → run
#     3. Set RUN_INDEX_APPEND = True,  date range = 2022/JAN–2022/DEC → run
#     ... (repeat for each year)
#
# ─────────────────────────────────────────────────────────────────────────────
NEW_FOLDERS       = []      # e.g. ["2025/MAR", "2025/APR"] — leave [] for date-range mode

BATCH_START_YEAR  = 2020
BATCH_START_MONTH = "JAN"   # JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
BATCH_END_YEAR    = 2025
BATCH_END_MONTH   = "DEC"
BATCH_ORDER       = "asc"   # "asc" | "desc"
BATCH_WORKERS     = 4

RUN_INDEX_APPEND  = False   # True = extend existing index; False = fresh build
# ─────────────────────────────────────────────────────────────────────────────


def _file_stem(path: str) -> str:
    """Return the filename without extension."""
    return os.path.splitext(os.path.basename(path or ""))[0]


def _sanitize_id_part(s: str) -> str:
    """Sanitize a string for use in IDs: uppercase, collapse spaces, remove invalid chars."""
    if not s:
        return ""
    s = re.sub(r"\s+", "_", str(s).upper().strip())
    s = re.sub(r"[^\w\-]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def _build_case_id(court: str, year: str, case_name: str | None, fallback_file_stem: str) -> str:
    """Build case ID like SC_2020_ABHILASHA_PARKASH."""
    if "SUPREME COURT" in (court or "").upper():
        prefix = "SC"
    elif "HIGH COURT" in (court or "").upper():
        prefix = "HC"
    else:
        prefix = "COURT"
    yr = (year or "UNKNOWN").strip()
    if not yr or not re.match(r"^(19|20)\d{2}$", yr):
        yr = "UNKNOWN"
    if case_name and case_name.strip():
        raw = case_name.strip().upper()
        for sep in (" v ", " V ", " versus ", " Versus "):
            if sep in raw:
                parts = raw.split(sep, 1)
                left = _sanitize_id_part(parts[0])
                right = _sanitize_id_part(parts[1])
                name_part = f"{left}_{right}" if left and right else (left or right)
                break
        else:
            name_part = _sanitize_id_part(raw)
    else:
        name_part = _sanitize_id_part(fallback_file_stem) or "UNKNOWN"
    return f"{prefix}_{yr}_{name_part}"[:200]


def _build_chunk_id(case_id: str, paragraph_id: int) -> str:
    """Build chunk ID like {case_id}_P001_C01."""
    return f"{case_id}_P{int(paragraph_id):03d}_C01"


def _build_act_id(act_name: str, year_str: str | None) -> str:
    """Build act ID like INDIAN_PENAL_CODE_1860."""
    name_part = _sanitize_id_part(act_name or "ACT")
    if year_str and str(year_str).strip():
        return f"{name_part}_{str(year_str).strip()}"
    return name_part


def _build_section_id(act_id: str, sec_num: str) -> str:
    """Build section ID like {act_id}_SECTION_302."""
    sec = re.sub(r"[^\w\-]", "", str(sec_num))
    return f"{act_id}_SECTION_{sec}"


# ---------------------------------------------------------------------------
# End-to-end pipeline: PDFs → json_output → cases/statutes index → vector store
# ---------------------------------------------------------------------------
# 1. PDFs in raw_data/BareActs and raw_data/CaseLaws → extracted to json_output (case-law and statute JSON).
# 2. cases/case_metadata.json and statutes/statutes.json are aggregated for listing.
# 3. json_output is converted to retrieval chunks and used to build:
#    - vector_store/bareacts_v2.index + bareacts_v2_chunks.json + bareacts_bm25.json
#    - vector_store/caselaws_v2.index + caselaws_v2_chunks.json + caselaws_bm25.json
#    - vector_store/citation_graph.json (case→section, case→case for precedent expansion)
# Set NYAYMALAW_DATA_SOURCE=legal_database and use this vector_store for high-accuracy hybrid retrieval.

INPUT_DIR          = os.path.join(os.path.dirname(__file__), "raw_data")
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "json_output")   # bare-act JSONs (flat)
CASES_DIR          = os.path.join(os.path.dirname(__file__), "cases")
STATUTES_DIR       = os.path.join(os.path.dirname(__file__), "statutes")
# Vector store (FAISS + BM25 + citation graph) — written by pipeline when building indexes
VECTOR_STORE_DIR   = os.path.join(os.path.dirname(__file__), "vector_store")
# .txt case law inputs and mirrored JSON outputs
CASELAWS_INPUT_DIR = os.path.join(INPUT_DIR, "CaseLaws")
CASE_OUTPUT_DIR    = os.path.join(OUTPUT_DIR, "caselaws")   # case-law JSONs in YYYY/MON/ tree

# Month abbreviation → integer (for date-range sorting/filtering)
_MONTH_ORDER = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH_ABBREVS = list(_MONTH_ORDER.keys())


# ------------------------------------------------
# BareActs PDF text extraction (pdfplumber)
# ------------------------------------------------

def _extract_text_pdfplumber(pdf_path: str) -> list:
    """Extract text page-by-page via pdfplumber (used for BareActs)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def extract_text_normal(pdf_path: str) -> list:
    """Extract text from a bare-act PDF via pdfplumber."""
    return _extract_text_pdfplumber(pdf_path)

# ------------------------------------------------
# Clean page noise
# ------------------------------------------------

def clean_text(text: str) -> str:
    """Remove page headers/footers, fix spacing, normalise line breaks."""
    if not text:
        return ""
    text = text.replace("\t", " ")
    # Remove standalone page numbers on their own line (e.g. "Page 3" or just "3")
    text = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", text)
    # Remove bracketed page-number stamps common in Telangana HC PDFs e.g. "[ 3385 ]"
    text = re.sub(r"(?m)^\s*\[\s*\d{3,5}\s*\]\s*$", "", text)
    # Remove bare page numbers on their own line
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Collapse multiple spaces / tabs on a single line (preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Paragraph splitting: split on "N. " at line start (MULTILINE); then merge when
# previous segment ends with a digit to avoid false splits (e.g. "Section 2. ", "2020. 2. ")
_PARA_NUM_PATTERN = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_SENT_BOUNDARY = re.compile(r"(?<=[a-z]{3}[.!?])\s+(?=[A-Z\(])")
_CASE_CHUNK_LIMIT = 1200
_MIN_PREAMBLE_LEN = 100
_MIN_PARA_LEN = 50
_MIN_BLOCK_LEN = 80


def _clean_ocr_noise(text: str) -> str:
    """Strip common OCR artifacts: standalone page numbers, single-letter lines, excess blanks."""
    if not (text or "").strip():
        return text or ""
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if len(s) <= 2 and re.match(r"^[A-Za-z]$", s):
            continue
        if len(s) == 2 and s.isalpha():
            continue
        out.append(line)
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", "\n".join(out))
    return result.strip()


def _split_para_to_limit(text: str, para_label: str) -> list:
    """
    Split a case-law paragraph into chunks of at most _CASE_CHUNK_LIMIT chars,
    breaking at sentence boundaries. Returns list of (label, chunk_text).
    """
    if len(text) <= _CASE_CHUNK_LIMIT:
        return [(para_label, text)]
    raw_sentences = _SENT_BOUNDARY.split(text)
    sentences = []
    for sent in raw_sentences:
        while len(sent) > _CASE_CHUNK_LIMIT:
            cut = sent.rfind(" ", 0, _CASE_CHUNK_LIMIT)
            if cut == -1:
                cut = _CASE_CHUNK_LIMIT
            sentences.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            sentences.append(sent)
    result = []
    buffer = ""
    split_idx = 0
    for sent in sentences:
        if buffer and len(buffer) + 1 + len(sent) > _CASE_CHUNK_LIMIT:
            label = para_label if split_idx == 0 else f"{para_label}_{split_idx}"
            result.append((label, buffer.strip()))
            split_idx += 1
            buffer = sent
        else:
            buffer = (buffer + " " + sent).strip() if buffer else sent
    if buffer.strip():
        label = para_label if split_idx == 0 else f"{para_label}_{split_idx}"
        result.append((label, buffer.strip()))
    return result


def _split_numbered_subparas(text: str) -> list[str]:
    """
    Within a block of text, split when a new numbered paragraph starts
    (e.g. '2. Background facts…'). This prevents sequences like
    '... for 3 years. 2. Background facts...' from being treated as a
    single paragraph.
    """
    parts: list[str] = []
    current = ""

    # Split but keep the delimiters (the 'N. ' tokens) so we can attach
    # them to the following text.
    tokens = re.split(r"(\d+\.\s+)", text)
    for tok in tokens:
        if not tok:
            continue
        if re.fullmatch(r"\d+\.\s+", tok):
            # Start of a new numbered paragraph
            if current.strip():
                parts.append(current.strip())
            current = tok
        else:
            current += tok

    if current.strip():
        parts.append(current.strip())

    return parts


# ------------------------------------------------
# Extract metadata — scans first 3 pages for header, extracts all fields
# ------------------------------------------------

# Petition / case-number patterns (ordered most-specific → least-specific)
# Each entry: (label, pattern, format_string)
_PETITION_PATTERNS = [
    ("WRIT_PETITION",
     r"WRIT\s+PETITION\s*(?:\([A-Z]+\))?\s*NO\.?S?\s*:?\s*([\d\s,ANDand]+)\s+OF\s+(\d{4})",
     "Writ Petition {} of {}"),
    ("CRIMINAL_PETITION",
     r"CRIMINAL\s+PETITION\s*NO\.?S?\s*:?\s*([\d\s,ANDand]+)\s+OF\s+(\d{4})",
     "Criminal Petition {} of {}"),
    ("CIVIL_APPEAL",
     r"CIVIL\s+APPEAL\s+NO\.?S?\s*\.?\s*([\d\s\-]+)\s+OF\s+(\d{4})",
     "Civil Appeal {} of {}"),
    ("CRIMINAL_APPEAL",
     r"CRIMINAL\s+APPEAL\s+NO\.?S?\s*\.?\s*([\d\s\-]+)\s+OF\s+(\d{4})",
     "Criminal Appeal {} of {}"),
    ("WP_SHORT",
     r"W\.?P\.?\s*(?:NO\.?|NOS?\.?)?\s*:?\s*(\d+)\s+OF\s+(\d{4})",
     "WP {} of {}"),
    ("SLP",
     r"S\.?L\.?P\.?\s*\(?[A-Z]*\)?\s*No\.?\s*([\d\-]+)\s*/?\s*(\d{4})",
     "SLP {} of {}"),
    ("GENERIC_PETITION",
     r"[A-Z][A-Z\s\(\)\.]*?(?:PETITION|APPEAL|APPLICATION)[A-Z\s\(\)\.]*NO\.?S?\s*:?\s*(\d+)\s+OF\s+(\d{4})",
     "Petition {} of {}"),
]

# Reporter citation patterns: AIR, SCC, SCR, INSC, SLT, etc.
_CITATION_PAT = re.compile(
    r"(?:"
    r"\[?\d{4}\]?\s+\d+\s+SCC\s+\d+"      # [2024] 7 SCC 1
    r"|\[?\d{4}\]?\s+\d+\s+SCR\s+\d+"      # [2024] 7 SCR 1236
    r"|AIR\s+\d{4}\s+[A-Z]+\s+\d+"         # AIR 2024 SC 100
    r"|\d{4}\s+INSC\s+\d+"                  # 2024 INSC 506
    r"|\d{4}\s+SCC\s+OnLine\s+[A-Z]+\s+\d+"
    r"|\[\d{4}\]\s+\d+\s+S\.C\.R\.\s+\d+"
    r")",
    re.I,
)

# Spelled-out date (High Court style): "TENTH DAY OF SEPTEMBER TWO THOUSAND AND TWENTY FIVE"
_SPELLED_DATE_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty": 20,
    "thirty": 30, "thirty-first": 31, "thirty first": 31,
    "twenty-first": 21, "twenty first": 21,
    "twenty-second": 22, "twenty second": 22,
    "twenty-third": 23, "twenty third": 23,
    "twenty-fourth": 24, "twenty fourth": 24,
    "twenty-fifth": 25, "twenty fifth": 25,
    "twenty-sixth": 26, "twenty sixth": 26,
    "twenty-seventh": 27, "twenty seventh": 27,
    "twenty-eighth": 28, "twenty eighth": 28,
    "twenty-ninth": 29, "twenty ninth": 29,
    "twenty-fi": 25,  # "twenty fi'e" OCR artefact for "twenty five"
}
_MONTH_WORDS = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June",
    "july": "July", "august": "August", "september": "September",
    "october": "October", "november": "November", "december": "December",
}
_YEAR_WORDS = {
    "nineteen": 1900, "twenty": 2000,
    "hundred": 100,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

_SPELLED_DATE_PAT = re.compile(
    r"(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)"
    r"\s*,?\s*THE\s+(.+?)\s+DAY\s+OF\s+(\w+)\s+(.+?)(?:\n|$)",
    re.I,
)
_NUMERIC_DATE_PAT = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})\b",
    re.I,
)

# Pattern A (High Court): "THE HONOURABLE SMT JUSTICE K. SUJANA"
_HONBLE_PAT = re.compile(
    r"(?:THE\s+)?HON(?:OU?RABLE)?\s+(?:SMT\.?\s+|SRI\.?\s+|MR\.?\s+|MRS\.?\s+|DR\.?\s+)?"
    r"(?:THE\s+CHIEF\s+JUSTICE\s+|JUSTICE\s+)([A-Z][A-Z\s\.\(\)]+?)(?=\s+J\.|\s+JJ\.|\n|,|$)",
    re.I,
)
# Pattern B (Supreme Court SCC/SCR): "[B.V. Nagarathna* and Augustine George Masih,* JJ.]"
_SC_JUDGE_PAT = re.compile(
    r"\[([A-Z][A-Za-z\s\.\,\*]+?)\s*JJ?\.\]",
    re.I,
)


def _parse_spelled_year(text: str) -> str | None:
    """Convert 'TWO THOUSAND AND TWENTY FIVE' → '2025'."""
    t = text.lower().strip()
    t = re.sub(r"\band\b", " ", t)
    tokens = t.split()
    total = 0
    for tok in tokens:
        v = _YEAR_WORDS.get(tok)
        if v is not None:
            total += v
    if 1900 < total <= 2100:
        return str(total)
    return None


def _extract_date_of_judgment(header: str) -> str | None:
    """Try spelled-out date first, then numeric date."""
    # Numeric date (most reliable): "10 July 2024"
    m = _NUMERIC_DATE_PAT.search(header)
    if m:
        return f"{int(m.group(1))} {m.group(2).capitalize()} {m.group(3)}"
    # Spelled-out date: "MONDAY,THE NINTH DAY OF JUNE TWO THOUSAND AND TWENTY FIVE"
    m = _SPELLED_DATE_PAT.search(header)
    if m:
        day_word = m.group(1).lower().strip()
        month_word = m.group(2).lower().strip()
        year_text = m.group(3).strip()
        day = _SPELLED_DATE_WORDS.get(day_word)
        month = _MONTH_WORDS.get(month_word)
        year = _parse_spelled_year(year_text)
        if day and month and year:
            return f"{day} {month} {year}"
    return None


def _extract_judges(header: str) -> list:
    """
    Extract judge names using two patterns:
      A) High Court style: 'THE HONOURABLE SMT JUSTICE K. SUJANA'
      B) SC/SCC style: '[B.V. Nagarathna* and Augustine George Masih,* JJ.]'
    """
    judges = []
    seen = set()

    def _add(name: str) -> None:
        # Strip trailing asterisks and extra whitespace
        name = re.sub(r"[\*]+", "", name).strip()
        name = re.sub(r"\s+", " ", name).strip().title()
        if name and name not in seen and len(name) > 3:
            seen.add(name)
            judges.append(name)

    # Pattern A — HC format
    for m in _HONBLE_PAT.finditer(header):
        _add(m.group(1))

    # Pattern B — SC SCC/SCR format: split on " and " or ","
    for m in _SC_JUDGE_PAT.finditer(header):
        raw = m.group(1)
        for part in re.split(r"\s+and\s+|,\s*", raw):
            part = part.strip()
            if part:
                _add(part)

    return judges


def _extract_case_number(header: str) -> tuple[str | None, str | None]:
    """
    Return (case_number, regex_label) for the first petition / appeal
    number found on the header pages.
    """
    for label, pat, fmt in _PETITION_PATTERNS:
        m = re.search(pat, header, re.I)
        if m:
            num = re.sub(r"\s+", " ", m.group(1)).strip()
            return fmt.format(num, m.group(2)), label
    return None, None


def _extract_reporter_citations(text: str) -> list:
    """Return all reporter citations found (deduplicated, ordered)."""
    seen, result = set(), []
    for m in _CITATION_PAT.finditer(text):
        c = re.sub(r"\s+", " ", m.group()).strip()
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _extract_outcome(pages: list) -> str | None:
    """Scan the last 3 pages for disposal language."""
    _OUTCOME_PAT = re.compile(
        r"\b(?:appeal|petition|application|revision)\s+is\s+"
        r"(allowed|dismissed|disposed\s+of|partly\s+allowed|allowed\s+in\s+part)",
        re.I,
    )
    for page in reversed(pages[-3:]):
        m = _OUTCOME_PAT.search(page or "")
        if m:
            return m.group(1).lower().replace("  ", " ")
    return None


# ------------------------------------------------
# Build paragraph chunks (ingestion-style: Path A numbered / Path B double-newline / Path C line-accum)
# ------------------------------------------------

def build_paragraph_chunks(case_id, pages):
    """
    Split case-law text into logical paragraphs. One paragraph = one chunk.
    Path A: split on "N. " at line start; merge when previous segment ends with
    a digit (avoids "Section 2. ", "2020. 2. " as new para). Path B/C fallback.
    """
    full_text = "\n".join(clean_text(p or "") for p in pages)
    full_text = _clean_ocr_noise(full_text)
    paragraphs = []
    paragraph_id = 0

    def emit_para(para_text: str, _para_label: str = "") -> None:
        """Emit one chunk per logical paragraph (no sub-splitting by length)."""
        nonlocal paragraph_id
        para_text = para_text.strip()
        if len(para_text) < _MIN_PARA_LEN:
            return
        paragraph_id += 1
        paragraphs.append({
            "paragraph_id": paragraph_id,
            "chunk_id": _build_chunk_id(case_id, paragraph_id),
            "text": para_text,
        })

    # Path A: numbered paragraphs at line start, with merge to avoid false splits
    para_splits = _PARA_NUM_PATTERN.split(full_text)
    if len(para_splits) > 3:
        merged = []
        current = para_splits[0].strip()
        for i in range(1, len(para_splits) - 1, 2):
            num = para_splits[i].strip()
            text = para_splits[i + 1].strip() if i + 1 < len(para_splits) else ""
            block = num + ". " + text
            if current.rstrip() and current.rstrip()[-1].isdigit():
                current = current + "\n" + block
            else:
                if current and (len(current) >= _MIN_PARA_LEN or (len(merged) == 0 and len(current) > _MIN_PREAMBLE_LEN)):
                    merged.append(current)
                current = block
        if current and (len(current) >= _MIN_PARA_LEN or (len(merged) == 0 and len(current) > _MIN_PREAMBLE_LEN)):
            merged.append(current)
        for para_text in merged:
            emit_para(para_text)
    else:
        # Path B: double-newline paragraphs
        blocks = [p.strip() for p in re.split(r"\n\s*\n", full_text) if len(p.strip()) >= _MIN_BLOCK_LEN]
        for i, block in enumerate(blocks):
            emit_para(block, str(i + 1))

    # Path C: last-resort line accumulator if we got too few chunks
    if len(paragraphs) < 3:
        paragraphs = []
        paragraph_id = 0
        lines = full_text.split("\n")
        current_chunk = []
        for line in lines:
            current_chunk.append(line)
            joined = "\n".join(current_chunk).strip()
            if len(joined) > _CASE_CHUNK_LIMIT:
                paragraph_id += 1
                paragraphs.append({
                    "paragraph_id": paragraph_id,
                    "chunk_id": _build_chunk_id(case_id, paragraph_id),
                    "text": joined,
                })
                current_chunk = []
        if current_chunk:
            joined = "\n".join(current_chunk).strip()
            if len(joined) >= _MIN_BLOCK_LEN:
                paragraph_id += 1
                paragraphs.append({
                    "paragraph_id": paragraph_id,
                    "chunk_id": _build_chunk_id(case_id, paragraph_id),
                    "text": joined,
                })

    return paragraphs


# ------------------------------------------------
# Case-law: annotate paragraphs (type, sections_cited, cited_cases) and chunk IDs
# ------------------------------------------------

def _is_bare_act(pdf_path: str) -> bool:
    """True if PDF is under a BareActs (or Bare Acts) folder."""
    path_norm = (pdf_path or "").replace("\\", "/")
    return "BareActs" in path_norm or "bare_act" in path_norm.lower()


# =============================================================================
# .TXT CASE LAW PROCESSING  (primary data source)
# =============================================================================
#
# File format (Indian Kanoon / portal scrape):
#
#   Line  1: [Cites
#   Line  2: <N>           ← integer
#   Line  3: , Cited by
#   Line  4: <M>           ← integer
#   Line  5: ]
#   Line  6: Supreme Court of India
#   Line  7: Party1 vs Party2 on DD Month, YYYY
#   Line  8: Author:
#   Line  9: <judge name>  (may be blank)
#   Line 10: Bench:
#   Lines 11+: <judge> / , / <judge> / ...  until INSC or body
#
# Body: numbered paragraphs, digital-signature noise blocks, page refs.
# =============================================================================

# ── Digital signature noise patterns ─────────────────────────────────────────
_TXT_DIG_SIG_PATS = [
    re.compile(
        r"Digitally signed by[^\n]*\n(?:[^\n]*\n){1,8}",
        re.IGNORECASE,
    ),
    re.compile(r"(?m)^Reason:\s*.+$"),
    re.compile(r"(?m)^Location:\s*.+$"),
    re.compile(r"(?m)^Date:\s*\d{4}\.\d{2}\.\d{2}[^\n]*$"),
    re.compile(r"(?m)^Signature\s+Not\s+Verified[^\n]*$", re.IGNORECASE),
    re.compile(r"(?m)^SMITHA\s+MANMATH\s+GANU[^\n]*$", re.IGNORECASE),
]

# ── Header field patterns ─────────────────────────────────────────────────────
_TXT_INSC_PAT      = re.compile(r"\b(\d{4})\s+INSC\s+(\d+)\b", re.IGNORECASE)
_TXT_DATE_LINE_PAT = re.compile(
    r"\bon\s+(\d{1,2})\s+(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s*,?\s*(\d{4})\b",
    re.IGNORECASE,
)
_TXT_EQUIV_PAT     = re.compile(r"^Equivalent citations?:\s*(.+)$", re.IGNORECASE)
_TXT_JURIS_PAT     = re.compile(
    r"\b(CIVIL\s+APPELLATE|CRIMINAL\s+APPELLATE|CIVIL\s+ORIGINAL|"
    r"CRIMINAL\s+ORIGINAL|WRIT|TRANSFER)\s+JURISDICTION\b",
    re.IGNORECASE,
)

# ── BareAct enrichment patterns ───────────────────────────────────────────────
# Cross-references: Section N, Sections N-M, s.N, Article N
_BA_XREF_PAT = re.compile(
    r"\b(?:Sections?\s+|s\.\s*|Articles?\s+|Clause\s+)(\d+(?:[A-Z])?(?:\s*(?:to|-)\s*\d+[A-Z]?)?)",
    re.IGNORECASE,
)
# Defined terms: "XYZ" means ..., 'XYZ' means, XYZ means  (within definition sections)
_BA_DEFINED_TERM_PAT = re.compile(
    r'(?:"([^"]{2,60})"|\'([^\'{2,60}]*)\'|([A-Z][a-z][A-Za-z\s]{1,50}))\s+(?:means?|includes?|shall\s+mean|shall\s+include)\b',
    re.IGNORECASE,
)


_BENCH_FROM_BODY_PAT = re.compile(
    r"HON['']BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z][A-Za-z\.\s]+?)(?:\s*,?\s*J\.|\n|$)",
    re.IGNORECASE,
)
_CORAM_PAT = re.compile(
    r"(?:CORAM|BEFORE)\s*:\s*(.+?)(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Signature lines at end of judgment: "……………J." followed by "(Judge Name)"
_JUDGE_SIG_PAT = re.compile(
    r"[.\u2026]+\s*J\.\s*\n\s*\(([A-Z][A-Za-z\.\s,]+?)\)",
    re.MULTILINE,
)


def _try_extract_bench_from_body(body_lines: list, result: dict) -> None:
    """
    Attempt to extract judge names from early body lines when the header has
    no explicit Author:/Bench: section (common in 2025+ format files).

    Modifies result["bench"] and result["author"] in-place if names are found.
    Only runs when bench is currently empty.
    """
    if result["bench"]:
        return  # already parsed from header
    snippet = "\n".join(body_lines)
    # Try HON'BLE JUSTICE patterns
    names = []
    seen = set()
    for m in _BENCH_FROM_BODY_PAT.finditer(snippet):
        name = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",. ")
        if name and name not in seen and 3 < len(name) < 60:
            seen.add(name)
            names.append(name)
    if names:
        result["bench"] = names
        if not result["author"]:
            result["author"] = names[0]
        return
    # Try CORAM: block
    cm = _CORAM_PAT.search(snippet[:1500])
    if cm:
        raw_coram = cm.group(1).strip()
        parts = re.split(r"[\n,]+", raw_coram)
        for part in parts:
            part = part.strip().rstrip(".").strip()
            if 3 < len(part) < 60 and part not in seen:
                seen.add(part)
                names.append(part)
        if names:
            result["bench"] = names[:5]
            if not result["author"]:
                result["author"] = names[0]


def _extract_judges_from_signatures(all_lines: list) -> list:
    """
    Extract judge names from signature blocks at the bottom of a judgment.
    Handles the common format:
        ………………………………………J.
                                 (J.B. Pardiwala)
    Returns a list of judge names, or [] if none found.
    """
    # Scan last 100 lines for signature pattern
    tail_text = "\n".join(all_lines[-100:])
    names = []
    seen = set()
    for m in _JUDGE_SIG_PAT.finditer(tail_text):
        name = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",. ")
        if name and name not in seen and 3 < len(name) < 60:
            seen.add(name)
            names.append(name)
    return names


def _parse_txt_header(lines: list) -> dict:
    """
    Parse the structured preamble of a scraped .txt case file.

    Returns a dict with:
      cites_count, cited_by_count, court, case_name, date_of_judgment,
      year, author, bench (list), neutral_citation, equivalent_citations (list),
      reportable (bool), body_start_index (line index where body begins).
    """
    result = {
        "cites_count":          None,
        "cited_by_count":       None,
        "court":                "Supreme Court of India",
        "case_name":            None,
        "date_of_judgment":     None,
        "year":                 None,
        "author":               None,
        "bench":                [],
        "neutral_citation":     None,
        "equivalent_citations": [],
        "reportable":           None,
        "body_start_index":     0,
    }
    if not lines:
        return result

    i = 0
    # ── [Cites N, Cited by M] block ──────────────────────────────────────────
    if i < len(lines) and lines[i].strip() == "[Cites":
        i += 1
        try:
            result["cites_count"] = int(lines[i].strip())
        except (ValueError, IndexError):
            pass
        i += 1  # skip ", Cited by"
        i += 1
        try:
            result["cited_by_count"] = int(lines[i].strip())
        except (ValueError, IndexError):
            pass
        i += 1  # skip "]"
        i += 1

    # ── Court line ───────────────────────────────────────────────────────────
    if i < len(lines) and "Supreme Court" in lines[i]:
        result["court"] = "Supreme Court of India"
        i += 1

    # ── Case name + date line ─────────────────────────────────────────────────
    if i < len(lines):
        line7 = lines[i].strip()
        i += 1
        dm = _TXT_DATE_LINE_PAT.search(line7)
        if dm:
            day, month, year = dm.group(1), dm.group(2), dm.group(3)
            result["date_of_judgment"] = f"{int(day)} {month.capitalize()} {year}"
            result["year"]             = year
            # case_name = everything before " on DD Month, YYYY"
            case_name = line7[:dm.start()].strip().rstrip("on").strip()
        else:
            case_name = line7
        result["case_name"] = case_name

    # ── Optional: Equivalent citations line (older files) ────────────────────
    if i < len(lines):
        eq_m = _TXT_EQUIV_PAT.match(lines[i].strip())
        if eq_m:
            result["equivalent_citations"] = [c.strip() for c in eq_m.group(1).split(",") if c.strip()]
            i += 1

    # ── 2024/2025+ format: neutral citation on the line immediately after the case
    #    name line (e.g. "2025 INSC 73   REPORTABLE").  These files have no
    #    separate Author:/Bench: lines — the body starts right after this line.
    if i < len(lines):
        line8 = lines[i].strip()
        insc_m8 = _TXT_INSC_PAT.search(line8)
        if (
            insc_m8
            and not _TXT_EQUIV_PAT.match(line8)
            and not line8.startswith("Author:")
            and not line8.startswith("Bench:")
        ):
            result["neutral_citation"] = f"{insc_m8.group(1)} INSC {insc_m8.group(2)}"
            if "NON-REPORTABLE" in line8.upper() or "NON\u00adREPORTABLE" in line8.upper():
                result["reportable"] = False
            elif "REPORTABLE" in line8.upper():
                result["reportable"] = True
            i += 1
            # No Author/Bench lines follow — skip to body start
            while i < len(lines) and not lines[i].strip():
                i += 1
            result["body_start_index"] = i
            # Try to extract bench from early body lines (CORAM / HON'BLE patterns)
            _try_extract_bench_from_body(lines[i:i + 60], result)
            if not result["author"] and result["bench"]:
                result["author"] = result["bench"][0]
            return result

    # ── Author ────────────────────────────────────────────────────────────────
    if i < len(lines) and lines[i].strip().startswith("Author:"):
        i += 1
        if i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("Bench:"):
            result["author"] = lines[i].strip()
            i += 1

    # ── Bench (multi-line, comma-separated) ───────────────────────────────────
    if i < len(lines) and lines[i].strip().startswith("Bench:"):
        i += 1
        seen = set()
        while i < len(lines):
            raw = lines[i].strip()
            # Stop conditions: INSC citation, REPORTABLE marker
            if _TXT_INSC_PAT.search(raw):
                insc_m = _TXT_INSC_PAT.search(raw)
                result["neutral_citation"] = f"{insc_m.group(1)} INSC {insc_m.group(2)}"
                if "NON-REPORTABLE" in raw.upper() or "NON\u00adREPORTABLE" in raw.upper():
                    result["reportable"] = False
                elif "REPORTABLE" in raw.upper():
                    result["reportable"] = True
                i += 1
                break
            if raw.upper() in ("REPORTABLE", "NON-REPORTABLE", "NON\u00adREPORTABLE"):
                result["reportable"] = "NON" not in raw.upper()
                i += 1
                break
            if raw == ",":
                i += 1
                continue
            # Stop if line looks like body start (IN THE SUPREME COURT / numbered para)
            if raw.startswith("IN THE SUPREME COURT") or re.match(r"^\d+\.", raw):
                break
            # Stop if line looks like a case number (e.g. "Civil Appeal Nos. 2769-2770 of 2023")
            if re.search(r"\b(Appeal|Petition|Application|Suit)\b.*\bof\s+\d{4}\b", raw, re.I):
                break
            # Stop at old-format structured fields (2004-2010 style)
            if re.match(r"^CASE\s+NO\.?:?", raw, re.I) or re.match(
                r"^(?:PETITIONER|RESPONDENT|DATE\s+OF\s+JUDGMENT)\s*:?", raw, re.I
            ):
                break
            if raw and raw != "Author:" and not raw.startswith("Bench:"):
                name = raw.rstrip(",").strip()
                if name and name not in seen and len(name) > 3:
                    seen.add(name)
                    result["bench"].append(name)
            i += 1

    # If author not set yet, derive from bench[0]
    if not result["author"] and result["bench"]:
        result["author"] = result["bench"][0]

    # Skip any remaining blank/header lines before body
    while i < len(lines) and not lines[i].strip():
        i += 1
    result["body_start_index"] = i
    return result


def _clean_txt_body(text: str) -> str:
    """
    Strip digital signature noise, page number artifacts, and excessive
    whitespace from a scraped .txt case law body.
    """
    # 1. Digital signature blocks
    for pat in _TXT_DIG_SIG_PATS:
        text = pat.sub("", text)
    # 2. Standalone page numbers / "Page N of M"
    text = re.sub(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # 3. Repeated "IN THE SUPREME COURT" header blocks mid-text
    text = re.sub(
        r"(?m)^[ \t]*IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA\s*$",
        "",
        text,
    )
    # 4. Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_txt_case_number(body: str) -> tuple:
    """Return (case_number, case_number_regex) from body text."""
    return _extract_case_number(body[:3000])


def _extract_txt_parties(case_name: str) -> tuple:
    """Split 'Party1 vs Party2' into (petitioner, respondent)."""
    if not case_name:
        return None, None
    for sep in (" vs ", " Vs ", " VS ", " versus ", " Versus "):
        if sep in case_name:
            parts = case_name.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return case_name.strip(), None


def _extract_txt_outcome(body: str) -> str | None:
    """Scan last 4000 chars for disposal language; fall back to full body."""
    _OUT_PASSIVE = re.compile(
        r"\b(?:appeal|petition|application|revision|suit|writ|case)\s+(?:is\s+|are\s+|stands?\s+|be\s+)?"
        r"(allowed|dismissed|disposed\s+of|partly\s+allowed|allowed\s+in\s+part"
        r"|discharged|quashed|set\s+aside|upheld|confirmed|partly\s+dismissed)",
        re.I,
    )
    _OUT_ACTIVE = re.compile(
        r"\bwe\s+(?:hereby\s+)?(allow|dismiss|dispose\s+of|partly\s+allow|quash|set\s+aside|uphold|confirm)\b",
        re.I,
    )
    _OUT_FAIL = re.compile(
        r"\b(?:appeals?|petitions?|applications?|writ)\s+(?:fail(?:s)?\s+and\s+are|stands?\s+)"
        r"(?:hereby\s+)?(dismissed|allowed|disposed\s+of)\b"
        r"|\bare\s+hereby\s+(dismissed|allowed|disposed\s+of)\b",
        re.I,
    )
    _OUT_RESULT = re.compile(
        r"\b(allowed|dismissed|disposed\s+of|partly\s+allowed|allowed\s+in\s+part"
        r"|quashed|set\s+aside|upheld)\s+accordingly\b",
        re.I,
    )
    _NORMALIZE = {
        "dispose of": "disposed of",
        "allow": "allowed",
        "dismiss": "dismissed",
        "partly allow": "partly allowed",
        "quash": "quashed",
        "set aside": "set aside",
        "uphold": "upheld",
        "confirm": "upheld",
        "set_aside": "set aside",
    }
    def _best_group(m):
        """Return the first non-None group from a match (handles alternation)."""
        for g in m.groups():
            if g is not None:
                return g
        return m.group(0)

    tail = body[-4000:]
    for pat in (_OUT_PASSIVE, _OUT_ACTIVE, _OUT_FAIL, _OUT_RESULT):
        m = pat.search(tail)
        if m:
            raw = _best_group(m).lower().strip()
            raw = re.sub(r"\s+", " ", raw)
            return _NORMALIZE.get(raw, raw)
    # Wider scan — full body (catches cases where order is embedded mid-text)
    for pat in (_OUT_PASSIVE, _OUT_ACTIVE, _OUT_FAIL, _OUT_RESULT):
        last_m = None
        for last_m in pat.finditer(body):
            pass
        if last_m:
            raw = _best_group(last_m).lower().strip()
            raw = re.sub(r"\s+", " ", raw)
            return _NORMALIZE.get(raw, raw)
    return None


def process_txt_case(filepath: str) -> dict:
    """
    Parse a scraped .txt case law file and return a structured JSON dict
    matching the existing case law schema (plus new txt-only fields).

    New fields: cites_count, cited_by_count, neutral_citation,
                equivalent_citations, jurisdiction_type, source_file, source_type.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as e:
        raise RuntimeError(f"Cannot open {filepath}: {e}") from e

    all_lines = raw.splitlines()

    # ── Parse header ──────────────────────────────────────────────────────────
    hdr = _parse_txt_header(all_lines)

    # ── Extract body ──────────────────────────────────────────────────────────
    body_lines = all_lines[hdr["body_start_index"]:]
    body_raw   = "\n".join(body_lines)
    body_clean = _clean_txt_body(body_raw)

    # ── Jurisdiction type ─────────────────────────────────────────────────────
    jm = _TXT_JURIS_PAT.search(body_clean[:2000])
    jurisdiction_type = jm.group(0).upper() if jm else None

    # ── Case number ───────────────────────────────────────────────────────────
    case_number, case_number_regex = _extract_txt_case_number(body_clean)

    # ── Parties ───────────────────────────────────────────────────────────────
    petitioner, respondent = _extract_txt_parties(hdr["case_name"] or "")

    # ── Year from date (fallback: filename) ──────────────────────────────────
    year = hdr["year"]
    if not year:
        fname = os.path.basename(filepath)
        ym = re.match(r"^(\d{4})_", fname)
        if ym:
            year = ym.group(1)

    court = hdr["court"] or "Supreme Court of India"

    # ── case_id: try case number first, fallback to YYYY_MON_N ───────────────
    fname_stem = os.path.splitext(os.path.basename(filepath))[0]   # e.g. 2024_JAN_5
    if case_number:
        case_id = _build_case_id(court, year or "", hdr["case_name"], fname_stem)
    else:
        safe_stem = _sanitize_id_part(fname_stem)
        case_id   = f"SC_{year or 'UNKNOWN'}_{safe_stem}"

    # ── Bench type ────────────────────────────────────────────────────────────
    bench_size  = len(hdr["bench"])
    if bench_size == 1:
        bench_type = "single"
    elif bench_size == 2:
        bench_type = "division"
    elif bench_size >= 3:
        bench_type = "constitution"
    else:
        bench_type = None

    # ── Paragraphs (reuse existing splitter — pass body as single "page") ─────
    paragraphs = build_paragraph_chunks(case_id, [body_clean])

    # ── Reporter citations (from body) ────────────────────────────────────────
    reporter_citations = _extract_reporter_citations(body_clean[:5000])
    if hdr["neutral_citation"] and hdr["neutral_citation"] not in reporter_citations:
        reporter_citations.insert(0, hdr["neutral_citation"])

    # ── Outcome ───────────────────────────────────────────────────────────────
    outcome = _extract_txt_outcome(body_clean)

    # ── Disposition — normalized outcome enum ────────────────────────────────
    _DISP_MAP = {
        "allowed":          "allowed",
        "dismissed":        "dismissed",
        "disposed of":      "disposed",
        "disposed":         "disposed",
        "partly allowed":   "partly_allowed",
        "allowed in part":  "partly_allowed",
        "quashed":          "quashed",
        "set aside":        "set_aside",
        "upheld":           "upheld",
        "discharged":       "discharged",
        "confirmed":        "upheld",
    }
    disposition = _DISP_MAP.get((outcome or "").lower().strip()) if outcome else None

    # ── Bench fallback: signature lines at end of file (2025+ format) ────────
    judges = hdr["bench"]
    author = hdr["author"]
    if not judges:
        sig_judges = _extract_judges_from_signatures(all_lines)
        if sig_judges:
            judges = sig_judges
            author = author or sig_judges[0]
    # Recompute bench_type based on final judge list
    bench_size = len(judges)
    if bench_size == 1:
        bench_type = "single"
    elif bench_size == 2:
        bench_type = "division"
    elif bench_size >= 3:
        bench_type = "constitution"
    else:
        bench_type = None

    # ── Assemble ──────────────────────────────────────────────────────────────
    # Source file stored as relative path from CASELAWS_INPUT_DIR
    try:
        rel_src = os.path.relpath(filepath, CASELAWS_INPUT_DIR)
    except ValueError:
        rel_src = os.path.basename(filepath)

    data = {
        "case_id":               case_id,
        "case_name":             hdr["case_name"],
        "normalized_case_name":  re.sub(r"\s+", "_", (hdr["case_name"] or "").upper()) or fname_stem,
        "court":                 court,
        "year":                  year,
        "judgment_template":     None,
        "case_number":           case_number,
        "case_number_regex":     case_number_regex,
        "date_of_judgment":      hdr["date_of_judgment"],
        "judges":                judges,
        "author":                author,
        "bench_type":            bench_type,
        "petitioner":            petitioner,
        "respondent":            respondent,
        "reporter_citations":    reporter_citations,
        "outcome":               outcome,
        "disposition":           disposition,   # normalized enum (allowed/dismissed/disposed/etc.)
        # New .txt-only fields
        "cites_count":           hdr["cites_count"],
        "cited_by_count":        hdr["cited_by_count"],
        "neutral_citation":      hdr["neutral_citation"],
        "equivalent_citations":  hdr["equivalent_citations"],
        "jurisdiction_type":     jurisdiction_type,
        "reportable":            hdr["reportable"],
        "source_file":           rel_src,
        "source_type":           "txt_scraped",
        "paragraphs":            [],   # filled below after annotation
    }

    data["paragraphs"] = paragraphs
    data = _split_long_paragraphs(data, limit=_CHUNK_LIMIT)
    data = _annotate_paragraphs(data)
    data = _normalize_chunk_ids(data)
    data["case_summary"] = build_case_summary(data)
    return data


def _txt_sort_key(filepath: str) -> tuple:
    """Return (year, month_int, file_index) for YYYY_MON_N.txt path."""
    fname = os.path.basename(filepath)
    m = re.match(r"^(\d{4})_([A-Z]{3})_(\d+)\.txt$", fname, re.IGNORECASE)
    if m:
        y   = int(m.group(1))
        mon = _MONTH_ORDER.get(m.group(2).upper(), 0)
        idx = int(m.group(3))
        return (y, mon, idx)
    return (9999, 99, 99999)


def _in_date_range(filepath: str, start_year: int, start_month: str,
                   end_year: int, end_month: str) -> bool:
    """Return True if this file falls within [start_year/start_month, end_year/end_month]."""
    key = _txt_sort_key(filepath)
    if key[0] == 9999:
        return True   # unrecognised filename — include anyway
    s_mon = _MONTH_ORDER.get(start_month.upper(), 1)
    e_mon = _MONTH_ORDER.get(end_month.upper(), 12)
    start_key = (start_year, s_mon, 0)
    end_key   = (end_year,   e_mon, 999999)
    return start_key <= key <= end_key


def run_extraction_txt_caselaws(
    start_year:  int        = None,
    start_month: str        = None,
    end_year:    int        = None,
    end_month:   str        = None,
    folders:     list       = None,
    order:       str        = None,
    workers:     int        = None,
) -> None:
    """
    Walk CaseLaws/YYYY/MON/*.txt, parse each file, write JSON to
    json_output/caselaws/YYYY/MON/YYYY_MON_N.json.

    SAFE TO RUN ANY TIME — existing JSON files are NEVER overwritten.
    Only .txt files that do not yet have a matching JSON output are processed.
    To reprocess a file, delete its JSON manually first.

    TWO WAYS TO SPECIFY WHICH FILES TO PROCESS
    -------------------------------------------
    Option A — folder list (preferred for incremental additions):
        folders=["2025/MAR", "2025/APR"]
        Pass the YYYY/MON folder names you just added.  The date-range
        parameters are ignored when folders is supplied.

        Example (programmatic):
            run_extraction_txt_caselaws(folders=["2025/MAR", "2025/APR"])

        Example (from __main__ / command line):
            Set NEW_FOLDERS at the top of this file (see below).

    Option B — date range (for initial bulk load or a range of months):
        start_year=2024, start_month="JAN", end_year=2024, end_month="JUN"
        Defaults to the BATCH_* constants at the top of this file.

    Logs errors to json_output/caselaws/txt_extraction_errors.log.
    """
    ord_ = (order   or BATCH_ORDER).lower()
    w    = workers if workers is not None else BATCH_WORKERS

    os.makedirs(CASE_OUTPUT_DIR, exist_ok=True)
    err_log = os.path.join(CASE_OUTPUT_DIR, "txt_extraction_errors.log")

    # ── Build the list of candidate input directories ─────────────────────────
    if folders:
        # Option A: explicit folder list — e.g. ["2025/MAR", "2025/APR"]
        scan_dirs = []
        for f in folders:
            # Accept both "2025/MAR" and "2025\\MAR" and "2025/mar"
            f_norm = f.replace("\\", "/").upper()
            full   = os.path.join(CASELAWS_INPUT_DIR, *f_norm.split("/"))
            if os.path.isdir(full):
                scan_dirs.append(full)
            else:
                print(f"  ⚠ Folder not found, skipping: {full}")
        label = "folders=[" + ", ".join(folders) + "]"
    else:
        # Option B: date range
        sy  = start_year  if start_year  is not None else BATCH_START_YEAR
        sm  = start_month if start_month is not None else BATCH_START_MONTH
        ey  = end_year    if end_year    is not None else BATCH_END_YEAR
        em  = end_month   if end_month   is not None else BATCH_END_MONTH
        scan_dirs = [CASELAWS_INPUT_DIR]   # walk full tree, filter by range
        label = f"range={sy}/{sm}–{ey}/{em}"

    # ── Collect all .txt files from the scan directories ─────────────────────
    all_txt = []
    for scan_dir in scan_dirs:
        for root, dirs, files in os.walk(scan_dir):
            dirs.sort()
            for fname in sorted(files):
                if not fname.lower().endswith(".txt"):
                    continue
                fpath = os.path.join(root, fname)
                if folders:
                    all_txt.append(fpath)          # folders mode: include all
                elif _in_date_range(fpath, sy, sm, ey, em):
                    all_txt.append(fpath)          # range mode: filter by date

    all_txt.sort(key=_txt_sort_key, reverse=(ord_ == "desc"))

    if not all_txt:
        print(f"[TXT extractor] No .txt files found for {label}")
        return

    # ── Skip files that already have JSON output (NEVER overwrite) ───────────
    pending = []
    for fp in all_txt:
        fname     = os.path.basename(fp)           # 2024_JAN_5.txt
        stem      = os.path.splitext(fname)[0]     # 2024_JAN_5
        # Mirror the exact YYYY/MON folder structure under CASE_OUTPUT_DIR
        rel_parts = os.path.relpath(fp, CASELAWS_INPUT_DIR).split(os.sep)
        # rel_parts example: ['2024', 'JAN', '2024_JAN_5.txt']
        if len(rel_parts) >= 3:
            out_dir = os.path.join(CASE_OUTPUT_DIR, rel_parts[0], rel_parts[1])
        elif len(rel_parts) >= 2:
            out_dir = os.path.join(CASE_OUTPUT_DIR, rel_parts[0])
        else:
            out_dir = CASE_OUTPUT_DIR
        out_path = os.path.join(out_dir, stem + ".json")
        if not os.path.exists(out_path):
            pending.append((fp, out_dir, out_path))
        # If out_path already exists → silently skip (existing JSON untouched)

    skipped = len(all_txt) - len(pending)
    print(f"[TXT extractor] {label} | order={ord_} | "
          f"found={len(all_txt)} | already done={skipped} | to process={len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    errors = []

    def _process_one(item):
        fp, out_dir, out_path = item
        try:
            data = process_txt_case(fp)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return (fp, None)
        except Exception as exc:
            return (fp, str(exc))

    if w <= 1:
        # Single-threaded (easier to debug)
        for item in tqdm(pending, desc="Extracting .txt"):
            fp, err = _process_one(item)
            if err:
                errors.append((fp, err))
    else:
        with ThreadPoolExecutor(max_workers=w) as pool:
            futs = {pool.submit(_process_one, item): item for item in pending}
            for fut in tqdm(as_completed(futs), total=len(pending), desc="Extracting .txt"):
                fp, err = fut.result()
                if err:
                    errors.append((fp, err))

    if errors:
        with open(err_log, "a", encoding="utf-8") as ef:
            for fp, err in errors:
                ef.write(f"{fp}\t{err}\n")
        print(f"  \u26a0 {len(errors)} errors logged to {err_log}")
    else:
        print("  \u2713 All files processed successfully")

    _build_cases_and_statutes_index()
    print("Extraction complete. Run build_indexes.py (or run_build_indexes_batch) for FAISS/BM25.")



# Match run_sample_pipeline._split_text_to_limit (sentence-boundary split)
_CHUNK_LIMIT = 1200


def _split_text_to_limit(text: str, limit: int = None) -> list:
    """Split long paragraph text into chunks at sentence boundaries (<= limit chars)."""
    limit = limit or _CHUNK_LIMIT
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\(])", text)
    result = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + 1 + len(sent) > limit:
            result.append(buf.strip())
            buf = sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf.strip():
        result.append(buf.strip())
    # Hard-split any chunk still over limit
    final = []
    for chunk in result:
        if len(chunk) <= limit:
            final.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = min(start + limit, len(chunk))
                if end < len(chunk):
                    cut = chunk.rfind(" ", start, end)
                    if cut == -1 or cut <= start:
                        cut = end
                else:
                    cut = end
                final.append(chunk[start:cut].strip())
                start = cut
    return final


# ── Paragraph type classification patterns (ported from Ingestion/smart_chunker.py) ──
_PARA_TYPE_ORDER = re.compile(
    r"\bin\s+the\s+result\b"
    r"|\bappeal\s+is\s+(?:accordingly\s+)?(?:allowed|dismissed)\b"
    r"|\bpetition\s+is\s+(?:allowed|dismissed)\b"
    r"|\bwrit\s+petition\s+is\s+(?:allowed|dismissed)\b"
    r"|\bordered\s+that\b"
    r"|\baccordingly\s+(?:allowed|dismissed|disposed\s+of)\b"
    r"|\bremanded\s+(?:back\s+)?to\b"
    r"|\binterim\s+(?:stay|relief|order)\b"
    r"|\bdirected\s+to\s+pay\b"
    r"|\bcosts?\s+of\s+rupees?\b"
    r"|\bno\s+order\s+as\s+to\s+costs?\b"
    r"|\bthe\s+above\s+(?:appeal|petition|application)\s+is\b"
    r"|\bdisposed\s+of\s+(?:accordingly|as\s+above)\b",
    re.I,
)
_PARA_TYPE_RATIO = re.compile(
    r"\bwe\s+hold\b|\bit\s+is\s+held\b|\bthis\s+court\s+holds?\b"
    r"|\bthe\s+law\s+(?:is|has\s+been)\s+(?:well\s+)?settled\b"
    r"|\bthe\s+(?:core|crux|question)\s+(?:issue\s+|that\s+falls\s+for\s+consideration)?"
    r"|\bthe\s+question\s+(?:that\s+)?(?:falls|arises)\s+for\s+(?:consideration|decision)\b"
    r"|\bprincipal\s+question\s+of\s+law\b"
    r"|\bsubstantial\s+question\s+of\s+law\b"
    r"|\blegal\s+proposition\b|\bthe\s+ratio\s+(?:decidendi)?\b"
    r"|\bprecedent\s+(?:has\s+been\s+|is\s+)(?:set|established|followed|overruled)\b"
    r"|\boverruled\b|\bdistinguished\b|\breferred\s+to\s+a\s+(?:larger|constitution)\s+bench\b",
    re.I,
)
_PARA_TYPE_REASONING = re.compile(
    r"\bfor\s+the\s+(?:aforesaid|foregoing)\s+reasons\b"
    r"|\bwe\s+find\b|\bwe\s+agree\b|\bwe\s+are\s+(?:not\s+)?(?:satisfied|persuaded|convinced)\b"
    r"|\bwe\s+are\s+of\s+the\s+view\b|\bwe\s+are\s+of\s+(?:the\s+)?opinion\b"
    r"|\bin\s+our\s+(?:considered\s+)?(?:view|opinion|judgment)\b"
    r"|\bit\s+(?:appears?|seems?|is\s+(?:clear|evident|apparent))\b"
    r"|\bfrom\s+the\s+(?:record|evidence|material|perusal)\b"
    r"|\bthe\s+evidence\s+(?:shows?|establishes?|demonstrates?|discloses?)\b"
    r"|\bin\s+the\s+light\s+of\b|\bin\s+view\s+of\s+the\s+above\b"
    r"|\bconsidering\s+(?:the\s+)?(?:facts?|evidence|materials?|submissions?)\b"
    r"|\bupon\s+consideration\b|\bon\s+(?:a\s+)?(?:careful\s+)?(?:perusal|examination|reading)\b"
    r"|\bour\s+(?:analysis|examination|consideration|reasoning)\b"
    r"|\bthus\s+(?:the\s+)?(?:court|we|this)\b|\bhaving\s+regard\s+to\b"
    r"|\bno\s+merit\b|\bwithout\s+merit\b|\bmeritorious\b",
    re.I,
)
_PARA_TYPE_ARGUMENTS = re.compile(
    r"\blearned\s+(?:counsel|senior\s+counsel|advocate|solicitor|attorney)\b"
    r"|\bsubmitted\s+that\b|\bcontended\s+that\b|\bit\s+was\s+(?:argued|submitted|urged|contended)\b"
    r"|\bvehemently\s+(?:argued|opposed|objected)\b"
    r"|\bfor\s+the\s+(?:appellant|respondent|petitioner|defendant|plaintiff|state)\b"
    r"|\bstrenuously\s+argued\b|\bforcefully\s+argued\b"
    r"|\bappearing\s+for\s+the\b|\bon\s+behalf\s+of\s+the\b"
    r"|\brelied\s+(?:upon|on)\b|\bplaced\s+reliance\s+(?:upon|on)\b"
    r"|\bopposed\s+the\s+(?:prayer|petition|appeal|writ)\b"
    r"|\bper\s+(?:contra|contra)\b",
    re.I,
)
_PARA_TYPE_FACTS = re.compile(
    r"\bthe\s+(?:facts?\s+(?:of\s+the\s+case|are|(?:in\s+)?brief))\b"
    r"|\bbriefly\s+(?:stated|put)\b|\bshort\s+facts\b"
    r"|\bthe\s+case\s+of\s+the\s+(?:petitioner|appellant|complainant|prosecution)\b"
    r"|\bthe\s+prosecution\s+case\b|\bthe\s+plaintiff\b|\bthe\s+defendant\b"
    r"|\bfactual\s+(?:matrix|background|position)\b"
    r"|\bFIR\s+(?:was\s+)?(?:lodged|filed|registered)\b"
    r"|\baccused\s+(?:was|were)\s+(?:arrested|charged|convicted)\b"
    r"|\bappellant\s+(?:was|has\s+been)\s+(?:convicted|sentenced|acquitted)\b"
    r"|\bthe\s+(?:trial\s+)?(?:court|sessions\s+court)\s+(?:had\s+)?(?:convicted|acquitted|dismissed)\b"
    r"|\bpetitioner\s+(?:is|was)\s+(?:a\s+\w+\s+)?(?:aggrieved|seeking|employed|dismissed|terminated|denied|suspended)\b"
    r"|\bpetitioner\s+was\s+(?:a\s+)?\w"
    r"|\bpermission\s+was\s+(?:granted|refused|denied)\b"
    r"|\bthe\s+(?:present|instant)\s+(?:appeal|petition|case)\s+(?:arises?|is)\b"
    r"|\bwas\s+(?:dismissed|terminated|removed)\s+from\s+(?:service|employment|office)\b"
    r"|\bwithout\s+(?:any\s+)?(?:notice|show\s+cause|hearing|opportunity)\b",
    re.I,
)


def _classify_paragraph_type(para_text: str, para_label: str, para_index: int, total_paras: int) -> str:
    """
    Heuristic classifier: facts | arguments | reasoning | ratio | order | unknown.
    Strengthened version with ported patterns from Ingestion/smart_chunker.py.
    Uses both regex signals and positional heuristics.
    """
    if not (para_text or "").strip():
        return "unknown"
    text = para_text[:3000]  # extended sample window (was 2000)

    # Positional signals: last 10% of document → likely order/ratio
    is_early = total_paras and para_index <= max(3, total_paras // 5)
    is_late  = total_paras and para_index >= int(total_paras * 0.85)

    # Strongest signal wins — try in priority order
    if _PARA_TYPE_ORDER.search(text):
        return "order"
    if _PARA_TYPE_RATIO.search(text):
        return "ratio"
    if _PARA_TYPE_REASONING.search(text):
        return "reasoning"
    if _PARA_TYPE_ARGUMENTS.search(text):
        return "arguments"
    if _PARA_TYPE_FACTS.search(text) or (is_early and para_label not in ("preamble", "0", "1")):
        return "facts"
    # Late paragraphs without strong signal → probably order/reasoning
    if is_late:
        return "reasoning"
    return "unknown"


# Section citation patterns (clean act+section only)
_SECTION_IPC_FULL = re.compile(r"Section\s+(\d+[A-Za-z]?)\s+of\s+(?:the\s+)?Indian\s+Penal\s+Code", re.IGNORECASE)
_SECTION_ARMS_FULL = re.compile(r"Section\s+(\d+[A-Za-z]?)\s+of\s+(?:the\s+)?Arms\s+Act", re.IGNORECASE)
_SECTION_ACT_AFTER = re.compile(r"Section\s+(\d+[A-Za-z]?)\s+(IPC|BNS|CrPC|CPC|TPA|BNSS|IEA|Evidence\s+Act|Contract\s+Act|NI\s+Act|SRA|HMA|Arms\s+Act|MV\s+Act)\b", re.IGNORECASE)
_ACT_SECTION_BEFORE = re.compile(r"\b(IPC|BNS|CrPC|CPC|TPA|BNSS|IEA|Arms\s+Act|MV\s+Act)\s+Section\s+(\d+[A-Za-z]?)", re.IGNORECASE)
_ARTICLE_CONSTITUTION = re.compile(r"Article\s+(\d+[A-Za-z]?)\s+(?:of\s+)?(?:the\s+)?Constitution", re.IGNORECASE)


def _normalize_act_name(act: str) -> str:
    a = act.strip().replace(" ", "")
    if not a:
        return act.strip()
    if a.upper() in ("IPC", "BNS", "CRPC", "CPC", "TPA", "BNSS", "IEA"):
        return a.upper() if a.upper() != "CRPC" else "CrPC"
    if "Arms" in act or a.upper() == "ARMSACT":
        return "Arms Act"
    if "Evidence" in act:
        return "Evidence Act"
    if "Contract" in act:
        return "Contract Act"
    return act.strip()


def _extract_sections_cited(text: str) -> list:
    if not (text or "").strip():
        return []
    seen = set()
    result = []

    def add(act: str, sec: str):
        act = _normalize_act_name(act)
        key = f"{act} {sec}".strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)

    for m in _SECTION_IPC_FULL.finditer(text):
        add("IPC", m.group(1))
    for m in _SECTION_ARMS_FULL.finditer(text):
        add("Arms Act", m.group(1))
    for m in _SECTION_ACT_AFTER.finditer(text):
        add(m.group(2), m.group(1))
    for m in _ACT_SECTION_BEFORE.finditer(text):
        add(m.group(1), m.group(2))
    for m in _ARTICLE_CONSTITUTION.finditer(text):
        add("Constitution", m.group(1))
    return result[:30]


_CITED_FRAGMENT_MARKERS = re.compile(r"\b(supra|infra|ibid\.?|hereinafter|aforementioned)\b", re.I)


# Words that indicate the end of a party name (sentence verbs / connectors)
_PARTY_TRAILING_STOP = re.compile(
    r"\s+(?:held|wherein|where|said|found|stated|decided|observed|noted|"
    r"ruled|opined|directed|ordered|dismissed|allowed|denied|rejected|"
    r"that\b|therein\b|thereof\b|thereto\b|thus\b|hence\b|therefore\b)\b.*$",
    re.I,
)


def _clean_party_name(s: str) -> str:
    """Trim trailing verb words from a captured party name."""
    s = _PARTY_TRAILING_STOP.sub("", s).strip().strip(".,; ")
    return s


def _extract_cited_cases(text: str) -> list:
    """
    Extract cited case names from paragraph text.
    Normalizes to 'Party A v Party B' form for citation graph matching.
    Rejects noise: OCR garbage, narrative fragments, case-number-only strings.
    """
    if not (text or "").strip():
        return []
    seen = set()
    result = []

    # Narrative / garbage words in party names indicate false positives
    _NARRATIVE = frozenset({
        "near", "saw", "undergo", "years", "year", "month", "days", "stated",
        "according", "therefore", "however", "wherein", "whereas", "hence",
        "thus", "thereafter", "then", "after", "before", "when", "while",
        "because", "although", "though", "said", "told", "asked", "replied",
    })

    def _add(party_a: str, party_b: str):
        a = _clean_party_name(party_a.strip())
        b = _clean_party_name(party_b.strip())
        # Basic sanity: each side 3–70 chars, 1–6 words
        if not a or not b or len(a) < 3 or len(b) < 3:
            return
        if len(a) > 70 or len(b) > 70:
            return
        words_a = a.split()
        words_b = b.split()
        if len(words_a) > 6 or len(words_b) > 6:
            return
        # Reject narrative words
        all_words = set(w.lower() for w in words_a + words_b)
        if all_words & _NARRATIVE:
            return
        # Reject reporter tokens
        joined = f"{a} {b}"
        if any(tok in joined for tok in ("SCR", "SCC", "AIR", "INSC", "MANU", "ILR")):
            return
        if _CITED_FRAGMENT_MARKERS.search(joined):
            return
        name = f"{a} v {b}"
        if name not in seen:
            seen.add(name)
            result.append(name)

    # Pattern: "X v Y" / "X versus Y" / "X vs. Y"
    # Use lazy G1 and stop G2 at sentence-end characters
    for m in re.finditer(
        r"([A-Z][A-Za-z0-9\s\.\,&']{3,50}?)\s+v(?:s\.?|ersus)?\s+([A-Z][A-Za-z0-9\s\.\,&']{3,50})",
        text,
    ):
        _add(m.group(1), m.group(2))

    return result[:25]


def _split_long_paragraphs(data: dict, limit: int = 1800) -> dict:
    paragraphs = data.get("paragraphs", []) or []
    new_paragraphs = []
    for p in paragraphs:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= limit:
            new_paragraphs.append(p)
            continue
        chunks = _split_text_to_limit(text, limit=limit)
        for chunk_text in chunks:
            q = dict(p)
            q["text"] = chunk_text
            new_paragraphs.append(q)
    data["paragraphs"] = new_paragraphs
    return data


def _annotate_paragraphs(data: dict) -> dict:
    paragraphs = data.get("paragraphs", []) or []
    total_paras = len(paragraphs)
    for idx, p in enumerate(paragraphs, start=1):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        cleaned = _clean_ocr_noise(text)
        p["text"] = cleaned.strip()
        para_label = str(p.get("paragraph_id") or idx)
        p["paragraph_type"] = _classify_paragraph_type(cleaned, para_label, idx, total_paras)
        p["sections_cited"] = _extract_sections_cited(cleaned)
        p["cited_cases"] = _extract_cited_cases(cleaned)
    return data


def build_case_summary(data: dict) -> dict:
    """
    Build structured case summary from paragraphs for semantic layers / case-level embedding.
    Aggregates facts, legal issues (arguments), ratio, and order by paragraph_type.
    """
    paragraphs = data.get("paragraphs") or []
    facts = []
    ratio = []
    issues = []
    reasoning = []
    order = []
    all_sections = []
    for p in paragraphs:
        t = (p.get("paragraph_type") or "unknown").strip().lower()
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if t == "facts":
            facts.append(text)
        elif t == "ratio":
            ratio.append(text)
        elif t in ("arguments", "argument"):
            issues.append(text)
        elif t == "reasoning":
            reasoning.append(text)
        elif t == "order":
            order.append(text)
        for s in (p.get("sections_cited") or []):
            if s and s not in all_sections:
                all_sections.append(s)
    # ── Aggregate cases_cited from paragraph cited_cases ─────────────────────
    all_cited_cases: list = []
    seen_cases: set = set()
    for p in paragraphs:
        for c in (p.get("cited_cases") or []):
            if c and c not in seen_cases:
                seen_cases.add(c)
                all_cited_cases.append(c)

    case_name = (data.get("case_name") or "").strip()
    court = (data.get("court") or "").strip()
    year = str(data.get("year") or "").strip()

    # ── outcome_summary: normalize from top-level outcome + final order ───────
    outcome_raw = (data.get("outcome") or "").strip()
    final_order_text = " ".join(order[:2])[:800] if order else ""
    if outcome_raw:
        outcome_summary = outcome_raw.capitalize()
        if final_order_text:
            outcome_summary = f"{outcome_summary}. {final_order_text[:300]}"
    else:
        outcome_summary = final_order_text[:400]

    return {
        "case_name": case_name,
        "case_id": data.get("case_id"),
        "court": court,
        "year": year,
        "facts_summary": " ".join(facts[:3])[:1500] if facts else "",
        "legal_issues": " ".join(issues[:3])[:1500] if issues else " ".join(reasoning[:2])[:1000] or "",
        "ratio_summary": " ".join(ratio[:3])[:1500] if ratio else "",
        "final_order": final_order_text,
        "outcome_summary": outcome_summary,
        "sections_cited": all_sections[:30],
        "cases_cited": all_cited_cases[:50],
    }


def _sanitize_filename(s: str) -> str:
    """Safe filename: spaces to _, remove invalid chars, collapse underscores."""
    if not s:
        return ""
    s = s.strip().replace(" ", "_")
    for c in '<>:"/\\|?*':
        s = s.replace(c, "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180]  # avoid path length issues on Windows


def _normalize_chunk_ids(data: dict) -> dict:
    case_id = data.get("case_id", "UNKNOWN_CASE")
    paragraphs = data.get("paragraphs", []) or []
    chunk_counter = {}
    for p in paragraphs:
        pid = int(p.get("paragraph_id", 0))
        chunk_counter[pid] = chunk_counter.get(pid, 0) + 1
        cid = chunk_counter[pid]
        p["chunk_id"] = f"{case_id}_P{pid:03d}_C{cid:02d}"
        p["chunk_number"] = cid
    # normalized_case_name for renaming source PDF (e.g. BALMIKI_SINGH_V_RAM_CHANDER_SINGH_AND_ORS)
    case_name = (data.get("case_name") or "").strip()
    if case_name:
        norm = case_name.upper()
        for sep in (" v ", " V ", " versus ", " Versus "):
            if sep in norm:
                parts = norm.split(sep, 1)
                left = _sanitize_filename(parts[0].replace(" ", "_"))
                right = _sanitize_filename(parts[1].replace(" ", "_"))
                data["normalized_case_name"] = f"{left}_V_{right}"
                break
        else:
            data["normalized_case_name"] = _sanitize_filename(norm.replace(" ", "_"))
    else:
        data["normalized_case_name"] = case_id
    return data


# ------------------------------------------------
# Bare Act: section/subsection splitting and statute schema
# (Patterns and helpers ported from Ingestion/smart_chunker.py v4)
# ------------------------------------------------

# Char limit per section chunk — matches smart_chunker's _SUB_CHUNK_THRESHOLD
_BARE_ACT_SECTION_CHAR_LIMIT = 1800   # raised from 1200
_BARE_ACT_MIN_SECTION_LEN    = 30     # ignore sections shorter than this
_BARE_ACT_HARD_SPLIT_LIMIT   = 6000   # para-fallback fires above this

# ── Section boundary patterns (priority: 1A → 1B → 2) ──────────────────────
# Pattern 1A (primary): "NNN. Title.—" or "NNN. Title.—(1)" — Indian bare act format
# Supports: "34. Duty...report.—(1)", "164. Procedure...peace.—(1)", "304. Officer...contingencies.—"
# Also: "304. Officer...contingencies.\n(1)" — no dash when subsection starts next line
# Dash chars: em-dash \u2014, en-dash \u2013, hyphen, minus \u2212, hyphen \u2010
_BA_DASH = r"[\u2014\u2013\-\u2212\u2010\u2011]"
# Variant A: title ends with dash (.— or .-)
_BA_PAT_1A_DASH = re.compile(
    r"^[ \t]*(?:\d{1,4}\[|[¹²³⁴⁵⁶⁷⁸⁹⁰]{1,3}\[?|\[\s]*)?"
    r"(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\.\s+"
    r"([A-Z][^\n\u2014\u2013]{1,350}?(?:\n[A-Za-z][^\n\u2014\u2013]{0,150})??)"
    r"\.?\s*" + _BA_DASH + r"+",
    re.MULTILINE,
)
# Variant B: title ends with period, next line is "(1)" or "(2)" etc (no dash in PDF)
_BA_PAT_1A_NODASH = re.compile(
    r"^[ \t]*(?:\d{1,4}\[|[¹²³⁴⁵⁶⁷⁸⁹⁰]{1,3}\[?|\[\s]*)?"
    r"(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\.\s+"
    r"([A-Z][^\n]{1,400}?\.)\s*\n\s*\((\d+)\)",
    re.MULTILINE,
)
# Pattern 1A = dash variant; 1A_NODASH used to add sections that lack dash (period + newline + "(1)")
# Pattern 1B (secondary): "Section 498A." keyword style (older drafting)
_BA_PAT_1B = re.compile(
    r"^[\s]*(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?(?:-[A-Za-z])?)"
    r"[\.\s\u2014\-:]+(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
# Pattern 2 (fallback): plain "498A. Husband or relative…" (with min-gap filter)
_BA_PAT_2 = re.compile(
    r"^[\s]*(\d+[A-Za-z]?(?:-[A-Za-z])?)[\.\s\u2014\-:]+\s*([A-Z].*?)$",
    re.MULTILINE,
)
# Schedule boundary: stop collecting sections when a SCHEDULE heading appears.
# Indian statutes use: THE FIRST SCHEDULE, FIRST SCHEDULE, SCHEDULE I, SCHEDULE II, etc.
_BA_SCHEDULE_HDR = re.compile(
    r"(?:^|\n)[ \t]*(?:THE\s+)?(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|"
    r"SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+|[IVX]+\.?\s+)?SCHEDULES?\.?\b"
    r"(?:\s+(?:[IVX]+|\d+))?\s*(?:\n|$)",
    re.IGNORECASE,
)
# Additional schedule variants (SCHEDULE I, SCHEDULE II, SCHEDULE A, etc.)
_BA_SCHEDULE_HDR_ROMAN = re.compile(
    r"(?im)^\s*SCHEDULE\s+[IVXLC]+\b",
)

# ── Sub-section boundary patterns ───────────────────────────────────────────
_BA_SUB_COMBINED = re.compile(
    r"(?m)^[ \t]*(?:\((\d+)\)|\(([a-z]+)\)|(Proviso|Explanation\s*\d*|Illustration\s*\d*)[.:\-\u2014]?)[ \t]+",
    re.IGNORECASE,
)

# ── Editorial noise patterns ─────────────────────────────────────────────────
_BA_NOISE_PATTERNS = [
    re.compile(
        r"\b(?:Subs\.?|Substituted|Inserted|Ins\.?|Added|Omitted|Omit\.?|Deleted|Rep\.?)\s+"
        r"by\s+(?:the\s+)?(?:Act\s+)?\d+\s+of\s+\d{4}[^.\n]*(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(r"Published\s+in\s+Gazette\s+of\s+India[^.\n]*(?:\.|$)", re.IGNORECASE),
    re.compile(r"\[?\s*Vide\s+[^\]]+\]\s*", re.IGNORECASE),
    re.compile(r"\s*\[\d{1,3}\]\s*(?=\s|$|\n)"),
    re.compile(r"^\s*\d{1,3}\[\s*", re.MULTILINE),
]
_BA_PAGE_LINE = re.compile(r"^(?:\s*Page\s+)?\d{1,4}\s*$", re.MULTILINE)

# ── Legal keywords for bare act section enrichment ──────────────────────────
_BA_LEGAL_TERMS = {
    "bail", "fir", "arrest", "custody", "remand", "charge sheet", "complaint",
    "petition", "appeal", "revision", "writ", "mandamus", "certiorari",
    "injunction", "specific performance", "breach", "contract", "tort",
    "negligence", "defamation", "fraud", "misrepresentation", "dowry",
    "cruelty", "maintenance", "divorce", "custody", "adoption", "succession",
    "inheritance", "property", "possession", "eviction", "rent", "lease",
    "mortgage", "easement", "partition", "compensation", "damages",
    "limitation", "jurisdiction", "maintainability", "locus standi",
    "res judicata", "estoppel", "natural justice", "fundamental rights",
    "constitutional", "land acquisition", "eminent domain", "environmental",
    "consumer", "arbitration", "mediation", "insolvency", "bankruptcy",
}


def _ba_extract_keywords(text: str) -> list:
    """Extract legal keywords from bare act section text."""
    text_lower = text.lower()
    return [term for term in _BA_LEGAL_TERMS if term in text_lower][:20]


def _ba_safe_id(name: str) -> str:
    """Convert act name to safe ID prefix."""
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return safe[:50] if safe else "unknown"


def _ba_strip_noise(text: str) -> str:
    """Remove amendment notices, gazette refs, footnote markers, page-only lines."""
    out = text or ""
    for pat in _BA_NOISE_PATTERNS:
        out = pat.sub(" ", out)
    out = _BA_PAGE_LINE.sub("", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return out.strip()


def _ba_detect_chapter(text_before: str) -> str:
    """Find the most recent CHAPTER/PART heading before this section."""
    chapters = re.findall(
        r"(CHAPTER|PART)\s+([IVXLCDM\d]+[\s\u2014\-:]*[^\n]*)",
        text_before,
        re.IGNORECASE,
    )
    if chapters:
        kind, detail = chapters[-1]
        return f"{kind.title()} {detail.strip()}"
    return ""


def _extract_schedules(full_text: str, schedule_boundary: int) -> list:
    """
    Extract schedules as separate objects. Schedules run from schedule_boundary to end of document.
    Returns list of {schedule_number, title, text}.
    """
    if schedule_boundary >= len(full_text):
        return []
    schedule_text = full_text[schedule_boundary:].strip()
    if not schedule_text or len(schedule_text) < 50:
        return []
    # Find all schedule headers in the tail
    starts = []
    for m in _BA_SCHEDULE_HDR.finditer(schedule_text):
        starts.append((m.start(), m.group(0).strip()))
    for m in _BA_SCHEDULE_HDR_ROMAN.finditer(schedule_text):
        if not any(s[0] == m.start() for s in starts):
            starts.append((m.start(), m.group(0).strip()))
    starts.sort(key=lambda x: x[0])
    if not starts:
        return [{"schedule_number": "I", "title": "Schedule", "text": schedule_text[:5000]}]
    schedules = []
    for i, (pos, header) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(schedule_text)
        text = schedule_text[pos:end].strip()
        text = _ba_strip_noise(text)
        if len(text) < 30:
            continue
        # Parse schedule number from header (e.g. "THE FIRST SCHEDULE" -> "FIRST")
        num = "I"
        if "FIRST" in header.upper():
            num = "I"
        elif "SECOND" in header.upper():
            num = "II"
        elif "THIRD" in header.upper():
            num = "III"
        elif "FOURTH" in header.upper():
            num = "IV"
        elif "FIFTH" in header.upper():
            num = "V"
        else:
            rom = re.search(r"SCHEDULE\s+([IVXLC]+)", header, re.I)
            if rom:
                num = rom.group(1).upper()
        schedules.append({"schedule_number": num, "title": header[:80], "text": text[:8000]})
    return schedules


def _truncate_at_schedule(text: str) -> str:
    """
    Truncate section text at the first schedule header. Prevents schedule content
    (forms, tables, restarted numbering) from leaking into section body.
    Must be applied BEFORE subsection parsing.
    """
    if not text or not text.strip():
        return text
    m = _BA_SCHEDULE_HDR.search(text)
    if m:
        return text[: m.start()].rstrip()
    m = _BA_SCHEDULE_HDR_ROMAN.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text


def _ba_filter_min_gap(starts: list, min_gap: int = 300) -> list:
    """Remove section-start candidates fewer than min_gap chars apart."""
    if not starts:
        return starts
    filtered = [starts[0]]
    for s in starts[1:]:
        if s["pos"] - filtered[-1]["pos"] >= min_gap:
            filtered.append(s)
    return filtered


def _ba_title_from_match(m) -> str:
    """Extract clean section title from a regex match."""
    raw = (m.group(2) or "").strip()
    first_line = raw.split("\n")[0].strip()
    first_line = re.sub(r"[\s\u2014\u2013.]+$", "", first_line)
    first_line = re.sub(r"^\d+[A-Za-z]?\.\s+", "", first_line)
    return first_line[:120]


def _split_bare_act_into_sections(full_text: str) -> tuple[list, list]:
    """
    Split a bare act into sections and schedules.

    Priority:
      1A) em-dash anchor "NNN. Title.—"  (dominant Indian bare act format)
      1B) "Section NNN." keyword style   (older drafting)
      2)  Plain "NNN. Text"              (fallback, min-gap filtered)

    Schedule boundary: first SCHEDULE in document — section text is truncated there.
    Returns (section_dicts, schedule_dicts). Sections: {sec_num, title, text}. Schedules: {schedule_number, title, text}.
    """
    # --- Find section boundaries ---
    section_starts = []

    p1a = []
    for m in _BA_PAT_1A_DASH.finditer(full_text):
        p1a.append({"pos": m.start(), "sec_num": m.group(1).strip(), "title": _ba_title_from_match(m)})
    # Add sections that end with ".\n(1)" (no dash) — common in BNSS/CrPC when PDF lacks em-dash
    for m in _BA_PAT_1A_NODASH.finditer(full_text):
        pos, sec_num = m.start(), m.group(1).strip()
        title = (m.group(2) or "").strip()
        if title.endswith("."):
            title = title[:-1].strip()
        title = re.sub(r"^\d+[A-Za-z]?\.\s+", "", title)[:120]
        # Skip if already found by dash pattern (within 20 chars)
        if not any(abs(s["pos"] - pos) < 20 for s in p1a):
            p1a.append({"pos": pos, "sec_num": sec_num, "title": title})

    p1b = []
    for m in _BA_PAT_1B.finditer(full_text):
        p1b.append({"pos": m.start(), "sec_num": m.group(1).strip(), "title": _ba_title_from_match(m)})

    # Compute Pattern 2 results (needed for fallback decision)
    p2_raw = []
    for m in _BA_PAT_2.finditer(full_text):
        p2_raw.append({"pos": m.start(), "sec_num": m.group(1).strip(), "title": _ba_title_from_match(m)})
    p2_gapped = _ba_filter_min_gap(p2_raw, min_gap=300)

    if p1a and len(p1a) >= 3:
        # Pattern 1A (dash + no-dash) — use exclusively for BNSS, BNS, CrPC, TPA
        section_starts = p1a
    elif p1a and p2_gapped and len(p2_gapped) > 2 * len(p1a):
        # Pattern 1A found very few, Pattern 2 finds much more — prefer Pattern 2
        # (e.g., Advocates Act 1961 uses plain "N. Title." format)
        section_starts = p2_gapped
    elif p1a:
        section_starts = p1a
    elif p1b:
        section_starts = p1b
    else:
        section_starts = p2_gapped

    if not section_starts:
        return [], []

    # Sort and deduplicate by position first
    section_starts.sort(key=lambda x: x["pos"])
    seen_pos = set()
    unique = []
    for s in section_starts:
        if s["pos"] not in seen_pos:
            seen_pos.add(s["pos"])
            unique.append(s)
    section_starts = unique

    # Deduplicate by section number: keep the body (em-dash) entry over mini-TOC entries.
    # When the same section number appears at multiple positions (e.g. once in a chapter
    # mini-TOC and once in the actual body), prefer the one whose title/surrounding text
    # contains an em-dash "—" (body format). If none has an em-dash, keep the last
    # occurrence (closest to where the actual text appears).
    _EMDASH = re.compile(r"[\u2014\u2013]")
    by_secnum: dict = {}
    for s in section_starts:
        sn = s["sec_num"]
        if sn not in by_secnum:
            by_secnum[sn] = s
        else:
            prev = by_secnum[sn]
            # Prefer whichever has an em-dash in its detected context
            ctx_new  = full_text[s["pos"]:s["pos"] + 300]
            ctx_prev = full_text[prev["pos"]:prev["pos"] + 300]
            new_has_dash  = bool(_EMDASH.search(ctx_new))
            prev_has_dash = bool(_EMDASH.search(ctx_prev))
            if new_has_dash and not prev_has_dash:
                by_secnum[sn] = s   # new is real body — prefer it
            elif not new_has_dash and not prev_has_dash:
                by_secnum[sn] = s   # both TOC-like — take the later (body) one
            # else: prev already has em-dash — keep it
    section_starts = sorted(by_secnum.values(), key=lambda x: x["pos"])

    # Schedule boundary: locate where real schedule content begins.
    # Indian act PDFs often mention schedule titles in the TOC (near the start)
    # before the body sections appear. To avoid treating a TOC schedule reference
    # as the boundary, we search for the first SCHEDULE header that appears AFTER
    # the bulk of the detected sections. Specifically:
    #   - Anchor = position of the last detected section (or 50% of doc if none).
    #   - Search for a SCHEDULE header from that anchor onward.
    # This correctly skips TOC schedule listings and finds the real body schedule.
    schedule_boundary = len(full_text)
    if section_starts:
        anchor = section_starts[-1]["pos"]  # last section's start position
    else:
        anchor = len(full_text) // 2
    sched_m = _BA_SCHEDULE_HDR.search(full_text, pos=anchor)
    if not sched_m:
        sched_m = _BA_SCHEDULE_HDR_ROMAN.search(full_text, pos=anchor)
    if sched_m:
        schedule_boundary = sched_m.start()
    section_starts = [s for s in section_starts if s["pos"] < schedule_boundary]

    if not section_starts:
        return [], []

    # Extract text for each section
    results = []
    for i, sec in enumerate(section_starts):
        start = sec["pos"]
        end = section_starts[i + 1]["pos"] if i + 1 < len(section_starts) else min(schedule_boundary, len(full_text))
        section_text = full_text[start:end].strip()
        section_text = _truncate_at_schedule(section_text)  # Hard stop: no schedule leakage
        section_text = _ba_strip_noise(section_text)
        if len(section_text) < _BARE_ACT_MIN_SECTION_LEN:
            continue
        results.append({
            "sec_num": sec["sec_num"],
            "title": sec["title"],
            "text": section_text,
        })
    schedules = _extract_schedules(full_text, schedule_boundary)
    return results, schedules


def _split_section_into_subchunks(section_text: str, sec_num: str, sec_title: str) -> list:
    """
    Split a single section text (> _BARE_ACT_SECTION_CHAR_LIMIT) into sub-section chunks.
    Splits at (1)/(2)/… numeric sub-sections, (a)/(b)/… clauses, and Proviso/Explanation/Illustration.
    Returns list of (sub_label, text) tuples.
    """
    boundaries = []
    for m in _BA_SUB_COMBINED.finditer(section_text):
        if m.group(1):
            lbl = f"({m.group(1)})"
        elif m.group(2):
            lbl = f"({m.group(2)})"
        else:
            lbl = m.group(3).strip()
        boundaries.append((m.start(), lbl))

    if not boundaries:
        return [("", section_text)]

    segments = []
    prev_end = 0
    prev_label = "intro"
    for pos, lbl in boundaries:
        seg_text = section_text[prev_end:pos].strip()
        if seg_text:
            segments.append((prev_label, seg_text))
        prev_label = lbl
        prev_end = pos
    tail = section_text[prev_end:].strip()
    if tail:
        segments.append((prev_label, tail))

    if not segments:
        return [("", section_text)]

    # Merge very short intro into first real sub-section
    result = []
    carry = ""
    for lbl, seg in segments:
        combined = (carry + "\n\n" + seg).strip() if carry else seg
        carry = ""
        if lbl == "intro" and len(combined) < 100:
            carry = combined
            continue
        if len(combined) >= 50:
            result.append((lbl, combined))

    return result if result else [("", section_text)]


def _para_fallback_split_section(section_text: str) -> list:
    """Last-resort: split large section blob on double-newlines / paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if len(paragraphs) <= 2:
        paragraphs = [ln.strip() for ln in section_text.split("\n")
                      if ln.strip() and len(ln.strip()) > 60]
    results = []
    buffer = ""
    part_idx = 1
    target = 2500
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > target:
            results.append((f"para_{part_idx}", buffer))
            part_idx += 1
            buffer = para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
    if buffer:
        results.append((f"para_{part_idx}", buffer))
    return results if results else [("", section_text)]


def build_act_summary(data: dict) -> dict:
    """
    Build per-act summary for structured embedding.
    Extracts:
      - preamble: introductory text before the first numbered section (or first section text)
      - chapters: chapter headings with section ranges
      - key_sections: list of "Section N: Title" for meaningful sections
      - total_sections count
    """
    act_name = (data.get("act_name") or "Act").strip()
    act_id   = (data.get("act_id") or "").strip()
    year     = data.get("year")
    year_str = str(year) if year is not None else ""
    sections = data.get("sections") or []

    # ── Preamble: use §1's text (short title / commencement) as the preamble anchor.
    # Sort sections by integer section number to find §1 reliably regardless of
    # list ordering — the first section in the sorted order is the canonical opener.
    preamble = ""
    preamble_pat = re.compile(
        r"(?:preamble|whereas|an\s+act\s+to|be\s+it\s+enacted|short\s+title)\b",
        re.IGNORECASE,
    )

    def _sec_sort_key(s):
        raw = re.sub(r"[^0-9]", "", str(s.get("section_number") or "0")) or "0"
        return int(raw)

    sections_sorted = sorted(
        [s for s in sections if not (s.get("sub_section") or "").strip()],
        key=_sec_sort_key,
    )
    if sections_sorted:
        first_text = (sections_sorted[0].get("text") or "").strip()
        if preamble_pat.search(first_text[:300]):
            preamble = first_text[:800]
        elif first_text:
            preamble = first_text[:300]

    # ── Chapters: collect unique chapter headings with their section ranges
    chapter_map = {}  # chapter_heading -> [min_sec_num, max_sec_num]
    for sec in sections:
        ch = (sec.get("chapter") or "").strip()
        if not ch:
            continue
        sn_raw = sec.get("section_number") or "0"
        try:
            sn_int = int(re.sub(r"[^0-9]", "", sn_raw) or "0")
        except Exception:
            sn_int = 0
        if ch not in chapter_map:
            chapter_map[ch] = [sn_int, sn_int]
        else:
            chapter_map[ch][0] = min(chapter_map[ch][0], sn_int)
            chapter_map[ch][1] = max(chapter_map[ch][1], sn_int)
    chapters_list = []
    for ch, (s_min, s_max) in chapter_map.items():
        if s_min == s_max:
            chapters_list.append(f"{ch} (Section {s_min})")
        else:
            chapters_list.append(f"{ch} (Sections {s_min}–{s_max})")

    # ── Key sections: skip preamble/definitions section (usually sec 2/3), pick title
    # Use section_title field if present (new schema), else derive from text first line
    key_sections = []
    seen_titles = set()
    for sec in sections[:50]:  # scan up to 50 sections
        num = sec.get("section_number") or ""
        sub = (sec.get("sub_section") or "").strip()
        if sub:
            continue  # skip sub-chunks — they share the parent section title
        title = (
            (sec.get("section_title") or sec.get("title") or "").strip()
            or (sec.get("text") or "").split("\n")[0][:80].strip()
        )
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        key_sections.append(f"Section {num}: {title}")
        if len(key_sections) >= 30:
            break

    return {
        "act_name": act_name,
        "act_id": act_id,
        "year": year_str,
        "preamble": preamble,
        "chapters": chapters_list,
        "key_sections": key_sections,
        "total_sections": len(set(s.get("section_number") for s in sections if s.get("section_number"))),
    }


def _to_statute_schema(raw: dict, pdf_path: str) -> dict:
    """
    Convert raw PDF extraction to structured bare-act schema.
    Uses multi-pattern section detection and enriched section schema:
      section_number, section_title, sub_section, chapter, keywords, text
    Sections > _BARE_ACT_SECTION_CHAR_LIMIT are split into multiple top-level
    entries (one per sub-section), each inheriting the parent's metadata.
    """
    paragraphs = raw.get("paragraphs", [])
    full_text = "\n".join((p.get("text") or "").strip() for p in paragraphs if (p.get("text") or "").strip())

    # Detect act name from text (try common patterns first)
    act_name = None
    act_year = None
    # Priority-0: first 300 chars title line
    for line in full_text[:300].strip().splitlines()[:3]:
        line = line.strip()
        if not line:
            continue
        title_m = re.match(
            r"^(THE\s+[A-Z][A-Za-z\s,()]+\b(?:ACT|CODE|BILL|SANHITA|ADHINIYAM)\b(?:\s*,?\s*\d{4})?)\s*$",
            line, re.IGNORECASE,
        )
        if title_m:
            act_name = re.sub(r"\s+", " ", title_m.group(1)).strip().title()
            yr_m = re.search(r"(19|20)\d{2}", act_name)
            if yr_m:
                act_year = yr_m.group()
            break
    # Fallback: scan first 5000 chars
    if not act_name:
        match = re.search(r"(THE\s+[A-Z][A-Z\s,()]+\b(?:ACT|CODE|BILL|SANHITA|ADHINIYAM)\b(?:\s*,?\s*\d{4})?)", full_text[:5000], re.IGNORECASE)
        if match:
            act_name = re.sub(r"\s+", " ", match.group(0)).strip().title()
            yr_m = re.search(r"(19|20)\d{2}", act_name)
            if yr_m:
                act_year = yr_m.group()
    if not act_name:
        act_name = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title()
        yr_m = re.search(r"(19|20)\d{2}", act_name)
        if yr_m:
            act_year = yr_m.group()

    year_str = act_year or str(raw.get("year") or "")
    act_id = _build_act_id(act_name, year_str)

    section_dicts, schedules = _split_bare_act_into_sections(full_text)
    sections = []

    for sec_dict in section_dicts:
        sec_num      = sec_dict["sec_num"]
        sec_title    = sec_dict["title"]
        section_text = sec_dict["text"]

        # Find chapter heading for this section
        sec_pos   = full_text.find(section_text[:60]) if section_text else 0
        chapter   = _ba_detect_chapter(full_text[:max(0, sec_pos)])
        section_id = _build_section_id(act_id, sec_num)

        if len(section_text) <= _BARE_ACT_SECTION_CHAR_LIMIT:
            # Section fits in one chunk
            keywords = _ba_extract_keywords(section_text)
            sections.append({
                "section_id":     section_id,
                "section_number": sec_num,
                "section_title":  sec_title,
                "sub_section":    "",
                "chapter":        chapter,
                "keywords":       keywords,
                "text":           section_text,
            })
        else:
            # Section is large — split into sub-chunks (emitted as top-level entries)
            sub_chunks = _split_section_into_subchunks(section_text, sec_num, sec_title)

            if len(sub_chunks) > 1:
                for sub_label, sub_text in sub_chunks:
                    if len(sub_text) < 50:
                        continue
                    # Para-fallback for any sub-chunk still above hard limit
                    if len(sub_text) > _BARE_ACT_HARD_SPLIT_LIMIT:
                        para_chunks = _para_fallback_split_section(sub_text)
                        for plab, ptxt in para_chunks:
                            combined_sub = f"{sub_label}.{plab}" if sub_label else plab
                            sections.append({
                                "section_id":     f"{section_id}_{re.sub(r'[^a-z0-9]', '_', (combined_sub or 'x').lower())}",
                                "section_number": sec_num,
                                "section_title":  sec_title,
                                "sub_section":    combined_sub,
                                "chapter":        chapter,
                                "keywords":       _ba_extract_keywords(ptxt),
                                "text":           ptxt,
                            })
                    else:
                        safe_sub = re.sub(r"[^a-z0-9]", "_", (sub_label or "x").lower()).strip("_")
                        sections.append({
                            "section_id":     f"{section_id}_{safe_sub}" if sub_label else section_id,
                            "section_number": sec_num,
                            "section_title":  sec_title,
                            "sub_section":    sub_label,
                            "chapter":        chapter,
                            "keywords":       _ba_extract_keywords(sub_text),
                            "text":           sub_text,
                        })
            else:
                # No sub-section markers at all — para-fallback split
                para_chunks = _para_fallback_split_section(section_text)
                for plab, ptxt in para_chunks:
                    safe_plab = re.sub(r"[^a-z0-9]", "_", (plab or "x").lower()).strip("_")
                    sections.append({
                        "section_id":     f"{section_id}_{safe_plab}" if plab else section_id,
                        "section_number": sec_num,
                        "section_title":  sec_title,
                        "sub_section":    plab,
                        "chapter":        chapter,
                        "keywords":       _ba_extract_keywords(ptxt),
                        "text":           ptxt,
                    })

    # ── Enrich sections with cross_references and defined_terms ─────────────
    for sec in sections:
        txt = sec.get("text") or ""
        # cross_references: section/article numbers explicitly cited in the text
        xrefs = []
        seen_xrefs: set = set()
        for m in _BA_XREF_PAT.finditer(txt):
            ref = m.group(1).strip()
            # exclude the section's own number
            if ref and ref != sec.get("section_number") and ref not in seen_xrefs:
                seen_xrefs.add(ref)
                xrefs.append(ref)
        sec["cross_references"] = xrefs[:20]
        # defined_terms: terms explicitly defined ("X" means ..., X means ...)
        dterms = []
        seen_dterms: set = set()
        for m in _BA_DEFINED_TERM_PAT.finditer(txt[:3000]):
            term = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            if term and term not in seen_dterms:
                seen_dterms.add(term)
                dterms.append(term)
        sec["defined_terms"] = dterms[:15]

    out = {
        "act_id":       act_id,
        "act_name":     act_name,
        "year":         int(year_str) if year_str and str(year_str).isdigit() else None,
        "jurisdiction": "India",
        "sections":     sections,
        "schedules":    schedules,
    }
    out["act_summary"] = build_act_summary(out)
    return out


# ------------------------------------------------
# Process PDF
# ------------------------------------------------

# ------------------------------------------------
# Run pipeline
# ------------------------------------------------

def run_extraction_only() -> None:
    """Process BareActs PDFs from raw_data/BareActs/ → json_output/ (flat).
    For .txt case law extraction use run_extraction_txt_caselaws().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_paths = []
    bare_acts_root = os.path.join(INPUT_DIR, "BareActs")
    if os.path.isdir(bare_acts_root):
        for root, _, files in os.walk(bare_acts_root):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, name))
    else:
        # Fallback: scan all PDFs in INPUT_DIR (backward-compat)
        for root, _, files in os.walk(INPUT_DIR):
            for name in files:
                if name.lower().endswith(".pdf") and _is_bare_act(os.path.join(root, name)):
                    pdf_paths.append(os.path.join(root, name))

    for pdf_path in tqdm(pdf_paths, desc="Processing BareActs PDFs"):
        rel = os.path.relpath(pdf_path, INPUT_DIR)
        try:
            pages = extract_text_normal(pdf_path)
            # Build a minimal raw dict that _to_statute_schema can consume:
            # it joins p["text"] for each paragraph to get the full act text.
            raw = {"paragraphs": [{"text": clean_text(p or "")} for p in pages]}
            data = _to_statute_schema(raw, pdf_path)
            base = _sanitize_filename((data.get("act_name") or "").strip()) or "bare_act"
            out_path = os.path.join(OUTPUT_DIR, base + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Error:", rel, e)

    _build_cases_and_statutes_index()
    logger.info("BareActs extraction complete. Run build_indexes.py to build vector store.")

def run_build_indexes_only(
    start_year:  int  = None,
    start_month: str  = None,
    end_year:    int  = None,
    end_month:   str  = None,
    append:      bool = None,
) -> None:
    """
    Build (or extend) FAISS + BM25 + citation graph from json_output.

    Date-range parameters filter WHICH case-law JSONs are included in this
    index build, letting you build in year-by-year batches.  Statute
    (bare-act) JSONs are ALWAYS included in full — they are small and must
    appear in every index build.

    Parameters
    ----------
    start_year / start_month / end_year / end_month
        Date range for case-law chunks.  Default to BATCH_* constants.
    append
        Whether to extend the existing FAISS index instead of rebuilding.
        Defaults to the RUN_INDEX_APPEND constant at the top of this file.
        Pass True/False explicitly to override for a single call.

    Typical batched workflow
    ------------------------
    Batch 1 (first ever):  run_build_indexes_only(2020, "JAN", 2020, "DEC", append=False)
    Batch 2:               run_build_indexes_only(2021, "JAN", 2021, "DEC", append=True)
    Batch 3:               run_build_indexes_only(2022, "JAN", 2022, "DEC", append=True)
    """
    sy  = start_year  if start_year  is not None else BATCH_START_YEAR
    sm  = start_month if start_month is not None else BATCH_START_MONTH
    ey  = end_year    if end_year    is not None else BATCH_END_YEAR
    em  = end_month   if end_month   is not None else BATCH_END_MONTH
    app = append      if append      is not None else RUN_INDEX_APPEND

    mode_label = "APPEND (extend existing)" if app else "FRESH (overwrite)"
    logger.info("Index build — range: %s/%s – %s/%s | mode: %s", sy, sm, ey, em, mode_label)

    _build_cases_and_statutes_index(append=app)
    para_chunks, case_summary_chunks = _json_to_case_chunks(sy, sm, ey, em)
    section_chunks, act_summary_chunks = _json_to_statute_chunks()
    _build_vector_store_from_chunks(
        section_chunks, para_chunks,
        case_summary_chunks=case_summary_chunks,
        act_summary_chunks=act_summary_chunks,
        append=app,
    )
    logger.info("Index build complete (case-law range: %s/%s – %s/%s | append=%s).", sy, sm, ey, em, app)


def run_build_indexes_batch(
    start_year: int, start_month: str,
    end_year:   int, end_month:   str,
    append:     bool = None,
) -> None:
    """Convenience alias — build/extend index for an explicit date range batch.

    append defaults to the RUN_INDEX_APPEND constant if not specified.
    """
    run_build_indexes_only(start_year, start_month, end_year, end_month, append=append)

def run_pipeline() -> None:
    """Extraction only: PDF → json_output + cases/statutes index. Run build_indexes.py for FAISS/BM25."""
    run_extraction_only()


def _build_cases_and_statutes_index(append: bool = False):
    """Aggregate json_output into cases/case_metadata.json and statutes/statutes.json.

    Parameters
    ----------
    append : if True and case_metadata.json already exists, load it first and
             only walk case JSONs whose ``base`` is NOT already in the existing
             metadata.  This avoids re-opening 57 K JSON files when adding a
             single year's worth of new extractions.
             Statutes are ALWAYS rebuilt from scratch (there are very few of
             them and they can change when a new BareAct PDF is processed).
    """
    os.makedirs(CASES_DIR, exist_ok=True)
    os.makedirs(STATUTES_DIR, exist_ok=True)
    statutes = []

    # ── Statute (bare-act) JSONs — always rebuilt (small set) ────────────
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if not f.lower().endswith(".json"):
                continue
            path = os.path.join(OUTPUT_DIR, f)
            try:
                with open(path, encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                continue
            base = os.path.splitext(f)[0]
            if data.get("act_id") or (data.get("sections") and data.get("act_name")):
                statutes.append({
                    "act_id":   data.get("act_id"),
                    "act_name": data.get("act_name"),
                    "year":     data.get("year"),
                    "base":     base,
                })
    with open(os.path.join(STATUTES_DIR, "statutes.json"), "w", encoding="utf-8") as fp:
        json.dump(statutes, fp, indent=2, ensure_ascii=False)

    # ── Case-law JSONs — mirrored tree in CASE_OUTPUT_DIR ────────────────
    case_meta_path = os.path.join(CASES_DIR, "case_metadata.json")
    existing_bases: set = set()
    cases: list = []

    if append and os.path.exists(case_meta_path):
        # Load existing metadata; remember which bases are already indexed
        try:
            with open(case_meta_path, encoding="utf-8") as fp:
                cases = json.load(fp)
            existing_bases = {c.get("base") for c in cases if c.get("base")}
            logger.info(
                "_build_cases_and_statutes_index: append mode — "
                "%d cases already in metadata; will only add new entries.", len(cases),
            )
        except Exception as e:
            logger.warning("Could not load existing case_metadata.json (%s) — rebuilding.", e)
            cases = []
            existing_bases = set()

    new_count = 0
    if os.path.isdir(CASE_OUTPUT_DIR):
        for root, _, files in os.walk(CASE_OUTPUT_DIR):
            for f in sorted(files):
                if not f.lower().endswith(".json"):
                    continue
                base = os.path.splitext(f)[0]
                if base in existing_bases:
                    continue  # already in metadata — skip
                path = os.path.join(root, f)
                try:
                    with open(path, encoding="utf-8") as fp:
                        data = json.load(fp)
                except Exception:
                    continue
                if data.get("case_name") or data.get("paragraphs"):
                    cases.append({
                        "case_id":              data.get("case_id"),
                        "case_name":            data.get("case_name"),
                        "normalized_case_name": data.get("normalized_case_name"),
                        "court":                data.get("court"),
                        "year":                 data.get("year"),
                        "date_of_judgment":     data.get("date_of_judgment"),
                        "cites_count":          data.get("cites_count"),
                        "cited_by_count":       data.get("cited_by_count"),
                        "neutral_citation":     data.get("neutral_citation"),
                        "source_type":          data.get("source_type", "txt_scraped"),
                        "base":                 base,
                    })
                    new_count += 1

    logger.info(
        "_build_cases_and_statutes_index: wrote %d total cases (+%d new), %d statutes.",
        len(cases), new_count, len(statutes),
    )
    with open(case_meta_path, "w", encoding="utf-8") as fp:
        json.dump(cases, fp, indent=2, ensure_ascii=False)


def _json_to_case_chunks(
    start_year:  int = None,
    start_month: str = None,
    end_year:    int = None,
    end_month:   str = None,
):
    """
    Convert case-law JSON files to (paragraph_chunks, case_summary_chunks).

    Walks CASE_OUTPUT_DIR (json_output/caselaws/YYYY/MON/) recursively.
    If date-range parameters are provided, only files whose filename matches
    YYYY_MON_N.json within the range are included — enabling batched indexing.
    All four date params must be supplied together to filter; if any is None
    the filter is skipped (all files included).
    """
    sy = start_year  if start_year  is not None else None
    sm = start_month if start_month is not None else None
    ey = end_year    if end_year    is not None else None
    em = end_month   if end_month   is not None else None
    use_filter = all(x is not None for x in [sy, sm, ey, em])

    para_chunks    = []
    summary_chunks = []

    if not os.path.isdir(CASE_OUTPUT_DIR):
        return para_chunks, summary_chunks

    for root, dirs, files in os.walk(CASE_OUTPUT_DIR):
        dirs.sort()
        for fname in sorted(files):
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(root, fname)

            # Date-range filter (treats JSON filename like .txt filename)
            if use_filter:
                txt_like = fname.replace(".json", ".txt")
                fake_path = os.path.join(root, txt_like)
                if not _in_date_range(fake_path, sy, sm, ey, em):
                    continue

            try:
                with open(fpath, encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            if data.get("act_id") or (data.get("sections") and not data.get("paragraphs")):
                continue  # statute — handled separately

            case_name         = (data.get("case_name") or "").strip()
            court             = (data.get("court") or "").strip()
            year              = str(data.get("year") or "").strip()
            case_id           = (data.get("case_id") or "").strip()
            reporter_citations = data.get("reporter_citations") or []
            citation = " | ".join(reporter_citations)[:200] if reporter_citations else ""
            neutral  = data.get("neutral_citation") or ""
            if neutral and neutral not in citation:
                citation = (neutral + " | " + citation).strip(" | ")[:200]

            cites_count    = data.get("cites_count")
            cited_by_count = data.get("cited_by_count")

            summary        = data.get("case_summary") or {}
            ratio_summary  = (summary.get("ratio_summary") or "").strip()
            issues_summary = (summary.get("legal_issues") or "").strip()
            sections_summary = summary.get("sections_cited") or []
            sections_str = ", ".join(sections_summary[:15]) if sections_summary else ""

            # One case-summary chunk per case (for case-level FAISS index)
            summary_parts = [f"Case: {case_name}", f"Court: {court}", f"Year: {year}"]
            if cited_by_count is not None:
                summary_parts.append(f"Cited by: {cited_by_count}")
            if issues_summary:
                summary_parts.append(f"Issues: {issues_summary[:800]}")
            if ratio_summary:
                summary_parts.append(f"Ratio: {ratio_summary[:800]}")
            if sections_str:
                summary_parts.append(f"Sections cited: {sections_str}")
            summary_search_text = "\n\n".join(summary_parts)
            summary_chunks.append({
                "chunk_id":      f"{case_id}_SUMMARY",
                "case_id":       case_id,
                "case_name":     case_name,
                "court":         court,
                "year":          year,
                "citation":      citation,
                "cites_count":   cites_count,
                "cited_by_count": cited_by_count,
                "full_text":     summary_search_text,
                "search_text":   summary_search_text,
                "doc_type":      "case_summary",
            })

            paragraphs = data.get("paragraphs") or []
            for p in paragraphs:
                text = (p.get("text") or "").strip()
                if not text or len(text) < 30:
                    continue
                cid   = p.get("chunk_id") or f"{case_id}_P{p.get('paragraph_id', 0):03d}_C{p.get('chunk_number', 1):02d}"
                ptype = p.get("paragraph_type", "unknown")
                secs  = list(p.get("sections_cited") or [])
                struct_parts = [f"Case: {case_name}"]
                if court:
                    struct_parts.append(f"Court: {court}")
                if year:
                    struct_parts.append(f"Year: {year}")
                if issues_summary and ptype in ("arguments", "reasoning"):
                    struct_parts.append(f"Issue: {issues_summary[:400]}")
                if ratio_summary and ptype == "ratio":
                    struct_parts.append(f"Ratio: {ratio_summary[:400]}")
                elif ptype == "ratio":
                    struct_parts.append(f"Ratio: {text[:400]}")
                if secs:
                    struct_parts.append(f"Sections cited: {', '.join(secs[:10])}")
                struct_parts.append(text[:1200])
                search_text = "\n\n".join(struct_parts)
                para_chunks.append({
                    "chunk_id":       cid,
                    "case_id":        case_id,
                    "case_name":      case_name,
                    "citation":       citation,
                    "court":          court,
                    "year":           year,
                    "cites_count":    cites_count,
                    "cited_by_count": cited_by_count,
                    "paragraph_type": ptype,
                    "sections_cited": secs,
                    "cited_cases":    list(p.get("cited_cases") or []),
                    "full_text":      text,
                    "search_text":    search_text,
                    "doc_type":       "case_law",
                })
    return para_chunks, summary_chunks


def _json_to_statute_chunks() -> tuple[list, list]:
    """Convert statute JSON files in json_output to (section_chunks, act_summary_chunks)."""
    section_chunks = []
    act_summary_chunks = []
    if not os.path.isdir(OUTPUT_DIR):
        return section_chunks, act_summary_chunks
    for f in os.listdir(OUTPUT_DIR):
        if not f.lower().endswith(".json"):
            continue
        path = os.path.join(OUTPUT_DIR, f)
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        if not data.get("act_id") and not (data.get("sections") and data.get("act_name")):
            continue
        act_name = (data.get("act_name") or "Act").strip()
        act_id = (data.get("act_id") or "").strip()
        sections = data.get("sections") or []
        year_str = str(data.get("year", "")) if data.get("year") is not None else ""
        # Section-level chunks with structured search_text
        for sec in sections:
            sec_num = sec.get("section_number") or ""
            section_text = (sec.get("text") or "").strip()
            if not section_text:
                continue
            # Support both old schema ("title") and new schema ("section_title")
            title = (
                sec.get("section_title") or sec.get("title") or
                section_text.split("\n")[0][:150]
            ).strip()
            sub_section = (sec.get("sub_section") or "").strip()
            chapter     = (sec.get("chapter") or "").strip()
            keywords    = sec.get("keywords") or []
            sec_id = sec.get("section_id") or f"{act_id}_SECTION_{sec_num}"
            # Build rich search_text
            search_text = f"Act: {act_name} | Section {sec_num}: {title}"
            if sub_section:
                search_text += f" {sub_section}"
            if chapter:
                search_text += f" [{chapter}]"
            if keywords:
                search_text += f" | Keywords: {', '.join(keywords[:10])}"
            search_text += f" | {section_text[:1500]}"
            section_chunks.append({
                "chunk_id":       sec_id,
                "act_name":       act_name,
                "act_id":         act_id,
                "year":           year_str,
                "section_number": sec_num,
                "section_title":  title,
                "sub_section":    sub_section,
                "chapter":        chapter,
                "keywords":       keywords,
                "section_id":     sec_id,
                "full_text":      section_text,
                "search_text":    search_text,
                "doc_type":       "bare_act",
            })
        # One act-summary chunk per act (for two-tier retrieval)
        act_sum = data.get("act_summary") or build_act_summary(data)
        preamble     = (act_sum.get("preamble") or "").strip()
        key_sections = act_sum.get("key_sections") or []
        chapters     = act_sum.get("chapters") or []
        total_secs   = act_sum.get("total_sections") or 0
        summary_search = f"Act: {act_name} | Year: {year_str}"
        if total_secs:
            summary_search += f" | Total sections: {total_secs}"
        if preamble:
            summary_search += f" | Preamble: {preamble[:400]}"
        if chapters:
            summary_search += " | Chapters: " + "; ".join(chapters[:10])
        if key_sections:
            summary_search += " | Sections: " + " | ".join(key_sections[:20])
        act_summary_chunks.append({
            "chunk_id":      f"{act_id}_SUMMARY",
            "act_name":      act_name,
            "act_id":        act_id,
            "year":          year_str,
            "total_sections": total_secs,
            "full_text":     preamble,
            "search_text":   summary_search,
            "doc_type":      "act_summary",
        })
    return section_chunks, act_summary_chunks


def _build_vector_store_from_chunks(
    bare_chunks: list,
    case_chunks: list,
    case_summary_chunks: list = None,
    act_summary_chunks: list = None,
    append: bool = False,
) -> None:
    """Build (or extend) FAISS + BM25 indexes.

    Parameters
    ----------
    append : passed through to build_index().  True = extend existing indexes;
             False = fresh build.  Bare-act indexes are ALWAYS rebuilt fresh
             (they are small and rarely change).  Only case-law indexes honour
             the append flag so that year-by-year batches accumulate correctly.
    """
    case_summary_chunks = case_summary_chunks or []
    act_summary_chunks = act_summary_chunks or []
    if not bare_chunks and not case_chunks and not case_summary_chunks and not act_summary_chunks:
        logger.warning("No chunks to index — skipping vector store build")
        return
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        from Ingestion.build_v2_index import build_index, _get_embedder
        from retrieval.citation_graph import build_citation_graph_from_chunks
        from retrieval.hybrid_retriever import BM25, save_bm25_index
    except ImportError as e:
        logger.warning("Vector store build skipped (missing deps: %s). Set PYTHONPATH to project root and install deps.", e)
        return
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    bare_index              = os.path.join(VECTOR_STORE_DIR, "bareacts_v2.index")
    bare_chunks_path        = os.path.join(VECTOR_STORE_DIR, "bareacts_v2_chunks.json")
    bare_bm25               = os.path.join(VECTOR_STORE_DIR, "bareacts_bm25.json")
    case_index              = os.path.join(VECTOR_STORE_DIR, "caselaws_v2.index")
    case_chunks_path        = os.path.join(VECTOR_STORE_DIR, "caselaws_v2_chunks.json")
    case_bm25               = os.path.join(VECTOR_STORE_DIR, "caselaws_bm25.json")
    case_summary_index      = os.path.join(VECTOR_STORE_DIR, "case_summaries_v2.index")
    case_summary_chunks_path= os.path.join(VECTOR_STORE_DIR, "case_summaries_v2_chunks.json")
    case_summary_bm25       = os.path.join(VECTOR_STORE_DIR, "case_summaries_bm25.json")
    act_summary_index       = os.path.join(VECTOR_STORE_DIR, "act_summaries_v2.index")
    act_summary_chunks_path = os.path.join(VECTOR_STORE_DIR, "act_summaries_v2_chunks.json")
    act_summary_bm25        = os.path.join(VECTOR_STORE_DIR, "act_summaries_bm25.json")
    citation_path           = os.path.join(VECTOR_STORE_DIR, "citation_graph.json")

    embedder = _get_embedder()

    # Bare-act indexes: always fresh (small, rarely change)
    if bare_chunks:
        logger.info("Building bare acts index: %d chunks", len(bare_chunks))
        build_index(bare_chunks, bare_index, bare_chunks_path, bare_bm25, embedder, append=False)

    # Case-law paragraph index: honours the append flag
    if case_chunks:
        logger.info(
            "%s case law paragraph index: %d chunks",
            "Extending" if append else "Building", len(case_chunks),
        )
        build_index(case_chunks, case_index, case_chunks_path, case_bm25, embedder, append=append)
        try:
            # Citation graph: rebuild from the chunk list passed this run
            # (graph edges can only reference cases in the current batch,
            #  which is acceptable — the graph is used for re-ranking not lookup)
            chunks_dict = {str(i): c for i, c in enumerate(case_chunks)}
            build_citation_graph_from_chunks(chunks_dict, citation_path)
            logger.info("Citation graph written to %s", citation_path)
        except Exception as e:
            logger.warning("Citation graph build failed (non-fatal): %s", e)

    # Case-summary index: honours the append flag
    if case_summary_chunks:
        logger.info(
            "%s case summary index: %d cases",
            "Extending" if append else "Building", len(case_summary_chunks),
        )
        build_index(
            case_summary_chunks, case_summary_index,
            case_summary_chunks_path, case_summary_bm25, embedder, append=append,
        )

    # Act-summary index: always fresh (mirrors bare-act rebuild)
    if act_summary_chunks:
        logger.info("Building act summary index: %d acts", len(act_summary_chunks))
        build_index(
            act_summary_chunks, act_summary_index,
            act_summary_chunks_path, act_summary_bm25, embedder, append=False,
        )


if __name__ == "__main__":
    import sys

    # ── Logging setup ──────────────────────────────────────────────────────
    _log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    _fmt    = "%(asctime)s  %(levelname)-8s  %(message)s"
    _datefmt = "%H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, _log_level, logging.WARNING))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, _log_level, logging.WARNING))
    console_handler.setFormatter(logging.Formatter(_fmt, datefmt=_datefmt))
    root.addHandler(console_handler)
    # Dedicated audit logger → pipeline.log (overwrite each run)
    audit_path = os.path.join(os.path.dirname(__file__), "pipeline.log")
    audit_handler = logging.FileHandler(audit_path, mode="w", encoding="utf-8")
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger = logging.getLogger("nyaymalaw.audit")
    audit_logger.handlers.clear()
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False
    # ───────────────────────────────────────────────────────────────────────

    print("=" * 64)
    print("Nyaymalaw Pipeline")
    if NEW_FOLDERS:
        print(f"  Mode    : folder list → {NEW_FOLDERS}")
    else:
        print(f"  Mode    : date range  → {BATCH_START_YEAR}/{BATCH_START_MONTH} – {BATCH_END_YEAR}/{BATCH_END_MONTH}")
    print(f"  Order   : {BATCH_ORDER}")
    print(f"  Workers : {BATCH_WORKERS}")
    print("  Note    : existing JSON files are NEVER overwritten")
    print("=" * 64)

    if NEW_FOLDERS:
        # Targeted mode: only process the folders you explicitly listed
        run_extraction_txt_caselaws(folders=NEW_FOLDERS)
    else:
        # Date-range mode: process everything in the configured BATCH_* window
        run_extraction_txt_caselaws()

    print("Done — run build_indexes.py (or call run_build_indexes_only()) for FAISS/BM25.")
