"""
Smart Chunker v3 — Section-level chunking for Bare Acts, paragraph-level for Case Laws.

v3 changes vs v2:
  1. Pattern 1A (em-dash anchor) added as the primary section detector — matches the
     dominant Indian bare act format "NNN. Title.—" used by BNSS, BNS, TPA, CPC,
     Evidence, Contract, Companies Acts etc.
  2. Sub-section splitter: sections >3 000 chars are split at (1)/(2)/… level;
     sub-sections still >3 000 chars are split at (a)/(b)/… level.  Proviso,
     Explanation and Illustration are treated as named sub-chunks.  Every sub-chunk
     inherits the parent section's metadata and carries a 'sub_section' field.
  3. _detect_act_name_from_text now recognises SANHITA / ADHINIYAM / NIYAMAWALI
     keywords (needed for BNSS, BNS, BSA) and scans the first 5 000 chars.
  4. _detect_case_metadata: expanded citation patterns (INSC, SCC Online, ILR,
     MANU); court name OCR-garble correction via fuzzy state-name matching.
  5. process_case_laws_directory deduplicates chunks on (case_name, para_num,
     first-200-chars) before returning, eliminating ~1 500 duplicate case chunks.
"""

import re
import os
import json
import logging
import pdfplumber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
    return text


# Approximate chars for "first two pages" when page boundaries aren't available (e.g. .txt)
FIRST_TWO_PAGES_CHARS = 3000


def extract_text_from_pdf_first_n_pages(pdf_path: str, n: int = 2) -> str:
    """Extract text from only the first n pages of a PDF. Used for act vs judgment detection."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:n]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Failed to extract first {n} pages from {pdf_path}: {e}")
    return text


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or text file."""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    else:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""


# ---------------------------------------------------------------------------
# Bare Act Section-Level Chunking
# ---------------------------------------------------------------------------

# Patterns to detect section headings in Indian bare acts
_SECTION_PATTERNS = [
    # Pattern 1A (primary — highest confidence):
    #   "53. Fraudulent transfer.—"  /  "53A. Part performance.—"
    #   Matches em-dash U+2014 AND en-dash U+2013 because pdfplumber substitutes
    #   U+2013 for U+2014 on many Indian government PDFs (BNSS, BNS, etc.).
    #   The exclusion class [^\n\u2014\u2013] prevents the title from swallowing
    #   a dash that belongs to the next section.
    #
    #   Amendment-bracket prefix: India Code PDFs mark amendment-inserted sections
    #   with a footnote number + opening bracket before the section number, e.g.:
    #     "1[106. Duration of certain leases.—"   (TPA s.106, inserted by Act 1)
    #     "10[3. Definitions.—"                   (CPC, inserted by 10th amendment)
    #   pdfplumber extracts superscripts as plain digits, so the prefix is always
    #   "\d{1,3}[" (1–3 digit footnote ref + "[").  We also handle bare "[" for
    #   cases where the footnote number is absent, and Unicode superscripts
    #   (¹²³…) for PDFs that preserve them.
    re.compile(
        r"^[ \t]*(?:\d{1,3}\[|[¹²³⁴⁵⁶⁷⁸⁹]{1,2}\[?|\[)?"
        r"(\d+[A-Za-z]{0,3}(?:-[A-Za-z])?)\.\s+"
        r"([A-Z][^\n\u2014\u2013]{1,150}?(?:\n[A-Za-z][^\n\u2014\u2013]{0,80})??)\.?\s*[\u2014\u2013]",
        re.MULTILINE,
    ),
    # Pattern 1B (secondary):  "Section 498A." / "S. 498A —"  (older drafting style)
    re.compile(
        r"^[\s]*(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?(?:-[A-Za-z])?)"
        r"[\.\s—\-:]+(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Pattern 2 (fallback):  plain "498A. Husband or relative…"
    # Only used when 1A + 1B both find nothing.
    re.compile(
        r"^[\s]*(\d+[A-Za-z]?(?:-[A-Za-z])?)[\.\s—\-:]+\s*([A-Z].*?)$",
        re.MULTILINE,
    ),
    # Article pattern (Constitution / state acts with Articles)
    re.compile(
        r"^[\s]*(?:Article|Art\.?)\s*(\d+[A-Za-z]?)"
        r"[\.\s—\-:]+(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# ---------------------------------------------------------------------------
# Short-name aliases for common Indian acts (used to enrich search_text so
# that queries like "BNS Section 117" retrieve the correct chunks even though
# the act is stored as "The Bharatiya Nyaya Sanhita 2023").
# ---------------------------------------------------------------------------
_ACT_ALIASES: dict = {
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
    """Return the short alias for an act, or empty string if none known."""
    name_lower = act_name.lower()
    for key, alias in _ACT_ALIASES.items():
        if key in name_lower:
            return alias
    return ""


def _filter_min_gap(starts: list, min_gap: int = 300) -> list:
    """
    Remove section-start candidates that are fewer than min_gap characters
    apart from the previous accepted candidate.

    This prevents numbered sub-items within a section (e.g. the list of
    'grievous hurt' types in BNS §117, each starting with "1.", "2." …)
    from being treated as separate section boundaries when Pattern 2 fires.
    """
    if not starts:
        return starts
    filtered = [starts[0]]
    for s in starts[1:]:
        if s["pos"] - filtered[-1]["pos"] >= min_gap:
            filtered.append(s)
    return filtered


# ---------------------------------------------------------------------------
# Act-name false-positive blocklist
# Phrases that LOOK like act names (end with "Act"/"Code") but are actually
# sentence fragments extracted from section body text.
# ---------------------------------------------------------------------------
_ACT_NAME_BLOCKLIST = frozenset({
    "the purpose of this act",
    "the purpose of this code",
    "the following act",
    "the said act",
    "the said code",
    "the commencement of this act",
    "the commencement of this code",
    "the provisions of this act",
    "the provisions of this code",
    "the lines of right",
    "the list as may be prescribed",
    "the power to make rules",
    "the territories which",
    "the university grants commission",
})

_ACT_NAME_SKIP_WORDS = frozenset({
    "the", "act", "code", "bill", "this", "that", "such", "said",
    "following", "above", "aforesaid", "relevant", "applicable",
    "respective", "sanhita", "adhiniyam", "niyam", "niyamawali",
})


def _is_valid_act_name(name: str) -> bool:
    """
    Return True if `name` looks like a genuine act title.
    Rejects:
      • Blocklisted fragment phrases.
      • Names with fewer than 2 'meaningful' words (i.e. words ≥ 4 chars
        that are not common stop-words).
    """
    name_lower = name.lower().strip()
    # Blocklist check (prefix match so "the purpose of this act, 2020" also caught)
    for blocked in _ACT_NAME_BLOCKLIST:
        if name_lower.startswith(blocked):
            return False
    # Must contain at least 2 meaningful content words
    meaningful = [
        w for w in name.split()
        if len(w) >= 4 and w.lower() not in _ACT_NAME_SKIP_WORDS
    ]
    return len(meaningful) >= 2


def _detect_act_name_from_text(text: str, filename: str) -> str:
    """
    Try to extract the act name from the document text or filename.

    v3: scans first 5 000 chars (not 2 000) and adds SANHITA / ADHINIYAM /
    NIYAMAWALI keywords so that BNSS, BNS and BSA are named correctly.
    v4: validates candidates through _is_valid_act_name() to reject fragment
    phrases like "The Purpose Of This Act" or "The Following Act".
    v5: Priority-0 check on first 3 lines so the real document title wins over
    amending-acts lists (e.g. CPC 1908 consolidated PDF listing "Amendment Act,
    1914" near the top).
    """
    # Priority 0: check the very first 3 lines for a document title.
    # This prevents the amending-acts list near the top of a consolidated PDF
    # (e.g. CPC 1908 which lists "Amendment Act, 1914" at line 4) from
    # overriding the real title on line 1.
    _title_lines = text[:300].strip().splitlines()
    for line in _title_lines[:3]:
        line = line.strip()
        if not line:
            continue
        # Must look like "The Foo Bar Act, YYYY" or "THE FOO SANHITA YYYY"
        title_act = re.match(
            r"^(THE\s+[A-Z][A-Za-z\s,()]+\b(?:ACT|CODE|BILL|ORDINANCE|REGULATION)\b"
            r"(?:\s*,?\s*\d{4})?)\s*$",
            line, re.IGNORECASE,
        )
        if title_act:
            name = re.sub(r"\s+", " ", title_act.group(1)).strip()
            if len(name) > 10 and _is_valid_act_name(name):
                return name.title()
        title_ind = re.match(
            r"^((?:THE\s+)?[A-Z][A-Za-z\s,()]+\b(?:SANHITA|ADHINIYAM|NIYAMAWALI|NIYAM)\b"
            r"(?:[\s,]*(?:19|20)\d{2})?)\s*$",
            line, re.IGNORECASE,
        )
        if title_ind:
            name = re.sub(r"\s+", " ", title_ind.group(1)).strip()
            if len(name) > 10 and _is_valid_act_name(name):
                return name.title()
        # "Name, YYYY" form (e.g. "The Code of Civil Procedure, 1908")
        title_year = re.match(
            r"^((?:THE\s+)?[A-Za-z][A-Za-z\s,\'\-()]+,\s*(?:19|20)\d{2})\s*$",
            line, re.IGNORECASE,
        )
        if title_year:
            name = re.sub(r"\s+", " ", title_year.group(1)).strip()
            if (len(name) > 12
                    and not re.match(r"^(?:Section|Article|Sec\.?|Art\.?)\s", name, re.I)
                    and _is_valid_act_name(name)):
                return name.title()

    sample = text[:5000]

    # 1) Standard keywords: ACT / CODE / BILL / ORDINANCE / REGULATION
    #    Requires "THE" prefix so that section titles like "accident in doing a
    #    lawful act" don't match.  \b prevents "act" inside "practitioner".
    act_pattern = re.compile(
        r"(THE\s+[A-Z][A-Z\s,()]+\b(?:ACT|CODE|BILL|ORDINANCE|REGULATION)\b"
        r"(?:\s*,?\s*\d{4})?)",
        re.IGNORECASE,
    )
    match = act_pattern.search(sample)
    if match:
        name = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(name) > 10 and _is_valid_act_name(name):
            return name.title()

    # 2) Indian-language enactment keywords: SANHITA / ADHINIYAM / NIYAMAWALI / NIYAM
    indian_kw_pattern = re.compile(
        r"(?:THE\s+)?([A-Z][A-Z\s,()]+\b(?:SANHITA|ADHINIYAM|NIYAMAWALI|NIYAM)\b"
        r"(?:[\s,]*(?:19|20)\d{2})?)",
        re.IGNORECASE,
    )
    match_ik = indian_kw_pattern.search(sample)
    if match_ik:
        name = re.sub(r"\s+", " ", match_ik.group(0)).strip()
        if len(name) > 10 and _is_valid_act_name(name):
            return name.title()

    # 3) "Name, YYYY" fallback when ACT/CODE etc. is missing
    name_year_pattern = re.compile(
        r"\b((?:THE\s+)?[A-Za-z][A-Za-z0-9\s,\'\-()]+,\s*(?:19|20)\d{2})\b",
        re.IGNORECASE,
    )
    match_ny = name_year_pattern.search(sample)
    if match_ny:
        name = re.sub(r"\s+", " ", match_ny.group(1)).strip()
        if (len(name) > 12
                and not re.match(r"^(?:Section|Article|Sec\.?|Art\.?)\s", name, re.I)
                and _is_valid_act_name(name)):
            return name.title()

    # 4) Fall back to filename
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace("_", " ").replace("-", " ")
    tokens = [t for t in name.split() if not re.fullmatch(r"\d+", t)]
    return " ".join(tokens).strip().title() or "Unknown Act"


def _detect_chapter(text_before: str) -> str:
    """Find the most recent CHAPTER/PART heading before this section."""
    chapters = re.findall(
        r"(CHAPTER|PART)\s+([IVXLCDM\d]+[\s—\-:]*[^\n]*)",
        text_before,
        re.IGNORECASE,
    )
    if chapters:
        kind, detail = chapters[-1]
        return f"{kind.title()} {detail.strip()}"
    return ""


# ---------------------------------------------------------------------------
# Sub-section splitter (Change 2)
# ---------------------------------------------------------------------------

# Patterns for sub-section boundaries within a section body
_SUB_SEC_PATTERN = re.compile(r"(?m)^[ \t]*\((\d+)\)[ \t]+")   # (1), (2), …
_CLAUSE_PATTERN  = re.compile(r"(?m)^[ \t]*\(([a-z]+)\)[ \t]+") # (a), (b), …

# Named structural elements treated as sub-chunks
_NAMED_SUBCHUNK_PATTERN = re.compile(
    r"(?m)^[ \t]*(Proviso|Explanation\s*\d*|Illustration\s*\d*)\b[.:\-—]?[ \t]*",
    re.IGNORECASE,
)

_SUB_CHUNK_THRESHOLD = 3000   # chars; split section if longer than this
_SUB_CHUNK_MIN       = 100    # don't emit sub-chunks shorter than this


def _split_section_into_subchunks(
    section_text: str,
    act_name: str,
    act_alias: str,
    sec_num: str,
    sec_title: str,
    chapter: str,
    sec_type: str,
    source_file: str,
) -> list:
    """
    Split a single section text that is longer than _SUB_CHUNK_THRESHOLD into
    sub-section level chunks:
      Level 1 — at (1), (2), (3) … sub-sections
      Level 2 — at (a), (b) … clauses within a sub-section still >threshold
      Also splits at Proviso / Explanation / Illustration headings.

    Every sub-chunk inherits the parent section's metadata and adds a
    'sub_section' field describing which part it covers.

    Returns a list of chunk dicts (same schema as chunk_bare_act output).
    """
    alias_part = f" ({act_alias})" if act_alias else ""

    def _make_sub(text: str, sub_label: str) -> dict:
        text = text.strip()
        search_text = (
            f"{act_name}{alias_part} {sec_type.title()} {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + (f" {sub_label}" if sub_label else "")
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{text}"
        )
        sub_safe = re.sub(r"[^a-z0-9]+", "_", sub_label.lower()).strip("_")
        return {
            "chunk_id": f"{_safe_id(act_name)}_{sec_type}_{sec_num}_{sub_safe}",
            "act_name": act_name,
            "section_number": sec_num,
            "section_title": sec_title,
            "sub_section": sub_label,
            "chapter": chapter,
            "full_text": text,
            "search_text": search_text,
            "keywords": _extract_keywords(text),
            "source_file": source_file,
            "doc_type": "bare_act",
        }

    def _split_at_pattern(pat: re.Pattern, text: str) -> list:
        """Split text at regex boundaries; return list of (label, text) pairs."""
        parts = []
        last_end = 0
        label = "intro"
        for m in pat.finditer(text):
            chunk_text = text[last_end:m.start()].strip()
            if chunk_text:
                parts.append((label, chunk_text))
            label = f"({m.group(1)})"
            last_end = m.start()
        # Remainder
        remainder = text[last_end:].strip()
        if remainder:
            parts.append((label, remainder))
        return parts

    # --- First: check for named structural elements (Proviso/Explanation/Illustration)
    # We'll split at ALL boundary types in a single pass using a combined pattern.
    combined_pat = re.compile(
        r"(?m)^[ \t]*(?:\((\d+)\)|\(([a-z]+)\)|(Proviso|Explanation\s*\d*|Illustration\s*\d*)[.:\-—]?)[ \t]+",
        re.IGNORECASE,
    )

    boundaries = []
    for m in combined_pat.finditer(section_text):
        if m.group(1):
            lbl = f"({m.group(1)})"
        elif m.group(2):
            lbl = f"({m.group(2)})"
        else:
            lbl = m.group(3).strip()
        boundaries.append((m.start(), lbl))

    if not boundaries:
        # Nothing to split on — return as single chunk
        return [_make_sub(section_text, "")]

    # Build segments
    segments: list = []  # list of (label, text)
    prev_end = 0
    prev_label = "intro"
    for pos, lbl in boundaries:
        seg_text = section_text[prev_end:pos].strip()
        if seg_text:
            segments.append((prev_label, seg_text))
        prev_label = lbl
        prev_end = pos
    # Last segment
    tail = section_text[prev_end:].strip()
    if tail:
        segments.append((prev_label, tail))

    if not segments:
        return [_make_sub(section_text, "")]

    # Now emit sub-chunks; merge tiny intro into first real sub-section
    result = []
    carry = ""
    for lbl, seg in segments:
        combined_text = (carry + "\n\n" + seg).strip() if carry else seg
        carry = ""
        if lbl == "intro" and len(combined_text) < _SUB_CHUNK_MIN:
            # Very short intro — carry it into the next sub-chunk
            carry = combined_text
            continue
        if len(combined_text) >= _SUB_CHUNK_MIN:
            result.append(_make_sub(combined_text, lbl))

    if not result:
        result.append(_make_sub(section_text, ""))

    return result


_HARD_SPLIT_THRESHOLD = 6000    # chars; paragraph-fallback fires above this
_HARD_SPLIT_PARA_TARGET = 2500  # aim for ~2.5 k char paragraphs in fallback


def _para_fallback_split(chunk: dict) -> list:
    """
    Last-resort splitter for blobs that have NO (1)/(2) sub-section markers
    (e.g. a BNSS section that absorbed 20 un-detected sections due to missing
    em-dashes in the PDF, or a large Schedule table).

    Splits on double-newlines into ~2.5 k char chunks.  Each chunk gets the
    parent section's metadata plus sub_section="para_N".
    """
    text = chunk["full_text"]
    act_name  = chunk["act_name"]
    act_alias = _act_alias(act_name)
    sec_num   = chunk.get("section_number", "")
    sec_title = chunk.get("section_title", "")
    chapter   = chunk.get("chapter", "")
    source    = chunk.get("source_file", "")
    alias_part = f" ({act_alias})" if act_alias else ""

    # Try double-newline splitting first; fall back to single-newline when the
    # text has no blank lines (common in pdfplumber output for some PDFs).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 2:
        paragraphs = [ln.strip() for ln in text.split("\n")
                      if ln.strip() and len(ln.strip()) > 60]
    results = []
    buffer = ""
    part_idx = 1

    def _emit(buf: str, idx: int) -> dict:
        lbl = f"para_{idx}"
        search_text = (
            f"{act_name}{alias_part} Section {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + f" {lbl}"
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{buf}"
        )
        return {
            "chunk_id": f"{_safe_id(act_name)}_section_{sec_num}_{lbl}",
            "act_name": act_name,
            "section_number": sec_num,
            "section_title": sec_title,
            "sub_section": lbl,
            "chapter": chapter,
            "full_text": buf,
            "search_text": search_text,
            "keywords": _extract_keywords(buf),
            "source_file": source,
            "doc_type": "bare_act",
        }

    for para in paragraphs:
        if len(buffer) + len(para) > _HARD_SPLIT_PARA_TARGET and buffer:
            results.append(_emit(buffer, part_idx))
            part_idx += 1
            buffer = para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    if buffer:
        results.append(_emit(buffer, part_idx))

    logger.debug(
        "Para-fallback split Section %s of %s into %d parts (was %d chars)",
        sec_num, act_name, len(results), len(text),
    )
    return results if results else [chunk]


def _maybe_split_chunk(chunk: dict) -> list:
    """
    If chunk['full_text'] exceeds _SUB_CHUNK_THRESHOLD, split it into
    sub-section level chunks.  Otherwise return [chunk] unchanged.

    Cascade:
      1. Try structured sub-section splitting ((1)/(2)/Proviso etc.)
      2. If no markers found AND chunk still > _HARD_SPLIT_THRESHOLD,
         apply paragraph-level fallback to avoid leaving multi-section blobs.
    """
    full_text = chunk.get("full_text", "")
    if len(full_text) <= _SUB_CHUNK_THRESHOLD:
        return [chunk]

    sub_chunks = _split_section_into_subchunks(
        section_text=full_text,
        act_name=chunk["act_name"],
        act_alias=_act_alias(chunk["act_name"]),
        sec_num=chunk.get("section_number", ""),
        sec_title=chunk.get("section_title", ""),
        chapter=chunk.get("chapter", ""),
        sec_type="section",
        source_file=chunk.get("source_file", ""),
    )

    if len(sub_chunks) > 1:
        # Apply paragraph fallback to any sub-chunk that is still above the
        # hard threshold (e.g. a spurious "section" blob that had one (b) marker
        # early on, leaving a 78k tail labelled sub=(b)).
        final: list = []
        for sc in sub_chunks:
            if len(sc.get("full_text", "")) > _HARD_SPLIT_THRESHOLD:
                final.extend(_para_fallback_split(sc))
            else:
                final.append(sc)
        logger.debug(
            "Section %s of %s → %d sub-chunks (was %d chars)",
            chunk.get("section_number", "?"),
            chunk.get("act_name", "?"),
            len(final),
            len(full_text),
        )
        return final

    # No sub-section markers — if still very large, use paragraph fallback
    if len(full_text) > _HARD_SPLIT_THRESHOLD:
        return _para_fallback_split(chunk)

    return [chunk]


# ---------------------------------------------------------------------------
# Core bare-act chunker
# ---------------------------------------------------------------------------

def chunk_bare_act(text: str, filename: str = "") -> list:
    """
    Split a bare act into section-level chunks with rich metadata.

    Each chunk = one complete section (or sub-section for large sections) with:
    - act_name, section_number, section_title, chapter, full_text
    - keywords extracted from content
    - sub_section field (empty for full-section chunks)
    """
    act_name = _detect_act_name_from_text(text, filename)
    act_alias = _act_alias(act_name)
    chunks = []

    def _title_from_match(m) -> str:
        """Cap section title to first line, max 120 chars."""
        raw = (m.group(2) or "").strip()
        first_line = raw.split("\n")[0].strip()
        # Remove trailing em/en-dash and period
        first_line = re.sub(r"[\s\u2014\u2013.]+$", "", first_line)
        # Strip a spurious leading "NNN. " prefix that appears when pdfplumber
        # concatenates a page-number or sub-item ("6.") with the real heading
        # ("29. Title"), producing Group 2 = "29. Title" with sec_num = "6".
        first_line = re.sub(r"^\d+[A-Za-z]?\.\s+", "", first_line)
        return first_line[:120]

    # -----------------------------------------------------------------------
    # Find all section boundaries.
    # Priority:
    #   1A) Em-dash pattern  "NNN. Title.—"  (dominant Indian bare act format)
    #   1B) "Section NNN."   (explicit keyword, older style)
    #   2)  Plain "NNN. Title"  (fallback with min-gap filter)
    # -----------------------------------------------------------------------
    section_starts = []

    # --- Pattern 1A: em-dash anchor
    p1a_starts = []
    for match in _SECTION_PATTERNS[0].finditer(text):
        p1a_starts.append({
            "pos": match.start(),
            "section_number": match.group(1).strip(),
            "section_title": _title_from_match(match),
            "type": "section",
        })

    # --- Pattern 1B: "Section X." keyword
    p1b_starts = []
    for match in _SECTION_PATTERNS[1].finditer(text):
        p1b_starts.append({
            "pos": match.start(),
            "section_number": match.group(1).strip(),
            "section_title": _title_from_match(match),
            "type": "section",
        })

    if p1a_starts:
        # Use Pattern 1A exclusively when it fires.
        # DO NOT merge Pattern 1B results: 1B matches inline references like
        # "section 84 may, for reasons..." inside section 85's body text, which
        # would cut sections in half and mislabel the continuation as a new section.
        section_starts = p1a_starts
        logger.debug(f"Pattern 1A (em-dash) found {len(p1a_starts)} sections in {filename}")
    elif p1b_starts:
        section_starts = p1b_starts
        logger.debug(f"Pattern 1A found nothing; Pattern 1B found {len(p1b_starts)} sections in {filename}")
    else:
        # Pattern 2 fallback: plain number at line start
        p2_starts = []
        for match in _SECTION_PATTERNS[2].finditer(text):
            p2_starts.append({
                "pos": match.start(),
                "section_number": match.group(1).strip(),
                "section_title": _title_from_match(match),
                "type": "section",
            })
        section_starts = _filter_min_gap(p2_starts, min_gap=300)
        if section_starts:
            logger.debug(
                f"Patterns 1A+1B found nothing; Pattern 2 fallback yielded "
                f"{len(section_starts)} sections (after gap filter) in {filename}"
            )

    # Also find article patterns (Constitution / state acts with Articles)
    for match in _SECTION_PATTERNS[3].finditer(text):
        section_starts.append({
            "pos": match.start(),
            "section_number": match.group(1).strip(),
            "section_title": _title_from_match(match),
            "type": "article",
        })

    # Sort by position
    section_starts.sort(key=lambda x: x["pos"])

    # Deduplicate sections at same position
    seen_positions = set()
    unique_starts = []
    for s in section_starts:
        if s["pos"] not in seen_positions:
            seen_positions.add(s["pos"])
            unique_starts.append(s)
    section_starts = unique_starts

    # -----------------------------------------------------------------------
    # Schedule boundary: find the first standalone SCHEDULE heading that
    # appears AFTER the last detected section start, and cap the document
    # there.  Schedules use clause numbers (1., 2., 3.) or table-row numbers
    # that Pattern 2 / Pattern 1B treat as section boundaries, contaminating
    # the index with schedule table rows (e.g. Companies Act Schedule III has
    # 479 numbered clauses that were all labelled S.3).
    #
    # Searching only AFTER the last section start ensures that:
    #   (a) Table-of-contents references to schedules ("THE FIRST SCHEDULE"
    #       listed in the Arrangement of Sections near the top) are never
    #       mistaken for the actual schedule boundary.
    #   (b) Inline references ("see the First Schedule") embedded in section
    #       body text are ignored.
    # -----------------------------------------------------------------------
    _SCHEDULE_HDR = re.compile(
        r"(?:^|\n)[ \t]*(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|"
        r"SEVENTH|EIGHTH|NINTH|TENTH|[IVX]+\.?\s+)?SCHEDULES?\.?\b"
        r"(?:\s+(?:[IVX]+|\d+))?\s*(?:\n|$)",
        re.IGNORECASE,
    )
    schedule_boundary = len(text)
    if section_starts:
        search_from = section_starts[-1]["pos"]
        sched_match = _SCHEDULE_HDR.search(text, pos=search_from)
        if sched_match:
            schedule_boundary = sched_match.start()
            before = len(section_starts)
            section_starts = [s for s in section_starts if s["pos"] < schedule_boundary]
            dropped = before - len(section_starts)
            logger.debug(
                f"Schedule boundary at char {schedule_boundary:,} — "
                f"dropped {dropped} spurious section starts from schedule content in {filename}"
            )

    if not section_starts:
        return _fallback_chunk(text, act_name, filename)

    # Extract text for each section
    for i, sec in enumerate(section_starts):
        start = sec["pos"]
        # Cap last section at schedule boundary so its body text doesn't include
        # the entire schedule blob
        if i + 1 < len(section_starts):
            end = section_starts[i + 1]["pos"]
        else:
            end = min(schedule_boundary, len(text))
        section_text = text[start:end].strip()

        if len(section_text) < 30:
            continue

        chapter = _detect_chapter(text[:start])
        keywords = _extract_keywords(section_text)

        sec_type  = sec["type"]
        sec_num   = sec["section_number"]
        sec_title = sec["section_title"]

        alias_part = f" ({act_alias})" if act_alias else ""
        search_text = (
            f"{act_name}{alias_part} {sec_type.title()} {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{section_text}"
        )

        base_chunk = {
            "chunk_id": f"{_safe_id(act_name)}_{sec_type}_{sec_num}",
            "act_name": act_name,
            "section_number": sec_num,
            "section_title": sec_title,
            "sub_section": "",          # filled in by splitter when applicable
            "chapter": chapter,
            "full_text": section_text,
            "search_text": search_text,
            "keywords": keywords,
            "source_file": os.path.basename(filename),
            "doc_type": "bare_act",
        }

        # Apply sub-section splitter for large sections
        chunks.extend(_maybe_split_chunk(base_chunk))

    logger.info(f"Chunked bare act '{act_name}' into {len(chunks)} sections from {filename}")
    return chunks


def _fallback_chunk(text: str, act_name: str, filename: str) -> list:
    """
    Fallback: split by paragraphs when no sections are detected.
    """
    _MIN_CHUNK = 500
    _MAX_CHUNK = 2000

    paragraphs = re.split(r"\n\s*\n", text)
    act_alias = _act_alias(act_name)
    alias_part = f" ({act_alias})" if act_alias else ""

    chunks = []
    buffer = []
    buf_len = 0
    chunk_idx = 0

    def _flush(buf, idx):
        combined = "\n\n".join(buf).strip()
        if len(combined) < 50:
            return None
        return {
            "chunk_id": f"{_safe_id(act_name)}_para_{idx}",
            "act_name": act_name,
            "section_number": "",
            "section_title": "",
            "sub_section": "",
            "chapter": "",
            "full_text": combined,
            "search_text": f"{act_name}{alias_part}\n\n{combined}",
            "keywords": _extract_keywords(combined),
            "source_file": os.path.basename(filename),
            "doc_type": "bare_act",
        }

    for para in paragraphs:
        para = para.strip()
        if len(para) < 30:
            continue

        if buf_len + len(para) > _MAX_CHUNK and buffer:
            c = _flush(buffer, chunk_idx)
            if c:
                chunks.append(c)
                chunk_idx += 1
            buffer = []
            buf_len = 0

        buffer.append(para)
        buf_len += len(para)

        if buf_len >= _MIN_CHUNK:
            c = _flush(buffer, chunk_idx)
            if c:
                chunks.append(c)
                chunk_idx += 1
            buffer = []
            buf_len = 0

    if buffer:
        c = _flush(buffer, chunk_idx)
        if c:
            chunks.append(c)

    return chunks


# ---------------------------------------------------------------------------
# Case Law Paragraph-Level Chunking
# ---------------------------------------------------------------------------

_CASE_NAME_PATTERN = re.compile(
    r"([A-Z][a-zA-Z\s\.]+)\s+(?:v\.?s?\.?|versus)\s+([A-Z][a-zA-Z\s\.]+)",
    re.IGNORECASE,
)

_COURT_PATTERNS = {
    "Supreme Court of India": re.compile(r"supreme\s+court", re.IGNORECASE),
    "High Court": re.compile(r"high\s+court", re.IGNORECASE),
    "District Court": re.compile(r"district\s+(?:court|judge)", re.IGNORECASE),
}

_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Expanded citation patterns (v3)
_CITATION_PATTERN = re.compile(
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+"           # (2022) 5 SCC 123
    r"|\d{4}\s*\(\d+\)\s*SCC\s*\d+"           # 2022 (5) SCC 123
    r"|AIR\s*\d{4}\s*SC\s*\d+"                # AIR 2022 SC 123
    r"|\d{4}\s*SCC\s*\(Cri\)\s*\d+"           # 2022 SCC (Cri) 123
    r"|\d{4}\s+INSC\s+\d+"                    # 2024 INSC 506
    r"|SCC\s*Online\s*(?:SC|HC)?\s*\d+"       # SCC Online SC 123
    r"|ILR\s*\d{4}\s*\w+\s*\d+"              # ILR 2022 SC 123
    r"|MANU/\w+/\d+/\d+",                     # MANU/SC/0123/2022
    re.IGNORECASE,
)

_PARA_NUM_PATTERN = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)

# ---------------------------------------------------------------------------
# Case-law paragraph size limiter
# ---------------------------------------------------------------------------

# Sentence boundary: period/!? preceded by ≥3 lowercase letters (excludes
# single-letter abbreviations like "J.", "v.", "S.") followed by whitespace
# and an uppercase letter or opening parenthesis.
_SENT_BOUNDARY = re.compile(r"(?<=[a-z]{3}[.!?])\s+(?=[A-Z\(])")

# Hard limit for a single case-law chunk (chars).  Matches ~512 BERT tokens.
_CASE_CHUNK_LIMIT = 1800


def _split_para_to_limit(text: str, para_label: str) -> list:
    """
    Split a case-law paragraph into chunks of at most _CASE_CHUNK_LIMIT chars,
    breaking at sentence boundaries where possible.

    Labelling scheme (matching user spec):
      - first sub-chunk  → para_label          (e.g. "12")
      - second sub-chunk → para_label + "_1"   (e.g. "12_1")
      - third sub-chunk  → para_label + "_2"   (e.g. "12_2")
      ...

    Two-level split:
      Level 1: split at sentence boundaries (SENT_BOUNDARY regex).
      Level 2: if a single "sentence" still exceeds the limit (e.g. a long
               quoted block), hard-split at the nearest whitespace before the
               limit so no chunk ever escapes the cap.
    """
    if len(text) <= _CASE_CHUNK_LIMIT:
        return [(para_label, text)]

    # Level-1: sentence boundary split
    raw_sentences = _SENT_BOUNDARY.split(text)

    # Level-2: hard-split any sentence that is itself oversized
    sentences: list[str] = []
    for sent in raw_sentences:
        while len(sent) > _CASE_CHUNK_LIMIT:
            # Split at last whitespace before the limit
            cut = sent.rfind(" ", 0, _CASE_CHUNK_LIMIT)
            if cut == -1:
                cut = _CASE_CHUNK_LIMIT      # no whitespace — hard cut
            sentences.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            sentences.append(sent)

    # Accumulate sentences into ≤ _CASE_CHUNK_LIMIT chunks
    result: list[tuple[str, str]] = []
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

# Known Indian state names for court-name garble correction
_INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Calcutta", "Bombay", "Madras", "Allahabad",
    "Patna", "Hyderabad", "Lucknow", "Chandigarh",
]
_STATE_LOOKUP = {s.lower(): s for s in _INDIAN_STATES}


def _sanitise_court_name(raw: str) -> str:
    """
    Clean up OCR-garbled court names.
    e.g. "High Court of Guatemala"        → "High Court of Gujarat"
         "High Court of Gujarat AT Hyderabad" → "High Court of Gujarat"
         "High Court of JUDICATURE AT Madras" → "High Court of Madras"
    Uses simple character-distance matching against known state/city names.
    """
    raw = raw.strip()
    # Collapse whitespace and newlines
    raw = re.sub(r"[\n\r\t]+", " ", raw).strip()
    raw = re.sub(r"\s+", " ", raw)
    # Strip "JUDICATURE"/"JUDICDATURE" OCR artifacts FIRST so that
    # "JUDICATURE AT Madras" becomes "AT Madras" before the AT-truncation step.
    raw = re.sub(r"\bJUDIC[A-Z]+\b\s*", "", raw, flags=re.IGNORECASE).strip()
    # Strip leading "AT " left over after JUDICATURE removal
    raw = re.sub(r"^AT\s+", "", raw, flags=re.IGNORECASE).strip()
    # Truncate at bench/location qualifiers that appear after the state name:
    # "AT <city>", "FOR THE STATE OF", "TO DISPOSE", "AND <next-word>", etc.
    raw = re.sub(
        r"\s+(?:AT|FOR\s+THE\s+STATE(?:\s+OF)?|TO\s+\w|AND\s+[A-Z])\b.*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()

    # Check if a word in raw matches a known state (case-insensitive)
    words = raw.split()
    corrected = []
    i = 0
    while i < len(words):
        word = words[i]
        word_lower = word.lower().rstrip(".,;:")
        # Try 1-word match
        if word_lower in _STATE_LOOKUP:
            corrected.append(_STATE_LOOKUP[word_lower])
            i += 1
            continue
        # Try 2-word match (e.g. "Andhra Pradesh")
        if i + 1 < len(words):
            two = (word + " " + words[i + 1]).lower().rstrip(".,;:")
            if two in _STATE_LOOKUP:
                corrected.append(_STATE_LOOKUP[two])
                i += 2
                continue
        # Fuzzy: find closest state by Levenshtein-like char overlap
        best_state = None
        best_score = 0
        for state_lower, state_name in _STATE_LOOKUP.items():
            # Simple overlap score: matching chars / max length
            if abs(len(word_lower) - len(state_lower)) > 4:
                continue
            matches = sum(c in state_lower for c in word_lower)
            score = matches / max(len(word_lower), len(state_lower))
            if score > 0.82 and score > best_score:   # 82% char overlap threshold
                best_score = score
                best_state = state_name
        if best_state:
            corrected.append(best_state)
        else:
            corrected.append(word)
        i += 1

    return " ".join(corrected)


def _detect_case_metadata(text: str, filename: str) -> dict:
    """Extract case name, court, year, citation from judgment text (v3)."""
    metadata = {
        "case_name": "",
        "court": "",
        "year": "",
        "citation": "",
        "bench": "",
    }

    scan = text[:5000]  # v3: extended from 3 000 to 5 000

    # Case name (X v. Y)
    match = _CASE_NAME_PATTERN.search(scan)
    if match:
        metadata["case_name"] = f"{match.group(1).strip()} v. {match.group(2).strip()}"

    # Court
    for court_name, pattern in _COURT_PATTERNS.items():
        if pattern.search(scan):
            if court_name == "High Court":
                hc_match = re.search(
                    r"(?:Hon'?ble\s+)?(?:the\s+)?(?:High\s+Court\s+of\s+)([A-Za-z\s]+)",
                    scan,
                    re.IGNORECASE,
                )
                if hc_match:
                    raw_state = hc_match.group(1).strip().split("\n")[0].strip()[:60]
                    metadata["court"] = f"High Court of {_sanitise_court_name(raw_state)}"
                else:
                    metadata["court"] = "High Court"
            else:
                metadata["court"] = court_name
            break

    # Year
    years = _YEAR_PATTERN.findall(scan)
    if years:
        metadata["year"] = years[0]

    # Citation (v3: expanded patterns)
    cite_match = _CITATION_PATTERN.search(text[:6000])
    if cite_match:
        metadata["citation"] = cite_match.group(0).strip()

    # Bench (justices)
    bench_match = re.search(
        r"(?:Bench|Coram|Before)[\s:]+(.+?)(?:\n|$)",
        scan,
        re.IGNORECASE,
    )
    if bench_match:
        metadata["bench"] = bench_match.group(1).strip()[:200]

    # Fallback: derive from filename
    if not metadata["case_name"]:
        name = os.path.splitext(os.path.basename(filename))[0]
        name = name.replace("_", " ").replace("-", " ")
        if " v " in name.lower() or " vs " in name.lower():
            metadata["case_name"] = name.title()

    return metadata


def _determine_binding_authority(court: str) -> str:
    """Classify the binding authority level."""
    court_lower = court.lower()
    if "supreme court" in court_lower:
        return "supreme_court"
    elif "high court" in court_lower:
        return "high_court"
    elif "district" in court_lower or "sessions" in court_lower:
        return "district_court"
    elif "tribunal" in court_lower:
        return "tribunal"
    return "unknown"


def chunk_case_law(text: str, filename: str = "") -> list:
    """
    Split a case law judgment into paragraph-level chunks with metadata.

    Every emitted chunk is capped at _CASE_CHUNK_LIMIT (1 800) chars so the
    full text always fits within the embedding model's 512-token window.

    When a numbered paragraph is split the sub-chunks are labelled:
        para_N, para_N_1, para_N_2, …
    so downstream code can reconstruct the original paragraph if needed.
    """
    metadata = _detect_case_metadata(text, filename)
    binding = _determine_binding_authority(metadata.get("court", ""))

    chunks = []

    def _emit(para_text: str, para_label: str) -> None:
        """Split para_text to limit and append all sub-chunks to `chunks`."""
        for label, chunk_text in _split_para_to_limit(para_text, para_label):
            if len(chunk_text) >= 50:
                chunks.append(_make_case_chunk(
                    chunk_text, metadata, binding, label, filename
                ))

    # ── Path A: numbered paragraphs ("1. …", "2. …") ────────────────────────
    para_splits = _PARA_NUM_PATTERN.split(text)

    if len(para_splits) > 3:
        preamble = para_splits[0].strip()
        if preamble and len(preamble) > 100:
            _emit(preamble, "preamble")

        for i in range(1, len(para_splits) - 1, 2):
            para_num  = para_splits[i].strip()
            para_text = para_splits[i + 1].strip() if i + 1 < len(para_splits) else ""
            if len(para_text) < 50:
                continue
            _emit(para_text, para_num)

    # ── Path B: double-newline paragraphs ────────────────────────────────────
    else:
        paragraphs = re.split(r"\n\s*\n", text)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 80:
                continue
            _emit(para, str(i + 1))

    # ── Path C: last-resort line accumulator ─────────────────────────────────
    if len(chunks) < 3:
        chunks = []
        lines = text.split("\n")
        current_chunk: list[str] = []
        for line in lines:
            current_chunk.append(line)
            joined = "\n".join(current_chunk).strip()
            if len(joined) > _CASE_CHUNK_LIMIT:
                _emit(joined, str(len(chunks) + 1))
                current_chunk = []
        if current_chunk:
            joined = "\n".join(current_chunk).strip()
            if len(joined) > 80:
                _emit(joined, str(len(chunks) + 1))

    logger.info(
        f"Chunked case law '{metadata.get('case_name', filename)}' "
        f"into {len(chunks)} paragraphs"
    )
    return chunks


def _make_case_chunk(
    text: str, metadata: dict, binding: str, para_num: str, filename: str
) -> dict:
    """Create a single case law chunk with full metadata."""
    case_name = (
        (metadata.get("case_name") or "").strip()
        or os.path.basename(filename).replace(".pdf", "")
        or "Judgment"
    )
    court    = metadata.get("court", "")
    year     = metadata.get("year", "")
    citation = metadata.get("citation", "")

    header = f"{case_name}" if case_name else ""
    if citation:
        header += f" ({citation})"
    if court:
        header += f" - {court}"
    if year and year not in header:
        header += f", {year}"

    search_text = f"{header}\n\n{text}" if header else text

    return {
        "chunk_id": f"{_safe_id(case_name or filename)}_para_{para_num}",
        "case_name": case_name,
        "citation": citation,
        "court": court,
        "bench": metadata.get("bench", ""),
        "year": year,
        "paragraph_num": para_num,
        "full_text": text,
        "search_text": search_text,
        "keywords": _extract_keywords(text),
        "binding_authority": binding,
        "source_file": os.path.basename(filename),
        "doc_type": "case_law",
    }


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

_LEGAL_TERMS = {
    "bail", "fir", "arrest", "custody", "remand", "charge sheet", "complaint",
    "petition", "appeal", "revision", "writ", "mandamus", "certiorari",
    "injunction", "specific performance", "breach", "contract", "tort",
    "negligence", "defamation", "fraud", "misrepresentation", "dowry",
    "cruelty", "maintenance", "divorce", "custody", "adoption", "succession",
    "inheritance", "property", "possession", "eviction", "rent", "lease",
    "mortgage", "easement", "partition", "compensation", "damages",
    "limitation", "jurisdiction", "maintainability", "locus standi",
    "res judicata", "estoppel", "natural justice", "fundamental rights",
    "constitutional", "article 14", "article 19", "article 21",
    "section 302", "section 498a", "section 420", "section 138",
    "cheating", "murder", "culpable homicide", "theft", "robbery",
    "land acquisition", "eminent domain", "environmental", "consumer",
    "arbitration", "mediation", "insolvency", "bankruptcy",
}


def _extract_keywords(text: str) -> list:
    """Extract legal keywords from text."""
    text_lower = text.lower()
    found = [term for term in _LEGAL_TERMS if term in text_lower]
    return found[:20]


def _safe_id(name: str) -> str:
    """Convert a name to a safe chunk ID prefix."""
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return safe[:50] if safe else "unknown"


# ---------------------------------------------------------------------------
# Judgment vs Bare Act detection (prevent misclassification)
# ---------------------------------------------------------------------------

def _looks_like_judgment(text: str, filename: str = "") -> bool:
    """
    Return True if the document content looks like a court judgment rather than a bare act.
    """
    if not (text or "").strip():
        return False
    sample = ((text or "").strip() + " " + (filename or "")).lower()
    judgment_indicators = [
        "in the supreme court of india",
        "in the high court of",
        "civil appeal no",
        "criminal appeal no",
        "writ petition",
        "appellant",
        "respondent",
        "petitioner",
        "j u d g m e n t",
        "judgment",
        "judgement",
        "hon'ble",
        "honble",
        "coram",
        "civil appellate jurisdiction",
        "criminal appellate jurisdiction",
    ]
    bare_act_indicators = [
        "bare act",
        "act,",
        "act no.",
        "code,",
        "ordinance",
        "legislative department",
        "indiacode",
        "gazette of india",
        "schedule",
    ]
    j_count = sum(1 for i in judgment_indicators if i in sample)
    a_count = sum(1 for i in bare_act_indicators if i in sample)
    return j_count > a_count


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def process_bare_acts_directory(bare_acts_dir: str) -> list:
    """Process all PDFs in a bare acts directory. Returns list of all chunks."""
    all_chunks = []
    if not os.path.isdir(bare_acts_dir):
        logger.warning(f"Bare acts directory not found: {bare_acts_dir}")
        return all_chunks

    files = [f for f in os.listdir(bare_acts_dir) if f.lower().endswith((".pdf", ".txt"))]
    logger.info(f"Found {len(files)} bare act files in {bare_acts_dir}")

    for filename in sorted(files):
        filepath = os.path.join(bare_acts_dir, filename)
        if filepath.lower().endswith(".pdf"):
            intro_text = extract_text_from_pdf_first_n_pages(filepath, n=2)
        else:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    full = f.read()
            except Exception as e:
                logger.error(f"Failed to read {filepath}: {e}")
                continue
            intro_text = full[:FIRST_TWO_PAGES_CHARS]

        if not intro_text.strip():
            logger.warning(f"Empty text from {filename}, skipping")
            continue
        if _looks_like_judgment(intro_text, filename):
            logger.warning(
                "Skipping '%s': content looks like a court judgment, not a bare act.",
                filename,
            )
            continue

        text = full if filepath.lower().endswith(".txt") else extract_text_from_file(filepath)
        if not text.strip():
            logger.warning(f"Empty text from {filename}, skipping")
            continue

        chunks = chunk_bare_act(text, filepath)
        all_chunks.extend(chunks)

    logger.info(f"Total bare act chunks: {len(all_chunks)}")
    return all_chunks


def process_case_laws_directory(case_laws_dir: str) -> list:
    """
    Process all files in a case laws directory.

    v3: deduplicates chunks on (case_name + para_num + first 200 chars of
    full_text) before returning, eliminating ~1 500 duplicate chunks caused
    by duplicate PDF files in the directory.
    """
    all_chunks = []
    if not os.path.isdir(case_laws_dir):
        logger.warning(f"Case laws directory not found: {case_laws_dir}")
        return all_chunks

    files = [f for f in os.listdir(case_laws_dir) if f.lower().endswith((".pdf", ".txt"))]
    logger.info(f"Found {len(files)} case law files in {case_laws_dir}")

    for filename in sorted(files):
        filepath = os.path.join(case_laws_dir, filename)
        text = extract_text_from_file(filepath)
        if not text.strip():
            logger.warning(f"Empty text from {filename}, skipping")
            continue
        chunks = chunk_case_law(text, filepath)
        all_chunks.extend(chunks)

    # Deduplicate on (case_name, paragraph_num, first-200-chars of full_text)
    seen: set = set()
    deduped: list = []
    dup_count = 0
    for chunk in all_chunks:
        key = (
            chunk.get("case_name", ""),
            chunk.get("paragraph_num", ""),
            chunk.get("full_text", "")[:200],
        )
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(chunk)

    if dup_count:
        logger.info(f"Removed {dup_count} duplicate case law chunks")

    logger.info(f"Total case law chunks (after dedup): {len(deduped)}")
    return deduped
