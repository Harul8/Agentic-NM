"""
core/chunker.py — Smart section/paragraph-level chunker for bare acts and case laws.
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

# Editorial noise patterns to strip from bare-act section text (not legal content)
_BARE_ACT_NOISE_PATTERNS = [
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
_BARE_ACT_PAGE_LINE = re.compile(r"^(?:\s*Page\s+)?\d{1,4}\s*$", re.MULTILINE)


def _strip_bare_act_editorial_noise(text: str) -> str:
    """Remove amendment notices, gazette refs, footnote markers, page-only lines."""
    if not (text or "").strip():
        return text or ""
    out = text
    for pat in _BARE_ACT_NOISE_PATTERNS:
        out = pat.sub(" ", out)
    out = _BARE_ACT_PAGE_LINE.sub("", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return out.strip()


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

# Hard limit for a single bare-act chunk (chars).  We keep this aligned with
# the case-law limit so that every chunk (section or paragraph) is <= 1 800.
_BARE_ACT_CHUNK_LIMIT = 1800


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
_HARD_SPLIT_PARA_TARGET = 1800  # aim for ~1.8 k char paragraphs in fallback


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


def _split_bareact_text_to_limit(chunk: dict, limit: int = _BARE_ACT_CHUNK_LIMIT) -> list[dict]:
    """
    Split a bare-act section/sub-section chunk into <=limit-char parts,
    preferring to cut at the latest newline before the limit, then at
    whitespace, finally hard-cutting when needed.
    """
    text = (chunk.get("full_text") or "").strip()
    if not text or len(text) <= limit:
        return [chunk]

    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        parts.append(remaining.strip())

    if not parts:
        return [chunk]

    act_name  = chunk.get("act_name", "")
    act_alias = _act_alias(act_name)
    alias_part = f" ({act_alias})" if act_alias else ""
    sec_num   = chunk.get("section_number", "")
    sec_title = chunk.get("section_title", "")
    chapter   = chunk.get("chapter", "")
    source    = chunk.get("source_file", "")
    base_id   = chunk.get("chunk_id") or f"{_safe_id(act_name)}_section_{sec_num}"
    base_sub  = (chunk.get("sub_section") or "").strip()

    result: list[dict] = []
    for idx, part_text in enumerate(parts):
        if idx == 0:
            sub_label = base_sub
            suffix = ""
        else:
            suffix = f"_P{idx}"
            sub_label = f"{base_sub}_{idx}" if base_sub else f"para_{idx}"
        search_text = (
            f"{act_name}{alias_part} Section {sec_num}"
            + (f" — {sec_title}" if sec_title else "")
            + (f" {sub_label}" if sub_label else "")
            + (f" [{chapter}]" if chapter else "")
            + f"\n\n{part_text}"
        )
        new_chunk = {
            "chunk_id":      f"{base_id}{suffix}",
            "act_name":      act_name,
            "section_number": sec_num,
            "section_title":  sec_title,
            "sub_section":    sub_label,
            "chapter":        chapter,
            "full_text":      part_text,
            "search_text":    search_text,
            "keywords":       _extract_keywords(part_text),
            "source_file":    source,
            "doc_type":       "bare_act",
        }
        result.append(new_chunk)
    return result

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
        # For each structured sub-chunk, ensure it does not exceed
        # _BARE_ACT_CHUNK_LIMIT by splitting at newline/whitespace where needed.
        final: list[dict] = []
        for sc in sub_chunks:
            text_len = len(sc.get("full_text", "") or "")
            if text_len > _BARE_ACT_CHUNK_LIMIT:
                final.extend(_split_bareact_text_to_limit(sc, limit=_BARE_ACT_CHUNK_LIMIT))
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
        section_text = _strip_bare_act_editorial_noise(section_text)

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

# Narrative words that indicate a false-positive case name (e.g. "he saw the versus undergo")
_CASE_NAME_NARRATIVE_WORDS = frozenset({
    "near", "saw", "undergo", "years", "year", "month", "days", "stated",
    "according", "therefore", "however", "wherein", "whereas", "hence",
    "thus", "thereafter", "then", "after", "before", "when", "while",
    "because", "although", "though", "said", "told", "asked", "replied",
})

# Words/phrases that indicate argument headings or non-title text, not a case name
_CASE_NAME_BAD_PHRASES = frozenset({
    "counsel for the", "counsel for", "petitioner", "respondent", "informant",
    "the accused", "the petitioner", "the respondent", "scobserver",
    "writ petition", "wp no", "crl.", "crl ", "civil appeal", "criminal appeal",
})

# Max reasonable length for a single party name in "X v. Y"
_MAX_PARTY_NAME_CHARS = 60

# Header zone: only first N chars used for case name extraction (avoids body/arguments)
_CASE_NAME_HEADER_CHARS = 1800

# Administrative prefixes to strip from case names (identifiers, not party names) — roadmap Rule 2
_ADMIN_PREFIX_PATTERN = re.compile(
    r"^(?:HC_|SC_|WP_|Crl_|W\.P\.\s*No\.?|Crl\.?\s*)\s*",
    re.IGNORECASE,
)

# OCR garbage: reject case names containing these — roadmap Rule 3
_OCR_GARBAGE_PATTERN = re.compile(r"[#;']|\d[\s,.';\"]+[a-z]|[a-z][\s,.';\"]+\d", re.IGNORECASE)


def _strip_admin_prefix(name: str) -> str:
    """Strip leading HC_, SC_, WP_, Crl_ etc. from case name (roadmap: Rule 2)."""
    if not name or not isinstance(name, str):
        return name
    return _ADMIN_PREFIX_PATTERN.sub("", name.strip()).strip()


def _is_likely_bad_case_name(name: str) -> bool:
    """
    Return True if the string looks like OCR garbage, argument headings, or non-case text.
    Used to reject bad case names and cited-case strings for graph quality.
    Roadmap: Rule 1 (valid A v B), Rule 3 (OCR garbage).
    """
    if not name or len(name) < 4:
        return True
    lower = name.lower().strip()
    for phrase in _CASE_NAME_BAD_PHRASES:
        if phrase in lower:
            return True
    # OCR garbage: #, ;, ', digits mixed with punctuation — roadmap Rule 3
    if _OCR_GARBAGE_PATTERN.search(name):
        return True
    # Case numbers / docket numbers (e.g. "6985 and 11988 of 2023")
    if re.search(r"\d{4,}\s+and\s+\d{4,}", lower):
        return True
    if re.search(r"\bwp\s*no\.?\s*\d+|crl\.?\s*\d+", lower):
        return True
    # Address-like (e.g. "2023_Hyderabad-500073 and Others")
    if re.search(r"\d{4}_[A-Za-z]+-\d{5}", lower):
        return True
    # Truncated or single-letter party (e.g. "rosy jacob v ja")
    parts = re.split(r"\s+v\.?\s*|vs\.?\s*|versus\s*", name, flags=re.IGNORECASE, maxsplit=1)
    if len(parts) >= 2:
        left, right = parts[0].strip(), parts[1].strip()
        if len(right) <= 2 or len(left) <= 2:
            return True
    return False


def _first_party_only_from_side(party_side: str) -> str:
    """
    When there are multiple parties on one side (e.g. "A, B and C" or "X & Ors."),
    return only the first full name: take segment before first "," or "(", then strip & Ors.
    Dots remain part of the name (e.g. "Dr. A. B. Rao" is kept intact).
    """
    if not party_side or not isinstance(party_side, str):
        return ""
    cut_idx = len(party_side)
    for ch in [",", "("]:
        idx = party_side.find(ch)
        if idx != -1 and idx < cut_idx:
            cut_idx = idx
    first = party_side[:cut_idx].strip()
    first = re.sub(r"\s*[&,]\s*(Ors\.?|Others?|Anr\.?|Another)\s*$", "", first, flags=re.IGNORECASE).strip()
    return first[: _MAX_PARTY_NAME_CHARS] if first else ""


def _normalize_case_name_for_display(name: str) -> str:
    """
    Normalize case name for storage and display: strip reporter prefixes, unify v./vs/versus,
    collapse spaces, and use only the first full name from each side when multiple parties.
    e.g. "S.C.R. A ABHILASHA v PARKASH" -> "A Abhilasha v Parkash"
    e.g. "A, B and C v X, Y and Z" -> "A v X"
    """
    if not name or not isinstance(name, str):
        return ""
    s = re.sub(r"\s+", " ", name.strip()).strip()
    # Strip leading reporter/source abbreviations
    s = re.sub(r"^(?:S\.C\.R\.|AIR|SCC|SCR)\s*\.?\s*", "", s, flags=re.IGNORECASE).strip()
    # Unify v. / vs / versus to single " v "
    s = re.sub(r"\s+v\.?\s*|vs\.?\s*|versus\s*", " v ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    # When multiple parties on either side, keep only first full name from each
    if " v " in s:
        left, _, right = s.partition(" v ")
        left = _first_party_only_from_side(left.strip()).strip()
        right = _first_party_only_from_side(right.strip()).strip()
        if left and right:
            s = f"{left} v {right}"
    # Simple title-case for party names (capitalise first letter of each word)
    words = s.split()
    out = []
    for w in words:
        if w.upper() == "V" or w.lower() == "v":
            out.append("v")
        elif len(w) > 1:
            out.append(w[0].upper() + w[1:].lower())
        else:
            out.append(w)
    return " ".join(out)[:200]


def _looks_like_narrative_case_name(name: str) -> bool:
    """Return True if the extracted 'case name' looks like narrative text, not party names."""
    if not name or len(name) > 120:
        return True
    lower = name.lower()
    words = set(lower.split())
    if words & _CASE_NAME_NARRATIVE_WORDS:
        return True
    # "X v. Y" — each part should be short and not a full sentence
    if " v. " in name or " v " in name or " vs " in name or " versus " in lower:
        parts = re.split(r"\s+v\.?\s*|vs\.?\s*|versus\s*", name, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) >= 2:
            left, right = parts[0].strip(), parts[1].strip()
            if len(left) > _MAX_PARTY_NAME_CHARS or len(right) > _MAX_PARTY_NAME_CHARS:
                return True
            # Party names are usually 2–5 words (e.g. "State of Bihar", "Ramchandra")
            if len(left.split()) > 8 or len(right.split()) > 8:
                return True
    return False


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
    """
    Extract case name, court, year, citation from judgment text (v3+).

    v4: Prefer title/cover area (first ~1500 chars); reject narrative-looking
    matches; add BETWEEN X AND Y pattern; prefer line with citation.
    """
    metadata = {
        "case_name": "",
        "court": "",
        "year": "",
        "citation": "",
        "bench": "",
    }

    scan = text[:5000]  # Court, year, citation can use slightly more text
    # Header only for case name: avoids "Counsel for the vs Counsel for the" and body text (graph quality)
    title_zone = text[: _CASE_NAME_HEADER_CHARS]

    # --- Case name: only from header; strip admin prefixes, normalize, reject bad/OCR ---
    def _set_case_name(candidate: str) -> bool:
        if not candidate:
            return False
        # Roadmap Rule 2: strip HC_, SC_, WP_, Crl_ before normalizing
        candidate = _strip_admin_prefix(candidate)
        if not candidate or len(candidate) < 10:
            return False
        normalized = _normalize_case_name_for_display(candidate)
        if not normalized or _looks_like_narrative_case_name(normalized) or _is_likely_bad_case_name(normalized):
            return False
        metadata["case_name"] = normalized
        return True

    # 1) BETWEEN ... AND ... (common in Indian judgments) — header only
    between_pat = re.compile(
        r"BETWEEN\s+([A-Z][A-Za-z\s\.]+?)\s+AND\s+([A-Z][A-Za-z\s\.]+?)(?:\s*\.|\s*\(|\s*\n|$)",
        re.IGNORECASE,
    )
    m = between_pat.search(title_zone)
    if m:
        left = _strip_admin_prefix(re.sub(r"\s+", " ", m.group(1)).strip())
        right = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(left) <= _MAX_PARTY_NAME_CHARS and len(right) <= _MAX_PARTY_NAME_CHARS and left and right:
            candidate = f"{left} v. {right}"
            if _set_case_name(candidate):
                pass  # use this

    # 2) X v. / vs. / versus Y — header zone only (no fallback to body); strip admin prefix from raw line
    if not metadata["case_name"]:
        match = _CASE_NAME_PATTERN.search(title_zone)
        if match:
            candidate = f"{match.group(1).strip()} v. {match.group(2).strip()}"
            candidate = _strip_admin_prefix(candidate)
            if candidate and " v " in candidate:
                _set_case_name(candidate)

    # 3) Line containing both " v." and citation — restrict to first 30 lines of header
    if not metadata["case_name"] and _CITATION_PATTERN.search(title_zone):
        lines = title_zone.splitlines()
        for line in lines[:30]:
            line = line.strip()
            if len(line) < 20 or len(line) > 200:
                continue
            if " v." not in line and " v " not in line and " vs " not in line.lower():
                continue
            m = _CASE_NAME_PATTERN.search(line)
            if m:
                candidate = _strip_admin_prefix(f"{m.group(1).strip()} v. {m.group(2).strip()}")
                if candidate and " v " in candidate and _set_case_name(candidate):
                    break

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

    # Year (prefer first 4-digit year in title zone); roadmap Rule 4: sanity 1850–current year
    years = _YEAR_PATTERN.findall(scan)
    if years:
        try:
            y = int(years[0])
            from datetime import date
            current_year = date.today().year
            if 1850 <= y <= current_year:
                metadata["year"] = years[0]
        except (ValueError, TypeError):
            pass

    # Court: if header has HC_/WP/Crl, do not assign Supreme Court (roadmap: court misclassification)
    if metadata["court"] == "Supreme Court of India" and re.search(
        r"\b(HC_|SC_|WP\s*No\.?|W\.P\.|Crl\.?)", title_zone[:1000], re.IGNORECASE
    ):
        metadata["court"] = "High Court"

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

    # Fallback: derive from filename (normalize and reject bad)
    if not metadata["case_name"] and filename:
        name = os.path.splitext(os.path.basename(filename))[0]
        name = name.replace("_", " ").replace("-", " ")
        name = _strip_admin_prefix(name)
        if name and (" v " in name.lower() or " vs " in name.lower()):
            candidate = _normalize_case_name_for_display(name)
            if candidate and not _looks_like_narrative_case_name(candidate) and not _is_likely_bad_case_name(candidate):
                metadata["case_name"] = candidate

    return metadata


def _clean_ocr_noise(text: str) -> str:
    """
    Strip common OCR artifacts from case-law text: standalone page numbers,
    single-letter lines, repeated header/footer lines, excess blank lines.
    """
    if not (text or "").strip():
        return text or ""
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        # Drop lines that are only 1–4 digits (page numbers)
        if re.fullmatch(r"\d{1,4}", s):
            continue
        # Drop single-letter or two-letter lines (OCR noise)
        if len(s) <= 2 and re.match(r"^[A-Za-z]$", s):
            continue
        if len(s) == 2 and s.isalpha():
            continue
        out.append(line)
    # Collapse multiple consecutive blank lines to at most two
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", "\n".join(out))
    return result.strip()


# Heuristic keywords for paragraph-type classification (case law)
_PARA_TYPE_FACTS = re.compile(
    r"\b(?:the\s+)?(?:petitioner|respondent|appellant|plaintiff|defendant)\b"
    r"|stated\s+that|according\s+to\s+(?:the\s+)?(?:petitioner|respondent)"
    r"|facts\s+of\s+the\s+case|brief\s+facts|case\s+of\s+the\s+petitioner"
    r"|in\s+the\s+present\s+case\s+",
    re.IGNORECASE,
)
_PARA_TYPE_ARGUMENTS = re.compile(
    r"\b(?:submitted|contended|argued)\s+that\b|learned\s+counsel"
    r"|it\s+was\s+argued|submission\s+of\s+the\s+(?:petitioner|respondent)",
    re.IGNORECASE,
)
_PARA_TYPE_REASONING = re.compile(
    r"\bwe\s+are\s+of\s+(?:the\s+)?view\b|in\s+our\s+view\b|considering\s+(?:the\s+)?"
    r"|in\s+the\s+light\s+of\b|in\s+view\s+of\s+the\s+above"
    r"|(?:we\s+)?hold\s+that\b|the\s+court\s+held",
    re.IGNORECASE,
)
_PARA_TYPE_RATIO = re.compile(
    r"\bratio\s+decidendi\b|the\s+law\s+is\s+that\b|it\s+is\s+held\b"
    r"|accordingly\s+we\s+hold\b|the\s+court\s+holds\b",
    re.IGNORECASE,
)
_PARA_TYPE_ORDER = re.compile(
    r"\bin\s+the\s+result\b|appeal\s+is\s+(?:accordingly\s+)?(?:allowed|dismissed)"
    r"|petition\s+is\s+(?:allowed|dismissed)|ordered\s+that\b"
    r"|writ\s+petition\s+is\s+(?:allowed|dismissed)",
    re.IGNORECASE,
)


def _classify_paragraph_type(para_text: str, para_label: str, para_index: int, total_paras: int) -> str:
    """
    Classify case-law paragraph as facts | arguments | reasoning | ratio | order | unknown.
    Uses heuristics (keywords + position). Early paragraphs often contain facts.
    """
    if not (para_text or "").strip():
        return "unknown"
    text = para_text[:2000]  # first 2k chars enough for signals
    lower = text.lower()
    is_early = total_paras and para_index < max(3, total_paras // 5)

    if _PARA_TYPE_ORDER.search(text):
        return "order"
    if _PARA_TYPE_RATIO.search(text):
        return "ratio"
    if _PARA_TYPE_REASONING.search(text):
        return "reasoning"
    if _PARA_TYPE_ARGUMENTS.search(text):
        return "arguments"
    if _PARA_TYPE_FACTS.search(text) or (is_early and para_label not in ("preamble", "0", "1")):
        # Early paras without other signals → often facts
        if not _PARA_TYPE_REASONING.search(text) and not _PARA_TYPE_RATIO.search(text):
            return "facts"
    if is_early and ("petitioner" in lower or "respondent" in lower or "stated" in lower):
        return "facts"
    return "unknown"


# Section reference patterns: "Section 307 IPC", "Section 27 of Arms Act", "IPC Section 302"
_SECTION_CITED_PATTERNS = [
    re.compile(r"Section\s+(\d+[A-Za-z]?)\s+(?:of\s+)?([A-Za-z][A-Za-z\s]+?)(?:\s+Act|\s*$)", re.IGNORECASE),
    re.compile(r"Section\s+(\d+[A-Za-z]?)\s+(IPC|BNS|CrPC|CPC|TPA|BNSS|IEA|Evidence\s+Act|Contract\s+Act|NI\s+Act|SRA|HMA|Arms\s+Act|MV\s+Act)\b", re.IGNORECASE),
    re.compile(r"\b(IPC|BNS|CrPC|CPC|TPA|BNSS|IEA|Arms\s+Act|MV\s+Act)\s+Section\s+(\d+[A-Za-z]?)", re.IGNORECASE),
    re.compile(r"Article\s+(\d+[A-Za-z]?)\s+(?:of\s+)?(?:the\s+)?Constitution", re.IGNORECASE),
]


# Reject statute refs that are clearly not act+section (e.g. "the 6", "thesaid 4") — roadmap statute–case linking
_VALID_SECTION_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s+(\d+[A-Za-z]?)$|^Constitution\s+(\d+[A-Za-z]?)$", re.IGNORECASE)
_SECTION_ACT_BLOCKLIST = frozenset({"the", "thesaid", "said", "that", "this", "under", "as", "per", "see"})


def _extract_sections_cited(text: str) -> list:
    """
    Extract statute references from case-law text, e.g. "Section 307 IPC" → "IPC 307".
    Returns list of normalized "ActName SectionNum" for retrieval/filtering.
    Rejects invalid extractions like "the 6", "thesaid 4" (roadmap: statute–case linking).
    """
    if not (text or "").strip():
        return []
    seen = set()
    result = []
    for pat in _SECTION_CITED_PATTERNS:
        for m in pat.finditer(text):
            if pat == _SECTION_CITED_PATTERNS[0]:
                num, act = m.group(1).strip(), m.group(2).strip()
                act_short = act.replace(" ", "")[:20] if act else "Act"
                key = f"{act_short} {num}"
            elif pat == _SECTION_CITED_PATTERNS[1]:
                num, act = m.group(1).strip(), m.group(2).strip().replace(" ", "")
                key = f"{act} {num}"
            elif pat == _SECTION_CITED_PATTERNS[2]:
                act, num = m.group(1).strip().replace(" ", ""), m.group(2).strip()
                key = f"{act} {num}"
            else:
                num = m.group(1).strip()
                key = f"Constitution {num}"
            # Only keep valid act+section shape; reject "the 6", "thesaid 4" and other non-act words
            match = _VALID_SECTION_KEY.match(key) if len(key) >= 4 else None
            if match and key not in seen:
                act_part = (match.group(1) or "").lower()
                if act_part and act_part in _SECTION_ACT_BLOCKLIST:
                    continue
                seen.add(key)
                result.append(key)
    return result[:30]  # cap per chunk


# Pattern to find cited case names in body text (X v Y / X vs Y). Used for citation graph.
_CITED_CASE_PATTERN = re.compile(
    r"(?:in|as\s+held\s+in|following|relying\s+on|see\s+)\s*"
    r"(?:\(?\s*)?"
    r"([A-Z][A-Za-z\s\.]+?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z\s\.]+?)(?:\s*\(?\s*\d{4}\s*\)?)?(?:\s+\d+\s*SCC\s*\d+)?",
    re.IGNORECASE,
)

# Leading phrases to strip from cited-case text (roadmap: citation extraction normalization)
_CITED_STRIP_PREFIXES = (
    "in ", "of this court in ", "of that court in ", "but in a recent decision in ",
    "as held in ", "as observed in ", "following ", "relying on ", "see ", "referring to ",
)

# Reject cited strings that look like sentence fragments (too long or contain these)
_CITED_FRAGMENT_MARKERS = re.compile(
    r"\b(but|however|therefore|cpc\.|in a recent|decision in|wherein|whereas)\b",
    re.IGNORECASE,
)
_CITED_MAX_WORDS = 12  # reasonable "Party1 v Party2" is at most ~8 words total
_CITED_MAX_TOTAL_CHARS = 100  # reject very long extractions


def _normalize_case_id(name: str) -> str:
    """Normalize case name for graph key: strip, collapse spaces, lowercase."""
    if not name or not isinstance(name, str):
        return ""
    s = re.sub(r"\s+", " ", name.strip()).strip()
    return s.lower()[:200]


def _strip_citation_prefix(s: str) -> str:
    """Strip leading citation context words so we get clean 'Party1 v Party2'."""
    if not s or not isinstance(s, str):
        return s
    t = s.strip()
    for prefix in _CITED_STRIP_PREFIXES:
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
            break
    return t.strip()


def _is_citation_fragment(name: str) -> bool:
    """True if the extracted string looks like a sentence fragment, not a case name."""
    if not name or len(name) > _CITED_MAX_TOTAL_CHARS:
        return True
    words = name.split()
    if len(words) > _CITED_MAX_WORDS:
        return True
    if _CITED_FRAGMENT_MARKERS.search(name):
        return True
    return False


def extract_cited_cases(text: str) -> list:
    """
    Extract cited case names from case-law text (e.g. "as held in X v Y (2020)").
    Returns list of normalized "Party1 v Party2" strings for citation graph.
    Rejects truncated, OCR, fragment-like, and non-case strings for graph quality.
    Uses prefix stripping and fragment rejection per roadmap (citation extraction normalization).
    """
    if not (text or "").strip():
        return []
    seen = set()
    result = []

    def _add_cited(name: str) -> None:
        if len(name) < 10:
            return
        # Strip leading "in ", "of this court in ", etc.
        name = _strip_citation_prefix(name)
        if len(name) < 10:
            return
        # Reject sentence fragments and overly long
        if _is_citation_fragment(name):
            return
        # Reject truncated (e.g. "rosy jacob v ja") and bad/OCR
        if _is_likely_bad_case_name(name):
            return
        normalized = _normalize_case_name_for_display(name)
        if not normalized or len(normalized) < 10:
            return
        if _is_citation_fragment(normalized):
            return
        key = _normalize_case_id(normalized)
        if key and key not in seen:
            seen.add(key)
            result.append(normalized)

    for m in _CITED_CASE_PATTERN.finditer(text):
        p1, p2 = m.group(1).strip(), m.group(2).strip()
        # Strip prefixes that may have been captured into party 1
        p1 = _strip_citation_prefix(p1)
        if len(p1) < 2 or len(p2) < 3 or len(p1) > 80 or len(p2) > 80:
            continue
        if len(p1.split()) > 6 or len(p2.split()) > 6:
            continue
        _add_cited(f"{p1} v {p2}")
    # Also catch standalone "X v. Y" / "X vs Y" without leading phrase
    alt = re.compile(
        r"\b([A-Z][A-Za-z\s\.]{2,40}?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z\s\.]{2,40}?)\b",
        re.IGNORECASE,
    )
    for m in alt.finditer(text):
        p1, p2 = m.group(1).strip(), m.group(2).strip()
        p1 = _strip_citation_prefix(p1)
        if len(p2) < 3 or len(p1) < 2:
            continue
        if len(p1.split()) > 6 or len(p2.split()) > 6:
            continue
        _add_cited(f"{p1} v {p2}")
    return result[:25]  # cap per chunk


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

    v4: OCR cleaning, paragraph_type (facts/arguments/reasoning/ratio/order),
    and sections_cited extracted per chunk.
    """
    text = _clean_ocr_noise(text)
    metadata = _detect_case_metadata(text, filename)
    binding = _determine_binding_authority(metadata.get("court", ""))

    chunks = []

    def _emit(para_text: str, para_label: str, para_index: int = 0, total_paras: int = 0) -> None:
        """Split para_text to limit and append all sub-chunks with paragraph_type, sections_cited, cited_cases."""
        ptype = _classify_paragraph_type(para_text, para_label, para_index, total_paras)
        for label, chunk_text in _split_para_to_limit(para_text, para_label):
            if len(chunk_text) >= 50:
                sections_cited = _extract_sections_cited(chunk_text)
                cited_cases = extract_cited_cases(chunk_text)
                chunks.append(_make_case_chunk(
                    chunk_text, metadata, binding, label, filename,
                    paragraph_type=ptype, sections_cited=sections_cited, cited_cases=cited_cases,
                ))

    # ── Path A: numbered paragraphs ("1. …", "2. …") ────────────────────────
    para_splits = _PARA_NUM_PATTERN.split(text)

    if len(para_splits) > 3:
        total_paras = 1 + (len(para_splits) - 1) // 2
        preamble = para_splits[0].strip()
        if preamble and len(preamble) > 100:
            _emit(preamble, "preamble", 0, total_paras)

        for i in range(1, len(para_splits) - 1, 2):
            para_num  = para_splits[i].strip()
            para_text = para_splits[i + 1].strip() if i + 1 < len(para_splits) else ""
            if len(para_text) < 50:
                continue
            para_index = (i + 1) // 2  # 1-based index after preamble
            _emit(para_text, para_num, para_index, total_paras)

    # ── Path B: double-newline paragraphs ────────────────────────────────────
    else:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 80]
        total_paras = len(paragraphs)
        for i, para in enumerate(paragraphs):
            _emit(para, str(i + 1), i, total_paras)

    # ── Path C: last-resort line accumulator ─────────────────────────────────
    if len(chunks) < 3:
        chunks = []
        lines = text.split("\n")
        current_chunk: list[str] = []
        for line in lines:
            current_chunk.append(line)
            joined = "\n".join(current_chunk).strip()
            if len(joined) > _CASE_CHUNK_LIMIT:
                _emit(joined, str(len(chunks) + 1), 0, 0)
                current_chunk = []
        if current_chunk:
            joined = "\n".join(current_chunk).strip()
            if len(joined) > 80:
                _emit(joined, str(len(chunks) + 1), 0, 0)

    logger.info(
        f"Chunked case law '{metadata.get('case_name', filename)}' "
        f"into {len(chunks)} paragraphs"
    )
    return chunks


def _make_case_chunk(
    text: str, metadata: dict, binding: str, para_num: str, filename: str,
    paragraph_type: str = "unknown", sections_cited: list = None, cited_cases: list = None,
) -> dict:
    """Create a single case law chunk with full metadata."""
    if sections_cited is None:
        sections_cited = []
    if cited_cases is None:
        cited_cases = []
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
        "paragraph_type": paragraph_type,
        "sections_cited": list(sections_cited),
        "cited_cases": list(cited_cases),
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
