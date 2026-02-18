"""
Smart Chunker v2 — Section-level chunking for Bare Acts, paragraph-level for Case Laws.

Replaces the old 800-char blind chunking with legally-aware splitting that preserves
complete sections and meaningful metadata.
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
    # "Section 498A." or "Section 498-A." or "Section 498A —"
    re.compile(
        r"^[\s]*(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?(?:-[A-Za-z])?)"
        r"[\.\s—\-:]+(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "498A. Husband or relative..." (just number at start of line)
    re.compile(
        r"^[\s]*(\d+[A-Za-z]?(?:-[A-Za-z])?)[\.\s—\-:]+\s*([A-Z].*?)$",
        re.MULTILINE,
    ),
    # "CHAPTER IV" or "PART II" headers (used for grouping)
    re.compile(
        r"^[\s]*(CHAPTER|PART|SCHEDULE)\s+([IVXLCDM\d]+[\.\s—\-:]*.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Article 21." (for Constitution)
    re.compile(
        r"^[\s]*(?:Article|Art\.?)\s*(\d+[A-Za-z]?)"
        r"[\.\s—\-:]+(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _detect_act_name_from_text(text: str, filename: str) -> str:
    """Try to extract the act name from the document text or filename."""
    # Look for common patterns like "THE INDIAN PENAL CODE, 1860"
    act_pattern = re.compile(
        r"(?:THE\s+)?([A-Z][A-Z\s,]+(?:ACT|CODE|BILL|ORDINANCE|REGULATION)"
        r"(?:\s*,?\s*\d{4})?)",
        re.IGNORECASE,
    )
    match = act_pattern.search(text[:2000])
    if match:
        name = match.group(0).strip()
        # Clean up
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) > 10:
            return name.title()

    # Fall back to filename
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace("_", " ").replace("-", " ")
    # Remove purely numeric tokens
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


def chunk_bare_act(text: str, filename: str = "") -> list:
    """
    Split a bare act into section-level chunks with rich metadata.

    Each chunk = one complete section with:
    - act_name, section_number, section_title, chapter, full_text
    - keywords extracted from content
    """
    act_name = _detect_act_name_from_text(text, filename)
    chunks = []

    # Find all section boundaries
    section_starts = []
    for pattern in _SECTION_PATTERNS[:2]:  # Only section patterns, not chapter
        for match in pattern.finditer(text):
            section_starts.append({
                "pos": match.start(),
                "section_number": match.group(1).strip(),
                "section_title": match.group(2).strip() if match.group(2) else "",
                "type": "section",
            })

    # Also find article patterns (Constitution)
    for match in _SECTION_PATTERNS[3].finditer(text):
        section_starts.append({
            "pos": match.start(),
            "section_number": match.group(1).strip(),
            "section_title": match.group(2).strip() if match.group(2) else "",
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

    if not section_starts:
        # No sections detected — fall back to paragraph-level chunking
        return _fallback_chunk(text, act_name, filename)

    # Extract text for each section
    for i, sec in enumerate(section_starts):
        start = sec["pos"]
        end = section_starts[i + 1]["pos"] if i + 1 < len(section_starts) else len(text)
        section_text = text[start:end].strip()

        # Skip very short sections (likely noise)
        if len(section_text) < 30:
            continue

        # Detect which chapter this section is in
        chapter = _detect_chapter(text[:start])

        # Extract keywords from section text
        keywords = _extract_keywords(section_text)

        sec_type = sec["type"]
        sec_num = sec["section_number"]
        sec_title = sec["section_title"]

        # Build a search-friendly text representation
        search_text = (
            f"{act_name} {sec_type.title()} {sec_num}"
            + (f" - {sec_title}" if sec_title else "")
            + (f" ({chapter})" if chapter else "")
            + f"\n\n{section_text}"
        )

        chunks.append({
            "chunk_id": f"{_safe_id(act_name)}_{sec_type}_{sec_num}",
            "act_name": act_name,
            "section_number": sec_num,
            "section_title": sec_title,
            "chapter": chapter,
            "full_text": section_text,
            "search_text": search_text,
            "keywords": keywords,
            "source_file": os.path.basename(filename),
            "doc_type": "bare_act",
        })

    logger.info(f"Chunked bare act '{act_name}' into {len(chunks)} sections from {filename}")
    return chunks


def _fallback_chunk(text: str, act_name: str, filename: str) -> list:
    """Fallback: split by paragraphs when no sections are detected."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 50:
            continue
        chunks.append({
            "chunk_id": f"{_safe_id(act_name)}_para_{i}",
            "act_name": act_name,
            "section_number": "",
            "section_title": "",
            "chapter": "",
            "full_text": para,
            "search_text": f"{act_name}\n\n{para}",
            "keywords": _extract_keywords(para),
            "source_file": os.path.basename(filename),
            "doc_type": "bare_act",
        })
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

_CITATION_PATTERN = re.compile(
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+|\d{4}\s*\(\d+\)\s*SCC\s*\d+"
    r"|AIR\s*\d{4}\s*SC\s*\d+"
    r"|\d{4}\s*SCC\s*\(Cri\)\s*\d+",
    re.IGNORECASE,
)

_PARA_NUM_PATTERN = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)


def _detect_case_metadata(text: str, filename: str) -> dict:
    """Extract case name, court, year, citation from judgment text."""
    metadata = {
        "case_name": "",
        "court": "",
        "year": "",
        "citation": "",
        "bench": "",
    }

    # Case name (X v. Y)
    match = _CASE_NAME_PATTERN.search(text[:3000])
    if match:
        metadata["case_name"] = f"{match.group(1).strip()} v. {match.group(2).strip()}"

    # Court
    for court_name, pattern in _COURT_PATTERNS.items():
        if pattern.search(text[:3000]):
            # Try to be more specific
            if court_name == "High Court":
                hc_match = re.search(
                    r"(?:Hon'?ble\s+)?(?:the\s+)?(?:High\s+Court\s+of\s+)([A-Za-z\s]+)",
                    text[:3000],
                    re.IGNORECASE,
                )
                if hc_match:
                    metadata["court"] = f"High Court of {hc_match.group(1).strip()}"
                else:
                    metadata["court"] = "High Court"
            else:
                metadata["court"] = court_name
            break

    # Year
    years = _YEAR_PATTERN.findall(text[:3000])
    if years:
        metadata["year"] = years[0]

    # Citation
    cite_match = _CITATION_PATTERN.search(text[:5000])
    if cite_match:
        metadata["citation"] = cite_match.group(0).strip()

    # Bench (justices)
    bench_match = re.search(
        r"(?:Bench|Coram|Before)[\s:]+(.+?)(?:\n|$)",
        text[:5000],
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

    Each chunk = one meaningful paragraph/section with:
    - case_name, court, year, citation, bench, binding_authority
    - paragraph_num, legal_principle extracted
    """
    metadata = _detect_case_metadata(text, filename)
    binding = _determine_binding_authority(metadata.get("court", ""))

    chunks = []

    # Split by numbered paragraphs (e.g., "1. ...", "2. ...")
    para_splits = _PARA_NUM_PATTERN.split(text)

    if len(para_splits) > 3:
        # We have numbered paragraphs
        # para_splits = [preamble, num1, text1, num2, text2, ...]
        preamble = para_splits[0].strip()
        if preamble and len(preamble) > 100:
            chunks.append(_make_case_chunk(
                preamble, metadata, binding, "preamble", filename
            ))

        for i in range(1, len(para_splits) - 1, 2):
            para_num = para_splits[i].strip()
            para_text = para_splits[i + 1].strip() if i + 1 < len(para_splits) else ""
            if len(para_text) < 50:
                continue
            chunks.append(_make_case_chunk(
                para_text, metadata, binding, para_num, filename
            ))
    else:
        # No numbered paragraphs — split by double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 80:
                continue

            # Merge very short consecutive paragraphs
            chunks.append(_make_case_chunk(
                para, metadata, binding, str(i + 1), filename
            ))

    # If very few chunks, try splitting by single newlines with min length
    if len(chunks) < 3:
        chunks = []
        lines = text.split("\n")
        current_chunk = []
        for line in lines:
            current_chunk.append(line)
            joined = "\n".join(current_chunk).strip()
            if len(joined) > 500:
                chunks.append(_make_case_chunk(
                    joined, metadata, binding, str(len(chunks) + 1), filename
                ))
                current_chunk = []
        if current_chunk:
            joined = "\n".join(current_chunk).strip()
            if len(joined) > 80:
                chunks.append(_make_case_chunk(
                    joined, metadata, binding, str(len(chunks) + 1), filename
                ))

    logger.info(
        f"Chunked case law '{metadata.get('case_name', filename)}' "
        f"into {len(chunks)} paragraphs"
    )
    return chunks


def _make_case_chunk(
    text: str, metadata: dict, binding: str, para_num: str, filename: str
) -> dict:
    """Create a single case law chunk with full metadata."""
    case_name = (metadata.get("case_name") or "").strip() or os.path.basename(filename).replace(".pdf", "") or "Judgment"
    court = metadata.get("court", "")
    year = metadata.get("year", "")
    citation = metadata.get("citation", "")

    # Build search-friendly text
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

# Common Indian legal terms for keyword extraction
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
    found = []
    for term in _LEGAL_TERMS:
        if term in text_lower:
            found.append(term)
    return found[:20]  # cap to avoid noise


def _safe_id(name: str) -> str:
    """Convert a name to a safe chunk ID prefix."""
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return safe[:50] if safe else "unknown"


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def process_bare_acts_directory(bare_acts_dir: str) -> list:
    """Process all PDFs in a bare acts directory. Returns list of all chunks."""
    all_chunks = []
    if not os.path.isdir(bare_acts_dir):
        logger.warning(f"Bare acts directory not found: {bare_acts_dir}")
        return all_chunks

    files = [
        f for f in os.listdir(bare_acts_dir)
        if f.lower().endswith((".pdf", ".txt"))
    ]
    logger.info(f"Found {len(files)} bare act files in {bare_acts_dir}")

    for filename in sorted(files):
        filepath = os.path.join(bare_acts_dir, filename)
        text = extract_text_from_file(filepath)
        if not text.strip():
            logger.warning(f"Empty text from {filename}, skipping")
            continue
        chunks = chunk_bare_act(text, filepath)
        all_chunks.extend(chunks)

    logger.info(f"Total bare act chunks: {len(all_chunks)}")
    return all_chunks


def process_case_laws_directory(case_laws_dir: str) -> list:
    """Process all files in a case laws directory. Returns list of all chunks."""
    all_chunks = []
    if not os.path.isdir(case_laws_dir):
        logger.warning(f"Case laws directory not found: {case_laws_dir}")
        return all_chunks

    files = [
        f for f in os.listdir(case_laws_dir)
        if f.lower().endswith((".pdf", ".txt"))
    ]
    logger.info(f"Found {len(files)} case law files in {case_laws_dir}")

    for filename in sorted(files):
        filepath = os.path.join(case_laws_dir, filename)
        text = extract_text_from_file(filepath)
        if not text.strip():
            logger.warning(f"Empty text from {filename}, skipping")
            continue
        chunks = chunk_case_law(text, filepath)
        all_chunks.extend(chunks)

    logger.info(f"Total case law chunks: {len(all_chunks)}")
    return all_chunks
