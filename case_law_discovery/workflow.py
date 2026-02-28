"""
Case law discovery workflow: dynamic (LLM-driven) except hardcoded limits.

Uses existing modules (LLM, config, vector store read, tiered_search, hybrid_retriever)
without modifying them. Only limits in limits.py are fixed.
"""

import json
import logging
import math
import os
import re
import sys
import time
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level locks for parallel act processing (Fix 5)
# ---------------------------------------------------------------------------
# Protects bare-act summary file writes and summary index file writes
_file_write_lock = threading.Lock()
# Protects the shared index_signatures set across parallel threads
_signatures_lock = threading.Lock()

# Project root for imports
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from case_law_discovery.limits import (
    SIMILARITY_THRESHOLD_LOW,
    SIMILARITY_THRESHOLD_HIGH,
    TOP_N_STATEMENT,
    TOP_N_PER_ACT,
    BEST_PER_CHUNK,
    MAX_FETCH_PER_ACT,
    ACT_SUMMARY_MAX_CHARS,
    CASE_LAW_SUMMARY_MAX_CHARS,
    TOP_N_ACT_NAMED,
    ACT_NAMED_FETCH_BUFFER,
)
from case_law_discovery.store import (
    load_pending,
    save_pending,
    load_summary_index,
    save_summary_index,
    get_bare_act_summary,
    set_bare_act_summary,
    load_bare_act_summary_index,
)

# Approximate first two pages for classification and signature (match auto_enricher)
_FIRST_TWO_PAGES_CHARS = 3000

# ---------------------------------------------------------------------------
# Keyword extraction from raw bare act chunks (used for clean web search queries)
# ---------------------------------------------------------------------------

_LEGAL_STOP_WORDS = frozenset({
    "the", "this", "that", "act", "any", "all", "for", "such", "where", "when",
    "which", "who", "shall", "may", "not", "has", "have", "india", "indian",
    "government", "state", "central", "every", "person", "persons", "order",
    "section", "article", "clause", "rule", "schedule", "provided", "under",
    "made", "and", "or", "by", "to", "in", "of", "on", "at", "be", "is", "are",
    "was", "were", "been", "being", "with", "from", "as", "an", "a", "its",
    "their", "them", "these", "those", "also", "other", "another", "each",
    "above", "below", "herein", "thereof", "thereto", "thereunder", "hereinafter",
    "following", "prescribed", "applicable", "specified", "aforesaid",
})


def _extract_act_keywords(act_name: str, chunks: list | None = None) -> str:
    """
    Extract 4-5 legally meaningful keywords directly from bare act vector store chunks.
    Deterministic — no LLM call, no markdown. Returns a space-joined keyword string
    suitable for use as a supplementary DDG/IK search query.

    Priority: defined terms ("X" means...) first, then capitalized noun phrases.
    Falls back to empty string if chunks are unavailable.
    """
    # Load chunks from vector store if not provided
    if not chunks:
        try:
            from config import BARE_CHUNKS_V2
            import json as _json
            if os.path.isfile(BARE_CHUNKS_V2):
                with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as _f:
                    _all = _json.load(_f)
                if isinstance(_all, dict):
                    _all = list(_all.values())
                act_norm = act_name.lower().strip()
                chunks = [
                    c for c in _all
                    if (c.get("act_name") or c.get("source") or "").lower().strip() == act_norm
                ][:20]
        except Exception:
            pass

    if not chunks:
        return ""

    # Combine text from first 20 chunks (500 chars each to keep it light)
    text = " ".join((_chunk_text(c) or "")[:500] for c in chunks[:20])
    if not text.strip():
        return ""

    # 1. Extract defined terms: "Term" means / includes / refers
    defined = []
    for m in re.finditer(r'"([A-Za-z][a-zA-Z\s\-]{2,40})"', text):
        term = m.group(1).strip()
        end = m.end()
        snippet = text[end:end + 60].lower()
        if any(kw in snippet for kw in (" means", " includes", " refers", " denotes")):
            defined.append(term)
    # Also single-quoted
    for m in re.finditer(r"'([A-Z][a-zA-Z\s\-]{2,40})'", text):
        term = m.group(1).strip()
        end = m.end()
        snippet = text[end:end + 60].lower()
        if any(kw in snippet for kw in (" means", " includes", " refers", " denotes")):
            defined.append(term)

    # 2. Extract capitalized noun phrases (2-3 words, e.g. "Welfare Board", "Unpaid Accumulations")
    noun_phrases = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})\b', text)

    # Build candidate list — defined terms first (most specific), then noun phrases
    candidates = []
    seen = set()
    for term in defined + noun_phrases:
        term = term.strip()
        if not term or len(term) < 4:
            continue
        words = term.lower().split()
        # Skip if every word is a stop word
        if all(w in _LEGAL_STOP_WORDS for w in words):
            continue
        norm = " ".join(words)
        if norm in seen:
            continue
        seen.add(norm)
        candidates.append(term)

    # Take top 5 candidates, total keyword string ≤ 80 chars
    keywords = []
    total = 0
    for term in candidates[:20]:
        if total + len(term) + 1 > 80:
            break
        keywords.append(term)
        total += len(term) + 1

    return " ".join(keywords[:5])


# ---------------------------------------------------------------------------
# P3: Section-anchored score_query
# ---------------------------------------------------------------------------

# Words that signal operative/substantive legal content in a section
_OPERATIVE_WORDS = frozenset({
    "shall", "punishable", "liable", "offence", "offense", "penalty",
    "imprisonment", "fine", "cognizable", "bailable", "non-bailable",
    "compoundable", "warrant", "summon", "arrest", "conviction",
    "means", "includes", "definitions", "defined",
    "right", "duty", "obligation", "entitled", "prohibited", "unlawful",
    "void", "voidable", "enforceable", "prescribed", "compensation",
    "damages", "forfeiture", "disqualification", "cancellation",
})


def _score_chunk_substantiveness(chunk: dict) -> int:
    """
    Score a chunk by how many operative legal words appear in its full_text.
    Higher = more substantively significant for a score_query.
    """
    text = (_chunk_text(chunk) or "").lower()
    return sum(1 for w in _OPERATIVE_WORDS if w in text)


def _build_section_anchored_query(
    act_name: str,
    chunks: list | None,
    top_n: int = 5,
    snippet_chars: int = 150,
) -> str:
    """
    Build a focused cross-encoder query from the most legally significant sections
    of the act (P3 improvement). Instead of the broad act summary, picks the top
    `top_n` sections by operative-word density and includes their section number,
    title, and a short opening snippet.

    Format:
        "{act_name} | Sec {N} {title}: {snippet} | Sec {M} {title}: {snippet} ..."

    Falls back to empty string if chunks are unavailable (caller should fall back
    to act_name + act_summary).
    """
    # Load chunks from vector store if not provided
    if not chunks:
        try:
            from config import BARE_CHUNKS_V2
            import json as _json
            if os.path.isfile(BARE_CHUNKS_V2):
                with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as _f:
                    _all = _json.load(_f)
                if isinstance(_all, dict):
                    _all = list(_all.values())
                act_norm = act_name.lower().strip()
                chunks = [
                    c for c in _all
                    if (c.get("act_name") or c.get("source") or "").lower().strip() == act_norm
                ]
        except Exception:
            pass

    if not chunks:
        return ""

    # Score every chunk and pick top_n
    scored = sorted(chunks, key=_score_chunk_substantiveness, reverse=True)
    top_chunks = scored[:top_n]

    if not top_chunks:
        return ""

    parts = [act_name]
    for c in top_chunks:
        sec_num = (c.get("section_number") or "").strip()
        sec_title = (c.get("section_title") or "").strip()
        full_text = (_chunk_text(c) or "").strip()
        # Build a compact label: "Sec 498A Cruelty by husband: <first 150 chars>"
        label_parts = []
        if sec_num:
            label_parts.append(f"Sec {sec_num}")
        if sec_title:
            label_parts.append(sec_title)
        label = " ".join(label_parts)
        snippet = full_text[:snippet_chars].replace("\n", " ").strip()
        if label and snippet:
            parts.append(f"{label}: {snippet}")
        elif label:
            parts.append(label)
        elif snippet:
            parts.append(snippet)

    query = " | ".join(parts)
    # Cross-encoder works best under ~512 tokens (~2000 chars); trim if needed
    return query[:2000]


# ---------------------------------------------------------------------------
# P1: Section-anchored IK search queries
# ---------------------------------------------------------------------------

def _build_section_ik_queries(act_name: str, chunks: list | None, top_n: int = 3) -> list[str]:
    """
    Build additional Indian Kanoon search queries that include the most operative
    section numbers from the act's chunks.

    Example: "Indian Penal Code" → [
        "Indian Penal Code section 302 Punishment for murder",
        "Indian Penal Code section 498A Cruelty by husband",
        "Indian Penal Code section 376 Rape",
    ]

    These section-specific queries surface judgments that cite the exact provision,
    rather than relying on the act name alone which may return administrative results.
    """
    if not chunks:
        return []

    # Score chunks and take top_n most operative
    scored = sorted(chunks, key=_score_chunk_substantiveness, reverse=True)
    queries = []
    seen_secs = set()
    for c in scored:
        if len(queries) >= top_n:
            break
        sec_num = (c.get("section_number") or "").strip()
        if not sec_num or sec_num in seen_secs:
            continue
        seen_secs.add(sec_num)
        sec_title = (c.get("section_title") or "").strip()
        # Build focused IK query: "Act section N Title"
        parts = [act_name, "section", sec_num]
        if sec_title:
            # First 4 words of title to keep query short
            title_words = sec_title.split()[:4]
            parts.append(" ".join(title_words))
        queries.append(" ".join(parts)[:200])

    return queries


# ---------------------------------------------------------------------------
# P2: Judgment-type filter — reject interlocutory/procedural documents
# ---------------------------------------------------------------------------

_RE_INTERLOCUTORY = re.compile(
    r'\b('
    r'interlocutory\s+application'
    r'|i\.a\.\s*(?:no\.?\s*)?\d'
    r'|i\.a\.\s+\w'          # "I.A. filed"
    r'|office\s+report'
    r'|listing\s+order'
    r'|defect\s+(?:no\.?|notice)'
    r'|show\s+cause\s+notice'
    r'|writ\s+of\s+summons'
    r'|chamber\s+summons'
    r'|master\s+summons'
    r'|contempt\s+notice'
    r'|registered\s+notice'
    r'|office\s+objection'
    r'|this\s+is\s+not\s+a\s+judgment'
    r'|suit\s+summons'
    r'|summons\s+for\s+judgment'
    r'|adjournment\s+(?:order|application)'
    r'|returnable\s+(?:on|before)'   # summons language
    r'|interim\s+(?:stay|injunction)\s+(?:application|motion)'
    r')',
    re.IGNORECASE,
)

# Positive signals that strongly suggest a final judgment
_RE_FINAL_SIGNALS = re.compile(
    r'\b('
    r'(?:this\s+)?(?:appeal|petition|suit|revision|writ\s+petition)\s+is\s+(?:allowed|dismissed|disposed)'
    r'|for\s+the\s+(?:above\s+)?reasons.*judgment\s+is'
    r'|accordingly\s+(?:the\s+)?(?:appeal|petition|suit)\s+(?:is\s+)?(?:allowed|dismissed)'
    r'|we\s+(?:therefore\s+)?(?:allow|dismiss|affirm|set\s+aside)\s+the'
    r'|in\s+the\s+result.*(?:allowed|dismissed)'
    r'|operative\s+part\s+of\s+the\s+(?:order|judgment)'
    r')',
    re.IGNORECASE,
)


def _is_final_judgment(title: str, first_two_pages: str) -> bool:
    """
    Return True if the document looks like a final judgment or substantive order,
    False if it appears to be an interlocutory order, summons, notice, or office report.

    Uses a two-step check:
    1. If clear interlocutory/procedural markers are present → reject.
    2. Otherwise accept (we prefer false-negatives to false-positives here).
    """
    combined = (title + " " + first_two_pages[:1200]).lower()
    if _RE_INTERLOCUTORY.search(combined):
        return False
    return True


# ---------------------------------------------------------------------------
# P5: Court-tier pre-sort for IK TIDs
# ---------------------------------------------------------------------------

def _ik_court_tier(doc: dict) -> int:
    """
    Return court tier integer for sorting IK TIDs before document fetch:
        0 = Supreme Court  (highest priority)
        1 = High Court
        2 = Other / unknown

    Uses the 'docsource' field returned by Indian Kanoon search API.
    """
    ds = (doc.get("docsource") or "").lower()
    if "supreme court" in ds or ds.strip() in ("sc", "supremecourt"):
        return 0
    if "high court" in ds or " hc" in ds or ds.endswith("hc") or "highcourt" in ds:
        return 1
    return 2


def is_case_law_discovery_request(message: str) -> bool:
    """
    Lightweight check: should this message be handled by the case law discovery workflow
    instead of the main chat pipeline? No LLM call.
    """
    if not message or not isinstance(message, str):
        return False
    lower = message.lower().strip()
    # Explicit mention of "case law discovery" always routes to this workflow
    if "case law discovery" in lower:
        return True
    # "find case laws for the first bare act in vector store" / "first bare act in our database" etc.
    has_first_act = ("first" in lower and ("bare act" in lower or "bare acts" in lower or ("act" in lower and ("vector" in lower or "database" in lower))))
    has_vector_store = "vector store" in lower or "vectorstore" in lower or ("database" in lower and ("bare act" in lower or "act" in lower))
    has_case_law = "case law" in lower or "case laws" in lower
    if has_case_law and (has_vector_store or ("first" in lower and "act" in lower)):
        return True
    if has_first_act and (has_vector_store or has_case_law):
        return True
    return False


def _ask_llm(prompt: str, system: str | None = None) -> str:
    """Call existing LLM (Ollama) without changing existing modules."""
    try:
        from llm.ollama_client import ask_llm
        full = (system + "\n\n" + prompt) if system else prompt
        out = ask_llm(full)
        return (out or "").strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return ""


def _line_looks_like_act_title(line: str) -> bool:
    """True if line looks like an act/sanhita/code/constitution title (for extraction or topic check)."""
    if not line or len(line.strip()) < 10:
        return False
    lower = line.strip().lower()
    if re.match(r"^(find|get|case law|discovery|for the|below|following)", lower):
        return False
    # Must contain at least one act-like keyword or a 4-digit year
    has_keyword = any(kw in lower for kw in ("act", "sanhita", "code", "constitution", "ordinance", "regulation"))
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", line))
    if has_year and len(lower) > 15:
        return True
    if has_keyword and (has_year or len(lower) > 20):
        return True
    return False


def _extract_act_names_from_message(user_message: str) -> list[str]:
    """
    Extract act name(s) directly from the user message so we don't rely on LLM
    returning the correct act. Accepts lines that look like act/sanhita/code/constitution
    titles (with or without year), e.g. 'The Indian Evidence Act, 1872', 'Bharatiya Nagarik
    Suraksha Sanhita, 2023', 'The Code on Wages, 2019', 'THE CONSTITUTION OF INDIA'.
    """
    if not user_message or not user_message.strip():
        return []
    candidates = []
    # Split by newlines and by common intro separators
    parts = re.split(r"\n+|below acts|following acts|:\s*", user_message.strip(), flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not _line_looks_like_act_title(part):
            continue
        candidates.append(part)
    return candidates


def _parse_act_range_from_message(user_message: str) -> tuple[int, int] | None:
    """
    Parse act range patterns from user message without an LLM call.
    Returns (act_start, act_end) as 1-based integers, or None if no range found.

    Understands patterns like:
      "acts 32 to 78"         → (32, 78)
      "acts from 32 to 78"    → (32, 78)
      "act numbers 32-78"     → (32, 78)
      "first 10 acts"         → (1, 10)
      "first 5 bare acts"     → (1, 5)
    """
    msg = user_message.lower()
    # Explicit range: "acts 32 to 78", "acts from 32 to 78", "act numbers 32-78", "32 to 78"
    m = re.search(r'acts?\s+(?:from\s+)?(?:number[s]?\s+)?(\d+)\s*(?:to|-)\s*(\d+)', msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "numbers 32 to 78" / "numbered 32 to 78"
    m = re.search(r'number(?:ed|s)?\s+(\d+)\s*(?:to|-)\s*(\d+)', msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "first N acts" / "first 10 bare acts"
    m = re.search(r'first\s+(\d+)\s+(?:bare\s+)?acts?', msg)
    if m:
        return 1, int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Search Constraints — parse user-specified filters (court type, result count,
# extra terms) from the message and apply them throughout the discovery pipeline.
# All constraints are *optional*: if not specified, existing defaults apply unchanged.
# ---------------------------------------------------------------------------

# Court type detection patterns: (compiled_regex, canonical_label)
_COURT_PATTERNS = [
    (re.compile(r'\b(supreme\s+court|sc\s+judg(?:ment|ement))\b', re.I), "Supreme Court"),
    (re.compile(r'\b(telangana\s+high\s+court|high\s+court\s+of\s+telangana|tshc)\b', re.I), "Telangana High Court"),
    (re.compile(r'\b(andhra\s+(?:pradesh\s+)?high\s+court|aphc)\b', re.I), "Andhra Pradesh High Court"),
    (re.compile(r'\b(delhi\s+high\s+court|high\s+court\s+of\s+delhi)\b', re.I), "Delhi High Court"),
    (re.compile(r'\b(bombay\s+high\s+court|high\s+court\s+of\s+bombay)\b', re.I), "Bombay High Court"),
    (re.compile(r'\b(madras\s+high\s+court|high\s+court\s+of\s+madras)\b', re.I), "Madras High Court"),
    (re.compile(r'\b(calcutta\s+high\s+court|high\s+court\s+at\s+calcutta)\b', re.I), "Calcutta High Court"),
    (re.compile(r'\b(allahabad\s+high\s+court|high\s+court\s+of\s+judicature\s+at\s+allahabad)\b', re.I), "Allahabad High Court"),
    (re.compile(r'\b(kerala\s+high\s+court)\b', re.I), "Kerala High Court"),
    (re.compile(r'\b(karnataka\s+high\s+court|high\s+court\s+of\s+karnataka)\b', re.I), "Karnataka High Court"),
    (re.compile(r'\b(gujarat\s+high\s+court)\b', re.I), "Gujarat High Court"),
    (re.compile(r'\b(rajasthan\s+high\s+court)\b', re.I), "Rajasthan High Court"),
    (re.compile(r'\b(punjab\s+(?:and\s+)?haryana\s+high\s+court)\b', re.I), "Punjab and Haryana High Court"),
    (re.compile(r'\b(madhya\s+pradesh\s+high\s+court|m\.?p\.?\s+high\s+court)\b', re.I), "Madhya Pradesh High Court"),
    (re.compile(r'\b(orissa\s+high\s+court|odisha\s+high\s+court)\b', re.I), "Orissa High Court"),
    (re.compile(r'\b(patna\s+high\s+court)\b', re.I), "Patna High Court"),
    (re.compile(r'\b(gauhati\s+high\s+court)\b', re.I), "Gauhati High Court"),
    (re.compile(r'\b(himachal\s+(?:pradesh\s+)?high\s+court)\b', re.I), "Himachal Pradesh High Court"),
    (re.compile(r'\b(jharkhand\s+high\s+court)\b', re.I), "Jharkhand High Court"),
    (re.compile(r'\b(uttarakhand\s+high\s+court|nainital\s+high\s+court)\b', re.I), "Uttarakhand High Court"),
    (re.compile(r'\b(chhattisgarh\s+high\s+court)\b', re.I), "Chhattisgarh High Court"),
    (re.compile(r'\b(manipur\s+high\s+court|tripura\s+high\s+court|meghalaya\s+high\s+court)\b', re.I), "North-East High Court"),
    # Generic "High Court" (any HC, no state specified)
    (re.compile(r'\bonly\s+(?:from\s+)?high\s+courts?\b', re.I), "High Court"),
    (re.compile(r'\b(?:only\s+)?high\s+court\s+(?:only|judgments?|judgements?|cases?|rulings?)\b', re.I), "High Court"),
    (re.compile(r'\bhigh\s+court\s+only\b', re.I), "High Court"),
]

# Patterns for "only N case laws" / "find 5 judgments" / "top 10 results"
_MAX_RESULTS_RE = re.compile(
    r'\b(?:only|find|get|top|limit\s+to|maximum|max|fetch|show|give\s+me|restrict\s+to)\s+(\d+)\s*'
    r'(?:case\s+laws?|judgments?|judgements?|results?|documents?|docs?)\b',
    re.I,
)
_MAX_RESULTS_RE2 = re.compile(
    r'\b(\d+)\s+(?:case\s+laws?|judgments?|judgements?|results?)\s+(?:only|max|maximum|at\s+most)\b',
    re.I,
)

# Pattern for extra topic terms: "about land acquisition", "relating to dowry"
_EXTRA_TERMS_RE = re.compile(
    r'\b(?:about|related\s+to|relating\s+to|concerning|regarding|specific\s+to|on\s+topic\s+of|on\s+the\s+topic\s+of)\s+(.+?)(?=\s*$|\s*[,;]|\s+and\s+(?:only|just|please|also)\b)',
    re.I,
)


def parse_search_constraints(user_message: str) -> dict:
    """
    Parse user-specified search constraints from the message (regex, no LLM call).

    Returns a dict with:
      court_type (str|None)  — canonical court label, e.g. "High Court", "Telangana High Court"
      max_results (int|None) — user-specified result cap (overrides TOP_N_* defaults)
      extra_terms (str)      — additional terms to append to search queries
      court_query_prefix (str) — what to prepend to queries (same as court_type if set)

    All fields default to None/"" — no-op if not specified.
    """
    msg = user_message or ""

    # 1. Court type (first match wins; specific courts checked before generic)
    court_type = None
    for pattern, label in _COURT_PATTERNS:
        if pattern.search(msg):
            court_type = label
            break

    # 2. Max results
    max_results = None
    m = _MAX_RESULTS_RE.search(msg) or _MAX_RESULTS_RE2.search(msg)
    if m:
        try:
            max_results = max(1, min(int(m.group(1)), 100))  # clamp 1–100
        except (ValueError, IndexError):
            pass

    # 3. Extra terms (skip if it looks like an act name — already handled by act resolution)
    extra_terms = ""
    m = _EXTRA_TERMS_RE.search(msg)
    if m:
        raw = m.group(1).strip().rstrip(".,;:")
        if raw and len(raw) < 80 and not _line_looks_like_act_title(raw):
            extra_terms = raw

    return {
        "court_type": court_type,
        "max_results": max_results,
        "extra_terms": extra_terms,
        "court_query_prefix": court_type or "",
    }


def _effective_limit(constraints: dict | None, default: int) -> int:
    """Return user-specified max_results or the default if not set."""
    if constraints and constraints.get("max_results"):
        return int(constraints["max_results"])
    return default


def _build_constrained_query(base_query: str, constraints: dict | None) -> str:
    """Append court type and extra terms to a query string (only if not already present)."""
    if not constraints:
        return base_query
    base = base_query.strip()
    court_prefix = (constraints.get("court_query_prefix") or "").strip()
    extra_terms = (constraints.get("extra_terms") or "").strip()
    parts = [base]
    if court_prefix and court_prefix.lower() not in base.lower():
        parts.append(court_prefix)
    if extra_terms and extra_terms.lower() not in base.lower():
        parts.append(extra_terms)
    return " ".join(parts)


def _matches_court_type(candidate: dict, court_type: str) -> bool:
    """
    Return True if candidate's title + first 1 KB of content matches the requested court type.
    Gracefully returns True if court_type is empty (no-op).
    """
    if not court_type:
        return True
    text = (
        (candidate.get("title") or "") + " " + (candidate.get("content") or "")[:1000]
    ).lower()
    ct_lower = court_type.lower()
    if ct_lower == "supreme court":
        return "supreme court" in text
    if ct_lower == "high court":
        return "high court" in text
    # Specific named court: require all meaningful words to appear
    words = [w for w in ct_lower.split() if len(w) > 2 and w not in ("and", "the", "of", "at")]
    return all(w in text for w in words)


def _filter_by_court(candidates: list[dict], constraints: dict | None) -> list[dict]:
    """
    Filter candidate list to those matching the requested court_type.
    Falls back to returning all candidates if the filter would yield nothing
    (prevents accidentally returning 0 results due to missing content).
    """
    court_type = (constraints or {}).get("court_type") if constraints else None
    if not court_type:
        return candidates
    filtered = [c for c in candidates if _matches_court_type(c, court_type)]
    if not filtered and candidates:
        logger.info(
            "Court filter '%s' matched 0 of %d candidates; relaxing filter (no content available for check)",
            court_type, len(candidates),
        )
        return candidates  # Graceful fallback
    if len(filtered) < len(candidates):
        logger.info(
            "Court filter '%s': kept %d of %d candidates",
            court_type, len(filtered), len(candidates),
        )
    return filtered


def interpret_user_input(user_message: str) -> dict:
    """
    Use LLM to interpret: statement-based vs acts-range, and what to search for.
    Returns dict with: flow_type ("statement" | "first_10_acts"), query/topic,
    act_start (1-based), act_end (1-based inclusive).
    """
    # Try fast regex parse first (avoids LLM call for common range patterns)
    range_match = _parse_act_range_from_message(user_message)
    if range_match:
        act_start, act_end = range_match
        return {"flow_type": "first_10_acts", "topic": None, "act_start": act_start, "act_end": act_end}

    system = (
        "You are a legal research assistant. Classify the user's request into exactly one of two flows. "
        "Reply with a single JSON object only, no markdown, no explanation. "
        "Keys: flow_type (either 'statement' or 'first_10_acts'), topic (short search topic or null), "
        "act_start (1-based start position integer, only when flow_type is first_10_acts, default 1), "
        "act_end (1-based end position integer, only when flow_type is first_10_acts). "
        "Use first_10_acts ONLY when the user explicitly asks for case laws for acts by position "
        "(e.g. 'first 10 bare acts', 'first 5 acts', 'acts 32 to 78', 'acts from 10 to 50'). "
        "When the user names a specific act (e.g. 'THE FAMILY COURTS ACT 1984', 'Hindu Marriage Act'), "
        "always use flow_type statement and set topic to that act name. "
        "Otherwise use statement and set topic to the legal subject they want case laws for."
    )
    prompt = f"User request: {user_message}\n\nReply with one JSON object: flow_type, topic, act_start, act_end."
    raw = _ask_llm(prompt, system=system)
    try:
        # Extract JSON if wrapped in markdown
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        obj = json.loads(raw)
        flow = (obj.get("flow_type") or "statement").strip().lower()
        if flow == "first_10_acts":
            act_start = int(obj.get("act_start") or 1)
            act_end = int(obj.get("act_end") or obj.get("act_count") or 10)
            return {"flow_type": "first_10_acts", "topic": None, "act_start": act_start, "act_end": act_end}
        return {
            "flow_type": "statement",
            "topic": (obj.get("topic") or user_message[:200]).strip() or "case laws",
            "act_start": None,
            "act_end": None,
        }
    except Exception:
        return {"flow_type": "statement", "topic": user_message[:200].strip() or "case laws", "act_start": None, "act_end": None}


def get_acts_from_vector_store(act_start: int = 1, act_end: int = 10) -> list[dict]:
    """
    Read bare act chunks from the vector store; return acts in the range [act_start, act_end]
    (both 1-based, inclusive) sorted alphabetically. For example:
      act_start=1,  act_end=10  → first 10 acts
      act_start=32, act_end=78  → acts numbered 32 to 78 in alphabetical order

    Logs the total act count and the slice being processed so the caller can see
    how many acts are in the store (e.g. "Acts 32-78 of 100 total").
    """
    try:
        from config import BARE_CHUNKS_V2
        if not os.path.isfile(BARE_CHUNKS_V2):
            return []
        with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            return []
        # Chunks may be dict keyed by id or list
        if isinstance(chunks, dict):
            chunk_list = list(chunks.values())
        else:
            chunk_list = chunks
        by_act = {}
        for c in chunk_list:
            name = (c.get("act_name") or c.get("source") or "Unknown").strip()
            by_act.setdefault(name, []).append(c)
        sorted_acts = sorted(by_act.keys(), key=lambda x: x.lower())
        total = len(sorted_acts)
        # Convert to 0-based slice: act_start=1 → index 0; act_end=10 → index 9 (inclusive)
        start_idx = max(0, act_start - 1)
        end_idx = min(total, act_end)  # slice end is exclusive, act_end is 1-based inclusive
        selected = sorted_acts[start_idx:end_idx]
        logger.info("Acts range %d-%d of %d total acts in vector store (%d selected)",
                    act_start, act_end, total, len(selected))
        return [{"act_name": name, "chunks": by_act[name]} for name in selected]
    except Exception as e:
        logger.warning("Could not read bare chunks: %s", e)
        return []


def get_first_n_acts_from_vector_store(n: int = 10) -> list[dict]:
    """Backward-compatible wrapper: returns first n acts. Use get_acts_from_vector_store() directly."""
    return get_acts_from_vector_store(act_start=1, act_end=n)


def _chunk_text(chunk: dict) -> str:
    """Extract searchable text from a bare act chunk."""
    return (
        (chunk.get("content") or chunk.get("search_text") or chunk.get("full_text") or chunk.get("text") or "")
        .strip()
    )


def _generate_act_summary(act_name: str, chunks: list[dict]) -> str:
    """Generate a summary of the act from its chunks (LLM), capped at ACT_SUMMARY_MAX_CHARS."""
    combined = []
    for c in chunks[:50]:
        t = _chunk_text(c)
        if t:
            combined.append(t[:800])
    text = "\n\n".join(combined)[:15000]
    if not text.strip():
        return act_name
    system = (
        "You are a legal assistant. Summarize the following Indian bare act text in clear, concise language. "
        f"Include: short title, purpose, key definitions, and main provisions. Keep the summary under {ACT_SUMMARY_MAX_CHARS} characters. "
        "Output only the summary, no preamble."
    )
    prompt = f"Act: {act_name}\n\nText:\n{text}\n\nSummary:"
    out = _ask_llm(prompt, system=system)
    return (out or act_name).strip()[:ACT_SUMMARY_MAX_CHARS]


def _generate_case_law_summary(doc_text: str, title: str = "") -> str:
    """Generate a summary of a case law document (LLM), capped at CASE_LAW_SUMMARY_MAX_CHARS."""
    text = (doc_text or "").strip()[:12000]
    if not text:
        return title or "Case law"
    system = (
        "You are a legal assistant. Summarize the following Indian court judgment in clear language. "
        f"Include: parties, court, key facts, and main holding. Keep under {CASE_LAW_SUMMARY_MAX_CHARS} characters. "
        "Output only the summary."
    )
    prompt = f"Judgment:\n{text}\n\nSummary:"
    out = _ask_llm(prompt, system=system)
    return (out or title or "Case law").strip()[:CASE_LAW_SUMMARY_MAX_CHARS]


def get_or_create_act_summary(act_name: str, chunks: list[dict]) -> str:
    """Return act summary from index, or generate from chunks and save."""
    summary = get_bare_act_summary(act_name)
    if summary:
        return summary
    summary = _generate_act_summary(act_name, chunks)
    if summary:
        set_bare_act_summary(act_name, summary)
    return summary or act_name


def _is_case_law_doc(url: str, title: str, first_two_pages: str) -> bool:
    """Heuristic: is this document a case law (judgment) vs bare act?"""
    combined = f"{url} {title} {first_two_pages}".lower()
    case_indicators = [
        " v. ", " v/s ", " vs ", "appellant", "respondent",
        "petitioner", "judgment", "judgement", "hon'ble",
        "supreme court", "high court", "bench", "coram",
    ]
    bare_act_indicators = [
        "bare act", "act,", "code,", "ordinance", "regulation",
        "section", "chapter", "schedule", "notification",
        "indiacode", "legislative", "gazette",
    ]
    case_score = sum(1 for i in case_indicators if i in combined)
    act_score = sum(1 for i in bare_act_indicators if i in combined)
    return case_score > act_score


def _extract_signature(first_two_pages: str) -> str:
    """
    Extract duplicate-check signature from first two pages: court + parties (appellant vs respondent).
    Returns a short string for dedup; empty if extraction fails.
    """
    if not first_two_pages or len(first_two_pages.strip()) < 100:
        return ""
    text = first_two_pages.strip()[:_FIRST_TWO_PAGES_CHARS]
    try:
        system = (
            "You extract a unique signature for an Indian court judgment from the first two pages. "
            "Reply with exactly one line: Court (e.g. Supreme Court or High Court of X) then | then "
            "Parties (Appellant vs Respondent, e.g. 'XYZ Ltd vs State of Maharashtra'). "
            "Use only the exact format: COURT | PARTIES. If you cannot determine, reply UNKNOWN."
        )
        prompt = f"Document excerpt:\n{text[:2500]}\n\nOne-line signature (COURT | PARTIES):"
        raw = _ask_llm(prompt, system=system)
        if raw and "UNKNOWN" not in raw.upper() and "|" in raw:
            return raw.strip()[:200]
    except Exception as e:
        logger.debug("Signature extraction failed: %s", e)
    return ""


def _is_likely_document_url(url: str) -> bool:
    """
    True if URL looks like an official document (PDF or judgment page), not a home/generic page.
    Used so only document URLs are proposed for indexing; home pages and index pages are skipped.
    """
    if not url or not url.strip():
        return False
    u = url.strip().lower()
    if u.endswith(".pdf"):
        return True
    # Path after domain (skip protocol and domain)
    for prefix in ("https://", "http://", "www."):
        if u.startswith(prefix):
            u = u[len(prefix):]
            break
    path = u.split("/", 1)[-1].split("?")[0] if "/" in u else ""
    path = path.strip("/")
    # Reject generic/home
    if not path or path in ("", "index", "index.html", "search", "pdfsearch"):
        return False
    # Accept document-like paths
    doc_indicators = ("/pdf", "/judgment", "/judgement", "/doc", "/view_judgment", "/order", "/judgments", "/ehcr", "/bitstream", "/supremecourt/", "/doc/")
    if any(ind in u for ind in doc_indicators):
        return True
    # sci.gov.in judgment PDFs often have path like /year/num/...
    if "sci.gov.in" in u and ("supremecourt" in u or "/20" in u):
        return True
    return False


def _is_pdf_url(url: str) -> bool:
    """
    True only if URL clearly points to a PDF document.
    Case law discovery presents only PDF docs for indexing, not web page links.
    """
    if not url or not url.strip():
        return False
    u = url.strip().lower()
    if u.endswith(".pdf") or u.rstrip("/").endswith(".pdf"):
        return True
    if ".pdf?" in u or ".pdf&" in u:
        return True
    if "/pdf/" in u or "/pdf?" in u:
        return True
    return False


def _fetch_and_score_candidate(
    result: dict,
    query: str,
    include_content: bool = False,
) -> dict | None:
    """
    Fetch URL, check it's case law from first two pages, score vs query.
    Returns dict with title, source_url, score, signature; optionally content (for summary generation).
    Only fetches URLs that look like documents (PDF/judgment pages); skips home/generic pages.
    """
    from retrieval.tiered_search import fetch_content_and_pdf
    from retrieval.hybrid_retriever import score_query_document

    url = result.get("url", "")
    title = result.get("title", "Unknown")
    if not url:
        return None
    if not _is_likely_document_url(url):
        return None
    try:
        text_content, _ = fetch_content_and_pdf(url, timeout=25)
    except Exception as e:
        logger.debug("Fetch failed %s: %s", url[:60], e)
        return None
    text_content = (text_content or "").strip()
    if not text_content or len(text_content) < 200:
        return None
    first_two = text_content[:_FIRST_TWO_PAGES_CHARS]
    if not _is_case_law_doc(url, title, first_two):
        return None
    # P2: reject interlocutory orders / summons / notices
    if not _is_final_judgment(title, first_two):
        logger.debug("P2 filter: skipping non-final document %s", url[:60])
        return None
    score = score_query_document(query, text_content)
    if score <= SIMILARITY_THRESHOLD_LOW:
        return None
    signature = _extract_signature(first_two)
    out = {
        "title": title,
        "source_url": url,
        "suggested_category": "case_law",
        "score": float(score),
        "signature": signature or f"{url[:80]}",
        "act_name": None,
    }
    if include_content:
        out["content"] = text_content[:12000]
    return out


def _score_candidate_content(
    title: str,
    source_url: str,
    text_content: str,
    query: str,
    include_content: bool = False,
) -> dict | None:
    """
    Score pre-fetched content (e.g. from Indian Kanoon API). No URL fetch.
    Returns same shape as _fetch_and_score_candidate or None if below threshold.
    """
    from retrieval.hybrid_retriever import score_query_document

    text_content = (text_content or "").strip()
    if not text_content or len(text_content) < 200:
        return None
    first_two = text_content[:_FIRST_TWO_PAGES_CHARS]
    # P2: reject interlocutory orders / summons / notices
    if not _is_final_judgment(title, first_two):
        logger.debug("P2 filter: skipping non-final IK document %s", source_url[:60])
        return None
    score = score_query_document(query, text_content)
    if score <= SIMILARITY_THRESHOLD_LOW:
        return None
    signature = _extract_signature(first_two)
    out = {
        "title": title,
        "source_url": source_url,
        "suggested_category": "case_law",
        "score": float(score),
        "signature": signature or source_url[:80],
        "act_name": None,
    }
    if include_content:
        out["content"] = text_content[:12000]
    return out


def _indian_kanoon_candidates(query: str, max_results: int, include_content: bool, max_pages: int | None = None) -> list[dict]:
    """
    Search Indian Kanoon API and return scored candidates (same shape as web candidates).

    P5: Fetches up to IK_SEARCH_MAX_PAGES pages (default 5; was 3) to collect a larger
    candidate pool. Each IK page returns ~10-20 docs.

    P5: After collecting TIDs, pre-sorts by court tier (SC first, HC second) so the
    most authoritative judgments get full-text fetched first — the expensive step.

    P0: Applies citation-count boost to final scores:
        boosted_score = ce_score * (1 + CITATION_BOOST_WEIGHT * log(1 + numciting))
    """
    try:
        from retrieval.indian_kanoon_client import search as ik_search, get_document
    except ImportError as e:
        logger.debug("Indian Kanoon client not available: %s", e)
        return []
    from config import INDIAN_KANOON_API_TOKEN, IK_SEARCH_MAX_PAGES, CITATION_BOOST_WEIGHT
    if not INDIAN_KANOON_API_TOKEN:
        logger.warning("Indian Kanoon API token not set (INDIAN_KANOON_API_TOKEN); act-named flow will have no IK results")
        return []

    if max_pages is None:
        max_pages = IK_SEARCH_MAX_PAGES  # P5: default 5 pages (was hardcoded 3)

    # --- Phase 1: Collect TIDs from multiple pages (fast — just metadata) ---
    all_tids = []
    seen_tids = set()
    for page in range(max_pages):
        page_results = ik_search(query, pagenum=page, max_results=20)
        if not page_results:
            break  # No more results for this query
        new = [r for r in page_results if r.get("tid") and r.get("tid") not in seen_tids]
        if not new:
            break  # IK returned duplicates — we've likely exhausted results
        for r in new:
            seen_tids.add(r.get("tid"))
        all_tids.extend(new)
        if page < max_pages - 1:
            time.sleep(0.5)  # Brief pause between page requests

    if not all_tids:
        return []

    # P5: Sort TIDs by court tier before document fetch so SC judgments are fetched first.
    # This means when we truncate to max_results, we preferentially fetch SC > HC > other.
    all_tids.sort(key=_ik_court_tier)

    logger.debug(
        "IK query '%s': collected %d TIDs across %d pages (SC=%d HC=%d other=%d)",
        query[:60], len(all_tids), min(max_pages, len(all_tids) // 10 + 1),
        sum(1 for t in all_tids if _ik_court_tier(t) == 0),
        sum(1 for t in all_tids if _ik_court_tier(t) == 1),
        sum(1 for t in all_tids if _ik_court_tier(t) == 2),
    )

    # --- Phase 2: Fetch full document content for up to max_results TIDs ---
    candidates = []
    for r in all_tids[:max_results]:
        tid = r.get("tid")
        if not tid:
            continue
        time.sleep(0.3)
        doc = get_document(tid)
        if not doc or not (doc.get("doc_text") or "").strip():
            continue
        text = (doc.get("doc_text") or "").strip()
        title = doc.get("title") or r.get("title") or "Judgment"
        url = doc.get("url") or r.get("url") or f"https://indiankanoon.org/doc/{tid}/"
        c = _score_candidate_content(title, url, text, query, include_content=include_content)
        if c:
            # P0: apply citation-count boost to elevate heavily-cited judgments
            numciting = int(r.get("numciting") or 0)
            if numciting > 0 and CITATION_BOOST_WEIGHT > 0:
                c["score"] = c["score"] * (1.0 + CITATION_BOOST_WEIGHT * math.log(1.0 + numciting))
            c["numciting"] = numciting
            candidates.append(c)
    return candidates


def _normalize_act_name_for_match(name: str) -> str:
    """Lowercase, collapse spaces, remove commas so 'The Family Courts Act, 1984' matches 'THE FAMILY COURTS ACT 1984'."""
    if not name:
        return ""
    s = re.sub(r"[,\.]", " ", name.lower().strip())
    return " ".join(s.split())


def _resolve_act_from_summary_index(user_topic: str) -> tuple[str, str]:
    """
    Resolve user's act name (e.g. 'Indian Penal Code') to (act_name, summary) from bare act summary index.
    Returns ("", "") if no match. Matches exact key or topic contained in act name / act name in topic.
    Normalizes punctuation so "THE FAMILY COURTS ACT 1984" matches index key "The Family Courts Act, 1984".
    """
    if not user_topic or not user_topic.strip():
        return ("", "")
    topic_norm = _normalize_act_name_for_match(user_topic)
    index = load_bare_act_summary_index()
    if not index:
        return ("", "")
    # Exact match (case-insensitive, punctuation normalized)
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        if _normalize_act_name_for_match(act_name) == topic_norm:
            return (act_name.strip(), (summary or "").strip())
    # Topic contained in act name (e.g. "Indian Penal Code" in "The Indian Penal Code, 1860")
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        act_norm = _normalize_act_name_for_match(act_name)
        if topic_norm in act_norm:
            return (act_name.strip(), (summary or "").strip())
    # Act name contained in topic (e.g. user said full name)
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        act_norm = _normalize_act_name_for_match(act_name)
        if act_norm in topic_norm:
            return (act_name.strip(), (summary or "").strip())
    return ("", "")


def _topic_looks_like_act_name(topic: str) -> bool:
    """True if topic looks like an act/sanhita/code/constitution name so we run act-named flow even when not in index."""
    return _line_looks_like_act_title(topic)


def _resolve_all_acts_from_summary_index(user_topic: str) -> list[tuple[str, str]]:
    """
    Parse topic for multiple act names (e.g. "Act A and Act B" or "Act A, Act B"), resolve each
    from the bare act summary index, and return a list of (act_name, summary) in order, deduped by act_name.
    """
    if not user_topic or not user_topic.strip():
        return []
    # Split by common separators: " and ", ", ", " & ", newline, semicolon
    parts = re.split(r"\s+and\s+|\s*,\s*|\s+&\s+|\n|;", user_topic.strip(), flags=re.IGNORECASE)
    seen_act_names = set()
    result = []
    for part in parts:
        candidate = " ".join(part.split()).strip()
        if len(candidate) < 3:
            continue
        act_name, summary = _resolve_act_from_summary_index(candidate)
        if act_name and act_name not in seen_act_names:
            seen_act_names.add(act_name)
            result.append((act_name, summary or ""))
    return result


def _apply_statement_selection(candidates: list[dict], limit: int = TOP_N_STATEMENT) -> list[dict]:
    """Apply limits: all > HIGH, else top `limit` above LOW. limit defaults to TOP_N_STATEMENT."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= limit:
        return sorted(above_high, key=lambda x: -x["score"])[:limit]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:limit]


def _apply_per_act_selection(candidates: list[dict], limit: int = TOP_N_PER_ACT) -> list[dict]:
    """Per act: all >HIGH, else top `limit` above LOW. limit defaults to TOP_N_PER_ACT."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= limit:
        return sorted(above_high, key=lambda x: -x["score"])[:limit]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:limit]


def _web_search_act_name_and_summary(act_name: str, act_summary: str, score_query: str, seen_urls: dict, max_per_query: int = 12, chunks: list | None = None) -> None:
    """
    Run tiered web search with (1) act name, (2) act name + legal keywords from vector store.
    Replaces the old raw-markdown summary query that caused 400/429 errors on all search engines.
    Uses _extract_act_keywords() to get 4-5 clean legal terms from the act's chunks.
    Merges scored results into seen_urls (in-place).
    """
    from retrieval.tiered_search import tiered_search
    if len(seen_urls) >= TOP_N_ACT_NAMED:
        return
    time.sleep(2)

    # Build a clean keyword-enriched query (replaces raw markdown summary)
    keywords = _extract_act_keywords(act_name, chunks)
    keyword_query = f"{act_name} {keywords}".strip() if keywords else ""

    # Two queries: plain act name, then act name + legal keywords (different result angles)
    queries = [act_name.strip()]
    if keyword_query and keyword_query.strip() != act_name.strip():
        queries.append(keyword_query[:150])

    for query in queries:
        if not query:
            continue
        if len(seen_urls) >= TOP_N_ACT_NAMED:
            break
        time.sleep(1)
        results = tiered_search(
            query=query,
            search_type="case_law",
            jurisdiction_state="Telangana",
            max_per_tier=max_per_query,
            discovery_mode=True,
        )
        for r in results[:ACT_NAMED_FETCH_BUFFER]:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            if not _is_pdf_url(url):
                continue
            time.sleep(0.4)
            doc = _fetch_and_score_candidate(r, score_query, include_content=True)
            if doc:
                seen_urls[url] = doc


def run_act_named_flow(act_name: str, act_summary: str, constraints: dict | None = None) -> tuple[list[dict], int]:
    """
    User named an act: search Indian Kanoon (1) with act name, (2) with summary from index if present;
    then web search (1) with act name, (2) with summary; merge by URL, score descending, deduplicate vs index,
    keep first n_results unique (default TOP_N_ACT_NAMED=25).
    Returns (items_for_pending, duplicates_discarded_count).

    constraints (optional): parsed from user message via parse_search_constraints().
      court_type  → filters + adds court name to queries.
      max_results → overrides TOP_N_ACT_NAMED default.
      extra_terms → extra terms appended to all queries.
    """
    n_results = _effective_limit(constraints, TOP_N_ACT_NAMED)
    logger.info(
        "Act-named flow: act=%s, fetch up to %s from IK + web, dedup, target %s unique%s",
        act_name[:50], ACT_NAMED_FETCH_BUFFER, n_results,
        f", court_filter={constraints['court_type']}" if constraints and constraints.get("court_type") else "",
    )

    # P3: Use section-anchored query (most operative sections) for better cross-encoder scoring.
    # chunks=None causes _build_section_anchored_query to load them from the vector store.
    score_query = (
        _build_section_anchored_query(act_name, chunks=None)
        or (act_name + " " + (act_summary or "")[:500]).strip()
    )
    logger.info(
        "Act-named flow: score_query first 80 chars: %s",
        score_query[:80],
    )
    seen_urls = {}

    # Extract legal keywords from vector store (used for both IK and web search)
    act_keywords = _extract_act_keywords(act_name)
    keyword_query = f"{act_name} {act_keywords}".strip() if act_keywords else ""

    # P1: load chunks from vector store for section-anchored queries
    _named_chunks: list | None = None
    try:
        from config import BARE_CHUNKS_V2
        import json as _json_named
        if os.path.isfile(BARE_CHUNKS_V2):
            with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as _f:
                _all = _json_named.load(_f)
            if isinstance(_all, dict):
                _all = list(_all.values())
            act_norm = act_name.lower().strip()
            _named_chunks = [c for c in _all if (c.get("act_name") or c.get("source") or "").lower().strip() == act_norm]
    except Exception:
        pass

    section_queries = _build_section_ik_queries(act_name, _named_chunks, top_n=2)

    # Indian Kanoon: (1) act name [+ constraints], (2) keywords [+ constraints],
    #               (3)+(4) section-anchored queries [P1]
    ik_queries = [_build_constrained_query(act_name.strip(), constraints)]
    kq_constrained = _build_constrained_query(keyword_query[:200], constraints) if keyword_query and keyword_query.strip() != act_name.strip() else ""
    if kq_constrained and kq_constrained != ik_queries[0]:
        ik_queries.append(kq_constrained)
    # P1: section-specific queries (max 2 to avoid rate-limit spikes)
    for sq in section_queries[:2]:
        sq_c = _build_constrained_query(sq, constraints)
        if sq_c and sq_c not in ik_queries:
            ik_queries.append(sq_c)
    for query in ik_queries:
        if not query:
            continue
        batch = _indian_kanoon_candidates(query, max_results=ACT_NAMED_FETCH_BUFFER, include_content=True)
        for c in batch:
            url = (c.get("source_url") or "").strip()
            if not url:
                continue
            if url not in seen_urls or (c.get("score") or 0) > (seen_urls[url].get("score") or 0):
                seen_urls[url] = c

    # Web (eCourts + DDG): only if we don't already have enough (avoids 429s when IK gives n_results+)
    if len(seen_urls) < n_results:
        constrained_act = _build_constrained_query(act_name, constraints)
        _web_search_act_name_and_summary(constrained_act, act_summary, score_query, seen_urls, max_per_query=12, chunks=None)
    else:
        logger.info("Already have %d candidates from IK; skipping web search to avoid rate limits", len(seen_urls))

    # Apply court-type filter (no-op if not specified)
    all_candidates = list(seen_urls.values())
    all_candidates = _filter_by_court(all_candidates, constraints)
    scored = sorted(all_candidates, key=lambda x: -(x.get("score") or 0))

    if not scored:
        logger.warning("No Indian Kanoon or web results for act-named query")
        return [], 0

    # Deduplicate vs existing index (and within-list): check_indexing_candidates marks already_in_store
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in scored
    ]
    deduped = check_indexing_candidates(dedup_list)
    # Merge already_in_store back into scored (same order)
    for i, s in enumerate(scored):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False

    # Take first n_results that are unique (not already_in_store) and PDF URLs only
    unique = []
    discarded = 0
    for s in scored:
        if len(unique) >= n_results:
            break
        if s.get("already_in_store"):
            discarded += 1
            continue
        if not _is_pdf_url(s.get("source_url") or ""):
            discarded += 1
            continue
        unique.append(s)
    if discarded:
        logger.info("Act-named: %s unique selected (target %s), %s duplicates discarded, next in rank used",
                    len(unique), n_results, discarded)
    if not unique and scored:
        logger.warning("Act-named flow: 0 proposed for indexing (all %s candidates already in index)", len(scored))

    for s in unique:
        s["act_name"] = act_name
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))
    return unique, discarded


def run_statement_flow(topic: str, constraints: dict | None = None) -> list[dict]:
    """
    Statement-based flow: Indian Kanoon API first (if token set), then tiered web search →
    fetch → score → apply limits (all >5, else top 10 above 3). Returns list for pending.

    constraints (optional): parsed from user message via parse_search_constraints().
      court_type  → filters results to matching court; adds court name to queries.
      max_results → overrides TOP_N_STATEMENT default.
      extra_terms → extra terms appended to search queries.
    """
    from retrieval.tiered_search import tiered_search

    n_results = _effective_limit(constraints, TOP_N_STATEMENT)
    constrained_topic = _build_constrained_query(topic, constraints)

    logger.info(
        "Statement flow (topic=%s): limits TOP_N=%s, score>%s or top above %s%s",
        topic, n_results, SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_LOW,
        f", court_filter={constraints['court_type']}" if constraints and constraints.get("court_type") else "",
    )
    candidates = []
    seen_urls = set()

    # First: Indian Kanoon API (if configured) — use constrained query for better targeting
    ik_list = _indian_kanoon_candidates(constrained_topic, max_results=15, include_content=True)
    for c in ik_list:
        url = c.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            candidates.append(c)

    # Then: tiered web search (official only: eCourts, SCI, HC)
    results = tiered_search(
        query=constrained_topic,
        search_type="case_law",
        jurisdiction_state="Telangana",
        max_per_tier=12,
    )
    for r in results[:20]:
        url = r.get("url") or ""
        if url in seen_urls:
            continue
        if not _is_pdf_url(url):
            continue
        seen_urls.add(url)
        time.sleep(0.5)
        doc = _fetch_and_score_candidate(r, constrained_topic, include_content=True)
        if doc:
            candidates.append(doc)

    # Apply court-type filter (no-op if not specified)
    candidates = _filter_by_court(candidates, constraints)

    selected = _apply_statement_selection(candidates, limit=n_results)
    selected = [s for s in selected if _is_pdf_url(s.get("source_url") or "")]
    # Deduplicate vs existing index (same logic as main indexing)
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in selected
    ]
    deduped = check_indexing_candidates(dedup_list)
    for i, s in enumerate(selected):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False
    for s in selected:
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))
    return selected


def _signatures_from_summary_index(summary_index: dict) -> set:
    """Collect all signatures already in the summary index (any act)."""
    seen = set()
    for sig_list in summary_index.values():
        if isinstance(sig_list, list):
            for s in sig_list:
                if isinstance(s, str):
                    seen.add(s)
                elif isinstance(s, dict) and s.get("signature"):
                    seen.add(str(s["signature"]))
        elif isinstance(sig_list, str):
            seen.add(sig_list)
    return seen


def run_first_10_acts_flow(act_count: int) -> list[dict]:
    """Backward-compatible wrapper. Use run_acts_range_flow(act_start, act_end) directly."""
    return run_acts_range_flow(act_start=1, act_end=act_count)


def _process_single_act(
    act_info: dict,
    stagger_idx: int,
    stagger_seconds: float,
    summary_index: dict,
    index_signatures: set,
    constraints: dict | None = None,
) -> list[dict]:
    """
    Process one act: IK multi-page fetch + keyword web search + score + dedup.
    Thread-safe: uses module-level _file_write_lock and _signatures_lock for shared state.

    stagger_idx > 0 causes an initial sleep of (stagger_idx * stagger_seconds) to spread
    thread DDG bursts over time and reduce rate-limit collisions.

    constraints (optional): parsed from user message via parse_search_constraints().
      court_type  → filters + appends to search queries.
      max_results → overrides TOP_N_PER_ACT default.
      extra_terms → extra terms appended to queries.

    Returns list of scored candidates for this act (content still attached for later summary gen).
    """
    from retrieval.tiered_search import tiered_search

    if stagger_idx > 0:
        time.sleep(stagger_idx * stagger_seconds)

    n_results = _effective_limit(constraints, TOP_N_PER_ACT)
    act_name = act_info.get("act_name", "Unknown")
    chunks = act_info.get("chunks", [])

    # Get act summary — read from cache first; only hold the write lock when generating new
    act_summary = get_bare_act_summary(act_name) or ""
    if not act_summary:
        # LLM inference happens outside the lock (parallel Ollama queue is fine)
        act_summary = _generate_act_summary(act_name, chunks) or act_name
        with _file_write_lock:
            set_bare_act_summary(act_name, act_summary)

    # P3: Use section-anchored query (most operative sections) for better cross-encoder scoring.
    # Falls back to act_name + summary if no chunks are available.
    score_query = (
        _build_section_anchored_query(act_name, chunks)
        or (act_name + " " + act_summary[:1500]).strip()
    )
    logger.info(
        "Act %s: search with act name, then summary (IK + web), top %s judgments%s [score_query first 80: %s]",
        act_name[:50], n_results,
        f", court_filter={constraints['court_type']}" if constraints and constraints.get("court_type") else "",
        score_query[:80],
    )

    seen_urls: dict = {}

    # Extract legal keywords from act chunks
    act_keywords = _extract_act_keywords(act_name, chunks)
    keyword_query = f"{act_name} {act_keywords}".strip() if act_keywords else ""

    # P1: Build section-anchored IK queries (act + specific section numbers).
    # These surface judgments that cite the exact operative section, not just the act name.
    section_queries = _build_section_ik_queries(act_name, chunks, top_n=2)

    # Indian Kanoon: (1) act name [+ constraints], (2) keywords [+ constraints],
    #               (3)+(4) section-anchored queries [P1]
    ik_queries = [_build_constrained_query(act_name.strip(), constraints)]
    kq_constrained = _build_constrained_query(keyword_query[:200], constraints) if keyword_query and keyword_query.strip() != act_name.strip() else ""
    if kq_constrained and kq_constrained != ik_queries[0]:
        ik_queries.append(kq_constrained)
    # P1: add section-specific queries (max 2 to avoid rate-limit spikes)
    for sq in section_queries[:2]:
        sq_c = _build_constrained_query(sq, constraints)
        if sq_c and sq_c not in ik_queries:
            ik_queries.append(sq_c)
    for query in ik_queries:
        if not query:
            continue
        ik_list = _indian_kanoon_candidates(query, max_results=MAX_FETCH_PER_ACT, include_content=True)
        for c in ik_list:
            url = (c.get("source_url") or "").strip()
            if not url:
                continue
            if url not in seen_urls or (c.get("score") or 0) > (seen_urls[url].get("score") or 0):
                seen_urls[url] = c
    scored = list(seen_urls.values())

    # Web: act name then keyword-enriched query (both with constraints appended)
    if len(scored) < n_results:
        time.sleep(1.0)
        web_queries = [_build_constrained_query(act_name.strip(), constraints)]
        kq_web = _build_constrained_query(keyword_query[:150], constraints) if keyword_query and keyword_query.strip() != act_name.strip() else ""
        if kq_web and kq_web != web_queries[0]:
            web_queries.append(kq_web)
        for query in web_queries:
            if not query:
                continue
            time.sleep(0.5)
            results = tiered_search(
                query=query,
                search_type="case_law",
                jurisdiction_state="Telangana",
                max_per_tier=10,
                discovery_mode=True,
            )
            for r in results[:MAX_FETCH_PER_ACT]:
                url = (r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                if not _is_pdf_url(url):
                    continue
                time.sleep(0.4)
                doc = _fetch_and_score_candidate(r, score_query, include_content=True)
                if doc:
                    seen_urls[url] = doc
        scored = list(seen_urls.values())

    # Apply court-type filter (no-op if not specified)
    scored = _filter_by_court(scored, constraints)

    selected = _apply_per_act_selection(scored, limit=n_results)

    # Thread-safe: check and update shared index_signatures in one atomic block
    with _signatures_lock:
        selected = [s for s in selected if (s.get("signature") or "") not in index_signatures]
        by_sig: dict = {}
        for s in selected:
            sig = s.get("signature") or s["source_url"]
            if sig not in by_sig or by_sig[sig]["score"] < s["score"]:
                by_sig[sig] = s
        selected = list(by_sig.values())
        for s in selected:
            if s.get("signature"):
                index_signatures.add(s["signature"])

    act_items = []
    for s in selected:
        if not _is_pdf_url(s.get("source_url") or ""):
            continue
        s["act_name"] = act_name
        act_items.append(s)

    return act_items


def run_acts_range_flow(act_start: int = 1, act_end: int = 10, constraints: dict | None = None) -> list[dict]:
    """
    Acts-range flow: process acts numbered act_start to act_end (1-based, inclusive) in alphabetical
    order from the vector store. For each act: IK search + keyword web search, score, dedup, keep
    top n_results per act (default TOP_N_PER_ACT=25). Generates case law summaries; deduplicates vs
    act-case-law index. Returns list for pending (title, source_url, signature, act_name, summary).

    Uses ThreadPoolExecutor(max_workers=2) for batches > 1 act. Acts are staggered 45s apart to
    avoid simultaneous DDG bursts. Shared state (index_signatures, file writes) is protected by
    module-level locks. Single-act requests run sequentially with no overhead.

    constraints (optional): parsed from user message via parse_search_constraints().
      court_type  → filters results + adds court name to queries.
      max_results → overrides TOP_N_PER_ACT default.
      extra_terms → extra terms appended to queries.

    Examples:
      run_acts_range_flow(1, 10)                               → first 10 acts (default limits)
      run_acts_range_flow(32, 78)                              → acts 32 to 78
      run_acts_range_flow(1, 5, constraints={"court_type": "High Court", "max_results": 5})
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Parallel processing config
    MAX_PARALLEL_WORKERS = 2   # Start at 2; bump to 3 once confirmed stable on your hardware
    STAGGER_SECONDS = 45.0     # Each thread starts its first DDG call N*45s after the previous

    summary_index = load_summary_index()
    index_signatures = _signatures_from_summary_index(summary_index)
    all_items = []
    acts = get_acts_from_vector_store(act_start=act_start, act_end=act_end)
    if not acts:
        logger.warning("No acts found in vector store for range %d-%d", act_start, act_end)
        return []

    if len(acts) == 1:
        # Single act — run sequentially, no threading overhead
        all_items.extend(
            _process_single_act(acts[0], 0, 0.0, summary_index, index_signatures, constraints)
        )
    else:
        logger.info(
            "Parallel processing %d acts with %d workers (%.0fs stagger between starts)",
            len(acts), MAX_PARALLEL_WORKERS, STAGGER_SECONDS,
        )
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(
                    _process_single_act, act_info, idx, STAGGER_SECONDS, summary_index, index_signatures, constraints
                ): act_info
                for idx, act_info in enumerate(acts)
            }
            for future in as_completed(futures):
                act_info = futures[future]
                try:
                    act_items = future.result()
                    all_items.extend(act_items)
                    logger.info(
                        "Act '%s' completed in parallel: %d candidates",
                        act_info.get("act_name", "?")[:50], len(act_items),
                    )
                except Exception as exc:
                    logger.error(
                        "Act '%s' failed in parallel processing: %s",
                        act_info.get("act_name", "?")[:50], exc,
                    )

    # Deduplicate vs existing index (same logic as main indexing)
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in all_items
    ]
    deduped = check_indexing_candidates(dedup_list)
    for i, s in enumerate(all_items):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))

    return all_items


def run(user_message: str) -> dict:
    """
    Main entry: interpret message, run the chosen flow, append results to pending store.
    Returns { "flow_type", "act_start", "act_end", "added": int, "pending_total": int, "message" }.

    User-specified constraints (court type, result count, extra terms) are parsed from the
    message and threaded into all sub-flows. Example messages:
      "find only high court judgments for first 5 acts"
      "find 5 case laws for Hindu Marriage Act"
      "case law discovery — Telangana High Court only, limit 10"
    """
    interpreted = interpret_user_input(user_message)
    constraints = parse_search_constraints(user_message)
    # Log active constraints so the operator can see what was parsed
    if constraints.get("court_type") or constraints.get("max_results") or constraints.get("extra_terms"):
        logger.info(
            "Search constraints parsed: court_type=%s, max_results=%s, extra_terms=%s",
            constraints.get("court_type"), constraints.get("max_results"), constraints.get("extra_terms"),
        )

    flow_type = interpreted.get("flow_type", "statement")
    added = 0
    duplicates_discarded = None
    if flow_type == "first_10_acts":
        act_start = int(interpreted.get("act_start") or 1)
        act_end = int(interpreted.get("act_end") or interpreted.get("act_count") or 10)
        logger.info("Acts-range flow: processing acts %d to %d (alphabetical order)", act_start, act_end)
        items = run_acts_range_flow(act_start=act_start, act_end=act_end, constraints=constraints)
        if items:
            pending = load_pending()
            for it in items:
                pending.append({
                    "title": it.get("title") or "Case law",
                    "source_url": it.get("source_url") or "",
                    "suggested_category": it.get("suggested_category", "case_law"),
                    "signature": it.get("signature"),
                    "act_name": it.get("act_name"),
                    "summary": it.get("summary"),
                    "already_in_store": it.get("already_in_store", False),
                })
            save_pending(pending)
            added = len(items)
    else:
        topic = interpreted.get("topic") or "case laws"
        # Prefer act name(s) extracted from user message over LLM topic (LLM can hallucinate wrong act, e.g. Evidence Act instead of Christian Marriage Act)
        from_message = _extract_act_names_from_message(user_message)
        if from_message:
            topic = from_message[0].strip()
        acts = _resolve_all_acts_from_summary_index(topic)
        # If no acts resolved from index but topic looks like an act name, run act-named flow with act name only (Indian Kanoon)
        if not acts and _topic_looks_like_act_name(topic):
            acts = [(topic.strip(), "")]
        # If we had multiple act names in message, resolve each and merge (dedupe by act name)
        if from_message and len(from_message) > 1:
            seen = {a[0] for a in acts}
            for candidate in from_message[1:]:
                if not candidate or not _topic_looks_like_act_name(candidate):
                    continue
                resolved = _resolve_all_acts_from_summary_index(candidate)
                if not resolved:
                    resolved = [(candidate.strip(), "")]
                for act_name, act_summary in resolved:
                    if act_name and act_name not in seen:
                        seen.add(act_name)
                        acts.append((act_name, act_summary))
        if acts:
            # Multi-act: complete search + dedup + append for each act, then move to next
            for act_name, act_summary in acts:
                items, disc = run_act_named_flow(act_name, act_summary, constraints=constraints)
                if duplicates_discarded is None:
                    duplicates_discarded = 0
                duplicates_discarded += disc
                if not items:
                    continue
                pending = load_pending()
                urls_in_pending = {(p.get("source_url") or "").strip() for p in pending}
                for it in items:
                    url = (it.get("source_url") or "").strip()
                    if url and url in urls_in_pending:
                        continue
                    if url:
                        urls_in_pending.add(url)
                    pending.append({
                        "title": it.get("title") or "Case law",
                        "source_url": it.get("source_url") or "",
                        "suggested_category": it.get("suggested_category", "case_law"),
                        "signature": it.get("signature"),
                        "act_name": it.get("act_name"),
                        "summary": it.get("summary"),
                        "already_in_store": it.get("already_in_store", False),
                    })
                    added += 1
                save_pending(pending)
        else:
            items = run_statement_flow(topic, constraints=constraints)
            if items:
                pending = load_pending()
                for it in items:
                    pending.append({
                        "title": it.get("title") or "Case law",
                        "source_url": it.get("source_url") or "",
                        "suggested_category": it.get("suggested_category", "case_law"),
                        "signature": it.get("signature"),
                        "act_name": it.get("act_name"),
                        "summary": it.get("summary"),
                        "already_in_store": it.get("already_in_store", False),
                    })
                save_pending(pending)
                added = len(items)

    pending = load_pending()
    message = f"Added {added} document(s) for indexing. Total pending: {len(pending)}."
    if duplicates_discarded is not None and duplicates_discarded > 0:
        message += f" {duplicates_discarded} duplicate(s) discarded; {added} unique case laws proposed."
    return {
        "flow_type": flow_type,
        "topic": interpreted.get("topic"),
        "act_start": interpreted.get("act_start"),
        "act_end": interpreted.get("act_end"),
        "added": added,
        "pending_total": len(pending),
        "duplicates_discarded": duplicates_discarded,
        "message": message,
        "constraints": {
            "court_type": constraints.get("court_type"),
            "max_results": constraints.get("max_results"),
            "extra_terms": constraints.get("extra_terms") or None,
        },
    }


def retrieve_first_gate(query: str = "") -> dict:
    """
    First-gate retrieval: resolve act(s) from query (or all acts with case laws), return
    act name + act summary + list of case law (signature, summary) from summary indexes.
    No vector store access; use for fast "case laws for this act/text" response.
    """
    act_map = load_summary_index()
    bare_summaries = load_bare_act_summary_index()
    case_summaries = load_case_law_summary_index()
    q = (query or "").strip().lower()
    acts_with_cases = [
        (act_name, sig_list)
        for act_name, sig_list in act_map.items()
        if isinstance(sig_list, list) and len(sig_list) > 0
    ]
    if q:
        words = [w for w in q.split() if len(w) > 1]
        def matches(act_name: str) -> bool:
            if q in act_name.lower():
                return True
            summary = bare_summaries.get(act_name) or ""
            if q in summary.lower():
                return True
            if words and any(w in act_name.lower() or w in summary.lower() for w in words):
                return True
            return False
        acts_with_cases = [(a, lst) for a, lst in acts_with_cases if matches(a)]
    result = []
    for act_name, sig_list in acts_with_cases:
        act_summary = bare_summaries.get(act_name) or get_bare_act_summary(act_name)
        case_laws = []
        for sig in sig_list:
            if isinstance(sig, dict):
                sig = sig.get("signature") or ""
            if not isinstance(sig, str):
                continue
            summary = case_summaries.get(sig) or get_case_law_summary(sig)
            case_laws.append({"signature": sig, "summary": summary or ""})
        result.append({
            "act_name": act_name,
            "act_summary": (act_summary or "")[:2000],
            "case_laws": case_laws,
        })
    return {"acts": result, "query": query or None}
