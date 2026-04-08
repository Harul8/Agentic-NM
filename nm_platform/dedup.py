"""
platform/dedup.py — Deduplication for indexing candidates.
"""
import re
import logging
import time
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache for existing signatures
# ---------------------------------------------------------------------------
# get_existing_signatures() reads the full chunk JSON(s) from disk — expensive for large corpora.
# Cache the result for _SIGS_CACHE_TTL_SEC so repeated calls during a single discovery session
# (processing 50+ acts) only hit disk once. Invalidated after actual indexing so the next
# discovery session always sees fresh data.
_SIGS_CACHE_TTL_SEC = 300.0  # 5 minutes — covers a full bulk-discovery session
_sigs_cache: dict = {"data": None, "ts": 0.0}


def invalidate_signatures_cache() -> None:
    """Force-expire the signatures cache. Call after successfully indexing new documents."""
    _sigs_cache["data"] = None
    _sigs_cache["ts"] = 0.0
    logger.debug("Signatures cache invalidated")

# Year: 4 consecutive digits (e.g. 1948, 1984)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Similarity thresholds
_TITLE_SIMILARITY_HIGH = 0.88  # above this: duplicate by title alone
_TITLE_SIMILARITY_LOW = 0.75   # below this: not duplicate; between: use content
_CONTENT_SIMILARITY_MIN = 0.65  # content must exceed this when title is borderline


def _normalize_short_title(title: str) -> str:
    """Normalize for comparison: lowercase, remove 'the', punctuation, collapse spaces."""
    if not title:
        return ""
    try:
        s = str(title).strip().lower()
    except (TypeError, ValueError):
        return ""
    # Remove leading "the"
    if s.startswith("the "):
        s = s[4:].strip()
    try:
        # Keep only letters, digits, spaces (fixed patterns; never use user data as regex)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    except re.error:
        return ""
    return s[:120]  # cap length for signature


def _extract_year(text: str) -> Optional[str]:
    """Return first 4-digit year (19xx or 20xx) found in text, or None."""
    if not text:
        return None
    try:
        s = str(text)[:2000]
    except (TypeError, ValueError):
        return None
    try:
        m = _YEAR_RE.search(s)
        return m.group(0) if m else None
    except re.error:
        return None


def _title_similarity(a: str, b: str) -> float:
    """Return similarity ratio between two normalized short titles (0–1)."""
    if not a or not b:
        return 0.0
    try:
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _content_similarity(a: str, b: str, max_chars: int = 2000) -> float:
    """Return similarity ratio between first ~2 pages of content (0–1)."""
    if not a or not b:
        return 0.0
    try:
        sa = _normalize_content_for_compare(str(a)[:max_chars])
        sb = _normalize_content_for_compare(str(b)[:max_chars])
        if not sa or not sb:
            return 0.0
        return SequenceMatcher(None, sa, sb).ratio()
    except Exception:
        return 0.0


def _normalize_content_for_compare(text: str) -> str:
    """Normalize content for similarity: collapse whitespace, lowercase."""
    if not text:
        return ""
    try:
        s = str(text).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s
    except Exception:
        return ""


def get_existing_signatures(force_reload: bool = False) -> dict:
    """
    Load bare act and case law chunks and return sets of (normalized_short_name, year)
    so we can detect duplicates by act name + year / case name + year.

    Results are cached in memory for _SIGS_CACHE_TTL_SEC (default 5 min) so bulk discovery
    sessions processing many acts don't re-read the chunk JSON files from disk on every act.
    Call invalidate_signatures_cache() after successfully indexing new documents to force a
    fresh load on the next call.

    force_reload=True bypasses the cache (used internally by invalidate_signatures_cache).
    """
    now = time.time()
    if not force_reload and _sigs_cache["data"] is not None and (now - _sigs_cache["ts"]) < _SIGS_CACHE_TTL_SEC:
        logger.debug("Signatures cache hit (age %.1fs)", now - _sigs_cache["ts"])
        return _sigs_cache["data"]

    from config import BARE_CHUNKS_V2, CASE_CHUNKS_V2
    from core.retriever import load_chunks

    out: Dict[str, List[Dict[str, str]]] = {"bare_act": [], "case_law": []}
    seen: Dict[str, set] = {"bare_act": set(), "case_law": set()}
    try:
        for chunks_path, category in [(BARE_CHUNKS_V2, "bare_act"), (CASE_CHUNKS_V2, "case_law")]:
            try:
                chunks = load_chunks(chunks_path)
            except Exception as e:
                logger.debug("Skip chunk file %s: %s", chunks_path, e)
                continue
            if not chunks:
                continue
            name_key = "act_name" if category == "bare_act" else "case_name"
            for chunk in chunks.values():
                try:
                    raw_name = chunk.get(name_key) or chunk.get("source") or chunk.get("source_file") or ""
                    name = str(raw_name).strip() if raw_name is not None else ""
                    if not name:
                        continue
                    short = _normalize_short_title(name)
                    if not short:
                        continue
                    content = chunk.get("full_text") or chunk.get("text") or ""
                    content_slice = str(content)[:2000] if content else ""
                    year = _extract_year(name) or _extract_year(content_slice)
                    key = (short, year or "")
                    if key in seen[category]:
                        continue
                    seen[category].add(key)
                    out[category].append({"short": short, "year": year or "", "content": content_slice})
                except Exception as e:
                    logger.debug("Skip chunk for duplicate check: %s", e)
                    continue
    except Exception as e:
        logger.warning("get_existing_signatures failed: %s", e)
    logger.debug("Existing signatures loaded from disk: %d bare act, %d case law", len(out["bare_act"]), len(out["case_law"]))

    # Store in cache
    _sigs_cache["data"] = out
    _sigs_cache["ts"] = time.time()
    return out


def _is_similar_to_internal(
    short: str,
    year: str,
    content: str,
    sigs: List[Dict[str, str]],
) -> bool:
    """
    Return True if (short, year, content) is similar to any internal signature.
    Uses title similarity; when borderline (0.75-0.88), uses first-two-pages comparison.
    """
    for sig in sigs:
        s_year = sig.get("year", "")
        if year and s_year and year != s_year:
            continue
        sim = _title_similarity(short, sig.get("short", ""))
        if sim >= _TITLE_SIMILARITY_HIGH:
            return True
        if sim >= _TITLE_SIMILARITY_LOW:
            s_content = sig.get("content", "")
            if content and s_content:
                csim = _content_similarity(content, s_content)
                if csim >= _CONTENT_SIMILARITY_MIN:
                    return True
    return False


def _is_similar_to_candidate(
    short: str,
    year: str,
    content: str,
    others: List[Dict[str, Any]],
) -> bool:
    """
    Return True if (short, year, content) is similar to any previous candidate (web vs web).
    """
    for o in others:
        o_short = o.get("_short", "")
        o_year = o.get("_year", "")
        o_content = o.get("content", "")
        if year and o_year and year != o_year:
            continue
        sim = _title_similarity(short, o_short)
        if sim >= _TITLE_SIMILARITY_HIGH:
            return True
        if sim >= _TITLE_SIMILARITY_LOW and content and o_content:
            csim = _content_similarity(content, o_content)
            if csim >= _CONTENT_SIMILARITY_MIN:
                return True
    return False


def is_duplicate_of_existing(
    title: str,
    content_start: Optional[str] = None,
    suggested_category: str = "bare_act",
    existing: Optional[dict] = None,
) -> bool:
    """
    Return True if this candidate appears to be already in the internal store
    (same short title + year; optionally use first ~2 pages of content to confirm year).

    suggested_category: "bare_act" or "case_law"
    """
    if not title or not title.strip():
        return False
    short = _normalize_short_title(title)
    if not short:
        return False
    year = _extract_year(title)
    if not year and content_start:
        year = _extract_year((content_start or "")[:2000])
    year = year or ""

    if existing is None:
        existing = get_existing_signatures()
    sigs = existing.get(suggested_category) or []
    content = (content_start or "")[:2000] if content_start else ""
    return _is_similar_to_internal(short, year, content, sigs)


def check_indexing_candidates(candidates: list) -> list:
    """
    For each candidate dict with title, source_url, suggested_category, optionally content,
    add "already_in_store": True if it duplicates an existing indexed document (internal vs web)
    or a previous candidate (web vs web). Uses similarity on short title and first two pages.
    """
    if not candidates:
        return candidates
    try:
        existing = get_existing_signatures()
    except Exception as e:
        logger.warning("Could not load existing index for duplicate check: %s", e)
        return candidates

    result = []
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        title = c.get("title", "")
        content = c.get("content", "") or ""
        content_slice = str(content)[:2000] if content else ""
        cat = c.get("suggested_category", "bare_act")
        short = _normalize_short_title(title)
        year = _extract_year(title) or _extract_year(content_slice) or ""

        # Internal vs web: similarity check
        dup_internal = False
        if short:
            sigs = existing.get(cat) or []
            dup_internal = _is_similar_to_internal(short, year, content_slice, sigs)

        # Web vs web: similarity check against previously kept candidates
        dup_web = False
        if short and not dup_internal:
            dup_web = _is_similar_to_candidate(short, year, content_slice, kept)

        already_in_store = dup_internal or dup_web
        entry = {**c, "already_in_store": already_in_store, "_short": short, "_year": year}
        result.append({**c, "already_in_store": already_in_store})
        if not already_in_store:
            kept.append(entry)
    return result
