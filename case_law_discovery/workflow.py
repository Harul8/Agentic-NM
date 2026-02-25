"""
Case law discovery workflow: dynamic (LLM-driven) except hardcoded limits.

Uses existing modules (LLM, config, vector store read, tiered_search, hybrid_retriever)
without modifying them. Only limits in limits.py are fixed.
"""

import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)

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
    score = score_query_document(query, text_content)
    if score <= SIMILARITY_THRESHOLD_LOW:
        return None
    first_two = text_content[:_FIRST_TWO_PAGES_CHARS]
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


def _indian_kanoon_candidates(query: str, max_results: int, include_content: bool, max_pages: int = 3) -> list[dict]:
    """
    Search Indian Kanoon API and return scored candidates (same shape as web candidates).

    Fetches up to max_pages pages of search results (page 0, 1, 2) to collect a larger
    pool of TIDs before fetching full document content. Each IK page returns ~10-20 docs,
    so 3 pages gives ~60 TIDs, from which we fetch documents for the first max_results.
    This triples the candidate pool without increasing the number of document fetches.
    """
    try:
        from retrieval.indian_kanoon_client import search as ik_search, get_document
    except ImportError as e:
        logger.debug("Indian Kanoon client not available: %s", e)
        return []
    from config import INDIAN_KANOON_API_TOKEN
    if not INDIAN_KANOON_API_TOKEN:
        logger.warning("Indian Kanoon API token not set (INDIAN_KANOON_API_TOKEN); act-named flow will have no IK results")
        return []

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

    logger.debug("IK query '%s': collected %d TIDs across %d pages", query[:60], len(all_tids), min(max_pages, len(all_tids) // 10 + 1))

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


def _apply_statement_selection(candidates: list[dict]) -> list[dict]:
    """Apply limits: all > HIGH, else top TOP_N_STATEMENT above LOW."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= TOP_N_STATEMENT:
        return sorted(above_high, key=lambda x: -x["score"])[:TOP_N_STATEMENT]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:TOP_N_STATEMENT]


def _apply_per_act_selection(candidates: list[dict]) -> list[dict]:
    """Per act: all >5, else top TOP_N_PER_ACT above 3."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= TOP_N_PER_ACT:
        return sorted(above_high, key=lambda x: -x["score"])[:TOP_N_PER_ACT]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:TOP_N_PER_ACT]


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
            discovery_mode=True,  # Don't short-circuit at 5 results; search all tiers
        )
        for r in results[:ACT_NAMED_FETCH_BUFFER]:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            time.sleep(0.4)
            doc = _fetch_and_score_candidate(r, score_query, include_content=True)
            if doc:
                seen_urls[url] = doc


def run_act_named_flow(act_name: str, act_summary: str) -> tuple[list[dict], int]:
    """
    User named an act: search Indian Kanoon (1) with act name, (2) with summary from index if present;
    then web search (1) with act name, (2) with summary; merge by URL, score descending, deduplicate vs index,
    keep first TOP_N_ACT_NAMED (25) unique.
    Returns (items_for_pending, duplicates_discarded_count).
    """
    logger.info("Act-named flow: act=%s, fetch up to %s from Indian Kanoon + web (act name, then summary), dedup, target %s unique",
                act_name[:50], ACT_NAMED_FETCH_BUFFER, TOP_N_ACT_NAMED)

    score_query = (act_name + " " + (act_summary or "")[:500]).strip()
    seen_urls = {}

    # Extract legal keywords from vector store (used for both IK and web search)
    act_keywords = _extract_act_keywords(act_name)
    keyword_query = f"{act_name} {act_keywords}".strip() if act_keywords else ""

    # Indian Kanoon: (1) plain act name, (2) act name + legal keywords
    # Keywords query replaces raw markdown summary — avoids IK 502s on long/markdown queries
    ik_queries = [act_name.strip()]
    if keyword_query and keyword_query.strip() != act_name.strip():
        ik_queries.append(keyword_query[:200])
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
    # Web (eCourts + DDG): only if we don't already have enough (avoids 429s when IK gives 25+)
    if len(seen_urls) < TOP_N_ACT_NAMED:
        # Pass chunks=None; _web_search_act_name_and_summary will load them from vector store
        _web_search_act_name_and_summary(act_name, act_summary, score_query, seen_urls, max_per_query=12, chunks=None)
    else:
        logger.info("Already have %d candidates from IK; skipping web search to avoid rate limits", len(seen_urls))

    scored = sorted(seen_urls.values(), key=lambda x: -(x.get("score") or 0))

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

    # Take first TOP_N_ACT_NAMED that are unique (not already_in_store) and document URLs only
    unique = []
    discarded = 0
    for s in scored:
        if len(unique) >= TOP_N_ACT_NAMED:
            break
        if s.get("already_in_store"):
            discarded += 1
            continue
        if not _is_likely_document_url(s.get("source_url") or ""):
            discarded += 1
            continue
        unique.append(s)
    if discarded:
        logger.info("Act-named: %s unique selected (target %s), %s duplicates discarded, next in rank used",
                    len(unique), TOP_N_ACT_NAMED, discarded)
    if not unique and scored:
        logger.warning("Act-named flow: 0 proposed for indexing (all %s candidates already in index)", len(scored))

    for s in unique:
        s["act_name"] = act_name
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))
    return unique, discarded


def run_statement_flow(topic: str) -> list[dict]:
    """
    Statement-based flow: Indian Kanoon API first (if token set), then tiered web search →
    fetch → score → apply limits (all >5, else top 10 above 3). Returns list for pending.
    """
    from retrieval.tiered_search import tiered_search

    logger.info("Statement flow (topic=%s): limits TOP_N=%s, score>%s or top above %s",
                topic, TOP_N_STATEMENT, SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_LOW)
    candidates = []
    seen_urls = set()

    # First: Indian Kanoon API (if configured)
    ik_list = _indian_kanoon_candidates(topic, max_results=15, include_content=True)
    for c in ik_list:
        url = c.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            candidates.append(c)

    # Then: tiered web search (official → legal portals → newspapers)
    results = tiered_search(
        query=topic,
        search_type="case_law",
        jurisdiction_state="Telangana",
        max_per_tier=12,
    )
    for r in results[:20]:
        if r.get("url") in seen_urls:
            continue
        seen_urls.add(r.get("url"))
        time.sleep(0.5)
        doc = _fetch_and_score_candidate(r, topic, include_content=True)
        if doc:
            candidates.append(doc)

    selected = _apply_statement_selection(candidates)
    selected = [s for s in selected if _is_likely_document_url(s.get("source_url") or "")]
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


def run_acts_range_flow(act_start: int = 1, act_end: int = 10) -> list[dict]:
    """
    Acts-range flow: process acts numbered act_start to act_end (1-based, inclusive) in alphabetical
    order from the vector store. For each act: IK search + keyword web search, score, dedup, keep
    top TOP_N_PER_ACT (25) per act. Generates case law summaries; deduplicates vs act-case-law index.
    Returns list for pending (title, source_url, signature, act_name, summary).

    Examples:
      run_acts_range_flow(1, 10)   → first 10 acts (same as old run_first_10_acts_flow(10))
      run_acts_range_flow(32, 78)  → acts 32 to 78 in alphabetical order
    """
    from retrieval.tiered_search import tiered_search

    summary_index = load_summary_index()
    index_signatures = _signatures_from_summary_index(summary_index)
    all_items = []
    acts = get_acts_from_vector_store(act_start=act_start, act_end=act_end)
    if not acts:
        logger.warning("No acts found in vector store for range %d-%d", act_start, act_end)
        return []

    for act_info in acts:
        act_name = act_info.get("act_name", "Unknown")
        chunks = act_info.get("chunks", [])
        act_summary = get_or_create_act_summary(act_name, chunks) or ""
        score_query = (act_name + " " + act_summary[:1500]).strip()
        logger.info("Act %s: search with act name, then summary (IK + web), top %s judgments", act_name[:50], TOP_N_PER_ACT)

        scored = []
        seen_urls = {}

        # Extract legal keywords from act chunks (available in this flow)
        act_keywords = _extract_act_keywords(act_name, chunks)
        keyword_query = f"{act_name} {act_keywords}".strip() if act_keywords else ""

        # Indian Kanoon: (1) act name, (2) act name + legal keywords
        # Replaces old raw markdown summary query that caused IK 502s / 400s on all search engines
        ik_queries = [act_name.strip()]
        if keyword_query and keyword_query.strip() != act_name.strip():
            ik_queries.append(keyword_query[:200])
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

        # Web: act name, then keyword-enriched query (no raw markdown summary)
        if len(scored) < TOP_N_PER_ACT:
            time.sleep(1.0)
            web_queries = [act_name.strip()]
            if keyword_query and keyword_query.strip() != act_name.strip():
                web_queries.append(keyword_query[:150])
            for query in web_queries:
                if not query:
                    continue
                time.sleep(0.5)
                results = tiered_search(
                    query=query,
                    search_type="case_law",
                    jurisdiction_state="Telangana",
                    max_per_tier=10,
                    discovery_mode=True,  # Don't short-circuit at 5 results
                )
                for r in results[:MAX_FETCH_PER_ACT]:
                    url = (r.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    time.sleep(0.4)
                    doc = _fetch_and_score_candidate(r, score_query, include_content=True)
                    if doc:
                        seen_urls[url] = doc
            scored = list(seen_urls.values())
        selected = _apply_per_act_selection(scored)
        selected = [s for s in selected if (s.get("signature") or "") not in index_signatures]
        by_sig = {}
        for s in selected:
            sig = s.get("signature") or s["source_url"]
            if sig not in by_sig or by_sig[sig]["score"] < s["score"]:
                by_sig[sig] = s
        selected = list(by_sig.values())
        for s in selected:
            if s.get("signature"):
                index_signatures.add(s["signature"])
        for s in selected:
            if not _is_likely_document_url(s.get("source_url") or ""):
                continue
            s["act_name"] = act_name
            all_items.append(s)  # keep content for dedup

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
    """
    interpreted = interpret_user_input(user_message)
    flow_type = interpreted.get("flow_type", "statement")
    added = 0
    duplicates_discarded = None
    if flow_type == "first_10_acts":
        act_start = int(interpreted.get("act_start") or 1)
        act_end = int(interpreted.get("act_end") or interpreted.get("act_count") or 10)
        logger.info("Acts-range flow: processing acts %d to %d (alphabetical order)", act_start, act_end)
        items = run_acts_range_flow(act_start=act_start, act_end=act_end)
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
                items, disc = run_act_named_flow(act_name, act_summary)
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
            items = run_statement_flow(topic)
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
