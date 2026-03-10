"""
Tiered Internet Search — Strict domain-governed web search for legal materials.

Search order is fixed and never bypassed:
  1. Official sources (Tier 2): courts (SCI, HC), India Code, legislative.gov.in — tag: OFFICIAL
  2. Legal portals (Tier 3): only if needed — Indian Kanoon, Live Law, Bar and Bench, etc. — tag: LEGAL_PORTAL
  3. Newspapers (Tier 4): only if still needed — context only, NOT authority — tag: NEWS_REFERENCE
  4. Stop there — no other sources are used. Blocked and unknown domains are always discarded.

All web search for legal content must go through tiered_search() so this order is enforced.
When original PDFs are found, they are auto-saved to Google Drive and indexed.
"""

import os
import logging
import json
import time
import threading
from typing import Optional

from config import (
    TIER2_OFFICIAL_COURT_DOMAINS,
    TIER3_LEGAL_PORTAL_DOMAINS,
    TIER4_NEWSPAPER_DOMAINS,
    BLOCKED_DOMAINS,
    ALL_ALLOWED_DOMAINS,
    HC_DOMAIN_BY_STATE,
    OFFICIAL_SOURCE_TAG,
)

logger = logging.getLogger(__name__)

# Small delay before each DDG request to reduce 429 (Too Many Requests) from search backends
DDGS_REQUEST_DELAY_SEC = 2.0

# Global cross-thread DDG rate limiter — ensures minimum DDGS_REQUEST_DELAY_SEC between ANY
# DDG call across all threads. Replaces per-thread time.sleep() which allowed simultaneous
# calls when multiple acts ran in parallel.
_ddgs_rate_lock = threading.Lock()
_ddgs_last_call_ts: list[float] = [0.0]  # mutable list so all threads share one reference

# ---------------------------------------------------------------------------
# P5: URL text content cache — avoids re-fetching the same URL during
# discovery (skip_index=True) and again during manual indexing.
# Only caches text (not PDF bytes, which may be large); TTL = 2 hours.
# ---------------------------------------------------------------------------
_URL_CONTENT_CACHE_TTL_SEC = 7200.0  # 2 hours
_url_content_cache: dict = {}       # url -> {"text": str, "ts": float}
_url_cache_lock = threading.Lock()


def _url_cache_get(url: str):
    """Return cached text for url, or None if missing / expired."""
    with _url_cache_lock:
        entry = _url_content_cache.get(url)
        if entry and (time.time() - entry["ts"]) < _URL_CONTENT_CACHE_TTL_SEC:
            return entry["text"]
        return None


def _url_cache_set(url: str, text: str) -> None:
    """Store text in the URL content cache."""
    with _url_cache_lock:
        _url_content_cache[url] = {"text": text, "ts": time.time()}


# ---------------------------------------------------------------------------
# Domain Classification
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    url = url.lower().strip()
    # Remove protocol
    for prefix in ("https://", "http://", "www."):
        if url.startswith(prefix):
            url = url[len(prefix):]
    # Take domain part
    return url.split("/")[0].split("?")[0]


def classify_source(url: str) -> str:
    """
    Classify a URL into a source tier.
    Returns: "tier2_official", "tier3_legal_portal", "tier4_newspaper", "blocked", or "unknown"
    """
    domain = _get_domain(url)
    if not domain:
        return "blocked"

    # Check blocked first
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return "blocked"

    # Check tiers
    for official in TIER2_OFFICIAL_COURT_DOMAINS:
        if official in domain or domain in official:
            return "tier2_official"

    for portal in TIER3_LEGAL_PORTAL_DOMAINS:
        if portal in domain or domain in portal:
            return "tier3_legal_portal"

    for newspaper in TIER4_NEWSPAPER_DOMAINS:
        if newspaper in domain or domain in newspaper:
            return "tier4_newspaper"

    return "unknown"


def is_allowed_source(url: str) -> bool:
    """Check if URL is from any allowed domain (Tier 2, 3, or 4)."""
    tier = classify_source(url)
    return tier in ("tier2_official", "tier3_legal_portal", "tier4_newspaper")


def is_blocked_source(url: str) -> bool:
    """Check if URL is from a blocked domain."""
    tier = classify_source(url)
    return tier in ("blocked", "unknown")


# Only these tiers are allowed in search results. Order: official → legal portals → newspapers; stop there.
# Only official sources (courts, India Code); legal portals and newspapers removed.
ALLOWED_TIERS = ("tier2_official",)


def get_source_tag(url: str) -> str:
    """Get the source tag for a URL: OFFICIAL (government sources), LEGAL_PORTAL, NEWS_REFERENCE, or UNKNOWN."""
    tier = classify_source(url)
    return {
        "tier2_official": OFFICIAL_SOURCE_TAG,
        "tier3_legal_portal": "LEGAL_PORTAL",
        "tier4_newspaper": "NEWS_REFERENCE",
    }.get(tier, "UNKNOWN")


# ---------------------------------------------------------------------------
# Tiered Search Execution
# ---------------------------------------------------------------------------

def _ddgs_search(query: str, max_results: int = 10) -> list:
    """
    Execute a DuckDuckGo search. Returns list of {title, url, snippet}.
    Strictly filters out blocked domains (Wikipedia, social media, etc.) per tier hierarchy.
    
    Note: DuckDuckGo/primp may query Wikipedia's API internally as part of its search aggregation,
    but Wikipedia results are filtered out here and never returned. Only official sources
    (Tier 2: courts, Tier 3: legal portals, Tier 4: newspapers) are allowed.
    
    Suppresses primp's internal logging to avoid noise from Wikipedia/other search engine API calls.
    """
    import logging as std_logging
    # Cross-thread rate limiter: enforce minimum DDGS_REQUEST_DELAY_SEC between any DDG call
    # globally (not just within a single thread). Safe for parallel act processing.
    with _ddgs_rate_lock:
        now = time.time()
        wait = DDGS_REQUEST_DELAY_SEC - (now - _ddgs_last_call_ts[0])
        if wait > 0:
            time.sleep(wait)
        _ddgs_last_call_ts[0] = time.time()
    # Suppress primp's verbose logging (it logs all internal API calls to Wikipedia, Google, etc.)
    primp_logger = std_logging.getLogger("primp")
    original_level = primp_logger.level
    primp_logger.setLevel(std_logging.WARNING)  # Only show warnings/errors, not INFO
    
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href", r.get("url", ""))
                if not url:
                    continue
                # Strict filtering: block Wikipedia and other blocked domains immediately
                if is_blocked_source(url):
                    continue  # Skip Wikipedia, social media, blogs, etc.
                results.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("body", r.get("snippet", "")),
                })
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for '{query}': {e}")
        return []
    finally:
        # Restore original logging level
        primp_logger.setLevel(original_level)


def search_tier2_official(
    query: str,
    jurisdiction_state: str = "",
    max_results: int = 10,
    search_type: str = "both",
) -> list:
    """
    Tier 2: Search official court websites and India Code.

    search_type: "bare_act" = only legislation (India Code, legislative.gov.in); no court judgments.
    "case_law" = only courts (Supreme Court, High Court judgments). "both" = all.
    """
    results = []

    # India Code / legislation (for bare acts) — only when we want acts, not judgments
    if search_type in ("bare_act", "both"):
        ic_query = f"{query} site:indiacode.nic.in"
        ic_results = _ddgs_search(ic_query, max_results=max_results if search_type == "bare_act" else 5)
        for r in ic_results:
            url = r.get("url", "")
            if is_blocked_source(url):
                continue
            if "indiacode.nic.in" not in url.lower():
                continue
            r["source_tag"] = OFFICIAL_SOURCE_TAG
            r["tier"] = 2
            results.append(r)

        # Legislative.gov.in (central acts)
        leg_query = f"{query} site:legislative.gov.in"
        leg_results = _ddgs_search(leg_query, max_results=3)
        for r in leg_results:
            url = r.get("url", "")
            if is_blocked_source(url):
                continue
            if "legislative.gov.in" not in url.lower():
                continue
            r["source_tag"] = OFFICIAL_SOURCE_TAG
            r["tier"] = 2
            results.append(r)

    # Supreme Court & High Courts (judgments only) — only when we want case laws
    if search_type in ("case_law", "both"):
        sc_query = f"{query} site:sci.gov.in judgment"
        sc_results = _ddgs_search(sc_query, max_results=max_results)
        for r in sc_results:
            url = r.get("url", "")
            if is_blocked_source(url):
                continue
            url_lower = url.lower()
            if "sci.gov.in" not in url_lower:
                continue
            # Filter out non-judgment SCI pages: cause lists, case status trackers,
            # NJDG portals, and SCI home/navigation pages all score -10 to -11
            # in the cross-encoder and waste fetch time.  Only accept URLs whose
            # path suggests actual judgment or order content.
            _JUDGMENT_PATH_MARKERS = (
                "/judgment", "/judgement", "/judgments", "/judgements",
                "/order", "/supct", ".pdf",
            )
            if not any(marker in url_lower for marker in _JUDGMENT_PATH_MARKERS):
                logger.debug("sci.gov.in: skipping non-judgment URL: %s", url[:100])
                continue
            r["source_tag"] = OFFICIAL_SOURCE_TAG
            r["tier"] = 2
            results.append(r)

        # Relevant High Court
        if jurisdiction_state:
            state_lower = jurisdiction_state.lower().strip()
            hc_domain = HC_DOMAIN_BY_STATE.get(state_lower)
            if hc_domain:
                hc_query = f"{query} site:{hc_domain}"
                hc_results = _ddgs_search(hc_query, max_results=max_results)
                for r in hc_results:
                    url = r.get("url", "")
                    if is_blocked_source(url):
                        continue
                    domain_clean = hc_domain.replace("https://", "").replace("http://", "").replace("www.", "")
                    if domain_clean not in url.lower():
                        continue
                    r["source_tag"] = OFFICIAL_SOURCE_TAG
                    r["tier"] = 2
                    results.append(r)

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    logger.info(f"Tier 2 search: {len(unique)} results for '{query[:80]}'")
    return unique


def tiered_search(
    query: str,
    search_type: str = "both",
    jurisdiction_state: str = "",
    max_per_tier: int = 10,
    discovery_mode: bool = False,
) -> list:
    """
    Execute web search using only official sources (eCourts + Tier 2: courts, India Code).
    Legal portals and newspapers have been removed; all callers get official-only results.

    Args:
        query: Search query (e.g. from sufficiency analysis or case law discovery)
        search_type: "bare_act", "case_law", or "both"
        jurisdiction_state: State for HC-specific search (e.g., "Telangana", "Karnataka")
        max_per_tier: Max results to fetch per tier
        discovery_mode: Unused (kept for API compatibility).

    Returns:
        List of search results, each with source_tag and tier. Only official (tier2_official).
    """
    all_results = []

    # 1) Official sources (courts for case_law, India Code for bare_act, or both)
    if search_type == "case_law":
        time.sleep(2)
    tier2 = search_tier2_official(query, jurisdiction_state, max_per_tier, search_type=search_type)
    all_results.extend(tier2)

    logger.info("Tiered search complete (official only): %d results", len(all_results))
    return _filter_and_dedupe(all_results)


def _filter_and_dedupe(results: list, allowed_tiers: tuple = None) -> list:
    """
    Keep only allowed tiers; remove duplicates. Default is official only (tier2_official).
    When allowed_tiers is set, only those tiers are kept.
    """
    tiers = allowed_tiers if allowed_tiers is not None else ALLOWED_TIERS
    seen = set()
    filtered = []
    for r in results:
        url = r.get("url", "")
        if not url or classify_source(url) not in tiers:
            continue
        if url in seen:
            continue
        seen.add(url)
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Content Fetching with PDF Detection
# ---------------------------------------------------------------------------

def _browser_headers() -> dict:
    """Headers that reduce 404s from sites (e.g. India Code) that block non-browser requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }


def fetch_content_and_pdf(url: str, timeout: int = 30) -> tuple:
    """
    Fetch content from a URL. Returns (text_content, pdf_bytes_or_None).

    If URL points to a PDF, downloads it and extracts text.
    If URL points to HTML, extracts readable text.

    P5: Text content (not PDF bytes) is cached for _URL_CONTENT_CACHE_TTL_SEC so the same
    URL is not re-fetched when a document is first discovered (skip_index=True) and then
    manually indexed shortly after.
    """
    import requests

    # SCI certificate is for api.sci.gov.in; www causes SSL hostname mismatch
    if "www.api.sci.gov.in" in url:
        url = url.replace("www.api.sci.gov.in", "api.sci.gov.in")

    # P5: Return cached text immediately (PDF bytes not cached — they may be large)
    cached_text = _url_cache_get(url)
    if cached_text is not None:
        logger.debug("URL content cache hit: %s", url[:80])
        return cached_text, None

    # tshc.gov.in uses a cert not in Python's certifi bundle (works fine in browsers)
    ssl_verify = False if "tshc.gov.in" in url else True
    if not ssl_verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    headers = _browser_headers()
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers, verify=ssl_verify)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return "", None

    content_type = response.headers.get("Content-Type", "").lower()

    # PDF
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        import io
        pdf_bytes = response.content
        # Extract text: pdfplumber first, then pypdf fallback
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for PDF at {url}: {e}")
        if not text.strip():
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                if text.strip():
                    logger.info("Extracted PDF text using pypdf fallback")
            except Exception as e2:
                logger.warning(f"pypdf fallback failed for PDF at {url}: {e2}")
        result_text = text.strip()
        # P5: Cache extracted PDF text (not the raw bytes)
        if result_text:
            _url_cache_set(url, result_text)
        return (result_text, pdf_bytes) if result_text else ("", pdf_bytes)

    # HTML
    if "html" in content_type or "text" in content_type:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Check if page links to a PDF we should download
            pdf_link = _find_pdf_link(soup, url)
            pdf_bytes = None
            if pdf_link:
                pdf_bytes = _download_pdf(pdf_link, timeout)
            # P5: Cache the extracted HTML text
            if text:
                _url_cache_set(url, text)
            return text, pdf_bytes
        except Exception as e:
            logger.warning(f"Failed to parse HTML from {url}: {e}")
            fallback = response.text[:5000]
            _url_cache_set(url, fallback)
            return fallback, None

    return "", None


def _find_pdf_link(soup, base_url: str) -> Optional[str]:
    """Look for PDF download links on a page (e.g. India Code 'View PDF', bitstream links)."""
    from urllib.parse import urljoin
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        # Direct PDF URL, or link text suggests PDF (View PDF, Download PDF, etc.)
        if href.lower().endswith(".pdf"):
            return urljoin(base_url, href)
        if "pdf" in text and ("download" in text or "view" in text or "pdf" in href.lower() or "/bitstream/" in href.lower()):
            return urljoin(base_url, href)
    return None


def _download_pdf(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download a PDF file. Returns bytes or None."""
    import requests
    # tshc.gov.in uses a cert not in Python's certifi bundle
    ssl_verify = False if "tshc.gov.in" in url else True
    if not ssl_verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        resp = requests.get(url, timeout=timeout, headers=_browser_headers(), verify=ssl_verify)
        if resp.ok and len(resp.content) > 1000:
            return resp.content
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# High-Level: Search for Gaps
# ---------------------------------------------------------------------------

def search_for_gaps(
    gaps: list,
    jurisdiction_state: str = "",
) -> dict:
    """
    Execute targeted internet searches for each identified gap.

    Args:
        gaps: List from sufficiency_analyzer.get_targeted_search_queries()
              Each: {"query": "...", "type": "bare_act"|"case_law"|"both"}
        jurisdiction_state: For HC-specific searches

    Returns:
        {
            "bare_act_results": [...],
            "case_law_results": [...],
            "news_results": [...],
        }
    """
    bare_results = []
    case_results = []
    news_results = []

    for gap in gaps:
        query = gap.get("query", "")
        gap_type = gap.get("type", "both")

        if not query:
            continue

        logger.info(f"Searching for gap: '{query}' (type={gap_type})")
        # Short delay between gap searches to reduce 429 rate limiting from search engines
        time.sleep(1.5)
        results = tiered_search(
            query=query,
            search_type=gap_type,
            jurisdiction_state=jurisdiction_state,
        )

        for r in results:
            source_tag = r.get("source_tag", "")
            url = (r.get("url") or "").lower()
            title = (r.get("title") or "").lower()

            if source_tag == "NEWS_REFERENCE":
                news_results.append(r)
            elif gap_type == "bare_act":
                # Only keep results that look like legislation, not court judgments or case names
                if "sci.gov.in" in url or "judgment" in title or "judgement" in title:
                    continue
                # Skip case-law-style titles (e.g. "Appellant vs Respondent", "X vs State of Y")
                if " vs " in title or " v. " in title or " v/s " in title:
                    continue
                bare_results.append(r)
            elif gap_type == "case_law":
                case_results.append(r)
            else:
                # "both" — classify by content
                title_lower = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                if any(p in title_lower for p in [" v. ", " v/s ", " vs ", "judgment", "judgement"]):
                    case_results.append(r)
                else:
                    bare_results.append(r)

    logger.info(
        f"Gap search complete: {len(bare_results)} bare acts, "
        f"{len(case_results)} case laws, {len(news_results)} news refs"
    )

    return {
        "bare_act_results": bare_results,
        "case_law_results": case_results,
        "news_results": news_results,
    }
