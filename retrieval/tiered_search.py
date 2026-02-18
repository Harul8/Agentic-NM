"""
Tiered Internet Search — Strict domain-governed web search for legal materials.

Follows the 4-tier data sourcing hierarchy:
  Tier 2: Official court websites (SC, HCs, India Code)
  Tier 3: Trusted legal portals (Indian Kanoon, Live Law, etc.)
  Tier 4: Mainstream newspapers (context only, NOT authority)

Blocked: All social media, blogs, Wikipedia, user-generated content.

When original PDFs are found, they are auto-saved to Google Drive and indexed.
"""

import os
import logging
import json
from typing import Optional

from config import (
    TIER2_OFFICIAL_COURT_DOMAINS,
    TIER3_LEGAL_PORTAL_DOMAINS,
    TIER4_NEWSPAPER_DOMAINS,
    BLOCKED_DOMAINS,
    ALL_ALLOWED_DOMAINS,
    HC_DOMAIN_BY_STATE,
)

logger = logging.getLogger(__name__)


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


def get_source_tag(url: str) -> str:
    """Get the display source tag for a URL."""
    tier = classify_source(url)
    return {
        "tier2_official": "OFFICIAL_COURT",
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
) -> list:
    """
    Tier 2: Search official court websites and India Code.

    Constructs site-specific queries for:
    - Supreme Court (sci.gov.in)
    - Relevant High Court (based on jurisdiction_state)
    - India Code (indiacode.nic.in)
    """
    results = []

    # Supreme Court - strict domain check: only sci.gov.in results
    sc_query = f"{query} site:sci.gov.in judgment"
    sc_results = _ddgs_search(sc_query, max_results=max_results)
    for r in sc_results:
        url = r.get("url", "")
        if is_blocked_source(url):
            continue
        # Ensure result is actually from sci.gov.in (site: operator isn't always perfect)
        if "sci.gov.in" not in url.lower():
            continue
        r["source_tag"] = "OFFICIAL_COURT"
        r["tier"] = 2
        results.append(r)

    # India Code (for bare acts) - strict domain check: only indiacode.nic.in results
    ic_query = f"{query} site:indiacode.nic.in"
    ic_results = _ddgs_search(ic_query, max_results=5)
    for r in ic_results:
        url = r.get("url", "")
        if is_blocked_source(url):
            continue
        # Ensure result is actually from indiacode.nic.in
        if "indiacode.nic.in" not in url.lower():
            continue
        r["source_tag"] = "OFFICIAL_COURT"
        r["tier"] = 2
        results.append(r)

    # Relevant High Court - strict domain check
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
                # Ensure result is actually from the specified HC domain
                domain_clean = hc_domain.replace("https://", "").replace("http://", "").replace("www.", "")
                if domain_clean not in url.lower():
                    continue
                r["source_tag"] = "OFFICIAL_COURT"
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


def search_tier3_legal_portals(query: str, max_results: int = 10) -> list:
    """
    Tier 3: Search trusted legal portals.
    Uses site-specific searches for each portal.
    """
    results = []
    portals = [
        ("indiankanoon.org", "judgment case law"),
        ("livelaw.in", "judgment legal news"),
        ("scobserver.in", "supreme court analysis"),
        ("barandbench.com", "legal news judgment"),
    ]

    for domain, suffix in portals:
        portal_query = f"{query} site:{domain} {suffix}"
        portal_results = _ddgs_search(portal_query, max_results=max_results // 2)
        for r in portal_results:
            if not is_blocked_source(r["url"]):
                r["source_tag"] = "LEGAL_PORTAL"
                r["tier"] = 3
                results.append(r)

    # Also do a general search filtered to allowed portals
    general_results = _ddgs_search(
        f"{query} India judgment bare act section", max_results=max_results
    )
    for r in general_results:
        tier = classify_source(r["url"])
        if tier == "tier3_legal_portal":
            r["source_tag"] = "LEGAL_PORTAL"
            r["tier"] = 3
            results.append(r)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    logger.info(f"Tier 3 search: {len(unique)} results for '{query[:80]}'")
    return unique


def search_tier4_newspapers(query: str, max_results: int = 5) -> list:
    """
    Tier 4: Search mainstream newspapers (context only, NOT legal authority).
    """
    results = []
    news_query = f"{query} India court verdict legal"
    news_results = _ddgs_search(news_query, max_results=max_results * 2)
    for r in news_results:
        tier = classify_source(r["url"])
        if tier == "tier4_newspaper":
            r["source_tag"] = "NEWS_REFERENCE"
            r["tier"] = 4
            results.append(r)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    logger.info(f"Tier 4 search: {len(unique)} results for '{query[:80]}'")
    return unique[:max_results]


def tiered_search(
    query: str,
    search_type: str = "both",
    jurisdiction_state: str = "",
    max_per_tier: int = 10,
) -> list:
    """
    Execute the full tiered search, stopping when sufficient results are found.

    Args:
        query: Specific search query (from sufficiency analysis gap)
        search_type: "bare_act", "case_law", or "both"
        jurisdiction_state: State for HC-specific search (e.g., "Karnataka")
        max_per_tier: Max results to fetch per tier

    Returns:
        List of search results, each tagged with source_tag and tier.
        Results from blocked sources are silently discarded.
    """
    all_results = []

    # Tier 2: Official court websites
    tier2 = search_tier2_official(query, jurisdiction_state, max_per_tier)
    all_results.extend(tier2)

    # If we got enough from Tier 2, we can skip lower tiers for case laws
    # But for bare acts, we should still check Tier 3 (Indian Kanoon has good bare act coverage)
    if search_type == "case_law" and len(tier2) >= 5:
        logger.info(f"Tier 2 provided {len(tier2)} case law results, skipping lower tiers")
        return _filter_and_dedupe(all_results)

    # Tier 3: Legal portals
    tier3 = search_tier3_legal_portals(query, max_per_tier)
    all_results.extend(tier3)

    if len(all_results) >= 8:
        logger.info(f"Tiers 2+3 provided {len(all_results)} results, skipping newspapers")
        return _filter_and_dedupe(all_results)

    # Tier 4: Newspapers (context only)
    tier4 = search_tier4_newspapers(query, max_per_tier // 2)
    all_results.extend(tier4)

    return _filter_and_dedupe(all_results)


def _filter_and_dedupe(results: list) -> list:
    """Remove blocked sources and duplicates."""
    seen = set()
    filtered = []
    for r in results:
        url = r.get("url", "")
        if is_blocked_source(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Content Fetching with PDF Detection
# ---------------------------------------------------------------------------

def fetch_content_and_pdf(url: str, timeout: int = 30) -> tuple:
    """
    Fetch content from a URL. Returns (text_content, pdf_bytes_or_None).

    If URL points to a PDF, downloads it and extracts text.
    If URL points to HTML, extracts readable text.
    """
    import requests

    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
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
        return (text.strip(), pdf_bytes) if text.strip() else ("", pdf_bytes)

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
            return text, pdf_bytes
        except Exception as e:
            logger.warning(f"Failed to parse HTML from {url}: {e}")
            return response.text[:5000], None

    return "", None


def _find_pdf_link(soup, base_url: str) -> Optional[str]:
    """Look for PDF download links on a page."""
    from urllib.parse import urljoin
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if href.lower().endswith(".pdf") or "download" in text and "pdf" in text:
            return urljoin(base_url, href)
    return None


def _download_pdf(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download a PDF file. Returns bytes or None."""
    import requests
    try:
        resp = requests.get(url, timeout=timeout)
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
        results = tiered_search(
            query=query,
            search_type=gap_type,
            jurisdiction_state=jurisdiction_state,
        )

        for r in results:
            tier = r.get("tier", 99)
            source_tag = r.get("source_tag", "")

            if source_tag == "NEWS_REFERENCE":
                news_results.append(r)
            elif gap_type == "bare_act":
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
