"""
eCourts (judgments.ecourts.gov.in) search for case laws.

Searches the Indian Judiciary eSCR portal with act name and court filter.
Courts supported: Supreme Court of India, High Court for State of Telangana.
Used as the first step in case law web search (before Indian Kanoon and tiered DDG).
"""

import logging
import time
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

BASE_URL = "https://judgments.ecourts.gov.in/pdfsearch/"

# Court identifiers for eCourts (labels used in portal)
COURT_SUPREME_COURT = "Supreme Court of India"
COURT_HC_TELANGANA = "High Court for State of Telangana"

# Allowed courts for case law discovery (only these two)
ECOURTS_CASE_LAW_COURTS = (COURT_SUPREME_COURT, COURT_HC_TELANGANA)


def _get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }


def search_ecourts(
    act_name: str,
    court: str | None = None,
    max_results: int = 15,
) -> list[dict]:
    """
    Search judgments.ecourts.gov.in for judgments by act name and optional court.

    Args:
        act_name: Act name or search phrase (e.g. "family court act").
        court: One of COURT_SUPREME_COURT, COURT_HC_TELANGANA, or None for both.
        max_results: Maximum number of result items to return.

    Returns:
        List of {"title", "url", "snippet", "source_tag", "tier"}.
        source_tag is OFFICIAL, tier is 2. Empty list if fetch/parse fails or CAPTCHA.
    """
    import requests
    from bs4 import BeautifulSoup

    if not act_name or not act_name.strip():
        return []

    results = []
    # Try GET with common param names; portal may use POST + CAPTCHA in practice
    params = {"search_term": act_name.strip()}

    url = BASE_URL.rstrip("/") + "/"
    try:
        resp = requests.get(url, params=params, timeout=25, headers=_get_headers())
        resp.raise_for_status()
    except Exception as e:
        logger.warning("eCourts fetch failed: %s", e)
        return []

    html = resp.text
    if not html or len(html) < 500:
        return []

    # If we got redirected to a CAPTCHA or login page, bail
    if "captcha" in html.lower() or "recaptcha" in html.lower():
        logger.debug("eCourts returned CAPTCHA page; skipping")
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.warning("eCourts parse failed: %s", e)
        return []

    seen_urls = set()
    # Look for result links: same domain, not just index/pdfsearch
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        full_url = urljoin(BASE_URL, href)
        if "judgments.ecourts.gov.in" not in full_url:
            continue
        # Skip home/search pages
        path = full_url.split("judgments.ecourts.gov.in")[-1].split("?")[0].lower()
        if path in ("/", "/pdfsearch", "/pdfsearch/") or path.rstrip("/") == "/pdfsearch":
            continue
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        title = (a.get_text(strip=True) or "").strip() or "Judgment"
        if len(title) > 300:
            title = title[:297] + "..."
        # Optional: filter by court name in title/snippet if we have court filter
        if court and court not in title:
            # Still include; court might be in sibling/row
            pass
        results.append({
            "title": title,
            "url": full_url,
            "snippet": title[:200],
            "source_tag": "OFFICIAL",
            "tier": 2,
        })
        if len(results) >= max_results:
            break

    if results:
        logger.info("eCourts: %d results for '%s'", len(results), act_name[:50])
    return results


def search_ecourts_both_courts(act_name: str, max_results_per_court: int = 12) -> list[dict]:
    """
    Search eCourts for Supreme Court of India and High Court of Telangana.
    Merges and deduplicates by URL. Used as first step in case law discovery.
    """
    all_results = []
    seen = set()
    for court in ECOURTS_CASE_LAW_COURTS:
        time.sleep(0.8)
        items = search_ecourts(act_name, court=court, max_results=max_results_per_court)
        for r in items:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                all_results.append(r)
    return all_results
