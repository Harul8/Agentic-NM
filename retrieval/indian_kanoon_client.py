"""
Indian Kanoon API client (api.indiankanoon.org).

Uses Token authentication. Set INDIAN_KANOON_API_TOKEN in .env.
Search and fetch document by id; returns JSON. Document content may be HTML — stripped to text for scoring.
"""

import logging
import re

logger = logging.getLogger(__name__)

BASE_URL = "https://api.indiankanoon.org"


def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; Nyaymalaw/1.0)",
    }


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities to get plain text."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return text.strip()


def search(query: str, pagenum: int = 0, max_results: int = 20, token: str | None = None) -> list[dict]:
    """
    Search Indian Kanoon API. Returns list of {tid, title, headline, docsource, url}.

    Args:
        query: Search query (act name, keywords, etc.).
        pagenum: Page number (0-based).
        max_results: Max number of results to return (API returns a page; we slice).
        token: API token. If None, config.INDIAN_KANOON_API_TOKEN is used.
    """
    if token is None:
        try:
            from config import INDIAN_KANOON_API_TOKEN
            token = INDIAN_KANOON_API_TOKEN
        except Exception:
            token = ""
    if not token:
        logger.warning("Indian Kanoon API token not set")
        return []

    import requests

    # Keep query short to avoid 502 / URI/body limits; act name + short phrase is enough
    query = (query or "").strip()
    if len(query) > 350:
        query = query[:347].rstrip() + "..."
    if not query:
        return []

    # API expects POST; GET returns 405 Method Not Allowed
    url = f"{BASE_URL}/search/"
    payload = {"formInput": query, "pagenum": pagenum}
    try:
        resp = requests.post(url, data=payload, headers=_get_headers(token), timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Indian Kanoon search failed: %s", e)
        return []

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("Indian Kanoon search JSON parse failed: %s", e)
        return []

    docs = data.get("docs") or data.get("doc") or []
    if isinstance(docs, dict):
        docs = [docs]
    results = []
    for d in docs[:max_results]:
        tid = d.get("tid") or d.get("docid")
        if not tid:
            continue
        title = (d.get("title") or "").strip() or "Judgment"
        results.append({
            "tid": str(tid),
            "title": title,
            "headline": d.get("headline", ""),
            "docsource": d.get("docsource", ""),
            "url": f"https://indiankanoon.org/doc/{tid}/",
        })
    return results


def get_document(doc_id: str, token: str | None = None) -> dict | None:
    """
    Fetch a single document by id. Returns dict with doc_text, title, url.

    doc_text is plain text (HTML stripped). If API returns only "doc" (HTML), we strip tags.
    """
    if token is None:
        try:
            from config import INDIAN_KANOON_API_TOKEN
            token = INDIAN_KANOON_API_TOKEN
        except Exception:
            token = ""
    if not token:
        return None

    import requests

    url = f"{BASE_URL}/doc/{doc_id}/"
    try:
        resp = requests.get(url, headers=_get_headers(token), timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.debug("Indian Kanoon get_document failed for %s: %s", doc_id, e)
        return None

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("Indian Kanoon doc JSON parse failed: %s", e)
        return None

    # API may return "doc" (HTML) or "doc_text" or similar
    raw_doc = data.get("doc_text") or data.get("doc") or data.get("content") or ""
    if isinstance(raw_doc, str) and raw_doc.strip().startswith("<"):
        doc_text = _strip_html(raw_doc)
    else:
        doc_text = (raw_doc or "").strip()

    title = (data.get("title") or "").strip() or "Judgment"
    url_public = data.get("url") or f"https://indiankanoon.org/doc/{doc_id}/"

    return {
        "doc_text": doc_text,
        "title": title,
        "url": url_public,
    }
