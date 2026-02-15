"""
Response Generator - Pulls relevant bare acts, case laws (vector store + internet),
explains relevance, and returns structured response.

Flow:
1. Expand facts to legal search query
2. Retrieve bare act sections from vector store, filter by relevance, explain each
3. Retrieve case laws from vector store, filter by relevance, explain each
4. If no local case laws: search internet for HC/SC judgments, fetch content,
   extract RELEVANT PORTIONS only, explain each
5. Return structured response with clear relevance explanations
"""

import os
import json
import logging
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    EXPAND_LEGAL_QUERY_SYSTEM,
    EXTRACT_BARE_ACT_PORTIONS_SYSTEM,
    EXTRACT_CASE_PORTIONS_SYSTEM,
    RELEVANCE_EXPLANATION_SYSTEM,
    CONVERSATIONAL_SUMMARY_SYSTEM,
)

# Paths – from config (DATA_ROOT, e.g. Google Drive)
from config import (
    VECTOR_STORE,
    BARE_INDEX,
    BARE_CHUNKS,
    CASE_INDEX,
    CASE_CHUNKS,
    BARE_ACTS_DIR,
    CASELAW_DIR,
)

logger = logging.getLogger(__name__)
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def expand_legal_query(facts: str) -> str:
    """Convert plain-language facts to a precise legal research query (advocate-style)."""
    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}

FACTS:
{facts[:1500]}

Query:"""
    try:
        return ask_llm(prompt).strip()[:500] or facts[:300]
    except Exception:
        return facts[:300]


# Similarity thresholds (IndexFlatIP with normalised vectors)
# Bare acts: high bar so we get only clearly relevant sections; no limit on count
MIN_SIMILARITY_BARE_ACTS = 0.52
# Case laws: same relevance bar; we take best 5 from Drive
MIN_SIMILARITY_CASE_LAWS = 0.45
BARE_ACTS_TOP_K = 50   # pull as many relevant sections as pass the threshold
CASE_LAWS_MAX = 5      # best 5 from Drive (or combined Drive + web)


def _safe_read_faiss_index(index_path: str):
    """Read FAISS index; return (index, True) or (None, False) on path/IO errors (e.g. Windows Drive path)."""
    try:
        if not os.path.exists(index_path):
            return None, False
        idx = faiss.read_index(index_path)
        return idx, True
    except Exception:
        return None, False


def _safe_write_faiss_index(index, path: str) -> bool:
    """Write FAISS index; return True on success. On failure (e.g. path with spaces on Windows), return False."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        return True
    except Exception:
        return False


def _clean_source_name(source: str) -> str:
    """Turn a filename like 'THE INDIAN CONTRACT ACT 1872.pdf' into a readable act name.
    Returns empty string for meaningless filenames (e.g. '250884_2_english_01042024.pdf')."""
    if not source:
        return ""
    import re
    name = source
    # Strip file extension
    for ext in (".pdf", ".txt", ".json"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")
    # Remove purely numeric/date-like tokens
    tokens = name.split()
    cleaned = [t for t in tokens if not re.fullmatch(r"\d+", t)]
    # Filter out noise words that come from gazette filenames
    noise = {"english", "hindi", "cg", "dl", "gide", "registered", "extraordinary"}
    cleaned = [t for t in cleaned if t.lower() not in noise]
    name = " ".join(cleaned).strip()
    # If after cleaning we have fewer than 2 meaningful chars, it was a meaningless filename
    if len(name) < 3:
        return ""
    # Title-case if all-caps or all-lower
    if name == name.upper() or name == name.lower():
        name = name.title()
    return name


def retrieve_bare_acts(query: str, top_k: int = None, min_similarity: float = None) -> list:
    """Retrieve relevant bare act sections from vector store. High similarity, no count limit."""
    index, ok = _safe_read_faiss_index(BARE_INDEX)
    if not ok or not index or not os.path.exists(BARE_CHUNKS):
        return []
    k = top_k if top_k is not None else BARE_ACTS_TOP_K
    min_sim = min_similarity if min_similarity is not None else MIN_SIMILARITY_BARE_ACTS

    try:
        with open(BARE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        return []

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue  # not relevant enough
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])  # copy so we can add fields
            text = (chunk.get("text") or "").strip()
            if len(text) > 50 and "Not Acceptable" not in text:
                chunk["_score"] = score
                # Derive readable act_name from source if missing
                if not chunk.get("act_name"):
                    chunk["act_name"] = _clean_source_name(chunk.get("source", ""))
                results.append(chunk)
    return results


def retrieve_case_laws(query: str, top_k: int = None, min_similarity: float = None) -> list:
    """Retrieve relevant case laws from vector store. Returns best up to CASE_LAWS_MAX (5)."""
    index, ok = _safe_read_faiss_index(CASE_INDEX)
    if not ok or not index or not os.path.exists(CASE_CHUNKS):
        return []
    k = max(top_k if top_k is not None else CASE_LAWS_MAX, 15)
    min_sim = min_similarity if min_similarity is not None else MIN_SIMILARITY_CASE_LAWS

    try:
        with open(CASE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        return []

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            text = (chunk.get("text") or "").strip()
            if len(text) > 50 and "Not Acceptable" not in text:
                chunk["_score"] = score
                if not chunk.get("source") or chunk["source"] == "Unknown":
                    chunk["source"] = _clean_source_name(chunk.get("source", ""))
                results.append(chunk)
    return results[:CASE_LAWS_MAX]


def search_internet_bare_acts(query: str, max_results: int = 5) -> list:
    """Search internet for Indian bare act sections."""
    queries = [
        f"{query} India bare act section",
        f"{query} site:indiankanoon.org",
        f"{query} act section India legislation",
    ]
    seen_urls = set()
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for q in queries:
                if len(results) >= max_results:
                    break
                try:
                    for r in ddgs.text(q, max_results=max_results):
                        url = r.get("href", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append({
                                "title": r.get("title", "Unknown"),
                                "url": url,
                                "snippet": (r.get("body") or "")[:400],
                            })
                            if len(results) >= max_results:
                                break
                except Exception:
                    continue
        return results[:max_results]
    except Exception:
        return []


def fetch_bare_act_content(url: str) -> str:
    """Fetch and extract text from a bare act URL (handles both HTML and PDF)."""
    text, _ = fetch_bare_act_content_with_pdf(url)
    return text


def fetch_bare_act_content_with_pdf(url: str):
    """
    Fetch bare act URL. Returns (text, pdf_bytes_or_None).
    Only returns pdf_bytes when the response is actually a PDF (so caller can save+index only PDFs).
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or url.lower().endswith(".pdf")
            or (len(r.content) >= 5 and r.content[:5] == b"%PDF-")
        )

        if is_pdf:
            text = _extract_text_from_pdf(r.content)
            return (text, r.content) if text and _is_readable_text(text) else (text or "", r.content)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return (" ".join(text.split())[:5000], None)
    except Exception:
        return ("", None)


def extract_relevant_bare_act_portions(facts: str, title: str, content: str) -> str:
    """Use LLM to extract relevant bare act provisions from fetched content (advocate-style)."""
    if not content or len(content) < 100:
        return content[:1500] if content else ""

    prompt = f"""{EXTRACT_BARE_ACT_PORTIONS_SYSTEM}

CASE FACTS:
{facts[:600]}

DOCUMENT: {title}
---
{content[:3500]}
---

Relevant provisions only:"""
    try:
        return ask_llm(prompt).strip()[:2500] or content[:1500]
    except Exception:
        return content[:1500]


def _is_supreme_court_query(query: str) -> bool:
    """Detect if the user is specifically asking for Supreme Court judgments."""
    q = query.lower()
    sc_keywords = [
        "supreme court", "sc judgment", "sc case", "sc ruling",
        "apex court", "hon'ble supreme", "honble supreme",
        "supreme court of india", "sci judgment", "sci case",
    ]
    return any(kw in q for kw in sc_keywords)


def _is_land_acquisition_related_query(query: str) -> bool:
    """True if query is about land acquisition / compensation (for static fallback)."""
    if not query:
        return False
    q = query.lower()
    return any(term in q for term in ("land", "acquisition", "compensation", "acquired", "government"))


# Known Supreme Court judgments on land acquisition / compensation (official api.sci.gov.in PDFs).
# Used when DDGS returns no results so the user still gets relevant case laws.
_FALLBACK_SC_LAND_ACQUISITION_CASE_LAWS = [
    {
        "title": "Supreme Court – Land acquisition compensation (Section 28-A redetermination)",
        "url": "https://api.sci.gov.in/supremecourt/2017/39949/39949_2017_4_1501_42571_Judgement_13-Mar-2023.pdf",
        "sci_pdf": "https://api.sci.gov.in/supremecourt/2017/39949/39949_2017_4_1501_42571_Judgement_13-Mar-2023.pdf",
        "snippet": "Right to redetermination of compensation under Section 28-A of the Land Acquisition Act, 1894; beneficial interpretation for marginalized landowners.",
    },
    {
        "title": "Supreme Court – Land acquisition lapse (Section 24(2) of 2013 Act)",
        "url": "https://api.sci.gov.in/supremecourt/2022/9229/9229_2022_16_1501_44075_Judgement_01-May-2023.pdf",
        "sci_pdf": "https://api.sci.gov.in/supremecourt/2022/9229/9229_2022_16_1501_44075_Judgement_01-May-2023.pdf",
        "snippet": "Lapse of acquisition where neither possession taken nor compensation paid; Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act, 2013.",
    },
    {
        "title": "Supreme Court – Land acquisition and compensation",
        "url": "https://api.sci.gov.in/supremecourt/2019/25495/25495_2019_4_1502_17423_Judgement_14-Oct-2019.pdf",
        "sci_pdf": "https://api.sci.gov.in/supremecourt/2019/25495/25495_2019_4_1502_17423_Judgement_14-Oct-2019.pdf",
        "snippet": "Supreme Court judgment on land acquisition and compensation.",
    },
    {
        "title": "Supreme Court – Land acquisition compensation / post-notification purchasers",
        "url": "https://api.sci.gov.in/supremecourt/2022/17822/17822_2022_2_6_57617_Judgement_05-Dec-2024.pdf",
        "sci_pdf": "https://api.sci.gov.in/supremecourt/2022/17822/17822_2022_2_6_57617_Judgement_05-Dec-2024.pdf",
        "snippet": "Rights of post-notification purchasers; Section 4 notification and Section 24 of the 2013 Act.",
    },
]


def _get_fallback_sc_land_acquisition_case_laws(max_results: int) -> list:
    """Return up to max_results from known SC land acquisition judgments when search returns nothing."""
    return [dict(r) for r in _FALLBACK_SC_LAND_ACQUISITION_CASE_LAWS[:max_results]]


# URLs that are generic navigation pages, NOT judgments
_SCI_GENERIC_PAGES = {
    "https://www.sci.gov.in/", "https://sci.gov.in/",
    "https://www.sci.gov.in/judgements-case-no/",
    "https://www.sci.gov.in/free-text-judgements/",
    "https://www.sci.gov.in/cause-list/",
    "https://scr.sci.gov.in/", "https://scr.sci.gov.in/scrsearch/",
}


def _is_judgment_url(url: str) -> bool:
    """Return True if the URL points to an actual judgment (not a generic navigation page)."""
    if not url:
        return False
    url_lower = url.lower()
    # Reject known generic pages
    clean = url.rstrip("/") + "/"
    if clean in _SCI_GENERIC_PAGES or url.rstrip("/") + "/" in _SCI_GENERIC_PAGES:
        return False
    # sci.gov.in judgment PDFs
    if "api.sci.gov.in/supremecourt/" in url_lower:
        return True
    # scr.sci.gov.in with a specific case
    if "scr.sci.gov.in" in url_lower and len(url) > 30:
        return True
    # indiankanoon.org: accept any path (doc, docfragment, or other judgment pages)
    if "indiankanoon.org" in url_lower:
        path = url.split("indiankanoon.org", 1)[-1].strip("/")
        return len(path) > 0 and "search" not in path[:20]
    # Other legal portals that might return judgments
    if "sci.gov.in" in url_lower and ("/doc/" in url_lower or "/supremecourt/" in url_lower):
        return True
    if "sci.gov.in" in url_lower:
        return False
    return True


def _find_sci_pdf_url(case_title: str) -> str:
    """Try to find the official sci.gov.in PDF for a given case title."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(f"{case_title} site:api.sci.gov.in judgment pdf", max_results=3):
                url = r.get("href", "")
                if "api.sci.gov.in/supremecourt/" in url and url.lower().endswith(".pdf"):
                    return url
    except Exception:
        pass
    return ""


def _format_case_title_as_vs(raw_title: str) -> str:
    """Format case name as 'Appellant v/s Respondent' (e.g. 'X vs Y on 12 April 2023' -> 'X v/s Y')."""
    import re
    if not raw_title or len(raw_title) < 5:
        return raw_title or "Unknown"
    # Remove trailing " on DD Month, YYYY" or " on DD Month YYYY"
    t = re.sub(r"\s+on\s+\d{1,2}\s+\w+\s*,?\s*\d{4}\s*$", "", raw_title, flags=re.IGNORECASE).strip()
    # Normalise " vs ", " vs. ", " v. " to " v/s "
    t = re.sub(r"\s+vs\.?\s+", " v/s ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+v\.\s+", " v/s ", t, flags=re.IGNORECASE)
    # Trim to reasonable length
    return t[:120] if len(t) > 120 else t


def search_internet_case_laws(query: str, max_results: int = 5) -> list:
    """Search credible legal resources: Supreme Court, Indian Kanoon, legal portals. Lead with short site: queries."""
    is_sc = _is_supreme_court_query(query)

    # Lead with short, high-yield site-restricted queries so we get results even when long query fails
    if is_sc:
        queries = [
            "land acquisition compensation Supreme Court site:indiankanoon.org",
            "Supreme Court land acquisition government compensation site:indiankanoon.org",
            "land acquisition no compensation Supreme Court India site:indiankanoon.org",
            f"{query} site:indiankanoon.org",
            f"{query} Supreme Court judgment site:indiankanoon.org",
            f"{query} site:scr.sci.gov.in",
            "Supreme Court India land acquisition compensation judgment",
        ]
    else:
        queries = [
            f"{query} site:indiankanoon.org",
            f"{query} High Court India case law judgment",
            f"{query} India case law",
        ]

    seen_urls = set()
    results = []

    def _accept_url(url: str) -> bool:
        if not url or url in seen_urls or _is_excluded_source(url):
            return False
        return _is_judgment_url(url) or _is_allowed_legal_or_news_source(url)

    def _collect(ddgs, query_list, limit):
        for q in query_list:
            if len(results) >= limit:
                return
            try:
                for r in ddgs.text(q, max_results=max(limit * 4, 15)):
                    url = r.get("href", "")
                    if not _accept_url(url):
                        continue
                    seen_urls.add(url)
                    title = r.get("title", "Unknown")
                    sci_pdf = _find_sci_pdf_url(title) if is_sc else ""
                    results.append({
                        "title": title,
                        "url": url,
                        "sci_pdf": sci_pdf,
                        "snippet": (r.get("body") or "")[:400],
                    })
                    if len(results) >= limit:
                        return
            except Exception:
                continue

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            _collect(ddgs, queries, max_results)

            if len(results) < max_results and is_sc:
                fallback = [
                    "land acquisition compensation Supreme Court India judgment",
                    "Supreme Court land acquisition compensation site:livelaw.in",
                    "Supreme Court land acquisition site:indiankanoon.org",
                ]
                for q in fallback:
                    if len(results) >= max_results:
                        break
                    try:
                        for r in ddgs.text(q, max_results=max(max_results * 4, 15)):
                            url = r.get("href", "")
                            if not _accept_url(url):
                                continue
                            seen_urls.add(url)
                            title = r.get("title", "Unknown")
                            sci_pdf = _find_sci_pdf_url(title)
                            results.append({
                                "title": title,
                                "url": url,
                                "sci_pdf": sci_pdf,
                                "snippet": (r.get("body") or "")[:400],
                            })
                            if len(results) >= max_results:
                                break
                    except Exception:
                        continue
            # Last resort: minimal query without site: to get any legal result
            if len(results) < max_results and is_sc:
                try:
                    for r in ddgs.text("land acquisition Supreme Court India judgment indiankanoon", max_results=15):
                        url = r.get("href", "")
                        if not _accept_url(url):
                            continue
                        seen_urls.add(url)
                        title = r.get("title", "Unknown")
                        results.append({
                            "title": title,
                            "url": url,
                            "sci_pdf": _find_sci_pdf_url(title),
                            "snippet": (r.get("body") or "")[:400],
                        })
                        if len(results) >= max_results:
                            break
                except Exception:
                    pass
            # When DDGS returns nothing, use known SC land acquisition judgments so user still gets results
            if len(results) == 0 and is_sc and _is_land_acquisition_related_query(query):
                logger.info("Case law search returned 0 results; using static fallback SC land acquisition judgments.")
                results = _get_fallback_sc_land_acquisition_case_laws(max_results)
        if len(results) == 0:
            logger.info("Case law search returned 0 results for query: %s", query[:100])
        return results[:max_results]
    except Exception as e:
        logger.debug("Case law search failed: %s", e, exc_info=True)
        if _is_supreme_court_query(query) and _is_land_acquisition_related_query(query):
            logger.info("Using static fallback SC land acquisition judgments after search exception.")
            return _get_fallback_sc_land_acquisition_case_laws(max_results)
        return []


def search_internet_bare_acts_pdf_preferred(query: str, max_results: int = 5) -> list:
    """Search for bare act PDFs only (filetype:pdf). Avoids generic HTML pages."""
    seen = set()
    results = []
    try:
        from ddgs import DDGS
        queries = [
            f"{query} India bare act filetype:pdf",
            f"{query} act section India legislation filetype:pdf",
            f"{query} site:indiankanoon.org filetype:pdf",
        ]
        with DDGS() as ddgs:
            for q in queries:
                if len(results) >= max_results:
                    break
                try:
                    for r in ddgs.text(q, max_results=max_results * 2):
                        url = r.get("href", "")
                        if not url or url in seen or _is_generic_or_landing_url(url):
                            continue
                        if not _is_pdf_url(url):
                            continue
                        seen.add(url)
                        results.append({
                            "title": r.get("title", "Unknown"),
                            "url": url,
                            "snippet": (r.get("body") or "")[:400],
                        })
                        if len(results) >= max_results:
                            break
                except Exception:
                    continue
        return results[:max_results]
    except Exception:
        return []


def search_internet_case_laws_pdf_preferred(query: str, max_results: int = 5) -> list:
    """Search for case law / judgment PDFs. Prefers sci.gov.in PDFs; accepts filetype:pdf from legal sites."""
    seen = set()
    results = []
    try:
        from ddgs import DDGS
        is_sc = _is_supreme_court_query(query)
        queries = [
            f"{query} Supreme Court India judgment filetype:pdf",
            f"{query} site:api.sci.gov.in pdf",
            f"{query} India case law judgment filetype:pdf",
            f"{query} site:indiankanoon.org filetype:pdf",
        ] if is_sc else [
            f"{query} High Court India judgment filetype:pdf",
            f"{query} India case law filetype:pdf",
            f"{query} site:indiankanoon.org filetype:pdf",
        ]
        with DDGS() as ddgs:
            for q in queries:
                if len(results) >= max_results:
                    break
                try:
                    for r in ddgs.text(q, max_results=max_results * 2):
                        url = r.get("href", "")
                        if not url or url in seen or _is_excluded_source(url) or _is_generic_or_landing_url(url):
                            continue
                        if not _is_pdf_url(url) and not _is_judgment_url(url):
                            continue
                        seen.add(url)
                        title = r.get("title", "Unknown")
                        sci_pdf = _find_sci_pdf_url(title) if is_sc else ""
                        results.append({
                            "title": title,
                            "url": url,
                            "sci_pdf": sci_pdf,
                            "snippet": (r.get("body") or "")[:400],
                        })
                        if len(results) >= max_results:
                            break
                except Exception:
                    continue
        return results[:max_results]
    except Exception:
        return []


def _extract_text_from_pdf(content_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content_bytes))
        text_parts = []
        for page in reader.pages[:30]:  # limit to 30 pages
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
        text = " ".join(" ".join(text_parts).split())[:8000]
        return text if _is_readable_text(text) else ""
    except Exception:
        return ""


def _is_pdf_url(url: str) -> bool:
    """True if URL points to a PDF (by extension or known host)."""
    if not url:
        return False
    u = url.lower().split("?")[0]
    if u.endswith(".pdf"):
        return True
    if "api.sci.gov.in/supremecourt/" in u and ".pdf" in u:
        return True
    return False


def _is_generic_or_landing_url(url: str) -> bool:
    """True if URL is a generic/landing page we should not store or index."""
    if not url:
        return True
    u = url.lower().rstrip("/")
    # Known generic pages
    if u in _SCI_GENERIC_PAGES or u + "/" in _SCI_GENERIC_PAGES:
        return True
    # Home/root paths
    for base in ("https://www.sci.gov.in", "https://sci.gov.in", "https://indiankanoon.org"):
        if u == base or u == base + "/" or u.startswith(base + "/search") or u.startswith(base + "/?") or u == base.rstrip("/"):
            return True
    # Obvious non-document paths
    if "/search" in u and u.count("/") <= 4:
        return True
    return False


def _search_legal_news_or_analysis(query: str, max_results: int = 5) -> list:
    """Search only legal portals + mainstream newspapers (no Quora, LinkedIn, social). For response only; never stored."""
    try:
        from ddgs import DDGS
        results = []
        seen = set()
        # Prefer legal portals and mainstream news via site: queries
        site_queries = [
            f"{query} Supreme Court judgment site:livelaw.in",
            f"{query} Supreme Court India site:thehindu.com",
            f"{query} Supreme Court site:indianexpress.com",
            f"{query} Supreme Court judgment site:scobserver.in",
            f"{query} land acquisition Supreme Court site:indiankanoon.org",
        ]
        with DDGS() as ddgs:
            for q in site_queries:
                if len(results) >= max_results:
                    break
                try:
                    for r in ddgs.text(q, max_results=max_results):
                        url = r.get("href", "")
                        if not url or url in seen or _is_excluded_source(url):
                            continue
                        if not _is_allowed_legal_or_news_source(url):
                            continue
                        seen.add(url)
                        results.append({
                            "source": r.get("title", "Unknown"),
                            "text": (r.get("body") or "")[:500] or "See source link for details.",
                            "url": url,
                        })
                        if len(results) >= max_results:
                            break
                except Exception:
                    continue
        return results[:max_results]
    except Exception:
        return []


# Domains we treat as official legal sources; only PDFs from these are stored in Drive/vector DB
_OFFICIAL_LEGAL_DOMAINS = (
    "sci.gov.in",
    "api.sci.gov.in",
    "indiankanoon.org",
    "scobserver.in",
    "livelaw.in",
    "livelaw.in/",
    "www.livelaw.in",
    "lawcommissionofindia.nic.in",
    "legislative.gov.in",
    "indiacode.nic.in",
)

# Never use these for case laws or legal research (social / Q&A)
_EXCLUDED_SOURCE_DOMAINS = (
    "quora.com",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "instagram.com",
    "youtube.com",
    "medium.com",
    "pinterest.com",
)

# For news fallback only: legal portals + mainstream newspapers (no social media)
_LEGAL_AND_MAINSTREAM_NEWS_DOMAINS = (
    "livelaw.in",
    "scobserver.in",
    "barandbench.com",
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "economictimes.indiatimes.com",
    "ndtv.com",
    "firstpost.com",
    "sci.gov.in",
    "indiankanoon.org",
)


def _is_excluded_source(url: str) -> bool:
    """True if URL is from a source we never use (Quora, LinkedIn, social media)."""
    if not url:
        return True
    u = url.lower()
    return any(d in u for d in _EXCLUDED_SOURCE_DOMAINS)


def _is_allowed_legal_or_news_source(url: str) -> bool:
    """True if URL is from an allowed fallback: legal portal or mainstream newspaper only."""
    if not url:
        return False
    u = url.lower()
    return any(d in u for d in _LEGAL_AND_MAINSTREAM_NEWS_DOMAINS)


def _is_official_legal_source(url: str) -> bool:
    """True if URL is from an official legal source (SC, Indian Kanoon, SC Observer, LiveLaw, etc.). Only these get stored in Drive/vector DB."""
    if not url:
        return False
    u = url.lower().strip()
    return any(domain in u for domain in _OFFICIAL_LEGAL_DOMAINS)


# Bare acts = government gazette / legislation. Case laws = court orders (appellant v/s respondent, judgment).
_CASE_LIKE_PHRASES = (" v. ", " v/s ", " vs ", " appellant ", " respondent ", " petitioner ", " judgment ", " judgement ", " order ", " supreme court ", " high court ", " hon'ble ", " honble ")
_BARE_ACT_DOMAINS = ("indiacode.nic.in", "legislative.gov.in", "egazette.nic.in", "lawcommissionofindia.nic.in")


def _looks_like_case_law(url: str, title: str) -> bool:
    """True if source appears to be a court order (appellant v/s respondent, judgment), not a bare act/gazette."""
    u = (url or "").lower()
    t = (title or "").lower()
    if "sci.gov.in" in u or "api.sci.gov.in" in u or "scr.sci.gov.in" in u:
        return True
    if any(p in t for p in _CASE_LIKE_PHRASES):
        return True
    return False


def _looks_like_bare_act(url: str, title: str) -> bool:
    """True if source appears to be legislation/gazette (bare act), not a court judgment."""
    u = (url or "").lower()
    t = (title or "").lower()
    if any(d in u for d in _BARE_ACT_DOMAINS):
        return True
    if any(p in t for p in _CASE_LIKE_PHRASES):
        return False  # clearly a case
    if " act " in t or " section " in t or " gazette " in t or "legislation" in t:
        return True
    return False


def _safe_filename(title: str, max_len: int = 100) -> str:
    """Produce a safe filename from a title; ensure .pdf extension."""
    import re
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in (title or "document"))
    safe = re.sub(r"_+", "_", safe).strip("_")[:max_len] or "document"
    return safe + ".pdf" if not safe.lower().endswith(".pdf") else safe


def save_pdf_to_drive(content_bytes: bytes, filename: str, folder_kind: str) -> str:
    """
    Save PDF bytes to Drive (BareActs or CaseLaws). Creates dirs if needed.
    folder_kind: "BareActs" or "CaseLaws"
    Returns absolute path if saved, else empty string.
    """
    if not content_bytes or len(content_bytes) < 100 or content_bytes[:5] != b"%PDF-":
        return ""
    root = BARE_ACTS_DIR if folder_kind == "BareActs" else CASELAW_DIR
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    try:
        with open(path, "wb") as f:
            f.write(content_bytes)
        return os.path.abspath(path)
    except Exception:
        return ""


def _is_readable_text(text: str) -> bool:
    """Check if text is actually readable (not garbled PDF binary or encoding noise)."""
    if not text or len(text) < 20:
        return False
    sample = text[:500]
    printable = sum(1 for c in sample if c.isprintable() or c.isspace())
    return (printable / len(sample)) > 0.75


def _looks_like_navigation(text: str) -> bool:
    """True if text is mostly website navigation/chrome rather than judgment or act content."""
    if not text or len(text) < 100:
        return False
    sample = (text[:1500] or "").lower()
    nav_phrases = [
        "skip to main content",
        "indian kanoon",
        "search engine for indian law",
        "main navigation",
        "free features",
        "premium features",
        "prism ai",
        "pricing",
        "login",
        "mobile navigation",
        "legal document view",
        "tools for analyzing",
        "document options",
        "get in pdf",
        "print it!",
        "download court copy",
    ]
    matches = sum(1 for p in nav_phrases if p in sample)
    return matches >= 3


def _clean_scraped_text(text: str) -> str:
    """Remove common website navigation noise from scraped text."""
    import re
    # Common noise phrases from indiankanoon and other legal sites
    noise_patterns = [
        r"Skip to main content.*?(?=\n|$)",
        r"Indian Kanoon - Search engine.*?(?=\n|$)",
        r"Main Navigation.*?(?=\n|$)",
        r"Free features.*?(?=\n|$)",
        r"Premium features.*?(?=\n|$)",
        r"Prism AI.*?(?=\n|$)",
        r"Pricing\s*Login.*?(?=\n|$)",
        r"Mobile Navigation.*?(?=\n|$)",
        r"Legal Document View.*?(?=\n|$)",
        r"Tools for analyzing.*?(?=\n|$)",
        r"Document Options.*?(?=\n|$)",
        r"Get in PDF.*?(?=\n|$)",
        r"Print it!.*?(?=\n|$)",
        r"Download Court Copy.*?(?=\n|$)",
        r"Cites \d+.*?Cited by \d+.*?(?=\n|$)",
        r"Search\s+Search\s+",
        r"Accessibility Links.*?Cursor",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # Remove multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_case_content(url: str) -> str:
    """Fetch and extract text from a case law URL (handles both HTML and PDF)."""
    text, _ = fetch_case_content_with_pdf(url)
    return text


def fetch_case_content_with_pdf(url: str):
    """
    Fetch case law URL. Returns (text, pdf_bytes_or_None).
    Only returns pdf_bytes when the response is actually a PDF (so caller can save+index only PDFs).
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or url.lower().endswith(".pdf")
            or (len(r.content) >= 5 and r.content[:5] == b"%PDF-")
        )

        if is_pdf:
            text = _extract_text_from_pdf(r.content)
            return (text, r.content) if text and _is_readable_text(text) else (text or "", r.content)

        # HTML: extract text and clean navigation noise
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        judgment_div = (
            soup.find("div", {"id": "judgments"})
            or soup.find("div", {"id": "content"})
            or soup.find("div", class_=lambda c: c and "document" in " ".join(c).lower())
            or soup.find("div", class_=lambda c: c and "judgment" in " ".join(c).lower())
            or soup.find("pre")
            or soup.find("article")
            or soup.find("main")
            or soup
        )
        text = judgment_div.get_text(separator="\n", strip=True)
        text = _clean_scraped_text(text)
        return (" ".join(text.split())[:6000], None)
    except Exception:
        return ("", None)


def extract_relevant_case_portions(facts: str, case_title: str, case_content: str) -> str:
    """Use LLM to summarise a judgment in its own words, highlighting what's relevant to the user's query."""
    if not case_content or len(case_content) < 100:
        return case_content[:1500] if case_content else ""

    # Check if the content is garbled PDF binary (not properly extracted)
    printable_ratio = sum(1 for c in case_content[:500] if c.isprintable() or c.isspace()) / max(len(case_content[:500]), 1)
    if printable_ratio < 0.7:
        return ""  # unusable content

    prompt = f"""You are a legal research assistant. Read this judgment and write a clear, concise summary in your own words.

Structure your summary as:
1. **Case**: Who were the parties and which court decided it.
2. **Facts**: 2-3 sentences on what happened.
3. **Issue**: The key legal question(s) before the court.
4. **Held**: What the court decided and the key legal principle(s) established.
5. **Relevance**: 1-2 sentences on why this is relevant to the user's query.

Keep the total summary to 150-250 words. Write naturally — do NOT copy raw text from the judgment.

USER'S QUERY:
{facts[:600]}

JUDGMENT: {case_title}
---
{case_content[:4000]}
---

Summary:"""
    try:
        summary = ask_llm(prompt).strip()
        if summary and len(summary) > 50:
            return summary[:2500]
        return case_content[:1500]
    except Exception:
        return case_content[:1500]


def generate_relevance_explanation(
    facts: str,
    bare_sections: list,
    case_laws: list,
    internet_cases: list,
) -> str:
    """
    Generate structured legal opinion explanation (used for legal_opinion intent).
    """
    if not bare_sections and not case_laws and not internet_cases:
        try:
            no_mat_prompt = (
                f"You are a legal assistant. The user described the following situation:\n\n{facts[:1200]}\n\n"
                "After searching our database and the internet, no relevant bare act provisions or case laws were found. "
                "Write a short, helpful response to the user: acknowledge their query, explain that no matching legal materials were found, "
                "and suggest how they could refine their request (e.g. more specific terms, different jurisdiction, specific act names). "
                "Be concise and professional."
            )
            return ask_llm(no_mat_prompt).strip()
        except Exception:
            return ""

    bare_text = json.dumps([{"source": c.get("source"), "text": (c.get("text") or "")[:600]} for c in bare_sections], indent=2)[:2500]
    case_text = json.dumps([{"source": c.get("source"), "text": (c.get("text") or "")[:600]} for c in case_laws], indent=2)[:2500]
    net_text = json.dumps([{"title": c.get("title"), "relevant_portion": c.get("relevant_portion", c.get("content", ""))[:500]} for c in internet_cases], indent=2)[:2000]

    prompt = f"""{RELEVANCE_EXPLANATION_SYSTEM}

---
CLIENT'S CASE FACTS:
{facts[:1200]}

---
BARE ACT SECTIONS:
{bare_text}

---
CASE LAWS (from database):
{case_text}

---
CASE LAWS (from internet - if any):
{net_text}
---

Write the analysis as specified above."""

    try:
        return ask_llm(prompt).strip()
    except Exception:
        return ""


def generate_conversational_summary(
    facts: str,
    bare_sections: list,
    case_laws: list,
) -> str:
    """
    Generate a warm, conversational intro summary (used for search/lookup intents).
    Reads like a ChatGPT-style greeting + topic briefing.
    Always returns something meaningful — never empty.
    """
    num_cases = len(case_laws)
    num_bare = len(bare_sections)

    if not bare_sections and not case_laws:
        try:
            return ask_llm(
                f"You are a friendly legal research assistant. The user asked: \"{facts[:500]}\"\n"
                "Unfortunately no results were found. Write a short, warm response acknowledging their query "
                "and suggesting how to refine it. Be conversational."
            ).strip()
        except Exception:
            return f"I searched for materials related to your query but couldn't find matching results. You could try using more specific legal terms or mentioning particular acts or courts."

    # Build a brief digest of the retrieved content for the LLM to summarise
    material_digest = ""
    for i, c in enumerate(bare_sections[:5]):
        material_digest += f"Bare Act {i+1}: {c.get('act_name', c.get('source', ''))} — {(c.get('text') or '')[:400]}\n"
    for i, c in enumerate(case_laws[:5]):
        material_digest += f"Case Law {i+1}: {c.get('source', '')} — {(c.get('text') or '')[:400]}\n"

    prompt = f"""{CONVERSATIONAL_SUMMARY_SYSTEM}

USER'S QUERY:
{facts[:600]}

RETRIEVED MATERIALS (for context — summarise the topic, don't list these):
{material_digest[:3500]}

Write your response now:"""

    try:
        result = ask_llm(prompt).strip()
        if result and len(result) > 30:
            return result
    except Exception:
        pass

    # Fallback: generate a basic summary if LLM fails
    parts = []
    if num_cases > 0:
        parts.append(f"{num_cases} Supreme Court judgment{'s' if num_cases > 1 else ''}")
    if num_bare > 0:
        parts.append(f"{num_bare} relevant bare act provision{'s' if num_bare > 1 else ''}")
    materials_str = " and ".join(parts)
    return f"I found {materials_str} related to your query. Here's what I retrieved for you:"



def add_bare_act_to_index(bare_act_data: dict):
    if not bare_act_data or not bare_act_data.get("text"):
        return

    index, ok = _safe_read_faiss_index(BARE_INDEX)
    if not ok or not index:
        index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
    try:
        with open(BARE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        chunks = {}

    new_chunk = {
        "source": bare_act_data.get("title", "Internet"),
        "text": bare_act_data["text"],
        "act_name": bare_act_data.get("act_name", "Bare Act"),
    }
    text_to_embed = new_chunk["text"]
    embedding = embedder.encode(
        text_to_embed, convert_to_numpy=True, normalize_embeddings=True
    )
    index.add(np.array([embedding], dtype="float32"))
    new_id = str(len(chunks))
    chunks[new_id] = new_chunk

    if _safe_write_faiss_index(index, BARE_INDEX):
        try:
            with open(BARE_CHUNKS, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
        except Exception:
            pass
    print(f"Indexed new bare act: {bare_act_data.get('title', 'Internet')}")


def web_fallback_bare_acts_with_save(issue: str, max_results: int = 5) -> list:
    """
    Search web for bare acts; when we get an original PDF, save to Drive (BareActs) and index.
    Uses regular search first so we always get results. Returns list of {source, text, act_name, url} for display.
    """
    results = []
    bare_results = search_internet_bare_acts(issue, max_results=max_results)
    if not bare_results:
        bare_results = search_internet_bare_acts_pdf_preferred(issue, max_results=max_results)
    for r in bare_results:
        url = r.get("url", "")
        if _is_generic_or_landing_url(url):
            continue
        content, pdf_bytes = fetch_bare_act_content_with_pdf(url)
        relevant = ""
        if content and _is_readable_text(content):
            relevant = extract_relevant_bare_act_portions(issue, r.get("title", ""), content)
        text = relevant if _is_readable_text(relevant) else ""
        if not text and content and _is_readable_text(content):
            text = content[:1500]
        if not text:
            text = r.get("snippet", "")
        if not text or not _is_readable_text(text):
            text = f"Relevant content from: {r.get('title', 'Unknown')}. See source link for full text."
        title = r.get("title", "Internet")
        if pdf_bytes and _is_official_legal_source(url) and _looks_like_bare_act(url, title):
            fname = _safe_filename(title)
            save_pdf_to_drive(pdf_bytes, fname, "BareActs")
            add_bare_act_to_index({
                "title": title,
                "text": text,
                "act_name": r.get("title", "Unknown"),
                "url": url,
            })
        results.append({
            "source": title,
            "text": text,
            "act_name": r.get("title", "Unknown"),
            "url": url,
        })
    return results


def web_fallback_case_laws_with_save(issue: str, max_results: int = 5) -> list:
    """
    Search web for case laws; when we get an original PDF, save to Drive (CaseLaws) and index.
    Uses regular search first so we always get results. Returns list of {source, text, url} for display.
    """
    results = []
    web_results = search_internet_case_laws(issue, max_results=max_results)
    if not web_results:
        web_results = search_internet_case_laws_pdf_preferred(issue, max_results=max_results)
    for r in web_results:
        url = r.get("url", "")
        pdf_url = r.get("sci_pdf") or url
        if _is_generic_or_landing_url(pdf_url) and _is_generic_or_landing_url(url):
            continue
        content, pdf_bytes = fetch_case_content_with_pdf(pdf_url)
        if not content and pdf_url != url:
            content, pdf_bytes = fetch_case_content_with_pdf(url)
        relevant_portion = ""
        if content and _is_readable_text(content) and not _looks_like_navigation(content):
            relevant_portion = extract_relevant_case_portions(issue, r.get("title", ""), content)
        text = relevant_portion if _is_readable_text(relevant_portion) else ""
        if not text and content and _is_readable_text(content) and not _looks_like_navigation(content):
            text = content[:1500]
        if not text:
            text = r.get("snippet", "")
        if not text or not _is_readable_text(text):
            text = "Summary of this judgment is available at the source link below."
        display_url = r.get("sci_pdf") or url
        title = r.get("title", "Internet")
        if pdf_bytes and _is_official_legal_source(display_url) and _looks_like_case_law(display_url, title):
            fname = _safe_filename(title)
            save_pdf_to_drive(pdf_bytes, fname, "CaseLaws")
            add_case_law_to_index({
                "title": title,
                "relevant_portion": text,
                "url": display_url,
            })
        results.append({
            "source": title,
            "text": text,
            "url": display_url,
        })
    return results


def add_case_law_to_index(case_law_data: dict):
    if not case_law_data or not case_law_data.get("relevant_portion"):
        return

    index, ok = _safe_read_faiss_index(CASE_INDEX)
    if not ok or not index:
        index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
    try:
        with open(CASE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        chunks = {}

    new_chunk = {
        "source": case_law_data.get("title", "Internet"),
        "text": case_law_data["relevant_portion"],
    }
    text_to_embed = new_chunk["text"]
    embedding = embedder.encode(
        text_to_embed, convert_to_numpy=True, normalize_embeddings=True
    )
    index.add(np.array([embedding], dtype="float32"))
    new_id = str(len(chunks))
    chunks[new_id] = new_chunk

    if _safe_write_faiss_index(index, CASE_INDEX):
        try:
            with open(CASE_CHUNKS, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
        except Exception:
            pass
    print(f"Indexed new case law: {case_law_data.get('title', 'Internet')}")


def _generate_title(existing_name: str, text: str, doc_type: str) -> str:
    """Generate a short, meaningful title from the content using the LLM.
    For bare acts: extracts the actual act name (e.g. 'The Indian Contract Act, 1872 — Section 73').
    For case laws: extracts the case citation (e.g. 'State of Bihar v. Kameshwar Singh')."""
    # If existing_name is already a proper act/case name, use it
    if existing_name and len(existing_name) > 8:
        # Check it's not a raw filename (contains meaningful words, not just numbers)
        import re
        words = [w for w in existing_name.split() if not re.fullmatch(r"\d+", w)]
        if len(words) >= 2:
            return existing_name

    # Ask LLM to extract the official name from the content
    try:
        snippet = text[:800].replace("\n", " ").strip()
        if doc_type == "bare act provision":
            prompt = (
                "From this bare act excerpt, extract the OFFICIAL ACT NAME and the specific section/provision "
                "it deals with. Format: '<Act Name, Year> — <Section/Provision>'. "
                "Example: 'The Indian Contract Act, 1872 — Section 73 (Compensation for breach)'. "
                "Output ONLY the title, nothing else.\n\n"
                f"Excerpt: {snippet}"
            )
        else:
            prompt = (
                "From this judgment excerpt, extract the CASE NAME (parties) and court. "
                "Format: '<Party 1> v. <Party 2> (<Court>, <Year>)'. "
                "Example: 'State of Bihar v. Kameshwar Singh (Supreme Court, 1952)'. "
                "Output ONLY the title, nothing else.\n\n"
                f"Excerpt: {snippet}"
            )
        title = ask_llm(prompt).strip().strip('"').strip("'").strip()
        if title and 5 < len(title) < 150:
            return title
    except Exception:
        pass
    # Fallback
    if existing_name and len(existing_name) > 3:
        return existing_name
    first_line = text.split("\n")[0].strip()[:80]
    return first_line or "Untitled"


def _merge_and_rank_case_laws(drive_list: list, web_list: list, query: str) -> list:
    """Combine Drive and web case laws; score web items by similarity to query; return best 5 by score."""
    if not web_list:
        return drive_list[:CASE_LAWS_MAX]
    try:
        query_vec = embedder.encode(query or " ", convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        return drive_list[:CASE_LAWS_MAX]
    for item in web_list:
        text = (item.get("text") or item.get("source") or "")[:2000]
        if not text:
            item["_score"] = 0.0
            continue
        try:
            vec = embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            score = float(np.dot(query_vec, vec))
            item["_score"] = score
        except Exception:
            item["_score"] = 0.0
    combined = list(drive_list) + list(web_list)
    combined.sort(key=lambda x: float(x.get("_score", 0)), reverse=True)
    return combined[:CASE_LAWS_MAX]


def generate_response(facts_summary: str, confirmed_materials: dict = None, top_k: int = 5, intent: str = "legal_opinion") -> dict:
    """
    Full legal research response.
    Drive first: bare acts (high similarity, no limit); case laws (best 5).
    If < 5 case laws from Drive, search web, combine, take best 5. Only official PDFs stored.
    """
    legal_query = expand_legal_query(facts_summary)
    search_query = f"{facts_summary} {legal_query}"[:500]

    # 1) Drive first: bare acts (all relevant, high similarity); case laws (best 5)
    bare_sections = list(retrieve_bare_acts(search_query, top_k=BARE_ACTS_TOP_K))
    case_laws_local = list(retrieve_case_laws(search_query))

    # If user confirmed materials, add them and generate full response
    if confirmed_materials:
        for b in confirmed_materials.get("bare_acts", []):
            add_bare_act_to_index(b)  # Index the new bare act
            bare_sections.append({
                "source": b.get("title", "Internet"),
                "text": b.get("text", b.get("content", b.get("snippet", ""))),
                "act_name": b.get("title", b.get("act_name", "Bare Act")),
            })
        for c in confirmed_materials.get("case_laws", []):
            add_case_law_to_index(c)  # Index the new case law
            case_laws_local.append({
                "source": c.get("title", "Internet"),
                "text": c.get("relevant_portion", c.get("content", c.get("snippet", ""))),
            })
        # Fall through to generate full response below

    # 2) Bare acts: if Drive had none, search web; store only official PDFs
    if not bare_sections:
        bare_results = search_internet_bare_acts(legal_query, max_results=top_k)
        if not bare_results:
            bare_results = search_internet_bare_acts_pdf_preferred(legal_query, max_results=top_k)
        if not bare_results:
            bare_results = search_internet_bare_acts(facts_summary[:250].strip() or legal_query[:200], max_results=top_k)
        for r in bare_results:
            url = r.get("url", "")
            if _is_generic_or_landing_url(url):
                continue
            content, pdf_bytes = fetch_bare_act_content_with_pdf(url)
            relevant = ""
            if content and _is_readable_text(content):
                relevant = extract_relevant_bare_act_portions(facts_summary, r.get("title", ""), content)
            text = relevant if _is_readable_text(relevant) else ""
            if not text and content and _is_readable_text(content):
                text = content[:1500]
            if not text:
                text = r.get("snippet", "")
            if not text or not _is_readable_text(text):
                text = f"Relevant content from: {r.get('title', 'Unknown')}. See source link for full text."
            # Store only if official PDF and clearly a bare act (gazette/legislation), not a court order
            title = r.get("title", "Internet")
            if pdf_bytes and _is_official_legal_source(url) and _looks_like_bare_act(url, title):
                fname = _safe_filename(title)
                save_pdf_to_drive(pdf_bytes, fname, "BareActs")
                add_bare_act_to_index({
                    "title": title,
                    "text": text,
                    "act_name": r.get("title", "Unknown"),
                    "url": url,
                })
            bare_sections.append({
                "source": r.get("title", "Internet"),
                "text": text,
                "act_name": r.get("title", "Unknown"),
                "url": url,
            })

    # 3) Case laws: if fewer than 5 from Drive, search web (min 5); combine and take best 5; store only official PDFs
    case_law_limit = max(top_k, CASE_LAWS_MAX)  # always request at least 5 case laws
    if len(case_laws_local) < CASE_LAWS_MAX:
        web_results = search_internet_case_laws(legal_query, max_results=case_law_limit)
        if not web_results:
            web_results = search_internet_case_laws_pdf_preferred(legal_query, max_results=case_law_limit)
        if not web_results:
            web_results = search_internet_case_laws(facts_summary[:250].strip() or legal_query[:200], max_results=case_law_limit)
        web_case_list = []
        for r in web_results:
            url = r.get("url", "")
            if _is_excluded_source(url):
                continue
            pdf_url = r.get("sci_pdf") or url
            if _is_generic_or_landing_url(pdf_url) and _is_generic_or_landing_url(url):
                continue
            content, pdf_bytes = fetch_case_content_with_pdf(pdf_url)
            if not content and pdf_url != url:
                content, pdf_bytes = fetch_case_content_with_pdf(url)
            relevant_portion = ""
            if content and _is_readable_text(content) and not _looks_like_navigation(content):
                relevant_portion = extract_relevant_case_portions(facts_summary, r.get("title", ""), content)
            text = relevant_portion if _is_readable_text(relevant_portion) else ""
            if not text and content and _is_readable_text(content) and not _looks_like_navigation(content):
                text = content[:1500]
            if not text:
                text = r.get("snippet", "")
            if not text or not _is_readable_text(text):
                text = "Summary of this judgment is available at the source link below."
            display_url = r.get("sci_pdf") or url
            title = r.get("title", "Internet")
            if pdf_bytes and _is_official_legal_source(display_url) and _looks_like_case_law(display_url, title):
                fname = _safe_filename(title)
                save_pdf_to_drive(pdf_bytes, fname, "CaseLaws")
                add_case_law_to_index({
                    "title": title,
                    "relevant_portion": text,
                    "url": display_url,
                })
            web_case_list.append({
                "source": title,
                "text": text,
                "url": display_url,
            })
        if not web_case_list:
            web_case_list = _search_legal_news_or_analysis(legal_query, max_results=CASE_LAWS_MAX)
        case_laws_local = _merge_and_rank_case_laws(case_laws_local, web_case_list, search_query)

    # Hide bare acts when user asked only for case laws/judgments (regardless of intent)
    query_lower = facts_summary.lower()
    case_law_only = any(
        phrase in query_lower
        for phrase in ["case law", "case laws", "caselaws", "judgment", "judgments", "judgement", "judgements", "ruling", "pull", "find", "get", "show me"]
    ) and not any(
        phrase in query_lower
        for phrase in ["bare act", "bare acts", "sections", "provisions", "act sections"]
    )

    bare_for_display = bare_sections
    if case_law_only:
        bare_for_display = []
    elif intent == "search" and bare_sections:
        # Filter to only acts that match query keywords
        stop = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "by", "from", "with", "is", "are", "was", "were", "be", "been", "have", "has", "can", "could", "that", "this", "it", "as", "at"}
        words = [w for w in query_lower.split() if len(w) > 2 and w not in stop][:15]
        if words:
            def _bare_relevant(c):
                combined = f"{(c.get('act_name') or '')} {(c.get('source') or '')} {(c.get('text') or '')[:400]}".lower()
                return sum(1 for w in words if w in combined) >= max(1, len(words) // 3)
            bare_for_display = [c for c in bare_sections if _bare_relevant(c)]
        if not bare_for_display:
            bare_for_display = []

    # Generate explanation: conversational for search/lookup, formal for legal_opinion
    if intent in ("search", "lookup"):
        explanation = generate_conversational_summary(
            facts_summary, bare_for_display, case_laws_local
        )
    else:
        explanation = generate_relevance_explanation(
            facts_summary, bare_sections, case_laws_local, []
        )

    bare_formatted = []
    for c in bare_for_display:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        act_name = c.get("act_name") or _clean_source_name(c.get("source", "")) or ""
        title = _generate_title(act_name, text, "bare act provision")
        bare_formatted.append({
            "source": c.get("source", ""),
            "text": text,
            "act_name": act_name,
            "title": title,
            "url": c.get("url", ""),
        })

    case_formatted = []
    for c in case_laws_local:
        text = (c.get("text") or "").strip()
        source = c.get("source") or ""
        # Heading: "Appellant v/s Respondent" (from search title / source)
        display_title = _format_case_title_as_vs(source)
        # Brief summary: use existing text (LLM summary or snippet); ensure it's not empty
        if not text:
            text = "Summary of this judgment is available at the source link below."
        case_formatted.append({
            "source": source,
            "text": text,
            "title": display_title,
            "url": c.get("url", ""),  # PDF link when available (sci.gov.in), else source page
        })

    if not (explanation or "").strip():
        explanation = "Here's what I found for your query. Below are any relevant case laws and provisions."
    return {
        "needs_confirmation": False,
        "bare_act_sections": bare_formatted,
        "case_laws": case_formatted,
        "internet_case_laws": [],
        "explanation": explanation,
    }
