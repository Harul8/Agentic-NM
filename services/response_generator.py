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

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE = os.path.join(BASE_DIR, "data", "vector_store")
BARE_INDEX = os.path.join(VECTOR_STORE, "bareacts.index")
BARE_CHUNKS = os.path.join(VECTOR_STORE, "bareacts_chunks.json")
CASE_INDEX = os.path.join(VECTOR_STORE, "caselaws.index")
CASE_CHUNKS = os.path.join(VECTOR_STORE, "caselaws_chunks.json")

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


# Minimum cosine similarity to consider a result relevant (IndexFlatIP with normalised vectors)
# 0.45 filters out clearly irrelevant results while keeping reasonably related ones
MIN_SIMILARITY = 0.45


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


def retrieve_bare_acts(query: str, top_k: int = 10) -> list:
    """Retrieve relevant bare act sections from vector store. Filters by similarity threshold."""
    if not os.path.exists(BARE_INDEX) or not os.path.exists(BARE_CHUNKS):
        return []

    index = faiss.read_index(BARE_INDEX)
    with open(BARE_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < MIN_SIMILARITY:
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


def retrieve_case_laws(query: str, top_k: int = 10) -> list:
    """Retrieve relevant case laws from vector store. Filters by similarity threshold."""
    if not os.path.exists(CASE_INDEX) or not os.path.exists(CASE_CHUNKS):
        return []

    index = faiss.read_index(CASE_INDEX)
    with open(CASE_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([query_vec], dtype="float32"), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < MIN_SIMILARITY:
            continue  # not relevant enough
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            text = (chunk.get("text") or "").strip()
            if len(text) > 50 and "Not Acceptable" not in text:
                chunk["_score"] = score
                if not chunk.get("source") or chunk["source"] == "Unknown":
                    chunk["source"] = _clean_source_name(chunk.get("source", ""))
                results.append(chunk)
    return results


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
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or url.lower().endswith(".pdf")
            or r.content[:5] == b"%PDF-"
        )

        if is_pdf:
            return _extract_text_from_pdf(r.content)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return " ".join(text.split())[:5000]
    except Exception:
        return ""


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
    """Search credible legal resources: Supreme Court (sci.gov.in), Indian Kanoon, legal portals."""
    is_sc = _is_supreme_court_query(query)

    if is_sc:
        queries = [
            f"{query} Supreme Court of India site:indiankanoon.org",
            f"{query} Supreme Court judgment site:indiankanoon.org",
            f"{query} land acquisition compensation Supreme Court India",
            f"{query} site:scr.sci.gov.in",
            "Supreme Court India land acquisition compensation judgment",
        ]
    else:
        queries = [
            f"{query} High Court India case law judgment",
            f"{query} site:indiankanoon.org",
            f"{query} India case law",
        ]

    seen_urls = set()
    results = []

    def _collect(ddgs, query_list, limit):
        for q in query_list:
            if len(results) >= limit:
                return
            try:
                for r in ddgs.text(q, max_results=limit * 2):
                    url = r.get("href", "")
                    if not url or url in seen_urls:
                        continue
                    if not _is_judgment_url(url):
                        continue
                    seen_urls.add(url)
                    title = r.get("title", "Unknown")
                    sci_pdf = ""
                    if is_sc:
                        sci_pdf = _find_sci_pdf_url(title)
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

            # If still no results, try with very permissive URL acceptance (any indiankanoon/sci.gov)
            if len(results) < max_results and is_sc:
                fallback = [
                    f"{query} Supreme Court judgment India",
                    "land acquisition compensation Supreme Court India judgment",
                    "Supreme Court land acquisition compensation",
                ]
                for q in fallback:
                    if len(results) >= max_results:
                        break
                    try:
                        for r in ddgs.text(q, max_results=max_results * 2):
                            url = r.get("href", "")
                            if not url or url in seen_urls:
                                continue
                            # Accept any URL from known legal sources
                            if "indiankanoon" in url.lower() or "sci.gov.in" in url.lower() or "scr.sci.gov.in" in url.lower():
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
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or url.lower().endswith(".pdf")
            or r.content[:5] == b"%PDF-"
        )

        if is_pdf:
            return _extract_text_from_pdf(r.content)

        # HTML: extract text and clean navigation noise
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        # Indian Kanoon / legal portals: judgment body is often in id="judgments", class="document", pre, or main content
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
        return " ".join(text.split())[:6000]
    except Exception:
        return ""


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

    # Load existing index and chunks
    try:
        index = faiss.read_index(BARE_INDEX)
        with open(BARE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
        chunks = {}

    # Prepare new chunk
    new_chunk = {
        "source": bare_act_data.get("title", "Internet"),
        "text": bare_act_data["text"],
        "act_name": bare_act_data.get("act_name", "Bare Act"),
    }

    # Embed and add to index
    text_to_embed = new_chunk["text"]
    embedding = embedder.encode(
        text_to_embed, convert_to_numpy=True, normalize_embeddings=True
    )
    index.add(np.array([embedding], dtype="float32"))

    # Add to chunks
    new_id = str(len(chunks))
    chunks[new_id] = new_chunk

    # Save updated index and chunks
    faiss.write_index(index, BARE_INDEX)
    with open(BARE_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Indexed new bare act: {bare_act_data.get('title', 'Internet')}")


def add_case_law_to_index(case_law_data: dict):
    if not case_law_data or not case_law_data.get("relevant_portion"):
        return

    # Load existing index and chunks
    try:
        index = faiss.read_index(CASE_INDEX)
        with open(CASE_CHUNKS, encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
        chunks = {}

    # Prepare new chunk
    new_chunk = {
        "source": case_law_data.get("title", "Internet"),
        "text": case_law_data["relevant_portion"],
    }

    # Embed and add to index
    text_to_embed = new_chunk["text"]
    embedding = embedder.encode(
        text_to_embed, convert_to_numpy=True, normalize_embeddings=True
    )
    index.add(np.array([embedding], dtype="float32"))

    # Add to chunks
    new_id = str(len(chunks))
    chunks[new_id] = new_chunk

    # Save updated index and chunks
    faiss.write_index(index, CASE_INDEX)
    with open(CASE_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

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


def generate_response(facts_summary: str, confirmed_materials: dict = None, top_k: int = 5, intent: str = "legal_opinion") -> dict:
    """
    Full legal research response.
    top_k: how many results to retrieve per category (bare acts, case laws). Default 5.
    intent: "search", "lookup", or "legal_opinion" — controls the explanation style.
    confirmed_materials: {bare_acts: [], case_laws: []} - from user confirmation
    """
    legal_query = expand_legal_query(facts_summary)
    search_query = f"{facts_summary} {legal_query}"[:500]

    bare_sections = list(retrieve_bare_acts(search_query, top_k=top_k))
    case_laws_local = list(retrieve_case_laws(search_query, top_k=top_k))

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

    # When local vector DB has no suitable data: search internet and USE results directly
    if not bare_sections or not case_laws_local:
        if not bare_sections:
            bare_results = search_internet_bare_acts(legal_query, max_results=top_k)
            for r in bare_results:
                content = fetch_bare_act_content(r.get("url", ""))
                relevant = ""
                if content and _is_readable_text(content):
                    relevant = extract_relevant_bare_act_portions(
                        facts_summary, r.get("title", ""), content
                    )
                text = relevant if _is_readable_text(relevant) else ""
                if not text and content and _is_readable_text(content):
                    text = content[:1500]
                if not text:
                    text = r.get("snippet", "")
                if text and _is_readable_text(text):
                    bare_sections.append({
                        "source": r.get("title", "Internet"),
                        "text": text,
                        "act_name": r.get("title", "Unknown"),
                        "url": r.get("url", ""),
                    })

        if not case_laws_local:
            web_results = search_internet_case_laws(legal_query, max_results=top_k)
            for r in web_results:
                # Try fetching from the source URL (indiankanoon, scr.sci.gov.in, etc.)
                content = fetch_case_content(r.get("url", ""))
                # If source URL gave no content and we have a sci_pdf, try the PDF
                if not content and r.get("sci_pdf"):
                    content = fetch_case_content(r.get("sci_pdf"))
                relevant_portion = ""
                if content and _is_readable_text(content) and not _looks_like_navigation(content):
                    relevant_portion = extract_relevant_case_portions(
                        facts_summary, r.get("title", ""), content
                    )
                text = relevant_portion if _is_readable_text(relevant_portion) else ""
                # Never use raw scraped content if it's navigation/chrome — use snippet only
                if not text and content and _is_readable_text(content) and not _looks_like_navigation(content):
                    text = content[:1500]
                if not text:
                    text = r.get("snippet", "")
                if not text or not _is_readable_text(text):
                    # Still add the result with title and link; show short placeholder for text
                    text = "Summary of this judgment is available at the source link below."
                display_url = r.get("sci_pdf") or r.get("url", "")
                case_laws_local.append({
                    "source": r.get("title", "Internet"),
                    "text": text,
                    "url": display_url,
                })

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

    return {
        "needs_confirmation": False,
        "bare_act_sections": bare_formatted,
        "case_laws": case_formatted,
        "internet_case_laws": [],
        "explanation": explanation,
    }
