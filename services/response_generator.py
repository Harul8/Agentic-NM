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
    # Reject known generic pages
    clean = url.rstrip("/") + "/"
    if clean in _SCI_GENERIC_PAGES or url.rstrip("/") + "/" in _SCI_GENERIC_PAGES:
        return False
    # sci.gov.in judgment PDFs: api.sci.gov.in/supremecourt/...
    if "api.sci.gov.in/supremecourt/" in url:
        return True
    # scr.sci.gov.in with a specific case ID
    if "scr.sci.gov.in" in url and len(url) > 30:
        return True
    # indiankanoon.org doc pages (contain /doc/ or /docfragment/)
    if "indiankanoon.org" in url and ("/doc/" in url or "/docfragment/" in url):
        return True
    # Generic sci.gov.in pages without specific case content
    if "sci.gov.in" in url and "/doc/" not in url and "/supremecourt/" not in url:
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


def search_internet_case_laws(query: str, max_results: int = 5) -> list:
    """Search internet for Indian case law judgments.
    For Supreme Court queries: find judgments via indiankanoon, then link to sci.gov.in PDFs."""
    is_sc = _is_supreme_court_query(query)
    print(f"[CASE_SEARCH] query='{query[:80]}', is_sc={is_sc}, max={max_results}")

    if is_sc:
        queries = [
            f"{query} Supreme Court site:indiankanoon.org",
            f"{query} Supreme Court of India judgment site:indiankanoon.org",
        ]
    else:
        queries = [
            f"{query} High Court India case law judgment",
            f"{query} site:indiankanoon.org",
            f"{query} India case law",
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
                    print(f"[CASE_SEARCH] Searching: {q[:80]}...")
                    for r in ddgs.text(q, max_results=max_results * 2):
                        url = r.get("href", "")
                        if not url or url in seen_urls:
                            continue
                        if not _is_judgment_url(url):
                            print(f"[CASE_SEARCH] Rejected URL: {url[:80]}")
                            continue
                        seen_urls.add(url)
                        title = r.get("title", "Unknown")
                        print(f"[CASE_SEARCH] Found: {title[:60]} -> {url[:60]}")

                        results.append({
                            "title": title,
                            "url": url,
                            "sci_pdf": "",  # Will try to find PDF later, don't block search
                            "snippet": (r.get("body") or "")[:400],
                        })
                        if len(results) >= max_results:
                            break
                except Exception as e:
                    print(f"[CASE_SEARCH] Query failed: {e}")
                    continue

        # After collecting results, try to find SC PDF links (best-effort, don't fail on error)
        if is_sc:
            for r in results:
                try:
                    sci_pdf = _find_sci_pdf_url(r["title"])
                    if sci_pdf:
                        r["sci_pdf"] = sci_pdf
                        print(f"[CASE_SEARCH] PDF found: {sci_pdf[:60]}")
                except Exception:
                    pass  # PDF link is optional, don't fail

        print(f"[CASE_SEARCH] Total results: {len(results)}")
        return results[:max_results]
    except Exception as e:
        print(f"[CASE_SEARCH] FATAL: {e}")
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


def _clean_scraped_text(text: str) -> str:
    """Remove common website navigation noise from scraped text."""
    import re
    # Noise substrings to cut out (works on single-line text too)
    noise_phrases = [
        "Skip to main content",
        "Indian Kanoon - Search engine for Indian Law",
        "Search Indian laws and judgments",
        "Main Navigation",
        "Free features",
        "Premium Premium features",
        "Premium features",
        "Prism AI",
        "Pricing Login",
        "Mobile Navigation",
        "Legal Document View",
        "Tools for analyzing structure and cite text of judgments",
        "Tools for analyzing structure and cite",
        "Document Options",
        "Get in PDF",
        "Print it!",
        "Download Court Copy",
        "Search Results Page",
        "Filter Results by Document Types",
        "All Laws Judgments Tribunals",
        "Accessibility Links",
        "Accessibility Tools",
        "Color Contrast High Contrast Normal Contrast",
        "Highlight Links Invert Saturation",
        "Text Size Font Size Increase Font Size Decrease Normal Font",
        "Text Spacing Line Height",
        "Others Hide Images Big Cursor",
    ]
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    # Regex: "Cites N , Cited by N" patterns
    text = re.sub(r"\[?Cites?\s*\d+\s*,?\s*Cited\s*by\s*\d+[^\]]*\]?", "", text, flags=re.IGNORECASE)
    # Regex: repeated "Search Search" patterns
    text = re.sub(r"(?:Search\s+){2,}", "", text)
    # Regex: "Login" appearing alone
    text = re.sub(r"\bLogin\b\s*", "", text)
    # Clean up leftover multiple spaces
    text = re.sub(r"  +", " ", text)
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

        # HTML: extract judgment text specifically
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove all non-content elements first
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "input", "select"]):
            tag.decompose()

        # For indiankanoon: try specific judgment containers
        judgment_text = ""
        # Method 1: div with id containing 'judgment'
        for div_id in ["judgments", "judgment", "judgment_content"]:
            div = soup.find("div", {"id": div_id})
            if div:
                judgment_text = div.get_text(separator="\n", strip=True)
                break
        # Method 2: look for <pre> blocks (indiankanoon uses these for judgment text)
        if not judgment_text:
            pre_blocks = soup.find_all("pre")
            if pre_blocks:
                judgment_text = "\n".join(p.get_text(separator="\n", strip=True) for p in pre_blocks)
        # Method 3: look for the main content area
        if not judgment_text:
            for cls in ["doc_content", "result_content", "main-content"]:
                div = soup.find("div", class_=cls)
                if div:
                    judgment_text = div.get_text(separator="\n", strip=True)
                    break
        # Fallback: full page text
        if not judgment_text:
            judgment_text = soup.get_text(separator="\n", strip=True)

        judgment_text = _clean_scraped_text(judgment_text)
        return " ".join(judgment_text.split())[:6000]
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
                f"You are a friendly legal research assistant. The user's topic is: \"{facts[:500]}\"\n"
                "Unfortunately no results were found. Write a short, warm response (do NOT repeat the user's words) "
                "suggesting how to refine their search. Be conversational and helpful."
            ).strip()
        except Exception:
            return "I searched through relevant legal databases but couldn't find matching results for this topic. You could try using more specific legal terms, mentioning particular acts, or specifying the court."

    # Build a brief digest of the retrieved content for the LLM to summarise
    # Clean the text snippets before sending to LLM to avoid navigation noise
    material_digest = ""
    for i, c in enumerate(bare_sections[:5]):
        raw = _clean_scraped_text((c.get('text') or '')[:500])
        material_digest += f"Bare Act {i+1}: {c.get('title', c.get('act_name', c.get('source', '')))} — {raw[:400]}\n"
    for i, c in enumerate(case_laws[:5]):
        raw = _clean_scraped_text((c.get('text') or '')[:500])
        material_digest += f"Case Law {i+1}: {c.get('title', c.get('source', ''))} — {raw[:400]}\n"

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

    # Fallback: generate a basic greeting + summary if LLM fails
    # NEVER repeat the user's query back to them
    parts = []
    if num_cases > 0:
        parts.append(f"{num_cases} Supreme Court judgment{'s' if num_cases > 1 else ''}")
    if num_bare > 0:
        parts.append(f"{num_bare} relevant bare act provision{'s' if num_bare > 1 else ''}")
    materials_str = " and ".join(parts)
    return (
        f"Hello! I've searched through relevant legal databases and found {materials_str} on this topic. "
        f"Here's what the law and the courts have to say:"
    )



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


def _extract_case_name_from_text(text: str) -> str:
    """Extract 'Party A v/s Party B' style case name from judgment text."""
    import re
    # Match patterns: "X vs Y", "X v. Y", "X v/s Y", "X Vs. Y" etc.
    # Search a wider window (first 800 chars) since case name may appear after noise
    search_text = text[:800]
    vs_match = re.search(
        r"([A-Z][A-Za-z\s.,&'()]+?)\s+(?:vs\.?|v\.?|v/s\.?|Vs\.?|VS\.?|V/S\.?)\s+([A-Z][A-Za-z\s.,&'()]+?)(?:\s+on\s+\d|\s*$|\s*\n|\s*\()",
        search_text
    )
    if vs_match:
        p1 = vs_match.group(1).strip().rstrip(",. ")
        p2 = vs_match.group(2).strip().rstrip(",. ")
        if len(p1) > 2 and len(p2) > 2:
            return f"{p1} v/s {p2}"
    # Simpler fallback: just find "X vs Y" or "X v/s Y"
    vs_simple = re.search(
        r"([A-Z][A-Za-z.\s&]+?)\s+(?:vs\.?|v\.?|v/s)\s+([A-Z][A-Za-z.\s&]+)",
        search_text, re.IGNORECASE
    )
    if vs_simple:
        p1 = vs_simple.group(1).strip().rstrip(",. ")
        p2 = vs_simple.group(2).strip().rstrip(",. ")
        if len(p1) > 2 and len(p2) > 2:
            return f"{p1} v/s {p2}"
    return ""


def _is_proper_case_name(name: str) -> bool:
    """Check if a string looks like a proper 'X vs Y' case name."""
    import re
    return bool(re.search(r"(?:vs\.?|v\.?|v/s)", name, re.IGNORECASE))


def _generate_title(existing_name: str, text: str, doc_type: str) -> str:
    """Generate a short, meaningful title from existing name or text content.
    Avoids LLM calls to prevent timeouts — uses smart text extraction instead."""
    import re

    # === CASE LAWS ===
    if doc_type == "case law / judgment":
        # First: check if existing_name already has a proper "X vs Y" pattern
        if existing_name and _is_proper_case_name(existing_name):
            # Clean trailing date like " on 12 April, 2023"
            cleaned = re.sub(r"\s+on\s+\d+\s+\w+,?\s*\d{4}.*$", "", existing_name).strip()
            return cleaned if len(cleaned) > 5 else existing_name

        # Second: try to extract case name from the text content
        case_name = _extract_case_name_from_text(text)
        if case_name:
            return case_name

        # Third: try to extract from existing_name if it has parties but weird format
        # e.g. "National Highway Authority Of India vs Resham Singh And Ors ..."
        if existing_name:
            case_from_name = _extract_case_name_from_text(existing_name)
            if case_from_name:
                return case_from_name

        # Fallback: use existing_name cleaned up, or first meaningful line
        if existing_name and len(existing_name) > 8:
            # Remove common suffixes like "- Supreme Court of India", "- Indian Kanoon"
            cleaned = re.sub(r"\s*[-–—]\s*(Supreme Court|Indian Kanoon|High Court).*$", "", existing_name, flags=re.IGNORECASE).strip()
            if len(cleaned) > 5:
                return cleaned

    # === BARE ACTS ===
    if doc_type == "bare act provision":
        # Look for patterns like "THE SOMETHING ACT, YEAR" or "Something Act 1972"
        act_match = re.search(
            r"(?:THE\s+)?([A-Z][A-Za-z\s]+(?:ACT|SANHITA|CODE|RULES|REGULATION)[,\s]*\d{4})",
            text[:500], re.IGNORECASE
        )
        if act_match:
            act_name = act_match.group(0).strip().rstrip(",")
            sec_match = re.search(r"[Ss]ection\s+(\d+[A-Za-z]?)", text[:500])
            if sec_match:
                return f"{act_name.title()} — Section {sec_match.group(1)}"
            return act_name.title()
        # Try existing name
        if existing_name and len(existing_name) > 8:
            words = [w for w in existing_name.split() if not re.fullmatch(r"\d+", w)]
            if len(words) >= 2:
                return existing_name

    # Final fallback
    if existing_name and len(existing_name) > 3:
        return existing_name
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 15 and not any(w in line.lower() for w in ["skip", "search", "login", "navigation"]):
            return line[:80]
    return "Untitled"


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
    print(f"[GENERATE] Local: {len(bare_sections)} bare acts, {len(case_laws_local)} case laws")

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
    print(f"[GENERATE] Need web search? bare={not bare_sections}, case={not case_laws_local}")
    if not bare_sections or not case_laws_local:
        if not bare_sections:
            bare_results = search_internet_bare_acts(legal_query, max_results=top_k)
            for r in bare_results:
                content = fetch_bare_act_content(r.get("url", ""))
                # Clean content of navigation noise
                if content:
                    content = _clean_scraped_text(content)
                relevant = ""
                if content and _is_readable_text(content):
                    relevant = extract_relevant_bare_act_portions(
                        facts_summary, r.get("title", ""), content
                    )
                text = relevant if _is_readable_text(relevant) else ""
                if not text and content and _is_readable_text(content):
                    text = content[:1500]
                if not text:
                    text = _clean_scraped_text(r.get("snippet", ""))
                if text and _is_readable_text(text):
                    bare_sections.append({
                        "source": r.get("title", "Internet"),
                        "text": text,
                        "act_name": r.get("title", "Unknown"),
                        "url": r.get("url", ""),
                    })

        if not case_laws_local:
            print(f"[GENERATE] Starting web case law search...")
            web_results = search_internet_case_laws(legal_query, max_results=top_k)
            print(f"[GENERATE] Web search returned {len(web_results)} case law results")
            for idx, r in enumerate(web_results):
                try:
                    print(f"[GENERATE] Processing case {idx+1}: {r.get('title', '?')[:60]}")
                    # Try fetching from the source URL (indiankanoon, scr.sci.gov.in, etc.)
                    content = fetch_case_content(r.get("url", ""))
                    print(f"[GENERATE]   Content from URL: {len(content)} chars")
                    # If source URL gave no content and we have a sci_pdf, try the PDF
                    if not content and r.get("sci_pdf"):
                        content = fetch_case_content(r.get("sci_pdf"))
                        print(f"[GENERATE]   Content from PDF: {len(content)} chars")
                    # Always clean the content of navigation noise
                    if content:
                        content = _clean_scraped_text(content)
                    relevant_portion = ""
                    if content and _is_readable_text(content):
                        relevant_portion = extract_relevant_case_portions(
                            facts_summary, r.get("title", ""), content
                        )
                        print(f"[GENERATE]   LLM summary: {len(relevant_portion)} chars")
                    text = relevant_portion if _is_readable_text(relevant_portion) else ""
                    if not text and content and _is_readable_text(content):
                        text = content[:1500]
                    if not text:
                        text = _clean_scraped_text(r.get("snippet", ""))
                    if text and _is_readable_text(text):
                        display_url = r.get("sci_pdf") or r.get("url", "")
                        case_laws_local.append({
                            "source": r.get("title", "Internet"),
                            "text": text,
                            "url": display_url,
                        })
                        print(f"[GENERATE]   ADDED case law #{len(case_laws_local)}")
                    else:
                        print(f"[GENERATE]   SKIPPED - no readable text")
                except Exception as e:
                    print(f"[GENERATE]   ERROR processing case: {e}")
            print(f"[GENERATE] Final case laws: {len(case_laws_local)}")

    # Generate explanation: conversational for search/lookup, formal for legal_opinion
    if intent in ("search", "lookup"):
        explanation = generate_conversational_summary(
            facts_summary, bare_sections, case_laws_local
        )
    else:
        explanation = generate_relevance_explanation(
            facts_summary, bare_sections, case_laws_local, []
        )

    bare_formatted = []
    for c in bare_sections:
        text = _clean_scraped_text((c.get("text") or "").strip())
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
        text = _clean_scraped_text((c.get("text") or "").strip())
        if not text:
            continue
        source = c.get("source") or _clean_source_name(c.get("source", "")) or ""
        title = _generate_title(source, text, "case law / judgment")
        case_formatted.append({
            "source": source,
            "text": text,
            "title": title,
            "url": c.get("url", ""),
        })

    return {
        "needs_confirmation": False,
        "bare_act_sections": bare_formatted,
        "case_laws": case_formatted,
        "internet_case_laws": [],
        "explanation": explanation,
    }
