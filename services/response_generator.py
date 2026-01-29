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
    """Convert plain-language facts to legal research query with acts, sections, terms."""
    prompt = f"""You are an Indian legal research expert. Convert these case facts into a concise legal research query.

FACTS:
{facts[:1500]}

Output ONLY a single search query (1-2 sentences) that includes:
- Relevant Bare Acts (e.g., Specific Relief Act, Contract Act, Transfer of Property Act)
- Legal terms and concepts
- Section numbers if mentioned
- Key legal issues

Example: "Specific performance of contract Section 10 Specific Relief Act 1963 breach of contract remedy"

Query:"""
    try:
        return ask_llm(prompt).strip()[:500] or facts[:300]
    except Exception:
        return facts[:300]


def retrieve_bare_acts(query: str, top_k: int = 10) -> list:
    """Retrieve relevant bare act sections from vector store."""
    if not os.path.exists(BARE_INDEX) or not os.path.exists(BARE_CHUNKS):
        return []

    index = faiss.read_index(BARE_INDEX)
    with open(BARE_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(np.array([query_vec], dtype="float32"), top_k)

    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        key = str(idx)
        if key in chunks:
            chunk = chunks[key]
            text = (chunk.get("text") or "").strip()
            if len(text) > 50 and "Not Acceptable" not in text:
                results.append(chunk)
    return results


def retrieve_case_laws(query: str, top_k: int = 10) -> list:
    """Retrieve relevant case laws from vector store."""
    if not os.path.exists(CASE_INDEX) or not os.path.exists(CASE_CHUNKS):
        return []

    index = faiss.read_index(CASE_INDEX)
    with open(CASE_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(np.array([query_vec], dtype="float32"), top_k)

    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        key = str(idx)
        if key in chunks:
            chunk = chunks[key]
            text = (chunk.get("text") or "").strip()
            if len(text) > 50 and "Not Acceptable" not in text:
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
    """Fetch and extract text from a bare act URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return " ".join(text.split())[:5000]
    except Exception:
        return ""


def extract_relevant_bare_act_portions(facts: str, title: str, content: str) -> str:
    """Use LLM to extract relevant bare act provisions from fetched content."""
    if not content or len(content) < 100:
        return content[:1500] if content else ""

    prompt = f"""Extract ONLY the relevant statutory provisions/sections from this legal document that apply to the case facts.
Include: section numbers, definitions, substantive provisions.
Exclude: preamble, footnotes, unrelated sections.
Keep 2-4 paragraphs max.

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


def search_internet_case_laws(query: str, max_results: int = 5) -> list:
    """Search internet for Indian Supreme Court / High Court judgments."""
    queries = [
        f"{query} Supreme Court of India judgment",
        f"{query} High Court India case law",
        f"{query} site:indiankanoon.org",
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


def fetch_case_content(url: str) -> str:
    """Fetch and extract text from a case law URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return " ".join(text.split())[:6000]
    except Exception:
        return ""


def extract_relevant_case_portions(facts: str, case_title: str, case_content: str) -> str:
    """Use LLM to extract only the RELEVANT portions of a case law for the user's facts."""
    if not case_content or len(case_content) < 100:
        return case_content[:1500] if case_content else ""

    prompt = f"""Extract ONLY the portions of this judgment that are RELEVANT to the case facts below.
Include: ratio decidendi, key holdings, applicable legal principles, relevant observations.
Exclude: procedural details, unrelated facts, boilerplate.
Keep 2-4 paragraphs max. Be precise.

CASE FACTS:
{facts[:800]}

JUDGMENT: {case_title}
---
{case_content[:4000]}
---

Relevant portions only:"""
    try:
        return ask_llm(prompt).strip()[:2500] or case_content[:1500]
    except Exception:
        return case_content[:1500]


def generate_relevance_explanation(
    facts: str,
    bare_sections: list,
    case_laws: list,
    internet_cases: list,
) -> str:
    """
    Generate structured explanation: for each bare act section and case law,
    explain WHY it is relevant to the user's facts.
    """
    if not bare_sections and not case_laws and not internet_cases:
        return "No relevant bare act provisions or case laws were found for your query. Try rephrasing with more specific legal terms or section references."

    bare_text = json.dumps([{"source": c.get("source"), "text": (c.get("text") or "")[:600]} for c in bare_sections], indent=2)[:2500]
    case_text = json.dumps([{"source": c.get("source"), "text": (c.get("text") or "")[:600]} for c in case_laws], indent=2)[:2500]
    net_text = json.dumps([{"title": c.get("title"), "relevant_portion": c.get("relevant_portion", c.get("content", ""))[:500]} for c in internet_cases], indent=2)[:2000]

    prompt = f"""You are an Indian legal research assistant. For each legal provision and case law below, explain in 2-3 sentences WHY it is relevant to the user's case facts. Be specific - connect the law to the facts.

USER'S CASE FACTS:
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

Provide a structured response:

## Relevant Bare Act Provisions
For each section: [Act/Source] - [Why this section applies to the user's situation]

## Relevant Case Laws  
For each case: [Case name/source] - [Why this precedent applies - what principle from the case helps the user]

Be concise. Use bullet points. Focus on practical relevance."""

    try:
        return ask_llm(prompt).strip()
    except Exception:
        return "Relevance analysis could not be generated. Please review the retrieved materials above."


def generate_summary_for_confirmation(bare_acts: list, case_laws: list) -> str:
    """Generate a brief summary of internet-sourced materials for user confirmation."""
    parts = []
    if bare_acts:
        parts.append(f"**Bare Act Sections** ({len(bare_acts)} found):")
        for b in bare_acts[:3]:
            parts.append(f"• {b.get('title', 'Unknown')}")
        if len(bare_acts) > 3:
            parts.append(f"  ... and {len(bare_acts) - 3} more")
    if case_laws:
        parts.append(f"\n**Case Laws** ({len(case_laws)} found):")
        for c in case_laws[:3]:
            parts.append(f"• {c.get('title', 'Unknown')}")
        if len(case_laws) > 3:
            parts.append(f"  ... and {len(case_laws) - 3} more")
    parts.append("\n\nIf these are relevant to your matter, please confirm to index them and include them in the full legal research response.")
    return "\n".join(parts)


def generate_response(facts_summary: str, confirmed_materials: dict = None) -> dict:
    """
    Full legal research response.
    When no local bare acts or case laws: search internet, return summary + materials_to_confirm.
    When user confirms: index materials, then include in full response.
    confirmed_materials: {bare_acts: [], case_laws: []} - from user confirmation
    """
    legal_query = expand_legal_query(facts_summary)
    search_query = f"{facts_summary} {legal_query}"[:500]

    bare_sections = list(retrieve_bare_acts(search_query, top_k=10))
    case_laws_local = list(retrieve_case_laws(search_query, top_k=10))

    # If user confirmed materials, add them and generate full response
    if confirmed_materials:
        for b in confirmed_materials.get("bare_acts", []):
            bare_sections.append({
                "source": b.get("title", "Internet"),
                "text": b.get("text", b.get("content", b.get("snippet", ""))),
                "act_name": b.get("title", b.get("act_name", "Bare Act")),
            })
        for c in confirmed_materials.get("case_laws", []):
            case_laws_local.append({
                "source": c.get("title", "Internet"),
                "text": c.get("relevant_portion", c.get("content", c.get("snippet", ""))),
            })
        # Fall through to generate full response below

    # If no local bare acts OR no local case laws (and no confirmed): search internet
    elif not bare_sections or not case_laws_local:
        internet_bare_acts = []
        internet_cases = []
        if not bare_sections:
            bare_results = search_internet_bare_acts(legal_query, max_results=5)
            for r in bare_results:
                content = fetch_bare_act_content(r["url"])
                relevant = ""
                if content:
                    relevant = extract_relevant_bare_act_portions(
                        facts_summary, r.get("title", ""), content
                    )
                internet_bare_acts.append({
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "content": content[:2000] if content else r.get("snippet", ""),
                    "text": relevant or (content[:1500] if content else r.get("snippet", "")),
                    "act_name": r.get("title", "Unknown"),
                    "source": "internet",
                })

        if not case_laws_local:
            web_results = search_internet_case_laws(legal_query, max_results=5)
            for r in web_results:
                content = fetch_case_content(r["url"])
                relevant_portion = ""
                if content:
                    relevant_portion = extract_relevant_case_portions(
                        facts_summary, r.get("title", ""), content
                    )
                internet_cases.append({
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "content": content[:2000] if content else r.get("snippet", ""),
                    "relevant_portion": relevant_portion or (content[:1500] if content else r.get("snippet", "")),
                    "source": "internet",
                })

        # If we found internet materials, return summary for confirmation
        if internet_bare_acts or internet_cases:
            summary = generate_summary_for_confirmation(internet_bare_acts, internet_cases)
            return {
                "needs_confirmation": True,
                "summary": summary,
                "materials_to_confirm": {
                    "bare_acts": internet_bare_acts,
                    "case_laws": internet_cases,
                },
                "bare_act_sections": [],
                "case_laws": [],
                "internet_case_laws": [],
                "explanation": summary,
            }

    # Generate full response with explanation (local materials + any confirmed materials)
    explanation = generate_relevance_explanation(
        facts_summary, bare_sections, case_laws_local, []
    )

    bare_formatted = []
    for c in bare_sections:
        src = c.get("source", "Unknown")
        text = (c.get("text") or "").strip()
        if text:
            bare_formatted.append({
                "source": src,
                "text": text,
                "act_name": (c.get("act_name") or src).replace(".pdf", "").replace("_", " ") if src else "Bare Act",
            })

    case_formatted = []
    for c in case_laws_local:
        case_formatted.append({
            "source": c.get("source", "Unknown"),
            "text": (c.get("text") or "").strip(),
        })

    # internet_formatted only when we have unconfirmed internet cases (legacy path)
    internet_formatted = []

    return {
        "needs_confirmation": False,
        "bare_act_sections": bare_formatted,
        "case_laws": case_formatted,
        "internet_case_laws": internet_formatted,
        "explanation": explanation,
    }
