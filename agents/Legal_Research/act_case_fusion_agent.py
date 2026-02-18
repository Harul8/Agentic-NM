"""
Legal Research Fusion — Combines bare act retrieval + case law retrieval.

Phase 3 rewrite: Uses v2 hybrid_retriever (FAISS + BM25 + cross-encoder)
instead of the old bare_act_agent/case_law_agent (v1 FAISS only).
Web fallback uses tiered_search + auto_enricher.

This module is used by:
- api_server.py /search endpoint
- mcp_server.py legal_research tool
"""

from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM


def _search_bare_acts(issue: str, top_k: int = 20) -> list:
    """Search bare acts using v2 hybrid retriever (FAISS + BM25 + cross-encoder)."""
    try:
        from retrieval.hybrid_retriever import search_bare_acts_auto
        results = search_bare_acts_auto(issue, top_k=top_k)
        return [
            {
                "source": r.get("source", ""),
                "text": r.get("text", ""),
                "act_name": r.get("act_name", ""),
                "section_number": r.get("section_number", ""),
                "rerank_score": r.get("rerank_score", 0),
            }
            for r in results if r.get("text")
        ]
    except Exception:
        return []


def _search_case_laws(issue: str, top_k: int = 20) -> list:
    """Search case laws using v2 hybrid retriever (FAISS + BM25 + cross-encoder)."""
    try:
        from retrieval.hybrid_retriever import search_case_laws_auto
        results = search_case_laws_auto(issue, top_k=top_k)
        return [
            {
                "source": r.get("source", ""),
                "text": r.get("text", ""),
                "case_name": r.get("case_name", ""),
                "court": r.get("court", ""),
                "year": r.get("year", ""),
                "rerank_score": r.get("rerank_score", 0),
            }
            for r in results if r.get("text")
        ]
    except Exception:
        return []


def _web_fallback_bare_acts(issue: str, max_results: int = 5) -> list:
    """Search web for bare acts via tiered search + auto-enricher (v2 pipeline)."""
    try:
        from retrieval.tiered_search import search_for_gaps
        from retrieval.auto_enricher import enrich_from_gap_results
        gaps = [{"aspect": "bare_act", "search_queries": [f"{issue} India bare act section"]}]
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrich_from_gap_results(gap_results, search_type="bare_act")
        results = []
        for gap in gap_results.get("results", []):
            for r in gap.get("search_results", []):
                results.append({
                    "source": r.get("title", "Internet"),
                    "text": r.get("snippet", ""),
                    "act_name": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                })
                if len(results) >= max_results:
                    break
        return results[:max_results]
    except Exception:
        return []


def _web_fallback_case_laws(issue: str, max_results: int = 5) -> list:
    """Search web for case laws via tiered search + auto-enricher (v2 pipeline)."""
    try:
        from retrieval.tiered_search import search_for_gaps
        from retrieval.auto_enricher import enrich_from_gap_results
        gaps = [{"aspect": "case_law", "search_queries": [f"{issue} Supreme Court India judgment"]}]
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrich_from_gap_results(gap_results, search_type="case_law")
        results = []
        for gap in gap_results.get("results", []):
            for r in gap.get("search_results", []):
                results.append({
                    "source": r.get("title", "Internet"),
                    "text": r.get("snippet", ""),
                    "url": r.get("url", ""),
                })
                if len(results) >= max_results:
                    break
        return results[:max_results]
    except Exception:
        return []


@tool
def fuse_bare_act_and_case_law(issue: str):
    """
    Combine relevant Bare Act sections and Case Laws for a given legal issue.
    Uses v2 hybrid retriever (FAISS + BM25 + cross-encoder) first;
    falls back to tiered web search if local results are insufficient.
    """
    result = {
        "issue": issue,
        "bare_act_sections": [],
        "case_laws": [],
    }

    # ---- Bare Act Retrieval (v2 hybrid, then web fallback) ----
    result["bare_act_sections"] = _search_bare_acts(issue)
    if not result["bare_act_sections"]:
        result["bare_act_sections"] = _web_fallback_bare_acts(issue, max_results=5)

    # ---- Case Law Retrieval (v2 hybrid, then web fallback) ----
    result["case_laws"] = _search_case_laws(issue)
    if not result["case_laws"]:
        result["case_laws"] = _web_fallback_case_laws(issue, max_results=5)

    return result


act_case_fusion_agent = Agent(
    role="Legal Research Synthesizer",
    goal="Combine Bare Act provisions and Case Laws into a single structured legal context",
    backstory=(
        "You strictly aggregate authoritative legal material. "
        "You never reason, interpret, retry, or invent. "
        "If data is missing, you return empty sections."
    ),
    llm=CREWAI_LLM,
    tools=[fuse_bare_act_and_case_law],
    verbose=True
)
