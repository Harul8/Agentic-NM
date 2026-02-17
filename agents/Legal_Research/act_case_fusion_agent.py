from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM
from agents.Legal_Research.bare_act_agent import retrieve_bare_act_section
from agents.Legal_Research.case_law_agent import retrieve_case_law


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
    Uses local vector store first; if no results, searches the web for Indian bare acts and case laws.
    """

    result = {
        "issue": issue,
        "bare_act_sections": [],
        "case_laws": [],
    }

    # ---- Bare Act Retrieval (local, then web fallback) ----
    try:
        bare_acts = retrieve_bare_act_section.run(query=issue)
        if bare_acts:
            result["bare_act_sections"] = bare_acts
    except Exception:
        result["bare_act_sections"] = []

    if not result["bare_act_sections"]:
        result["bare_act_sections"] = _web_fallback_bare_acts(issue, max_results=5)

    # ---- Case Law Retrieval (local, then web fallback) ----
    try:
        case_laws = retrieve_case_law.run(query=issue)
        if case_laws:
            result["case_laws"] = case_laws
    except Exception:
        result["case_laws"] = []

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
