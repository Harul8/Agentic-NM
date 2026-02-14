from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM
from agents.Legal_Research.bare_act_agent import retrieve_bare_act_section
from agents.Legal_Research.case_law_agent import retrieve_case_law


def _web_fallback_bare_acts(issue: str, max_results: int = 5) -> list:
    """Search the web for relevant bare acts when local vector store has none."""
    from services.response_generator import (
        search_internet_bare_acts,
        fetch_bare_act_content,
        extract_relevant_bare_act_portions,
    )
    results = []
    try:
        web_results = search_internet_bare_acts(issue, max_results=max_results)
        for r in web_results:
            content = fetch_bare_act_content(r.get("url", ""))
            title = r.get("title", "Unknown")
            relevant = ""
            if content:
                relevant = extract_relevant_bare_act_portions(issue, title, content)
            text = relevant or (content[:1500] if content else (r.get("snippet", ""))[:800])
            if not text:
                continue
            results.append({
                "source": title,
                "text": text,
                "act_name": title,
                "url": r.get("url", ""),
            })
    except Exception:
        pass
    return results


def _web_fallback_case_laws(issue: str, max_results: int = 5) -> list:
    """Search the web for relevant case laws when local vector store has none."""
    from services.response_generator import (
        search_internet_case_laws,
        fetch_case_content,
        extract_relevant_case_portions,
    )
    results = []
    try:
        web_results = search_internet_case_laws(issue, max_results=max_results)
        for r in web_results:
            content = fetch_case_content(r.get("url", ""))
            title = r.get("title", "Unknown")
            relevant = ""
            if content:
                relevant = extract_relevant_case_portions(issue, title, content)
            text = relevant or (content[:1500] if content else (r.get("snippet", ""))[:800])
            if not text:
                continue
            results.append({
                "source": title,
                "text": text,
                "url": r.get("url", ""),
            })
    except Exception:
        pass
    return results


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
