from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM
from agents.Legal_Research.bare_act_agent import retrieve_bare_act_section
from agents.Legal_Research.case_law_agent import retrieve_case_law


@tool
def fuse_bare_act_and_case_law(issue: str):
    """
    Combine relevant Bare Act sections and Case Laws for a given legal issue.
    This tool is deterministic and does not retry or hallucinate.
    """

    result = {
        "issue": issue,
        "bare_act_sections": [],
        "case_laws": []
    }

    # ---- Bare Act Retrieval ----
    try:
        bare_acts = retrieve_bare_act_section.run(
            query=issue
        )
        if bare_acts:
            result["bare_act_sections"] = bare_acts
    except Exception as e:
        result["bare_act_sections"] = []

    # ---- Case Law Retrieval ----
    try:
        case_laws = retrieve_case_law.run(
            query=issue
        )
        if case_laws:
            result["case_laws"] = case_laws
    except Exception as e:
        result["case_laws"] = []

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
