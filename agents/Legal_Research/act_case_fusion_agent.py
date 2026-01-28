# agents/Legal_Research/act_case_fusion_agent.py

from crewai import Agent
from crewai.tools import tool

# import the TOOLS (not agents)
from agents.Legal_Research.bare_act_agent import retrieve_bare_act_section
from agents.Legal_Research.case_law_agent import retrieve_case_law


@tool
def fuse_bare_act_and_case_law(issue: str):
    """
    Combine relevant Bare Act sections and Case Laws for a given legal issue.
    Uses existing retrieval tools. Does not generate opinions.
    """

    bare_act_context = retrieve_bare_act_section(issue)
    case_law_context = retrieve_case_law(issue)

    return {
        "issue": issue,
        "bare_act_sections": bare_act_context,
        "case_laws": case_law_context
    }


act_case_fusion_agent = Agent(
    role="Legal Research Synthesizer",
    goal="Combine Bare Act provisions and Case Laws into a single legal context",
    backstory=(
        "You strictly aggregate authoritative legal material. "
        "You do not interpret, argue, or draft. "
        "You only consolidate statutory law and judicial precedent."
    ),
    llm="ollama/llama3.1:8b",
    tools=[fuse_bare_act_and_case_law],
    verbose=True
)
