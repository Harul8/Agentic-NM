from crewai import Agent
from crewai.tools import tool
import json

@tool
def draft_petition(opinion_json: str):
    """Draft a legal petition or pleading based on the generated legal opinion."""
    try:
        data = json.loads(opinion_json)
    except:
        return "Invalid legal opinion."

    opinion = data.get("opinion", "")

    draft = f"""LEGAL DRAFT DOCUMENT

Based on the legal analysis, the following petition is drafted:

{opinion}

This draft is prepared for further review and formatting.
"""

    return draft


drafting_agent = Agent(
    role="Drafting Agent",
    goal="Draft petitions, pleadings, and legal documents.",
    backstory="You convert legal opinions into court-ready drafts.",
    llm="ollama/mistral:7b",
    tools=[draft_petition],
    verbose=True
)
