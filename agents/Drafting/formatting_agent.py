from crewai import Agent
from crewai.tools import tool

@tool
def format_for_court(draft_text: str):
    """Format the drafted legal document into a clean, professional court-ready structure."""
    
    formatted = f"""===============================
COURT DRAFT FORMAT
===============================

{draft_text}

===============================
END OF DOCUMENT
===============================
"""

    return formatted


formatting_agent = Agent(
    role="Formatting Agent",
    goal="Format drafted legal petitions and pleadings for final presentation.",
    backstory="You ensure legal drafts are clean, structured, and court-ready.",
    llm="ollama/mistral:7b",
    tools=[format_for_court],
    verbose=True
)
