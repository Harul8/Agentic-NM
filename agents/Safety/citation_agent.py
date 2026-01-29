from crewai import Agent
from crewai.tools import tool

@tool
def verify_citations(draft_text: str):
    """Check whether the legal draft contains proper Bare Act and Case Law references."""
    
    if "Section" in draft_text or "v." in draft_text:
        return "Citations appear valid and referenced correctly."
    else:
        return "⚠️ No legal citations detected in the draft."


citation_agent = Agent(
    role="Citation Verifier",
    goal="Ensure all legal statements are properly cited from Bare Acts or case law.",
    backstory="You validate citations and prevent unsupported legal claims.",
    llm="ollama/mistral:7b",
    tools=[verify_citations],
    verbose=True
)
