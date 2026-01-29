from crewai import Agent
from crewai.tools import tool
import json

@tool
def frame_issues(normalized_facts: str):
    """Identify and list the core legal issues based on normalized case facts."""
    try:
        data = json.loads(normalized_facts)
    except:
        return json.dumps({"issues": ["Invalid facts"]}, indent=2)

    summary = data.get("case_summary", "").lower()
    issues = []

    if "tenant" in summary:
        issues.append("Landlord–tenant dispute")
    if "contract" in summary:
        issues.append("Contract validity or breach")

    return json.dumps({"issues": issues or ["General legal dispute"]}, indent=2)

issue_framing_agent = Agent(
    role="Issue Framing Agent",
    goal="Extract the key legal issues from case facts.",
    backstory="You help lawyers understand the main disputes in a case.",
    llm="ollama/mistral:7b",
    tools=[frame_issues],
    verbose=True
)
