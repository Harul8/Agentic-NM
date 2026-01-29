from crewai import Agent
from crewai.tools import tool
import json

@tool
def check_section_applicability(research_json: str):
    """Check whether the retrieved Bare Act section applies to the case facts."""
    try:
        data = json.loads(research_json)
    except:
        return json.dumps({"applicable": False, "reason": "Invalid research input"}, indent=2)

    facts = data.get("case_summary", "").lower()
    section = data.get("section_text", "").lower()

    if "tenant" in facts and "lease" in section:
        return json.dumps({"applicable": True, "reason": "Lease law applies"}, indent=2)

    return json.dumps({"applicable": False, "reason": "No clear match"}, indent=2)


applicability_agent = Agent(
    role="Section Applicability Agent",
    goal="Validate whether the retrieved Bare Act section applies to the case.",
    backstory="You match legal sections with the facts of the case.",
    llm="ollama/mistral:7b",
    tools=[check_section_applicability],
    verbose=True
)
