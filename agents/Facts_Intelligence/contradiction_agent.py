from crewai import Agent
from crewai.tools import tool
import json

@tool
def detect_contradictions(normalized_facts: str):
    """Detect contradictions or logical inconsistencies in normalized legal case facts."""
    try:
        data = json.loads(normalized_facts)
    except:
        return json.dumps({"valid": False, "issues": ["Invalid normalized facts"]}, indent=2)

    summary = data.get("case_summary", "").lower()
    issues = []

    if "tenant" in summary and "owner" in summary and "same person" in summary:
        issues.append("Tenant and owner cannot be the same party.")
    if "not paid rent" in summary and "paid all dues" in summary:
        issues.append("Both unpaid and paid rent mentioned.")

    return json.dumps({
        "valid": len(issues) == 0,
        "issues": issues or ["No contradictions found"]
    }, indent=2)

contradiction_agent = Agent(
    role="Contradiction Checker",
    goal="Validate normalized case facts for contradictions.",
    backstory="You ensure only logically consistent case facts proceed further.",
    llm="ollama/llama3.1:8b",
    tools=[detect_contradictions],
    verbose=True
)
