from crewai import Agent
from crewai.tools import tool
import json

@tool
def determine_jurisdiction(normalized_facts: str):
    """Determine the likely court jurisdiction based on locations in the case facts."""
    try:
        data = json.loads(normalized_facts)
    except:
        return json.dumps({"jurisdiction": "Unknown"}, indent=2)

    locations = data.get("locations", [])

    if any("hyderabad" in loc.lower() for loc in locations):
        return json.dumps({"jurisdiction": "Hyderabad Civil Court"}, indent=2)
    elif any("telangana" in loc.lower() for loc in locations):
        return json.dumps({"jurisdiction": "Telangana State Courts"}, indent=2)

    return json.dumps({"jurisdiction": "General Civil Court"}, indent=2)

jurisdiction_agent = Agent(
    role="Jurisdiction Agent",
    goal="Identify the appropriate legal jurisdiction.",
    backstory="You determine where the case should be filed.",
    llm="ollama/llama3.1:8b",
    tools=[determine_jurisdiction],
    verbose=True
)
