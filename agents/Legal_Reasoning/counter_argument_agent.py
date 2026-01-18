from crewai import Agent
from crewai.tools import tool
import json

@tool
def generate_counter_arguments(opinion_json: str):
    """Generate possible counter-arguments and weaknesses from the legal opinion."""
    try:
        data = json.loads(opinion_json)
    except:
        return json.dumps({"counter_arguments": ["Invalid opinion"]}, indent=2)

    opinion = data.get("opinion", "").lower()
    counters = []

    if "breach" in opinion:
        counters.append("Opposing party may argue there was no breach.")
    if "tenant" in opinion:
        counters.append("Opposing party may claim tenancy rights are protected.")

    return json.dumps({"counter_arguments": counters or ["No major counterpoints found"]}, indent=2)


counter_argument_agent = Agent(
    role="Counter-Argument Agent",
    goal="Highlight weaknesses and opposing viewpoints.",
    backstory="You help lawyers anticipate the opposing side.",
    llm="ollama/llama3.1:8b",
    tools=[generate_counter_arguments],
    verbose=True
)
