from crewai import Agent
from crewai.tools import tool
import json

@tool
def generate_opinion(research_data: str):
    """Generate a plain-language legal opinion based on research and facts."""
    try:
        data = json.loads(research_data)
    except:
        return json.dumps({"opinion": "Invalid research input"}, indent=2)

    sections = data.get("sections", [])
    issues = data.get("issues", [])

    opinion_text = "Based on the case facts and research:\n"

    if issues:
        opinion_text += f"- The dispute concerns: {', '.join(issues)}.\n"

    if sections:
        opinion_text += f"- Relevant legal provisions include: {', '.join(sections)}.\n"

    opinion_text += "This provides a foundational legal analysis."

    return json.dumps({"opinion": opinion_text}, indent=2)


opinion_agent = Agent(
    role="Legal Opinion Agent",
    goal="Provide a reasoned legal opinion from facts and research.",
    backstory="You synthesize Bare Act sections and legal issues into a professional legal opinion.",
    llm="ollama/llama3.1:8b",
    tools=[generate_opinion],
    verbose=True
)
