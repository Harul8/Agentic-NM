from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM

@tool
def human_gate(confidence_level: str):
    """Decide whether the legal answer should be sent to a human reviewer."""
    
    if "Low" in confidence_level:
        return "Escalate to Human Reviewer"
    elif "Medium" in confidence_level:
        return "Optional Human Review"
    else:
        return "Proceed Automatically"


gatekeeper_agent = Agent(
    role="Human Gatekeeper",
    goal="Ensure low-confidence legal answers are escalated for human review.",
    backstory="You act as the final safety layer before drafting or filing.",
    llm=CREWAI_LLM,
    tools=[human_gate],
    verbose=True
)
