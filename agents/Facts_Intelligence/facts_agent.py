from crewai import Agent
from crewai.tools import tool
import json

from llm.config import CREWAI_LLM

@tool
def structure_facts(raw_facts: str):
    """
    Convert plain language case facts into structured JSON.
    """
    structured = {
        "summary": raw_facts.strip(),
        "entities": [],
        "dates": [],
        "locations": []
    }
    return json.dumps(structured, indent=2)

facts_agent = Agent(
    role="Facts Collector",
    goal="Collect, clean, and structure case facts provided by the user.",
    backstory=(
        "You are the first legal intake assistant. "
        "You take plain-language case facts and convert them into structured legal JSON "
        "so that research and drafting agents can use it."
    ),
    llm=CREWAI_LLM,
    tools=[structure_facts],
    verbose=True
)
