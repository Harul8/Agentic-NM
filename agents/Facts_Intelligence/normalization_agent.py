from crewai import Agent
from crewai.tools import tool
import json

from llm.config import CREWAI_LLM

@tool
def normalize_facts(structured_facts: str):
    """Normalize structured case facts into standard legal JSON schema."""
    try:
        data = json.loads(structured_facts)
    except:
        return json.dumps({"error": "Invalid facts format"}, indent=2)

    normalized = {
        "case_summary": data.get("summary", ""),
        "key_entities": data.get("entities", []),
        "important_dates": data.get("dates", []),
        "locations": data.get("locations", [])
    }
    return json.dumps(normalized, indent=2)

normalization_agent = Agent(
    role="Fact Normalizer",
    goal="Normalize and standardize structured case facts.",
    backstory="You clean structured facts into normalized legal JSON.",
    llm=CREWAI_LLM,
    tools=[normalize_facts],
    verbose=True
)
