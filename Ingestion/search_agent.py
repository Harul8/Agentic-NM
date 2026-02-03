from crewai import Agent

from llm.config import CREWAI_LLM

search_agent = Agent(
    role="Case Law Search Agent",
    goal="Search and collect relevant case law sources for a given legal query",
    backstory="You are a legal researcher who finds authoritative judgments.",
    llm=CREWAI_LLM,
    verbose=True
)
