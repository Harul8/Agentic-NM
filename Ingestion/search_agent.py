from crewai import Agent

search_agent = Agent(
    role="Case Law Search Agent",
    goal="Search and collect relevant case law sources for a given legal query",
    backstory="You are a legal researcher who finds authoritative judgments.",
    llm="ollama/llama3.1:8b",
    verbose=True
)
