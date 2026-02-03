from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM

@tool
def detect_hallucinations(answer_text: str):
    """Check whether the generated legal answer contains unsupported or fabricated content."""
    
    suspicious_terms = ["may imply", "probably", "appears to", "might be"]
    
    for term in suspicious_terms:
        if term.lower() in answer_text.lower():
            return "⚠️ Potential hallucination detected. Needs human review."
    
    return "Answer appears grounded and factual."


hallucination_agent = Agent(
    role="Hallucination Detector",
    goal="Detect and flag hallucinated or unsupported legal reasoning.",
    backstory="You ensure the LLM only returns verifiable legal information.",
    llm=CREWAI_LLM,
    tools=[detect_hallucinations],
    verbose=True
)
