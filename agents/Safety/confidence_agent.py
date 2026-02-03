from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM

@tool
def score_confidence(answer_text: str):
    """Assign a confidence score to the legal answer based on clarity and specificity."""
    
    score = 100
    
    if len(answer_text) < 50:
        score -= 30
    if "may" in answer_text.lower() or "might" in answer_text.lower():
        score -= 20
    if "section" not in answer_text.lower():
        score -= 20
    
    if score >= 80:
        return "High Confidence"
    elif score >= 50:
        return "Medium Confidence"
    else:
        return "Low Confidence – Needs Review"


confidence_agent = Agent(
    role="Confidence Evaluator",
    goal="Evaluate and assign a confidence level to the generated legal response.",
    backstory="You check if the legal answer is reliable enough to proceed.",
    llm=CREWAI_LLM,
    tools=[score_confidence],
    verbose=True
)
