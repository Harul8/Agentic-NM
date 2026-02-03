from crewai import Agent
from crewai.tools import tool
import json

from llm.config import CREWAI_LLM

@tool
def rank_precedents(case_law_json: str):
    """Rank retrieved case law precedents as binding or persuasive."""
    try:
        data = json.loads(case_law_json)
    except:
        return json.dumps({"ranked_cases": []}, indent=2)

    cases = data.get("case_law", [])

    ranked = []
    for c in cases:
        if "supreme court" in c.lower():
            ranked.append({"case": c, "strength": "Binding"})
        else:
            ranked.append({"case": c, "strength": "Persuasive"})

    return json.dumps({"ranked_cases": ranked}, indent=2)


precedent_ranking_agent = Agent(
    role="Precedent Ranking Agent",
    goal="Classify and rank legal precedents by authority level.",
    backstory="You determine whether judgments are binding or persuasive.",
    llm=CREWAI_LLM,
    tools=[rank_precedents],
    verbose=True
)
