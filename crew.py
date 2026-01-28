import sys
from crewai import Crew, Task

# 🧠 Fact Intelligence
from agents.Facts_Intelligence.facts_agent import facts_agent
from agents.Facts_Intelligence.normalization_agent import normalization_agent
from agents.Facts_Intelligence.contradiction_agent import contradiction_agent

# 📚 Legal Research
from agents.Legal_Research.bare_act_agent import bare_act_agent
from agents.Legal_Research.case_law_agent import case_law_agent
from agents.Legal_Research.applicability_agent import applicability_agent
from agents.Legal_Research.precedent_ranking_agent import precedent_ranking_agent

# ⚖️ Legal Reasoning
from agents.Legal_Reasoning.issue_framing_agent import issue_framing_agent
from agents.Legal_Reasoning.jurisdiction_agent import jurisdiction_agent
from agents.Legal_Reasoning.opinion_agent import opinion_agent
from agents.Legal_Reasoning.counter_argument_agent import counter_argument_agent

# 📝 Drafting
from agents.Drafting.drafting_agent import drafting_agent
from agents.Drafting.formatting_agent import formatting_agent

# 🛡️ Safety
from agents.Safety.citation_agent import citation_agent
from agents.Safety.hallucination_agent import hallucination_agent
from agents.Safety.confidence_agent import confidence_agent
from agents.Safety.gatekeeper_agent import gatekeeper_agent


# 🧩 ---------------- TASK DEFINITIONS ---------------- #

task_facts = Task(
    description="Use ONLY the provided raw_facts input. Do not invent or simulate any example facts. Convert that input into JSON.",
    expected_output="Structured case facts in JSON format",
    agent=facts_agent
)


task_normalize = Task(
    description="Normalize the structured facts into clean legal JSON.",
    expected_output="A clean normalized version of the case facts.",
    agent=normalization_agent
)

task_contradiction = Task(
    description="Check the normalized facts for internal contradictions.",
    expected_output="A contradiction analysis report.",
    agent=contradiction_agent
)

task_bare_act = Task(
    description="Retrieve relevant Bare Act sections.",
    expected_output="Relevant Bare Act legal provisions.",
    agent=bare_act_agent
)

task_case_law = Task(
    description="Retrieve similar legal case precedents.",
    expected_output="Relevant case law references.",
    agent=case_law_agent
)

task_applicability = Task(
    description="Check applicability of legal provisions.",
    expected_output="Applicable sections for the case.",
    agent=applicability_agent
)

task_precedent_rank = Task(
    description="Rank the case precedents by relevance.",
    expected_output="Ranked legal precedents.",
    agent=precedent_ranking_agent
)

task_issue_frame = Task(
    description="Frame core legal issues.",
    expected_output="List of framed legal issues.",
    agent=issue_framing_agent
)

task_jurisdiction = Task(
    description="Determine case jurisdiction.",
    expected_output="Appropriate court jurisdiction.",
    agent=jurisdiction_agent
)

task_opinion = Task(
    description="Provide legal opinion.",
    expected_output="A professional legal analysis.",
    agent=opinion_agent
)

task_counter_argument = Task(
    description="Generate counter-arguments.",
    expected_output="Possible counter-legal reasoning.",
    agent=counter_argument_agent
)

task_drafting = Task(
    description="Draft a legal petition.",
    expected_output="A structured court-ready petition.",
    agent=drafting_agent
)

task_formatting = Task(
    description="Format petition for court.",
    expected_output="Final formatted petition document.",
    agent=formatting_agent
)

task_citation = Task(
    description="Verify citations.",
    expected_output="Validated legal citations.",
    agent=citation_agent
)

task_hallucination = Task(
    description="Check hallucinations.",
    expected_output="Hallucination validation result.",
    agent=hallucination_agent
)

task_confidence = Task(
    description="Score answer confidence.",
    expected_output="Confidence level of legal output.",
    agent=confidence_agent
)

task_gatekeeper = Task(
    description="Decide human escalation.",
    expected_output="Decision on whether to escalate.",
    agent=gatekeeper_agent
)


# 🤖 ---------------- CREW EXECUTION ---------------- #

crew = Crew(
    agents=[
        facts_agent, normalization_agent, contradiction_agent,
        bare_act_agent, case_law_agent, applicability_agent, precedent_ranking_agent,
        issue_framing_agent, jurisdiction_agent, opinion_agent, counter_argument_agent,
        drafting_agent, formatting_agent,
        citation_agent, hallucination_agent, confidence_agent, gatekeeper_agent
    ],
    tasks=[
        task_facts, task_normalize, task_contradiction,
        task_bare_act, task_case_law, task_applicability, task_precedent_rank,
        task_issue_frame, task_jurisdiction, task_opinion, task_counter_argument,
        task_drafting, task_formatting,
        task_citation, task_hallucination, task_confidence, task_gatekeeper
    ],
    verbose=True
)


if __name__ == "__main__":
    user_input = sys.argv[1] if len(sys.argv) > 1 else "No facts provided"

    result = crew.kickoff(inputs={
        "raw_facts": user_input
    })
