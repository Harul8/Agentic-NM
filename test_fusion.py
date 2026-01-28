from crewai import Crew, Task
from agents.Legal_Research.act_case_fusion_agent import act_case_fusion_agent

fusion_task = Task(
    description="Find relevant Bare Act sections and Case Laws for specific performance of contract",
    expected_output="Structured Bare Act sections and Case Laws",
    agent=act_case_fusion_agent
)

crew = Crew(
    agents=[act_case_fusion_agent],
    tasks=[fusion_task],
    verbose=True
)

result = crew.kickoff()

print("\n=== FUSION OUTPUT ===\n")
print(result)
