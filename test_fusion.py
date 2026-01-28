from crewai import Crew, Task

from agents.Legal_Research.act_case_fusion_agent import act_case_fusion_agent

# Define task
fusion_task = Task(
    description="Find relevant Bare Act sections and Case Laws for specific performance of contract",
    expected_output="Combined statutory provisions and case laws",
    agent=act_case_fusion_agent
)

# Create crew
crew = Crew(
    agents=[act_case_fusion_agent],
    tasks=[fusion_task],
    verbose=True
)

# Run
result = crew.kickoff()
print("\n=== FUSION OUTPUT ===\n")
print(result)
