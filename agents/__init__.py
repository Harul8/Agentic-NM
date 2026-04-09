"""
agents/ — Nyaymalaw agentic layer.

The agents package replaces the hardcoded pipeline/chat.py two-phase loop
with an LLM-driven orchestrator that holds the full tool set and decides
dynamically what to do: intake, research, or both — in parallel when useful.

Primary entry points:

    from agents.orchestrator import OrchestratorAgent

    agent = OrchestratorAgent()

    # Blocking call — returns final text
    result = agent.run(message, conversation, workflow_state)

    # Streaming call — yields SSE-compatible event dicts
    for event in agent.run_stream(message, conversation, workflow_state):
        ...
"""
