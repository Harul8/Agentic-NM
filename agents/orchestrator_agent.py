"""
Orchestrator Agent — CrewAI agent that understands the user query and returns a routing decision.

Output shape matches fact_collector.get_next_question_or_complete so it can be used as a drop-in:
- {"action": "ask", "question": "..."} or
- {"action": "complete", "intent": "search"|"lookup"|"legal_opinion", "facts_summary": "...", "message": "...", "result_count": int}

The agent output is raw text (JSON + optional reasoning). Caller should parse with
fact_collector._parse_llm_response(response, user_message) for full normalization.
"""

import logging
from typing import Optional

from crewai import Agent, Crew, Task
from llm.config import CREWAI_LLM
from prompts.advocate_prompts import FACT_COLLECTION_SYSTEM

logger = logging.getLogger(__name__)

ORCHESTRATOR_ROLE = "Query Router for Legal Research"
ORCHESTRATOR_GOAL = (
    "Analyze the user message and conversation to decide: (1) Is this a greeting, a direct search/lookup request, or a legal opinion request? "
    "(2) If search/lookup with a topic — output action=complete immediately with intent=search or lookup. "
    "(3) If legal opinion — either ask one clarifying question (action=ask) or complete with facts_summary if you have enough. "
    "Output exactly one JSON object per the system instructions; the client sees only your reply_to_client."
)
ORCHESTRATOR_BACKSTORY = (
    "You are the front-line router for an Indian legal research app. You never do research yourself; "
    "you only classify the request and either ask one short question or hand off to research with a clear intent and facts_summary. "
    "You follow the advocate prompts strictly: search/lookup with a topic must be complete immediately, no follow-up questions."
)


def _build_task_context(conversation_history: list, user_message: str) -> str:
    """Build the context string for the orchestrator task."""
    lines = []
    for m in conversation_history:
        role = "Client" if m.get("role") == "user" else "Lawyer"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"Client: {user_message}")
    return "\n".join(lines)


def run_orchestrator(conversation_history: list, user_message: str) -> Optional[str]:
    """
    Run the CrewAI orchestrator agent. Returns the raw agent output text (contains JSON)
    so the caller can parse it with fact_collector._parse_llm_response for normalization.

    Returns None if CrewAI fails (import error, runtime error, etc.) so caller can fall back to direct LLM.
    """
    try:
        context = _build_task_context(conversation_history, user_message)
        agent = Agent(
            role=ORCHESTRATOR_ROLE,
            goal=ORCHESTRATOR_GOAL,
            backstory=ORCHESTRATOR_BACKSTORY,
            llm=CREWAI_LLM,
            verbose=False,
        )
        task = Task(
            description=(
                "Using the system instructions below, analyze the conversation and latest user message. "
                "Output your reasoning on the first line if needed, then on the next line output exactly one valid JSON object. "
                "Use 'reply_to_client' in the JSON for the message the user sees; use 'question' only when action is 'ask'. "
                "System instructions:\n\n" + FACT_COLLECTION_SYSTEM
                + "\n\nConversation so far:\n" + context
                + "\n\nNow output REASONING: (optional) then your JSON with reply_to_client."
            ),
            expected_output="One valid JSON object with action, and reply_to_client or question, plus intent/facts_summary/result_count when action is complete.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        if result is None:
            return None
        # crew.kickoff() may return a CrewOutput or the final task output as string
        output = getattr(result, "raw", None) or getattr(result, "output", None) or str(result)
        if output and isinstance(output, str) and output.strip():
            return output.strip()
        return None
    except Exception as e:
        logger.warning("Orchestrator agent failed, falling back to direct LLM: %s", e, exc_info=False)
        return None
