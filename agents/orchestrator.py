"""
agents/orchestrator.py — LangGraph-powered orchestrator for Nyaymalaw.

Replaces the hand-rolled tool-use loop in the previous OrchestratorAgent class
with a proper LangGraph StateGraph.  The graph has four nodes:

    START
      │
      ▼
  [safety_check]  ── unsafe ──► END
      │ safe
      ▼
    [agent]  ◄──────────────────┐
      │                         │
      ├── tool calls ──► [tools]─┘
      │
      └── final answer ──► [grounding_guard] ──► END

Key properties preserved from the original implementation
---------------------------------------------------------
- Parallel tool execution   : LangGraph's ToolNode fires all requested tools
                              concurrently via its built-in thread pool.
- Dynamic model selection   : fast vs. regular model chosen per-turn based on
                              total context size.
- Safety checks             : input safety_check node + output grounding_guard node.
- Session continuity        : session_id travels in NyaymalaState; no manual
                              plumbing through workflow_state dicts.
- Streaming                 : run_stream() yields the same event dict schema as
                              before so api_server.py requires no changes.
- Drop-in compatibility     : OrchestratorAgent.run() / run_stream() signatures
                              are preserved exactly.

LangSmith tracing is automatic — every graph execution appears as a traced run
in the LangSmith UI when LANGCHAIN_TRACING_V2=true is set in .env.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.state import NyaymalaState
from agents.tool_registry import TOOLS
from retrieval.guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger("nyaymalaw.orchestrator")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Nyaymalaw, a senior Indian legal advocate AI.

PERSONA
You conduct legal intake and analysis like a calm, experienced advocate — not like a chatbot form and not like a law textbook. You are warm, clear, and structured. You speak in plain English for lay users and can use tighter legal phrasing for legal professionals. You never lecture. You move matters forward.

TOOLS YOU HAVE
Research tools:
  search_bare_acts   — find relevant statutory sections from the local database
  search_case_laws   — find relevant case law paragraphs from the local database
  lookup_section     — fetch verbatim text of a known act section
  lookup_case        — fetch verbatim text of a known case judgment

Intake and opinion tools:
  start_intake       — begin a new structured legal opinion intake session
  continue_intake    — pass the client's next message to an active intake session
  get_intake_state   — inspect collected facts without advancing the session
  draft_opinion      — generate the full structured legal opinion from a completed intake

WHEN TO USE EACH MODE

Legal opinion request (client has a dispute, wants advice):
  1. Call start_intake with the client's first message OR continue_intake if a session already exists.
  2. Present the AI's reply from the tool. Keep doing this until advance_to_stage2 is true.
  3. When advance_to_stage2 is true, call draft_opinion to produce the structured opinion.
  4. While intake is underway, you MAY fire research tools in parallel to get a head start.

Research request (client wants statutes or case law, no personal dispute):
  Fire search_bare_acts and search_case_laws IN PARALLEL. Synthesize and explain.
  If the client names a specific section or case, use lookup_section / lookup_case instead.

Mixed request (personal dispute + explicit research question):
  Run intake AND research in parallel from the first turn.

PARALLEL TOOL USE
When you need both bare acts and case laws, request BOTH tools in a single response — the system executes them concurrently.

INTAKE CONVERSATION PRINCIPLES
- Open with brief empathy when the facts call for it.
- Identify the legal issue in plain human language.
- Explain why the next details matter.
- Ask 2–3 closely related questions in one cluster when they belong to the same factual theme.
- Track urgency, prior actions, evidence posture, and relief sought.
- Never cite statutes or case law from memory during intake — use the tools.
- Never ask the client to repeat what they already said.
- Never jump to legal conclusions before intake is complete.

GROUNDED FINAL ANSWERS
All legal analysis must be grounded in tool results. Do not cite law you have not retrieved. If the tools return no relevant material, say so plainly rather than guessing.

SESSION CONTINUITY
The current intake session_id (if any) will be provided in the workflow context. Always pass the existing session_id to continue_intake rather than calling start_intake again mid-conversation.
"""

# Max recursion rounds (agent + tools counts as 2 per round → 6 rounds = 12 steps)
_MAX_RECURSION = 12


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _safety_check_node(state: NyaymalaState) -> dict:
    """Sanitize input and run safety + PII checks before touching the LLM."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        return {"safe": True, "pii_warning": None}

    text = sanitize_input(last_human.content or "")
    safety = check_query_safety(text)

    if not safety["safe"]:
        blocked_reply = AIMessage(content=safety["reason"])
        return {
            "messages": [blocked_reply],
            "safe": False,
            "pii_warning": None,
            "final_reply": safety["reason"],
        }

    return {
        "safe": True,
        "pii_warning": safety.get("pii_warning"),
    }


def _agent_node(state: NyaymalaState) -> dict:
    """
    Core LLM node.  Selects fast or regular model based on context size,
    binds all tools, and invokes the model.  Returns the AI message (which
    may contain tool_calls) for LangGraph to route onward.
    """
    from platform_pkg.llm import OPENAI_MODEL, OPENAI_MODEL_FAST

    # Build full message list: system prompt (with session context) + history
    session_id = state.get("session_id")
    system_content = _SYSTEM_PROMPT
    if session_id:
        system_content += f"\n\nCURRENT INTAKE SESSION: session_id = {session_id!r}. Use continue_intake with this session_id."
    else:
        system_content += "\n\nCURRENT INTAKE SESSION: None. Call start_intake to begin a new intake session if the user has a legal dispute."

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    # Dynamic model selection based on total context size
    total_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    model_name = OPENAI_MODEL if total_chars > 8000 else OPENAI_MODEL_FAST

    llm = ChatOpenAI(model=model_name, max_tokens=4000, timeout=120)
    llm_with_tools = llm.bind_tools(TOOLS)

    response: AIMessage = llm_with_tools.invoke(messages)

    # Extract updated session_id from any tool results already in state
    # (will be overwritten properly after tool execution in next cycle)
    return {"messages": [response]}


def _grounding_guard_node(state: NyaymalaState) -> dict:
    """
    Post-generation safety check on the LLM's final text output.
    Replaces the unsafe draft with a neutral fallback if the guard fires.
    """
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is None:
        return {"final_reply": ""}

    text = last_ai.content or ""
    safety = check_response_safety(text)

    if not safety["safe"]:
        logger.warning("Grounding guard rejected orchestrator output")
        fallback = (
            "I was unable to produce a fully grounded response for this matter. "
            "Please rephrase your query or provide additional facts."
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "final_reply": fallback,
        }

    # Extract session_id and intake_state from the most recent ToolMessages
    session_id = state.get("session_id")
    intake_state = state.get("intake_state")

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content or "{}")
            if "session_id" in data and data["session_id"]:
                session_id = data["session_id"]
            if "intake_state" in data and data["intake_state"]:
                intake_state = data["intake_state"]
        except Exception:
            pass

    return {
        "final_reply": text,
        "session_id": session_id,
        "intake_state": intake_state,
    }


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def _route_after_safety(state: NyaymalaState) -> Literal["agent", "__end__"]:
    return "agent" if state.get("safe", True) else END


def _route_after_agent(state: NyaymalaState) -> Literal["tools", "grounding_guard"]:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai and getattr(last_ai, "tool_calls", None):
        return "tools"
    return "grounding_guard"


# ---------------------------------------------------------------------------
# Build the graph (compiled once at module import)
# ---------------------------------------------------------------------------

def _build_graph():
    tools_node = ToolNode(TOOLS)

    builder = StateGraph(NyaymalaState)
    builder.add_node("safety_check",    _safety_check_node)
    builder.add_node("agent",           _agent_node)
    builder.add_node("tools",           tools_node)
    builder.add_node("grounding_guard", _grounding_guard_node)

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges("safety_check", _route_after_safety,
                                  {"agent": "agent", END: END})
    builder.add_conditional_edges("agent", _route_after_agent,
                                  {"tools": "tools", "grounding_guard": "grounding_guard"})
    builder.add_edge("tools", "agent")
    builder.add_edge("grounding_guard", END)

    return builder.compile(recursion_limit=_MAX_RECURSION)


_graph = _build_graph()


# ---------------------------------------------------------------------------
# OrchestratorAgent — drop-in replacement with identical public API
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """
    Thin wrapper around the compiled LangGraph so api_server.py and
    pipeline/chat.py need zero changes.

    Public methods
    --------------
    run(message, conversation, workflow_state)  → result dict
    run_stream(message, conversation, workflow_state)  → Generator[event dict]
    """

    def run(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> dict:
        result = {
            "reply": "",
            "workflow_state": workflow_state or {},
            "intake_state": None,
            "session_id": None,
            "error": None,
        }
        events = list(self.run_stream(message, conversation, workflow_state))
        for ev in events:
            if ev["type"] == "done":
                result.update(ev.get("payload", {}))
            elif ev["type"] == "error":
                result["error"] = ev.get("message", "Unknown error")
        return result

    def run_stream(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> Generator[dict, None, None]:
        t0 = time.perf_counter()
        workflow_state = dict(workflow_state or {})

        # Build initial LangGraph state
        history = self._build_history(conversation)
        initial_state: NyaymalaState = {
            "messages": history + [HumanMessage(content=message)],
            "safe": True,
            "pii_warning": None,
            "session_id": workflow_state.get("intake_session_id"),
            "intake_state": None,
            "final_reply": None,
        }

        yield {"type": "step", "message": "Thinking…"}

        final_state: NyaymalaState | None = None
        try:
            for chunk in _graph.stream(initial_state, stream_mode="values"):
                final_state = chunk

                # Emit progress steps from the latest AI message
                last_ai = next(
                    (m for m in reversed(chunk.get("messages", []))
                     if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
                    None,
                )
                if last_ai:
                    names = [tc["name"] for tc in last_ai.tool_calls]
                    yield {"type": "step", "message": self._step_label(names)}

        except Exception as exc:
            logger.exception("LangGraph execution failed: %s", exc)
            yield {"type": "error", "message": str(exc)}

        # Extract final reply and updated state
        reply = ""
        session_id = workflow_state.get("intake_session_id")
        intake_state = None

        if final_state:
            reply = final_state.get("final_reply") or ""
            if final_state.get("session_id"):
                session_id = final_state["session_id"]
            intake_state = final_state.get("intake_state")

            # PII warning as a step event
            if final_state.get("pii_warning"):
                yield {"type": "step", "message": final_state["pii_warning"]}

        # Stream reply tokens (true token-level streaming is available via
        # _graph.astream_events — kept as chunked for sync compatibility)
        if reply:
            chunk_size = 24
            for i in range(0, len(reply), chunk_size):
                yield {"type": "token", "text": reply[i:i + chunk_size]}

        # Update workflow_state with new session_id
        if session_id:
            workflow_state["intake_session_id"] = session_id

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Orchestrator completed in %.0f ms", elapsed)

        yield {
            "type": "done",
            "payload": {
                "reply": reply,
                "workflow_state": workflow_state,
                "intake_state": intake_state,
                "session_id": session_id,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_history(self, conversation: list[dict]) -> list:
        """Convert the raw conversation list to LangChain message objects."""
        messages = []
        for m in (conversation or [])[-12:]:
            role = m.get("role")
            content = m.get("content") or ""
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _step_label(self, tool_names: list[str]) -> str:
        labels = {
            "search_bare_acts":         "Searching legislation…",
            "search_case_laws":         "Searching case law…",
            "lookup_section":           "Looking up statutory section…",
            "lookup_case":              "Looking up judgment…",
            "start_intake":             "Starting intake…",
            "continue_intake":          "Processing your response…",
            "get_intake_state":         "Reviewing collected facts…",
            "draft_opinion":            "Drafting legal opinion…",
            "extract_document_facts":   "Reading your document…",
            "cross_reference_document": "Cross-referencing document…",
            "identify_forum":           "Identifying the right forum…",
            "check_limitation":         "Checking limitation period…",
        }
        if len(tool_names) == 1:
            return labels.get(tool_names[0], f"Running {tool_names[0]}…")
        is_research = all(n in ("search_bare_acts", "search_case_laws") for n in tool_names)
        if is_research:
            return "Searching legislation and case law in parallel…"
        return " + ".join(labels.get(n, n) for n in tool_names)
