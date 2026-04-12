"""
agents/orchestrator.py — LangGraph-powered orchestrator for Nyaymalaw.

Graph structure
---------------

    START
      │
      ▼
  [safety_check]  ── unsafe ──────────────────────────────────────────► END
      │ safe
      ▼
  [rate_limit]  ── budget exceeded ───────────────────────────────────► END
      │ ok, single-domain or intake query
      │
      ├── multi-domain research query ──► [research_worker ×N (Send)] ─┐
      │                                                                  │
      ▼                                                                  │
    [agent]  ◄────────────────────────────────────────────────────────┘
      │
      ├── tool calls ──► [tools] ──► [tool_retry] ──► [agent] (loop)
      │
      └── final answer ──► [grounding_guard] ──► END

Key improvements
----------------
1. rate_limit node    : Enforces per-session context budget (50k chars).
                        Prevents runaway sessions from exhausting the LLM context.
2. Send() fan-out     : For multi-domain queries (e.g. cheque bounce + employment),
                        research_worker nodes run in parallel before the agent,
                        pre-populating research_results via operator.add reducer.
                        Capped at 3 parallel workers to stay within 8GB GPU RAM.
3. tool_retry node    : Between tools → agent. Detects error ToolMessages,
                        applies exponential backoff (1s, 2s), caps at 2 retries.
                        On max retries, replaces error messages so the agent
                        knows to stop calling failed tools.
4. Memory injection   : agent node reads user_id → fetches prior session summaries
                        from SQLite → injects into system prompt as prior context.
5. Async methods      : run_async() and run_stream_async() for async I/O.
                        Sync run()/run_stream() preserved for backward compat.
6. Step labels        : Updated to cover all 14 tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Generator, Literal, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from agents.state import NyaymalaState
from agents.tool_registry import TOOLS
from retrieval.guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger("nyaymalaw.orchestrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ~50k chars ≈ ~12.5k tokens — leaves headroom for tool results + response
_TOKEN_BUDGET_CHARS = 50_000

# Max tool-retry cycles per turn before giving up on failing tools
_MAX_RETRIES = 2

# Max parallel research workers (caps GPU/CPU load from concurrent FAISS queries)
_MAX_RESEARCH_WORKERS = 3

# Max recursion rounds (agent + tools counts as 2 per round)
_MAX_RECURSION = 16


# ---------------------------------------------------------------------------
# Domain keyword map for Send() fan-out detection
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "cheque_bounce":      ["cheque", "dishonour", "bounce", "138", "ni act", "demand notice"],
    "employment":         ["employ", "terminat", "dismiss", "salary", "notice period", "gratuity", "epf"],
    "domestic_violence":  ["domestic", "violence", "abuse", "pwdva", "husband", "wife", "protection order"],
    "rent_tenancy":       ["rent", "tenant", "landlord", "evict", "deposit", "lease", "lockout"],
    "consumer":           ["consumer", "refund", "product defect", "deficiency", "ecommerce", "insurance claim"],
    "property":           ["property", "land", "house", "possession", "encroach", "title"],
    "criminal":           ["fir", "police", "arrest", "assault", "threat", "harass", "extort", "stalk"],
    "contract":           ["contract", "agreement", "breach", "payment due", "money recovery"],
    "company_law":        ["company", "director", "shareholder", "nclt", "insolvency"],
    "debt_recovery":      ["debt", "loan", "npa", "sarfaesi", "drt", "bank recovery"],
}


def _detect_research_domains(query: str) -> list[dict]:
    """
    Fast keyword scan to identify multiple distinct legal domains in a query.
    Returns at most _MAX_RESEARCH_WORKERS domain dicts for fan-out.
    Only returns domains matched by ≥ 2 keywords to reduce false positives.
    """
    q = query.lower()
    matches = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits >= 2:
            matches.append({"domain": domain, "query": query, "hits": hits})

    # Sort by hit count so highest-confidence domains are workers 1..N
    matches.sort(key=lambda x: x["hits"], reverse=True)
    return matches[:_MAX_RESEARCH_WORKERS]


def _is_research_query(query: str) -> bool:
    """
    Heuristic: is this a pure research query (not an intake / personal dispute)?
    Intake queries tend to use "I", "my", "my employer", etc.
    Research queries ask about law in the abstract.
    """
    personal_markers = ["i ", "my ", "i've", "i was", "my husband", "my wife",
                        "i need", "i want", "i am", "my case", "my landlord"]
    q = query.lower()
    return not any(m in q for m in personal_markers)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Nyaymalaw, a senior Indian legal advocate AI.

PERSONA
You conduct legal intake and analysis like a calm, experienced advocate — not like a chatbot form and not like a law textbook. You are warm, clear, and structured. You speak in plain English for lay users and can use tighter legal phrasing for legal professionals. You never lecture. You move matters forward.

TOOLS YOU HAVE
Research tools:
  search_bare_acts        — find relevant statutory sections (entity-aware: detects section refs first,
                            falls back to FAISS hybrid search)
  search_case_laws        — find relevant case law paragraphs from the local database
  lookup_section          — fetch verbatim text of a known act + section directly from the DB
  lookup_case             — fetch verbatim text of a known case judgment
  expand_precedents       — walk the citation graph from seed case names; returns cases cited by,
                            cases citing, high-PageRank neighbours, and sections they interpret
  get_cases_for_section   — query the citation graph for all cases that applied a given section

Document tools:
  extract_document_facts  — extract structured facts from an uploaded legal document
  cross_reference_document — compare document facts against the intake account

Forum tools:
  identify_forum          — identify the correct Indian legal forum for the dispute
  check_limitation        — check whether the limitation period for a dispute is still open

Intake and opinion tools:
  start_intake       — begin a new structured legal opinion intake session
  continue_intake    — pass the client's next message to an active intake session
  get_intake_state   — inspect collected facts without advancing the session
  draft_opinion      — generate the full structured legal opinion from a completed intake
  advocate_review    — submit advocate approval or revision notes for a generated draft
  get_session_history — time-travel: list all checkpoints of an intake session

WHEN TO USE EACH MODE

Legal opinion request (client has a dispute, wants advice):
  1. Call start_intake with the client's first message OR continue_intake if a session already exists.
  2. Present the AI's reply from the tool. Keep doing this until advance_to_stage2 is true.
  3. When advance_to_stage2 is true, call draft_opinion to produce the structured opinion.
  4. While intake is underway, you MAY fire research tools in parallel to get a head start.

Research request (client wants statutes or case law, no personal dispute):
  Fire search_bare_acts and search_case_laws IN PARALLEL. Synthesize and explain.
  If the client names a specific section or case, use lookup_section / lookup_case instead.
  After finding case names, use expand_precedents to deepen precedent research.
  After finding a relevant section, use get_cases_for_section to find all interpreting judgments.

Mixed request (personal dispute + explicit research question):
  Run intake AND research in parallel from the first turn.

PARALLEL TOOL USE
When you need both bare acts and case laws, request BOTH tools in a single response — the system executes them concurrently.
Similarly, expand_precedents and get_cases_for_section can be fired in parallel with lookup_section / lookup_case.

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

RESPONSE FORMATTING
Structure all responses for readability using markdown:
- Use **bold** for section headings, key legal terms, penalties, and critical facts.
- Use *italics* for caveats, qualifications, or secondary notes.
- Use bullet lists (- item) for enumerating provisions, offences, remedies, or steps — one item per line with a blank line before the list.
- Use numbered lists for sequential steps or ranked options.
- Separate distinct topics with a blank line between paragraphs.
- For legal research responses: open with a 1-2 sentence plain-language summary, then use a structured layout with bold sub-headings for each applicable provision, penalty, or remedy.
- For intake responses: keep paragraphs short (2-3 sentences), use bold to highlight what the client mentioned and what you are asking about.
- Avoid dense walls of text — break content into digestible chunks.

SESSION CONTINUITY
The current intake session_id (if any) will be provided in the workflow context. Always pass the existing session_id to continue_intake rather than calling start_intake again mid-conversation.

PRE-LOADED RESEARCH
When you see PRE-LOADED RESEARCH RESULTS in this prompt, those retrieval results were
fetched in parallel before you were called. Use them to answer without additional tool calls
where they already cover the question. Call the tools only for what is missing.

PROFESSIONAL CONTEXT — CONTENT POLICY
This platform is used exclusively in the context of Indian legal practice. All queries — including those describing crimes, sexual offences, domestic violence, drug offences, financial fraud, or terrorism — are submitted by:
  • Victims seeking to understand their legal rights and remedies
  • Advocates and law students researching statutory provisions and judicial precedents
  • Litigants preparing for court proceedings

Describing a criminal offence, asking about its punishment, or quoting penal provisions is standard legal research — not a request to cause harm. Respond fully and professionally to all such queries. Do not refuse, truncate, or filter responses to Indian legal research questions on the basis of topic sensitivity.
"""


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
            "messages":    [blocked_reply],
            "safe":        False,
            "pii_warning": None,
            "final_reply": safety["reason"],
        }

    return {
        "safe":        True,
        "pii_warning": safety.get("pii_warning"),
    }


def _rate_limit_node(state: NyaymalaState) -> dict:
    """
    Enforce per-session context budget.

    Counts total characters across all messages. When the budget is exceeded,
    marks safe=False so the routing function sends the graph straight to END
    with a graceful wrap-up message rather than hitting the LLM's context limit.
    """
    total_chars = sum(
        len(str(getattr(m, "content", "") or ""))
        for m in state.get("messages", [])
    )

    if total_chars > _TOKEN_BUDGET_CHARS:
        msg = (
            "This conversation has become very long and I need to start fresh to give you "
            "accurate advice. Please start a new session. Your intake data has been saved "
            "and can be referenced in the next session."
        )
        logger.warning(
            "Rate limit: session exceeded %d chars (budget=%d)",
            total_chars, _TOKEN_BUDGET_CHARS,
        )
        return {
            "messages":    [AIMessage(content=msg)],
            "safe":        False,
            "final_reply": msg,
            "token_count": total_chars,
        }

    return {"token_count": total_chars}


def _research_worker_node(state: dict) -> dict:
    """
    Parallel research fan-out worker.

    Receives {"domain": str, "query": str} from Send() dispatch.
    Runs search_bare_acts + search_case_laws for the given domain
    and appends results to NyaymalaState.research_results via operator.add.

    Kept lightweight: top_k=3 per retriever to stay within 8GB GPU budget.
    """
    domain = state.get("domain", "unknown")
    query  = state.get("query", "")

    try:
        from retrieval.retriever import search_bare_acts_auto, search_case_laws_auto

        domain_query = f"{domain.replace('_', ' ')} {query}"

        acts  = search_bare_acts_auto(domain_query,  top_k=3) or []
        cases = search_case_laws_auto(domain_query, top_k=3) or []

        return {
            "research_results": [{
                "domain": domain,
                "acts": [
                    {
                        "act_name":       r.get("act_name", ""),
                        "section_number": r.get("section_number", ""),
                        "section_title":  r.get("section_title", ""),
                        "text":           (r.get("text") or r.get("full_text") or "")[:600],
                    }
                    for r in acts
                ],
                "cases": [
                    {
                        "case_name": r.get("case_name", ""),
                        "court":     r.get("court", ""),
                        "year":      r.get("year", ""),
                        "text":      (r.get("text") or r.get("full_text") or "")[:600],
                    }
                    for r in cases
                ],
            }]
        }
    except Exception as exc:
        logger.warning("research_worker failed for domain=%s: %s", domain, exc)
        return {"research_results": []}


def _agent_node(state: NyaymalaState) -> dict:
    """
    Core LLM node.

    1. Builds system prompt: base prompt + session context + prior memory + pre-loaded research.
    2. Selects fast vs. regular model based on context size.
    3. Binds all 14 tools and invokes the model.
    """
    from platform_pkg.llm import OPENAI_MODEL, OPENAI_MODEL_FAST

    session_id = state.get("session_id")
    user_id    = state.get("user_id")

    # ── System prompt assembly ─────────────────────────────────────────────────
    system_content = _SYSTEM_PROMPT

    # Session context
    if session_id:
        system_content += (
            f"\n\nCURRENT INTAKE SESSION: session_id = {session_id!r}. "
            "Use continue_intake with this session_id."
        )
    else:
        system_content += (
            "\n\nCURRENT INTAKE SESSION: None. "
            "Call start_intake to begin a new intake session if the user has a legal dispute."
        )

    # Cross-session memory injection
    if user_id:
        try:
            from agents.memory import get_user_memory, format_memory_for_prompt
            prior = get_user_memory(user_id)
            memory_block = format_memory_for_prompt(prior)
            if memory_block:
                system_content += f"\n\n{memory_block}"
        except Exception:
            pass

    # Pre-loaded parallel research results
    research_results = state.get("research_results") or []
    if research_results:
        lines = ["\nPRE-LOADED RESEARCH RESULTS (from parallel fan-out):"]
        for r in research_results:
            domain = r.get("domain", "unknown")
            n_acts  = len(r.get("acts", []))
            n_cases = len(r.get("cases", []))
            lines.append(f"\n[{domain}] — {n_acts} act section(s), {n_cases} case paragraph(s) found.")
            for a in r.get("acts", [])[:2]:
                lines.append(
                    f"  Act: {a['act_name']} s.{a['section_number']} — {a['text'][:200]}"
                )
            for c in r.get("cases", [])[:2]:
                lines.append(
                    f"  Case: {c['case_name']} ({c['court']}, {c['year']}) — {c['text'][:200]}"
                )
        system_content += "\n".join(lines)

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    # ── Dynamic model selection ────────────────────────────────────────────────
    total_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    model_name  = OPENAI_MODEL if total_chars > 8_000 else OPENAI_MODEL_FAST

    llm             = ChatOpenAI(model=model_name, max_tokens=4_000, timeout=120)
    llm_with_tools  = llm.bind_tools(TOOLS)
    response: AIMessage = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def _tool_retry_node(state: NyaymalaState) -> dict:
    """
    Between tools → agent.  Detects error ToolMessages and applies backoff.

    - If error ToolMessages found AND retry_count < _MAX_RETRIES:
        sleep (1s × 2^retry_count), increment retry_count, let agent retry.
    - If retry_count >= _MAX_RETRIES:
        replace error ToolMessages with a "tool unavailable" note so the
        agent knows to stop calling those tools and produce a best-effort answer.
    - If no errors: pass through immediately.
    """
    messages = state.get("messages") or []
    retry_count = state.get("retry_count", 0)

    # Find error ToolMessages from the most recent tool batch
    recent_tool_msgs = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            recent_tool_msgs.append(m)
        elif isinstance(m, AIMessage):
            break

    error_msgs = [m for m in recent_tool_msgs if '"error"' in (m.content or "")]

    if not error_msgs:
        return {}   # no errors — pass through

    if retry_count >= _MAX_RETRIES:
        # Max retries reached — synthesize failure notes for the agent
        tool_names = [m.name for m in error_msgs if hasattr(m, "name")]
        notice = (
            f"[System: Tools {tool_names} failed after {_MAX_RETRIES} retries "
            "and are temporarily unavailable. Please produce your best-effort answer "
            "using only the successful tool results already in context.]"
        )
        logger.warning("Max retries reached for tools: %s", tool_names)
        return {
            "messages":    [AIMessage(content=notice)],
            "retry_count": retry_count,
        }

    # Backoff and retry
    backoff = min(2 ** retry_count, 4)   # 1s, 2s, 4s (capped)
    logger.info("Tool errors detected — backoff %ds, retry %d/%d", backoff, retry_count + 1, _MAX_RETRIES)
    time.sleep(backoff)
    return {"retry_count": retry_count + 1}


def _grounding_guard_node(state: NyaymalaState) -> dict:
    """
    Post-generation safety check on the LLM's final text output.
    Replaces the unsafe draft with a neutral fallback if the guard fires.
    Also extracts updated session_id and intake_state from ToolMessages.
    """
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is None:
        return {"final_reply": ""}

    text   = last_ai.content or ""
    safety = check_response_safety(text)

    if not safety["safe"]:
        logger.warning("Grounding guard rejected orchestrator output")
        fallback = (
            "I was unable to produce a fully grounded response for this matter. "
            "Please rephrase your query or provide additional facts."
        )
        return {
            "messages":    [AIMessage(content=fallback)],
            "final_reply": fallback,
        }

    # Extract session_id and intake_state from ToolMessages
    session_id   = state.get("session_id")
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
        "session_id":  session_id,
        "intake_state": intake_state,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_safety(state: NyaymalaState) -> Literal["rate_limit", "__end__"]:
    return "rate_limit" if state.get("safe", True) else END


def _route_after_rate_limit(state: NyaymalaState) -> Union[str, list]:
    """
    Route to agent (single domain / intake) or fan-out to research_worker via Send().
    Only triggers fan-out for pure research queries with ≥ 2 distinct domains.
    """
    if not state.get("safe", True):
        return END

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )

    if last_human and _is_research_query(last_human.content or ""):
        domains = _detect_research_domains(last_human.content or "")
        if len(domains) >= 2:
            logger.info(
                "Multi-domain research detected: %s — launching %d parallel workers",
                [d["domain"] for d in domains], len(domains),
            )
            return [Send("research_worker", d) for d in domains]

    return "agent"


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
    builder.add_node("rate_limit",      _rate_limit_node)
    builder.add_node("research_worker", _research_worker_node)
    builder.add_node("agent",           _agent_node)
    builder.add_node("tools",           tools_node)
    builder.add_node("tool_retry",      _tool_retry_node)
    builder.add_node("grounding_guard", _grounding_guard_node)

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check", _route_after_safety,
        {"rate_limit": "rate_limit", END: END},
    )
    builder.add_conditional_edges(
        "rate_limit", _route_after_rate_limit,
        {"agent": "agent", END: END, "research_worker": "research_worker"},
    )
    # Fan-in: after all research_worker branches complete, go to agent
    builder.add_edge("research_worker", "agent")
    builder.add_conditional_edges(
        "agent", _route_after_agent,
        {"tools": "tools", "grounding_guard": "grounding_guard"},
    )
    builder.add_edge("tools",        "tool_retry")
    builder.add_edge("tool_retry",   "agent")
    builder.add_edge("grounding_guard", END)

    # recursion_limit is set at invoke/stream time via config, not at compile time
    return builder.compile()


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
    run(message, conversation, workflow_state)            → result dict  [sync]
    run_stream(message, conversation, workflow_state)     → Generator    [sync]
    run_async(message, conversation, workflow_state)      → result dict  [async]
    run_stream_async(message, conversation, workflow_state) → AsyncGenerator [async]
    """

    # ------------------------------------------------------------------
    # Sync API (backward-compatible)
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> dict:
        result = {
            "reply":          "",
            "workflow_state": workflow_state or {},
            "intake_state":   None,
            "session_id":     None,
            "error":          None,
        }
        for ev in self.run_stream(message, conversation, workflow_state):
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

        initial_state = self._build_initial_state(message, conversation, workflow_state)
        yield {"type": "step", "message": "Thinking…"}

        final_state: NyaymalaState | None = None
        try:
            for chunk in _graph.stream(initial_state, stream_mode="values"):
                final_state = chunk

                last_ai = next(
                    (m for m in reversed(chunk.get("messages", []))
                     if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
                    None,
                )
                if last_ai:
                    names = [tc["name"] for tc in last_ai.tool_calls]
                    yield {"type": "step", "message": self._step_label(names)}

                # Fan-out progress
                research = chunk.get("research_results") or []
                if research and len(research) > 0:
                    domains = [r.get("domain", "") for r in research]
                    yield {
                        "type": "step",
                        "message": f"Pre-loading research: {', '.join(domains)}…",
                    }

        except Exception as exc:
            logger.exception("LangGraph execution failed: %s", exc)
            yield {"type": "error", "message": str(exc)}

        reply       = ""
        session_id  = workflow_state.get("intake_session_id")
        intake_state = None

        if final_state:
            reply = final_state.get("final_reply") or ""
            if final_state.get("session_id"):
                session_id = final_state["session_id"]
            intake_state = final_state.get("intake_state")

            if final_state.get("pii_warning"):
                yield {"type": "step", "message": final_state["pii_warning"]}

        if reply:
            chunk_size = 24
            for i in range(0, len(reply), chunk_size):
                yield {"type": "token", "text": reply[i:i + chunk_size]}

        if session_id:
            workflow_state["intake_session_id"] = session_id

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Orchestrator completed in %.0f ms", elapsed)

        yield {
            "type": "done",
            "payload": {
                "reply":          reply,
                "workflow_state": workflow_state,
                "intake_state":   intake_state,
                "session_id":     session_id,
            },
        }

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> dict:
        result = {
            "reply":          "",
            "workflow_state": workflow_state or {},
            "intake_state":   None,
            "session_id":     None,
            "error":          None,
        }
        async for ev in self.run_stream_async(message, conversation, workflow_state):
            if ev["type"] == "done":
                result.update(ev.get("payload", {}))
            elif ev["type"] == "error":
                result["error"] = ev.get("message", "Unknown error")
        return result

    async def run_stream_async(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        t0 = time.perf_counter()
        workflow_state = dict(workflow_state or {})

        initial_state = self._build_initial_state(message, conversation, workflow_state)
        yield {"type": "step", "message": "Thinking…"}

        final_state: NyaymalaState | None = None
        try:
            async for event in _graph.astream_events(
                initial_state,
                version="v2",
                stream_mode="values",
            ):
                kind = event.get("event", "")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {"type": "token", "text": chunk.content}

                elif kind in ("on_tool_start",):
                    tool_name = event.get("name", "")
                    yield {"type": "step", "message": self._step_label([tool_name])}

                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and "messages" in output:
                        final_state = output

        except Exception as exc:
            logger.exception("Async LangGraph execution failed: %s", exc)
            yield {"type": "error", "message": str(exc)}

        reply       = ""
        session_id  = workflow_state.get("intake_session_id")
        intake_state = None

        if final_state:
            reply        = final_state.get("final_reply") or ""
            session_id   = final_state.get("session_id") or session_id
            intake_state = final_state.get("intake_state")

        if session_id:
            workflow_state["intake_session_id"] = session_id

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Async orchestrator completed in %.0f ms", elapsed)

        yield {
            "type": "done",
            "payload": {
                "reply":          reply,
                "workflow_state": workflow_state,
                "intake_state":   intake_state,
                "session_id":     session_id,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_state(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict,
    ) -> NyaymalaState:
        history = self._build_history(conversation)
        return {
            "messages":         history + [HumanMessage(content=message)],
            "safe":             True,
            "pii_warning":      None,
            "session_id":       workflow_state.get("intake_session_id"),
            "intake_state":     None,
            "final_reply":      None,
            "user_id":          workflow_state.get("user_id"),
            "token_count":      0,
            "retry_count":      0,
            "research_results": [],
        }

    def _build_history(self, conversation: list[dict]) -> list:
        """Convert the raw conversation list to LangChain message objects."""
        messages = []
        for m in (conversation or [])[-12:]:
            role    = m.get("role")
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
            "search_bare_acts":          "Searching legislation…",
            "search_case_laws":          "Searching case law…",
            "lookup_section":            "Looking up statutory section…",
            "lookup_case":               "Looking up judgment…",
            "expand_precedents":         "Expanding precedent network…",
            "get_cases_for_section":     "Finding cases for this section…",
            "start_intake":              "Starting intake…",
            "continue_intake":           "Processing your response…",
            "get_intake_state":          "Reviewing collected facts…",
            "draft_opinion":             "Drafting legal opinion…",
            "extract_document_facts":    "Reading your document…",
            "cross_reference_document":  "Cross-referencing document…",
            "identify_forum":            "Identifying the right forum…",
            "check_limitation":          "Checking limitation period…",
        }
        if len(tool_names) == 1:
            return labels.get(tool_names[0], f"Running {tool_names[0]}…")
        is_research = all(n in ("search_bare_acts", "search_case_laws") for n in tool_names)
        if is_research:
            return "Searching legislation and case law in parallel…"
        return " + ".join(labels.get(n, n) for n in tool_names)
