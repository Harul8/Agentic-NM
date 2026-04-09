"""
agents/orchestrator.py — LLM-driven orchestrator agent.

Replaces the hardcoded pipeline/chat.py two-phase loop.

The orchestrator holds the full 8-tool set and runs a tool-use loop:
  1. Send conversation + tools to the LLM
  2. If the model requests tools: execute them (in parallel when multiple)
  3. Append tool results, loop back
  4. When the model produces final text: apply grounding guard, return

Key design properties:
  - Parallel tool execution: research calls (bare acts + case laws) fire concurrently
  - Session continuity: intake session_id travels in workflow_state across turns
  - Safety: guard.py checks input before the loop, grounding guard checks draft output
  - Streaming: run_stream() yields event dicts for the SSE layer in api_server.py
  - Drop-in: same signature as pipeline/chat.py's process_chat()

Usage:

    from agents.orchestrator import OrchestratorAgent

    agent = OrchestratorAgent()

    # Blocking
    result = agent.run(message, conversation, workflow_state)
    # result = {
    #     "reply": str,
    #     "workflow_state": dict,   # updated state to persist
    #     "intake_state": dict,     # structured facts for frontend
    #     "session_id": str | None,
    # }

    # Streaming
    for event in agent.run_stream(message, conversation, workflow_state):
        # event = {"type": "step"|"token"|"done"|"error", ...}
        ...
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Optional

from agents.tool_registry import TOOL_DEFINITIONS, dispatch_tool
from retrieval.guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger("nyaymalaw.orchestrator")

# ---------------------------------------------------------------------------
# System prompt — the orchestrator's persona and tool-use strategy
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
  4. While intake is underway, you MAY fire research tools in parallel to get a head start — especially if the legal category is already clear from the first message.

Research request (client wants statutes or case law, no personal dispute):
  Fire search_bare_acts and search_case_laws IN PARALLEL. Synthesize and explain.
  If the client names a specific section or case, use lookup_section / lookup_case instead.

Mixed request (personal dispute + explicit research question):
  Run intake AND research in parallel from the first turn.

PARALLEL TOOL USE
When you need both bare acts and case laws, request BOTH tools in a single response — the system executes them concurrently. Do not wait for one to finish before calling the other.

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
All legal analysis must be grounded in tool results. Do not cite law you have not retrieved. If the tools return no relevant material, say so plainly rather than guessing. The final answer should be flowing prose: short fact framing, grounded legal analysis, practical next-step guidance — no rigid heading-heavy templates.

SESSION CONTINUITY
The current intake session_id (if any) will be provided in the workflow context below. Always pass the existing session_id to continue_intake rather than calling start_intake again mid-conversation.
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """
    LLM-driven orchestrator that replaces the hardcoded pipeline/chat.py loop.

    The agent runs an OpenAI tool-use loop, executing tools in parallel when
    the model requests multiple at once, until it produces a final text answer.
    """

    MAX_TOOL_ROUNDS = 6        # hard cap on tool-use rounds per request
    MAX_WORKERS = 4            # max parallel tool threads

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        conversation: list[dict],
        workflow_state: dict | None = None,
    ) -> dict:
        """
        Blocking call. Returns a result dict:
        {
            "reply": str,
            "workflow_state": dict,
            "intake_state": dict | None,
            "session_id": str | None,
            "error": str | None,
        }
        """
        result = {"reply": "", "workflow_state": workflow_state or {}, "intake_state": None, "session_id": None, "error": None}
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
        """
        Streaming call. Yields event dicts:
          {"type": "step",  "message": str}               — progress update
          {"type": "token", "text": str}                  — streamed text fragment
          {"type": "done",  "payload": {...}}              — final result
          {"type": "error", "message": str}               — error (non-fatal if intake reply available)
        """
        t0 = time.perf_counter()
        workflow_state = dict(workflow_state or {})
        session_id: str | None = workflow_state.get("intake_session_id")

        # ── 1. Safety check ───────────────────────────────────────────
        message = sanitize_input(message)
        safety = check_query_safety(message)
        if not safety["safe"]:
            yield {"type": "done", "payload": {
                "reply": safety["reason"],
                "workflow_state": workflow_state,
                "intake_state": None,
                "session_id": session_id,
            }}
            return

        if safety.get("pii_warning"):
            yield {"type": "step", "message": safety["pii_warning"]}

        # ── 2. Build the initial message list ─────────────────────────
        yield {"type": "step", "message": "Thinking…"}
        messages = self._build_messages(message, conversation, session_id)

        # ── 3. Tool-use loop ──────────────────────────────────────────
        final_text = ""
        updated_session_id = session_id
        updated_intake_state: dict | None = None
        round_count = 0

        try:
            client = self._get_client()

            while round_count < self.MAX_TOOL_ROUNDS:
                round_count += 1
                logger.debug("Orchestrator tool-use round %d", round_count)

                response = client.chat.completions.create(
                    model=self._pick_model(messages),
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_completion_tokens=4000,
                    timeout=120,
                )

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                msg = choice.message

                # Append assistant message to history
                messages.append(msg.model_dump() if hasattr(msg, "model_dump") else {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc
                                   for tc in (msg.tool_calls or [])],
                })

                if finish_reason == "stop" or not msg.tool_calls:
                    # Model produced its final answer
                    final_text = (msg.content or "").strip()
                    break

                # ── Execute all requested tools (in parallel) ─────────
                tool_calls = msg.tool_calls
                tool_names = [tc.function.name for tc in tool_calls]
                yield {"type": "step", "message": self._step_label(tool_names)}

                tool_results = self._execute_tools_parallel(tool_calls)

                # Collect updated session_id and intake_state from results
                for tool_name, result_json in tool_results.items():
                    try:
                        result_data = json.loads(result_json)
                        if "session_id" in result_data:
                            updated_session_id = result_data["session_id"]
                        if "intake_state" in result_data and result_data["intake_state"]:
                            updated_intake_state = result_data["intake_state"]
                    except Exception:
                        pass

                # Append tool result messages
                for tc in tool_calls:
                    tid = tc.id
                    result_json = tool_results.get(tc.function.name, json.dumps({"error": "no result"}))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": result_json,
                    })

            else:
                logger.warning("Orchestrator hit MAX_TOOL_ROUNDS (%d)", self.MAX_TOOL_ROUNDS)

        except Exception as exc:
            logger.exception("Orchestrator tool-use loop failed: %s", exc)
            yield {"type": "error", "message": str(exc)}
            # Fall through — if we have partial text, still try to return it

        # ── 4. Grounding guard on any draft output ────────────────────
        if final_text:
            safety_out = check_response_safety(final_text)
            if not safety_out["safe"]:
                logger.warning("Grounding guard rejected orchestrator output")
                final_text = (
                    "I was unable to produce a fully grounded response for this matter. "
                    "Please rephrase your query or provide additional facts."
                )

        # ── 5. Stream tokens + done ───────────────────────────────────
        if final_text:
            chunk_size = 24
            for i in range(0, len(final_text), chunk_size):
                yield {"type": "token", "text": final_text[i:i + chunk_size]}

        # Update workflow state
        if updated_session_id:
            workflow_state["intake_session_id"] = updated_session_id

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Orchestrator completed in %.0f ms (rounds=%d)", elapsed, round_count)

        yield {"type": "done", "payload": {
            "reply": final_text,
            "workflow_state": workflow_state,
            "intake_state": updated_intake_state,
            "session_id": updated_session_id,
        }}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        from platform.llm import _get_openai_client
        return _get_openai_client()

    def _pick_model(self, messages: list[dict]) -> str:
        """Choose fast or regular model based on total message length."""
        from platform.llm import OPENAI_MODEL_FAST, OPENAI_MODEL
        total_chars = sum(len(str(m.get("content") or "")) for m in messages)
        # Use regular model once the context grows (intake complete + research)
        return OPENAI_MODEL if total_chars > 8000 else OPENAI_MODEL_FAST

    def _build_messages(
        self,
        message: str,
        conversation: list[dict],
        session_id: str | None,
    ) -> list[dict]:
        """
        Build the messages list for the first LLM call.

        Includes:
          - system prompt (with session context injected)
          - recent conversation history (last 12 turns)
          - current user message
        """
        # Inject session context into system prompt
        system = _SYSTEM_PROMPT
        if session_id:
            system += f"\n\nCURRENT INTAKE SESSION: session_id = {session_id!r}. Use continue_intake with this session_id."
        else:
            system += "\n\nCURRENT INTAKE SESSION: None. Call start_intake to begin a new intake session if the user has a legal dispute."

        msgs: list[dict] = [{"role": "system", "content": system}]

        # Include recent history (last 12 turns, skip system messages)
        history = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in (conversation or [])[-12:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        msgs.extend(history)

        # Current message
        msgs.append({"role": "user", "content": message})
        return msgs

    def _execute_tools_parallel(self, tool_calls: list) -> dict[str, str]:
        """
        Execute all tool calls concurrently.
        Returns {tool_name: json_result_string}.
        """
        results: dict[str, str] = {}

        def _run_one(tc) -> tuple[str, str]:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = dispatch_tool(name, args)
            except Exception as exc:
                logger.exception("Tool %r failed: %s", name, exc)
                result = json.dumps({"error": str(exc), "tool": name})
            return name, result

        if len(tool_calls) == 1:
            name, result = _run_one(tool_calls[0])
            results[name] = result
        else:
            with ThreadPoolExecutor(max_workers=min(self.MAX_WORKERS, len(tool_calls))) as executor:
                futures = {executor.submit(_run_one, tc): tc for tc in tool_calls}
                for future in as_completed(futures):
                    try:
                        name, result = future.result(timeout=60)
                        results[name] = result
                    except Exception as exc:
                        tc = futures[future]
                        logger.error("Parallel tool %r raised: %s", tc.function.name, exc)
                        results[tc.function.name] = json.dumps({"error": str(exc)})

        return results

    def _step_label(self, tool_names: list[str]) -> str:
        """Generate a human-readable progress label for the set of tools being called."""
        labels = {
            "search_bare_acts":          "Searching legislation…",
            "search_case_laws":          "Searching case law…",
            "lookup_section":            "Looking up statutory section…",
            "lookup_case":               "Looking up judgment…",
            "start_intake":              "Starting intake…",
            "continue_intake":           "Processing your response…",
            "get_intake_state":          "Reviewing collected facts…",
            "draft_opinion":             "Drafting legal opinion…",
            "extract_document_facts":    "Reading your document…",
            "cross_reference_document":  "Cross-referencing document with your account…",
            "identify_forum":            "Identifying the right forum…",
            "check_limitation":          "Checking limitation period…",
        }
        if len(tool_names) == 1:
            return labels.get(tool_names[0], f"Running {tool_names[0]}…")

        # Multiple tools in parallel
        is_research = all(n in ("search_bare_acts", "search_case_laws") for n in tool_names)
        if is_research:
            return "Searching legislation and case law in parallel…"

        parts = [labels.get(n, n) for n in tool_names]
        return " + ".join(parts)
