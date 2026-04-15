"""
pipeline/chat.py — LEGACY orchestrator. Do not add new logic here.

This module is kept for backward compatibility with the existing frontend endpoints:
  /submit_case, /submit_case/stream
  /conversation/continue, /conversation/continue/stream
  /interview_step, /interview_step/stream

All new integrations must use:
  POST /agent/stream  →  agents.orchestrator.OrchestratorAgent (LangGraph)

Migration status
----------------
The OrchestratorAgent has been fully migrated to LangGraph (agents/orchestrator.py).
  - Tool definitions  : agents/tool_registry.py  (@tool decorators, auto-schemas)
  - Graph state       : agents/state.py           (NyaymalaState TypedDict)
  - Execution graph   : agents/orchestrator.py    (StateGraph: safety→agent→tools→guard)
  - Observability     : LangSmith auto-traces every /agent/stream call when
                        LANGCHAIN_TRACING_V2=true is set in .env

process_chat_agent() below is the thin bridge that lets any code in this module
delegate a single turn to the LangGraph orchestrator without touching api_server.py.

This file will be removed once the React frontend is fully migrated to /agent/stream.
"""
import logging
import os
import time
import hashlib

from platform_pkg.llm import ask_llm
from agents.intake.collector import get_next_question_or_complete, is_stop_signal
from agents.intake.stage1_opening import process_turn as _intake_process_turn
from platform_pkg.memory import guard_activity
from platform_pkg.warmup import kickoff_runtime_warmup, kickoff_ollama_warmup_if_qwen
from retrieval.generator import (
    generate_response_v2 as generate_response,
    generate_legal_opinion_from_cache,
)
from retrieval.guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph bridge — delegates a single turn to the new OrchestratorAgent.
# Use this instead of process_chat() for any new internal callers.
# ---------------------------------------------------------------------------

def process_chat_agent(
    message: str,
    conversation: list[dict],
    workflow_state: dict | None = None,
) -> dict:
    """
    Thin bridge to the LangGraph OrchestratorAgent.

    Returns the same result dict shape as process_chat() so callers can
    switch between the two without changing their own code:
        {
            "reply":          str,
            "workflow_state": dict,
            "intake_state":   dict | None,
            "session_id":     str | None,
            "error":          str | None,
        }
    """
    from agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent().run(message, conversation, workflow_state)

# Set PIPELINE_TIMING=1 in env to log elapsed ms for each step (debug slow follow-ups)
_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")

GENERIC_CHAT_SYSTEM = (
    "You are a concise, helpful assistant. "
    "Answer clearly and accurately. "
    "Do not provide legal advice in this mode; suggest switching to legal mode for legal analysis."
)


def _log_step(step_name: str, elapsed_ms: float, extra: str = "") -> None:
    """Log a pipeline step and elapsed time when PIPELINE_TIMING is enabled."""
    if _PIPELINE_TIMING:
        msg = f"PIPELINE_TIMING {step_name}: {elapsed_ms:.0f} ms"
        if extra:
            msg += f" | {extra}"
        logger.info(msg)


def _ensure_message(msg: str, facts: str, intent: str) -> str:
    """
    Ensure the transition message to the user is substantive (not empty/emoji-only).
    If the LLM's reply_to_client was empty or too short, generate a sensible default.
    """
    if msg and len(msg) >= 15:
        # Check it has at least 3 real words (not just symbols/emoji)
        words = [w for w in msg.split() if any(c.isalnum() for c in w)]
        if len(words) >= 3:
            return msg

    # Generate a contextual fallback without forcing one fixed template.
    seed = f"{intent}|{(facts or '')[:120]}"
    idx = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16) % 3
    if intent == "search":
        opts = (
            "I checked the most relevant local materials and will walk you through the strongest results.",
            "I pulled the strongest matched materials for your query and will summarize them clearly.",
            "I reviewed the top matching materials and will now explain the key points.",
        )
        return opts[idx]
    if intent == "lookup":
        opts = (
            "I located the most relevant provisions and will explain what is most useful here.",
            "I found the key statutory materials for this query and will summarize them briefly.",
            "I retrieved the strongest provision-level matches and will now explain their relevance.",
        )
        return opts[idx]
    opts = (
        "I have enough to begin grounded analysis and will now explain the strongest legal points.",
        "I can now proceed with a grounded legal view based on the materials available.",
        "I am ready to provide a grounded analysis from the current factual record and retrieved materials.",
    )
    return opts[idx]


_ANALYSIS_READY_MARKERS = (
    "say 'proceed'",
    "one last",
)


def _build_analysis_ready_prompt(conversation: list, intake_state: dict | None = None) -> str:
    """
    Human-feeling handoff message shown when Stage 1 intake is complete.
    Personalises using intake_state when available; falls back to generic variants.
    """
    if intake_state:
        issue = (intake_state.get("issue_summary") or "").strip()
        case_summary = str(((intake_state.get("case_file") or {}).get("summary") or "")).strip()
        lead = case_summary or issue
        if lead:
            short_issue = lead[:140].rstrip(".")
            return (
                f"On the present record, I have enough to assess the position — {short_issue}. "
                f"If there is one last material fact or document I should account for, share it now. "
                f"Otherwise just say 'proceed' and I'll prepare the legal position, likely routes, and immediate next steps."
            )
    idx = len([m for m in (conversation or []) if m.get("role") == "assistant"]) % 3
    opts = (
        "On the present record, I have enough to begin. If there is one last material detail or document I should consider, share it now — otherwise say 'proceed' and I'll set out the legal position and next steps.",
        "I have enough to assess the matter as it presently stands. Feel free to add one last material point, or say 'proceed' and I'll start framing the likely legal routes.",
        "The record is now sufficient for a measured analysis. Add one last material detail if needed, or say 'proceed' and I'll prepare the legal position, risks, and next steps.",
    )
    return opts[idx]


def _analysis_confirmation_already_asked(conversation: list) -> bool:
    for msg in reversed(conversation or []):
        if msg.get("role") == "assistant":
            low = (msg.get("content") or "").lower()
            return all(marker in low for marker in _ANALYSIS_READY_MARKERS)
    return False


def _last_assistant_is_analysis_ready(conversation: list) -> bool:
    """True only when the most recent assistant turn is the analysis-ready handoff."""
    for msg in reversed(conversation or []):
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            low = content.lower()
            return all(marker in low for marker in _ANALYSIS_READY_MARKERS)
        if role == "user":
            return False
    return False



def _get_analysis_stage(workflow_state: dict | None) -> str:
    if not isinstance(workflow_state, dict):
        return "intake"
    stage = str(workflow_state.get("analysisStage") or "").strip().lower()
    return stage or "intake"


def _get_stored_facts_summary(workflow_state: dict | None) -> str:
    if not isinstance(workflow_state, dict):
        return ""
    return str(workflow_state.get("factsSummary") or "").strip()


def _user_requests_precedents(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    precedent_terms = (
        "precedent", "precedents", "case law", "case laws", "case-law",
        "judgment", "judgments", "judgement", "judgements",
        "authority", "authorities", "citation", "citations",
    )
    if any(term in low for term in precedent_terms):
        return True
    if low in {"yes", "yes please", "ok", "okay", "sure", "go ahead", "please do", "proceed", "continue"}:
        return True
    tokens = set(low.replace("?", " ").replace("!", " ").split())
    return len(tokens) <= 4 and bool(tokens & {"yes", "ok", "okay", "sure", "proceed"})


def _user_declines_precedents(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    if low in {"no", "no thanks", "not now", "later", "skip", "not needed", "no need", "leave it"}:
        return True
    return low.startswith("no ") or low.startswith("not now")


def _looks_like_new_case_opening(current_message: str, conversation: list) -> bool:
    """
    Detect when a user is starting a fresh matter inside an already-completed chat.

    Keep this narrow: only trigger on substantive factual narratives, not short follow-ups
    like "proceed", "what about bail?", or "summarize this".
    """
    text = (current_message or "").strip()
    if len(text) < 35:
        return False
    low = text.lower()
    if is_stop_signal(text) or low in {"ok", "okay", "yes", "no", "thanks", "thank you"}:
        return False
    if "?" in text:
        return False

    factual_markers = (
        "my husband", "my wife", "my employer", "my landlord", "my tenant",
        "my brother", "my sister", "my father", "my mother", "my neighbour",
        "has been", "have been", "is harassing", "is threatening",
        "beats me", "assault", "abuse", "evict", "terminated me", "fired me",
        "refused", "cheated", "for last", "for the last", "for three months",
        "comes home", "living with", "police", "fir", "maintenance",
    )
    if not any(marker in low for marker in factual_markers):
        return False

    assistant_text = " ".join(
        (msg.get("content") or "").lower()
        for msg in (conversation or [])
        if msg.get("role") == "assistant"
    )
    return (
        all(marker in assistant_text for marker in _ANALYSIS_READY_MARKERS)
        or "legal opinion" in assistant_text
        or "applicable laws" in assistant_text
    )
def _build_chat_window_summary(conversation: list, current_message: str = "") -> str:
    """Deterministic summary of the current chat window for downstream analysis."""
    user_points: list[str] = []
    asked_questions: list[str] = []
    seen_user: set[str] = set()
    seen_q: set[str] = set()

    for msg in conversation or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if role == "user":
            if content not in seen_user:
                seen_user.add(content)
                user_points.append(content[:260])
        elif role == "assistant" and "?" in content:
            if content not in seen_q:
                seen_q.add(content)
                asked_questions.append(content[:220])

    current = (current_message or "").strip().replace("\n", " ")
    if current and not is_stop_signal(current) and current.lower() not in {"ok", "okay", "yes"} and current not in seen_user:
        user_points.append(current[:260])

    lines = ["CHAT WINDOW SUMMARY"]
    if user_points:
        lines.append("User facts and statements:")
        for idx, item in enumerate(user_points[-8:], 1):
            lines.append(f"{idx}. {item}")
    if asked_questions:
        lines.append("Questions already asked in this chat:")
        for idx, item in enumerate(asked_questions[-6:], 1):
            lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def _build_user_fact_history(conversation: list, current_message: str = "") -> str:
    """Join substantive user facts so response generation never starts from an empty base."""
    user_points: list[str] = []
    seen: set[str] = set()
    for msg in conversation or []:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content or is_stop_signal(content) or content.lower() in {"ok", "okay", "yes", "no"}:
            continue
        if content not in seen:
            seen.add(content)
            user_points.append(content)
    current = (current_message or "").strip()
    if current and not is_stop_signal(current) and current.lower() not in {"ok", "okay", "yes", "no"} and current not in seen:
        user_points.append(current)
    return "\n".join(user_points).strip()


def _augment_facts_with_chat_summary(facts_summary: str, conversation: list, current_message: str = "") -> str:
    """Append a compact chat-window summary so final reasoning carries the whole matter forward."""
    base = (facts_summary or "").strip() or _build_user_fact_history(conversation, current_message=current_message)
    summary = _build_chat_window_summary(conversation, current_message=current_message).strip()
    if not summary:
        return base
    if summary in base:
        return base
    return f"{base}\n\n{summary}".strip() if base else summary


def _run_search_or_lookup(
    facts_summary: str,
    intent: str,
    msg: str,
    result_count: int = None,
    progress_callback=None,
    search_strategy: str = "local_then_web",
    step_callback=None,
    token_callback=None,
    document_types: str | None = None,
    model_override: str | None = None,
) -> dict:
    """Handle search/lookup intents: go straight to research and return results."""
    # Always retrieve both bare acts AND case laws regardless of intent.
    # The original "acts_only"/"case_laws_only" split was too aggressive:
    #   â€¢ "search" â†’ "case_laws_only": if user asks "which sections apply", gets 0 bare acts
    #   â€¢ "lookup" â†’ "acts_only": never shows any supporting case laws
    # With an experiment corpus that may have acts but no case laws (or vice versa),
    # exclusive selection causes false "No results" responses.
    effective_document_types = (document_types or "").strip().lower() or "both"
    try:
        resp = generate_response(
            facts_summary,
            jurisdiction_state="",
            intent=intent,
            result_count=result_count,
            progress_callback=progress_callback,
            document_types=effective_document_types,
            search_strategy=search_strategy,
            step_callback=step_callback,
            token_callback=token_callback,
            model_override=model_override,
        )
    except Exception as e:
        logger.error("Research generation failed: %s", e, exc_info=True)
        resp = {
            "bare_act_sections": [],
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": f"I couldn't complete your search right now. Please try again or rephrase your query.",
        }

    response_type = "search_results" if intent == "search" else "lookup_results"
    explanation = (resp.get("explanation") or "").strip()
    if not explanation:
        explanation = _ensure_message("", facts_summary, intent)

    return {
        "phase": "done",
        "message": msg,
        "facts_summary": facts_summary,
        "response": {
            "bare_act_sections": resp.get("bare_act_sections", []),
            "case_laws": resp.get("case_laws", []),
            "internet_case_laws": resp.get("internet_case_laws", []),
            "explanation": explanation,
            "progress": resp.get("progress"),
            "indexing_candidates": resp.get("indexing_candidates", []),
        },
        "response_type": response_type,
        "materials_to_confirm": None,
        "indexed": False,
    }


def _default_search_strategy_for_intent(intent: str | None, requested_strategy: str | None) -> str:
    """
    Keep interactive legal opinions on the fast local path by default.

    Deep search/research flows can still opt into slower local+web behavior.
    """
    if (intent or "").strip().lower() == "legal_opinion":
        return "local_only"
    requested = (requested_strategy or "").strip().lower()
    if requested in ("local_only", "local_then_web"):
        return requested
    return "local_then_web"


def _run_generic_chat(
    conversation: list,
    current_message: str,
    token_callback=None,
    model_override: str | None = None,
) -> dict:
    """Generalist Agent: handle any query not covered by legal agents (search, lookup, legal_opinion). Answers like ChatGPT/Perplexity; no legal retrieval."""
    try:
        context = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in conversation[-6:]
        )
        prompt = f"{GENERIC_CHAT_SYSTEM}\n\nConversation:\n{context}\n\nUser: {current_message}\n\nAssistant:"
        if token_callback:
            from platform_pkg.llm import ask_llm_stream
            parts = []
            for tok in ask_llm_stream(prompt, task_hint="fast", model=model_override):
                token_callback(tok)
                parts.append(tok)
            reply = "".join(parts).strip()
        else:
            reply = ask_llm(prompt, task_hint="fast", model=model_override).strip()
        if not reply:
            reply = "I'm not sure how to answer that. Could you rephrase or ask something else?"
    except Exception as e:
        logger.error("Generic chat failed: %s", e, exc_info=True)
        reply = "I couldn't generate a response right now. Please try again."
    return {
        "phase": "done",
        "message": reply,
        "facts_summary": current_message,
        "response": {
            "bare_act_sections": [],
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": reply,
            "progress": None,
            "indexing_candidates": [],
        },
        "response_type": "generic_chat",
        "materials_to_confirm": None,
        "indexed": False,
    }


def _run_preliminary_retrieval(facts: str, model_override: str | None = None) -> dict | None:
    """
    Run full dispute decomposition + retrieval without opinion synthesis.

    Called in parallel with intake on Turn 1 so the retrieval cache is ready
    before the second client turn.  Uses the full decomposition path
    (retrieval_only=True forces interactive_fast_path off) so the vocabulary
    is translated to proper legal terms before searching the index.

    Returns a dict suitable for storage in workflow_state["preliminaryRetrievalContext"],
    or None if retrieval fails.
    """
    try:
        result = generate_response(
            facts,
            jurisdiction_state="",
            intent="legal_opinion",
            document_types="both",
            search_strategy="local_only",
            model_override=model_override,
            retrieval_only=True,
        )
        if result.get("retrieval_only"):
            logger.info(
                "Preliminary retrieval complete: %d sections, %d case laws, %d disputes",
                len(result.get("bare_act_sections") or []),
                len(result.get("flattened_case_laws") or []),
                len(result.get("disputes") or []),
            )
            return result
    except Exception as exc:
        logger.warning("Preliminary retrieval failed (non-fatal): %s", exc)
    return None


def _empty_result(phase: str = "done", facts_summary: str = None) -> dict:
    """Return a safe empty result dict."""
    return {
        "phase": phase,
        "message": "",
        "facts_summary": facts_summary,
        "response": None,
        "response_type": None,
        "materials_to_confirm": None,
        "indexed": False,
    }


def process_chat(
    conversation: list,
    current_message: str,
    phase: str,
    facts_summary: str = None,
    progress_callback=None,
    intent: str = None,
    document_types: str = None,
    search_strategy: str = None,
    result_count: int = None,
    chat_mode: str | None = None,
    step_callback=None,
    token_callback=None,
    model_override: str | None = None,
    workflow_state: dict | None = None,
    analysis_mode: str | None = None,
) -> dict:
    """
    Process a chat message and return the appropriate response.

    Args:
        conversation: List of {role, content} messages
        current_message: User's current message
        phase: "fact_collection" | "response_generation"
        facts_summary: Collected facts (when phase is response_generation)

    Returns dict with: phase, message, facts_summary, response, response_type,
    materials_to_confirm, indexed
    """

    t_pipeline_start = time.perf_counter()
    _log_step("process_chat START", 0, f"phase={phase}")

    try:
        if phase == "fact_collection" and (chat_mode or "").strip().lower() != "general":
            kickoff_runtime_warmup("intake")
            # Do not warm Ollama at startup/background; warm only if UI selected
            # Qwen and this is the first user turn.
            prior_user_turns = sum(1 for m in (conversation or []) if m.get("role") == "user")
            if prior_user_turns == 0:
                kickoff_ollama_warmup_if_qwen(model_override, "first_qwen_message")
    except Exception:
        pass

    # ---- Safety gate: check input before any processing ----
    current_message = sanitize_input(current_message)
    safety = check_query_safety(current_message)
    _log_step("safety_check", (time.perf_counter() - t_pipeline_start) * 1000)

    if not safety.get("safe"):
        logger.warning("Blocked unsafe query (risk=%s): %s", safety.get("risk_level"), current_message[:80])
        return {
            "phase": "fact_collection",
            "message": safety.get("reason", "I cannot process this request. Please rephrase your legal question."),
            "facts_summary": None,
            "response": None,
            "response_type": None,
            "materials_to_confirm": None,
            "indexed": False,
        }

    # Normalise chat_mode (manual override from UI)
    mode = (chat_mode or "").strip().lower()

    # ---- Phase: Fact Collection ----
    if phase == "fact_collection":
        # Manual override: general chat â†’ skip legal routing entirely
        if mode == "general":
            with guard_activity("fact_collection:generic_chat"):
                return _run_generic_chat(
                    conversation, current_message, token_callback=token_callback, model_override=model_override
                )

        # Manual override: direct legal research (search-style workflow)
        if mode == "legal_research":
            # Treat message as a single research query; no multi-step interview.
            msg = _ensure_message("", current_message, intent="search")
            with guard_activity("fact_collection:legal_research_search"):
                return _run_search_or_lookup(
                    current_message,
                    intent="search",
                    msg=msg,
                    result_count=None,
                    progress_callback=progress_callback,
                    # Use local-first retrieval with Indiankanoon fallback only when local retrieval is empty.
                    search_strategy="local_then_web",
                    step_callback=step_callback,
                    token_callback=token_callback,
                    model_override=model_override,
                )

        analysis_stage = _get_analysis_stage(workflow_state)

        if analysis_stage == "await_precedent_confirmation":
            stored_facts = _get_stored_facts_summary(workflow_state) or _build_user_fact_history(conversation)
            if _user_requests_precedents(current_message):
                _log_step("fact_collection PRECEDENT_STAGE_ACK", (time.perf_counter() - t_pipeline_start) * 1000)
                return {
                    "phase": "response_generation",
                    "message": "",
                    "facts_summary": stored_facts,
                    "intent": "legal_opinion",
                    "document_types": "both",
                    "search_strategy": "local_only",
                    "result_count": None,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "precedent_generation",
                    "analysis_mode": "precedents_only",
                }
            if _user_declines_precedents(current_message):
                return {
                    "phase": "done",
                    "message": "Understood. If you later want supporting judicial precedents for these disputes, just ask for the relevant case laws and I will continue from here.",
                    "facts_summary": stored_facts,
                    "response": None,
                    "response_type": "bare_act_guidance",
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "await_precedent_confirmation",
                }
            if not _looks_like_new_case_opening(current_message, conversation):
                merged_facts = _augment_facts_with_chat_summary(
                    stored_facts,
                    conversation,
                    current_message=current_message,
                )
                _log_step("fact_collection BARE_ACT_REFRESH -> grounded_analysis", (time.perf_counter() - t_pipeline_start) * 1000)
                return {
                    "phase": "response_generation",
                    "message": "",
                    "facts_summary": merged_facts,
                    "intent": "legal_opinion",
                    "document_types": "both",
                    "search_strategy": "local_only",
                    "result_count": None,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "grounded_analysis",
                    "analysis_mode": "full_opinion",
                }

        # If an older chat still contains an analysis-ready handoff, treat the
        # user's next message as additional facts and move straight to grounded
        # analysis instead of re-entering intake.
        if _last_assistant_is_analysis_ready(conversation):
            base_facts = _build_user_fact_history(conversation, current_message=current_message)
            merged_facts = _augment_facts_with_chat_summary(
                facts_summary or base_facts,
                conversation,
                current_message=current_message,
            )
            _log_step("fact_collection ANALYSIS_READY_ACK -> grounded_analysis", (time.perf_counter() - t_pipeline_start) * 1000)
            return {
                "phase": "response_generation",
                "message": "",
                "facts_summary": merged_facts,
                "intent": "legal_opinion",
                "document_types": "both",
                "search_strategy": "local_only",
                "result_count": None,
                "response": None,
                "response_type": None,
                "materials_to_confirm": None,
                "indexed": False,
                "analysis_stage": "grounded_analysis",
                "analysis_mode": "full_opinion",
            }
        if _looks_like_new_case_opening(current_message, conversation):
            logger.info("Detected fresh case opening inside completed chat; re-entering intake with reset conversation")
            conversation = []

        # Explicit legal opinion mode: use the 6-stage legal_opinion_intake.process_turn()
        # which has a deterministic, model-driven readiness gate (all 4 anchor fields
        # populated) instead of a hard turn limit.
        force_legal = mode == "legal_opinion"

        if force_legal:
            from concurrent.futures import ThreadPoolExecutor as _TPE

            t_before_intake = time.perf_counter()
            session = {
                "history": conversation,
                "intake_state": (workflow_state or {}).get("intakeState") or None,
            }

            # Read any retrieval cache stored from a prior turn.
            prelim_context = (workflow_state or {}).get("preliminaryRetrievalContext") or None

            # On the very first user turn with no cache: run intake + preliminary
            # retrieval in parallel.  Retrieval uses full dispute decomposition so
            # queries are translated to proper legal vocabulary before searching the
            # index.  The result is stored as a session-level cache — subsequent turns
            # skip the expensive retrieval step entirely.
            _prior_user_turns = sum(1 for m in (conversation or []) if m.get("role") == "user")
            _should_prelim = prelim_context is None and _prior_user_turns == 0

            if _should_prelim:
                def _run_intake_guarded():
                    with guard_activity("fact_collection:intake_process_turn"):
                        return _intake_process_turn(session, current_message)

                def _run_prelim_guarded():
                    with guard_activity("fact_collection:preliminary_retrieval"):
                        return _run_preliminary_retrieval(current_message, model_override)

                with _TPE(max_workers=2) as _pool:
                    _intake_fut = _pool.submit(_run_intake_guarded)
                    _prelim_fut = _pool.submit(_run_prelim_guarded)
                    intake_result = _intake_fut.result()
                    prelim_context = _prelim_fut.result()

                _log_step(
                    "intake_process_turn + preliminary_retrieval (parallel)",
                    (time.perf_counter() - t_before_intake) * 1000,
                    f"advance={intake_result.get('advance_to_stage2')} cache={'ok' if prelim_context else 'miss'}",
                )
            else:
                with guard_activity("fact_collection:intake_process_turn"):
                    intake_result = _intake_process_turn(session, current_message)
                _log_step(
                    "intake_process_turn",
                    (time.perf_counter() - t_before_intake) * 1000,
                    f"advance={intake_result.get('advance_to_stage2')} urgency={intake_result.get('urgency_signal')} cache={'hit' if prelim_context else 'miss'}",
                )

            intake_state_out = intake_result.get("intake_state") or {}
            stop_requested = is_stop_signal(current_message)
            # "immediate" only — "near_term" means client confirmed safe; allow normal flow
            urgency_immediate = intake_result.get("urgency_signal") == "immediate"

            if intake_result.get("advance_to_stage2") or stop_requested:
                # If urgency is immediate, the process_turn reply is a safety-first
                # response — surface that instead of jumping to the analysis handoff.
                # The client needs safety guidance before we talk about legal analysis.
                if urgency_immediate and not stop_requested:
                    _log_step("fact_collection URGENT (hold analysis-ready)", (time.perf_counter() - t_pipeline_start) * 1000)
                    return {
                        "phase": "fact_collection",
                        "message": intake_result.get("reply") or current_message,
                        "facts_summary": None,
                        "intake_state": intake_state_out,
                        "preliminary_retrieval_context": prelim_context,
                        "response": None,
                        "response_type": None,
                        "materials_to_confirm": None,
                        "indexed": False,
                        "analysis_stage": "intake",
                    }

                # Intake is complete — carry the retrieval cache into Stage 2 so
                # the response_generation phase can skip re-retrieval.
                facts = _build_user_fact_history(conversation, current_message=current_message)
                facts = _augment_facts_with_chat_summary(facts, conversation, current_message=current_message)
                _log_step("fact_collection ADVANCE -> grounded_analysis (process_turn)", (time.perf_counter() - t_pipeline_start) * 1000)
                return {
                    "phase": "response_generation",
                    "message": "",
                    "facts_summary": facts,
                    "intake_state": intake_state_out,
                    "preliminary_retrieval_context": prelim_context,
                    "intent": "legal_opinion",
                    "document_types": "both",
                    "search_strategy": "local_only",
                    "result_count": None,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "grounded_analysis",
                    "analysis_mode": "full_opinion",
                }

            # Safety gate: if all 4 anchor fields are populated but the model
            # still returned advance_to_stage2=False (common false negative on
            # first-turn comprehensive messages), force-advance to Stage 2.
            _anchors = (
                str(intake_state_out.get("issue_summary") or "").strip(),
                str(intake_state_out.get("relationship_to_other_party") or "").strip(),
                str(intake_state_out.get("timeframe_status") or "").strip(),
                str(intake_state_out.get("client_goal_initial") or "").strip(),
            )
            _all_anchors_present = all(
                v and v.lower() not in ("unknown", "null", "none", "")
                for v in _anchors
            )
            if _all_anchors_present and not urgency_immediate:
                _log_step("fact_collection ANCHOR_GATE -> grounded_analysis (all 4 anchors present)", (time.perf_counter() - t_pipeline_start) * 1000)
                facts = _build_user_fact_history(conversation, current_message=current_message)
                facts = _augment_facts_with_chat_summary(facts, conversation, current_message=current_message)
                return {
                    "phase": "response_generation",
                    "message": "",
                    "facts_summary": facts,
                    "intake_state": intake_state_out,
                    "preliminary_retrieval_context": prelim_context,
                    "intent": "legal_opinion",
                    "document_types": "both",
                    "search_strategy": "local_only",
                    "result_count": None,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "grounded_analysis",
                    "analysis_mode": "full_opinion",
                }

            # Still collecting — persist the retrieval cache so the next turn can
            # use it for law-informed gap questions and eventually skip re-retrieval.
            _log_step("fact_collection CONTINUE (process_turn)", (time.perf_counter() - t_pipeline_start) * 1000)
            return {
                "phase": "fact_collection",
                "message": intake_result.get("reply") or current_message,
                "facts_summary": None,
                "intake_state": intake_state_out,
                "preliminary_retrieval_context": prelim_context,
                "response": None,
                "response_type": None,
                "materials_to_confirm": None,
                "indexed": False,
                "analysis_stage": "intake",
            }

        # Default (non-legal-opinion) mode: use fact collector for intent routing.
        t_before_fact = time.perf_counter()
        with guard_activity("fact_collection:get_next_question_or_complete"):
            result = get_next_question_or_complete(
                conversation,
                current_message,
                force_legal=False,
                token_callback=token_callback,
                model_override=model_override,
            )
        _log_step("get_next_question_or_complete", (time.perf_counter() - t_before_fact) * 1000, f"action={result.get('action')}")

        if result.get("action") == "complete":
            intent = result.get("intent", "legal_opinion")
            facts = result.get("facts_summary", current_message)
            facts = _augment_facts_with_chat_summary(facts, conversation, current_message=current_message)
            msg = _ensure_message(result.get("message", ""), facts, intent)

            if intent in ("search", "lookup"):
                count = result.get("result_count")
                strategy = result.get("search_strategy", "local_then_web")
                _log_step("FACT_COLLECTION â†’ retrieval (search/lookup)", (time.perf_counter() - t_pipeline_start) * 1000, f"facts_len={len(facts or '')}")
                with guard_activity(f"fact_collection:run_{intent}"):
                    return _run_search_or_lookup(
                        facts,
                        intent,
                        msg,
                        result_count=count,
                        progress_callback=progress_callback,
                        search_strategy=strategy,
                        step_callback=step_callback,
                        token_callback=token_callback,
                        document_types=result.get("document_types", "both"),
                    )

            if intent == "bulk_ingest":
                # Bulk ingest removed; treat as generic chat
                with guard_activity("fact_collection:bulk_ingest_to_generic_chat"):
                    return _run_generic_chat(
                        conversation, current_message, token_callback=token_callback, model_override=model_override
                    )

            if intent == "generic_chat":
                with guard_activity("fact_collection:intent_generic_chat"):
                    return _run_generic_chat(
                        conversation, current_message, token_callback=token_callback, model_override=model_override
                    )

            # Once enough facts exist, move directly into grounded analysis
            # instead of pausing for a confirmation-only turn.
            _log_step(
                "FACT_COLLECTION -> grounded_analysis",
                (time.perf_counter() - t_pipeline_start) * 1000,
                f"facts_len={len(facts or '')}",
            )
            return {
                "phase": "response_generation",
                "message": "",
                "facts_summary": facts,
                "intent": intent,
                "document_types": result.get("document_types", "both"),
                "search_strategy": _default_search_strategy_for_intent(
                    intent,
                    result.get("search_strategy"),
                ),
                "result_count": result.get("result_count"),
                "response": None,
                "response_type": None,
                "materials_to_confirm": None,
                "indexed": False,
                "analysis_stage": "grounded_analysis",
                "analysis_mode": "full_opinion",
            }

        # Still collecting facts — the compact intake path already streamed any tokens.
        # Just return the structured question result to the caller.
        _log_step("fact_collection DONE (ask)", (time.perf_counter() - t_pipeline_start) * 1000)
        return {
            "phase": "fact_collection",
            "message": result.get("question") or result.get("message", current_message),
            "facts_summary": None,
            "response": None,
            "response_type": None,
            "materials_to_confirm": None,
            "indexed": False,
        }

    # ---- Phase: Response Generation ----
    elif phase == "response_generation":
        _log_step("response_generation START", (time.perf_counter() - t_pipeline_start) * 1000)
        facts = _augment_facts_with_chat_summary(
            facts_summary or current_message,
            conversation,
            current_message=current_message,
        )
        use_intent = intent or "legal_opinion"
        use_document_types = document_types or "both"
        use_search_strategy = _default_search_strategy_for_intent(use_intent, search_strategy)
        use_result_count = result_count
        use_analysis_mode = (analysis_mode or "full_opinion").strip().lower() or "full_opinion"
        use_intake_state = (workflow_state or {}).get("intakeState") or None

        # Use session-level retrieval cache when available (set during intake Turn 1).
        # This skips the expensive dispute decomposition + retrieval step entirely.
        _prelim_ctx = (workflow_state or {}).get("preliminaryRetrievalContext") or None
        _cache_eligible = (
            _prelim_ctx
            and _prelim_ctx.get("retrieval_only")
            and use_intent == "legal_opinion"
            and use_analysis_mode in ("full_opinion", "bare_acts_only", "precedents_only")
        )

        t_before_gen = time.perf_counter()
        try:
            if _cache_eligible:
                _log_step("response_generation CACHE HIT — skipping retrieval", 0)
                with guard_activity("response_generation:generate_response_from_cache"):
                    resp = generate_legal_opinion_from_cache(
                        _prelim_ctx,
                        facts,
                        intent=use_intent,
                        analysis_mode=use_analysis_mode,
                        token_callback=token_callback,
                        model_override=model_override,
                        step_callback=step_callback,
                        search_strategy=use_search_strategy,
                    )
            else:
                with guard_activity("response_generation:generate_response"):
                    resp = generate_response(
                        facts,
                        jurisdiction_state="",
                        intent=use_intent,
                        progress_callback=progress_callback,
                        document_types=use_document_types,
                        search_strategy=use_search_strategy,
                        result_count=use_result_count,
                        step_callback=step_callback,
                        token_callback=token_callback,
                        model_override=model_override,
                        analysis_mode=use_analysis_mode,
                        intake_state=use_intake_state,
                    )
        except Exception as e:
            logger.error("Response generation failed: %s", e, exc_info=True)
            return {
                "phase": "done",
                "message": "I encountered an issue while preparing your analysis. Please try again.",
                "facts_summary": facts,
                "response": None,
                "response_type": "legal_opinion",
                "materials_to_confirm": None,
                "indexed": False,
                "analysis_stage": _get_analysis_stage(workflow_state),
            }
        _log_step("generate_response (retrieval+LLM)", (time.perf_counter() - t_before_gen) * 1000)

        explanation = (resp.get("explanation") or "").strip()
        if explanation:
            resp_safety = check_response_safety(explanation)
            if not resp_safety.get("safe"):
                logger.warning("Unsafe LLM output blocked")
                explanation = "I was unable to generate a safe response for this query. Please try rephrasing."

        if use_analysis_mode == "bare_acts_only":
            response_type = "bare_act_guidance"
            next_analysis_stage = "await_precedent_confirmation"
        elif use_analysis_mode == "precedents_only":
            response_type = "precedent_support"
            next_analysis_stage = "precedents_ready"
        else:
            response_type = "legal_opinion"
            next_analysis_stage = "precedents_ready"

        return {
            "phase": "done",
            "message": "",
            "facts_summary": facts,
            "response": {
                "bare_act_sections": resp.get("bare_act_sections", []),
                "case_laws": resp.get("case_laws", []),
                "internet_case_laws": resp.get("internet_case_laws", []),
                "next_steps": resp.get("next_steps", []),
                "next_steps_summary": (resp.get("next_steps_summary") or "").strip(),
                "explanation": explanation,
                "progress": resp.get("progress"),
                "indexing_candidates": resp.get("indexing_candidates", []),
            },
            "response_type": response_type,
            "materials_to_confirm": None,
            "indexed": False,
            "analysis_stage": next_analysis_stage,
        }


    return _empty_result()


