"""
Interactive Chat Orchestrator â€” Handles multi-phase legal chat flow:
1. Fact collection (professional advocate intake) with intent detection
2. Response generation (bare acts + case laws + structured opinion)

Phase 2 cleanup:
- Greeting logic moved to fact_collector (is_greeting + generate_greeting_response)
- No more inline _is_proper_greeting() or ad-hoc LLM calls for greetings
- Direct v2 pipeline import, no fallbacks
- Disclaimer appended to legal opinions
- generic_chat: non-legal topics (politics, science, tech) answered like ChatGPT/Perplexity.
"""

import logging
import os
import time

from llm.ollama_client import ask_llm
from services.fact_collector import get_next_question_or_complete, is_stop_signal
from services.runtime_warmup import kickoff_runtime_warmup
from services.response_generator_v2 import (
    generate_response_v2 as generate_response,
)
from services.content_guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger(__name__)

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

    # Generate contextual default based on intent
    if intent == "search":
        return "I've searched for relevant case laws and judgments for your query. Here's what I found."
    elif intent == "lookup":
        return "I've looked up the relevant bare act provisions for your query. Here's what I found."
    else:
        return "Thank you for sharing the details. I've researched the applicable bare acts and case laws. Here's my analysis."


_ANALYSIS_READY_PREFIX = "I have enough to identify the applicable bare act sections"


def _build_analysis_ready_prompt() -> str:
    return (
        "I have enough to identify the applicable bare act sections for the dispute or disputes that emerge from what you have shared. "
        "If you want, you can add one last important fact now. "
        "Otherwise, just say 'proceed' and I will first show the relevant bare act sections from the local legal database."
    )


def _analysis_confirmation_already_asked(conversation: list) -> bool:
    for msg in reversed(conversation or []):
        if msg.get("role") == "assistant":
            return _ANALYSIS_READY_PREFIX.lower() in (msg.get("content") or "").lower()
    return False


def _last_assistant_is_analysis_ready(conversation: list) -> bool:
    """True only when the most recent assistant turn is the analysis-ready handoff."""
    for msg in reversed(conversation or []):
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            return _ANALYSIS_READY_PREFIX.lower() in content.lower()
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
    return _ANALYSIS_READY_PREFIX.lower() in assistant_text or "legal opinion" in assistant_text or "applicable laws" in assistant_text
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
        explanation = "Here's what I found for your query."

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


def _run_generic_chat(conversation: list, current_message: str, token_callback=None) -> dict:
    """Generalist Agent: handle any query not covered by legal agents (search, lookup, legal_opinion). Answers like ChatGPT/Perplexity; no legal retrieval."""
    try:
        context = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in conversation[-6:]
        )
        prompt = f"{GENERIC_CHAT_SYSTEM}\n\nConversation:\n{context}\n\nUser: {current_message}\n\nAssistant:"
        if token_callback:
            from llm.ollama_client import ask_llm_stream
            parts = []
            for tok in ask_llm_stream(prompt, task_hint="fast"):
                token_callback(tok)
                parts.append(tok)
            reply = "".join(parts).strip()
        else:
            reply = ask_llm(prompt, task_hint="fast").strip()
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
            return _run_generic_chat(conversation, current_message, token_callback=token_callback)

        # Manual override: direct legal research (search-style workflow)
        if mode == "legal_research":
            # Treat message as a single research query; no multi-step interview.
            msg = _ensure_message("", current_message, intent="search")
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
                _log_step("fact_collection BARE_ACT_REFRESH", (time.perf_counter() - t_pipeline_start) * 1000)
                return {
                    "phase": "response_generation",
                    "message": "",
                    "facts_summary": merged_facts,
                    "intent": "legal_opinion",
                    "document_types": "acts_only",
                    "search_strategy": "local_only",
                    "result_count": None,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                    "analysis_stage": "bare_acts_only",
                    "analysis_mode": "bare_acts_only",
                }

        # If the immediately previous assistant turn was the analysis-ready handoff,
        # the user is either confirming to proceed or giving one last material fact.
        # In either case, do not re-enter intake.
        if _last_assistant_is_analysis_ready(conversation):
            base_facts = _build_user_fact_history(conversation, current_message=current_message)
            merged_facts = _augment_facts_with_chat_summary(
                facts_summary or base_facts,
                conversation,
                current_message=current_message,
            )
            _log_step("fact_collection ANALYSIS_READY_ACK", (time.perf_counter() - t_pipeline_start) * 1000)
            return {
                "phase": "response_generation",
                "message": "",
                "facts_summary": merged_facts,
                "intent": "legal_opinion",
                "document_types": "acts_only",
                "search_strategy": "local_only",
                "result_count": None,
                "response": None,
                "response_type": None,
                "materials_to_confirm": None,
                "indexed": False,
                "analysis_stage": "bare_acts_only",
                "analysis_mode": "bare_acts_only",
            }
        if _looks_like_new_case_opening(current_message, conversation):
            logger.info("Detected fresh case opening inside completed chat; re-entering intake with reset conversation")
            conversation = []

        # Default / explicit legal opinion: use fact collector, but force LEGAL
        # so Gate 1 can never downgrade to GENERALIST when the user chose legal mode.
        force_legal = mode == "legal_opinion"
        t_before_fact = time.perf_counter()
        result = get_next_question_or_complete(conversation, current_message, force_legal=force_legal, token_callback=token_callback)
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
                return _run_generic_chat(conversation, current_message, token_callback=token_callback)

            if intent == "generic_chat":
                return _run_generic_chat(conversation, current_message, token_callback=token_callback)

            # Keep intake fast: once enough facts exist, ask for confirmation to proceed
            # to full research instead of launching retrieval in the same turn.
            if not is_stop_signal(current_message) and not _analysis_confirmation_already_asked(conversation):
                _log_step("fact_collection ANALYSIS_READY", (time.perf_counter() - t_pipeline_start) * 1000)
                return {
                    "phase": "fact_collection",
                    "message": _build_analysis_ready_prompt(),
                    "facts_summary": facts,
                    "response": None,
                    "response_type": None,
                    "materials_to_confirm": None,
                    "indexed": False,
                "analysis_stage": "ready_for_bare_acts",
                }

            _log_step(
                "FACT_COLLECTION â†’ response_generation",
                (time.perf_counter() - t_pipeline_start) * 1000,
                f"facts_len={len(facts or '')}",
            )
            return {
                "phase": "response_generation",
                "message": result.get("message", ""),
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
        t_before_gen = time.perf_counter()
        try:
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




