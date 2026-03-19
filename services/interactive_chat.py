"""
Interactive Chat Orchestrator — Handles multi-phase legal chat flow:
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
from services.fact_collector import get_next_question_or_complete
from services.response_generator_v2 import (
    generate_response_v2 as generate_response,
    retrieve_bare_acts_phase,
    generate_final_opinion_with_case_laws,
)
from services.content_guard import check_query_safety, sanitize_input, check_response_safety

logger = logging.getLogger(__name__)

# Set PIPELINE_TIMING=1 in env to log elapsed ms for each step (debug slow follow-ups)
_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


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


def _run_bare_acts_phase(facts_summary: str, progress_callback=None, states: list = None, conversation_history: list = None, step_callback=None) -> dict:
    """
    New Phase A: retrieve and explain bare acts, then ask a targeted follow-up question
    (or proceed directly if no follow-up is needed).
    Returns phase="bare_acts_presented" so the API can show sections + question to the user.

    states: list of state names detected from the query (e.g. ['Telangana']).
      Used to search BOTH Union/central acts AND state-specific acts.
    conversation_history: full chat history so the followup-question generator can avoid
      repeating questions that were already asked during intake.
    """
    t_phase = time.perf_counter()
    try:
        result = retrieve_bare_acts_phase(
            facts_summary,
            progress_callback=progress_callback,
            states=states or [],
            conversation_history=conversation_history or [],
            step_callback=step_callback,
        )
    except Exception as e:
        logger.error("_run_bare_acts_phase: retrieval failed: %s", e, exc_info=True)
        result = {"bare_acts": [], "followup_question": None, "intro_text": "I couldn't retrieve bare act sections right now. Proceeding with general analysis."}

    bare_acts = result.get("bare_acts", [])
    disputes   = result.get("disputes", [])   # per-dispute groupings for UI rendering
    followup_question = result.get("followup_question")
    intro_text = result.get("intro_text", "Here are the relevant bare act sections I found.")
    _log_step(
        "_run_bare_acts_phase",
        (time.perf_counter() - t_phase) * 1000,
        f"bare_acts={len(bare_acts)} disputes={len(disputes)} followup={'yes' if followup_question else 'no'}",
    )

    # If no follow-up needed and we have sections, we can still present them before
    # the user confirms to proceed — keep phase as bare_acts_presented in all cases.
    return {
        "phase": "bare_acts_presented",
        "message": intro_text,
        "facts_summary": facts_summary,
        "bare_acts": bare_acts,
        "disputes": disputes,             # forwarded to API → frontend for per-dispute layout
        "states": states or [],           # so frontend can send back for Phase B (case-law search)
        "followup_question": followup_question,
        "response": {
            "bare_act_sections": bare_acts,
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": intro_text,
            "progress": None,
            "indexing_candidates": [],
        },
        "response_type": "bare_acts_presented",
        "materials_to_confirm": None,
        "indexed": False,
    }


def _run_final_with_case_laws(facts_summary: str, bare_acts: list, additional_info: str, progress_callback=None, states: list = None, step_callback=None, token_callback=None) -> dict:
    """
    New Phase B: given collected facts + already-retrieved bare acts + any additional user info,
    retrieve case laws, associate them with sections, and generate the structured final opinion.
    states: optional list of state names (e.g. from Phase A) so web search is jurisdiction-aware.
    """
    t_phase = time.perf_counter()
    try:
        resp = generate_final_opinion_with_case_laws(
            facts_summary,
            bare_acts,
            additional_info=additional_info,
            progress_callback=progress_callback,
            states=states or [],
            step_callback=step_callback,
            token_callback=token_callback,
        )
    except Exception as e:
        logger.error("_run_final_with_case_laws failed: %s", e, exc_info=True)
        resp = {
            "bare_act_sections": bare_acts,
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": "I encountered an issue while preparing the final analysis. Please try again.",
            "progress": None,
            "indexing_candidates": [],
        }

    explanation = (resp.get("explanation") or "").strip()
    if explanation:
        from services.content_guard import check_response_safety
        resp_safety = check_response_safety(explanation)
        if not resp_safety.get("safe"):
            explanation = "I was unable to generate a safe response for this query. Please rephrase."
    _log_step(
        "_run_final_with_case_laws",
        (time.perf_counter() - t_phase) * 1000,
        f"bare_acts={len(resp.get('bare_act_sections', []))} case_laws={len(resp.get('case_laws', []))}",
    )

    return {
        "phase": "done",
        "message": "",
        "facts_summary": facts_summary,
        "response": {
            "bare_act_sections": resp.get("bare_act_sections", []),
            "case_laws": resp.get("case_laws", []),
            "internet_case_laws": resp.get("internet_case_laws", []),
            "explanation": explanation,
            "progress": resp.get("progress"),
            "indexing_candidates": resp.get("indexing_candidates", []),
        },
        "response_type": "legal_opinion",
        "materials_to_confirm": None,
        "indexed": False,
    }


def _run_search_or_lookup(facts_summary: str, intent: str, msg: str, result_count: int = None, progress_callback=None, search_strategy: str = "local_only", step_callback=None, token_callback=None) -> dict:
    """Handle search/lookup intents: go straight to research and return results."""
    # Always retrieve both bare acts AND case laws regardless of intent.
    # The original "acts_only"/"case_laws_only" split was too aggressive:
    #   • "search" → "case_laws_only": if user asks "which sections apply", gets 0 bare acts
    #   • "lookup" → "acts_only": never shows any supporting case laws
    # With an experiment corpus that may have acts but no case laws (or vice versa),
    # exclusive selection causes false "No results" responses.
    document_types = "both"
    try:
        resp = generate_response(
            facts_summary,
            jurisdiction_state="",
            intent=intent,
            result_count=result_count,
            progress_callback=progress_callback,
            document_types=document_types,
            search_strategy=search_strategy,
            step_callback=step_callback,
            token_callback=token_callback,
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


def _run_legal_opinion_simple(
    facts_summary: str,
    progress_callback=None,
    search_strategy: str = "local_only",
    step_callback=None,
    token_callback=None,
) -> dict:
    """
    Single-pass legal opinion: hybrid retrieval (local only) + LLM reasoning.

    This bypasses the older two-phase bare-acts-then-caselaw flow and skips web
    enrichment so the hot path is:
      facts → local retrieval (acts + cases) → opinion.
    """
    try:
        resp = generate_response(
            facts_summary,
            jurisdiction_state="",
            intent="legal_opinion",
            progress_callback=progress_callback,
            document_types="both",
            search_strategy=search_strategy or "local_only",
            result_count=None,
            step_callback=step_callback,
            token_callback=token_callback,
        )
    except Exception as e:
        logger.error("Legal opinion generation failed: %s", e, exc_info=True)
        return {
            "phase": "done",
            "message": "I encountered an issue while preparing your legal opinion. Please try again or rephrase your query.",
            "facts_summary": facts_summary,
            "response": None,
            "response_type": "legal_opinion",
            "materials_to_confirm": None,
            "indexed": False,
        }

    # Output safety check
    explanation = (resp.get("explanation") or "").strip()
    if explanation:
        resp_safety = check_response_safety(explanation)
        if not resp_safety.get("safe"):
            logger.warning("Unsafe legal-opinion output blocked")
            explanation = "I was unable to generate a safe response for this query. Please try rephrasing."

    return {
        "phase": "done",
        "message": "",
        "facts_summary": facts_summary,
        "response": {
            "bare_act_sections": resp.get("bare_act_sections", []),
            "case_laws": resp.get("case_laws", []),
            "internet_case_laws": resp.get("internet_case_laws", []),
            "explanation": explanation or "Here's my analysis based on the most relevant bare acts and case laws I found.",
            "progress": resp.get("progress"),
            "indexing_candidates": resp.get("indexing_candidates", []),
        },
        "response_type": "legal_opinion",
        "materials_to_confirm": None,
        "indexed": False,
    }


# Generalist Agent — handles all conversations and queries NOT covered by legal agents:
# non-legal topics (politics, science, tech, general knowledge), how-to, trivia, and any unclear request.
GENERIC_CHAT_SYSTEM = """You are a helpful, knowledgeable generalist assistant. You handle any question or conversation that is not a legal research request (case laws, bare acts, legal advice, or bulk indexing from a URL). Answer clearly and conversationally — like ChatGPT or Perplexity. Topics you handle include: politics, science, technology, history, how-to, trivia, general knowledge, and any other non-legal or ambiguous query. If the question is clearly about law (Indian law, cases, acts, legal advice), briefly say you're better suited for legal research and suggest they ask for case laws or legal opinion in this app. Otherwise answer from your training knowledge. Keep responses informative and concise. If you don't know something, say so. Do not use legal disclaimers for non-legal topics."""


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
            for tok in ask_llm_stream(prompt):
                token_callback(tok)
                parts.append(tok)
            reply = "".join(parts).strip()
        else:
            reply = ask_llm(prompt).strip()
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
    bare_acts: list = None,
    states: list = None,
    pending_indexing_list: list = None,
    chat_mode: str | None = None,
    step_callback=None,
    token_callback=None,
) -> dict:
    """
    Process a chat message and return the appropriate response.

    Args:
        conversation: List of {role, content} messages
        current_message: User's current message
        phase: "fact_collection" | "response_generation" | "confirm_index"
        facts_summary: Collected facts (when phase is response_generation)

    Returns dict with: phase, message, facts_summary, response, response_type,
    materials_to_confirm, indexed
    """

    t_pipeline_start = time.perf_counter()
    _log_step("process_chat START", 0, f"phase={phase}")

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
        # Manual override: general chat → skip legal routing entirely
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
                # Use local-only retrieval in hot path; caller can still request web via API-level overrides.
                search_strategy="local_only",
                step_callback=step_callback,
                token_callback=token_callback,
            )

        # Default / explicit legal opinion: use fact collector, but force LEGAL
        # so Gate 1 can never downgrade to GENERALIST when the user chose legal mode.
        force_legal = mode == "legal_opinion"
        t_before_fact = time.perf_counter()
        result = get_next_question_or_complete(conversation, current_message, force_legal=force_legal, token_callback=token_callback)
        _log_step("get_next_question_or_complete", (time.perf_counter() - t_before_fact) * 1000, f"action={result.get('action')}")

        if result.get("action") == "complete":
            intent = result.get("intent", "legal_opinion")
            facts = result.get("facts_summary", current_message)
            msg = _ensure_message(result.get("message", ""), facts, intent)

            if intent in ("search", "lookup"):
                count = result.get("result_count")
                strategy = result.get("search_strategy", "local_only")
                _log_step("FACT_COLLECTION → retrieval (search/lookup)", (time.perf_counter() - t_pipeline_start) * 1000, f"facts_len={len(facts or '')}")
                return _run_search_or_lookup(facts, intent, msg, result_count=count, progress_callback=progress_callback, search_strategy=strategy, step_callback=step_callback, token_callback=token_callback)

            if intent == "bulk_ingest":
                # Bulk ingest removed; treat as generic chat
                return _run_generic_chat(conversation, current_message, token_callback=token_callback)

            if intent == "generic_chat":
                return _run_generic_chat(conversation, current_message, token_callback=token_callback)

            # Legal opinion: two-phase flow.
            # Phase A: retrieve bare act sections, present legal protection summary,
            # ask client if they want a detailed opinion (or ask for any critical gaps).
            # Phase B (bare_acts_review): retrieve case laws + generate full structured opinion.
            facts_len = len(facts or "")
            states = result.get("states", [])
            _log_step(
                "FACT_COLLECTION → _run_bare_acts_phase",
                (time.perf_counter() - t_pipeline_start) * 1000,
                f"facts_len={facts_len}",
            )
            return _run_bare_acts_phase(facts, progress_callback=progress_callback, states=states, conversation_history=conversation, step_callback=step_callback)

        # Still collecting facts — tokens were already streamed inside _run_single_gate
        # when token_callback was provided.  Just return the structured result.
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
        facts = facts_summary or current_message
        use_intent = intent or "legal_opinion"
        use_document_types = document_types or "both"
        use_search_strategy = search_strategy or "local_only"
        use_result_count = result_count
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
            }
        _log_step("generate_response (retrieval+LLM)", (time.perf_counter() - t_before_gen) * 1000)

        # Check if materials need user confirmation first
        if resp.get("needs_confirmation"):
            materials = resp.get("materials_to_confirm", {})
            return {
                "phase": "confirm_materials",
                "message": resp.get("summary", resp.get("explanation", "")),
                "facts_summary": facts,
                "response": {
                    "bare_act_sections": resp.get("bare_act_sections", []),
                    "case_laws": resp.get("case_laws", []),
                    "internet_case_laws": [],
                    "explanation": resp.get("summary", resp.get("explanation", "")),
                },
                "response_type": "legal_opinion",
                "materials_to_confirm": {
                    "bare_acts": materials.get("bare_acts", []),
                    "case_laws": materials.get("case_laws", []),
                },
                "indexed": False,
            }

        # Full response ready — check output safety
        explanation = (resp.get("explanation") or "").strip()
        if explanation:
            resp_safety = check_response_safety(explanation)
            if not resp_safety.get("safe"):
                logger.warning("Unsafe LLM output blocked")
                explanation = "I was unable to generate a safe response for this query. Please try rephrasing."

        return {
            "phase": "done",
            "message": "",  # transition text only — explanation goes in response.explanation
            "facts_summary": facts,
            "response": {
                "bare_act_sections": resp.get("bare_act_sections", []),
                "case_laws": resp.get("case_laws", []),
                "internet_case_laws": resp.get("internet_case_laws", []),
                "explanation": explanation,
                "progress": resp.get("progress"),
                "indexing_candidates": resp.get("indexing_candidates", []),
            },
            "response_type": "legal_opinion",
            "materials_to_confirm": None,
            "indexed": False,
        }

    # ---- Phase: Bare Acts Review (user answered the follow-up question) ----
    elif phase == "bare_acts_review":
        # The user replied to the follow-up question shown after bare act presentation.
        # additional_info = user's answer; bare_acts = sections from Phase A (passed by frontend).
        # states = from Phase A response so case-law web search is jurisdiction-aware.
        facts = facts_summary or current_message
        additional_info = current_message
        stored_bare_acts = bare_acts or []
        return _run_final_with_case_laws(
            facts, stored_bare_acts, additional_info, progress_callback=progress_callback, states=states or [], step_callback=step_callback, token_callback=token_callback,
        )

    # ---- Phase: Confirm Index ----
    elif phase == "confirm_index":
        return _empty_result("done", facts_summary)

    return _empty_result()
