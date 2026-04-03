from __future__ import annotations

import copy
import json
import logging

from prompts.advocate_prompts import (
    ENTRY_ROUTER_OPENING_SYSTEM,
    ENTRY_ROUTER_INTENT_SYSTEM,
    ENTRY_ROUTER_LOOKUP_CONFIRMATION_SYSTEM,
    ENTRY_ROUTER_GENERAL_REPLY_SYSTEM,
    ENTRY_ROUTER_STATE_SCHEMA,
)

logger = logging.getLogger(__name__)


def _ask_llm(prompt: str, task_hint: str = "fast", model_override: str | None = None) -> str:
    from llm.ollama_client import ask_llm
    return ask_llm(prompt, task_hint=task_hint, model=model_override) or ""


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _build_context(session: dict, max_turns: int = 8) -> str:
    hist = session.get("history") or []
    lines = []
    for turn in hist[-max_turns:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:240]}")
    return "\n".join(lines)


def new_entry_state() -> dict:
    return copy.deepcopy(ENTRY_ROUTER_STATE_SCHEMA)


def generate_entry_opening(model_override: str | None = None) -> str:
    try:
        msg = _ask_llm(ENTRY_ROUTER_OPENING_SYSTEM, task_hint="quality", model_override=model_override).strip()
        if msg:
            return msg
    except Exception as exc:
        logger.warning("Entry opening generation failed: %s", exc)
    return "Tell me what you need — I can do a quick legal lookup or a full legal-opinion workflow for your case."


def _classify_route(user_message: str, conversation_context: str, model_override: str | None = None) -> str:
    prompt = (
        ENTRY_ROUTER_INTENT_SYSTEM
        .replace("{user_message}", (user_message or "").strip())
        .replace("{conversation_context}", conversation_context or "(none)")
    )
    try:
        out = _extract_json(_ask_llm(prompt, task_hint="fast", model_override=model_override))
        route = str((out or {}).get("route") or "").strip().lower()
        if route in {"legal_opinion_workflow", "quick_legal_lookup", "general_non_legal"}:
            return route
    except Exception:
        pass

    low = (user_message or "").lower()
    if any(k in low for k in ("section", "provision", "act", "ipc", "bnss", "bns", "lookup", "quick")):
        return "quick_legal_lookup"
    # High-signal legal questions that usually want provisions first.
    if any(k in low for k in (
        "punishment", "penalty", "liable", "what happens", "what is the punishment",
        "harass", "harassing", "harassment", "cruelty", "abuse",
        "husband", "in-law", "in-laws", "inlaws", "relatives", "dowry",
        "molestation", "assault", "threat",
        "f.i.r", "fir", "complaint",
    )):
        return "quick_legal_lookup"
    if any(k in low for k in ("draft", "case", "petition", "notice", "opinion", "legal advice")):
        return "legal_opinion_workflow"
    return "general_non_legal"


def _parse_lookup_confirmation(user_message: str, model_override: str | None = None) -> tuple[str, str]:
    prompt = ENTRY_ROUTER_LOOKUP_CONFIRMATION_SYSTEM.replace("{user_message}", (user_message or "").strip())
    try:
        out = _extract_json(_ask_llm(prompt, task_hint="fast", model_override=model_override))
        decision = str((out or {}).get("decision") or "").strip().lower()
        extra = str((out or {}).get("extra_context") or "").strip()
        if decision in {"proceed_lookup", "add_more", "route_legal_opinion"}:
            return decision, extra
    except Exception:
        pass
    low = (user_message or "").lower()
    if "just information" in low or "information about the topic" in low:
        return "proceed_lookup", ""
    if "legal support" in low:
        return "route_legal_opinion", ""
    if any(k in low for k in ("proceed", "go ahead", "yes", "continue", "do it")):
        return "proceed_lookup", ""
    if any(k in low for k in ("full opinion", "draft", "complete case", "legal opinion")):
        return "route_legal_opinion", ""
    return "add_more", (user_message or "").strip()


def _parse_web_confirmation(user_message: str) -> str:
    low = (user_message or "").strip().lower()
    if any(k in low for k in ("yes", "yeah", "yep", "ok", "okay", "go ahead", "search web", "web")):
        return "proceed_web"
    if any(k in low for k in ("no", "not now", "skip", "local only")):
        return "skip_web"
    if any(k in low for k in ("full opinion", "legal opinion", "draft", "complete case")):
        return "route_legal_opinion"
    return "unclear"


def process_turn(session: dict, user_message: str, model_override: str | None = None) -> dict:
    state = session.get("entry_state") if isinstance(session.get("entry_state"), dict) else new_entry_state()
    context = _build_context(session)
    msg = (user_message or "").strip()

    # If awaiting quick-lookup confirmation, parse that first
    if state.get("pending_lookup_confirmation"):
        decision, extra = _parse_lookup_confirmation(msg, model_override=model_override)
        topic = str(state.get("lookup_topic") or "").strip()
        if decision == "route_legal_opinion":
            state["pending_lookup_confirmation"] = False
            return {
                "reply": "Understood. I will route this into the full legal-opinion workflow and start detailed intake now.",
                "entry_state": state,
                "route_to": "legal_opinion_workflow",
                "lookup_query": None,
            }
        if decision == "proceed_lookup":
            query = f"{topic}. {extra}".strip(". ").strip() if extra else topic
            state["pending_lookup_confirmation"] = False
            return {
                "reply": "Great — I’m pulling the relevant legal provisions now.",
                "entry_state": state,
                "route_to": "quick_legal_lookup",
                "lookup_query": query or msg,
            }
        # add_more
        merged = f"{topic}. {extra}".strip(". ").strip()
        state["lookup_topic"] = merged or topic or msg
        return {
            "reply": "What are you looking for?",
            "entry_options": [
                {"id": "just_information", "label": "Just information about the topic"},
                {"id": "legal_support", "label": "Legal support"},
            ],
            "entry_state": state,
            "route_to": "entry_router",
            "lookup_query": None,
        }

    if state.get("pending_web_confirmation"):
        decision = _parse_web_confirmation(msg)
        if decision == "proceed_web":
            state["pending_web_confirmation"] = False
            return {
                "reply": "Understood — I’ll run a web search now to find additional relevant materials.",
                "entry_state": state,
                "route_to": "quick_legal_lookup_web",
                "lookup_query": state.get("lookup_topic") or msg,
            }
        if decision == "route_legal_opinion":
            state["pending_web_confirmation"] = False
            return {
                "reply": "Understood. I’ll move this into the full legal-opinion workflow and begin structured intake.",
                "entry_state": state,
                "route_to": "legal_opinion_workflow",
                "lookup_query": None,
            }
        if decision == "skip_web":
            state["pending_web_confirmation"] = False
            return {
                "reply": "No problem. We’ll keep it to local materials only. If you later want web-backed references, just ask.",
                "entry_state": state,
                "route_to": "entry_router",
                "lookup_query": None,
            }
        return {
            "reply": "Would you like me to search the web for additional legal materials now, or keep it local-only?",
            "entry_state": state,
            "route_to": "entry_router",
            "lookup_query": None,
        }

    route = _classify_route(msg, context, model_override=model_override)
    state["route"] = route

    if route == "legal_opinion_workflow":
        return {
            "reply": "Understood — we’ll do the full legal-opinion workflow. I’ll begin structured intake now.",
            "entry_state": state,
            "route_to": "legal_opinion_workflow",
            "lookup_query": None,
        }

    if route == "quick_legal_lookup":
        state["pending_lookup_confirmation"] = True
        state["lookup_topic"] = msg
        return {
            "reply": "What are you looking for?",
            "entry_options": [
                {"id": "just_information", "label": "Just information about the topic"},
                {"id": "legal_support", "label": "Legal support"},
            ],
            "entry_state": state,
            "route_to": "entry_router",
            "lookup_query": None,
        }

    # general_non_legal
    try:
        prompt = ENTRY_ROUTER_GENERAL_REPLY_SYSTEM.replace("{user_message}", msg)
        reply = _ask_llm(prompt, task_hint="fast", model_override=model_override).strip()
        if reply:
            return {
                "reply": reply,
                "entry_state": state,
                "route_to": "entry_router",
                "lookup_query": None,
            }
    except Exception:
        pass
    return {
        "reply": "Here’s a quick answer based on general knowledge. I am a legal AI agent and specialise in Indian legal matters.",
        "entry_state": state,
        "route_to": "entry_router",
        "lookup_query": None,
    }

