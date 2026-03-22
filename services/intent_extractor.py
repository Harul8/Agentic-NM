"""
Intent Extractor — Extracts structured research intent from the user's message.

Single source of truth for model-driven behavior: states, domains, topics, scope,
document_types, search_strategy, result_count. Used by fact_collector and
response_generator; static/keyword logic only as fallback when extraction fails.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXTRACT_RESEARCH_INTENT_PROMPT = """You are an Indian legal research assistant. From the user's message, extract structured intent for legal research. Output ONLY what the user actually mentioned or clearly implied. Do NOT add states, domains, or topics they did not ask for.

USER MESSAGE:
{user_message}

Extract and output valid JSON only (no preamble):
{{
  "states": ["list of Indian states or jurisdictions mentioned"],
  "domains": ["legal domains or areas they asked for"],
  "topics": ["specific matters or subjects"],
  "scope": "broad or specific",
  "central_or_state": "both or central or state",
  "document_types": "acts_only or case_laws_only or both",
  "search_strategy": "local_only or local_then_web",
  "result_count": null or integer
}}

RULES:
- states: only states/regions the user named. Empty list if none. Do NOT include "India" in states—India = Central.
- domains: only legal areas they asked for. Empty list if none.
- topics: specific subjects. Empty if none.
- scope: "broad" if they want "all acts", "list all", "every act", "pull all"; "specific" otherwise.
- central_or_state: "both" if they mention BOTH India/Central AND a state; "central" if only India/Central; "state" if only state acts.
- document_types: "acts_only" if they want only acts/laws (no judgments)—e.g. "only acts", "bare acts", "laws by government", "no case laws". "case_laws_only" if they want only judgments—e.g. "only case laws", "only judgments", "no acts". "both" otherwise.
- search_strategy: "local_only" if they say only local, no web, local database only. "local_then_web" otherwise.
- "local_then_web" means: search the local vector store first, and use Indiankanoon only if the local store has no relevant result for a material type.
- result_count: set to an integer ONLY if they explicitly asked for a number (e.g. "5 case laws", "top 10", "three judgments")—use that number, cap at 30. Set to null if they did not specify a count (then the system will show all highly relevant results).
- Use only what the user said. Do not infer or add examples."""


def extract_research_intent(user_message: str, llm_fn=None) -> dict[str, Any]:
    """
    Extract structured intent from the user's message. Single source of truth for
    model-driven behavior; static/keyword logic used only as fallback when this fails.

    Returns:
        dict with keys: states, domains, topics, scope, central_or_state,
        document_types ("acts_only"|"case_laws_only"|"both"),
        search_strategy ("local_only"|"local_then_web"),
        result_count (int or None; None = not specified → pipeline uses flexible limit).
        On failure returns safe defaults.
    """
    if llm_fn is None:
        from llm.ollama_client import ask_llm
        llm_fn = lambda prompt: ask_llm(prompt, task_hint="fast")

    default = {
        "states": [],
        "domains": [],
        "topics": [],
        "scope": "specific",
        "central_or_state": "both",
        "document_types": "both",
        "search_strategy": "local_then_web",
        "result_count": None,
    }

    if not (user_message or "").strip():
        return default

    try:
        prompt = EXTRACT_RESEARCH_INTENT_PROMPT.format(
            user_message=(user_message or "")[:2000].strip()
        )
        response = (llm_fn(prompt) or "").strip()
        # Try to parse JSON from response (may be wrapped in markdown or text)
        text = response
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if "{" in part and "}" in part:
                    start, end = part.find("{"), part.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = part[start:end]
                        break
        if "{" in text and "}" in text:
            start, end = text.find("{"), text.rfind("}") + 1
            out = json.loads(text[start:end])
        else:
            out = json.loads(text)

        if not isinstance(out, dict):
            return default

        states = out.get("states")
        domains = out.get("domains")
        topics = out.get("topics")
        scope = out.get("scope", "specific")
        central_or_state = out.get("central_or_state", "both")
        document_types = out.get("document_types", "both")
        search_strategy = out.get("search_strategy", "local_then_web")
        result_count = out.get("result_count")

        if not isinstance(states, list):
            states = []
        if not isinstance(domains, list):
            domains = []
        if not isinstance(topics, list):
            topics = []

        states = [str(s).strip() for s in states if s]
        domains = [str(d).strip() for d in domains if d]
        topics = [str(t).strip() for t in topics if t]
        if scope not in ("broad", "specific"):
            scope = "specific"
        if central_or_state not in ("both", "central", "state"):
            central_or_state = "both"
        if document_types not in ("acts_only", "case_laws_only", "both"):
            document_types = "both"
        if search_strategy not in ("local_only", "local_then_web"):
            search_strategy = "local_then_web"
        if result_count is not None:
            try:
                n = int(result_count)
                result_count = max(1, min(n, 30)) if n > 0 else None
            except (TypeError, ValueError):
                result_count = None

        result = {
            "states": states,
            "domains": domains,
            "topics": topics,
            "scope": scope,
            "central_or_state": central_or_state,
            "document_types": document_types,
            "search_strategy": search_strategy,
            "result_count": result_count,
        }
        logger.info(
            "Extracted intent: document_types=%s search_strategy=%s result_count=%s states=%s scope=%s",
            document_types, search_strategy, result_count, states, scope,
        )
        return result
    except Exception as e:
        logger.warning("Intent extraction failed: %s", e)
        return default
