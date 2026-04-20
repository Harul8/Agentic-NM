"""
pipeline/mode.py — Authoritative contract for pipeline mode flags.

PURPOSE
-------
Four flags thread through the entire chat pipeline but their interactions were
previously undocumented and only discoverable by reading chat.py + generator.py:

  intent             "legal_opinion" | "search" | "lookup" | "bulk_ingest" | "generic_chat"
  search_strategy    "local_only"    | "local_then_web"
  analysis_mode      "full_opinion"  | "bare_acts_only" | "precedents_only"
  interactive_fast_path  bool — derived, never set directly by callers

This module provides:
  • Named constants for valid values
  • PipelineMode  — typed container holding all resolved flags
  • resolve_pipeline_mode()  — single function that validates inputs, applies
    defaults, and computes interactive_fast_path deterministically

INTERACTION RULES (previously implicit, now explicit)
------------------------------------------------------
1. intent="legal_opinion"
     search_strategy defaults to "local_only" (fast local index).
     analysis_mode is meaningful ("full_opinion" / "bare_acts_only" / "precedents_only").
     interactive_fast_path may be True (see rule 4).

2. intent in ("search", "lookup")
     search_strategy defaults to "local_then_web" (research flow needs broader coverage).
     analysis_mode is ignored (these intents produce retrieval results, not opinions).
     interactive_fast_path is always False.

3. intent in ("bulk_ingest", "generic_chat")
     Neither search_strategy nor analysis_mode is meaningful.
     interactive_fast_path is always False.

4. interactive_fast_path = True  iff  ALL of:
     - retrieval_only is False
     - ENABLE_INTERACTIVE_FAST_PATH env var is "1" / "true" / "yes"
     - intent == "legal_opinion"
     - search_strategy == "local_only"

   When True: single dispute path, live token streaming, LLM opinion on first
   available local sections.  When False: full multi-dispute parallel retrieval.

5. "web_only" search_strategy is deprecated; it is silently promoted to
   "local_then_web" to preserve backwards-compatibility.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid value sets
# ---------------------------------------------------------------------------

VALID_INTENTS: frozenset[str] = frozenset({
    "legal_opinion",
    "search",
    "lookup",
    "bulk_ingest",
    "generic_chat",
})

VALID_SEARCH_STRATEGIES: frozenset[str] = frozenset({
    "local_only",
    "local_then_web",
})

VALID_ANALYSIS_MODES: frozenset[str] = frozenset({
    "full_opinion",
    "bare_acts_only",
    "precedents_only",
})

# Intents that produce a legal opinion (analysis_mode and fast path apply)
_OPINION_INTENTS: frozenset[str] = frozenset({"legal_opinion"})

# Intents that use the research / retrieval path (broader search strategy)
_RESEARCH_INTENTS: frozenset[str] = frozenset({"search", "lookup"})

# ---------------------------------------------------------------------------
# PipelineMode — resolved, validated mode for one pipeline invocation
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PipelineMode:
    """
    Resolved pipeline mode flags for a single process_chat / generate_response_v2
    invocation.  All fields are validated and have correct defaults applied.

    Do not construct directly — use resolve_pipeline_mode().
    """
    intent: str
    search_strategy: str
    analysis_mode: str
    interactive_fast_path: bool
    retrieval_only: bool

    # Human-readable log string for debugging
    summary: str = field(compare=False)

    def is_opinion(self) -> bool:
        return self.intent in _OPINION_INTENTS

    def is_research(self) -> bool:
        return self.intent in _RESEARCH_INTENTS


# ---------------------------------------------------------------------------
# resolve_pipeline_mode — the single source of truth
# ---------------------------------------------------------------------------

def resolve_pipeline_mode(
    intent: str | None = None,
    search_strategy: str | None = None,
    analysis_mode: str | None = None,
    retrieval_only: bool = False,
    *,
    _env_fast_path: bool | None = None,  # override for tests; None reads env
) -> PipelineMode:
    """
    Validate and resolve all pipeline mode flags.

    Parameters
    ----------
    intent           : raw intent string from caller (may be None / invalid)
    search_strategy  : raw strategy string (may be None / "web_only" / invalid)
    analysis_mode    : raw analysis mode string (may be None / invalid)
    retrieval_only   : True → skip opinion synthesis, return retrieval cache only
    _env_fast_path   : test override for ENABLE_INTERACTIVE_FAST_PATH env var

    Returns
    -------
    PipelineMode with all fields validated, defaults applied, and
    interactive_fast_path computed deterministically.
    """
    # --- Resolve intent ---
    resolved_intent = (intent or "").strip().lower() or "legal_opinion"
    if resolved_intent not in VALID_INTENTS:
        logger.warning(
            "resolve_pipeline_mode: unknown intent %r — defaulting to 'legal_opinion'",
            resolved_intent,
        )
        resolved_intent = "legal_opinion"

    # --- Resolve search_strategy ---
    raw_strategy = (search_strategy or "").strip().lower()
    if raw_strategy == "web_only":
        logger.info("resolve_pipeline_mode: 'web_only' is deprecated — using 'local_then_web'")
        raw_strategy = "local_then_web"

    if raw_strategy in VALID_SEARCH_STRATEGIES:
        resolved_strategy = raw_strategy
    else:
        # Apply intent-based default (rule 1 / 2 above)
        if resolved_intent in _OPINION_INTENTS:
            resolved_strategy = "local_only"
        else:
            resolved_strategy = "local_then_web"

    # --- Resolve analysis_mode ---
    raw_mode = (analysis_mode or "").strip().lower() or "full_opinion"
    if raw_mode not in VALID_ANALYSIS_MODES:
        logger.warning(
            "resolve_pipeline_mode: unknown analysis_mode %r — defaulting to 'full_opinion'",
            raw_mode,
        )
        raw_mode = "full_opinion"
    # analysis_mode only meaningful for opinion intents
    resolved_analysis_mode = raw_mode if resolved_intent in _OPINION_INTENTS else "full_opinion"

    # --- Compute interactive_fast_path (rule 4) ---
    if _env_fast_path is None:
        env_flag = os.environ.get("ENABLE_INTERACTIVE_FAST_PATH", "1").lower() in ("1", "true", "yes")
    else:
        env_flag = _env_fast_path

    interactive_fast_path = (
        not retrieval_only
        and env_flag
        and resolved_intent == "legal_opinion"
        and resolved_strategy == "local_only"
    )

    summary = (
        f"intent={resolved_intent} strategy={resolved_strategy} "
        f"analysis={resolved_analysis_mode} fast={interactive_fast_path} "
        f"retrieval_only={retrieval_only}"
    )

    return PipelineMode(
        intent=resolved_intent,
        search_strategy=resolved_strategy,
        analysis_mode=resolved_analysis_mode,
        interactive_fast_path=interactive_fast_path,
        retrieval_only=retrieval_only,
        summary=summary,
    )
