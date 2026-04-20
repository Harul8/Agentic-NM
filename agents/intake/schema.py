"""
agents/intake/schema.py — Formal contract for the IntakeState shared dict.

PURPOSE
-------
IntakeState is a plain dict that flows through every Stage 1 function.
Previously there was no single definition of its fields — keys were added
wherever needed, causing silent KeyErrors, wrong type assumptions, and
stage-coupling that was invisible until a page-refresh broke it.

This module provides:
  • IntakeState  — TypedDict with all fields and their types (for IDE + mypy)
  • INTAKE_DEFAULTS  — canonical default values for every field
  • make_intake_state()  — factory that returns a fresh, fully-populated state
  • coerce_intake_state(raw)  — fills missing keys, normalises types, warns on
    unknown keys; call at every stage boundary that receives an incoming state

FIELD GROUPS
------------
Category
  primary_issue_cluster    str | None   — primary detected legal category
  secondary_issue_clusters list[str]    — up to 2 secondary categories
  category_confidence      str | None   — "high" | "medium" | "low" | None

Anchor fields (all four required before Stage 2 transition)
  issue_summary            str | None   — what happened (core events)
  relationship_to_other_party  str | None  — spouse / employer / landlord / etc.
  timeframe_status         str | None   — ongoing / recent / historical / unknown
  client_goal_initial      str | None   — plain-language: what client wants

Urgency & risk
  urgency_signal           str | None   — immediate / near_term / no_urgency / unknown
  risk_flags               list[str]    — imminent_harm / active_arrest / child_at_risk / etc.
  immediate_need           str | None   — safety / shelter / protection_order / bail / money / none
  emotional_ask            str | None   — validation / information / action / unknown

Parties
  jurisdiction             str | None   — Indian state or "unknown"
  client_role              str | None   — victim / accused / claimant / etc.
  other_party              str | None   — brief description of opposite party

Facts & gaps
  known_facts              list         — structured fact dicts (see stage1_opening.py)
  open_questions           list[str]    — live unresolved gaps
  detail_request_issued    bool         — True after grouped detail request sent
  detail_groups_requested  list[str]    — information points requested from client
  missing_detail_groups    list[str]    — gaps remaining after reviewing client bundle
  followup_questions       list[str]    — short focused follow-up questions post gap-review
  deferred_questions       list[str]    — lower-priority gaps deferred to next turn
  analysis_ready           bool         — True when intake is sufficient for Stage 2

Summary
  latest_intake_summary    str | None   — latest concise matter summary

Case file (senior-advocate framing)
  case_file                dict         — structured case file (see make_case_file_template)

Remedy (Stage 4)
  stated_remedy            str | None   — verbatim what client asked for
  assessed_remedy          str | None   — system-assessed achievable relief
  remedy_detail            dict         — faster_alternative, remedy_gap, recommended_lead, etc.

Progress
  turn_count               int          — number of substantive client turns
  ready_for_stage2         bool         — True when all four anchor fields are populated
  stage                    str          — always "stage1"
"""
from __future__ import annotations

import logging
from typing import TypedDict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TypedDict — for IDE type-checking and documentation
# ---------------------------------------------------------------------------

class IntakeState(TypedDict, total=False):
    # Metadata
    stage: str

    # Category
    primary_issue_cluster: Optional[str]
    secondary_issue_clusters: list
    category_confidence: Optional[str]

    # Anchor fields
    issue_summary: Optional[str]
    relationship_to_other_party: Optional[str]
    timeframe_status: Optional[str]
    client_goal_initial: Optional[str]

    # Urgency & risk
    urgency_signal: Optional[str]
    risk_flags: list
    immediate_need: Optional[str]
    emotional_ask: Optional[str]

    # Parties
    jurisdiction: Optional[str]
    client_role: Optional[str]
    other_party: Optional[str]

    # Facts & gaps
    known_facts: list
    open_questions: list
    detail_request_issued: bool
    detail_groups_requested: list
    missing_detail_groups: list
    followup_questions: list
    deferred_questions: list
    analysis_ready: bool

    # Summary
    latest_intake_summary: Optional[str]

    # Case file
    case_file: dict

    # Remedy
    stated_remedy: Optional[str]
    assessed_remedy: Optional[str]
    remedy_detail: dict

    # Progress
    turn_count: int
    ready_for_stage2: bool


# ---------------------------------------------------------------------------
# INTAKE_DEFAULTS — canonical defaults for every field
# ---------------------------------------------------------------------------

INTAKE_DEFAULTS: dict = {
    "stage": "stage1",
    # Category
    "primary_issue_cluster": None,
    "secondary_issue_clusters": [],
    "category_confidence": None,
    # Anchor fields
    "issue_summary": None,
    "relationship_to_other_party": None,
    "timeframe_status": None,
    "client_goal_initial": None,
    # Urgency & risk
    "urgency_signal": None,
    "risk_flags": [],
    "immediate_need": None,
    "emotional_ask": None,
    # Parties
    "jurisdiction": None,
    "client_role": None,
    "other_party": None,
    # Facts & gaps
    "known_facts": [],
    "open_questions": [],
    "detail_request_issued": False,
    "detail_groups_requested": [],
    "missing_detail_groups": [],
    "followup_questions": [],
    "deferred_questions": [],
    "analysis_ready": False,
    # Summary
    "latest_intake_summary": None,
    # Case file (populated by make_case_file_template in prompts/intake.py)
    "case_file": {},
    # Remedy
    "stated_remedy": None,
    "assessed_remedy": None,
    "remedy_detail": {},
    # Progress
    "turn_count": 0,
    "ready_for_stage2": False,
}

# Fields that must be list (not None, not str, not dict)
_LIST_FIELDS: frozenset[str] = frozenset({
    "secondary_issue_clusters",
    "risk_flags",
    "known_facts",
    "open_questions",
    "detail_groups_requested",
    "missing_detail_groups",
    "followup_questions",
    "deferred_questions",
})

# Fields that must be bool
_BOOL_FIELDS: frozenset[str] = frozenset({
    "detail_request_issued",
    "analysis_ready",
    "ready_for_stage2",
})

# Fields that must be int
_INT_FIELDS: frozenset[str] = frozenset({
    "turn_count",
})

# Known field names — used for unknown-key warnings in development
_KNOWN_FIELDS: frozenset[str] = frozenset(INTAKE_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# make_intake_state — factory
# ---------------------------------------------------------------------------

def make_intake_state() -> dict:
    """Return a fresh intake state dict with all fields set to their defaults."""
    state = {}
    for key, default in INTAKE_DEFAULTS.items():
        if isinstance(default, (list, dict)):
            state[key] = type(default)()
        else:
            state[key] = default
    return state


# ---------------------------------------------------------------------------
# coerce_intake_state — normalise an incoming state at stage boundaries
# ---------------------------------------------------------------------------

def coerce_intake_state(raw: dict | None) -> dict:
    """
    Ensure every known field exists and has the right type.

    - Missing fields: filled from INTAKE_DEFAULTS
    - List fields that are None/str/other: reset to []
    - Bool fields that are non-bool: coerced with bool()
    - Int fields that are non-int: coerced with int() (default 0 on failure)
    - dict fields (case_file, remedy_detail) that are non-dict: reset to {}
    - Unknown keys: logged at DEBUG level (helps catch typos across stages)
    - Existing valid values: preserved unchanged

    Call this at the top of every stage function that receives an incoming
    intake_state from an external source (API boundary, page refresh, etc.).
    """
    if not isinstance(raw, dict):
        logger.debug("coerce_intake_state: received non-dict (%r) — starting fresh", type(raw).__name__)
        return make_intake_state()

    state = dict(raw)  # shallow copy — do not mutate caller's dict

    # Fill missing keys
    for key, default in INTAKE_DEFAULTS.items():
        if key not in state:
            state[key] = type(default)() if isinstance(default, (list, dict)) else default

    # Normalise list fields
    for key in _LIST_FIELDS:
        val = state.get(key)
        if not isinstance(val, list):
            if val is not None:
                logger.debug("coerce_intake_state: field %r expected list, got %r — resetting to []", key, type(val).__name__)
            state[key] = []

    # Normalise bool fields
    for key in _BOOL_FIELDS:
        val = state.get(key)
        if not isinstance(val, bool):
            state[key] = bool(val)

    # Normalise int fields
    for key in _INT_FIELDS:
        val = state.get(key)
        if not isinstance(val, int):
            try:
                state[key] = int(val)
            except (TypeError, ValueError):
                state[key] = 0

    # Normalise dict fields
    for key in ("case_file", "remedy_detail"):
        val = state.get(key)
        if not isinstance(val, dict):
            state[key] = {}

    # Warn on unknown keys (development-time catch for typos across stages)
    unknown = set(state.keys()) - _KNOWN_FIELDS
    if unknown:
        logger.debug("coerce_intake_state: unknown fields %s — keeping as-is", sorted(unknown))

    return state


# ---------------------------------------------------------------------------
# anchor_fields_present — readiness check
# ---------------------------------------------------------------------------

_BLANK_SENTINELS: frozenset[str] = frozenset({"unknown", "null", "none", ""})


def anchor_fields_present(state: dict) -> bool:
    """
    Return True iff all four anchor fields are populated with meaningful values.

    Used by the Stage 1→2 gate in pipeline/chat.py.
    Extracted here so the check is defined alongside the schema.
    """
    anchors = (
        str(state.get("issue_summary") or "").strip(),
        str(state.get("relationship_to_other_party") or "").strip(),
        str(state.get("timeframe_status") or "").strip(),
        str(state.get("client_goal_initial") or "").strip(),
    )
    return all(v and v.lower() not in _BLANK_SENTINELS for v in anchors)
