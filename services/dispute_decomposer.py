"""
Dispute Decomposer — breaks a composite legal query into distinct dispute components.

A single client query often contains multiple grievances (e.g. assault + eviction +
encroachment). Each grievance needs separate bare act and case law research because
different statutes and different reliefs apply.

This module produces a structured list of disputes that drives the per-dispute
retrieval loop in response_generator_v2.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def decompose_disputes(facts_summary: str, llm_fn=None) -> list:
    """
    Break a legal query into distinct dispute components via LLM.

    Returns a list of dispute dicts:
        [{"id": "d1", "dispute": "...", "legal_nature": "criminal|civil|both",
          "keywords": ["Act name", "legal term", ...]}, ...]

    Fallback: if LLM fails or returns unusable output, returns a single
    dispute covering the full query — retrieval always continues.

    Args:
        facts_summary: Plain-language description of the legal situation
        llm_fn: LLM callable (default: ollama_client.ask_llm)
    """
    if llm_fn is None:
        from llm.ollama_client import ask_llm
        llm_fn = ask_llm

    from prompts.advocate_prompts import DISPUTE_DECOMPOSITION_PROMPT

    prompt = DISPUTE_DECOMPOSITION_PROMPT.format(query=facts_summary[:2000])

    try:
        response = llm_fn(prompt)
        disputes = _parse_disputes(response)
        if disputes:
            # Ensure IDs are set and unique
            disputes = _normalise_ids(disputes)
            logger.info(
                "Dispute decomposition: %d component(s) — %s",
                len(disputes),
                [d.get("dispute", "")[:60] for d in disputes],
            )
            return disputes
    except Exception as e:
        logger.error("Dispute decomposition LLM call failed: %s", e)

    # Fallback: single dispute = full query
    logger.warning("Dispute decomposition failed; treating query as single dispute")
    return _single_dispute_fallback(facts_summary)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_disputes(response: str) -> list:
    """Extract and validate disputes list from LLM JSON response."""
    if not response:
        return []

    # Try direct parse
    try:
        data = json.loads(response.strip())
        return _extract_disputes_from_data(data)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in response
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        try:
            data = json.loads(match.group(0))
            return _extract_disputes_from_data(data)
        except json.JSONDecodeError:
            pass

    return []


def _extract_disputes_from_data(data: dict) -> list:
    """Validate and extract the disputes list from parsed JSON."""
    if not isinstance(data, dict):
        return []
    disputes = data.get("disputes", [])
    if not isinstance(disputes, list) or not disputes:
        return []

    valid = []
    for d in disputes:  # no cap — all disputes are captured
        if not isinstance(d, dict):
            continue
        dispute_text = (d.get("dispute") or "").strip()
        if not dispute_text:
            continue
        valid.append({
            "id": str(d.get("id", "")),
            "dispute": dispute_text,
            "legal_nature": str(d.get("legal_nature", "both")).lower(),
            "keywords": [str(k).strip() for k in d.get("keywords", []) if str(k).strip()],
        })

    return valid


def _normalise_ids(disputes: list) -> list:
    """Ensure every dispute has a unique id in the form d1, d2, ..."""
    for i, d in enumerate(disputes):
        if not d.get("id") or d["id"] in {x["id"] for x in disputes[:i]}:
            d["id"] = f"d{i + 1}"
    return disputes


def _single_dispute_fallback(facts_summary: str) -> list:
    """Return a single dispute object wrapping the full query."""
    return [{
        "id": "d1",
        "dispute": facts_summary[:300],
        "legal_nature": "both",
        "keywords": [],
    }]
