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
    """Extract and validate disputes list from LLM JSON response.

    Tries multiple extraction strategies to handle common LLM output variations:
      1. Direct JSON parse (clean output)
      2. Strip markdown fences (```json ... ```)
      3. Largest {...} object in the response (LLM preamble before JSON)
      4. Largest [...] array in the response (LLM returns array directly)
    """
    if not response:
        return []

    # 1. Direct parse
    stripped = response.strip()
    try:
        data = json.loads(stripped)
        result = _extract_disputes_from_data(data)
        if result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            result = _extract_disputes_from_data(data)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Find the outermost {...} object block
    obj_match = re.search(r"\{[\s\S]*\}", stripped)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            result = _extract_disputes_from_data(data)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Find a [...] array block (LLM sometimes returns the disputes array directly)
    arr_match = re.search(r"\[[\s\S]*\]", stripped)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            if isinstance(arr, list):
                # Treat as if it were {"disputes": [...]}
                result = _extract_disputes_from_data({"disputes": arr})
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.debug("Dispute decomposer: could not extract valid JSON from LLM response (len=%d)", len(response))
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
            # New fields from improved decomposition prompt (may be absent in old LLM output)
            "bare_act_hints": [str(h).strip() for h in d.get("bare_act_hints", []) if str(h).strip()],
            "search_angles": [str(s).strip() for s in d.get("search_angles", []) if str(s).strip()],
        })

    return valid


def _normalise_ids(disputes: list) -> list:
    """Ensure every dispute has a unique id in the form d1, d2, ..."""
    for i, d in enumerate(disputes):
        if not d.get("id") or d["id"] in {x["id"] for x in disputes[:i]}:
            d["id"] = f"d{i + 1}"
    return disputes


def _single_dispute_fallback(facts_summary: str) -> list:
    """Return a single dispute object wrapping the full query.

    Ensures all fields that downstream code expects are present so the fallback
    dispute dict has the same shape as a fully decomposed one.
    """
    return [{
        "id": "d1",
        "dispute": facts_summary[:300],
        "legal_nature": "both",
        "keywords": [],       # _build_bare_act_queries will use Q5 LLM fallback to generate
        "bare_act_hints": [], # never inject LLM-knowledge act names
        "search_angles": [],  # empty → _web_search_bare_acts uses round1_queries
    }]
