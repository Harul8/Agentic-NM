"""
Response Generator v2 — Full legal research pipeline with:
- Hybrid retrieval (FAISS + BM25 + cross-encoder re-ranking)
- LLM-driven sufficiency analysis (no hard numeric limits)
- Tiered internet search (official courts → legal portals → newspapers)
- Auto-enrichment (PDFs saved to Google Drive, indexed to vector store)
- Source tagging ([LOCAL_DB], [OFFICIAL_COURT], [LEGAL_PORTAL], [NEWS_REFERENCE])

This replaces the old response_generator.py's generate_response() function.
The old module is preserved for backward compatibility — this new module is
imported and used by the API server.
"""

import os
import json
import logging
from typing import Optional

from llm.ollama_client import ask_llm
from prompts.advocate_prompts import (
    EXPAND_LEGAL_QUERY_SYSTEM,
    EXTRACT_BARE_ACT_PORTIONS_SYSTEM,
    EXTRACT_CASE_PORTIONS_SYSTEM,
    RELEVANCE_EXPLANATION_SYSTEM,
    CONVERSATIONAL_SUMMARY_SYSTEM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Expansion (reused from v1, improved)
# ---------------------------------------------------------------------------

def expand_legal_query(facts: str) -> str:
    """Convert plain-language facts to a precise legal research query."""
    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}

FACTS:
{facts[:2000]}

Query:"""
    try:
        result = ask_llm(prompt).strip()[:500]
        return result or facts[:300]
    except Exception:
        return facts[:300]


# ---------------------------------------------------------------------------
# Extract relevant portions from fetched content
# ---------------------------------------------------------------------------

def extract_relevant_portions(facts: str, title: str, content: str, doc_type: str) -> str:
    """Use LLM to extract only the relevant portions of a document."""
    if doc_type == "bare_act":
        system = EXTRACT_BARE_ACT_PORTIONS_SYSTEM
    else:
        system = EXTRACT_CASE_PORTIONS_SYSTEM

    prompt = f"""{system}

CASE FACTS: {facts[:500]}
DOCUMENT TITLE: {title}
DOCUMENT CONTENT:
{content[:3000]}

Relevant portions:"""
    try:
        return ask_llm(prompt).strip()
    except Exception:
        return content[:1500]


# ---------------------------------------------------------------------------
# Main Response Generation Pipeline
# ---------------------------------------------------------------------------

def generate_response_v2(
    facts_summary: str,
    jurisdiction_state: str = "",
    intent: str = "legal_opinion",
    confirmed_materials: dict = None,
) -> dict:
    """
    Full legal research response using the v2 pipeline.

    Flow:
    1. Expand facts to legal search query
    2. Hybrid search local vector store (FAISS + BM25 + re-rank)
    3. LLM-driven sufficiency analysis
    4. If gaps found → tiered internet search → auto-enrich
    5. Generate explanation with source tags
    6. Return structured response

    Args:
        facts_summary: Plain language case facts
        jurisdiction_state: State for HC-specific searches (e.g., "Karnataka")
        intent: "legal_opinion", "search", or "lookup"
        confirmed_materials: Pre-confirmed materials to add (from user confirmation flow)

    Returns:
        {
            "bare_act_sections": [...],
            "case_laws": [...],
            "explanation": "...",
            "sufficiency": {...},
            "sources_used": [...],
            "needs_confirmation": False,
        }
    """
    from retrieval.hybrid_retriever import search_bare_acts_auto, search_case_laws_auto
    from retrieval.sufficiency_analyzer import (
        analyze_sufficiency,
        get_targeted_search_queries,
    )
    from retrieval.tiered_search import search_for_gaps, fetch_content_and_pdf
    from retrieval.auto_enricher import enrich_from_gap_results

    # Step 1: Expand query
    legal_query = expand_legal_query(facts_summary)
    search_query = f"{facts_summary} {legal_query}"[:500]
    logger.info(f"Expanded query: {legal_query[:200]}")

    # Step 2: Hybrid search local vector store
    logger.info("Searching local vector store (hybrid: FAISS + BM25 + re-rank)...")
    bare_acts = search_bare_acts_auto(search_query, top_k=30)
    case_laws = search_case_laws_auto(search_query, top_k=30)

    logger.info(f"Local results: {len(bare_acts)} bare act chunks, {len(case_laws)} case law chunks")

    # Add confirmed materials if provided
    if confirmed_materials:
        bare_acts, case_laws = _add_confirmed_materials(
            confirmed_materials, bare_acts, case_laws
        )

    # Step 3: Sufficiency analysis
    logger.info("Running sufficiency analysis...")
    sufficiency = analyze_sufficiency(facts_summary, bare_acts, case_laws)
    gaps = get_targeted_search_queries(sufficiency)

    # Step 4: If gaps found, search internet (tiered)
    enrichment_summary = None
    internet_bare_acts = []
    internet_case_laws = []

    if gaps and not sufficiency.get("overall_sufficient", False):
        logger.info(f"Gaps identified: {len(gaps)}. Searching internet (tiered)...")

        gap_results = search_for_gaps(gaps, jurisdiction_state)

        # Auto-enrich: download PDFs, save to Drive, index
        enrichment_summary = enrich_from_gap_results(gap_results)

        # Process internet results for display
        for enriched in enrichment_summary.get("enriched_bare_acts", []):
            content = enriched.get("content", "")
            if content:
                relevant = extract_relevant_portions(
                    facts_summary, enriched.get("title", ""), content, "bare_act"
                )
                internet_bare_acts.append({
                    "source": enriched.get("title", "Internet"),
                    "text": relevant or content[:1500],
                    "act_name": enriched.get("title", "Unknown"),
                    "url": enriched.get("url", ""),
                    "source_tag": enriched.get("source_tag", "LEGAL_PORTAL"),
                    "indexed": enriched.get("indexed", False),
                })

        for enriched in enrichment_summary.get("enriched_case_laws", []):
            content = enriched.get("content", "")
            if content:
                relevant = extract_relevant_portions(
                    facts_summary, enriched.get("title", ""), content, "case_law"
                )
                internet_case_laws.append({
                    "source": enriched.get("title", "Internet"),
                    "text": relevant or content[:1500],
                    "case_name": enriched.get("title", "Unknown"),
                    "url": enriched.get("url", ""),
                    "source_tag": enriched.get("source_tag", "LEGAL_PORTAL"),
                    "indexed": enriched.get("indexed", False),
                })

        logger.info(
            f"Internet enrichment: {len(internet_bare_acts)} bare acts, "
            f"{len(internet_case_laws)} case laws"
        )
    else:
        logger.info("Local results sufficient. Skipping internet search.")

    # Step 5: Combine all results
    all_bare_acts = _format_bare_acts(bare_acts + internet_bare_acts)
    all_case_laws = _format_case_laws(case_laws + internet_case_laws)

    # Handle intent-specific filtering
    if intent in ("search", "lookup"):
        all_bare_acts, all_case_laws = _filter_by_intent(
            facts_summary, intent, all_bare_acts, all_case_laws
        )

    # Step 6: Generate explanation
    if intent in ("search", "lookup"):
        explanation = _generate_conversational_summary(
            facts_summary, all_bare_acts, all_case_laws
        )
    else:
        explanation = _generate_legal_opinion(
            facts_summary, all_bare_acts, all_case_laws, sufficiency
        )

    if not (explanation or "").strip():
        explanation = "Here's what I found for your query. Below are the relevant legal provisions and case laws."

    # Collect source tags for transparency
    sources_used = set()
    for ba in all_bare_acts:
        sources_used.add(ba.get("source_tag", "LOCAL_DB"))
    for cl in all_case_laws:
        sources_used.add(cl.get("source_tag", "LOCAL_DB"))

    return {
        "bare_act_sections": all_bare_acts,
        "case_laws": all_case_laws,
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "needs_confirmation": False,
        "internet_case_laws": [],  # backward compat
    }


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def _format_bare_acts(bare_acts: list) -> list:
    """Format bare act results for display."""
    formatted = []
    seen = set()

    for ba in bare_acts:
        # Deduplicate by section number + act name
        key = f"{ba.get('act_name', '')}_{ba.get('section_number', '')}_{ba.get('source', '')}"
        if key in seen:
            continue
        seen.add(key)

        text = (
            ba.get("full_text")
            or ba.get("text")
            or ""
        ).strip()
        if not text or len(text) < 30:
            continue

        act_name = ba.get("act_name", "")
        section = ba.get("section_number", "")
        title = ba.get("section_title", "")

        display_title = act_name
        if section:
            display_title += f" — Section {section}"
        if title:
            display_title += f" ({title})"
        if not display_title:
            display_title = ba.get("source", "Unknown")

        formatted.append({
            "source": ba.get("source_file", ba.get("source", "")),
            "text": text,
            "act_name": act_name,
            "section_number": section,
            "title": display_title,
            "url": ba.get("url", ""),
            "source_tag": ba.get("source_tag", "LOCAL_DB"),
            "_rerank_score": ba.get("_rerank_score", 0),
        })

    # Sort by relevance score
    formatted.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    return formatted


def _format_case_laws(case_laws: list) -> list:
    """Format case law results for display."""
    formatted = []
    seen = set()

    for cl in case_laws:
        # Deduplicate by case name
        case_name = cl.get("case_name", cl.get("source", ""))
        key = f"{case_name}_{cl.get('paragraph_num', '')}"
        if key in seen:
            continue
        seen.add(key)

        text = (
            cl.get("full_text")
            or cl.get("text")
            or ""
        ).strip()
        if not text or len(text) < 30:
            continue

        court = cl.get("court", "")
        year = cl.get("year", "")
        citation = cl.get("citation", "")

        display_title = case_name or cl.get("source", "Unknown")
        if court:
            display_title += f" ({court}"
            if year:
                display_title += f", {year}"
            display_title += ")"
        elif citation:
            display_title += f" [{citation}]"

        formatted.append({
            "source": cl.get("source_file", cl.get("source", "")),
            "text": text,
            "case_name": case_name,
            "title": display_title,
            "court": court,
            "year": year,
            "citation": citation,
            "binding_authority": cl.get("binding_authority", ""),
            "url": cl.get("url", ""),
            "source_tag": cl.get("source_tag", "LOCAL_DB"),
            "_rerank_score": cl.get("_rerank_score", 0),
        })

    # Sort by: binding authority (SC first), then relevance score
    authority_order = {"supreme_court": 0, "high_court": 1, "tribunal": 2, "district_court": 3}
    formatted.sort(
        key=lambda x: (
            authority_order.get(x.get("binding_authority", ""), 5),
            -x.get("_rerank_score", 0),
        )
    )
    return formatted


def _filter_by_intent(facts: str, intent: str, bare_acts: list, case_laws: list):
    """Filter results based on user intent (search/lookup vs legal_opinion)."""
    query_lower = facts.lower()

    case_law_only = any(
        phrase in query_lower
        for phrase in ["case law", "case laws", "judgment", "judgments", "ruling", "pull", "find"]
    ) and not any(
        phrase in query_lower
        for phrase in ["bare act", "bare acts", "sections", "provisions"]
    )

    if case_law_only:
        return [], case_laws

    bare_act_only = any(
        phrase in query_lower
        for phrase in ["bare act", "bare acts", "section", "provisions", "act sections"]
    ) and not any(
        phrase in query_lower
        for phrase in ["case law", "judgment", "ruling"]
    )

    if bare_act_only:
        return bare_acts, []

    return bare_acts, case_laws


# ---------------------------------------------------------------------------
# Explanation Generation
# ---------------------------------------------------------------------------

def _generate_legal_opinion(
    facts: str,
    bare_acts: list,
    case_laws: list,
    sufficiency: dict,
) -> str:
    """Generate a formal legal opinion with citations and source tags."""
    bare_text = json.dumps(
        [{"title": b.get("title"), "text": b.get("text", "")[:500], "source_tag": b.get("source_tag")}
         for b in bare_acts[:15]],
        indent=2,
    )[:3000]

    case_text = json.dumps(
        [{"title": c.get("title"), "text": c.get("text", "")[:500], "source_tag": c.get("source_tag")}
         for c in case_laws[:15]],
        indent=2,
    )[:3000]

    confidence = sufficiency.get("confidence", "medium")

    prompt = f"""{RELEVANCE_EXPLANATION_SYSTEM}

CASE FACTS:
{facts[:1500]}

BARE ACT SECTIONS:
{bare_text}

CASE LAWS:
{case_text}

CONFIDENCE LEVEL: {confidence}

IMPORTANT: For each legal statement, cite the source. Tag each citation with its source type:
[LOCAL_DB] for materials from our verified database
[OFFICIAL_COURT] for materials from court websites
[LEGAL_PORTAL] for materials from legal portals
[NEWS_REFERENCE] for newspaper articles (context only)

Generate the legal analysis:"""

    try:
        return ask_llm(prompt).strip()
    except Exception as e:
        logger.error(f"Opinion generation failed: {e}")
        return ""


def _generate_conversational_summary(
    facts: str,
    bare_acts: list,
    case_laws: list,
) -> str:
    """Generate a conversational summary for search/lookup queries."""
    bare_text = json.dumps(
        [{"title": b.get("title"), "text": b.get("text", "")[:300]}
         for b in bare_acts[:10]],
        indent=2,
    )[:2000]

    case_text = json.dumps(
        [{"title": c.get("title"), "text": c.get("text", "")[:300]}
         for c in case_laws[:10]],
        indent=2,
    )[:2000]

    prompt = f"""{CONVERSATIONAL_SUMMARY_SYSTEM}

USER QUERY: {facts[:500]}
BARE ACTS FOUND: {bare_text}
CASE LAWS FOUND: {case_text}

Response:"""

    try:
        return ask_llm(prompt).strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Confirmed Materials (from user confirmation flow)
# ---------------------------------------------------------------------------

def _add_confirmed_materials(confirmed: dict, bare_acts: list, case_laws: list):
    """Add user-confirmed materials and index them."""
    from retrieval.auto_enricher import add_bare_act_chunks, add_case_law_chunks
    from Ingestion.smart_chunker import chunk_bare_act, chunk_case_law

    for b in confirmed.get("bare_acts", []):
        text = b.get("text", b.get("content", ""))
        title = b.get("title", "Confirmed")
        if text:
            chunks = chunk_bare_act(text, title)
            add_bare_act_chunks(chunks)
            bare_acts.append({
                "act_name": title,
                "text": text,
                "source": title,
                "source_tag": "LOCAL_DB",
            })

    for c in confirmed.get("case_laws", []):
        text = c.get("relevant_portion", c.get("content", c.get("text", "")))
        title = c.get("title", "Confirmed")
        if text:
            chunks = chunk_case_law(text, title)
            add_case_law_chunks(chunks)
            case_laws.append({
                "case_name": title,
                "text": text,
                "source": title,
                "source_tag": "LOCAL_DB",
            })

    return bare_acts, case_laws
