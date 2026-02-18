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
    CASE_SUMMARY_SYSTEM,
    RELEVANCE_EXPLANATION_SYSTEM,
    CONVERSATIONAL_SUMMARY_SYSTEM,
)
from services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# Relevance and quality thresholds — keep only high-quality, recent materials
MIN_RERANK_SCORE = 3.0  # Minimum to include in pool (lowered from 4.0 to allow more local results)
HIGH_QUALITY_SCORE = 5.0  # Case laws with score > this: "highly relevant"; stop web search if we have 5+; index web PDFs only if > this
WEB_MIN_SCORE = 2.0  # Minimum rerank score to include a web-sourced case law (lowered to allow more results after full PDF scoring)
TARGET_HIGH_QUALITY_CASE_LAWS = 5  # Stop search once we have this many case laws with score > HIGH_QUALITY_SCORE
MIN_CASE_YEAR = 1975 # Prefer cases from last 40 years; older treated as low priority
JUNK_CASE_PATTERNS = ("unknown", "unknown (", "high court, 1908", "high court, 1972")


def _is_quality_case_law(chunk: dict) -> bool:
    """
    Reject junk case law chunks: template text, very old (when year known).
    Allow results with real judgment text; use source/source_file as display if case_name missing.
    """
    case_name = (chunk.get("case_name") or chunk.get("source") or chunk.get("source_file") or "").strip()
    # Allow chunks that have at least a source/file name for display (not strictly require case_name)
    if not case_name:
        return False
    name_lower = case_name.lower()
    if name_lower == "unknown":
        return False
    for bad in JUNK_CASE_PATTERNS:
        if bad in name_lower:
            return False
    # Only reject when year is present and too old; missing year is OK (we sort by year preference)
    year_raw = chunk.get("year")
    if year_raw is not None and year_raw != "":
        try:
            y = int(str(year_raw).strip()[:4])
            if y < MIN_CASE_YEAR:
                return False
        except (ValueError, TypeError):
            pass
    text = (chunk.get("full_text") or chunk.get("text") or "").strip()
    if len(text) < 50:
        return False
    return True


def _is_quality_bare_act(chunk: dict) -> bool:
    """Reject bare act chunks without proper act/section metadata."""
    act_name = (chunk.get("act_name") or "").strip()
    if not act_name or act_name.lower() == "unknown":
        return False
    section = (chunk.get("section_number") or "").strip()
    if not section:
        return False
    text = (chunk.get("full_text") or chunk.get("text") or "").strip()
    return len(text) >= 50


def _case_year_for_sort(chunk: dict) -> int:
    """Return year for sorting; 0 = unknown, use for 'prefer recent' ordering."""
    raw = chunk.get("year")
    if raw is None or raw == "":
        return 0
    try:
        return int(str(raw).strip()[:4])
    except (ValueError, TypeError):
        return 0


# Court name → display acronym for case title (e.g. [SC] Appellant v/s Respondent)
COURT_ACRONYMS = {
    "supreme_court": "[SC]",
    "supreme court": "[SC]",
    "high_court": "[HC]",
    "high court": "[HC]",
    "sessions court": "[Sess. Ct.]",
    "sessions court of": "[Sess. Ct.]",
    "civil court": "[Civil Ct.]",
    "district court": "[DC]",
    "tribunal": "[Trib.]",
    "nclt": "[NCLT]",
    "nclat": "[NCLAT]",
    "consumer": "[Consumer]",
    "family court": "[Family Ct.]",
}


def _court_acronym(court: str, binding_authority: str, case_name_or_source: str = "") -> str:
    """Return court acronym for display: [SC], [HC], [Sess. Ct.], [Civil Ct.], [DC], [Trib.], etc."""
    combined = " ".join(
        filter(None, [
            (binding_authority or "").strip().lower(),
            (court or "").strip().lower(),
            (case_name_or_source or "").strip().lower(),
        ])
    )
    for key, acronym in COURT_ACRONYMS.items():
        if key in combined:
            return acronym
    if "supreme" in combined:
        return "[SC]"
    if "high court" in combined:
        return "[HC]"
    if "session" in combined:
        return "[Sess. Ct.]"
    if "civil" in combined and "court" in combined:
        return "[Civil Ct.]"
    if "district" in combined:
        return "[DC]"
    if "tribunal" in combined:
        return "[Trib.]"
    # Fallback: "State of X" or "v/s State of" in case name usually indicates state/High Court litigation
    if "state of " in combined or " v/s state" in combined or " vs state" in combined:
        return "[HC]"
    return "[Court]"


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
    result_count: int = None,
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

    # Initialize progress tracker
    progress = ProgressTracker()

    # Step 1: Expand query
    legal_query = expand_legal_query(facts_summary)
    search_query = f"{facts_summary} {legal_query}"[:500]
    logger.info(f"Expanded query: {legal_query[:200]}")

    # Step 2: Hybrid search local vector store
    progress.start_group("Internal Search", "Searching local vector database")
    progress.add_step("Searching local vector store (FAISS + BM25 + re-ranking)...")
    logger.info("Searching local vector store (hybrid: FAISS + BM25 + re-rank)...")
    bare_acts = search_bare_acts_auto(search_query, top_k=30)
    case_laws = search_case_laws_auto(search_query, top_k=30)

    logger.info(f"Local results (raw): {len(bare_acts)} bare act chunks, {len(case_laws)} case law chunks")
    progress.add_step(f"Found {len(case_laws)} case law documents in local database", {"total_found": len(case_laws)})

    # Track ALL document scans (before filtering) - this shows what was actually scanned
    progress.add_step("Scanning case law documents for similarity...")
    all_case_laws_scanned = case_laws.copy()  # Keep original list for tracking
    for cl in all_case_laws_scanned:
        score = cl.get("_rerank_score", 0)
        doc_name = cl.get("case_name") or cl.get("source", "Unknown")
        included = score >= MIN_RERANK_SCORE
        progress.add_document_scan(doc_name, score, included, threshold=MIN_RERANK_SCORE)

    # Filter by relevance then by quality — no junk or very old cases
    progress.add_step(f"Filtering documents (relevance score >= {MIN_RERANK_SCORE})...")
    bare_acts_before = len(bare_acts)
    case_laws_before = len(case_laws)
    
    bare_acts = [ba for ba in bare_acts if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE]
    case_laws = [cl for cl in case_laws if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE]
    
    bare_acts = [ba for ba in bare_acts if _is_quality_bare_act(ba)]
    case_laws = [cl for cl in case_laws if _is_quality_case_law(cl)]
    
    passed_threshold = len([cl for cl in case_laws if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE])
    progress.add_step(f"Quality filter applied: {passed_threshold} case laws passed threshold", {
        "passed_threshold": passed_threshold,
        "total_searched": case_laws_before
    })

    # Sort: bare acts by relevance; case laws by year (prefer last 40 years) then relevance
    bare_acts.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    case_laws.sort(
        key=lambda x: (
            0 if _case_year_for_sort(x) >= MIN_CASE_YEAR else 1,
            -x.get("_rerank_score", 0),
        )
    )
    logger.info(
        f"Local results (after relevance ≥{MIN_RERANK_SCORE} + quality filter): "
        f"{len(bare_acts)} bare acts, {len(case_laws)} case laws"
    )

    # Case law early exit: if we have 5+ with score > 5.0, skip internet search
    case_laws_high = [cl for cl in case_laws if cl.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    local_high_quality_count = len(case_laws_high)
    skip_web_search = local_high_quality_count >= TARGET_HIGH_QUALITY_CASE_LAWS
    if skip_web_search:
        logger.info(
            f"Local has {local_high_quality_count} case laws with score > {HIGH_QUALITY_SCORE}. "
            "Skipping internet search."
        )
        progress.add_step(f"Found {local_high_quality_count} high-quality case laws (score > {HIGH_QUALITY_SCORE}). Skipping web search.", {
            "high_quality_count": local_high_quality_count,
            "skipped_web": True
        })
        case_laws = case_laws_high[:TARGET_HIGH_QUALITY_CASE_LAWS]
    else:
        # Keep local case laws with score >= WEB_MIN for later merge
        case_laws = [cl for cl in case_laws if cl.get("_rerank_score", 0) >= WEB_MIN_SCORE]
        progress.add_step(f"Found {len(case_laws)} case laws passing threshold (score >= {WEB_MIN_SCORE}). Proceeding to web search.", {
            "passed_threshold": len(case_laws),
            "skipped_web": False
        })

    # Add confirmed materials if provided
    if confirmed_materials:
        bare_acts, case_laws = _add_confirmed_materials(
            confirmed_materials, bare_acts, case_laws
        )

    # Step 3: Sufficiency analysis (only if we're doing web search)
    sufficiency = analyze_sufficiency(facts_summary, bare_acts, case_laws)
    gaps = get_targeted_search_queries(sufficiency)

    # Step 4: If not early-exit and gaps found, search internet (tiered); only official PDFs, index only if score > 5.0
    enrichment_summary = None
    internet_bare_acts = []
    internet_case_laws = []

    if not skip_web_search and gaps and not sufficiency.get("overall_sufficient", False):
        progress.finish_group()  # Finish Internal Search group
        progress.start_group("Web Search", "Searching official court websites and legal portals")
        logger.info(f"Gaps identified: {len(gaps)}. Searching internet (tiered, official PDFs only)...")
        progress.add_step(f"Gaps identified: {len(gaps)}. Searching official sources...", {"gap_count": len(gaps)})
        
        for g in gaps:
            q = (g.get("query") or "").strip()
            if not q or q.lower() in ("relevant indian court judgments", "relevant indian bare act sections"):
                g["query"] = f"{legal_query} Supreme Court High Court judgment"
        gap_results = search_for_gaps(gaps, jurisdiction_state)
        
        # Track web search results
        web_case_law_results = gap_results.get("case_law_results", [])
        progress.add_step(f"Found {len(web_case_law_results)} case law results from web search", {
            "total_found": len(web_case_law_results)
        })
        progress.add_step("Downloading and extracting PDFs...")

        enrichment_summary = enrich_from_gap_results(
            gap_results,
            original_query=facts_summary,
            local_high_quality_count=local_high_quality_count,
            target_high_quality=TARGET_HIGH_QUALITY_CASE_LAWS,
        )
        
        # Track web document scans
        progress.add_step("Scanning web documents for similarity...")
        for enriched in enrichment_summary.get("enriched_case_laws", []):
            score = enriched.get("_rerank_score", 0)
            doc_name = enriched.get("title", "Unknown")
            included = score >= WEB_MIN_SCORE
            progress.add_document_scan(doc_name, score, included, threshold=WEB_MIN_SCORE, metadata={
                "url": enriched.get("url", ""),
                "source_tag": enriched.get("source_tag", "")
            })
        
        web_passed = len([e for e in enrichment_summary.get("enriched_case_laws", []) if e.get("_rerank_score", 0) >= WEB_MIN_SCORE])
        progress.add_step(f"Web search complete: {web_passed} case laws passed threshold (score >= {WEB_MIN_SCORE})", {
            "passed_threshold": web_passed,
            "total_searched": len(web_case_law_results)
        })

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
            score = enriched.get("_rerank_score", 0)
            if content and score >= WEB_MIN_SCORE:
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
                    "_rerank_score": score,
                })

        logger.info(
            f"Internet enrichment: {len(internet_bare_acts)} bare acts, "
            f"{len(internet_case_laws)} case laws"
        )
    else:
        logger.info("Local results sufficient. Skipping internet search.")
        progress.finish_group()  # Finish Internal Search group if web search skipped

    # Finish any remaining group
    if progress.current_group:
        progress.finish_group()

    # Step 5: Format bare acts first, then match case laws to them
    all_bare_acts = _format_bare_acts(bare_acts + internet_bare_acts)
    
    # Apply result_count limit for bare acts
    if result_count and result_count > 0:
        all_bare_acts = all_bare_acts[:result_count]
        logger.info(f"Applied result_count limit: {result_count} bare acts")

    # Step 6: Match case laws to bare act sections
    # Combine and format case laws
    combined_case_laws = case_laws + internet_case_laws
    combined_case_laws.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    # Get enough case laws to match to bare acts (need at least 2 per bare act)
    format_input_size = max(len(all_bare_acts) * 4, 20)
    case_laws_for_format = combined_case_laws[:format_input_size]
    all_case_laws = _format_case_laws(case_laws_for_format, user_query=facts_summary)

    # Match case laws to bare act sections (2 per section, no duplicates)
    all_bare_acts = _match_case_laws_to_bare_acts(all_bare_acts, all_case_laws)

    # Handle intent-specific filtering
    if intent in ("search", "lookup"):
        all_bare_acts, _ = _filter_by_intent(
            facts_summary, intent, all_bare_acts, []
        )

    # Step 7: Generate explanation
    # Flatten case laws for explanation generation (backward compat)
    flattened_case_laws = []
    for ba in all_bare_acts:
        flattened_case_laws.extend(ba.get("related_case_laws", []))
    
    if intent in ("search", "lookup"):
        explanation = _generate_conversational_summary(
            facts_summary, all_bare_acts, flattened_case_laws
        )
    else:
        explanation = _generate_legal_opinion(
            facts_summary, all_bare_acts, flattened_case_laws, sufficiency
        )

    if not (explanation or "").strip():
        explanation = "Here's what I found for your query. Below are the relevant legal provisions with related case laws."

    # Collect source tags for transparency
    sources_used = set()
    for ba in all_bare_acts:
        sources_used.add(ba.get("source_tag", "LOCAL_DB"))
        for cl in ba.get("related_case_laws", []):
            sources_used.add(cl.get("source_tag", "LOCAL_DB"))

    return {
        "bare_act_sections": all_bare_acts,  # Now includes nested case_laws
        "case_laws": flattened_case_laws,  # Flattened for backward compat
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "needs_confirmation": False,
        "internet_case_laws": [],  # backward compat
        "progress": progress.get_progress(),  # Add progress tracking data
    }


# ---------------------------------------------------------------------------
# Matching Case Laws to Bare Acts
# ---------------------------------------------------------------------------

def _match_case_laws_to_bare_acts(bare_acts: list, case_laws: list, max_per_section: int = 2) -> list:
    """
    Match case laws to bare act sections based on relevance.
    Each case law is assigned to the bare act section with highest relevance score.
    Limits to max_per_section case laws per bare act section.
    Ensures no duplicates (each case law appears only once).
    
    Returns bare_acts list with 'related_case_laws' field added to each section.
    """
    from retrieval.hybrid_retriever import score_query_document
    
    if not bare_acts or not case_laws:
        # If no bare acts or case laws, return as-is
        for ba in bare_acts:
            ba["related_case_laws"] = []
        return bare_acts
    
    # Score each case law against each bare act section
    # Create query from bare act: act_name + section_number + text
    matches = []  # List of (bare_act_idx, case_law_idx, score)
    
    for ba_idx, ba in enumerate(bare_acts):
        act_name = ba.get("act_name", "")
        section = ba.get("section_number", "")
        ba_text = ba.get("text", "")
        # Create query from bare act section
        ba_query = f"{act_name} Section {section} {ba_text[:500]}".strip()
        
        for cl_idx, cl in enumerate(case_laws):
            cl_text = cl.get("text", "")
            cl_title = cl.get("title", cl.get("case_name", ""))
            # Score case law relevance to this bare act section
            score = score_query_document(ba_query, f"{cl_title} {cl_text[:2000]}")
            matches.append((ba_idx, cl_idx, score))
    
    # Sort matches by score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Assign case laws to bare act sections (no duplicates, max 2 per section)
    assigned_case_laws = set()  # Track which case laws have been assigned
    bare_act_case_counts = {}  # Track how many case laws each bare act has
    
    for ba_idx, cl_idx, score in matches:
        # Skip if this case law already assigned
        if cl_idx in assigned_case_laws:
            continue
        # Skip if this bare act already has max case laws
        if bare_act_case_counts.get(ba_idx, 0) >= max_per_section:
            continue
        
        # Assign this case law to this bare act
        if "related_case_laws" not in bare_acts[ba_idx]:
            bare_acts[ba_idx]["related_case_laws"] = []
        bare_acts[ba_idx]["related_case_laws"].append(case_laws[cl_idx])
        assigned_case_laws.add(cl_idx)
        bare_act_case_counts[ba_idx] = bare_act_case_counts.get(ba_idx, 0) + 1
    
    # Ensure all bare acts have the field (even if empty)
    for ba in bare_acts:
        if "related_case_laws" not in ba:
            ba["related_case_laws"] = []
    
    logger.info(f"Matched {len(assigned_case_laws)} case laws to {len(bare_acts)} bare act sections")
    return bare_acts


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def _format_bare_acts(bare_acts: list) -> list:
    """Format bare act results for display. Skip junk (no act/section)."""
    formatted = []
    seen = set()

    for ba in bare_acts:
        if not _is_quality_bare_act(ba):
            continue
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


def _summarize_case_paragraphs(
    user_query: str, case_display_name: str, paragraph_texts: list
) -> tuple:
    """
    Generate a 150-200 word summary and extract "Appellant v/s Respondent" from the model.
    Returns (parties_line, summary_body). parties_line is used for title; summary_body for card text.
    """
    if not paragraph_texts:
        return ("", "")
    excerpts = "\n\n".join(
        (t.strip()[:2000] for t in paragraph_texts[:3] if (t or "").strip())
    )
    if not excerpts.strip():
        return ("", "")
    prompt = f"""{CASE_SUMMARY_SYSTEM}

USER'S QUERY / WHAT THEY CARE ABOUT:
{user_query[:800]}

CASE: {case_display_name}

EXCERPTS FROM THE JUDGMENT (most relevant paragraphs):
{excerpts[:6000]}

Your response (first line = Appellant v/s Respondent, then blank line, then 150-200 word summary):"""
    try:
        raw = ask_llm(prompt).strip()
        if not raw:
            return ("", paragraph_texts[0][:800] if paragraph_texts else "")
        lines = raw.split("\n")
        parties_line = ""
        if lines:
            first = lines[0].strip()
            if first and ("v/s" in first or " vs " in first.lower() or " v. " in first.lower()):
                parties_line = first[:200]
        start = 1
        while start < len(lines) and not lines[start].strip():
            start += 1
        summary_body = "\n".join(lines[start:]).strip()[:2500] if start < len(lines) else raw[:2500]
        return (parties_line, summary_body or raw[:2500])
    except Exception as e:
        logger.warning(f"Case summary generation failed: {e}")
        return ("", paragraph_texts[0][:800] if paragraph_texts else "")


def _format_case_laws(case_laws: list, user_query: str = "") -> list:
    """
    Group case law chunks by case, take top 3 most relevant paragraphs per case,
    and replace raw text with a 150-200 word summary in the model's own words.
    Returns one row per case.
    """
    # Group chunks by case (same judgment = same case_name + source)
    groups = {}
    for cl in case_laws:
        if not _is_quality_case_law(cl):
            continue
        text = (cl.get("full_text") or cl.get("text") or "").strip()
        if not text or len(text) < 30:
            continue
        case_name = cl.get("case_name", cl.get("source", ""))
        court = cl.get("court", "")
        year = cl.get("year", "")
        group_key = (case_name or "", court, str(year or ""), cl.get("source", ""))
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append({
            "cl": cl,
            "text": text,
            "score": cl.get("_rerank_score", 0),
        })

    formatted = []
    for group_key, chunks in groups.items():
        case_name, court, year, _ = group_key
        chunks.sort(key=lambda x: -x["score"])
        top3 = chunks[:3]
        if not top3:
            continue
        first = top3[0]["cl"]
        court = first.get("court", court)
        year = first.get("year", year)
        citation = first.get("citation", "")
        binding_authority = first.get("binding_authority", "")
        fallback_title = case_name or first.get("source", "Unknown")
        if court:
            fallback_title += f" ({court}"
            if year:
                fallback_title += f", {year}"
            fallback_title += ")"
        elif citation:
            fallback_title += f" [{citation}]"

        paragraph_texts = [c["text"] for c in top3]
        parties_line = ""
        summary_body = ""
        if user_query.strip():
            parties_line, summary_body = _summarize_case_paragraphs(
                user_query, fallback_title, paragraph_texts
            )
        if not summary_body:
            summary_body = paragraph_texts[0][:800] if paragraph_texts else ""

        acronym = _court_acronym(court, binding_authority, case_name or first.get("source", ""))
        if parties_line.strip():
            display_title = f"{acronym} {parties_line.strip()}"
        else:
            display_title = f"{acronym} {fallback_title}".strip() if acronym != "[Court]" else fallback_title

        # Preserve URL - check multiple possible fields
        url = first.get("url") or first.get("source_url") or ""
        # For local case laws without URL, we could construct a file:// URL, but for now leave empty
        # Internet case laws should have URL from enrichment
        
        formatted.append({
            "source": first.get("source_file", first.get("source", "")),
            "text": summary_body,
            "case_name": case_name,
            "title": display_title,
            "court": court,
            "year": year,
            "citation": citation,
            "binding_authority": first.get("binding_authority", ""),
            "url": url,  # Preserve URL from original chunk
            "source_tag": first.get("source_tag", "LOCAL_DB"),
            "_rerank_score": top3[0]["score"],
            "_year": _case_year_for_sort(first),
        })

    authority_order = {"supreme_court": 0, "high_court": 1, "tribunal": 2, "district_court": 3}
    formatted.sort(
        key=lambda x: (
            authority_order.get(x.get("binding_authority", ""), 5),
            0 if x.get("_year", 0) >= MIN_CASE_YEAR else 1,
            -x.get("_rerank_score", 0),
        )
    )
    for x in formatted:
        x.pop("_year", None)
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
