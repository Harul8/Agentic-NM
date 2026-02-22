"""
Response Generator v2 — Full legal research pipeline with:
- Hybrid retrieval (FAISS + BM25 + cross-encoder re-ranking)
- LLM-driven sufficiency analysis (no hard numeric limits)
- Tiered internet search (official courts → legal portals → newspapers)
- Auto-enrichment (PDFs saved to Google Drive, indexed to vector store)
- Source tagging ([LOCAL_DB], [OFFICIAL] = government sources, [LEGAL_PORTAL], [NEWS_REFERENCE])

This replaces the old response_generator.py's generate_response() function.
The old module is preserved for backward compatibility — this new module is
imported and used by the API server.
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from config import (
    BARE_ACTS_DIR,
    CASELAW_DIR,
    GOOGLE_DRIVE_BARE_ACTS_FOLDER_URL,
    GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL,
)
from llm.ollama_client import ask_llm, get_model_display_for_prompt, get_gpu_info
from prompts.advocate_prompts import (
    EXPAND_LEGAL_QUERY_SYSTEM,
    EXTRACT_BARE_ACT_PORTIONS_SYSTEM,
    EXTRACT_CASE_PORTIONS_SYSTEM,
    CASE_SUMMARY_SYSTEM,
    RELEVANCE_EXPLANATION_SYSTEM,
    RELEVANCE_EXPLANATION_NO_MATERIALS,
    CONVERSATIONAL_SUMMARY_SYSTEM,
    BARE_ACT_ONLY_SUMMARY,
    CASE_LAW_DISPUTE_ORDER_SUMMARY,
)
from services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# Relevance and quality thresholds — keep only high-quality, recent materials
MIN_RERANK_SCORE = 3.0  # Minimum to include in pool (lowered from 4.0 to allow more local results)
HIGH_QUALITY_SCORE = 5.0  # Case laws with score > this: "highly relevant"; stop web search if we have 5+; index web PDFs only if > this
WEB_MIN_SCORE = 1.0  # Minimum cross-encoder score for web case laws (ms-marco logits; 1.0 keeps on-topic official SC judgments)
TARGET_HIGH_QUALITY_CASE_LAWS = 5  # Stop search once we have this many case laws with score > HIGH_QUALITY_SCORE
MIN_CASE_YEAR = 1975 # Prefer cases from last 40 years; older treated as low priority
JUNK_CASE_PATTERNS = ("unknown", "unknown (", "high court, 1908", "high court, 1972")

# When user does NOT set a limit: include all with score > HIGH_QUALITY_SCORE; if fewer than this, add from >= MIN_RERANK_SCORE up to this many
FLEXIBLE_MIN_FALLBACK = 10  # Minimum results to show when no user limit and not enough high-similarity (all must pass MIN_RERANK_SCORE)


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


def _apply_flexible_result_limit(
    items: list,
    score_key: str = "_rerank_score",
    user_limit: Optional[int] = None,
    high_score: float = HIGH_QUALITY_SCORE,
    min_score: float = MIN_RERANK_SCORE,
    min_fallback: int = FLEXIBLE_MIN_FALLBACK,
) -> list:
    """
    When user did not set a limit (user_limit is None): include all items with score > high_score;
    if there are fewer than min_fallback, add from items with score >= min_score up to min_fallback total.
    When user set a limit: return first user_limit items (caller ensures items already pass min_score).
    Items without score are treated as min_score so they can be included in the fallback tier.
    """
    if not items:
        return []
    if user_limit is not None and user_limit > 0:
        return items[:user_limit]
    # Sort by score desc (missing score treated as min_score for inclusion in pool)
    def _score(i):
        s = i.get(score_key)
        return (float(s) if s is not None else min_score)
    sorted_items = sorted(items, key=_score, reverse=True)
    high = [i for i in sorted_items if _score(i) > high_score]
    rest = [i for i in sorted_items if min_score <= _score(i) <= high_score]
    if len(high) >= min_fallback:
        return high
    need = min_fallback - len(high)
    return high + rest[:need]


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
    # Also check for "v/s State" or "vs State" (without "of")
    if ("state of " in combined or " v/s state" in combined or " vs state" in combined or 
        " v/s state" in combined or " vs state" in combined or
        "/state" in combined.lower()):
        return "[HC]"
    # If case name contains "State" and looks like a court case, default to HC
    if "state" in combined and ("v/s" in combined or "vs" in combined or "v." in combined):
        return "[HC]"
    return "[Court]"


# ---------------------------------------------------------------------------
# Gap query generation (LLM-driven when empty or generic)
# ---------------------------------------------------------------------------

def _generate_gap_search_query(facts_summary: str, legal_query: str, gap_type: str) -> Optional[str]:
    """Generate one web search query for a gap via LLM. Returns None on failure."""
    kind = "bare act / legislation" if gap_type == "bare_act" else "case law / court judgment"
    prompt = f"""You are an Indian legal research assistant. We need a short web search phrase to find {kind}.

User request: {facts_summary[:400]}
Legal query: {legal_query[:300]}

Generate ONE short search phrase (5-12 words) suitable for a search engine to find relevant Indian {kind}. Include India Code or Supreme Court/High Court if appropriate. Output only the search phrase, no preamble or explanation."""
    try:
        out = (ask_llm(prompt) or "").strip()[:200]
        return out if out else None
    except Exception as e:
        logger.debug("Gap query LLM failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Query Expansion (reused from v1, improved)
# ---------------------------------------------------------------------------

def expand_legal_query(facts: str, intent: dict = None) -> str:
    """Convert plain-language facts to a precise legal research query. If intent is provided (from extract_research_intent), use it to enrich the query dynamically."""
    from prompts.advocate_prompts import EXPAND_LEGAL_QUERY_SYSTEM, EXPAND_LEGAL_QUERY_INTENT_BLOCK
    intent_block = ""
    if intent and isinstance(intent, dict) and (intent.get("states") or intent.get("domains") or intent.get("topics")):
        intent_block = EXPAND_LEGAL_QUERY_INTENT_BLOCK.format(intent_json=json.dumps(intent, indent=0))
    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}
{intent_block}

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
    progress_callback=None,
    document_types: str = "both",
    search_strategy: str = "local_then_web",
) -> dict:
    """
    Full legal research response using the v2 pipeline.

    Flow depends on search_strategy:
    - local_then_web (default): local hybrid search → sufficiency → web for gaps
    - web_only: skip local; build gaps from query → tiered web search only
    - local_only: local search only; no web search

    1. Expand facts to legal search query
    2. Hybrid search local vector store (unless web_only)
    3. LLM-driven sufficiency analysis
    4. If gaps found and not local_only → tiered internet search → auto-enrich
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
        get_broad_discovery_queries,
    )
    from retrieval.tiered_search import search_for_gaps, fetch_content_and_pdf
    from retrieval.auto_enricher import enrich_from_gap_results

    # Initialize progress tracker
    progress = ProgressTracker()

    def _emit_progress():
        if progress_callback:
            try:
                progress_callback(progress.get_progress_snapshot())
            except Exception:
                pass

    # Normalize search_strategy
    if search_strategy not in ("local_only", "web_only", "local_then_web"):
        search_strategy = "local_then_web"

    # Intent extraction for every request: extract states, domains, topics so expansion and (when web_only) broad discovery are intent-aware.
    research_intent = None
    try:
        from services.intent_extractor import extract_research_intent
        research_intent = extract_research_intent(facts_summary)
    except Exception as e:
        logger.debug("Intent extraction skipped: %s", e)

    # Step 1: Expand query (intent-aware so expansion reflects user intent dynamically)
    legal_query = expand_legal_query(facts_summary, intent=research_intent)
    search_query = f"{facts_summary} {legal_query}"[:500]
    logger.info(f"Expanded query: {legal_query[:200]}")

    # Intent + document_types: only retrieve what the user asked for (acts = government; case laws = courts)
    retrieve_acts = intent == "lookup" or document_types == "acts_only" or (intent == "legal_opinion" and document_types != "case_laws_only")
    retrieve_case_laws = intent == "search" or document_types == "case_laws_only" or (intent == "legal_opinion" and document_types != "acts_only")
    if not retrieve_acts and not retrieve_case_laws:
        retrieve_acts, retrieve_case_laws = True, True  # fallback

    # Step 2: Hybrid search local vector store (skip entirely if web_only)
    if search_strategy == "web_only":
        bare_acts = []
        case_laws = []
        progress.start_group("Web Search", "User requested web-only search (skipping local database)")
        _emit_progress()
        progress.add_step("Skipping local search. Building web search from your query...", {"search_strategy": "web_only"})
        _emit_progress()
        logger.info("Search strategy: web_only — skipping local vector store")
    else:
        progress.start_group("Internal Search", "Local vector store (FAISS + BM25 + re-ranking)")
        _emit_progress()
        progress.add_step("Searching local vector store (FAISS + BM25 + re-ranking)...")
        _emit_progress()
        logger.info("Searching local vector store (hybrid: FAISS + BM25 + re-rank)...")
        if not retrieve_case_laws:
            bare_acts = search_bare_acts_auto(search_query, top_k=30)
            case_laws = []
            logger.info("Retrieval: bare-act-only (user asked for acts/laws only, no case laws)")
        elif not retrieve_acts:
            bare_acts = []
            case_laws = search_case_laws_auto(search_query, top_k=45)
            logger.info("Retrieval: case-law-only (user asked for judgments only, no acts)")
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_ba = executor.submit(search_bare_acts_auto, search_query, 30)
                fut_cl = executor.submit(search_case_laws_auto, search_query, 45)
                bare_acts = fut_ba.result()
                case_laws = fut_cl.result()

    logger.info(f"Local results (raw): {len(bare_acts)} bare act chunks, {len(case_laws)} case law chunks")

    # When web_only we already have empty local results; skip local progress/filter/sort and set skip_web_search=False
    if search_strategy != "web_only":
        # Progress messages reflect what we're actually retrieving (acts only / case laws only / both)
        if retrieve_acts and retrieve_case_laws:
            progress.add_step(f"Found {len(bare_acts)} bare act sections, {len(case_laws)} case law documents in local database", {"bare_acts": len(bare_acts), "case_laws": len(case_laws), "total_found": len(bare_acts) + len(case_laws)})
        elif retrieve_acts:
            progress.add_step(f"Found {len(bare_acts)} bare act sections in local database (acts only)", {"total_found": len(bare_acts), "bare_acts": len(bare_acts)})
        else:
            progress.add_step(f"Found {len(case_laws)} case law documents in local database", {"total_found": len(case_laws), "case_laws": len(case_laws)})
        _emit_progress()
        # Track document scans: one line per phase that updates in place (use prefix so we don't overwrite doc-scan steps)
        if retrieve_acts and bare_acts:
            bare_to_scan = bare_acts[:20]  # cap to avoid too many steps
            bare_scan_total = len(bare_to_scan)
            _bare_prefix = "Scanning bare act sections for relevance"
            for idx, ba in enumerate(bare_to_scan, 1):
                msg = f"{_bare_prefix} ({idx}/{bare_scan_total}) done"
                if idx == 1:
                    progress.add_step(msg, {"current": idx, "total": bare_scan_total})
                else:
                    progress.update_last_step_with_prefix(_bare_prefix, msg, {"current": idx, "total": bare_scan_total})
                _emit_progress()
                score = ba.get("_rerank_score", 0)
                doc_name = ba.get("act_name") or "Unknown"
                included = score >= MIN_RERANK_SCORE
                progress.add_document_scan(doc_name, score, included, threshold=MIN_RERANK_SCORE)
                _emit_progress()
        if retrieve_case_laws and case_laws:
            case_scan_total = len(case_laws)
            _case_prefix = "Scanning case law documents for similarity"
            for idx, cl in enumerate(case_laws, 1):
                msg = f"{_case_prefix} ({idx}/{case_scan_total}) done"
                if idx == 1:
                    progress.add_step(msg, {"current": idx, "total": case_scan_total})
                else:
                    progress.update_last_step_with_prefix(_case_prefix, msg, {"current": idx, "total": case_scan_total})
                _emit_progress()
                score = cl.get("_rerank_score", 0)
                doc_name = cl.get("case_name") or cl.get("source", "Unknown")
                included = score >= MIN_RERANK_SCORE
                progress.add_document_scan(doc_name, score, included, threshold=MIN_RERANK_SCORE)
                _emit_progress()

        # Filter by relevance then by quality — no junk or very old cases
        progress.add_step(f"Filtering documents (relevance score >= {MIN_RERANK_SCORE})...")
        _emit_progress()
        bare_acts_before = len(bare_acts)
        case_laws_before = len(case_laws)
        
        bare_acts = [ba for ba in bare_acts if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE]
        case_laws = [cl for cl in case_laws if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE]
        
        bare_acts = [ba for ba in bare_acts if _is_quality_bare_act(ba)]
        case_laws = [cl for cl in case_laws if _is_quality_case_law(cl)]
        
        passed_threshold = len([cl for cl in case_laws if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE])
        bare_passed = len(bare_acts)
        if retrieve_acts and retrieve_case_laws:
            progress.add_step(f"Quality filter applied: {bare_passed} bare acts, {passed_threshold} case laws passed threshold", {
                "passed_threshold": passed_threshold,
                "total_searched": bare_acts_before + case_laws_before,
                "bare_acts_passed": bare_passed,
                "case_laws_passed": passed_threshold,
            })
        elif retrieve_acts:
            progress.add_step(f"Quality filter applied: {bare_passed} bare act sections passed threshold", {
                "total_searched": bare_acts_before,
                "bare_acts_passed": bare_passed,
            })
        else:
            progress.add_step(f"Quality filter applied: {passed_threshold} case laws passed threshold", {
                "passed_threshold": passed_threshold,
                "total_searched": case_laws_before,
                "case_laws_passed": passed_threshold,
            })
        _emit_progress()
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

    # Case law early exit: only when we're actually retrieving case laws and have enough (and not local_only)
    case_laws_high = [cl for cl in case_laws if cl.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    local_high_quality_count = len(case_laws_high)
    skip_web_search = (
        retrieve_case_laws
        and local_high_quality_count >= TARGET_HIGH_QUALITY_CASE_LAWS
    )
    if search_strategy == "local_only":
        skip_web_search = True
        logger.info("Search strategy: local_only — skipping web search")
        if progress.current_group:
            progress.add_step("User requested local database only. Skipping web search.", {"search_strategy": "local_only"})
            _emit_progress()
    if search_strategy != "web_only":
        if skip_web_search:
            logger.info(
                f"Local has {local_high_quality_count} case laws with score > {HIGH_QUALITY_SCORE}. "
                "Skipping internet search."
            )
            progress.add_step(f"Found {local_high_quality_count} high-quality case laws (score > {HIGH_QUALITY_SCORE}). Skipping web search.", {
                "high_quality_count": local_high_quality_count,
                "total_found": local_high_quality_count,
                "skipped_web": True,
            })
            _emit_progress()
            # When user set a limit, cap; otherwise include all high-similarity (no rigid cap)
            if result_count is not None and result_count > 0:
                case_laws = case_laws_high[:result_count]
            else:
                case_laws = case_laws_high
        else:
            # Keep local case laws with score >= WEB_MIN for later merge (only if we're retrieving case laws)
            if retrieve_case_laws:
                case_laws = [cl for cl in case_laws if cl.get("_rerank_score", 0) >= WEB_MIN_SCORE]
                case_passing = len(case_laws)
                progress.add_step(f"Found {case_passing} case laws passing threshold (score >= {WEB_MIN_SCORE}). Proceeding to web search.", {
                    "passed_threshold": case_passing,
                    "total_found": case_passing,
                    "skipped_web": False,
                })
            elif retrieve_acts:
                bare_passing = len(bare_acts)
                progress.add_step(f"Found {bare_passing} bare act sections. Proceeding to web search for more acts if needed.", {
                    "passed_threshold": bare_passing,
                    "total_found": bare_passing,
                    "skipped_web": False,
                })
            else:
                progress.add_step("Proceeding to web search.", {"skipped_web": False})
            _emit_progress()
    else:
        # web_only: always do web search (gaps will come from sufficiency with empty local)
        skip_web_search = False
        progress.add_step("Identifying web search queries from your question...", {"skipped_web": False})
        _emit_progress()

    # Add confirmed materials if provided
    if confirmed_materials:
        bare_acts, case_laws = _add_confirmed_materials(
            confirmed_materials, bare_acts, case_laws
        )

    # Step 3: Sufficiency analysis vs broad discovery (web_only with empty local gets multiple broad queries from intent)
    if search_strategy == "web_only" and not bare_acts and not case_laws:
        # Broad discovery: queries built dynamically from extracted intent (states, domains, topics)—no hardcoded scenarios.
        gaps = get_broad_discovery_queries(
            facts_summary, legal_query, document_types, intent=research_intent
        )
        sufficiency = {"overall_sufficient": False, "gaps": [], "aspects": [], "confidence": "low"}
        logger.info("Web search: using %d broad discovery queries (web_only, no local)", len(gaps))
    else:
        sufficiency = analyze_sufficiency(facts_summary, bare_acts, case_laws)
        gaps = get_targeted_search_queries(sufficiency)
        # Respect document_types: only search for what the user asked for (acts vs case laws)
        if document_types == "acts_only":
            gaps = [g for g in gaps if g.get("type") == "bare_act"]
            if not gaps and sufficiency.get("gaps"):
                gaps = [{"query": (facts_summary or legal_query)[:200], "type": "bare_act"}]
            logger.info("Web search: acts_only — only bare act gaps")
        elif document_types == "case_laws_only":
            gaps = [g for g in gaps if g.get("type") == "case_law"]
            if not gaps and sufficiency.get("gaps"):
                gaps = [{"query": (facts_summary or legal_query)[:200], "type": "case_law"}]
            logger.info("Web search: case_laws_only — only case law gaps")

    # Step 4: If not early-exit and gaps found, search internet (tiered); only official PDFs, index only if score > 5.0
    enrichment_summary = None
    web_bare_count = 0
    web_case_count = 0
    web_search_stats = None  # For data-driven no-materials summary: web_found, shortlisted, proposed, already_in_library
    internet_bare_acts = []
    internet_case_laws = []

    if not skip_web_search and gaps and not sufficiency.get("overall_sufficient", False):
        progress.finish_group()  # Finish Internal Search group
        _emit_progress()
        progress.start_group("Web Search", "External sources (official court websites & legal portals)")
        _emit_progress()
        logger.info("Gaps identified: %d. Searching internet (tiered, official PDFs only)...", len(gaps))
        for g in gaps:
            logger.info("Gap query: %s (type=%s)", (g.get("query") or "")[:120], g.get("type", "both"))
        gap_count = len(gaps)
        progress.add_step(f"Gaps identified: {gap_count}. Searching official sources...", {"gap_count": gap_count})
        _emit_progress()

        for g in gaps:
            q = (g.get("query") or "").strip()
            if not q or q.lower() in ("relevant indian court judgments", "relevant indian bare act sections"):
                gap_type = g.get("type", "both")
                # Model-driven: generate query from user intent; static template as fallback
                generated = _generate_gap_search_query(facts_summary, legal_query, gap_type)
                if generated:
                    g["query"] = generated
                    logger.info("Gap query from LLM: %s", generated[:80])
                elif gap_type == "bare_act":
                    g["query"] = f"{legal_query} India Code bare act state act"
                else:
                    g["query"] = f"{legal_query} Supreme Court High Court judgment"
        gap_results = search_for_gaps(gaps, jurisdiction_state)
        
        # Track web search results — dynamic counts for all scenarios (acts_only, case_laws_only, both)
        web_bare_results = gap_results.get("bare_act_results", [])
        web_case_law_results = gap_results.get("case_law_results", [])
        web_bare_count = len(web_bare_results)
        web_case_count = len(web_case_law_results)
        if document_types == "acts_only":
            progress.add_step(f"Found {web_bare_count} bare act results from web search", {"total_found": web_bare_count, "bare_found": web_bare_count})
        elif document_types == "case_laws_only":
            progress.add_step(f"Found {web_case_count} case law results from web search", {"total_found": web_case_count, "case_found": web_case_count})
        else:
            progress.add_step(f"Found {web_bare_count} bare act, {web_case_count} case law results from web search", {
                "bare_found": web_bare_count,
                "case_found": web_case_count,
                "total_found": web_bare_count + web_case_count,
            })
        _emit_progress()

        # When user asked for "pull all" / "get all" / scope broad, enrich more (cap 300 bare acts, 100 case laws).
        pull_all = (
            (research_intent and research_intent.get("scope") == "broad")
            or (result_count is not None and result_count >= 25)
        )
        max_bare = 300 if pull_all else None
        max_case = 100 if pull_all else None

        def _download_progress(current: int, total: int):
            msg = f"Downloading and extracting PDFs ({current}/{total}) done"
            if current == 1:
                progress.add_step(msg, {"current": current, "total": total})
            else:
                progress.update_last_step(msg, {"current": current, "total": total})
            _emit_progress()

        enrichment_summary = enrich_from_gap_results(
            gap_results,
            original_query=facts_summary,
            local_high_quality_count=local_high_quality_count,
            target_high_quality=TARGET_HIGH_QUALITY_CASE_LAWS,
            skip_index=True,  # Do not auto-index; surface as indexing_candidates for user-triggered indexing
            max_bare_acts_to_enrich=max_bare,
            max_case_laws_to_enrich=max_case,
            progress_callback=_download_progress,
        )

        # Track web document scans: one "Scanning (n/total) done" line that updates in place (don't overwrite doc-scan steps)
        _scan_prefix = "Scanning web documents for relevance"
        enriched_bare = enrichment_summary.get("enriched_bare_acts", [])
        enriched_case = enrichment_summary.get("enriched_case_laws", [])
        scan_total = len(enriched_bare) + len(enriched_case)
        scan_n = 0
        for enriched in enriched_bare:
            scan_n += 1
            msg = f"{_scan_prefix} ({scan_n}/{scan_total}) done"
            if scan_n == 1:
                progress.add_step(msg, {"current": scan_n, "total": scan_total})
            else:
                progress.update_last_step_with_prefix(_scan_prefix, msg, {"current": scan_n, "total": scan_total})
            _emit_progress()
            doc_name = enriched.get("title", "Unknown")
            progress.add_document_scan(doc_name, 1.0, True, threshold=WEB_MIN_SCORE, metadata={
                "url": enriched.get("url", ""),
                "source_tag": enriched.get("source_tag", "")
            })
            _emit_progress()
        for enriched in enriched_case:
            scan_n += 1
            msg = f"{_scan_prefix} ({scan_n}/{scan_total}) done"
            if scan_n == 1:
                progress.add_step(msg, {"current": scan_n, "total": scan_total})
            else:
                progress.update_last_step_with_prefix(_scan_prefix, msg, {"current": scan_n, "total": scan_total})
            _emit_progress()
            score = enriched.get("_rerank_score", 0)
            doc_name = enriched.get("title", "Unknown")
            included = score >= WEB_MIN_SCORE
            progress.add_document_scan(doc_name, score, included, threshold=WEB_MIN_SCORE, metadata={
                "url": enriched.get("url", ""),
                "source_tag": enriched.get("source_tag", "")
            })
            _emit_progress()
        web_bare_passed = len(enrichment_summary.get("enriched_bare_acts", []))
        web_passed = len([e for e in enrichment_summary.get("enriched_case_laws", []) if e.get("_rerank_score", 0) >= WEB_MIN_SCORE])
        if document_types == "acts_only":
            progress.add_step(f"Web search complete: {web_bare_passed} bare act documents", {
                "passed_threshold": web_bare_passed,
                "total_searched": web_bare_count,
                "total_found": web_bare_count,
            })
        elif document_types == "case_laws_only":
            progress.add_step(f"Web search complete: {web_passed} case laws passed threshold (score >= {WEB_MIN_SCORE})", {
                "passed_threshold": web_passed,
                "total_searched": web_case_count,
                "total_found": web_case_count,
            })
        else:
            progress.add_step(f"Web search complete: {web_bare_passed} bare acts, {web_passed} case laws passed threshold", {
                "passed_threshold": web_bare_passed + web_passed,
                "total_searched": web_bare_count + web_case_count,
                "bare_passed": web_bare_passed,
                "case_passed": web_passed,
            })
        _emit_progress()
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
        _emit_progress()
    # Finish any remaining group
    if progress.current_group:
        progress.finish_group()
        _emit_progress()

    # Step 4b: Post-search progress — building response, opinion, indexing list
    progress.start_group("Response", "Building final response, opinion, and indexing list")
    progress.add_step("Building final response...", {})
    _emit_progress()

    # Step 5: Apply flexible result limit (no rigid cap when user didn't set limit: all > 5.0, else up to 10 with >= 3.0)
    combined_bare = bare_acts + internet_bare_acts
    limited_bare = _apply_flexible_result_limit(combined_bare, user_limit=result_count)
    all_bare_acts = _format_bare_acts(limited_bare)

    # Step 6: Match case laws to bare act sections (or for search-only, wrap case laws in one section)
    combined_case_laws = case_laws + internet_case_laws
    combined_case_laws.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    limited_case_laws = _apply_flexible_result_limit(combined_case_laws, user_limit=result_count)
    format_input_size = max(len(all_bare_acts) * 6, 30) if all_bare_acts else (result_count or FLEXIBLE_MIN_FALLBACK)
    # Cap chunks per case so one judgment does not dominate (more distinct cases in the list)
    case_laws_for_format = _diversify_case_laws_by_case(
        limited_case_laws,
        max_chunks_per_case=4,
        max_total=max(format_input_size, 24),
    )
    all_case_laws = _format_case_laws(case_laws_for_format, user_query=facts_summary)

    # Match case laws to bare act sections (2 per section, no duplicates); search-only has no bare acts
    all_bare_acts = _match_case_laws_to_bare_acts(all_bare_acts, all_case_laws)

    # Search-only: we have no bare acts; wrap all case laws in one section so UI gets them
    if intent == "search" and not all_bare_acts and all_case_laws:
        # When user set a limit, cap; otherwise show all we have (already limited by flexible rule)
        limit = max(1, result_count) if result_count is not None else max(FLEXIBLE_MIN_FALLBACK, len(all_case_laws))
        all_bare_acts = [{
            "act_name": "Case laws",
            "title": "Case laws (search results)",
            "text": "",
            "related_case_laws": all_case_laws[:limit],
            "source_tag": "LOCAL_DB",
        }]

    # Step 6b: Generating explanation (flow-dependent)
    if intent in ("search", "lookup"):
        progress.add_step("Generating search summary...", {})
    else:
        progress.add_step("Generating legal opinion...", {})
    _emit_progress()

    # Step 7: Generate explanation
    # Flatten case laws for explanation generation (backward compat)
    flattened_case_laws = []
    for ba in all_bare_acts:
        flattened_case_laws.extend(ba.get("related_case_laws", []))

    # Add "Calling X model, GPU: Y" step here (before blocking on LLM) so it always appears in the progress tracker
    _bare = (facts_summary or "")[:1500]
    _bare += json.dumps([{"t": b.get("title", ""), "x": (b.get("text") or "")[:500]} for b in all_bare_acts[:15]])[:3000]
    _bare += json.dumps([{"t": c.get("title", ""), "x": (c.get("text") or "")[:500]} for c in flattened_case_laws[:15]])[:3000]
    _display, _switched = get_model_display_for_prompt(_bare)
    if _display:
        _action = "Switching to" if _switched else "Calling"
        _msg = f"{_action} {_display} model"
        _gpu = get_gpu_info()
        if _gpu:
            _msg += f", GPU: {', '.join(_gpu)}"
        progress.add_step(_msg)
        _emit_progress()

    def _on_before_llm(prompt: str):
        pass  # Model step already added above so it appears before ask_llm blocks

    if intent in ("search", "lookup"):
        explanation = _generate_conversational_summary(
            facts_summary, all_bare_acts, flattened_case_laws, intent=intent, on_before_llm=_on_before_llm
        )
    else:
        explanation = _generate_legal_opinion(
            facts_summary, all_bare_acts, flattened_case_laws, sufficiency, on_before_llm=_on_before_llm
        )

    if not (explanation or "").strip():
        explanation = "Here's what I found for your query. Below are the relevant legal provisions with related case laws."

    # Build indexing candidates for UI: strictly official PDF documents only (government sources; no legal portals / HTML-only).
    def _is_official_pdf(e):
        if not e.get("url") or not e.get("title"):
            return False
        tag = (e.get("source_tag") or "").upper()
        if tag not in ("OFFICIAL", "OFFICIAL_COURT"):
            return False
        if e.get("pdf_saved"):
            return True
        url = (e.get("url") or "").lower()
        return ".pdf" in url or "/bitstream/" in url

    indexing_candidates = []
    if enrichment_summary:
        raw_candidates = []
        enriched_bare = enrichment_summary.get("enriched_bare_acts", [])
        enriched_case = enrichment_summary.get("enriched_case_laws", [])
        for e in enriched_bare:
            if _is_official_pdf(e):
                raw_candidates.append({
                    "title": e.get("title", ""),
                    "source_url": e.get("url", ""),
                    "suggested_category": "bare_act",
                    "content": (e.get("content") or "")[:2000],  # first ~2 pages for duplicate check
                })
        for e in enriched_case:
            if _is_official_pdf(e):
                raw_candidates.append({
                    "title": e.get("title", ""),
                    "source_url": e.get("url", ""),
                    "suggested_category": "case_law",
                    "content": (e.get("content") or "")[:2000],
                })
        n_bare = sum(1 for c in raw_candidates if c.get("suggested_category") == "bare_act")
        n_case = sum(1 for c in raw_candidates if c.get("suggested_category") == "case_law")
        logger.info(
            "Indexing candidates (official PDF only): %d bare acts, %d case laws from %d/%d enriched",
            n_bare, n_case, len(enriched_bare), len(enriched_case),
        )
        # Cross-check with internal store (short title + year); mark duplicates so UI can highlight and skip index
        try:
            progress.add_step(f"Preparing Pending indexing list (0/{len(raw_candidates)})...", {"current": 0, "total": len(raw_candidates)})
            _emit_progress()
            from services.indexing_duplicate_check import check_indexing_candidates
            indexing_candidates = check_indexing_candidates(raw_candidates)
            # Drop content from payload; keep only title, source_url, suggested_category, already_in_store
            indexing_candidates = [
                {"title": c["title"], "source_url": c["source_url"], "suggested_category": c["suggested_category"], "already_in_store": c.get("already_in_store", False)}
                for c in indexing_candidates
            ]
            n_ready = sum(1 for c in indexing_candidates if not c.get("already_in_store"))
            n_in_store = len(indexing_candidates) - n_ready
            total = len(raw_candidates)
            already_titles = [c["title"] for c in indexing_candidates if c.get("already_in_store")]
            progress.add_step(
                f"Pending indexing list ({n_ready}/{total}) — {n_ready} ready for indexing, {n_in_store} already in library",
                {"current": n_ready, "total": total, "ready": n_ready, "already_in_store": n_in_store, "already_in_library_titles": already_titles},
            )
            _emit_progress()
            # Only send ready-to-index items to frontend; "already in library" shown in Progress tracker only
            indexing_candidates = [c for c in indexing_candidates if not c.get("already_in_store")]
            # Data-driven no-materials summary: what we actually found/shortlisted/proposed/already in library
            web_search_stats = {
                "web_found": web_bare_count + web_case_count,
                "shortlisted": len(enriched_bare) + len(enriched_case),
                "proposed": n_ready,
                "already_in_library": n_in_store,
            }
        except Exception as ex:
            logger.warning("Indexing duplicate check failed: %s", ex)
            indexing_candidates = [{"title": c["title"], "source_url": c["source_url"], "suggested_category": c["suggested_category"], "already_in_store": False} for c in raw_candidates]
            total = len(raw_candidates)
            progress.add_step(
                f"Pending indexing list ({total}/{total}) — {total} ready for indexing (duplicate check skipped)",
                {"current": total, "total": total, "ready": total, "already_in_store": 0},
            )
            _emit_progress()
            web_search_stats = {
                "web_found": web_bare_count + web_case_count,
                "shortlisted": len(enriched_bare) + len(enriched_case),
                "proposed": total,
                "already_in_library": 0,
            }

    # When user has no materials for the answer: use data-driven summary if we have web search stats, else fix message for web_only
    if explanation and "I don't have any data" in explanation:
        explanation = _format_no_materials_message(search_strategy, web_search_stats)

    # Collect source tags for transparency
    sources_used = set()
    for ba in all_bare_acts:
        sources_used.add(ba.get("source_tag", "LOCAL_DB"))
        for cl in ba.get("related_case_laws", []):
            sources_used.add(cl.get("source_tag", "LOCAL_DB"))

    return {
        "bare_act_sections": all_bare_acts,  # Now includes nested case_laws
        "case_laws": [],  # Empty - case laws are now nested under bare acts to avoid duplicates
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "needs_confirmation": False,
        "internet_case_laws": [],  # backward compat
        "progress": progress.get_progress(),  # Add progress tracking data
        "indexing_candidates": indexing_candidates,
    }


# ---------------------------------------------------------------------------
# Case-level diversity (avoid one case dominating the list)
# ---------------------------------------------------------------------------

def _diversify_case_laws_by_case(
    case_laws: list,
    max_chunks_per_case: int = 4,
    max_total: int = 50,
) -> list:
    """
    Reduce dominance of a single case: cap chunks per case so more distinct
    cases appear in the result. Input must be sorted by _rerank_score desc.
    """
    if not case_laws:
        return []
    per_case_count = {}
    result = []
    for cl in case_laws:
        if len(result) >= max_total:
            break
        case_key = (
            (cl.get("case_name") or cl.get("source") or "").strip(),
            (cl.get("court") or "").strip(),
            str(cl.get("year") or ""),
            (cl.get("source") or cl.get("source_file") or "").strip(),
        )
        n = per_case_count.get(case_key, 0)
        if n >= max_chunks_per_case:
            continue
        per_case_count[case_key] = n + 1
        result.append(cl)
    if result != case_laws[:len(result)]:
        logger.info(
            "Case diversity: %d chunks from %d cases (max %d per case, cap %d total)",
            len(result), len(per_case_count), max_chunks_per_case, max_total,
        )
    return result


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
    
    # For each bare act, get its top max_per_section case laws by score (highest match first)
    # No duplicates: each case law appears at most once (under its best-matching section)
    assigned_case_laws = set()
    
    for ba_idx in range(len(bare_acts)):
        bare_acts[ba_idx]["related_case_laws"] = []
        # All matches for this bare act: (cl_idx, score)
        ba_matches = [(cl_idx, score) for bai, cl_idx, score in matches if bai == ba_idx]
        ba_matches.sort(key=lambda x: x[1], reverse=True)  # Highest score first
        count = 0
        for cl_idx, score in ba_matches:
            if count >= max_per_section:
                break
            if cl_idx in assigned_case_laws:
                continue
            bare_acts[ba_idx]["related_case_laws"].append(case_laws[cl_idx])
            assigned_case_laws.add(cl_idx)
            count += 1
    
    total_assigned = sum(len(ba.get("related_case_laws", [])) for ba in bare_acts)
    logger.info(f"Matched {total_assigned} case laws to {len(bare_acts)} bare act sections (up to {max_per_section} per section)")
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

        # Use web URL if present; else file:// link to local/Drive PDF so hyperlinks always work
        url = ba.get("url", "")
        if not url:
            src_file = ba.get("source_file") or ba.get("source") or ""
            if src_file:
                file_path = Path(BARE_ACTS_DIR) / src_file
                url = file_path.as_uri()
            if not url:
                url = GOOGLE_DRIVE_BARE_ACTS_FOLDER_URL
        formatted.append({
            "source": ba.get("source_file", ba.get("source", "")),
            "text": text,
            "act_name": act_name,
            "section_number": section,
            "title": display_title,
            "url": url,
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

        # Use web URL if present; else file:// link to local/Drive PDF so hyperlinks always work
        url = first.get("url") or first.get("source_url") or ""
        if not url:
            src_file = first.get("source_file") or first.get("source") or ""
            if src_file:
                file_path = Path(CASELAW_DIR) / src_file
                url = file_path.as_uri()
            if not url:
                url = GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL

        formatted.append({
            "source": first.get("source_file", first.get("source", "")),
            "text": summary_body,
            "case_name": case_name,
            "title": display_title,
            "court": court,
            "year": year,
            "citation": citation,
            "binding_authority": first.get("binding_authority", ""),
            "url": url,
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
    on_before_llm=None,
) -> str:
    """Generate a formal legal opinion with citations and source tags. on_before_llm(prompt) is called before LLM if provided."""
    # Check if we have any materials at all
    has_bare_acts = bool(bare_acts and len(bare_acts) > 0)
    has_case_laws = bool(case_laws and len(case_laws) > 0)
    
    # If no materials at all, return fixed "I don't have any data" — do NOT call LLM (no hallucination risk)
    if not has_bare_acts and not has_case_laws:
        return f"""## Brief Facts
{facts[:200]}

## Analysis and Conclusion
{RELEVANCE_EXPLANATION_NO_MATERIALS}"""

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
    
    # Add explicit empty array indicators if needed
    bare_array_note = ""
    case_array_note = ""
    if not has_bare_acts:
        bare_array_note = "\n⚠️ NOTE: The BARE ACT SECTIONS array above is EMPTY ([]). Do NOT create an 'Applicable Statutory Provisions' section."
    if not has_case_laws:
        case_array_note = "\n⚠️ NOTE: The CASE LAWS array above is EMPTY ([]). Do NOT create a 'Relevant Case Law' section."

    prompt = f"""{RELEVANCE_EXPLANATION_SYSTEM}

CASE FACTS:
{facts[:1500]}

BARE ACT SECTIONS:
{bare_text}
{bare_array_note}

CASE LAWS:
{case_text}
{case_array_note}

CONFIDENCE LEVEL: {confidence}

IMPORTANT: 
- Only cite sources that appear in the arrays above. If an array is empty ([]), do not create that section.
- For each legal statement that references retrieved materials, tag the citation with its source type:
  [LOCAL_DB] for materials from our verified database
  [OFFICIAL] for materials from government sources (legislation: state/central; judgments: courts)
  [LEGAL_PORTAL] for materials from legal portals
  [NEWS_REFERENCE] for newspaper articles (context only)
- If no materials were retrieved (empty arrays), do NOT add any source tags.

Generate the legal analysis:"""

    try:
        if callable(on_before_llm):
            on_before_llm(prompt)
        return ask_llm(prompt).strip()
    except Exception as e:
        logger.error(f"Opinion generation failed: {e}")
        return ""


def _generate_conversational_summary(
    facts: str,
    bare_acts: list,
    case_laws: list,
    intent: str = "search",
    on_before_llm=None,
) -> str:
    """Generate a conversational summary for search/lookup. For lookup use bare-act-only summary; for search use dispute+order/judgement in brief.
    When no materials exist for the requested type, return fixed 'I don't have any data' — do NOT call LLM (anti-hallucination). on_before_llm(prompt) is called before LLM if provided."""
    has_bare_acts = bool(bare_acts and len(bare_acts) > 0)
    has_case_laws = bool(case_laws and len(case_laws) > 0)

    # No-LLM path: when requested type has no materials, return fixed message
    if intent == "lookup" and not has_bare_acts:
        return RELEVANCE_EXPLANATION_NO_MATERIALS
    if intent == "search" and not has_case_laws:
        return RELEVANCE_EXPLANATION_NO_MATERIALS

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

    if intent == "lookup":
        system = BARE_ACT_ONLY_SUMMARY
        prompt = f"""{system}

USER QUERY: {facts[:500]}
BARE ACT SECTIONS FOUND: {bare_text if has_bare_acts else "[]"}

Summarise only the relevant bare act sections (no case laws):"""
    elif intent == "search":
        system = CASE_LAW_DISPUTE_ORDER_SUMMARY
        prompt = f"""{system}

USER QUERY: {facts[:500]}
CASE LAWS FOUND: {case_text if has_case_laws else "[]"}

For each case give: facts related to dispute + court order/judgement in brief:"""
    else:
        bare_array_note = "\n⚠️ NOTE: The BARE ACTS FOUND array above is EMPTY ([]). Do NOT claim you found bare act provisions." if not has_bare_acts else ""
        case_array_note = "\n⚠️ NOTE: The CASE LAWS FOUND array above is EMPTY ([]). Do NOT claim you found case laws." if not has_case_laws else ""
        prompt = f"""{CONVERSATIONAL_SUMMARY_SYSTEM}

USER QUERY: {facts[:500]}
BARE ACTS FOUND: {bare_text}
{bare_array_note}
CASE LAWS FOUND: {case_text}
{case_array_note}

Response:"""

    try:
        if callable(on_before_llm):
            on_before_llm(prompt)
        return ask_llm(prompt).strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return ""


def _format_no_materials_message(search_strategy: str, web_search_stats: Optional[dict]) -> str:
    """
    Return a truthful, data-driven message when no materials are shown in the answer.
    When web search was used, report actual counts (found, shortlisted, proposed, already in library)
    so the user sees what happened instead of a static 'I don't have any data'.
    When user asked for web_only, never claim we 'searched the internal vector store'.
    """
    if web_search_stats is not None:
        wf = web_search_stats.get("web_found", 0)
        sl = web_search_stats.get("shortlisted", 0)
        prop = web_search_stats.get("proposed", 0)
        already = web_search_stats.get("already_in_library", 0)
        if wf == 0 and sl == 0:
            if search_strategy == "web_only":
                return (
                    "I searched the web only (as you requested) but found no acts or laws. "
                    "Try rephrasing with specific section numbers, Act names, or a different legal angle."
                )
            return (
                "I searched the web but found no results. "
                "Try rephrasing with specific section numbers, Act names, or a different legal angle."
            )
        return (
            f"I searched the web and found {wf} act(s)/law(s). After relevance checks, {sl} were shortlisted. "
            f"I'm proposing {prop} for indexing; {already} are already in your library. "
            "None of the retrieved materials met the relevance threshold for the answer above—try rephrasing with specific section numbers, Act names, or a different legal angle."
        )
    if search_strategy == "web_only":
        return (
            "I searched the web only (as you requested) but found no relevant bare act provisions or case laws. "
            "Try rephrasing with specific section numbers, Act names, or a different legal angle."
        )
    return RELEVANCE_EXPLANATION_NO_MATERIALS


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
