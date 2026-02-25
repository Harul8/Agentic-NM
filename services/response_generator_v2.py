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
# Per-Dispute Retrieval Helpers (dispute-first pipeline)
# ---------------------------------------------------------------------------

def _summarize_bare_acts_brief(bare_acts: list) -> str:
    """Compact bare act summary for the sufficiency prompt (act + section only)."""
    if not bare_acts:
        return "NONE FOUND"
    lines = []
    for i, ba in enumerate(bare_acts[:25]):
        act = ba.get("act_name", "")
        sec = ba.get("section_number", "")
        title = ba.get("section_title", "")
        entry = f"{i + 1}. {act}"
        if sec:
            entry += f" Section {sec}"
        if title:
            entry += f" — {title}"
        lines.append(entry)
    return "\n".join(lines)


def _check_bare_act_sufficiency(dispute_text: str, bare_acts: list, llm_fn=None) -> bool:
    """
    Lightweight LLM check: are the retrieved bare act sections sufficient for this dispute?
    Returns True  → stop, do not go to web.
    Returns False → proceed to Round 2 web search.
    Defaults to False on any error so we err on the side of searching more.
    """
    if llm_fn is None:
        from llm.ollama_client import ask_llm
        llm_fn = ask_llm
    from prompts.advocate_prompts import BARE_ACT_DISPUTE_SUFFICIENCY_PROMPT

    ba_summary = _summarize_bare_acts_brief(bare_acts)
    prompt = BARE_ACT_DISPUTE_SUFFICIENCY_PROMPT.format(
        dispute=dispute_text[:500],
        bare_acts=ba_summary[:2000],
    )
    try:
        import re as _re
        response = llm_fn(prompt).strip()
        match = _re.search(r"\{[^}]+\}", response)
        if match:
            import json as _json
            result = _json.loads(match.group(0))
            sufficient = bool(result.get("sufficient", False))
            logger.info(
                "Bare act sufficiency: sufficient=%s reason=%s",
                sufficient, result.get("reason", "")
            )
            return sufficient
        # Plain-text fallback
        return "sufficient" in response.lower() and "not sufficient" not in response.lower()
    except Exception as e:
        logger.warning("Bare act sufficiency check failed (%s); defaulting to False (go to web)", e)
        return False


def _web_search_bare_acts(dispute: dict, full_query: str) -> list:
    """
    Web search for bare acts targeting one dispute component.
    Returns list of bare-act-like dicts with at least: act_name, text, source_tag, _rerank_score.
    """
    from retrieval.tiered_search import search_for_gaps
    from retrieval.auto_enricher import enrich_from_gap_results

    dispute_text = dispute.get("dispute", "")
    keywords = " ".join(dispute.get("keywords", []))
    gap_query = f"{dispute_text} {keywords} India bare act legislation section".strip()[:400]

    gaps = [{"query": gap_query, "type": "bare_act"}]
    try:
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrichment = enrich_from_gap_results(
            gap_results,
            original_query=full_query,
            local_high_quality_count=0,
            target_high_quality=0,
            skip_index=True,
        )
    except Exception as e:
        logger.error("Web search bare acts failed for dispute '%s': %s", dispute_text[:60], e)
        return []

    results = []
    for enriched in enrichment.get("enriched_bare_acts", []):
        content = enriched.get("content", "").strip()
        if not content:
            continue
        results.append({
            "act_name": enriched.get("title", "Unknown"),
            "section_number": "",          # web results rarely have parsed section numbers
            "section_title": "",
            "text": content[:2000],
            "full_text": content[:2000],
            "source_tag": enriched.get("source_tag", "LEGAL_PORTAL"),
            "url": enriched.get("url", ""),
            "_rerank_score": float(enriched.get("_rerank_score", MIN_RERANK_SCORE)),
        })
    logger.info(
        "Web bare acts for dispute '%s': %d results", dispute_text[:60], len(results)
    )
    return results


def retrieve_bare_acts_for_dispute(dispute: dict, full_query: str) -> list:
    """
    Retrieve ALL relevant bare act sections for a single dispute component.

    Round 1 — local hybrid search:
        → quality filter (score >= MIN_RERANK_SCORE, _is_quality_bare_act)
        → if any result score > HIGH_QUALITY_SCORE AND LLM says sufficient → STOP
        → else → Round 2

    Round 2 — tiered web search (bare acts only):
        → merge with local, deduplicate
        → STOP (hard ceiling; no further rounds)

    No result cap — every section that passes the quality filter is returned.
    """
    from retrieval.hybrid_retriever import search_bare_acts_auto

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    keywords = " ".join(dispute.get("keywords", []))
    search_query = f"{dispute_text} {keywords}".strip()[:400]

    # --- Round 1: local ---
    logger.info("[%s] Bare acts Round 1 — local hybrid search: '%s'", dispute_id, search_query[:80])
    local_raw = search_bare_acts_auto(search_query, top_k=50)
    local_results = [
        ba for ba in local_raw
        if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)
    ]
    logger.info("[%s] Bare acts Round 1 local: %d sections after quality filter", dispute_id, len(local_results))

    # Stop condition: high-quality results found AND LLM confirms sufficient
    high_quality_local = [ba for ba in local_results if ba.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    if high_quality_local and _check_bare_act_sufficiency(dispute_text, local_results):
        logger.info(
            "[%s] Bare acts Round 1 sufficient (%d sections, %d high-quality). Stopping.",
            dispute_id, len(local_results), len(high_quality_local),
        )
        return local_results

    # --- Round 2: web search ---
    logger.info("[%s] Bare acts Round 2 — web search (local insufficient)", dispute_id)
    web_results = _web_search_bare_acts(dispute, full_query)

    # Merge + deduplicate by act_name + section_number
    merged = _merge_deduplicate_bare_acts(local_results, web_results)
    logger.info(
        "[%s] Bare acts Round 2 complete: %d total (local=%d web=%d)",
        dispute_id, len(merged), len(local_results), len(web_results),
    )
    return merged


def _build_case_law_query(dispute_text: str, bare_act_sections: list) -> str:
    """
    Build case law search query: dispute + act names + section numbers.
    Graceful degradation:
      Tier 1 — dispute + act name + section number  (richest)
      Tier 2 — dispute + act name only              (section missing)
      Tier 3 — dispute only                         (no useful bare act metadata)
    Always returns a non-empty string.
    """
    parts = [dispute_text]
    for ba in bare_act_sections[:5]:   # cap to top 5 most relevant sections
        act = (ba.get("act_name") or "").strip()
        sec = (ba.get("section_number") or "").strip()
        if act and sec:
            parts.append(f"{act} Section {sec}")
        elif act:
            parts.append(act)
    return " ".join(parts).strip()[:500] or dispute_text[:300]


def _apply_dispute_case_law_limit(case_laws: list) -> list:
    """
    Threshold-based limit for per-dispute case laws:
      • If any result scores > HIGH_QUALITY_SCORE (5.0): keep top 5 of those.
      • Otherwise: keep top 2 that pass MIN_RERANK_SCORE.
    Input must already be quality-filtered.
    """
    if not case_laws:
        return []
    sorted_cls = sorted(case_laws, key=lambda x: x.get("_rerank_score", 0), reverse=True)
    high_quality = [cl for cl in sorted_cls if cl.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    if high_quality:
        return high_quality[:5]
    return sorted_cls[:2]


def _web_search_case_laws(dispute: dict, bare_act_sections: list, full_query: str) -> list:
    """
    Web search for case laws for one dispute, using the richer dispute+sections query.
    Returns list of case-law-like dicts.
    """
    from retrieval.tiered_search import search_for_gaps
    from retrieval.auto_enricher import enrich_from_gap_results

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    gap_query = _build_case_law_query(dispute_text, bare_act_sections)
    gap_query = f"{gap_query} Supreme Court High Court judgment India".strip()[:400]

    gaps = [{"query": gap_query, "type": "case_law"}]
    try:
        gap_results = search_for_gaps(gaps, jurisdiction_state="")
        enrichment = enrich_from_gap_results(
            gap_results,
            original_query=full_query,
            local_high_quality_count=0,
            target_high_quality=TARGET_HIGH_QUALITY_CASE_LAWS,
            skip_index=True,
        )
    except Exception as e:
        logger.error("Web search case laws failed for dispute '%s': %s", dispute_text[:60], e)
        return []

    results = []
    for enriched in enrichment.get("enriched_case_laws", []):
        content = enriched.get("content", "").strip()
        score = float(enriched.get("_rerank_score", 0))
        if not content or score < WEB_MIN_SCORE:
            continue
        results.append({
            "case_name": enriched.get("title", "Unknown"),
            "court": "",
            "year": "",
            "text": content[:2000],
            "full_text": content[:2000],
            "source_tag": enriched.get("source_tag", "LEGAL_PORTAL"),
            "url": enriched.get("url", ""),
            "_rerank_score": score,
        })
    logger.info(
        "[%s] Web case laws for dispute '%s': %d results", dispute_id, dispute_text[:60], len(results)
    )
    return results


def retrieve_case_laws_for_dispute(dispute: dict, bare_act_sections: list, full_query: str) -> list:
    """
    Retrieve case laws for a single dispute component.

    Query is built from dispute + act names + section numbers (graceful degradation
    to dispute-only if bare act metadata is missing).

    Round 1 — local hybrid search:
        → quality filter
        → apply threshold limit (top 5 if high-quality, else top 2)
        → if already at 5 → STOP

    Round 2 — tiered web search (case laws only):
        → merge with local, deduplicate
        → re-apply threshold limit
        → STOP (hard ceiling)
    """
    from retrieval.hybrid_retriever import search_case_laws_auto

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    search_query = _build_case_law_query(dispute_text, bare_act_sections)

    # --- Round 1: local ---
    logger.info("[%s] Case laws Round 1 — local hybrid search: '%s'", dispute_id, search_query[:80])
    local_raw = search_case_laws_auto(search_query, top_k=50)
    local_results = [
        cl for cl in local_raw
        if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
    ]
    limited = _apply_dispute_case_law_limit(local_results)
    logger.info("[%s] Case laws Round 1 local: %d after filter, %d after limit", dispute_id, len(local_results), len(limited))

    # If we got the maximum (5 high-quality), stop
    if len(limited) >= 5:
        logger.info("[%s] Case laws Round 1 sufficient (5 high-quality). Stopping.", dispute_id)
        return limited

    # --- Round 2: web search ---
    logger.info("[%s] Case laws Round 2 — web search (local had %d)", dispute_id, len(limited))
    web_results = _web_search_case_laws(dispute, bare_act_sections, full_query)

    # Merge, deduplicate, re-apply limit
    merged = _merge_deduplicate_case_laws(local_results, web_results)
    merged_filtered = [
        cl for cl in merged
        if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
    ]
    final = _apply_dispute_case_law_limit(merged_filtered)
    logger.info(
        "[%s] Case laws Round 2 complete: %d final (local=%d web=%d)",
        dispute_id, len(final), len(local_results), len(web_results),
    )
    return final


def _merge_deduplicate_bare_acts(local: list, web: list) -> list:
    """
    Merge local + web bare acts, deduplicating by act_name + section_number.
    Local results take precedence (web result dropped if local already has the section).
    Result is sorted by _rerank_score descending.
    """
    seen = set()
    merged = []
    for ba in local + web:
        key = (
            (ba.get("act_name") or "").strip().lower(),
            (ba.get("section_number") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(ba)
    merged.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    return merged


def _merge_deduplicate_case_laws(local: list, web: list) -> list:
    """
    Merge local + web case laws, deduplicating by normalised case name.
    Local results take precedence. Sorted by _rerank_score descending.
    """
    def _norm(name: str) -> str:
        n = (name or "").lower().strip()
        for sep in (" v. ", " vs. ", " v/s ", " versus "):
            n = n.replace(sep, " v ")
        return " ".join(n.split())

    seen = set()
    merged = []
    for cl in local + web:
        name = _norm(cl.get("case_name") or cl.get("source") or cl.get("title") or "")
        # Accept unnamed chunks (rare) but don't dedup them by empty key
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        merged.append(cl)
    merged.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    return merged


def _deduplicate_bare_acts(bare_acts: list) -> list:
    """Deduplicate a flat list of bare acts (across disputes) by act + section."""
    return _merge_deduplicate_bare_acts(bare_acts, [])


def _deduplicate_case_laws(case_laws: list) -> list:
    """Deduplicate a flat list of case laws (across disputes) by normalised name."""
    return _merge_deduplicate_case_laws(case_laws, [])


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
    Full legal research pipeline — dispute-first approach.

    New flow (replaces the old single-pass sufficiency-driven search):
    1. Expand the query via LLM for richer legal search terms
    2. Decompose the situation into distinct dispute components (LLM, legal_opinion only)
    3. For each dispute independently:
       a. retrieve_bare_acts_for_dispute  (local → sufficiency check → web, max 2 rounds)
       b. retrieve_case_laws_for_dispute  (local → threshold limit  → web, max 2 rounds)
    4. Aggregate + deduplicate across all disputes
    5. Format sections, match case laws, generate LLM opinion
    6. Return structured response (same shape as before + dispute_breakdown)

    search_strategy:
      local_then_web (default) — both rounds active per dispute
      local_only               — Round 2 (web) disabled; stops after local search
      web_only                 — Round 1 (local) disabled; web-only for each dispute

    Args:
        facts_summary:       Plain language case facts / user query
        jurisdiction_state:  State for HC-specific searches (e.g., "Karnataka")
        intent:              "legal_opinion" | "search" | "lookup"
        confirmed_materials: Pre-confirmed materials to inject (from confirmation flow)
        result_count:        User-specified limit (search/lookup only)
        progress_callback:   Optional callable(snapshot) for streaming progress
        document_types:      "both" | "acts_only" | "case_laws_only"
        search_strategy:     "local_then_web" | "local_only" | "web_only"
    """
    progress = ProgressTracker()

    def _emit_progress():
        if progress_callback:
            try:
                progress_callback(progress.get_progress_snapshot())
            except Exception:
                pass

    if search_strategy not in ("local_only", "web_only", "local_then_web"):
        search_strategy = "local_then_web"

    # Intent extraction — used for query expansion
    research_intent = None
    try:
        from services.intent_extractor import extract_research_intent
        research_intent = extract_research_intent(facts_summary)
    except Exception as e:
        logger.debug("Intent extraction skipped: %s", e)

    # Step 1: Query expansion
    legal_query = expand_legal_query(facts_summary, intent=research_intent)
    logger.info("Expanded query: %s", legal_query[:200])

    # What to retrieve (acts vs case laws vs both)
    retrieve_acts = (
        intent == "lookup"
        or document_types == "acts_only"
        or (intent == "legal_opinion" and document_types != "case_laws_only")
    )
    retrieve_case_laws_flag = (
        intent == "search"
        or document_types == "case_laws_only"
        or (intent == "legal_opinion" and document_types != "acts_only")
    )
    if not retrieve_acts and not retrieve_case_laws_flag:
        retrieve_acts = retrieve_case_laws_flag = True

    # Step 2: Dispute decomposition
    progress.start_group("Dispute Analysis", "Identifying distinct dispute components")
    _emit_progress()

    if intent in ("search", "lookup"):
        # Direct search/lookup — no decomposition needed
        disputes = [{"id": "d1", "dispute": facts_summary[:300], "legal_nature": "both", "keywords": []}]
        progress.add_step("Direct search/lookup — treating as single query", {"disputes": 1})
    else:
        from services.dispute_decomposer import decompose_disputes
        disputes = decompose_disputes(facts_summary)
        labels = [d.get("dispute", "")[:60] for d in disputes]
        progress.add_step(
            f"Identified {len(disputes)} dispute component(s)",
            {"count": len(disputes), "disputes": labels},
        )
        logger.info("Disputes identified: %s", labels)

    _emit_progress()
    progress.finish_group()
    _emit_progress()

    # Step 3: Per-dispute retrieval
    dispute_results = []

    for dispute in disputes:
        d_id = dispute.get("id", "?")
        d_text = dispute.get("dispute", "")

        progress.start_group(
            f"Research [{d_id}]: {d_text[:50]}",
            "Retrieving bare acts and case laws for this dispute",
        )
        _emit_progress()

        # 3a: Bare acts
        bare_d = []
        if retrieve_acts:
            progress.add_step(f"[{d_id}] Searching bare acts...")
            _emit_progress()

            if search_strategy == "web_only":
                bare_d = _web_search_bare_acts(dispute, facts_summary)
                bare_d = [
                    ba for ba in bare_d
                    if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)
                ]
            elif search_strategy == "local_only":
                from retrieval.hybrid_retriever import search_bare_acts_auto
                kw = " ".join(dispute.get("keywords", []))
                q = f"{d_text} {kw}".strip()[:400]
                raw = search_bare_acts_auto(q, top_k=50)
                bare_d = [
                    ba for ba in raw
                    if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)
                ]
            else:
                bare_d = retrieve_bare_acts_for_dispute(dispute, facts_summary)

            progress.add_step(
                f"[{d_id}] Found {len(bare_d)} bare act section(s)",
                {"count": len(bare_d), "dispute": d_text[:60]},
            )
            _emit_progress()

        # 3b: Case laws
        case_d = []
        if retrieve_case_laws_flag:
            progress.add_step(f"[{d_id}] Searching case laws...")
            _emit_progress()

            if search_strategy == "web_only":
                raw_cl = _web_search_case_laws(dispute, bare_d, facts_summary)
                case_d = _apply_dispute_case_law_limit([
                    cl for cl in raw_cl
                    if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
                ])
            elif search_strategy == "local_only":
                from retrieval.hybrid_retriever import search_case_laws_auto
                q = _build_case_law_query(d_text, bare_d)
                raw_cl = search_case_laws_auto(q, top_k=50)
                case_d = _apply_dispute_case_law_limit([
                    cl for cl in raw_cl
                    if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
                ])
            else:
                case_d = retrieve_case_laws_for_dispute(dispute, bare_d, facts_summary)

            progress.add_step(
                f"[{d_id}] Found {len(case_d)} case law(s)",
                {"count": len(case_d), "dispute": d_text[:60]},
            )
            _emit_progress()

        progress.finish_group()
        _emit_progress()

        dispute_results.append({"dispute": dispute, "bare_acts": bare_d, "case_laws": case_d})

    # Step 4: Aggregate + deduplicate across disputes
    all_bare_raw = []
    all_case_raw = []
    for dr in dispute_results:
        all_bare_raw.extend(dr["bare_acts"])
        all_case_raw.extend(dr["case_laws"])

    all_bare_raw = _deduplicate_bare_acts(all_bare_raw)
    all_case_raw = _deduplicate_case_laws(all_case_raw)

    if confirmed_materials:
        all_bare_raw, all_case_raw = _add_confirmed_materials(confirmed_materials, all_bare_raw, all_case_raw)

    # Step 5: Format sections, match case laws, generate opinion
    progress.start_group("Response", "Formatting results and generating legal opinion")
    progress.add_step("Formatting sections and matching case laws to bare acts...")
    _emit_progress()

    all_case_raw_div = _diversify_case_laws_by_case(
        all_case_raw, max_chunks_per_case=4, max_total=max(len(all_bare_raw) * 6, 30),
    )
    formatted_bare = _format_bare_acts(all_bare_raw)
    formatted_case = _format_case_laws(all_case_raw_div, user_query=facts_summary)
    formatted_bare = _match_case_laws_to_bare_acts(formatted_bare, formatted_case)

    # Search-only: wrap case laws in a bare-act shell so UI renders them
    if intent == "search" and not formatted_bare and formatted_case:
        limit = max(1, result_count) if result_count else max(FLEXIBLE_MIN_FALLBACK, len(formatted_case))
        formatted_bare = [{
            "act_name": "Case laws",
            "title": "Case laws (search results)",
            "text": "",
            "related_case_laws": formatted_case[:limit],
            "source_tag": "LOCAL_DB",
        }]

    # Step 6: LLM opinion / summary
    if intent in ("search", "lookup"):
        progress.add_step("Generating search summary...")
    else:
        progress.add_step("Generating legal opinion...")
    _emit_progress()

    flattened_case_laws = []
    for ba in formatted_bare:
        flattened_case_laws.extend(ba.get("related_case_laws", []))

    # Model display step (before blocking on LLM call)
    _ctx = (facts_summary or "")[:1500]
    _ctx += json.dumps([{"t": b.get("title", ""), "x": (b.get("text") or "")[:500]} for b in formatted_bare[:15]])[:3000]
    _ctx += json.dumps([{"t": c.get("title", ""), "x": (c.get("text") or "")[:500]} for c in flattened_case_laws[:15]])[:3000]
    _display, _switched = get_model_display_for_prompt(_ctx)
    if _display:
        _action = "Switching to" if _switched else "Calling"
        _msg = f"{_action} {_display} model"
        _gpu = get_gpu_info()
        if _gpu:
            _msg += f", GPU: {', '.join(_gpu)}"
        progress.add_step(_msg)
        _emit_progress()

    # Sufficiency is satisfied by design (each dispute ran its own rounds)
    sufficiency = {
        "overall_sufficient": bool(formatted_bare),
        "gaps": [],
        "aspects": [],
        "confidence": "medium",
    }

    if intent in ("search", "lookup"):
        explanation = _generate_conversational_summary(
            facts_summary, formatted_bare, flattened_case_laws, intent=intent
        )
    else:
        explanation = _generate_legal_opinion(
            facts_summary, formatted_bare, flattened_case_laws, sufficiency
        )

    if not (explanation or "").strip():
        explanation = "Here's what I found for your query. Below are the relevant legal provisions with related case laws."
    if explanation and "I don't have any data" in explanation:
        explanation = _format_no_materials_message(search_strategy, None)

    sources_used = set()
    for ba in formatted_bare:
        sources_used.add(ba.get("source_tag", "LOCAL_DB"))
        for cl in ba.get("related_case_laws", []):
            sources_used.add(cl.get("source_tag", "LOCAL_DB"))

    progress.finish_group()
    _emit_progress()

    return {
        "bare_act_sections": formatted_bare,
        "case_laws": [],                      # case laws are nested under bare acts
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "needs_confirmation": False,
        "internet_case_laws": [],             # backward compat
        "progress": progress.get_progress(),
        "indexing_candidates": [],            # TODO: collect from per-dispute web enrichments
        "dispute_breakdown": [
            {
                "dispute_id": dr["dispute"].get("id"),
                "dispute": dr["dispute"].get("dispute"),
                "bare_acts_count": len(dr["bare_acts"]),
                "case_laws_count": len(dr["case_laws"]),
            }
            for dr in dispute_results
        ],
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
