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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import (
    BARE_ACTS_DIR,
    CASELAW_DIR,
    GOOGLE_DRIVE_BARE_ACTS_FOLDER_URL,
    GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL,
)
from llm.ollama_client import ask_llm, ask_llm_stream, get_model_display_for_prompt, get_gpu_info
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
    STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT,
    ACT_SELECTION_PROMPT,
    BARE_ACT_SECTION_RELEVANCE_PROMPT,
    CASE_LAW_RELEVANCE_PROMPT,
)
from services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# Relevance and quality thresholds — keep only high-quality, recent materials
MIN_RERANK_SCORE = 0.5  # Minimum score to include in pool. ms-marco cross-encoder logits:
                        #   >2.0 = clearly relevant; 0.5-2 = borderline; <0.5 = noise / off-topic.
                        # 0.5 formally excludes near-zero scoring sections while keeping
                        # BNSS/BNS/BSA sections (typical scores 0.5–3.0 for plain-language queries).
                        # The per-dispute hard cap (MAX_SECTIONS_PER_DISPUTE_TOTAL) then keeps
                        # only the top N by score. Raised from 0.0 (allowed all noise) to 0.5.
HIGH_QUALITY_SCORE = 5.0  # Case laws with score > this: "highly relevant"; stop web search if we have 5+; index web PDFs only if > this
BARE_ACT_HIGH_QUALITY_SCORE = 2.0  # Bare-act equivalent. ms-marco scores for on-topic legal sections
                                   # cluster around 2–3; using 2.0 (not 5.0) as the HQ threshold for
                                   # bare acts so the auto-stop and LLM-sufficiency tiers fire correctly.
WEB_MIN_SCORE = 1.0  # Minimum cross-encoder score for web case laws (ms-marco logits; 1.0 keeps on-topic official SC judgments)
TARGET_HIGH_QUALITY_CASE_LAWS = 5  # Stop search once we have this many case laws with score > HIGH_QUALITY_SCORE
MIN_CASE_YEAR = 1975 # Prefer cases from last 40 years; older treated as low priority
JUNK_CASE_PATTERNS = ("unknown", "unknown (", "high court, 1908", "high court, 1972")

# When user does NOT set a limit: include all with score > HIGH_QUALITY_SCORE; if fewer than this, add from >= MIN_RERANK_SCORE up to this many
FLEXIBLE_MIN_FALLBACK = 10  # Minimum results to show when no user limit and not enough high-similarity (all must pass MIN_RERANK_SCORE)

# Section caps — keep retrieval focused while still allowing connected statutory provisions
# to appear together in the final opinion for one dispute.
MAX_WEB_SECTIONS_PER_DISPUTE = 5    # Top N chunks kept from web search per dispute (by doc/section score)
MAX_SECTIONS_PER_DISPUTE_TOTAL = 5  # Hard cap per dispute after local+web+xref merge (top N by score)
MAX_BARE_ACTS_OVERALL = 12          # Safety cap on total sections sent to UI across all disputes

# Act-level pre-filter (for 100+ act indexes)
# An act qualifies if its highest-scoring section clears this threshold.
# ms-marco cross-encoder logits: 5.0 captures the "correct act, right domain" range
# while cleanly dropping wrong-domain acts (Constitution, POCSO, Arms Act for an
# assault query typically score 0–2, well below this bar).
ACT_SELECTION_MIN_SCORE = 5.0
ACT_SELECTION_MAX_ACTS  = 4   # Never take more than 4 even if many qualify

# Web search fallback threshold.
# If the BEST local section score is below this, the local index doesn't have a
# confident match — trigger web search regardless of section count.
# ms-marco logits: 8+ = "exact match" (BNS 115 for assault, TP Act 54 for sale).
# Only trigger web when local result count is low (roadmap: not by score threshold).
# Score-based trigger caused almost every request to hit the internet.
WEB_SEARCH_MIN_LOCAL_COUNT = 3  # if len(local_results) < this, allow web search
# Legacy: kept for logging; no longer used to force web
WEB_SEARCH_FALLBACK_MIN_SCORE = 8.0

# ---------------------------------------------------------------------------
# New criminal codes (2023) — keyword sets for automatic query injection
# ---------------------------------------------------------------------------
# IndiaCode generic searches like "assault India central act" return the Arms Act
# because it is a well-established, heavily-indexed act whose title contains "arms"
# and whose provisions mention "assault / violence".  BNS 2023 is a much newer
# addition and only surfaces when queried by its *explicit name*.
#
# These sets let us detect which of the three new criminal codes is relevant to a
# dispute and inject a name-explicit query so IndiaCode can find the right act.
#   BNS  = Bharatiya Nyaya Sanhita 2023   (substantive — replaces IPC)
#   BNSS = Bharatiya Nagarik Suraksha Sanhita 2023 (procedure — replaces CrPC)
#   BSA  = Bharatiya Sakshya Adhiniyam 2023         (evidence  — replaces IEA)

_BNS_KEYWORDS: frozenset = frozenset({
    "assault", "hurt", "grievous", "battery", "injury", "murder",
    "culpable homicide", "theft", "robbery", "dacoity", "rioting",
    "rape", "kidnapping", "abduction", "extortion", "cheating",
    "fraud", "forgery", "criminal intimidation", "criminal force",
    "wrongful confinement", "wrongful restraint", "mischief", "trespass",
    "defamation", "unlawful assembly", "sedition", "waging war",
    "death", "bodily harm", "physical harm", "beat", "beaten", "attack",
    "voluntarily", "intentional harm", "criminal liability",
})

_BNSS_KEYWORDS: frozenset = frozenset({
    "fir", "first information report", "arrest", "bail", "anticipatory bail",
    "remand", "chargesheet", "charge sheet", "cognizance", "summons",
    "warrant", "police custody", "judicial custody", "magistrate",
    "sessions court", "trial", "complaint", "investigation", "challan",
    "bailable", "non-bailable", "compoundable",
})

_BSA_KEYWORDS: frozenset = frozenset({
    "evidence", "witness", "testimony", "confession", "admission",
    "documentary evidence", "electronic record", "hearsay", "expert",
    "examination", "cross examination", "burden of proof", "presumption",
    "oral evidence", "primary evidence", "secondary evidence",
})


def _detect_new_criminal_codes(dispute_text: str, legal_nature: str) -> dict:
    """
    Detect which of the three new 2023 criminal codes apply to this dispute.

    Returns a dict with boolean flags:
        {"bns": bool, "bnss": bool, "bsa": bool}

    Detection is keyword-based so it stays deterministic (no LLM guessing).
    Only fires for "criminal" or "both" legal_nature disputes.
    """
    result = {"bns": False, "bnss": False, "bsa": False}
    if legal_nature not in ("criminal", "both"):
        return result
    text_lower = dispute_text.lower()
    result["bns"]  = any(kw in text_lower for kw in _BNS_KEYWORDS)
    result["bnss"] = any(kw in text_lower for kw in _BNSS_KEYWORDS)
    result["bsa"]  = any(kw in text_lower for kw in _BSA_KEYWORDS)
    return result


def _cap_bare_acts_by_score(bare_acts: list, cap: int) -> list:
    """Return top `cap` bare acts sorted by _rerank_score descending."""
    if not bare_acts or cap <= 0:
        return bare_acts
    return sorted(bare_acts, key=lambda x: x.get("_rerank_score", 0), reverse=True)[:cap]


_RE_DISALLOWED_LEGACY_CODES = re.compile(
    r"\b(?:IPC|Indian Penal Code|CrPC|Code of Criminal Procedure|IEA|Indian Evidence Act)\b",
    re.IGNORECASE,
)
_RE_SECTION_CITATION = re.compile(r"\bsections?\s+([0-9A-Za-z,\sand]+)", re.IGNORECASE)


def _local_only_no_materials_message() -> str:
    """Strict fallback when an answer cannot be grounded in the local vector store."""
    return (
        "I don't have grounded data for this query in the local vector store. "
        "I can only answer legal questions from locally retrieved bare act provisions and case laws. "
        "Please index the relevant materials or rephrase the query."
    )


def _is_local_db_source(item: dict) -> bool:
    """True when a retrieved item came from the local vector store."""
    return (item.get("source_tag") or "").strip().upper() == "LOCAL_DB"


def _filter_materials_to_local_db(bare_acts: list | None, case_laws: list | None) -> tuple[list, list]:
    """Keep only LOCAL_DB materials and drop nested related case laws from other sources."""
    local_bare: list = []
    for ba in bare_acts or []:
        if not _is_local_db_source(ba):
            continue
        clone = dict(ba)
        clone["related_case_laws"] = [
            cl for cl in (clone.get("related_case_laws") or [])
            if _is_local_db_source(cl)
        ]
        local_bare.append(clone)

    local_case = [cl for cl in (case_laws or []) if _is_local_db_source(cl)]
    return local_bare, local_case


def _filter_dispute_results_to_local_db(dispute_results: list | None) -> list:
    """Keep only local bare acts and case laws inside dispute-level retrieval results."""
    filtered: list = []
    for dr in dispute_results or []:
        local_bare, local_case = _filter_materials_to_local_db(
            dr.get("bare_acts"),
            dr.get("case_laws"),
        )
        clone = dict(dr)
        clone["bare_acts"] = local_bare
        clone["case_laws"] = local_case
        filtered.append(clone)
    return filtered


def _extract_cited_section_numbers(text: str) -> set[str]:
    """Extract section numbers mentioned in generated output."""
    found: set[str] = set()
    for match in _RE_SECTION_CITATION.finditer(text or ""):
        for sec in re.findall(r"\d+[A-Za-z]?", match.group(1), flags=re.IGNORECASE):
            found.add(sec.lower())
    return found


def _build_allowed_section_numbers(bare_acts: list | None) -> set[str]:
    """Build the set of section numbers actually present in retrieved local materials."""
    allowed: set[str] = set()
    for ba in bare_acts or []:
        sec = str(ba.get("section_number") or "").strip().lower()
        if sec:
            allowed.add(sec)
    return allowed


def _violates_local_grounding(text: str, bare_acts: list | None) -> bool:
    """Detect unsupported legal citations in model output."""
    if not (text or "").strip():
        return False
    if _RE_DISALLOWED_LEGACY_CODES.search(text):
        return True

    cited_sections = _extract_cited_section_numbers(text)
    allowed_sections = _build_allowed_section_numbers(bare_acts)
    if cited_sections and not allowed_sections:
        return True
    return any(sec not in allowed_sections for sec in cited_sections)


def _enforce_local_grounding(text: str, bare_acts: list | None) -> str:
    """Return a strict local-only fallback if the model output cites unsupported law."""
    cleaned = (text or "").strip()
    if _violates_local_grounding(cleaned, bare_acts):
        logger.warning("Blocked unsupported legal citations in generated output; returning local-only fallback")
        return _local_only_no_materials_message()
    return cleaned


# Keywords that identify rent/tenant/eviction disputes for act-preference (ToPA over Contract Act 294A)
_RENT_EVICTION_KEYWORDS = frozenset({
    "rent", "tenant", "landlord", "eviction", "lease", "tenancy",
    "non-payment", "non payment", "arrears", "vacat", "notice to quit",
    "lease agreement", "rental", "defaulted on rent",
})


def _is_rent_eviction_dispute(dispute: dict) -> bool:
    """True if this dispute is about rent, tenancy, eviction, or lease termination."""
    text = (dispute.get("dispute") or "").lower()
    keywords = [k.lower() for k in dispute.get("keywords") or []]
    combined = text + " " + " ".join(keywords)
    return any(kw in combined for kw in _RENT_EVICTION_KEYWORDS)


def _reorder_bare_acts_for_rent_disputes(bare_acts: list, dispute: dict) -> list:
    """
    For rent/eviction disputes: prefer Transfer of Property Act and state rent/eviction acts,
    demote Indian Contract Act § 294A (Contingent Contracts) since eviction is under ToPA/state law.
    Returns a new list (same items, reordered); does not change scores.
    """
    if not bare_acts or not _is_rent_eviction_dispute(dispute):
        return bare_acts

    def _act_priority(ba: dict) -> tuple:
        act = (ba.get("act_name") or "").strip().lower()
        sec = (ba.get("section_number") or "").strip().lower()
        score = ba.get("_rerank_score", 0.0)
        # Prefer ToPA and state acts with "rent" or "eviction" or "lease" in name
        if "transfer of property" in act:
            return (0, -score)   # first group, then by score desc
        if "rent" in act or "eviction" in act or "lease" in act or "tenancy" in act:
            return (1, -score)
        # Demote Contract Act 294A (not the primary remedy for eviction)
        if "contract act" in act or "indian contract" in act:
            if sec == "294a" or sec == "294":
                return (3, -score)   # last group
        return (2, -score)   # middle

    return sorted(bare_acts, key=_act_priority)


def _build_indexing_candidates_from_web_case_laws(case_laws: list) -> list:
    """
    Build indexing_candidates from web-sourced case laws for the Pending indexing UI.
    Only includes case laws with a fetchable URL (http/https) and source_tag != LOCAL_DB.
    Runs check_indexing_candidates to mark already_in_store.
    """
    if not case_laws:
        return []
    from services.indexing_duplicate_check import check_indexing_candidates

    candidates = []
    seen_urls = set()
    for cl in case_laws:
        url = (cl.get("url") or cl.get("source_url") or "").strip()
        if not url or not url.startswith(("http://", "https://")):
            continue
        if (cl.get("source_tag") or "").upper() == "LOCAL_DB":
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = (cl.get("case_name") or cl.get("title") or cl.get("source") or "Unknown").strip()
        content = (cl.get("text") or cl.get("full_text") or "")[:2000]
        candidates.append({
            "title": title,
            "source_url": url,
            "suggested_category": "case_law",
            "content": content,
        })
    return check_indexing_candidates(candidates)


def _build_indexing_candidates_from_pending_list(pending_list: list) -> list:
    """
    Build indexing_candidates from the pending_indexing_list (web search pending_only path).
    Each item is {title, source_url, suggested_category}. Dedupes by URL, runs
    check_indexing_candidates to mark already_in_store.
    """
    if not pending_list:
        return []
    from services.indexing_duplicate_check import check_indexing_candidates

    seen_urls = set()
    candidates = []
    for item in pending_list:
        url = (item.get("source_url") or "").strip()
        if not url or not url.startswith(("http://", "https://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        candidates.append({
            "title": (item.get("title") or "Unknown").strip(),
            "source_url": url,
            "suggested_category": (item.get("suggested_category") or "case_law").strip(),
            "content": item.get("content", ""),
        })
    return check_indexing_candidates(candidates)


def _select_top_acts(all_local: list) -> set:
    """
    Act-level pre-filter for a 100+ act index.

    With many acts indexed, a broad hybrid search returns sections from dozens of
    acts (POCSO, Arms Act, Constitution ...) that happen to share adjacent concepts.
    This function selects acts based on score quality, not a hard count:

    1. For each act, find its highest-scoring section across all queries.
    2. Keep only acts whose max score >= ACT_SELECTION_MIN_SCORE (5.0).
       ms-marco logits: 5+ = "confident match"; 0-2 = "adjacent terminology only".
    3. Cap at ACT_SELECTION_MAX_ACTS (4) to prevent runaway on edge cases.
    4. Fallback: if nothing clears the threshold (index has no strong local match),
       take the single best-scoring act rather than returning empty — empty would
       incorrectly suppress local results and skip straight to web search.

    Examples:
      BNS=9.8, BNSS=7.2, TP Act=6.1 → all three qualify (all ≥ 5.0)
      BNS=9.8, Arms Act=1.8          → only BNS qualifies; Arms Act dropped
      Succession Act=4.9, CPC=3.1    → nothing qualifies; take Succession Act as fallback
    """
    if not all_local:
        return set()

    act_max: dict[str, float] = {}
    for chunk in all_local:
        act = (chunk.get("act_name") or "").strip()
        if not act:
            continue
        score = chunk.get("_rerank_score", 0.0)
        if score > act_max.get(act, -999.0):
            act_max[act] = score

    # Acts that clear the quality bar
    qualifying = sorted(
        [(act, score) for act, score in act_max.items() if score >= ACT_SELECTION_MIN_SCORE],
        key=lambda x: x[1], reverse=True
    )

    if qualifying:
        top_acts = {act for act, _ in qualifying[:ACT_SELECTION_MAX_ACTS]}
    else:
        # Fallback: nothing clears the 5.0 threshold.
        # If the best act scores below 3.0 the whole local pool is the wrong domain
        # (e.g. Copyright Act at 1.45 for an assault query). Return empty set — the
        # caller treats empty as "no act filter applied", which means all local sections
        # survive. Since max_local_score < WEB_SEARCH_FALLBACK_MIN_SCORE (8.0), force_web
        # fires and web results dominate the capped list; the low-scoring local stragglers
        # rarely beat web results in the score-sorted cap.
        # Only use "best act" fallback when it's at least in the right ballpark (3.0–4.9):
        # that means roughly the right legal domain even if not the exact provision.
        best = max(act_max, key=act_max.get)
        best_score = act_max[best]
        if best_score < 3.0:
            logger.info(
                "Act pre-filter: best act '%s' scores %.2f < 3.0 — skip fallback (wrong domain); "
                "local results will not be filtered by act; web search will fill the gap.",
                best, best_score,
            )
            return set()
        top_acts = {best}
        logger.info(
            "Act pre-filter: no act cleared %.1f threshold; fallback to best act '%s' (score=%.2f)",
            ACT_SELECTION_MIN_SCORE, best, act_max[best],
        )

    logger.info(
        "Act pre-filter: %d/%d acts qualify (min_score=%.1f, cap=%d) → %s",
        len(qualifying), len(act_max), ACT_SELECTION_MIN_SCORE, ACT_SELECTION_MAX_ACTS, list(top_acts),
    )
    return top_acts


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


def _format_case_citation(cl: dict) -> str:
    """Format case for display: name + (year) + court when available, so LLM can cite fully."""
    name = (cl.get("case_name") or cl.get("title") or cl.get("source") or "Unknown").strip()
    year = cl.get("year")
    court = cl.get("court") or cl.get("binding_authority") or ""
    if year or court:
        parts = [name]
        if year:
            try:
                parts.append(f"({int(str(year).strip()[:4])})")
            except (ValueError, TypeError):
                pass
        if court:
            parts.append(f"— {court.strip()}")
        return " ".join(parts)
    return name


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

def expand_legal_query(facts: str, intent: dict = None) -> list[str]:
    """
    Convert plain-language facts to 1–3 legal research queries (roadmap: multi-query retrieval).

    Returns a list of up to 3 query strings: base (LLM or facts) + optional synonym variant
    + optional statute-style variant. Callers run hybrid retrieval per query, merge, dedupe, rerank.
    """
    from prompts.advocate_prompts import EXPAND_LEGAL_QUERY_SYSTEM, EXPAND_LEGAL_QUERY_INTENT_BLOCK

    # Legal synonym expansion (user phrase → statute-style terms)
    _SYNONYMS = {
        "rent": ["rent", "lease payment", "tenancy payment"],
        "eviction": ["eviction", "recovery of possession"],
        "threat": ["criminal intimidation", "threat of injury"],
        "land acquisition": ["compulsory acquisition", "state acquisition"],
        "tenant": ["tenant", "lessee"],
        "landlord": ["landlord", "lessor"],
        "defaulted": ["default in payment"],
        "not paid": ["default in payment"],
        "stolen": ["theft"],
        "cheat": ["cheating", "fraud"],
    }

    def _add_synonym_variant(text: str) -> str:
        t = text.lower()
        for phrase, replacements in _SYNONYMS.items():
            if phrase in t:
                for r in replacements:
                    if r not in t:
                        return text + " " + r  # append first missing synonym
        return ""

    intent_block = ""
    if intent and isinstance(intent, dict) and (intent.get("states") or intent.get("domains") or intent.get("topics")):
        intent_block = EXPAND_LEGAL_QUERY_INTENT_BLOCK.format(intent_json=json.dumps(intent, indent=0))

    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}
{intent_block}

FACTS:
{facts[:2000]}

Query:"""
    queries = []
    try:
        base = ask_llm(prompt).strip()[:500]
        if base:
            queries.append(base)
    except Exception:
        pass
    if not queries:
        queries.append(facts[:300])

    # Variant 2: synonym-expanded
    syn = _add_synonym_variant(queries[0])
    if syn and syn not in [q.lower() for q in queries]:
        queries.append(syn[:500])

    # Variant 3: statute-style (e.g. "default in payment of rent" if "rent not paid")
    if len(queries) < 3 and "default" not in queries[0].lower() and ("rent" in queries[0].lower() or "payment" in queries[0].lower()):
        statute_style = queries[0].replace("not paid", "default in payment").replace("did not pay", "default in payment")
        if statute_style != queries[0] and statute_style not in [q.lower() for q in queries]:
            queries.append(statute_style[:500])

    return queries[:3]


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


# ---------------------------------------------------------------------------
# Bare act multi-query helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IPC → BNS / CrPC → BNSS / IEA → BSA translation table
# Applied at query-build time so searches hit the NEW act names stored in the
# vector store rather than the old colonial-era names the LLM defaults to.
# ---------------------------------------------------------------------------
_OLD_TO_NEW_ACT = {
    # Criminal substantive
    "indian penal code":                  "Bharatiya Nyaya Sanhita",
    "ipc":                                "BNS",
    "ipc 1860":                           "Bharatiya Nyaya Sanhita 2023",
    "indian penal code, 1860":            "Bharatiya Nyaya Sanhita (BNS), 2023",
    "indian penal code 1860":             "Bharatiya Nyaya Sanhita (BNS), 2023",
    # Criminal procedure
    "code of criminal procedure":         "Bharatiya Nagarik Suraksha Sanhita",
    "crpc":                               "BNSS",
    "crpc, 1973":                         "Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023",
    "code of criminal procedure, 1973":   "Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023",
    # Evidence
    "indian evidence act":                "Bharatiya Sakshya Adhiniyam",
    "iea":                                "BSA",
    "indian evidence act, 1872":          "Bharatiya Sakshya Adhiniyam (BSA), 2023",
}

# IPC section number → BNS section number (most common criminal offences)
_IPC_SEC_TO_BNS = {
    "115": "115",   # already BNS
    "323": "115",   "324": "117",   "325": "117",   "326": "118",
    "307": "109",   "302": "103",   "304": "105",   "304b": "80",
    "341": "126",   "342": "127",   "351": "131",   "352": "132",
    "354": "74",    "376": "64",
    "378": "303",   "379": "303",   "380": "305",   "381": "306",
    "406": "316",   "420": "318",   "415": "318",
    "425": "324",   "426": "324",   "427": "324",   "436": "328",
    "441": "329",   "447": "329",   "448": "330",   "452": "332",
    "499": "356",   "500": "356",
    "503": "351",   "504": "352",   "506": "351",
}

_RE_IPC_SEC = re.compile(r'\b(?:ipc|i\.p\.c\.?)\s*s(?:ection)?\s*(\d+[a-z]?)\b', re.IGNORECASE)
_RE_SEC_NUM = re.compile(r'\bsection\s+(\d+[a-z]?)\b', re.IGNORECASE)


def _translate_act_hint(hint: str) -> str:
    """Translate one old-act name string to the new equivalent (or return as-is)."""
    low = hint.lower().strip()
    return _OLD_TO_NEW_ACT.get(low, hint)


def _inject_bns_equivalents(text: str) -> str:
    """
    For a query/hint that contains IPC section references, append BNS equivalents.
    E.g. "IPC Section 323 assault" → "IPC Section 323 assault BNS Section 115"
    """
    additions = []
    for m in _RE_IPC_SEC.finditer(text):
        ipc_sec = m.group(1).lower()
        bns_sec = _IPC_SEC_TO_BNS.get(ipc_sec)
        if bns_sec:
            additions.append(f"BNS Section {bns_sec}")
    if additions:
        return text + " " + " ".join(additions)
    return text


def _build_bare_act_queries(dispute: dict) -> list[str]:
    """
    Build 3-5 diverse search queries for bare act retrieval from a single dispute.

    All queries are derived from the user's own language — no IPC/BNS section numbers
    or act names are injected from training knowledge. Grounding in retrieved data only.

    Sources:
    1. dispute_text + keywords (always included — broadest query)
    2. search_angles from dispute decomposition (plain-language angles, LLM-generated)
    3. bare_act_hints (act names — only if the USER explicitly mentioned an act name;
       DISPUTE_DECOMPOSITION_PROMPT outputs [] by default so this rarely fires)
    4. dispute_text alone (no keywords — helps when keywords are too noisy)
    5. LLM-generated search angles if dispute has no pre-built angles (lazy fallback)

    Returns deduplicated list of queries, longest-first.
    """
    dispute_text = dispute.get("dispute", "")
    keywords = dispute.get("keywords", [])
    search_angles = dispute.get("search_angles", [])   # from decomposition prompt
    bare_act_hints = dispute.get("bare_act_hints", []) # act names from decomposition
    legal_concepts = dispute.get("legal_concepts", []) # for structured statute/concept queries

    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str):
        q = q.strip()[:400]
        norm = q.lower()
        if q and norm not in seen:
            seen.add(norm)
            queries.append(q)

    # Q0: legal concepts (structured lookup — concept terms as queries, e.g. "criminal intimidation")
    for c in legal_concepts[:2]:
        if c and isinstance(c, str):
            _add(c.strip())

    # Q1: full dispute + keywords (broadest, catches anything related)
    # Use only the user's own language — no LLM-knowledge BNS/IPC enrichment
    base = f"{dispute_text} {' '.join(keywords)}".strip()
    _add(base)

    # Q2+: pre-built search angles from the decomposition LLM (plain-language angles)
    for angle in search_angles[:3]:
        _add(angle)

    # Q3: each act hint as its own short query (only if user explicitly mentioned an act name)
    # bare_act_hints is [] by default from DISPUTE_DECOMPOSITION_PROMPT (no LLM injection)
    for hint in bare_act_hints[:3]:
        _add(hint)
        # Also act + first keyword
        if keywords:
            _add(f"{hint} {keywords[0]}")

    # Q4: dispute text only (no keywords — helps when keywords are too noisy)
    _add(dispute_text)

    # Q4b: deterministic statute-style injections for common criminal-code disputes.
    # This strengthens local retrieval when the user describes facts in plain language
    # ("beat me", "threatened to break my knees") but the indexed act is stored under
    # formal statute names like Bharatiya Nyaya Sanhita, 2023.
    legal_nature = str(dispute.get("legal_nature", "both")).lower()
    new_codes = _detect_new_criminal_codes(dispute_text, legal_nature)
    concept_text = " ".join(
        [str(x).strip() for x in (legal_concepts or []) + (keywords or []) if str(x).strip()]
    ).lower()
    text_lower = dispute_text.lower()

    def _has_any(text: str, phrases: list[str]) -> bool:
        return any(p in text for p in phrases)

    bns_terms: list[str] = []
    if _has_any(concept_text + " " + text_lower, ["criminal intimidation", "threat", "threatened", "threat of injury"]):
        bns_terms.append("criminal intimidation")
    if _has_any(concept_text + " " + text_lower, ["assault", "attack", "beat", "beaten", "physical harm", "injury", "hurt", "metal rod"]):
        bns_terms.append("hurt assault physical harm")
    if new_codes.get("bns"):
        primary_bns = " ".join(dict.fromkeys(bns_terms)) if bns_terms else "criminal intimidation hurt assault"
        _add(f"Bharatiya Nyaya Sanhita 2023 {primary_bns}")
        _add(f"BNS 2023 {primary_bns}")
    if new_codes.get("bnss"):
        _add("Bharatiya Nagarik Suraksha Sanhita 2023 FIR complaint investigation")
    if new_codes.get("bsa"):
        _add("Bharatiya Sakshya Adhiniyam 2023 evidence witness electronic record")

    # Q4c: deterministic tenancy/rent injections for local act retrieval.
    # This is especially useful when the user describes a lease/rent default in
    # plain language but does not name the applicable act.
    if _is_rent_eviction_dispute(dispute):
        _add("Transfer of Property Act lease forfeiture non payment of rent eviction")
        _add("rent tenancy eviction non payment of rent lease agreement landlord tenant")
        if _has_any(text_lower, ["written agreement", "lease agreement", "vacating the property", "vacate the property"]):
            _add("lease agreement forfeiture termination of lease non payment of rent")

    # Q5: If we still have < 3 queries, generate via LLM
    if len(queries) < 3:
        try:
            from prompts.advocate_prompts import BARE_ACT_SEARCH_QUERIES_PROMPT
            import json as _json
            prompt = BARE_ACT_SEARCH_QUERIES_PROMPT.format(
                dispute=dispute_text[:300],
                act_hints=", ".join(bare_act_hints[:3]) or "unknown",
                keywords=", ".join(keywords[:5]) or "none",
            )
            raw = ask_llm(prompt).strip()
            # Strip markdown fence if present
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            data = _json.loads(raw)
            for q in (data.get("queries") or []):
                _add(str(q))
        except Exception as e:
            logger.debug("Bare act LLM query gen failed: %s", e)

    # Cap at 3 queries for Tier 1 fast path (roadmap: max_queries = 3)
    final_queries = queries[:3]
    try:
        logger.info(
            "Bare-act query builder [%s]: legal_nature=%s queries=%s",
            (dispute.get("id") or "?"),
            str(dispute.get("legal_nature", "both")).lower(),
            [q[:140] for q in final_queries],
        )
    except Exception:
        pass
    return final_queries


# Matches cross-references that are WITHIN the same act
# (e.g. "subject to Section 13B", "as defined under Section 2(a)")
# Deliberately excludes references like "Section 325 of the Indian Penal Code"
# by requiring that "of the <ActName>" does NOT follow the section number.
_RE_SECTION_REF = re.compile(
    r'\b(?:under|subject to|as per|defined (?:in|under)|referred to in|'
    r'provided (?:in|under)|see also|as mentioned in|in accordance with)\s+'
    r'[Ss]ection[s]?\s+(\d+[A-Za-z]*(?:\s*(?:and|,|to)\s*\d+[A-Za-z]*)*)'
    r'(?!\s+of\s+the\s+[A-Z])',   # negative lookahead: not "of the <Xyz Act>"
    re.IGNORECASE,
)
_RE_BARE_SEC_NUM = re.compile(r'\bsections?\s+(\d+[A-Za-z]*)', re.IGNORECASE)

# Phrases that signal an external-act citation — section numbers appearing
# in this context are NOT same-act cross-references and must be ignored.
_RE_EXTERNAL_ACT_CITATION = re.compile(
    r'[Ss]ection\s+\d+[A-Za-z]*\s+of\s+the\s+(?:'
    r'Indian Penal Code|Code of Criminal Procedure|Indian Evidence Act|'
    r'Transfer of Property Act|Civil Procedure Code|Limitation Act|'
    r'Registration Act|Stamp Act|Companies Act|Income Tax Act|'
    r'Bharatiya Nyaya Sanhita|BNS|BNSS|BSA|IPC|CrPC'
    r')',
    re.IGNORECASE,
)


def _extract_cross_references(chunks: list) -> list[str]:
    """
    Scan retrieved bare act chunks for SAME-ACT cross-references
    (e.g. "as defined under Section 2", "subject to Section 13B").

    Critical fix: previously the regex matched "provided for by section 335"
    inside IEA illustrative examples that cited IPC 335 as an EXAMPLE.
    That caused the cross-reference expander to search for IEA Section 335,
    which found more IEA sections mentioning IPC examples — a feedback loop
    that flooded results with Evidence Act chunks instead of BNS/BNSS sections.

    Two-layer guard:
    1. Negative lookahead in the regex: rejects "Section X of the <ActName>"
    2. Post-filter: remove numbers that appear near external-act citation phrases
    """
    referenced: set[str] = set()
    already_have: set[str] = set()

    for c in chunks:
        sec = (c.get("section_number") or "").strip()
        if sec:
            already_have.add(sec.lower())

        text = (c.get("full_text") or c.get("search_text") or c.get("text") or "")

        # Collect section numbers mentioned near external-act phrases — these are
        # citations to another Act's sections, NOT same-act cross-references.
        external_sec_nums: set[str] = set()
        for em in _RE_EXTERNAL_ACT_CITATION.finditer(text):
            # Back-scan the match to pull the section number out
            snippet = text[max(0, em.start() - 5): em.end()]
            for nm in _RE_BARE_SEC_NUM.finditer(snippet):
                external_sec_nums.add(nm.group(1).lower())

        for m in _RE_SECTION_REF.finditer(text):
            raw = m.group(1)
            for part in re.split(r'\s*(?:and|,|to)\s*', raw):
                s = part.strip()
                if s and s.lower() not in already_have and s.lower() not in external_sec_nums:
                    referenced.add(s)

    return list(referenced)


def _fetch_cross_referenced_sections(
    act_names: list[str],
    section_numbers: list[str],
) -> list:
    """
    Direct vector-store lookup for specific act+section pairs.
    Used after finding cross-references in retrieved chunks.
    """
    from retrieval.hybrid_retriever import search_bare_acts_auto
    results = []
    seen_keys: set[str] = set()

    for sec in section_numbers[:6]:   # cap to avoid blowing up query time
        for act in act_names[:3]:
            q = f"{act} section {sec}".strip()
            raw = search_bare_acts_auto(q, top_k=5)
            for r in raw:
                key = (
                    (r.get("act_name") or "").strip().lower(),
                    (r.get("section_number") or "").strip().lower(),
                )
                if key not in seen_keys and _is_quality_bare_act(r):
                    seen_keys.add(key)
                    results.append(r)
    return results


def _web_search_bare_acts(dispute: dict, full_query: str, round1_queries: list = None, states: list = None, pending_indexing_list: list = None) -> list:
    """
    Web search for bare acts targeting one dispute component.
    Runs up to 3 targeted queries instead of one generic one.
    Returns list of bare-act-like dicts.

    Query priority:
      1. search_angles from decomposer (focused plain-language phrases)
      2. round1_queries[1:] — reuse the focused queries already run in Round 1
         (these include LLM-generated angles like "criminal assault causing grievous hurt
         with metal rod"; skip Q1 which is the broad dispute+keywords combination)
      3. keywords — keyword-level fallback
      4. dispute_text[:200] — last resort when nothing else is available
    """
    from retrieval.tiered_search import search_for_gaps
    from retrieval.auto_enricher import enrich_from_gap_results

    dispute_text    = dispute.get("dispute", "")
    keywords        = dispute.get("keywords", [])
    bare_act_hints  = dispute.get("bare_act_hints", [])
    search_angles   = dispute.get("search_angles", [])

    # Build targeted web queries.
    # Priority order:
    #   1. search_angles (focused 6-12 word phrases from decomposer, e.g.
    #      "criminal liability for physical assault causing serious injury") — use up to 3.
    #   2. keywords (3-5 descriptive words) + "legislation India" as a tighter fallback.
    #   3. If no angles and no keywords: raw dispute_text (last resort).
    # NOTE: bare_act_hints is intentionally left unused here — the decomposer is
    # configured to output plain language only; act names come from the local index
    # or web discovery, never from LLM guesses.
    web_gaps = []
    seen_q: set[str] = set()

    def _add_gap(q: str):
        q = q.strip()[:300]
        if q and q.lower() not in seen_q:
            seen_q.add(q.lower())
            web_gaps.append({"query": q, "type": "bare_act"})

    # Build the focused, state-agnostic base queries (these find central/Union acts).
    # Priority: decomposer search_angles → Round 1 focused queries → keywords → raw text.
    # We deliberately strip factual/location details (city names, incident dates) to
    # avoid biasing IndiaCode toward state-specific acts.
    focused: list[str] = []
    if search_angles:
        focused = search_angles[:3]
    elif round1_queries and len(round1_queries) > 1:
        # Skip index 0 (full dispute text which often contains city/state names);
        # indices 1+ are focused legal phrases from LLM or decomposer search_angles.
        focused = round1_queries[1:4]
    elif keywords:
        kws = " ".join(keywords[:4])
        focused = [kws]

    if focused:
        # ── Central / Union act queries (always first) ──────────────────────────
        # These are state-agnostic and will find BNS, BNSS, Transfer of Property Act,
        # Specific Relief Act, etc. on IndiaCode regardless of which state the dispute is in.
        for q in focused[:2]:
            _add_gap(f"{q} India central act")

        # ── State-specific queries (supplement, not replacement) ─────────────────
        # Indian law is layered: central acts apply everywhere, but states may have their
        # own concurrent legislation (Telangana Land Encroachment Act, Telangana Tenancy Act,
        # Karnataka Rent Control Act, etc.). We add ONE state-specific query for each
        # detected state so we capture relevant local legislation alongside central acts.
        if states:
            for state in states[:2]:   # cap at 2 states in case many were detected
                state_q = focused[0]   # use the primary angle for state search
                _add_gap(f"{state_q} {state} state act")
    else:
        # Absolute fallback: raw dispute text (fires only when there are no angles, no
        # round1_queries, and no keywords — unusual edge case on a bare one-word input)
        _add_gap(f"{dispute_text[:180]} India central act")

    # Always add a keyword-level query as a supplementary signal
    if keywords:
        kws = " ".join(keywords[:3])
        _add_gap(f"{kws} India legislation")

    # primary_term: most focused legal phrase available — used as suffix in all
    # act-name queries below so IndiaCode retrieves the *right sections* of each act.
    primary_term = (focused[0] if focused else "") or " ".join(keywords[:2])
    primary_term = primary_term[:120]  # guard against very long phrases

    # ── bare_act_hints → explicit act-name queries (general mechanism) ───────
    # The decomposer now generates bare_act_hints with actual Act names when it
    # is confident (e.g. "Transfer of Property Act 1882" for a property dispute,
    # "Negotiable Instruments Act 1881" for a cheque-bounce case).  Using the act
    # name explicitly in the query guarantees IndiaCode returns sections from the
    # correct act rather than whichever popular act shares adjacent vocabulary
    # (Arms Act for "assault", Copyright Act for "original work", etc.).
    # This is the GENERAL mechanism — it covers all domains without hardcoding.
    if bare_act_hints and primary_term:
        insert_pos = 0  # prepend before generic queries
        for hint in bare_act_hints[:3]:
            hint_q = f"{hint} {primary_term}"
            if hint_q.lower() not in seen_q:
                web_gaps.insert(insert_pos, {"query": hint_q, "type": "bare_act"})
                seen_q.add(hint_q.lower())
                logger.info("bare_act_hint query [%d]: %s", insert_pos, hint_q[:100])
                insert_pos += 1

    # ── Safety-net: new 2023 criminal codes ─────────────────────────────────
    # BNS / BNSS / BSA replace IPC / CrPC / IEA but IndiaCode generic searches
    # ("assault India central act") still surface the Arms Act.  If the
    # decomposer hinted at the new codes (via bare_act_hints), the block above
    # already handles it.  This block is a keyword-based fallback for the common
    # case where the LLM still uses old names ("IPC") or no hint was generated.
    legal_nature = dispute.get("legal_nature", "both")
    new_codes = _detect_new_criminal_codes(dispute_text, legal_nature)

    hints_lower = " ".join(bare_act_hints).lower()
    already_has_bns  = "bharatiya nyaya sanhita"    in hints_lower
    already_has_bnss = "bharatiya nagarik suraksha" in hints_lower
    already_has_bsa  = "bharatiya sakshya"          in hints_lower

    # How many hint queries we prepended (determines insert_pos for safety-net)
    hint_count = min(len(bare_act_hints), 3) if (bare_act_hints and primary_term) else 0

    if new_codes["bns"] and not already_has_bns and primary_term:
        bns_q = f"Bharatiya Nyaya Sanhita 2023 {primary_term}"
        if bns_q.lower() not in seen_q:
            web_gaps.insert(hint_count, {"query": bns_q, "type": "bare_act"})
            seen_q.add(bns_q.lower())
            logger.info("Safety-net BNS query: %s", bns_q[:100])

    if new_codes["bnss"] and not already_has_bnss and primary_term:
        bnss_q = f"Bharatiya Nagarik Suraksha Sanhita 2023 {primary_term}"
        if bnss_q.lower() not in seen_q:
            pos = hint_count + (1 if new_codes["bns"] and not already_has_bns else 0)
            web_gaps.insert(pos, {"query": bnss_q, "type": "bare_act"})
            seen_q.add(bnss_q.lower())
            logger.info("Safety-net BNSS query: %s", bnss_q[:100])

    if new_codes["bsa"] and not already_has_bsa and primary_term:
        bsa_q = f"Bharatiya Sakshya Adhiniyam 2023 {primary_term}"
        if bsa_q.lower() not in seen_q:
            pos = hint_count + sum([
                new_codes["bns"]  and not already_has_bns,
                new_codes["bnss"] and not already_has_bnss,
            ])
            web_gaps.insert(pos, {"query": bsa_q, "type": "bare_act"})
            seen_q.add(bsa_q.lower())
            logger.info("Safety-net BSA query: %s", bsa_q[:100])

    if not web_gaps:
        web_gaps = [{"query": f"{dispute_text[:180]} India bare act", "type": "bare_act"}]

    logger.info(
        "Web search queries for dispute '%s' (states=%s): %s",
        dispute_text[:50], states or [], [g["query"][:80] for g in web_gaps],
    )

    try:
        gap_results = search_for_gaps(web_gaps, jurisdiction_state=states[0] if states else "")
    except Exception as e:
        logger.error("Web search bare acts failed for dispute '%s': %s", dispute_text[:60], e)
        return []

    logger.info(
        "Indiankanoon bare-act search: %d results for dispute '%s'",
        len(gap_results.get("bare_act_results", [])),
        dispute_text[:60],
    )
    return []


def _filter_bare_acts_with_llm(dispute: dict, bare_acts: list, debug: dict | None = None) -> list:
    """
    Use an LLM to down-rank or drop obviously off-topic bare act sections for one dispute.
    Falls back to the original list on any error or empty filter output.
    """
    if not bare_acts:
        return bare_acts

    # Skip expensive LLM call when every section already scores above the high-quality
    # threshold — the cross-encoder is already confident they are all relevant.
    if all(ba.get("_rerank_score", 0) >= HIGH_QUALITY_SCORE for ba in bare_acts):
        logger.debug("Bare-act LLM filter skipped — all %d sections high quality", len(bare_acts))
        return bare_acts

    dispute_text = (dispute.get("dispute") or "")[:400]

    # Build a compact JSON payload for the top-N candidates only to control prompt size.
    candidates = []
    for ba in bare_acts[:20]:
        act_name = (ba.get("act_name") or "").strip()
        section_number = (ba.get("section_number") or "").strip()
        section_title = (ba.get("section_title") or "").strip()
        raw = (
            ba.get("search_text")
            or ba.get("full_text")
            or ba.get("text")
            or ""
        )
        # Normalise whitespace and truncate to a few sentences.
        snippet = " ".join(str(raw).split())[:400]
        candidates.append(
            {
                "act_name": act_name,
                "section_number": section_number,
                "section_title": section_title,
                "snippet": snippet,
            }
        )

    try:
        # Optional debug snapshot of input candidates
        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            dbg.setdefault("bare_act_llm_section_filters", []).append(
                {
                    "input_sections": [
                        {
                            "act_name": c["act_name"],
                            "section_number": c["section_number"],
                            "section_title": c["section_title"],
                        }
                        for c in candidates
                    ]
                }
            )
        prompt = BARE_ACT_SECTION_RELEVANCE_PROMPT.format(
            dispute=dispute_text,
            sections_json=json.dumps(candidates, ensure_ascii=False),
        )
        resp = ask_llm(prompt)
        data = json.loads(resp)
        keep_keys = set()
        for s in data.get("sections") or []:
            rel = (s.get("relevance") or "").strip().lower()
            if rel in ("high", "medium"):
                key = (
                    (s.get("act_name") or "").strip().lower(),
                    (s.get("section_number") or "").strip().lower(),
                )
                keep_keys.add(key)

        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            # Update the last entry with kept keys
            if dbg.get("bare_act_llm_section_filters"):
                dbg["bare_act_llm_section_filters"][-1]["kept_keys"] = sorted(list(keep_keys))

        if not keep_keys:
            return bare_acts

        filtered = [
            ba
            for ba in bare_acts
            if (
                (ba.get("act_name") or "").strip().lower(),
                (ba.get("section_number") or "").strip().lower(),
            )
            in keep_keys
        ]
        # Avoid returning an empty list when the LLM is over-aggressive.
        result = filtered or bare_acts
        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            if dbg.get("bare_act_llm_section_filters"):
                dbg["bare_act_llm_section_filters"][-1]["kept_count"] = len(result)
        return result
    except Exception as e:
        logger.warning("Bare-act section LLM relevance filter failed: %s", e)
        return bare_acts


def _filter_case_laws_with_llm(dispute: dict, bare_act_sections: list, case_laws: list, debug: dict | None = None) -> list:
    """
    Use an LLM to down-rank or drop obviously off-topic case law chunks for one dispute.
    Falls back to the original list on any error or empty filter output.
    """
    if not case_laws:
        return case_laws

    # Skip expensive LLM call when every result already scores above the high-quality
    # threshold — the cross-encoder is already confident they are all relevant.
    if all(cl.get("_rerank_score", 0) >= HIGH_QUALITY_SCORE for cl in case_laws):
        logger.debug("Case-law LLM filter skipped — all %d results high quality", len(case_laws))
        return case_laws

    dispute_text = (dispute.get("dispute") or "")[:400]

    # Optional context: a brief summary of the key bare act sections linked to this dispute.
    bare_ctx_parts = []
    for ba in bare_act_sections[:5]:
        act = (ba.get("act_name") or "").strip()
        sec = (ba.get("section_number") or "").strip()
        title = (ba.get("section_title") or "").strip()
        if act and sec:
            label = f"{act} Section {sec}"
            if title:
                label += f" — {title}"
            bare_ctx_parts.append(label)
    bare_act_context = "\n".join(bare_ctx_parts) if bare_ctx_parts else "[]"

    # Build compact JSON payload for the top-N candidates only.
    candidates = []
    for cl in case_laws[:20]:
        case_name = (cl.get("case_name") or cl.get("source") or "").strip()
        court = (cl.get("court") or "").strip()
        year = str(cl.get("year") or "").strip()
        binding = (cl.get("binding_authority") or cl.get("binding") or "").strip()
        raw = (
            cl.get("search_text")
            or cl.get("full_text")
            or cl.get("text")
            or ""
        )
        snippet = " ".join(str(raw).split())[:400]
        candidates.append(
            {
                "case_name": case_name,
                "court": court,
                "year": year,
                "binding": binding,
                "snippet": snippet,
            }
        )

    try:
        # Optional debug snapshot of input candidates
        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            dbg.setdefault("case_law_llm_filters", []).append(
                {
                    "input_cases": [
                        {
                            "case_name": c["case_name"],
                            "court": c["court"],
                            "year": c["year"],
                            "binding": c["binding"],
                        }
                        for c in candidates
                    ]
                }
            )
        prompt = CASE_LAW_RELEVANCE_PROMPT.format(
            dispute=dispute_text,
            bare_act_context=bare_act_context,
            cases_json=json.dumps(candidates, ensure_ascii=False),
        )
        resp = ask_llm(prompt)
        data = json.loads(resp)
        keep_names = set()
        for c in data.get("cases") or []:
            rel = (c.get("relevance") or "").strip().lower()
            if rel in ("high", "medium"):
                name_key = (c.get("case_name") or "").strip().lower()
                if name_key:
                    keep_names.add(name_key)

        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            if dbg.get("case_law_llm_filters"):
                dbg["case_law_llm_filters"][-1]["kept_names"] = sorted(list(keep_names))

        if not keep_names:
            return case_laws

        filtered = []
        for cl in case_laws:
            name_key = (
                (cl.get("case_name") or cl.get("source") or "").strip().lower()
            )
            if name_key in keep_names:
                filtered.append(cl)

        result = filtered or case_laws
        if debug is not None:
            d_id = dispute.get("id", "?")
            dbg = debug.setdefault(d_id, {})
            if dbg.get("case_law_llm_filters"):
                dbg["case_law_llm_filters"][-1]["kept_count"] = len(result)
        return result
    except Exception as e:
        logger.warning("Case-law LLM relevance filter failed: %s", e)
        return case_laws


def retrieve_bare_acts_for_dispute(dispute: dict, full_query: str, states: list = None, debug: dict | None = None, pending_indexing_list: list = None) -> list:
    """
    Retrieve ALL relevant bare act sections for a single dispute component.

    Act-first search (new):
        Before any section-level queries, identify the 3-5 most relevant acts using
        ActProfileIndex (cheap BM25 over per-act profile docs built from section
        titles + key phrase samples).  Pass the result as ``allowed_acts`` to each
        hybrid_search call so the cross-encoder only sees candidates from those acts
        (~20-30 candidates instead of ~90) → ~3-4× faster per query.
        Falls back to unfiltered search if the profile index is unavailable.

    Round 1 — local multi-query hybrid search:
        Runs 4-6 queries from diverse legal angles (primary, act-specific,
        remedy-focused, section-anchored). Merges by chunk key, keeping the
        highest score for each unique section. Deduplicates, then quality-filters.
        → if high-quality found AND LLM says sufficient → STOP

    Round 1b — cross-reference expansion:
        Scans retrieved chunks for "subject to Section X / as defined under Section Y"
        references and fetches those sections if not already retrieved.

    Round 2 — tiered web search (bare acts only):
        Targeted queries (act names + keywords + India Code) instead of generic.
        → merge, deduplicate → STOP

    No result cap — every section that passes quality filter is returned.
    """
    from retrieval.hybrid_retriever import search_bare_acts_auto, search_bare_acts_filtered

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    debug_entry = None
    if debug is not None:
        debug_entry = debug.setdefault(dispute_id, {})

    # Act filtering removed (act_profile_index + statute_concept_index deleted).
    # hybrid_retriever cross-encoder handles relevance filtering directly.
    # Keep this as an empty frozenset rather than None so downstream diagnostics
    # and candidate-act loops remain iterable even when no act filter is active.
    allowed_acts = frozenset()

    # Diagnostic: compare decomposer hints against profile-identified acts.
    # When both are non-empty but completely disjoint, the two systems disagree
    # on which acts apply — a strong signal for targeted tuning of either the
    # decomposer prompts or the act profile vocabulary.
    bare_act_hints = dispute.get("bare_act_hints", [])
    if debug_entry is not None:
        debug_entry["bm25_acts"] = sorted(list(allowed_acts))
        debug_entry["bare_act_hints"] = list(bare_act_hints)
    if bare_act_hints and allowed_acts:
        hints_lower = {h.strip().lower() for h in bare_act_hints}
        acts_lower  = {a.strip().lower() for a in allowed_acts}
        if not hints_lower.intersection(acts_lower):
            logger.warning(
                "[%s] hints/profile MISMATCH — decomposer hints=%s, profile acts=%s. "
                "Consider updating act profile vocabulary or decomposer prompts.",
                dispute_id, list(bare_act_hints)[:3], sorted(allowed_acts),
            )

    # LLM refinement: unify BM25-identified acts and decomposer hints, then let the LLM
    # rate relevance per Act for THIS dispute (high/medium/low) in a domain-agnostic way.
    candidate_acts = []
    seen_act_names = set()
    for name in sorted(allowed_acts):
        n = (name or "").strip()
        if n and n.lower() not in seen_act_names:
            candidate_acts.append({"act_name": n, "source": "profile", "note": ""})
            seen_act_names.add(n.lower())
    for hint in bare_act_hints:
        n = str(hint or "").strip()
        if n and n.lower() not in seen_act_names:
            candidate_acts.append({"act_name": n, "source": "hint", "note": ""})
            seen_act_names.add(n.lower())

    if candidate_acts:
        try:
            prompt = ACT_SELECTION_PROMPT.format(
                dispute=dispute_text[:400],
                candidate_acts_json=json.dumps(candidate_acts, ensure_ascii=False),
            )
            resp = ask_llm(prompt)
            data = json.loads(resp)
            refined = [
                (a.get("act_name") or "").strip()
                for a in (data.get("acts") or [])
                if (a.get("relevance") or "").strip().lower() in ("high", "medium")
            ]
            refined = [n for n in refined if n]
            if refined:
                allowed_acts = frozenset(refined)
                logger.info(
                    "[%s] Act refinement via LLM: %d high/medium acts → %s",
                    dispute_id, len(allowed_acts), sorted(allowed_acts),
                )
                if debug_entry is not None:
                    debug_entry["llm_acts"] = sorted(list(allowed_acts))
        except Exception as e:
            logger.warning(
                "[%s] Act-selection LLM refinement failed: %s",
                dispute_id, e,
            )

    # Build diverse query set
    queries = _build_bare_act_queries(dispute)
    logger.info(
        "[%s] Bare acts Round 1 — %d queries: %s",
        dispute_id, len(queries), [q[:60] for q in queries],
    )

    # --- Round 1: local multi-query ---
    # Use act-filtered search when profile index identified relevant acts:
    #   search_bare_acts_filtered passes allowed_acts to hybrid_search so the
    #   cross-encoder only scores candidates from the pre-identified acts.
    # Falls back to search_bare_acts_auto when allowed_acts is empty (profile unavailable).
    # top_k=15 per query: with act-filter in place, ~15 candidates after filtering
    # vs ~90 without → significant cross-encoder speedup.
    def _search(q: str) -> list:
        if allowed_acts:
            return search_bare_acts_filtered(q, allowed_acts, top_k=15)
        return search_bare_acts_auto(q, top_k=15)

    seen_chunk_keys: dict[str, dict] = {}   # chunk_key -> best-scored chunk
    for q in queries:
        raw = _search(q)
        for ba in raw:
            key = ba.get("_chunk_key") or (
                (ba.get("act_name") or "").lower() + "|" + (ba.get("section_number") or "").lower()
            )
            prev_score = (seen_chunk_keys.get(key) or {}).get("_rerank_score", -999)
            if ba.get("_rerank_score", 0) > prev_score:
                seen_chunk_keys[key] = ba

    all_local = list(seen_chunk_keys.values())

    # Act-level pre-filter: with 100+ acts indexed, restrict to the top 3 acts
    # whose best-scoring section wins the cross-encoder race for this dispute.
    # This eliminates sections from irrelevant acts (POCSO, Arms Act, Constitution,
    # Indian Succession Act ...) before the score-threshold filter runs.
    top_acts = _select_top_acts(all_local)

    # Secondary guard: if LLM already identified relevant acts (allowed_acts), drop any
    # reranker-selected act that is NOT a fuzzy substring match of a LLM-allowed act.
    # This prevents false positives like Telangana Excise Act sneaking through because
    # its drunkenness provisions scored well on a domestic violence query.
    if top_acts and allowed_acts:
        def _act_in_llm_list(act: str, llm_acts: frozenset) -> bool:
            al = act.lower()
            for la in llm_acts:
                la_l = la.lower()
                # 1. Direct substring match (original check)
                if la_l in al or al in la_l:
                    return True
                # 2. Word-overlap match — catches aliases like "Land Acquisition Act 1894"
                #    vs "...Land Acquisition...Rules 2014".  Strip stop-words by requiring
                #    4+ character words so "act", "the", "and" don't create false positives.
                al_words = set(re.findall(r'\b[a-z]{4,}\b', al))
                la_words = set(re.findall(r'\b[a-z]{4,}\b', la_l))
                if len(al_words & la_words) >= 1:
                    return True
            return False
        llm_intersect = {a for a in top_acts if _act_in_llm_list(a, allowed_acts)}
        if llm_intersect:
            dropped = top_acts - llm_intersect
            if dropped:
                logger.info(
                    "[%s] LLM-act guard: dropped %s (not in LLM-identified acts %s)",
                    dispute_id, list(dropped), list(allowed_acts),
                )
            top_acts = llm_intersect
        else:
            # llm_intersect empty: LLM-identified act (e.g. "Land Acquisition Act 1894")
            # is not indexed locally. Log it so we can investigate, but don't silently
            # keep all reranker acts — that allows completely unrelated acts (Prisons,
            # HMDA) to slip through. Fall back to the top 1 reranker act only.
            best_act = max(top_acts, key=lambda a: max(
                (c.get("_rerank_score", 0) for c in all_local if (c.get("act_name") or "").strip() == a),
                default=0,
            ))
            logger.info(
                "[%s] LLM-act guard: no word-overlap match found — LLM acts=%s; "
                "falling back to top reranker act '%s' only (was: %s)",
                dispute_id, list(allowed_acts), best_act, list(top_acts),
            )
            top_acts = {best_act}

    if top_acts:
        before = len(all_local)
        all_local = [ba for ba in all_local if (ba.get("act_name") or "").strip() in top_acts]
        logger.info("[%s] Act pre-filter: %d → %d sections (kept acts: %s)", dispute_id, before, len(all_local), list(top_acts))

    local_results = [
        ba for ba in all_local
        if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)
    ]
    local_results.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    logger.info(
        "[%s] Bare acts Round 1: %d unique sections across %d queries (after act pre-filter + quality filter)",
        dispute_id, len(local_results), len(queries),
    )

    # Cross-reference expansion is DISABLED.
    # With 100+ acts indexed, extracted section numbers (e.g. 10, 85, 233) appear in dozens
    # of unrelated acts, causing Constitution / IBC / Succession Act sections to flood results.
    # Re-enable only once the index is restricted to a curated set of acts.

    # Stop condition: skip Round 2 web search when local results are sufficient.
    # Sufficiency tiers (fastest-first, avoids unnecessary LLM calls):
    #
    #   Tier A1 — AUTO-STOP (no LLM): ≥ 5 sections score > BARE_ACT_HIGH_QUALITY_SCORE (2.0).
    #             ms-marco logits for relevant legal sections cluster around 2–3, so this
    #             threshold fires reliably on a good local corpus.
    #
    #   Tier A2 — AUTO-STOP (volume, no LLM): ≥ 10 sections pass MIN_RERANK_SCORE (0.0).
    #             When the local store has broad coverage (e.g. full BNSS / IPC indexed),
    #             many sections return with moderate scores. 10+ is sufficient coverage.
    #
    #   Tier B — LLM CONFIRM: 1-4 high-quality sections. Ask the LLM whether coverage is
    #             adequate before spending time on web search.
    #
    #   Tier C — ALWAYS WEB: 0 high-quality sections → proceed to Round 2 regardless.
    high_quality_local = [ba for ba in local_results if ba.get("_rerank_score", 0) > BARE_ACT_HIGH_QUALITY_SCORE]
    _AUTO_STOP_COUNT = 3   # ≥ 3 HQ sections (score > 2.0) → skip LLM and web search entirely
    _AUTO_STOP_VOLUME = 5  # ≥ 5 any-score sections → stop (matches MAX_SECTIONS_PER_DISPUTE_TOTAL)

    # Web search gate: only when local result count is low (roadmap: not by score).
    # If we have at least WEB_SEARCH_MIN_LOCAL_COUNT sections, do not force web.
    max_local_score = max((ba.get("_rerank_score", 0) for ba in local_results), default=0.0)
    force_web = (len(local_results) < WEB_SEARCH_MIN_LOCAL_COUNT)
    if force_web:
        logger.info(
            "[%s] Local sections %d < %d — allowing web search (best score %.2f).",
            dispute_id, len(local_results), WEB_SEARCH_MIN_LOCAL_COUNT, max_local_score,
        )
    else:
        if len(high_quality_local) >= _AUTO_STOP_COUNT:
            logger.info(
                "[%s] Bare acts Round 1 auto-sufficient (%d HQ sections ≥ %d, score>%.1f). "
                "Skipping LLM sufficiency call and web search.",
                dispute_id, len(high_quality_local), _AUTO_STOP_COUNT, BARE_ACT_HIGH_QUALITY_SCORE,
            )
            filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
            return _cap_bare_acts_by_score(
                _reorder_bare_acts_for_rent_disputes(filtered_local, dispute),
                MAX_SECTIONS_PER_DISPUTE_TOTAL,
            )

        if len(local_results) >= _AUTO_STOP_VOLUME:
            logger.info(
                "[%s] Bare acts Round 1 auto-sufficient by volume (%d sections ≥ %d). "
                "Skipping LLM sufficiency call and web search.",
                dispute_id, len(local_results), _AUTO_STOP_VOLUME,
            )
            filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
            return _cap_bare_acts_by_score(
                _reorder_bare_acts_for_rent_disputes(filtered_local, dispute),
                MAX_SECTIONS_PER_DISPUTE_TOTAL,
            )

        if high_quality_local and local_results and _check_bare_act_sufficiency(dispute_text, local_results):
            logger.info(
                "[%s] Bare acts Round 1 sufficient (%d sections, %d high-quality). Stopping.",
                dispute_id, len(local_results), len(high_quality_local),
            )
            filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
            return _cap_bare_acts_by_score(
                _reorder_bare_acts_for_rent_disputes(filtered_local, dispute),
                MAX_SECTIONS_PER_DISPUTE_TOTAL,
            )

    # --- Round 2: web search ---
    logger.info("[%s] Bare acts Round 2 — web search (local had %d, force_web=%s)", dispute_id, len(local_results), force_web)
    # Pass `queries` so the web search can reuse the focused Round 1 queries
    # (including LLM-generated ones) instead of falling back to raw dispute text.
    # Pass `states` so web search covers both central/Union acts AND state-specific acts.
    web_results = _web_search_bare_acts(dispute, full_query, round1_queries=queries, states=states or [], pending_indexing_list=pending_indexing_list)

    merged = _merge_deduplicate_bare_acts(local_results, web_results)
    merged = _filter_bare_acts_with_llm(dispute, merged, debug)
    merged = _reorder_bare_acts_for_rent_disputes(merged, dispute)
    capped = _cap_bare_acts_by_score(merged, MAX_SECTIONS_PER_DISPUTE_TOTAL)
    logger.info(
        "[%s] Bare acts Round 2 complete: %d total → %d after cap (local=%d web=%d)",
        dispute_id, len(merged), len(capped), len(local_results), len(web_results),
    )
    return capped


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


def _apply_dispute_case_law_limit(case_laws: list, num_sections: int = 1) -> list:
    """
    Threshold-based limit for per-dispute case laws:
      • If there is one main section: keep up to 3 strongest cases.
      • If there are multiple connected sections: keep up to 5 strongest cases so they
        can be distributed across sections in the final opinion.
      • Prefer high-quality cases first; otherwise fall back to top relevant cases.
    Input must already be quality-filtered.
    """
    if not case_laws:
        return []
    cap = 5 if int(num_sections or 1) > 1 else 3
    sorted_cls = sorted(case_laws, key=lambda x: x.get("_rerank_score", 0), reverse=True)
    high_quality = [cl for cl in sorted_cls if cl.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    if high_quality:
        return high_quality[:cap]
    return sorted_cls[:cap]


def _web_search_case_laws(dispute: dict, bare_act_sections: list, full_query: str, states: list = None, pending_indexing_list: list = None) -> list:
    """
    Web search for case laws for one dispute, using the richer dispute+sections query.
    Runs an Indiankanoon-only web search for case laws.
    This path no longer builds pending-indexing candidates.
    Returns [] so answer remains grounded in locally indexed materials only.
    """
    from retrieval.tiered_search import search_for_gaps
    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    gap_query = _build_case_law_query(dispute_text, bare_act_sections)
    gap_query = f"{gap_query} Supreme Court High Court judgment India".strip()[:400]

    gaps = [{"query": gap_query, "type": "case_law"}]
    try:
        gap_results = search_for_gaps(gaps, jurisdiction_state=states[0] if states else "")
    except Exception as e:
        logger.error("Web search case laws failed for dispute '%s': %s", dispute_text[:60], e)
        return []

    logger.info(
        "[%s] Indiankanoon case-law search: %d results.",
        dispute_id, len(gap_results.get("case_law_results", [])),
    )
    return []


def retrieve_case_laws_for_dispute(dispute: dict, bare_act_sections: list, full_query: str, states: list = None, debug: dict | None = None, pending_indexing_list: list = None) -> list:
    """
    Retrieve case laws for a single dispute component.

    Statute-anchored retrieval: in addition to a dispute-based query, run case-law
    search on statute-anchored queries (e.g. "Section 10 ActName case law interpretation")
    so cases that interpret the retrieved provisions rank higher.

    Round 1 — local hybrid search:
        Queries: (1) dispute + sections, (2–3) per top section "Section X ActName case law".
        Merge by chunk key, keep best score, quality filter, limit.
        → if already at 5 → STOP

    Round 2 — tiered web search (case laws only):
        → merge with local, deduplicate, re-apply limit.
    """
    from retrieval.hybrid_retriever import search_case_laws_auto

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")
    debug_entry = None
    if debug is not None:
        debug_entry = debug.setdefault(dispute_id, {})

    # Build queries: dispute+sections + statute-anchored (roadmap: cases interpreting those sections)
    queries = [_build_case_law_query(dispute_text, bare_act_sections)]
    for ba in bare_act_sections[:2]:  # top 2 sections
        act = (ba.get("act_name") or "").strip()
        sec = (ba.get("section_number") or "").strip()
        if act and sec:
            queries.append(f"Section {sec} {act} case law interpretation")
    queries = list(dict.fromkeys(queries))[:3]  # dedupe, cap 3

    # --- Round 1: local (multi-query merge) ---
    seen_chunk_keys = {}
    for search_query in queries:
        logger.debug("[%s] Case law query: '%s'", dispute_id, search_query[:80])
        local_raw = search_case_laws_auto(search_query, top_k=20)
        for cl in local_raw:
            key = cl.get("_chunk_key") or (
                (cl.get("case_name") or "").lower() + "|" + (cl.get("paragraph_num") or cl.get("full_text", "")[:100])
            )
            prev_score = (seen_chunk_keys.get(key) or {}).get("_rerank_score", -999)
            if (cl.get("_rerank_score", 0) > prev_score):
                seen_chunk_keys[key] = cl

    local_raw = list(seen_chunk_keys.values())
    local_results = [
        cl for cl in local_raw
        if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
    ]
    limited = _apply_dispute_case_law_limit(local_results, num_sections=len(bare_act_sections))
    logger.info("[%s] Case laws Round 1 local: %d after filter, %d after limit", dispute_id, len(local_results), len(limited))

    # If we got the maximum (5 high-quality), stop — skip LLM filter, already sufficient.
    if len(limited) >= 5:
        logger.info("[%s] Case laws Round 1 sufficient (5 high-quality). Stopping.", dispute_id)
        return limited

    # --- Round 2: web search ---
    logger.info("[%s] Case laws Round 2 — web search (local had %d)", dispute_id, len(limited))
    web_results = _web_search_case_laws(dispute, bare_act_sections, full_query, states=states, pending_indexing_list=pending_indexing_list)

    # Merge, deduplicate, re-apply numeric filter, then LLM relevance filter + per-dispute limit
    merged = _merge_deduplicate_case_laws(local_results, web_results)
    merged_filtered = [
        cl for cl in merged
        if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)
    ]
    llm_filtered = _filter_case_laws_with_llm(dispute, bare_act_sections, merged_filtered, debug)
    final = _apply_dispute_case_law_limit(llm_filtered, num_sections=len(bare_act_sections))
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
# BARE ACTS PHASE — present sections, explain, ask follow-up, then final opinion
# ---------------------------------------------------------------------------

def _explain_sections_and_get_followup(dispute_text: str, bare_acts: list, conversation_history: list = None) -> dict:
    """
    Single LLM call: add a contextual explanation to each retrieved section
    and generate ONE targeted follow-up question (or None if facts are sufficient).
    Returns {"bare_acts": [...with "explanation" field...], "followup_question": str|None}
    """
    if not bare_acts:
        return {"bare_acts": [], "followup_question": None}

    from prompts.advocate_prompts import BARE_ACT_EXPLAIN_AND_FOLLOWUP_PROMPT

    # Build a concise list — cap at 10 sections to keep prompt manageable
    lines = []
    for ba in bare_acts[:10]:
        section_text = (ba.get("full_text") or ba.get("text") or "")[:250]
        lines.append(
            f"- {ba.get('act_name', 'Unknown Act')} § {ba.get('section_number', '?')} "
            f"({ba.get('section_title', '')}): {section_text}"
        )

    # Build "questions already asked" block from conversation history so the prompt
    # cannot re-ask anything covered during intake.
    questions_already_asked_block = ""
    if conversation_history:
        asked: list[str] = []
        user_facts: list[str] = []
        for msg in conversation_history:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                import re as _re
                for sentence in _re.split(r"(?<=[.!?])\s+", content):
                    s = sentence.strip()
                    if "?" in s and len(s) > 12:
                        asked.append(s)
            elif role == "user" and len(content) > 5:
                user_facts.append(content[:300])
        lines_block = []
        if user_facts:
            lines_block.append("FACTS ALREADY STATED BY CLIENT — do NOT ask about any of these again:")
            for i, f in enumerate(user_facts, 1):
                lines_block.append(f"  [{i}] {f}")
        if asked:
            lines_block.append("QUESTIONS ALREADY ASKED IN THIS SESSION — NEVER repeat these:")
            for i, q in enumerate(asked, 1):
                lines_block.append(f"  [{i}] {q}")
            lines_block.append("Your additional_info_items MUST NOT duplicate any of the above questions.")
        questions_already_asked_block = "\n".join(lines_block)

    prompt = BARE_ACT_EXPLAIN_AND_FOLLOWUP_PROMPT.format(
        dispute_facts=dispute_text[:800],
        bare_acts_list="\n".join(lines),
        questions_already_asked=questions_already_asked_block,
    )

    try:
        raw = ask_llm(prompt)
        parsed = _extract_json(raw)
        if not parsed or not isinstance(parsed, dict):
            raise ValueError("No valid JSON dict found in LLM response")
    except Exception as e:
        logger.warning("_explain_sections_and_get_followup: LLM/JSON failed (%s). Using sections as-is.", e)
        return {"bare_acts": bare_acts, "followup_question": None}

    # Build a lookup: (act_name_lower, section_number_str) → explanation
    explanation_map = {}
    for item in parsed.get("section_explanations", []):
        key = (
            (item.get("act_name") or "").strip().lower(),
            str(item.get("section_number") or "").strip(),
        )
        explanation_map[key] = (item.get("explanation") or "").strip()

    enriched = []
    for ba in bare_acts:
        key = (
            (ba.get("act_name") or "").strip().lower(),
            str(ba.get("section_number") or "").strip(),
        )
        enriched.append({**ba, "explanation": explanation_map.get(key, "")})

    # Build one "request for additional information" from bullet list (additional_info_items)
    items = parsed.get("additional_info_items") or []
    if isinstance(items, list) and len(items) > 0:
        items = [str(x).strip() for x in items if str(x).strip()]
    if items:
        followup = (
            "Before I proceed to the full legal opinion, there are a few details that would help "
            "me apply the right provisions more precisely. Could you clarify:\n"
            + "\n".join("• " + x for x in items)
        )
    else:
        followup = (parsed.get("followup_question") or "").strip() or None
    if followup and len(followup) < 15:
        followup = None

    return {"bare_acts": enriched, "followup_question": followup}


def _link_case_laws_to_sections(bare_acts: list, case_laws: list) -> list:
    """
    Associate each case law with the most relevant bare act section.
    Strategy: scan case law text for section numbers that appear in our bare acts list.
    Falls back to assigning unmatched case laws to the top-scored bare act section.
    Returns a copy of bare_acts with "related_case_laws" populated on each entry.
    """
    result = [{**ba, "related_case_laws": []} for ba in bare_acts]
    if not case_laws or not result:
        return result

    unmatched = []
    for cl in case_laws:
        cl_text = ((cl.get("text") or cl.get("full_text") or "") + " " +
                   (cl.get("case_name") or cl.get("title") or "")).lower()
        matched = False
        for ba_entry in result:
            sec_num = str(ba_entry.get("section_number") or "").strip()
            act_keywords = [
                w for w in (ba_entry.get("act_name") or "").lower().split()
                if len(w) > 3 and w not in ("the", "and", "of", "for", "act,", "act")
            ]
            if sec_num and sec_num in cl_text:
                # Require at least one act keyword too (avoids false matches on common numbers)
                if not act_keywords or any(w in cl_text for w in act_keywords[:3]):
                    ba_entry["related_case_laws"].append(cl)
                    matched = True
                    break
        if not matched:
            unmatched.append(cl)

    # Assign unmatched case laws to the highest-scored bare act section
    if unmatched and result:
        result[0]["related_case_laws"].extend(unmatched)

    return result


def _distribute_case_laws_across_sections(sections_with_cases: list) -> list:
    """
    Rebalance linked case laws across sections within one dispute.

    Goals:
    - 1 section: keep up to 3 most relevant case laws
    - 2+ sections: keep up to 5 total across the dispute
    - give the strongest section 2-3 case laws when available
    - allow remaining sections to keep 1-2 each when relevant
    """
    if not sections_with_cases:
        return sections_with_cases

    sections = [{**s, "related_case_laws": list(s.get("related_case_laws", []))} for s in sections_with_cases]
    sections.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)

    total_cap = 5 if len(sections) > 1 else 3
    per_section_caps = [3] + [2] * max(0, len(sections) - 1)
    used_case_keys: set[str] = set()
    total_kept = 0

    def _case_key(cl: dict) -> str:
        return (
            (cl.get("case_name") or cl.get("title") or cl.get("source") or "").strip().lower()
            + "|"
            + str(cl.get("paragraph_num") or "")[:20]
            + "|"
            + (cl.get("_chunk_key") or "")[:80]
        )

    for idx, sec in enumerate(sections):
        if total_kept >= total_cap:
            sec["related_case_laws"] = []
            continue
        cap = per_section_caps[idx] if idx < len(per_section_caps) else 1
        cap = min(cap, total_cap - total_kept)
        ranked = sorted(sec.get("related_case_laws", []), key=lambda x: x.get("_rerank_score", 0), reverse=True)
        kept = []
        for cl in ranked:
            key = _case_key(cl)
            if key in used_case_keys:
                continue
            kept.append(cl)
            used_case_keys.add(key)
            if len(kept) >= cap:
                break
        sec["related_case_laws"] = kept
        total_kept += len(kept)

    return sections


def retrieve_bare_acts_phase(facts_summary: str, progress_callback=None, states: list = None, pending_indexing_list: list = None, conversation_history: list = None, step_callback=None) -> dict:
    """
    Phase A of the new two-phase flow.
    1. Decompose disputes.
    2. Retrieve relevant bare act sections for all disputes.
    3. Add contextual explanations to each section.
    4. Generate one targeted follow-up question (or None if not needed).

    Returns:
      {
        "bare_acts": [list with "explanation" field on each entry],
        "followup_question": str | None,
        "intro_text": str,   # human-readable intro shown in chat
      }
    """
    from services.dispute_decomposer import decompose_disputes

    def _step(msg: str, icon: str = ""):
        if step_callback:
            try:
                step_callback({"message": msg, "icon": icon})
            except Exception:
                pass

    if progress_callback:
        progress_callback({"step": "bare_acts", "message": "Retrieving relevant bare act sections…"})

    _step("Analysing your legal situation...", "🔍")
    try:
        disputes = decompose_disputes(facts_summary)
        _step(f"Identified {len(disputes)} legal dispute component{'s' if len(disputes) != 1 else ''}", "📋")
    except Exception as e:
        logger.warning("retrieve_bare_acts_phase: decompose_disputes failed (%s). Using single dispute.", e)
        disputes = [{"id": "d1", "dispute": facts_summary[:300], "legal_nature": "both",
                     "keywords": [], "bare_act_hints": [], "search_angles": []}]

    _step("Retrieving relevant legal provisions...", "📖")
    # Retrieve bare acts per dispute in parallel; tag each section with its dispute origin
    # so Phase B can reconstruct per-dispute groupings without re-running decomposition.
    all_bare_acts: list = []
    dispute_errors: list[dict] = []

    _states = states or []

    def _fetch_for_dispute(d):
        sections = retrieve_bare_acts_for_dispute(d, facts_summary, states=_states, pending_indexing_list=pending_indexing_list)
        for s in sections:
            # setdefault: first dispute tag wins if a section appears in multiple disputes
            s.setdefault("_dispute_id", d["id"])
            s.setdefault("_dispute_text", d.get("dispute", ""))
        return sections

    with ThreadPoolExecutor(max_workers=min(len(disputes), 3)) as ex:
        futures = {ex.submit(_fetch_for_dispute, d): d for d in disputes}
        for fut in futures:
            try:
                all_bare_acts.extend(fut.result())
            except Exception as exc:
                dispute = futures[fut]
                dispute_errors.append({
                    "dispute_id": dispute.get("id", "?"),
                    "dispute": (dispute.get("dispute", "") or "")[:200],
                    "error": repr(exc),
                })
                logger.exception(
                    "retrieve_bare_acts_phase: dispute retrieval error for %s: %s",
                    dispute.get("id", "?"),
                    exc,
                )

    # Deduplicate across disputes (first-seen dispute tag is preserved via setdefault above)
    all_bare_acts = _merge_deduplicate_bare_acts(all_bare_acts, [])
    # Safety cap: max 15 total across all disputes (5 per dispute × 3 disputes typical)
    all_bare_acts = _cap_bare_acts_by_score(all_bare_acts, MAX_BARE_ACTS_OVERALL)

    if not all_bare_acts:
        if dispute_errors:
            logger.error(
                "retrieve_bare_acts_phase: no bare acts returned because all dispute workers failed. "
                "errors=%s",
                dispute_errors,
            )
        else:
            logger.warning(
                "retrieve_bare_acts_phase: no bare acts found after local/web retrieval and filtering. "
                "facts_summary=%r disputes=%s states=%s",
                facts_summary[:300],
                [d.get("dispute", "")[:120] for d in disputes],
                _states,
            )

    if all_bare_acts:
        _step(f"Found {len(all_bare_acts)} relevant legal provision{'s' if len(all_bare_acts) != 1 else ''} — preparing summary...", "✅")
    else:
        _step("No provisions found in local database — checking web sources...", "⚠️")

    if progress_callback:
        progress_callback({"step": "bare_acts_explain",
                           "message": f"Explaining {len(all_bare_acts)} section(s)…"})

    # Explain sections + generate follow-up
    enriched = _explain_sections_and_get_followup(facts_summary, all_bare_acts, conversation_history=conversation_history or [])

    flat_bare_acts = enriched["bare_acts"]  # flat list (kept for Phase B backward compat)

    # Group sections by dispute for the new per-dispute UI layout
    from collections import OrderedDict as _OD
    _dispute_map: _OD = _OD()
    for ba in flat_bare_acts:
        did = ba.get("_dispute_id") or "d_main"
        dtext = ba.get("_dispute_text") or facts_summary[:200]
        if did not in _dispute_map:
            _dispute_map[did] = {"id": did, "dispute": dtext, "sections": []}
        _dispute_map[did]["sections"].append(ba)
    disputes_grouped = list(_dispute_map.values())

    # Build a friendly advocate-voice intro
    n = len(flat_bare_acts)
    nd = len(disputes_grouped)
    if n == 0:
        intro = (
            "I've reviewed your situation carefully. At this stage I wasn't able to locate specific statutory provisions "
            "in the database for your query, but let me walk you through the general legal position and what you can do next."
        )
    elif nd == 1:
        section_word = "provision" if n == 1 else "provisions"
        intro = (
            f"I've had a chance to look into your situation. Based on what you've shared, "
            f"I've identified {n} legal {section_word} that are directly relevant to your case. "
            "Here is the legal protection available to you — and what each provision means in your specific circumstances."
        )
    else:
        dispute_word = "distinct legal issues" if nd > 1 else "legal issue"
        section_word = "provision" if n == 1 else "provisions"
        intro = (
            f"I've reviewed your situation carefully. I can see {nd} {dispute_word} arising from the facts you've described, "
            f"and I've identified {n} legal {section_word} across these. "
            "Let me walk you through the legal protection available to you for each."
        )

    # Ensure we always offer a clear next step to the client.
    followup = enriched.get("followup_question")
    if not followup:
        if n == 0:
            # No sections found: explicitly offer judgments + full opinion as the next step.
            followup = (
                "Even though I couldn't match a specific bare act section in the database right now, "
                "I can still search for relevant court judgments and prepare a full legal opinion for you. "
                "Would you like me to go ahead and do that next?"
            )
        else:
            # Sections found but no follow-up suggested — offer to proceed with detailed opinion.
            followup = (
                "Based on these provisions, I can now prepare a detailed opinion that also brings in key judgments "
                "on situations like yours. Shall I proceed with that for you?"
            )

    return {
        "disputes": disputes_grouped,        # new: grouped by dispute for UI rendering
        "bare_acts": flat_bare_acts,        # kept: flat list used by Phase B (case laws + opinion)
        "followup_question": followup,
        "intro_text": intro,
    }


def generate_final_opinion_with_case_laws(
    facts_summary: str,
    bare_acts: list,
    additional_info: str = "",
    progress_callback=None,
    states: list = None,
    pending_indexing_list: list = None,
    step_callback=None,
    token_callback=None,
) -> dict:
    """
    Phase B of the new two-phase flow.
    1. Build enriched facts (original + any additional info user provided).
    2. Reconstruct per-dispute groupings from _dispute_id/_dispute_text tags on bare_acts
       (no new LLM decomposition — reuse Phase A's decomposition directly).
    3. Retrieve case laws per dispute group in parallel.
    4. Link case laws to sections within each dispute group.
    5. Build dispute_blocks_text and generate structured opinion:
       Dispute N → Section (verbatim + how it applies) → Case laws (what parts apply and how).

    Returns the standard generate_response_v2() response dict.
    """
    from collections import OrderedDict
    from prompts.advocate_prompts import STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT

    bare_acts, _ = _filter_materials_to_local_db(bare_acts, None)
    full_facts = facts_summary
    if (additional_info or "").strip():
        full_facts = f"{facts_summary}\n\nAdditional information provided by client: {additional_info.strip()}"

    if progress_callback:
        progress_callback({"step": "case_laws", "message": "Searching for relevant case laws…"})

    # --- Reconstruct per-dispute groupings from Phase A tags (no re-decomposition) ---
    dispute_groups: OrderedDict = OrderedDict()  # dispute_id → {id, dispute, sections}
    for ba in bare_acts:
        did = ba.get("_dispute_id") or "d_main"
        dtext = ba.get("_dispute_text") or full_facts[:300]
        if did not in dispute_groups:
            dispute_groups[did] = {
                "id": did,
                "dispute": dtext,
                "keywords": [],
                "bare_act_hints": [],
                "search_angles": [],
                "sections": [],
            }
        dispute_groups[did]["sections"].append(ba)

    # Fallback: single group with all bare acts if Phase A tags are absent
    if not dispute_groups:
        dispute_groups["d_main"] = {
            "id": "d_main",
            "dispute": full_facts[:300],
            "keywords": [],
            "bare_act_hints": [],
            "search_angles": [],
            "sections": list(bare_acts),
        }

    # --- Retrieve case laws per dispute group in parallel ---
    _states = states or []
    _pending = pending_indexing_list if pending_indexing_list is not None else []

    def _fetch_case_laws_for_group(grp):
        cls = retrieve_case_laws_for_dispute(grp, grp["sections"], full_facts, states=_states, pending_indexing_list=_pending)
        return grp["id"], cls

    all_case_laws: list = []
    case_laws_by_dispute: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(dispute_groups), 3)) as ex:
        futures = {ex.submit(_fetch_case_laws_for_group, grp): grp
                   for grp in dispute_groups.values()}
        for fut in futures:
            try:
                did, cls = fut.result()
                case_laws_by_dispute[did] = cls
                all_case_laws.extend(cls)
            except Exception as exc:
                logger.warning("generate_final_opinion_with_case_laws: case law error: %s", exc)

    all_case_laws = _deduplicate_case_laws(all_case_laws)
    _, all_case_laws = _filter_materials_to_local_db(None, all_case_laws)
    case_laws_by_dispute = {
        did: [cl for cl in cls if _is_local_db_source(cl)]
        for did, cls in case_laws_by_dispute.items()
    }

    # --- Link case laws to sections within each dispute group ---
    all_bare_acts_with_cases: list = []
    for did, grp in dispute_groups.items():
        grp_case_laws = case_laws_by_dispute.get(did, [])
        grp["sections_with_cases"] = _link_case_laws_to_sections(grp["sections"], grp_case_laws)
        grp["sections_with_cases"] = _distribute_case_laws_across_sections(grp["sections_with_cases"])
        grp["sections_with_cases"], _ = _filter_materials_to_local_db(grp["sections_with_cases"], None)
        all_bare_acts_with_cases.extend(grp["sections_with_cases"])

    # --- Build dispute_blocks_text for STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT ---
    # Format: Dispute → Section (verbatim + explanation) → Case laws (what parts apply and how)
    # Use section chunk limit for verbatim (not entire act)
    dispute_blocks = []
    for i, (did, grp) in enumerate(dispute_groups.items(), start=1):
        lines = [f"--- DISPUTE COMPONENT {i} ---"]
        lines.append(f"Dispute: {grp['dispute']}")
        lines.append("")
        for ba in grp.get("sections_with_cases", grp["sections"]):
            act = ba.get("act_name", "Unknown Act")
            sec = ba.get("section_number", "?")
            title = ba.get("section_title", "")
            verbatim = (ba.get("full_text") or ba.get("text") or "").strip()[:600]
            expl = (ba.get("explanation") or "").strip()
            lines.append(f"[{act}, Section {sec} — {title}]")
            lines.append(f'Verbatim section text: "{verbatim}"')
            if expl:
                lines.append(f"Applicability note (from Phase A): {expl}")
            related = ba.get("related_case_laws", [])
            if related:
                lines.append("Case laws linked to this section:")
                for cl in related[:3]:
                    name = _format_case_citation(cl)
                    body = (cl.get("text") or cl.get("full_text") or "")[:350]
                    lines.append(f"  - [{name}]: {body}")
            lines.append("")
        dispute_blocks.append("\n".join(lines))

    dispute_blocks_text = "\n".join(dispute_blocks) or "None retrieved."
    if not all_bare_acts_with_cases and not all_case_laws:
        return {
            "bare_act_sections": [],
            "case_laws": [],
            "internet_case_laws": [],
            "explanation": _local_only_no_materials_message(),
            "progress": None,
            "indexing_candidates": [],
        }

    # --- Build strict citation allowlist ---
    _final_sec_allowlist = [
        f"{ba.get('act_name', '')} § {ba.get('section_number', '')}"
        for ba in bare_acts[:20] if ba.get("section_number")
    ]
    _final_case_allowlist = [
        _format_case_citation(cl)
        for cl in all_case_laws[:10]
        if (cl.get("case_name") or cl.get("title") or "").strip()
    ]
    _allowlist_suffix = (
        f"\n\n🚨 STRICT CITATION RULES — NO EXCEPTIONS:\n"
        f"• Sections you may cite: {_final_sec_allowlist if _final_sec_allowlist else ['NONE']}\n"
        f"• Cases you may cite: {_final_case_allowlist if _final_case_allowlist else ['NONE']}\n"
        f"• Do NOT add any IPC, CrPC, or other section numbers / case names from training knowledge "
        f"that are NOT in the above lists. Every section and case in the opinion must be from "
        f"RETRIEVED MATERIALS BY DISPUTE COMPONENT above."
    )

    opinion_prompt = STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.format(
        dispute_facts=full_facts[:800],
        additional_info=(additional_info or "None provided").strip(),
        dispute_blocks_text=dispute_blocks_text,
    ) + _allowlist_suffix

    if progress_callback:
        progress_callback({"step": "opinion", "message": "Generating structured legal opinion…"})
    if step_callback:
        try:
            step_callback({"message": "Now I have everything I need — composing your legal analysis...", "icon": "💡"})
        except Exception:
            pass

    try:
        if token_callback:
            opinion_text = ""
            for _tok in ask_llm_stream(opinion_prompt):
                token_callback(_tok)
                opinion_text += _tok
            opinion_text = opinion_text.strip()
        else:
            opinion_text = ask_llm(opinion_prompt).strip()
    except Exception as e:
        logger.error("generate_final_opinion_with_case_laws: opinion LLM failed: %s", e)
        opinion_text = "I was unable to generate a structured opinion at this time. Please try again."

    opinion_text = _enforce_local_grounding(opinion_text, all_bare_acts_with_cases)

    return {
        "bare_act_sections": all_bare_acts_with_cases,
        "case_laws": [],           # Nested under bare acts — no top-level duplicates
        "internet_case_laws": [],
        "explanation": opinion_text,
        "progress": None,
        "indexing_candidates": [],
    }


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
    step_callback=None,
    token_callback=None,
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

    def _emit_step(message: str, icon: str = ""):
        """Emit a single clean user-facing step message (separate from ProgressTracker)."""
        if step_callback:
            try:
                step_callback({"message": message, "icon": icon})
            except Exception:
                pass

    if intent in ("legal_opinion", "search", "lookup") and search_strategy != "local_only":
        logger.info(
            "Overriding legal search strategy '%s' -> 'local_only' for strict local vector store grounding",
            search_strategy,
        )
        search_strategy = "local_only"
    if search_strategy not in ("local_only", "web_only", "local_then_web"):
        search_strategy = "local_only"

    # Intent extraction — used for query expansion
    research_intent = None
    try:
        from services.intent_extractor import extract_research_intent
        research_intent = extract_research_intent(facts_summary)
    except Exception as e:
        logger.debug("Intent extraction skipped: %s", e)

    # Step 1: Query expansion (returns 1–3 queries for multi-query retrieval)
    legal_queries = expand_legal_query(facts_summary, intent=research_intent)
    legal_query = legal_queries[0] if legal_queries else facts_summary[:300]
    logger.info("Expanded query(s): %s", legal_query[:200] if legal_query else "none")

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
    _emit_step("Analysing your legal situation...", "🔍")
    progress.start_group("Dispute Analysis", "Identifying distinct dispute components")
    _emit_progress()

    if intent in ("search", "lookup"):
        # Direct search/lookup — no decomposition needed
        disputes = [{"id": "d1", "dispute": facts_summary[:300], "legal_nature": "both", "keywords": []}]
        progress.add_step("Direct search/lookup — treating as single query", {"disputes": 1})
        _emit_step("Searching legal database...", "🔎")
    else:
        from services.dispute_decomposer import decompose_disputes
        disputes = decompose_disputes(facts_summary)
        labels = [d.get("dispute", "")[:60] for d in disputes]
        progress.add_step(
            f"Identified {len(disputes)} dispute component(s)",
            {"count": len(disputes), "disputes": labels},
        )
        logger.info("Disputes identified: %s", labels)
        _emit_step(
            f"Broken down into {len(disputes)} legal dispute component{'s' if len(disputes) != 1 else ''}",
            "📋",
        )

    _emit_progress()
    progress.finish_group()
    _emit_progress()

    # State list for jurisdiction-aware web search (HC domain, IndiaCode state filter)
    _states = [jurisdiction_state] if (jurisdiction_state or "").strip() else []

    # Step 3: Per-dispute retrieval — all disputes run in parallel.
    # _emit_step / step_callback wraps queue.put() which is thread-safe.
    # Each worker uses its own local pending/debug lists; we merge them back
    # to the shared structures in the main thread after all futures complete.
    pending_indexing_list = []
    dispute_results = []
    debug_pipeline = {
        "disputes": disputes,
        "per_dispute": {},
    }

    def _retrieve_dispute(dispute):
        """Worker: retrieve bare acts + case laws for one dispute component."""
        d_id = dispute.get("id", "?")
        d_text = dispute.get("dispute", "")
        _local_pending: list = []
        _local_debug: dict = {}

        _emit_step(f"[{d_id}] Identifying legal provisions...", "📖")

        # 3a: Bare acts
        bare_d = []
        if retrieve_acts:
            if search_strategy == "web_only":
                bare_d = _web_search_bare_acts(dispute, facts_summary, states=_states, pending_indexing_list=_local_pending)
                bare_d = [ba for ba in bare_d if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)]
            elif search_strategy == "local_only":
                from retrieval.hybrid_retriever import search_bare_acts_auto
                # Multi-query: run all _build_bare_act_queries and merge by best score per section.
                queries_ba = _build_bare_act_queries(dispute)
                seen_ba: dict = {}
                for _q_ba in queries_ba:
                    for _ba in search_bare_acts_auto(_q_ba, top_k=30):
                        _key = (
                            (_ba.get("act_name") or "").strip().lower(),
                            (_ba.get("section_number") or "").strip().lower(),
                        )
                        if _ba.get("_rerank_score", 0) > (seen_ba.get(_key) or {}).get("_rerank_score", -999):
                            seen_ba[_key] = _ba
                raw = list(seen_ba.values())
                bare_d = [ba for ba in raw if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)]
            else:
                bare_d = retrieve_bare_acts_for_dispute(dispute, facts_summary, states=_states, debug=_local_debug, pending_indexing_list=_local_pending)

            if bare_d:
                act_names = list(dict.fromkeys(
                    ba.get("act_name") or ba.get("title", "").split(" §")[0]
                    for ba in bare_d if ba.get("act_name") or ba.get("title")
                ))[:4]
                acts_str = ", ".join(act_names) if act_names else "legal provisions"
                _emit_step(f"[{d_id}] Found {len(bare_d)} section{'s' if len(bare_d) != 1 else ''} — {acts_str}", "✅")
            else:
                _emit_step(f"[{d_id}] No bare act sections found — trying web...", "⚠️")

        # 3b: Case laws
        case_d = []
        if retrieve_case_laws_flag:
            _emit_step(f"[{d_id}] Searching for judicial precedents...", "⚖️")
            if search_strategy == "web_only":
                raw_cl = _web_search_case_laws(dispute, bare_d, facts_summary, states=_states, pending_indexing_list=_local_pending)
                case_d = _apply_dispute_case_law_limit([cl for cl in raw_cl if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)])
            elif search_strategy == "local_only":
                from retrieval.hybrid_retriever import search_case_laws_auto
                # Multi-query: base dispute query + up to 2 section-anchored queries, merge best per chunk.
                _q_base = _build_case_law_query(d_text, bare_d)
                _queries_cl = [_q_base]
                for _ba in bare_d[:2]:
                    _act = (_ba.get("act_name") or "").strip()
                    _sec = (_ba.get("section_number") or "").strip()
                    if _act and _sec:
                        _queries_cl.append(f"Section {_sec} {_act} case law interpretation")
                _queries_cl = list(dict.fromkeys(_queries_cl))[:3]
                seen_cl: dict = {}
                for _q_cl in _queries_cl:
                    for _cl in search_case_laws_auto(_q_cl, top_k=20):
                        _ck = _cl.get("_chunk_key") or (
                            (_cl.get("case_name") or "").lower() + "|" + (_cl.get("full_text", "")[:100])
                        )
                        if _cl.get("_rerank_score", 0) > (seen_cl.get(_ck) or {}).get("_rerank_score", -999):
                            seen_cl[_ck] = _cl
                raw_cl = list(seen_cl.values())
                case_d = _apply_dispute_case_law_limit([cl for cl in raw_cl if cl.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_case_law(cl)])
            else:
                case_d = retrieve_case_laws_for_dispute(dispute, bare_d, facts_summary, states=_states, debug=_local_debug, pending_indexing_list=_local_pending)

            # Citation graph expansion: boost recall by fetching chunks for cases
            # that the retrieved cases cite (forward) or that cite them (backward).
            # Runs after all retrieval paths so it always supplements the best results.
            if case_d:
                try:
                    from retrieval.citation_graph import (
                        expand_case_names_by_precedent,
                        get_chunks_by_case_names,
                    )
                    from retrieval.hybrid_retriever import load_chunks as _hr_load_chunks
                    from config import CASE_CHUNKS_V2
                    _top_names = [
                        cl.get("case_name", "")
                        for cl in case_d
                        if cl.get("case_name")
                    ]
                    _extra_names, _ = expand_case_names_by_precedent(_top_names, max_extra=8)
                    if _extra_names:
                        _chunks_dict = _hr_load_chunks(CASE_CHUNKS_V2)
                        _exp_chunks = get_chunks_by_case_names(_chunks_dict, _extra_names, max_total=6)
                        if _exp_chunks:
                            _existing_names = {
                                (cl.get("case_name") or "").strip().lower()
                                for cl in case_d
                            }
                            _new = [
                                c for c in _exp_chunks
                                if (c.get("case_name") or "").strip().lower() not in _existing_names
                                and _is_quality_case_law(c)
                            ]
                            if _new:
                                logger.info(
                                    "[%s] Citation graph expansion: +%d cases added", d_id, len(_new)
                                )
                                case_d = case_d + _new
                except Exception as _cg_err:
                    logger.debug("[%s] Citation graph expansion skipped: %s", d_id, _cg_err)

            if case_d:
                sc_count = sum(1 for cl in case_d if "supreme" in (cl.get("court") or cl.get("source") or "").lower())
                detail = f"including {sc_count} Supreme Court" if sc_count else "from High Courts"
                _emit_step(f"[{d_id}] Found {len(case_d)} judgement{'s' if len(case_d) != 1 else ''} ({detail})", "✅")
            else:
                _emit_step(f"[{d_id}] No judgements found in local database", "ℹ️")

        # Tag with dispute ID before returning
        for _ba in bare_d:
            _ba.setdefault("_dispute_id", d_id)
        for _cl in case_d:
            _cl.setdefault("_dispute_id", d_id)

        return {
            "dispute": dispute,
            "bare_acts": bare_d,
            "case_laws": case_d,
            "_pending": _local_pending,
            "_debug": _local_debug,
        }

    # Submit all disputes to the thread pool; collect as each completes.
    progress.start_group("Research", f"Retrieving bare acts and case laws for {len(disputes)} dispute(s)")
    _emit_progress()
    with ThreadPoolExecutor(max_workers=min(len(disputes), 3)) as executor:
        futures = {executor.submit(_retrieve_dispute, d): d for d in disputes}
        for fut in as_completed(futures):
            try:
                dr = fut.result()
                pending_indexing_list.extend(dr.pop("_pending", []))
                debug_pipeline["per_dispute"].update(dr.pop("_debug", {}))
                dispute_results.append(dr)
                progress.add_step(
                    f"[{dr['dispute'].get('id','?')}] Retrieved {len(dr['bare_acts'])} sections, {len(dr['case_laws'])} judgements",
                    {"dispute": dr["dispute"].get("dispute", "")[:60]},
                )
                _emit_progress()
            except Exception as exc:
                logger.error("Dispute retrieval failed: %s", exc)
    progress.finish_group()
    _emit_progress()

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
    formatted_bare, _ = _filter_materials_to_local_db(formatted_bare, None)
    local_dispute_results = _filter_dispute_results_to_local_db(dispute_results)

    # Search-only: wrap case laws in a bare-act shell so UI renders them
    if intent == "search" and not formatted_bare and formatted_case:
        limit = max(1, result_count) if result_count else max(FLEXIBLE_MIN_FALLBACK, len(formatted_case))
        formatted_bare = [{
            "act_name": "Case laws",
            "title": "Case laws (search results)",
            "text": "",
            "related_case_laws": [cl for cl in formatted_case[:limit] if _is_local_db_source(cl)],
            "source_tag": "LOCAL_DB",
        }]

    # Step 6: LLM opinion / summary
    if intent in ("search", "lookup"):
        progress.add_step("Generating search summary...")
        _emit_step("Now I have everything I need — composing your summary...", "💡")
    else:
        progress.add_step("Generating legal opinion...")
        _emit_step("Now I have everything I need — composing your legal analysis...", "💡")
    _emit_progress()

    flattened_case_laws = []
    for ba in formatted_bare:
        flattened_case_laws.extend(ba.get("related_case_laws", []))

    # Determine whether we have any grounded materials at all. If both bare acts
    # and case laws are empty, we must NOT generate a substantive legal opinion
    # from the model's prior training — only a truthful "no materials" message.
    has_materials = bool(formatted_bare or flattened_case_laws)

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

    if not has_materials:
        # Hard stop: no sections or case laws → do not fabricate a legal opinion.
        explanation = _format_no_materials_message(search_strategy, None)
    else:
        if intent in ("search", "lookup"):
            explanation = _generate_conversational_summary(
                facts_summary, formatted_bare, flattened_case_laws, intent=intent,
                token_callback=token_callback,
            )
        else:
            # Legal opinion: structured by dispute (Dispute Summary → Section + why + precedents per component → Legal Position)
            explanation = _generate_structured_opinion_by_dispute(
                facts_summary, local_dispute_results, additional_info="",
                token_callback=token_callback,
            )
            if not (explanation or "").strip():
                explanation = _generate_legal_opinion(
                    facts_summary, formatted_bare, flattened_case_laws, sufficiency
                )

        if not (explanation or "").strip():
            explanation = "Here's what I found for your query. Below are the relevant legal provisions with related case laws."
        if explanation and "I don't have any data" in explanation:
            explanation = _format_no_materials_message(search_strategy, None)
        explanation = _enforce_local_grounding(explanation, formatted_bare)

    sources_used = set()
    for ba in formatted_bare:
        sources_used.add(ba.get("source_tag", "LOCAL_DB"))
        for cl in ba.get("related_case_laws", []):
            sources_used.add(cl.get("source_tag", "LOCAL_DB"))

    # Finish the main research/opinion group
    progress.finish_group()
    _emit_progress()

    # Optional: add a separate debug group with intermediate stages per dispute
    try:
        progress.start_group(
            "Debug pipeline",
            "Intermediate retrieval stages per dispute (acts, sections, case laws)",
        )
        for d in disputes:
            d_id = d.get("id", "?")
            d_text = d.get("dispute", "")[:120]
            dbg = debug_pipeline.get("per_dispute", {}).get(d_id, {})

            progress.add_step(f"[{d_id}] Dispute: {d_text}")

            bm25_acts = dbg.get("bm25_acts") or []
            if bm25_acts:
                progress.add_step(
                    f"[{d_id}] BM25 act candidates: " + ", ".join(bm25_acts[:5])
                )

            llm_acts = dbg.get("llm_acts") or []
            if llm_acts:
                progress.add_step(
                    f"[{d_id}] LLM-refined acts (high/medium): " + ", ".join(llm_acts[:5])
                )

            for i, ent in enumerate(dbg.get("bare_act_llm_section_filters") or []):
                ins = ent.get("input_sections") or []
                kept_count = ent.get("kept_count")
                msg = f"[{d_id}] Bare-act LLM filter #{i+1}: {len(ins)} → {kept_count if kept_count is not None else '?'} sections"
                progress.add_step(msg)

            for i, ent in enumerate(dbg.get("case_law_llm_filters") or []):
                ins = ent.get("input_cases") or []
                kept_count = ent.get("kept_count")
                msg = f"[{d_id}] Case-law LLM filter #{i+1}: {len(ins)} → {kept_count if kept_count is not None else '?'} results"
                progress.add_step(msg)

        progress.finish_group()
        _emit_progress()
    except Exception as _dbg_exc:
        logger.debug("Debug pipeline progress group failed: %s", _dbg_exc)

    return {
        "bare_act_sections": formatted_bare,
        "case_laws": [],                      # case laws are nested under bare acts
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "needs_confirmation": False,
        "internet_case_laws": [],             # backward compat
        "progress": progress.get_progress(),
        "indexing_candidates": [],
        "dispute_breakdown": [
            {
                "dispute_id": dr["dispute"].get("id"),
                "dispute": dr["dispute"].get("dispute"),
                "bare_acts_count": len(dr["bare_acts"]),
                "case_laws_count": len(dr["case_laws"]),
            }
            for dr in dispute_results
        ],
        "debug_pipeline": debug_pipeline,
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
    Match case laws to bare act sections using dispute-tag grouping and
    section-number / act-keyword text matching.  Zero cross-encoder calls.

    Previous implementation scored every (section, case_law) pair with the
    cross-encoder — O(sections × case_laws) predictions after all retrieval was
    already complete (e.g. 3 sections × 15 case laws = 45 extra predictions).
    The information needed to make the match already exists as the _dispute_id
    tag placed on both bare acts and case laws during retrieval, so we simply
    use that tag to group the lists and then apply lightweight text matching
    within each group.

    Strategy (in order):
      1. Group bare_acts and case_laws by _dispute_id.
      2. Within each dispute group, scan each case law's concatenated text+title
         for the section number of each bare act AND at least one act keyword.
         If found and the section has capacity (< max_per_section), assign it.
      3. Unmatched case laws in a dispute group fall back to the highest-scored
         section in that same group (first entry, already sorted by score desc).
      4. Case laws whose dispute group has no corresponding bare act group
         (edge case: all sections for that dispute were filtered out) are
         pushed to a _global fallback pool and matched against all remaining
         sections using the same text-based logic.

    Returns bare_acts list with 'related_case_laws' populated on each entry.
    """
    from collections import defaultdict

    # Initialise related_case_laws on every section
    for ba in bare_acts:
        ba["related_case_laws"] = []

    if not case_laws or not bare_acts:
        return bare_acts

    # ── Step 1: Group by dispute ID ──────────────────────────────────────────
    # bare_acts is already sorted by _rerank_score desc (from _format_bare_acts);
    # preserving that order means "first in group = highest-scored" for fallback.
    _GLOBAL = "_global"
    ba_by_dispute: dict = defaultdict(list)
    for ba in bare_acts:
        ba_by_dispute[ba.get("_dispute_id") or _GLOBAL].append(ba)

    cl_by_dispute: dict = defaultdict(list)
    for cl in case_laws:
        cl_by_dispute[cl.get("_dispute_id") or _GLOBAL].append(cl)

    # ── Step 2: Text-based matching within each dispute group ────────────────
    def _text_match_group(ba_group: list, cl_group: list) -> list:
        """Assign case laws in cl_group to sections in ba_group; return unmatched."""
        unmatched = []
        for cl in cl_group:
            cl_text = (
                (cl.get("text") or cl.get("full_text") or "") + " " +
                (cl.get("case_name") or cl.get("title") or "")
            ).lower()

            placed = False
            for ba in ba_group:
                if len(ba.get("related_case_laws", [])) >= max_per_section:
                    continue
                sec_num = str(ba.get("section_number") or "").strip()
                act_keywords = [
                    w for w in (ba.get("act_name") or "").lower().split()
                    if len(w) > 3 and w not in ("the", "and", "of", "for", "act,", "act")
                ]
                # Section number must appear in the case law text AND at least
                # one act keyword must match (guards against false hits on common
                # numbers like "10" or "2" that appear in many unrelated judgments).
                if sec_num and sec_num in cl_text:
                    if not act_keywords or any(w in cl_text for w in act_keywords[:3]):
                        ba["related_case_laws"].append(cl)
                        placed = True
                        break
            if not placed:
                unmatched.append(cl)
        return unmatched

    # ── Step 3: Process each dispute group ───────────────────────────────────
    all_groups = set(ba_by_dispute.keys()) | set(cl_by_dispute.keys())
    orphan_case_laws: list = []  # case laws whose ba group is empty

    for group_id in all_groups:
        ba_group = ba_by_dispute.get(group_id, [])
        cl_group = cl_by_dispute.get(group_id, [])

        if not cl_group:
            continue

        if not ba_group:
            # No bare acts for this dispute — collect for global fallback
            orphan_case_laws.extend(cl_group)
            continue

        unmatched = _text_match_group(ba_group, cl_group)

        # Fallback: assign unmatched to the highest-scored section in this group
        if unmatched and ba_group:
            top_ba = ba_group[0]
            for cl in unmatched:
                if len(top_ba.get("related_case_laws", [])) < max_per_section:
                    top_ba["related_case_laws"].append(cl)

    # ── Step 4: Global fallback for orphaned case laws ───────────────────────
    if orphan_case_laws:
        remaining = _text_match_group(bare_acts, orphan_case_laws)
        # Anything still unmatched goes to the overall top section
        if remaining and bare_acts:
            top_ba = bare_acts[0]
            for cl in remaining:
                if len(top_ba.get("related_case_laws", [])) < max_per_section:
                    top_ba["related_case_laws"].append(cl)

    total_assigned = sum(len(ba.get("related_case_laws", [])) for ba in bare_acts)
    logger.info(
        "Matched %d case laws to %d bare act sections via dispute-tag + text matching "
        "(0 cross-encoder calls; replaced O(sections×case_laws) scoring)",
        total_assigned, len(bare_acts),
    )
    return bare_acts


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def _extract_leading_section_number(text: str) -> str:
    """Extract the section number from the opening line of verbatim bare-act text.

    Bare act text frequently starts with its own section header, e.g.:
        "115. Voluntarily causing grievous hurt..."
        "[115A. Punishment for…]"
        "Section 115 — Voluntarily causing…"

    When a chunk spans multiple sections and _trim_to_one_section() crops the
    text, the displayed section may differ from the chunk metadata's
    section_number field.  This function parses the *actual* leading section
    number so the UI card title and the text stay in sync.

    Returns the section number string (e.g. "115", "115A") or "" if the text
    does not begin with a recognisable section header.
    """
    import re as _re
    if not text:
        return ""
    text = text.strip()
    # Pattern 1: optional "[", 1-3 digits + optional uppercase letter, ". " + uppercase
    # Matches: "115. Voluntarily" / "[115A. Some title"
    m = _re.match(r'^\[?(\d{1,3}[A-Z]?)\.\s+[A-Z]', text)
    if m:
        return m.group(1)
    # Pattern 2: "Section 115" or "section 115A" at start of text
    m = _re.match(r'^[Ss]ection\s+(\d{1,3}[A-Z]?)\b', text)
    if m:
        return m.group(1)
    return ""


def _trim_to_one_section(text: str, max_chars: int = 1200) -> str:
    """Trim verbatim bare-act text to roughly one section's worth.

    Strategy (in order of preference):
    1. If text fits within max_chars already — return as-is.
    2. Look for the next section-number header after position 300
       (e.g. "\\n52. ", "\\n[52A. ") — cut just before it.
    3. Fall back to the last paragraph break (\\n\\n) before max_chars.
    4. Fall back to the last sentence boundary ('. ') before max_chars.
    5. Hard-cap at max_chars and append an ellipsis.
    """
    import re as _re
    if len(text) <= max_chars:
        return text
    # Pattern: newline(s) followed by an optional '[', 1-3 digits, optional letter, '. ', uppercase
    _NEXT_SEC = _re.compile(r'\n{1,2}\[?\d{1,3}[A-Z]?\.\s+[A-Z]')
    m = _NEXT_SEC.search(text, 300)
    if m and m.start() <= max_chars + 500:
        return text[:m.start()].rstrip()
    cut = text.rfind('\n\n', 200, max_chars)
    if cut > 200:
        return text[:cut].rstrip()
    cut = text.rfind('. ', 200, max_chars)
    if cut > 200:
        return text[:cut + 1].rstrip()
    return text[:max_chars].rstrip() + "…"


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
        # Trim to one section's worth — avoids dumping entire chapters into the
        # UI verbatim box when the underlying chunk carries multi-section text.
        text = _trim_to_one_section(text, max_chars=1200)

        act_name = ba.get("act_name", "")
        section = ba.get("section_number", "")
        title = ba.get("section_title", "")

        # Reconcile section number with actual text content.
        # When a chunk spans multiple sections, _trim_to_one_section() may expose
        # text from a section whose number differs from the chunk metadata's
        # section_number field.  Parse the leading header from the trimmed text
        # and prefer it — this keeps the UI card title in sync with what's shown.
        _sec_from_text = _extract_leading_section_number(text)
        if _sec_from_text and str(_sec_from_text) != str(section):
            logger.debug(
                "Section# reconciled: metadata=%r → text=%r (%s)",
                section, _sec_from_text, act_name,
            )
            section = _sec_from_text

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
            # Preserve dispute tag so _match_case_laws_to_bare_acts() can group
            # by dispute instead of re-scoring pairs with the cross-encoder.
            "_dispute_id": ba.get("_dispute_id", ""),
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
            "signature": (first.get("signature") or "").strip() or None,
            "_rerank_score": top3[0]["score"],
            "_year": _case_year_for_sort(first),
            # Carry dispute tag from the highest-scored chunk so the matching
            # step can group case laws by dispute without cross-encoder calls.
            "_dispute_id": first.get("_dispute_id", ""),
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


# Max sections per dispute to pass to structured opinion (most relevant only)
MAX_SECTIONS_PER_DISPUTE_FOR_OPINION = 5
MAX_CASE_LAWS_PER_DISPUTE = 5

# ---------------------------------------------------------------------------
# Explanation Generation
# ---------------------------------------------------------------------------

def _build_dispute_blocks_text(dispute_results: list) -> str:
    """
    Build dispute-wise text for STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.
    For each dispute: take top MAX_SECTIONS_PER_DISPUTE_FOR_OPINION bare sections,
    link case laws to sections by section number, then format as a block.
    """
    blocks = []
    for dr in dispute_results:
        dispute = dr.get("dispute", {})
        d_id = dispute.get("id", "?")
        d_text = (dispute.get("dispute") or "").strip()
        bare_d = dr.get("bare_acts") or []
        case_d = dr.get("case_laws") or []

        top_bare = sorted(
            bare_d,
            key=lambda x: x.get("_rerank_score", 0),
            reverse=True,
        )[:MAX_SECTIONS_PER_DISPUTE_FOR_OPINION]
        if not top_bare:
            blocks.append(f"### Component [{d_id}]: {d_text[:200]}\n(No bare act sections retrieved for this component.)")
            continue

        formatted_bare = _format_bare_acts(top_bare)
        linked = _link_case_laws_to_sections(formatted_bare, case_d[:MAX_CASE_LAWS_PER_DISPUTE])

        lines = [f"### Component [{d_id}]: {d_text[:200]}"]
        for ba in linked:
            act = ba.get("act_name", "")
            sec = ba.get("section_number", "")
            title = ba.get("title", ba.get("section_title", ""))
            verbatim_text = (ba.get("text") or ba.get("full_text") or "").strip()
            # Provide enough for verbatim quoting (up to 600 chars per section)
            verbatim_snippet = verbatim_text[:600] if verbatim_text else ""
            lines.append(f"\n**{act}, Section {sec}** — {title}")
            lines.append(f"Verbatim section text (quote this in the opinion):\n{verbatim_snippet}")
            related = ba.get("related_case_laws") or []
            if related:
                lines.append("Case laws under this section (explain what parts apply and how):")
                for cl in related[:3]:
                    name = _format_case_citation(cl)
                    snippet = (cl.get("text") or cl.get("full_text") or "")[:300]
                    lines.append(f"- {name}: {snippet}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks) if blocks else "No dispute components with retrieved materials."


def _generate_structured_opinion_by_dispute(
    facts_summary: str,
    dispute_results: list,
    additional_info: str = "",
    token_callback=None,
) -> str:
    """
    Generate structured legal opinion: Dispute Summary → Section + why it applies + precedents (per component) → Legal Position.
    Uses STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT. Strictly grounded in retrieved materials only.

    If token_callback is provided, streams tokens to it and returns the collected text.
    """
    if not dispute_results:
        return _local_only_no_materials_message()

    dispute_blocks_text = _build_dispute_blocks_text(dispute_results)
    if dispute_blocks_text.strip() == "No dispute components with retrieved materials.":
        return _local_only_no_materials_message()

    from prompts.advocate_prompts import STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT

    prompt = STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.format(
        dispute_facts=facts_summary[:1200],
        additional_info=(additional_info or "None provided").strip(),
        dispute_blocks_text=dispute_blocks_text,
    )
    try:
        if token_callback:
            full_text = ""
            for token in ask_llm_stream(prompt):
                token_callback(token)
                full_text += token
            local_bare = []
            for dr in dispute_results:
                local_bare.extend(dr.get("bare_acts") or [])
            return _enforce_local_grounding(full_text.strip(), local_bare)
        local_bare = []
        for dr in dispute_results:
            local_bare.extend(dr.get("bare_acts") or [])
        return _enforce_local_grounding(ask_llm(prompt).strip(), local_bare)
    except Exception as e:
        logger.error("Structured opinion by dispute failed: %s", e)
        return ""


def _generate_legal_opinion(
    facts: str,
    bare_acts: list,
    case_laws: list,
    sufficiency: dict,
    on_before_llm=None,
) -> str:
    """Generate a formal legal opinion with citations and source tags. on_before_llm(prompt) is called before LLM if provided."""
    # Check if we have any materials at all
    bare_acts, case_laws = _filter_materials_to_local_db(bare_acts, case_laws)
    bare_acts, case_laws = _filter_materials_to_local_db(bare_acts, case_laws)
    bare_acts, case_laws = _filter_materials_to_local_db(bare_acts, case_laws)
    has_bare_acts = bool(bare_acts and len(bare_acts) > 0)
    has_case_laws = bool(case_laws and len(case_laws) > 0)
    
    # If no materials at all, return fixed "I don't have any data" — do NOT call LLM (no hallucination risk)
    if not has_bare_acts and not has_case_laws:
        return f"""## Brief Facts
{facts[:200]}

## Analysis and Conclusion
{_local_only_no_materials_message()}"""

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
        return _enforce_local_grounding(ask_llm(prompt).strip(), bare_acts)
    except Exception as e:
        logger.error(f"Opinion generation failed: {e}")
        return ""


def _generate_conversational_summary(
    facts: str,
    bare_acts: list,
    case_laws: list,
    intent: str = "search",
    on_before_llm=None,
    token_callback=None,
) -> str:
    """Generate a conversational summary for search/lookup. For lookup use bare-act-only summary; for search use dispute+order/judgement in brief.
    When no materials exist for the requested type, return fixed 'I don't have any data' — do NOT call LLM (anti-hallucination). on_before_llm(prompt) is called before LLM if provided."""
    has_bare_acts = bool(bare_acts and len(bare_acts) > 0)
    has_case_laws = bool(case_laws and len(case_laws) > 0)

    # No-LLM path: only return fixed "no materials" when NOTHING at all was found.
    # Previously: "search" → return no-materials if no case laws (even with bare acts).
    #             "lookup" → return no-materials if no bare acts (even with case laws).
    # This caused false failures when the corpus has acts but no case laws yet (or vice versa).
    # Fix: only return no-materials when the specific requested type is empty AND nothing
    # else is available either (truly empty response).
    if not has_bare_acts and not has_case_laws:
        return _local_only_no_materials_message()

    # For lookup: bare-acts are primary. For search: case laws are primary.
    # If the primary type is missing but the secondary is present, fall through
    # to generate a summary of whatever was found (intent treated as generic).
    if intent == "lookup" and not has_bare_acts:
        return _local_only_no_materials_message()
    if intent == "search" and not has_case_laws and not has_bare_acts:
        return _local_only_no_materials_message()

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

    # --- Anti-hallucination: build explicit citation allowlists ---
    section_allowlist = []
    for ba in bare_acts[:20]:
        sec = str(ba.get("section_number") or "").strip()
        act = (ba.get("act_name") or "").strip()
        if sec:
            section_allowlist.append(f"{act} § {sec}" if act else f"§ {sec}")
    case_allowlist = []
    for cl in case_laws[:10]:
        name = (cl.get("case_name") or cl.get("title") or "").strip()
        if name:
            case_allowlist.append(name)

    if section_allowlist or case_allowlist:
        _sec_list = section_allowlist if section_allowlist else ["NONE"]
        _case_list = case_allowlist if case_allowlist else ["NONE"]
        allowlist_block = (
            f"\n\n🚨 STRICT CITATION RULES — NO EXCEPTIONS:\n"
            f"1. You may ONLY reference these sections (verbatim): {_sec_list}\n"
            f"2. You may ONLY reference these cases: {_case_list}\n"
            f"3. Do NOT add any IPC / CrPC / old-act section numbers or case names "
            f"from your training knowledge that are NOT in the above lists.\n"
            f"4. If a section or case is not in the above lists, do NOT mention it."
        )
    else:
        allowlist_block = (
            "\n\n🚨 STRICT CITATION RULES: No sections or cases were retrieved. "
            "Do NOT cite any section numbers, act names, or case names. "
            "Only describe the legal topic without citing specific provisions."
        )

    if intent == "lookup" or (intent == "search" and has_bare_acts and not has_case_laws):
        # lookup: summarise bare acts (primary).
        # search with bare acts but no case laws: fall back to bare-act summary so
        # the user still sees the retrieved sections instead of a blank "no results".
        system = BARE_ACT_ONLY_SUMMARY
        prompt = f"""{system}

USER QUERY: {facts[:500]}
BARE ACT SECTIONS FOUND: {bare_text if has_bare_acts else "[]"}
{allowlist_block}

Summarise only the relevant bare act sections (no case laws):"""
    elif intent == "search":
        system = CASE_LAW_DISPUTE_ORDER_SUMMARY
        prompt = f"""{system}

USER QUERY: {facts[:500]}
CASE LAWS FOUND: {case_text if has_case_laws else "[]"}
{allowlist_block}

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
{allowlist_block}

Response:"""

    try:
        if callable(on_before_llm):
            on_before_llm(prompt)
        if token_callback:
            full_text = ""
            for token in ask_llm_stream(prompt):
                token_callback(token)
                full_text += token
            return _enforce_local_grounding(full_text.strip(), bare_acts)
        return _enforce_local_grounding(ask_llm(prompt).strip(), bare_acts)
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
    if search_strategy == "local_only":
        return (
            "I searched the local vector store only and found no relevant bare act provisions or case laws. "
            "Try rephrasing with specific section numbers, Act names, or a different legal angle."
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
    """Add user-confirmed materials to the in-memory response lists (indexing removed)."""
    for b in confirmed.get("bare_acts", []):
        text = b.get("text", b.get("content", ""))
        title = b.get("title", "Confirmed")
        if text:
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
            case_laws.append({
                "case_name": title,
                "text": text,
                "source": title,
                "source_tag": "LOCAL_DB",
            })

    return bare_acts, case_laws
