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
    STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT,
)
from services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# Relevance and quality thresholds — keep only high-quality, recent materials
MIN_RERANK_SCORE = 0.0  # Minimum score to include in pool. ms-marco cross-encoder logits:
                        #   >2.0 = clearly relevant; 0-2 = borderline; <0 = low relevance.
                        # 0.0 lets BNSS/TPA/BSA sections (typical scores 0.1–3.0 for plain-
                        # language queries) pass the quality filter. The per-dispute hard cap
                        # (MAX_SECTIONS_PER_DISPUTE_TOTAL) then keeps only the top N by score.
                        # Previously 3.0 → then 1.0 → both silently discarded local BNSS sections.
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

# Section caps — keep retrieval focused: goal is 1–2 really on-point sections per dispute.
# Cross-encoder (local) + doc-level scoring (web) pick the BEST N by score; caps prevent 100+ sections per act.
MAX_WEB_SECTIONS_PER_DISPUTE = 3    # Top N chunks kept from web search per dispute (by doc/section score)
MAX_SECTIONS_PER_DISPUTE_TOTAL = 3  # Hard cap per dispute after local+web+xref merge (top N by score)
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
# Scores of 5-7 mean we found the right act but coverage may be incomplete.
# Since our local index is still growing, 8.0 ensures we supplement with web
# whenever the local store only has partial or procedure-level coverage.
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

    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str):
        q = q.strip()[:400]
        norm = q.lower()
        if q and norm not in seen:
            seen.add(norm)
            queries.append(q)

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

    return queries


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


def _web_search_bare_acts(dispute: dict, full_query: str, round1_queries: list = None, states: list = None) -> list:
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
        gap_results = search_for_gaps(web_gaps, jurisdiction_state="")
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

    # Run each fetched document through chunk_bare_act() to extract section-level chunks
    # with proper act_name / section_number / section_title metadata.
    # Previously we returned flat document dicts with section_number="" which caused
    # _is_quality_bare_act() to silently drop all web results.
    from Ingestion.smart_chunker import chunk_bare_act

    results = []
    for enriched in enrichment.get("enriched_bare_acts", []):
        content = enriched.get("content", "").strip()
        if not content or len(content) < 100:
            continue

        doc_title = enriched.get("title", "")
        doc_url   = enriched.get("url", "")
        doc_tag   = enriched.get("source_tag", "WEB")
        doc_score = float(enriched.get("_rerank_score", MIN_RERANK_SCORE))

        # Attempt section-level chunking
        try:
            chunks = chunk_bare_act(content, doc_title)
        except Exception as _ce:
            logger.debug("chunk_bare_act failed for '%s': %s", doc_title[:40], _ce)
            chunks = []

        if not chunks:
            # Chunker found no section patterns — treat whole doc as one synthetic chunk
            # so at least the act-level content is surfaced.
            act_guess = doc_title or "Unknown Act"
            chunks = [{
                "act_name": act_guess,
                "section_number": "1",       # synthetic — lets quality filter pass
                "section_title": "",
                "full_text": content[:2000],
                "search_text": content[:500],
                "source": doc_title,
            }]

        for chunk in chunks:
            if not _is_quality_bare_act(chunk):
                continue
            # Assign URL, source tag, and propagate doc-level score so section caps keep the best-matching Acts.
            # If chunk already has a more specific score, keep it; otherwise inherit from parent document.
            if "_rerank_score" not in chunk:
                chunk["_rerank_score"] = doc_score
            chunk["source_tag"]   = doc_tag
            chunk["url"]          = doc_url
            chunk["_web_sourced"] = True   # signals UI to show "Add to local index" option
            results.append(chunk)

    # Keep only top N sections by score so we don't flood the UI with every section from every act.
    results = _cap_bare_acts_by_score(results, MAX_WEB_SECTIONS_PER_DISPUTE)
    logger.info(
        "Web bare acts for dispute '%s': %d section chunks from %d docs (%d queries)",
        dispute_text[:60], len(results), len(enrichment.get("enriched_bare_acts", [])), len(web_gaps),
    )
    return results


def retrieve_bare_acts_for_dispute(dispute: dict, full_query: str, states: list = None) -> list:
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
    from retrieval.act_profile_index import identify_relevant_acts

    dispute_text = dispute.get("dispute", "")
    dispute_id = dispute.get("id", "?")

    # --- Act-first: identify relevant acts before any section-level queries ---
    # Use ONLY dispute_text (e.g. "Neighbor attacked causing severe leg injury"),
    # NOT full_query (facts_summary).  dispute_text is already a concise one-line
    # legal label produced by the dispute decomposer — it's the ideal BM25 query.
    # full_query (facts summary) adds 2-3 sentences of case facts that introduce
    # noise tokens ("yesterday", "police complaint", "original documents") that
    # dilute the legal signal and cause wrong act matches.
    # The profile BM25 is fast (no GPU, < 50ms), defaults: top_k=3, min_score=0.15.
    act_query = dispute_text.strip()
    allowed_acts = identify_relevant_acts(act_query)
    if allowed_acts:
        logger.info(
            "[%s] Act-first profile match: %d acts identified → %s",
            dispute_id, len(allowed_acts), sorted(allowed_acts),
        )
    else:
        logger.info(
            "[%s] Act-first: profile index unavailable or no match — falling back to unfiltered search",
            dispute_id,
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

    # Score-based web search gate:
    # If no local section clears WEB_SEARCH_FALLBACK_MIN_SCORE (8.0), the index doesn't
    # have a confident match — skip all auto-stop tiers and go straight to web search.
    # Scores 5-7 mean we found the right act category but likely only procedure/definition
    # sections, not the exact substantive provision (e.g. BNSS procedure sections scoring 6
    # for an assault query, while BNS substantive section is missing entirely).
    max_local_score = max((ba.get("_rerank_score", 0) for ba in local_results), default=0.0)
    force_web = (max_local_score < WEB_SEARCH_FALLBACK_MIN_SCORE)
    if force_web:
        logger.info(
            "[%s] Best local score %.2f < %.1f threshold — forcing web search regardless of section count.",
            dispute_id, max_local_score, WEB_SEARCH_FALLBACK_MIN_SCORE,
        )
    else:
        if len(high_quality_local) >= _AUTO_STOP_COUNT:
            logger.info(
                "[%s] Bare acts Round 1 auto-sufficient (%d HQ sections ≥ %d, score>%.1f). "
                "Skipping LLM sufficiency call and web search.",
                dispute_id, len(high_quality_local), _AUTO_STOP_COUNT, BARE_ACT_HIGH_QUALITY_SCORE,
            )
            return _cap_bare_acts_by_score(local_results, MAX_SECTIONS_PER_DISPUTE_TOTAL)

        if len(local_results) >= _AUTO_STOP_VOLUME:
            logger.info(
                "[%s] Bare acts Round 1 auto-sufficient by volume (%d sections ≥ %d). "
                "Skipping LLM sufficiency call and web search.",
                dispute_id, len(local_results), _AUTO_STOP_VOLUME,
            )
            return _cap_bare_acts_by_score(local_results, MAX_SECTIONS_PER_DISPUTE_TOTAL)

        if high_quality_local and local_results and _check_bare_act_sufficiency(dispute_text, local_results):
            logger.info(
                "[%s] Bare acts Round 1 sufficient (%d sections, %d high-quality). Stopping.",
                dispute_id, len(local_results), len(high_quality_local),
            )
            return _cap_bare_acts_by_score(local_results, MAX_SECTIONS_PER_DISPUTE_TOTAL)

    # --- Round 2: web search ---
    logger.info("[%s] Bare acts Round 2 — web search (local had %d, force_web=%s)", dispute_id, len(local_results), force_web)
    # Pass `queries` so the web search can reuse the focused Round 1 queries
    # (including LLM-generated ones) instead of falling back to raw dispute text.
    # Pass `states` so web search covers both central/Union acts AND state-specific acts.
    web_results = _web_search_bare_acts(dispute, full_query, round1_queries=queries, states=states or [])

    merged = _merge_deduplicate_bare_acts(local_results, web_results)
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
# BARE ACTS PHASE — present sections, explain, ask follow-up, then final opinion
# ---------------------------------------------------------------------------

def _explain_sections_and_get_followup(dispute_text: str, bare_acts: list) -> dict:
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

    prompt = BARE_ACT_EXPLAIN_AND_FOLLOWUP_PROMPT.format(
        dispute_facts=dispute_text[:800],
        bare_acts_list="\n".join(lines),
    )

    try:
        raw = ask_llm(prompt)
        # Strip any preamble before the first '{'
        raw = raw[raw.find("{"):].strip() if "{" in raw else raw
        parsed = json.loads(raw)
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

    followup = (parsed.get("followup_question") or "").strip() or None
    if followup and len(followup) < 10:
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


def retrieve_bare_acts_phase(facts_summary: str, progress_callback=None, states: list = None) -> dict:
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

    if progress_callback:
        progress_callback({"step": "bare_acts", "message": "Retrieving relevant bare act sections…"})

    try:
        disputes = decompose_disputes(facts_summary)
    except Exception as e:
        logger.warning("retrieve_bare_acts_phase: decompose_disputes failed (%s). Using single dispute.", e)
        disputes = [{"id": "d1", "dispute": facts_summary[:300], "legal_nature": "both",
                     "keywords": [], "bare_act_hints": [], "search_angles": []}]

    # Retrieve bare acts per dispute in parallel; tag each section with its dispute origin
    # so Phase B can reconstruct per-dispute groupings without re-running decomposition.
    all_bare_acts: list = []

    _states = states or []

    def _fetch_for_dispute(d):
        sections = retrieve_bare_acts_for_dispute(d, facts_summary, states=_states)
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
                logger.warning("retrieve_bare_acts_phase: dispute retrieval error: %s", exc)

    # Deduplicate across disputes (first-seen dispute tag is preserved via setdefault above)
    all_bare_acts = _merge_deduplicate_bare_acts(all_bare_acts, [])
    # Safety cap: max 15 total across all disputes (5 per dispute × 3 disputes typical)
    all_bare_acts = _cap_bare_acts_by_score(all_bare_acts, MAX_BARE_ACTS_OVERALL)

    if progress_callback:
        progress_callback({"step": "bare_acts_explain",
                           "message": f"Explaining {len(all_bare_acts)} section(s)…"})

    # Explain sections + generate follow-up
    enriched = _explain_sections_and_get_followup(facts_summary, all_bare_acts)

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

    # Build a friendly intro
    n = len(flat_bare_acts)
    nd = len(disputes_grouped)
    if n == 0:
        intro = (
            "I searched the legal database but couldn't find specific sections for your query. "
            "I'll proceed with a general analysis."
        )
    else:
        dispute_word = "dispute" if nd == 1 else "disputes"
        section_word = "section" if n == 1 else "sections"
        intro = (
            f"I found {n} relevant {section_word} across {nd} {dispute_word}. "
            "Here's what each section says and how it applies to your situation."
        )

    return {
        "disputes": disputes_grouped,        # new: grouped by dispute for UI rendering
        "bare_acts": flat_bare_acts,          # kept: flat list used by Phase B (case laws + opinion)
        "followup_question": enriched["followup_question"],
        "intro_text": intro,
    }


def generate_final_opinion_with_case_laws(
    facts_summary: str,
    bare_acts: list,
    additional_info: str = "",
    progress_callback=None,
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
    def _fetch_case_laws_for_group(grp):
        cls = retrieve_case_laws_for_dispute(grp, grp["sections"], full_facts)
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

    # --- Link case laws to sections within each dispute group ---
    all_bare_acts_with_cases: list = []
    for did, grp in dispute_groups.items():
        grp_case_laws = case_laws_by_dispute.get(did, [])
        grp["sections_with_cases"] = _link_case_laws_to_sections(grp["sections"], grp_case_laws)
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
                    name = cl.get("case_name") or cl.get("title") or "Unknown"
                    body = (cl.get("text") or cl.get("full_text") or "")[:250]
                    lines.append(f"  - [{name}]: {body}")
            lines.append("")
        dispute_blocks.append("\n".join(lines))

    dispute_blocks_text = "\n".join(dispute_blocks) or "None retrieved."

    # --- Build strict citation allowlist ---
    _final_sec_allowlist = [
        f"{ba.get('act_name', '')} § {ba.get('section_number', '')}"
        for ba in bare_acts[:20] if ba.get("section_number")
    ]
    _final_case_allowlist = [
        (cl.get("case_name") or cl.get("title") or "").strip()
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

    try:
        opinion_text = ask_llm(opinion_prompt).strip()
    except Exception as e:
        logger.error("generate_final_opinion_with_case_laws: opinion LLM failed: %s", e)
        opinion_text = "I was unable to generate a structured opinion at this time. Please try again."

    # Collect web case laws for indexing proposals
    web_case_laws = []
    for ba in all_bare_acts_with_cases:
        for cl in ba.get("related_case_laws", []):
            if cl.get("url") or cl.get("source_url"):
                web_case_laws.append(cl)
    indexing_candidates = _build_indexing_candidates_from_web_case_laws(web_case_laws)

    return {
        "bare_act_sections": all_bare_acts_with_cases,
        "case_laws": [],           # Nested under bare acts — no top-level duplicates
        "internet_case_laws": [],
        "explanation": opinion_text,
        "progress": None,
        "indexing_candidates": indexing_candidates,
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
        # Legal opinion: structured by dispute (Dispute Summary → Section + why + precedents per component → Legal Position)
        explanation = _generate_structured_opinion_by_dispute(
            facts_summary, dispute_results, additional_info=""
        )
        if not (explanation or "").strip():
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

    # Collect web case laws for indexing proposals (from raw case laws before formatting)
    indexing_candidates = _build_indexing_candidates_from_web_case_laws(all_case_raw)

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
        "indexing_candidates": indexing_candidates,
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
                    name = cl.get("case_name") or cl.get("title") or "Unknown"
                    snippet = (cl.get("text") or cl.get("full_text") or "")[:300]
                    lines.append(f"- {name}: {snippet}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks) if blocks else "No dispute components with retrieved materials."


def _generate_structured_opinion_by_dispute(
    facts_summary: str,
    dispute_results: list,
    additional_info: str = "",
) -> str:
    """
    Generate structured legal opinion: Dispute Summary → Section + why it applies + precedents (per component) → Legal Position.
    Uses STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT. Strictly grounded in retrieved materials only.
    """
    if not dispute_results:
        return RELEVANCE_EXPLANATION_NO_MATERIALS

    dispute_blocks_text = _build_dispute_blocks_text(dispute_results)
    if dispute_blocks_text.strip() == "No dispute components with retrieved materials.":
        return RELEVANCE_EXPLANATION_NO_MATERIALS

    from prompts.advocate_prompts import STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT

    prompt = STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.format(
        dispute_facts=facts_summary[:1200],
        additional_info=(additional_info or "None provided").strip(),
        dispute_blocks_text=dispute_blocks_text,
    )
    try:
        return ask_llm(prompt).strip()
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

    # No-LLM path: only return fixed "no materials" when NOTHING at all was found.
    # Previously: "search" → return no-materials if no case laws (even with bare acts).
    #             "lookup" → return no-materials if no bare acts (even with case laws).
    # This caused false failures when the corpus has acts but no case laws yet (or vice versa).
    # Fix: only return no-materials when the specific requested type is empty AND nothing
    # else is available either (truly empty response).
    if not has_bare_acts and not has_case_laws:
        return RELEVANCE_EXPLANATION_NO_MATERIALS

    # For lookup: bare-acts are primary. For search: case laws are primary.
    # If the primary type is missing but the secondary is present, fall through
    # to generate a summary of whatever was found (intent treated as generic).
    if intent == "lookup" and not has_bare_acts:
        return RELEVANCE_EXPLANATION_NO_MATERIALS
    if intent == "search" and not has_case_laws and not has_bare_acts:
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
