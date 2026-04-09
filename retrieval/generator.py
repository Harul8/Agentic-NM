"""
pipeline/generator.py — Main response generation pipeline.
"""
import os
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import (
    BARE_ACTS_DIR,
    CASELAW_DIR,
    GOOGLE_DRIVE_BARE_ACTS_FOLDER_URL,
    GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL,
)
from platform.llm import (
    ask_llm,
    ask_llm_stream,
    get_model_display_for_prompt,
    get_gpu_info,
    set_request_model_override,
)
from prompts.research import (
    EXPAND_LEGAL_QUERY_SYSTEM,
    CASE_SUMMARY_SYSTEM,
    RELEVANCE_EXPLANATION_SYSTEM,
    RELEVANCE_EXPLANATION_NO_MATERIALS,
    CONVERSATIONAL_SUMMARY_SYSTEM,
    BARE_ACT_ONLY_SUMMARY,
    CASE_LAW_DISPUTE_ORDER_SUMMARY,
    STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT,
    FAST_INTERACTIVE_OPINION_PROMPT,
    BARE_ACT_STAGE_SUMMARY_PROMPT,
    BARE_ACT_NEXT_STEPS_REPAIR_PROMPT,
    PRECEDENT_STAGE_SUMMARY_PROMPT,
    ACT_SELECTION_PROMPT,
    BARE_ACT_SECTION_RELEVANCE_PROMPT,
    CASE_LAW_RELEVANCE_PROMPT,
)
from platform.progress import ProgressTracker

logger = logging.getLogger(__name__)
_ENABLE_CITATION_GRAPH_EXPANSION = os.environ.get("ENABLE_CITATION_GRAPH_EXPANSION", "1").lower() in ("1", "true", "yes")
_ENABLE_CASE_SUMMARY_LLM = os.environ.get("ENABLE_CASE_SUMMARY_LLM", "").lower() in ("1", "true", "yes")
_ENABLE_BARE_ACT_LLM_FILTER = os.environ.get("ENABLE_BARE_ACT_LLM_FILTER", "1").lower() in ("1", "true", "yes")
_ENABLE_CASE_LAW_LLM_FILTER = os.environ.get("ENABLE_CASE_LAW_LLM_FILTER", "1").lower() in ("1", "true", "yes")
_ENABLE_RUNTIME_FEWSHOT = os.environ.get("ENABLE_RUNTIME_FEWSHOT", "1").lower() in ("1", "true", "yes")
_ENABLE_INTERACTIVE_FAST_PATH = os.environ.get("ENABLE_INTERACTIVE_FAST_PATH", "1").lower() in ("1", "true", "yes")
_ENABLE_INTERACTIVE_FAST_LLM = os.environ.get("ENABLE_INTERACTIVE_FAST_LLM", "0").lower() in ("1", "true", "yes")
_FAST_BARE_QUERY_LIMIT = 3
_FAST_CASE_QUERY_LIMIT = 2
_FAST_BARE_TOP_K = 5
_FAST_CASE_TOP_K = 4
INTERACTIVE_MIN_RERANK_SCORE = 0.15

# Relevance and quality thresholds — keep only high-quality, recent materials
MIN_RERANK_SCORE = 0.5  # Minimum score to include in pool. ms-marco cross-encoder logits:
                        #   >2.0 = clearly relevant; 0.5-2 = borderline; <0.5 = noise / off-topic.
                        # 0.5 formally excludes near-zero scoring sections while keeping
                        # BNSS/BNS/BSA sections (typical scores 0.5–3.0 for plain-language queries).
                        # The per-dispute hard cap (MAX_SECTIONS_PER_DISPUTE_TOTAL) then keeps
                        # only the top N by score. Raised from 0.0 (allowed all noise) to 0.5.
HIGH_QUALITY_SCORE = 5.0  # Case laws with score > this: "highly relevant"
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
MAX_SECTIONS_PER_DISPUTE_TOTAL = 9  # Hard cap per dispute after local+web+xref merge (top N by score)
MAX_BARE_ACTS_OVERALL = 12          # Safety cap on total sections sent to UI across all disputes
MAX_SECTIONS_PER_DISPUTE_FOR_OPINION = 3
MAX_CASE_LAWS_PER_DISPUTE = 3

# Retrieved text budget — how many characters of a chunk to send to the LLM.
# Local Llama (8K–16K context) needed aggressive truncation; API models have 200K+
# context windows and must receive full chunks to reason correctly over legal text.
# Indian bare act sections and case law chunks are stored at ≤2000 chars each.
# Set to a large value so full chunks always pass through; override at call site
# only if a specific prompt has its own hard size limit.
_MAX_SECTION_TEXT = 2000    # chars per bare act section
_MAX_CASE_TEXT    = 2000    # chars per case law chunk
_MAX_JSON_PAYLOAD = 60_000  # chars for JSON-serialised payload (replaces old 2000/3000 caps)

# Act-level pre-filter (for 100+ act indexes)
# An act qualifies if its highest-scoring section clears this threshold.
# ms-marco cross-encoder logits: 5.0 captures the "correct act, right domain" range
# while cleanly dropping wrong-domain acts (Constitution, POCSO, Arms Act for an
# assault query typically score 0–2, well below this bar).
ACT_SELECTION_MIN_SCORE = 5.0
ACT_SELECTION_MAX_ACTS  = 4   # Never take more than 4 even if many qualify

WEB_SEARCH_MIN_LOCAL_COUNT = 3  # legacy constant retained for compatibility
WEB_SEARCH_FALLBACK_MIN_SCORE = 8.0

# ---------------------------------------------------------------------------
# New criminal codes (2023) — keyword sets for automatic query injection
# ---------------------------------------------------------------------------
# IndiaCode generic searches like "assault India central act" return the Arms Act
# because it is a well-established, heavily-indexed act whose title contains "arms"
# and whose provisions mention "assault / violence".  BNS 2023 is a much newer
# addition and only surfaces when queried by its *explicit name*.
# Criminal code injection (BNS/BNSS/BSA) is now driven by the dispute's legal_nature
# field (set from agents.intake state) rather than keyword matching — see _inject_criminal_code_queries.


def _is_quality_bare_act(section: dict) -> bool:
    """
    Lightweight bare-act quality guard for the final analysis lane.

    Keep sections that have a plausible act name, section identifier, and usable text.
    This avoids obvious junk while letting the reranker score decide relevance.
    """
    if not isinstance(section, dict):
        return False
    act_name = (section.get("act_name") or "").strip()
    sec_num = str(section.get("section_number") or "").strip()
    text = (
        section.get("text")
        or section.get("full_text")
        or section.get("search_text")
        or ""
    ).strip()
    if not act_name or len(act_name) < 4:
        return False
    if not sec_num:
        return False
    if not text or len(text) < 25:
        return False
    junk_markers = ("unknown", "not available", "no text", "n/a")
    low = f"{act_name} {text[:120]}".lower()
    if any(marker in low for marker in junk_markers):
        return False
    return True


def _is_quality_case_law(case_law: dict) -> bool:
    """
    Lightweight case-law quality guard for the final analysis lane.

    Keep final-judgment-style results with a real case/source name and usable text.
    Strong ranking and court-ordering still happen elsewhere.
    """
    if not isinstance(case_law, dict):
        return False
    case_name = (
        case_law.get("case_name")
        or case_law.get("title")
        or case_law.get("source")
        or ""
    ).strip()
    text = (
        case_law.get("text")
        or case_law.get("full_text")
        or case_law.get("search_text")
        or ""
    ).strip()
    if not case_name or len(case_name) < 6:
        return False
    if not text or len(text) < 60:
        return False
    low_name = case_name.lower()
    if any(p in low_name for p in JUNK_CASE_PATTERNS):
        return False
    return True


def _cap_bare_acts_by_score(bare_acts: list, cap: int) -> list:
    """Return top `cap` bare acts sorted by _rerank_score descending."""
    if not bare_acts or cap <= 0:
        return bare_acts
    return sorted(bare_acts, key=lambda x: x.get("_rerank_score", 0), reverse=True)[:cap]


_RE_SECTION_CITATION = re.compile(r"\bsections?\s+([0-9A-Za-z,\sand]+)", re.IGNORECASE)


def _local_only_no_materials_message() -> str:
    """Strict fallback when an answer cannot be grounded in the local vector store."""
    return (
        "I don't have grounded data for this query in the local vector store. "
        "I can only answer legal questions from locally retrieved bare act provisions and case laws. "
        "Please index the relevant materials or rephrase the query."
    )


def _strip_chat_window_summary(text: str) -> str:
    """Remove appended chat-window scaffolding before retrieval or final prose."""
    value = (text or "").strip()
    marker = "CHAT WINDOW SUMMARY"
    if marker in value:
        value = value.split(marker, 1)[0].strip()
    return value


def _normalize_fact_text(text: str) -> str:
    """Collapse repeated whitespace while preserving the user's actual facts."""
    return " ".join(_strip_chat_window_summary(text).split()).strip()


def _split_fact_segments(text: str) -> list[str]:
    """Break a fact summary into compact, de-duplicated segments."""
    normalized = _normalize_fact_text(text)
    if not normalized:
        return []
    raw_segments = re.split(r'(?<=[.!?])\s+|\s*\n+\s*', normalized)
    segments: list[str] = []
    seen: set[str] = set()
    for raw in raw_segments:
        seg = " ".join(raw.split()).strip(" -")
        key = seg.lower()
        if len(seg) < 8 or key in seen:
            continue
        seen.add(key)
        segments.append(seg)
    return segments


def _build_focus_fact_text(text: str, max_chars: int = 420) -> str:
    """
    Preserve both the opening fact pattern and the latest material fact.
    This keeps late-turn updates from being clipped out of fast retrieval.
    """
    segments = _split_fact_segments(text)
    if not segments:
        return ""
    selected: list[str] = []
    used: set[str] = set()

    def _add(segment: str):
        key = segment.lower()
        if segment and key not in used:
            used.add(key)
            selected.append(segment)

    for seg in segments[:2]:
        _add(seg)
    for seg in segments[-2:]:
        _add(seg)

    merged = " ".join(selected).strip()
    if len(merged) <= max_chars:
        return merged

    if len(selected) >= 2:
        budget = max(40, (max_chars - 5) // 2)
        head = selected[0][:budget].rsplit(" ", 1)[0].strip(" .,;:")
        tail = selected[-1][:budget].rsplit(" ", 1)[0].strip(" .,;:")
        compact = f"{head} ... {tail}".strip()
        return compact[:max_chars].strip()

    return merged[:max_chars].rsplit(" ", 1)[0].strip()


_FAST_ALIGNMENT_STOPWORDS = frozenset({
    "about", "above", "after", "against", "already", "also", "although", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "done", "during", "each", "for", "from", "further", "get", "got", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "need", "no", "not", "now", "of",
    "on", "or", "our", "out", "over", "please", "right", "same", "she", "should", "since", "so", "some",
    "still", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "want", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "without", "would", "you", "your",
    "legal", "opinion", "case", "law", "facts", "fact", "issue", "matter", "shared",
})


def _extract_salient_terms(text: str, limit: int = 8) -> list[str]:
    """Simple lexical term extractor for fast-path fact alignment."""
    tokens = re.findall(r"[a-z][a-z0-9_-]{3,}", (text or "").lower())
    ranked: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _FAST_ALIGNMENT_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        ranked.append(tok)
        if len(ranked) >= limit:
            break
    return ranked


def _material_family_key(item: dict) -> str:
    raw = " ".join(
        str(item.get(k) or "")
        for k in ("act_name", "title", "case_name")
    ).lower()
    if not raw:
        return ""
    raw = re.sub(r"\d{4}", " ", raw)
    tokens = [
        tok for tok in re.findall(r"[a-z][a-z0-9_-]{3,}", raw)
        if tok not in _FAST_ALIGNMENT_STOPWORDS and tok not in {"section", "sections", "act", "code", "rule", "rules", "law", "laws", "judgment", "judgement", "court"}
    ]
    return " ".join(tokens[:4]).strip()


def _short_dispute_label(text: str, max_words: int = 8) -> str:
    cleaned = " ".join((text or "").split()).strip(" .:;,-")
    if not cleaned:
        return "Dispute"
    first = re.split(r"(?<=[.!?])\s+|;|:", cleaned, maxsplit=1)[0].strip(" .:;,-")
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]).strip(" .:;,-") + "?"
    return first or "Dispute"


def _best_alignment_score(item: dict) -> float:
    if not isinstance(item, dict):
        return 0.0
    candidates = (
        "_case_aligned_score",
        "_family_aligned_score",
        "_fact_alignment_score",
        "_case_support_score",
        "_rerank_score",
    )
    best = 0.0
    for field in candidates:
        try:
            best = max(best, float(item.get(field, 0) or 0))
        except Exception:
            continue
    return best


def _apply_family_consensus_sort(items: list, base_field: str, cap: int | None = None) -> list:
    if not items:
        return []

    family_totals: dict[str, float] = {}
    for idx, item in enumerate(items):
        family = _material_family_key(item)
        if not family:
            continue
        weight = float(item.get(base_field, item.get("_rerank_score", 0)) or 0)
        family_totals[family] = family_totals.get(family, 0.0) + weight + max(0.0, 0.18 - (idx * 0.02))

    rescored: list[dict] = []
    for item in items:
        clone = dict(item)
        family = _material_family_key(item)
        clone["_material_family"] = family
        consensus = family_totals.get(family, 0.0) if family else 0.0
        clone["_family_consensus_score"] = consensus
        clone["_family_aligned_score"] = float(clone.get(base_field, clone.get("_rerank_score", 0)) or 0) + min(consensus * 0.12, 1.35)
        rescored.append(clone)

    rescored.sort(
        key=lambda x: (
            x.get("_family_aligned_score", 0),
            x.get(base_field, 0),
            x.get("_rerank_score", 0),
        ),
        reverse=True,
    )
    return rescored[:cap] if cap else rescored


def _apply_fact_alignment_sort(items: list, _facts_summary: str, cap: int | None = None) -> list:
    """
    Sort retrieved items by cross-encoder score then apply family-consensus grouping.
    Lexical overlap arithmetic removed — the cross-encoder already captures relevance.
    """
    if not items:
        return []
    for item in items:
        if "_fact_alignment_score" not in item:
            item["_fact_alignment_score"] = float(item.get("_rerank_score", 0) or 0)
    ranked = _apply_family_consensus_sort(items, "_fact_alignment_score", cap=None)
    return ranked[:cap] if cap else ranked


# ---------------------------------------------------------------------------
# Intake-state-driven dispute building — replaces the decompose_disputes LLM call.
# The intake conversation already identified primary + secondary issue clusters;
# we use those to build the dispute list with category-seeded search angles.
# ---------------------------------------------------------------------------

_CATEGORY_SEEDS: dict[str, list[str]] = {
    "domestic_violence": [
        "domestic violence cruelty wife husband PWDVA protection order",
        "498A BNS cruelty mental physical harassment matrimonial home",
    ],
    "matrimonial": [
        "divorce maintenance spouse alimony Hindu Marriage Act",
        "child custody welfare minor guardian",
    ],
    "property": [
        "property dispute title ownership possession injunction",
        "specific performance sale agreement Transfer of Property Act",
    ],
    "criminal": [
        "FIR complaint cognizable offence bail BNS BNSS",
        "criminal intimidation threat quash high court",
    ],
    "employment": [
        "wrongful termination reinstatement service matter labour court",
        "gratuity provident fund dues industrial dispute",
    ],
    "consumer": [
        "consumer complaint deficiency service compensation NCDRC",
        "unfair trade practice consumer protection redressal forum",
    ],
    "motor_accident": [
        "motor accident claim compensation MACT negligence rash driving",
        "personal injury insurance liability third party",
    ],
    "cheque_dishonour": [
        "cheque bounce dishonour NI Act section 138 demand notice",
        "cheque dishonour criminal complaint drawer prosecution",
    ],
    "land_acquisition": [
        "land acquisition compensation market value LARR Act 2013",
        "solatium annuity enhanced compensation acquisition award",
    ],
    "general": [],
}

_DISPUTE_LEGAL_NATURE: dict[str, str] = {
    "domestic_violence": "criminal",
    "criminal": "criminal",
    "cheque_dishonour": "criminal",
    "matrimonial": "civil",
    "property": "civil",
    "employment": "civil",
    "consumer": "civil",
    "motor_accident": "civil",
    "land_acquisition": "civil",
    "general": "both",
}


def _category_search_angles(cluster: str, facts_summary: str, max_seeds: int = 2) -> list[str]:
    """Return search angles for a dispute cluster: facts text + category seeds (no LLM)."""
    text = (_build_focus_fact_text(facts_summary, max_chars=350) or facts_summary[:350]).strip()
    seeds = _CATEGORY_SEEDS.get(cluster, [])[:max_seeds]
    angles = [text] + seeds
    return list(dict.fromkeys(a for a in angles if a))[:3]


def _build_disputes_from_intake_state(intake_state: dict | None, facts_summary: str) -> list:
    """
    Build dispute list from agents.intake state clusters — no LLM call.

    Reads primary_issue_cluster + secondary_issue_clusters from the intake state
    (already populated by Stage 1) and returns one dispute dict per cluster,
    each with category-seeded search_angles for targeted parallel retrieval.

    Falls back to _single_dispute_fallback when intake state is absent or empty.
    """
    from retrieval.decomposer import _single_dispute_fallback

    if not intake_state:
        return _single_dispute_fallback(facts_summary)

    primary = intake_state.get("primary_issue_cluster")
    secondaries = intake_state.get("secondary_issue_clusters") or []

    clusters = [primary] + [s for s in secondaries if s and s != primary]
    clusters = [c for c in clusters if c]

    if not clusters:
        return _single_dispute_fallback(facts_summary)

    disputes = []
    for i, cluster in enumerate(clusters[:3]):  # cap at 3 disputes
        angles = _category_search_angles(cluster, facts_summary)
        disputes.append({
            "id": f"d{i + 1}",
            "dispute": facts_summary[:700],
            "legal_nature": _DISPUTE_LEGAL_NATURE.get(cluster, "both"),
            "keywords": [],
            "legal_concepts": [],
            "bare_act_hints": [],
            "search_angles": angles,
        })

    logger.info(
        "Disputes from agents.intake state: %d — %s",
        len(disputes),
        [f"d{i+1}:{c}" for i, c in enumerate(clusters[:3])],
    )
    return disputes


def _build_single_dispute_from_facts(facts_summary: str) -> dict:
    """Use the full fact pattern as one dispute component for the fast chat path."""
    text = _build_focus_fact_text(facts_summary, max_chars=700) or _normalize_fact_text(facts_summary)
    segments = _split_fact_segments(facts_summary)
    keywords = _extract_salient_terms(text, limit=8)
    search_angles: list[str] = []

    for candidate in (
        text,
        " ".join(segments[:2]).strip(),
        " ".join([segments[0], segments[-1]]).strip() if len(segments) >= 2 else "",
        " ".join(keywords[:6]).strip(),
    ):
        normalized = " ".join((candidate or "").split()).strip()[:420]
        if normalized and normalized not in search_angles:
            search_angles.append(normalized)
        if len(search_angles) >= 3:
            break

    return {
        "id": "d1",
        "dispute": text[:700],
        "legal_nature": "mixed",
        "keywords": list(dict.fromkeys(keywords))[:8],
        "legal_concepts": [],
        "bare_act_hints": [],
        "search_angles": list(dict.fromkeys(search_angles))[:3],
    }


def _build_runtime_rescue_queries(
    facts_summary: str,
    limit: int = 2,
    bare_act_sections: list | None = None,
    model_override: str | None = None,
) -> list[str]:
    """Use model-backed query expansion only as a rescue when the first local pass is thin."""
    queries: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        q = " ".join((value or "").split()).strip()[:420]
        if not q:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(q)

    _add(_build_focus_fact_text(facts_summary, max_chars=360) or facts_summary[:320])
    if bare_act_sections:
        for ba in (bare_act_sections or [])[:2]:
            act = (ba.get("act_name") or "").strip()
            sec = str(ba.get("section_number") or "").strip()
            if act and sec:
                _add(f"Section {sec} {act} interpretation judgment India")
            elif act:
                _add(f"{act} interpretation judgment India")
    try:
        for query in expand_legal_query(facts_summary, model_override=model_override)[:12]:
            _add(query[:320])
    except Exception as e:
        logger.debug("Interactive rescue query expansion failed: %s", e)
    return queries[: max(1, limit)]


def _select_interactive_grounded_results(
    raw_results: list,
    facts_summary: str,
    *,
    quality_fn,
    cap: int,
    min_score: float = INTERACTIVE_MIN_RERANK_SCORE,
) -> list:
    """
    Generalized runtime selector for interactive retrieval.

    Keep clearly relevant results first, but when the local corpus is sparse do
    not collapse to zero merely because the reranker score is borderline.
    """
    aligned = _apply_fact_alignment_sort(raw_results or [], facts_summary)
    quality_items = [item for item in aligned if quality_fn(item)]
    if not quality_items:
        return []

    strong = [
        item for item in quality_items
        if max(float(item.get("_rerank_score", 0) or 0), float(item.get("_fact_alignment_score", 0) or 0))
        >= min_score
    ]
    if strong:
        return strong[:cap]

    softened = [
        item for item in quality_items
        if (item.get("_fact_alignment_overlap", 0) or 0) > 0 or float(item.get("_rerank_score", 0) or 0) >= 0.0
    ]
    return softened[: min(cap, 3)]


def _align_bare_acts_with_case_support(bare_acts: list, case_laws: list, facts_summary: str, cap: int) -> list:
    """
    Use retrieved precedent text as an additional general relevance signal.

    If multiple case laws repeatedly reference one statute family, that act
    should outrank one-off sections from unrelated statutes.
    """
    if not bare_acts:
        return []
    if not case_laws:
        return _apply_fact_alignment_sort(bare_acts, facts_summary, cap=cap)

    case_blob = " ".join(
        " ".join(
            str(item.get(field) or "") for field in ("title", "case_name", "text", "full_text")
        )
        for item in (case_laws or [])
    ).lower()

    rescored = []
    for item in bare_acts:
        clone = dict(item)
        act_name = (item.get("act_name") or "").strip().lower()
        sec = str(item.get("section_number") or "").strip().lower()
        # Structural presence check — act/section named in case text is a signal
        # the cases consider this provision relevant; used for tie-breaking only,
        # not as a multiplier on top of the cross-encoder score.
        act_mentioned = bool(act_name and act_name in case_blob)
        sec_mentioned = bool(sec and re.search(rf"\bsection\s+{re.escape(sec)}\b", case_blob))
        clone["_case_support_hits"] = int(act_mentioned) * 2 + int(sec_mentioned)
        clone["_case_aligned_score"] = float(clone.get("_fact_alignment_score", clone.get("_rerank_score", 0)) or 0)
        rescored.append(clone)

    rescored.sort(
        key=lambda item: (
            item.get("_case_aligned_score", 0),
            item.get("_case_support_hits", 0),
            item.get("_rerank_score", 0),
        ),
        reverse=True,
    )
    rescored = _apply_family_consensus_sort(rescored, "_case_aligned_score", cap=None)
    return rescored[:cap]


def _align_case_laws_with_statute_support(case_laws: list, bare_acts: list, _facts_summary: str, cap: int) -> list:
    """
    Re-rank case laws using the statute shortlist as a structural tie-breaker.

    The cross-encoder score is the primary relevance signal. Act/section name
    presence in the case text is used for tie-breaking only — not as an additive
    bonus that overrides the model's judgment.
    """
    if not case_laws:
        return []

    act_names = [(ba.get("act_name") or "").strip().lower() for ba in (bare_acts or []) if (ba.get("act_name") or "").strip()]
    section_numbers = [str(ba.get("section_number") or "").strip().lower() for ba in (bare_acts or []) if str(ba.get("section_number") or "").strip()]

    rescored = []
    for item in case_laws:
        clone = dict(item)
        blob = " ".join(
            str(item.get(field) or "")
            for field in ("title", "case_name", "text", "full_text", "search_text")
        ).lower()
        act_hits = sum(1 for act_name in act_names[:3] if act_name and act_name in blob)
        section_hits = sum(1 for sec in section_numbers[:3] if sec and re.search(rf"\bsection\s+{re.escape(sec)}\b", blob))
        clone["_statute_case_support"] = act_hits * 2 + section_hits
        clone["_case_support_score"] = float(clone.get("_rerank_score", 0) or 0)
        rescored.append(clone)

    rescored.sort(
        key=lambda item: (
            item.get("_case_support_score", 0),
            item.get("_statute_case_support", 0),
            item.get("_rerank_score", 0),
        ),
        reverse=True,
    )
    rescored = _apply_family_consensus_sort(rescored, "_case_support_score", cap=None)
    return rescored[:cap]


def _build_grounded_bare_act_fallback(formatted_bare: list, facts_summary: str = "") -> str:
    """
    Deterministic grounded fallback when we have local statutory material but the
    model output is unusable or too slow to rely on.

    Returns clean framing prose only — section labels (Act + Section + title) but NO
    raw section text. Embedding raw section text produces garbled output because the
    DB text already begins with the section number itself, e.g.:
      "Section 18 supports protection through 18. Protection orders. The Magistrate may..."
    The full section text is shown separately in bare_act_sections cards.
    """
    if not formatted_bare:
        return _local_only_no_materials_message()

    stage_bare = _prepare_bare_act_stage_entries(formatted_bare, facts_summary=facts_summary)
    if not stage_bare:
        stage_bare = _prepare_bare_act_stage_entries(formatted_bare)

    # Group sections by Act name
    act_groups: dict[str, list[dict]] = {}
    for ba in stage_bare:
        act_name = (ba.get("act_name") or ba.get("title") or "the retrieved statute").strip() or "the retrieved statute"
        act_groups.setdefault(act_name, []).append(ba)

    parts: list[str] = []
    parts.append(
        "On the facts presently shared, the locally retrieved statutory materials "
        "point to a supportable legal position on the record."
    )

    body_parts = []
    for act_name, sections in list(act_groups.items())[:3]:
        # Reference sections by label (number + title) only — text is in cards
        section_labels = []
        for ba in sections[:3]:
            sec = str(ba.get("section_number") or "").strip()
            title = (ba.get("section_title") or "").strip()
            if sec and title:
                section_labels.append(f"Section {sec} ({title})")
            elif sec:
                section_labels.append(f"Section {sec}")
        if section_labels:
            body_parts.append(
                f"{act_name}: the most directly applicable provisions are {', '.join(section_labels)}."
            )
        else:
            body_parts.append(
                f"{act_name} appears to be one of the most material retrieved Acts for this dispute."
            )

    if body_parts:
        parts.append("\n".join(body_parts))

    parts.append(
        "The full text of each provision is set out in the cards below. "
        "The safest next step is to test these materials against the notice, "
        "communications, and timeline already mentioned before taking a firmer position."
    )

    return "\n\n".join(parts).strip()


def _build_grounded_interactive_fallback(
    facts_summary: str,
    formatted_bare: list,
    formatted_case: list,
) -> str:
    """
    Deterministic grounded opinion_text for the fast interactive lane.

    Returns clean framing prose only (2-4 sentences). Section text is shown in
    bare_act_sections cards and must NOT be embedded here — embedding truncated
    section text (which starts with its own section number) produces garbled output
    like "Section 18 points to 18. Protection orders. The Magistrate may..." because
    the DB text already begins with the section number. Case law text truncated at
    220 characters also reads as mid-sentence fragments.
    """
    local_bare, local_case = _filter_materials_to_local_db(formatted_bare, formatted_case)
    if not local_bare and not local_case:
        return _local_only_no_materials_message()
    if local_bare and not local_case:
        return _build_grounded_bare_act_fallback(local_bare, facts_summary=facts_summary)

    # Build section labels (act + section number + title where available).
    # Do NOT embed raw section text — it belongs in the bare_act_sections cards.
    bare_labels: list[str] = []
    for ba in local_bare[:3]:
        act = (ba.get("act_name") or ba.get("title") or "the retrieved statute").strip()
        sec = str(ba.get("section_number") or "").strip()
        title = (ba.get("section_title") or "").strip()
        if sec and title:
            bare_labels.append(f"{act}, Section {sec} ({title})")
        elif sec:
            bare_labels.append(f"{act}, Section {sec}")
        else:
            bare_labels.append(act)

    # Build case labels (name + court).
    # Do NOT embed raw case text — truncated text reads as fragments.
    case_labels: list[str] = []
    for cl in local_case[:2]:
        name = (cl.get("title") or cl.get("case_name") or cl.get("source") or "the retrieved case").strip()
        court = (cl.get("court") or "").strip()
        case_labels.append(f"{name} ({court})" if court and court not in name else name)

    parts: list[str] = []

    # Opening framing — reference the type of dispute, not raw facts_summary which
    # may contain backtick-wrapped chat history or scaffold markers.
    parts.append(
        "On the facts presently shared, the locally retrieved materials point to a "
        "supportable legal position on the record."
    )

    if bare_labels:
        if len(bare_labels) == 1:
            parts.append(f"The principal statutory provision retrieved is {bare_labels[0]}.")
        else:
            joined = ", ".join(bare_labels[:-1]) + f", and {bare_labels[-1]}"
            parts.append(f"The primary statutory provisions retrieved are {joined}.")

    if case_labels:
        if len(case_labels) == 1:
            parts.append(f"The strongest locally retrieved precedent is {case_labels[0]}.")
        else:
            joined = " and ".join(case_labels)
            parts.append(f"The locally retrieved precedents include {joined}.")

    parts.append(
        "The full text of these provisions and cases is set out in the cards below. "
        "The safest next step is to test these materials against the notice, "
        "communications, and timeline already mentioned before taking a firmer position."
    )

    return "\n\n".join(parts).strip()


def _stream_precomposed_text(text: str, token_callback=None, words_per_chunk: int = 3) -> None:
    """Emit a prebuilt fallback response in small word chunks for the live UI."""
    if not token_callback or not (text or "").strip():
        return
    words = text.split()
    if not words:
        return
    for i in range(0, len(words), max(1, words_per_chunk)):
        chunk = " ".join(words[i:i + words_per_chunk])
        if i + words_per_chunk < len(words):
            chunk += " "
        token_callback(chunk)


def _build_live_streamer(token_callback=None):
    """
    Build a tiny stateful helper that can stream short provisional phrases only
    once each. The final persisted answer still comes from the actual result.
    """
    emitted: set[str] = set()

    def _emit_once(key: str, text: str, words_per_chunk: int = 3) -> None:
        if key in emitted or not token_callback:
            return
        emitted.add(key)
        _stream_precomposed_text(text, token_callback=token_callback, words_per_chunk=words_per_chunk)

    return _emit_once


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


def _format_indiankanoon_fallback_results(results: list | None, kind: str) -> list:
    """Shape Indiankanoon fallback results for the UI without mixing them into grounded analysis."""
    formatted: list = []
    seen: set[str] = set()
    label = "Bare Act" if kind == "bare_act" else "Case Law"
    for item in results or []:
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (item.get("title") or "").strip() or f"{label} fallback result"
        snippet = " ".join((item.get("snippet") or item.get("text") or "").split()).strip()
        text = snippet[:700]
        formatted.append({
            "title": f"[{label}] {title}",
            "text": text,
            "url": url,
            "source_tag": "INDIANKANOON",
        })
    return formatted


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


def _enforce_local_grounding(text: str, _bare_acts: list | None = None) -> str:
    """Pass-through — grounding is enforced via prompt instruction, not post-hoc regex."""
    return (text or "").strip()


def expand_legal_query(
    facts: str,
    intent: dict = None,
    expansion_debug: dict | None = None,
    model_override: str | None = None,
) -> list[str]:
    """
    Convert plain-language facts to legal research queries for hybrid retrieval.

    The model emits distinct legal issues; each issue may contribute up to three
    queries. There is no fixed global cap on how many issues or total queries.
    Heuristic fallbacks apply only when the model returns nothing usable.
    """
    from prompts.research import EXPAND_LEGAL_QUERY_SYSTEM, EXPAND_LEGAL_QUERY_INTENT_BLOCK

    _MAX_QUERIES_PER_ISSUE = 3
    # Vector/BM25: short key-phrases (prompt targets ~5–10 words); hard cap 12 words + char ceiling
    _EXPANDED_QUERY_MAX_WORDS = 12
    _EXPANDED_QUERY_MAX_CHARS = 200
    _MODEL_RAW_RESPONSE_DEBUG_MAX_CHARS = 2500

    def _normalize_fact_query(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        cleaned = re.sub(r"[\"'`]+", "", cleaned)
        words = cleaned.split()
        if len(words) > _EXPANDED_QUERY_MAX_WORDS:
            cleaned = " ".join(words[:_EXPANDED_QUERY_MAX_WORDS])
        return cleaned[:_EXPANDED_QUERY_MAX_CHARS]

    fact_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]{4,}", (facts or "").lower())
        if tok not in {"with", "from", "that", "this", "have", "there", "their", "about", "which", "would", "could"}
    }
    fact_mentions_statute = bool(re.search(r"\b(act|code|rules|regulation|regulations|section)\b", (facts or "").lower()))

    def _is_grounded_query(query_text: str) -> bool:
        low = (query_text or "").lower()
        if not low:
            return False
        if any(marker in low for marker in ("case law search", "bare act search", "search query", "query:", "search:")):
            return False
        query_tokens = {
            tok for tok in re.findall(r"[a-zA-Z]{4,}", low)
            if tok not in {"with", "from", "that", "this", "have", "there", "their", "about", "which", "would", "could"}
        }
        overlap = len(query_tokens & fact_tokens)
        act_mentions = re.findall(
            r"\b[A-Z][A-Za-z0-9,&(). -]{0,80}\b(?:Act|Code|Rules|Regulation(?:s)?|Procedure)\b(?:,?\s*\d{4})?",
            query_text,
        )
        distinct_acts = {item.strip().lower() for item in act_mentions if item.strip()}
        if not fact_mentions_statute and len(distinct_acts) > 1:
            return False
        if (query_text.count("+") >= 1 or query_text.count(";") >= 2) and len(distinct_acts) > 1:
            return False
        # Terse key-phrases (≤12 meaningful tokens) need lighter overlap than long queries
        if fact_tokens and len(fact_tokens) >= 5:
            nq = len(query_tokens)
            if nq <= 12:
                if nq >= 2 and overlap < 1:
                    return False
                if nq == 1 and overlap < 1:
                    return False
            elif nq >= 4 and overlap < 2:
                return False
        return True

    def _add_query(bucket: list[str], seen: set[str], q: str) -> None:
        normalized = _normalize_fact_query(q)
        key = normalized.lower()
        if not normalized or key in seen or not _is_grounded_query(normalized):
            return
        seen.add(key)
        bucket.append(normalized)

    def _parse_model_expansion(raw: str) -> tuple[list[str], list[dict]]:
        """Returns (flat_queries_in_order, issues_metadata_for_debug)."""
        if not (raw or "").strip():
            return [], []
        flat: list[str] = []
        seen_keys: set[str] = set()
        issues_meta: list[dict] = []
        text = raw.strip()
        possible_chunks = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            possible_chunks.insert(0, match.group(0))
        for chunk in possible_chunks:
            try:
                parsed = json.loads(chunk)
            except Exception:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("issues"), list):
                for item in parsed["issues"]:
                    if not isinstance(item, dict):
                        continue
                    label = (
                        str(item.get("issue_label") or item.get("label") or item.get("issue") or "").strip()
                    )
                    qraw = item.get("queries") or item.get("expanded_queries") or item.get("search_queries")
                    if not isinstance(qraw, list):
                        continue
                    issue_queries: list[str] = []
                    for value in qraw[:_MAX_QUERIES_PER_ISSUE]:
                        _add_query(issue_queries, seen_keys, str(value))
                    for q in issue_queries:
                        flat.append(q)
                    if label or issue_queries:
                        issues_meta.append({"issue_label": label, "queries": issue_queries})
                if flat:
                    return flat, issues_meta
            # Legacy: flat "queries" array (single blob)
            if isinstance(parsed, dict):
                values = parsed.get("queries")
            elif isinstance(parsed, list):
                values = parsed
            else:
                values = []
            legacy_flat: list[str] = []
            for value in values or []:
                _add_query(legacy_flat, seen_keys, str(value))
            if legacy_flat:
                issues_meta.append({"issue_label": "", "queries": legacy_flat})
                return legacy_flat, issues_meta

        # Plain-text fallback: lines / semicolon-separated candidates
        candidate_keys: set[str] = set()
        fallback: list[str] = []
        for part in re.split(r"[\r\n]+|;\s*", text):
            _add_query(fallback, candidate_keys, part.lstrip("-*0123456789. ").strip())
        if fallback:
            issues_meta.append({"issue_label": "", "queries": fallback})
        return fallback, issues_meta

    intent_block = ""
    if intent and isinstance(intent, dict) and (intent.get("states") or intent.get("domains") or intent.get("topics")):
        intent_block = EXPAND_LEGAL_QUERY_INTENT_BLOCK.format(intent_json=json.dumps(intent, indent=0))

    queries: list[str] = []
    raw_model = ""
    model_parsed: list[str] = []
    issues_from_model: list[dict] = []
    normalized_facts = _normalize_fact_query(facts)
    focus_query = _normalize_fact_query(_build_focus_fact_text(facts, max_chars=420) or normalized_facts)

    prompt = f"""{EXPAND_LEGAL_QUERY_SYSTEM}
{intent_block}

FACTS:
{facts[:2000]}

Return JSON only:"""
    try:
        raw_model = ask_llm(prompt, task_hint="fast", model=model_override).strip()
        model_parsed, issues_from_model = _parse_model_expansion(raw_model)
        queries = list(model_parsed)
    except Exception:
        queries = []

    if not queries:
        seen_fb: set[str] = set()
        _add_query(queries, seen_fb, normalized_facts)
        if focus_query and focus_query.lower() not in seen_fb:
            _add_query(queries, seen_fb, focus_query)

    def _consolidate(qs: list[str]) -> list[str]:
        """
        Remove near-duplicate queries using Jaccard token overlap.
        Two queries are considered duplicates when they share >= 60% of
        their meaningful tokens (4+ chars, ignoring common stopwords).
        Earlier queries (higher issue priority) are kept; later near-duplicates
        are dropped.  Single-token queries are never consolidated away.
        """
        _STOP = {"with", "from", "that", "this", "have", "there", "their",
                 "about", "which", "would", "could", "been", "were", "into"}
        _JACCARD_THRESHOLD = 0.60

        def _tokens(text: str) -> frozenset[str]:
            return frozenset(
                t for t in re.findall(r"[a-zA-Z]{4,}", text.lower())
                if t not in _STOP
            )

        kept: list[str] = []
        kept_tokens: list[frozenset] = []
        for q in qs:
            qt = _tokens(q)
            if not qt:
                kept.append(q)
                kept_tokens.append(qt)
                continue
            duplicate = False
            for kt in kept_tokens:
                if not kt:
                    continue
                intersection = len(qt & kt)
                union = len(qt | kt)
                if union and intersection / union >= _JACCARD_THRESHOLD:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(q)
                kept_tokens.append(qt)
        return kept

    out = _consolidate(queries)
    if expansion_debug is not None:
        expansion_debug["model_raw_response"] = (raw_model or "")[:_MODEL_RAW_RESPONSE_DEBUG_MAX_CHARS]
        expansion_debug["issues_from_model"] = list(issues_from_model)
        expansion_debug["parsed_queries_from_model"] = list(model_parsed)
        expansion_debug["consolidated_dropped"] = [q for q in queries if q not in out]
        expansion_debug["final_expanded_queries"] = list(out)
        expansion_debug["queries_added_after_model"] = [q for q in out if q not in model_parsed]
    return out


# ---------------------------------------------------------------------------
# Extract relevant portions from fetched content
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


def _extract_json(raw: str):
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty JSON payload")

    try:
        return json.loads(text)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in '{[':
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            return parsed
        except Exception:
            continue

    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    raise ValueError("No JSON object found in model output")


def _check_bare_act_sufficiency(dispute_text: str, bare_acts: list, llm_fn=None) -> bool:
    """
    Lightweight LLM check: are the retrieved bare act sections sufficient for this dispute?
    Returns True  → stop, local coverage is sufficient.
    Returns False → local coverage may be incomplete.
    Defaults to False on any error.
    """
    if llm_fn is None:
        from platform.llm import ask_llm
        llm_fn = lambda prompt: ask_llm(prompt, task_hint="fast")
    from prompts.research import BARE_ACT_DISPUTE_SUFFICIENCY_PROMPT

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



def _build_bare_act_queries(dispute: dict, expanded_queries: list[str] | None = None) -> list[str]:
    """
    Build 3-5 diverse search queries for bare act retrieval from a single dispute.

    All queries are derived from the user's own language — no IPC/BNS section numbers
    or act names are injected from platform.training knowledge. Grounding in retrieved data only.

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

    # Q-1: LLM-expanded issue phrases (short, legally focused) — run first for vector/BM25.
    for eq in (expanded_queries or [])[:24]:
        if isinstance(eq, str) and eq.strip():
            _add(eq.strip()[:400])

    # Q0: legal_concepts + dispute text — contextual queries that carry the relational
    # domain (e.g. "domestic violence marital cruelty wife husband protection") rather
    # than surface-act words alone. Each concept is joined with the dispute text so the
    # vector search retrieves from the correct statutory domain, not just surface keywords.
    for c in legal_concepts[:2]:
        if c and isinstance(c, str):
            # Concept alone (structured lookup — e.g. "criminal intimidation")
            _add(c.strip())
            # Concept + dispute text (domain-anchored query — surfaces correct Act family)
            concept_anchored = f"{c.strip()} {dispute_text}".strip()
            _add(concept_anchored[:400])

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
            from prompts.research import BARE_ACT_SEARCH_QUERIES_PROMPT
            import json as _json
            prompt = BARE_ACT_SEARCH_QUERIES_PROMPT.format(
                dispute=dispute_text[:300],
                act_hints=", ".join(bare_act_hints[:3]) or "unknown",
                keywords=", ".join(keywords[:5]) or "none",
            )
            raw = ask_llm(prompt, task_hint="fast").strip()
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
    from retrieval.retriever import search_bare_acts_auto
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
    Returns generic Indiankanoon fallback results for bare acts.

    Query priority:
      1. search_angles from decomposer (focused plain-language phrases)
      2. round1_queries[1:] — reuse the focused queries already run in Round 1
         (these include LLM-generated angles like "criminal assault causing grievous hurt
         with metal rod"; skip Q1 which is the broad dispute+keywords combination)
      3. keywords — keyword-level fallback
      4. dispute_text[:200] — last resort when nothing else is available
    """
    from retrieval.enricher import search_for_gaps
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
        # These are state-agnostic and surface major central statutes on IndiaCode
        # regardless of which state the dispute is in.
        for q in focused[:2]:
            _add_gap(f"{q} India central act")

        # ── State-specific queries (supplement, not replacement) ─────────────────
        # Indian law is layered: central acts apply everywhere, but states may have
        # concurrent legislation. We add ONE state-specific query per detected state
        # so local acts are searched alongside central acts.
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

    # ── Structural injection: new 2023 criminal codes ────────────────────────
    # Drive BNS/BNSS/BSA injection from dispute.legal_nature (set by intake state)
    # rather than keyword matching the dispute text.
    legal_nature = dispute.get("legal_nature", "both")
    is_criminal = legal_nature in ("criminal", "both")

    hints_lower = " ".join(bare_act_hints).lower()
    already_has_bns  = "bharatiya nyaya sanhita"    in hints_lower
    already_has_bnss = "bharatiya nagarik suraksha" in hints_lower
    already_has_bsa  = "bharatiya sakshya"          in hints_lower

    hint_count = min(len(bare_act_hints), 3) if (bare_act_hints and primary_term) else 0

    if is_criminal and not already_has_bns and primary_term:
        bns_q = f"Bharatiya Nyaya Sanhita 2023 {primary_term}"
        if bns_q.lower() not in seen_q:
            web_gaps.insert(hint_count, {"query": bns_q, "type": "bare_act"})
            seen_q.add(bns_q.lower())
            logger.info("BNS query (criminal dispute): %s", bns_q[:100])

    if is_criminal and not already_has_bnss and primary_term:
        bnss_q = f"Bharatiya Nagarik Suraksha Sanhita 2023 {primary_term}"
        if bnss_q.lower() not in seen_q:
            pos = hint_count + (1 if is_criminal and not already_has_bns else 0)
            web_gaps.insert(pos, {"query": bnss_q, "type": "bare_act"})
            seen_q.add(bnss_q.lower())
            logger.info("BNSS query (criminal dispute): %s", bnss_q[:100])

    if is_criminal and not already_has_bsa and primary_term:
        bsa_q = f"Bharatiya Sakshya Adhiniyam 2023 {primary_term}"
        if bsa_q.lower() not in seen_q:
            web_gaps.insert(hint_count + 2, {"query": bsa_q, "type": "bare_act"})
            seen_q.add(bsa_q.lower())
            logger.info("BSA query (criminal dispute): %s", bsa_q[:100])

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

    bare_results = gap_results.get("bare_act_results", [])
    logger.info(
        "Indiankanoon bare-act fallback: %d results for dispute '%s'",
        len(bare_results),
        dispute_text[:60],
    )
    return bare_results


def _filter_bare_acts_with_llm(dispute: dict, bare_acts: list, debug: dict | None = None) -> list:
    """
    Use an LLM to down-rank or drop obviously off-topic bare act sections for one dispute.
    Falls back to the original list on any error or empty filter output.
    """
    if not bare_acts:
        return bare_acts
    if not _ENABLE_BARE_ACT_LLM_FILTER:
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
        resp = ask_llm(prompt, task_hint="fast")
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
    if not _ENABLE_CASE_LAW_LLM_FILTER:
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
        resp = ask_llm(prompt, task_hint="fast")
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


def retrieve_bare_acts_for_dispute(
    dispute: dict,
    full_query: str,
    states: list = None,
    debug: dict | None = None,
    pending_indexing_list: list = None,
    bare_act_trace: list | None = None,
    expanded_legal_queries: list[str] | None = None,
) -> list:
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

    External fallback:
        If local retrieval returns zero relevant sections, the caller may query
        Indiankanoon for fallback links/results, but those are not merged into
        grounded local analysis here.

    No result cap — every section that passes quality filter is returned.
    """
    from retrieval.retriever import search_bare_acts_auto, search_bare_acts_filtered

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

    # Early LLM act refinement removed.
    # Act profile index was deleted; allowed_acts stays frozenset() and queries run
    # unfiltered. LLM act refinement now runs AFTER cross-encoder retrieval (below)
    # so it can contextually filter whatever the retriever actually surfaced.

    # Build diverse query set (expanded issue phrases first)
    queries = _build_bare_act_queries(dispute, expanded_queries=expanded_legal_queries)
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
        if bare_act_trace is None:
            if allowed_acts:
                return search_bare_acts_filtered(q, allowed_acts, top_k=15)
            return search_bare_acts_auto(q, top_k=15)
        stage_trace: list = []
        if allowed_acts:
            out = search_bare_acts_filtered(q, allowed_acts, top_k=15, trace=stage_trace)
        else:
            out = search_bare_acts_auto(q, top_k=15, trace=stage_trace)
        bare_act_trace.append({"retrieval_query": q, "hybrid_stages": stage_trace})
        return out

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

    # ── Always-on LLM act refinement (post-retrieval) ────────────────────────
    # Build candidates from three sources (priority order):
    #   1. "hint"      — Acts the decomposer identified from dispute context
    #   2. "procedural_companion" — procedural Acts (BNSS, CPC, BSA) auto-injected
    #   3. "retrieved" — Acts the cross-encoder surfaced from the index
    # The LLM then rates each candidate "high", "medium", or "low" against the
    # FULL dispute context (text + legal_concepts). This eliminates surface-word
    # false positives (e.g. POCSO for a marital assault query) without any
    # hardcoded domain heuristics.
    candidate_acts = []
    seen_act_names: set = set()

    def _add_candidate(act_name: str, source: str, note: str = "") -> None:
        n = (act_name or "").strip()
        if n and n.lower() not in seen_act_names:
            candidate_acts.append({"act_name": n, "source": source, "note": note})
            seen_act_names.add(n.lower())

    for hint in bare_act_hints:
        _add_candidate(str(hint or ""), "hint")
    for act_name in sorted(top_acts):
        _add_candidate(act_name, "retrieved")

    if candidate_acts:
        try:
            # Pass dispute text enriched with legal_concepts so the LLM can use
            # relational/domain context (not just surface words) to judge relevance.
            legal_concepts = dispute.get("legal_concepts", [])
            dispute_context = dispute_text
            if legal_concepts:
                dispute_context += f"\nLegal concepts: {', '.join(legal_concepts)}"
            prompt = ACT_SELECTION_PROMPT.format(
                dispute=dispute_context[:600],
                candidate_acts_json=json.dumps(candidate_acts, ensure_ascii=False),
            )
            resp = ask_llm(prompt, task_hint="fast")
            data = json.loads(resp)
            refined = [
                (a.get("act_name") or "").strip()
                for a in (data.get("acts") or [])
                if (a.get("relevance") or "").strip().lower() in ("high", "medium")
            ]
            refined = [n for n in refined if n]
            if refined:
                top_acts = set(refined)
                logger.info(
                    "[%s] LLM act refinement (post-retrieval): %d high/medium acts → %s",
                    dispute_id, len(top_acts), sorted(top_acts),
                )
                if debug_entry is not None:
                    debug_entry["llm_acts"] = sorted(list(top_acts))
            else:
                logger.warning(
                    "[%s] LLM act refinement returned no high/medium acts — keeping cross-encoder top_acts %s",
                    dispute_id, sorted(top_acts),
                )
        except Exception as e:
            logger.warning(
                "[%s] LLM act refinement failed: %s — keeping cross-encoder top_acts",
                dispute_id, e,
            )

    if top_acts:
        # Use word-overlap fuzzy match instead of exact string equality.
        # LLM-returned act names and index act names often differ in punctuation,
        # year format, or short-name variants (e.g. "BNS 2023" vs "Bharatiya Nyaya
        # Sanhita, 2023"). Requiring ≥1 meaningful word overlap (4+ chars, non-stopword)
        # handles these variants without hardcoded aliases.
        def _act_matches_refined(indexed_name: str, refined_set: set) -> bool:
            il = indexed_name.lower()
            # Use 5+ char words: drops short connectors ('from', 'that', 'with', 'act')
            # while keeping domain-specific identifiers ('bharatiya', 'sanhita',
            # 'domestic', 'violence', 'sakshya', 'criminal', 'protection', etc.).
            # Require ≥2 such words to match, so single shared generic words like
            # "protection" (in both PWDVA and POCSO) don't create false positives.
            il_words = set(re.findall(r'\b[a-z]{5,}\b', il))
            for r in refined_set:
                rl = r.lower()
                # Exact substring match handles perfect name/alias hits
                if rl in il or il in rl:
                    return True
                rl_words = set(re.findall(r'\b[a-z]{5,}\b', rl))
                if len(il_words & rl_words) >= 2:
                    return True
            return False

        before = len(all_local)
        filtered = [
            ba for ba in all_local
            if _act_matches_refined((ba.get("act_name") or "").strip(), top_acts)
        ]
        # Safety: if fuzzy filter drops everything (e.g. hint-only acts not yet indexed),
        # keep all_local unfiltered so retrieval doesn't silently return zero sections.
        if filtered:
            all_local = filtered
            logger.info(
                "[%s] Act pre-filter: %d → %d sections (kept acts: %s)",
                dispute_id, before, len(all_local), sorted(top_acts),
            )
        else:
            logger.warning(
                "[%s] Act pre-filter produced 0 matches (refined=%s) — keeping all %d sections unfiltered",
                dispute_id, sorted(top_acts), before,
            )

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

    # Stop condition: if local retrieval found anything relevant, return it.
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
    _AUTO_STOP_COUNT = 3
    _AUTO_STOP_VOLUME = 5

    if len(high_quality_local) >= _AUTO_STOP_COUNT:
        logger.info(
            "[%s] Bare acts Round 1 auto-sufficient (%d HQ sections ≥ %d, score>%.1f).",
            dispute_id, len(high_quality_local), _AUTO_STOP_COUNT, BARE_ACT_HIGH_QUALITY_SCORE,
        )
        filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
        return _cap_bare_acts_by_score(
            _apply_fact_alignment_sort(
                filtered_local, (dispute.get("dispute") or "").strip(), cap=len(filtered_local)
            ),
            MAX_SECTIONS_PER_DISPUTE_TOTAL,
        )

    if len(local_results) >= _AUTO_STOP_VOLUME:
        logger.info(
            "[%s] Bare acts Round 1 auto-sufficient by volume (%d sections ≥ %d).",
            dispute_id, len(local_results), _AUTO_STOP_VOLUME,
        )
        filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
        return _cap_bare_acts_by_score(
            _apply_fact_alignment_sort(
                filtered_local, (dispute.get("dispute") or "").strip(), cap=len(filtered_local)
            ),
            MAX_SECTIONS_PER_DISPUTE_TOTAL,
        )

    if high_quality_local and local_results and _check_bare_act_sufficiency(dispute_text, local_results):
        logger.info(
            "[%s] Bare acts Round 1 sufficient (%d sections, %d high-quality).",
            dispute_id, len(local_results), len(high_quality_local),
        )
        filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
        return _cap_bare_acts_by_score(
            _apply_fact_alignment_sort(
                filtered_local, (dispute.get("dispute") or "").strip(), cap=len(filtered_local)
            ),
            MAX_SECTIONS_PER_DISPUTE_TOTAL,
        )

    if local_results:
        logger.info("[%s] Bare acts Round 1 returning %d local section(s); no external fallback needed.", dispute_id, len(local_results))
        filtered_local = _filter_bare_acts_with_llm(dispute, local_results, debug)
        return _cap_bare_acts_by_score(
            _apply_fact_alignment_sort(
                filtered_local, (dispute.get("dispute") or "").strip(), cap=len(filtered_local)
            ),
            MAX_SECTIONS_PER_DISPUTE_TOTAL,
        )

    logger.info("[%s] Bare acts Round 1 found no relevant local sections.", dispute_id)
    return []


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


def _build_fast_case_queries(
    dispute: dict,
    bare_act_sections: list,
    expanded_queries: list[str] | None = None,
) -> list[str]:
    """Small, diverse case-law query set for the fast interactive path."""
    dispute_text = (dispute.get("dispute") or "").strip()
    queries: list[str] = []
    seen: set[str] = set()

    def _add(value: str):
        q = " ".join((value or "").split()).strip()[:500]
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            queries.append(q)

    for eq in (expanded_queries or [])[:12]:
        if isinstance(eq, str) and eq.strip():
            _add(f"{eq.strip()[:320]} case law India")

    _add(_build_case_law_query(dispute_text, bare_act_sections))

    if bare_act_sections:
        _ba = bare_act_sections[0]
        _act = (_ba.get("act_name") or "").strip()
        _sec = (_ba.get("section_number") or "").strip()
        if _act and _sec:
            _add(f"Section {_sec} {_act} case law interpretation")

    for angle in (dispute.get("search_angles") or [])[:2]:
        _add(f"{angle} case law")

    keywords = [str(k).strip() for k in (dispute.get("keywords") or []) if str(k).strip()]
    if keywords:
        _add(f"{dispute_text[:220]} {' '.join(keywords[:4])} case law")

    return queries[:_FAST_CASE_QUERY_LIMIT]


def _select_top_acts(bare_acts: list, max_acts: int = 4) -> set[str]:
    """
    Keep the strongest act clusters for one dispute, as candidates for LLM refinement.

    Acts are ranked by:
    1. best section rerank score
    2. number of supporting sections

    max_acts is raised to 4 (from 2) so the always-on LLM refinement receives
    enough candidates to surface the contextually correct Act even when it is not
    the single highest cross-encoder scorer (e.g. PWDVA 2005 behind BNS for a
    marital assault query). The LLM then narrows these to high/medium relevance.
    """
    act_scores: dict[str, dict] = {}
    for ba in bare_acts or []:
        act_name = (ba.get("act_name") or "").strip()
        if not act_name:
            continue
        score = float(ba.get("_rerank_score", 0) or 0)
        entry = act_scores.setdefault(act_name, {"best": score, "count": 0})
        entry["best"] = max(entry["best"], score)
        entry["count"] += 1

    ranked = sorted(
        act_scores.items(),
        key=lambda item: (item[1]["best"], item[1]["count"]),
        reverse=True,
    )
    return {act for act, _meta in ranked[:max_acts] if act}


def _apply_dispute_case_law_limit(case_laws: list, num_sections: int = 1) -> list:
    """
    Threshold-based limit for per-dispute case laws.
    Always keep a tight cap of 3 strongest paragraphs for consistency.
    Input must already be quality-filtered.
    """
    if not case_laws:
        return []
    _ = num_sections
    cap = 3
    sorted_cls = sorted(
        case_laws,
        key=lambda x: (
            x.get("_paragraph_precision_score", 0),
            x.get("_case_support_score", 0),
            x.get("_fact_alignment_score", 0),
            x.get("_rerank_score", 0),
        ),
        reverse=True,
    )
    high_quality = [cl for cl in sorted_cls if cl.get("_rerank_score", 0) > HIGH_QUALITY_SCORE]
    if high_quality:
        return high_quality[:cap]
    return sorted_cls[:cap]


def _web_search_case_laws(dispute: dict, bare_act_sections: list, full_query: str, states: list = None, pending_indexing_list: list = None) -> list:
    """
    Web search for case laws for one dispute, using the richer dispute+sections query.
    Runs an Indiankanoon-only web search for case laws.
    This path no longer builds pending-indexing candidates.
    Returns generic Indiankanoon fallback results for case laws.
    """
    from retrieval.enricher import search_for_gaps
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

    case_results = gap_results.get("case_law_results", [])
    logger.info(
        "[%s] Indiankanoon case-law fallback: %d results.",
        dispute_id, len(case_results),
    )
    return case_results


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

    No external results are merged into grounded analysis here. Indiankanoon
    fallback is handled by the caller only when local retrieval returns zero.
    """
    from retrieval.retriever import search_case_laws_auto

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
    if local_results:
        local_results = _align_case_laws_with_statute_support(
            local_results,
            bare_act_sections,
            full_query,
            cap=max(len(local_results), MAX_CASE_LAWS_PER_DISPUTE),
        )
    limited = _apply_dispute_case_law_limit(local_results, num_sections=len(bare_act_sections))
    logger.info("[%s] Case laws Round 1 local: %d after filter, %d after limit", dispute_id, len(local_results), len(limited))

    # If we got the maximum (5 high-quality), stop — already sufficient.
    if len(limited) >= 5:
        logger.info("[%s] Case laws Round 1 sufficient (5 high-quality). Stopping.", dispute_id)
        return limited
    if local_results:
        logger.info("[%s] Case laws Round 1 returning %d local result(s); no external fallback needed.", dispute_id, len(limited))
        llm_filtered = _filter_case_laws_with_llm(dispute, bare_act_sections, local_results, debug)
        llm_filtered = _align_case_laws_with_statute_support(
            llm_filtered,
            bare_act_sections,
            full_query,
            cap=max(len(llm_filtered), MAX_CASE_LAWS_PER_DISPUTE),
        )
        return _apply_dispute_case_law_limit(llm_filtered, num_sections=len(bare_act_sections))

    logger.info("[%s] Case laws Round 1 found no relevant local judgments.", dispute_id)
    return []


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

    from prompts.research import BARE_ACT_EXPLAIN_AND_FOLLOWUP_PROMPT

    # Build a concise list — cap at 10 sections to keep prompt manageable
    lines = []
    for ba in bare_acts[:10]:
        section_text = (ba.get("full_text") or ba.get("text") or "")[:_MAX_SECTION_TEXT]
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
        raw = ask_llm(prompt, task_hint="fast")
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
    model_override: str | None = None,
    analysis_mode: str = "full_opinion",
    intake_state: dict | None = None,
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
    # Ensure helper LLM calls (intent extraction/query expansion/etc.) follow
    # the frontend-selected model/provider for this request context.
    set_request_model_override(model_override)

    def _emit_progress():
        if progress_callback:
            try:
                progress_callback(progress.get_progress_snapshot())
            except Exception:
                pass

    def _emit_step(message: str, icon: str = "", detail: dict | None = None):
        """Emit a single clean user-facing step message (separate from ProgressTracker)."""
        if step_callback:
            try:
                payload = {"message": message, "icon": icon}
                if detail:
                    payload["detail"] = detail
                step_callback(payload)
            except Exception:
                pass

    if search_strategy == "web_only":
        logger.info("Overriding deprecated legal search strategy 'web_only' -> 'local_then_web'")
        search_strategy = "local_then_web"
    if search_strategy not in ("local_only", "local_then_web"):
        search_strategy = "local_then_web"
    interactive_fast_path = (
        _ENABLE_INTERACTIVE_FAST_PATH
        and intent == "legal_opinion"
        and search_strategy == "local_only"
    )
    _emit_live_token = _build_live_streamer(token_callback if interactive_fast_path else None)

    # Intent extraction — used for query expansion
    research_intent = None
    if not interactive_fast_path:
        try:
            from retrieval.intent import extract_research_intent
            research_intent = extract_research_intent(facts_summary, model_override=model_override)
            if research_intent:
                _emit_step(
                    "Intent parsed: "
                    f"docs={research_intent.get('document_types', 'both')}, "
                    f"strategy={research_intent.get('search_strategy', search_strategy)}, "
                    f"scope={research_intent.get('scope', 'specific')}",
                    "🧭",
                )
        except Exception as e:
            logger.debug("Intent extraction skipped: %s", e)

    # Step 1: Query expansion (returns 1–3 queries for multi-query retrieval)
    fast_query_seed = _build_focus_fact_text(facts_summary, max_chars=520) or _strip_chat_window_summary(facts_summary)
    expansion_debug: dict = {}
    if interactive_fast_path:
        legal_queries = expand_legal_query(
            fast_query_seed[:400],
            intent=research_intent,
            expansion_debug=expansion_debug,
            model_override=model_override,
        ) or [
            fast_query_seed[:400]
        ]
        expansion_debug.setdefault("expansion_input_preview", (fast_query_seed[:400] or "").strip())
    else:
        legal_queries = expand_legal_query(
            facts_summary,
            intent=research_intent,
            expansion_debug=expansion_debug,
            model_override=model_override,
        )
        expansion_debug.setdefault("expansion_input_preview", (facts_summary[:800] or "").strip())
    legal_query = legal_queries[0] if legal_queries else facts_summary[:300]
    logger.info("Expanded query(s): %s", legal_query[:200] if legal_query else "none")
    progress.update_retrieval_diagnostics(
        {
            "query_expansion": {
                "kind": "query_expansion",
                **expansion_debug,
            }
        }
    )
    _emit_progress()
    _emit_step(
        f"Query expansion complete: {len(legal_queries)} retrieval quer{'y' if len(legal_queries) == 1 else 'ies'}",
        "🔎",
    )

    analysis_mode = (analysis_mode or "full_opinion").strip().lower() or "full_opinion"

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
    if analysis_mode == "bare_acts_only":
        retrieve_acts = True
        retrieve_case_laws_flag = False
    elif analysis_mode == "precedents_only":
        retrieve_acts = True
        retrieve_case_laws_flag = True
    if not retrieve_acts and not retrieve_case_laws_flag:
        retrieve_acts = retrieve_case_laws_flag = True
    _flow = (
        f"Flow selected: intent={intent}, mode={analysis_mode}, "
        f"retrieve_acts={retrieve_acts}, retrieve_case_laws={retrieve_case_laws_flag}, "
        f"strategy={search_strategy}"
    )
    _emit_step(_flow, "⚙️")

    # Step 2: Dispute decomposition
    _emit_step("Analysing your legal situation...", "🔍")
    progress.start_group("Dispute Analysis", "Identifying distinct dispute components")
    _emit_progress()

    if intent in ("search", "lookup"):
        # Direct search/lookup — no decomposition needed
        disputes = [{"id": "d1", "dispute": facts_summary[:300], "legal_nature": "both", "keywords": []}]
        _emit_step("Searching legal database...", "🔎")
    elif interactive_fast_path:
        disputes = [_build_single_dispute_from_facts(facts_summary)]
        _emit_step("Preparing a fast grounded opinion from local materials...", "⚡")
        _emit_live_token(
            "analysis_start",
            "Reviewing the local statutory provisions and the strongest precedents for your facts. ",
            words_per_chunk=2,
        )
    else:
        disputes = _build_disputes_from_intake_state(intake_state, facts_summary)
        labels = [d.get("dispute", "")[:60] for d in disputes]
        logger.info("Disputes identified: %s", labels)
        for _d in disputes[:6]:
            _did = _d.get("id", "?")
            _dtext = (_d.get("dispute") or "").strip()
            if len(_dtext) > 140:
                _dtext = _dtext[:137] + "..."
            _nature = _d.get("legal_nature", "both")
            _emit_step(f"[{_did}] Decomposition: {_dtext} (nature: {_nature})", "🧩")
        if len(disputes) > 6:
            _emit_step(
                f"...and {len(disputes) - 6} more decomposition item(s)",
                "🧩",
            )
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
    external_fallback_results = []
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
        _external_results: list = []
        _bare_hybrid_trace: list = []

        _emit_step(f"[{d_id}] Identifying legal provisions...", "📖")

        # 3a: Bare acts
        bare_d = []
        if retrieve_acts:
            if interactive_fast_path:
                from retrieval.retriever import search_bare_acts_fast, search_bare_acts_runtime
                queries_ba = (
                    _build_bare_act_queries(dispute, expanded_queries=legal_queries)
                    or [d_text or facts_summary[:300]]
                )[:_FAST_BARE_QUERY_LIMIT]
                seen_ba: dict = {}
                for _q_ba in queries_ba:
                    qt: list = []
                    found_for_query = False
                    for _ba in search_bare_acts_fast(_q_ba, top_k=_FAST_BARE_TOP_K, trace=qt):
                        _key = (
                            (_ba.get("act_name") or "").strip().lower(),
                            (_ba.get("section_number") or "").strip().lower(),
                        )
                        if _ba.get("_rerank_score", 0) > (seen_ba.get(_key) or {}).get("_rerank_score", -999):
                            seen_ba[_key] = _ba
                            found_for_query = True
                    _bare_hybrid_trace.append(
                        {"retrieval_query": _q_ba, "path": "interactive_fast", "hybrid_stages": qt}
                    )
                    if found_for_query:
                        break
                raw = list(seen_ba.values())
                bare_d = _select_interactive_grounded_results(
                    raw,
                    facts_summary,
                    quality_fn=_is_quality_bare_act,
                    cap=MAX_SECTIONS_PER_DISPUTE_TOTAL,
                )
                if not bare_d:
                    logger.info("[%s] Interactive fast bare-act rescue: broadening local search", d_id)
                    rescue_queries = _build_runtime_rescue_queries(
                        facts_summary, limit=2, bare_act_sections=bare_d, model_override=model_override
                    )
                    for _q_ba in rescue_queries:
                        qt2: list = []
                        for _ba in search_bare_acts_runtime(_q_ba, top_k=8, trace=qt2):
                            _key = (
                                (_ba.get("act_name") or "").strip().lower(),
                                (_ba.get("section_number") or "").strip().lower(),
                            )
                            if _ba.get("_rerank_score", 0) > (seen_ba.get(_key) or {}).get("_rerank_score", -999):
                                seen_ba[_key] = _ba
                        _bare_hybrid_trace.append(
                            {
                                "retrieval_query": _q_ba,
                                "path": "interactive_runtime_rescue",
                                "hybrid_stages": qt2,
                            }
                        )
                    raw = list(seen_ba.values())
                    bare_d = _select_interactive_grounded_results(
                        raw,
                        facts_summary,
                        quality_fn=_is_quality_bare_act,
                        cap=MAX_SECTIONS_PER_DISPUTE_TOTAL,
                    )
                if bare_d:
                    bare_d = _apply_fact_alignment_sort(
                        _filter_bare_acts_with_llm(dispute, bare_d, _local_debug),
                        facts_summary,
                        cap=MAX_SECTIONS_PER_DISPUTE_TOTAL,
                    )
                    top_acts = _select_top_acts(bare_d, max_acts=3)
                    if top_acts:
                        bare_d = _apply_fact_alignment_sort(
                            [ba for ba in bare_d if (ba.get("act_name") or "").strip() in top_acts],
                            facts_summary,
                            cap=MAX_SECTIONS_PER_DISPUTE_TOTAL,
                        )
            elif search_strategy == "local_only":
                from retrieval.retriever import search_bare_acts_auto
                # Multi-query: run all _build_bare_act_queries and merge by best score per section.
                queries_ba = _build_bare_act_queries(dispute, expanded_queries=legal_queries)
                seen_ba: dict = {}
                for _q_ba in queries_ba:
                    qt: list = []
                    for _ba in search_bare_acts_auto(_q_ba, top_k=30, trace=qt):
                        _key = (
                            (_ba.get("act_name") or "").strip().lower(),
                            (_ba.get("section_number") or "").strip().lower(),
                        )
                        if _ba.get("_rerank_score", 0) > (seen_ba.get(_key) or {}).get("_rerank_score", -999):
                            seen_ba[_key] = _ba
                    _bare_hybrid_trace.append(
                        {"retrieval_query": _q_ba, "path": "local_only_auto", "hybrid_stages": qt}
                    )
                raw = list(seen_ba.values())
                bare_d = [ba for ba in raw if ba.get("_rerank_score", 0) >= MIN_RERANK_SCORE and _is_quality_bare_act(ba)]
            else:
                bare_d = retrieve_bare_acts_for_dispute(
                    dispute,
                    facts_summary,
                    states=_states,
                    debug=_local_debug,
                    pending_indexing_list=_local_pending,
                    bare_act_trace=_bare_hybrid_trace,
                    expanded_legal_queries=legal_queries,
                )
                if not bare_d:
                    _emit_step(f"[{d_id}] No local bare act sections found — checking Indiankanoon fallback...", "🌐")
                    for _r in _web_search_bare_acts(dispute, facts_summary, states=_states, pending_indexing_list=_local_pending):
                        _rr = dict(_r)
                        _rr["_fallback_kind"] = "bare_act"
                        _external_results.append(_rr)

            if bare_d:
                act_names = list(dict.fromkeys(
                    ba.get("act_name") or ba.get("title", "").split(" §")[0]
                    for ba in bare_d if ba.get("act_name") or ba.get("title")
                ))[:4]
                acts_str = ", ".join(act_names) if act_names else "legal provisions"
                _emit_step(f"[{d_id}] Found {len(bare_d)} section{'s' if len(bare_d) != 1 else ''} — {acts_str}", "✅")
                if interactive_fast_path:
                    _emit_live_token(
                        "acts_found",
                        "I found grounded statutory provisions and I am checking the closest precedents against them. ",
                        words_per_chunk=2,
                    )
            else:
                _emit_step(f"[{d_id}] No local bare act sections found", "⚠️")

        # 3b: Case laws
        case_d = []
        if retrieve_case_laws_flag:
            _emit_step(f"[{d_id}] Searching for judicial precedents...", "⚖️")
            if interactive_fast_path:
                from retrieval.retriever import search_case_summaries_fast, search_case_laws_runtime
                _queries_cl = _build_fast_case_queries(dispute, bare_d, expanded_queries=legal_queries)
                seen_cl: dict = {}
                for _q_cl in _queries_cl:
                    for _cl in search_case_summaries_fast(_q_cl, top_k=_FAST_CASE_TOP_K):
                        _ck = _cl.get("_chunk_key") or (
                            (_cl.get("case_name") or "").lower() + "|" + (_cl.get("full_text", "")[:100])
                        )
                        if _cl.get("_rerank_score", 0) > (seen_cl.get(_ck) or {}).get("_rerank_score", -999):
                            seen_cl[_ck] = _cl
                raw_cl = list(seen_cl.values())
                initial_case = _select_interactive_grounded_results(
                    raw_cl,
                    facts_summary,
                    quality_fn=_is_quality_case_law,
                    cap=MAX_CASE_LAWS_PER_DISPUTE,
                    min_score=0.0,
                )
                case_d = _apply_dispute_case_law_limit(initial_case, num_sections=len(bare_d))
                if not case_d:
                    logger.info("[%s] Interactive fast case-law rescue: broadening local case search", d_id)
                    rescue_queries = _build_runtime_rescue_queries(
                        facts_summary, limit=2, bare_act_sections=bare_d, model_override=model_override
                    )
                    for _q_cl in rescue_queries:
                        for _cl in search_case_laws_runtime(_q_cl, top_k=8):
                            _ck = _cl.get("_chunk_key") or (
                                (_cl.get("case_name") or "").lower() + "|" + (_cl.get("full_text", "")[:100])
                            )
                            if _cl.get("_rerank_score", 0) > (seen_cl.get(_ck) or {}).get("_rerank_score", -999):
                                seen_cl[_ck] = _cl
                    raw_cl = list(seen_cl.values())
                    rescued_case = _select_interactive_grounded_results(
                        raw_cl,
                        facts_summary,
                        quality_fn=_is_quality_case_law,
                        cap=MAX_CASE_LAWS_PER_DISPUTE,
                    )
                    case_d = _apply_dispute_case_law_limit(rescued_case, num_sections=len(bare_d))
                if case_d:
                    case_d = _apply_dispute_case_law_limit(
                        _filter_case_laws_with_llm(dispute, bare_d, case_d, _local_debug),
                        num_sections=len(bare_d),
                    )
                case_d = _apply_fact_alignment_sort(
                    case_d,
                    facts_summary,
                    cap=max(len(case_d), MAX_CASE_LAWS_PER_DISPUTE),
                )
            elif search_strategy == "local_only":
                from retrieval.retriever import search_case_laws_auto
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
                if not case_d:
                    for _r in _web_search_case_laws(dispute, bare_d, facts_summary, states=_states, pending_indexing_list=_local_pending):
                        _rr = dict(_r)
                        _rr["_fallback_kind"] = "case_law"
                        _external_results.append(_rr)

            if case_d:
                case_cap = max(len(case_d), MAX_CASE_LAWS_PER_DISPUTE)
                case_d = _apply_fact_alignment_sort(case_d, facts_summary, cap=case_cap)
                if bare_d:
                    case_d = _align_case_laws_with_statute_support(case_d, bare_d, facts_summary, cap=case_cap)
                    case_d = _apply_dispute_case_law_limit(case_d, num_sections=len(bare_d))
                    bare_d = _align_bare_acts_with_case_support(
                        bare_d,
                        case_d,
                        facts_summary,
                        cap=MAX_SECTIONS_PER_DISPUTE_TOTAL,
                    )
                else:
                    case_d = case_d[:MAX_CASE_LAWS_PER_DISPUTE]

            # Citation graph expansion: boost recall by fetching chunks for cases
            # that the retrieved cases cite (forward) or that cite them (backward).
            # Runs after all retrieval paths so it always supplements the best results.
            if case_d and _ENABLE_CITATION_GRAPH_EXPANSION and not interactive_fast_path:
                try:
                    from retrieval.citations import (
                        expand_case_names_by_precedent,
                        get_chunks_by_case_names,
                    )
                    from retrieval.retriever import load_chunks as _hr_load_chunks
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
                if interactive_fast_path:
                    _emit_live_token(
                        "cases_found",
                        "I now have grounded precedent support as well and I am drafting the opinion. ",
                        words_per_chunk=2,
                    )
            else:
                _emit_step(f"[{d_id}] No judgements found in local database", "ℹ️")

        # Tag with dispute metadata before returning
        dispute_label = _short_dispute_label(d_text)
        for _ba in bare_d:
            _ba.setdefault("_dispute_id", d_id)
            _ba.setdefault("_dispute_label", dispute_label)
            _ba.setdefault("_dispute_text", d_text)
        for _cl in case_d:
            _cl.setdefault("_dispute_id", d_id)
            _cl.setdefault("_dispute_label", dispute_label)
            _cl.setdefault("_dispute_text", d_text)

        out_dr = {
            "dispute": dispute,
            "bare_acts": bare_d,
            "case_laws": case_d,
            "external_results": _external_results,
            "_pending": _local_pending,
            "_debug": _local_debug,
        }
        if _bare_hybrid_trace:
            out_dr["_bare_hybrid_trace"] = _bare_hybrid_trace
        return out_dr

    # Submit all disputes to the thread pool; collect as each completes.
    progress.start_group("Research", f"Retrieving bare acts and case laws for {len(disputes)} dispute(s)")
    _emit_progress()
    merged_bare_hybrid_traces: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=1 if interactive_fast_path else min(len(disputes), 2)) as executor:
        futures = {executor.submit(_retrieve_dispute, d): d for d in disputes}
        for fut in as_completed(futures):
            try:
                dr = fut.result()
                pending_indexing_list.extend(dr.pop("_pending", []))
                debug_pipeline["per_dispute"].update(dr.pop("_debug", {}))
                external_fallback_results.extend(dr.pop("external_results", []))
                _bt = dr.pop("_bare_hybrid_trace", None)
                if _bt:
                    merged_bare_hybrid_traces[str(dr.get("dispute", {}).get("id", "?"))] = _bt
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
    if merged_bare_hybrid_traces:
        progress.update_retrieval_diagnostics(
            {
                "bare_act_hybrid_trace": {
                    "kind": "bare_act_hybrid_trace",
                    "by_dispute": merged_bare_hybrid_traces,
                }
            }
        )
        _emit_progress()

    # Step 4: Aggregate + deduplicate across disputes
    all_bare_raw = []
    all_case_raw = []
    for dr in dispute_results:
        all_bare_raw.extend(dr["bare_acts"])
        all_case_raw.extend(dr["case_laws"])

    all_bare_raw = _deduplicate_bare_acts(all_bare_raw)
    all_case_raw = _deduplicate_case_laws(all_case_raw)
    _external_bare = [r for r in external_fallback_results if r.get("_fallback_kind") == "bare_act"]
    _external_case = [r for r in external_fallback_results if r.get("_fallback_kind") != "bare_act"]
    external_fallback_results = (
        _format_indiankanoon_fallback_results(_external_bare, kind="bare_act")
        + _format_indiankanoon_fallback_results(_external_case, kind="case_law")
    )

    if confirmed_materials:
        all_bare_raw, all_case_raw = _add_confirmed_materials(confirmed_materials, all_bare_raw, all_case_raw)

    # Step 5: Format sections, match case laws, and prepare the staged response
    progress.start_group("Response", "Formatting grounded materials for the next legal stage")
    if analysis_mode == "bare_acts_only":
        progress.add_step("Formatting applicable bare act sections...")
    elif analysis_mode == "precedents_only":
        progress.add_step("Formatting dispute-wise sections and precedents...")
    else:
        progress.add_step("Formatting sections and matching case laws to bare acts...")
    _emit_progress()

    all_case_raw_div = _diversify_case_laws_by_case(
        all_case_raw, max_chunks_per_case=3, max_total=max(len(all_bare_raw) * 4, 18),
    )
    formatted_bare = _format_bare_acts(all_bare_raw)
    if analysis_mode == "precedents_only":
        formatted_case = _format_case_law_excerpts(all_case_raw_div)
    elif retrieve_case_laws_flag:
        formatted_case = _format_case_laws(all_case_raw_div, user_query=facts_summary)
    else:
        formatted_case = []
    formatted_bare = _match_case_laws_to_bare_acts(formatted_bare, formatted_case)
    formatted_bare, _ = _filter_materials_to_local_db(formatted_bare, None)
    local_dispute_results = _filter_dispute_results_to_local_db(dispute_results)

    if intent == "search" and not formatted_bare and formatted_case:
        limit = max(1, result_count) if result_count else max(FLEXIBLE_MIN_FALLBACK, len(formatted_case))
        formatted_bare = [{
            "act_name": "Case laws",
            "title": "Case laws (search results)",
            "text": "",
            "related_case_laws": [cl for cl in formatted_case[:limit] if _is_local_db_source(cl)],
            "source_tag": "LOCAL_DB",
        }]

    # Step 6: Stage-specific response drafting
    if intent in ("search", "lookup"):
        progress.add_step("Generating search summary for retrieved materials...")
        _emit_step("Switching to summary generation for search/lookup results...", "🧠")
    elif analysis_mode == "bare_acts_only":
        progress.add_step("Preparing grounded bare-act-only analysis...")
        _emit_step("Switching to bare-act-only legal drafting...", "🧠")
    elif analysis_mode == "precedents_only":
        progress.add_step("Preparing precedent-focused analysis...")
        _emit_step("Switching to precedent-focused legal drafting...", "🧠")
    else:
        progress.add_step("Generating full legal opinion...")
        _emit_step("Switching to full legal-opinion drafting...", "🧠")
    _emit_progress()

    flattened_case_laws = []
    for ba in formatted_bare:
        flattened_case_laws.extend(ba.get("related_case_laws", []))
    if interactive_fast_path and not flattened_case_laws and formatted_case:
        flattened_case_laws = [cl for cl in formatted_case if _is_local_db_source(cl)]

    has_materials = bool(formatted_bare or flattened_case_laws)

    _ctx = (facts_summary or "")[:1500]
    _ctx += json.dumps([{"t": b.get("title", ""), "x": (b.get("text") or "")[:250]} for b in formatted_bare[:8]])[:1800]
    _ctx += json.dumps([{"t": c.get("title", ""), "x": (c.get("text") or "")[:220]} for c in flattened_case_laws[:8]])[:1800]
    _display, _switched = get_model_display_for_prompt(_ctx, explicit_model=model_override)
    if _display:
        _action = "Switching to" if _switched else "Calling"
        _msg = f"{_action} {_display} model"
        _gpu = get_gpu_info()
        if _gpu:
            _msg += f", GPU: {', '.join(_gpu)}"
        progress.add_step(_msg)
        _emit_progress()

    sufficiency = {
        "overall_sufficient": bool(formatted_bare),
        "gaps": [],
        "aspects": [],
        "confidence": "medium",
    }
    next_steps: list = []
    next_steps_summary: str = ""

    if not has_materials:
        fallback_stats = None
        if external_fallback_results:
            fallback_stats = {
                "web_found": len(external_fallback_results),
                "shortlisted": len(external_fallback_results),
                "proposed": 0,
                "already_in_library": 0,
            }
        explanation = _format_no_materials_message(search_strategy, fallback_stats)
        _stream_precomposed_text(explanation, token_callback=token_callback)
    else:
        if intent in ("search", "lookup"):
            explanation = _generate_conversational_summary(
                facts_summary, formatted_bare, flattened_case_laws, intent=intent,
                token_callback=token_callback, model_override=model_override,
            )
        elif analysis_mode == "bare_acts_only":
            explanation, formatted_bare, next_steps, next_steps_summary = _generate_bare_act_stage_text(
                facts_summary,
                local_dispute_results,
                formatted_bare,
                model_override=model_override,
                token_callback=token_callback,
            )
            flattened_case_laws = []
        elif analysis_mode == "precedents_only":
            explanation, formatted_bare, flattened_case_laws = _generate_precedent_stage_text(
                facts_summary,
                formatted_bare,
                formatted_case,
                model_override=model_override,
                token_callback=token_callback,
            )
        else:
            if interactive_fast_path:
                explanation = _generate_interactive_fast_opinion(
                    facts_summary,
                    formatted_bare,
                    formatted_case,
                    token_callback=token_callback,
                    model_override=model_override,
                )
            else:
                explanation = _generate_structured_opinion_by_dispute(
                    facts_summary, local_dispute_results, additional_info="",
                    token_callback=token_callback, model_override=model_override,
                )
                if not (explanation or "").strip():
                    explanation = _generate_legal_opinion(
                        facts_summary, formatted_bare, flattened_case_laws, sufficiency, model_override=model_override
                    )

        if not (explanation or "").strip():
            explanation = (
                "I reviewed the strongest local materials and summarized the key provisions and supporting precedents below."
            )
        if explanation and "I don't have any data" in explanation:
            explanation = _format_no_materials_message(search_strategy, None)
        if analysis_mode == "precedents_only":
            grounded_explanation = (explanation or "").strip()
        else:
            grounded_explanation = _enforce_local_grounding(explanation, formatted_bare)
        if grounded_explanation == _local_only_no_materials_message():
            if analysis_mode == "bare_acts_only" and formatted_bare:
                grounded_explanation = _build_grounded_bare_act_fallback(formatted_bare, facts_summary=facts_summary)
            elif analysis_mode == "precedents_only" and flattened_case_laws:
                grounded_explanation = _build_precedent_only_fallback(facts_summary, flattened_case_laws)
            elif formatted_bare or flattened_case_laws:
                grounded_explanation = _build_grounded_interactive_fallback(
                    facts_summary,
                    formatted_bare,
                    flattened_case_laws,
                )
        explanation = grounded_explanation

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

    try:
        from retrieval.retriever import _trace_label_chunk

        def _uniq_doc_labels(chunks: list) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for ch in chunks or []:
                if not isinstance(ch, dict):
                    continue
                lab = (_trace_label_chunk(ch) or "").strip()
                if not lab or lab == "chunk":
                    continue
                key = lab.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(lab)
                if len(out) >= 120:
                    break
            return out

        progress.update_retrieval_diagnostics(
            {
                "retrieved_documents": {
                    "bare_acts": _uniq_doc_labels(all_bare_raw),
                    "case_laws": _uniq_doc_labels(all_case_raw_div),
                }
            }
        )
        _emit_progress()
    except Exception as _rd_exc:
        logger.debug("Progress retrieved_documents failed: %s", _rd_exc)

    return {
        "bare_act_sections": formatted_bare,
        "case_laws": [],                      # case laws are nested under bare acts
        "explanation": explanation,
        "sufficiency": sufficiency,
        "sources_used": list(sources_used),
        "internet_case_laws": external_fallback_results,
        "progress": progress.get_progress(),
        "indexing_candidates": [],
        "next_steps": next_steps,
        "next_steps_summary": next_steps_summary,
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

    # ── Step 2: Text-based matching with proximity-weighted holding score ────
    def _holding_proximity_score(cl_text_lower: str, sec_num: str, window: int = 300) -> float:
        """
        Score how 'legally operative' the section-number mention is.
        Finds all positions where sec_num appears, then counts holding-signal
        words/phrases within ±window chars around each hit. Returns the max
        score across all positions, so a single strong hit is enough.
        """
        if not sec_num:
            return 0.0
        score = 0.0
        start = 0
        while True:
            pos = cl_text_lower.find(sec_num, start)
            if pos == -1:
                break
            window_text = cl_text_lower[max(0, pos - window): pos + len(sec_num) + window]
            hit_score = 0.0
            for phrase in _LEGAL_HOLDING_SIGNALS_PHRASE:
                if phrase in window_text:
                    hit_score += 0.35
            for word in _LEGAL_HOLDING_SIGNALS_WORD:
                if word in window_text:
                    hit_score += 0.15
            score = max(score, hit_score)
            start = pos + 1
        return score

    def _text_match_group(ba_group: list, cl_group: list) -> list:
        """
        Assign case laws in cl_group to sections in ba_group; return unmatched.

        Matching logic (in priority order):
        1. Section number must appear in case law text AND at least one act keyword
           matches — same as before.
        2. Among all qualifying bare act sections for a case law, prefer the one
           where the section-number mention is closest to holding-signal language
           (_holding_proximity_score). This avoids attaching a case law to a section
           it merely mentioned in narration vs. one it actually decided.
        3. Very low-scored case laws (below INTERACTIVE_MIN_RERANK_SCORE) are skipped
           in the unmatched fallback — don't pollute sections with noise.
        """
        unmatched = []
        for cl in cl_group:
            # Use para_texts[0] if available (full raw para, not collapsed summary)
            para_texts = cl.get("para_texts") or []
            raw_text = para_texts[0] if para_texts else (cl.get("text") or cl.get("full_text") or "")
            cl_text = (
                raw_text + " " +
                (cl.get("case_name") or cl.get("title") or "")
            ).lower()

            # Find all qualifying bare act sections for this case law
            candidates = []
            for ba in ba_group:
                if len(ba.get("related_case_laws", [])) >= max_per_section:
                    continue
                sec_num = str(ba.get("section_number") or "").strip()
                act_keywords = [
                    w for w in (ba.get("act_name") or "").lower().split()
                    if len(w) > 3 and w not in ("the", "and", "of", "for", "act,", "act")
                ]
                if sec_num and sec_num in cl_text:
                    if not act_keywords or any(w in cl_text for w in act_keywords[:3]):
                        proximity = _holding_proximity_score(cl_text, sec_num)
                        candidates.append((proximity, ba.get("_rerank_score", 0), ba))

            if candidates:
                # Pick the section with the highest proximity score; break ties by rerank_score
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_ba = candidates[0][2]
                best_ba["related_case_laws"].append(cl)
            else:
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

        # Fallback: assign unmatched to the highest-scored section in this group.
        # Only attach case laws that clear INTERACTIVE_MIN_RERANK_SCORE — do not
        # let very low-scored (noise) case laws pollute sections they didn't match.
        if unmatched and ba_group:
            top_ba = ba_group[0]
            for cl in unmatched:
                if cl.get("_rerank_score", 0) < INTERACTIVE_MIN_RERANK_SCORE:
                    continue
                if len(top_ba.get("related_case_laws", [])) < max_per_section:
                    top_ba["related_case_laws"].append(cl)

    # ── Step 4: Global fallback for orphaned case laws ───────────────────────
    if orphan_case_laws:
        remaining = _text_match_group(bare_acts, orphan_case_laws)
        # Anything still unmatched goes to the overall top section (score-filtered)
        if remaining and bare_acts:
            top_ba = bare_acts[0]
            for cl in remaining:
                if cl.get("_rerank_score", 0) < INTERACTIVE_MIN_RERANK_SCORE:
                    continue
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
            "section_title": title,
            "title": display_title,
            "url": url,
            "source_tag": ba.get("source_tag", "LOCAL_DB"),
            "_rerank_score": ba.get("_rerank_score", 0),
            "_sort_score": _best_alignment_score(ba),
            # Preserve dispute tag so _match_case_laws_to_bare_acts() can group
            # by dispute instead of re-scoring pairs with the cross-encoder.
            "_dispute_id": ba.get("_dispute_id", ""),
            "_dispute_label": ba.get("_dispute_label", ""),
            "_dispute_text": ba.get("_dispute_text", ""),
        })

    # Sort by the strongest aligned score available, falling back to rerank score.
    formatted.sort(key=lambda x: (x.get("_sort_score", 0), x.get("_rerank_score", 0)), reverse=True)
    return formatted


def _extract_verbatim_display_excerpt(text: str, max_lines: int = 4, max_chars: int = 650, focus_terms: list[str] | None = None) -> str:
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if len(lines) >= 2:
        units = lines
        joiner = "\n"
    else:
        units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        joiner = " "
    if not units:
        return raw[:max_chars].strip()

    scored = []
    for idx, unit in enumerate(units):
        compact = unit.strip()
        if not compact:
            continue
        score = _stage_text_priority_score(compact, extra_terms=focus_terms)
        if compact.lower().startswith("section "):
            score += 0.12
        if len(compact) >= 48:
            score += 0.03
        scored.append((score, idx, compact))

    if scored and max(item[0] for item in scored) > 0:
        chosen = sorted(
            sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[:max_lines],
            key=lambda item: item[1],
        )
        excerpt = joiner.join(item[2] for item in chosen)
    else:
        excerpt = joiner.join(units[:max_lines])

    excerpt = excerpt[:max_chars].strip()
    return excerpt or raw[:max_chars].strip()


_STAGE_PROTECTION_PRIORITY_TERMS = (
    "urgent", "immediate", "interim", "protection", "protect", "protective", "relief", "restrain",
    "prohibit", "injunction", "residence", "reside", "shelter", "medical", "hospital", "complaint",
    "police", "custody", "maintenance", "compensation", "damages", "access", "return", "restore",
    "liberty", "rights", "fundamental", "safety", "violence", "abuse", "harassment", "assistance",
    "order", "magistrate", "court", "support", "aid", "monetary", "injury", "proof", "evidence",
    "document", "record", "witness", "notice", "summons", "investigation", "jurisdiction", "injunction",
)

_STAGE_PROCEDURAL_DOWNWEIGHT_TERMS = (
    "short title", "commencement", "extent", "definition", "definitions", "form", "forms", "schedule",
    "annexure", "appendix", "instruction", "instructions", "application under", "prescribed form",
    "fees", "fee", "format", "register", "registers",
)

# Holding/ratio signals — multi-word phrases that mark operative legal text in Indian SC/HC judgments.
# Multi-word phrases score higher than single-word signals to reduce false positives.
_LEGAL_HOLDING_SIGNALS_PHRASE = (
    "we hold", "it is held", "the court holds", "accordingly held", "it is therefore held",
    "it is directed", "we direct", "the court directs", "it is hereby directed",
    "it is observed", "we observe", "the court observed", "the court notes",
    "ratio decidendi", "the ratio", "the legal principle",
    "is entitled to", "shall be entitled", "are entitled to",
    "for the foregoing reasons", "in the result", "in the circumstances",
    "the appeal is allowed", "the appeal is dismissed", "the petition is allowed",
    "the petition is dismissed", "is set aside", "is quashed", "are quashed",
    "it is well settled", "settled law", "settled position", "the law is well settled",
    "must be read", "ought to be", "is bound to",
    "we allow", "we dismiss", "we set aside", "we quash",
    "accordingly ordered", "accordingly directed",
    "the impugned", "impugned order", "impugned judgment",
    "this court holds", "this court finds", "this court directs",
    "the high court erred", "the tribunal erred",
    "the accused is", "the respondent is", "the appellant is",
    "conviction is", "sentence is", "bail is",
    "interim relief", "stay granted", "stay refused",
)

_LEGAL_HOLDING_SIGNALS_WORD = (
    "accordingly", "consequently", "therefore", "thus", "hence",
    "held", "directed", "ordered", "decreed", "allowed", "dismissed",
    "entitled", "liable", "convicted", "acquitted",
)


def _stage_text_priority_score(text: str, extra_terms: list[str] | None = None) -> float:
    low = " ".join((text or "").lower().split())
    if not low:
        return 0.0
    score = 0.0
    protection_hits = 0
    procedural_hits = 0
    holding_hits = 0
    for term in _STAGE_PROTECTION_PRIORITY_TERMS:
        if term in low:
            protection_hits += 1
            score += 0.12 if " " in term else 0.05
    for term in _STAGE_PROCEDURAL_DOWNWEIGHT_TERMS:
        if term in low:
            procedural_hits += 1
            score -= 0.2 if " " in term else 0.08
    if procedural_hits and not protection_hits:
        score -= 0.55 + min(procedural_hits * 0.08, 0.32)
    # Holding signal boost — multi-word phrases score higher than single words
    for phrase in _LEGAL_HOLDING_SIGNALS_PHRASE:
        if phrase in low:
            holding_hits += 1
            score += 0.35
    for word in _LEGAL_HOLDING_SIGNALS_WORD:
        if word in low:
            holding_hits += 1
            score += 0.15
    for term in (extra_terms or []):
        t = str(term or "").strip().lower()
        if t and t in low:
            score += 0.06
    return score


def _extract_holding_lines(text: str, max_lines: int = 5) -> str:
    """
    From a case law paragraph (or bare act section), extract the most legally
    operative lines using holding-signal scoring.

    Rules:
    - If the text has ≤ 8 lines: return it unchanged (no extraction needed).
    - Otherwise: score each line, pick the top `max_lines` by score,
      re-sort them into their original reading order, and join with ' ... '.
    - No character cap is applied — lines are returned as-is.
    - Sentence splits are used when natural line breaks are absent.
    """
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""

    # Split into units (prefer natural line breaks, fall back to sentences)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if len(lines) < 2:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]

    # Short text — return as-is
    if len(lines) <= 8:
        return raw

    # Score each line; track original index to restore reading order
    scored = []
    for idx, line in enumerate(lines):
        if not line:
            continue
        s = _stage_text_priority_score(line)
        scored.append((s, idx, line))

    # Take top max_lines, restore original order
    top = sorted(scored, key=lambda x: -x[0])[:max_lines]
    top_in_order = sorted(top, key=lambda x: x[1])

    # If all scores are the same (no signal found), just return first max_lines lines
    if len(set(round(x[0], 4) for x in top_in_order)) == 1:
        return "\n".join(lines[:max_lines])

    return " ... ".join(x[2] for x in top_in_order)


def _stage_material_impact_score(item: dict, focus_terms: list[str] | None = None) -> float:
    if not isinstance(item, dict):
        return 0.0
    base = float(item.get("_sort_score") or item.get("_rerank_score") or 0.0)
    text = " ".join(
        str(item.get(key) or "")
        for key in ("act_name", "section_title", "title", "text", "full_text")
    )
    return base + _stage_text_priority_score(text, extra_terms=focus_terms)


def _prepare_bare_act_stage_entries(
    formatted_bare: list,
    facts_summary: str = "",
    max_acts_per_dispute: int = 3,
    max_sections_per_act: int = 4,
) -> list:
    prepared_rows = []
    focus_terms = _extract_salient_terms(_build_focus_fact_text(facts_summary, max_chars=420), limit=10)
    for ba in formatted_bare or []:
        clone = dict(ba)
        # Keep full section text for both LLM reasoning and UI display.
        # No truncation — the section/subsection text is shown in its entirety.
        clone["is_verbatim_excerpt"] = True
        clone["_stage_score"] = _stage_material_impact_score(clone, focus_terms=focus_terms)
        prepared_rows.append(clone)

    grouped: dict[str, dict[str, dict]] = {}
    for item in prepared_rows:
        dispute_id = str(item.get("_dispute_id") or "general").strip() or "general"
        act_name = str(item.get("act_name") or item.get("source") or item.get("title") or "Applicable Act").strip() or "Applicable Act"
        act_key = act_name.lower()
        dispute_bucket = grouped.setdefault(dispute_id, {})
        act_bucket = dispute_bucket.setdefault(
            act_key,
            {"act_name": act_name, "score": float("-inf"), "items": []},
        )
        act_bucket["items"].append(item)
        act_bucket["score"] = max(act_bucket["score"], item.get("_stage_score", 0))

    selected: list[dict] = []
    for _dispute_id, act_map in grouped.items():
        ranked_acts = sorted(act_map.values(), key=lambda bucket: bucket.get("score", 0), reverse=True)[:max_acts_per_dispute]
        for act_bucket in ranked_acts:
            act_items = sorted(
                act_bucket.get("items", []),
                key=lambda item: (item.get("_stage_score", 0), item.get("_sort_score", 0), item.get("_rerank_score", 0)),
                reverse=True,
            )
            kept = []
            seen_sections: set[tuple[str, str]] = set()
            for item in act_items:
                section_key = (
                    str(item.get("section_number") or "").strip().lower(),
                    str(item.get("title") or "").strip().lower(),
                )
                if section_key in seen_sections:
                    continue
                if item.get("_stage_score", 0) < 0 and kept:
                    continue
                seen_sections.add(section_key)
                kept.append(item)
                if len(kept) >= max_sections_per_act:
                    break
            selected.extend(kept)

    selected.sort(
        key=lambda item: (
            str(item.get("_dispute_id") or "general"),
            -(item.get("_stage_score", 0)),
            -(item.get("_sort_score", 0)),
            -(item.get("_rerank_score", 0)),
        )
    )
    for item in selected:
        item.pop("_stage_score", None)
    return selected


def _format_case_law_excerpts(case_laws: list) -> list:
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
        groups.setdefault(group_key, []).append({
            "cl": cl,
            "text": text,
            "score": cl.get("_rerank_score", 0),
        })

    formatted = []
    for group_key, chunks in groups.items():
        case_name, court, year, _ = group_key
        chunks.sort(key=lambda x: -x["score"])
        first = chunks[0]["cl"]
        citation = first.get("citation", "")
        fallback_title = case_name or first.get("source", "Unknown")
        if court:
            fallback_title += f" ({court}"
            if year:
                fallback_title += f", {year}"
            fallback_title += ")"
        elif citation:
            fallback_title += f" [{citation}]"

        url = first.get("url") or first.get("source_url") or ""
        if not url:
            src_file = first.get("source_file") or first.get("source") or ""
            if src_file:
                file_path = Path(CASELAW_DIR) / src_file
                url = file_path.as_uri()
            if not url:
                url = GOOGLE_DRIVE_CASE_LAWS_FOLDER_URL

        excerpt = _extract_verbatim_display_excerpt(chunks[0]["text"], max_lines=5, max_chars=760)
        para_num = first.get("paragraph_num")
        para_id = first.get("paragraph_id")
        formatted.append({
            "source": first.get("source_file", first.get("source", "")),
            "text": excerpt,
            "case_name": case_name,
            "title": fallback_title,
            "court": court,
            "year": year,
            "citation": citation,
            "binding_authority": first.get("binding_authority", ""),
            "url": url,
            "source_tag": first.get("source_tag", "LOCAL_DB"),
            "signature": (first.get("signature") or "").strip() or None,
            "paragraph_num": para_num,
            "paragraph_id": para_id,
            "_rerank_score": chunks[0]["score"],
            "_sort_score": _best_alignment_score(first),
            "_year": _case_year_for_sort(first),
            "_dispute_id": first.get("_dispute_id", ""),
            "_dispute_label": first.get("_dispute_label", ""),
            "_dispute_text": first.get("_dispute_text", ""),
            "is_verbatim_excerpt": True,
        })

    authority_order = {"supreme_court": 0, "high_court": 1, "tribunal": 2, "district_court": 3}
    formatted.sort(
        key=lambda x: (
            authority_order.get(x.get("binding_authority", ""), 5),
            0 if x.get("_year", 0) >= MIN_CASE_YEAR else 1,
            -x.get("_sort_score", x.get("_rerank_score", 0)),
        )
    )
    for x in formatted:
        x.pop("_year", None)
    return formatted



def _build_bare_act_stage_dispute_blocks(dispute_results: list) -> str:
    blocks = []
    for dr in dispute_results or []:
        dispute = dr.get("dispute", {}) or {}
        d_id = dispute.get("id", "?")
        d_text = (dispute.get("dispute") or "").strip()
        d_label = _short_dispute_label(d_text)
        blocks.append(f"DISPUTE [{d_id}] {d_label}: {d_text}")
        bare = _prepare_bare_act_stage_entries(_format_bare_acts(dr.get("bare_acts") or []))
        if not bare:
            blocks.append("(No local bare act sections were retrieved for this dispute.)")
            continue
        for ba in bare[:8]:
            label = ba.get("title") or ba.get("act_name") or "Bare Act"
            # Pass full section text to LLM — no truncation.
            excerpt = (ba.get("text") or "").strip()
            blocks.append(f"- {label}: {excerpt}")
        blocks.append("")
    return "\n".join(blocks).strip()



def _build_precedent_stage_compact_context(linked: list) -> str:
    """Precedent-only cues for the LLM: no dispute headers, no bare-act anchors.

    Uses _extract_holding_lines on para_texts (the preserved list of top-3 raw
    paragraphs from _format_case_laws) so the LLM sees operative holding lines,
    not a truncated or collapsed summary. For short paragraphs, the full text
    is passed through unchanged.
    """
    lines: list[str] = []
    n = 0
    for ba in linked or []:
        for cl in ba.get("related_case_laws") or []:
            n += 1
            title = str(cl.get("title") or cl.get("case_name") or "Judgment").strip()
            lines.append(f"[{n}] {title}")

            # Prefer para_texts (list of raw paras preserved from _format_case_laws).
            # Use the highest-scored para (index 0 — already sorted by score desc).
            # Fall back to cl["text"] if para_texts is absent (web fallback rows etc.)
            para_texts = cl.get("para_texts") or []
            if para_texts:
                best_para = para_texts[0]
            else:
                best_para = cl.get("text") or ""

            if best_para:
                cue = _extract_holding_lines(best_para, max_lines=5)
                if cue:
                    lines.append(f"    Holding/operative paragraph: {cue}")

    return "\n".join(lines).strip() or "(No retrieved precedent paragraphs.)"


def _build_precedent_only_fallback(facts_summary: str, case_rows: list) -> str:
    """Grounding fallback for precedents_only: no statutory repetition."""
    rows = [c for c in (case_rows or []) if isinstance(c, dict)]
    if not rows:
        return _local_only_no_materials_message()
    fact_line = _build_focus_fact_text(facts_summary, max_chars=260) or "the facts shared"
    lead = (
        f"Based on the retrieved judgments, here is how the closest local precedents bear on {fact_line}. "
        "The judgment extracts are listed below for your review."
    )
    parts: list[str] = []
    for cl in rows[:5]:
        name = (cl.get("title") or cl.get("case_name") or "Judgment").strip()
        text = " ".join((cl.get("text") or "").split()).strip()[:220]
        if text:
            parts.append(f"{name} highlights, on this record, that {text}")
        else:
            parts.append(f"{name} is among the strongest locally retrieved precedents here.")
    return "\n\n".join([lead, " ".join(parts)]).strip()



def _build_stage_retained_sections_text(stage_bare: list) -> str:
    grouped: dict[str, dict] = {}
    for item in stage_bare or []:
        dispute_id = str(item.get("_dispute_id") or "general").strip() or "general"
        bucket = grouped.setdefault(
            dispute_id,
            {
                "label": str(item.get("_dispute_label") or "").strip(),
                "text": str(item.get("_dispute_text") or "").strip(),
                "items": [],
            },
        )
        bucket["items"].append(item)

    blocks: list[str] = []
    for dispute_id, bucket in grouped.items():
        dispute_text = bucket.get("text") or ""
        dispute_label = bucket.get("label") or _short_dispute_label(dispute_text)
        blocks.append(f"DISPUTE [{dispute_id}] {dispute_label}: {dispute_text}".strip())
        for item in bucket.get("items", [])[:12]:
            act_name = str(item.get("act_name") or item.get("source") or "Applicable Act").strip()
            section_number = str(item.get("section_number") or "").strip()
            section_title = str(item.get("section_title") or "").strip()
            explanation = str(item.get("explanation") or "").strip()
            excerpt = " ".join(str(item.get("text") or "").split())[:380]
            label = act_name
            if section_number:
                label += f" - Section {section_number}"
            if section_title:
                label += f" ({section_title})"
            line = f"- {label}"
            if explanation:
                line += f": {explanation}"
            elif excerpt:
                line += f": {excerpt}"
            blocks.append(line)
            if explanation and excerpt:
                blocks.append(f"  Excerpt: {excerpt}")
        blocks.append("")
    return "\n".join(blocks).strip()



def _merge_next_step_granular_fields(item: dict) -> str:
    parts: list[str] = []
    for key in ("what_to_do", "precautions", "why_it_helps", "challenge_to_watch", "legal_protection"):
        t = str(item.get(key) or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def _flatten_structured_next_steps_to_paragraph(steps: list | None) -> str:
    """Plain-text fallback from structured next steps (e.g. for legacy clients)."""
    if not steps:
        return ""
    chunks: list[str] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not summary:
            summary = _merge_next_step_granular_fields(item)
        if title and summary:
            chunks.append(f"{title} {summary}".strip())
        elif title:
            chunks.append(title)
        elif summary:
            chunks.append(summary)
    return " ".join(chunks).strip()


def _derive_next_steps_summary(
    parsed: dict | None,
    steps: list,
    repair_summary: str = "",
) -> str:
    if isinstance(parsed, dict):
        s = str(parsed.get("next_steps_summary") or "").strip()
        if s:
            return s
    rs = (repair_summary or "").strip()
    if rs:
        return rs
    return _flatten_structured_next_steps_to_paragraph(steps)


def _normalize_stage_next_steps(items: list) -> list[dict]:
    """Each item: title + summary (preferred), or legacy granular fields merged into summary."""
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        what_to_do = str(item.get("what_to_do") or "").strip()
        precautions = str(item.get("precautions") or "").strip()
        why_it_helps = str(item.get("why_it_helps") or "").strip()
        challenge_to_watch = str(item.get("challenge_to_watch") or "").strip()
        legal_protection = str(item.get("legal_protection") or "").strip()

        if summary and len(title) >= 4:
            dedupe_key = (title.lower(), summary[:160].lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append({
                "title": title,
                "summary": summary,
                "what_to_do": "",
                "precautions": "",
                "why_it_helps": "",
                "challenge_to_watch": "",
                "legal_protection": "",
            })
            continue

        if len(title) < 6:
            continue
        if not any([what_to_do, precautions, why_it_helps, challenge_to_watch, legal_protection]):
            continue
        merged = _merge_next_step_granular_fields(item)
        dedupe_key = (title.lower(), merged[:160].lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append({
            "title": title,
            "summary": merged,
            "what_to_do": what_to_do,
            "precautions": precautions,
            "why_it_helps": why_it_helps,
            "challenge_to_watch": challenge_to_watch,
            "legal_protection": legal_protection,
        })
    return normalized[:12]



def _build_bare_act_next_steps_fallback(stage_bare: list, facts_summary: str = "") -> list:
    ranked_items = sorted(
        [item for item in (stage_bare or []) if isinstance(item, dict)],
        key=lambda item: (
            float(item.get("_sort_score") or item.get("_rerank_score") or 0.0),
            len(str(item.get("explanation") or item.get("text") or "")),
        ),
        reverse=True,
    )[:6]
    if not ranked_items:
        return []

    focus_terms = _extract_salient_terms(_build_focus_fact_text(facts_summary, max_chars=360), limit=8)
    act_names = list(
        dict.fromkeys(
            str(item.get("act_name") or "Applicable Act").strip()
            for item in ranked_items
            if str(item.get("act_name") or "").strip()
        )
    )
    section_labels = [
        label for label in [
            (
                f"Section {str(item.get('section_number') or '').strip()}"
                + (f" of {str(item.get('act_name') or '').strip()}" if str(item.get('act_name') or '').strip() else "")
            ).strip()
            for item in ranked_items
        ]
        if label
    ]
    explanation_snippets = [
        str(item.get("explanation") or "").strip()
        for item in ranked_items
        if str(item.get("explanation") or "").strip()
    ]
    dispute_labels = list(
        dict.fromkeys(
            str(item.get("_dispute_label") or "").strip()
            for item in ranked_items
            if str(item.get("_dispute_label") or "").strip()
        )
    )

    act_phrase = ", ".join(act_names[:3]) if act_names else "the retrieved Acts"
    section_phrase = ", ".join(section_labels[:4]) if section_labels else "the strongest retained sections"
    focus_hint = ", ".join(focus_terms[:3]) if focus_terms else ", ".join(dispute_labels[:2]) or "the present case"
    precaution_focus = ", ".join(focus_terms[:4]) if focus_terms else "documents, messages, records, and timelines"

    s1_body = " ".join(
        [
            f"Bring together the records that best connect your facts to {section_phrase} under {act_phrase}, keep the chronology clear, and organise the materials likely to matter in the first authority or court-facing step.",
            f"Keep the supporting record on {precaution_focus} intact, avoid altering originals, and preserve the timeline exactly as it presently stands.",
            explanation_snippets[0] if explanation_snippets else f"The retained sections are most useful when the facts, timing, and supporting record line up clearly on the present case.",
            f"The other side may dispute the factual sequence, urgency, or sufficiency of the current record relating to {focus_hint}.",
            f"The retained provisions under {act_phrase} provide the clearest presently available statutory footing for the next formal step if the record is presented carefully.",
        ]
    )
    s2_body = " ".join(
        [
            f"Use the retained sections to identify the first filing, complaint, application, police-facing step, magistrate-facing step, or court-facing step that the present record can responsibly support, and move promptly where delay may weaken urgency.",
            "Before filing, check that the immediate relief sought matches the current facts and that the most relevant documents for jurisdiction, notice, injury, possession, payment, communication, or proof are ready.",
            explanation_snippets[1] if len(explanation_snippets) > 1 else f"The strongest retained sections appear to support not just the substantive right but also the immediate route to seek protection or relief.",
            "The first forum or authority may examine urgency, maintainability, and whether the relief sought is broader than the present record can presently justify.",
            f"The retained statutory scheme helps frame both the immediate remedy and the procedural route, especially where the present facts already engage {section_phrase}.",
        ]
    )
    steps = [
        {"title": "Preserve and organise the strongest supporting record", "summary": s1_body},
        {"title": "Take the first procedural step without delay", "summary": s2_body},
    ]
    return _normalize_stage_next_steps(steps)


def _repair_bare_act_next_steps(
    facts_summary: str, stage_bare: list, model_override: str | None = None
) -> tuple[list, str]:
    if not stage_bare:
        return [], ""
    retained_sections_text = _build_stage_retained_sections_text(stage_bare)
    prompt = BARE_ACT_NEXT_STEPS_REPAIR_PROMPT.format(
        facts_summary=(facts_summary or "").strip()[:1400],
        retained_sections_text=retained_sections_text[:9000],
    )
    try:
        if model_override:
            raw = ask_llm(prompt, model=model_override)
        else:
            raw = ask_llm(prompt, task_hint="fast")
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("No valid JSON returned for bare-act next-step repair")
        summary = str(parsed.get("next_steps_summary") or "").strip()
        steps = _normalize_stage_next_steps(parsed.get("next_steps") or [])
        return steps, summary
    except Exception as exc:
        logger.warning("Bare-act next-step repair failed: %s", exc)
        return [], ""



_BARE_ACT_LLM_MAX_ATTEMPTS = 10   # max retries for transient errors (timeouts / connectivity)
_BARE_ACT_PARSE_RETRY_MAX  = 3    # max retries for JSON parse / schema failures


def _is_llm_transient_error(exc: Exception) -> bool:
    """True when the error is likely a timeout or connectivity issue worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ("timeout", "timed out", "unreachable", "connection", "connect"))


# Appended to the prompt on parse-failure retries so the model understands why it is being retried.
_BARE_ACT_JSON_REPAIR_HINT = """

IMPORTANT — your previous response could not be parsed as valid JSON.
Return ONLY a single valid JSON object with no prose, no markdown, no code fences, no explanation.
The object must have exactly these top-level keys:
  "summary_text"         — string, 3-6 sentences of flowing prose
  "section_explanations" — array of {act_name, section_number, explanation}
  "next_steps"           — array of {title, summary}
  "next_steps_summary"   — string
Start your response with { and end with }. Nothing before or after the JSON object."""


def _generate_bare_act_stage_text(
    facts_summary: str,
    dispute_results: list,
    formatted_bare: list,
    model_override: str | None = None,
    token_callback=None,
) -> tuple[str, list, list, str]:
    """
    Generate the bare-act stage summary, per-section explanations, and next steps.

    Retries the LLM call up to _BARE_ACT_LLM_MAX_ATTEMPTS times total.
    On a transient failure (timeout / unreachable) a short message is streamed
    to the UI so the user knows what is happening.  After all attempts are
    exhausted the deterministic fallback is returned instead.
    """
    if not formatted_bare:
        return _local_only_no_materials_message(), [], [], ""

    stage_bare = _prepare_bare_act_stage_entries(formatted_bare, facts_summary=facts_summary)
    dispute_blocks_text = _build_bare_act_stage_dispute_blocks(dispute_results)
    prompt = BARE_ACT_STAGE_SUMMARY_PROMPT.format(
        facts_summary=(facts_summary or "").strip()[:1400],
        dispute_blocks_text=dispute_blocks_text[:9000],
    )

    last_exc: Exception | None = None
    transient_attempts = 0   # how many times we retried due to timeout/connectivity
    parse_attempts     = 0   # how many times we retried due to JSON parse / schema failure
    current_prompt     = prompt  # may grow a JSON repair hint on parse failures

    while True:
        try:
            raw = ask_llm(current_prompt, model=model_override)
            parsed = _extract_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("No valid JSON returned for bare-act stage")

            explanation_map = {
                (
                    str(item.get("act_name") or "").strip().lower(),
                    str(item.get("section_number") or "").strip(),
                ): str(item.get("explanation") or "").strip()
                for item in (parsed.get("section_explanations") or [])
                if isinstance(item, dict)
            }
            enriched = []
            for ba in stage_bare:
                key = (
                    str(ba.get("act_name") or "").strip().lower(),
                    str(ba.get("section_number") or "").strip(),
                )
                clone = dict(ba)
                clone["explanation"] = explanation_map.get(key, "")
                enriched.append(clone)

            if not any(str(item.get("explanation") or "").strip() for item in enriched):
                try:
                    repaired = _explain_sections_and_get_followup(
                        (facts_summary or "").strip()[:1200],
                        [dict(item) for item in stage_bare],
                    )
                    repaired_map = {
                        (
                            str(item.get("act_name") or "").strip().lower(),
                            str(item.get("section_number") or "").strip(),
                        ): str(item.get("explanation") or "").strip()
                        for item in (repaired.get("bare_acts") or [])
                        if isinstance(item, dict)
                    }
                    for clone in enriched:
                        if str(clone.get("explanation") or "").strip():
                            continue
                        key = (
                            str(clone.get("act_name") or "").strip().lower(),
                            str(clone.get("section_number") or "").strip(),
                        )
                        clone["explanation"] = repaired_map.get(key, "")
                except Exception as repair_exc:
                    logger.warning("Bare-act stage explanation repair failed: %s", repair_exc)

            repair_summary = ""
            next_steps = _normalize_stage_next_steps(parsed.get("next_steps") or [])
            if not next_steps:
                next_steps, repair_summary = _repair_bare_act_next_steps(
                    facts_summary, enriched, model_override=model_override
                )
            if not next_steps:
                next_steps = _build_bare_act_next_steps_fallback(enriched, facts_summary=facts_summary)
            next_steps_summary = _derive_next_steps_summary(parsed, next_steps, repair_summary=repair_summary)
            summary_text = str(parsed.get("summary_text") or "").strip()
            if not summary_text:
                raise ValueError("Empty bare-act stage summary")

            _stream_precomposed_text(summary_text, token_callback=token_callback)
            return summary_text, enriched, next_steps, next_steps_summary

        except Exception as exc:
            last_exc = exc
            is_transient = _is_llm_transient_error(exc)

            if is_transient:
                # ── Timeout / connectivity issue ─────────────────────────────
                transient_attempts += 1
                logger.warning(
                    "Bare-act stage LLM attempt %d/%d failed (transient): %s",
                    transient_attempts, _BARE_ACT_LLM_MAX_ATTEMPTS, exc,
                )
                if transient_attempts >= _BARE_ACT_LLM_MAX_ATTEMPTS:
                    break  # exhausted transient budget → fall through to deterministic fallback
                # Tell the user the model is slow — they can see the wait is real
                retry_msg = (
                    f"The model is taking longer than expected "
                    f"(attempt {transient_attempts} of {_BARE_ACT_LLM_MAX_ATTEMPTS} did not complete). "
                    f"Retrying now\u2026 "
                )
                _stream_precomposed_text(retry_msg, token_callback=token_callback)
                # Retry with the same prompt — connectivity may recover
                current_prompt = prompt

            else:
                # ── Parse / schema failure — model responded but output was wrong ─
                parse_attempts += 1
                logger.warning(
                    "Bare-act stage LLM parse failure %d/%d: %s",
                    parse_attempts, _BARE_ACT_PARSE_RETRY_MAX, exc,
                )
                if parse_attempts >= _BARE_ACT_PARSE_RETRY_MAX:
                    break  # exhausted parse budget → fall through to deterministic fallback
                # Silent retry — user does not need to see JSON format errors.
                # Append the repair hint so the model understands what went wrong.
                current_prompt = prompt + _BARE_ACT_JSON_REPAIR_HINT

    # All attempts exhausted — surface the final error and return the deterministic fallback.
    # Only show a UI message for transient failures (model was unreachable).
    # Parse failures are silent from the user's perspective — just show the fallback content.
    logger.error(
        "Bare-act stage summary generation failed (transient_attempts=%d, parse_attempts=%d): %s",
        transient_attempts, parse_attempts, last_exc,
    )
    if transient_attempts > 0:
        # Model was genuinely unreachable — tell the user
        error_notice = (
            "The model did not respond in time after multiple attempts. "
            "Showing the grounded statutory summary from the local database instead.\n\n"
        )
        _stream_precomposed_text(error_notice, token_callback=token_callback)
    # For parse failures only: fall through silently — the fallback content is streamed below
    fallback = _build_grounded_bare_act_fallback(stage_bare, facts_summary=facts_summary)
    next_steps, repair_summary = _repair_bare_act_next_steps(
        facts_summary, stage_bare, model_override=model_override
    )
    if not next_steps:
        next_steps = _build_bare_act_next_steps_fallback(stage_bare, facts_summary=facts_summary)
    next_steps_summary = _derive_next_steps_summary(None, next_steps, repair_summary=repair_summary)
    _stream_precomposed_text(fallback, token_callback=token_callback)
    return fallback, stage_bare, next_steps, next_steps_summary



def _repair_case_explanation_gaps(facts_summary: str, linked: list) -> list:
    repaired = []
    for ba in linked or []:
        clone_ba = dict(ba)
        repaired_cases = []
        for cl in (ba.get("related_case_laws") or []):
            clone_cl = dict(cl)
            if not str(clone_cl.get("explanation") or "").strip():
                case_label = clone_cl.get("title") or clone_cl.get("case_name") or "Case law"
                _, summary_body = _summarize_case_paragraphs(
                    facts_summary or "",
                    case_label,
                    [clone_cl.get("text") or ""],
                )
                summary_body = (summary_body or "").strip()
                low = summary_body.lower()
                if (not summary_body) or low in {"i don't have any data", "i do not have any data", "no data", "no information available", "insufficient information"} or len(summary_body) < 24:
                    summary_body = _build_case_summary_from_paragraphs([clone_cl.get("text") or ""], max_chars=320).strip()
                clone_cl["explanation"] = summary_body[:700]
            repaired_cases.append(clone_cl)
        clone_ba["related_case_laws"] = repaired_cases
        repaired.append(clone_ba)
    return repaired


def _generate_precedent_stage_text(facts_summary: str, formatted_bare: list, formatted_case: list, model_override: str | None = None, token_callback=None) -> tuple[str, list, list]:
    stage_bare = _prepare_bare_act_stage_entries(formatted_bare)
    stage_case = [dict(c) for c in (formatted_case or [])]
    linked = _match_case_laws_to_bare_acts([dict(b) for b in stage_bare], [dict(c) for c in stage_case], max_per_section=2)
    flattened = []
    for ba in linked:
        flattened.extend(ba.get("related_case_laws") or [])
    if not flattened:
        message = (
            "I reviewed the local precedents as the next step, but I did not find a closely matching local judicial extract strong enough to present as grounded precedent support on the present record. "
            "The statutory protections already shown remain the clearest grounded starting point unless we broaden the search or refine the facts further."
        )
        _stream_precomposed_text(message, token_callback=token_callback)
        return message, linked, []

    precedent_extracts_text = _build_precedent_stage_compact_context(linked)
    prompt = PRECEDENT_STAGE_SUMMARY_PROMPT.format(
        facts_summary=(facts_summary or "").strip()[:1400],
        precedent_extracts_text=precedent_extracts_text[:12000],
    )
    try:
        raw = ask_llm(prompt, model=model_override)
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("No valid JSON returned for precedent stage")
        explanation_map = {
            str(item.get("title") or "").strip().lower(): str(item.get("explanation") or "").strip()
            for item in (parsed.get("case_explanations") or [])
            if isinstance(item, dict)
        }
        for ba in linked:
            enriched_cases = []
            for cl in (ba.get("related_case_laws") or []):
                clone = dict(cl)
                t_key = str(clone.get("title") or "").strip().lower()
                cn_key = str(clone.get("case_name") or "").strip().lower()
                clone["explanation"] = (
                    explanation_map.get(t_key, "")
                    or explanation_map.get(cn_key, "")
                )
                enriched_cases.append(clone)
            ba["related_case_laws"] = enriched_cases
        linked = _repair_case_explanation_gaps(facts_summary, linked)
        summary_text = str(parsed.get("summary_text") or "").strip()
        if not summary_text:
            raise ValueError("Empty precedent stage summary")
        _stream_precomposed_text(summary_text, token_callback=token_callback)
        return summary_text, linked, flattened
    except Exception as exc:
        logger.warning("Precedent stage summary generation failed: %s", exc)
        linked = _repair_case_explanation_gaps(facts_summary, linked)
        fallback = _build_precedent_only_fallback(facts_summary, flattened)
        _stream_precomposed_text(fallback, token_callback=token_callback)
        return fallback, linked, flattened


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
        raw = ask_llm(prompt, task_hint="fast").strip()
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


def _build_case_summary_from_paragraphs(paragraph_texts: list, max_chars: int = 700) -> str:
    """Build a compact extractive summary without an extra LLM call."""
    cleaned = []
    seen = set()
    for txt in paragraph_texts:
        t = " ".join((txt or "").split())
        if not t:
            continue
        t = t[:max_chars]
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
        if len(cleaned) >= 2:
            break
    if not cleaned:
        return ""
    summary = "\n\n".join(cleaned)
    return summary[: max_chars * 2]


def _case_year_for_sort(case_row: dict) -> int:
    """Parse a sortable year from case metadata (fallback 0 when unknown)."""
    if not isinstance(case_row, dict):
        return 0

    raw = case_row.get("year")
    if raw is not None:
        y = str(raw).strip()
        if y.isdigit():
            yi = int(y)
            if 1800 <= yi <= 2100:
                return yi

    citation = str(case_row.get("citation") or "")
    m = re.search(r"\b(18|19|20)\d{2}\b", citation)
    if m:
        return int(m.group(0))

    source = str(case_row.get("source") or case_row.get("source_file") or "")
    m = re.search(r"\b(18|19|20)\d{2}\b", source)
    if m:
        return int(m.group(0))

    return 0


def _format_case_citation(case_law: dict) -> str:
    """Stable display string for case-law bullets in opinion prompts."""
    if not isinstance(case_law, dict):
        return "Unknown case"
    title = (
        case_law.get("title")
        or case_law.get("case_name")
        or case_law.get("source")
        or "Unknown case"
    )
    citation = (case_law.get("citation") or "").strip()
    year = str(case_law.get("year") or "").strip()
    if citation:
        return f"{title} [{citation}]"
    if year:
        return f"{title} ({year})"
    return str(title)


def _format_case_laws(case_laws: list, user_query: str = "") -> list:
    """
    Group case law chunks by case, take top 3 most relevant paragraphs per case.

    Key change: we no longer collapse para_texts into a single summary_body
    before selection. Instead:
    - para_texts (list of up to 3 raw paragraph texts) is preserved on the
      formatted dict so downstream stages (_build_precedent_stage_compact_context,
      _match_case_laws_to_bare_acts) can work on real content.
    - 'text' is set to the holding-extracted lines of the highest-scored para
      via _extract_holding_lines — so even the display text reflects operative
      legal content rather than the first/second paragraph blindly.
    - If _ENABLE_CASE_SUMMARY_LLM is on, the LLM summary path runs in addition
      and its output is stored as 'summary_text' for any UI component that wants
      the full prose version. It does NOT replace 'text'.
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
        uses_pre_summarized_chunks = any(bool(c["cl"].get("is_case_summary")) for c in top3)
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

        # Preserve all top-3 para texts for downstream reasoning stages.
        paragraph_texts = [c["text"] for c in top3]

        # Primary display text: holding-extracted lines from the highest-scored para.
        # For short paras (≤ 8 lines), _extract_holding_lines returns the full text.
        # For long paras, it surfaces the most operative 5 lines.
        display_text = _extract_holding_lines(top3[0]["text"], max_lines=5)
        if not display_text:
            display_text = top3[0]["text"]

        # Optional LLM summary (only when enabled) — stored separately, does NOT
        # replace display_text. UI can use summary_text for a "read more" / prose view.
        summary_text = ""
        if user_query.strip() and _ENABLE_CASE_SUMMARY_LLM and not uses_pre_summarized_chunks:
            _, summary_text = _summarize_case_paragraphs(
                user_query, fallback_title, paragraph_texts
            )

        # Vector store already provides clean case names; fallback_title adds court/year.
        display_title = fallback_title

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
            # 'text' = holding-extracted lines of the top-scored para (not a collapsed summary)
            "text": display_text,
            # 'para_texts' = all top-3 raw para texts, kept alive for downstream stages
            "para_texts": paragraph_texts,
            # 'summary_text' = LLM prose summary if enabled, else empty
            "summary_text": summary_text,
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
            verbatim_snippet = verbatim_text[:_MAX_SECTION_TEXT] if verbatim_text else ""
            lines.append(f"\n**{act}, Section {sec}** — {title}")
            lines.append(f"Section text excerpt:\n{verbatim_snippet}")
            related = ba.get("related_case_laws") or []
            if related:
                lines.append("Case laws under this section (explain what parts apply and how):")
                for cl in related[:2]:
                    name = _format_case_citation(cl)
                    snippet = (cl.get("text") or cl.get("full_text") or "")[:_MAX_CASE_TEXT]
                    lines.append(f"- {name}: {snippet}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks) if blocks else "No dispute components with retrieved materials."


def _generate_interactive_fast_opinion(
    facts_summary: str,
    bare_acts: list,
    case_laws: list,
    token_callback=None,
    model_override: str | None = None,
) -> str:
    """
    Compact grounded opinion path for interactive chat.

    Uses already-formatted local bare acts and case-law summaries directly, so we
    avoid the larger dispute-block prompt and keep first-token latency lower.
    """
    local_bare, local_case = _filter_materials_to_local_db(bare_acts, case_laws)
    if not local_bare and not local_case:
        result = _local_only_no_materials_message()
        _stream_precomposed_text(result, token_callback=token_callback)
        return result
    if local_bare and not local_case:
        result = _build_grounded_bare_act_fallback(local_bare, facts_summary=facts_summary)
        _stream_precomposed_text(result, token_callback=token_callback)
        return result
    if not _ENABLE_INTERACTIVE_FAST_LLM or len(local_case) <= 2:
        result = _build_grounded_interactive_fallback(facts_summary, local_bare, local_case)
        _stream_precomposed_text(result, token_callback=token_callback)
        return result

    bare_payload = json.dumps(
        [
            {
                "title": ba.get("title", ""),
                "text": (ba.get("text") or "")[:_MAX_SECTION_TEXT],
            }
            for ba in local_bare[:3]
        ],
        ensure_ascii=False,
        indent=2,
    )[:2200]
    case_payload = json.dumps(
        [
            {
                "title": cl.get("title", ""),
                "court": cl.get("court", ""),
                "year": cl.get("year", ""),
                "text": (cl.get("text") or "")[:420],
            }
            for cl in local_case[:3]
        ],
        ensure_ascii=False,
        indent=2,
    )[:2400]

    few_shot_block = ""
    if _ENABLE_RUNTIME_FEWSHOT:
        try:
            from platform.training.few_shot import get_opinion_example
            packed = get_opinion_example(facts_summary)
            if packed:
                few_shot_block = f"\n\n{packed}\n"
        except Exception:
            few_shot_block = ""

    prompt = FAST_INTERACTIVE_OPINION_PROMPT.format(
        facts_summary=(facts_summary or "").strip()[:1200],
        bare_acts_json=bare_payload or "[]",
        case_laws_json=case_payload or "[]",
    ) + few_shot_block

    try:
        if token_callback:
            full_text = ""
            for token in ask_llm_stream(prompt, model=model_override):
                token_callback(token)
                full_text += token
            grounded = _enforce_local_grounding(full_text.strip(), local_bare)
            if grounded == _local_only_no_materials_message():
                return _build_grounded_interactive_fallback(facts_summary, local_bare, local_case)
            return grounded
        grounded = _enforce_local_grounding(ask_llm(prompt, model=model_override).strip(), local_bare)
        if grounded == _local_only_no_materials_message():
            return _build_grounded_interactive_fallback(facts_summary, local_bare, local_case)
        return grounded
    except Exception as e:
        logger.error("Interactive fast opinion generation failed: %s", e)
        if local_bare or local_case:
            return _build_grounded_interactive_fallback(facts_summary, local_bare, local_case)
        return _local_only_no_materials_message()


def _generate_structured_opinion_by_dispute(
    facts_summary: str,
    dispute_results: list,
    additional_info: str = "",
    token_callback=None,
    model_override: str | None = None,
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

    from prompts.research import STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT

    few_shot_block = ""
    if _ENABLE_RUNTIME_FEWSHOT:
        try:
            from platform.training.few_shot import get_opinion_example
            few_shot_query = "\n".join(part for part in [facts_summary, additional_info] if part).strip()
            packed = get_opinion_example(few_shot_query)
            if packed:
                few_shot_block = f"\n\n{packed}\n"
        except Exception:
            few_shot_block = ""

    prompt = STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT.format(
        dispute_facts=facts_summary[:1200],
        additional_info=(additional_info or "None provided").strip(),
        dispute_blocks_text=dispute_blocks_text,
    ) + few_shot_block
    logger.info(
        "Final structured opinion prompt size: %d chars across %d dispute component(s)",
        len(prompt),
        len(dispute_results),
    )
    t_start = time.perf_counter()
    try:
        if token_callback:
            full_text = ""
            for token in ask_llm_stream(prompt, model=model_override):
                token_callback(token)
                full_text += token
            local_bare = []
            for dr in dispute_results:
                local_bare.extend(dr.get("bare_acts") or [])
            result = _enforce_local_grounding(full_text.strip(), local_bare)
            if result == _local_only_no_materials_message() and local_bare:
                result = _build_grounded_bare_act_fallback(_format_bare_acts(local_bare))
            logger.info(
                "Final structured opinion generated in %.0f ms (%d chars output)",
                (time.perf_counter() - t_start) * 1000,
                len(result),
            )
            return result
        local_bare = []
        for dr in dispute_results:
            local_bare.extend(dr.get("bare_acts") or [])
        result = _enforce_local_grounding(ask_llm(prompt, model=model_override).strip(), local_bare)
        if result == _local_only_no_materials_message() and local_bare:
            result = _build_grounded_bare_act_fallback(_format_bare_acts(local_bare))
        logger.info(
            "Final structured opinion generated in %.0f ms (%d chars output)",
            (time.perf_counter() - t_start) * 1000,
            len(result),
        )
        return result
    except Exception as e:
        logger.error("Structured opinion by dispute failed: %s", e)
        return ""


def _generate_legal_opinion(
    facts: str,
    bare_acts: list,
    case_laws: list,
    sufficiency: dict,
    on_before_llm=None,
    model_override: str | None = None,
) -> str:
    """Generate a formal legal opinion with citations and source tags. on_before_llm(prompt) is called before LLM if provided."""
    # Check if we have any materials at all
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
        [{"title": b.get("title"), "text": b.get("text", "")[:_MAX_SECTION_TEXT], "source_tag": b.get("source_tag")}
         for b in bare_acts[:15]],
        indent=2,
    )[:_MAX_JSON_PAYLOAD]

    case_text = json.dumps(
        [{"title": c.get("title"), "text": c.get("text", "")[:_MAX_CASE_TEXT], "source_tag": c.get("source_tag")}
         for c in case_laws[:15]],
        indent=2,
    )[:_MAX_JSON_PAYLOAD]

    confidence = sufficiency.get("confidence", "medium")
    
    # Add explicit empty array indicators if needed
    bare_array_note = ""
    case_array_note = ""
    if not has_bare_acts:
        bare_array_note = "\n⚠️ NOTE: The BARE ACT SECTIONS array above is EMPTY ([]). Do NOT create an 'Applicable Statutory Provisions' section."
    if not has_case_laws:
        case_array_note = "\n⚠️ NOTE: The CASE LAWS array above is EMPTY ([]). Do NOT create a 'Relevant Case Law' section."

    few_shot_block = ""
    if _ENABLE_RUNTIME_FEWSHOT:
        try:
            from platform.training.few_shot import get_opinion_example
            packed = get_opinion_example(facts)
            if packed:
                few_shot_block = f"\n\n{packed}\n"
        except Exception:
            few_shot_block = ""

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
{few_shot_block}
Generate the legal analysis:"""

    try:
        if callable(on_before_llm):
            on_before_llm(prompt)
        return _enforce_local_grounding(ask_llm(prompt, model=model_override).strip(), bare_acts)
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
    model_override: str | None = None,
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
        [{"title": b.get("title"), "text": b.get("text", "")[:_MAX_SECTION_TEXT]}
         for b in bare_acts[:10]],
        indent=2,
    )[:_MAX_JSON_PAYLOAD]

    case_text = json.dumps(
        [{"title": c.get("title"), "text": c.get("text", "")[:_MAX_CASE_TEXT]}
         for c in case_laws[:10]],
        indent=2,
    )[:_MAX_JSON_PAYLOAD]

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
            for token in ask_llm_stream(prompt, model=model_override):
                token_callback(token)
                full_text += token
            return _enforce_local_grounding(full_text.strip(), bare_acts)
        return _enforce_local_grounding(ask_llm(prompt, model=model_override).strip(), bare_acts)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return ""


def _format_no_materials_message(search_strategy: str, web_search_stats: Optional[dict]) -> str:
    """
    Return a truthful, data-driven message when no grounded local materials were found.
    When Indiankanoon fallback was used, report actual counts so the user sees what
    happened instead of a generic "I don't have any data" message.
    """
    if web_search_stats is not None:
        wf = web_search_stats.get("web_found", 0)
        sl = web_search_stats.get("shortlisted", 0)
        if wf == 0 and sl == 0:
            return (
                "I searched the local vector store first and then checked Indiankanoon, but found no relevant acts or judgments. "
                "Try rephrasing with specific section numbers, Act names, or a different legal angle."
            )
        return (
            f"I found no grounded local materials in the vector store. I then checked Indiankanoon and found {wf} fallback result(s), "
            f"with {sl} shown below for reference. These external results are not used as grounded local citations in the legal opinion above."
        )
    if search_strategy == "local_only":
        return (
            "I searched the local vector store only and found no relevant bare act provisions or case laws. "
            "Try rephrasing with specific section numbers, Act names, or a different legal angle."
        )
    return (
        "I searched the local vector store first and then checked Indiankanoon, but found no relevant bare act provisions or case laws. "
        "Try rephrasing with specific section numbers, Act names, or a different legal angle."
    )


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
