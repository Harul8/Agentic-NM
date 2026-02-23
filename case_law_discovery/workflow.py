"""
Case law discovery workflow: dynamic (LLM-driven) except hardcoded limits.

Uses existing modules (LLM, config, vector store read, tiered_search, hybrid_retriever)
without modifying them. Only limits in limits.py are fixed.
"""

import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Project root for imports
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from case_law_discovery.limits import (
    SIMILARITY_THRESHOLD_LOW,
    SIMILARITY_THRESHOLD_HIGH,
    TOP_N_STATEMENT,
    TOP_N_PER_ACT,
    BEST_PER_CHUNK,
    MAX_FETCH_PER_ACT,
    ACT_SUMMARY_MAX_CHARS,
    CASE_LAW_SUMMARY_MAX_CHARS,
    TOP_N_ACT_NAMED,
    ACT_NAMED_FETCH_BUFFER,
)
from case_law_discovery.store import (
    load_pending,
    save_pending,
    load_summary_index,
    save_summary_index,
    get_bare_act_summary,
    set_bare_act_summary,
    load_bare_act_summary_index,
)

# Approximate first two pages for classification and signature (match auto_enricher)
_FIRST_TWO_PAGES_CHARS = 3000


def is_case_law_discovery_request(message: str) -> bool:
    """
    Lightweight check: should this message be handled by the case law discovery workflow
    instead of the main chat pipeline? No LLM call.
    """
    if not message or not isinstance(message, str):
        return False
    lower = message.lower().strip()
    # Explicit mention of "case law discovery" always routes to this workflow
    if "case law discovery" in lower:
        return True
    # "find case laws for the first bare act in vector store" / "first bare act in our database" etc.
    has_first_act = ("first" in lower and ("bare act" in lower or "bare acts" in lower or ("act" in lower and ("vector" in lower or "database" in lower))))
    has_vector_store = "vector store" in lower or "vectorstore" in lower or ("database" in lower and ("bare act" in lower or "act" in lower))
    has_case_law = "case law" in lower or "case laws" in lower
    if has_case_law and (has_vector_store or ("first" in lower and "act" in lower)):
        return True
    if has_first_act and (has_vector_store or has_case_law):
        return True
    return False


def _ask_llm(prompt: str, system: str | None = None) -> str:
    """Call existing LLM (Ollama) without changing existing modules."""
    try:
        from llm.ollama_client import ask_llm
        full = (system + "\n\n" + prompt) if system else prompt
        out = ask_llm(full)
        return (out or "").strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return ""


def interpret_user_input(user_message: str) -> dict:
    """
    Use LLM to interpret: statement-based vs first-N-acts, and what to search for.
    Returns dict with: flow_type ("statement" | "first_10_acts"), query/topic, act_names (if applicable).
    """
    system = (
        "You are a legal research assistant. Classify the user's request into exactly one of two flows. "
        "Reply with a single JSON object only, no markdown, no explanation. "
        "Keys: flow_type (either 'statement' or 'first_10_acts'), topic (short search topic or null), "
        "act_count (number for first N acts, default 10). "
        "If the user asks for case laws for 'first 10 bare acts' or 'first N acts in vector store', use first_10_acts. "
        "Otherwise use statement and set topic to the legal subject they want case laws for."
    )
    prompt = f"User request: {user_message}\n\nReply with one JSON object: flow_type, topic, act_count."
    raw = _ask_llm(prompt, system=system)
    try:
        # Extract JSON if wrapped in markdown
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        obj = json.loads(raw)
        flow = (obj.get("flow_type") or "statement").strip().lower()
        if "first" in flow or "10" in str(obj.get("act_count", "")):
            return {"flow_type": "first_10_acts", "topic": None, "act_count": int(obj.get("act_count", 10))}
        return {
            "flow_type": "statement",
            "topic": (obj.get("topic") or user_message[:200]).strip() or "case laws",
            "act_count": None,
        }
    except Exception:
        if "first" in user_message.lower() and ("10" in user_message or "act" in user_message.lower()):
            return {"flow_type": "first_10_acts", "topic": None, "act_count": 10}
        return {"flow_type": "statement", "topic": user_message[:200].strip() or "case laws", "act_count": None}


def get_first_n_acts_from_vector_store(n: int = 10) -> list[dict]:
    """Read bare act chunks from vector store; return first n acts (alphabetically) with their chunks."""
    try:
        from config import BARE_CHUNKS_V2
        if not os.path.isfile(BARE_CHUNKS_V2):
            return []
        with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            return []
        # Chunks may be dict keyed by id or list
        if isinstance(chunks, dict):
            chunk_list = list(chunks.values())
        else:
            chunk_list = chunks
        by_act = {}
        for c in chunk_list:
            name = (c.get("act_name") or c.get("source") or "Unknown").strip()
            by_act.setdefault(name, []).append(c)
        sorted_acts = sorted(by_act.keys(), key=lambda x: x.lower())
        return [{"act_name": name, "chunks": by_act[name]} for name in sorted_acts[:n]]
    except Exception as e:
        logger.warning("Could not read bare chunks: %s", e)
        return []


def _chunk_text(chunk: dict) -> str:
    """Extract searchable text from a bare act chunk."""
    return (
        (chunk.get("content") or chunk.get("search_text") or chunk.get("full_text") or chunk.get("text") or "")
        .strip()
    )


def _generate_act_summary(act_name: str, chunks: list[dict]) -> str:
    """Generate a summary of the act from its chunks (LLM), capped at ACT_SUMMARY_MAX_CHARS."""
    combined = []
    for c in chunks[:50]:
        t = _chunk_text(c)
        if t:
            combined.append(t[:800])
    text = "\n\n".join(combined)[:15000]
    if not text.strip():
        return act_name
    system = (
        "You are a legal assistant. Summarize the following Indian bare act text in clear, concise language. "
        f"Include: short title, purpose, key definitions, and main provisions. Keep the summary under {ACT_SUMMARY_MAX_CHARS} characters. "
        "Output only the summary, no preamble."
    )
    prompt = f"Act: {act_name}\n\nText:\n{text}\n\nSummary:"
    out = _ask_llm(prompt, system=system)
    return (out or act_name).strip()[:ACT_SUMMARY_MAX_CHARS]


def _generate_case_law_summary(doc_text: str, title: str = "") -> str:
    """Generate a summary of a case law document (LLM), capped at CASE_LAW_SUMMARY_MAX_CHARS."""
    text = (doc_text or "").strip()[:12000]
    if not text:
        return title or "Case law"
    system = (
        "You are a legal assistant. Summarize the following Indian court judgment in clear language. "
        f"Include: parties, court, key facts, and main holding. Keep under {CASE_LAW_SUMMARY_MAX_CHARS} characters. "
        "Output only the summary."
    )
    prompt = f"Judgment:\n{text}\n\nSummary:"
    out = _ask_llm(prompt, system=system)
    return (out or title or "Case law").strip()[:CASE_LAW_SUMMARY_MAX_CHARS]


def get_or_create_act_summary(act_name: str, chunks: list[dict]) -> str:
    """Return act summary from index, or generate from chunks and save."""
    summary = get_bare_act_summary(act_name)
    if summary:
        return summary
    summary = _generate_act_summary(act_name, chunks)
    if summary:
        set_bare_act_summary(act_name, summary)
    return summary or act_name


def _is_case_law_doc(url: str, title: str, first_two_pages: str) -> bool:
    """Heuristic: is this document a case law (judgment) vs bare act?"""
    combined = f"{url} {title} {first_two_pages}".lower()
    case_indicators = [
        " v. ", " v/s ", " vs ", "appellant", "respondent",
        "petitioner", "judgment", "judgement", "hon'ble",
        "supreme court", "high court", "bench", "coram",
    ]
    bare_act_indicators = [
        "bare act", "act,", "code,", "ordinance", "regulation",
        "section", "chapter", "schedule", "notification",
        "indiacode", "legislative", "gazette",
    ]
    case_score = sum(1 for i in case_indicators if i in combined)
    act_score = sum(1 for i in bare_act_indicators if i in combined)
    return case_score > act_score


def _extract_signature(first_two_pages: str) -> str:
    """
    Extract duplicate-check signature from first two pages: court + parties (appellant vs respondent).
    Returns a short string for dedup; empty if extraction fails.
    """
    if not first_two_pages or len(first_two_pages.strip()) < 100:
        return ""
    text = first_two_pages.strip()[:_FIRST_TWO_PAGES_CHARS]
    try:
        system = (
            "You extract a unique signature for an Indian court judgment from the first two pages. "
            "Reply with exactly one line: Court (e.g. Supreme Court or High Court of X) then | then "
            "Parties (Appellant vs Respondent, e.g. 'XYZ Ltd vs State of Maharashtra'). "
            "Use only the exact format: COURT | PARTIES. If you cannot determine, reply UNKNOWN."
        )
        prompt = f"Document excerpt:\n{text[:2500]}\n\nOne-line signature (COURT | PARTIES):"
        raw = _ask_llm(prompt, system=system)
        if raw and "UNKNOWN" not in raw.upper() and "|" in raw:
            return raw.strip()[:200]
    except Exception as e:
        logger.debug("Signature extraction failed: %s", e)
    return ""


def _fetch_and_score_candidate(
    result: dict,
    query: str,
    include_content: bool = False,
) -> dict | None:
    """
    Fetch URL, check it's case law from first two pages, score vs query.
    Returns dict with title, source_url, score, signature; optionally content (for summary generation).
    """
    from retrieval.tiered_search import fetch_content_and_pdf
    from retrieval.hybrid_retriever import score_query_document

    url = result.get("url", "")
    title = result.get("title", "Unknown")
    if not url:
        return None
    try:
        text_content, _ = fetch_content_and_pdf(url, timeout=25)
    except Exception as e:
        logger.debug("Fetch failed %s: %s", url[:60], e)
        return None
    text_content = (text_content or "").strip()
    if not text_content or len(text_content) < 200:
        return None
    first_two = text_content[:_FIRST_TWO_PAGES_CHARS]
    if not _is_case_law_doc(url, title, first_two):
        return None
    score = score_query_document(query, text_content)
    if score <= SIMILARITY_THRESHOLD_LOW:
        return None
    signature = _extract_signature(first_two)
    out = {
        "title": title,
        "source_url": url,
        "suggested_category": "case_law",
        "score": float(score),
        "signature": signature or f"{url[:80]}",
        "act_name": None,
    }
    if include_content:
        out["content"] = text_content[:12000]
    return out


def _score_candidate_content(
    title: str,
    source_url: str,
    text_content: str,
    query: str,
    include_content: bool = False,
) -> dict | None:
    """
    Score pre-fetched content (e.g. from Indian Kanoon API). No URL fetch.
    Returns same shape as _fetch_and_score_candidate or None if below threshold.
    """
    from retrieval.hybrid_retriever import score_query_document

    text_content = (text_content or "").strip()
    if not text_content or len(text_content) < 200:
        return None
    score = score_query_document(query, text_content)
    if score <= SIMILARITY_THRESHOLD_LOW:
        return None
    first_two = text_content[:_FIRST_TWO_PAGES_CHARS]
    signature = _extract_signature(first_two)
    out = {
        "title": title,
        "source_url": source_url,
        "suggested_category": "case_law",
        "score": float(score),
        "signature": signature or source_url[:80],
        "act_name": None,
    }
    if include_content:
        out["content"] = text_content[:12000]
    return out


def _indian_kanoon_candidates(query: str, max_results: int, include_content: bool) -> list[dict]:
    """Search Indian Kanoon API and return scored candidates (same shape as web candidates)."""
    try:
        from retrieval.indian_kanoon_client import search as ik_search, get_document
    except ImportError:
        return []
    from config import INDIAN_KANOON_API_TOKEN
    if not INDIAN_KANOON_API_TOKEN:
        return []
    raw = ik_search(query, pagenum=0, max_results=max_results)
    candidates = []
    for r in raw[:max_results]:
        tid = r.get("tid")
        if not tid:
            continue
        time.sleep(0.3)
        doc = get_document(tid)
        if not doc or not (doc.get("doc_text") or "").strip():
            continue
        text = (doc.get("doc_text") or "").strip()
        title = doc.get("title") or r.get("title") or "Judgment"
        url = doc.get("url") or r.get("url") or f"https://indiankanoon.org/doc/{tid}/"
        c = _score_candidate_content(title, url, text, query, include_content=include_content)
        if c:
            candidates.append(c)
    return candidates


def _resolve_act_from_summary_index(user_topic: str) -> tuple[str, str]:
    """
    Resolve user's act name (e.g. 'Indian Penal Code') to (act_name, summary) from bare act summary index.
    Returns ("", "") if no match. Matches exact key or topic contained in act name / act name in topic.
    """
    if not user_topic or not user_topic.strip():
        return ("", "")
    topic_norm = " ".join(user_topic.strip().lower().split())
    index = load_bare_act_summary_index()
    if not index:
        return ("", "")
    # Exact match (case-insensitive)
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        if act_name.strip().lower() == topic_norm:
            return (act_name.strip(), (summary or "").strip())
    # Topic contained in act name (e.g. "Indian Penal Code" in "The Indian Penal Code, 1860")
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        if topic_norm in act_name.strip().lower():
            return (act_name.strip(), (summary or "").strip())
    # Act name contained in topic (e.g. user said full name)
    for act_name, summary in index.items():
        if not act_name or not summary:
            continue
        if act_name.strip().lower() in topic_norm:
            return (act_name.strip(), (summary or "").strip())
    return ("", "")


def _apply_statement_selection(candidates: list[dict]) -> list[dict]:
    """Apply limits: all > HIGH, else top TOP_N_STATEMENT above LOW."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= TOP_N_STATEMENT:
        return sorted(above_high, key=lambda x: -x["score"])[:TOP_N_STATEMENT]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:TOP_N_STATEMENT]


def _apply_per_act_selection(candidates: list[dict]) -> list[dict]:
    """Per act: all >5, else top TOP_N_PER_ACT above 3."""
    above_high = [c for c in candidates if c["score"] > SIMILARITY_THRESHOLD_HIGH]
    above_low = [c for c in candidates if SIMILARITY_THRESHOLD_LOW < c["score"] <= SIMILARITY_THRESHOLD_HIGH]
    if len(above_high) >= TOP_N_PER_ACT:
        return sorted(above_high, key=lambda x: -x["score"])[:TOP_N_PER_ACT]
    combined = above_high + sorted(above_low, key=lambda x: -x["score"])
    return combined[:TOP_N_PER_ACT]


def run_act_named_flow(act_name: str, act_summary: str) -> tuple[list[dict], int]:
    """
    User named an act: use bare act summary index → search Indian Kanoon with (act name + summary),
    get top results by similarity, deduplicate, keep first TOP_N_ACT_NAMED (10) unique. If duplicates
    in the first 10, they are discarded and we take the next in rank (11, 12, ...) until we have 10 unique.
    Renaming on indexing is done by the enricher (case law filename logic).
    Returns (items_for_pending, duplicates_discarded_count).
    """
    query = (act_name + " " + (act_summary or "")[:1500]).strip()
    logger.info("Act-named flow: act=%s, fetch up to %s from Indian Kanoon, dedup, target %s unique",
                act_name[:50], ACT_NAMED_FETCH_BUFFER, TOP_N_ACT_NAMED)

    # Fetch more than TOP_N so we can fill 10 after dedup
    scored = _indian_kanoon_candidates(query, max_results=ACT_NAMED_FETCH_BUFFER, include_content=True)
    if not scored:
        logger.warning("No Indian Kanoon results for act-named query")
        return [], 0

    # Sort by score descending (highest first)
    scored.sort(key=lambda x: -x["score"])

    # Deduplicate vs existing index (and within-list): check_indexing_candidates marks already_in_store
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in scored
    ]
    deduped = check_indexing_candidates(dedup_list)
    # Merge already_in_store back into scored (same order)
    for i, s in enumerate(scored):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False

    # Take first TOP_N_ACT_NAMED that are unique (not already_in_store); duplicates in top 10 → take next in rank (11, 12, ...)
    unique = []
    discarded = 0
    for s in scored:
        if len(unique) >= TOP_N_ACT_NAMED:
            break
        if s.get("already_in_store"):
            discarded += 1
            continue
        unique.append(s)
    if discarded:
        logger.info("Act-named: %s unique selected (target %s), %s duplicates discarded, next in rank used",
                    len(unique), TOP_N_ACT_NAMED, discarded)

    for s in unique:
        s["act_name"] = act_name
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))
    return unique, discarded


def run_statement_flow(topic: str) -> list[dict]:
    """
    Statement-based flow: Indian Kanoon API first (if token set), then tiered web search →
    fetch → score → apply limits (all >5, else top 10 above 3). Returns list for pending.
    """
    from retrieval.tiered_search import tiered_search

    logger.info("Statement flow (topic=%s): limits TOP_N=%s, score>%s or top above %s",
                topic, TOP_N_STATEMENT, SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_LOW)
    candidates = []
    seen_urls = set()

    # First: Indian Kanoon API (if configured)
    ik_list = _indian_kanoon_candidates(topic, max_results=15, include_content=True)
    for c in ik_list:
        url = c.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            candidates.append(c)

    # Then: tiered web search (official → legal portals → newspapers)
    results = tiered_search(
        query=topic,
        search_type="case_law",
        jurisdiction_state="Telangana",
        max_per_tier=12,
    )
    for r in results[:20]:
        if r.get("url") in seen_urls:
            continue
        seen_urls.add(r.get("url"))
        time.sleep(0.5)
        doc = _fetch_and_score_candidate(r, topic, include_content=True)
        if doc:
            candidates.append(doc)

    selected = _apply_statement_selection(candidates)
    # Deduplicate vs existing index (same logic as main indexing)
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in selected
    ]
    deduped = check_indexing_candidates(dedup_list)
    for i, s in enumerate(selected):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False
    for s in selected:
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))
    return selected


def _signatures_from_summary_index(summary_index: dict) -> set:
    """Collect all signatures already in the summary index (any act)."""
    seen = set()
    for sig_list in summary_index.values():
        if isinstance(sig_list, list):
            for s in sig_list:
                if isinstance(s, str):
                    seen.add(s)
                elif isinstance(s, dict) and s.get("signature"):
                    seen.add(str(s["signature"]))
        elif isinstance(sig_list, str):
            seen.add(sig_list)
    return seen


def run_first_10_acts_flow(act_count: int) -> list[dict]:
    """
    First-N-acts flow: one summary per act (from bare act summary index or generated), one web search
    per act (act name + summary), fetch up to MAX_FETCH_PER_ACT, score each vs whole-act summary,
    top TOP_N_PER_ACT (10) per act. Generate case law summary for each; dedup vs act-case-law map.
    Returns list for pending (title, source_url, signature, act_name, summary).
    """
    from retrieval.tiered_search import tiered_search

    summary_index = load_summary_index()
    index_signatures = _signatures_from_summary_index(summary_index)
    all_items = []
    acts = get_first_n_acts_from_vector_store(act_count)
    if not acts:
        logger.warning("No acts found in vector store")
        return []

    for act_info in acts:
        act_name = act_info.get("act_name", "Unknown")
        chunks = act_info.get("chunks", [])
        act_summary = get_or_create_act_summary(act_name, chunks)
        query = (act_name + " " + act_summary[:1500]).strip()
        logger.info("Act %s: one search (summary), top %s judgments", act_name[:50], TOP_N_PER_ACT)

        scored = []
        # First: Indian Kanoon API (if configured)
        ik_list = _indian_kanoon_candidates(query, max_results=MAX_FETCH_PER_ACT, include_content=True)
        scored.extend(ik_list)

        # Then: tiered web search if we want more candidates
        if len(scored) < TOP_N_PER_ACT:
            time.sleep(1.0)
            results = tiered_search(
                query=query,
                search_type="case_law",
                jurisdiction_state="Telangana",
                max_per_tier=10,
            )
            seen_urls = {c.get("source_url") for c in scored}
            for r in results[:MAX_FETCH_PER_ACT]:
                if r.get("url") in seen_urls:
                    continue
                seen_urls.add(r.get("url"))
                time.sleep(0.4)
                doc = _fetch_and_score_candidate(r, act_summary, include_content=True)
                if doc:
                    scored.append(doc)
        selected = _apply_per_act_selection(scored)
        selected = [s for s in selected if (s.get("signature") or "") not in index_signatures]
        by_sig = {}
        for s in selected:
            sig = s.get("signature") or s["source_url"]
            if sig not in by_sig or by_sig[sig]["score"] < s["score"]:
                by_sig[sig] = s
        selected = list(by_sig.values())
        for s in selected:
            if s.get("signature"):
                index_signatures.add(s["signature"])
        for s in selected:
            s["act_name"] = act_name
            all_items.append(s)  # keep content for dedup

    # Deduplicate vs existing index (same logic as main indexing)
    from services.indexing_duplicate_check import check_indexing_candidates
    dedup_list = [
        {"title": s.get("title", ""), "source_url": s.get("source_url", ""), "suggested_category": "case_law", "content": (s.get("content") or "")[:2000]}
        for s in all_items
    ]
    deduped = check_indexing_candidates(dedup_list)
    for i, s in enumerate(all_items):
        s["already_in_store"] = deduped[i].get("already_in_store", False) if i < len(deduped) else False
        content = s.pop("content", None)
        s["summary"] = _generate_case_law_summary(content or "", s.get("title", ""))

    return all_items


def run(user_message: str) -> dict:
    """
    Main entry: interpret message, run the chosen flow, append results to pending store.
    Returns { "flow_type", "topic" or "act_count", "added": int, "pending_total": int, "message" }.
    """
    interpreted = interpret_user_input(user_message)
    flow_type = interpreted.get("flow_type", "statement")
    added = 0
    duplicates_discarded = None
    if flow_type == "first_10_acts":
        act_count = interpreted.get("act_count") or 10
        items = run_first_10_acts_flow(act_count)
    else:
        topic = interpreted.get("topic") or "case laws"
        act_name, act_summary = _resolve_act_from_summary_index(topic)
        if act_name and act_summary:
            items, duplicates_discarded = run_act_named_flow(act_name, act_summary)
        else:
            items = run_statement_flow(topic)

    if items:
        pending = load_pending()
        for it in items:
            pending.append({
                "title": it.get("title") or "Case law",
                "source_url": it.get("source_url") or "",
                "suggested_category": it.get("suggested_category", "case_law"),
                "signature": it.get("signature"),
                "act_name": it.get("act_name"),
                "summary": it.get("summary"),
                "already_in_store": it.get("already_in_store", False),
            })
        save_pending(pending)
        added = len(items)

    pending = load_pending()
    message = f"Added {added} document(s) for indexing. Total pending: {len(pending)}."
    if duplicates_discarded is not None and duplicates_discarded > 0:
        message += f" {duplicates_discarded} duplicate(s) discarded; {added} unique case laws proposed."
    return {
        "flow_type": flow_type,
        "topic": interpreted.get("topic"),
        "act_count": interpreted.get("act_count"),
        "added": added,
        "pending_total": len(pending),
        "duplicates_discarded": duplicates_discarded,
        "message": message,
    }


def retrieve_first_gate(query: str = "") -> dict:
    """
    First-gate retrieval: resolve act(s) from query (or all acts with case laws), return
    act name + act summary + list of case law (signature, summary) from summary indexes.
    No vector store access; use for fast "case laws for this act/text" response.
    """
    act_map = load_summary_index()
    bare_summaries = load_bare_act_summary_index()
    case_summaries = load_case_law_summary_index()
    q = (query or "").strip().lower()
    acts_with_cases = [
        (act_name, sig_list)
        for act_name, sig_list in act_map.items()
        if isinstance(sig_list, list) and len(sig_list) > 0
    ]
    if q:
        words = [w for w in q.split() if len(w) > 1]
        def matches(act_name: str) -> bool:
            if q in act_name.lower():
                return True
            summary = bare_summaries.get(act_name) or ""
            if q in summary.lower():
                return True
            if words and any(w in act_name.lower() or w in summary.lower() for w in words):
                return True
            return False
        acts_with_cases = [(a, lst) for a, lst in acts_with_cases if matches(a)]
    result = []
    for act_name, sig_list in acts_with_cases:
        act_summary = bare_summaries.get(act_name) or get_bare_act_summary(act_name)
        case_laws = []
        for sig in sig_list:
            if isinstance(sig, dict):
                sig = sig.get("signature") or ""
            if not isinstance(sig, str):
                continue
            summary = case_summaries.get(sig) or get_case_law_summary(sig)
            case_laws.append({"signature": sig, "summary": summary or ""})
        result.append({
            "act_name": act_name,
            "act_summary": (act_summary or "")[:2000],
            "case_laws": case_laws,
        })
    return {"acts": result, "query": query or None}
