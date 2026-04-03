"""
Legal Opinion Intake — Stage 5: Draft Document Generation

The final autonomous stage before advocate review.

Workflow
--------
1. Resolve document type from the confirmed remedy plan (Stage 4).
2. Generate targeted retrieval queries (bare acts + case laws) via LLM.
3. Run hybrid retrieval in parallel against the vector indexes.
4. Format the retrieved materials for the draft prompt.
5. Stream the complete draft document token by token.
6. Return the final draft text + retrieval citations.

Public API
----------
  new_stage5_state(stage4_state)             → dict
  generate_stage5_draft(
      stage4_state,
      model_override=None,
      token_callback=None,          # callable(token_str) — for SSE streaming
      step_callback=None,           # callable(step_dict) — for progress steps
  )                                 → dict
      Returns: {draft_text, stage5_state, retrieved_bare_acts, retrieved_case_laws,
                document_type, document_type_label, citations}
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM helpers
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "fast", model_override: str | None = None) -> str:
    from llm.ollama_client import ask_llm
    return ask_llm(prompt, task_hint=task_hint, model=model_override) or ""


def _stream_llm(prompt: str, token_callback: Callable | None, model_override: str | None = None) -> str:
    """Stream LLM output token by token; returns the full text."""
    from llm.ollama_client import ask_llm_stream
    full = []
    for token in ask_llm_stream(prompt, model=model_override):
        full.append(token)
        if token_callback:
            try:
                token_callback(token)
            except Exception:
                pass
    return "".join(full)


# ---------------------------------------------------------------------------
# Prompt + schema imports
# ---------------------------------------------------------------------------

from prompts.advocate_prompts import (
    STAGE5_RETRIEVAL_QUERIES_SYSTEM,
    STAGE5_DRAFT_SYSTEM,
    STAGE5_STATE_SCHEMA,
    resolve_document_type,
)

# ---------------------------------------------------------------------------
# Debug timing
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage5.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | list | None:
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if "{" in part or "[" in part:
                text = part
                break
    start = min(
        (text.find("{") if "{" in text else len(text)),
        (text.find("[") if "[" in text else len(text)),
    )
    end_brace  = text.rfind("}")
    end_bracket = text.rfind("]")
    end = max(end_brace, end_bracket)
    if start < len(text) and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _run_bare_act_retrieval(query: str, top_k: int = 6) -> list[dict]:
    """Search bare acts with safe fallback."""
    try:
        from retrieval.hybrid_retriever import search_bare_acts_auto
        return search_bare_acts_auto(query, top_k=top_k) or []
    except Exception as exc:
        logger.warning("Bare act retrieval failed for %r: %s", query[:60], exc)
        return []


def _run_case_law_retrieval(query: str, top_k: int = 5) -> list[dict]:
    """Search case laws with safe fallback."""
    try:
        from retrieval.hybrid_retriever import search_case_laws_auto
        return search_case_laws_auto(query, top_k=top_k) or []
    except Exception as exc:
        logger.warning("Case law retrieval failed for %r: %s", query[:60], exc)
        return []


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by text content."""
    seen: set[str] = set()
    result: list[dict] = []
    for chunk in chunks:
        key = (chunk.get("text") or chunk.get("content") or "")[:120].strip()
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def _format_bare_act_chunk(chunk: dict, idx: int) -> str:
    """Format one bare act chunk for the draft prompt."""
    act  = chunk.get("act_name") or chunk.get("source") or "Act"
    sec  = chunk.get("section_number") or chunk.get("section") or ""
    subtitle = chunk.get("section_title") or chunk.get("title") or ""
    subsection = chunk.get("subsection") or chunk.get("sub_section") or chunk.get("clause") or ""
    text = (chunk.get("text") or chunk.get("content") or "").strip()
    header = f"[{idx}] {act}"
    if sec:
        header += f" — {sec}"
    return f"{header}\n{text}"


def _format_case_law_chunk(chunk: dict, idx: int) -> str:
    """Format one case law chunk for the draft prompt."""
    case = chunk.get("case_name") or chunk.get("source") or "Judgment"
    year = chunk.get("year") or ""
    court= chunk.get("court") or ""
    text = (chunk.get("text") or chunk.get("content") or "").strip()
    header = f"[{idx}] {case}"
    if court:
        header += f" — {court}"
    if year:
        header += f" ({year})"
    return f"{header}\n{text}"


def _format_retrieved_bare_acts(chunks: list[dict]) -> str:
    if not chunks:
        return "(No statutory provisions retrieved — note in Grounds section that specific sections are to be cited after verification)"
    parts = [_format_bare_act_chunk(c, i + 1) for i, c in enumerate(chunks[:10])]
    return "\n\n".join(parts)


def _format_retrieved_case_laws(chunks: list[dict]) -> str:
    if not chunks:
        return "(No case laws retrieved — note in Case Laws section that supporting judgments are to be inserted after research)"
    parts = [_format_case_law_chunk(c, i + 1) for i, c in enumerate(chunks[:8])]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Format helpers for the draft prompt
# ---------------------------------------------------------------------------

def _facts_from_state(stage4_state: dict) -> tuple[str, str]:
    """Returns (high_confidence_facts_text, medium_confidence_facts_text)."""
    vr = stage4_state.get("vetting_report") or {}
    high  = vr.get("high_confidence_facts") or []
    med   = vr.get("medium_confidence_facts") or []
    unc   = vr.get("uncertain_facts") or []

    # Fallback: use known_facts if vetting_report is empty
    if not high and not med:
        all_facts = stage4_state.get("known_facts") or []
        high = all_facts
        med  = []

    # If medium is empty, pick uncertain facts and label them as medium
    # (they still get hedged language in the draft)
    if not med and unc:
        med = unc

    high_text = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(high)) or "  (see conversation history)"
    med_text  = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(med)) if med else "  (none)"
    return high_text, med_text


def _evidence_text(stage4_state: dict) -> str:
    items = stage4_state.get("evidence_items") or []
    if not items:
        return "(none mentioned — note in draft that relevant documents to be filed)"
    return "\n".join(f"  - {e}" for e in items)


def _supporting_remedies_text(remedy_plan: dict) -> str:
    items = remedy_plan.get("supporting_remedies") or []
    return ", ".join(items) if items else "none"


def _statutory_basis_from_plan(remedy_plan: dict) -> str:
    """Extract a statutory basis hint from the strategy note."""
    return remedy_plan.get("strategy_note") or "(as per confirmed remedy plan)"


# ===========================================================================
# Public: new_stage5_state
# ===========================================================================

def new_stage5_state(stage4_state: dict) -> dict:
    state = copy.deepcopy(STAGE5_STATE_SCHEMA)
    remedy_plan = stage4_state.get("remedy_plan") or {}
    lead_remedy = remedy_plan.get("lead_remedy") or (
        (stage4_state.get("confirmed_remedies") or [""])[0]
    )
    category = stage4_state.get("category") or "general"
    doc_type, doc_label = resolve_document_type(category, lead_remedy or "")

    state.update({
        "category":           category,
        "category_label":     stage4_state.get("category_label") or "General Matter",
        "jurisdiction":       stage4_state.get("jurisdiction") or "unknown",
        "client_role":        stage4_state.get("client_role") or "Petitioner",
        "other_party":        stage4_state.get("other_party") or "Respondent",
        "known_facts":        list(stage4_state.get("known_facts") or []),
        "evidence_items":     list(stage4_state.get("evidence_items") or []),
        "vetting_report":     dict(stage4_state.get("vetting_report") or {}),
        "remedy_plan":        dict(remedy_plan),
        "document_type":      doc_type,
        "document_type_label": doc_label,
    })
    return state


# ===========================================================================
# Internal: generate retrieval queries
# ===========================================================================

def _generate_retrieval_queries(
    stage5_state: dict,
    model_override: str | None = None,
) -> dict:
    t0 = time.perf_counter()

    remedy_plan = stage5_state.get("remedy_plan") or {}
    high_facts, _ = _facts_from_state(stage5_state)

    prompt = (
        STAGE5_RETRIEVAL_QUERIES_SYSTEM
        .replace("{document_type_label}", stage5_state.get("document_type_label") or "Legal Document")
        .replace("{category_label}",      stage5_state.get("category_label") or "General")
        .replace("{jurisdiction}",        stage5_state.get("jurisdiction") or "India")
        .replace("{key_facts}",           high_facts[:800])
        .replace("{lead_remedy}",         remedy_plan.get("lead_remedy") or "")
        .replace("{supporting_remedies}", _supporting_remedies_text(remedy_plan))
        .replace("{statutory_basis}",     _statutory_basis_from_plan(remedy_plan))
    )

    _DEFAULT = {
        "bare_act_queries": [
            f"{stage5_state.get('category_label', 'legal')} {stage5_state.get('document_type', '')}",
        ],
        "case_law_queries": [
            f"{stage5_state.get('category_label', 'legal')} rights protection India",
        ],
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("retrieval_queries", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage5 retrieval query generation failed: %s", exc)

    return _DEFAULT


# ===========================================================================
# Internal: run parallel retrieval
# ===========================================================================

def _run_parallel_retrieval(
    bare_act_queries: list[str],
    case_law_queries: list[str],
    step_callback: Callable | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run all bare-act and case-law queries in parallel.
    Returns (bare_act_chunks, case_law_chunks) — deduplicated.
    """
    if step_callback:
        step_callback({"message": "Retrieving relevant statutes and judgments…", "icon": ""})

    bare_chunks: list[dict] = []
    case_chunks: list[dict] = []

    all_tasks = (
        [("bare", q) for q in (bare_act_queries or [])[:4]] +
        [("case", q) for q in (case_law_queries or [])[:4]]
    )

    if not all_tasks:
        return [], []

    with ThreadPoolExecutor(max_workers=min(8, len(all_tasks))) as pool:
        futures = {}
        for kind, query in all_tasks:
            if kind == "bare":
                fut = pool.submit(_run_bare_act_retrieval, query, 6)
            else:
                fut = pool.submit(_run_case_law_retrieval, query, 5)
            futures[fut] = kind

        for fut in as_completed(futures):
            kind = futures[fut]
            try:
                chunks = fut.result()
                if kind == "bare":
                    bare_chunks.extend(chunks)
                else:
                    case_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Retrieval task failed: %s", exc)

    bare_chunks = _deduplicate_chunks(bare_chunks)
    case_chunks = _deduplicate_chunks(case_chunks)

    # Sort by rerank score descending
    bare_chunks.sort(key=lambda c: float(c.get("rerank_score") or c.get("score") or 0), reverse=True)
    case_chunks.sort(key=lambda c: float(c.get("rerank_score") or c.get("score") or 0), reverse=True)

    logger.info("Stage5 retrieval: %d bare-act chunks, %d case-law chunks", len(bare_chunks), len(case_chunks))
    return bare_chunks, case_chunks


# ===========================================================================
# Internal: build the draft prompt
# ===========================================================================

def _build_draft_prompt(stage5_state: dict) -> str:
    remedy_plan  = stage5_state.get("remedy_plan") or {}
    high_facts, med_facts = _facts_from_state(stage5_state)

    return (
        STAGE5_DRAFT_SYSTEM
        .replace("{document_type_label}",  stage5_state.get("document_type_label") or "Legal Document")
        .replace("{high_confidence_facts}", high_facts)
        .replace("{medium_confidence_facts}", med_facts)
        .replace("{evidence_items}",       _evidence_text(stage5_state))
        .replace("{client_role}",          stage5_state.get("client_role") or "Petitioner")
        .replace("{other_party}",          stage5_state.get("other_party") or "Respondent")
        .replace("{jurisdiction}",         stage5_state.get("jurisdiction") or "India")
        .replace("{lead_remedy}",          remedy_plan.get("lead_remedy") or "relief as prayed")
        .replace("{supporting_remedies}",  _supporting_remedies_text(remedy_plan))
        .replace("{strategy_note}",        remedy_plan.get("strategy_note") or "")
        .replace("{retrieved_bare_acts}",  _format_retrieved_bare_acts(stage5_state.get("retrieved_bare_acts") or []))
        .replace("{retrieved_case_laws}",  _format_retrieved_case_laws(stage5_state.get("retrieved_case_laws") or []))
    )


# ===========================================================================
# Internal: build citations summary
# ===========================================================================

def _build_citations(bare_chunks: list[dict], case_chunks: list[dict]) -> list[dict]:
    citations = []
    for c in bare_chunks[:10]:
        citations.append({
            "type":    "bare_act",
            "source":  c.get("act_name") or c.get("source") or "Statute",
            "section": c.get("section_number") or c.get("section") or "",
            "subsection": c.get("subsection") or c.get("sub_section") or c.get("clause") or "",
            "section_title": c.get("section_title") or c.get("title") or "",
            "score":   round(float(c.get("rerank_score") or c.get("score") or 0), 3),
        })
    for c in case_chunks[:8]:
        citations.append({
            "type":    "case_law",
            "source":  c.get("case_name") or c.get("source") or "Judgment",
            "court":   c.get("court") or "",
            "year":    c.get("year") or "",
            "paragraph": c.get("paragraph_num") or c.get("para_num") or c.get("paragraph_id") or "",
            "citation": c.get("citation") or "",
            "score":   round(float(c.get("rerank_score") or c.get("score") or 0), 3),
        })
    return citations


# ===========================================================================
# Public: generate_stage5_draft
# ===========================================================================

def generate_stage5_draft(
    stage4_state: dict,
    model_override: str | None = None,
    token_callback: Callable | None = None,
    step_callback: Callable | None = None,
) -> dict:
    """
    Full Stage 5 pipeline: retrieval + draft generation (streaming).

    Parameters
    ----------
    stage4_state : dict
        The final stage4_state from Stage 4.
    model_override : str | None
    token_callback : callable(str) | None
        Called with each streamed token — for SSE real-time output.
    step_callback : callable(dict) | None
        Called with step progress dicts.

    Returns
    -------
    dict:
      - draft_text           : str  — complete draft document
      - stage5_state         : dict
      - retrieved_bare_acts  : list
      - retrieved_case_laws  : list
      - document_type        : str
      - document_type_label  : str
      - citations            : list
    """
    t_total = time.perf_counter()

    # ── 1. Initialise Stage 5 state ──────────────────────────────────────────
    stage5_state = new_stage5_state(stage4_state)
    doc_type  = stage5_state["document_type"]
    doc_label = stage5_state["document_type_label"]

    if step_callback:
        step_callback({"message": f"Preparing {doc_label}…", "icon": ""})

    # ── 2. Generate retrieval queries ─────────────────────────────────────────
    queries = _generate_retrieval_queries(stage5_state, model_override=model_override)
    stage5_state["retrieval_queries"] = queries

    # ── 3. Parallel retrieval ─────────────────────────────────────────────────
    bare_chunks, case_chunks = _run_parallel_retrieval(
        queries.get("bare_act_queries") or [],
        queries.get("case_law_queries") or [],
        step_callback=step_callback,
    )
    stage5_state["retrieved_bare_acts"]  = bare_chunks
    stage5_state["retrieved_case_laws"]  = case_chunks
    _t("retrieval_complete", t_total)

    # ── 4. Build the draft prompt ─────────────────────────────────────────────
    if step_callback:
        step_callback({"message": "Drafting your document…", "icon": "✍"})

    prompt = _build_draft_prompt(stage5_state)

    # ── 5. Stream the draft ───────────────────────────────────────────────────
    draft_text = _stream_llm(prompt, token_callback=token_callback, model_override=model_override)
    _t("draft_complete", t_total)

    if not draft_text or len(draft_text.strip()) < 200:
        # Fallback: non-streaming call
        logger.warning("Stage5 streaming produced short output (%d chars) — retrying non-streamed", len(draft_text))
        from llm.ollama_client import ask_llm
        draft_text = ask_llm(prompt, task_hint="quality", model=model_override) or draft_text

    # ── 6. Finalise state ─────────────────────────────────────────────────────
    stage5_state["draft_text"]        = draft_text.strip()
    stage5_state["draft_complete"]    = True
    stage5_state["ready_for_review"]  = True

    citations = _build_citations(bare_chunks, case_chunks)

    logger.info(
        "Stage5 complete | doc_type=%s | bare_chunks=%d | case_chunks=%d | draft_len=%d | total=%.0fs",
        doc_type,
        len(bare_chunks),
        len(case_chunks),
        len(draft_text),
        time.perf_counter() - t_total,
    )

    if step_callback:
        step_callback({"message": "Draft complete — ready for advocate review", "icon": "✓"})

    return {
        "draft_text":          stage5_state["draft_text"],
        "stage5_state":        stage5_state,
        "retrieved_bare_acts": bare_chunks,
        "retrieved_case_laws": case_chunks,
        "document_type":       doc_type,
        "document_type_label": doc_label,
        "citations":           citations,
    }
