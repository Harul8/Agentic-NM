"""
Legal Opinion — Stage 6: Advocate Review

The final stage. The AI's role shifts completely:
  - Audience is the SENIOR ADVOCATE, not the client
  - AI acts as a junior associate who prepared the draft
  - The advocate can question, revise, research, and finalise

Capabilities per turn
---------------------
  qa         — Answer advocate's questions about the case, strategy, draft choices
  revise     — Rewrite a specific section per the advocate's instruction
  research   — Find additional case laws / statutes on a specific legal point
  strengthen — Make an argument stronger (research + targeted revision)
  new_fact   — Incorporate a new fact the advocate provides
  finalize   — Close out and confirm the document is ready to file
  other      — General conversational response

Public API
----------
  new_stage6_state(stage5_state, draft_text)  → dict
  generate_stage6_opening(stage5_state, model_override)  → str
  process_turn(session, advocate_message, model_override)  → dict
      Returns: {reply, updated_draft, stage6_state, finalized}
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
# LLM helpers
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "fast", model_override: str | None = None) -> str:
    from llm.ollama_client import ask_llm
    return ask_llm(prompt, task_hint=task_hint, model=model_override) or ""


def _ask_llm_quality(prompt: str, model_override: str | None = None) -> str:
    return _ask_llm(prompt, task_hint="quality", model_override=model_override)


# ---------------------------------------------------------------------------
# Prompt imports
# ---------------------------------------------------------------------------

from prompts.advocate_prompts import (
    STAGE6_INTENT_CLASSIFIER_SYSTEM,
    STAGE6_QA_SYSTEM,
    STAGE6_REVISION_SYSTEM,
    STAGE6_RESEARCH_SUMMARY_SYSTEM,
    STAGE6_FINALIZE_SYSTEM,
    STAGE6_STATE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Debug timing
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _t(label: str, t0: float) -> None:
    if _TIMING:
        logger.info("PIPELINE_TIMING stage6.%s: %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if "{" in part and "}" in part:
                text = part
                break
    start = text.find("{")
    end   = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(session: dict, max_turns: int = 8, max_chars: int = 400) -> str:
    history = session.get("history", [])
    lines: list[str] = []
    for turn in history[-max_turns:]:
        role    = "Advocate" if turn.get("role") == "user" else "AI Associate"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "…"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Case summary builder
# ---------------------------------------------------------------------------

def _build_case_summary(stage6_state: dict) -> str:
    """Compact summary of the case for advocate-facing prompts."""
    parts = []
    category = stage6_state.get("category_label") or stage6_state.get("category") or "General"
    parts.append(f"Matter type: {category}")
    if stage6_state.get("jurisdiction"):
        parts.append(f"Jurisdiction: {stage6_state['jurisdiction']}")
    summary = stage6_state.get("case_summary")
    if summary:
        parts.append(summary)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Draft excerpt helper (avoids feeding 4000-word draft to every prompt)
# ---------------------------------------------------------------------------

def _draft_excerpt(draft: str, max_chars: int = 3000) -> str:
    """Return the draft, truncated to max_chars with a note if cut."""
    if not draft:
        return "(draft not available)"
    if len(draft) <= max_chars:
        return draft
    return draft[:max_chars].rstrip() + f"\n\n[… draft continues — {len(draft) - max_chars} more characters …]"


# ---------------------------------------------------------------------------
# Retrieval helpers (reuse from stage 5)
# ---------------------------------------------------------------------------

def _retrieve_bare_acts(query: str, top_k: int = 5) -> list[dict]:
    try:
        from retrieval.hybrid_retriever import search_bare_acts_auto
        return search_bare_acts_auto(query, top_k=top_k) or []
    except Exception as exc:
        logger.warning("Stage6 bare act retrieval failed: %s", exc)
        return []


def _retrieve_case_laws(query: str, top_k: int = 5) -> list[dict]:
    try:
        from retrieval.hybrid_retriever import search_case_laws_auto
        return search_case_laws_auto(query, top_k=top_k) or []
    except Exception as exc:
        logger.warning("Stage6 case law retrieval failed: %s", exc)
        return []


def _format_chunk(chunk: dict, kind: str) -> str:
    if kind == "bare":
        act  = chunk.get("act_name") or chunk.get("source") or "Act"
        sec  = chunk.get("section") or chunk.get("section_title") or ""
        text = (chunk.get("text") or chunk.get("content") or "").strip()
        return f"[{act}{' — ' + sec if sec else ''}]\n{text}"
    else:
        case  = chunk.get("case_name") or chunk.get("source") or "Judgment"
        year  = chunk.get("year") or ""
        court = chunk.get("court") or ""
        text  = (chunk.get("text") or chunk.get("content") or "").strip()
        return f"[{case}{' — ' + court if court else ''}{' (' + year + ')' if year else ''}]\n{text}"


def _run_research(query: str) -> tuple[list[dict], list[dict]]:
    """Run bare-act + case-law retrieval in parallel for a given research query."""
    bare: list[dict] = []
    case: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_bare = pool.submit(_retrieve_bare_acts, query, 5)
        f_case = pool.submit(_retrieve_case_laws, query, 5)
        for fut in as_completed([f_bare, f_case]):
            if fut is f_bare:
                bare = fut.result()
            else:
                case = fut.result()
    return bare, case


# ===========================================================================
# Public: new_stage6_state
# ===========================================================================

def new_stage6_state(stage5_state: dict, draft_text: str = "") -> dict:
    """Create Stage 6 state from Stage 5 final state and the generated draft."""
    state = copy.deepcopy(STAGE6_STATE_SCHEMA)
    remedy_plan = stage5_state.get("remedy_plan") or {}

    # Build compact case summary from known facts + remedy plan
    known_facts = stage5_state.get("known_facts") or []
    facts_text  = "; ".join(known_facts[:5]) or "(see draft)"
    strategy    = remedy_plan.get("strategy_note") or ""
    lead        = remedy_plan.get("lead_remedy") or ""
    summary_parts = [f"Facts: {facts_text}"]
    if lead:
        summary_parts.append(f"Lead remedy: {lead}")
    if strategy:
        summary_parts.append(f"Strategy: {strategy}")

    state.update({
        "category":            stage5_state.get("category") or "general",
        "category_label":      stage5_state.get("category_label") or "General Matter",
        "jurisdiction":        stage5_state.get("jurisdiction") or "India",
        "document_type":       stage5_state.get("document_type"),
        "document_type_label": stage5_state.get("document_type_label") or "Legal Document",
        "current_draft":       draft_text.strip(),
        "original_draft":      draft_text.strip(),
        "case_summary":        "\n".join(summary_parts),
        "strategy_note":       strategy,
    })
    return state


# ===========================================================================
# Public: generate_stage6_opening
# ===========================================================================

def generate_stage6_opening(
    stage5_state: dict,
    draft_text: str = "",
    model_override: str | None = None,
) -> str:
    """
    Generate the opening message the AI sends to the advocate when handing
    over the draft for review.
    """
    doc_label  = stage5_state.get("document_type_label") or "Legal Document"
    category   = stage5_state.get("category_label") or "matter"
    remedy_plan = stage5_state.get("remedy_plan") or {}
    lead       = remedy_plan.get("lead_remedy") or "the primary relief"
    caveat     = remedy_plan.get("honest_caveat") or ""
    known_facts = stage5_state.get("known_facts") or []
    fact_count  = len(known_facts)

    vr  = stage5_state.get("vetting_report") or {}
    unc = vr.get("uncertain_facts") or []
    gaps = vr.get("evidence_gaps") or []

    prompt = (
        f"You are an AI legal associate presenting a completed draft to a senior advocate for review.\n\n"
        f"DOCUMENT PREPARED: {doc_label}\n"
        f"MATTER: {category}\n"
        f"PRIMARY RELIEF: {lead}\n"
        f"FACTS GATHERED: {fact_count} facts across intake stages\n"
        f"UNCERTAIN FACTS: {', '.join(unc[:3]) if unc else 'none'}\n"
        f"EVIDENCE GAPS: {', '.join(gaps[:3]) if gaps else 'none'}\n"
        f"CAVEAT: {caveat or 'none'}\n\n"
        f"Write a brief professional handover note to the senior advocate (5–8 sentences):\n"
        f"1. Confirm the document is ready for review\n"
        f"2. Summarise what was drafted and the lead remedy\n"
        f"3. Flag any uncertain facts or evidence gaps they should verify\n"
        f"4. Note the caveat if any\n"
        f"5. Invite them to ask questions, request revisions, or mark the document as final\n\n"
        f"Professional tone — you are speaking to a senior colleague, not a client.\n"
        f"Output ONLY the handover note."
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        if reply and len(reply) > 80:
            return reply
    except Exception as exc:
        logger.warning("Stage6 opening generation failed: %s", exc)

    # Structured fallback
    flags = []
    if unc:
        flags.append(f"uncertain facts: {', '.join(unc[:2])}")
    if gaps:
        flags.append(f"evidence gaps: {', '.join(gaps[:2])}")
    flag_str = " — please verify: " + "; ".join(flags) if flags else ""
    return (
        f"The {doc_label} has been prepared and is ready for your review{flag_str}. "
        f"The document leads with {lead}. "
        f"You can ask me questions about the drafting choices, request revisions to any section, "
        f"ask me to find stronger case laws on any point, or mark the document as final when you're satisfied."
    )


# ===========================================================================
# Internal: classify advocate's intent
# ===========================================================================

def _classify_intent(
    stage6_state: dict,
    advocate_message: str,
    model_override: str | None = None,
) -> dict:
    t0 = time.perf_counter()

    prompt = (
        STAGE6_INTENT_CLASSIFIER_SYSTEM
        .replace("{document_type_label}", stage6_state.get("document_type_label") or "Legal Document")
        .replace("{category_label}",      stage6_state.get("category_label") or "General")
        .replace("{advocate_message}",    (advocate_message or "").strip())
    )

    _DEFAULT = {
        "intent": "qa",
        "target_section": None,
        "specific_instruction": advocate_message,
        "research_query": None,
    }

    try:
        raw = _ask_llm(prompt, task_hint="fast", model_override=model_override)
        _t("classify_intent", t0)
        out = _extract_json(raw)
        if out and isinstance(out, dict):
            return out
    except Exception as exc:
        logger.warning("Stage6 intent classification failed: %s", exc)

    # Heuristic fallback
    low = advocate_message.lower()
    if any(w in low for w in ("final", "approve", "done", "file it", "sign off", "ready")):
        _DEFAULT["intent"] = "finalize"
    elif any(w in low for w in ("revise", "change", "update", "rewrite", "modify", "edit", "fix", "replace", "add")):
        _DEFAULT["intent"] = "revise"
    elif any(w in low for w in ("case", "judgment", "citation", "authority", "find", "search", "more")):
        _DEFAULT["intent"] = "research"
    elif any(w in low for w in ("stronger", "strengthen", "weaker", "better argument")):
        _DEFAULT["intent"] = "strengthen"
    return _DEFAULT


# ===========================================================================
# Internal: handle qa intent
# ===========================================================================

def _handle_qa(
    stage6_state: dict,
    advocate_message: str,
    context: str,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()
    current_draft = stage6_state.get("current_draft") or ""

    prompt = (
        STAGE6_QA_SYSTEM
        .replace("{document_type_label}", stage6_state.get("document_type_label") or "Legal Document")
        .replace("{category_label}",      stage6_state.get("category_label") or "General")
        .replace("{jurisdiction}",        stage6_state.get("jurisdiction") or "India")
        .replace("{case_summary}",        _build_case_summary(stage6_state))
        .replace("{strategy_note}",       stage6_state.get("strategy_note") or "(see draft)")
        .replace("{draft_excerpt}",       _draft_excerpt(current_draft, max_chars=2500))
        .replace("{advocate_message}",    (advocate_message or "").strip())
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("qa", t0)
        if reply and len(reply) > 10:
            return reply
    except Exception as exc:
        logger.warning("Stage6 QA failed: %s", exc)

    return "I'll need to review the draft more carefully to answer that. Could you point me to the specific section or paragraph you're asking about?"


# ===========================================================================
# Internal: handle revise / new_fact intent
# ===========================================================================

def _handle_revision(
    stage6_state: dict,
    instruction: dict,
    advocate_message: str,
    additional_materials_text: str = "",
    model_override: str | None = None,
) -> tuple[str, str]:
    """
    Returns (reply_to_advocate, revised_section_text).
    The caller splices the revised section back into current_draft if appropriate.
    """
    t0 = time.perf_counter()
    current_draft = stage6_state.get("current_draft") or ""
    target        = instruction.get("target_section") or "the relevant section"
    specific_inst = instruction.get("specific_instruction") or advocate_message

    # Extract the section being revised if we can identify it
    section_to_revise = _extract_section(current_draft, target)

    prompt = (
        STAGE6_REVISION_SYSTEM
        .replace("{document_type_label}",   stage6_state.get("document_type_label") or "Legal Document")
        .replace("{category_label}",        stage6_state.get("category_label") or "General")
        .replace("{jurisdiction}",          stage6_state.get("jurisdiction") or "India")
        .replace("{case_summary}",          _build_case_summary(stage6_state))
        .replace("{section_to_revise}",     section_to_revise)
        .replace("{revision_instruction}",  specific_inst)
        .replace("{additional_materials}",  additional_materials_text or "(none)")
    )

    try:
        revised_text = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("revision", t0)
        if revised_text and len(revised_text) > 20:
            reply = (
                f"I've revised {target}. Here's the updated text:\n\n"
                f"---\n{revised_text}\n---\n\n"
                "Please review and let me know if you'd like further changes, or ask me to apply this to the document."
            )
            return reply, revised_text
    except Exception as exc:
        logger.warning("Stage6 revision failed: %s", exc)

    return "I had difficulty generating the revision. Could you clarify which section you'd like changed and exactly how?", ""


# ===========================================================================
# Internal: extract section from draft
# ===========================================================================

def _extract_section(draft: str, target: str | None) -> str:
    """Try to extract a named section from the draft for targeted revision."""
    if not target or not draft:
        return _draft_excerpt(draft, max_chars=2000)

    target_lower = target.lower()
    # Common section headers to look for
    section_keywords = {
        "prayer": ["prayer", "relief sought", "d. relief", "d. prayer"],
        "facts":  ["facts of the case", "a. facts", "b. facts"],
        "grounds": ["legal grounds", "b. legal grounds", "grounds"],
        "case laws": ["case laws relied", "c. case laws"],
    }

    # Try to find the matching section in the draft
    lines = draft.split("\n")
    start_idx: int | None = None
    end_idx:   int | None = None

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        for kw_group, keywords in section_keywords.items():
            if kw_group in target_lower or any(k in target_lower for k in keywords):
                if any(k in line_lower for k in keywords):
                    start_idx = i
                    break
        if start_idx is not None:
            break

    if start_idx is not None:
        # Find end of section (next heading or end of document)
        for i in range(start_idx + 1, len(lines)):
            line_stripped = lines[i].strip()
            if line_stripped.startswith("**") and line_stripped.endswith("**") and len(line_stripped) > 4:
                end_idx = i
                break
        section = "\n".join(lines[start_idx:end_idx or (start_idx + 40)])
        return section[:2000]

    # Fallback: return the whole draft excerpt
    return _draft_excerpt(draft, max_chars=2000)


# ===========================================================================
# Internal: apply a revision to the current draft
# ===========================================================================

def _apply_revision_to_draft(draft: str, target: str | None, revised_text: str) -> str:
    """
    Attempt to splice revised_text into the draft at the target section.
    If we can't locate the target precisely, append a note at the end.
    """
    if not draft or not revised_text:
        return draft

    target_lower = (target or "").lower()
    lines = draft.split("\n")
    start_idx: int | None = None
    end_idx:   int | None = None

    # Simple heading matcher
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        if target_lower and (target_lower in ll or ll in target_lower):
            start_idx = i
            break

    if start_idx is not None:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip().startswith("**") and lines[i].strip().endswith("**") and len(lines[i].strip()) > 4:
                end_idx = i
                break
        end_idx = end_idx or (start_idx + 40)
        new_lines = lines[:start_idx + 1] + ["", revised_text, ""] + lines[end_idx:]
        return "\n".join(new_lines)

    # Can't locate — append as amendment note
    return draft + f"\n\n---\n**AMENDED SECTION — {target or 'Revision'}:**\n{revised_text}\n---"


# ===========================================================================
# Internal: handle research / strengthen intent
# ===========================================================================

def _handle_research(
    stage6_state: dict,
    research_query: str,
    model_override: str | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run retrieval + format research note.
    Returns (reply, bare_chunks, case_chunks).
    """
    t0 = time.perf_counter()

    if not research_query:
        research_query = (
            f"{stage6_state.get('category_label', '')} "
            f"{stage6_state.get('document_type_label', '')} legal grounds"
        )

    bare_chunks, case_chunks = _run_research(research_query)
    _t("retrieval", t0)

    bare_text = "\n\n".join(
        f"[{c.get('act_name') or c.get('source', 'Act')} — {c.get('section', '')}]\n{(c.get('text') or c.get('content', '')).strip()}"
        for c in bare_chunks[:5]
    ) or "(none found)"

    case_text = "\n\n".join(
        f"[{c.get('case_name') or c.get('source', 'Case')} — {c.get('court', '')} ({c.get('year', '')})]\n{(c.get('text') or c.get('content', '')).strip()}"
        for c in case_chunks[:5]
    ) or "(none found)"

    prompt = (
        STAGE6_RESEARCH_SUMMARY_SYSTEM
        .replace("{research_query}",   research_query)
        .replace("{category_label}",   stage6_state.get("category_label") or "General")
        .replace("{retrieved_bare_acts}", bare_text)
        .replace("{retrieved_case_laws}", case_text)
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("research_summary", t0)
        if reply and len(reply) > 40:
            return reply, bare_chunks, case_chunks
    except Exception as exc:
        logger.warning("Stage6 research summary failed: %s", exc)

    summary_lines = []
    if bare_chunks:
        summary_lines.append("**Statutory provisions found:**")
        for c in bare_chunks[:3]:
            summary_lines.append(f"- {c.get('act_name') or c.get('source', '')} — {c.get('section', '')}")
    if case_chunks:
        summary_lines.append("\n**Case laws found:**")
        for c in case_chunks[:3]:
            summary_lines.append(f"- {c.get('case_name') or c.get('source', '')} ({c.get('year', '')})")

    return "\n".join(summary_lines) or "No relevant materials found.", bare_chunks, case_chunks


# ===========================================================================
# Internal: handle finalize intent
# ===========================================================================

def _handle_finalize(
    stage6_state: dict,
    model_override: str | None = None,
) -> str:
    t0 = time.perf_counter()

    prompt = (
        STAGE6_FINALIZE_SYSTEM
        .replace("{document_type_label}", stage6_state.get("document_type_label") or "Legal Document")
        .replace("{category_label}",      stage6_state.get("category_label") or "General")
    )

    try:
        reply = _ask_llm_quality(prompt, model_override=model_override).strip()
        _t("finalize", t0)
        if reply and len(reply) > 40:
            return reply
    except Exception as exc:
        logger.warning("Stage6 finalize response failed: %s", exc)

    return (
        f"The {stage6_state.get('document_type_label', 'document')} is now finalised. "
        "Before filing, please verify the party names, addresses, and dates against the client's original documents. "
        "Ensure the court seal and case number placeholder are updated. "
        "A covering letter to the court registry may also be required."
    )


# ===========================================================================
# Public: process_turn
# ===========================================================================

def process_turn(
    session: dict,
    advocate_message: str,
    model_override: str | None = None,
) -> dict:
    """
    Process one advocate turn in Stage 6 review.

    Parameters
    ----------
    session : dict
        Must contain:
          - "history"       : list of {role, content}
          - "stage6_state"  : running Stage 6 state
    advocate_message : str
    model_override : str | None

    Returns
    -------
    dict:
      - "reply"            : str   — response to the advocate
      - "updated_draft"    : str   — current_draft (may have been revised)
      - "stage6_state"     : dict
      - "finalized"        : bool
      - "intent"           : str   — what was detected
    """
    msg = (advocate_message or "").strip()

    stage6_state: dict = session.get("stage6_state") or {}
    if not stage6_state.get("category"):
        logger.warning("Stage6 process_turn called without initialised stage6_state")
        stage6_state = copy.deepcopy(STAGE6_STATE_SCHEMA)
        stage6_state["category"] = "general"

    stage6_state["turn_count"] = stage6_state.get("turn_count", 0) + 1
    context = _build_context(session, max_turns=8)

    # ── 1. Classify intent ────────────────────────────────────────────────────
    intent_data = _classify_intent(stage6_state, msg, model_override=model_override)
    intent      = intent_data.get("intent", "qa")
    target_sec  = intent_data.get("target_section")
    specific_inst = intent_data.get("specific_instruction") or msg
    research_q  = intent_data.get("research_query")

    reply = ""
    revised_text = ""
    bare_found: list[dict] = []
    case_found: list[dict] = []

    # ── 2. Dispatch by intent ─────────────────────────────────────────────────
    if intent == "finalize":
        reply = _handle_finalize(stage6_state, model_override=model_override)
        stage6_state["finalized"] = True

    elif intent in ("research", "strengthen"):
        query = research_q or specific_inst or msg
        research_note, bare_found, case_found = _handle_research(
            stage6_state, query, model_override=model_override
        )
        # For "strengthen", follow up with revision using the new material
        if intent == "strengthen" and target_sec:
            bare_text = "\n\n".join(f"[{c.get('act_name') or c.get('source','')}]\n{(c.get('text') or '').strip()}" for c in bare_found[:3])
            case_text = "\n\n".join(f"[{c.get('case_name') or c.get('source','')} ({c.get('year','')})]\n{(c.get('text') or '').strip()}" for c in case_found[:3])
            materials = "\n\n".join(filter(None, [bare_text, case_text]))
            rev_reply, revised_text = _handle_revision(
                stage6_state,
                {"target_section": target_sec, "specific_instruction": f"Strengthen this section using the additional materials provided"},
                msg,
                additional_materials_text=materials,
                model_override=model_override,
            )
            reply = f"**Research findings:**\n\n{research_note}\n\n---\n\n**Strengthened section:**\n\n{rev_reply}"
        else:
            reply = research_note

        # Store new materials in state
        stage6_state.setdefault("additional_case_laws", []).extend(case_found[:5])
        stage6_state.setdefault("additional_bare_acts", []).extend(bare_found[:5])

    elif intent in ("revise", "new_fact"):
        bare_text = ""
        case_text = ""
        # If we have additional materials already researched this session, include them
        existing_case = stage6_state.get("additional_case_laws") or []
        if existing_case and target_sec:
            case_text = "\n\n".join(
                f"[{c.get('case_name') or c.get('source','')}]\n{(c.get('text') or '').strip()}"
                for c in existing_case[:3]
            )
        materials = case_text or ""
        reply, revised_text = _handle_revision(
            stage6_state,
            intent_data,
            msg,
            additional_materials_text=materials,
            model_override=model_override,
        )

    else:  # qa / other
        reply = _handle_qa(stage6_state, msg, context, model_override=model_override)

    # ── 3. Apply revision to draft if we have revised text ────────────────────
    current_draft = stage6_state.get("current_draft") or ""
    if revised_text and current_draft:
        updated_draft = _apply_revision_to_draft(current_draft, target_sec, revised_text)
        stage6_state["current_draft"] = updated_draft
        # Audit trail
        stage6_state.setdefault("revision_history", []).append({
            "turn":           stage6_state["turn_count"],
            "instruction":    specific_inst[:200],
            "section_changed": target_sec,
            "revised_text":   revised_text[:500],
        })

    # ── 4. Store advocate notes if they provide info ─────────────────────────
    if intent == "new_fact":
        stage6_state.setdefault("advocate_notes", []).append(msg[:300])

    logger.info(
        "Stage6 turn %d | intent=%s | target=%s | finalized=%s",
        stage6_state["turn_count"],
        intent,
        target_sec,
        stage6_state.get("finalized"),
    )

    return {
        "reply":         reply,
        "updated_draft": stage6_state.get("current_draft") or current_draft,
        "stage6_state":  stage6_state,
        "finalized":     bool(stage6_state.get("finalized")),
        "intent":        intent,
    }
