"""
Legal Draft Stage 5 — Structured Draft Builder

Assembles the final two-part output from the intake state + retrieved materials:

  formatted_draft  — Markdown-formatted draft document with:
                       1. Statement of Facts
                       2. Legal Framework  (verbatim statutory subsection text)
                       3. Case Law Support (paragraph_num + verbatim judgment text)
                       4. Prayer / Relief Sought
                       5. Documents Checklist

  advocate_review  — Structured JSON brief for the advocate-review UI panel:
                       {disputes[], summary_facts[], overall_prayer[], overall_documents_checklist[]}

Items 15-18 of the implementation plan:
  15 — Statement of Facts from structured known_facts with evidence hooks
  16 — Legal Framework: verbatim subsection text passed to LLM draft prompt
  17 — Case Law Support: paragraph_num + verbatim judgment text per section
  18 — Dual output (formatted_draft + advocate_review JSON)

Public API
----------
  build_legal_draft(
      intake_state, bare_act_sections, facts_summary, model_override
  ) → {"formatted_draft": str, "advocate_review": dict}
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM import
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "quality") -> str:
    from llm.ollama_client import ask_llm
    return (ask_llm(prompt, task_hint=task_hint) or "").strip()


# ---------------------------------------------------------------------------
# LLM prompt: Prayer + Documents Checklist
# ---------------------------------------------------------------------------

_PRAYER_AND_DOCS_PROMPT = """You are a senior Indian legal advocate drafting the Prayer and Documents Checklist sections of a legal opinion.

FACTS SUMMARY:
{facts_summary}

DISPUTES AND APPLICABLE LAW:
{disputes_text}

STATED REMEDY (what the client wants):
{stated_remedy}

ASSESSED REMEDY (what is practically achievable):
{assessed_remedy}

Output ONLY valid JSON with this exact structure:
{{
  "prayer": [
    {{"relief": "...", "forum": "...", "urgency": "immediate|normal"}}
  ],
  "documents_checklist": [
    {{"document": "...", "purpose": "...", "priority": "essential|supporting|optional"}}
  ]
}}

RULES:
- prayer: list up to 5 specific reliefs, each with the recommended forum (e.g. "Protection Officer", "Family Court", "Labour Court")
- documents_checklist: list 6–10 documents; mark essential ones (without which the case cannot proceed)
- No preamble. Output JSON only."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_facts_section(intake_state: dict, facts_summary: str) -> tuple[str, list]:
    """
    Build Statement of Facts markdown text and structured facts list.

    Uses structured known_facts objects when available; falls back to plain
    facts_summary string.
    """
    known_facts: list[dict] = [
        f for f in (intake_state.get("known_facts") or [])
        if isinstance(f, dict) and f.get("fact")
    ]

    structured: list[dict] = []
    md_lines: list[str] = []

    if known_facts:
        for i, f in enumerate(known_facts, 1):
            fact_text = f.get("fact", "").strip()
            if not fact_text:
                continue
            turn     = f.get("source_turn", "")
            ftype    = f.get("fact_type", "other")
            timeref  = f.get("time_reference") or ""
            evidence = f.get("evidence_hook") or ""
            conf     = f.get("confidence_seed", "medium")

            # Markdown bullet
            suffix_parts = []
            if timeref:
                suffix_parts.append(f"*{timeref}*")
            if evidence:
                suffix_parts.append(f"[Evidence: {evidence}]")
            suffix = "  " + "  ".join(suffix_parts) if suffix_parts else ""
            md_lines.append(f"{i}. {fact_text}{suffix}")

            structured.append({
                "index":        i,
                "fact":         fact_text,
                "fact_type":    ftype,
                "source_turn":  turn,
                "time_reference": timeref or None,
                "evidence_hook":  evidence or None,
                "confidence":   conf,
            })
    else:
        # Fallback: split plain facts_summary into numbered lines
        for i, line in enumerate(
            (l.strip() for l in (facts_summary or "").split("\n") if l.strip()), 1
        ):
            md_lines.append(f"{i}. {line}")
            structured.append({"index": i, "fact": line})

    md = "## Statement of Facts\n\n" + "\n".join(md_lines) if md_lines else ""
    return md, structured


def _build_legal_framework_section(bare_act_sections: list) -> tuple[str, list]:
    """
    Build Legal Framework section from bare_act_sections.

    Groups by dispute (_dispute_label), picks up to 3 sections per dispute,
    and includes the verbatim statutory text under each section heading
    (Item 16: verbatim subsection text in the draft).
    """
    if not bare_act_sections:
        return "", []

    # Group by dispute
    from collections import defaultdict, OrderedDict
    by_dispute: dict = OrderedDict()
    for ba in bare_act_sections:
        d_label = (ba.get("_dispute_label") or ba.get("_dispute_id") or "General").strip()
        if d_label not in by_dispute:
            by_dispute[d_label] = []
        by_dispute[d_label].append(ba)

    md_parts: list[str] = ["## Legal Framework"]
    structured_disputes: list[dict] = []

    for dispute_label, sections in by_dispute.items():
        # Cap at 3 per dispute (Item 16)
        top3 = sections[:3]

        md_parts.append(f"\n### {dispute_label}\n")
        dispute_sections_structured: list[dict] = []

        for ba in top3:
            act      = (ba.get("act_name") or "").strip()
            sec_num  = str(ba.get("section_number") or "").strip()
            sec_title = (ba.get("section_title") or "").strip()
            text     = (ba.get("text") or ba.get("full_text") or "").strip()

            # Heading: Act — Section N (Title)
            heading_parts = [act]
            if sec_num:
                heading_parts.append(f"Section {sec_num}")
            if sec_title:
                heading_parts.append(f"({sec_title})")
            heading = " — ".join(heading_parts[:2])
            if sec_title:
                heading += f" ({sec_title})"

            # Truncate verbatim text to 1500 chars per section (budget ceiling)
            verbatim = text[:1500] if text else "(text unavailable)"

            md_parts.append(f"**{heading}**\n\n> {verbatim}\n")

            dispute_sections_structured.append({
                "act_name":      act,
                "section_number": sec_num,
                "section_title":  sec_title,
                "verbatim_text":  verbatim,
                "url":            ba.get("url", ""),
                "source_tag":     ba.get("source_tag", "LOCAL_DB"),
                "rerank_score":   round(ba.get("_rerank_score", 0), 3),
            })

        structured_disputes.append({
            "dispute_label": dispute_label,
            "sections":      dispute_sections_structured,
        })

    return "\n".join(md_parts), structured_disputes


def _build_case_law_section(bare_act_sections: list) -> tuple[str, list]:
    """
    Build Case Law Support section.

    Iterates related_case_laws on each bare_act_section.  Includes paragraph_num
    and verbatim judgment text per case (Item 17).

    Deduplicates by case_name so the same case doesn't appear twice across sections.
    """
    seen_cases: set[str] = set()
    md_parts: list[str] = ["## Case Law Support"]
    structured: list[dict] = []
    has_any = False

    for ba in bare_act_sections:
        related = ba.get("related_case_laws") or []
        if not related:
            continue

        act     = (ba.get("act_name") or "").strip()
        sec_num = str(ba.get("section_number") or "").strip()
        section_label = f"{act} § {sec_num}" if sec_num else act

        for cl in related:
            case_name  = (cl.get("case_name") or cl.get("title") or cl.get("source") or "Unknown Case").strip()
            court      = (cl.get("court") or "").strip()
            year       = str(cl.get("year") or "").strip()
            para_num   = str(cl.get("paragraph_num") or "").strip()
            binding    = (cl.get("binding_authority") or cl.get("binding") or "").strip()
            raw_text   = (
                cl.get("search_text") or cl.get("full_text") or cl.get("text") or ""
            ).strip()

            dedup_key = case_name.lower()
            if dedup_key in seen_cases:
                continue
            seen_cases.add(dedup_key)
            has_any = True

            # Citation line
            citation_parts = [case_name]
            if court:
                citation_parts.append(court)
            if year:
                citation_parts.append(year)
            citation = ", ".join(citation_parts)
            if binding:
                citation += f" [{binding}]"

            para_label = f"¶{para_num}" if para_num else ""

            # Verbatim judgment text — up to 1500 chars
            verbatim = raw_text[:1500] if raw_text else "(text unavailable)"

            md_parts.append(f"\n**{citation}**  \n*Relevant to: {section_label}*")
            if para_label:
                md_parts.append(f"*Paragraph {para_num}:*")
            md_parts.append(f"\n> {verbatim}\n")

            structured.append({
                "case_name":       case_name,
                "court":           court,
                "year":            year,
                "paragraph_num":   para_num or None,
                "binding_authority": binding or None,
                "verbatim_text":   verbatim,
                "relevant_to_section": section_label,
            })

    if not has_any:
        return "", []

    return "\n".join(md_parts), structured


def _generate_prayer_and_docs(
    facts_summary: str,
    structured_disputes: list,
    intake_state: dict,
    model_override: str | None,
) -> tuple[str, str, list, list]:
    """
    LLM call to produce Prayer and Documents Checklist.
    Returns (prayer_md, docs_md, prayer_structured, docs_structured).
    """
    remedy_block   = (intake_state or {}).get("assessed_remedy") or {}
    stated_remedy  = (intake_state or {}).get("stated_remedy") or "not stated"
    assessed_rem   = (
        remedy_block.get("assessed_remedy")
        or (intake_state or {}).get("client_goal_initial")
        or "not assessed"
    )

    # Summarise disputes and their sections for the prompt
    disputes_lines: list[str] = []
    for d in (structured_disputes or []):
        label = d.get("dispute_label", "")
        secs  = [
            f"{s.get('act_name')} Section {s.get('section_number')} ({s.get('section_title', '')})"
            for s in (d.get("sections") or [])
        ]
        disputes_lines.append(f"- {label}: {'; '.join(secs)}")
    disputes_text = "\n".join(disputes_lines) or "No disputes identified"

    prompt = (
        _PRAYER_AND_DOCS_PROMPT
        .replace("{facts_summary}",  (facts_summary or "")[:600])
        .replace("{disputes_text}",  disputes_text[:600])
        .replace("{stated_remedy}",  str(stated_remedy)[:200])
        .replace("{assessed_remedy}", str(assessed_rem)[:200])
    )

    prayer_structured: list[dict] = []
    docs_structured:   list[dict] = []

    try:
        raw = _ask_llm(prompt, task_hint="quality")
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            prayer_structured = data.get("prayer") or []
            docs_structured   = data.get("documents_checklist") or []
    except Exception as exc:
        logger.warning("Stage5 prayer/docs LLM failed: %s", exc)

    # Deterministic fallbacks
    if not prayer_structured:
        prayer_structured = [
            {"relief": assessed_rem or "relief as appropriate", "forum": "competent court", "urgency": "normal"}
        ]
    if not docs_structured:
        # Pull from evidence_hooks in known_facts
        hooks = list({
            f.get("evidence_hook")
            for f in ((intake_state or {}).get("known_facts") or [])
            if isinstance(f, dict) and f.get("evidence_hook")
        })
        docs_structured = [
            {"document": h, "purpose": "evidentiary support", "priority": "essential"}
            for h in (hooks or ["Identity proof", "Incident report"])
        ]

    # Build markdown
    prayer_lines = ["## Prayer / Relief Sought\n"]
    for i, p in enumerate(prayer_structured, 1):
        relief  = p.get("relief", "")
        forum   = p.get("forum", "")
        urgency = p.get("urgency", "normal")
        tag     = " *(urgent)*" if urgency == "immediate" else ""
        prayer_lines.append(f"{i}. {relief} — *Forum: {forum}*{tag}")

    docs_lines = ["## Documents Checklist\n"]
    for d in docs_structured:
        doc      = d.get("document", "")
        purpose  = d.get("purpose", "")
        priority = d.get("priority", "supporting")
        tick     = "☑" if priority == "essential" else "☐"
        docs_lines.append(f"- {tick} **{doc}** — {purpose}")

    prayer_md = "\n".join(prayer_lines)
    docs_md   = "\n".join(docs_lines)

    return prayer_md, docs_md, prayer_structured, docs_structured


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_legal_draft(
    intake_state: Optional[dict] = None,
    bare_act_sections: Optional[list] = None,
    facts_summary: str = "",
    model_override: Optional[str] = None,
) -> dict:
    """
    Build the full structured legal draft from intake state and retrieved materials.

    Parameters
    ----------
    intake_state : dict | None
        Stage 1 intake state (known_facts, remedies, etc.).
    bare_act_sections : list | None
        Retrieved bare act sections, each with optional 'related_case_laws' list.
        Each section must have: act_name, section_number, section_title, text,
        _dispute_label, related_case_laws[].
    facts_summary : str
        Plain-text facts string (fallback when intake_state has no known_facts).
    model_override : str | None
        Optional model string to pass to the LLM.

    Returns
    -------
    dict with keys:
      "formatted_draft"  : str  — complete Markdown draft
      "advocate_review"  : dict — structured JSON for the UI advocate-review panel
    """
    t0 = time.perf_counter()
    state     = intake_state or {}
    sections  = bare_act_sections or []

    # ── 1. Statement of Facts ──────────────────────────────────────────────
    facts_md, facts_structured = _build_facts_section(state, facts_summary)

    # ── 2. Legal Framework (verbatim statutory text, 3 per dispute) ────────
    framework_md, disputes_structured = _build_legal_framework_section(sections)

    # ── 3. Case Law Support (paragraph_num + verbatim judgment text) ───────
    caselaw_md, caselaw_structured = _build_case_law_section(sections)

    # ── 4. Prayer + 5. Documents Checklist (single LLM call) ───────────────
    prayer_md, docs_md, prayer_structured, docs_structured = _generate_prayer_and_docs(
        facts_summary=facts_summary or "\n".join(f["fact"] for f in facts_structured if "fact" in f),
        structured_disputes=disputes_structured,
        intake_state=state,
        model_override=model_override,
    )

    # ── Assemble Markdown draft ────────────────────────────────────────────
    primary   = state.get("primary_issue_cluster") or "Legal Matter"
    heading   = f"# Legal Opinion — {primary.replace('_', ' ').title()}\n"
    separator = "\n---\n"

    sections_md = [heading, facts_md, framework_md]
    if caselaw_md:
        sections_md.append(caselaw_md)
    sections_md += [prayer_md, docs_md]

    formatted_draft = separator.join(s for s in sections_md if s)

    # ── Assemble advocate-review JSON ──────────────────────────────────────
    advocate_review = {
        "primary_category":          primary,
        "urgency_signal":            state.get("urgency_signal", "normal"),
        "summary_facts":             facts_structured,
        "disputes":                  [
            {
                **d,
                "case_laws": [
                    cl for cl in caselaw_structured
                    if any(
                        s.get("act_name", "") in cl.get("relevant_to_section", "")
                        for s in d.get("sections", [])
                    )
                ],
            }
            for d in disputes_structured
        ],
        "all_case_laws":             caselaw_structured,
        "overall_prayer":            prayer_structured,
        "overall_documents_checklist": docs_structured,
        "stated_remedy":             state.get("stated_remedy"),
        "assessed_remedy":           (state.get("assessed_remedy") or {}).get("assessed_remedy"),
        "recommended_lead":          (state.get("assessed_remedy") or {}).get("recommended_lead"),
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Stage5 draft built in %.0f ms | disputes=%d sections=%d cases=%d",
        elapsed_ms,
        len(disputes_structured),
        len(sections),
        len(caselaw_structured),
    )

    return {
        "formatted_draft": formatted_draft,
        "advocate_review": advocate_review,
    }
