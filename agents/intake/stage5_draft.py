"""
intake/stage5_draft.py — Stage 5: Structured draft builder.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from agents.intake.casefile import (
    compute_quality_review_signals,
    ensure_case_file_structure,
    set_pipeline_layer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM import
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str, task_hint: str = "quality") -> str:
    from platform_pkg.llm import ask_llm
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


_RELIEF_STRATEGY_PROMPT = """You are a senior Indian legal advocate preparing the decision layer of a legal opinion.

CASE FILE SUMMARY:
{case_summary}

FORUM ANALYSIS:
{forum_analysis}

LEGAL FRAMEWORK SUMMARY:
{legal_framework}

CASE LAW SUPPORT SUMMARY:
{case_law_summary}

YOUR TASK:
Return ONLY valid JSON with this exact structure:
{
  "relief_first_analysis": [
    {"relief": "...", "why_it_matters": "...", "legal_basis_note": "...", "support_level": "strong|moderate|limited"}
  ],
  "forum_selection": {
    "primary_forum": "...",
    "alternative_forum": "...",
    "why_primary_forum": "...",
    "forum_risk": "..."
  },
  "strategy_comparison": [
    {"route": "...", "best_for": "...", "tradeoff": "...", "speed": "fast|moderate|slow", "support_level": "strong|moderate|limited"}
  ],
  "practical_sequencing": [
    {"window": "today|next_7_days|before_filing|during_proceedings", "step": "...", "why": "..."}
  ],
  "draft_suitability": {
    "status": "ready|mostly_ready|not_ready",
    "score": 0,
    "blocking_gaps": ["..."],
    "why": "..."
  }
}

RULES:
- Think relief-first, not statute-first.
- Use the forum analysis where available; if uncertain, say so conservatively.
- Strategy comparison should reflect genuinely different routes or sequencing choices, not paraphrases.
- Practical sequencing should be concrete and action-oriented.
- Draft suitability should be strict: if core identity, timeline, supporting record, or forum clarity is weak, say so.
- Do not invent law beyond what the legal framework and case law summaries support.
- Keep every field concise and high signal."""


_COUNSEL_REVIEW_PROMPT = """You are senior counsel reviewing a client-facing legal opinion prepared by a junior advocate.

CASE FILE:
{case_file}

RELIEF AND FORUM ANALYSIS:
{relief_strategy}

CLIENT-FACING DRAFT:
{client_draft}

Return ONLY valid JSON with this exact structure:
{
  "approval": "pass|revise",
  "tone": "measured|needs_softening",
  "overstatement_risks": ["..."],
  "unsupported_points": ["..."],
  "relief_cautions": ["..."],
  "relevance_gaps": ["..."],
  "rewrite_notes": ["..."],
  "client_safe_note": "..."
}

RULES:
- Be strict about overstatement, missing proof, and relief that runs ahead of the record.
- Prefer concise, surgical feedback over broad criticism.
- If the draft is already measured, return empty lists where appropriate.
- Keep the client_safe_note calm, accurate, and responsibly worded."""


_RESEARCH_PACKET_PROMPT = """You are a senior Indian legal advocate organizing retrieved law into issue-wise research packets.

CASE FILE:
{case_file}

RELIEF AND FORUM ANALYSIS:
{relief_strategy}

DISPUTE-SECTION MAP:
{disputes_json}

CASE LAW SUPPORT:
{case_law_json}

Return ONLY valid JSON with this exact structure:
{
  "packets": [
    {
      "issue": "...",
      "relevant_reliefs": ["..."],
      "primary_sections": [
        {"act_name": "...", "section_number": "...", "section_title": "...", "why_it_matters": "..."}
      ],
      "case_law_ranked": [
        {
          "case_name": "...",
          "usefulness_for": "threshold issue|maintainability|burden of proof|interim relief|final relief|general support",
          "rank": "high|medium|low",
          "why": "..."
        }
      ],
      "likely_objections": ["..."],
      "strengthening_steps": [
        {"step": "...", "basis": "...", "why": "..."}
      ]
    }
  ]
}

RULES:
- Build packets issue-wise, not as one flat list of authorities.
- Tie sections and cases to the reliefs they most naturally support on the present record.
- Rank case law by practical usefulness, not prestige.
- Likely objections should come from the weak points in the record and the retrieved law.
- Strengthening steps must be tied to a section or case name where possible.
- Do not invent sections or cases not present in the supplied material.
- Keep the packets concise and high signal."""


_ACTION_DRAFTING_PROMPT = """You are a senior Indian legal advocate deciding which drafting actions, if any, should follow the present legal analysis.

CASE FILE:
{case_file}

RELIEF AND FORUM ANALYSIS:
{relief_strategy}

RESEARCH PACKETS:
{research_packets}

DOCUMENTS CHECKLIST:
{documents_json}

OVERALL PRAYER:
{prayer_json}

Return ONLY valid JSON with this exact structure:
{
  "analysis_status": "analysis_only|analysis_and_drafting",
  "overall_readiness": "ready|mostly_ready|not_ready",
  "readiness_reason": "...",
  "drafts": [
    {
      "draft_type": "FIR|complaint|notice|reply|application|affidavit_checklist|evidence_bundle_checklist|other",
      "recommended": true,
      "readiness": "ready|mostly_ready|not_ready",
      "forum": "...",
      "purpose": "...",
      "missing_fields": ["..."],
      "forum_specific_prayers": ["..."],
      "annexures": ["..."]
    }
  ],
  "hold_back_note": "..."
}

RULES:
- Keep legal analysis separate from action drafting. If the record is not ready for responsible drafting, return analysis_only.
- Recommend only the draft types that actually fit the current forum, relief, and evidentiary posture.
- Missing fields should identify legally important blockers such as identity, chronology, forum facts, prior steps, signatures, authorizations, or supporting materials.
- Forum-specific prayers must be tailored to the named forum or authority, not generic relief labels.
- Annexures should be practical document packs that can accompany the relevant draft.
- Do not force all listed draft types into every matter.
- Keep the output concise and decision-oriented."""


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
    remedy_block   = (intake_state or {}).get("remedy_detail") or {}
    stated_remedy  = (intake_state or {}).get("stated_remedy") or "not stated"
    assessed_rem   = (
        (intake_state or {}).get("assessed_remedy")
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


def _build_gap_section(intake_state: dict) -> tuple[str, list[str]]:
    """
    Build a short section covering factual gaps and how the record can be strengthened.
    """
    missing_groups = [
        str(x).strip() for x in (
            (intake_state or {}).get("missing_detail_groups")
            or (intake_state or {}).get("open_questions")
            or []
        )
        if str(x).strip()
    ]

    evidence_hooks = [
        f.get("evidence_hook")
        for f in ((intake_state or {}).get("known_facts") or [])
        if isinstance(f, dict) and f.get("evidence_hook")
    ]
    witness_hooks = [
        f.get("witness_hook")
        for f in ((intake_state or {}).get("known_facts") or [])
        if isinstance(f, dict) and f.get("witness_hook")
    ]
    case_file = (intake_state or {}).get("case_file") or {}
    proof_recommendations = [
        item for item in (case_file.get("missing_proof_recommendations") or [])
        if isinstance(item, dict) and str(item.get("point") or "").strip()
    ]
    limitation_notes = [
        str(x).strip()
        for x in (((case_file.get("procedural_posture") or {}).get("limitation_notes")) or [])
        if str(x).strip()
    ]

    strengthen_points: list[str] = []
    if evidence_hooks:
        strengthen_points.append(
            "Organize and preserve the supporting record already mentioned: "
            + ", ".join(dict.fromkeys(evidence_hooks))
            + "."
        )
    if witness_hooks:
        strengthen_points.append(
            "Secure clear statements or contact details for witnesses such as "
            + ", ".join(dict.fromkeys(witness_hooks))
            + "."
        )
    for item in proof_recommendations[:4]:
        point = str(item.get("point") or "").strip()
        why = str(item.get("why_it_matters") or "").strip()
        source = str(item.get("best_source") or "").strip()
        line = point
        if why:
            line += f" This matters because {why}"
        if source:
            line += f" Best source: {source}."
        strengthen_points.append(line)
    for note in limitation_notes[:2]:
        strengthen_points.append(f"Address the timing issue early: {note}")
    if missing_groups:
        strengthen_points.append(
            "Fill the remaining factual gaps below so the legal position can be framed more precisely."
        )
    if not strengthen_points:
        strengthen_points.append(
            "Keep the timeline, supporting documents, and communications arranged chronologically to strengthen the record."
        )

    lines = ["## Gaps And Strengthening\n"]
    if missing_groups:
        lines.append("**Current gaps**")
        for item in missing_groups:
            lines.append(f"- {item}")
    else:
        lines.append("No major factual gaps are presently flagged from intake.")

    lines.append("\n**How to strengthen the matter**")
    for item in strengthen_points:
        lines.append(f"- {item}")

    return "\n".join(lines), missing_groups


def _build_case_file_sections(intake_state: dict) -> tuple[str, dict]:
    """
    Render the internal case_file into advocate-style sections for the draft.
    """
    case_file = ensure_case_file_structure((intake_state or {}).get("case_file") or {})
    case_theory = case_file.get("case_theory") or {}
    evidence = case_file.get("evidence_posture") or {}
    risks = case_file.get("risk_map") or {}
    timeline = case_file.get("timeline") or {}
    procedural = case_file.get("procedural_posture") or {}
    proof_matrix = case_file.get("fact_proof_matrix") or []
    missing_proof = case_file.get("missing_proof_recommendations") or []
    contradictions = case_file.get("contradictions") or []

    lines: list[str] = []

    summary = str(case_file.get("summary") or "").strip()
    if summary:
        lines.append("## Case Theory\n")
        lines.append(summary)
        lines.append("")
    else:
        lines.append("## Case Theory\n")

    immediate_concerns = _string_list(case_file.get("immediate_concerns") or [], limit=6)
    if immediate_concerns:
        lines.append("**Immediate concerns**")
        for item in immediate_concerns:
            lines.append(f"- {item}")

    theory_points = [
        ("Core grievance", case_theory.get("core_grievance")),
        ("Client position", case_theory.get("client_position")),
        ("Likely opposing position", case_theory.get("opposing_position")),
        ("Immediate relief", case_theory.get("immediate_relief")),
        ("Long-term relief", case_theory.get("long_term_relief")),
    ]
    for label, value in theory_points:
        if value:
            lines.append(f"- **{label}:** {value}")
    for item in (case_theory.get("strongest_facts") or []):
        if item:
            lines.append(f"- **Strongest fact:** {item}")
    for item in (case_theory.get("weakest_facts") or []):
        if item:
            lines.append(f"- **Weak point:** {item}")

    lines.append("\n## Evidence Posture\n")
    evidence_groups = [
        ("Document-backed", evidence.get("document_backed") or []),
        ("Witness-backed", evidence.get("witness_backed") or []),
        ("Asserted but unproven", evidence.get("asserted_but_unproven") or []),
        ("Needs contemporaneous proof", evidence.get("needs_contemporaneous_proof") or []),
        ("Credibility notes", evidence.get("credibility_notes") or []),
    ]
    for label, items in evidence_groups:
        if items:
            lines.append(f"**{label}**")
            for item in items:
                if item:
                    lines.append(f"- {item}")

    timeline_events = timeline.get("events") or []
    latest_material_event = str(timeline.get("latest_material_event") or "").strip()
    timeline_gaps = _string_list(timeline.get("timeline_gaps") or [], limit=5)
    lines.append("\n## Timeline And Procedural Posture\n")
    if latest_material_event:
        lines.append(f"- **Latest material event:** {latest_material_event}")
    for item in timeline_events[:6]:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event") or "").strip()
        if not event:
            continue
        date_or_period = str(item.get("date_or_period") or "").strip()
        significance = str(item.get("significance") or "").strip()
        materials = [
            str(x).strip()
            for x in (item.get("supporting_materials") or [])
            if str(x).strip()
        ]
        line = f"- **{date_or_period or 'Timeline'}:** {event}"
        if significance:
            line += f" ({significance})"
        if materials:
            line += f" [Support: {', '.join(materials[:3])}]"
        lines.append(line)
    if timeline_gaps:
        lines.append("**Chronology gaps**")
        for item in timeline_gaps:
            if item:
                lines.append(f"- {item}")

    posture_points = [
        ("Current stage", procedural.get("current_stage")),
        ("Current forum or authority", procedural.get("current_forum_or_authority")),
        ("Next deadline or trigger", procedural.get("next_deadline_or_trigger")),
    ]
    for label, value in posture_points:
        if value:
            lines.append(f"- **{label}:** {value}")
    for item in _string_list((procedural.get("steps_already_taken") or []), limit=6):
        if item:
            lines.append(f"- **Step already taken:** {item}")
    for item in _string_list((procedural.get("limitation_notes") or []), limit=4):
        if item:
            lines.append(f"- **Timing note:** {item}")

    if proof_matrix:
        lines.append("\n## Fact-To-Proof Matrix\n")
        for item in proof_matrix[:8]:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact") or "").strip()
            if not fact:
                continue
            status = str(item.get("support_status") or "").strip()
            materials = [
                str(x).strip()
                for x in (item.get("supporting_materials") or [])
                if str(x).strip()
            ]
            witnesses = [
                str(x).strip()
                for x in (item.get("witness_support") or [])
                if str(x).strip()
            ]
            gap = str(item.get("proof_gap") or "").strip()
            lines.append(f"- **{fact}**")
            if status:
                lines.append(f"  Support status: {status}")
            if materials:
                lines.append(f"  Materials: {', '.join(materials[:4])}")
            if witnesses:
                lines.append(f"  Witness support: {', '.join(witnesses[:3])}")
            if gap:
                lines.append(f"  Proof gap: {gap}")

    if missing_proof:
        lines.append("\n## Proof Strengthening Priorities\n")
        for item in missing_proof[:6]:
            if not isinstance(item, dict):
                continue
            point = str(item.get("point") or "").strip()
            if not point:
                continue
            why = str(item.get("why_it_matters") or "").strip()
            source = str(item.get("best_source") or "").strip()
            line = f"- **{point}**"
            if why:
                line += f": {why}"
            if source:
                line += f" Best source: {source}."
            lines.append(line)

    lines.append("\n## Risk Assessment\n")
    risk_groups = [
        ("Maintainability risks", risks.get("maintainability_risks") or []),
        ("Proof risks", risks.get("proof_risks") or []),
        ("Timeline risks", risks.get("timeline_risks") or []),
        ("Relief risks", risks.get("relief_risks") or []),
        ("Likely objections", risks.get("other_side_objections") or []),
    ]
    for label, items in risk_groups:
        if items:
            lines.append(f"**{label}**")
            for item in items:
                if item:
                    lines.append(f"- {item}")

    if contradictions:
        lines.append("\n## Contradictions To Resolve\n")
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            issue = str(item.get("issue") or "").strip()
            if not issue:
                continue
            severity = str(item.get("severity") or "low").strip().lower()
            note = str(item.get("note") or "").strip()
            suffix = f" ({severity})" if severity else ""
            lines.append(f"- **{issue}**{suffix}" + (f": {note}" if note else ""))

    return "\n".join(lines), case_file


def _get_forum_analysis(intake_state: dict) -> dict:
    """Run forum identification from the current intake state, with safe fallback."""
    try:
        from agents.forum.agent import identify_forum
        raw = identify_forum(json.dumps(intake_state or {}, ensure_ascii=False))
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Forum analysis failed during draft build: %s", exc)
        return {}


def _summarize_legal_framework(disputes_structured: list) -> str:
    lines: list[str] = []
    for dispute in (disputes_structured or [])[:4]:
        label = dispute.get("dispute_label", "General")
        sections = []
        for sec in (dispute.get("sections") or [])[:3]:
            act = sec.get("act_name", "")
            num = sec.get("section_number", "")
            title = sec.get("section_title", "")
            part = " ".join(x for x in [act, f"Section {num}" if num else "", f"({title})" if title else ""] if x)
            if part:
                sections.append(part)
        lines.append(f"- {label}: {'; '.join(sections)}")
    return "\n".join(lines) or "No legal framework retrieved."


def _summarize_case_law(caselaw_structured: list) -> str:
    lines: list[str] = []
    for case in (caselaw_structured or [])[:6]:
        case_name = case.get("case_name", "Unknown case")
        relevant = case.get("relevant_to_section", "")
        court = case.get("court", "")
        year = case.get("year", "")
        meta = ", ".join(x for x in [court, year] if x)
        tail = f" ({meta})" if meta else ""
        lines.append(f"- {case_name}{tail}: relevant to {relevant}")
    return "\n".join(lines) or "No case law support retrieved."


def _build_relief_strategy(
    intake_state: dict,
    disputes_structured: list,
    caselaw_structured: list,
) -> tuple[str, dict]:
    """Build relief-first analysis, forum selection, strategy comparison, sequencing, and draft readiness."""
    case_file = (intake_state or {}).get("case_file") or {}
    forum_analysis = _get_forum_analysis(intake_state or {})

    prompt = (
        _RELIEF_STRATEGY_PROMPT
        .replace("{case_summary}", json.dumps(case_file, ensure_ascii=False, indent=2)[:3500])
        .replace("{forum_analysis}", json.dumps(forum_analysis, ensure_ascii=False, indent=2)[:2000] or "{}")
        .replace("{legal_framework}", _summarize_legal_framework(disputes_structured)[:1800])
        .replace("{case_law_summary}", _summarize_case_law(caselaw_structured)[:1800])
    )

    data: dict = {}
    try:
        raw = _ask_llm(prompt, task_hint="quality")
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
        if isinstance(parsed, dict):
            data = parsed
    except Exception as exc:
        logger.warning("Relief strategy synthesis failed: %s", exc)

    if not data:
        primary_forum = ((forum_analysis.get("primary_forum") or {}).get("forum")) or "Competent forum to be confirmed"
        alternative_forum = ((forum_analysis.get("alternative_forum") or {}).get("forum")) or None
        assessed = (intake_state or {}).get("assessed_remedy") or (intake_state or {}).get("client_goal_initial") or "Appropriate relief"
        missing = (intake_state or {}).get("missing_detail_groups") or []
        status = "ready" if not missing else "mostly_ready"
        score = 85 if not missing else 65
        data = {
            "relief_first_analysis": [
                {
                    "relief": assessed,
                    "why_it_matters": "This appears to be the most practical immediate objective on the current record.",
                    "legal_basis_note": "Support depends on the retrieved framework and the present evidence posture.",
                    "support_level": "moderate",
                }
            ],
            "forum_selection": {
                "primary_forum": primary_forum,
                "alternative_forum": alternative_forum,
                "why_primary_forum": "This forum appears most aligned with the present relief and dispute framing.",
                "forum_risk": "Forum choice should be rechecked if the facts, value, or procedural posture materially shift.",
            },
            "strategy_comparison": [
                {
                    "route": "Immediate relief route",
                    "best_for": "urgent protective or interim steps",
                    "tradeoff": "Faster movement, but may depend on a narrower factual threshold.",
                    "speed": "fast",
                    "support_level": "moderate",
                },
                {
                    "route": "Full merits route",
                    "best_for": "final determination and broader relief",
                    "tradeoff": "More complete, but slower and more document-heavy.",
                    "speed": "slow",
                    "support_level": "moderate",
                },
            ],
            "practical_sequencing": [
                {"window": "today", "step": "Preserve the current record", "why": "Early preservation improves consistency and proof value."},
                {"window": "next_7_days", "step": "Organize the factual timeline and supporting materials", "why": "This sharpens forum choice and relief framing."},
                {"window": "before_filing", "step": "Close any material factual gaps", "why": "A tighter record reduces avoidable objections."},
            ],
            "draft_suitability": {
                "status": status,
                "score": score,
                "blocking_gaps": list(missing)[:5],
                "why": "Readiness depends on whether the present record is sufficient for a specific filing without avoidable ambiguity.",
            },
        }

    md_lines = ["## Relief-First Analysis\n"]
    for item in (data.get("relief_first_analysis") or []):
        if not isinstance(item, dict):
            continue
        relief = item.get("relief", "")
        why = item.get("why_it_matters", "")
        note = item.get("legal_basis_note", "")
        support = item.get("support_level", "")
        md_lines.append(f"- **{relief}**")
        if why:
            md_lines.append(f"  Why it matters: {why}")
        if note:
            md_lines.append(f"  Legal basis note: {note}")
        if support:
            md_lines.append(f"  Support level: {support}")

    forum = data.get("forum_selection") or {}
    md_lines.append("\n## Forum Selection\n")
    if forum:
        if forum.get("primary_forum"):
            md_lines.append(f"- **Primary forum:** {forum.get('primary_forum')}")
        if forum.get("alternative_forum"):
            md_lines.append(f"- **Alternative forum:** {forum.get('alternative_forum')}")
        if forum.get("why_primary_forum"):
            md_lines.append(f"- **Why this forum:** {forum.get('why_primary_forum')}")
        if forum.get("forum_risk"):
            md_lines.append(f"- **Forum risk:** {forum.get('forum_risk')}")

    md_lines.append("\n## Strategy Comparison\n")
    for item in (data.get("strategy_comparison") or []):
        if not isinstance(item, dict):
            continue
        route = item.get("route", "")
        best_for = item.get("best_for", "")
        tradeoff = item.get("tradeoff", "")
        speed = item.get("speed", "")
        support = item.get("support_level", "")
        md_lines.append(f"- **{route}:** best for {best_for}. Tradeoff: {tradeoff}. Speed: {speed}. Support: {support}.")

    md_lines.append("\n## Practical Sequencing\n")
    for item in (data.get("practical_sequencing") or []):
        if not isinstance(item, dict):
            continue
        md_lines.append(f"- **{item.get('window', '')}:** {item.get('step', '')} — {item.get('why', '')}")

    suitability = data.get("draft_suitability") or {}
    md_lines.append("\n## Draft Suitability\n")
    if suitability:
        md_lines.append(f"- **Status:** {suitability.get('status', 'unknown')}")
        md_lines.append(f"- **Score:** {suitability.get('score', 0)}/100")
        if suitability.get("why"):
            md_lines.append(f"- **Why:** {suitability.get('why')}")
        gaps = suitability.get("blocking_gaps") or []
        if gaps:
            md_lines.append("- **Blocking gaps:**")
            for gap in gaps:
                md_lines.append(f"  - {gap}")

    return "\n".join(md_lines), {
        "forum_analysis": forum_analysis,
        "relief_strategy": data,
    }


def _string_list(items: list, limit: int | None = None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if limit and len(cleaned) >= limit:
            break
    return cleaned


def _build_research_packets(
    case_file: dict,
    disputes_structured: list,
    caselaw_structured: list,
    relief_strategy_structured: dict,
) -> tuple[str, dict]:
    """Organize retrieved materials into issue-wise research packets."""
    prompt = (
        _RESEARCH_PACKET_PROMPT
        .replace("{case_file}", json.dumps(case_file or {}, ensure_ascii=False, indent=2)[:3500])
        .replace("{relief_strategy}", json.dumps((relief_strategy_structured or {}).get("relief_strategy") or {}, ensure_ascii=False, indent=2)[:2500])
        .replace("{disputes_json}", json.dumps(disputes_structured or [], ensure_ascii=False, indent=2)[:4500])
        .replace("{case_law_json}", json.dumps(caselaw_structured or [], ensure_ascii=False, indent=2)[:4500])
    )

    packets: list[dict] = []
    try:
        raw = _ask_llm(prompt, task_hint="quality")
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
        if isinstance(parsed, dict):
            packets = parsed.get("packets") or []
    except Exception as exc:
        logger.warning("Research packet synthesis failed: %s", exc)

    cleaned_packets: list[dict] = []
    if not packets:
        reliefs = _string_list(
            [item.get("relief") for item in (((relief_strategy_structured or {}).get("relief_strategy") or {}).get("relief_first_analysis") or []) if isinstance(item, dict)],
            limit=3,
        )
        objections = _string_list(
            ((case_file or {}).get("risk_map") or {}).get("other_side_objections") or [],
            limit=4,
        )
        strengthening = (case_file or {}).get("missing_proof_recommendations") or []
        for dispute in (disputes_structured or [])[:4]:
            if not isinstance(dispute, dict):
                continue
            label = str(dispute.get("dispute_label") or "Issue").strip()
            sections = []
            act_names = []
            for sec in (dispute.get("sections") or [])[:3]:
                if not isinstance(sec, dict):
                    continue
                act_name = str(sec.get("act_name") or "").strip()
                section_number = str(sec.get("section_number") or "").strip()
                if act_name:
                    act_names.append(act_name)
                sections.append({
                    "act_name": act_name,
                    "section_number": section_number,
                    "section_title": str(sec.get("section_title") or "").strip() or None,
                    "why_it_matters": str(sec.get("section_title") or "This section appears central to the issue framing.").strip(),
                })
            linked_cases = []
            for case in (caselaw_structured or []):
                if not isinstance(case, dict):
                    continue
                relevant = str(case.get("relevant_to_section") or "").strip()
                if act_names and not any(act in relevant for act in act_names):
                    continue
                linked_cases.append({
                    "case_name": str(case.get("case_name") or "Unknown case").strip(),
                    "usefulness_for": "general support",
                    "rank": "high" if not linked_cases else "medium",
                    "why": f"Appears tied to {relevant or label} on the present record.",
                })
                if len(linked_cases) >= 3:
                    break
            steps = []
            for item in strengthening[:3]:
                if not isinstance(item, dict):
                    continue
                point = str(item.get("point") or "").strip()
                if not point:
                    continue
                steps.append({
                    "step": point,
                    "basis": linked_cases[0]["case_name"] if linked_cases else (sections[0]["act_name"] if sections else label),
                    "why": str(item.get("why_it_matters") or "").strip() or "This addresses a current proof gap.",
                })
            cleaned_packets.append({
                "issue": label,
                "relevant_reliefs": reliefs,
                "primary_sections": sections,
                "case_law_ranked": linked_cases,
                "likely_objections": objections,
                "strengthening_steps": steps,
            })
    else:
        for packet in packets[:6]:
            if not isinstance(packet, dict):
                continue
            issue = str(packet.get("issue") or "").strip()
            if not issue:
                continue
            primary_sections = []
            for sec in (packet.get("primary_sections") or [])[:4]:
                if not isinstance(sec, dict):
                    continue
                act_name = str(sec.get("act_name") or "").strip()
                section_number = str(sec.get("section_number") or "").strip()
                if not (act_name or section_number):
                    continue
                primary_sections.append({
                    "act_name": act_name or None,
                    "section_number": section_number or None,
                    "section_title": str(sec.get("section_title") or "").strip() or None,
                    "why_it_matters": str(sec.get("why_it_matters") or "").strip()[:260] or None,
                })
            case_law_ranked = []
            for case in (packet.get("case_law_ranked") or [])[:5]:
                if not isinstance(case, dict):
                    continue
                case_name = str(case.get("case_name") or "").strip()
                if not case_name:
                    continue
                usefulness = str(case.get("usefulness_for") or "general support").strip().lower()
                if usefulness not in (
                    "threshold issue",
                    "maintainability",
                    "burden of proof",
                    "interim relief",
                    "final relief",
                    "general support",
                ):
                    usefulness = "general support"
                rank = str(case.get("rank") or "medium").strip().lower()
                if rank not in ("high", "medium", "low"):
                    rank = "medium"
                case_law_ranked.append({
                    "case_name": case_name[:220],
                    "usefulness_for": usefulness,
                    "rank": rank,
                    "why": str(case.get("why") or "").strip()[:260] or None,
                })
            strengthening_steps = []
            for step in (packet.get("strengthening_steps") or [])[:5]:
                if not isinstance(step, dict):
                    continue
                text = str(step.get("step") or "").strip()
                if not text:
                    continue
                strengthening_steps.append({
                    "step": text[:260],
                    "basis": str(step.get("basis") or "").strip()[:220] or None,
                    "why": str(step.get("why") or "").strip()[:260] or None,
                })
            cleaned_packets.append({
                "issue": issue[:220],
                "relevant_reliefs": _string_list(packet.get("relevant_reliefs") or [], limit=4),
                "primary_sections": primary_sections,
                "case_law_ranked": case_law_ranked,
                "likely_objections": _string_list(packet.get("likely_objections") or [], limit=5),
                "strengthening_steps": strengthening_steps,
            })

    md_lines = ["## Issue-Wise Research Packets\n"]
    if not cleaned_packets:
        md_lines.append("No issue-wise research packets could be assembled from the current material.")
    for packet in cleaned_packets:
        md_lines.append(f"### {packet.get('issue', 'Issue')}\n")
        reliefs = packet.get("relevant_reliefs") or []
        if reliefs:
            md_lines.append("- **Relevant reliefs:** " + ", ".join(reliefs))
        sections = packet.get("primary_sections") or []
        if sections:
            md_lines.append("- **Primary sections:**")
            for sec in sections:
                label = " ".join(
                    x for x in [
                        str(sec.get("act_name") or "").strip(),
                        f"Section {sec.get('section_number')}" if sec.get("section_number") else "",
                        f"({sec.get('section_title')})" if sec.get("section_title") else "",
                    ]
                    if x
                )
                why = str(sec.get("why_it_matters") or "").strip()
                md_lines.append(f"  - {label or 'Section'}" + (f": {why}" if why else ""))
        cases = packet.get("case_law_ranked") or []
        if cases:
            md_lines.append("- **Ranked case law:**")
            for case in cases:
                case_name = str(case.get("case_name") or "").strip()
                usefulness = str(case.get("usefulness_for") or "").strip()
                rank = str(case.get("rank") or "").strip()
                why = str(case.get("why") or "").strip()
                md_lines.append(f"  - {case_name} [{rank}, {usefulness}]" + (f": {why}" if why else ""))
        objections = packet.get("likely_objections") or []
        if objections:
            md_lines.append("- **Likely objections:**")
            for item in objections:
                md_lines.append(f"  - {item}")
        steps = packet.get("strengthening_steps") or []
        if steps:
            md_lines.append("- **How to strengthen from the law:**")
            for step in steps:
                text = str(step.get("step") or "").strip()
                basis = str(step.get("basis") or "").strip()
                why = str(step.get("why") or "").strip()
                line = f"  - {text}"
                if basis:
                    line += f" [Basis: {basis}]"
                if why:
                    line += f": {why}"
                md_lines.append(line)
        md_lines.append("")

    return "\n".join(md_lines).rstrip(), {"packets": cleaned_packets}


def _build_action_plan(
    case_file: dict,
    relief_strategy_structured: dict,
    research_packets_structured: dict,
    prayer_structured: list,
    docs_structured: list,
) -> tuple[str, dict]:
    """Decide whether to stay in analysis or move into draft-specific action planning."""
    prompt = (
        _ACTION_DRAFTING_PROMPT
        .replace("{case_file}", json.dumps(case_file or {}, ensure_ascii=False, indent=2)[:3500])
        .replace("{relief_strategy}", json.dumps((relief_strategy_structured or {}).get("relief_strategy") or {}, ensure_ascii=False, indent=2)[:2500])
        .replace("{research_packets}", json.dumps((research_packets_structured or {}).get("packets") or [], ensure_ascii=False, indent=2)[:4000])
        .replace("{documents_json}", json.dumps(docs_structured or [], ensure_ascii=False, indent=2)[:2000])
        .replace("{prayer_json}", json.dumps(prayer_structured or [], ensure_ascii=False, indent=2)[:2000])
    )

    plan: dict = {}
    try:
        raw = _ask_llm(prompt, task_hint="quality")
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
        if isinstance(parsed, dict):
            plan = parsed
    except Exception as exc:
        logger.warning("Action drafting planner failed: %s", exc)

    if not plan:
        relief_strategy = (relief_strategy_structured or {}).get("relief_strategy") or {}
        suitability = relief_strategy.get("draft_suitability") or {}
        status = str(suitability.get("status") or "mostly_ready").strip().lower()
        if status not in ("ready", "mostly_ready", "not_ready"):
            status = "mostly_ready"
        forum = str((relief_strategy.get("forum_selection") or {}).get("primary_forum") or "Competent forum").strip()
        blocking = _string_list(suitability.get("blocking_gaps") or [], limit=5)
        reliefs = _string_list(
            [item.get("relief") for item in (relief_strategy.get("relief_first_analysis") or []) if isinstance(item, dict)],
            limit=3,
        )
        essential_docs = _string_list(
            [item.get("document") for item in (docs_structured or []) if isinstance(item, dict) and str(item.get("priority") or "").strip().lower() == "essential"],
            limit=5,
        )
        drafts = [
            {
                "draft_type": "evidence_bundle_checklist",
                "recommended": True,
                "readiness": "ready",
                "forum": forum,
                "purpose": "Organize the documentary record before any filing or notice is finalized.",
                "missing_fields": [],
                "forum_specific_prayers": [],
                "annexures": essential_docs,
            }
        ]
        if status == "ready":
            drafts.insert(0, {
                "draft_type": "application",
                "recommended": True,
                "readiness": "ready",
                "forum": forum,
                "purpose": "Prepare the first substantive filing aligned to the present relief and forum choice.",
                "missing_fields": [],
                "forum_specific_prayers": [f"Seek {relief} before {forum}" for relief in reliefs] or [f"Seek the principal relief before {forum}"],
                "annexures": essential_docs,
            })
            analysis_status = "analysis_and_drafting"
            hold_back_note = ""
        elif status == "mostly_ready":
            drafts.insert(0, {
                "draft_type": "application",
                "recommended": True,
                "readiness": "mostly_ready",
                "forum": forum,
                "purpose": "A draft can be prepared in outline, but the present blockers should be closed before finalization.",
                "missing_fields": blocking,
                "forum_specific_prayers": [f"Seek {relief} before {forum}" for relief in reliefs[:2]] or [f"Seek appropriate interim or final relief before {forum}"],
                "annexures": essential_docs,
            })
            analysis_status = "analysis_only"
            hold_back_note = str(suitability.get("why") or "The drafting route should wait until the remaining gaps are resolved.").strip()
        else:
            drafts = drafts[:1]
            analysis_status = "analysis_only"
            hold_back_note = str(suitability.get("why") or "The matter can be analyzed now, but drafting should wait until core blockers are cured.").strip()
        plan = {
            "analysis_status": analysis_status,
            "overall_readiness": status,
            "readiness_reason": str(suitability.get("why") or "Drafting readiness depends on the present record and forum clarity.").strip(),
            "drafts": drafts,
            "hold_back_note": hold_back_note or None,
        }

    cleaned_drafts: list[dict] = []
    for item in (plan.get("drafts") or [])[:7]:
        if not isinstance(item, dict):
            continue
        draft_type = str(item.get("draft_type") or "").strip()
        if not draft_type:
            continue
        readiness = str(item.get("readiness") or "mostly_ready").strip().lower()
        if readiness not in ("ready", "mostly_ready", "not_ready"):
            readiness = "mostly_ready"
        raw_recommended = item.get("recommended", False)
        if isinstance(raw_recommended, str):
            recommended = raw_recommended.strip().lower() in ("true", "yes", "1", "recommended")
        else:
            recommended = bool(raw_recommended)
        cleaned_drafts.append({
            "draft_type": draft_type[:120],
            "recommended": recommended,
            "readiness": readiness,
            "forum": str(item.get("forum") or "").strip()[:180] or None,
            "purpose": str(item.get("purpose") or "").strip()[:260] or None,
            "missing_fields": _string_list(item.get("missing_fields") or [], limit=6),
            "forum_specific_prayers": _string_list(item.get("forum_specific_prayers") or [], limit=5),
            "annexures": _string_list(item.get("annexures") or [], limit=6),
        })

    cleaned_plan = {
        "analysis_status": str(plan.get("analysis_status") or "analysis_only").strip().lower(),
        "overall_readiness": str(plan.get("overall_readiness") or "mostly_ready").strip().lower(),
        "readiness_reason": str(plan.get("readiness_reason") or "").strip()[:320] or None,
        "drafts": cleaned_drafts,
        "hold_back_note": str(plan.get("hold_back_note") or "").strip()[:320] or None,
    }
    if cleaned_plan["analysis_status"] not in ("analysis_only", "analysis_and_drafting"):
        cleaned_plan["analysis_status"] = "analysis_only"
    if cleaned_plan["overall_readiness"] not in ("ready", "mostly_ready", "not_ready"):
        cleaned_plan["overall_readiness"] = "mostly_ready"

    md_lines = ["## Action Drafting Plan\n"]
    md_lines.append(f"- **Analysis status:** {cleaned_plan.get('analysis_status')}")
    md_lines.append(f"- **Overall readiness:** {cleaned_plan.get('overall_readiness')}")
    if cleaned_plan.get("readiness_reason"):
        md_lines.append(f"- **Readiness reason:** {cleaned_plan.get('readiness_reason')}")
    if cleaned_plan.get("hold_back_note"):
        md_lines.append(f"- **Hold-back note:** {cleaned_plan.get('hold_back_note')}")
    for draft in cleaned_drafts:
        md_lines.append(f"\n### {draft.get('draft_type', 'Draft')}\n")
        md_lines.append(f"- **Recommended:** {'yes' if draft.get('recommended') else 'no'}")
        md_lines.append(f"- **Readiness:** {draft.get('readiness')}")
        if draft.get("forum"):
            md_lines.append(f"- **Forum:** {draft.get('forum')}")
        if draft.get("purpose"):
            md_lines.append(f"- **Purpose:** {draft.get('purpose')}")
        if draft.get("missing_fields"):
            md_lines.append("- **Missing-field blockers:**")
            for field in draft.get("missing_fields") or []:
                md_lines.append(f"  - {field}")
        if draft.get("forum_specific_prayers"):
            md_lines.append("- **Forum-specific prayers:**")
            for prayer in draft.get("forum_specific_prayers") or []:
                md_lines.append(f"  - {prayer}")
        if draft.get("annexures"):
            md_lines.append("- **Annexures / document pack:**")
            for annexure in draft.get("annexures") or []:
                md_lines.append(f"  - {annexure}")

    return "\n".join(md_lines), cleaned_plan


def _build_client_facing_draft(
    primary: str,
    case_file: dict,
    facts_structured: list,
    relief_strategy_structured: dict,
    disputes_structured: list,
    caselaw_structured: list,
    missing_groups: list[str],
    prayer_structured: list,
    docs_structured: list,
    action_plan_structured: dict | None = None,
) -> str:
    case_theory = (case_file or {}).get("case_theory") or {}
    evidence = (case_file or {}).get("evidence_posture") or {}
    timeline = (case_file or {}).get("timeline") or {}
    procedural = (case_file or {}).get("procedural_posture") or {}
    missing_proof = (case_file or {}).get("missing_proof_recommendations") or []
    immediate_concerns = _string_list((case_file or {}).get("immediate_concerns") or [], limit=4)
    relief_strategy = (relief_strategy_structured or {}).get("relief_strategy") or {}
    forum_selection = relief_strategy.get("forum_selection") or {}
    sequencing = relief_strategy.get("practical_sequencing") or []
    reliefs = relief_strategy.get("relief_first_analysis") or []
    suitability = relief_strategy.get("draft_suitability") or {}
    action_plan = action_plan_structured or {}

    lines = [
        f"# Legal Position - {primary.replace('_', ' ').title()}",
        "",
        "On the present facts shared, and subject to the supporting documents and fuller record, this is the legal position as it presently appears.",
    ]

    lines.append("\n## Facts As Understood\n")
    summary = str((case_file or {}).get("summary") or "").strip()
    if summary:
        lines.append(summary)
    for item in (case_theory.get("strongest_facts") or [])[:4]:
        if item:
            lines.append(f"- {item}")
    if not summary and not case_theory.get("strongest_facts"):
        for item in (facts_structured or [])[:4]:
            fact = str(item.get("fact") or "").strip()
            if fact:
                lines.append(f"- {fact}")

    lines.append("\n## Immediate Concerns\n")
    if immediate_concerns:
        for item in immediate_concerns:
            lines.append(f"- {item}")
    latest_material_event = str(timeline.get("latest_material_event") or "").strip()
    if latest_material_event:
        lines.append(f"- The latest material development appears to be: {latest_material_event}.")
    next_trigger = str(procedural.get("next_deadline_or_trigger") or "").strip()
    if next_trigger:
        lines.append(f"- A timing point to keep in view is: {next_trigger}.")
    for item in _string_list((procedural.get("limitation_notes") or []), limit=2):
        lines.append(f"- Timing note: {item}.")
    if not immediate_concerns and not latest_material_event and not next_trigger:
        lines.append("- No extraordinary immediate concern is clearly established beyond preserving the record and acting without avoidable delay.")

    lines.append("\n## Legal Position\n")
    if reliefs:
        for item in reliefs[:3]:
            if not isinstance(item, dict):
                continue
            relief = str(item.get("relief") or "").strip()
            why = str(item.get("why_it_matters") or "").strip()
            note = str(item.get("legal_basis_note") or "").strip()
            support = str(item.get("support_level") or "").strip()
            if relief:
                line = f"- {relief}"
                if why:
                    line += f": {why}"
                if note:
                    line += f" On the present record, {note.lower()}"
                if support:
                    line += f" Support presently appears {support}."
                lines.append(line)
    else:
        immediate_relief = str(case_theory.get("immediate_relief") or "").strip()
        if immediate_relief:
            lines.append(f"- The immediate legal focus appears to be {immediate_relief}, subject to the record supporting that route.")
    if forum_selection.get("primary_forum"):
        lines.append(
            f"- The presently suitable forum appears to be {forum_selection.get('primary_forum')}, "
            f"subject to any additional facts affecting jurisdiction or procedure."
        )
    if procedural.get("current_stage"):
        lines.append(f"- The matter presently looks to be at the {procedural.get('current_stage')} stage.")
    if forum_selection.get("forum_risk"):
        lines.append(f"- Forum caution: {forum_selection.get('forum_risk')}")

    lines.append("\n## Supporting Law\n")
    added_support = False
    for dispute in (disputes_structured or [])[:3]:
        label = dispute.get("dispute_label", "Relevant issue")
        sections = []
        for sec in (dispute.get("sections") or [])[:2]:
            act = sec.get("act_name", "")
            sec_num = sec.get("section_number", "")
            if act or sec_num:
                sections.append(" ".join(x for x in [act, f"Section {sec_num}" if sec_num else ""] if x))
        if sections:
            lines.append(f"- {label}: " + "; ".join(sections))
            added_support = True
    for case in (caselaw_structured or [])[:2]:
        case_name = str(case.get("case_name") or "").strip()
        relevant = str(case.get("relevant_to_section") or "").strip()
        if case_name:
            lines.append(f"- Case law support includes {case_name}" + (f", relevant to {relevant}" if relevant else "") + ".")
            added_support = True
    if not added_support:
        lines.append("- The final legal basis will depend on the fuller research record, but the present analysis is tied to the issue framing and relief identified above.")

    lines.append("\n## Evidentiary Gaps\n")
    evidence_gaps = _string_list((evidence.get("asserted_but_unproven") or []), limit=4)
    if evidence_gaps:
        for item in evidence_gaps:
            lines.append(f"- {item}")
    for item in missing_groups[:4]:
        if item:
            lines.append(f"- Further clarity is still needed on: {item}.")
    for item in missing_proof[:3]:
        if not isinstance(item, dict):
            continue
        point = str(item.get("point") or "").strip()
        why = str(item.get("why_it_matters") or "").strip()
        source = str(item.get("best_source") or "").strip()
        if point:
            line = f"- To strengthen the present position, {point}"
            if why:
                line += f" because {why}"
            if source:
                line += f". The best source appears to be {source}"
            line += "."
            lines.append(line)
    if not evidence_gaps and not missing_groups and not missing_proof:
        weak_points = _string_list((case_theory.get("weakest_facts") or []), limit=2)
        if weak_points:
            for item in weak_points:
                lines.append(f"- A present weakness to keep in view is: {item}.")
        else:
            lines.append("- No major factual gap is presently flagged, though the record should still be checked against the final filing strategy.")

    lines.append("\n## Next Steps\n")
    for item in sequencing[:4]:
        if not isinstance(item, dict):
            continue
        window = str(item.get("window") or "").strip()
        step = str(item.get("step") or "").strip()
        why = str(item.get("why") or "").strip()
        if step:
            window_label = window.replace("_", " ").title() if window else "Next"
            line = f"- {window_label}: {step}"
            if why:
                line += f" because {why}"
            line += "."
            lines.append(line)
    for item in (prayer_structured or [])[:2]:
        if not isinstance(item, dict):
            continue
        relief = str(item.get("relief") or "").strip()
        forum = str(item.get("forum") or "").strip()
        if relief:
            lines.append(f"- If the record remains consistent, one filing route may seek {relief}" + (f" before {forum}" if forum else "") + ".")
    essential_docs = [
        str(item.get("document") or "").strip()
        for item in (docs_structured or [])
        if isinstance(item, dict) and str(item.get("priority") or "").strip().lower() == "essential"
    ]
    if essential_docs:
        lines.append("- Keep ready: " + ", ".join(_string_list(essential_docs, limit=4)) + ".")
    if suitability.get("status") and suitability.get("status") != "ready":
        why = str(suitability.get("why") or "").strip()
        if why:
            lines.append(f"- The matter does not yet look fully filing-ready. {why}")
    if action_plan.get("analysis_status") == "analysis_only":
        hold_back = str(action_plan.get("hold_back_note") or "").strip()
        if hold_back:
            lines.append(f"- For now, the legal analysis can proceed, but responsible drafting should wait. {hold_back}")
    else:
        recommended_drafts = [
            item for item in (action_plan.get("drafts") or [])
            if isinstance(item, dict) and item.get("recommended")
        ]
        for item in recommended_drafts[:2]:
            draft_type = str(item.get("draft_type") or "").strip()
            forum = str(item.get("forum") or "").strip()
            if draft_type:
                lines.append(
                    f"- If the record stays consistent, the next drafting step may be a {draft_type}"
                    + (f" for {forum}" if forum else "")
                    + "."
                )

    return "\n".join(lines)


def _run_counsel_review(
    case_file: dict,
    relief_strategy_structured: dict,
    client_facing_draft: str,
) -> tuple[str, dict]:
    prompt = (
        _COUNSEL_REVIEW_PROMPT
        .replace("{case_file}", json.dumps(case_file or {}, ensure_ascii=False, indent=2)[:3500])
        .replace("{relief_strategy}", json.dumps((relief_strategy_structured or {}).get("relief_strategy") or {}, ensure_ascii=False, indent=2)[:2500])
        .replace("{client_draft}", client_facing_draft[:5000])
    )

    review: dict = {}
    try:
        raw = _ask_llm(prompt, task_hint="quality")
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw and "}" in raw else {}
        if isinstance(parsed, dict):
            review = parsed
    except Exception as exc:
        logger.warning("Counsel review synthesis failed: %s", exc)

    if not review:
        risk_map = (case_file or {}).get("risk_map") or {}
        review = {
            "approval": "revise" if ((risk_map.get("proof_risks") or []) or (risk_map.get("relief_risks") or [])) else "pass",
            "tone": "measured",
            "overstatement_risks": _string_list(risk_map.get("relief_risks") or [], limit=3),
            "unsupported_points": _string_list(((case_file or {}).get("evidence_posture") or {}).get("asserted_but_unproven") or [], limit=3),
            "relief_cautions": _string_list(risk_map.get("relief_risks") or [], limit=3),
            "relevance_gaps": [],
            "rewrite_notes": ["Keep the analysis tied to the present record and avoid language that sounds conclusive where proof is still developing."],
            "client_safe_note": "The advice should stay measured and be presented as the position on the present record, subject to documents and fuller instructions.",
        }

    md_lines = ["## Counsel Review\n"]
    approval = str(review.get("approval") or "pass").strip().lower()
    tone = str(review.get("tone") or "measured").strip().lower()
    md_lines.append(f"- **Approval:** {approval}")
    md_lines.append(f"- **Tone:** {tone}")
    for label, key in (
        ("Overstatement risks", "overstatement_risks"),
        ("Unsupported points", "unsupported_points"),
        ("Relief cautions", "relief_cautions"),
        ("Relevance gaps", "relevance_gaps"),
        ("Rewrite notes", "rewrite_notes"),
    ):
        items = _string_list(review.get(key) or [], limit=5)
        if items:
            md_lines.append(f"- **{label}:**")
            for item in items:
                md_lines.append(f"  - {item}")
    client_safe_note = str(review.get("client_safe_note") or "").strip()
    if client_safe_note:
        md_lines.append(f"- **Client-safe note:** {client_safe_note}")

    return "\n".join(md_lines), review


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
    Build the full structured legal draft from agents.intake state and retrieved materials.

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
    state     = dict(intake_state or {})
    state["case_file"] = ensure_case_file_structure(state.get("case_file"))
    sections  = bare_act_sections or []

    # ── 1. Statement of Facts ──────────────────────────────────────────────
    facts_md, facts_structured = _build_facts_section(state, facts_summary)
    case_file_md, case_file_structured = _build_case_file_sections(state)

    # ── 2. Legal Framework (verbatim statutory text, 3 per dispute) ────────
    framework_md, disputes_structured = _build_legal_framework_section(sections)

    # ── 3. Case Law Support (paragraph_num + verbatim judgment text) ───────
    caselaw_md, caselaw_structured = _build_case_law_section(sections)
    relief_strategy_md, relief_strategy_structured = _build_relief_strategy(
        state,
        disputes_structured,
        caselaw_structured,
    )

    # ── 4. Prayer + 5. Documents Checklist (single LLM call) ───────────────
    gaps_md, missing_groups = _build_gap_section(state)

    prayer_md, docs_md, prayer_structured, docs_structured = _generate_prayer_and_docs(
        facts_summary=facts_summary or "\n".join(f["fact"] for f in facts_structured if "fact" in f),
        structured_disputes=disputes_structured,
        intake_state=state,
        model_override=model_override,
    )
    research_packets_md, research_packets_structured = _build_research_packets(
        case_file_structured,
        disputes_structured,
        caselaw_structured,
        relief_strategy_structured,
    )
    case_file_structured["research_packets"] = research_packets_structured.get("packets") or []
    action_plan_md, action_plan_structured = _build_action_plan(
        case_file_structured,
        relief_strategy_structured,
        research_packets_structured,
        prayer_structured,
        docs_structured,
    )
    case_file_structured["action_plan"] = action_plan_structured

    # ── Assemble Markdown draft ────────────────────────────────────────────
    primary   = state.get("primary_issue_cluster") or "Legal Matter"
    client_facing_draft = _build_client_facing_draft(
        primary,
        case_file_structured,
        facts_structured,
        relief_strategy_structured,
        disputes_structured,
        caselaw_structured,
        missing_groups,
        prayer_structured,
        docs_structured,
        action_plan_structured,
    )
    counsel_review_md, counsel_review_structured = _run_counsel_review(
        case_file_structured,
        relief_strategy_structured,
        client_facing_draft,
    )
    case_file_structured = set_pipeline_layer(
        case_file_structured,
        "research",
        "complete" if (research_packets_structured.get("packets") or []) else "in_progress",
        "Issue-wise research packets synthesized from retrieved sections and case law.",
    )
    case_file_structured = set_pipeline_layer(
        case_file_structured,
        "opinion",
        "complete",
        "Client-facing and chamber-note opinions generated with counsel review.",
    )
    case_file_structured = set_pipeline_layer(
        case_file_structured,
        "action",
        "complete" if (action_plan_structured.get("drafts") or []) else "in_progress",
        "Action drafting readiness, blockers, prayers, and annexures assessed.",
    )
    heading   = f"# Chamber Note - {primary.replace('_', ' ').title()}\n"
    separator = "\n---\n"

    sections_md = [heading, facts_md, case_file_md, relief_strategy_md, framework_md]
    if caselaw_md:
        sections_md.append(caselaw_md)
    sections_md += [research_packets_md, action_plan_md, gaps_md, prayer_md, docs_md, counsel_review_md]

    lawyer_facing_draft = separator.join(s for s in sections_md if s)
    formatted_draft = client_facing_draft
    quality_signals = compute_quality_review_signals(
        client_facing_draft=client_facing_draft,
        lawyer_facing_draft=lawyer_facing_draft,
        counsel_review=counsel_review_structured,
        missing_detail_groups=missing_groups,
        contradictions=case_file_structured.get("contradictions") or [],
        action_plan=action_plan_structured,
    )
    case_file_structured.setdefault("audit_metadata", {})["quality_review_signals"] = quality_signals
    logger.info("QUALITY_REVIEW_SIGNAL %s", json.dumps(quality_signals, ensure_ascii=False))

    # ── Assemble advocate-review JSON ──────────────────────────────────────
    advocate_review = {
        "primary_category":          primary,
        "urgency_signal":            state.get("urgency_signal", "normal"),
        "summary_facts":             facts_structured,
        "case_file":                 case_file_structured,
        "schema_versions":           case_file_structured.get("schema_versions"),
        "pipeline_layers":           case_file_structured.get("pipeline_layers"),
        "audit_metadata":            case_file_structured.get("audit_metadata"),
        "quality_review_signals":    (case_file_structured.get("audit_metadata") or {}).get("quality_review_signals"),
        "client_facing_draft":       client_facing_draft,
        "lawyer_facing_draft":       lawyer_facing_draft,
        "default_output_mode":       "client_facing",
        "counsel_review":            counsel_review_structured,
        "forum_analysis":            relief_strategy_structured.get("forum_analysis"),
        "relief_strategy":           relief_strategy_structured.get("relief_strategy"),
        "research_packets":          research_packets_structured.get("packets") or [],
        "action_plan":               action_plan_structured,
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
        "missing_detail_groups":     missing_groups,
        "overall_prayer":            prayer_structured,
        "overall_documents_checklist": docs_structured,
        "stated_remedy":             state.get("stated_remedy"),
        "assessed_remedy":           state.get("assessed_remedy"),
        "recommended_lead":          (state.get("remedy_detail") or {}).get("recommended_lead"),
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
        "client_facing_draft": client_facing_draft,
        "lawyer_facing_draft": lawyer_facing_draft,
        "advocate_review": advocate_review,
    }


def build_draft(
    intake_state: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    revision_notes: Optional[str] = None,
) -> dict:
    """
    Backward-compatible wrapper expected by the intake graph.
    """
    _ = conversation_history, revision_notes
    return build_legal_draft(intake_state=intake_state)
