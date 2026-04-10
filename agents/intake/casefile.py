"""
Canonical case-file helpers for the Nyaymalaw intake and opinion pipeline.
"""
from __future__ import annotations

import copy
from collections import Counter

CASE_FILE_SCHEMA_VERSION = "2026-04-10.case_file.v1"
EVIDENCE_POSTURE_SCHEMA_VERSION = "2026-04-10.evidence_posture.v1"
RISK_MAP_SCHEMA_VERSION = "2026-04-10.risk_map.v1"
RESEARCH_PACKET_SCHEMA_VERSION = "2026-04-10.research_packets.v1"
ACTION_PLAN_SCHEMA_VERSION = "2026-04-10.action_plan.v1"
AUDIT_METADATA_SCHEMA_VERSION = "2026-04-10.audit_metadata.v1"

PIPELINE_LAYER_DEFAULTS = (
    ("intake", "conversation_and_known_facts"),
    ("case_framing", "canonical_case_file"),
    ("research", "retrieved_law_plus_model_synthesis"),
    ("opinion", "client_facing_and_chamber_outputs"),
    ("action", "draft_readiness_and_action_plan"),
)

FACT_SOURCE_TYPES = (
    "user_statement",
    "document_extraction",
    "model_inference",
    "retrieved_law",
    "advocate_edit",
    "system_inference",
)


def make_case_file_template() -> dict:
    """Return the canonical case-file scaffold used across intake and drafting."""
    return {
        "schema_versions": {
            "case_file": CASE_FILE_SCHEMA_VERSION,
            "evidence_posture": EVIDENCE_POSTURE_SCHEMA_VERSION,
            "risk_map": RISK_MAP_SCHEMA_VERSION,
            "research_packets": RESEARCH_PACKET_SCHEMA_VERSION,
            "action_plan": ACTION_PLAN_SCHEMA_VERSION,
            "audit_metadata": AUDIT_METADATA_SCHEMA_VERSION,
        },
        "pipeline_layers": [
            {
                "layer": layer,
                "status": "pending",
                "source_of_truth": source,
                "note": None,
            }
            for layer, source in PIPELINE_LAYER_DEFAULTS
        ],
        "summary": None,
        "immediate_concerns": [],
        "case_theory": {
            "core_grievance": None,
            "client_position": None,
            "opposing_position": None,
            "immediate_relief": None,
            "long_term_relief": None,
            "strongest_facts": [],
            "weakest_facts": [],
        },
        "evidence_posture": {
            "document_backed": [],
            "witness_backed": [],
            "asserted_but_unproven": [],
            "needs_contemporaneous_proof": [],
            "credibility_notes": [],
        },
        "risk_map": {
            "maintainability_risks": [],
            "proof_risks": [],
            "timeline_risks": [],
            "relief_risks": [],
            "other_side_objections": [],
        },
        "timeline": {
            "events": [],
            "latest_material_event": None,
            "timeline_gaps": [],
        },
        "procedural_posture": {
            "current_stage": None,
            "steps_already_taken": [],
            "current_forum_or_authority": None,
            "next_deadline_or_trigger": None,
            "limitation_notes": [],
        },
        "fact_proof_matrix": [],
        "missing_proof_recommendations": [],
        "research_packets": [],
        "action_plan": {
            "analysis_status": None,
            "overall_readiness": None,
            "readiness_reason": None,
            "drafts": [],
            "hold_back_note": None,
        },
        "audit_metadata": {
            "canonical_record": "case_file",
            "fact_source_counts": {source: 0 for source in FACT_SOURCE_TYPES},
            "fact_provenance": [],
            "state_sources": {
                "known_facts": "user_statement",
                "case_theory": "model_inference",
                "evidence_posture": "model_inference",
                "risk_map": "model_inference",
                "timeline": "user_statement_plus_model_inference",
                "procedural_posture": "user_statement_plus_model_inference",
                "research_packets": "retrieved_law_plus_model_synthesis",
                "action_plan": "model_synthesis_from_relief_and_readiness",
            },
            "inference_notes": [],
            "quality_review_signals": {},
        },
        "contradictions": [],
    }


def _deep_merge(template: dict, value: dict) -> dict:
    merged = copy.deepcopy(template)
    for key, current in (value or {}).items():
        if key not in merged:
            merged[key] = copy.deepcopy(current)
            continue
        if isinstance(merged[key], dict) and isinstance(current, dict):
            merged[key] = _deep_merge(merged[key], current)
        else:
            merged[key] = copy.deepcopy(current)
    return merged


def ensure_case_file_structure(case_file: dict | None) -> dict:
    """Return a normalized canonical case file with version and audit metadata."""
    normalized = _deep_merge(make_case_file_template(), case_file or {})
    normalized["schema_versions"] = _deep_merge(
        make_case_file_template()["schema_versions"],
        normalized.get("schema_versions") or {},
    )
    normalized["audit_metadata"] = _deep_merge(
        make_case_file_template()["audit_metadata"],
        normalized.get("audit_metadata") or {},
    )

    existing_layers = {
        str(item.get("layer") or ""): item
        for item in (normalized.get("pipeline_layers") or [])
        if isinstance(item, dict) and str(item.get("layer") or "").strip()
    }
    normalized["pipeline_layers"] = []
    for layer, source in PIPELINE_LAYER_DEFAULTS:
        existing = existing_layers.get(layer) or {}
        normalized["pipeline_layers"].append({
            "layer": layer,
            "status": str(existing.get("status") or "pending").strip().lower(),
            "source_of_truth": str(existing.get("source_of_truth") or source).strip() or source,
            "note": str(existing.get("note") or "").strip() or None,
        })
    return normalized


def set_pipeline_layer(case_file: dict, layer: str, status: str, note: str | None = None) -> dict:
    """Update the status of a canonical pipeline layer in-place and return the case file."""
    case_file = ensure_case_file_structure(case_file)
    for item in case_file.get("pipeline_layers") or []:
        if item.get("layer") == layer:
            item["status"] = str(status or "pending").strip().lower()
            item["note"] = str(note or "").strip() or None
            break
    return case_file


def build_fact_audit(known_facts: list) -> dict:
    """Build provenance metadata from structured known_facts."""
    counts = Counter()
    entries: list[dict] = []
    for fact in known_facts or []:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("fact") or "").strip()
        if not text:
            continue
        source_type = str(fact.get("source_type") or "user_statement").strip().lower()
        if source_type not in FACT_SOURCE_TYPES:
            source_type = "user_statement"
        counts[source_type] += 1
        entries.append({
            "fact": text[:220],
            "source_type": source_type,
            "source_turn": fact.get("source_turn"),
            "source_detail": str(fact.get("source_detail") or "").strip() or None,
            "time_reference": str(fact.get("time_reference") or "").strip() or None,
            "evidence_hook": str(fact.get("evidence_hook") or "").strip() or None,
            "witness_hook": str(fact.get("witness_hook") or "").strip() or None,
        })
    base_counts = {source: 0 for source in FACT_SOURCE_TYPES}
    base_counts.update({k: int(v) for k, v in counts.items()})
    return {
        "fact_source_counts": base_counts,
        "fact_provenance": entries[:20],
    }


def apply_fact_audit(case_file: dict, known_facts: list, note: str | None = None) -> dict:
    """Attach known-fact provenance and optional inference notes to the case file."""
    case_file = ensure_case_file_structure(case_file)
    audit = case_file.get("audit_metadata") or {}
    fact_audit = build_fact_audit(known_facts)
    audit["fact_source_counts"] = fact_audit["fact_source_counts"]
    audit["fact_provenance"] = fact_audit["fact_provenance"]
    if note:
        notes = [str(x).strip() for x in (audit.get("inference_notes") or []) if str(x).strip()]
        if note not in notes:
            notes.append(note)
        audit["inference_notes"] = notes[:8]
    case_file["audit_metadata"] = audit
    return case_file


def compute_quality_review_signals(
    *,
    client_facing_draft: str,
    lawyer_facing_draft: str,
    counsel_review: dict | None,
    missing_detail_groups: list[str] | None,
    contradictions: list | None,
    action_plan: dict | None,
) -> dict:
    """Heuristic quality signals for review dashboards and runtime logging."""
    client_text = (client_facing_draft or "").lower()
    lawyer_text = (lawyer_facing_draft or "").lower()
    counsel_review = counsel_review or {}
    robotic_markers = [
        "i think",
        "tell me more",
        "anything else",
        "proceed",
        "on the present record",
    ]
    robotic_hits = [marker for marker in robotic_markers if marker in client_text]
    unsupported_points = counsel_review.get("unsupported_points") or []
    overstatement_risks = counsel_review.get("overstatement_risks") or []
    rewrite_notes = counsel_review.get("rewrite_notes") or []
    return {
        "counsel_requires_revision": str(counsel_review.get("approval") or "").strip().lower() == "revise",
        "counsel_tone": str(counsel_review.get("tone") or "").strip().lower() or None,
        "unsupported_point_count": len(unsupported_points),
        "overstatement_risk_count": len(overstatement_risks),
        "rewrite_note_count": len(rewrite_notes),
        "missing_gap_count": len(missing_detail_groups or []),
        "contradiction_count": len(contradictions or []),
        "robotic_phrase_hits": robotic_hits,
        "has_confidence_discipline": any(
            phrase in client_text
            for phrase in ("subject to", "on the present facts", "presently appears", "present record")
        ),
        "has_research_packets": "## issue-wise research packets" in lawyer_text,
        "has_action_plan": "## action drafting plan" in lawyer_text,
        "analysis_only_hold": str((action_plan or {}).get("analysis_status") or "").strip().lower() == "analysis_only",
    }
