"""
Deterministic tests for canonical case-file helpers and quality signals.
"""
from __future__ import annotations

import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.intake.casefile import (  # noqa: E402
    apply_fact_audit,
    compute_quality_review_signals,
    ensure_case_file_structure,
    make_case_file_template,
    set_pipeline_layer,
)


class TestCaseFileStructure(unittest.TestCase):
    def test_template_contains_versions_and_layers(self):
        case_file = make_case_file_template()
        self.assertIn("schema_versions", case_file)
        self.assertIn("audit_metadata", case_file)
        layers = case_file.get("pipeline_layers") or []
        self.assertGreaterEqual(len(layers), 5)
        self.assertEqual(layers[0].get("layer"), "intake")

    def test_ensure_case_file_preserves_custom_values(self):
        case_file = ensure_case_file_structure({
            "summary": "Custom summary",
            "case_theory": {"core_grievance": "Custom grievance"},
        })
        self.assertEqual(case_file["summary"], "Custom summary")
        self.assertEqual(case_file["case_theory"]["core_grievance"], "Custom grievance")
        self.assertIn("schema_versions", case_file)

    def test_set_pipeline_layer_updates_status(self):
        case_file = set_pipeline_layer(make_case_file_template(), "research", "complete", "done")
        layers = {item["layer"]: item for item in case_file.get("pipeline_layers") or []}
        self.assertEqual(layers["research"]["status"], "complete")
        self.assertEqual(layers["research"]["note"], "done")


class TestFactAudit(unittest.TestCase):
    def test_apply_fact_audit_counts_sources(self):
        case_file = make_case_file_template()
        known_facts = [
            {
                "fact": "The client received a termination email.",
                "source_turn": 2,
                "source_type": "user_statement",
                "source_detail": "user_turn_2",
            },
            {
                "fact": "The appointment letter shows employment began in 2017.",
                "source_turn": 0,
                "source_type": "document_extraction",
                "source_detail": "doc_appointment_letter",
            },
        ]
        audited = apply_fact_audit(case_file, known_facts, note="note")
        audit = audited["audit_metadata"]
        self.assertEqual(audit["fact_source_counts"]["user_statement"], 1)
        self.assertEqual(audit["fact_source_counts"]["document_extraction"], 1)
        self.assertEqual(len(audit["fact_provenance"]), 2)
        self.assertIn("note", audit["inference_notes"])


class TestQualitySignals(unittest.TestCase):
    def test_quality_signals_capture_revision_and_sections(self):
        signals = compute_quality_review_signals(
            client_facing_draft=(
                "# Legal Position\n"
                "On the present record, this is the position.\n"
                "## Evidentiary Gaps\n"
                "- One gap.\n"
            ),
            lawyer_facing_draft=(
                "## Issue-Wise Research Packets\n"
                "## Action Drafting Plan\n"
            ),
            counsel_review={
                "approval": "revise",
                "tone": "measured",
                "unsupported_points": ["Point A"],
                "overstatement_risks": ["Risk A", "Risk B"],
                "rewrite_notes": ["Tone it down"],
            },
            missing_detail_groups=["Timeline details"],
            contradictions=[{"issue": "Email timing mismatch"}],
            action_plan={"analysis_status": "analysis_only"},
        )
        self.assertTrue(signals["counsel_requires_revision"])
        self.assertEqual(signals["unsupported_point_count"], 1)
        self.assertEqual(signals["overstatement_risk_count"], 2)
        self.assertTrue(signals["has_research_packets"])
        self.assertTrue(signals["has_action_plan"])
        self.assertTrue(signals["analysis_only_hold"])


if __name__ == "__main__":
    unittest.main()
