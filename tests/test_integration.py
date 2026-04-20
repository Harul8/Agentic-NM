"""
tests/test_integration.py — Integration tests for behavioral fixes.

Covers:
  Issue #9  — pipeline/mode.py  : PipelineMode contract + resolve_pipeline_mode() interaction rules
  Issue #10 — agents/intake/schema.py : IntakeState contract, coerce_intake_state(), anchor_fields_present()
  Cross-cut  — schema + mode used together as chat.py does at stage boundaries

Run with:
    cd "Agentic NM"
    python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import os
import sys
import types
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Stub heavy native modules so tests run without GPU / FAISS / numpy.
# ---------------------------------------------------------------------------
for _mod_name, _attrs in [
    ("faiss", {"read_index": lambda *a, **k: None, "write_index": lambda *a, **k: None}),
    ("numpy", {"array": lambda *a, **k: a[0] if a else None, "float32": float}),
]:
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_mod_name] = _m

for _mod_name in (
    "sentence_transformers", "sentence_transformers.models",
    "torch", "torch.cuda",
):
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))


# ===========================================================================
# Issue #9 — pipeline/mode.py
# ===========================================================================

class TestResolvePipelineModeDefaults(unittest.TestCase):
    """resolve_pipeline_mode() defaults and validation."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_no_args_defaults_to_legal_opinion_local_only_full(self):
        m = self.resolve()
        self.assertEqual(m.intent, "legal_opinion")
        self.assertEqual(m.search_strategy, "local_only")
        self.assertEqual(m.analysis_mode, "full_opinion")

    def test_unknown_intent_falls_back_to_legal_opinion(self):
        m = self.resolve(intent="garbage_intent")
        self.assertEqual(m.intent, "legal_opinion")

    def test_none_intent_defaults_to_legal_opinion(self):
        m = self.resolve(intent=None)
        self.assertEqual(m.intent, "legal_opinion")

    def test_unknown_search_strategy_gets_intent_based_default_opinion(self):
        m = self.resolve(intent="legal_opinion", search_strategy="nonexistent")
        self.assertEqual(m.search_strategy, "local_only")

    def test_unknown_search_strategy_gets_intent_based_default_search(self):
        m = self.resolve(intent="search", search_strategy="nonexistent")
        self.assertEqual(m.search_strategy, "local_then_web")

    def test_unknown_analysis_mode_falls_back_to_full_opinion(self):
        m = self.resolve(intent="legal_opinion", analysis_mode="bad_mode")
        self.assertEqual(m.analysis_mode, "full_opinion")


class TestResolvePipelineModeRule1(unittest.TestCase):
    """Rule 1 — legal_opinion defaults search_strategy to local_only."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_legal_opinion_defaults_to_local_only(self):
        m = self.resolve(intent="legal_opinion")
        self.assertEqual(m.search_strategy, "local_only")

    def test_legal_opinion_explicit_local_then_web_honored(self):
        m = self.resolve(intent="legal_opinion", search_strategy="local_then_web")
        self.assertEqual(m.search_strategy, "local_then_web")

    def test_analysis_mode_honored_for_opinion(self):
        for mode in ("full_opinion", "bare_acts_only", "precedents_only"):
            m = self.resolve(intent="legal_opinion", analysis_mode=mode)
            self.assertEqual(m.analysis_mode, mode)


class TestResolvePipelineModeRule2(unittest.TestCase):
    """Rule 2 — search/lookup default search_strategy to local_then_web."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_search_defaults_to_local_then_web(self):
        m = self.resolve(intent="search")
        self.assertEqual(m.search_strategy, "local_then_web")

    def test_lookup_defaults_to_local_then_web(self):
        m = self.resolve(intent="lookup")
        self.assertEqual(m.search_strategy, "local_then_web")

    def test_analysis_mode_normalized_to_full_for_non_opinion(self):
        m = self.resolve(intent="search", analysis_mode="bare_acts_only")
        # analysis_mode is not meaningful for non-opinion intents; normalized to full_opinion
        self.assertEqual(m.analysis_mode, "full_opinion")


class TestResolvePipelineModeRule4(unittest.TestCase):
    """Rule 4 — interactive_fast_path is a derived flag, never set directly."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_fast_path_true_when_all_conditions_met(self):
        m = self.resolve(
            intent="legal_opinion",
            search_strategy="local_only",
            retrieval_only=False,
            _env_fast_path=True,
        )
        self.assertTrue(m.interactive_fast_path)

    def test_fast_path_false_when_retrieval_only(self):
        m = self.resolve(
            intent="legal_opinion",
            search_strategy="local_only",
            retrieval_only=True,
            _env_fast_path=True,
        )
        self.assertFalse(m.interactive_fast_path)

    def test_fast_path_false_when_env_disabled(self):
        m = self.resolve(
            intent="legal_opinion",
            search_strategy="local_only",
            retrieval_only=False,
            _env_fast_path=False,
        )
        self.assertFalse(m.interactive_fast_path)

    def test_fast_path_false_for_non_opinion_intent(self):
        m = self.resolve(
            intent="search",
            search_strategy="local_only",
            retrieval_only=False,
            _env_fast_path=True,
        )
        self.assertFalse(m.interactive_fast_path)

    def test_fast_path_false_when_strategy_is_local_then_web(self):
        m = self.resolve(
            intent="legal_opinion",
            search_strategy="local_then_web",
            retrieval_only=False,
            _env_fast_path=True,
        )
        self.assertFalse(m.interactive_fast_path)

    def test_fast_path_false_for_bulk_ingest(self):
        m = self.resolve(intent="bulk_ingest", _env_fast_path=True)
        self.assertFalse(m.interactive_fast_path)

    def test_fast_path_false_for_generic_chat(self):
        m = self.resolve(intent="generic_chat", _env_fast_path=True)
        self.assertFalse(m.interactive_fast_path)


class TestResolvePipelineModeRule5(unittest.TestCase):
    """Rule 5 — web_only is deprecated and promoted to local_then_web."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_web_only_promoted_to_local_then_web(self):
        m = self.resolve(intent="legal_opinion", search_strategy="web_only")
        self.assertEqual(m.search_strategy, "local_then_web")

    def test_web_only_promotion_disables_fast_path(self):
        m = self.resolve(
            intent="legal_opinion",
            search_strategy="web_only",
            _env_fast_path=True,
        )
        # After promotion to local_then_web, fast_path requires local_only — so False
        self.assertFalse(m.interactive_fast_path)


class TestPipelineModeHelpers(unittest.TestCase):
    """PipelineMode.is_opinion() and is_research() classify correctly."""

    def setUp(self):
        from pipeline.mode import resolve_pipeline_mode
        self.resolve = resolve_pipeline_mode

    def test_is_opinion_true_for_legal_opinion(self):
        m = self.resolve(intent="legal_opinion")
        self.assertTrue(m.is_opinion())
        self.assertFalse(m.is_research())

    def test_is_research_true_for_search_and_lookup(self):
        for intent in ("search", "lookup"):
            m = self.resolve(intent=intent)
            self.assertTrue(m.is_research(), f"{intent} should be research")
            self.assertFalse(m.is_opinion(), f"{intent} should not be opinion")

    def test_neither_for_bulk_ingest(self):
        m = self.resolve(intent="bulk_ingest")
        self.assertFalse(m.is_opinion())
        self.assertFalse(m.is_research())

    def test_summary_contains_all_flags(self):
        m = self.resolve(intent="legal_opinion", _env_fast_path=True)
        s = m.summary
        self.assertIn("intent=", s)
        self.assertIn("strategy=", s)
        self.assertIn("analysis=", s)
        self.assertIn("fast=", s)

    def test_mode_is_frozen(self):
        from pipeline.mode import resolve_pipeline_mode
        m = resolve_pipeline_mode()
        with self.assertRaises((AttributeError, TypeError)):
            m.intent = "hacked"  # type: ignore[misc]


# ===========================================================================
# Issue #10 — agents/intake/schema.py
# ===========================================================================

class TestMakeIntakeState(unittest.TestCase):
    """make_intake_state() returns a fully populated fresh dict."""

    def setUp(self):
        from agents.intake.schema import make_intake_state, INTAKE_DEFAULTS
        self.make = make_intake_state
        self.defaults = INTAKE_DEFAULTS

    def test_all_default_keys_present(self):
        state = self.make()
        for key in self.defaults:
            self.assertIn(key, state, f"missing key: {key}")

    def test_list_fields_are_empty_lists_not_shared(self):
        s1 = self.make()
        s2 = self.make()
        s1["known_facts"].append("x")
        self.assertEqual(s2["known_facts"], [], "list fields must not share references")

    def test_dict_fields_are_empty_dicts_not_shared(self):
        s1 = self.make()
        s2 = self.make()
        s1["case_file"]["k"] = "v"
        self.assertEqual(s2["case_file"], {})

    def test_stage_defaults_to_stage1(self):
        self.assertEqual(self.make()["stage"], "stage1")

    def test_turn_count_is_zero(self):
        self.assertEqual(self.make()["turn_count"], 0)

    def test_bool_fields_are_false(self):
        state = self.make()
        for key in ("detail_request_issued", "analysis_ready", "ready_for_stage2"):
            self.assertIs(state[key], False, f"{key} should be False")


class TestCoerceIntakeStateMissingKeys(unittest.TestCase):
    """coerce_intake_state() fills in missing keys without overwriting present ones."""

    def setUp(self):
        from agents.intake.schema import coerce_intake_state
        self.coerce = coerce_intake_state

    def test_empty_dict_gets_all_defaults(self):
        from agents.intake.schema import INTAKE_DEFAULTS
        state = self.coerce({})
        for key in INTAKE_DEFAULTS:
            self.assertIn(key, state)

    def test_none_input_returns_fresh_state(self):
        from agents.intake.schema import INTAKE_DEFAULTS
        state = self.coerce(None)
        for key in INTAKE_DEFAULTS:
            self.assertIn(key, state)

    def test_non_dict_input_returns_fresh_state(self):
        state = self.coerce("oops")  # type: ignore[arg-type]
        self.assertIn("stage", state)

    def test_existing_string_value_preserved(self):
        state = self.coerce({"issue_summary": "Someone hit me"})
        self.assertEqual(state["issue_summary"], "Someone hit me")

    def test_existing_list_value_preserved(self):
        facts = [{"fact": "x"}]
        state = self.coerce({"known_facts": facts})
        self.assertEqual(state["known_facts"], facts)

    def test_does_not_mutate_input_dict(self):
        raw = {"issue_summary": "test"}
        original_keys = set(raw.keys())
        self.coerce(raw)
        self.assertEqual(set(raw.keys()), original_keys)


class TestCoerceIntakeStateTypeNormalisation(unittest.TestCase):
    """coerce_intake_state() fixes wrong types on known fields."""

    def setUp(self):
        from agents.intake.schema import coerce_intake_state
        self.coerce = coerce_intake_state

    def test_none_list_field_becomes_empty_list(self):
        state = self.coerce({"known_facts": None})
        self.assertEqual(state["known_facts"], [])

    def test_string_list_field_becomes_empty_list(self):
        state = self.coerce({"open_questions": "some string"})
        self.assertEqual(state["open_questions"], [])

    def test_non_bool_truthy_becomes_true(self):
        state = self.coerce({"detail_request_issued": 1})
        self.assertIs(state["detail_request_issued"], True)

    def test_non_bool_falsy_becomes_false(self):
        state = self.coerce({"analysis_ready": 0})
        self.assertIs(state["analysis_ready"], False)

    def test_string_int_field_coerced_to_int(self):
        state = self.coerce({"turn_count": "3"})
        self.assertEqual(state["turn_count"], 3)

    def test_invalid_int_falls_back_to_zero(self):
        state = self.coerce({"turn_count": "not_a_number"})
        self.assertEqual(state["turn_count"], 0)

    def test_none_dict_field_becomes_empty_dict(self):
        state = self.coerce({"case_file": None})
        self.assertEqual(state["case_file"], {})

    def test_string_dict_field_becomes_empty_dict(self):
        state = self.coerce({"remedy_detail": "broken"})
        self.assertEqual(state["remedy_detail"], {})

    def test_all_list_fields_reset_when_none(self):
        from agents.intake.schema import _LIST_FIELDS
        broken = {k: None for k in _LIST_FIELDS}
        state = self.coerce(broken)
        for key in _LIST_FIELDS:
            self.assertIsInstance(state[key], list, f"{key} should be list")

    def test_unknown_keys_preserved(self):
        state = self.coerce({"__custom_ui_flag": True})
        self.assertIn("__custom_ui_flag", state)
        self.assertIs(state["__custom_ui_flag"], True)


class TestAnchorFieldsPresent(unittest.TestCase):
    """anchor_fields_present() gates Stage 1→2 transition correctly."""

    def setUp(self):
        from agents.intake.schema import anchor_fields_present, make_intake_state
        self.check = anchor_fields_present
        self.make = make_intake_state

    def test_fresh_state_is_not_ready(self):
        self.assertFalse(self.check(self.make()))

    def test_all_anchors_populated_returns_true(self):
        state = self.make()
        state["issue_summary"] = "Got assaulted"
        state["relationship_to_other_party"] = "stranger"
        state["timeframe_status"] = "recent"
        state["client_goal_initial"] = "file a complaint"
        self.assertTrue(self.check(state))

    def test_partial_anchors_returns_false(self):
        state = self.make()
        state["issue_summary"] = "Got assaulted"
        state["relationship_to_other_party"] = "stranger"
        # timeframe_status and client_goal_initial still None
        self.assertFalse(self.check(state))

    def test_blank_sentinel_unknown_is_not_ready(self):
        state = self.make()
        state["issue_summary"] = "Got assaulted"
        state["relationship_to_other_party"] = "stranger"
        state["timeframe_status"] = "unknown"
        state["client_goal_initial"] = "file a complaint"
        self.assertFalse(self.check(state))

    def test_blank_sentinel_none_str_is_not_ready(self):
        state = self.make()
        state["issue_summary"] = "Got assaulted"
        state["relationship_to_other_party"] = "stranger"
        state["timeframe_status"] = "recent"
        state["client_goal_initial"] = "none"
        self.assertFalse(self.check(state))

    def test_whitespace_only_is_not_ready(self):
        state = self.make()
        state["issue_summary"] = "   "
        state["relationship_to_other_party"] = "stranger"
        state["timeframe_status"] = "recent"
        state["client_goal_initial"] = "file a complaint"
        self.assertFalse(self.check(state))

    def test_empty_string_is_not_ready(self):
        state = self.make()
        state["issue_summary"] = ""
        state["relationship_to_other_party"] = "stranger"
        state["timeframe_status"] = "recent"
        state["client_goal_initial"] = "file a complaint"
        self.assertFalse(self.check(state))

    def test_null_string_sentinel_is_not_ready(self):
        state = self.make()
        state["issue_summary"] = "Got assaulted"
        state["relationship_to_other_party"] = "null"
        state["timeframe_status"] = "recent"
        state["client_goal_initial"] = "file a complaint"
        self.assertFalse(self.check(state))


# ===========================================================================
# Cross-cut — schema + mode working together as chat.py does
# ===========================================================================

class TestSchemaModeIntegration(unittest.TestCase):
    """
    Simulate the chat.py pipeline boundary:
      1. coerce_intake_state() on incoming session state
      2. anchor_fields_present() to decide if Stage 2 is allowed
      3. resolve_pipeline_mode() to build a PipelineMode for response generation
    """

    def setUp(self):
        from agents.intake.schema import coerce_intake_state, anchor_fields_present
        from pipeline.mode import resolve_pipeline_mode
        self.coerce = coerce_intake_state
        self.anchors_ok = anchor_fields_present
        self.resolve = resolve_pipeline_mode

    def test_incomplete_intake_blocks_stage2(self):
        raw = {"issue_summary": "Harassment at work", "turn_count": "2"}
        state = self.coerce(raw)
        self.assertFalse(self.anchors_ok(state))

    def test_complete_intake_allows_stage2(self):
        raw = {
            "issue_summary": "Employer terminated without notice",
            "relationship_to_other_party": "employer",
            "timeframe_status": "recent",
            "client_goal_initial": "compensation and reinstatement",
            "turn_count": "4",
        }
        state = self.coerce(raw)
        self.assertTrue(self.anchors_ok(state))

    def test_mode_resolves_for_opinion_after_intake(self):
        """Once intake is complete, pipeline resolves a legal_opinion mode."""
        raw = {
            "issue_summary": "Property dispute",
            "relationship_to_other_party": "neighbour",
            "timeframe_status": "ongoing",
            "client_goal_initial": "recover possession",
        }
        state = self.coerce(raw)
        self.assertTrue(self.anchors_ok(state))
        mode = self.resolve(intent="legal_opinion", _env_fast_path=True)
        self.assertTrue(mode.interactive_fast_path)
        self.assertTrue(mode.is_opinion())

    def test_coerce_then_mode_search_never_fast_path(self):
        raw = {}
        state = self.coerce(raw)
        # Search intent never triggers fast path regardless of intake completeness
        mode = self.resolve(intent="search", _env_fast_path=True)
        self.assertFalse(mode.interactive_fast_path)

    def test_retrieval_only_disables_fast_path_even_with_complete_intake(self):
        raw = {
            "issue_summary": "Assault",
            "relationship_to_other_party": "stranger",
            "timeframe_status": "recent",
            "client_goal_initial": "file FIR",
        }
        state = self.coerce(raw)
        self.assertTrue(self.anchors_ok(state))
        mode = self.resolve(
            intent="legal_opinion",
            search_strategy="local_only",
            retrieval_only=True,
            _env_fast_path=True,
        )
        self.assertFalse(mode.interactive_fast_path)

    def test_coerced_turn_count_is_int(self):
        state = self.coerce({"turn_count": "7"})
        self.assertIsInstance(state["turn_count"], int)
        self.assertEqual(state["turn_count"], 7)

    def test_bad_state_from_session_gets_all_defaults(self):
        """Simulates a corrupt/missing session blob arriving at the pipeline."""
        from agents.intake.schema import INTAKE_DEFAULTS
        state = self.coerce(None)
        for key in INTAKE_DEFAULTS:
            self.assertIn(key, state)
        self.assertFalse(self.anchors_ok(state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
