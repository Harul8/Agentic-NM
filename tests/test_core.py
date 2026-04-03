"""
Nyaymalaw 4.0 — Core Unit Test Suite
======================================
Tests for:
  1. fact_collector  — _is_duplicate_question, dedup guard behaviour, _run_single_gate (mocked)
  2. hybrid_retriever — legal_term_boost, BM25 tokenise/score, _is_final_judgment_chunk
  3. response_generator_v2 — quality filters, flexible limit, _build_bare_act_queries helpers
  4. citation_graph — stable_id, normalisation helpers, get_case_authority_score stub

Run with:
    cd "Nyaymalaw 4.0"
    python -m pytest tests/test_core.py -v
"""

import sys
import os
import json
import types
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Stub heavy native modules so tests work in environments without GPU libs.
# faiss, numpy, sentence_transformers are not available in the CI/test VM.
# ---------------------------------------------------------------------------
_faiss_stub = types.ModuleType("faiss")
_faiss_stub.read_index = lambda *a, **kw: None
_faiss_stub.write_index = lambda *a, **kw: None
sys.modules.setdefault("faiss", _faiss_stub)

_numpy_stub = types.ModuleType("numpy")
_numpy_stub.array = lambda *a, **kw: a[0] if a else None
_numpy_stub.float32 = float
sys.modules.setdefault("numpy", _numpy_stub)

for _mod in (
    "sentence_transformers", "sentence_transformers.models",
    "torch", "torch.cuda",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)


# ===========================================================================
# 1. fact_collector tests
# ===========================================================================

class TestIsDuplicateQuestion(unittest.TestCase):
    """Tests for _is_duplicate_question (code-level dedup guard)."""

    def _load(self):
        from services.fact_collector import _is_duplicate_question
        return _is_duplicate_question

    def test_exact_topic_overlap_returns_true(self):
        fn = self._load()
        # Both questions contain "location" which is a dedup keyword
        self.assertTrue(fn(
            "Can you tell me the location of the assault?",
            ["What is the location where this happened?"],
        ))

    def test_no_shared_keywords_returns_false(self):
        fn = self._load()
        # "location" is not in "When did you sign the contract?"
        self.assertFalse(fn(
            "What is the location of the incident?",
            ["What is the total amount owed?"],
        ))

    def test_empty_asked_list_returns_false(self):
        fn = self._load()
        self.assertFalse(fn("What is the location?", []))

    def test_empty_proposed_returns_false(self):
        fn = self._load()
        self.assertFalse(fn("", ["Where did the incident happen?"]))

    def test_multiple_asked_questions_any_match(self):
        fn = self._load()
        # "injury" is a dedup keyword; appears in both proposed and asked
        self.assertTrue(fn(
            "Can you describe the injury in detail?",
            ["What were you earning before the injury?", "How was the injured party treated?"],
        ))

    def test_fir_topic_matches(self):
        fn = self._load()
        # "fir" is in the dedup keywords set
        self.assertTrue(fn(
            "Have you filed a FIR with the police?",
            ["Did you submit a FIR at the local station?"],
        ))


class TestRunSingleGateMocked(unittest.TestCase):
    """Tests for _run_single_gate with mocked LLM."""

    def _load(self):
        from services.fact_collector import _run_single_gate
        return _run_single_gate

    @patch("services.fact_collector.ask_llm")
    def test_returns_ask_action(self, mock_llm):
        mock_llm.return_value = '{"action":"ask","reply_to_client":"What is the location?"}'
        fn = self._load()
        result = fn([], "Someone hit me")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "ask")

    @patch("services.fact_collector.ask_llm")
    def test_returns_complete_action(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "action": "complete",
            "intent": "legal_opinion",
            "facts_summary": "Assault case",
            "reply_to_client": "Understood.",
        })
        fn = self._load()
        result = fn([], "Someone assaulted me last month in Hyderabad")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "complete")

    @patch("services.fact_collector.ask_llm")
    def test_dedup_guard_forces_complete(self, mock_llm):
        """If LLM proposes a duplicate question, the dedup guard forces complete."""
        # Conversation already has an assistant question asking about location
        history = [
            {"role": "user", "content": "Someone hit me"},
            {"role": "assistant", "content": "What is the location where this happened?"},
        ]
        # LLM still proposes an ask about location (same dedup keyword)
        mock_llm.return_value = '{"action":"ask","reply_to_client":"Can you tell me the location of the assault?"}'
        fn = self._load()
        result = fn(history, "It was in Hyderabad")
        # Dedup guard should intercept and force complete
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "complete")

    @patch("services.fact_collector.ask_llm")
    def test_greeting_passthrough(self, mock_llm):
        mock_llm.return_value = '{"action":"greeting","reply_to_client":"Hello!"}'
        fn = self._load()
        result = fn([], "hi")
        # greeting action must pass through _parse_llm_response without being rejected
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "greeting")

    def test_streaming_collects_full_response(self):
        """When token_callback is provided, tokens are collected and full response is parsed."""
        full_json = '{"action":"ask","reply_to_client":"What is the location?"}'
        tokens = [full_json[:10], full_json[10:25], full_json[25:]]

        collected = []

        def fake_stream(prompt, **kw):
            yield from tokens

        # Patch at the source module level (imported inside the if-block)
        with patch("llm.ollama_client.ask_llm_stream", side_effect=fake_stream):
            fn = self._load()
            result = fn([], "assault happened", token_callback=lambda t: collected.append(t))

        self.assertEqual("".join(collected), full_json)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "ask")


class TestGetNextQuestionAcceptsTokenCallback(unittest.TestCase):
    """get_next_question_or_complete must accept and pass token_callback."""

    @patch("services.fact_collector.ask_llm")
    def test_signature_accepts_token_callback(self, mock_llm):
        mock_llm.return_value = '{"action":"complete","intent":"legal_opinion","facts_summary":"test"}'
        from services.fact_collector import get_next_question_or_complete
        tokens = []
        # Should not raise TypeError
        result = get_next_question_or_complete([], "test query", token_callback=lambda t: tokens.append(t))
        self.assertIn("action", result)


# ===========================================================================
# 2. hybrid_retriever tests
# ===========================================================================

class TestLegalTermBoost(unittest.TestCase):
    """Tests for legal_term_boost."""

    def _load(self):
        from retrieval.hybrid_retriever import legal_term_boost
        return legal_term_boost

    def test_matching_section_returns_positive_boost(self):
        fn = self._load()
        boost = fn("IPC section 302 murder", "This case involves section 302 of Indian Penal Code")
        self.assertGreater(boost, 0.0)

    def test_no_match_returns_zero(self):
        fn = self._load()
        boost = fn("section 302 IPC", "The petitioner filed a writ petition under Article 226")
        self.assertEqual(boost, 0.0)

    def test_act_name_match(self):
        fn = self._load()
        boost = fn("Transfer of Property Act lease", "The Transfer of Property Act governs lease agreements")
        self.assertGreater(boost, 0.0)

    def test_empty_query_returns_zero(self):
        fn = self._load()
        self.assertEqual(fn("", "some document text"), 0.0)

    def test_empty_text_returns_zero(self):
        fn = self._load()
        self.assertEqual(fn("section 302 IPC", ""), 0.0)

    def test_result_bounded_between_zero_and_one(self):
        fn = self._load()
        boost = fn(
            "section 302 section 307 section 323 IPC Indian Penal Code",
            "section 302 section 307 section 323 Indian Penal Code"
        )
        self.assertGreaterEqual(boost, 0.0)
        self.assertLessEqual(boost, 1.0)


class TestBM25(unittest.TestCase):
    """Tests for the BM25 implementation."""

    def _load(self):
        from retrieval.hybrid_retriever import BM25
        return BM25

    def test_fit_and_score_basic(self):
        BM25 = self._load()
        bm25 = BM25()
        docs = [
            "The Indian Penal Code section 302 deals with murder",
            "Transfer of Property Act governs lease and rent",
            "Arbitration and Conciliation Act",
        ]
        bm25.fit(docs)
        results = bm25.score("Indian Penal Code murder", top_k=3)
        # Doc 0 should rank first
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], 0)

    def test_hyphenated_section_normalised(self):
        BM25 = self._load()
        bm25 = BM25()
        bm25.fit(["Section 498-A of IPC deals with cruelty"])
        results = bm25.score("section 498a ipc")
        self.assertGreater(len(results), 0)

    def test_serialise_roundtrip(self):
        BM25 = self._load()
        bm25 = BM25()
        bm25.fit(["Section 302 murder", "Section 307 attempt to murder"])
        d = bm25.to_dict()
        bm25b = BM25.from_dict(d)
        r1 = bm25.score("302 murder")
        r2 = bm25b.score("302 murder")
        self.assertEqual(len(r1), len(r2))
        if r1 and r2:
            self.assertAlmostEqual(r1[0][1], r2[0][1], places=5)

    def test_empty_corpus_returns_no_results(self):
        BM25 = self._load()
        bm25 = BM25()
        bm25.fit([])
        self.assertEqual(bm25.score("test"), [])


class TestNormalizeLegalQuery(unittest.TestCase):
    """normalize_legal_query should expand abbreviations."""

    def _load(self):
        from retrieval.hybrid_retriever import normalize_legal_query
        return normalize_legal_query

    def test_ipc_expanded(self):
        fn = self._load()
        result = fn("IPC section 302")
        self.assertIn("Indian Penal Code", result)

    def test_bns_expanded(self):
        fn = self._load()
        result = fn("BNS section 115")
        self.assertIn("Bharatiya Nyaya Sanhita", result)

    def test_no_abbreviation_unchanged(self):
        fn = self._load()
        result = fn("Transfer of Property Act lease")
        self.assertEqual(result, "Transfer of Property Act lease")


# ===========================================================================
# 3. response_generator_v2 tests
# ===========================================================================

class TestIsQualityCaseLaw(unittest.TestCase):
    def _load(self):
        from services.response_generator_v2 import _is_quality_case_law
        return _is_quality_case_law

    def test_valid_case_passes(self):
        fn = self._load()
        self.assertTrue(fn({
            "case_name": "State v. Ramu",
            "year": 2015,
            "full_text": "The Supreme Court held that the accused was guilty under Section 302 IPC." * 3,
        }))

    def test_missing_case_name_fails(self):
        fn = self._load()
        self.assertFalse(fn({"year": 2010, "full_text": "x" * 100}))

    def test_unknown_case_name_fails(self):
        fn = self._load()
        self.assertFalse(fn({"case_name": "Unknown", "year": 2010, "full_text": "x" * 100}))

    def test_too_old_fails(self):
        fn = self._load()
        self.assertFalse(fn({"case_name": "Old Case", "year": 1950, "full_text": "x" * 100}))

    def test_short_text_fails(self):
        fn = self._load()
        self.assertFalse(fn({"case_name": "Test Case", "year": 2010, "full_text": "short"}))


class TestIsQualityBareAct(unittest.TestCase):
    def _load(self):
        from services.response_generator_v2 import _is_quality_bare_act
        return _is_quality_bare_act

    def test_valid_section_passes(self):
        fn = self._load()
        self.assertTrue(fn({
            "act_name": "Indian Penal Code",
            "section_number": "302",
            "full_text": "Whoever commits murder shall be punished with death or life imprisonment." * 2,
        }))

    def test_missing_act_name_fails(self):
        fn = self._load()
        self.assertFalse(fn({"section_number": "302", "full_text": "x" * 100}))

    def test_missing_section_fails(self):
        fn = self._load()
        self.assertFalse(fn({"act_name": "IPC", "full_text": "x" * 100}))

    def test_short_text_fails(self):
        fn = self._load()
        self.assertFalse(fn({"act_name": "IPC", "section_number": "302", "full_text": "short"}))

    def test_unknown_act_name_fails(self):
        fn = self._load()
        self.assertFalse(fn({"act_name": "unknown", "section_number": "302", "full_text": "x" * 100}))


class TestApplyFlexibleResultLimit(unittest.TestCase):
    def _load(self):
        from services.response_generator_v2 import _apply_flexible_result_limit
        return _apply_flexible_result_limit

    def _make_items(self, scores):
        return [{"_rerank_score": s, "id": i} for i, s in enumerate(scores)]

    def test_user_limit_respected(self):
        fn = self._load()
        items = self._make_items([9, 8, 7, 6, 5, 4, 3])
        result = fn(items, user_limit=3)
        self.assertEqual(len(result), 3)

    def test_high_score_items_all_included_when_above_fallback(self):
        fn = self._load()
        items = self._make_items([6.0] * 12)  # 12 items all above HIGH_QUALITY_SCORE (5.0)
        result = fn(items)
        self.assertGreaterEqual(len(result), 10)  # all qualify; >= FLEXIBLE_MIN_FALLBACK

    def test_empty_list_returns_empty(self):
        fn = self._load()
        self.assertEqual(fn([]), [])

    def test_fallback_fills_to_min(self):
        fn = self._load()
        # Only 2 items above HIGH_QUALITY_SCORE; 8 between min and high
        items = self._make_items([6.0, 5.5] + [1.0] * 8)
        result = fn(items)
        # Should fill up to FLEXIBLE_MIN_FALLBACK (10)
        self.assertGreaterEqual(len(result), 10)


class TestExpandLegalQueryMocked(unittest.TestCase):
    """expand_legal_query returns a non-empty list when the model or fallback supplies queries."""

    @patch("services.response_generator_v2.ask_llm")
    def test_returns_list_of_queries(self, mock_llm):
        mock_llm.return_value = "assault and battery under Indian Penal Code section 323"
        from services.response_generator_v2 import expand_legal_query
        queries = expand_legal_query("Someone beat me up in Hyderabad")
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)

    @patch("services.response_generator_v2.ask_llm")
    def test_llm_failure_returns_fallback(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        from services.response_generator_v2 import expand_legal_query
        queries = expand_legal_query("property dispute")
        self.assertGreaterEqual(len(queries), 1)
        # Fallback is the raw facts[:300]
        self.assertIn("property dispute", queries[0])

    @patch("services.response_generator_v2.ask_llm")
    def test_issues_shape_flattens_queries(self, mock_llm):
        mock_llm.return_value = (
            '{"issues":['
            '{"issue_label":"rent default","queries":['
            '"tenant rent arrears Mumbai lease",'
            '"landlord recovery unpaid rent"]},'
            '{"issue_label":"eviction","queries":['
            '"eviction notice possession Mumbai",'
            '"lease termination breach tenant"]}'
            "]}"
        )
        from services.response_generator_v2 import expand_legal_query
        dbg = {}
        facts = (
            "My tenant in Mumbai stopped paying rent under the lease; "
            "landlord wants eviction and recovery of possession."
        )
        queries = expand_legal_query(facts, expansion_debug=dbg)
        self.assertGreaterEqual(len(queries), 2)
        self.assertEqual(len(dbg.get("issues_from_model") or []), 2)


# ===========================================================================
# 4. citation_graph tests
# ===========================================================================

class TestCitationGraphHelpers(unittest.TestCase):

    def test_stable_case_id_deterministic(self):
        from retrieval.citation_graph import _stable_case_id
        id1 = _stable_case_id("State v. Ramu", "2015")
        id2 = _stable_case_id("State v. Ramu", "2015")
        self.assertEqual(id1, id2)

    def test_stable_case_id_different_cases_differ(self):
        from retrieval.citation_graph import _stable_case_id
        id1 = _stable_case_id("State v. Ramu", "2015")
        id2 = _stable_case_id("Ramu v. State", "2015")
        self.assertNotEqual(id1, id2)

    def test_normalize_case_name_key_lowercases(self):
        from retrieval.citation_graph import _normalize_case_name_key
        self.assertEqual(
            _normalize_case_name_key("State v. RAMU"),
            "state v. ramu",
        )

    def test_normalize_for_cites_lookup_vs_to_v(self):
        from retrieval.citation_graph import _normalize_for_cites_lookup
        result = _normalize_for_cites_lookup("State vs. Ramu")
        # "vs." should be replaced with "v"; the dot after v comes from the period
        # in the original string — the key requirement is "vs." is gone
        self.assertNotIn("vs.", result)
        # Result should still contain "v" representing the separator
        self.assertIn(" v", result)

    def test_clean_cited_name_strips_noise(self):
        from retrieval.citation_graph import _clean_cited_name
        self.assertEqual(_clean_cited_name("  State v. Ramu.  "), "State v. Ramu")

    def test_get_case_authority_score_no_graph_returns_zero(self):
        from retrieval.citation_graph import get_case_authority_score, invalidate_graph
        invalidate_graph()
        with patch("retrieval.citation_graph.get_graph", return_value=None):
            score = get_case_authority_score("State v. Ramu", "2015")
        self.assertEqual(score, 0.0)

    def test_build_and_query_citation_graph(self):
        """Build a minimal graph from synthetic chunks and verify PageRank / edges."""
        from retrieval.citation_graph import build_citation_graph_from_chunks
        import tempfile, json

        chunks = {
            "0": {
                "doc_type": "case_law",
                "case_name": "Alpha v. Beta",
                "court": "supreme_court",
                "year": "2020",
                "sections_cited": ["IPC 302"],
                "cited_cases": ["Gamma v. Delta"],
            },
            "1": {
                "doc_type": "case_law",
                "case_name": "Gamma v. Delta",
                "court": "high_court",
                "year": "2018",
                "sections_cited": ["IPC 307"],
                "cited_cases": [],
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out_path = f.name

        try:
            graph = build_citation_graph_from_chunks(chunks, out_path)
            # Two cases should be in the graph
            self.assertGreaterEqual(len(graph["case_info"]), 2)
            # At least one "interprets" edge (IPC 302 or IPC 307)
            self.assertGreater(len(graph["interprets"]), 0)
            # Alpha→Gamma cite edge
            self.assertGreater(len(graph["cites"]), 0)
            # Authority scores assigned
            for cid, info in graph["case_info"].items():
                if not info.get("is_stub"):
                    self.assertIn("authority_score", info)
        finally:
            os.unlink(out_path)


# ===========================================================================
# 5. Multi-query retrieval integration check (no real index needed)
# ===========================================================================

class TestBuildBareActQueries(unittest.TestCase):
    """_build_bare_act_queries should return ≥1 and ≤6 diverse queries."""

    @patch("services.response_generator_v2.ask_llm")
    def test_returns_multiple_queries(self, mock_llm):
        mock_llm.return_value = '{"queries":["criminal assault","hurt","bodily harm"]}'
        from services.response_generator_v2 import _build_bare_act_queries
        dispute = {
            "id": "d1",
            "dispute": "Someone beat me with a rod and threatened to kill me",
            "keywords": ["assault", "threat"],
            "legal_nature": "criminal",
            "search_angles": ["criminal assault causing hurt", "criminal intimidation threat of injury"],
        }
        queries = _build_bare_act_queries(dispute)
        self.assertGreaterEqual(len(queries), 1)
        self.assertLessEqual(len(queries), 6)
        # Every query must be a non-empty string
        for q in queries:
            self.assertIsInstance(q, str)
            self.assertGreater(len(q.strip()), 0)

    @patch("services.response_generator_v2.ask_llm")
    def test_no_duplicates(self, mock_llm):
        mock_llm.return_value = '{"queries":[]}'
        from services.response_generator_v2 import _build_bare_act_queries
        dispute = {
            "id": "d2",
            "dispute": "Tenant did not pay rent",
            "keywords": ["rent", "tenant", "eviction"],
            "legal_nature": "civil",
            "search_angles": [],
        }
        queries = _build_bare_act_queries(dispute)
        self.assertEqual(len(queries), len(set(q.lower() for q in queries)))


# ===========================================================================
# 6. Streaming intake integration check
# ===========================================================================

class TestStreamingIntake(unittest.TestCase):
    """Verify token_callback is invoked when streaming is requested."""

    def test_tokens_streamed_for_ask_response(self):
        """Tokens emitted by ask_llm_stream are forwarded to token_callback."""
        full_json = '{"action":"ask","reply_to_client":"Where did the incident occur?"}'

        def fake_stream(prompt, **kw):
            yield full_json[:15]
            yield full_json[15:]

        from services.fact_collector import _run_single_gate
        collected = []
        with patch("llm.ollama_client.ask_llm_stream", side_effect=fake_stream):
            result = _run_single_gate([], "Someone hit me", token_callback=lambda t: collected.append(t))

        self.assertEqual("".join(collected), full_json)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "ask")

    @patch("services.fact_collector.ask_llm")
    def test_no_callback_uses_regular_ask_llm(self, mock_llm):
        """Without token_callback, ask_llm (not ask_llm_stream) is called."""
        mock_llm.return_value = '{"action":"ask","reply_to_client":"What is the nature of the dispute?"}'
        from services.fact_collector import _run_single_gate
        result = _run_single_gate([], "I have a legal problem")
        mock_llm.assert_called_once()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
