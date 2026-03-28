"""
End-to-end smoke test — Domestic Violence case scenario
========================================================
Tests the full pipeline without a running server by:
  1. Patching ask_llm with deterministic mock responses
  2. Driving the intake (fact_collector) through a realistic DV conversation
  3. Driving the bare-acts response stage
  4. Asserting all the properties we care about

Checks covered
--------------
[A] Intake — no cheque/finance topics injected for a DV case
[B] Intake — duplicate question detection (preamble-diluted case)
[C] Intake — conversation principles (empathy → issue → why → question structure)
[D] Routing — analysis_mode="bare_acts_only" set correctly after confirmation
[E] Bare acts response — opinion_text is clean prose (no "Section N points to N.")
[F] Bare acts response — no case laws in the response
[G] Bare acts response — next_steps array present and non-empty
[H] Fallback prose — _build_grounded_bare_act_fallback labels only, no raw text
[I] Fallback prose — _build_grounded_interactive_fallback labels only, no raw text
[J] Retry logic — _generate_bare_act_stage_text retries on LLM failure
"""

import sys, json, re, copy, textwrap, traceback
sys.path.insert(0, ".")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
HEAD = "\033[1;34m"
RESET = "\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((label, condition, detail))
    print(f"  {status}  {label}" + (f"\n       {detail}" if detail and not condition else ""))

def section(title):
    print(f"\n{HEAD}{'─'*60}{RESET}")
    print(f"{HEAD}  {title}{RESET}")
    print(f"{HEAD}{'─'*60}{RESET}")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _has_cheque_topic(text: str) -> bool:
    cheque_signals = [
        "cheque", "dishonour", "negotiable instruments", "section 138",
        "finance arrangement", "security cheque", "financing papers",
        "demand draft", "promissory note",
    ]
    t = text.lower()
    return any(s in t for s in cheque_signals)


def _has_embedded_section_text(text: str) -> bool:
    """Detect the garbled pattern: 'Section N points to N.' or 'Section N supports protection through N.'"""
    return bool(re.search(
        r"Section\s+\d+\s+(?:points to|supports protection through)\s+\d+[\.\s]",
        text,
        re.IGNORECASE,
    ))


# ──────────────────────────────────────────────
# [A-C] Intake unit tests (no LLM needed)
# ──────────────────────────────────────────────
section("A–C: Intake layer (fact_collector)")

from services.fact_collector import (
    _is_duplicate_question,
    _assess_model_next_reply,
)
from training.few_shot_retriever import _rank_examples

# [A] Few-shot retriever — DV query must not return cheque examples
dv_query = (
    "My husband has been physically assaulting me for the past 6 months. "
    "He threw objects at me last night and I had to call my parents. "
    "I have WhatsApp messages and a doctor's report."
)
try:
    # _rank_examples(query, desired_layers) — pass all layers to get everything ranked
    ranked = _rank_examples(dv_query, desired_layers={"intake_reply", "intake_state", "next_question"})
    domains = [ex.get("domain", "") for ex in ranked]
    cheque_hit = any("cheque" in d.lower() or "negotiable" in d.lower() for d in domains)
    check("[A] No cheque examples injected for DV query",
          not cheque_hit,
          f"Got domains: {domains}")
except Exception as e:
    check("[A] No cheque examples injected for DV query", False, str(e))

# [B] Duplicate detection — preamble must not let a duplicate through
try:
    prev_question = "What specific details about the assault are most relevant for legal action?"
    full_reply = (
        "I understand the gravity of this situation. "
        + prev_question
    )
    is_dup = False
    # Replicate what _assess_model_next_reply now does
    _reply_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_reply) if s.strip()]
    _question_sentences = [s for s in _reply_sentences if "?" in s and len(s) > 12]
    _check_texts = _question_sentences if _question_sentences else [full_reply]
    is_dup = any(_is_duplicate_question(t, [prev_question]) for t in _check_texts)
    check("[B] Preamble-wrapped duplicate correctly caught", is_dup,
          f"check_texts={_check_texts}")

    # Also confirm a genuinely new question is NOT flagged
    new_reply = "I understand. Can you tell me whether you have filed a complaint with the police yet?"
    _new_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", new_reply) if s.strip()]
    _new_qs = [s for s in _new_sentences if "?" in s and len(s) > 12]
    _new_checks = _new_qs if _new_qs else [new_reply]
    is_new_dup = any(_is_duplicate_question(t, [prev_question]) for t in _new_checks)
    check("[B] Genuinely new question NOT flagged as duplicate", not is_new_dup,
          f"check_texts={_new_checks}")
except Exception as e:
    check("[B] Duplicate detection", False, traceback.format_exc())

# [C] _assess_model_next_reply — grounding check with a well-structured reply
try:
    intake_state = {
        "facts_summary": (
            "Client reports ongoing physical violence by husband. "
            "Incidents escalating — objects thrown, bruises visible. "
            "Has WhatsApp messages and doctor's certificate."
        ),
        "known_facts": [
            "ongoing physical violence by husband",
            "escalating — objects thrown",
            "has WhatsApp messages",
            "has doctor's certificate",
        ],
        "open_points": [
            "whether police complaint has been filed",
            "whether client has a safe place to stay",
        ],
        "client_objective": "protection and safety",
        "urgency_level": "high",
        "enough_to_proceed": False,
    }
    asked = ["What specific details about the assault are most relevant?"]
    good_reply = (
        "I hear how frightening this has been. "
        "What you have described is a pattern of physical violence at home. "
        "The next detail will help me assess whether urgent protection is presently available. "
        "Have you or anyone on your behalf filed a police complaint yet, "
        "and if so, do you have the FIR or complaint number?"
    )
    ok, reasons = _assess_model_next_reply(good_reply, intake_state, asked)
    check("[C] Well-structured DV reply passes quality gate", ok,
          f"Reasons: {reasons}")
except Exception as e:
    check("[C] Quality gate", False, traceback.format_exc())


# ──────────────────────────────────────────────
# [D] Routing — analysis_mode
# ──────────────────────────────────────────────
section("D: Routing — analysis_mode=bare_acts_only")

from services.interactive_chat import _last_assistant_is_analysis_ready, _build_analysis_ready_prompt

try:
    ready_msg = _build_analysis_ready_prompt()
    conversation_at_confirmation = [
        {"role": "user", "content": "My husband beats me regularly. I have doctor reports and WhatsApp messages."},
        {"role": "assistant", "content": "I understand this is difficult. Have you filed a police complaint?"},
        {"role": "user", "content": "Yes, I filed an FIR last week. FIR number is 245/2025."},
        {"role": "assistant", "content": ready_msg},       # analysis-ready handoff
        {"role": "user", "content": "proceed"},            # user confirms
    ]
    is_ready = _last_assistant_is_analysis_ready(conversation_at_confirmation[:-1])  # exclude last user msg
    check("[D] _last_assistant_is_analysis_ready detects handoff correctly", is_ready,
          f"ready_msg prefix present: {ready_msg[:80]}")
except Exception as e:
    check("[D] Routing", False, traceback.format_exc())


# ──────────────────────────────────────────────
# [E-G] Response generation with mocked LLM
# ──────────────────────────────────────────────
section("E–G: Bare acts response (mocked LLM)")

import unittest.mock as mock

# Realistic mock data for a DV case — Protection of Women from DV Act sections
MOCK_BARE_ACTS = [
    {
        "act_name": "Protection of Women from Domestic Violence Act, 2005",
        "section_number": "18",
        "section_title": "Protection orders",
        "text": "18. Protection orders. The Magistrate may, after giving the aggrieved person and the respondent an opportunity of being heard and on being prima facie satisfied that domestic violence has taken place or is likely to take place, pass a protection order in favour of the aggrieved person and prohibit the respondent from.",
        "source_tag": "LOCAL_DB",
        "score": 0.92,
        "related_case_laws": [],
    },
    {
        "act_name": "Protection of Women from Domestic Violence Act, 2005",
        "section_number": "19",
        "section_title": "Residence orders",
        "text": "19. Residence orders. While disposing of an application under sub-section (1) of section 12, the Magistrate may, on being satisfied that domestic violence has taken place, pass a residence order.",
        "source_tag": "LOCAL_DB",
        "score": 0.87,
        "related_case_laws": [],
    },
    {
        "act_name": "Protection of Women from Domestic Violence Act, 2005",
        "section_number": "22",
        "section_title": "Compensation orders",
        "text": "22. Compensation orders. In addition to other reliefs as may be granted under this Act, the Magistrate may on an application being made by the aggrieved person, pass an order directing the respondent to pay compensation and damages for the injuries, including mental torture and emotional distress, caused by the acts of domestic violence.",
        "source_tag": "LOCAL_DB",
        "score": 0.81,
        "related_case_laws": [],
    },
]

MOCK_LLM_BARE_ACT_RESPONSE = json.dumps({
    "summary_text": (
        "The facts you have shared — ongoing physical violence, escalating incidents, "
        "and an FIR already on record — engage the Protection of Women from Domestic "
        "Violence Act, 2005 directly. The Act provides for immediate protective relief "
        "through the Magistrate, residence protection, and compensation for physical and "
        "emotional harm. Given the existing FIR and documentary evidence, the position on "
        "the record is currently supportable."
    ),
    "section_explanations": [
        {
            "act_name": "Protection of Women from Domestic Violence Act, 2005",
            "section_number": "18",
            "explanation": (
                "Section 18 allows the Magistrate to issue a protection order prohibiting the "
                "respondent from committing further acts of violence. This is the most immediate "
                "relief available and can be obtained on a prima facie showing — which the FIR "
                "and doctor's certificate together provide."
            ),
        },
        {
            "act_name": "Protection of Women from Domestic Violence Act, 2005",
            "section_number": "19",
            "explanation": (
                "Section 19 allows the Magistrate to pass a residence order ensuring you cannot "
                "be dispossessed from the shared household, or directing the respondent to provide "
                "alternate accommodation. This is directly relevant given the ongoing safety concern."
            ),
        },
        {
            "act_name": "Protection of Women from Domestic Violence Act, 2005",
            "section_number": "22",
            "explanation": (
                "Section 22 enables compensation for physical injury and emotional distress. "
                "The doctor's certificate goes directly to quantifying the harm and supports "
                "a compensation claim alongside the protection order application."
            ),
        },
    ],
    "next_steps": [
        {
            "title": "File application under PWDVA",
            "summary": (
                "File an application under Section 12 of the Protection of Women from Domestic "
                "Violence Act before the Magistrate. The FIR, WhatsApp messages, and doctor's "
                "certificate should accompany the application as supporting material."
            ),
        },
        {
            "title": "Seek interim protection order",
            "summary": (
                "At the first hearing, apply for an interim protection order under Section 18. "
                "Courts routinely grant these on prima facie material — your existing FIR and "
                "medical evidence are sufficient for this stage."
            ),
        },
        {
            "title": "Preserve and organise evidence",
            "summary": (
                "Secure certified copies of the FIR, printouts of WhatsApp messages with "
                "timestamps, and the original doctor's certificate. Keep duplicates with a "
                "trusted family member."
            ),
        },
    ],
    "next_steps_summary": (
        "File a Section 12 application under the PWDVA, seek an interim protection order "
        "at the first hearing, and secure your documentary evidence before the next date."
    ),
})

facts_summary = (
    "Client's husband has been physically assaulting her for 6 months. "
    "Incidents are escalating — objects thrown, visible bruising documented. "
    "Client has WhatsApp message evidence and a doctor's certificate. "
    "FIR filed last week, FIR number 245/2025. "
    "Client wants protection and is concerned about safety in the shared home."
)

try:
    from services.response_generator_v2 import _generate_bare_act_stage_text

    with mock.patch("services.response_generator_v2.ask_llm", return_value=MOCK_LLM_BARE_ACT_RESPONSE):
        streamed_tokens = []
        def capture_token(t):
            streamed_tokens.append(t)

        summary_text, enriched, next_steps, next_steps_summary = _generate_bare_act_stage_text(
            facts_summary=facts_summary,
            dispute_results=[],
            formatted_bare=MOCK_BARE_ACTS,
            model_override=None,
            token_callback=capture_token,
        )

    # [E] opinion_text is clean prose — no "Section N points to N." pattern
    garbled = _has_embedded_section_text(summary_text)
    check("[E] opinion_text has no garbled 'Section N points to N.' pattern",
          not garbled, f"summary_text[:200]: {summary_text[:200]}")

    # [E2] summary_text is non-empty and substantive
    check("[E2] summary_text is non-empty (>50 chars)",
          len(summary_text) > 50, f"len={len(summary_text)}")

    # [F] No case laws in enriched bare acts
    has_case_laws = any(
        bool(ba.get("related_case_laws")) for ba in enriched
    )
    check("[F] No case laws nested in bare act cards", not has_case_laws)

    # [G] next_steps present and non-empty
    check("[G] next_steps array is non-empty", len(next_steps) > 0,
          f"next_steps count: {len(next_steps)}")

    # [G2] each step has title and summary
    steps_well_formed = all(
        isinstance(s, dict) and s.get("title") and s.get("summary")
        for s in next_steps
    )
    check("[G2] Each next_step has title + summary", steps_well_formed,
          f"steps: {[list(s.keys()) for s in next_steps]}")

    # [G3] section explanations injected into enriched cards
    has_explanations = any(
        bool(ba.get("explanation")) for ba in enriched
    )
    check("[G3] Per-section explanations populated in cards", has_explanations)

    # Streaming tokens captured
    check("[G4] Tokens were streamed to token_callback",
          len(streamed_tokens) > 0, f"token count: {len(streamed_tokens)}")

    print(f"\n  Preview — opinion_text (first 300 chars):")
    print(textwrap.indent(textwrap.fill(summary_text[:300], 72), "    "))
    print(f"\n  Preview — next_steps[0]:")
    if next_steps:
        print(f"    title: {next_steps[0].get('title')}")
        print(textwrap.indent(textwrap.fill(next_steps[0].get('summary','')[:200], 68), "    "))

except Exception as e:
    check("[E-G] Bare acts response generation", False, traceback.format_exc())


# ──────────────────────────────────────────────
# [H] _build_grounded_bare_act_fallback
# ──────────────────────────────────────────────
section("H: _build_grounded_bare_act_fallback (deterministic fallback)")

try:
    from services.response_generator_v2 import _build_grounded_bare_act_fallback
    fallback_text = _build_grounded_bare_act_fallback(MOCK_BARE_ACTS, facts_summary=facts_summary)

    garbled = _has_embedded_section_text(fallback_text)
    check("[H1] No garbled 'Section N points to N.' in bare_act fallback",
          not garbled, f"fallback[:300]: {fallback_text[:300]}")

    has_act_ref = "Protection of Women" in fallback_text
    check("[H2] Act name referenced in fallback prose", has_act_ref)

    has_section_label = re.search(r"Section\s+\d+", fallback_text) is not None
    check("[H3] Section numbers referenced as labels (not embedded text)", has_section_label)

    has_card_pointer = "cards below" in fallback_text.lower() or "set out below" in fallback_text.lower()
    check("[H4] Fallback directs user to cards", has_card_pointer,
          f"fallback: {fallback_text}")

    print(f"\n  Preview — bare_act fallback:")
    print(textwrap.indent(textwrap.fill(fallback_text[:400], 72), "    "))

except Exception as e:
    check("[H] _build_grounded_bare_act_fallback", False, traceback.format_exc())


# ──────────────────────────────────────────────
# [I] _build_grounded_interactive_fallback
# ──────────────────────────────────────────────
section("I: _build_grounded_interactive_fallback (deterministic fallback)")

MOCK_CASE_LAWS = [
    {
        "title": "Indra Sarma v. V.K.V. Sarma",
        "court": "Supreme Court of India",
        "text": "12. The expression 'domestic relationship' in PWDVA is not restricted to relationships in the nature of marriage but includes live-in relationships as well. The court must consider the nature of the relationship holistically.",
        "source_tag": "LOCAL_DB",
        "score": 0.88,
        "_rerank_score": 0.88,
    },
    {
        "title": "Hiral P. Harsora v. Kusum Narottamdas Harsora",
        "court": "Supreme Court of India",
        "text": "15. The words 'adult male person' in Section 2(q) of PWDVA are unconstitutional. Any person irrespective of sex or age can be a respondent under the Act.",
        "source_tag": "LOCAL_DB",
        "score": 0.81,
        "_rerank_score": 0.81,
    },
]

try:
    from services.response_generator_v2 import _build_grounded_interactive_fallback
    interactive_fallback = _build_grounded_interactive_fallback(
        facts_summary=facts_summary,
        formatted_bare=MOCK_BARE_ACTS,
        formatted_case=MOCK_CASE_LAWS,
    )

    garbled = _has_embedded_section_text(interactive_fallback)
    check("[I1] No garbled 'Section N points to N.' in interactive fallback",
          not garbled, f"fallback[:300]: {interactive_fallback[:300]}")

    # Raw case text must not appear mid-sentence as a fragment
    raw_fragment_in_output = "12. The expression" in interactive_fallback or "15. The words" in interactive_fallback
    check("[I2] Raw case law text not embedded in interactive fallback prose",
          not raw_fragment_in_output)

    case_name_ref = "Indra Sarma" in interactive_fallback or "Hiral" in interactive_fallback
    check("[I3] Case names referenced by label in interactive fallback", case_name_ref)

    print(f"\n  Preview — interactive fallback:")
    print(textwrap.indent(textwrap.fill(interactive_fallback[:400], 72), "    "))

except Exception as e:
    check("[I] _build_grounded_interactive_fallback", False, traceback.format_exc())


# ──────────────────────────────────────────────
# [J] Retry logic in _generate_bare_act_stage_text
# ──────────────────────────────────────────────
section("J: LLM retry + UI surfacing on timeout")

try:
    from services.response_generator_v2 import _generate_bare_act_stage_text, _BARE_ACT_LLM_MAX_ATTEMPTS

    _state = {"call_count": 0}
    streamed_retry_msgs = []

    def flaky_llm(*args, **kwargs):
        _state["call_count"] += 1
        if _state["call_count"] < _BARE_ACT_LLM_MAX_ATTEMPTS:
            raise RuntimeError(f"Ollama unreachable after 3 attempts: timed out (call {_state['call_count']})")
        # Final attempt succeeds
        return MOCK_LLM_BARE_ACT_RESPONSE

    def capture_retry(t):
        streamed_retry_msgs.append(t)

    with mock.patch("services.response_generator_v2.ask_llm", side_effect=flaky_llm):
        summary_text_j, enriched_j, next_steps_j, _ = _generate_bare_act_stage_text(
            facts_summary=facts_summary,
            dispute_results=[],
            formatted_bare=MOCK_BARE_ACTS,
            model_override=None,
            token_callback=capture_retry,
        )

    call_count = _state["call_count"]
    check(f"[J1] LLM was called {_BARE_ACT_LLM_MAX_ATTEMPTS} times (all attempts used)",
          call_count == _BARE_ACT_LLM_MAX_ATTEMPTS,
          f"actual call_count={call_count}")

    retry_text = " ".join(streamed_retry_msgs)
    has_retry_msg = "taking longer" in retry_text.lower() or "retry" in retry_text.lower() or "retrying" in retry_text.lower()
    check("[J2] Retry message streamed to UI between attempts", has_retry_msg,
          f"streamed: {retry_text[:200]}")

    # Eventually succeeded — should have real summary text
    check("[J3] Final successful attempt returns real summary text",
          len(summary_text_j) > 50, f"summary_text: {summary_text_j[:100]}")

    # Now test all-failures path
    _state_fail = {"call_count": 0}
    streamed_fail_msgs = []

    def always_fail(*args, **kwargs):
        _state_fail["call_count"] += 1
        raise RuntimeError(f"Ollama unreachable: timed out (call {_state_fail['call_count']})")

    def capture_fail(t):
        streamed_fail_msgs.append(t)

    with mock.patch("services.response_generator_v2.ask_llm", side_effect=always_fail):
        summary_fail, enriched_fail, steps_fail, _ = _generate_bare_act_stage_text(
            facts_summary=facts_summary,
            dispute_results=[],
            formatted_bare=MOCK_BARE_ACTS,
            model_override=None,
            token_callback=capture_fail,
        )

    # Normalise whitespace before matching — _stream_precomposed_text sends 3-word
    # chunks so the joined token list has double spaces between chunk boundaries.
    fail_text = re.sub(r"\s+", " ", " ".join(streamed_fail_msgs))
    has_error_notice = "did not respond" in fail_text.lower() or "multiple attempts" in fail_text.lower()
    check("[J4] Error notice streamed to UI after all attempts exhausted", has_error_notice,
          f"streamed: {fail_text[:200]}")

    garbled_fallback = _has_embedded_section_text(summary_fail)
    check("[J5] Deterministic fallback returned (no garbled text) when all attempts fail",
          not garbled_fallback, f"summary_fail[:200]: {summary_fail[:200]}")

    check("[J6] next_steps still populated even when LLM fails completely",
          len(steps_fail) > 0, f"steps_fail: {steps_fail}")

except Exception as e:
    check("[J] Retry logic", False, traceback.format_exc())


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
section("RESULTS SUMMARY")

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"\n  Total: {total}   {PASS} Passed: {passed}   {FAIL} Failed: {failed}\n")

if failed:
    print(f"  Failed checks:")
    for label, ok, detail in results:
        if not ok:
            print(f"    {FAIL}  {label}")
            if detail:
                print(textwrap.indent(detail[:300], "       "))
    sys.exit(1)
else:
    print(f"  All checks passed.\n")
    sys.exit(0)
