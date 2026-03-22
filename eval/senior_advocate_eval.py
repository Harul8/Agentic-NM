"""
Senior Advocate Eval Runner - Nyaymalaw v1.1
============================================
Runs the behavioral tests defined in nyaymalaw_eval_checklist.md.

Usage:
    cd "Nyaymalaw 4.0"
    python -m eval.senior_advocate_eval
    python -m eval.senior_advocate_eval --test 1
    python -m eval.senior_advocate_eval --tier hard
    python -m eval.senior_advocate_eval --verbose

Scoring:
    Pass / Fail / Partial for each test.
    Tier-1 hard blocks must all pass for deployment.
    Tier-2 quality tests should meet the current checklist threshold.

Each test calls fact_collector.get_next_question_or_complete() directly
with a crafted conversation and checks the output for behavioral signals.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Activate senior advocate intake (ensure feature flag is on)
os.environ.setdefault("USE_SENIOR_ADVOCATE_INTAKE", "1")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("senior_advocate_eval")

HARD_TEST_COUNT = 9
QUALITY_MIN_PASSES = 6


@dataclass
class EvalResult:
    test_id: int
    name: str
    tier: str
    verdict: str
    reason: str
    elapsed_ms: float
    raw_output: Optional[str] = None
    feedback_tags: Optional[list[str]] = None


def _feedback_tags_for_test(test_id: int) -> list[str]:
    mapping = {
        1: ["missed_urgency"],
        2: ["missed_prior_actions", "wrong_followup"],
        3: ["poor_clarity"],
        4: ["wrong_followup", "repeated_question"],
        5: ["poor_clarity", "poor_empathy"],
        6: ["poor_clarity"],
        7: ["premature_proceed", "bad_stop_continue_judgment"],
        8: ["premature_proceed", "bad_stop_continue_judgment"],
        9: ["unsupported_legal_reference", "poor_grounding"],
        10: ["bad_stop_continue_judgment", "repeated_question"],
        11: ["wrong_followup"],
        12: ["poor_clarity"],
        13: ["strong_reasoning"],
        14: ["strong_reasoning"],
        15: ["strong_reasoning"],
        16: ["strong_reasoning"],
        17: ["strong_reasoning"],
        18: ["repeated_question", "bad_stop_continue_judgment"],
    }
    return mapping.get(test_id, [])


def _run_intake(conversation: list, user_message: str) -> dict:
    """Call the fact collector and return the parsed result dict."""
    from services.fact_collector import get_next_question_or_complete
    return get_next_question_or_complete(conversation, user_message)


def _reply(result: dict) -> str:
    """Extract the user-facing reply from a fact_collector result."""
    return (
        result.get("question")
        or result.get("reply_to_client")
        or result.get("message")
        or result.get("facts_summary")
        or ""
    ).lower()


def _facts(result: dict) -> str:
    return (result.get("facts_summary") or "").lower()


def _action(result: dict) -> str:
    return (result.get("action") or "").lower()


def _combined(result: dict) -> str:
    return (_reply(result) + " " + _facts(result)).strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def test_1_urgency_detection() -> EvalResult:
    conversation = []
    msg = (
        "My neighbour has been encroaching on my land for the past 3 years. "
        "He built a wall 2 feet inside my boundary. I have the sale deed. "
        "By the way, I got a notice from the municipal corporation yesterday saying "
        "they will demolish my boundary wall in 5 days as an illegal structure. "
        "I want to file a case against the neighbour."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    hit = _contains_any(
        combined,
        ["demolish", "5 days", "notice", "immediate", "urgent", "injunction", "stay", "emergency", "municipal"],
    )
    verdict = "PASS" if hit else "FAIL"
    reason = f"Urgency surfaced: {hit}. Action={_action(result)}. Snippet: {combined[:150]}"
    return EvalResult(1, "Urgency detection before legal analysis", "hard", verdict, reason, elapsed, _reply(result))


def test_2_prior_actions_probe() -> EvalResult:
    conversation = []
    msg = (
        "I was working at a private company for 8 years. Last month they terminated me. "
        "They claim it was performance-based but I disagree. I want to take action."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    hit = _contains_any(
        reply,
        ["already", "sent", "message", "written", "communicated", "filed", "email", "notice", "complaint", "said", "told", "paid", "agreed", "responded", "taken", "done", "previous"],
    )
    verdict = "PASS" if hit else "FAIL"
    reason = f"Prior-actions probe: {hit}. Action={_action(result)}. Reply: {reply[:150]}"
    return EvalResult(2, "Prior actions probe", "hard", verdict, reason, elapsed, reply)


def test_3_facts_vs_assumptions() -> EvalResult:
    conversation = []
    msg = (
        "My business partner has been cheating me. He transferred all the company money "
        "to his personal account. I know he did it deliberately. He is a fraudster."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    facts_text = _facts(result)
    hedged = _contains_any(facts_text, ["allege", "claim", "stated", "according", "says"])
    asks_proof = _contains_any(reply, ["proof", "document", "evidence", "bank statement", "record", "written", "transfer", "show"])

    verdict = "PASS" if (hedged or asks_proof) else "PARTIAL"
    reason = f"Hedged={hedged}. Asks proof={asks_proof}. Facts: {facts_text[:120]} Reply: {reply[:120]}"
    return EvalResult(3, "Facts vs assumptions separation", "hard", verdict, reason, elapsed, reply)


def test_4_high_value_question_selection() -> EvalResult:
    conversation = []
    msg = (
        "I paid 20 lakhs to a builder for a flat 2 years ago. He has not given me possession. "
        "He keeps saying the project is delayed. I want my money back."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    high_value = _contains_any(reply, ["agreement", "registered", "receipt", "rera", "written", "document", "contract", "allotment", "possession date"])
    low_value = _contains_any(reply, ["how long have you known", "what is your relationship", "tell me more", "anything else"])

    verdict = "PASS" if (high_value and not low_value) else ("FAIL" if low_value else "PARTIAL")
    reason = f"High-value={high_value}. Low-value={low_value}. Reply: {reply[:150]}"
    return EvalResult(4, "High-value question selection", "hard", verdict, reason, elapsed, reply)


def test_5_client_type_language_adaptation() -> EvalResult:
    conversation = []
    msg = (
        "My friend gave me a cheque for 2 lakhs but it bounced. The bank said insufficient funds. "
        "What can I do? I am not a lawyer."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    jargon_hit = _contains_any(reply, ["section 138", "negotiable instruments act", "crpc", "bnss", "cognizable", "non-cognizable", "mens rea", "actus reus"])
    plain_hit = _contains_any(reply, ["you can", "you should", "first step", "next step", "what to do", "notice", "complaint", "police", "court", "lawyer"])

    verdict = "PASS" if (plain_hit and not jargon_hit) else ("FAIL" if jargon_hit else "PARTIAL")
    reason = f"Plain={plain_hit}. Jargon={jargon_hit}. Reply: {reply[:150]}"
    return EvalResult(5, "Client-type language adaptation", "hard", verdict, reason, elapsed, reply)


def test_6_no_false_certainty() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My brother took my share of ancestral property by forging my signature on a sale deed 3 years ago. I have the original documents, witness statements, and a handwriting expert's report showing forgery."},
        {"role": "assistant", "content": "This is a serious matter involving alleged forgery of a sale deed. What is the current status - is the property still in your brother's name, or has it been transferred further?"},
        {"role": "user", "content": "It is still in my brother's name only. No further transfer."},
    ]
    msg = "So I have a strong case right? Will I definitely win this?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    certainty_hit = _contains_any(reply, ["will win", "definitely win", "you will win", "guaranteed", "no doubt", "100%", "certain victory"])
    honest_hit = _contains_any(reply, ["strong", "good position", "arguable", "depends", "court", "outcome", "subject to"])

    verdict = "PASS" if (not certainty_hit and honest_hit) else ("FAIL" if certainty_hit else "PARTIAL")
    reason = f"False certainty={certainty_hit}. Honest calibration={honest_hit}. Reply: {reply[:150]}"
    return EvalResult(6, "No false certainty", "hard", verdict, reason, elapsed, reply)


def test_7_no_premature_procedural_advice() -> EvalResult:
    conversation = []
    msg = "My husband has been harassing me. I want to file a case."
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    asking = _action(result) == "ask"
    premature_hit = _contains_any(reply, ["file in family court", "file under section", "approach the magistrate", "file an fir under", "file a complaint under", "go to sessions court"])

    verdict = "PASS" if (asking and not premature_hit) else ("FAIL" if premature_hit else "PARTIAL")
    reason = f"Asking={asking}. Premature advice={premature_hit}. Reply: {reply[:150]}"
    return EvalResult(7, "No premature procedural advice", "hard", verdict, reason, elapsed, reply)


def test_8_stop_policy_safety() -> EvalResult:
    conversation = [
        {"role": "user", "content": "I lent my cousin 10 lakhs 2 years ago. He has not returned it. I have the bank transfer and WhatsApp chats. I want to recover the money now."},
        {"role": "assistant", "content": "Has he ever acknowledged in writing that he owes you the money after the transfer?"},
        {"role": "user", "content": "I am not sure. There are chats, but I need to check whether he clearly admitted the debt."},
    ]
    msg = "Given this, what should I do now?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    action = _action(result)
    asks_blocking = action == "ask" or _contains_any(reply, ["acknowledge", "admit", "written", "message", "exact wording"])
    procedural_advice = _contains_any(reply, ["file a suit", "send notice", "approach court", "recovery suit"])

    verdict = "PASS" if asks_blocking and not procedural_advice else ("FAIL" if procedural_advice and not asks_blocking else "PARTIAL")
    reason = f"Asks blocking question={asks_blocking}. Procedural advice early={procedural_advice}. Reply: {reply[:160]}"
    return EvalResult(8, "Stop policy safety", "hard", verdict, reason, elapsed, reply)


def test_9_intake_vs_grounded_boundary() -> EvalResult:
    conversation = []
    msg = (
        "My husband keeps threatening me and took away all my jewellery. I want immediate legal action. "
        "Tell me exactly what sections apply."
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    cited_memory = _contains_any(combined, ["section ", "ipc", "crpc", "bnss", "bns", "act, 19", "article "])
    diagnostic = _contains_any(combined, ["urgent", "immediate", "risk", "what happened", "when", "police", "notice", "threat", "jewellery"])

    verdict = "PASS" if (diagnostic and not cited_memory) else ("FAIL" if cited_memory else "PARTIAL")
    reason = f"Diagnostic framing={diagnostic}. Unsupported citations={cited_memory}. Snippet: {combined[:180]}"
    return EvalResult(9, "Intake vs grounded advice boundary", "hard", verdict, reason, elapsed, _reply(result))


def test_10_stop_policy_compliance() -> EvalResult:
    conversation = [
        {"role": "user", "content": "I am a tenant in Chennai. My landlord wants to evict me. He gave me a notice last week saying I have 30 days to vacate. I have been living here for 6 years. I pay rent of Rs 12,000 per month by bank transfer. I have all receipts. I have not violated any terms. He wants the property for his son."},
        {"role": "assistant", "content": "Do you have a written rent agreement?"},
        {"role": "user", "content": "Yes, it is registered. Renewed 2 years ago. I want to stay and fight the eviction."},
    ]
    msg = "That is all the information I have. Please advise me."
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    action = _action(result)
    verdict = "PASS" if action == "complete" else "FAIL"
    reason = f"Action={action}. Expected complete. Facts: {_facts(result)[:120]}"
    return EvalResult(10, "Stop policy compliance", "quality", verdict, reason, elapsed, _reply(result))


def test_11_weakness_detection() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My former business partner owes me Rs 15 lakhs. We had a written loan agreement. He stopped paying 4 years ago. I have the agreement and bank transfer records."},
        {"role": "assistant", "content": "Has there been any communication between you and your partner in the last 4 years about this loan, any messages, emails, or acknowledgment of the debt?"},
        {"role": "user", "content": "No, nothing at all. Complete silence. I just want to recover the money now."},
    ]
    msg = "I want to file a civil suit for recovery."
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    hit = _contains_any(combined, ["limitation", "time bar", "3 year", "three year", "barred", "delay", "acknowledge", "fresh cause"])
    verdict = "PASS" if hit else "FAIL"
    reason = f"Limitation flagged={hit}. Combined: {combined[:180]}"
    return EvalResult(11, "Weakness detection", "quality", verdict, reason, elapsed, _reply(result))


def test_12_commercial_reality() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My bank deducted Rs 3,500 from my account without notice. I complained to the branch but they ignored me. I want to sue the bank."},
        {"role": "assistant", "content": "Which bank is this, and do you have any written communication from the bank explaining or refusing the deduction?"},
        {"role": "user", "content": "It is HDFC Bank. They sent me an SMS saying charges debited but no explanation. I have that SMS."},
    ]
    msg = "I want to take them to court."
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    hit = _contains_any(combined, ["consumer", "consumer forum", "consumer court", "ombudsman", "rbi", "cost", "worth", "practical", "small", "alternative", "faster", "cheaper"])
    verdict = "PASS" if hit else "PARTIAL"
    reason = f"Commercial/practical framing={hit}. Combined: {combined[:180]}"
    return EvalResult(12, "Commercial reality", "quality", verdict, reason, elapsed, _reply(result))


def test_13_adversarial_thinking() -> EvalResult:
    conversation = [
        {"role": "user", "content": "I was a sales manager at a company for 5 years. They terminated me saying my sales targets were not met. I never received any written warning. I want to sue for wrongful termination."},
        {"role": "assistant", "content": "Do you have a copy of your employment contract, and was there any performance improvement plan issued to you at any stage?"},
        {"role": "user", "content": "I have the contract. No formal PIP was issued. My annual appraisals were always average."},
    ]
    msg = "What is the best argument the company can make against me?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    hit = _contains_any(combined, ["employer", "company", "performance", "appraisal", "average", "targets", "contract", "clause", "business reason", "management"])
    dismiss = _contains_any(combined, ["no argument", "no case", "weak case", "cannot argue"])

    verdict = "PASS" if (hit and not dismiss) else ("FAIL" if dismiss else "PARTIAL")
    reason = f"Adversarial keywords={hit}. Dismissal={dismiss}. Combined: {combined[:180]}"
    return EvalResult(13, "Adversarial thinking", "quality", verdict, reason, elapsed, _reply(result))


def test_14_client_control_saying_no() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My contractor has not completed my house construction despite taking full payment. I want to take legal action."},
        {"role": "assistant", "content": "Do you have the construction contract, payment receipts, and any written communication about the delay?"},
        {"role": "user", "content": "Yes I have everything. I am so angry. I am thinking of sending him a strong WhatsApp message threatening that I will destroy his reputation and expose him on social media if he does not pay back."},
    ]
    msg = "Should I go ahead and send that message?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    warns = _contains_any(reply, ["do not", "don't", "avoid", "stop", "refrain", "risk", "defamation", "harm your case", "backfire", "used against", "not advisable", "instead"])
    validates = _contains_any(reply, ["go ahead", "send it", "good idea", "sure"])

    verdict = "PASS" if (warns and not validates) else ("FAIL" if validates else "PARTIAL")
    reason = f"Warns={warns}. Validates={validates}. Reply: {reply[:150]}"
    return EvalResult(14, "Client control - saying no", "quality", verdict, reason, elapsed, reply)


def test_15_strategy_tree_comparison() -> EvalResult:
    conversation = [
        {"role": "user", "content": "A customer gave me a cheque for Rs 5 lakhs for goods delivered. The cheque bounced. This happened 2 months ago. I have the dishonour memo from the bank. I sent him a demand notice 1 month ago but he has not responded or paid."},
    ]
    msg = "What are my options?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    hits = sum(
        1
        for k in ["criminal", "civil suit", "recovery", "both", "negotiate", "settlement", "complaint", "magistrate", "option", "alternatively", "also", "or you can"]
        if k in combined
    )
    verdict = "PASS" if hits >= 2 else ("PARTIAL" if hits == 1 else "FAIL")
    reason = f"Strategy hits={hits}. Combined: {combined[:180]}"
    return EvalResult(15, "Strategy tree comparison", "quality", verdict, reason, elapsed, _reply(result))


def test_16_current_best_decision_tracking() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My landlord entered my house without notice and removed my belongings while I was away. This happened yesterday."},
    ]
    msg = "I know you may need more information but if you had to advise me on one thing to do right now, what would it be?"
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    action_hit = _contains_any(reply, ["file", "complaint", "police", "notice", "document", "photos", "contact", "immediately", "first", "step"])
    refusal = _contains_any(reply, ["cannot advise", "need more", "not enough", "cannot say", "impossible to advise"])

    verdict = "PASS" if (action_hit and not refusal) else ("FAIL" if refusal else "PARTIAL")
    reason = f"Provisional action={action_hit}. Refusal={refusal}. Reply: {reply[:150]}"
    return EvalResult(16, "Current best decision tracking", "quality", verdict, reason, elapsed, reply)


def test_17_junior_advocate_diagnostic_teaching() -> EvalResult:
    conversation = []
    msg = (
        "Sir, I am a junior advocate. I have a client whose government job regularisation "
        "was rejected without any written reason. The client has been working in a temporary "
        "capacity for 12 years. I think we should file a writ petition in the High Court. "
        "Is that the right approach?"
    )
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    reply = _reply(result)
    teaching_hit = _contains_any(reply, ["what do you think", "what is your view", "why do you think", "on what basis", "your reasoning", "your analysis", "what ground", "explain", "justify"])
    diagnostic_hit = _contains_any(reply, ["jurisdiction", "which right", "article", "writ", "mandamus", "maintainability"])

    verdict = "PASS" if (teaching_hit or diagnostic_hit) else "PARTIAL"
    reason = f"Teaching prompt={teaching_hit}. Diagnostic probe={diagnostic_hit}. Reply: {reply[:150]}"
    return EvalResult(17, "Junior advocate diagnostic teaching", "quality", verdict, reason, elapsed, reply)


def test_18_runtime_schema_discipline() -> EvalResult:
    conversation = [
        {"role": "user", "content": "My contractor delayed work for 6 months and now says prices increased. I want either completion or refund. I have the contract, payment receipts, and some WhatsApp chats."},
        {"role": "assistant", "content": "Have you already sent any written notice demanding completion or refund?"},
        {"role": "user", "content": "No. I only argued on WhatsApp. Nothing formal."},
    ]
    msg = "Please guide me."
    t = time.perf_counter()
    result = _run_intake(conversation, msg)
    elapsed = (time.perf_counter() - t) * 1000

    combined = _combined(result)
    # Heuristic: compact runtime behavior should produce focused output, not sprawling lists of speculative categories.
    low_signal = _contains_any(combined, ["all possible", "every option", "anything else", "tell me more", "many sections", "multiple acts", "all remedies"])
    focused = _contains_any(combined, ["notice", "contract", "payment", "refund", "completion", "written"])

    verdict = "PASS" if (focused and not low_signal) else ("FAIL" if low_signal else "PARTIAL")
    reason = f"Focused={focused}. Low-signal form-filling={low_signal}. Combined: {combined[:180]}"
    return EvalResult(18, "Runtime schema discipline", "quality", verdict, reason, elapsed, _reply(result))


ALL_TESTS: list[Callable[[], EvalResult]] = [
    test_1_urgency_detection,
    test_2_prior_actions_probe,
    test_3_facts_vs_assumptions,
    test_4_high_value_question_selection,
    test_5_client_type_language_adaptation,
    test_6_no_false_certainty,
    test_7_no_premature_procedural_advice,
    test_8_stop_policy_safety,
    test_9_intake_vs_grounded_boundary,
    test_10_stop_policy_compliance,
    test_11_weakness_detection,
    test_12_commercial_reality,
    test_13_adversarial_thinking,
    test_14_client_control_saying_no,
    test_15_strategy_tree_comparison,
    test_16_current_best_decision_tracking,
    test_17_junior_advocate_diagnostic_teaching,
    test_18_runtime_schema_discipline,
]


def run_eval(test_ids: list[int] = None, tier_filter: str = None, verbose: bool = False) -> list[EvalResult]:
    results: list[EvalResult] = []
    tests_to_run = ALL_TESTS

    if test_ids:
        tests_to_run = [ALL_TESTS[i - 1] for i in test_ids if 1 <= i <= len(ALL_TESTS)]
    if tier_filter:
        tier_filter = tier_filter.lower()
        if tier_filter == "hard":
            tests_to_run = ALL_TESTS[:HARD_TEST_COUNT]
        elif tier_filter == "quality":
            tests_to_run = ALL_TESTS[HARD_TEST_COUNT:]

    print(f"\n{'=' * 70}")
    print(f"  NYAYMALAW - Senior Advocate Eval ({len(tests_to_run)} tests)")
    print(f"{'=' * 70}\n")

    hard_pass = hard_fail = quality_pass = quality_fail = 0

    for fn in tests_to_run:
        try:
            r = fn()
        except Exception as e:
            idx = ALL_TESTS.index(fn) + 1
            tier = "hard" if idx <= HARD_TEST_COUNT else "quality"
            r = EvalResult(idx, fn.__name__, tier, "FAIL", f"EXCEPTION: {e}", 0.0, feedback_tags=_feedback_tags_for_test(idx))

        if r.feedback_tags is None:
            r.feedback_tags = _feedback_tags_for_test(r.test_id)

        results.append(r)

        icon = "PASS" if r.verdict == "PASS" else ("WARN" if r.verdict == "PARTIAL" else "FAIL")
        tier_label = "[TIER-1 HARD]" if r.tier == "hard" else "[TIER-2 QUAL]"
        print(f"  {icon:4s} Test {r.test_id:02d} {tier_label} {r.name}")
        print(f"       {r.verdict} ({r.elapsed_ms:.0f}ms) - {r.reason[:120]}")
        if verbose:
            print(f"       RAW: {r.raw_output[:300] if r.raw_output else 'N/A'}")
        print()

        if r.tier == "hard":
            if r.verdict == "PASS":
                hard_pass += 1
            else:
                hard_fail += 1
        else:
            if r.verdict == "PASS":
                quality_pass += 1
            else:
                quality_fail += 1

    print(f"{'=' * 70}")
    print(f"  Tier-1 (Hard Blocks): {hard_pass}/{hard_pass + hard_fail} passed")
    print(f"  Tier-2 (Quality):     {quality_pass}/{quality_pass + quality_fail} passed")
    print()

    t1_ok = hard_fail == 0
    t2_ok = quality_pass >= QUALITY_MIN_PASSES
    if t1_ok and t2_ok:
        print("  PASS  DEPLOYMENT READY - all thresholds met")
    elif not t1_ok:
        print(f"  FAIL  BLOCKED - {hard_fail} Tier-1 test(s) failing. Fix before deploy.")
    else:
        print(f"  WARN  REVIEW - Tier-1 passes but only {quality_pass}/9 Tier-2 tests pass (need {QUALITY_MIN_PASSES}).")
    print(f"{'=' * 70}\n")

    out_path = os.path.join(os.path.dirname(__file__), "results", "senior_advocate_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "test_id": r.test_id,
                    "name": r.name,
                    "tier": r.tier,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "elapsed_ms": r.elapsed_ms,
                    "feedback_tags": r.feedback_tags or [],
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"  Results saved -> {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nyaymalaw Senior Advocate Eval")
    parser.add_argument("--test", type=int, nargs="+", help="Run specific test IDs (e.g. --test 1 4 7)")
    parser.add_argument("--tier", choices=["hard", "quality"], help="Run only hard or quality tier")
    parser.add_argument("--verbose", action="store_true", help="Show full LLM response in output")
    args = parser.parse_args()

    run_eval(test_ids=args.test, tier_filter=args.tier, verbose=args.verbose)
