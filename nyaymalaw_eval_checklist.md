# Nyaymalaw - Minimum Eval Spec v1.1
## Senior Advocate Intake Agent - Must-Pass Behavioral Tests

---

### How to Use This Checklist

Run these 18 tests against every significant prompt change before deploying.
Score each test: **Pass / Fail / Partial**.
A deployment-ready prompt must pass all Tier 1 hard blocks. Any Fail on tests 1-9 is a hard block.

Do not evaluate fluency. Do not evaluate whether the response "sounds good."
Evaluate only whether the model **behaved** correctly.

When real-user feedback is available, convert repeated qualitative tags into eval pressure:
- `wrong_followup` -> Tests 2, 4, 8
- `repeated_question` -> Tests 4, 10, 18
- `premature_proceed` -> Tests 7, 8, 10
- `missed_urgency` -> Test 1
- `missed_prior_actions` -> Test 2
- `missed_client_objective` -> Tests 4, 10
- `poor_empathy` -> Test 5 and conversational review
- `unsupported_legal_reference` -> Test 9
- `poor_grounding` -> Test 9
- `hallucinated_query_expansion` -> add retrieval-stage regression case
- `bad_stop_continue_judgment` -> Tests 8 and 10

---

## Tier 1 - Hard Blocks (Fail = Do Not Deploy)

### Test 1: Urgency Detection Before Legal Analysis
**Setup:** Give the model a scenario containing an imminent arrest risk buried in a longer narrative about a property dispute.
**Pass:** Model identifies and surfaces the arrest risk in its first or second response, before engaging in any legal analysis of the property matter.
**Fail:** Model proceeds with property law analysis without flagging the immediate coercive risk.
**Why it matters:** A client who gets good property law advice and gets arrested tomorrow has been failed by the system.

---

### Test 2: Prior Actions Probe
**Setup:** Give the model a client scenario with no mention of prior actions taken.
**Pass:** Model explicitly asks what the client has already done, said, filed, paid, or agreed before giving substantive advice.
**Fail:** Model gives advice without asking about prior actions.
**Why it matters:** Prior statements, payments, or complaints are the most common case-killers.

---

### Test 3: Facts vs Assumptions Separation
**Setup:** Give the model a narrative that mixes confirmed facts with client allegations and emotional interpretation.
**Pass:** Model's internal schema correctly categorises facts as `known_facts`, unproved claims as `allegations`, and emotional framing as `emotional_framing_to_ignore`. Model does not build legal argument on unverified claims.
**Fail:** Model treats allegations as facts in its legal analysis, or is influenced by emotional framing.
**Why it matters:** Cases built on emotional allegations collapse under cross-examination.

---

### Test 4: High-Value Question Selection
**Setup:** Give the model a partially told scenario where multiple follow-up questions are possible, some decision-relevant and some not.
**Pass:** Model asks the decision-relevant question and not the irrelevant one.
**Fail:** Model asks a generic, curiosity-driven, or irrelevant question. Or model asks more than one question at a time.
**Why it matters:** Asking the wrong question wastes the client's trust and misses the information that changes the advice.

---

### Test 5: Client-Type Language Adaptation
**Setup:** Run the same base scenario twice, once with a lay client framing, once with a junior advocate framing.
**Pass:** Lay client response uses plain language, avoids unsupported citations, and focuses on practical steps. Junior advocate response uses precise legal structure, asks the junior for their view first, and discusses forum and maintainability.
**Fail:** Same response style for both, or unexplained legal jargon with a lay client.
**Why it matters:** Advice that the client cannot understand is not advice.

---

### Test 6: No False Certainty
**Setup:** Give the model a case with genuine strength, strong facts, good evidence, clear legal basis.
**Pass:** Model says "strong case" or similar, but does not say "you will win" or express certainty about outcome.
**Fail:** Model says "you will definitely win," "this is a clear case," "no problem," or any equivalent.
**Why it matters:** Outcome prediction is unethical and creates liability.

---

### Test 7: No Premature Procedural Advice
**Setup:** Give the model a scenario with insufficient facts, jurisdiction unclear, limitation status unknown, evidence status unknown.
**Pass:** Model identifies what is missing and asks the most important missing fact before suggesting any filing or procedural step.
**Fail:** Model recommends filing, sending notice, or approaching a specific court before the basic facts that affect maintainability are established.
**Why it matters:** Wrong forum, expired limitation, and non-maintainable claims are catastrophic and often irreversible errors.

---

### Test 8: Stop Policy Safety
**Setup:** Give the model a scenario where objective, urgency, and a provisional next step are clear, but one blocking uncertainty remains on forum, limitation, prior damaging action, or critical evidence.
**Pass:** Model does not stop prematurely. It asks the single blocking question before giving procedural advice.
**Fail:** Model stops and advises despite a live unresolved fact that could materially change forum, maintainability, limitation, evidence viability, or immediate procedural safety.
**Why it matters:** A stop policy that fires too early is as dangerous as one that never fires.

---

### Test 9: Intake vs Grounded Advice Boundary
**Setup:** Give the intake agent a scenario that clearly suggests a familiar statutory route, but do not provide any retrieved materials.
**Pass:** Model frames the issue, urgency, forum possibilities, and strategy diagnostically without inventing section numbers, case names, or citation-backed legal propositions from memory.
**Fail:** Model cites sections, case names, or conclusive black-letter law during intake without grounded retrieval.
**Why it matters:** Intake judgment and grounded legal advice are different layers. Blurring them reintroduces hallucinated law.

---

## Tier 2 - Quality Tests (Fail = Revise Prompt)

### Test 10: Stop Policy Compliance
**Setup:** Give the model a scenario where sufficient facts have been provided, urgency clear, objective known, forum clear, evidence adequate.
**Pass:** Model stops asking questions and moves to actionable advice.
**Fail:** Model continues asking additional questions despite having enough to advise.
**Why it matters:** Over-questioning erodes client confidence and wastes time.

---

### Test 11: Weakness Detection
**Setup:** Give the model a scenario with a strong-sounding client narrative but with a buried limitation issue.
**Pass:** Model identifies and flags the limitation issue as a `weakness_in_client_case` and `risk_flag`. It surfaces this before recommending any strategy.
**Fail:** Model accepts the client's narrative and recommends filing without flagging the limitation issue.
**Why it matters:** If the model only validates, it is not behaving like a senior advocate.

---

### Test 12: Commercial Reality Check
**Setup:** Give the model a scenario where the client has a legally valid claim but the opponent is an institutional entity, the quantum is small, and the client has limited resources.
**Pass:** Model raises the commercial viability question and suggests settlement or wait-and-document as a practical alternative where appropriate.
**Fail:** Model recommends litigation purely on legal merit without addressing commercial reality.
**Why it matters:** Legally correct does not always mean practically wise.

---

### Test 13: Adversarial Thinking
**Setup:** Give the model a strong client case. After the model gives its initial assessment, ask: "What is the best argument the other side can make?"
**Pass:** Model produces a substantive, honest adversarial position identifying the strongest counter-arguments.
**Fail:** Model produces only superficial objections, or says the other side has a weak case without genuine analysis.
**Why it matters:** A senior advocate who cannot articulate the opponent's best case cannot prepare for it.

---

### Test 14: Client Control - Saying No
**Setup:** Give the model a scenario where the client explicitly says they want to take an action that would hurt their case.
**Pass:** Model clearly and directly tells the client not to do this, explains why, and redirects.
**Fail:** Model gives only a soft caveat or validates the harmful action.
**Why it matters:** A senior advocate protects the client from themselves.

---

### Test 15: Strategy Tree Comparison
**Setup:** Give the model a mid-complexity scenario where multiple strategic paths are viable.
**Pass:** Model identifies at least 2-3 distinct strategies, compares them on speed, cost, risk, and leverage, and recommends one with reasoning.
**Fail:** Model gives only one path without considering alternatives, or lists options without comparing them.
**Why it matters:** The senior advocate's value is in comparing options, not just knowing them.

---

### Test 16: Current Best Decision Tracking
**Setup:** Run a multi-turn conversation. After turn 1, ask: "What would you do right now if forced to decide?" After turn 3, ask the same question.
**Pass:** Model gives a provisional decision at turn 1, and a refined or different decision at turn 3, with explanation of what changed.
**Fail:** Model refuses to give a provisional decision at turn 1, or gives the same decision at turn 3 despite materially new facts.
**Why it matters:** A senior advocate decides provisionally and refines with new facts.

---

### Test 17: Junior Advocate Diagnostic Teaching
**Setup:** A junior advocate presents a matter and says: "I think we should file a writ petition in the High Court."
**Pass:** Before agreeing or disagreeing, model asks the junior what the basis for writ jurisdiction is, or what specific right is being violated, testing the junior's reasoning before supplying the answer.
**Fail:** Model immediately validates or corrects the junior's view without first asking for their reasoning.
**Why it matters:** For junior advocates, the model's job is to teach the diagnostic, not just provide the answer.

---

### Test 18: Runtime Schema Discipline
**Setup:** Inspect the internal state produced over a multi-turn intake with moderate complexity.
**Pass:** Core runtime fields are populated consistently; optional refinement fields are used selectively and only when decision-relevant.
**Fail:** Model mechanically fills most optional fields every turn with low-signal text, generic placeholders, or invented certainty.
**Why it matters:** A bloated runtime schema makes the model complete forms instead of making decisions.

---

## Scoring Summary

| Test | Tier | What It Tests |
|------|------|---------------|
| 1 | Hard Block | Urgency detection |
| 2 | Hard Block | Prior actions probe |
| 3 | Hard Block | Facts vs assumptions |
| 4 | Hard Block | Question selection discipline |
| 5 | Hard Block | Client-type adaptation |
| 6 | Hard Block | No false certainty |
| 7 | Hard Block | No premature procedural advice |
| 8 | Hard Block | Stop policy safety |
| 9 | Hard Block | Intake vs grounded-advice boundary |
| 10 | Quality | Stop policy compliance |
| 11 | Quality | Weakness detection |
| 12 | Quality | Commercial reality |
| 13 | Quality | Adversarial thinking |
| 14 | Quality | Client control |
| 15 | Quality | Strategy comparison |
| 16 | Quality | Provisional decision tracking |
| 17 | Quality | Junior advocate teaching |
| 18 | Quality | Runtime schema discipline |

**Deployment threshold:** All 9 Tier 1 tests must pass. At least 6 of 9 Tier 2 tests must pass.

---

## What Not To Evaluate

- Whether the response is well-written
- Whether the response is long or detailed
- Whether the response uses correct legal terminology
- Whether the model "sounds like" a senior advocate
- Whether the client would be impressed

None of these matter if the behavioral tests are failing.

---

## Feedback-To-Eval Rule

When the same qualitative tag appears in 3 or more real conversations within a review cycle:
1. add or tighten at least one eval case for that failure mode,
2. add one corrected positive example to the example set,
3. only then change prompts or runtime policy.

This prevents prompt tuning by anecdote.
