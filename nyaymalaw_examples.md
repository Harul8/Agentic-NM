# Nyaymalaw - Seed Examples v1.1
## Contrastive Training Examples for Senior Advocate Intake Agent

---

### How to Use These Examples

Each example shows:
1. The client's opening statement
2. The ideal follow-up question and why it was chosen
3. A plausible but wrong follow-up and why it loses
4. The internal JSON decision state at that moment
5. The final advice once enough facts are known

These are training anchors, not scripts. The model should learn the reasoning pattern, not memorise the dialogue.

Important boundary:
- Intake examples teach diagnostic behavior and question selection.
- Final legal advice examples should be treated as belonging to the grounded legal-advice layer.
- Do not learn from these examples that intake may freely cite unsupported law from memory.

---

## Example 1 - Lay Client: Property Dispute with Buried Urgency

### Client Opening
*"Sir, my neighbour has been encroaching on my land for years. He has put up a wall on my side. Last week I got a notice from the municipal corporation saying they will demolish my boundary wall next week because it is an illegal structure. I have my sale deed and all papers. I want to file a case."*

---

### Ideal Follow-Up

**Question:** "When exactly is the municipal corporation coming, what date is mentioned in the notice, and have you received any prior notice from them before this one?"

**Why this question was chosen:**
The demolition deadline is the live urgency. Everything else, the encroachment case, the neighbour's wall, the underlying title dispute, can wait. If the corporation demolishes next week, an injunction application must be filed within days. The date determines whether emergency relief is even possible. The prior-notice question tests whether there is a procedural defect in the corporation's action that strengthens an injunction application.

This question changes: urgency level, strategy (emergency writ vs. regular suit), and next procedural step.

---

### Plausible but Wrong Follow-Up

**Wrong question:** "How long has the neighbour been encroaching, and do you have any photos of the original boundary?"

**Why it loses:**
This question addresses the underlying encroachment dispute, which is real but not urgent. No decision about the encroachment case needs to be made today. The client faces demolition next week. Asking about the encroachment timeline first signals that the model has not detected the immediate risk.

This question changes nothing about what must be done in the next 48 hours.

---

### Internal JSON State (After Turn 1)

```json
{
  "client": { "client_type": "lay_client" },
  "objective": {
    "stated_objective": "file a case about encroachment",
    "real_objective": "prevent demolition of boundary wall next week",
    "objective_type": "interim_relief"
  },
  "urgency": {
    "urgency_level": "immediate",
    "urgency_flags": ["demolition_eviction_dispossession_risk"],
    "hours_to_act": "unknown - need exact date from notice"
  },
  "facts": {
    "known_facts": [
      "Client has sale deed and property documents",
      "Neighbour has encroached and built a wall on client's land",
      "Municipal corporation issued notice for demolition next week"
    ],
    "assumptions_or_uncertainties": [
      "Exact demolition date unknown",
      "Whether prior show-cause notice was issued",
      "Whether the municipal action is on client's structure or neighbour's encroachment"
    ]
  },
  "decision_state": {
    "enough_to_advise": false,
    "current_best_decision": "Emergency injunction application against demolition, but need date and notice details to confirm feasibility",
    "next_best_question": "What date is mentioned in the notice, and was any prior notice issued?",
    "question_purpose": "urgency + next_procedural_step",
    "question_decision_value": "high"
  }
}
```

---

### Final Advice (After Sufficient Facts Are Known; grounded legal-advice layer)

*Assume: Demolition date is in 5 days. No prior show-cause notice was given. Client's wall is on the client's own land. Municipal action targets client's structure, not the encroachment.*

**What to do today:**
File an emergency injunction application in the civil court or a writ petition in the High Court challenging the demolition notice on the ground that no prior show-cause or hearing was given. Attach the sale deed, property map, and the demolition notice.

**What to collect immediately:**
Sale deed, property survey or mutation records, the demolition notice, any prior correspondence with the municipal corporation, and photos of the current state of the boundary.

**What NOT to do:**
Do not approach the municipal corporation directly to "explain." This creates a record of your appearance and may waive the procedural objection. Do not begin construction or modification of any structure until the injunction is in place.

**On the encroachment against the neighbour:**
That is a separate matter and can be filed as a civil suit for mandatory injunction and damages after the immediate demolition threat is addressed. Do not mix the two issues in one filing.

**Realistic expectation:**
The procedural defect is a strong ground for interim stay. A stay buys time. It does not resolve the underlying title dispute.

---

## Example 2 - Lay Client: Employment Termination with Prior Actions Trap

### Client Opening
*"I was working in a private company for 8 years. Last month they terminated me without any notice. They said it was performance-based. I don't agree. I want to take action against them. I have been messaging my manager and HR asking for explanation but they are not replying."*

---

### Ideal Follow-Up

**Question:** "What exactly have you sent in those messages to your manager and HR, specifically, have you said anything like 'I accept that my performance was poor' or 'I'm willing to resign if you give me a settlement'?"

**Why this question was chosen:**
The client has already been communicating with the employer after termination. These messages are now part of the evidentiary record. A single message that contains an admission, an offer to resign, or an expression of acceptance of the termination characterisation can seriously damage a wrongful termination claim. This question must be asked before any strategy is recommended.

This question changes: evidence strength, strategy, and whether a notice or filing should go out now or after damage control.

---

### Plausible but Wrong Follow-Up

**Wrong question:** "Did the company have a standing orders policy or a service manual?"

**Why it loses:**
That may become relevant later, but it comes after understanding what evidence already exists. If the client has already made damaging admissions in writing, knowing the policy manual does not change the immediate priority.

---

### Internal JSON State (After Turn 1)

```json
{
  "client": { "client_type": "lay_client" },
  "objective": {
    "stated_objective": "take action against company for wrongful termination",
    "real_objective": "reinstatement or compensation",
    "objective_type": "filing_strategy"
  },
  "urgency": {
    "urgency_level": "moderate",
    "urgency_flags": ["evidence_disappearing"]
  },
  "prior_actions": {
    "prior_actions_taken": ["Messages sent to manager and HR post-termination"],
    "adverse_prior_actions": ["Content of messages unknown - potential admissions"]
  },
  "decision_state": {
    "enough_to_advise": false,
    "current_best_decision": "Possible wrongful termination claim, but cannot recommend strategy until prior message content is assessed",
    "next_best_question": "What exactly did you say in those messages, any admissions, settlement offers, or acceptance of the performance characterisation?",
    "question_purpose": "evidence_strength + defence_exposure",
    "question_decision_value": "high"
  }
}
```

---

### Final Advice (After Sufficient Facts Are Known; grounded legal-advice layer)

*Assume: Client's messages were only asking for explanation, no admissions. 8 years of service. No written warning in the last 6 months. No formal performance improvement plan on record.*

**What to do today:**
Send a formal legal notice to the company demanding the termination order in writing, the grounds for termination, and the opportunity to respond if none has been given.

**Forum:**
For a private sector employee with 8 years of service and no prior written warnings, both a civil suit for wrongful termination and a labour-law route may be worth examining depending on employment classification.

**What to collect:**
Appointment letter, last salary slip, all written communications with the company, the termination letter if issued, any performance appraisals from the past 3 years, and the employment contract.

**What NOT to do:**
Stop messaging the manager and HR informally. All communication from now should go through formal legal notice only.

**Realistic expectation:**
Without written warnings or a documented performance improvement plan, the company's factual position may be weak. Whether reinstatement or compensation is available depends on forum and facts. Do not promise reinstatement.

---

## Example 3 - Junior Advocate: Criminal Complaint with Forum Question

### Junior Advocate Opening
*"Sir, I have a client whose business partner transferred company funds into his personal account, approximately 40 lakhs over 8 months. My client has bank statements showing the transfers. The partnership deed clearly says no partner can withdraw above 2 lakhs without consent. I was thinking of filing an FIR for criminal breach of trust. Is that the right approach?"*

---

### Ideal Follow-Up

**Question:** "Before I give you my view, what is your analysis of whether this is primarily a civil wrong or a criminal one, and what do you think is the threshold that takes this from breach of contract to criminal breach of trust?"

**Why this question was chosen:**
The junior has already done some thinking and has a factual basis. The senior's job here is not to give the answer immediately but to test whether the junior understands the distinction between a civil breach and a criminal act. If the junior cannot articulate the threshold, they will not be able to defend the FIR if the accused challenges it, and they will not understand why a civil suit might serve the client better in some circumstances.

This also reveals whether the junior has considered the criminal-civil strategy comparison, filing a criminal complaint for leverage vs. filing it as the primary remedy.

---

### Plausible but Wrong Follow-Up

**Wrong question:** "Has the client already filed a complaint with the police?"

**Why it loses:**
That is a prior-actions check, which may become relevant, but it is not the highest-decision-value question here. The junior is asking about forum and strategy. The right intervention is to test the diagnostic first.

---

### Internal JSON State (After Turn 1)

```json
{
  "client": { "client_type": "junior_advocate" },
  "objective": {
    "stated_objective": "confirm whether criminal complaint is the right approach",
    "real_objective": "recover 40 lakhs from business partner who misappropriated funds"
  },
  "facts": {
    "known_facts": [
      "40 lakhs transferred from company account to partner's personal account over 8 months",
      "Partnership deed restricts withdrawals above 2 lakhs without consent",
      "Bank statements available showing transfers"
    ],
    "assumptions_or_uncertainties": [
      "Whether partner claims these were authorised drawings or loans",
      "Whether prior notice or demand has been made",
      "Whether any civil action has been initiated"
    ]
  },
  "legal_analysis": {
    "legal_issues": [
      "Criminal breach of trust",
      "Civil suit for recovery and accounts",
      "Dissolution of partnership"
    ]
  },
  "decision_state": {
    "enough_to_advise": false,
    "current_best_decision": "Both criminal and civil routes may be viable. Choice depends on whether the goal is recovery, punishment, or leverage. Need to test junior's reasoning before advising.",
    "next_best_question": "What is your analysis of the civil vs criminal distinction, and what do you see as the threshold for criminal breach of trust here?",
    "question_purpose": "forum + maintainability",
    "question_decision_value": "high"
  }
}
```

---

### Final Advice to Junior (After Testing Their Reasoning; grounded legal-advice layer)

*Assume retrieval-backed analysis confirms the governing criminal-breach-of-trust standard requires entrustment, misappropriation, and dishonest intent, and the junior correctly identifies the civil-criminal overlap.*

**Issue framing:**
The core issues are: (1) criminal breach of trust if dishonest intent can be established from the pattern of transfers, (2) civil recovery of 40 lakhs with interest, (3) dissolution of partnership and accounts. These are not alternatives. They can run simultaneously, but the sequencing matters.

**On the criminal route:**
Maintainable if the partner was entrusted with the funds in a fiduciary capacity and the transfers demonstrate dishonest intent. However, the accused will argue these were authorised drawings or loans. The criminal route may be strong but contested, so prepare for anticipatory-bail risk immediately after complaint.

**Strategic recommendation:**
File a civil suit for accounts, injunction, and recovery first. This establishes the civil record and preserves evidence. Send a legal notice demanding return of funds and an account statement. Then, if no response or settlement within a reasonable period, consider the criminal complaint. This sequencing may be stronger than filing criminal first because the civil record can support the complaint and avoid the appearance of using criminal law as a pure recovery tool.

**Likely objection to anticipate:**
The accused may argue this is a civil dispute dressed as criminal. Your response is the pattern of transfers, the clear breach of the partnership deed, and the fiduciary nature of the relationship.

**Drafting priority:**
Civil plaint for recovery and injunction restraining further withdrawals. Simultaneously, draft the legal notice. Keep the criminal complaint draft ready.

---

## Pattern Summary for Additional Examples

When creating more training examples, every example must contain:

1. A client opening with at least one buried element, urgency, prior bad action, or hidden weakness, that the model must detect.
2. One ideal follow-up with explicit reasoning about which decision it changes.
3. One plausible wrong follow-up that sounds reasonable but misses the priority.
4. A JSON state snapshot showing `current_best_decision` and `next_best_question` populated.
5. Final advice structured as: action today / documents to collect / what not to do / realistic expectation.

The contrast between ideal and wrong follow-up is the most important training signal. Without it, the model learns what good looks like but not why bad alternatives lose.
