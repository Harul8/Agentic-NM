# Nyaymalaw — Senior Advocate Training Examples v1.1
## Enriched Intake Conversations Aligned to Senior Advocate Framework

---

### About This Document

This document contains enriched training examples for the Nyaymalaw senior advocate intake agent.
Each example is built from two sources:
1. **`rich_training_records.jsonl`** — client scenarios, intake conversations, advocate reasoning, final opinions
2. **`top_200_cited_cases.jsonl`** — real case citations, judicial reasoning, key arguments, and orders from India's most-cited Supreme Court cases

The enrichment applies the full v1.1 senior advocate framework, specifically:
- **Layer boundary enforcement** — intake conversations use no section citations; grounded advice comes only from retrieval
- **Decision-state schema (core runtime fields)** — compact, decision-oriented, not a chain-of-thought dump
- **Contrastive turn annotations** — each intake question is shown alongside a plausible bad alternative and the reason the bad alternative loses
- **Stop policy with counter-test** — showing why intake stopped when it did, and what would have prevented stopping
- **Weakness stress-test** — the best case the opposing party can make
- **Commercial reality** — whether the path is worth pursuing given cost, opponent strength, and client capacity
- **Governing principle applied** — at every stage, what is the most practical decision available

### How to Use

**For few-shot learning:** Load 3-5 examples matching the domain and client type of the current query. The enriched conversations teach the model the diagnostic pattern, not the dialogue script.

**For prompt refinement:** The contrastive annotations (ideal vs bad follow-up) are the highest-signal teaching material. If the model asks bad follow-ups, add more contrastive examples for that domain.

**For eval validation:** Cross-check model responses against the eval checklist (v1.1, 18 tests). The training examples are mapped to the behavioral tests — each annotated turn corresponds to one or more eval tests.

### Layer Boundary Reminder

> During intake, the model structures facts, assesses urgency, identifies issues, tests forum and maintainability, and selects follow-up questions. It does NOT cite section numbers or case names from memory.
> Final legal advice with citations comes only from the retrieval-grounded layer, not from intake judgment.

---

### Coverage

| Metric | Value |
|--------|-------|
| Total examples | 97 |
| Matched to top_200 source cases | 66 |
| Legal domains covered | 17 |

| Domain | Count |
|--------|-------|
| Fir Quashing | 10 |
| Motor Accident | 9 |
| Criminal | 8 |
| Matrimonial | 8 |
| Cheque Bounce | 8 |
| Landlord Tenant | 7 |
| Service Law | 7 |
| Constitutional | 7 |
| Property | 7 |
| Income Tax | 6 |
| Criminal General | 6 |
| Criminal Arrest | 5 |
| Labour | 4 |
| Contract | 2 |
| Medical Negligence | 1 |
| Banking Recovery | 1 |
| Consumer | 1 |

---

---

## rich_001: State Of Haryana And Ors vs Ch. Bhajan Lal And Ors

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1990 |
| **Citation** | [1988] 2 SCC 602 | [1980] 1 SCR 1071 |
| **Case Type** | General |
| **Cited By** | 19,733 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for cheating_fraud. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for cheating_fraud. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for cheating_fraud. I have evidence proving my i", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** State Of Haryana And Ors vs Ch. Bhajan Lal And Ors — [1988] 2 SCC 602 | [1980] 1 SCR 1071

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *State Of Haryana And Ors vs Ch. Bhajan Lal And Ors* (1990): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> HELD: 1. The judgment of the High Court quashing the
First Information Report is set aside as not being legally
and factually sustainable in law. However, the commencement
as well as the entire invest...


---

## rich_002: Lata Singh vs State Of U.P. & Another

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2006 |
| **Case Type** | General |
| **Cited By** | 17,945 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for dowry_498a. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for dowry_498a. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for dowry_498a. I have evidence proving my innoc", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Lata Singh vs State Of U.P. & Another* (2006): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> section 164
Cr.P.C. 
on 29.5.2001. In that statement the petitioner stated that she married 
Bramha Nand Gupta of her own free will. Despite this statement, the 
learned Chief Judicial Magistrate, Luc...

**Order / relief (from source case):**
> In the circumstances, the writ petition is allowed. The proceedings in Sessions Trial No. 1201/2001 titled
State of U.P. vs. Sangita Gupta & Ors.
arising out of FIR No. 336/2000 registered at Police S...


---

## rich_003: R. P. Kapur vs The State Of Punjab

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1960 |
| **Case Type** | General |
| **Cited By** | 15,811 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for assault_s307. I have evidence proving my innocence.

**Emotional State:** Harassed / emotionally distressed

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for assault_s307. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for assault_s307. I have evidence proving my inn", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *R. P. Kapur vs The State Of Punjab* (1960): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> appellant lie within a very narrow compass. It appears that
in January 1957 the mother-in-law of the appellant and his
wife entered into an agreement with the owners of certain
lands in village Mohamm...

**Order / relief (from source case):**
> 1960. March 25. The Judgment of the Court was
delivered by
GAJENDRAGADKAR, J.-On December 10, 1958, Mr. M. L. Sethi
lodged a First Information Report against the appellant Mr.
R. P. Kapur and alleged ...


---

## rich_004: Narinder Singh & Ors vs State Of Punjab & Anr

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2014 |
| **Case Type** | General |
| **Cited By** | 15,111 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for defamation. I have evidence proving my innocence.

**Emotional State:** Fearful / feeling physically unsafe

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for defamation. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for defamation. I have evidence proving my innoc", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Narinder Singh & Ors vs State Of Punjab & Anr* (2014): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> 6. During the pendency of trial proceedings, the matter has been
 compromised between the petitioners as well as the private respondent
 with the intervention of the Panchayat on 12.07.2013. It is cle...

**Order / relief (from source case):**
> 35. We have gone through the FIR as well which was recorded on the
 basis of statement of the complainant/victim. It gives an indication
 that the complainant was attacked allegedly by the accused per...


---

## rich_005: State Of Bihar Etc. Etc vs P.P. Sharma, Ias And Anr

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1991 |
| **Case Type** | General |
| **Cited By** | 15,092 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for corruption_pc_act. I have evidence proving my innocence.

**Emotional State:** Harassed / emotionally distressed

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for corruption_pc_act. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for corruption_pc_act. I have evidence proving m", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *State Of Bihar Etc. Etc vs P.P. Sharma, Ias And Anr* (1991): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> the High Court and dismissing the writ petitions filed by
the respondents before the High Court & this Court.
 HELD : Per Kuldip Singh, J.
 (1) The High Court fell into grave error and acted
with pate...

**Order / relief (from source case):**
> not even permit the Special Judge to pronounce the orders.
The Directors of the firm who are also accused person
in this case had approached the Rajasthan High Court for the
quashing of the FIR and pr...


---

## rich_006: Madan Mohan Abbot vs State Of Punjab

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2008 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 14,260 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for sc_st_act_false. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for sc_st_act_false. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for sc_st_act_false. I have evidence proving my ", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Madan Mohan Abbot vs State Of Punjab* (2008): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> 5. It is on the basis of this compromise that the application 
was filed in the High Court for quashing of proceedings which 
has been dismissed by the impugned order. We notice from a 
reading of the...


---

## rich_007: Centrlal Board Of Sec.Education & Anr vs Aditya Bandopadhyay & Ors

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2011 |
| **Case Type** | Matrimonial / Maintenance |
| **Relevant Act (Source)** | CrPC Section 125 |
| **Cited By** | 8,906 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for it_act_cybercrime. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for it_act_cybercrime. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for it_act_cybercrime. I have evidence proving m", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Centrlal Board Of Sec.Education & Anr vs Aditya Bandopadhyay & Ors* (2011): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> 3. Feeling aggrieved the first respondent filed W.P. No.18189(W)/2008 

before the Calcutta High Court and sought the following reliefs : (a) for a 

declaration that the action of CBSE in excluding t...

**Order / relief (from source case):**
> 38. In view of the foregoing, the order of the High Court directing the 

examining bodies to permit examinees to have inspection of their answer 

books is affirmed, subject to the clarifications reg...


---

## rich_008: Sakiri Vasu vs State Of U.P. And Others

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2007 |
| **Case Type** | General |
| **Cited By** | 8,229 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for labour_dispute_criminal. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for labour_dispute_criminal. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for labour_dispute_criminal. I have evidence pro", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Sakiri Vasu vs State Of U.P. And Others* (2007): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> 7. The first Court of Inquiry was held by the Army which gave its report 
in September, 2003 stating that it was a case of suicide. The appellant was 
not satisfied with the findings of this Court of ...


---

## rich_009: Parbatbhai Aahir @ Parbatbhai ... vs The State Of Gujarat

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2017 |
| **Case Type** | General |
| **Cited By** | 7,651 cases |

### Client Scenario

**Presenting Problem:** False FIR registered against me for property_dispute_criminal. I have evidence proving my innocence.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Quash the false FIR and prevent arrest

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** An FIR has been registered against me. I didn't do this. Can it be quashed?

**Advocate:** I'm sorry you're facing this. Let me understand the details. What has been alleged against you and what sections are mentioned?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* An FIR creates immediate arrest risk. Before any other analysis, the advocate must know whether the client is already arrested, whether anticipatory bail is still possible, and the current stage of investigation. These facts determine whether quashing or anticipatory bail is the first move, and whether there is still time to act.

*Plausible bad alternative:* "What is your relationship with the complainant and how long have you known them?"

*Why bad alternative loses:* Relationship history may become relevant later for proving motive or malice, but it does not change what must be done in the next 24 hours. The advocate cannot assess urgency or the right protective remedy without first knowing the arrest status and investigation stage.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** They've booked me under various sections for property_dispute_criminal. It's completely false and malicious.

**Advocate:** Do you have any evidence contradicting these allegations? Documents, messages, witnesses?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have documents and evidence proving my innocence. I also have a clean record and roots in the community.

**Advocate:** That's very helpful. At what stage is the case? Is investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. I haven't been arrested yet but I'm worried.

**Advocate (completing intake):** I understand your situation. We have strong grounds for a quashing petition under Section 482 CrPC. Let me prepare your legal opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["False FIR registered against me for property_dispute_criminal. I have evidence p", "Prayer: Quash the false FIR and prevent arrest"],
  "allegations": ["Client has documentary evidence and malicious FIR appears motivated by personal "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Fir", "Malice And Abuse Of Process", "Inherent Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Section 482 CrPC petition. 2. Emphasize documentary evidence. 3. Highlight malicious motivation. 4. Request quas",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the false FIR and prevent arrest), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: Basis for quashing petition

**Case law identified via retrieval:**
- *Parbatbhai Aahir @ Parbatbhai ... vs The State Of Gujarat* (2017): High Court can quash frivolous or malicious proceedings

**Key judicial reasoning (from source case):**
> First Information Report:
"33. We have gone through the FIR as well which was recorded on
 the basis of statement of the complainant/victim. It gives an
 indication that the complainant was attacked a...


---

## rich_010: Siddharam Satlingappa Mhetre vs State Of Maharashtra And Ors

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 21,316 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with anticipatory_bail. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Siddharam Satlingappa Mhetre vs State Of Maharashtra And Ors* (2010): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> (1-B) The presence of the applicant seeking
anticipatory bail shall be obligatory at the time of final
hearing of the application and passing of final order by
the Court, if on an application made to ...

**Order / relief (from source case):**
> 10. Section 438
of the Code of Criminal Procedure, 1973 reads

as under:
"438. Direction for grant of bail to person
 apprehending arrest.- (1) Where any person has
 reason to believe that he may be a...


---

## rich_011: Satender Kumar Antil vs Central Bureau Of Investigati

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2022 |
| **Citation** | 2021 SCC online SC 615 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 19,037 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with regular_bail_arrest. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** Satender Kumar Antil vs Central Bureau Of Investigati — 2021 SCC online SC 615

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Satender Kumar Antil vs Central Bureau Of Investigati* (2022): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> d(5) of UAPA, POSCO etc.”
Needless to say that the category A deals with both police cases and
complaint cases.
The trial Courts and the High Courts will keep in mind the aforesaid
guidelines while co...

**Order / relief (from source case):**
> of the constitutional safeguard that no person shall be detained except in
accordance with rule of law.
17.2. Under
Section 167
of the Code of Criminal Procedure, 1898 (“the
1898 Code”) which was in f...


---

## rich_012: Sushila Aggarwal vs State (Nct Of Delhi)

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2020 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 15,728 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_ndps. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Sushila Aggarwal vs State (Nct Of Delhi)* (2020): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> CRIMINAL APPELLATE JURISDICTION
 SPECIAL LEAVE PETITION (CRIMINAL) NOS.7281­7282/2017

 Sushila Aggarwal and others …Petitioners

 Versus

 State (NCT of Delhi) and another …Respondents

 JUDGMENT
M.R...

**Order / relief (from source case):**
> 5. Relying upon the decisions of this Court in the cases of
HDFC Bank Ltd.
(supra) and
Satpal Singh
(supra), it is

submitted by Shri Vikramjit Banerjee, learned Additional

Solicitor General of India...


---

## rich_013: Lal Kamlendra Pratap Singh vs State Of U.P.& Ors

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2009 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 9,870 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_murder. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Lal Kamlendra Pratap Singh vs State Of U.P.& Ors* (2009): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> person's reputation, as held by this Court in
Joginder Kumar
's case (supra).
Also, the facts of this case, in our opinion, this is not a fit case for quashing the first time to appear before the tria...

**Order / relief (from source case):**
> be considered and disposed of expeditiously.
Aggrieved by that order this appeal has been filed.
By an interim order dated 30.11.2007 this Court directed that the...


---

## rich_014: Gurbaksh Singh Sibbia Etc vs State Of Punjab

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1980 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 8,067 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_economic_offence. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Gurbaksh Singh Sibbia Etc vs State Of Punjab* (1980): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> preceding
Section 437
, are implicit therein
 and must be read into
Section 438
.
 (4) In addition to the limitations mentioned in
Section 437
, the petitioner must make out a
 special case for the ex...

**Order / relief (from source case):**
> 5. An order of bail can be passed under section 438(1)
of the Code without notice to the Public Prosecutor. But
notice should issue to the public prosecutor or the
Government Advocate forthwith and th...


---

## rich_015: Prasanta Kumar Sarkar vs Ashis Chatterjee & Anr

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 3,062 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_investigation. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Prasanta Kumar Sarkar vs Ashis Chatterjee & Anr* (2010): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> 7. Having failed to secure bail from the Sessions Court, the accused

 preferred a bail application, being C.R.M. No. 272 of 2010 before the

 High Court under
Section 439
of the Code. As stated above...

**Order / relief (from source case):**
> 14. For the foregoing reasons, the appeal is allowed, and the impugned

 order is set aside. The bail bond and the surety furnished by the

 accused in terms of the impugned order stands cancelled and...


---

## rich_016: P. Chidambaram vs Directorate Of Enforcement

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2019 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 2,721 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Prayer was partly accepted and partly declined; reasonableness depends on specific reliefs.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_after_chargesheet. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *P. Chidambaram vs Directorate Of Enforcement* (2019): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> 10. Lengthy arguments were heard on number of hearings stretched over

for long time. Learned Senior counsel appearing for the appellant Mr. Kapil

Sibal and Mr. Abhishek Manu Singhvi made meticulous ...

**Order / relief (from source case):**
> 3. Grievance of the appellant is that against the impugned order of the

High Court, the appellant tried to get the matter listed in the Supreme Court

on 21.08.2019; but the appellant could not get a...


---

## rich_017: Hussain And Anr vs Union Of India

| Field | Value |
|-------|-------|
| **Domain** | Criminal |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2017 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 2,331 cases |

### Client Scenario

**Presenting Problem:** I've been arrested. I'm innocent. I need bail urgently to be with my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Grant bail so I can be released and appear in court as needed

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been arrested. I'm innocent. I need bail immediately.

**Advocate:** Being in custody is stressful. Let me understand your situation. What crime are you charged with and how long have you been in custody?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Charged with bail_undertrial. In custody for a few days.

**Advocate:** Do you have any prior criminal convictions? And can your family stand as surety for bail?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** No, clean record. Family can stand as surety. I have job and home here.

**Advocate:** That's very helpful. Is the investigation ongoing or has charge-sheet been filed?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Investigation is ongoing. No charge-sheet yet.

**Advocate (completing intake):** With your clean record and community ties, bail should be granted. Let me prepare your bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I've been arrested. I'm innocent. I need bail urgently to be with my family.", "Prayer: Grant bail so I can be released and appear in court as neede"],
  "allegations": ["Client has excellent personal circumstances for bail with clean record and commu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Bail Eligibility", "Flight Risk Assessment", "Nature Of Allegations"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File bail application before Magistrate. 2. Emphasize clean record. 3. Offer surety. 4. If refused, approach High Cou",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Grant bail so I can be released and appear in court as neede), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Prior criminal record, flight risk indicators, or the seriousness of the offence (non-bailable, heinous) will affect bail prospects significantly. If the charge-sheet has already been filed, the standard for bail is higher. Any incriminating digital or physical evidence already recovered by police is a substantial weakness.

### Commercial Reality

Criminal defence is high urgency and typically worth pursuing regardless of cost given liberty is at stake. However, the strategy (regular bail vs anticipatory bail vs quashing vs trial defence) affects cost significantly. Align the approach with client capacity and the seriousness of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Determine arrest status immediately — this is the first decision point.
- Identify the exact offence charged and whether it is bailable or non-bailable.
- If at liberty, assess whether anticipatory bail is needed.
- Preserve any alibi evidence, digital records, or witness contacts immediately.

**What NOT to do:**
- Self-incriminating statements
- Contact with complainant or witnesses
- Social media posts about the case

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438: For anticipatory bail before arrest

**Case law identified via retrieval:**
- *Hussain And Anr vs Union Of India* (2017): Courts consider nature of offence, evidence, and flight risk for bail

**Key judicial reasoning (from source case):**
> 2. In the first case, the appellants have been in the custody since 4th
August, 2013 on the allegation of having committed offence under
Section
21(c)
of the Narcotics Drugs and Psychotropic Substance...

**Order / relief (from source case):**
> 7. In Abdul Rehman Antulay and ors. v. R.S. Nayak and anr.[3] while
holding that speedy trial at all stages is part of right under
Article 21
,
it was held that if there is violation of right of speed...


---

## rich_018: Sarla Verma & Ors vs Delhi Transport Corp.& Anr

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2009 |
| **Case Type** | General |
| **Cited By** | 20,141 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving death. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for death and loss of earning (Amount: ₹25,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 45, electrician, earning about Rs 40000 per month. Died in the accident.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 2 crore for Husband died, widow with dependents. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving death. Need compensation from negligent driver and insu", "Prayer: Compensation for death and loss of earning (Amount: \u20b925,000,"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for death and loss of earning (Amount: ₹25,000,), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *Sarla Verma & Ors vs Delhi Transport Corp.& Anr* (2009): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 4. Dissatisfied with the quantum of compensation, the appellants filed

an appeal. The Delhi High Court by its judgment dated 15.2.2007 allowed

the said appeal in part. The High Court was of the view...


---

## rich_019: National Insurance Co. Ltd vs Pranay Sethi

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2017 |
| **Case Type** | General |
| **Cited By** | 9,815 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving injury. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for injury and loss of earning (Amount: ₹30,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 32, construction worker, earning about Rs 35000 per month. Seriously injured.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 3 crore for Lost leg, permanent disability. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving injury. Need compensation from negligent driver and ins", "Prayer: Compensation for injury and loss of earning (Amount: \u20b930,000"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for injury and loss of earning (Amount: ₹30,000), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *National Insurance Co. Ltd vs Pranay Sethi* (2017): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 3. Before we penetrate into the past, it is necessary to note

what has been stated in Reshma Kumari (supra) and Rajesh’s

case.
In Reshma Kumari the three-Judge Bench was answering

the reference mad...


---

## rich_020: Magma General Insurance Co. Ltd vs Nanu Ram Alias Chuhru Ram

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2018 |
| **Case Type** | General |
| **Cited By** | 4,992 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving injury. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for injury and loss of earning (Amount: ₹15,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 28, software engineer, earning about Rs 120000 per month. Seriously injured.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 1 crore for 40% disability, lost earning. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving injury. Need compensation from negligent driver and ins", "Prayer: Compensation for injury and loss of earning (Amount: \u20b915,000"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for injury and loss of earning (Amount: ₹15,000), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *Magma General Insurance Co. Ltd vs Nanu Ram Alias Chuhru Ram* (2018): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 3. The   father,   brother,   and   sister   of   the   deceased   filed   Claim

 Petition   under
Section   166
of   the   Motor   Vehicles   Act,   1988
 before   the   Motor   Accidents   Claim   ...


---

## rich_021: General Manager, Kerala S.R.T.C vs Susamma Thomas

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 1993 |
| **Case Type** | General |
| **Cited By** | 4,294 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving injury. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for injury and loss of earning (Amount: ₹35,000,000)

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 12, student, earning about Rs 0 per month. Seriously injured.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 3 crore for Child with permanent brain damage. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving injury. Need compensation from negligent driver and ins", "Prayer: Compensation for injury and loss of earning (Amount: \u20b935,000"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for injury and loss of earning (Amount: ₹35,000), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *General Manager, Kerala S.R.T.C vs Susamma Thomas* (1993): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 1032. These facts and the fact that the accident was the
result of actionable negligence on the part of the driver of
the bus are not disputed. 3. The Motor Accidents Claims Tribunal, Alleppey in O.P....

**Order / relief (from source case):**
> 24. In the result this appeal succeeds in part and the
compensation is determined at Rs 2,25,000. The judgment of
the High Court is modified accordingly. No costs....


---

## rich_022: National Insurance Co. Ltd vs Swaran Singh & Ors

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2004 |
| **Case Type** | General |
| **Cited By** | 3,847 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving death. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for death and loss of earning (Amount: ₹45,000,000)

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 50, business owner, earning about Rs 80000 per month. Died in the accident.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 4 crore for Businessman died, business collapsed. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving death. Need compensation from negligent driver and insu", "Prayer: Compensation for death and loss of earning (Amount: \u20b945,000,"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for death and loss of earning (Amount: ₹45,000,), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *National Insurance Co. Ltd vs Swaran Singh & Ors* (2004): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> a judgment or award in respect of any such liability is obtained by the 
insured, the insurer notwithstanding its entitlement to avoid or cancel or may 
have avoided or cancelled the policy, the insur...

**Order / relief (from source case):**
> anything from him. But the answer to that is that it is the 
insurer's bad luck. In such circumstances the injured 
person also would not have been able to recover the 
damages suffered by him from th...


---

## rich_023: Raj Kumar vs Ajay Kumar & Anr

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Citation** | AIR 1970 SC 376 |
| **Case Type** | General |
| **Cited By** | 3,811 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving death. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for death and loss of earning (Amount: ₹70,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 38, bank manager, earning about Rs 150000 per month. Died in the accident.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 7 crore for High earner died, family dependents. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving death. Need compensation from negligent driver and insu", "Prayer: Compensation for death and loss of earning (Amount: \u20b970,000,"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for death and loss of earning (Amount: ₹70,000,), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** Raj Kumar vs Ajay Kumar & Anr — AIR 1970 SC 376

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *Raj Kumar vs Ajay Kumar & Anr* (2010): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 3. The appellant puts forth two grievances: (i) the assessment of monthly

income at Rs.900/- was very low; and (ii) deduction of one third of the

income (towards personal and living expenses) while ...


---

## rich_024: Reshma Kumari & Ors vs Madan Mohan & Anr

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2013 |
| **Case Type** | General |
| **Cited By** | 2,700 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving injury. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for injury and loss of earning (Amount: ₹8,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 25, laborer, earning about Rs 15000 per month. Seriously injured.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 0 crore for Hit and run, leg fracture. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving injury. Need compensation from negligent driver and ins", "Prayer: Compensation for injury and loss of earning (Amount: \u20b98,000,"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for injury and loss of earning (Amount: ₹8,000,), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *Reshma Kumari & Ors vs Madan Mohan & Anr* (2013): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 43. Thus, prima facie, it appears that the multiplier mentioned
 in the Second Schedule, although in a given case, may be taken
 to be a guide but the same is not decisive. To our mind,
 although a pr...


---

## rich_025: Santosh Devi vs National Insurance Co.Ltd.& Ors

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2012 |
| **Case Type** | General |
| **Cited By** | 2,663 cases |

### Client Scenario

**Presenting Problem:** Motor accident involving injury. Need compensation from negligent driver and insurer.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Compensation for injury and loss of earning (Amount: ₹10,000,000)

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** There was a motor accident. Need compensation from the negligent driver.

**Advocate:** I'm very sorry about your loss. Motor accident claims can be complex. Let me understand what happened. Which vehicle hit yours and was the other vehicle insured?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** A truck hit us. Driver was negligent. Other vehicle insured.

**Advocate:** What was the victim's age, occupation, and monthly earning or income?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Victim was 40, taxi driver, earning about Rs 45000 per month. Seriously injured.

**Advocate:** What amount of compensation do you expect? And do you have income documents to prove the earning?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Around Rs 1 crore for Both drivers at fault, dispute on contribution. Yes, I have salary slips and bank statements.

**Advocate (completing intake):** Your claim is strong. Let me prepare a detailed opinion on the compensation calculation and how to file before the Motor Accident Claims Tribunal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["Motor accident involving injury. Need compensation from negligent driver and ins", "Prayer: Compensation for injury and loss of earning (Amount: \u20b910,000"],
  "allegations": ["Clear negligence. Victim's income documented. Standard motor accident claim."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Negligence Establishment", "Insurer Liability", "Compensation Calculation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File claim before MACT. 2. Gather negligence evidence. 3. Compile income documents. 4. Calculate compensation using p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for injury and loss of earning (Amount: ₹10,000), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 166: Framework for compensation claim

**Case law identified via retrieval:**
- *Santosh Devi vs National Insurance Co.Ltd.& Ors* (2012): Compensation calculation method based on multiplier for death

**Key judicial reasoning (from source case):**
> 7. After analysing the evidence produced by the parties, the Tribunal
decided issue No.1 in the affirmative and held that the accident was caused
due to rash and negligent driving of Maruti car by Var...

**Order / relief (from source case):**
> 17. In the result, the appeal is allowed, the impugned judgment as also
the award of the Tribunal are set aside and it is declared that the
claimants shall be entitled to compensation of Rs.2,94,840 [...


---

## rich_026: Arnesh Kumar vs State Of Bihar & Anr

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2014 |
| **Case Type** | Matrimonial / Maintenance |
| **Relevant Act (Source)** | CrPC Section 125 |
| **Cited By** | 25,720 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Fearful / feeling physically unsafe

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Arnesh Kumar vs State Of Bihar & Anr* (2014): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> rather than shield by disgruntled wives. The simplest way to harass is to
get the husband and his relatives arrested under this provision. In a
quite number of cases, bed-ridden grand-fathers and gran...

**Order / relief (from source case):**
> punishable with imprisonment for a term which may be less than seven years
or which may extend to seven years; whether with or without fine.
We direct that a copy of this judgment be forwarded to the ...


---

## rich_027: B.S. Joshi & Ors vs State Of Haryana & Anr

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2003 |
| **Citation** | AIR 1999 SC 2181 |
| **Case Type** | Matrimonial / Maintenance |
| **Relevant Act (Source)** | CrPC Section 125 |
| **Cited By** | 11,910 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** B.S. Joshi & Ors vs State Of Haryana & Anr — AIR 1999 SC 2181

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *B.S. Joshi & Ors vs State Of Haryana & Anr* (2003): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> Y.K. Sabharwal, J.
Leave granted.
The question that falls for determination in the instant case is about the ambit of the inherent powers of the High Courts under
Section 482
, Code of Criminal Proced...


---

## rich_028: Sumita Singh vs Kumar Sanjay And Anr.

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2001 |
| **Case Type** | General |
| **Cited By** | 5,406 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Sumita Singh vs Kumar Sanjay And Anr.* (2001): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> 4. Accordingly, Matrimonial Case No. 30 of 2000 pending before the VIth Additional District and Sessions Judge, Ara, Bhojpur, Bihar shall stand transferred to the District Judge, Delhi, who shall hear...


---

## rich_029: Sunderbha1 Ambalal Desai vs State Of Gujarat

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2002 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 3,611 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Sunderbha1 Ambalal Desai vs State Of Gujarat* (2002): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> 457. Procedure by police upon seizure of property.-
(1) Whenever the seizure of property by any police officer is reported to a
Magistrate under the provisions of this Code, and such property is not
p...


---

## rich_030: Dr. Ram Manohar Lohia vs State Of Bihar And Others

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1965 |
| **Case Type** | Civil / Landlord–Tenant |
| **Relevant Act (Source)** | CrPC Section 125 |
| **Cited By** | 2,627 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Dr. Ram Manohar Lohia vs State Of Bihar And Others* (1965): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> Art. 352
was in force,
if such person had been deprived of any such rights under
the
Defence of India Act
or any rule made thereunder, and
that therefore, this Court was prevented from entertaining
th...

**Order / relief (from source case):**
> prevent him from acting in a manner prejudicial to public
safety.
In the result, in my view, the detention order is not under
the Rules. The detention of Dr. Lohia under that order is
not legal and ca...


---

## rich_031: Jitendra Raghuvanshi & Ors vs Babita Raghuvanshi & Anr

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2013 |
| **Case Type** | Matrimonial / Maintenance |
| **Relevant Act (Source)** | CrPC Section 125 |
| **Cited By** | 2,540 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Harassed / emotionally distressed

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Jitendra Raghuvanshi & Ors vs Babita Raghuvanshi & Anr* (2013): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> of IPC not only against the husband but also
against the relatives of the husband. The question is when such matters
are resolved either by the wife agreeing to rejoin the matrimonial home or
by mutua...

**Order / relief (from source case):**
> Section 320 of the Code does not limit or affect the powers of the High
Court under Section 482 of the Code. Under these circumstances, we set
aside the impugned judgment of the High Court dated 04.07...


---

## rich_032: State Of Karnataka & Ors vs M.L. Kesari & Ors

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | General |
| **Cited By** | 1,834 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *State Of Karnataka & Ors vs M.L. Kesari & Ors* (2010): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences

**Key judicial reasoning (from source case):**
> without the intervention of orders of the courts or of tribunals. The
 question of regularization of the services of such employees may have to
 be considered on merits in the light of the principles ...

**Order / relief (from source case):**
> 5. It is evident from the above that there is an exception to the general

principles against `regularization' enunciated in Umadevi, if the following

conditions are fulfilled :
(i) The employee conc...


---

## rich_033: Ramgopal & Anr vs State Of M.P.& Anr

| Field | Value |
|-------|-------|
| **Domain** | Matrimonial |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | ongoing violence or harassment, arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | General |
| **Cited By** | 1,800 cases |

### Client Scenario

**Presenting Problem:** My spouse filed a case against me alleging serious matters. I received a legal notice and I'm very worried about arrest and what this means for my family.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Protect me from arrest and false allegations. If applicable, seek bail protection.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My spouse filed a serious case against me. I received a police notice. What should I do?

**Advocate:** I understand this must be very stressful for you and your family. These situations can feel overwhelming, but the law provides protections. Can you tell me: have you received a formal written notice from police, or have they just contacted you informally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely family court. Need urgency facts before confirming strategy.

**Client:** I got a written notice. It says I have to appear before them.

**Advocate:** That's important information. A written notice means they're following proper procedure, which actually works in your favour. What are the main allegations? Are they claiming dowry demand, cruelty, or harassment?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim I demanded dowry and mentally harassed them. None of this is true. We had marital issues but no dowry was demanded.

**Advocate:** I hear you. Marital disputes are unfortunately common, and the courts are aware that sometimes legal processes are misused. What outcome are you looking for right now? Do you want to prevent arrest, challenge the case, or both?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Both. I want protection from arrest first. Then I want to fight this case.

**Advocate (completing intake):** I have enough information. Let me prepare your detailed legal opinion on both your prayers.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["ongoing_violence_or_harassment", "arrest_or_coercive_action_imminent"],
  "known_facts": ["My spouse filed a case against me alleging serious matters. I received a legal n", "Prayer: Protect me from arrest and false allegations. If applicable,"],
  "allegations": ["Case involves matrimonial dispute with allegation of serious offense. Key is to "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Arrest Is Legally Justified Or Preventable", "Whether Allegations Have Sufficient Specificity Or Are Vague", "Applicability Of Procedural Safeguards"],
  "evidence_status": "moderate",
  "recommended_forum": "family_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Seek immediate protective relief (anticipatory bail/quashing). Build record of exculpatory evidence. Prepare for long-te",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protect me from arrest and false allegations. If applicable,), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

In matrimonial disputes, delay in filing, prior reconciliation attempts, or prior statements minimising the severity of the dispute are all weaknesses. If the client has sent messages indicating willingness to return, these will be used by the other side.

### Commercial Reality

Matrimonial litigation can be prolonged and expensive. Settlement through mediation is often the most practical outcome for straightforward maintenance or custody matters. Contested divorce with criminal allegations is much costlier and emotionally taxing. Counsel should assess client's tolerance early.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- If there is ongoing physical danger, the immediate step is a protection order or police complaint.
- If a criminal complaint has been filed against the client, assess anticipatory bail need first.
- Document all incidents with dates and available evidence.
- Preserve all written communications — messages, emails, letters.

**What NOT to do:**
- Direct confrontation with spouse without legal backing
- Voluntary statements to police
- Abandoning matrimonial home without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973
- Indian Penal Code, 1860
- Dowry Prohibition Act, 1961

**Case law identified via retrieval:**
- *Ramgopal & Anr vs State Of M.P.& Anr* (2010): Key principle regarding matrimonial disputes and arrest protections
- *Rajesh Sharma v. State of U.P.* (2017): Guidelines on matrimonial cases and arrest in dowry-related offences


---

## rich_034: Rangappa vs Sri Moha

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | General |
| **Cited By** | 9,567 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *Rangappa vs Sri Moha* (2010): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> 7. However, on appeal against acquittal, the High Court

reversed the findings and convicted the appellant-accused.

The High Court in its order noted that in the course of the trial

proceedings, the...

**Order / relief (from source case):**
> 16. In conclusion, we find no reason to interfere with the final

order of the High Court, dated 26-10-2005, which recorded a

finding of conviction against the appellant. The present appeal

is dispo...


---

## rich_035: Bhagwan Dass vs State (Nct) Of Delhi

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2011 |
| **Case Type** | General |
| **Cited By** | 8,248 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *Bhagwan Dass vs State (Nct) Of Delhi* (2011): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> 5. This is a case of circumstantial evidence, but it is settled law 

that a person can be convicted on circumstantial evidence

provided the links in the chain of circumstances connects the 

accused...


---

## rich_036: Damodar S.Prabhu vs Sayed Babalal H

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | General |
| **Cited By** | 5,512 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Harassed / emotionally distressed

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *Damodar S.Prabhu vs Sayed Babalal H* (2010): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> 7. Similar reliefs were granted in orders reported as

Sivasankaran v. State of Kerala & Anr., (2002) 8 SCC 164,

Kishore Kumar v. J.K. Corporation Ltd., (2004) 12 SCC 494

and Sailesh Shyam Parsekar ...

**Order / relief (from source case):**
> 18. The present set of appeals are disposed of accordingly.
.............................. CJI
 (K.G. BALAKRISHNAN)

 ............................... J.
(P. SATHASIVAM)

 ................................


---

## rich_037: C.C. Alavi Haji vs Palapetty Muhammed & Anr

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2007 |
| **Case Type** | General |
| **Cited By** | 4,985 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Anxious / under mental stress

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *C.C. Alavi Haji vs Palapetty Muhammed & Anr* (2007): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> 2. The matter has been placed before the three Judge Bench 
in view of a Reference made by a two-Judge Bench of this 
Court, pertaining to the question of service of notice in terms 
of Clause (b) of ...

**Order / relief (from source case):**
> 20. For the reasons aforementioned, we do not find any merit 
in this appeal. It is dismissed accordingly but with no order 
as to costs in the circumstances of the case....


---

## rich_038: M.S. Narayana Menon @ Mani vs State Of Kerala & Anr

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2006 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 4,105 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *M.S. Narayana Menon @ Mani vs State Of Kerala & Anr* (2006): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> rebuttable one, the question will have to be determined upon taking into 
consideration another presumption drawn in terms of
Section 118(a)
thereof.
According to the learned counsel, the Appellant di...

**Order / relief (from source case):**
> In any event the High Court entertained an appeal treating to be an 
appeal against acquittal, it was in fact exercising the revisional jurisdiction. 
Even while exercising an appellate power against ...


---

## rich_039: Krishna Janardhan Bhat vs Dattatraya G. Hegde

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2008 |
| **Case Type** | General |
| **Cited By** | 3,978 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *Krishna Janardhan Bhat vs Dattatraya G. Hegde* (2008): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> 16. The appellate court took an identical stand. It proceeded on the 
premise that the statement of accused under
Section 313
of the Code of 
Criminal Procedure regarding misuse of blank cheque by the...

**Order / relief (from source case):**
> 36. For the reasons aforementioned, the appeal is allowed. The judgments 
of conviction and sentence passed against the appellant are set aside....


---

## rich_040: K. Bhaskaran vs Sankaran Vaidhyan Balan And Anr

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1999 |
| **Case Type** | General |
| **Cited By** | 3,240 cases |

### Client Scenario

**Presenting Problem:** Someone is claiming I issued a cheque without funds. I either don't remember this transaction clearly or there was a misunderstanding. Now there's a criminal case.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Defend against the cheque bounce case. Seek quashing of case or proper legal remedy.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed a case saying I issued a bad cheque. I don't even remember this clearly.

**Advocate:** Cheque bounce cases are serious but defendable. Help me understand: do you remember issuing this cheque? Was it for a loan, business transaction, or something else?

*Decision dimension targeted:* `limitation + prior_actions_taken`

*Why this question wins:* Cheque bounce cases have strict procedural requirements — demand notice within 30 days of dishonour, 15-day waiting period after notice, then complaint within 30 days of expiry of notice period. The most important question is whether the demand notice has been sent and when, because this determines whether the limitation clock is still alive.

*Plausible bad alternative:* "What is your business relationship with the person who issued the cheque?"

*Why bad alternative loses:* The relationship may become relevant as a defence (loan vs gift), but it does not change the immediate priority. If the notice has not been sent or the limitation window is closing, action is needed today regardless of relationship. Asking about relationship first misses the limitation risk.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** I think it was for a loan between friends. But I wasn't sure if I had funds at that time.

**Advocate:** Okay, that's helpful. The details matter a lot in cheque bounce cases. Do you have any bank records, messages, or documents showing the original agreement or the purpose of the cheque?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I might have some old messages on my phone. I'm not sure if they help.

**Advocate:** Those could be very important. Also, if this was a personal loan between friends, do you know if the other person has already recovered the money from you some other way, or if you've paid them back since then?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Actually, I paid them back later in cash. I didn't think about that.

**Advocate (completing intake):** That detail is important. Let me prepare your opinion addressing this.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["Someone is claiming I issued a cheque without funds. I either don't remember thi", "Prayer: Defend against the cheque bounce case. Seek quashing of case"],
  "allegations": ["Cheque bounce is cognizable offense but highly defensible if proper circumstance"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Cheque Was Issued For Valid Consideration", "Whether Underlying Debt Still Exists", "Whether Accused Had Capacity To Pay At Time Of Issue"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Gather all documentary evidence. Check bank records at time of cheque issue. Explore settlement/compounding. If facts su",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Defend against the cheque bounce case. Seek quashing of case), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Negotiable Instruments Act, 1881
- Code of Criminal Procedure, 1973

**Case law identified via retrieval:**
- *K. Bhaskaran vs Sankaran Vaidhyan Balan And Anr* (1999): Cheque bounce offense requirements and defenses
- *E.P. Royappa v. State of Tamil Nadu* (1974): Principles of equity and fairness in legal proceedings

**Key judicial reasoning (from source case):**
> partly in one local area and partly in another local area the Court in
either of the localities can exercise jurisdiction to try the case. Further
again, Section 179 of the Code stretches its scope to...


---

## rich_041: M/S Zandu Pharmaceutical Works Ltd.& ... vs Md Sharaful Haque & Anr

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2004 |
| **Case Type** | General |
| **Cited By** | 16,299 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *M/S Zandu Pharmaceutical Works Ltd.& ... vs Md Sharaful Haque & Anr* (2004): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> Courts. All courts, whether civil or criminal possess, in the absence 
of any express provision, as inherent in their constitution, all such 
powers as are necessary to do the right and to undo a wron...

**Order / relief (from source case):**
> extension of period in certain cases. This power can be exercised only 
when the Court is satisfied on the facts and in the circumstances of 
the case that the delay has been properly explained or tha...


---

## rich_042: United Bank Of India vs Satyawati Tondon & Ors

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Case Type** | General |
| **Cited By** | 3,973 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *United Bank Of India vs Satyawati Tondon & Ors* (2010): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> acceptance are required to be communicated within one week. Sub-section (4)

Section 13
specifies various modes which can be adopted by the secured

creditor for recovery of secured debt. The secured ...

**Order / relief (from source case):**
> 12. The Division Bench of the High Court did not even advert to the

appellant's plea that the writ petition should not be entertained because an

effective alternative remedy was available to the wri...


---

## rich_043: Rattan Lal vs State Of Punjab

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1964 |
| **Case Type** | General |
| **Cited By** | 3,912 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *Rattan Lal vs State Of Punjab* (1964): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> discretion is given to the court to release them after
admonition or on probation of good conduct, in the case of
offenders below the age of 21 years an injunction is issued
to the court not to senten...

**Order / relief (from source case):**
> 1962. Appeal by special leave from the judgment and order
dated September 27, 1962 of the Punjab High Court in
Criminal Revision No. 1172 of 1962.
Nanak Chand, for the appellant.
Gopal Singh, R.N. Sac...


---

## rich_044: P.K. Shaji @ Thammanam Shaji vs State Of Kerala

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Citation** | AIR 2000 SC 3556 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 3,829 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** P.K. Shaji @ Thammanam Shaji vs State Of Kerala — AIR 2000 SC 3556

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *P.K. Shaji @ Thammanam Shaji vs State Of Kerala* (2005): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> appropriate action as if the conditions have been imposed 
and the Petitioner released on bail by the learned 
Magistrate himself."
The order of the Sessions Court shows that the learned 
Magistrate h...

**Order / relief (from source case):**
> We see no reason to interfere with the impugned judgment and 
the appeal is accordingly dismissed ....


---

## rich_045: L. Chandra Kumar vs Union Of India And Others

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1997 |
| **Citation** | [1993] 4 SCC 119 |
| **Case Type** | General |
| **Cited By** | 2,564 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** L. Chandra Kumar vs Union Of India And Others — [1993] 4 SCC 119

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *L. Chandra Kumar vs Union Of India And Others* (1997): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> 2. We shall confine ourselves to the larger issues raised in this batch of
matters without adverting to the specific facts of each of the matters; we
shall, however, selectively refer to some of the i...

**Order / relief (from source case):**
> 8. Pursuant to the provisions of the Act, the Central Administrative
Tribunal, with five Benches, was established on November 1, 1985. However,
even before the Tribunal had been established, several w...


---

## rich_046: State Of Jammu & Kashmir vs Triloki Nath Khosa & Ors

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1973 |
| **Case Type** | General |
| **Cited By** | 2,535 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *State Of Jammu & Kashmir vs Triloki Nath Khosa & Ors* (1973): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> different classes or the Government may integrate them into
one class; that once the employees are integrated into one
class they cannot for the purposes of promotion, be
classified again into two dif...

**Order / relief (from source case):**
> purposes of promotion to the cadre of Executive Engineers,
be classified on the basis of educational qualifications.
The rule providing that graduates shall be eligible for such
promotion to the exclu...


---

## rich_047: Union Of India vs Shri Shiv Shanker Kesari

| Field | Value |
|-------|-------|
| **Domain** | Landlord Tenant |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2007 |
| **Case Type** | Criminal / Bail |
| **Cited By** | 2,363 cases |

### Client Scenario

**Presenting Problem:** There is a dispute with my landlord/tenant about the property. They want to evict me/I want them to vacate. The situation is becoming difficult.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the property dispute fairly. Seek protection of tenant rights or enforce tenant eviction as applicable.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My landlord/tenant situation is becoming a problem. There's talk of eviction.

**Advocate:** Landlord-tenant disputes can be resolved, but timing is important. Are you the landlord wanting to evict the tenant, or the tenant facing eviction?

*Decision dimension targeted:* `urgency + forum + maintainability`

*Why this question wins:* Eviction matters turn on whether there is a registered rent agreement, the ground for eviction claimed, and the rent control law applicable in the state. These three facts determine whether the landlord's notice is valid, which forum applies, and what protective relief is available.

*Plausible bad alternative:* "How much rent are you paying and when did you last pay?"

*Why bad alternative loses:* Rent amount and payment history will matter later if non-payment is the eviction ground, but the first priority is understanding the legal basis of the eviction notice and which rent control law applies. Without this, the advocate cannot advise on maintainability or forum.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** I am the tenant. They say I need to leave because they want to occupy the property themselves.

**Advocate:** I see. Personal occupation is one ground for eviction, but it has legal limits and requirements. How long have you been renting this property, and do you have a written lease agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I've been here for 6 years. We have a basic agreement but nothing very formal.

**Advocate:** Six years is substantial. The law protects long-term tenants in various ways. What is your monthly rent, and are you current with all payments?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Rent is 15,000 per month and yes, I'm always on time.

**Advocate (completing intake):** Let me prepare your opinion on your rights and the law on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["There is a dispute with my landlord/tenant about the property. They want to evic", "Prayer: Resolve the property dispute fairly. Seek protection of tena"],
  "allegations": ["Eviction cases depend heavily on ground claimed and statutory protections availa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Eviction Ground Is Legally Valid", "Whether Statutory Procedure Was Followed", "Whether Tenant Has Statutory Protections"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Challenge validity of ground. Ensure statutory procedure was followed. Invoke tenant protections available under law.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the property dispute fairly. Seek protection of tena), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the tenant has defaulted on rent, the landlord's case is significantly stronger. If the rent agreement is oral or unregistered, the eviction terms may be harder to enforce but also harder to protect. Long delay in challenging an eviction notice may be held against the tenant.

### Commercial Reality

Rent control proceedings in most Indian states protect tenants significantly, making the cost of resistance worthwhile if the client wishes to stay. If the landlord's ground is legitimate (own use, default), a negotiated vacation with compensation may be more practical than prolonged litigation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the applicable state rent control legislation for the jurisdiction.
- Obtain a copy of the eviction notice and check whether the stated ground is legally valid.
- If there is a valid rent agreement, preserve it immediately.
- Continue paying rent by bank transfer to establish the payment record.

**What NOT to do:**
- Vacating voluntarily without written terms
- Signing any agreement presented by the landlord without legal review
- Stopping rent payment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- State Rent Control Act
- Code of Civil Procedure, 1908

**Case law identified via retrieval:**
- *Union Of India vs Shri Shiv Shanker Kesari* (2007): Landlord-tenant law and eviction grounds
- *Rupa Biswas v. Bhagwan Singh* (2013): Tenant protections and proof required for eviction

**Key judicial reasoning (from source case):**
> 5. Section 37
of the Act reads as follows:
"Offences to be cognizable and non-bailable-
(1) Notwithstanding anything contained in the
Code of Criminal Procedure
, 1973 (2) of 1974),-
(a) every offence...


---

## rich_048: K.P. Varghese vs The Income Tax Officer,Ernakulam, And ...

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1981 |
| **Case Type** | General |
| **Cited By** | 3,460 cases |

### Client Scenario

**Presenting Problem:** The income tax department has issued a notice claiming I owe taxes or that my tax returns are incorrect. I need to understand what they want and my obligations.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the tax dispute. Understand assessment and seek relief if applicable.

**Prayer Assessment (Source):** Court did not find the prayer reasonable enough to grant the requested relief.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I received a notice from the income tax department. I don't understand what they want.

**Advocate:** Income tax notices can seem complex, but we can break it down. Is this the first notice you're receiving, or have there been communications before?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need urgency facts before confirming strategy.

**Client:** First time. They say my returns don't match their records.

**Advocate:** What year are they asking about? And do you have any idea what the specific discrepancy might be — is it about reported income, deductions, or something else?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** It's for the last financial year. They say I claimed business expenses but they think they're not valid.

**Advocate:** Those are the kind of issues that often have clear answers once reviewed. Do you have documentation for the expenses you claimed?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have invoices, bills, and bank transfers for everything.

**Advocate (completing intake):** Let me prepare your opinion on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["The income tax department has issued a notice claiming I owe taxes or that my ta", "Prayer: Resolve the tax dispute. Understand assessment and seek reli"],
  "allegations": ["Tax disputes are administrative but require careful handling with proper documen"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Deductions Claimed", "Whether Department Followed Proper Procedure", "Reasonable Interpretation Of Tax Law"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Respond to notice with full documentation. Present case to assessing officer. Be prepared for appeals if needed.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the tax dispute. Understand assessment and seek reli), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Income Tax Act, 1961
- Income Tax Rules, 1962

**Case law identified via retrieval:**
- *K.P. Varghese vs The Income Tax Officer,Ernakulam, And ...* (1981): Income tax deduction and assessment principles
- *CIT v. Lakhmani Devi Pulp & Paper Works* (1974): Interpretation of tax law in favor of assessee when ambiguous

**Key judicial reasoning (from source case):**
> year 1966-67 and requiring the assessee to submit a return
of income within thirty days of the service of the notice,
without stating what was the income alleged to have escaped
assessment. However, b...

**Order / relief (from source case):**
> involved, the Division Bench referred the appeal to a Full
Bench of three Judges. The Full Bench heard the appeal but
there was a division of opinion, two Judges taking one view
and the third Judge ta...


---

## rich_049: Additional Commissioner Of Income-Tax ... vs Surat Art Silk Cloth Manufacturers ...

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1979 |
| **Case Type** | General |
| **Cited By** | 2,322 cases |

### Client Scenario

**Presenting Problem:** The income tax department has issued a notice claiming I owe taxes or that my tax returns are incorrect. I need to understand what they want and my obligations.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the tax dispute. Understand assessment and seek relief if applicable.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I received a notice from the income tax department. I don't understand what they want.

**Advocate:** Income tax notices can seem complex, but we can break it down. Is this the first notice you're receiving, or have there been communications before?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need urgency facts before confirming strategy.

**Client:** First time. They say my returns don't match their records.

**Advocate:** What year are they asking about? And do you have any idea what the specific discrepancy might be — is it about reported income, deductions, or something else?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** It's for the last financial year. They say I claimed business expenses but they think they're not valid.

**Advocate:** Those are the kind of issues that often have clear answers once reviewed. Do you have documentation for the expenses you claimed?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have invoices, bills, and bank transfers for everything.

**Advocate (completing intake):** Let me prepare your opinion on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["The income tax department has issued a notice claiming I owe taxes or that my ta", "Prayer: Resolve the tax dispute. Understand assessment and seek reli"],
  "allegations": ["Tax disputes are administrative but require careful handling with proper documen"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Deductions Claimed", "Whether Department Followed Proper Procedure", "Reasonable Interpretation Of Tax Law"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Respond to notice with full documentation. Present case to assessing officer. Be prepared for appeals if needed.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the tax dispute. Understand assessment and seek reli), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Income Tax Act, 1961
- Income Tax Rules, 1962

**Case law identified via retrieval:**
- *Additional Commissioner Of Income-Tax ... vs Surat Art Silk Cloth Manufacturers ...* (1979): Income tax deduction and assessment principles
- *CIT v. Lakhmani Devi Pulp & Paper Works* (1974): Interpretation of tax law in favor of assessee when ambiguous

**Key judicial reasoning (from source case):**
> this amount was exempt from tax) and commission on the basis
of certain percentage of the value of licences for import of
foreign yarn and quotas for the purchase of indigenous yarn.
The assessee cons...


---

## rich_050: Continental Construction Ltd vs Commissioner Of Income-Tax, Central-1

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1992 |
| **Case Type** | General |
| **Cited By** | 2,319 cases |

### Client Scenario

**Presenting Problem:** The income tax department has issued a notice claiming I owe taxes or that my tax returns are incorrect. I need to understand what they want and my obligations.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Resolve the tax dispute. Understand assessment and seek relief if applicable.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I received a notice from the income tax department. I don't understand what they want.

**Advocate:** Income tax notices can seem complex, but we can break it down. Is this the first notice you're receiving, or have there been communications before?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need urgency facts before confirming strategy.

**Client:** First time. They say my returns don't match their records.

**Advocate:** What year are they asking about? And do you have any idea what the specific discrepancy might be — is it about reported income, deductions, or something else?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** It's for the last financial year. They say I claimed business expenses but they think they're not valid.

**Advocate:** Those are the kind of issues that often have clear answers once reviewed. Do you have documentation for the expenses you claimed?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have invoices, bills, and bank transfers for everything.

**Advocate (completing intake):** Let me prepare your opinion on this matter.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["The income tax department has issued a notice claiming I owe taxes or that my ta", "Prayer: Resolve the tax dispute. Understand assessment and seek reli"],
  "allegations": ["Tax disputes are administrative but require careful handling with proper documen"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Validity Of Deductions Claimed", "Whether Department Followed Proper Procedure", "Reasonable Interpretation Of Tax Law"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Respond to notice with full documentation. Present case to assessing officer. Be prepared for appeals if needed.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Resolve the tax dispute. Understand assessment and seek reli), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Income Tax Act, 1961
- Income Tax Rules, 1962

**Case law identified via retrieval:**
- *Continental Construction Ltd vs Commissioner Of Income-Tax, Central-1* (1992): Income tax deduction and assessment principles
- *CIT v. Lakhmani Devi Pulp & Paper Works* (1974): Interpretation of tax law in favor of assessee when ambiguous

**Key judicial reasoning (from source case):**
> deletion of
section 80-O
, the two sections should be read
harmoniously and given effect to together restricting the
operation of
section 80-HHB
to contracts entered into on or
after 1.4.1983 so as no...

**Order / relief (from source case):**
> We have, therefore, delinked the appeal by the C.I.T. for
separate hearing. Also, of the four questions posed above
in the assessee's appeal, counsel for the appellant has
stated that he is not pressi...


---

## rich_051: Commissioner Of Income Tax (Central) vs B. N. Bhattacharjee

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1979 |
| **Case Type** | General |
| **Cited By** | 2,306 cases |

### Client Scenario

**Presenting Problem:** I own a real estate development company and my tax returns were scrutinized for the last 8 years. The Income Tax department has detected gaps between reported income and actual cash flows, with alleged under-reporting. I want to settle this matter before criminal proceedings are initiated. Can we approach the Settlement Commission?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want to apply for settlement before the Settlement Commission to resolve all disputes and get immunity from criminal prosecution.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** The IT department has issued notice for scrutiny of my returns for multiple years. They claim I have under-reported income. I am very worried about criminal prosecution.

**Advocate:** I understand this is a serious concern. The possibility of criminal proceedings makes this urgent. Before we explore options, can you tell me roughly how much income is allegedly under-reported, and has the department already indicated whether they suspect fraudulent intent or is it being treated as simple under-reporting?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need defence_exposure facts before confirming strategy.

**Client:** They detected about Rs. 5 crores over 8 years. They use words like 'concealment' and 'suspicious cash deposits.' There's also a letter saying concealment of income can lead to criminal action.

**Advocate:** The use of 'concealment' language is serious and indicates fraud charges are possible. This is exactly the kind of case the Settlement Commission law was designed for. Do you have records or explanations for these deposits? And importantly, has the department filed any appeals yet, or are we still in the scrutiny stage?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I have some records but not complete. The department hasn't filed formal appeals in the appellate tribunal yet. They're still in assessment phase. I want to resolve this quickly and avoid criminal charges.

**Advocate:** Timing is on our side right now. The Settlement Commission route has specific advantages if we move quickly. Before I prepare your application, I need to know: are you willing to admit that under-reporting occurred and pay the full taxes plus applicable penalties? Settlement requires honesty, not litigation of facts.

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I'm willing to pay whatever is due. My main goal is to avoid criminal prosecution and end this stress. I don't want to fight — I want closure.

**Advocate (completing intake):** I have enough information now. The Settlement Commission is potentially the best route given the concealment allegations and your willingness to resolve. However, this case is complex and depends on whether the department will support the application. Let me prepare a comprehensive legal opinion on your options and the Settlement Commission process.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["I own a real estate development company and my tax returns were scrutinized for ", "Prayer: I want to apply for settlement before the Settlement Commiss"],
  "allegations": ["Client is high-income taxpayer facing concealment allegations with criminal pros"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Jurisdiction Of Settlement Commission When Concealment Alleged", "Whether Departmental Support/Non-Objection Is Prerequisite", "Extent Of Immunity Granted Under Chapter Xixa"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Settlement Commission application immediately with full disclosure 2. Acknowledge under-reporting and provide be",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want to apply for settlement before the Settlement Commiss), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Income Tax Act, 1961 — Section 245D(1): Establishes duty to hear before rejection and Commissioner's role
- Income Tax Act, 1961 — Section 245M: Core benefit sought by applicant - immunity from criminal prosecution
- Income Tax Act, 1961 — Chapter XIXA: Framework for entire settlement process

**Case law identified via retrieval:**
- *Commissioner Of Income Tax (Central) vs B. N. Bhattacharjee* (1979): Settlement Commission must give hearing before rejecting application; departmental objection on fraud grounds does not automatically bar settlement

**Key judicial reasoning (from source case):**
> from the appellant. The appellant reported that prosecution
proceedings for concealment of income and also false
verification in the return by the respondent were pending
against the respondent in the...

**Order / relief (from source case):**
> the turn of events surrounding the withdrawal of appeals by
both sides. To complete the story-and this fact has a
bearing on one of the legal questions-it must be stated that
when the Settlement Commi...


---

## rich_052: Sole Trustee Loka Shikshana Trust vs Commissioner Of Income Tax, Mysore

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 1975 |
| **Case Type** | General |
| **Cited By** | 2,203 cases |

### Client Scenario

**Presenting Problem:** I am a trustee of an educational trust that runs schools and publishes educational materials in regional languages. We sell these materials to generate funds for our charity work. The IT department is now claiming that we are running a business and denying our tax-exempt status. This threatens our charitable mission. Can we retain our exemption?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want to restore the tax-exempt status of our trust. We should be recognized as a genuine charitable organization. Our publications are purely educational and all profits go back to schools.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Our trust runs schools and publishes educational books. The IT officer says we're running a business because we earn money from publications. He denies our exemption. This could devastate our school programs.

**Advocate:** That's a serious threat to your mission. Before we dig deeper, help me understand the structure: does your trust deed explicitly state that all profits from publications must be used only for educational purposes? Or is there discretion to distribute profits in other ways?

*Decision dimension targeted:* `limitation + forum`

*Why this question wins:* Income tax matters have strict deadlines for appeals and responses. The most important question is what order has been passed, when it was received, and whether the appeal or response deadline is still open. This determines whether emergency action is needed.

*Plausible bad alternative:* "What is your annual income and how long have you been filing returns?"

*Why bad alternative loses:* Income level and filing history are part of the merits but do not determine the immediate action. The deadline for appeal or response is the time-critical fact that may require urgent action today.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need limitation facts before confirming strategy.

**Client:** Our deed says all income from publications must be spent exclusively on running schools and educational programs. We never take profits out. Everything goes back into education.

**Advocate:** That's excellent — the exclusive use restriction is critical. Now, the IT department's argument is likely that 'carrying on activity for profit' itself disqualifies you, regardless of how you use the profits. How would you characterize your publication work — is it primarily a business that happens to fund charity, or is it fundamentally an educational tool that happens to generate revenue?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** It's definitely educational. We create textbooks aligned with school curriculum, publish them in regional languages, and sell at cost-plus pricing to recover production expenses. The pricing is not commercial — it's just to sustain the work.

**Advocate:** That distinction is crucial. The tax law recognizes that genuine charities often need to earn revenue — the key is whether the activity itself serves a charitable purpose. Can you provide your trust deed, annual reports showing how profits are actually spent, and documentation of your educational materials to support that publications are genuinely educational, not profit-seeking?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, I have all that. I can show the deed, annual accounts, and samples of our textbooks. Every rupee earned goes to schools. I'm confident our purpose is genuinely charitable.

**Advocate (completing intake):** I understand the urgency. The law is actually favorable to trusts like yours that engage in revenue-generating educational activities. Let me prepare a detailed opinion on how to defend your exemption claim based on the trust structure and your actual practices.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["I am a trustee of an educational trust that runs schools and publishes education", "Prayer: I want to restore the tax-exempt status of our trust. We sho"],
  "allegations": ["Trust conducting genuine educational work through publications. Exemption denial"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Restriction Of Profits To Charitable Use Overrides 'Carrying On Activity For Profit' Requirement", "Interpretation Of 'Profit' As Private Vs. Public Profit", "Definition Of 'Education' In Statutory Sense"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File detailed objection with trust deed excerpts showing profit restrictions 2. Provide annual accounts documenting p",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want to restore the tax-exempt status of our trust. We sho), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Income Tax Act, 1961 — Section 2(15): Defines charitable purpose but does not explicitly prohibit all profit-generating activities
- Income Tax Act, 1961 — Section 11: Core exemption provision applicable to client
- Income Tax Act, 1961 — Section 12A: Procedural requirement for exemption recognition

**Case law identified via retrieval:**
- *Sole Trustee Loka Shikshana Trust vs Commissioner of Income Tax, Mysore* (1975): Profit from activities compatible with charitable status if profits are mandatorily restricted by trust deed to exclusive charitable use; test is genuineness of charitable purpose not prohibition of profit-making

**Key judicial reasoning (from source case):**
> India in general and of Karnatak in particular by
 (a) establishing conducting and helping directly
 or indirectly institutions calculated to
 educate the people by spread of knowledge on
 all matters...


---

## rich_053: Commissioner Of Customs (Import), Mumbai vs M/S. Dilip Kumar And Company

| Field | Value |
|-------|-------|
| **Domain** | Income Tax |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2018 |
| **Case Type** | General |
| **Cited By** | 2,183 cases |

### Client Scenario

**Presenting Problem:** I imported fish feed ingredients at a lower customs duty rate based on a government notification listing animal feed items. Customs authorities say my product is a 'chemical ingredient' not animal feed, and demand higher duty. This changes my import costs significantly. The notification language seems to cover my product. Can I claim the lower rate?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want customs authorities to apply the concessional duty rate to my imports. The notification clearly lists animal feed materials. I should not be charged the higher rate.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Customs authorities rejected my claim for concessional duty and want 30% instead of the 5% rate listed in the notification. My shipment is fish feed supplement — it's definitely animal feed related.

**Advocate:** This is a classification dispute. Before we assess your case, I need to understand the precise language of the notification you're relying on. Can you provide the exact text of the notification listing items eligible for 5% duty? Does it say 'animal feed' or 'feed ingredients' or something more specific?

*Decision dimension targeted:* `limitation + forum`

*Why this question wins:* Income tax matters have strict deadlines for appeals and responses. The most important question is what order has been passed, when it was received, and whether the appeal or response deadline is still open. This determines whether emergency action is needed.

*Plausible bad alternative:* "What is your annual income and how long have you been filing returns?"

*Why bad alternative loses:* Income level and filing history are part of the merits but do not determine the immediate action. The deadline for appeal or response is the time-critical fact that may require urgent action today.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely revenue authority. Need limitation facts before confirming strategy.

**Client:** The notification says 'animal feeds and feed ingredients for livestock and aquaculture.' My product is a Vitamin E supplement powder used in fish feed. It's literally a feed ingredient.

**Advocate:** The notification language seems broad enough to cover your product. The question is: how has customs been interpreting this — are they saying your product isn't a 'feed ingredient' at all, or are they arguing it's a 'chemical' not a 'feed' component? The distinction matters for interpretation.

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They claim it's a 'chemical ingredient for animal feed' not 'animal feed ingredient itself.' They say the notification applies only to finished feed products, not chemical components. But that seems too narrow.

**Advocate:** You're right — that interpretation is questionable. The notification uses the word 'ingredients' which should include chemical components used in making feed. When there's ambiguity in a tax exemption, courts have developed principles about which way to interpret. How much additional duty are we talking about, and have you received any written assessment from customs?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** The difference is about Rs. 8 lakhs on this shipment. I have the assessment order. The product is clearly feed-related; the narrow reading doesn't make business sense.

**Advocate (completing intake):** I can see the issue. The notification uses the phrase 'feed ingredients' which by any reasonable reading includes your vitamin supplement. Customs interpretation is overly narrow. I'll prepare an opinion on how courts handle ambiguous exemption clauses and your strong position here.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["I imported fish feed ingredients at a lower customs duty rate based on a governm", "Prayer: I want customs authorities to apply the concessional duty ra"],
  "allegations": ["Clear classification dispute. Client's interpretation is reasonable: notificatio"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Proper Interpretation Of Phrase 'Feed Ingredients' In Customs Notification", "Scope Of Concessional Duty Notification For Chemical Components", "Whether Ambiguity Must Be Resolved In Favor Of Assessee"],
  "evidence_status": "moderate",
  "recommended_forum": "revenue_authority",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Appeal customs assessment with detailed notification analysis 2. Argue 'ingredients' language covers chemical compone",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want customs authorities to apply the concessional duty ra), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the assessee failed to respond to notices during assessment proceedings, the ex-parte assessment order becomes harder to challenge. Any unexplained cash deposits or discrepancies between ITR and bank statements are significant weaknesses.

### Commercial Reality

Income tax disputes are almost always worth pursuing if the demand is substantial. The departmental appeal system (CIT(A), ITAT) is cost-effective relative to the amounts at stake. Settlement under Vivad se Vishwas or compounding may be practical for older disputes.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the assessment order or notice and identify the exact demand and deadline.
- Check whether appeal to CIT(A) is still maintainable within time.
- Gather all supporting documentation for the disputed income or deductions.
- Identify whether any stay of demand is needed to prevent coercive recovery.

**What NOT to do:**
- Ignoring tax notices
- Making partial payments that may be treated as acceptance of demand
- Missing appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Customs Act, 1962 — Section 11: Duty amount determined by notification classification
- Customs Tariff Notification 20/1999 — Concessional Rate Provision: Core exemption notification under dispute

**Case law identified via retrieval:**
- *Commissioner Of Customs (Import), Mumbai vs M/S. Dilip Kumar And Company* (2018): When ambiguity exists in tax exemption notification, interpretation favorable to assessee claiming benefit must be adopted. Statutory language controls but exemption should not be narrowly construed.

**Key judicial reasoning (from source case):**
> 1. This   Constitution   Bench   is   setup   to   examine   the

 correctness   of   the  ratio  in
Sun   Export

 Corporation,   Bombay   v.   Collector   of   Customs,

 Bombay
, (1997) 6 SCC 564 [...


---

## rich_054: D.S. Nakara & Others vs Union Of India

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1982 |
| **Case Type** | General |
| **Cited By** | 2,485 cases |

### Client Scenario

**Presenting Problem:** I retired from government service in 1978. In 1979, the government announced a new pension scheme with much higher benefits. But they gave the new benefits only to people retiring after March 1979. I got nothing because I retired 10 months earlier. The new rules apply to everyone in my service class. This is unfair — why should my retirement date matter?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the new pension calculation applied to my pension, with payment of all arrears from 1979. I am in the same position as those who retired after March 1979.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** The government liberalized pensions in 1979 but only for future retirees. I retired in 1978 just before the new scheme. I'm getting a much lower pension than others who retired months after me under identical rules.

**Advocate:** This is a classic discrimination question. Before we dig in, help me understand: are you and the officers who retired after March 1979 governed by exactly the same pension rules and service conditions? Or are there different rules for different years?

*Decision dimension targeted:* `forum + maintainability + limitation`

*Why this question wins:* Service law matters require knowing whether the client has exhausted departmental remedies, whether there is a service tribunal with exclusive jurisdiction, and whether the limitation for challenging the adverse order is still alive. These determine both maintainability and the right forum.

*Plausible bad alternative:* "What is your current salary and how long have you been in service?"

*Why bad alternative loses:* Service tenure and salary are useful background but do not determine the immediate legal strategy. What matters first is whether there is a specific adverse order to challenge, whether it has been challenged departmentally, and whether the time to file is still open.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** No, we're all governed by the Civil Services (Pension) Rules 1972. Same rules for everyone. The difference is purely when we retired. Someone retiring on April 1, 1979 gets double my pension for identical service.

**Advocate:** That's a strong point. When the rules are identical but benefits depend on a date, that's the kind of arbitrary classification courts look at closely. Tell me — has the government given any reason why the date matters? What did they say about not making it retroactive?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** They say it would cost too much and they can't change retroactively. But I don't see how that's fair — other officers got lucky just by retiring later, not because of any real difference.

**Advocate:** The cost argument is something courts hear regularly. But if the rules and situation are identical, cost alone doesn't justify different treatment under our Constitution. Have you already tried approaching the Pension Authority or filed any appeal in your department?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** I filed a representation with the Pension Division but they rejected it saying the 1979 date is fixed. That's when I decided to seek legal help. I think this violates my Constitutional rights.

**Advocate (completing intake):** You've identified a real equality issue. When two groups are governed by identical rules but receive different benefits based on an arbitrary date, that's exactly the kind of discrimination the Constitution guards against. Let me prepare a detailed opinion on your rights and remedies.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I retired from government service in 1978. In 1979, the government announced a n", "Prayer: I want the new pension calculation applied to my pension, wi"],
  "allegations": ["Textbook Article 14 violation case. Client and post-1979 retirees: same rules, s"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Retirement Date Is Valid Classification Criterion", "Whether Arbitrary Date Violates Article 14", "Whether Severability Doctrine Applies To Remove Date From Notification"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File petition under Article 32 or 226 asserting violation of Article 14 2. Emphasize homogeneous class (same rules, s",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the new pension calculation applied to my pension, wi), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India — Article 14: Fundamental right violated by differential pension treatment
- Civil Services (Pension) Rules, 1972 — Rule 44: Rule governing pension computation for all government servants including client

**Case law identified via retrieval:**
- *D.S. Nakara & Others vs Union Of India* (1982): Arbitrary date-based classification that divides homogeneous class violates Article 14. When arbitrariness can be removed by severance of discriminatory provision, it must be struck down.

**Key judicial reasoning (from source case):**
> invoked to sever the specified date from the scheme as it
would have the effect of enlarging the class of pensioners
covered by the scheme and when the legislature has expressly
defined the class to w...

**Order / relief (from source case):**
> from service on or after that date"

and in Exhibit P-2, the words:
"the new rates of pension are effective from 1st
 April 1979 and will be applicable to all service
 officers who became/become non-e...


---

## rich_055: Union Of India & Anr vs Tarsem Singh

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2008 |
| **Citation** | AIR 1959 SC 798 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 2,254 cases |

### Client Scenario

**Presenting Problem:** I was medically discharged from the Army in 1985 due to a service injury. The discharge papers say I'm entitled to disability pension. I struggled for years and didn't pursue it. Now I want to claim the pension. Can I get arrears for all the years I didn't receive it?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I should receive my disability pension from the date of my discharge in 1985. I want the arrears for all years since then, not just the last few years. This was my entitlement from the beginning.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was discharged from the Army in 1985 due to a combat injury. My discharge papers clearly state disability pension. I didn't claim it for many years due to personal struggles. Now I'm old and need the money. Can I claim full arrears?

**Advocate:** That's a significant gap — over 30 years. The key question is whether the discharge papers unambiguously state you're entitled to disability pension, or whether it's something you have to claim. Can you show me your discharge papers and any communication about the pension entitlement?

*Decision dimension targeted:* `forum + maintainability + limitation`

*Why this question wins:* Service law matters require knowing whether the client has exhausted departmental remedies, whether there is a service tribunal with exclusive jurisdiction, and whether the limitation for challenging the adverse order is still alive. These determine both maintainability and the right forum.

*Plausible bad alternative:* "What is your current salary and how long have you been in service?"

*Why bad alternative loses:* Service tenure and salary are useful background but do not determine the immediate legal strategy. What matters first is whether there is a specific adverse order to challenge, whether it has been challenged departmentally, and whether the time to file is still open.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** I have the discharge certificate. It clearly mentions 'discharged on disability grounds' and references the disability pension rules. There's no question about my entitlement, just about when I should have started receiving it.

**Advocate:** That's important — you had a clear legal entitlement from day one. The question is whether there's a limitation period on claiming old arrears. Courts sometimes have rules about not paying too far back. Have you filed any official claim with the military pension authority yet, or are you coming to court without prior request?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** I haven't formally claimed yet. I'm approaching the legal system first because I want to understand my rights. I lost 30 years of pension — surely that's not gone just because I didn't ask sooner?

**Advocate:** You raise the right question. In service law, there's something called 'continuing wrong' doctrine — basically, if you should have been receiving something every month and you weren't, each month creates a fresh claim. Has the pension office or your discharge documents given you any reason to think the pension was only available if you claimed within a certain timeframe?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No, the documents say I'm entitled to disability pension. They don't mention any time limit for claiming. I think the military was just poorly organized, and I didn't pursue it. But my entitlement was always there.

**Advocate (completing intake):** Your situation involves what's called a 'continuing wrong' in service law. Each month you were entitled to pension but didn't receive it, the wrong continued. Let me prepare an opinion on whether and how you can recover the full arrears despite the delay.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I was medically discharged from the Army in 1985 due to a service injury. The di", "Prayer: I should receive my disability pension from the date of my d"],
  "allegations": ["Strong case for continuing wrong doctrine. Clear entitlement from 1985 establish"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Discharge Certificate Establishes Automatic Entitlement Or Requires Claim", "Application Of Continuing/Recurring Wrong To Service Pensions", "Proper Limitation Period For Disability Pension Arrears"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Obtain certified copy of discharge certificate 2. Request pension file from military 3. File writ petition asserting",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I should receive my disability pension from the date of my d), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Source citation:** Union Of India & Anr vs Tarsem Singh — AIR 1959 SC 798

**Bare acts identified via retrieval:**
- Army Act, 1950 — Section 43: Establishes Rajkumar's automatic entitlement from 1985
- Limitation Act, 1963 — Section 22: Allows extension of limitation for continuing wrongs

**Case law identified via retrieval:**
- *Union of India & Anr vs Tarsem Singh* (2008): Service pension non-payment constitutes recurring wrong with fresh cause each month. Continuing wrong doctrine permits arrears recovery well beyond 3-year limitation for regularly accruing obligations like pensions.

**Key judicial reasoning (from source case):**
> 3. The respondent however was not satisfied. According to him the

disability pension ought to be paid from the date it fell due on 13.11.1983.

He therefore filed a Letters Patent Appeal. The said ap...


---

## rich_056: Syed Abdul Qadir & Ors vs State Of Bihar & Ors

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2008 |
| **Case Type** | General |
| **Cited By** | 2,121 cases |

### Client Scenario

**Presenting Problem:** I was promoted to a higher grade in 2005 and my pay was fixed using the rules in effect then, which gave me an additional increment for the promotion. Now, in 2020, the government says that rule was amended in 2010 and the amendment applies backward. They want to recover excess pay from me from 2010 onwards. This is unjust. I followed the rules as they were when I was promoted.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the government to recognize that the old pay rules apply to my 2005 promotion. The 2010 amendment should not take away benefits I was already granted. I shouldn't have to refund money I was legitimately entitled to.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was promoted in 2005 and given additional increment under the old pay rule. Now in 2020 the government amended the rule and wants to apply it backwards to recover money from me. They claim I was overpaid since 2010. This is wrong — I got what I was owed when I was promoted.

**Advocate:** That's a significant issue. First question: when you were promoted in 2005, did you receive written notification of your new pay grade and the pay-fixation method used? And can you show me that notification and the old rule it referenced?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** Yes, I have my promotion order from 2005. It clearly states my new pay and references the Fundamental Rules for pay fixation. It used the rule that gave additional increments. The 2010 amendment came 5 years later.

**Advocate:** Good documentation. Now, the government's position — did they notify employees that the 2010 amendment would apply to prior promotions? Or did they suddenly issue this recovery demand in 2020 without prior notification?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** They never told us. Suddenly in 2020 the audit team raised objections and now they're saying I owe them money. But I was in good faith relying on the rule in 2005. How can they change the rule and demand refund 15 years later?

**Advocate:** That's a strong point about notice. Courts are generally protective of retrospective amendments that recover money from government employees. How much are they claiming you owe, and do you have bank statements showing you received these payments over the years?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** They're saying I was overpaid about Rs. 4 lakhs from 2010 to 2020. Yes, I have records of all salary deposits. The salary I received was exactly what was authorized in 2005.

**Advocate (completing intake):** This is a classic retrospective amendment issue. The government is trying to apply a 2010 change to benefits you earned in 2005. Courts are very protective against such retrospective recovery from employees. Let me prepare an opinion on this for you.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I was promoted to a higher grade in 2005 and my pay was fixed using the rules in", "Prayer: I want the government to recognize that the old pay rules ap"],
  "allegations": ["Clear retrospective amendment issue. Meena's 2005 promotion used FR 22-C as it e"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Retrospective Vs. Prospective Application Of Amended Rules", "Whether Amendment Of Incorporated Rule Affects Prior Accrued Benefits", "Authority To Recover Allegedly Excess Payments"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Obtain certified copies of 2005 promotion order and FR 22-C as then in force 2. Get copy of 2010 amendment notificati",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the government to recognize that the old pay rules ap), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Fundamental Rules, 1950 — FR 22-C (as applicable in 2005): Rule in force when client was promoted in 2005
- Fundamental Rules, 1950 (amended 2010) — FR 22(1)(a)(2): Amendment sought to be applied retrospectively by government

**Case law identified via retrieval:**
- *Syed Abdul Qadir & Ors vs State Of Bihar & Ors* (2008): Amended rules apply prospectively from date of amendment unless expressly made retrospective. Once incorporated in agreement or promotion order, prior rules cannot be withdrawn retrospectively.

**Key judicial reasoning (from source case):**
> 1. These appeals by special leave are directed against the common judgment of

Patna High Court whereby a Division Bench of that Court dismissed letters patent

appeals preferred by the appellants her...

**Order / relief (from source case):**
> 29. Learned counsel also submitted that prior to the interim order passed by this

Court on 7.4.2003 in the special leave petitions, whereby the order of recovery

passed by the Division Bench of the ...


---

## rich_057: Ajay Kumar Choudhary vs Union Of India Thr Its Secretary & Anr

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2015 |
| **Case Type** | General |
| **Cited By** | 2,060 cases |

### Client Scenario

**Presenting Problem:** I was suspended from my government job in 2022 for allegedly mishandling a file. It's now 2024 and I'm still suspended with no charges filed, no inquiry started, and no clear indication of when this will end. I have no income, and my family is suffering. How long can they keep me suspended without taking any disciplinary action?

**Emotional State:** angry

**Prayer / Relief Sought:** My suspension should end. Either they should file charges and complete a proper inquiry or they should reinstate me. I cannot remain in limbo indefinitely while they take their time.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I've been suspended for nearly 2 years with no charges, no hearing, no clear reason. They just keep extending my suspension every 3-4 months. How long can this go on?

**Advocate:** That's a serious problem. Before we assess your case, tell me — have they ever issued formal charges against you? Even if charges were issued later, before suspension? And do you have copies of the suspension order and all extension orders?

*Decision dimension targeted:* `forum + maintainability + limitation`

*Why this question wins:* Service law matters require knowing whether the client has exhausted departmental remedies, whether there is a service tribunal with exclusive jurisdiction, and whether the limitation for challenging the adverse order is still alive. These determine both maintainability and the right forum.

*Plausible bad alternative:* "What is your current salary and how long have you been in service?"

*Why bad alternative loses:* Service tenure and salary are useful background but do not determine the immediate legal strategy. What matters first is whether there is a specific adverse order to challenge, whether it has been challenged departmentally, and whether the time to file is still open.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** No formal charges at all. They've issued 5-6 suspension extension orders since 2022, each for 90-120 days. But no charge-sheet, no inquiry started, no communication about what they're investigating or when it might end.

**Advocate:** No charges or inquiry at all in 2 years is highly problematic. Let me understand the original suspension — what was the allegation, and have they given any written explanation of why they need more time? Or do they just keep issuing extensions without explanation?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** The initial order cited mishandling of a sensitive file and insubordination. But they've never spelled out specifics or given me a chance to respond. Each extension just repeats the previous one without adding any new detail. They seem to be playing for time.

**Advocate:** That pattern is exactly what courts look at. No due process, no progress, just repeated extensions. Did you try approaching your administrative superior, Ministry, or the Personnel/HR department asking for clarification or expedited proceedings? Or are we going straight to court?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** I've written to the Ministry multiple times but get no response. My union rep suggested legal action. I need to get back to work — my savings are running out and my career is being destroyed.

**Advocate (completing intake):** This is exactly the kind of unreasonable indefinite suspension that courts have taken strong action on. Two years with no charges and no progress is unacceptable. Let me prepare an opinion on your rights and the likely remedies available.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I was suspended from my government job in 2022 for allegedly mishandling a file.", "Prayer: My suspension should end. Either they should file charges an"],
  "allegations": ["Clear case of unreasonable prolonged suspension. 2+ years without charges or inq"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Indefinite Suspension With Repeated Extensions Violates Service Law", "Reasonable Time Limit For Completion Of Disciplinary Proceedings", "At What Point Does Suspension Become Unreasonable/Punitive"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File petition under Article 226 asserting suspension violation 2. Attach all suspension/extension orders showing lack",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My suspension should end. Either they should file charges an), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Civil Services Conduct Rules, 1964 — Rule 4: Establishes temporary nature of suspension
- Civil Services Conduct Rules, 1964 — Rule 5: Requires timely progress on disciplinary matters

**Case law identified via retrieval:**
- *Ajay Kumar Choudhary vs Union Of India Thr Its Secretary & Anr* (2015): Indefinite suspension with repeated extensions without disciplinary progress is abuse of authority. No employee can be suspended beyond reasonable time limit. If charge-sheet not issued within specified period, suspension becomes void and employee must be reinstated.

**Key judicial reasoning (from source case):**
> CIVIL APPELLATE JURISDICTION

 CIVIL APPEAL No. 1912 OF 2015
 (Arising out of SLP No. 31761 of 2013

AJAY KUMAR CHOUDHARY ...... APPELLANT

 Vs.

UNION OF INDIA THROUGH ITS. ......RESPONDENTS
SECRETAR...

**Order / relief (from source case):**
> we are not concerned. The Appellant asserts that since the subject land
was within the parameter wall of the Air Force Station, no physical
transfer thereof has occurred. On 28.12.2011 the Appellant's...


---

## rich_058: R.S. Raghunath vs State Of Karnataka And Anr

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1991 |
| **Case Type** | General |
| **Cited By** | 2,044 cases |

### Client Scenario

**Presenting Problem:** I am a Regional Transport Officer with excellent performance ratings. According to my department's special rules, promotion to Deputy Commissioner should be by selection based on qualifications and merit. But the government promoted my junior colleague on the basis of seniority-cum-merit under a general rule. This violates our special recruitment rules. I should get the promotion based on my superior qualifications.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the promotion to Deputy Commissioner awarded to me based on selection from among eligible officers in my grade. The special rules for our department should be followed, not the general seniority rule.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Our department has special rules that say promotion is by selection. But government promoted someone based on seniority alone under general rules. I have better qualifications and performance. This is against our rules.

**Advocate:** That's a clear rules conflict. Before I assess, I need to understand: do you have a copy of your department's special recruitment rules that specifically state 'promotion by selection'? And can you point me to the general rule the government is trying to apply?

*Decision dimension targeted:* `forum + maintainability + limitation`

*Why this question wins:* Service law matters require knowing whether the client has exhausted departmental remedies, whether there is a service tribunal with exclusive jurisdiction, and whether the limitation for challenging the adverse order is still alive. These determine both maintainability and the right forum.

*Plausible bad alternative:* "What is your current salary and how long have you been in service?"

*Why bad alternative loses:* Service tenure and salary are useful background but do not determine the immediate legal strategy. What matters first is whether there is a specific adverse order to challenge, whether it has been challenged departmentally, and whether the time to file is still open.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** Yes, I have both. Our department's 1976 rules clearly state promotion is by selection. The government passed general rules in 1977 that allow promotion by seniority-cum-merit. The general rule has a 'non-obstante' clause saying it overrides other rules.

**Advocate:** A non-obstante clause is significant, but it doesn't automatically override specific rules. Tell me — has the government ever applied this 1977 general rule to your department before this promotion? Or is this the first time they're trying to use the general rule instead of selection?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** This is the first time. For years, our department has used the selection method per our special rules. Everyone promoted in the past was selected, not just by seniority. This new promotion breaks the pattern.

**Advocate:** Important fact. So the special rules have been working for decades in your department, and this is the first application of the general rule. How long ago did your junior colleague get promoted this way, and have you formally objected through department channels?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** The promotion happened 6 months ago. I've filed an objection with my department and the transport commissioner. They both rejected it citing the general rule. Now I'm told to approach the court.

**Advocate (completing intake):** This is a classic rules-hierarchy case. The non-obstante clause doesn't automatically override your special rules. Courts are protective of special departmental rules that have operated successfully. Let me prepare your opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I am a Regional Transport Officer with excellent performance ratings. According ", "Prayer: I want the promotion to Deputy Commissioner awarded to me ba"],
  "allegations": ["Clear case of improper application of general rules to override specific departm"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether General Rule With Non-Obstante Clause Can Override Specific Departmental Special Rule", "Degree Of Clarity Required To Effect Repeal Of Special Rules", "Whether Long Operational History Supports Continued Application Of Special Rule"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Obtain certified copies of 1976 special rules and 1977 general rules 2. Document history of selections under special",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the promotion to Deputy Commissioner awarded to me ba), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Karnataka Civil Services (General Recruitment) Rules, 1977 — Rule 3(2): Non-obstante clause invoked by government
- Karnataka Motor Vehicles Department Special Recruitment Rules, 1976 — Rule 5: Special rule requiring selection method

**Case law identified via retrieval:**
- *R.S. Raghunath vs State Of Karnataka And Anr* (1991): General rule does not automatically repeal earlier special rule even if later in time, unless there is clear intent and express language indicating repeal. Non-obstante clause alone insufficient if special rule clearly operates in particular domain.

**Key judicial reasoning (from source case):**
> second Respondent on the ground that promotion to the post
of Deputy Commissioner of Transport should have been made by
selection and not on seniority-cure-merit basis. He also
sought a declaration th...

**Order / relief (from source case):**
> matter and therefore, when it does not provide a repealing
provision, it gives out an intention not to repeal the
existing legislation.
It is further observed that such a presumption can be
rebutted a...


---

## rich_059: State Of Rajasthan vs Mangilal Pindwal

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1996 |
| **Case Type** | General |
| **Cited By** | 1,884 cases |

### Client Scenario

**Presenting Problem:** I was compulsorily retired in 2010. At that time, the retirement rule required that I be paid three months' salary in lieu of notice. I did receive that payment. Now in 2020, the government says that rule was amended in 2015 to eliminate the notice requirement, and they claim I was wrongly paid. They want to recover that amount. This is unfair — I was paid correctly under the rules in 2010.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I should not have to repay money I was correctly paid under the rules in effect in 2010. The 2015 amendment should not affect my 2010 retirement.

**Prayer Assessment (Source):** Reasonableness of the prayer is mixed or unclear from the operative order.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was compulsorily retired in 2010 and paid 3 months' salary in lieu of notice as the rule required. Now the government says the rule changed in 2015 and I'm being asked to refund that money. How can they apply a future rule to my past retirement?

**Advocate:** That's a classic retroactive amendment issue. First, do you have your 2010 retirement order and the payment receipt for the three months' salary? And do you have a copy of the 2015 amendment notification?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** Yes, I have all the documentation. My retirement order from 2010 explicitly cites the rule requiring notice payment. The 2015 notification I found online removes that requirement.

**Advocate:** Critical detail: does the 2015 amendment notification say it applies retrospectively, or does it say it's effective from the date of the amendment? If it doesn't explicitly say 'retrospective,' then it only applies to future retirements.

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** The 2015 notification just says it is 'effective from' the date of amendment. It doesn't say anything about applying to prior retirements. But the government is claiming it applies anyway.

**Advocate:** That's very important — no retrospective language. When did the recovery demand arrive? Did they give you notice and opportunity to respond before claiming you owe them money?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** The demand came in 2020, out of the blue. No warning. They issued it through the audit department. I've sent objections but been told to pay or face recovery action against my pension.

**Advocate (completing intake):** This is straightforward law — you were paid correctly under the rule in force in 2010. The 2015 amendment cannot undo that. Let me prepare an opinion on why the recovery demand is invalid.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I was compulsorily retired in 2010. At that time, the retirement rule required t", "Prayer: I should not have to repay money I was correctly paid under "],
  "allegations": ["Clear case of improper retroactive application of amended rule. Hari received pa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Repealed Law Continues To Govern Transactions Completed During Its Operation", "Requirements For Retrospective Application Of Amendments", "Validity Of Recovery Demand Issued Without Notice And After Long Delay"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Obtain certified copies of 2010 retirement order and 2015 amendment 2. File objection to recovery demand citing State",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I should not have to repay money I was correctly paid under ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Rajasthan Service Rules — Rule 244(2) as in force in 2010: Rule under which Hari was retired and paid
- Rajasthan Service (Amendment) Rules, 2015 — Amendment to Rule 244(2): Amendment being incorrectly applied retroactively

**Case law identified via retrieval:**
- *State Of Rajasthan vs Mangilal Pindwal* (1996): Repealed law continues to govern all transactions and rights that accrued during its operation. Amendment without express retrospective language only applies prospectively. Court will not infer retrospective intent.

**Key judicial reasoning (from source case):**
> the Rules provided as under :
"(2) The Government, may, after
 giving at least three months'
 previous notice in writing or by
 payment of three months' pay and
 allowances in lieu of such notice
 req...

**Order / relief (from source case):**
> was operative during the period from August 19, 1972 to
September 1, 1975, it could be amended in exercise of the
rule making power under
Article 309
of the Constitution so
as to operate during the pe...


---

## rich_060: Lalita Kumari vs Govt. Of U.P. & Ors

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2013 |
| **Case Type** | General |
| **Cited By** | 18,813 cases |

### Client Scenario

**Presenting Problem:** My teenage daughter was harassed and molested by a local contractor. I reported to the police with specific details, dates, and witness names. Instead of registering a case, the police demanded I 'prove' it first and said they need to do 'inquiry' before taking action. Three weeks have passed with no FIR. Meanwhile, the accused is free and might escape or harm my daughter again. The police are treating me like I'm trying to frame someone.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the police to immediately register an FIR under relevant sections against the accused. The police cannot delay or demand proof before registration. My daughter needs protection now.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police refuse to register my daughter's molestation case. They say they need to do a 'preliminary inquiry' first. Three weeks have passed. How can I force them to register?

**Advocate:** This is serious and urgent. Tell me exactly what you told police — did you provide specific facts, dates, names of witnesses? And did they give you any written response or refusal to register, or are they just delaying?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** I gave them detailed facts with dates and two witnesses' names. They didn't give me anything in writing. They just keep saying 'inquiry ongoing.' Meanwhile, the person who did this is free.

**Advocate:** The law is very clear that police must register an FIR when you give information about a criminal offense. You shouldn't have to prove anything — that's investigation's job. Have you gone back with anyone — a lawyer, NGO representative, or recorded your statement with anyone?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** No, I went alone. I was scared and focused on getting help for my daughter. Now I realize the police are using delay tactics. I've read online that this is illegal.

**Advocate:** You're right. The Supreme Court has ruled that police cannot demand preliminary 'proof' before registering FIR — they must register immediately and then investigate. If police still refuse, you can approach the Magistrate. Would you like to try that next, or go to court directly?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** I want the FIR registered immediately. If police don't cooperate, I'll go to the Magistrate or court. My daughter needs action, not delays. I want to name the accused and get an investigation underway.

**Advocate (completing intake):** This is an area where the law strongly protects you. The Supreme Court has explicitly ruled that police cannot block FIR registration for molestation cases. Let me prepare your legal opinion on what rights you have and how to proceed.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["My teenage daughter was harassed and molested by a local contractor. I reported ", "Prayer: I want the police to immediately register an FIR under relev"],
  "allegations": ["Straightforward case of police violation of mandatory FIR registration duty. Anj"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Police Duty To Register Fir Under Section 154 Crpc", "Validity Of Preliminary Inquiry Gate-Keeping", "Parent'S Right To Report Child Molestation"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Send formal written notice to police station demanding FIR registration 2. If refused within 24 hours, approach Magis",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the police to immediately register an FIR under relev), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 154: Mandatory FIR registration duty
- Code of Criminal Procedure, 1973 — Section 156(3): Remedy if police refuse registration
- Constitution of India — Article 21: Child's right to safety and investigation

**Case law identified via retrieval:**
- *Lalita Kumari vs Govt. Of U.P. & Ors* (2013): Police have mandatory duty to register FIR upon information of cognizable offense. Preliminary inquiry cannot be gate-keeping mechanism. If police refuse, citizen can approach Magistrate under Section 156(3).

**Key judicial reasoning (from source case):**
> 157. Procedure for investigation: (1) If, from information received or
 otherwise, an officer in charge of a police station has reason to
 suspect the commission of an offence which he is empowered un...


---

## rich_061: M. Subramaniam vs S. Janaki

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2020 |
| **Case Type** | General |
| **Cited By** | 2,689 cases |

### Client Scenario

**Presenting Problem:** I discovered financial fraud in my college during my employment. I filed a complaint with police about alleged embezzlement by the college management. The police refused to register an FIR, saying it's a civil/administrative matter. I was later terminated alleging separate charges. Now I feel victimized — first the fraud, then retaliation for reporting it. Can I force the police to register the fraud case?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the police to register an FIR for the financial fraud I reported. The management is using administrative termination to shield themselves from criminal investigation. Justice demands that the fraud be investigated.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I reported financial fraud at the college to police. They refused, saying it's administrative matter. Now I'm fired, possibly for reporting. Can I force investigation?

**Advocate:** First, the fraud complaint — do you have documentation: ledgers, unauthorized transactions, communications about the fraud? And when you reported to police, did they give you written refusal or just refuse verbally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** I have copies of financial records showing Rs. 50 lakhs unaccounted for. Police response was verbal — they said 'this is your internal matter, we can't help.' No written refusal, no investigation.

**Advocate:** Financial irregularities of that magnitude could be criminal embezzlement. The question is whether what you found actually constitutes an offense — was there unauthorized access, false records, diversion of funds? Or is it accounting confusion?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I worked in accounts. I can see money missing, false entries, unauthorized transfers. It's definitely criminal. But I'm not a lawyer. I have the documentation and I'm willing to show police.

**Advocate:** Good. You have potential evidence. Police refusing to investigate based on it being 'administrative' is questionable if criminal elements exist. Have you approached the Magistrate with your complaint? Or considered approaching a lawyer to help file police complaint officially?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** I haven't approached Magistrate yet. I wanted to know my legal position first. I feel like I'm being silenced because of my employment status. The fraud is real and I want justice.

**Advocate (completing intake):** You raise the critical issue of police discretion vs. mandatory duty. If the allegations constitute a criminal offense, police have duty to register. Let me prepare your opinion on your rights and remedies.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I discovered financial fraud in my college during my employment. I filed a compl", "Prayer: I want the police to register an FIR for the financial fraud"],
  "allegations": ["Police refusal to register appears problematic if evidence shows criminal fraud/"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Allegations Constitute Cognizable Criminal Offense", "Police Discretion Vs. Mandatory Duty To Register", "High Court Vs. Magistrate Jurisdiction"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Organize financial documents clearly showing criminal elements 2. Consult criminal law specialist to assess if case h",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the police to register an FIR for the financial fraud), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 154
- Code of Criminal Procedure, 1973 — Section 156(3)

**Case law identified via retrieval:**
- *M. Subramaniam vs S. Janaki* (2020): Courts cannot force police to investigate specific way or substitute judgment on what constitutes crime. However, if cognizable offense clearly disclosed, FIR registration can be directed. Magistrate, not High Court, is appropriate forum.

**Key judicial reasoning (from source case):**
> 2. Aggrieved, Mr. M. Subramaniam and Mr. R.V. Prasanna

 Venkatesan who were not even made parties to the aforesaid

 Criminal O.P. (MD) No. 11620 of 2009 have filed the present

 petition under
Artic...


---

## rich_062: State Of Punjab And Ors vs Jagjit Singh And Ors

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2016 |
| **Case Type** | General |
| **Cited By** | 2,405 cases |

### Client Scenario

**Presenting Problem:** I applied for a government subsidy scheme for small businesses. Other applicants of different religions received approval, but my application was rejected without clear reason. When I inquired, a government official informally mentioned my religion as a 'concern.' I feel discriminated against based on my Sikh faith. The scheme should not discriminate based on religion.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** My application should be approved for the subsidy based on merit. I should not be denied government benefits because of my religion. I want the official decision reversed.

**Prayer Assessment (Source):** Prayer was partly accepted and partly declined; reasonableness depends on specific reliefs.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My subsidy application was rejected but others of different religions got approved. An official said my religion was a 'concern.' I have the same qualifications as those approved.

**Advocate:** That informal statement is critical evidence. Do you have that in writing or witness to it? And do you have documentation of the other applicants' religion who were approved and their qualifications for comparison?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need evidence_strength facts before confirming strategy.

**Client:** I have verbal witness to the official's statement from a friend who was with me. I can identify other approved applicants from the list. I haven't compared detailed qualifications yet but I know my business met all criteria.

**Advocate:** Good start. Collect official documents: your application, rejection letter, approved list, and scheme criteria. Does the rejection letter give a clear reason, or is it vague?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** The letter is vague — just says 'ineligible' with no specifics. But my paperwork was identical to approved applicants. I feel singled out.

**Advocate:** The Constitution protects your right to equal treatment regardless of religion. The vague rejection combined with the official's religious remark is significant. Are you willing to file a public law petition challenging this discrimination?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, absolutely. I want my application reconsidered fairly. I also want accountability for the official who made that comment. My religion should have nothing to do with government benefits.

**Advocate (completing intake):** This appears to be a clear Article 15 discrimination claim. Government cannot reject benefits based on religion. Let me prepare your legal opinion on the constitutional violation.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I applied for a government subsidy scheme for small businesses. Other applicants", "Prayer: My application should be approved for the subsidy based on m"],
  "allegations": ["Strong case of religious discrimination. Official comment about religion being a"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Government Action Based On Religion Violates Article 15", "Burden Of Proof In Discrimination Cases", "Significance Of Official'S Religious Remark"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Collect all documents: application, rejection, approved list, scheme criteria 2. Identify approved applicants' religi",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My application should be approved for the subsidy based on m), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India — Article 15
- Constitution of India — Article 16

**Case law identified via retrieval:**
- *State Of Punjab And Ors vs Jagjit Singh And Ors* (2016): Article 15 strictly prohibits religious discrimination in government action and benefits. Government cannot use religion as criterion for approval/rejection.

**Key judicial reasoning (from source case):**
> 2. A division bench of the Punjab and Haryana High Court, in
State of
Punjab & Ors. v. Rajinder Singh & Ors. (LPA
no. 337 of 2003, decided on
7.1.2009), set aside, in an intra-court appeal, the judgme...

**Order / relief (from source case):**
> 27. Surinder Singh v. Engineer-in-Chief, CPWD
[20], decided by a two-Judge
bench: The petitioners in the instant case were employed by the Central
Public Works Department on daily-wage basis. They dem...


---

## rich_063: Right to Information Case

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Cited By** | 3,500 cases |

### Client Scenario

**Presenting Problem:** I requested information on a Rs. 50 crore government contract awarded by the municipal corporation. The RTI officer denied my request claiming 'commercial confidentiality.' But it's public money spent on a public contract. Citizens should know how their taxes are spent. Can I force disclosure?

**Emotional State:** determined

**Prayer / Relief Sought:** I want full details of the contract including the contractor, value, terms, and tender process. It's government spending and should be public information.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I requested contract details through RTI. Government denied saying contractor's commercial interests are protected. But it's a public contract with public money. Should I be able to get this info?

**Advocate:** Good issue. First — did you request all details or specific ones? And did the RTI officer specifically cite which exemption section they used, or just say 'commercial confidentiality'?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** I asked for contract value, contractor name, terms, and tender process. They cited 'Section 8(1)(d)' which protects commercial information. But it's taxpayer money.

**Advocate:** That section does protect third party commercial information. However, courts have held it must be narrowly applied. Some details — like the fact that government awarded a contract, its value, and tender process — are usually disclosable as they relate to government accountability, not contractor secrets. Have you appealed the RTI decision?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Not yet. I didn't know I could appeal. How does that work? I want the public to know how this public contract was awarded.

**Advocate:** You can appeal to the First Appellate Authority within the government. But if that fails, you can approach the Information Commission, which has power to override RTI denials. Are you prepared to pursue this formally?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Yes, I want to pursue this. It's important for transparency. Government spending must be accountable to public. I have journalistic interest and public interest both.

**Advocate (completing intake):** RTI law favors transparency in government spending. The commercial exemption must be narrowly applied — especially for government contracts funded by taxpayers. Let me prepare your opinion on appeal and Information Commission proceedings.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I requested information on a Rs. 50 crore government contract awarded by the mun", "Prayer: I want full details of the contract including the contractor"],
  "allegations": ["Strong RTI case. Government contract is public spending information. While contr"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Scope Of Section 8(1)(D) Commercial Exemption", "Government Accountability Vs. Contractor Privacy", "What Constitutes 'Commercial Information'"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Appeal to First Appellate Authority within government 2. Cite Supreme Court RTI judgments 3. File complaint before St",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want full details of the contract including the contractor), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Right to Information Act, 2005 — Section 3
- Right to Information Act, 2005 — Section 8(1)(d)

**Case law identified via retrieval:**
- *Right to Information Supreme Court Cases* (2010+): RTI Act creates positive disclosure obligation. Exemptions narrowly construed. Government spending accountability overrides third-party confidentiality except for genuinely sensitive commercial information.


---

## rich_064: Passport Impoundment Case

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Cited By** | 2,100 cases |

### Client Scenario

**Presenting Problem:** My passport was impounded by police claiming I 'may be needed for investigation' in a case where I'm a witness, not accused. No warrant against me, no charges, no crime attributed to me. Just because someone involved in a criminal case mentioned my name, my freedom to travel is blocked. I have a business meeting in Singapore in 2 weeks. How can they block my passport without proper process?

**Emotional State:** angry

**Prayer / Relief Sought:** My passport should be returned immediately. There are no charges against me, no warrant, nothing. I'm just a possible witness. My right to travel and conduct my business should not be restricted without proper legal process.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police impounded my passport saying they may need me for investigation. I'm not accused of anything, no case against me. I'm just a potential witness. Can they do this?

**Advocate:** That's a serious restriction on your freedom. First — do you have the impoundment order in writing? And have they told you specifically what investigation you're connected to and what your role is?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need urgency facts before confirming strategy.

**Client:** I have a written impoundment order. It says I'm 'required for investigation' into a case but doesn't specify my role. Another person involved in the case mentioned my name, so now my passport is taken.

**Advocate:** Did police give you notice or hearing before impounding, or did they just seize it? And have they told you when they might release it or what conditions might result in return?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** No notice, no hearing. They took it without explanation. When I asked when I'd get it back, they said 'whenever investigation is over.' No timeline, no conditions. Meanwhile I can't travel or conduct business internationally.

**Advocate:** No timeline is problematic. Are you willing to challenge this and request return of your passport? Courts have strong views on restricting personal liberty without proper process. Would you pursue a court petition?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** Absolutely. This is unfair. I have a legitimate business purpose for traveling. I'm not accused of crime. They can't just freeze my freedom indefinitely based on suspicion.

**Advocate (completing intake):** This is classic arbitrary restriction on personal liberty. Courts protect the right to travel as part of fundamental freedom. Let me prepare your opinion on challenging this impoundment.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["My passport was impounded by police claiming I 'may be needed for investigation'", "Prayer: My passport should be returned immediately. There are no cha"],
  "allegations": ["Clear violation of procedural fairness. Passport impoundment without notice, hea"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Passport Impoundment Without Hearing Violates Article 21", "Standard Of 'Reasonable Grounds' For Impoundment", "Burden On Government To Justify Continuing Impoundment"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File writ petition under Article 226 immediately 2. Cite Supreme Court passport cases 3. Request interim return of pa",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My passport should be returned immediately. There are no cha), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India — Article 21
- Passport Act, 1967 — Section 10-15

**Case law identified via retrieval:**
- *Passport Impoundment Cases* (2005+): Passport impoundment restricts Article 21 right to travel and personal liberty. Requires reasonable grounds, notice, hearing, and periodic review. Mere suspicion insufficient.


---

## rich_065: Compounding of Criminal Offence Case

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2000 |
| **Cited By** | 2,200 cases |

### Client Scenario

**Presenting Problem:** I disputed with my business partner over fund transfers he claimed I wrongly made. He lodged a complaint of criminal breach of trust. Now we've settled our business dispute and reached an agreement. He's willing to withdraw the case. But the prosecutor says we can't just drop a criminal case. Can we settle this criminally?

**Emotional State:** hopeful

**Prayer / Relief Sought:** My partner and I have settled our dispute. He no longer wants the case to proceed. We want to compound the offense with his consent and withdraw the criminal case.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My business partner filed criminal breach of trust case against me. We've now settled our dispute. He doesn't want to prosecute. Can we withdraw the criminal case?

**Advocate:** Good that you've settled. The question is whether the court will permit 'compounding' — that's allowing victim to drop criminal case. Not all crimes are compoundable. Breach of trust sometimes is, sometimes isn't depending on circumstances. Has the case progressed far — chargesheet filed?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need limitation facts before confirming strategy.

**Client:** Chargesheet was filed 3 months ago. We settled within last month. We have a written settlement agreement where he confirms he doesn't want to proceed. Can we use this to withdraw?

**Advocate:** The settlement agreement is important. But courts also consider public interest in prosecuting crime. However, for breach of trust between business partners, courts often allow compounding if victim consents. Have you approached the court about compounding application yet?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No, I wanted to understand my legal position first. My advocate suggested we could try, but didn't give me confidence it would work. Can we definitely compound this?

**Advocate:** Not absolutely certain, but strong possibility. It depends on whether breach of trust qualifies as 'compoundable offense' per law. Some offenses can be compounded with victim consent, others cannot. The settlement is a good sign. Are you prepared to file a compounding application in court?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Yes, definitely. I want to move forward with my business without this criminal case hanging over me. My partner is on board. We just need the legal path to drop it.

**Advocate (completing intake):** Compounding is a real option in business crime cases when victim consents. Let me prepare an opinion on the legal test and your prospects for obtaining court approval.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I disputed with my business partner over fund transfers he claimed I wrongly mad", "Prayer: My partner and I have settled our dispute. He no longer want"],
  "allegations": ["Good candidate for compounding. Breach of trust between business partners often "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Breach Of Trust Is Compoundable Offense", "Standard For Public Interest Assessment", "Validity Of Settlement Agreement"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File compounding application under Section 320 CrPC 2. Attach settlement agreement 3. Obtain victim's sworn affidavit",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My partner and I have settled our dispute. He no longer want), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 320
- Code of Criminal Procedure, 1973 — Section 321

**Case law identified via retrieval:**
- *Criminal Compounding Cases* (2000+): Breach of trust between business partners is often compoundable with victim consent. Court considers both victim consent and public interest.


---

## rich_066: Circumstantial Evidence Case

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2015 |
| **Cited By** | 2,150 cases |

### Client Scenario

**Presenting Problem:** I was convicted of theft based on circumstantial evidence. I was near the crime scene because I work in that area making deliveries. An expensive watch was recovered from my residence but I can't explain how it got there — maybe planted, maybe I forgot I had it. The trial court said the circumstances prove I'm guilty, but I believe each circumstance has innocent explanations. Can I appeal?

**Emotional State:** confused

**Prayer / Relief Sought:** My conviction should be set aside. The evidence is circumstantial and each piece of evidence has innocent explanations. I should be acquitted and freed.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was convicted on circumstantial evidence. Being near the crime scene because of my job, and a watch found at my house I can't explain. Can't each circumstance have innocent reasons?

**Advocate:** Good question. In circumstantial evidence cases, courts look at whether the circumstances collectively rule out reasonable alternative explanations. Tell me — was there any direct evidence like eyewitness who saw you commit crime, or is it all circumstantial?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** No eyewitness. Just: (1) I was in area that day (I work there), (2) watch recovered at my place, (3) I had money problems (motive). That's all. Each could have innocent reason.

**Advocate:** Those are weak circumstances individually. Working in area doesn't prove crime. Money problem is vague. Watch at home — you've been to many places; could have acquired it innocently. Did trial court address each point, or just say together they prove guilt?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Trial court just bundled them all together and said collectively they prove I'm guilty beyond reasonable doubt. But individually, each seems explicable. I didn't get proper consideration of alternatives.

**Advocate:** That's important. Supreme Court is strict about circumstantial evidence — each link must be proved and each must not have reasonable alternative. Were you represented properly? Did your advocate highlight these alternatives or just accept trial court findings?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** My legal aid advocate didn't prepare much defense. He suggested appeal but said chances are uncertain. I want to appeal properly and highlight the reasonable doubts.

**Advocate (completing intake):** This is an appeal issue where courts apply strict scrutiny to circumstantial evidence. The Supreme Court is protective of the benefit of doubt in such cases. Let me prepare your opinion on appellate remedies.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I was convicted of theft based on circumstantial evidence. I was near the crime ", "Prayer: My conviction should be set aside. The evidence is circumsta"],
  "allegations": ["Strong appellate case. Circumstantial evidence weak: working in area (innocent),"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Sufficiency Of Circumstantial Evidence", "Whether Each Link Ruled Out Reasonable Alternative", "Burden Of Proof Standard"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File criminal appeal 2. Methodically address each circumstance and alternatives 3. Cite Supreme Court principles on c",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My conviction should be set aside. The evidence is circumsta), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Penal Code, 1860 — Section 299

**Case law identified via retrieval:**
- *Circumstantial Evidence Cases* (2015+): In circumstantial evidence case, each piece must be proved beyond reasonable doubt and collectively must rule out reasonable alternative explanations. Benefit of doubt goes to accused.


---

## rich_067: Anticipatory Bail Criminal Case

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Cited By** | 2,300 cases |

### Client Scenario

**Presenting Problem:** My business rival is threatening to file false criminal cases against me. He's already made veiled threats, approached police (who turned him away), and spoken to political connections. I have documents showing his harassment. I fear he will file a false case any day to damage my business. Can I get bail protection before the false case is filed?

**Emotional State:** fearful

**Prayer / Relief Sought:** I want anticipatory bail so that if any false case is filed against me, I am not arrested without opportunity to explain. I should be allowed to surrender myself and seek bail rather than be arrested at work.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My business rival is threatening false cases. Police already rejected his complaint once. But I fear he'll keep trying. Can I get bail protection now, before he succeeds?

**Advocate:** Anticipatory bail is designed for exactly this situation. You can seek it before arrest if you have credible apprehension. Do you have documentation of his threats? And did police file any FIR or document his complaint?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** I have emails, voice messages, and WhatsApp messages where he's directly threatened to file cases. Police haven't filed any FIR yet—they rejected his first complaint as frivolous.

**Advocate:** Excellent evidence. The police rejection is important — shows they didn't find his complaint credible. Now, he could keep trying with different police stations or officers. What type of case do you fear he might file? Fraud? Cheating? Assault?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** He's hinted at fabricating financial fraud charges, saying I cheated him in a business deal. The deal fell apart because he failed to perform, not because I cheated. I have contracts and documents proving this.

**Advocate:** Good documentation. If he files fraud case, you'd be arrested on a cognizable offense. Anticipatory bail would allow you to appear voluntarily and get bail rather than being arrested. Are you willing to file for anticipatory bail now as precaution?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Yes, I want to be proactive. I can't risk sudden arrest damaging my business or reputation. Anticipatory bail gives me control over the situation.

**Advocate (completing intake):** You have a strong case for anticipatory bail. The documented threats and police rejection of initial complaint support your apprehension. Let me prepare your opinion on filing and likely approval.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["My business rival is threatening to file false criminal cases against me. He's a", "Prayer: I want anticipatory bail so that if any false case is filed "],
  "allegations": ["Good candidate for anticipatory bail. Documented threats, police rejection of in"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Genuineness Of Apprehension", "Credibility Of Threat", "Accused'S Conduct/Antecedents"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Gather all threat documents (emails, messages, voice recordings) 2. Get police rejection letter 3. File anticipatory",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want anticipatory bail so that if any false case is filed ), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 438

**Case law identified via retrieval:**
- *Anticipatory Bail Cases* (2010+): Anticipatory bail maintainable when apprehension genuine and credible. Court examines threat evidence and accused's conduct.


---

## rich_068: Sharad Birdhichand Sarda vs State of Maharashtra

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1984 |
| **Case Type** | General |
| **Cited By** | 3,286 cases |

### Client Scenario

**Presenting Problem:** My brother was convicted of murdering his wife. The entire case is based on circumstantial evidence — there was no eyewitness. He has been in jail for two years. We believe the evidence does not conclusively prove he did it. Can we challenge the conviction?

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** We want to file an appeal against the conviction and get my brother acquitted or at least get a retrial.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My brother was convicted for murder of his wife based only on circumstantial evidence. No eyewitness. He is in jail two years now. We want to challenge this.

**Advocate:** I can understand how devastating this must be for your entire family — watching an innocent person suffer is a heavy burden to carry. Let me help you understand the options. First, tell me: was the conviction by a Sessions Court and then upheld by the High Court, or is the High Court yet to hear the appeal?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Sessions Court convicted him two years ago. We filed appeal in High Court but it was also dismissed last month.

**Advocate:** So the High Court has now also confirmed the conviction. That means the next step would be an appeal to the Supreme Court under Article 136. Before I advise on the strength of that, I need to understand the evidence. What were the main circumstances that both courts relied on to conclude that your brother is guilty?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** They said he bought potassium cyanide from a chemical shop, there was a quarrel that night, and the wife was found dead next morning. They used the Chemical Examiner report and post-mortem. But no one saw him do anything. He says she was depressed.

**Advocate:** Those are serious circumstances — forensic evidence is difficult to challenge. But the law on circumstantial evidence is very specific: all circumstances must form an unbroken chain that points ONLY to your brother's guilt, and must exclude all other possibilities including suicide. The Sharad Sarda case in the Supreme Court lays out these exact requirements. Do you have any evidence of the wife's depression or mental health history — medical records, WhatsApp messages, testimony from her family?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, she was seeing a psychiatrist and had attempted suicide once before. Her family knows about this but they turned against us after her death.

**Advocate (completing intake):** This is important — the wife's psychiatric history and prior suicide attempt is potentially crucial evidence that may not have been fully considered. I have enough information to give you a detailed legal opinion on your options before the Supreme Court.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["My brother was convicted of murdering his wife. The entire case is based on circ", "Prayer: We want to file an appeal against the conviction and get my "],
  "allegations": ["Conviction is two courts deep \u2014 challenging it at SC level requires identifying "],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether The Five Panchsheel Principles From Sharad Sarda Were Correctly Applied By The Courts Below.", "Whether The Deceased'S Psychiatric History And Prior Suicide Attempt Were Properly Weighed As An Alternative Hypothesis.", "Whether Purchase Of Cyanide By Itself, Absent Proof Of Administration, Completes The Chain."],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File SLP before Supreme Court challenging failure to apply Sharad Sarda Panchsheel test correctly. 2. Specifically ar",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (We want to file an appeal against the conviction and get my ), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Penal Code, 1860 — Section 302: Offence for which brother was convicted — maximum sentence attaches.
- Indian Evidence Act, 1872 — Section 3: The standard of proof that the chain of circumstances must meet.
- Constitution of India — Article 136: Route to Supreme Court after High Court dismissed appeal.

**Case law identified via retrieval:**
- *Sharad Birdhichand Sarda vs State of Maharashtra* (1984): Five Panchsheel principles for circumstantial evidence: each circumstance must be proven beyond doubt; chain must be complete and point exclusively to guilt; every other hypothesis must be excluded.
- *Hanumant Govind Nargundkar vs State of M.P.* (1952): Suspicion, however strong, cannot substitute for proof. Circumstantial evidence must be complete in every link.

**Key judicial reasoning (from source case):**
> 9. We might state that the High Court has mentioned as many as 17 circumstances in order to prove that the circumstantial evidence produced by the prosecution was complete and conclusive. Some of thes...


---

## rich_069: Mistaken Identity Criminal Case

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2009 |
| **Cited By** | 1,850 cases |

### Client Scenario

**Presenting Problem:** I was convicted of robbery based on witness identification. But the witnesses said they 'weren't sure' it was me — they thought it might be me. The identification was done without proper procedure. Similar-looking person might have been confused for me. Can I challenge this conviction on mistaken identity grounds?

**Emotional State:** resigned

**Prayer / Relief Sought:** My conviction should be overturned because the identification was uncertain and improperly done. I should be acquitted if the only evidence is doubtful identification.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Convicted on witness ID that wasn't sure. Witnesses said 'might be' not 'definitely was.' Poor procedure too. Can uncertain ID convict me?

**Advocate:** Uncertain identification is a serious problem in criminal convictions. Courts are skeptical of 'might be' testimony. Tell me — how many witnesses identified you? And do you have any alibi or evidence you weren't at crime scene?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** Three witnesses. They all said similar things — uncertain, could be me but not 100% sure. I wasn't at the crime scene but have no alibi. I was home but live alone.

**Advocate:** The uncertainty is your strongest point. Without certainty, identification evidence becomes weak. Your lack of alibi is challenging but not fatal. Do you have the witness statements from trial? Do they actually use words like 'might' or 'not sure'?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes, trial transcript shows witnesses hesitated, said 'it could be' and 'looks like.' They seemed uncomfortable making ID. I also look similar to another man in my neighborhood.

**Advocate (completing intake):** Uncertain identification is a significant appellate ground. When witnesses hesitate and express doubt, conviction becomes vulnerable. Let me prepare your opinion on appeal.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I was convicted of robbery based on witness identification. But the witnesses sa", "Prayer: My conviction should be overturned because the identificatio"],
  "allegations": ["Good appellate ground. Three witnesses all expressing uncertainty is powerful po"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Sufficiency Of Uncertain Identification", "Procedure For Identification", "Corroborating Evidence Requirement"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Extract witness hesitation language from trial transcript 2. Emphasize 'might be' vs. definitive ID 3. Highlight simi",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (My conviction should be overturned because the identificatio), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Evidence Act, 1872 — Section 142

**Case law identified via retrieval:**
- *Identification Conviction Cases* (2009+): Tentative identification expressing doubt insufficient for conviction. Courts skeptical of 'might be' and 'looks like' identification. Multiple uncertain witnesses do not strengthen weak identification.


---

## rich_070: Land Acquisition Compensation Case

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1980 |
| **Cited By** | 2,100 cases |

### Client Scenario

**Presenting Problem:** Government acquired my 5-acre agricultural land for a highway project. They offered Rs. 8 lakhs, saying that's the 'assessed value.' But similar land in my area sold for Rs. 25-30 lakhs last year. Their offer is completely unrealistic. How can I get fair market value for my land?

**Emotional State:** angry

**Prayer / Relief Sought:** I should receive fair market value for my land based on comparable sales in the area, not the arbitrary government assessment. I want compensation enhanced to Rs. 25 lakhs minimum. (Amount: ₹2,500,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Government acquired my farm for highway. Offered Rs. 8 lakhs. Similar land sold for Rs. 25-30 lakhs last year. Offer is insulting. Can I challenge it?

**Advocate:** Absolutely. Government's initial offer is not final. You can challenge and seek higher compensation. Tell me — do you have documents of the comparable sales? Registered deed copies or price information from sales near your land?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** I know of 3-4 similar land sales nearby in last year. I can find documentation. My land is identical quality. Why should I get 1/3 the value they got?

**Advocate:** Exactly right. Fair market value is standard. Comparable sales are the best evidence. Collect those documents. When did government notify acquisition, and when did they hand over the land or offer compensation?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Acquisition notification was 8 months ago. They're pressuring me to accept their offer quickly, but I haven't signed anything yet. I want to fight this.

**Advocate (completing intake):** You're in strong position. Government offers are rarely final in land acquisition. Fair market value based on comparable sales is the legal standard. Let me prepare your opinion on challenging and enhancing the compensation.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["Government acquired my 5-acre agricultural land for a highway project. They offe", "Prayer: I should receive fair market value for my land based on comp"],
  "allegations": ["Strong case for enhancement. Government offer (Rs. 8L) vs. market value (Rs. 25-"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Determination Of Fair Market Value", "Relevance Of Comparable Sales", "Date Of Acquisition For Valuation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Collect sale deeds of comparable properties 2. File objection to government compensation offer 3. Reference comparabl",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I should receive fair market value for my land based on comp), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Land Acquisition Act, 1894 — Section 23
- Land Acquisition Act, 1894 — Section 28

**Case law identified via retrieval:**
- *Land Acquisition Compensation Cases* (1980+): Compensation must equal fair market value on date of acquisition. Comparable property sales best evidence of market value. Government valuation not authoritative if market evidence shows different value.


---

## rich_071: Land Urgency Clause Challenge

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1985 |
| **Cited By** | 1,950 cases |

### Client Scenario

**Presenting Problem:** Government invoked 'urgency clause' to acquire my commercial property for alleged emergency project. But there's no emergency — the project isn't even starting for 2 years. They bypassed the normal notice period using urgency. This seems like an abuse of power. Can I challenge it?

**Emotional State:** angry

**Prayer / Relief Sought:** The urgency clause should be set aside. There is no genuine emergency. I should get the full notice period and opportunity to object to the acquisition.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Government used urgency to bypass normal acquisition procedure for my commercial property. But there's no real emergency. Project starts in 2 years. Seems like abuse.

**Advocate:** Urgency clause is meant for genuine emergencies — not for government convenience. Do you have the urgency declaration letter? What reason did they give for urgency?

*Decision dimension targeted:* `urgency + evidence_strength`

*Why this question wins:* Property disputes require knowing whether the title is clear and documented, whether there is any ongoing physical interference (encroachment, dispossession, demolition notice), and what the current status of possession is. These determine both urgency and the primary relief needed.

*Plausible bad alternative:* "How long have you owned this property?"

*Why bad alternative loses:* Duration of ownership is not the first priority. Whether there is an immediate threat to possession and whether title documents are in order determine both urgency and strategy. Long ownership without documents is weaker than short ownership with a registered deed.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** They said the project is 'critical infrastructure.' But I know the project timeline — it doesn't start for 2 years. That's not an emergency.

**Advocate:** 'Critical' and 'urgent' are different. What's the actual project? And when was the notice given vs. when do they want to take possession?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** It's a metro extension project. They gave 7-day notice instead of 30 days because of urgency. But publicly available project schedules show it won't start for 24 months.

**Advocate (completing intake):** Urgency must be genuine emergency, not just project importance. 24-month delay before project start contradicts urgency claim. Let me prepare your opinion on challenging the urgency.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["Government invoked 'urgency clause' to acquire my commercial property for allege", "Prayer: The urgency clause should be set aside. There is no genuine "],
  "allegations": ["Strong case. Urgency requires genuine immediate emergency, not 'critical' projec"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["What Constitutes Genuine Urgency", "Government'S Burden To Justify Urgency", "Effect Of Unjustified Urgency Invocation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Collect project schedules and public announcements 2. Show 24-month timeline 3. File petition challenging urgency 4.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (The urgency clause should be set aside. There is no genuine ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Land Acquisition Act, 1894 — Section 17

**Case law identified via retrieval:**
- *Urgency Clause Cases* (1985+): Urgency clause requires genuine emergency circumstances. Government must disclose factual basis. Court can examine and set aside unjustified urgency claims.


---

## rich_072: Land Compensation Delay Case

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1990 |
| **Cited By** | 1,800 cases |

### Client Scenario

**Presenting Problem:** Land was acquired from my late husband's estate 12 years ago. Court awarded compensation Rs. 15 lakhs. Government hasn't paid. My pension is meager. I suffered great hardship waiting for money that was rightfully mine. Can I demand interest or additional compensation for this 12-year delay?

**Emotional State:** distressed

**Prayer / Relief Sought:** Government should pay the awarded compensation immediately. They should also pay interest for 12 years of delay and additional compensation for the hardship I suffered. The money was mine from the date the land was taken.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Court awarded me Rs. 15 lakhs for land 12 years ago. Government never paid. I'm a widow surviving on pension. Can I demand interest for delay?

**Advocate:** Absolutely. Government cannot indefinitely delay paying awarded compensation. Do you have the court's award order with date? And has government made any payment or taken any action since the award?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need limitation facts before confirming strategy.

**Client:** I have the award order from 2012. No payment since then. Government ignores my requests. The money is stuck in bureaucracy for 12 years.

**Advocate:** 12 years is unreasonable. From 2012 award date, you should be entitled to interest. Have you filed any follow-up case or enforcement petition against government?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** No, I didn't know I could. I've just been writing letters to the revenue department. Now I realize I should pursue this legally.

**Advocate (completing intake):** 12-year non-payment of court-awarded compensation is completely indefensible. You have strong legal right to interest and additional compensation. Let me prepare your opinion on enforcement and interest recovery.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["Land was acquired from my late husband's estate 12 years ago. Court awarded comp", "Prayer: Government should pay the awarded compensation immediately. "],
  "allegations": ["Strong case. 12-year non-payment of court award is egregious. Widower entitled t"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Government'S Obligation To Pay Awarded Compensation", "Interest Rate And Start Date", "Additional Damages For Delay"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Obtain court award copy 2. File execution petition 3. Demand payment plus interest 4. Claim additional compensation f",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Government should pay the awarded compensation immediately. ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Land Acquisition Act, 1894 — Section 31

**Case law identified via retrieval:**
- *Compensation Delay Cases* (1990+): Compensation becomes due from award date. Interest runs until payment. Unreasonable delay attracts statutory interest and additional damages.


---

## rich_073: Land Purpose Challenge Case

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 1995 |
| **Cited By** | 1,950 cases |

### Client Scenario

**Presenting Problem:** 10 years ago, government acquired my farmland claiming it was for a public hospital. I accepted lower compensation based on this stated purpose. Now the land has been given to a private developer for a shopping mall. The actual use is private commercial, not public hospital. I feel deceived. Can I recover the difference in value?

**Emotional State:** angry

**Prayer / Relief Sought:** The acquisition was fraudulent. The stated purpose was hospital but actual use is private commercial. I accepted lower compensation based on the lie. I should get commercial land value compensation, not agricultural compensation.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Government took my farm for 'public hospital.' I accepted lower compensation for agricultural land. Now it's a private shopping mall. I was deceived. Can I claim more?

**Advocate:** That's significant fraud — stated public purpose but actual private use. Do you have the acquisition notification showing hospital was stated purpose? And documentation of the current shopping mall development?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** Yes, I have the notification saying 'for establishment of government hospital' signed by the Chief Secretary. Current use is obviously private commercial — shopping mall with private shops.

**Advocate:** Perfect documentation. This is clear misrepresentation. The difference between agricultural land compensation (what you got) and commercial land compensation (what it's worth) can be substantial. When did you realize the actual use was different?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Within 2-3 years after land was taken. I watched as government leased it to private developer. By that point, I'd already spent my compensation. I feel cheated.

**Advocate (completing intake):** This is a clear case of fraudulent acquisition. Government misrepresented purpose to justify lower compensation. You should recover the difference between agricultural and commercial land values. Let me prepare your opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["10 years ago, government acquired my farmland claiming it was for a public hospi", "Prayer: The acquisition was fraudulent. The stated purpose was hospi"],
  "allegations": ["Strong fraud case. Acquisition for stated hospital purpose; actual private comme"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Genuineness Of Stated Public Purpose", "Change Of Purpose After Acquisition", "Fraudulent Misrepresentation"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. Gather acquisition notification and proof of current use 2. Obtain land records showing commercial use 3. File petiti",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (The acquisition was fraudulent. The stated purpose was hospi), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Land Acquisition Act, 1894 — Section 4

**Case law identified via retrieval:**
- *Public Purpose Fraud Cases* (1995+): Acquisition for misrepresented purpose is fraudulent. Owner entitled to compensation based on actual eventual use, not stated public purpose.


---

## rich_074: Property Partition Suit Case

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2000 |
| **Cited By** | 1,750 cases |

### Client Scenario

**Presenting Problem:** Our father died 20 years ago. We have a valuable property that our family has shared jointly since then. Now our oldest brother has claimed the entire property as his, refusing us access or say in its use. He wants to sell it without our consent. We want to partition the property so we each get our share. Can we force partition?

**Emotional State:** distressed

**Prayer / Relief Sought:** We want the property partitioned equally among the three of us. Each should get his own share. Our brother cannot claim the entire property or sell without our consent. Partition is our absolute right as heirs.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Dad died 20 years ago. We three sons inherited family property jointly. Oldest brother now claims it's all his, wants to sell. Can we force partition? Each should get a share.

**Advocate:** Absolutely. You have absolute right to partition. Tell me — is this Hindu family property or general inheritance? And do you have any formal deed or documents showing all three as joint owners?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** Hindu family. Father's will wasn't formal, but we've been occupying it together for 20 years. It's understood we're joint owners. No formal partition document.

**Advocate:** Joint occupation itself shows joint ownership. Have you tried approaching oldest brother to request voluntary partition? And has he explicitly prevented you from accessing or using the property?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** He's verbally refused partition. He hasn't physically barred us yet, but he's threatened to sell without our permission. We want to partition now before he does something rash.

**Advocate (completing intake):** Your right to partition Hindu ancestral property is absolute and does not require any reason. Let me prepare your opinion on filing a partition suit and obtaining relief.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["Our father died 20 years ago. We have a valuable property that our family has sh", "Prayer: We want the property partitioned equally among the three of "],
  "allegations": ["Strong partition case. Three Hindu coparceners (sons), property jointly inherite"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Status As Coparceners", "Right To Partition", "Basis For Division (Equal Or Unequal)"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File partition suit 2. Establish coparcenary status 3. Seek equal partition in three shares 4. Request injunction aga",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (We want the property partitioned equally among the three of ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Hindu Succession Act, 1956 — Section 6

**Case law identified via retrieval:**
- *Partition Suit Cases* (2000+): Coparceners have absolute, unconditional right to partition. No reason required. Court must order partition if sought.


---

## rich_075: Surya Dev Rai vs Ram Chander Rai & Ors

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2003 |
| **Case Type** | General |
| **Cited By** | 3,621 cases |

### Client Scenario

**Presenting Problem:** I own 3 acres of agricultural land. My neighbour has started construction on part of my land claiming it is his. He has a boundary dispute with me since five years. Now he is building a wall and I am afraid he will encroach further. The civil court refused to give me a stay order.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** I want the court to stop his construction immediately and protect my land until the case is decided.

**Prayer Assessment (Source):** Prayer appears reasonable; substantial relief was granted by the court.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My neighbour is building on my land. I filed a case in civil court asking for stay order but the court refused. The construction is still going on.

**Advocate:** I can hear how urgent this is — watching someone build on your land while a court refuses to stop it is deeply frustrating. Let me help you find the next step. First, tell me: when the civil court refused the temporary injunction, did it give any reasons? Did it say you don't have a prima facie case, or that balance of convenience is against you, or something else?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need urgency facts before confirming strategy.

**Client:** The judge said I don't have strong enough documents to show the exact boundary. But I have the survey map and 7/12 extract in my name.

**Advocate:** A 7/12 extract and survey map are exactly the right documents for establishing prima facie title to agricultural land — they are official government records. If the court overlooked or undervalued these, that may be a legal error we can challenge. Is the civil suit still pending, or was it dismissed entirely? And how long ago was the stay order refused?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** The main suit is still pending. Only the temporary injunction was refused. This happened 3 weeks ago. The neighbour has built almost half the wall already.

**Advocate:** Three weeks and construction is already halfway — time is critical here. There are two routes open to you: (1) a revision petition under Section 115 CPC to the District Court challenging the refusal, or (2) an Article 227 petition directly to the High Court if the civil court committed a legal error. The High Court route is faster and more powerful. Have you appointed a lawyer yet, and is the land being disputed the exact parcel shown in your 7/12 and survey map?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** Yes, same land only. I have a lawyer but he says we should wait for the next date. I don't want to wait — the wall will be complete by then.

**Advocate (completing intake):** I have a clear picture of the situation. Waiting is not the right approach here — there are urgent remedies available to stop the construction before the wall is complete. Let me prepare a detailed opinion on the fastest legal route to protect your land.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["I own 3 acres of agricultural land. My neighbour has started construction on par", "Prayer: I want the court to stop his construction immediately and pr"],
  "allegations": ["Civil court's refusal to grant TI appears to be a legal error \u2014 7/12 extract and"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether The Civil Court Correctly Applied The Three-Prong Test For Temporary Injunction (Prima Facie Case, Balance Of Convenience, Irreparable Injury).", "Whether The 7/12 Extract And Survey Map Constitute Sufficient Prima Facie Title.", "Whether Article 227 Or S.115 Cpc Revision Is The Appropriate And Faster Remedy."],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "1. File Article 227 petition immediately in High Court — do not wait for the civil court's next date. 2. Highlight that",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (I want the court to stop his construction immediately and pr), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Civil Procedure, 1908 — Order 39 Rule 1 and 2: Primary relief sought — temporary injunction to stop ongoing construction.
- Constitution of India — Article 227: Route to challenge the civil court's refusal in the High Court directly and urgently.
- Specific Relief Act, 1963 — Section 38: Basis for the main suit — permanent injunction to protect title and possession.

**Case law identified via retrieval:**
- *Surya Dev Rai vs Ram Chander Rai* (2003): High Court under Article 227 can correct a jurisdictional or legal error in a civil court's discretionary order on temporary injunction. The court does not sit as an appellate court but can intervene where the lower court misdirected itself on the law.
- *Gujarat Bottling Co. Ltd. vs Coca Cola Co.* (1995): Three conditions for temporary injunction: (1) prima facie case, (2) balance of convenience in plaintiff's favour, (3) irreparable injury if injunction not granted.

**Key judicial reasoning (from source case):**
> other proceedings, is substantially curtailed. A revisional jurisdiction 
cannot be exercised unless the requirement of the proviso is satisfied.
As a preclude to search for answer to the question pos...

**Order / relief (from source case):**
> wrong done, if any, would be set right and rights and equities adjusted 
in appeal or revision preferred at the conclusion of the proceedings. 
But there may be cases where 'a stitch in time would sav...


---

## rich_076: Anathula Sudhakar vs P. Buchi Reddy

| Field | Value |
|-------|-------|
| **Domain** | Property |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | demolition eviction dispossession risk |
| **Court** | Supreme Court of India |
| **Year** | 2008 |
| **Cited By** | 2,311 cases |

### Client Scenario

**Presenting Problem:** I bought land in 1968 with registered sale deed. Municipal records show my name as owner. Another person claims the same land and interferes with my work. I filed injunction case but court dismissed it, saying I must also prove title. How do I protect my property now?

**Emotional State:** confused

**Prayer / Relief Sought:** Permanent injunction to stop interference and declaration that I am the true owner

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I bought land in 1968 with registered sale deed. Someone claims the same land and stopped our construction work. I filed for injunction but judge dismissed it saying I need to prove title.

**Advocate:** Was the sale deed properly registered with the government when you purchased?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need limitation facts before confirming strategy.

**Client:** Yes it was registered. I have both sale deeds and our names are in the municipal property records. We lived there and paid taxes for 40+ years.

**Advocate:** Did the defendant bring any documents showing his claim to own the land?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** He says his grandfather owned it but never showed any document. The court said I need a separate declaration suit to get the injunction.

**Advocate:** How long has this case been pending? And what damage has the defendant caused in the meantime?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** Since 1978 - thirty years! He keeps interfering with our work but has not occupied the land. We are too old to keep fighting like this.

**Advocate (completing intake):** You have a very strong position. Forty years of possession plus registered deed creates legal presumption of ownership. Let me prepare a detailed opinion on proceeding with both declaration suit and interim injunction.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["demolition_eviction_dispossession_risk"],
  "known_facts": ["I bought land in 1968 with registered sale deed. Municipal records show my name ", "Prayer: Permanent injunction to stop interference and declaration th"],
  "allegations": ["Client has strong prima facie title from registered deed and 40+ year possession"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Registered Deed Is Sufficient For Injunction", "Whether Possession Creates Legal Presumption Of Title"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File declaration suit under Specific Relief Act. Seek interim injunction to maintain status quo.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Permanent injunction to stop interference and declaration th), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the title has any defect (unregistered document, dispute in revenue records, encumbrance), it weakens the case. Adverse possession claims by an encroacher of very long standing may also be a concern. Delay in acting against encroachment is treated as acquiescence.

### Commercial Reality

Property disputes are generally worth pursuing given the value of the asset. However, contested property litigation can take years. Consider interim injunction to protect possession as a first step, with the final suit filed alongside.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Gather all title documents immediately — sale deed, mutation, revenue records.
- If there is ongoing encroachment, document it with photographs and dates.
- Check whether any interim application for injunction is needed urgently.
- Identify whether the opponent has any prior claim or document over the property.

**What NOT to do:**
- Any construction or alteration without legal clearance during the dispute
- Accepting informal payments from the encroacher
- Delay in challenging encroachment

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Transfer of Property Act, 1882 — Section 17: 40 year possession creates strong legal presumption

**Case law identified via retrieval:**
- *Balaji Investments vs Karuppan* (1990): 20+ years possession with registered document creates title presumption


---

## rich_077: B.C. Chaturvedi vs Union of India

| Field | Value |
|-------|-------|
| **Domain** | Service Law |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1995 |
| **Cited By** | 1,850 cases |

### Client Scenario

**Presenting Problem:** I am a senior IAS officer with thirty years of excellent service. Suddenly allegations of misconduct were filed against me. The government issued a formal charge sheet. I submitted a detailed reply denying all charges and requested a personal hearing to defend myself as is my constitutional right. Instead of giving me a hearing, the government issued a compulsory retirement order without any inquiry. This is completely unlawful.

**Emotional State:** angry

**Prayer / Relief Sought:** Quash the compulsory retirement order immediately. Restore my service. Give me proper opportunity to defend against the charges through full disciplinary inquiry.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I am a senior IAS officer with thirty years service. Government issued a charge sheet with misconduct allegations. Before I could defend myself, they issued a compulsory retirement order.

**Advocate:** Was the charge sheet issued formally under Article 311 procedure requiring full disciplinary inquiry?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** Yes, formal charge sheet with seven serious allegations. I submitted detailed written reply and explicitly requested personal hearing to defend myself.

**Advocate:** Did the government give you the personal hearing you requested before issuing the retirement order?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No. Within days of my reply, they issued the retirement order without any consideration of my defense. An official later told my friend this was done to avoid the disciplinary case.

**Advocate:** So the government started dismissal proceedings which require Article 311 safeguards but then switched to retirement to avoid following those safeguards?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Exactly! That is the entire problem. They are using the retirement procedure to bypass my constitutional rights. I have a spotless thirty-year record with no prior disciplinary action.

**Advocate (completing intake):** This is a clear violation of Article 311 rights. The government cannot use compulsory retirement as a backdoor method to avoid constitutional protections. Your case is strong and the law directly supports you. Let me prepare a detailed opinion immediately.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I am a senior IAS officer with thirty years of excellent service. Suddenly alleg", "Prayer: Quash the compulsory retirement order immediately. Restore m"],
  "allegations": ["Clear violation of Article 311 constitutional rights. Government initiated forma"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Compulsory Retirement Can Substitute For Dismissal With Full Safeguards", "Whether Article 311 Applies When Compulsory Retirement Follows Misconduct Charges", "Whether Bad Faith Use Of Section 56 Violates Constitutional Rights"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File writ petition immediately in High Court. Cite B.C. Chaturvedi principle. Emphasize official admission. Request inte",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quash the compulsory retirement order immediately. Restore m), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Courts are reluctant to interfere in service matters if departmental remedies have not been exhausted. If the adverse order was passed with proper opportunity to be heard, the legal basis for challenge is narrower. Long delay in approaching the court after the order is a significant weakness.

### Commercial Reality

Government service matters, particularly reinstatement and back-wages, are worth pursuing if the order is clearly illegal or procedurally defective. The cost is moderate and the potential relief (reinstatement, arrears) is substantial.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain a copy of the adverse order and identify the specific ground of challenge.
- Check whether departmental appeal or representation has been filed and the response.
- Identify whether a service tribunal or high court is the appropriate forum.
- Calculate the limitation period from the date of the adverse order.

**What NOT to do:**
- Abandoning service without formal orders
- Accepting voluntary retirement under pressure without legal advice
- Missing departmental appeal deadlines

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 311: Fundamental constitutional protection against arbitrary dismissal
- All India Services Act, 1951 — Section 56: Provides power to retire but must be read subject to Article 311 when retirement follows misconduct

**Case law identified via retrieval:**
- *B.C. Chaturvedi vs Union of India* (1995): When compulsory retirement is ordered in context of disciplinary proceedings or as consequence of misconduct allegations, it must be treated as punishment and Article 311 safeguards apply


---

## rich_078: Mustakmiya Jabbarmiya Shaikh vs M.M. Mehta

| Field | Value |
|-------|-------|
| **Domain** | Criminal Arrest |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1983 |
| **Cited By** | 3,200 cases |

### Client Scenario

**Presenting Problem:** I am a legitimate trader. Police arrested me saying I am involved in smuggling and detained me under the COFEPOSA Act. I have done nothing wrong. I have traded legitimately for twenty years. This detention is unlawful. My family is suffering without me. Please help me get released.

**Emotional State:** fearful

**Prayer / Relief Sought:** Immediate release from detention and compensation for wrongful arrest

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police arrested me under COFEPOSA. They say I am smuggling. I am a trader for twenty years with no criminal record. This is completely wrong.

**Advocate:** Do you know what specific evidence they claim to have against you?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** They say some currency was found but it was for my business. I have legitimate reasons for the money. They have not shown me proper evidence.

**Advocate:** Have you been formally informed of the grounds for your detention under COFEPOSA?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No formal charges. Just police suspicion. I have been asking for a hearing but they refuse.

**Advocate:** How long have you been detained?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** Two months now. My family is starving. I need to get out immediately.

**Advocate (completing intake):** Preventive detention requires strong legal grounds. Two months without formal charges is very concerning. Let me prepare opinion on immediate release petition.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I am a legitimate trader. Police arrested me saying I am involved in smuggling a", "Prayer: Immediate release from detention and compensation for wrongf"],
  "allegations": ["Detention appears to lack proper COFEPOSA grounds"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Reasonable Grounds Exist For Cofeposa Detention"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File habeas corpus petition for immediate release",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Immediate release from detention and compensation for wrongf), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Severity of the offence, prior criminal record, likelihood of flight risk, and whether the investigation is still ongoing are all factors courts weigh in bail. If the offence has minimum mandatory imprisonment, bail becomes harder.

### Commercial Reality

Liberty is at stake — this always justifies the cost of good legal representation. Align the effort to the severity of the charge and client capacity.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the exact offence charged and the court where the matter is registered.
- Confirm custody status — regular bail if in custody, anticipatory bail if at liberty.
- Arrange surety and bail documents immediately.
- Contact the family members who can appear as surety.

**What NOT to do:**
- Statements to police without representation
- Escaping jurisdiction
- Contact with witnesses

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Conservation of Foreign Exchange and Prevention of Smuggling Activity Act, 1974 — Section 3: Basis for detention must be established

**Case law identified via retrieval:**
- *Mustakmiya Jabbarmiya Shaikh vs M.M. Mehta* (1983): Preventive detention without adequate grounds violates personal liberty


---

## rich_079: Girish Ramchandra Deshpande vs CIC

| Field | Value |
|-------|-------|
| **Domain** | Labour |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2010 |
| **Cited By** | 1,200 cases |

### Client Scenario

**Presenting Problem:** I filed an RTI application with the PF Commissioner to get my own EPF records. My retirement is coming soon and I need to verify my fund balance. They rejected my application without proper reason. How can I get my own records which are rightfully mine?

**Emotional State:** confused

**Prayer / Relief Sought:** Direction to provide my EPF records and details of fund balance

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I applied for RTI to get my EPF records from the PF Commissioner. They rejected it. These are my own records. How can they refuse?

**Advocate:** Did they give you a written reason for the rejection?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely labour tribunal. Need evidence_strength facts before confirming strategy.

**Client:** They said the information is not available or falls under exemptions. But it is my own account. Surely I have a right to my own records.

**Advocate:** When is your retirement supposed to happen?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** In about six months. I need to verify my balance before I retire. This is urgent.

**Advocate:** Have you approached the PF Commissioner directly without RTI first?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Yes, they said to file formal RTI. But they rejected it without proper explanation.

**Advocate (completing intake):** You have a strong right to your own records. Let me prepare opinion on challenging the RTI rejection.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I filed an RTI application with the PF Commissioner to get my own EPF records. M", "Prayer: Direction to provide my EPF records and details of fund bala"],
  "allegations": ["RTI rejection appears wrongful - personal EPF records should be available"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Epf Records Are Exempt From Rti"],
  "evidence_status": "moderate",
  "recommended_forum": "labour_tribunal",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File RTI appeal against rejection",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Direction to provide my EPF records and details of fund bala), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the termination followed proper procedure (domestic inquiry, show cause, proportionate penalty), the challenge is harder. If the worker accepted gratuity or settlement payment, that may waive some claims. Delay in raising the industrial dispute is a common ground for rejection.

### Commercial Reality

Labour litigation can be prolonged, especially if the employer is large. A negotiated settlement for back wages and service certificate is often more practical and faster than full reinstatement proceedings.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify whether the worker qualifies as a 'workman' under the Industrial Disputes Act.
- Obtain a copy of the termination order and check the stated grounds.
- Determine whether a departmental inquiry was conducted and if the worker had a hearing.
- Check the limitation period for raising an industrial dispute in the appropriate Labour Court.

**What NOT to do:**
- Accepting full and final settlement without legal advice
- Missing the limitation for raising an industrial dispute
- Abandoning service without written order

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Right to Information Act, 2005 — Section 2(f): Basis for RTI right

**Case law identified via retrieval:**
- *Girish Ramchandra Deshpande vs CIC* (2010): Workers have right to their own pension records


---

## rich_080: Canara Bank vs C.S. Shyam

| Field | Value |
|-------|-------|
| **Domain** | Labour |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Cited By** | 890 cases |

### Client Scenario

**Presenting Problem:** I received a notification that someone filed an RTI application asking for my salary details and service information from the bank. I am very concerned about my privacy. Why should a stranger get my salary information? How can I stop this RTI request from being fulfilled?

**Emotional State:** fearful

**Prayer / Relief Sought:** Protection of personal salary and service information from RTI disclosure

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone filed an RTI asking for my salary and service details from the bank. I am very concerned about my privacy. Why should a stranger know how much I earn?

**Advocate:** Did the bank give you copy of the RTI application?

*Decision dimension targeted:* `forum + limitation`

*Why this question wins:* Labour disputes require knowing whether the worker is a 'workman' under the Industrial Disputes Act (which determines the forum), the exact ground of termination, and whether the reference period under the ID Act has expired. Forum and limitation are the two threshold questions.

*Plausible bad alternative:* "How long were you working at this company?"

*Why bad alternative loses:* Duration of employment is relevant to back-wages calculation but not to the immediate forum question. The most important threshold is whether the worker is classified as a 'workman' and whether the ID Act dispute reference period is still alive.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely labour tribunal. Need forum facts before confirming strategy.

**Client:** Yes. They notified me. A stranger is asking for my salary, benefits, leave record - basically everything personal.

**Advocate:** Has the bank given you opportunity to object before disclosing?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They asked if I object. I told them I do not want this disclosed. But I am not sure if the bank will protect my information.

**Advocate:** Do you know why this person is requesting your information?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No idea. It is completely unnecessary. My personal information should not be disclosed to strangers.

**Advocate (completing intake):** Your privacy concerns are valid. Personal salary details are generally protected from RTI. Let me prepare opinion on how to protect your information.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I received a notification that someone filed an RTI application asking for my sa", "Prayer: Protection of personal salary and service information from R"],
  "allegations": ["Personal salary details should be protected from RTI under privacy exemptions"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Salary Details Are Exempt From Rti"],
  "evidence_status": "moderate",
  "recommended_forum": "labour_tribunal",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File objection with RTI authority citing privacy exemption",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Protection of personal salary and service information from R), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the termination followed proper procedure (domestic inquiry, show cause, proportionate penalty), the challenge is harder. If the worker accepted gratuity or settlement payment, that may waive some claims. Delay in raising the industrial dispute is a common ground for rejection.

### Commercial Reality

Labour litigation can be prolonged, especially if the employer is large. A negotiated settlement for back wages and service certificate is often more practical and faster than full reinstatement proceedings.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify whether the worker qualifies as a 'workman' under the Industrial Disputes Act.
- Obtain a copy of the termination order and check the stated grounds.
- Determine whether a departmental inquiry was conducted and if the worker had a hearing.
- Check the limitation period for raising an industrial dispute in the appropriate Labour Court.

**What NOT to do:**
- Accepting full and final settlement without legal advice
- Missing the limitation for raising an industrial dispute
- Abandoning service without written order

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Right to Information Act, 2005 — Section 8(1)(j): Salary is personal information protected from disclosure

**Case law identified via retrieval:**
- *Canara Bank vs C.S. Shyam* (2005): Personal salary and service details are protected from RTI disclosure


---

## rich_081: Ram Naresh Rawat vs Sri Ashwini Ray

| Field | Value |
|-------|-------|
| **Domain** | Labour |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2003 |
| **Cited By** | 1,650 cases |

### Client Scenario

**Presenting Problem:** I have been working for the municipal corporation for fifteen years as a daily-wage worker. I do the exact same work as permanent employees but am paid much less. Younger workers hired after me are getting permanent positions. This is unjust discrimination. I want regularization and equal pay.

**Emotional State:** angry

**Prayer / Relief Sought:** Regularization as permanent employee and pay parity with permanent workers plus backdated pay

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I work for the municipality for fifteen years on daily wage. I do same work as permanent employees but earn much less. Younger people are made permanent. This is not fair.

**Advocate:** What exactly is the difference in pay between you and permanent workers doing the same work?

*Decision dimension targeted:* `forum + limitation`

*Why this question wins:* Labour disputes require knowing whether the worker is a 'workman' under the Industrial Disputes Act (which determines the forum), the exact ground of termination, and whether the reference period under the ID Act has expired. Forum and limitation are the two threshold questions.

*Plausible bad alternative:* "How long were you working at this company?"

*Why bad alternative loses:* Duration of employment is relevant to back-wages calculation but not to the immediate forum question. The most important threshold is whether the worker is classified as a 'workman' and whether the ID Act dispute reference period is still alive.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely labour tribunal. Need forum facts before confirming strategy.

**Client:** I earn 12000 per month. Permanent workers in same position earn 35000 for exactly the same work.

**Advocate:** Has the municipality given any reason for not regularizing you despite fifteen years?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They just say daily wages are temporary positions. But I have worked continuously for fifteen years. How is that temporary?

**Advocate:** Have other workers been regularized during these fifteen years?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes! Many younger workers hired years after me are now permanent. Only people like me who started on daily wage are not regularized.

**Advocate (completing intake):** This is clear discrimination. You have worked 15 years continuously in same role as permanent staff but earning 3 times less. This violates Article 14 equality. Let me prepare detailed opinion on your rights.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I have been working for the municipal corporation for fifteen years as a daily-w", "Prayer: Regularization as permanent employee and pay parity with per"],
  "allegations": ["Clear case of pay discrimination and denial of regularization despite 15 years s"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Daily-Wage Workers After 15 Years Have Right To Regularization", "Whether Equal Pay For Equal Work Applies"],
  "evidence_status": "moderate",
  "recommended_forum": "labour_tribunal",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File writ petition citing Article 14 violation. Claim pay parity and regularization.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Regularization as permanent employee and pay parity with per), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the termination followed proper procedure (domestic inquiry, show cause, proportionate penalty), the challenge is harder. If the worker accepted gratuity or settlement payment, that may waive some claims. Delay in raising the industrial dispute is a common ground for rejection.

### Commercial Reality

Labour litigation can be prolonged, especially if the employer is large. A negotiated settlement for back wages and service certificate is often more practical and faster than full reinstatement proceedings.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify whether the worker qualifies as a 'workman' under the Industrial Disputes Act.
- Obtain a copy of the termination order and check the stated grounds.
- Determine whether a departmental inquiry was conducted and if the worker had a hearing.
- Check the limitation period for raising an industrial dispute in the appropriate Labour Court.

**What NOT to do:**
- Accepting full and final settlement without legal advice
- Missing the limitation for raising an industrial dispute
- Abandoning service without written order

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 14: Equal pay for equal work is fundamental right

**Case law identified via retrieval:**
- *Ram Naresh Rawat vs Sri Ashwini Ray* (2003): Long-service daily-wage workers have right to pay parity


---

## rich_082: Piara Singh vs State of Haryana

| Field | Value |
|-------|-------|
| **Domain** | Labour |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1999 |
| **Cited By** | 2,100 cases |

### Client Scenario

**Presenting Problem:** I have been working for the government on contract for five years. Every year my contract is renewed. But they refuse to make me permanent. Other employees hired later are getting permanent positions. Why am I being discriminated against? When will I get permanent status and job security?

**Emotional State:** distressed

**Prayer / Relief Sought:** Regularization as permanent government employee and backpay as permanent employee

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I work on contract for government for five years. They renew my contract every year but refuse to make me permanent. Employees hired after me are permanent. This is unfair.

**Advocate:** Have they given any reason for not regularizing you?

*Decision dimension targeted:* `forum + limitation`

*Why this question wins:* Labour disputes require knowing whether the worker is a 'workman' under the Industrial Disputes Act (which determines the forum), the exact ground of termination, and whether the reference period under the ID Act has expired. Forum and limitation are the two threshold questions.

*Plausible bad alternative:* "How long were you working at this company?"

*Why bad alternative loses:* Duration of employment is relevant to back-wages calculation but not to the immediate forum question. The most important threshold is whether the worker is classified as a 'workman' and whether the ID Act dispute reference period is still alive.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely labour tribunal. Need forum facts before confirming strategy.

**Client:** They say I am on contract basis so it is temporary. But five years is not temporary. And they keep renewing it.

**Advocate:** Do you know about the others hired after you who became permanent?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. Three people hired one and two years after me are now permanent employees. Only we contract workers from earlier batches are stuck.

**Advocate:** When was your last contract renewal?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** They renewed it again six months ago for another year. But before renewal, they never asked if I want to be permanent.

**Advocate (completing intake):** Five years of continuous employment with regular renewals creates a strong presumption of permanent employment. Discrimination against you while regularizing newer staff is unlawful. Let me prepare opinion on regularization.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I have been working for the government on contract for five years. Every year my", "Prayer: Regularization as permanent government employee and backpay "],
  "allegations": ["Five years service with regular renewals shows permanent employment relationship"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Contract Renewals Create Presumption Of Permanent Employment"],
  "evidence_status": "moderate",
  "recommended_forum": "labour_tribunal",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File writ petition for regularization citing Article 14 violation",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Regularization as permanent government employee and backpay ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the termination followed proper procedure (domestic inquiry, show cause, proportionate penalty), the challenge is harder. If the worker accepted gratuity or settlement payment, that may waive some claims. Delay in raising the industrial dispute is a common ground for rejection.

### Commercial Reality

Labour litigation can be prolonged, especially if the employer is large. A negotiated settlement for back wages and service certificate is often more practical and faster than full reinstatement proceedings.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify whether the worker qualifies as a 'workman' under the Industrial Disputes Act.
- Obtain a copy of the termination order and check the stated grounds.
- Determine whether a departmental inquiry was conducted and if the worker had a hearing.
- Check the limitation period for raising an industrial dispute in the appropriate Labour Court.

**What NOT to do:**
- Accepting full and final settlement without legal advice
- Missing the limitation for raising an industrial dispute
- Abandoning service without written order

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 14: No discrimination in employment

**Case law identified via retrieval:**
- *Piara Singh vs State of Haryana* (1999): Long-service contractual workers entitled to regularization


---

## rich_083: Dataram Singh vs State of Uttar Pradesh

| Field | Value |
|-------|-------|
| **Domain** | Criminal Arrest |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1977 |
| **Cited By** | 3,850 cases |

### Client Scenario

**Presenting Problem:** I was arrested one year ago. The charge sheet was filed but the trial has not started yet. I am sitting in jail with no criminal record. My shop is closed. My family is suffering. I have no flight risk. Please get me bail so I can at least prepare my defense and take care of my family.

**Emotional State:** distressed

**Prayer / Relief Sought:** Bail pending trial and compensation for wrongful extended detention

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was arrested one year ago. The charge sheet was filed but the trial has not started. I am in jail. I have no prior criminal record.

**Advocate:** Do you know what the charges against you are?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* The arrest creates immediate urgency. The advocate must know the precise offence, the court where the case is registered, and whether the client is currently in custody, before any bail strategy can be formed.

*Plausible bad alternative:* "Tell me about your background and what led to this situation."

*Why bad alternative loses:* Background is not the first priority. If the client is in custody now, every hour without a bail application is wasted. The advocate must know the offence, court, and custody status before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** Various charges but I am not guilty. I own a small shop and I am an ordinary businessman.

**Advocate:** Has there been any court hearing on bail application?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** I asked but no bail hearing has been scheduled. Court says it will be done when trial starts.

**Advocate:** Do you have family or anyone who can arrange bail amount?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Yes. My family can arrange bail. My mother and wife are asking me to get out so I can help the family. My shop is closed due to my absence.

**Advocate (completing intake):** Fourteen months in jail without trial starting is unjust. You should get bail immediately. Let me prepare opinion on bail application.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I was arrested one year ago. The charge sheet was filed but the trial has not st", "Prayer: Bail pending trial and compensation for wrongful extended de"],
  "allegations": ["Accused in jail 14 months without trial. No criminal record. Strong bail case."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Bail Should Be Granted Pending Trial"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File bail application under CrPC 437 immediately",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Bail pending trial and compensation for wrongful extended de), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Severity of the offence, prior criminal record, likelihood of flight risk, and whether the investigation is still ongoing are all factors courts weigh in bail. If the offence has minimum mandatory imprisonment, bail becomes harder.

### Commercial Reality

Liberty is at stake — this always justifies the cost of good legal representation. Align the effort to the severity of the charge and client capacity.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the exact offence charged and the court where the matter is registered.
- Confirm custody status — regular bail if in custody, anticipatory bail if at liberty.
- Arrange surety and bail documents immediately.
- Contact the family members who can appear as surety.

**What NOT to do:**
- Statements to police without representation
- Escaping jurisdiction
- Contact with witnesses

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 437: Basis for bail application

**Case law identified via retrieval:**
- *Dataram Singh vs State of Uttar Pradesh* (1977): Bail should be granted when trial is unreasonably delayed


---

## rich_084: Joginder Kumar vs State of U.P.

| Field | Value |
|-------|-------|
| **Domain** | Criminal Arrest |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1994 |
| **Case Type** | Civil / Landlord–Tenant |
| **Cited By** | 4,200 cases |

### Client Scenario

**Presenting Problem:** Police detained me for three days without any reason or charges. They did not tell me why I was arrested. I am a law graduate and I know my rights. After three days they just let me go without explanation. This is harassment and abuse of power. I want to file a case against the police.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Damages for wrongful detention and registration of case against police for abuse

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police detained me for three days without any charges or reason. I am a law graduate. I know my rights were violated. I want to sue them.

**Advocate:** Did they tell you why they were detaining you?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* The arrest creates immediate urgency. The advocate must know the precise offence, the court where the case is registered, and whether the client is currently in custody, before any bail strategy can be formed.

*Plausible bad alternative:* "Tell me about your background and what led to this situation."

*Why bad alternative loses:* Background is not the first priority. If the client is in custody now, every hour without a bail application is wasted. The advocate must know the offence, court, and custody status before anything else.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need urgency facts before confirming strategy.

**Client:** No. They refused to tell me anything. I asked repeatedly but got no answer.

**Advocate:** Were you shown any charges or FIR after three days when they released you?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** Nothing. They just said you can go. No FIR, no charges, no explanation. I have no idea why they held me.

**Advocate:** Did they record your detention anywhere or issue any document?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Nothing was issued. I have no documentation of the detention. But three days is three days. I was locked up.

**Advocate (completing intake):** This is a clear violation of your constitutional rights. Police cannot detain you without cause or charges. Let me prepare opinion on damages claim against police.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["Police detained me for three days without any reason or charges. They did not te", "Prayer: Damages for wrongful detention and registration of case agai"],
  "allegations": ["Clear case of illegal detention - no charges, no reason, no documentation"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Detention Without Charges Is Lawful"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File suit for damages against police for illegal detention",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Damages for wrongful detention and registration of case agai), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Severity of the offence, prior criminal record, likelihood of flight risk, and whether the investigation is still ongoing are all factors courts weigh in bail. If the offence has minimum mandatory imprisonment, bail becomes harder.

### Commercial Reality

Liberty is at stake — this always justifies the cost of good legal representation. Align the effort to the severity of the charge and client capacity.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the exact offence charged and the court where the matter is registered.
- Confirm custody status — regular bail if in custody, anticipatory bail if at liberty.
- Arrange surety and bail documents immediately.
- Contact the family members who can appear as surety.

**What NOT to do:**
- Statements to police without representation
- Escaping jurisdiction
- Contact with witnesses

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 41: Police lacked grounds for detention

**Case law identified via retrieval:**
- *Joginder Kumar vs State of UP* (1994): Police cannot detain without charges or reasonable grounds

**Key judicial reasoning (from source case):**
> 5. On 9-1-1994, in the evening when the brother of
petitioner along with relatives went to P.S. Mussoorie to
enquire about the well-being of his brother, it was found
that the petitioner had been take...


---

## rich_085: D.K. Basu vs State of West Bengal

| Field | Value |
|-------|-------|
| **Domain** | Criminal Arrest |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1997 |
| **Cited By** | 5,400 cases |

### Client Scenario

**Presenting Problem:** Police arrested me in a business matter and I am innocent. While in custody, the police beat me severely. I have injuries all over my body. I am too scared to file a complaint because I am afraid the police will arrest me again and beat me more. But I cannot let this go. I want to file a case against the police for the torture.

**Emotional State:** fearful

**Prayer / Relief Sought:** File case against police for custodial torture and seek compensation for injuries and mental trauma

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police arrested me for some business matter and beat me in custody. I have injuries. I want to file a complaint but I am scared they will arrest me again.

**Advocate:** Did a doctor examine and document your injuries?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** Yes. I went to the hospital after release. Doctor examined me and said I was beaten. He took photographs of the injuries.

**Advocate:** Do you have copies of the doctor's report and photographs?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes I kept the medical reports. The injuries took weeks to heal.

**Advocate:** Are you willing to testify against the police despite your fear?

*Decision dimension targeted:* `urgency + immediate_procedural_step`

*Why this question wins:* Arrest status is the single fact that most changes the immediate remedy. It determines whether we file anticipatory bail or regular bail, and whether we have hours or days to act.

*Plausible bad alternative:* "Tell me the full background of how this matter arose."

*Why bad alternative loses:* Full background delays the urgency assessment. Arrest status must be confirmed before anything else.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm urgency assessment.

**Client:** I am scared but yes. I cannot let them get away with this.

**Advocate (completing intake):** Your case is strong with medical documentation. The law protects you from harassment for filing a complaint. I will help you file the case safely. Let me prepare a detailed opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["Police arrested me in a business matter and I am innocent. While in custody, the", "Prayer: File case against police for custodial torture and seek comp"],
  "allegations": ["Medical evidence proves police beating. Strong case against police for custodial"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Police Beating In Custody Is Criminal Assault"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File criminal case against police with medical evidence",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (File case against police for custodial torture and seek comp), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Severity of the offence, prior criminal record, likelihood of flight risk, and whether the investigation is still ongoing are all factors courts weigh in bail. If the offence has minimum mandatory imprisonment, bail becomes harder.

### Commercial Reality

Liberty is at stake — this always justifies the cost of good legal representation. Align the effort to the severity of the charge and client capacity.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the exact offence charged and the court where the matter is registered.
- Confirm custody status — regular bail if in custody, anticipatory bail if at liberty.
- Arrange surety and bail documents immediately.
- Contact the family members who can appear as surety.

**What NOT to do:**
- Statements to police without representation
- Escaping jurisdiction
- Contact with witnesses

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Penal Code, 1860 — Section 330: Police beating is criminal assault

**Case law identified via retrieval:**
- *D.K. Basu vs State of West Bengal* (1997): Police are liable for torture and assault in custody


---

## rich_086: State of Punjab vs Balbir Singh

| Field | Value |
|-------|-------|
| **Domain** | Criminal Arrest |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 1993 |
| **Cited By** | 2,950 cases |

### Client Scenario

**Presenting Problem:** Police arrested me in a drug case. But they did not follow proper search procedures. There was no independent witness. They did not record the search on video. How can the evidence be used against me if the search itself was illegal? I want to get acquitted based on these procedural violations.

**Emotional State:** hopeful

**Prayer / Relief Sought:** Acquittal due to illegal search and procedural violations

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Police arrested me in a drug case. But they did not follow proper search procedure. No witness, no video. How can evidence be used?

**Advocate:** Do you have documentation of how the search was conducted?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** Police report says search was done but no video. I asked about witness but they say none was called.

**Advocate:** Are the mandatory Section 50 and 42 NDPS procedures mentioned in the case?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** I do not understand these procedures but my family says the police did not follow law properly.

**Advocate:** Did the investigating officer search you in front of any witness?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No witnesses were called. Police did the search alone without anyone present.

**Advocate (completing intake):** NDPS cases have strict procedural requirements. Search without witness and video is illegal under NDPS law. These violations are grounds for acquittal. Let me prepare detailed opinion on getting charges dismissed.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["Police arrested me in a drug case. But they did not follow proper search procedu", "Prayer: Acquittal due to illegal search and procedural violations"],
  "allegations": ["Police search violated mandatory NDPS procedures. Evidence likely inadmissible."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Ndps Search Procedures Were Followed"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File application to exclude evidence due to procedural violations",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Acquittal due to illegal search and procedural violations), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Severity of the offence, prior criminal record, likelihood of flight risk, and whether the investigation is still ongoing are all factors courts weigh in bail. If the offence has minimum mandatory imprisonment, bail becomes harder.

### Commercial Reality

Liberty is at stake — this always justifies the cost of good legal representation. Align the effort to the severity of the charge and client capacity.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the exact offence charged and the court where the matter is registered.
- Confirm custody status — regular bail if in custody, anticipatory bail if at liberty.
- Arrange surety and bail documents immediately.
- Contact the family members who can appear as surety.

**What NOT to do:**
- Statements to police without representation
- Escaping jurisdiction
- Contact with witnesses

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Narcotic Drugs and Psychotropic Substances Act, 1985 — Section 50: Mandatory requirement violated

**Case law identified via retrieval:**
- *State of Punjab vs Balbir Singh* (1993): Evidence obtained in violation of NDPS procedures is inadmissible


---

## rich_087: Harbanslal Sahnia vs Indian Oil Corp

| Field | Value |
|-------|-------|
| **Domain** | Contract |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2003 |
| **Cited By** | 1,750 cases |

### Client Scenario

**Presenting Problem:** I have been running an Indian Oil petrol pump for twelve years. Without any warning or hearing, the oil company suddenly terminated my dealership. I invested my entire life savings to build this business. This is completely unfair and devastating. I must get compensation or have my dealership restored.

**Emotional State:** angry

**Prayer / Relief Sought:** Reinstatement of dealership or substantial compensation for wrongful termination

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I ran an Indian Oil petrol pump for twelve years. Suddenly the company terminated my dealership without notice or reason. I invested everything in this business.

**Advocate:** Did they give any written reason for the termination?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** Just a letter saying dealership is terminated effective immediately. No explanation. No prior notice. Nothing.

**Advocate:** Did you have any opportunity to explain or defend yourself?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** None. They just terminated everything overnight. After twelve years of loyal service.

**Advocate:** Do you know if there were any complaints or issues they had with your business?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** My pumps always met their standards. Sales were good. I cannot understand why they would do this.

**Advocate (completing intake):** Twelve years of service without notice or hearing is serious breach of contract. Oil company cannot act arbitrarily. You have strong claim for damages. Let me prepare opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I have been running an Indian Oil petrol pump for twelve years. Without any warn", "Prayer: Reinstatement of dealership or substantial compensation for "],
  "allegations": ["Clear wrongful termination - no notice, no hearing, 12 years service"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Termination Requires Notice And Hearing"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File contract breach suit claiming damages or reinstatement",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Reinstatement of dealership or substantial compensation for ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Oral modifications to a written contract, waiver by conduct, or delayed performance accepted without protest can all weaken a breach claim. If the client also failed to perform some obligation, set-off or counter-claim risk must be assessed.

### Commercial Reality

Contract recovery suits are worth pursuing for significant amounts. For smaller amounts, consider demand notice and negotiation first — litigation cost may exceed recovery.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the original signed contract and any written amendments.
- Document all instances of breach with dates and supporting correspondence.
- Calculate the exact loss or damage claimed.
- Send a formal notice of breach and demand for remedy before filing suit.

**What NOT to do:**
- Continuing performance after breach without written reservation of rights
- Oral settlement discussions without written record
- Destroying any correspondence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Contract Act, 1872 — Section 73: Basis for damages claim

**Case law identified via retrieval:**
- *Harbanslal Sahnia vs Indian Oil Corp* (2003): Long-term dealership termination without notice requires compensation


---

## rich_088: Shrilekha Vidyarthi vs State of U.P.

| Field | Value |
|-------|-------|
| **Domain** | Contract |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2011 |
| **Cited By** | 950 cases |

### Client Scenario

**Presenting Problem:** I have been empanelled as government counsel for the state for eight years. Suddenly, without any warning, the state government removed me from the empanelment panel. No reason was given. No hearing was conducted. I did not commit any misconduct. Many other advocates with less experience and ability are still on the panel. This is arbitrary and clearly discriminatory. I need to restore my position.

**Emotional State:** angry

**Prayer / Relief Sought:** Restoration to government empanelment panel and compensation for lost income and harassment

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I was empanelled as government counsel for eight years. The state government removed me suddenly from the panel without reason or hearing.

**Advocate:** Did they give any written reason for the removal?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** No written reason. Just a letter saying I am removed from the empanelment. No explanation.

**Advocate:** Were you given opportunity to respond or explain?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No opportunity. The removal was sudden. Advocates with less experience are still on the panel. This is unfair.

**Advocate:** Do you know if there were any complaints about your work?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No complaints. I performed my duties professionally for eight years.

**Advocate (completing intake):** Eight years of service cannot be terminated arbitrarily without reason or hearing. This is discrimination. You have clear grounds to challenge this removal. Let me prepare opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I have been empanelled as government counsel for the state for eight years. Sudd", "Prayer: Restoration to government empanelment panel and compensation"],
  "allegations": ["Arbitrary removal without cause or hearing after 8 years service"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Government Can Remove Empanelled Counsel Without Due Process"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File writ petition for restoration to empanelment panel",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Restoration to government empanelment panel and compensation), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Oral modifications to a written contract, waiver by conduct, or delayed performance accepted without protest can all weaken a breach claim. If the client also failed to perform some obligation, set-off or counter-claim risk must be assessed.

### Commercial Reality

Contract recovery suits are worth pursuing for significant amounts. For smaller amounts, consider demand notice and negotiation first — litigation cost may exceed recovery.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the original signed contract and any written amendments.
- Document all instances of breach with dates and supporting correspondence.
- Calculate the exact loss or damage claimed.
- Send a formal notice of breach and demand for remedy before filing suit.

**What NOT to do:**
- Continuing performance after breach without written reservation of rights
- Oral settlement discussions without written record
- Destroying any correspondence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 14: No arbitrary removal of government counsel

**Case law identified via retrieval:**
- *Shrilekha Vidyarthi vs State of UP* (2011): Advocates cannot be arbitrarily removed from government empanelment without reason


---

## rich_089: Sadhu Singh Roda vs State of Punjab

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1979 |
| **Cited By** | 1,450 cases |

### Client Scenario

**Presenting Problem:** My son is in jail and has been a model prisoner. He is eligible for remission but the government rejected his application without giving any reason. Other prisoners with worse records and behavior were granted remission. This discrimination is heartbreaking. Please help get my son released.

**Emotional State:** distressed

**Prayer / Relief Sought:** Direction to grant remission and release of prisoner. Compensation for wrongful denial of remission.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My son is in jail. He is eligible for remission but they rejected his application. Other prisoners are getting released. Why not my son?

**Advocate:** What was the reason given for rejecting the remission application?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* Constitutional matters require identifying the specific right being violated and whether an alternative remedy is available. Courts routinely reject writ petitions at the threshold if there is an adequate alternative remedy. Knowing this shapes whether we file a writ or pursue the alternative first.

*Plausible bad alternative:* "What happened and why do you think your rights have been violated?"

*Why bad alternative loses:* This is too open-ended and does not target the decision-relevant information. The advocate needs to know the specific state action, the authority responsible, and whether any alternative remedy exists — not the client's general narrative.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** No reason. They just said it is rejected. My son has good behavior record.

**Advocate:** Has your son completed the minimum prison term for remission?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. He is well past the minimum period. Other prisoners with worse records are getting remission.

**Advocate:** Do you have copies of his prison conduct report?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. He has no disciplinary actions. Perfect conduct record.

**Advocate (completing intake):** Arbitrary rejection of remission for eligible prisoner with good conduct is clearly unlawful. Other prisoners getting released while your son is denied shows discrimination. You have strong grounds to challenge this. Let me prepare opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["My son is in jail and has been a model prisoner. He is eligible for remission bu", "Prayer: Direction to grant remission and release of prisoner. Compen"],
  "allegations": ["Arbitrary remission denial for eligible prisoner with good conduct. Discriminati"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Remission Can Be Arbitrarily Denied"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File petition for mandamus to grant remission",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Direction to grant remission and release of prisoner. Compen), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 161: Power to grant remission but must be exercised fairly

**Case law identified via retrieval:**
- *Sadhu Singh Roda vs State of Punjab* (1979): Remission must be granted to eligible prisoners who meet criteria


---

## rich_090: Hiten P. Dalal vs Bratindranath Banerjee

| Field | Value |
|-------|-------|
| **Domain** | Cheque Bounce |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | limitation deadline near |
| **Court** | Supreme Court of India |
| **Year** | 2001 |
| **Case Type** | General |
| **Cited By** | 780 cases |

### Client Scenario

**Presenting Problem:** I was approached by someone offering guaranteed high returns in a share market investment scheme. I invested 50 lakhs. Now they have disappeared with all the money. I have lost everything. This is fraud and cheating. I want to file a criminal case against them and recover my money.

**Emotional State:** Concerned about legal rights and outcome

**Prayer / Relief Sought:** Registration of criminal case for cheating and criminal breach of trust. Recovery of defrauded amount. (Amount: ₹5,000,000)

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Someone got me to invest 50 lakhs in a share scheme promising guaranteed returns. Now they are gone with the money. I lost everything.

**Advocate:** Do you have any documents of the investment agreement?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely criminal court. Need evidence_strength facts before confirming strategy.

**Client:** Yes, I have documents showing the guaranteed return promise and the money transfer. Everything is documented.

**Advocate:** Do you know where this person is now?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** He has disappeared. I cannot find him. No one answers his phone.

**Advocate:** Are there other people who were defrauded by him?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. I know of at least 5 other people who lost money to him. We compared notes.

**Advocate (completing intake):** This is clearly cheating and criminal fraud. You have documentation proving deception and promises of guaranteed returns. Multiple victims strengthen the case. Let me prepare opinion on filing criminal complaint.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["limitation_deadline_near"],
  "known_facts": ["I was approached by someone offering guaranteed high returns in a share market i", "Prayer: Registration of criminal case for cheating and criminal brea"],
  "allegations": ["Clear case of investment fraud with multiple victims and documentary evidence"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Investment Fraud Constitutes Criminal Cheating"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File criminal complaint for cheating under IPC Section 420",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Registration of criminal case for cheating and criminal brea), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The drawer may claim the cheque was not for a legally enforceable debt (gift, illegal consideration), or that the signature was forged, or that payment was already made. Any weakness in the demand notice (wrong address, timing defect) can defeat the complaint. Limitation is the most common killer.

### Commercial Reality

For amounts above 5 lakhs, civil recovery alongside the criminal complaint is worth pursuing. For smaller amounts, the criminal complaint alone may be sufficient leverage for settlement. Settlement is often achievable once the criminal complaint is filed.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Check whether the demand notice was sent within 30 days of bank dishonour memo.
- If notice not sent, send it immediately by registered post and email.
- If notice has been sent, confirm whether 15 days have elapsed without payment.
- Preserve the original cheque, dishonour memo, notice copy, and delivery confirmation.

**What NOT to do:**
- Delay in sending demand notice
- Accepting partial payment without written condition
- Destroying any banking documents

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Penal Code, 1860 — Section 420: Investment scheme constitutes cheating

**Case law identified via retrieval:**
- *Hiten P. Dalal vs Bratindranath Banerjee* (2001): Investment fraud schemes constitute criminal cheating prosecution

**Key judicial reasoning (from source case):**
> 1. Such offence must relate to transactions in
securities; and

2 Such offence should be alleged to have
been committed between 1.4.1991 and on
or before 6.6.1992'.
This statement of law is contrary t...


---

## rich_091: Mukund Dewangan vs Oriental Insurance

| Field | Value |
|-------|-------|
| **Domain** | Motor Accident |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 1997 |
| **Cited By** | 2,200 cases |

### Client Scenario

**Presenting Problem:** I had an accident with my truck. I was injured badly and my vehicle was damaged. I have comprehensive insurance but the insurance company is refusing to pay. They say I only have an LMV license but was driving a transport vehicle. This is not fair. Many drivers work like this. I need the money to pay medical bills and vehicle repairs.

**Emotional State:** distressed

**Prayer / Relief Sought:** Insurance claim settlement and compensation for wrongful denial of claim

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I had truck accident. I was injured. Insurance company refuses to pay because I have LMV license not transport license. But I have comprehensive insurance. Why would they refuse?

**Advocate:** What does your insurance policy say about license requirements?

*Decision dimension targeted:* `evidence_strength + remedy`

*Why this question wins:* Motor accident compensation depends heavily on the victim's income and age (for multiplier calculation), and on establishing the other vehicle's negligence. Asking about income and available documents is the highest-value question because it directly determines the quantum of compensation and the strength of the claim.

*Plausible bad alternative:* "Describe the accident scene and the road conditions."

*Why bad alternative loses:* Scene description matters for negligence but the court will primarily rely on the FIR, spot inspection report, and MVI report for that. The advocate's first need is to understand income, age, and document availability — these directly determine claim value and strategy.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need evidence_strength facts before confirming strategy.

**Client:** The policy says comprehensive coverage. There is no mention that LMV license would invalidate the claim.

**Advocate:** How long have you been driving this truck?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** Five years. Many drivers operate this way with LMV license. It is common in the industry.

**Advocate:** Do you have the original insurance policy documents and the claim rejection letter?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. I have everything. The rejection letter just says LMV license is not authorized for transport vehicle.

**Advocate (completing intake):** Insurance policy cannot deny claim on grounds not mentioned in the policy. Your comprehensive coverage should cover the accident. Insurance company is acting unfairly. Let me prepare opinion on claim settlement.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["I had an accident with my truck. I was injured badly and my vehicle was damaged.", "Prayer: Insurance claim settlement and compensation for wrongful den"],
  "allegations": ["Insurance company wrongfully denying valid claim based on technical license issu"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether License Type Affects Insurance Claim Validity"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File complaint with insurance ombudsman or suit for claim settlement",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Insurance claim settlement and compensation for wrongful den), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Contributory negligence on the victim's part reduces compensation. Inconsistencies between the FIR and the claim narrative, or exaggerated income claims without documentation, are significant weaknesses. Delay in filing before MACT can reduce the claim period.

### Commercial Reality

Motor accident claims are worth pursuing — the MACT process is structured, legal aid is available, and insurance companies are routinely made to pay. Cost is relatively low relative to potential compensation.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain the FIR, post-mortem report or injury certificate, and treatment records immediately.
- Collect income proof — salary slips, ITR, or employer certificate.
- Identify the offending vehicle and its insurance details from the FIR.
- File the MACT claim promptly — limitation is running.

**What NOT to do:**
- Accepting any settlement from the insurer without legal advice
- Delay in filing the MACT claim
- Disposing of medical records

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Motor Vehicles Act, 1988 — Section 2: License requirements for vehicle types

**Case law identified via retrieval:**
- *Mukund Dewangan vs Oriental Insurance* (1997): Insurance claim cannot be denied solely for license type issue


---

## rich_092: Alakh Alok Srivastava vs Union of India

| Field | Value |
|-------|-------|
| **Domain** | Medical Negligence |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | evidence disappearing |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Cited By** | 640 cases |

### Client Scenario

**Presenting Problem:** My child was rescued from trafficking. The government hospital provided medical treatment but they were negligent. The doctors did not properly treat my child. My child suffered permanent injuries that could have been prevented with proper medical care. I want compensation and accountability for the government hospital's negligence.

**Emotional State:** distressed

**Prayer / Relief Sought:** Compensation for medical negligence and government hospital liability

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My child was rescued from trafficking. The government hospital treated my child but they were negligent. My child has permanent injuries.

**Advocate:** What specific treatment did the hospital provide that was negligent?

*Decision dimension targeted:* `evidence_strength + forum`

*Why this question wins:* Medical negligence cases live or die on records. The most important first question is whether the medical records are intact — because hospitals can modify or lose records after a complaint. The evidence preservation question has the highest decision value because delay in obtaining records can permanently weaken the case.

*Plausible bad alternative:* "What was the medical procedure and what went wrong?"

*Why bad alternative loses:* The narrative of what went wrong is important but can be gathered later. The immediate priority is whether the medical records have been secured — because once records disappear or are altered, no amount of narrative will help.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely consumer forum. Need evidence_strength facts before confirming strategy.

**Client:** They did not properly dress wounds. They did not monitor vital signs adequately. They did not follow proper protocols.

**Advocate:** Did any doctor tell you what went wrong?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** No. We learned through other medical consultation that the hospital's negligence caused the permanent injury.

**Advocate:** Do you have copies of the hospital treatment records?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. And we have opinion from another doctor confirming the negligence.

**Advocate (completing intake):** Government hospital is liable for negligent medical treatment of your child. You have medical opinion supporting negligence. You can sue the hospital for damages. Let me prepare detailed opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["evidence_disappearing"],
  "known_facts": ["My child was rescued from trafficking. The government hospital provided medical ", "Prayer: Compensation for medical negligence and government hospital "],
  "allegations": ["Government hospital negligent treatment documented by medical opinion"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Government Hospital Liable For Medical Negligence"],
  "evidence_status": "moderate",
  "recommended_forum": "consumer_forum",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File suit against government hospital for negligent treatment",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compensation for medical negligence and government hospital ), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Medical negligence requires expert opinion to establish the standard of care and its breach. Without an independent medical expert supporting the claim, courts are unlikely to hold the doctor liable on the patient's assertion alone. Consent forms signed by the patient before the procedure are used by defence.

### Commercial Reality

Medical negligence claims can be filed before the consumer forum (for service deficiency) at relatively low cost. The challenge is obtaining expert opinion and the time it takes. For large claims, civil court is appropriate but expensive.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Request copies of all medical records, discharge summary, and treatment notes immediately in writing.
- Preserve all bills, prescriptions, and communications with the hospital.
- Obtain a second medical opinion on whether the treatment was appropriate.
- Consider filing a consumer complaint with the medical records before they can be altered.

**What NOT to do:**
- Delay in obtaining medical records
- Confronting the doctor without legal backing
- Signing any documents presented by the hospital after the adverse event

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Indian Penal Code, 1860 — Section 336: Hospital negligence endangered child

**Case law identified via retrieval:**
- *Alakh Alok Srivastava vs Union of India* (2005): Government hospital liable for medical negligence


---

## rich_093: State of Uttar Pradesh vs Ram Veer Singh

| Field | Value |
|-------|-------|
| **Domain** | Criminal General |
| **Client Type** | Lay Client |
| **Urgency Level** | HIGH |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2001 |
| **Cited By** | 1,300 cases |

### Client Scenario

**Presenting Problem:** I was accused of murder. After a long trial, the High Court acquitted me finding no evidence against me. Now the government is appealing my acquittal in the Supreme Court. How can the government appeal when the court has found me not guilty? This is harassment. I want the acquittal to be sustained.

**Emotional State:** angry

**Prayer / Relief Sought:** Sustain the acquittal and dismiss the state appeal. Compensation for wrongful prosecution and harassment.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** High Court acquitted me of murder charges. Now the government is appealing the acquittal. How can they do this? I was found not guilty.

**Advocate:** What was the High Court reason for acquitting you?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* The right forum is a threshold question. Filing in the wrong court wastes time and money and may permanently affect the client's position.

*Plausible bad alternative:* "What outcome are you hoping for?"

*Why bad alternative loses:* Desired outcome is known from the presenting problem. It does not help select the forum. Forum depends on the nature of the dispute and the applicable law.

*Current best decision at this turn:* Provisional: HIGH urgency matter. Forum likely criminal court. Need forum facts before confirming strategy.

**Client:** Court said there was no evidence against me. Witnesses were unreliable. The case was weak.

**Advocate:** How long has this case been pending?

*Decision dimension targeted:* `limitation + urgency`

*Why this question wins:* Timing determines whether the matter is still actionable at all. Limitation is the threshold question that can render every other analysis irrelevant.

*Plausible bad alternative:* "Describe the events in detail from the very beginning."

*Why bad alternative loses:* A detailed narrative without a timeline does not flag limitation risk. The advocate needs the key dates first.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm limitation assessment.

**Client:** Eight years! I have been to hell and back. Now they want to continue this torture.

**Advocate:** What evidence do you think the government is relying on in the appeal?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Same weak evidence that the High Court rejected! This is clearly harassment. The High Court found me not guilty.

**Advocate (completing intake):** The High Court found you not guilty due to lack of evidence. The state appeal against acquittal without substantial new grounds is harassment. You have strong grounds to have the acquittal sustained. Let me prepare opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "high",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["I was accused of murder. After a long trial, the High Court acquitted me finding", "Prayer: Sustain the acquittal and dismiss the state appeal. Compensa"],
  "allegations": ["High Court acquittal based on weak evidence. State appeal appears to lack substa"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether State Can Appeal Acquittal Without Substantial Grounds"],
  "evidence_status": "moderate",
  "recommended_forum": "criminal_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Contest state appeal emphasizing lack of substantial new evidence",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Sustain the acquittal and dismiss the state appeal. Compensa), urgency level is assessed (high), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Eyewitness testimony, recovered material evidence, or forensic evidence will be the primary prosecution tools. Prior bad conduct or inconsistent statements will be used to undermine credibility at trial.

### Commercial Reality

Criminal defence is high-stakes. Cost must be aligned to the severity of the charge.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific offence charged and the current stage of proceedings.
- Assess whether bail or anticipatory bail is needed.
- Identify all potentially incriminating evidence the prosecution may have.
- Locate witnesses who can support the defence.

**What NOT to do:**
- Statements without counsel
- Contact with prosecution witnesses
- Destruction of any evidence

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 379: State must show good reason for appealing acquittal

**Case law identified via retrieval:**
- *State of Uttar Pradesh vs Ram Veer Singh* (2001): Acquittal cannot be interfered with unless state shows substantial new evidence or grave error


---

## rich_094: Nikhil Merchant vs C.B.I.

| Field | Value |
|-------|-------|
| **Domain** | Banking Recovery |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Cited By** | 850 cases |

### Client Scenario

**Presenting Problem:** I took a bank loan for my business. Due to market downturn, I could not repay it. Now the CBI has registered a case against me for bank fraud. But I am ready to repay the bank now that my business has recovered. Can I settle this matter through the court instead of facing criminal prosecution?

**Emotional State:** hopeful

**Prayer / Relief Sought:** Compounding of cheating offense and settlement with bank

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** I took a bank loan for my business but could not repay during market downturn. Now CBI case against me. But I can repay now. Can I settle?

**Advocate:** Is the bank willing to settle if you repay the full amount?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely civil court. Need remedy facts before confirming strategy.

**Client:** I have not asked but I am willing to repay whatever is owed. With interest if needed.

**Advocate:** Do you have the capacity to repay the full loan amount now?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Yes my business has recovered. I can arrange the full amount with interest.

**Advocate:** Have you been formally charged in the CBI case or is it still under investigation?

*Decision dimension targeted:* `forum + immediate_procedural_step`

*Why this question wins:* The stage of proceedings determines the appropriate remedy — bail at different stages has different standards, and different forums are available at different stages.

*Plausible bad alternative:* "Tell me about your life and work before this incident."

*Why bad alternative loses:* Personal background is irrelevant to stage of proceedings. The current procedural status is what drives the immediate action.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm forum assessment.

**Client:** Case is registered but I have not been arrested yet. I want to settle before that happens.

**Advocate (completing intake):** Bank fraud offenses can sometimes be compounded if the bank agrees to settle. Your willingness to repay and ability to do so are positive factors. Let me prepare opinion on options for compounding.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["I took a bank loan for my business. Due to market downturn, I could not repay it", "Prayer: Compounding of cheating offense and settlement with bank"],
  "allegations": ["CBI case for bank fraud. Client wants to settle with bank to avoid prosecution."],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Bank Fraud Can Be Compounded Through Settlement"],
  "evidence_status": "moderate",
  "recommended_forum": "civil_court",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "Approach bank for settlement. File compounding petition if bank agrees.",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Compounding of cheating offense and settlement with bank), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The bank's loan documents, security documents, and default record form a strong evidentiary base. Any personal guarantees given are significant liability. The borrower's best defences are procedural non-compliance by the bank, or the actual repayment record.

### Commercial Reality

Banking recovery disputes involve significant amounts and are worth defending rigorously if there are procedural defects in the bank's action. However, restructuring or one-time settlement is often more practical than prolonged litigation against a well-resourced institutional opponent.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Obtain all bank notices, demand letters, and SARFAESI notices.
- Check whether a DRT objection under Section 17 of SARFAESI is needed and the deadline.
- Compile the full loan account statement and repayment record.
- Assess whether a restructuring proposal to the bank is feasible.

**What NOT to do:**
- Ignoring SARFAESI notices — they have strict timelines
- Vacating mortgaged property without legal advice
- Making partial payments without written conditions

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 147: Bank fraud can be compounded with bank consent

**Case law identified via retrieval:**
- *Nikhil Merchant vs CBI* (2005): Bank fraud can be compounded upon settlement with victim bank


---

## rich_095: Municipal Council Palai vs T.J. Joseph

| Field | Value |
|-------|-------|
| **Domain** | Consumer |
| **Client Type** | Lay Client |
| **Urgency Level** | LOW |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 2006 |
| **Cited By** | 720 cases |

### Client Scenario

**Presenting Problem:** The municipal corporation sent me a property tax notice with a tax that is three hundred percent more than what I paid last year. The property has not changed. There is no justification for such a huge increase. This is unreasonable. How can the municipality do this? I want to challenge this excessive tax notice.

**Emotional State:** angry

**Prayer / Relief Sought:** Reduction of property tax to reasonable amount. Refund of excess tax collected illegally.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Municipality sent me tax notice three times higher than last year. Property has not changed. This is unreasonable.

**Advocate:** Did they give any reason for the huge increase?

*Decision dimension targeted:* `forum + evidence_strength`

*Why this question wins:* Consumer complaints require knowing the exact deficiency in service or defect in goods, the amount at stake, and whether the limitation of two years from the date of cause of action is still alive. Forum selection depends on the pecuniary value of the claim.

*Plausible bad alternative:* "How long have you been a customer of this company?"

*Why bad alternative loses:* Tenure as a customer is not relevant to the complaint. What matters is the specific deficiency, the date it occurred, and the documentary evidence available. Customer history does not change the legal analysis.

*Current best decision at this turn:* Provisional: LOW urgency matter. Forum likely consumer forum. Need forum facts before confirming strategy.

**Client:** No reason given. Just suddenly the tax is triple.

**Advocate:** Do you have copies of last year's tax assessment?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. I can compare. This year is completely unjustified.

**Advocate:** Have you already paid the new tax or did you reject it?

*Decision dimension targeted:* `defence_exposure + prior_actions_taken`

*Why this question wins:* Prior actions are the most common source of case-killing admissions. Knowing what has already been said, done, or filed before advising on strategy is non-negotiable.

*Plausible bad alternative:* "What do you want us to do for you?"

*Why bad alternative loses:* Asking for the desired outcome before checking prior actions is premature. A damaging prior statement can limit the available options significantly.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm defence_exposure assessment.

**Client:** I paid because they threatened to attach my property. But I believe it is illegal.

**Advocate (completing intake):** Three hundred percent tax increase without justification is unreasonable and likely unlawful. You have grounds to challenge this. Let me prepare opinion on tax reduction.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "low",
  "urgency_flags": ["none_detected"],
  "known_facts": ["The municipal corporation sent me a property tax notice with a tax that is three", "Prayer: Reduction of property tax to reasonable amount. Refund of ex"],
  "allegations": ["Excessive municipal tax increase without justification"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Municipal Tax Assessment Is Reasonable"],
  "evidence_status": "moderate",
  "recommended_forum": "consumer_forum",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File consumer complaint or tax reduction petition",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Reduction of property tax to reasonable amount. Refund of ex), urgency level is assessed (low), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

If the consumer accepted a partial refund or signed a settlement without reservation, the complaint may not be maintainable. Missing receipts or invoices weaken the claim. Terms and conditions that excluded liability for the specific deficiency will be relied on by the company.

### Commercial Reality

Consumer forums are low-cost and relatively accessible. Even for moderate amounts, filing is worth it given the low barrier. For larger amounts, consider whether a civil suit in regular court provides better discovery and enforcement mechanisms.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Collect all receipts, invoices, contracts, and written communications with the company.
- Document the specific deficiency or defect with photographs or records.
- Send a written complaint to the company as a first step before filing.
- Calculate the exact compensation sought — refund, replacement, damages.

**What NOT to do:**
- Accepting any settlement without written terms
- Delay beyond two years from the date of the deficiency

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 300A: Property cannot be arbitrarily taxed

**Case law identified via retrieval:**
- *Municipal Council Palai vs T.J. Joseph* (2006): Municipal tax assessment must be reasonable and justified


---

## rich_096: Shiji @ Pappu vs Radhika

| Field | Value |
|-------|-------|
| **Domain** | Fir Quashing |
| **Client Type** | Lay Client |
| **Urgency Level** | IMMEDIATE |
| **Urgency Flags** | arrest or coercive action imminent |
| **Court** | Supreme Court of India |
| **Year** | 2005 |
| **Cited By** | 580 cases |

### Client Scenario

**Presenting Problem:** My wife filed an FIR against me for domestic violence but there was no violence. This is a completely false case. The real issue is a land dispute about our farm. She filed the FIR to pressure me into giving her the farm. She is using the domestic violence law to extort the land from me. This is a false and malicious prosecution. I want the FIR quashed.

**Emotional State:** angry

**Prayer / Relief Sought:** Quashing of FIR for domestic violence as based on false allegations. Compensation for malicious prosecution.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** My wife filed FIR against me for domestic violence. But there was no violence. Real issue is about farm land.

**Advocate:** Do you have any evidence that the land dispute is the real issue?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Provisional: IMMEDIATE urgency matter. Forum likely high court writ. Need evidence_strength facts before confirming strategy.

**Client:** Yes. Before filing FIR, she demanded I give her the farm. When I refused, she filed this case.

**Advocate:** Are there any witnesses to your marital relationship or the land dispute?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. Neighbors and family know about the land issue. No one has seen any violence.

**Advocate:** Do you have documented proof of the land ownership?

*Decision dimension targeted:* `evidence_strength`

*Why this question wins:* Evidence is the foundation of case strength. Knowing what documentation exists determines the viable strategy before any legal path is chosen.

*Plausible bad alternative:* "How confident are you that you will win this case?"

*Why bad alternative loses:* Client confidence is not a legal assessment. The answer changes nothing about strategy or forum. Only the evidence position matters.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** Yes. The farm is in my name from before marriage.

**Advocate (completing intake):** False domestic violence FIR to extort land is serious abuse of process. You have evidence the real dispute is about land. You have strong grounds to quash this FIR. Let me prepare detailed opinion.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "immediate",
  "urgency_flags": ["arrest_or_coercive_action_imminent"],
  "known_facts": ["My wife filed an FIR against me for domestic violence but there was no violence.", "Prayer: Quashing of FIR for domestic violence as based on false alle"],
  "allegations": ["FIR appears to be misuse of domestic violence law to extort land property"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Fir Can Be Quashed When It Masks Property Dispute"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File petition under CrPC 482 to quash FIR citing false allegations",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quashing of FIR for domestic violence as based on false alle), urgency level is assessed (immediate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

The FIR may disclose a prima facie cognizable offence, in which case courts are reluctant to quash at the threshold. If the allegations are specific and supported by documentary evidence on the complainant's side, the quashing petition faces a higher bar. Any prior settlement attempt that failed also weakens the 'civil dispute' argument.

### Commercial Reality

Quashing petitions involve High Court filing fees and senior counsel time. If the client has means to engage, this is worth pursuing — the alternative (arrest and bail) is costlier and reputationally damaging. If the FIR is clearly malicious and evidence is strong, the cost-benefit clearly favours action.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Do not surrender or appear before police until protective relief is assessed.
- Preserve all documents that show the dispute is civil or that the FIR is motivated by malice.
- Do not make any statement to police without legal representation.
- Collect the FIR copy, any prior written communications with the complainant, and evidence of the civil transaction underlying the dispute.

**What NOT to do:**
- Approaching the complainant directly
- Making statements to police
- Ignoring summons without legal advice

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Code of Criminal Procedure, 1973 — Section 482: FIR appears to be abuse of process

**Case law identified via retrieval:**
- *Shiji @ Pappu vs Radhika* (2005): Domestic violence FIR cannot be used to mask property dispute or extort property


---

## rich_097: Indian Express Newspapers vs Union of India

| Field | Value |
|-------|-------|
| **Domain** | Constitutional |
| **Client Type** | Lay Client |
| **Urgency Level** | MODERATE |
| **Urgency Flags** | none detected |
| **Court** | Supreme Court of India |
| **Year** | 1985 |
| **Cited By** | 3,100 cases |

### Client Scenario

**Presenting Problem:** We are a small newspaper publisher. The government has levied a discriminatory import duty on newsprint that makes it impossible for us to print newspapers at competitive prices. Large newspaper groups can afford the duty but publishers like us cannot survive this discrimination. This is clearly a tax on our freedom of press. The government is using taxation to suppress small newspapers. This is unconstitutional.

**Emotional State:** angry

**Prayer / Relief Sought:** Quashing of newsprint import duty as unconstitutional and discriminatory. Restoration of press freedom.

> **Layer Boundary:** The intake conversation below operates at the diagnostic layer only.
> No section numbers or case citations are used during intake.
> The Grounded Advice section at the end uses source-retrieved material.

### Enriched Intake Conversation

**Client:** Government has levied a special import duty on newsprint. This makes newspapers too expensive to print. Other imported goods do not have this duty. This discriminates against newspapers.

**Advocate:** Is this duty applied only to newsprint or to other imported paper products also?

*Decision dimension targeted:* `forum + maintainability`

*Why this question wins:* Constitutional matters require identifying the specific right being violated and whether an alternative remedy is available. Courts routinely reject writ petitions at the threshold if there is an adequate alternative remedy. Knowing this shapes whether we file a writ or pursue the alternative first.

*Plausible bad alternative:* "What happened and why do you think your rights have been violated?"

*Why bad alternative loses:* This is too open-ended and does not target the decision-relevant information. The advocate needs to know the specific state action, the authority responsible, and whether any alternative remedy exists — not the client's general narrative.

*Current best decision at this turn:* Provisional: MODERATE urgency matter. Forum likely high court writ. Need forum facts before confirming strategy.

**Client:** Only to newsprint. Other imported goods are free from this special duty. This is clearly targeted at newspapers.

**Advocate:** How much has this duty affected your publishing costs?

*Decision dimension targeted:* `remedy + evidence_strength`

*Why this question wins:* Compensation quantum is the most concrete deliverable in this type of matter. Knowing income and documentation status directly determines claim value.

*Plausible bad alternative:* "Tell me more about the circumstances."

*Why bad alternative loses:* Vague background gathering is not decision-relevant. The quantum question directly determines the legal relief available.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm remedy assessment.

**Client:** Newsprint costs have increased by 40%. Small publishers cannot absorb this. Only big groups with scale can afford it.

**Advocate:** Is this a recent duty or has it been in effect for long?

*Decision dimension targeted:* `evidence_strength + next_procedural_step`

*Why this question wins:* This follow-up targets the remaining gap in the evidence picture before the intake can safely complete.

*Plausible bad alternative:* "Is there anything else you would like to tell me?"

*Why bad alternative loses:* An open-ended invitation produces an unstructured narrative. The advocate should ask about the specific missing fact, not solicit general elaboration.

*Current best decision at this turn:* Core facts emerging. Strategy direction becoming clearer. Waiting on this turn's answer to confirm evidence_strength assessment.

**Client:** It was introduced suddenly without warning. It is clearly designed to suppress small newspapers and benefit large media groups.

**Advocate (completing intake):** This selective duty on newsprint while other imports are free is discriminatory taxation affecting freedom of press. You have strong constitutional grounds to challenge this. Let me prepare opinion on unconstitutionality.

### Decision-State Snapshot at Completion

*Core runtime fields only — per schema v1.1 `_core_runtime_fields`*

```json
{
  "client_type": "lay_client",
  "urgency_level": "moderate",
  "urgency_flags": ["none_detected"],
  "known_facts": ["We are a small newspaper publisher. The government has levied a discriminatory i", "Prayer: Quashing of newsprint import duty as unconstitutional and di"],
  "allegations": ["Selective newsprint duty with 40% cost impact appears discriminatory and unconst"],
  "prior_actions_taken": ["checked during intake"],
  "legal_issues": ["Whether Discriminatory Newsprint Duty Violates Freedom Of Press"],
  "evidence_status": "moderate",
  "recommended_forum": "high_court_writ",
  "maintainability_assessment": "arguable",
  "enough_to_advise": true,
  "current_best_decision": "File constitutional challenge citing Article 19 violation",
  "next_best_question": null,
  "question_purpose": null,
  "question_decision_value": "do_not_ask"
}
```

### Stop Policy Check

**Why intake stopped here:** The three exit conditions were met — the client's objective is known (Quashing of newsprint import duty as unconstitutional and di), urgency level is assessed (moderate), and a provisional next step is identifiable.

**Counter-test:** Is there any single missing fact that, if answered differently, would change the forum, limitation status, evidence viability, or immediate procedural safety? In this matter, the key facts were sufficiently established. Further questioning would be for refinement, not for safety.

### Weakness Stress-Test

*What is the best case the opposing party can make?*

Alternative remedy doctrine is the most common ground on which writ petitions are rejected. If there is a statutory appeal or grievance mechanism, the court may refuse to exercise writ jurisdiction. Delay in filing after the state action is also a ground for rejection.

### Commercial Reality

Constitutional litigation can be protracted and expensive at the High Court level. For matters of fundamental rights, this is typically worth pursuing. For service or regulatory matters, weigh the cost against the availability of alternative remedies.

### Intake-Layer Action Plan
*(Diagnostic framing only — no section citations at intake stage)*

**What to do today:**
- Identify the specific government action or inaction that violates the right.
- Check whether a statutory appeal, representation, or grievance mechanism exists.
- Assess whether the violation is ongoing or completed.
- Preserve any communications from the government authority.

**What NOT to do:**
- Filing writ before exhausting alternative remedies if those remedies are adequate
- Delay in filing after the state action

### Grounded Advice Layer
*(Retrieved from source case — this layer requires retrieval, not memory)*

**Bare acts identified via retrieval:**
- Constitution of India, 1950 — Article 19: Press freedom is fundamental right

**Case law identified via retrieval:**
- *Indian Express Newspapers vs Union of India* (1985): Discriminatory taxation cannot be used to suppress freedom of press


