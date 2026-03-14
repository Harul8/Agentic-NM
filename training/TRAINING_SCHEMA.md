# Nyaymalaw Training Repository — JSON Schema
## Version 2.0 | Derived from Top-200 Cited Cases

---

## Overview

Each training record in the rich repository captures **three layers**:

1. **Case Analysis Layer** — what the court actually decided, with structured extraction of disputes, arguments, evidence, judicial reasoning, and outcome. This is the *ground truth*.

2. **Client Scenario Layer** — a realistic retail-client version of the same dispute (what an ordinary person would walk into a legal-aid office with).

3. **Training Conversation Layer** — the full client-advocate interaction: intake turns, follow-up questions, bare acts cited, case laws cited, advocate reasoning chain, and the final structured opinion. This is what the model learns from.

---

## File Structure

```
training/examples/
    intake_conversations.jsonl        ← EXISTING (5 records) → will grow to 100+
    final_opinions.jsonl              ← EXISTING (2 records) → will grow to 100+
    rich_cases/
        rich_training_records.jsonl   ← NEW: one record per case, all layers
```

---

## JSON Schema — Rich Training Record

```json
{
  // ──────────────────────────────────────────────────────────────────
  // LAYER 1: CASE METADATA
  // ──────────────────────────────────────────────────────────────────
  "id": "rich_001",                          // unique training record id
  "source_case_id": "SC_2014_ARNESH_KUMAR_VS_STATE_OF_BIHAR_ANR",
  "case_name": "Arnesh Kumar vs State of Bihar",
  "court": "Supreme Court of India",
  "year": "2014",
  "citation": "AIR 2014 SC 2756",            // null if not available
  "cited_by_count": 25720,                   // authority signal
  "legal_domain": "criminal",                // criminal | civil | matrimonial | labour | property | consumer | constitutional | motor_accident | cheque_bounce
  "dispute_type": "anticipatory_bail",       // fine-grained subtype
  "relevant_bare_acts": [
    {
      "act": "Code of Criminal Procedure, 1973",
      "sections": ["Section 41", "Section 41A", "Section 498A"],
      "purpose": "Conditions for arrest; anticipatory bail; cruelty by husband"
    },
    {
      "act": "Indian Penal Code, 1860",
      "sections": ["Section 498A"],
      "purpose": "Harassment by husband or in-laws for dowry"
    }
  ],

  // ──────────────────────────────────────────────────────────────────
  // LAYER 2: CASE ANALYSIS
  // ──────────────────────────────────────────────────────────────────
  "case_analysis": {
    "facts_of_case": "Brief narrative of the factual matrix — what happened, who are the parties, what relief was sought at the court level.",

    "disputes_identified": [
      {
        "dispute_no": 1,
        "dispute_label": "Validity of anticipated arrest under S.498A IPC",
        "dispute_summary": "Whether police can arrest the accused under Section 498A IPC without applying mind to the necessity of arrest per CrPC S.41."
      }
    ],

    "petitioner_arguments": [
      "Arrest under Section 498A is being used as a tool for personal vendetta.",
      "Mechanical arrests violate personal liberty under Article 21 of the Constitution.",
      "S.41 CrPC mandates the police officer to apply mind before arresting."
    ],

    "respondent_arguments": [
      "The allegations of cruelty and dowry harassment are serious cognizable offences.",
      "The accused may flee or tamper with evidence if not arrested.",
      "The FIR discloses prima facie offence — police discretion should not be fettered."
    ],

    "evidence_discussed": [
      {
        "evidence_type": "statistical",
        "description": "NCRB data showing 197,762 arrests in 2012 under S.498A IPC; conviction rate only 14.4%.",
        "relevance": "Court used this to establish rampant misuse of arrest powers."
      }
    ],

    "judicial_observations": [
      "Arrest brings humiliation, curtails freedom, and casts scars forever.",
      "Courts must ensure that police do not arrest accused unnecessarily.",
      "S.41A notice must be issued before arrest in offences punishable up to 7 years."
    ],

    "principles_laid_down": [
      "Police must record written reasons before making an arrest.",
      "Magistrate must apply independent mind before authorising detention.",
      "Courts may issue direction to police not to arrest without S.41A notice compliance."
    ],

    "order_outcome": "guilty | not_guilty | remanded | directions_issued | settlement | dismissed | partially_allowed",
    "order_summary": "The Supreme Court issued guidelines mandating police to serve S.41A notice before arrest in all offences punishable up to 7 years. State governments directed to train police and judicial officers. Magistrates directed to question necessity of arrest before remanding accused.",

    "relief_granted": "Anticipatory bail granted with conditions; guidelines issued to all states.",
    "relief_assessment": "Full relief — arrest stayed; important safeguard established."
  },

  // ──────────────────────────────────────────────────────────────────
  // LAYER 3: CLIENT SCENARIO (retail-client version of same dispute)
  // ──────────────────────────────────────────────────────────────────
  "client_scenario": {
    "scenario_id": "CS_001",
    "client_profile": {
      "name": "Rajesh",                        // anonymised first name
      "age": 34,
      "gender": "male",
      "location": "Pune, Maharashtra",
      "occupation": "software engineer",
      "monthly_income": 85000
    },
    "presenting_problem": "My wife has filed a case under Section 498A IPC against me and my parents. We received a police notice. I am scared we will be arrested. The marriage was troubled but there was no dowry demand. What should we do?",
    "client_emotional_state": "fearful",       // fearful | distressed | angry | confused | resigned | hopeful
    "prayer_mentioned": true,
    "prayer_stated": "I want anticipatory bail for me and my parents. I want to make sure we are not arrested.",
    "prayer_amount": null,                     // null if not monetary
    "prayer_reason": null
  },

  // ──────────────────────────────────────────────────────────────────
  // LAYER 4: TRAINING CONVERSATIONS
  // ──────────────────────────────────────────────────────────────────
  "intake_conversation": {
    "keywords": ["498A", "anticipatory bail", "arrest", "dowry", "police notice", "husband", "cruelty"],
    "conversation": [
      {
        "role": "user",
        "content": "My wife has filed a 498A case against me. Police came home. I'm scared of arrest."
      },
      {
        "role": "assistant",
        "content": "{\"action\": \"ask\", \"reply_to_client\": \"That must be incredibly frightening — having police at your door is one of the most stressful things anyone can face. You came to the right place and we can help you understand your options. Can you tell me: have you already received a formal written notice from the police (called a Section 41A notice), or did they just come without any written document?\"}"
      },
      {
        "role": "user",
        "content": "They gave a written notice to appear before them. Me and my parents both."
      },
      {
        "role": "assistant",
        "content": "{\"action\": \"ask\", \"reply_to_client\": \"Okay — that's actually an important detail that works in your favour right now. The fact that they issued a written notice rather than arresting you outright means they are (correctly) following Supreme Court guidelines from the Arnesh Kumar case. Before I explain what to do, tell me: what exactly has your wife alleged? Is it about dowry demands, physical violence, mental cruelty, or something else?\"}"
      },
      {
        "role": "user",
        "content": "She says my parents demanded dowry and I mentally harassed her. None of it is true. We had marital problems but no dowry was ever demanded. We have WhatsApp messages where she is very affectionate just 2 months ago."
      },
      {
        "role": "assistant",
        "content": "{\"action\": \"ask\", \"reply_to_client\": \"Thank you for sharing that. It's very common for S.498A cases to be filed during or after separation, and the courts are aware of this pattern — the Arnesh Kumar judgment directly addresses over-broad use of arrest powers in such cases. Your WhatsApp messages could be very relevant evidence. What outcome are you looking for right now? Are you looking for anticipatory bail so that you and your parents cannot be arrested, or do you also want to challenge the FIR itself?\"}"
      },
      {
        "role": "user",
        "content": "Both. First protect us from arrest, then fight the false case."
      },
      {
        "role": "assistant",
        "content": "{\"action\": \"complete\", \"intent\": \"legal_opinion\", \"facts_summary\": \"Client Rajesh, 34, software engineer in Pune. Wife has filed FIR under S.498A IPC alleging dowry demand and mental harassment against husband and in-laws. Police issued S.41A written notice to husband and parents. Client denies all allegations — says marriage was troubled but no dowry demand was ever made. Has WhatsApp messages showing wife was affectionate 2 months before the FIR. Prayer: (1) anticipatory bail for himself and parents to prevent arrest; (2) challenge/quash the FIR on merits. No monetary prayer.\", \"reply_to_client\": \"I have a clear picture now. The good news is that the law — especially the Supreme Court's Arnesh Kumar guidelines — provides strong protection against automatic arrest in S.498A cases. Let me prepare a detailed opinion on your two prayers: anticipatory bail and quashing of the FIR. I'll come back to you shortly.\"}"
      }
    ]
  },

  "advocate_reasoning": {
    "initial_assessment": "Client has received S.41A notice — this means police are following correct procedure. Arrest is not imminent but could happen after the notice reply. Priority: anticipatory bail before appearing to police. Secondary: explore quashing grounds under S.482 CrPC.",
    "key_legal_issues": [
      "Whether anticipatory bail under S.438 CrPC is maintainable at this pre-arrest stage.",
      "Whether the FIR discloses any cognizable offence beyond a matrimonial dispute.",
      "Applicability of Arnesh Kumar guidelines to restrict arbitrary arrest.",
      "Whether exculpatory WhatsApp evidence can support quashing."
    ],
    "risk_assessment": "Medium risk of arrest if AB application is not filed before appearing to police. Courts are sensitive to S.498A misuse (Rajesh Sharma, 2017 guidelines). Prior case law strongly supports granting AB in absence of criminal antecedents.",
    "strategy": "1. File AB application before Sessions Court under S.438 CrPC immediately. 2. Gather evidence: WhatsApp messages, character witnesses, proof no dowry articles were given or demanded. 3. Once AB is secured, consider S.482 CrPC quashing petition in High Court citing Arnesh Kumar and lack of specific allegations."
  },

  "final_opinion": {
    "bare_acts_cited": [
      {
        "act": "Code of Criminal Procedure, 1973",
        "section": "Section 438",
        "text_excerpt": "When any person has reason to believe that he may be arrested on an accusation of having committed a non-bailable offence, he may apply to the High Court or the Court of Session for a direction that in the event of such arrest he shall be released on bail.",
        "relevance": "Primary relief sought — anticipatory bail"
      },
      {
        "act": "Code of Criminal Procedure, 1973",
        "section": "Section 482",
        "text_excerpt": "Nothing in this Code shall be deemed to limit or affect the inherent powers of the High Court to make such orders as may be necessary to give effect to any order under this Code, or to prevent abuse of the process of any Court.",
        "relevance": "Quashing of FIR if allegations are ex-facie false"
      },
      {
        "act": "Indian Penal Code, 1860",
        "section": "Section 498A",
        "text_excerpt": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished...",
        "relevance": "The section under which FIR was registered"
      }
    ],

    "case_laws_cited": [
      {
        "case_name": "Arnesh Kumar vs State of Bihar",
        "citation": "AIR 2014 SC 2756",
        "court": "Supreme Court of India",
        "year": "2014",
        "principle": "Police must record reasons before arresting in offences punishable up to 7 years. S.41A notice must be issued first. Magistrate must apply independent mind before remanding accused.",
        "how_applied": "Client has already received S.41A notice — police are following the Arnesh Kumar mandate. AB court can be shown that police themselves have not found immediate arrest necessary."
      },
      {
        "case_name": "Rajesh Sharma & Ors. vs State of U.P.",
        "citation": "(2017) 12 SCC 256",
        "court": "Supreme Court of India",
        "year": "2017",
        "principle": "Family Welfare Committees to be set up; no automatic arrest in S.498A cases; cases to be scrutinised before arrest.",
        "how_applied": "Supports the argument that the FIR should not lead to immediate arrest and deserves careful scrutiny."
      }
    ],

    "opinion_text": "LEGAL OPINION\n\n**Facts of the Case**\nMr. Rajesh (34, software engineer, Pune) and his parents have been named in an FIR under Section 498A IPC filed by his wife. The FIR alleges dowry demand and mental harassment. The police have served a Section 41A CrPC written notice on the accused persons. The client denies all allegations and has documentary evidence (WhatsApp messages) showing the wife was affectionate as recently as two months before the FIR.\n\n**Disputes Identified**\n1. Immediate risk of arrest and need for pre-arrest bail protection\n2. Validity and merits of the S.498A FIR — whether it discloses a genuine offence\n\n**Legal Protection — Dispute 1: Anticipatory Bail**\nSection 438 CrPC provides for anticipatory bail when a person apprehends arrest in a non-bailable case. The Supreme Court in Arnesh Kumar v. State of Bihar (AIR 2014 SC 2756) mandated that police must issue S.41A notice (which has been done here) and record reasons before proceeding to arrest. The fact that the police have not arrested despite the notice indicates they do not presently find arrest necessary — this strengthens the AB application. An AB should be filed in the Sessions Court immediately, before the client appears before police.\n\n**Legal Protection — Dispute 2: Challenging the FIR**\nThe High Court's inherent jurisdiction under S.482 CrPC can be invoked to quash an FIR where allegations are either vague or constitute a matrimonial dispute dressed up as a criminal case. The Rajesh Sharma guidelines (2017) and the observations in Arnesh Kumar recognise that S.498A is misused to harass husbands and in-laws. The WhatsApp messages showing affection two months prior directly contradict the claim of continuous cruelty and can support a quashing petition.\n\n**Reliefs Sought & Assessment**\n- Anticipatory bail: Strong case — no criminal antecedents, police following S.41A protocol, Arnesh Kumar principles apply. Likelihood of grant: High.\n- Quashing FIR: Moderate case — success depends on whether the FIR contains specific dated allegations or is omnibus. Exculpatory WhatsApp messages are helpful but courts are cautious about quashing at investigation stage.\n\n**Next Steps**\n1. File anticipatory bail application in Sessions Court within 2–3 days — before appearing before police.\n2. Collect and preserve all WhatsApp messages, call records, and any evidence of the wife's conduct.\n3. Obtain character certificates and evidence that no dowry articles were given.\n4. After AB is secured, instruct a High Court advocate to evaluate the FIR for S.482 quashing.\n\nThe law provides you real protection here. The Supreme Court has consistently said that arrest is not a punishment — it is a necessity to be used sparingly. You are not powerless in this situation."
  },

  // ──────────────────────────────────────────────────────────────────
  // COMPATIBILITY — kept for few_shot_retriever.py and convert_to_finetune.py
  // ──────────────────────────────────────────────────────────────────
  "id": "rich_001",
  "case_type": "criminal_bail_498a",
  "description": "498A IPC case against husband and in-laws; police notice received; anticipatory bail and FIR quashing sought",
  "keywords": ["498A", "anticipatory bail", "CrPC 41A", "S.438", "S.482", "dowry", "false case", "arrest", "quashing"]
}
```

---

## Fields Reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | ✓ | `rich_NNN` format |
| `source_case_id` | string | ✓ | Matches `case_id` in top_200_cited_cases.jsonl |
| `legal_domain` | enum | ✓ | criminal / civil / matrimonial / labour / property / consumer / constitutional / motor_accident / cheque_bounce |
| `dispute_type` | string | ✓ | Fine-grained label |
| `relevant_bare_acts[]` | array | ✓ | Act name, sections, purpose |
| `case_analysis.disputes_identified[]` | array | ✓ | Numbered disputes |
| `case_analysis.petitioner_arguments[]` | array | ✓ | String list |
| `case_analysis.respondent_arguments[]` | array | ✓ | String list |
| `case_analysis.judicial_observations[]` | array | ✓ | Direct extracts or close paraphrase |
| `case_analysis.principles_laid_down[]` | array | ✓ | Ratio decidendi / obiter dicta |
| `case_analysis.order_outcome` | enum | ✓ | See values above |
| `client_scenario.emotional_state` | enum | ✓ | fearful / distressed / angry / confused / resigned / hopeful |
| `intake_conversation.conversation[]` | array | ✓ | role + content turns |
| `advocate_reasoning` | object | ✓ | initial_assessment, key_legal_issues, risk_assessment, strategy |
| `final_opinion.opinion_text` | string | ✓ | Full formatted opinion |
| `keywords` | array | ✓ | Used by few_shot_retriever.py |
| `case_type` | string | ✓ | Used by convert_to_finetune.py |

---

## Coverage Plan — 100 Cases (domain distribution)

| Domain | Target | Source from top_200 |
|---|---|---|
| Criminal (bail, quashing, S.498A, anticipatory bail) | 18 | Criminal (14) + General overlap |
| Matrimonial (maintenance, DV, custody, divorce) | 15 | Matrimonial (9) + General overlap |
| Property (encroachment, title, specific performance) | 12 | Civil property cases |
| Landlord–Tenant (eviction, rent control, deposit) | 12 | Civil Landlord-Tenant (25) |
| Labour (termination, gratuity, ESIC, workmen's comp) | 10 | General overlap |
| Motor Accident (MACT, compensation, multiplier) | 10 | General motor accident cases |
| Cheque Bounce (NI Act S.138, S.142, compounding) | 8 | General cheque bounce |
| Consumer (deficiency, NCDRC, jurisdiction) | 8 | General consumer cases |
| Constitutional (fundamental rights, writ, PIL) | 7 | General constitutional |

Total: **100**
