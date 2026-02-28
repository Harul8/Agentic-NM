# NyaymaLaw — Accuracy & Style Improvement Guide
## A Practical, Phase-by-Phase Plan (No Fine-Tuning)

> **Core principle**: The system improves in two directions — *breadth* (covering more of Indian law) and *depth* (reasoning better about what it already covers). Both are driven by real usage, not guesswork.

---

## The Feedback Loop That Drives Everything

Before diving into phases, understand the engine behind all improvement:

```
Real query → System answers → You review → Identify failure → Fix root cause → Re-test
```

Every improvement activity in this guide feeds back into this loop. The faster you run this loop, the faster the system improves. The goal by the end of Phase 3 is to be running this loop weekly.

---

## Phase 1 — Know Where You Stand (Weeks 1–2)

You cannot improve what you cannot measure. This phase is entirely about establishing a baseline.

### Activity 1.1 — Build Your Golden Test Set

Create a file called `eval/golden_test_set.json` with 50–80 realistic disputes. These should cover:
- Criminal assault / hurt (your current test case)
- Property sale and encroachment
- Cheque bounce / financial disputes
- Landlord-tenant
- Employment termination
- Consumer complaints
- Family disputes (maintenance, divorce)
- Contract breach

For each dispute, manually write down the *expected* outputs:
- Which acts should be identified (e.g., BNS 2023, TP Act 1882)
- Which sections are most relevant (e.g., Section 115, 117 for grievous hurt)
- What the correct legal nature is (criminal / civil / both)
- What the remedy is

This is your ground truth. You only need to do this once, and it becomes the measuring stick for every change you make from here on.

### Activity 1.2 — Run Your System Against the Test Set

Run all 50 queries through your current system and score each one:

| Criterion | Pass / Fail |
|---|---|
| Identified the correct primary act? | |
| Retrieved at least one relevant section? | |
| Legal nature correctly classified? | |
| Response mentioned the correct remedy? | |
| No hallucinated section numbers? | |

Calculate a baseline accuracy score — even a rough percentage per category is enough. Example: "Act identification: 60%, Section relevance: 45%, Remedy guidance: 70%."

Write this down. This is your baseline.

### Activity 1.3 — Categorise Your Failures

Group every failure from the test run into one of these root causes:

- **Coverage gap** — The act is simply not in your index (e.g., BNS was missing)
- **Retrieval failure** — The act is in the index but the wrong sections came back
- **Query understanding failure** — The decomposer misclassified the dispute or split it wrong
- **Prompt failure** — The right sections were retrieved but the response was poorly structured
- **Style failure** — The answer was factually okay but didn't read like a proper legal opinion

This categorisation tells you exactly where to spend your time in the phases that follow.

---

## Phase 2 — Fill the Knowledge Base (Weeks 2–6, then Ongoing)

The most impactful thing you can do right now is systematic coverage. Your system cannot retrieve what it hasn't indexed.

### Activity 2.1 — Priority Coverage Map

Rank the acts by how frequently they appear in Indian disputes. A practical starting priority:

**Tier 1 — Must have immediately** (highest dispute frequency):
- Bharatiya Nyaya Sanhita 2023 (replaces IPC — all criminal offences)
- Bharatiya Nagarik Suraksha Sanhita 2023 (replaces CrPC — all criminal procedure)
- Bharatiya Sakshya Adhiniyam 2023 (replaces Indian Evidence Act)
- Transfer of Property Act 1882
- Specific Relief Act 1963
- Negotiable Instruments Act 1881 (cheque bounce is 30%+ of civil litigation)
- Code of Civil Procedure 1908
- Indian Contract Act 1872
- Hindu Marriage Act 1955
- Registration Act 1908

**Tier 2 — Index within 2 months**:
- Consumer Protection Act 2019
- Arbitration & Conciliation Act 1996
- Hindu Succession Act 1956
- Protection of Women from Domestic Violence Act 2005
- Motor Vehicles Act 1988
- Limitation Act 1963

**Tier 3 — State-specific (Telangana)**:
- Telangana Land Encroachment Act
- Telangana Tenancy Act
- Telangana Rent Control Act
- AP / Telangana Municipalities Act

### Activity 2.2 — Case Laws Strategy

Case laws improve the system differently from bare acts. A bare act tells you *what the law says*; a case law tells you *how courts interpret it*. Both are needed for a complete answer.

For case laws, don't try to index everything — it's millions of judgments. Instead index *selectively*:

- **Landmark Supreme Court judgments** — The 20–30 most-cited cases per domain. These are the ones that every advocate references.
- **Section-level interpretation judgments** — For each important section in your index, add 2–3 judgments that interpreted that specific section. Tag them with the section number.
- **Recent High Court judgments** — The last 2 years of Telangana / Hyderabad High Court orders for your core dispute types. These are most relevant to your likely clients.

Source: IndiaKanoon.org allows bulk access. For landmark cases, LexisNexis and SCC Online are authoritative but paid.

### Activity 2.3 — Chunking Strategy for Case Laws

Case laws need different chunking than bare acts. A judgment has:
- Facts (what happened)
- Issues (what legal questions were decided)
- Held (the actual decision and its reasoning)
- Ratio (the principle of law established)

When you index a case law, chunk it so the *Held* and *Ratio* sections are their own chunks with metadata: `case_name`, `citation`, `court`, `year`, `acts_cited`, `sections_cited`. This way a query about "Section 138 NI Act dishonour" retrieves the *held* portion of relevant judgments, not the 10-page fact summary.

---

## Phase 3 — Improve Retrieval Quality (Month 2)

Once your coverage is good, the next bottleneck is whether the right sections are coming back for a given query.

### Activity 3.1 — Review the Cross-Encoder Score Distribution

Run your test set again after indexing the Tier 1 acts. Look at the rerank scores. You want to see:
- Scores > 8.0 for the most relevant sections (your current threshold for "exact match")
- Scores 5–7 for supporting sections
- Scores < 3.0 for anything that shouldn't have been retrieved

If you're seeing genuinely relevant sections coming back with scores of 3–4, your chunking may be too coarse (chunks are too long, diluting signal) or your query generation isn't covering the right vocabulary.

### Activity 3.2 — Chunk Size Tuning

For bare acts, the right chunk is typically one section at a time — not a group of sections, not a paragraph within a section. Each chunk should have:
- The full section number and title as metadata
- The complete text of that section
- A note of the parent act and chapter

If a section is very long (more than ~600 tokens), split it at the sub-section level. Retrieval degrades badly when chunks are too large.

### Activity 3.3 — Query Diversity in Round 1

Your Round 1 search currently generates 4 queries for a dispute. Review whether those 4 angles are genuinely diverse. For a cheque bounce dispute, you want:
1. The primary right ("right to payment on dishonour of negotiable instrument")
2. The offence angle ("criminal liability default payment drawer cheque")
3. The remedy angle ("compensation summary trial dishonoured cheque India")
4. The act-specific angle ("Negotiable Instruments Act Section 138 punishment")

If all 4 queries are paraphrases of the same idea, retrieval will be narrow. The `BARE_ACT_SEARCH_QUERIES_PROMPT` in your codebase controls this — review and improve it for the domains where you're seeing retrieval failure.

### Activity 3.4 — Metadata Filtering

Tag every indexed section with structured metadata and use it as a hard pre-filter:
- `legal_domain`: criminal / property / family / contract / consumer / procedure / evidence
- `applicable_states`: central / Telangana / AP / Karnataka etc.
- `act_year`: year of the act (helps distinguish BNS 2023 from IPC 1860)
- `in_force`: true / false (tracks whether the act has been repealed)

When the decomposer classifies a dispute as "civil / property", you can hard-filter to only retrieve from `legal_domain: property` chunks before running the expensive cross-encoder reranker. This dramatically improves both speed and precision.

---

## Phase 4 — Prompt Engineering Iteration (Month 2–3)

Your prompts are the intelligence layer. Every failure that isn't a coverage or retrieval failure is a prompt failure — and prompt failures are the cheapest to fix.

### Activity 4.1 — Maintain a Prompt Failure Log

Whenever the system gives a wrong answer and the right sections *were* retrieved, log it as a prompt failure. Note:
- What sections were retrieved (good)
- What the system said (wrong)
- What it should have said (correct)
- Which prompt caused the failure (decomposer / query generation / sufficiency check / response generation)

Review this log weekly. When you see 3+ failures from the same prompt, rewrite that prompt.

### Activity 4.2 — Decomposer Prompt Refinement

The decomposer is the most important prompt in your system — everything downstream depends on it getting the dispute components right. Things to test:

- Does it correctly split criminal + civil components in a single dispute?
- Does it use the new criminal code names (BNS/BNSS/BSA) in `bare_act_hints` rather than old names?
- Does it correctly classify purely civil property disputes as `legal_nature: civil` rather than `both`?
- Does it over-decompose? (splitting one dispute into 4 when 2 would be right)

Each of these is a testable behaviour. Add examples to your golden test set that specifically probe each one.

### Activity 4.3 — Response Style Iteration

The response generation prompt controls tone and style. Things to refine over time:
- Does it cite section numbers accurately (never hallucinate, always pull from retrieved text)?
- Does it give a practical "what to do next" alongside the legal analysis?
- Does it flag when a dispute needs actual legal representation (criminal matters, court filings)?
- Does it give the right level of detail — not too academic, not too vague?

The best way to improve this prompt is to take 10 responses the system gave that you were unhappy with and rewrite them as you would want an ideal answer to look like. Then look for the pattern in what the prompt needs to say to produce that style consistently.

---

## Phase 5 — Build a Feedback Loop into the Product (Month 3)

This is the phase that makes improvement self-sustaining.

### Activity 5.1 — Capture Every Query and Response

Log to a file or database:
- The full query text
- The decomposed disputes
- Which acts and sections were retrieved (with scores)
- The final response
- Timestamp

This is your training data for the future, and your audit trail for debugging failures.

### Activity 5.2 — Add Explicit User Feedback

Add a simple rating to your UI: 👍 / 👎 / ✏️ (correct / wrong / partially right). Even two options is enough. When a user marks something wrong, also capture *why* if possible (wrong act, missing a dispute, wrong remedy, etc.).

Every thumbs-down is a data point that tells you exactly where the system failed in production with a real query.

### Activity 5.3 — Weekly Review Ritual

Set aside 1 hour per week:
1. Review all 👎 responses from the past week
2. For each, trace the failure to its root cause (coverage / retrieval / prompt / style)
3. Pick the top 2–3 failures and fix them (index a new act, adjust a prompt, fix a chunk)
4. Re-run the golden test set to confirm the fix didn't break anything else

After 3 months of this, your system will be dramatically more accurate than it started — not because of any single big change, but because of 12 weeks of small, targeted corrections.

---

## Phase 6 — Style and Advocate Voice (Month 3–4)

This is about making the system sound like a knowledgeable Indian advocate, not a legal encyclopedia.

### Activity 6.1 — Advocate Style Checklist

Review 20 responses and check for these qualities:
- Does it lead with the most actionable answer, not the most general one?
- Does it speak to the client's specific facts, not just recite the law?
- Does it acknowledge when there are counter-arguments or grey areas?
- Does it give a practical path forward ("file a complaint under Section X at the local magistrate court")?
- Does it appropriately hedge on highly fact-dependent questions ("this would depend on whether...")?
- Does it use plain language for the remedy even when citing technical section numbers?

### Activity 6.2 — Add a Practical "Next Step" Layer

After the legal analysis, the system should always offer a concrete next step. This is what a real advocate does:
- For criminal assault: "File a complaint (FIR) at the local police station under BNS Section 115/117. If the police refuse, you can file a private complaint directly before the Magistrate under Section 223 BNSS."
- For cheque bounce: "Send a legal notice within 30 days of dishonour. If payment is not received within 15 days, file a complaint under Section 138 NI Act in the Magistrate court having jurisdiction."

This requires enriching your response generation prompt with instructions to always include a "what to do now" section.

### Activity 6.3 — Disclaimer and Scope Calibration

The system should be clear about what it can and cannot do:
- It can explain what the law says and what options exist
- It cannot replace a lawyer for actual court proceedings, drafting, or representation
- It should flag when a dispute is serious enough to need immediate legal representation (criminal matters involving arrest, custody, property of high value)

---

## Ongoing Cadence (Month 4+)

Once the phases above are complete, shift to a maintenance cadence:

**Weekly**: Review 👎 feedback, fix top 2 failures, update the prompt failure log.

**Monthly**: Add 1–2 new acts or a batch of case laws from the coverage gap list. Run the full golden test set. Check whether accuracy scores improved.

**Quarterly**: Do a full review of the golden test set — add new dispute types that came up in real usage, retire test cases that are no longer representative. Raise the bar on what "pass" means as the system improves.

**Annually**: Consider whether fine-tuning is now worthwhile. By this point you'll have hundreds of real (query, ideal response) pairs from production — which is exactly the data fine-tuning needs. The system you've built will also tell you precisely which failure modes fine-tuning would and wouldn't fix.

---

## Summary: What Drives Accuracy at Each Stage

| Stage | What limits accuracy | What fixes it |
|---|---|---|
| Today | Missing acts (BNS, TP Act, CPC) | Index Tier 1 acts immediately |
| Month 1 | Poor retrieval for covered acts | Chunking review + query diversity |
| Month 2 | Decomposer misclassifying disputes | Prompt iteration + failure log |
| Month 3 | Response style and hallucination | Response prompt + feedback loop |
| Month 4+ | Edge cases and rare dispute types | Production feedback + case laws |

The system improves fastest when you fix coverage gaps first (cheap, high impact), then retrieval quality (medium effort), then prompts (iterative, ongoing). Style and voice come last because they only matter once the factual accuracy is solid.
