# Nyaymalaw 3.0 — End-to-End Workflow (Proposed Overhaul)

---

## SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NYAYMALAW 3.0 ARCHITECTURE                       │
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────────┐ │
│  │  React    │──▶│  FastAPI      │──▶│ Consolidated│──▶│ Ollama        │ │
│  │  Frontend │◀──│  Backend      │◀──│ Agent Layer │◀──│ (Qwen2.5 7B) │ │
│  └──────────┘   └──────┬───────┘   └─────┬──────┘   └───────────────┘ │
│                        │                  │                             │
│                  ┌─────▼──────┐    ┌──────▼───────┐                    │
│                  │  SQLite    │    │ FAISS + BM25 │                    │
│                  │  (Users,   │    │ Hybrid Store │                    │
│                  │   Chats,   │    │ (Bare Acts,  │                    │
│                  │   Billing) │    │  Case Laws)  │                    │
│                  └────────────┘    └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DATA SOURCING HIERARCHY (Critical Governance Layer)

The system follows a strict, tiered data sourcing strategy. This ensures
accuracy, authority, and trustworthiness of every legal reference.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA SOURCE PRIORITY (Top to Bottom)                  │
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  TIER 1: GOOGLE DRIVE (Primary — Always searched first)          ║  │
│  ║                                                                   ║  │
│  ║  Location: NYAYMALAW_DATA_ROOT (Google Drive mounted folder)      ║  │
│  ║                                                                   ║  │
│  ║  ┌─────────────────────┐    ┌─────────────────────┐              ║  │
│  ║  │  BareActs/          │    │  CaseLaws/           │              ║  │
│  ║  │  (PDF files of      │    │  (PDF files of       │              ║  │
│  ║  │   Indian statutes)  │    │   court judgments)   │              ║  │
│  ║  └─────────┬───────────┘    └──────────┬──────────┘              ║  │
│  ║            │                            │                         ║  │
│  ║            ▼                            ▼                         ║  │
│  ║  ┌─────────────────────────────────────────────────┐             ║  │
│  ║  │  vector_store/                                   │             ║  │
│  ║  │  • bareacts.index  + bareacts_chunks.json        │             ║  │
│  ║  │  • caselaws.index  + caselaws_chunks.json        │             ║  │
│  ║  │                                                  │             ║  │
│  ║  │  FAISS + BM25 hybrid search over these files     │             ║  │
│  ║  └─────────────────────────────────────────────────┘             ║  │
│  ║                                                                   ║  │
│  ║  Rule: ALL queries hit Google Drive vector store FIRST.           ║  │
│  ║  The model determines sufficiency — NO hard numeric limits.       ║  │
│  ║  If all relevant bare acts for the dispute are found locally      ║  │
│  ║  AND case laws exist for each applicable section,                 ║  │
│  ║  STOP HERE. Do not go to internet.                               ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                         │
│                     Insufficient results?                              │
│                              │                                         │
│                              ▼                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  TIER 2: OFFICIAL COURT WEBSITES (First internet source)         ║  │
│  ║                                                                   ║  │
│  ║  Priority order:                                                  ║  │
│  ║                                                                   ║  │
│  ║  2a. Supreme Court of India                                       ║  │
│  ║      • main.sci.gov.in (judgments)                                ║  │
│  ║      • api.sci.gov.in  (API/PDF downloads)                        ║  │
│  ║                                                                   ║  │
│  ║  2b. Relevant High Court (based on user's jurisdiction)           ║  │
│  ║      • Karnataka HC: karnatakajudiciary.kar.nic.in                ║  │
│  ║      • Bombay HC: bombayhighcourt.nic.in                          ║  │
│  ║      • Delhi HC: delhihighcourt.nic.in                            ║  │
│  ║      • Madras HC: mhc.tn.gov.in                                  ║  │
│  ║      • Allahabad HC: allahabadhighcourt.in                        ║  │
│  ║      • (all other HCs mapped by state)                            ║  │
│  ║                                                                   ║  │
│  ║  2c. India Code (for bare acts)                                   ║  │
│  ║      • indiacode.nic.in (official legislative repository)         ║  │
│  ║                                                                   ║  │
│  ║  Rule: If original judgment PDFs are available on court            ║  │
│  ║  websites → DOWNLOAD PDF → Save to Google Drive CaseLaws/         ║  │
│  ║  → Index into vector store automatically.                         ║  │
│  ║  This grows the local database over time.                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                         │
│                     Still insufficient?                                │
│                              │                                         │
│                              ▼                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  TIER 3: TRUSTED LEGAL PORTALS (Verified legal sources)          ║  │
│  ║                                                                   ║  │
│  ║  ALLOWED PORTALS (whitelist):                                     ║  │
│  ║  ┌────────────────────────────────────────────────────────┐      ║  │
│  ║  │  • indiankanoon.org     — Largest Indian case law DB   │      ║  │
│  ║  │  • livelaw.in           — Legal news & judgments       │      ║  │
│  ║  │  • scobserver.in        — SC analysis & case tracker   │      ║  │
│  ║  │  • barandbench.com      — Legal journalism             │      ║  │
│  ║  │  • lawctopus.com        — Legal education & resources  │      ║  │
│  ║  │  • scconline.com        — SCC Online (if accessible)   │      ║  │
│  ║  │  • casemine.com         — Case law search engine       │      ║  │
│  ║  │  • legalbites.in        — Legal articles & analysis    │      ║  │
│  ║  │  • latestlaws.com       — Updated Indian laws          │      ║  │
│  ║  └────────────────────────────────────────────────────────┘      ║  │
│  ║                                                                   ║  │
│  ║  Rule: If portal links to original judgment PDF                   ║  │
│  ║  → Download PDF → Save to Google Drive → Index to vector store.   ║  │
│  ║  If no PDF available → Use article/analysis as reference          ║  │
│  ║  but MARK SOURCE clearly as "Legal Portal Analysis" (not          ║  │
│  ║  primary source).                                                 ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                         │
│                     Still insufficient?                                │
│                              │                                         │
│                              ▼                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  TIER 4: MAINSTREAM NEWSPAPERS (Last resort — analysis only)     ║  │
│  ║                                                                   ║  │
│  ║  ALLOWED NEWSPAPERS (whitelist):                                  ║  │
│  ║  ┌────────────────────────────────────────────────────────┐      ║  │
│  ║  │  • thehindu.com          — The Hindu                   │      ║  │
│  ║  │  • indianexpress.com     — Indian Express              │      ║  │
│  ║  │  • timesofindia.com      — Times of India              │      ║  │
│  ║  │  • ndtv.com              — NDTV                        │      ║  │
│  ║  │  • hindustantimes.com    — Hindustan Times             │      ║  │
│  ║  │  • deccanherald.com      — Deccan Herald               │      ║  │
│  ║  │  • theprint.in           — The Print                   │      ║  │
│  ║  │  • thewire.in            — The Wire                    │      ║  │
│  ║  │  • economictimes.com     — Economic Times              │      ║  │
│  ║  └────────────────────────────────────────────────────────┘      ║  │
│  ║                                                                   ║  │
│  ║  Rule: Newspaper articles are NEVER treated as primary legal      ║  │
│  ║  authority. They provide context and background only.             ║  │
│  ║  ALWAYS marked as "News Reference" in citations.                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  BLOCKED SOURCES (Never used, never referenced)                  ║  │
│  ║                                                                   ║  │
│  ║  ❌ Twitter/X — Individual opinions, unverified                   ║  │
│  ║  ❌ Facebook — Social media posts, unverified                     ║  │
│  ║  ❌ LinkedIn — Individual articles, not authoritative             ║  │
│  ║  ❌ Instagram — No legal authority                                ║  │
│  ║  ❌ YouTube — Transcripts unreliable for legal citation           ║  │
│  ║  ❌ Reddit — Anonymous opinions                                   ║  │
│  ║  ❌ Quora — User-generated answers, unverified                    ║  │
│  ║  ❌ Wikipedia — Not a primary legal source                        ║  │
│  ║  ❌ Personal blogs — Not authoritative                            ║  │
│  ║  ❌ Any domain not in the whitelists above                        ║  │
│  ║                                                                   ║  │
│  ║  If DuckDuckGo returns results from blocked sources,              ║  │
│  ║  they are SILENTLY DISCARDED. Never shown to user.                ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Enrichment Loop (Self-Growing Database)

```
Every time the system fetches from internet (Tier 2, 3, or 4):
        │
        ▼
┌───────────────────────────────────────────┐
│ Is original judgment/act PDF available?    │
│                                           │
│ YES:                                      │
│  1. Download PDF                          │
│  2. Save to Google Drive:                 │
│     • BareActs/ (if statute)              │
│     • CaseLaws/ (if judgment)             │
│  3. Chunk + embed + index into FAISS      │
│  4. Next time same topic is queried,      │
│     it will be found in Tier 1 (local)    │
│                                           │
│ NO (only article/analysis available):     │
│  1. Use content as reference              │
│  2. Mark clearly as "Portal Analysis"     │
│     or "News Reference" — NOT primary     │
│  3. Do NOT index articles as if they      │
│     were judgments or bare acts            │
│  4. Store article metadata separately     │
│     in a references table for audit       │
└───────────────────────────────────────────┘

Over time, this creates a SELF-GROWING legal database:
  • Week 1:  20 documents (manually uploaded)
  • Month 1: 100+ documents (auto-enriched from queries)
  • Month 6: 500+ documents (comprehensive coverage)
  • The more users query, the richer the database becomes.
```

### Source Attribution in Responses

```
Every citation in the final output carries a SOURCE TAG:

┌──────────────────────────────────────────────────────────┐
│ SOURCE TAGS (Visible to user)                            │
│                                                          │
│ [LOCAL DB]          — From Google Drive vector store     │
│                       (highest trust, already verified)  │
│                                                          │
│ [OFFICIAL COURT]    — Downloaded from SC/HC website      │
│                       (primary authority, auto-saved     │
│                        to Google Drive)                  │
│                                                          │
│ [LEGAL PORTAL]      — From Indian Kanoon, Live Law, etc. │
│                       (trusted but secondary source)     │
│                                                          │
│ [NEWS REFERENCE]    — From newspapers                    │
│                       (context only, NOT legal authority)│
│                                                          │
│ User sees exactly WHERE each piece of information        │
│ came from, building trust and verifiability.             │
└──────────────────────────────────────────────────────────┘
```

---

## PHASE 0: USER ENTRY & AUTHENTICATION

```
User visits Nyaymalaw.in
        │
        ▼
┌─────────────────────┐
│  LANDING PAGE        │
│  • What is Nyaymalaw │
│  • Features overview │
│  • Try Free / Login  │
└─────────┬───────────┘
          │
          ▼
    ┌───────────┐     ┌─────────────────────────┐
    │ Guest?    │─Yes─▶│ Limited session          │
    │           │      │ • 3 queries/day          │
    │           │      │ • No chat history saved  │
    │           │      │ • No document drafting   │
    └─────┬─────┘      └─────────────────────────┘
          │No
          ▼
    ┌───────────┐     ┌──────────────────────────┐
    │ Register/ │────▶│ Authentication            │
    │ Login     │     │ • Email + Password        │
    │           │     │ • OTP verification        │
    │           │     │ • JWT session token       │
    └───────────┘     └──────────┬───────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ USER TIER CHECK       │
                      │                      │
                      │ FREE tier:           │
                      │  • 5 queries/day     │
                      │  • Basic search      │
                      │  • Opinion generation│
                      │  • No doc drafting   │
                      │                      │
                      │ PREMIUM tier:        │
                      │  • 25 queries/day    │
                      │  • Advanced research │
                      │  • Full case analysis│
                      │  • Court-ready docs  │
                      │  • Priority queue    │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ CHAT INTERFACE LOADS  │
                      │ • Past chats sidebar │
                      │ • New chat ready     │
                      └──────────────────────┘
```

---

## PHASE 1: CONVERSATIONAL ENTRY (The Natural Advocate)

This is the critical UX improvement. The system should feel like walking into
a senior advocate's chamber — warm, professional, perceptive.

```
User sends first message
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                  INTENT CLASSIFIER                        │
│                                                          │
│  Classifies user message into one of:                    │
│                                                          │
│  1. GREETING        → "Hi", "Hello", "Good morning"     │
│  2. SMALL_TALK      → "How are you?", "What can you do?"│
│  3. LEGAL_QUERY     → Describes a legal situation        │
│  4. DIRECT_SEARCH   → "Find Section 498A IPC"           │
│  5. FOLLOW_UP       → Continues an existing conversation│
│  6. OFF_TOPIC       → Non-legal, unrelated queries       │
│  7. FEEDBACK        → Complaints, suggestions            │
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┼─────────────────────────┐
         │             │                         │
         ▼             ▼                         ▼
    ┌─────────┐  ┌───────────┐          ┌──────────────┐
    │GREETING │  │LEGAL_QUERY│          │DIRECT_SEARCH │
    │         │  │           │          │              │
    │Response:│  │Go to      │          │Skip intake,  │
    │Warm,    │  │Phase 2    │          │go directly   │
    │natural, │  │(Fact      │          │to retrieval  │
    │advocate-│  │Collection)│          │(Phase 3)     │
    │like     │  │           │          │              │
    │greeting │  └───────────┘          └──────────────┘
    │+ "How   │
    │can I    │
    │help     │
    │today?"  │
    └─────────┘

GREETING/SMALL_TALK EXAMPLES:
─────────────────────────────
User: "Hi"
Bot:  "Good morning! Welcome to Nyaymalaw. I'm here to help you with
      legal research and guidance. What legal matter can I assist
      you with today?"

User: "What can you do?"
Bot:  "I can help you research Indian law — from finding relevant
      sections in bare acts to analyzing case law precedents. I can
      also help draft court documents like petitions and affidavits.
      Just tell me about your legal situation in your own words,
      and we'll take it from there."

User: "Thanks for the help earlier"
Bot:  "You're most welcome! Happy I could help. Do you have another
      legal matter you'd like to discuss, or shall we continue where
      we left off?"
```

---

## PHASE 2: SMART FACT COLLECTION (Advocate Interview)

```
User describes legal situation (plain language)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              INTAKE AGENT (Consolidated)                   │
│                                                          │
│  Combines: Facts Collector + Normalizer + Contradiction  │
│                                                          │
│  Step 1: EXTRACT key facts from user's narrative         │
│          • Parties involved (names, relationships)       │
│          • Dates, locations, amounts                     │
│          • Nature of dispute/issue                       │
│          • Actions taken so far (FIR, notice, etc.)      │
│          • Desired outcome                               │
│                                                          │
│  Step 2: DETECT missing critical information             │
│          • Jurisdiction clues (state, district)          │
│          • Timeline gaps                                 │
│          • Contradictions in narrative                   │
│                                                          │
│  Step 3: DECIDE next action                              │
│          • Need more info? → Ask natural follow-up       │
│          • Facts sufficient? → Move to Phase 3           │
│          • User says "that's all"? → Proceed with what   │
│            we have, note gaps                            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ FOLLOW-UP QUESTIONS │
            │ (Natural, not       │
            │  templated)         │
            └─────────┬───────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   Need more     Facts ready    User stops
   info? ──┐     ──────────┐    ──────────┐
           │               │              │
    Ask next Q      Move to Phase 3  Move to Phase 3
    (loop back)     (complete)       (with gaps noted)

INTERVIEW EXAMPLE:
──────────────────
User: "My landlord is refusing to return my security deposit.
       It's been 3 months since I vacated."

Bot:  "I understand — that's a frustrating situation. Let me
       gather a few details so I can research this properly.

       Which city and state was the rented property in? And
       do you have a written rental agreement?"

User: "Bangalore, Karnataka. Yes, 11-month agreement."

Bot:  "Got it. And roughly how much is the security deposit
       we're talking about? Also, did your landlord give any
       reason for not returning it?"

User: "2 lakhs. He says there's damage but there isn't any."

Bot:  "Understood. One more thing — did you send any written
       notice or legal notice demanding the refund? That'll
       help me assess where things stand legally."

User: "No, not yet."

Bot:  "That's perfectly fine — we can work with this. Let me
       now research the relevant law and precedents for your
       situation."

      → PROCEED TO PHASE 3

FACTS STRUCTURED OUTPUT (Internal JSON):
─────────────────────────────────────────
{
  "parties": {
    "tenant": "User",
    "landlord": "Unnamed"
  },
  "dispute_type": "security_deposit_refund",
  "jurisdiction": {
    "state": "Karnataka",
    "city": "Bangalore"
  },
  "key_facts": [
    "11-month rental agreement existed",
    "Security deposit: INR 2,00,000",
    "Tenant vacated 3 months ago",
    "Landlord claims property damage",
    "Tenant disputes damage claim",
    "No legal notice sent yet"
  ],
  "missing_info": [
    "Exact date of vacating",
    "Whether agreement has deposit return clause"
  ],
  "desired_outcome": "Recovery of security deposit"
}
```

---

## PHASE 3: HYBRID RETRIEVAL (The Accuracy Fix)

```
Structured facts from Phase 2
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              QUERY EXPANSION (LLM-driven)                 │
│                                                          │
│  Converts plain facts into legal search queries:         │
│                                                          │
│  Input:  "Landlord won't return 2L deposit, Bangalore"   │
│  Output: [                                               │
│    "Karnataka Rent Control Act security deposit refund",  │
│    "Section 108 Transfer of Property Act obligations",    │
│    "tenant security deposit recovery civil suit",        │
│    "landlord deposit forfeiture damage deduction"        │
│  ]                                                       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│           HYBRID SEARCH (FAISS + BM25 + Re-rank)         │
│                                                          │
│  ┌─────────────────┐     ┌─────────────────┐            │
│  │  FAISS Vector    │     │  BM25 Keyword   │            │
│  │  Search          │     │  Search         │            │
│  │                  │     │                 │            │
│  │  Semantic match  │     │  Exact term     │            │
│  │  "meaning of     │     │  match on       │            │
│  │   the query"     │     │  section nums,  │            │
│  │                  │     │  act names,     │            │
│  │  Top 20 results  │     │  legal terms    │            │
│  └────────┬─────────┘     │                 │            │
│           │               │  Top 20 results │            │
│           │               └────────┬────────┘            │
│           │                        │                     │
│           └───────────┬────────────┘                     │
│                       ▼                                  │
│           ┌───────────────────────┐                      │
│           │  MERGE & DEDUPLICATE  │                      │
│           │  Combined candidate   │                      │
│           │  pool (~30-40 chunks) │                      │
│           └───────────┬───────────┘                      │
│                       │                                  │
│                       ▼                                  │
│           ┌───────────────────────┐                      │
│           │  CROSS-ENCODER        │                      │
│           │  RE-RANKER            │                      │
│           │                       │                      │
│           │  Scores each chunk    │                      │
│           │  against the original │                      │
│           │  query for relevance  │                      │
│           │                       │                      │
│           │  Output: Top 10       │                      │
│           │  most relevant chunks │                      │
│           └───────────┬───────────┘                      │
│                       │                                  │
└───────────────────────┼──────────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
   ┌────────────┐ ┌──────────┐
   │ BARE ACT   │ │ CASE LAW │
   │ RESULTS    │ │ RESULTS  │
   │            │ │          │
   │ Section-   │ │ Case     │
   │ level      │ │ name,    │
   │ chunks     │ │ court,   │
   │ with full  │ │ year,    │
   │ metadata   │ │ ratio    │
   └──────┬─────┘ └─────┬────┘
          │              │
          └──────┬───────┘
                 │
                 ▼
   ┌──────────────────────────────────────────────────────┐
   │          SUFFICIENCY ANALYSIS (LLM-Driven)           │
   │                                                      │
   │  NO hard numeric limits. The model analyzes:         │
   │                                                      │
   │  BARE ACTS:                                          │
   │  • What are ALL the legal aspects of this dispute?   │
   │  • Which bare acts / sections apply to EACH aspect?  │
   │  • Are ALL relevant aspects covered by local results?│
   │  • Would additional sections add meaningful value    │
   │    to the case? If not, stop.                        │
   │  • Pull every section that's relevant — no cap.      │
   │    But skip sections that don't add incremental      │
   │    value to the case.                                │
   │                                                      │
   │  CASE LAWS:                                          │
   │  • For EACH applicable bare act section (or group    │
   │    of related sections), find 2-3 most relevant      │
   │    case laws.                                        │
   │  • "Relevant" means: factually similar to current    │
   │    dispute + interprets the specific section.        │
   │  • Extract ONLY the portions of those judgments      │
   │    that directly apply to the current dispute.       │
   │  • If 5 sections from 3 bare acts apply →            │
   │    find 2-3 case laws PER section = up to 15 cases.  │
   │  • But if one strong SC judgment covers multiple     │
   │    sections, don't repeat it — cite it once.         │
   │                                                      │
   │  SUFFICIENCY DECISION:                               │
   │  • All aspects of dispute covered by bare acts? ──┐  │
   │  • Each section has supporting case law? ─────────┤  │
   │  • If BOTH yes → LOCAL RESULTS SUFFICIENT         │  │
   │  • If EITHER no → Need internet (identify gaps)   │  │
   └──────────────────────────┬───────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
          ALL COVERED                  GAPS IDENTIFIED
                │                           │
                ▼                           ▼
         Skip internet.     ┌─────────────────────────────┐
         Use local          │ TIERED INTERNET SEARCH       │
         results only.      │ (Only for the GAPS)          │
                            │                              │
                            │ The model identifies exactly │
                            │ what's missing:              │
                            │ • "Need Section 22 of       │
                            │   Karnataka Rent Act"        │
                            │ • "Need SC judgment on       │
                            │   deposit forfeiture"        │
                            │                              │
                            │ Then searches ONLY for the   │
                            │ missing pieces, not broadly. │
                            │                              │
                            │ Tier 2: Official Court       │
                            │ Websites (SC, HC by          │
                            │ user jurisdiction)           │
                            │         │                    │
                            │  Gap still unfilled?         │
                            │         ▼                    │
                            │ Tier 3: Legal Portals        │
                            │ (Indian Kanoon, Live         │
                            │ Law, SC Observer, etc.)      │
                            │         │                    │
                            │  Gap still unfilled?         │
                            │         ▼                    │
                            │ Tier 4: Mainstream           │
                            │ Newspapers (context          │
                            │ only, NOT authority)         │
                            │                              │
                            │ ❌ NEVER: Social media,      │
                            │ blogs, Wikipedia,            │
                            │ YouTube, Quora, Reddit       │
                            └──────────────┬───────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │ AUTO-ENRICH:                 │
                            │ If PDF found online →        │
                            │ Download → Save to           │
                            │ Google Drive → Index         │
                            │ to vector store.             │
                            │                              │
                            │ Database grows with          │
                            │ every query!                 │
                            └─────────────────────────────┘


RETRIEVAL LOGIC EXAMPLES:
─────────────────────────

Example 1: Security Deposit Dispute (Bangalore)
  Aspects identified by model:
    A. Tenant's right to deposit refund
    B. Landlord's right to deduct for damage
    C. Limitation period for recovery
    D. Forum for complaint (civil court vs consumer)

  Bare Acts pulled:
    • Transfer of Property Act — Section 108 (tenant rights)
    • Karnataka Rent Control Act — Section 22 (deposit rules)
    • Consumer Protection Act — Section 2(7), 35 (jurisdiction)
    • Limitation Act — Article 52 (recovery period)

  Case Laws per section/group:
    • S.108 TPA + S.22 KRA (grouped, same topic):
      - Ram Kumar v. Shyam (2019 Karnataka HC) — deposit refund
      - Satish v. Suresh (2021 SC) — landlord obligations
    • Consumer Protection Act:
      - Lucknow Development v. MK Gupta (SC) — housing disputes
      - Bangalore DC v. Tenant (2020 Karnataka HC)
    • Limitation:
      - ABC v. XYZ (SC) — limitation for rent disputes

  Total: 4 bare acts, ~8-10 sections, ~7 case laws
  All found locally? → Skip internet.
  Missing KRA Section 22? → Search internet for THAT specific section.

Example 2: Dowry Harassment Case
  Aspects:
    A. Criminal liability for cruelty
    B. Dowry demand as offense
    C. Domestic violence protection
    D. Maintenance rights

  Bare Acts:
    • BNS (new IPC) — Section 85, 86 (cruelty, dowry death)
    • Dowry Prohibition Act — Section 3, 4
    • Protection of Women from DV Act — Section 12, 18, 19
    • CrPC/BNSS — Section 144 (maintenance)

  Case Laws per group:
    • S.85-86 BNS: 2-3 cases on cruelty ingredients
    • Dowry Prohibition: 2-3 cases on what constitutes demand
    • DV Act: 2-3 cases on protection orders
    • Maintenance: 2-3 cases on quantum and entitlement

  Total: 4 bare acts, ~10 sections, ~10-12 case laws


NEW INDEXING STRUCTURE (Section-Level):
───────────────────────────────────────

BARE ACTS — Each chunk is ONE complete section:
{
  "chunk_id": "ipc_498a",
  "act_name": "Indian Penal Code, 1860",
  "section_number": "498A",
  "section_title": "Husband or relative of husband of a woman subjecting her to cruelty",
  "chapter": "Chapter XXA — Of Cruelty by Husband or Relatives of Husband",
  "full_text": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished...",
  "keywords": ["cruelty", "husband", "dowry", "harassment", "matrimonial"],
  "related_sections": ["304B", "406"],
  "embedding": [0.023, -0.145, ...]
}

CASE LAWS — Each chunk is a legal principle/ratio:
{
  "chunk_id": "sc_2019_dowry_001",
  "case_name": "Rajesh Sharma v. State of UP",
  "citation": "(2017) 8 SCC 543",
  "court": "Supreme Court of India",
  "bench": "Justice A.K. Goel, Justice U.U. Lalit",
  "year": 2017,
  "legal_principle": "Guidelines to prevent misuse of Section 498A...",
  "relevant_sections": ["498A IPC"],
  "paragraph_num": 19,
  "binding_authority": "supreme_court",
  "embedding": [0.034, -0.189, ...]
}
```

---

## PHASE 4: LEGAL ANALYSIS (Consolidated Agent Pipeline)

```
Retrieved results from Phase 3
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│         CONSOLIDATED AGENT PIPELINE (8 Agents)            │
│                                                          │
│  OLD (16 agents)          →  NEW (8 agents)              │
│  ─────────────────           ──────────────              │
│  Facts Collector     ┐                                   │
│  Fact Normalizer     ├─→  1. INTAKE AGENT                │
│  Contradiction Agent ┘       (Phase 2 — already ran)     │
│                                                          │
│  Issue Framing Agent ┐                                   │
│  Jurisdiction Agent  ├─→  2. ISSUE & JURISDICTION AGENT  │
│                      ┘                                   │
│  Bare Act Agent      ┐                                   │
│  Case Law Agent      ├─→  3. LEGAL RESEARCH AGENT        │
│  Act-Case Fusion     ┘       (Phase 3 — already ran)     │
│                                                          │
│  Applicability Agent ┐                                   │
│  Precedent Ranking   ├─→  4. RELEVANCE & RANKING AGENT   │
│                      ┘                                   │
│  Opinion Agent       ┐                                   │
│  Counter-Argument    ├─→  5. OPINION AGENT               │
│                      ┘                                   │
│  Drafting Agent      ┐                                   │
│  Formatting Agent    ├─→  6. DRAFTING AGENT (Premium)    │
│                      ┘                                   │
│  Citation Agent      ┐                                   │
│  Hallucination Agent ├─→  7. QA & SAFETY AGENT           │
│  Confidence Agent    ┘                                   │
│                                                          │
│  Gatekeeper Agent    ──→  8. GATEKEEPER AGENT            │
│                              (Unchanged — human gate)    │
└──────────────────────────────────────────────────────────┘

EXECUTION FLOW:
───────────────

Step 1: ISSUE & JURISDICTION AGENT
        ┌────────────────────────────────────────────┐
        │ Input: Structured facts from Intake Agent   │
        │                                            │
        │ Does:                                      │
        │  • Identifies legal issues                 │
        │    ("Recovery of security deposit",        │
        │     "Breach of rental agreement")          │
        │  • Determines applicable jurisdiction      │
        │    ("Civil Court, Bangalore" or            │
        │     "Rent Control Tribunal, Karnataka")    │
        │  • Checks maintainability                  │
        │    ("Is this a civil suit or consumer      │
        │     complaint?")                           │
        │                                            │
        │ Output: issues[], jurisdiction, court_type │
        └────────────────────────────────────────────┘
                         │
                         ▼
Step 2: RELEVANCE & RANKING AGENT
        ┌────────────────────────────────────────────┐
        │ Input: Retrieved chunks + identified issues │
        │                                            │
        │ Does:                                      │
        │  • Maps each legal aspect of the dispute   │
        │    to its applicable bare act sections     │
        │  • Validates "ingredients" of each section │
        │    against the facts — keeps only sections │
        │    that genuinely apply                    │
        │  • Drops sections that don't add           │
        │    incremental value to the case           │
        │  • For EACH section (or group of related   │
        │    sections), selects 2-3 best case laws:  │
        │    - Binding authority (SC > HC > District) │
        │    - Factual similarity to current case    │
        │    - Recency (newer precedent preferred)   │
        │  • Extracts ONLY the relevant portions of  │
        │    each judgment (specific paragraphs that  │
        │    interpret the section in a factually    │
        │    similar context)                        │
        │  • If one SC judgment covers multiple      │
        │    sections, cite it once (no duplication) │
        │                                            │
        │ Output:                                    │
        │  • bare_acts[] — all relevant sections,    │
        │    grouped by aspect of the dispute        │
        │  • case_laws_per_section[] — 2-3 cases per │
        │    section with extracted relevant portions│
        │  • gaps[] — aspects with no local coverage │
        │    (triggers targeted internet search)     │
        └────────────────────────────────────────────┘
                         │
                         ▼
Step 3: OPINION AGENT
        ┌────────────────────────────────────────────┐
        │ Input: Ranked results + facts + issues      │
        │                                            │
        │ Does:                                      │
        │  • Synthesizes a reasoned legal opinion    │
        │    in plain English (advocate's voice)     │
        │  • Every statement cites a source          │
        │    "[Section 108, TPA]" or                 │
        │    "[Rajesh Sharma v. State, (2017) 8 SCC  │
        │     543]"                                  │
        │  • Identifies strengths of the case        │
        │  • Identifies weaknesses / risks           │
        │  • Suggests counter-arguments opposing     │
        │    counsel might raise                     │
        │  • Recommends next steps                   │
        │                                            │
        │ Output: opinion_text, citations[],         │
        │         strengths[], weaknesses[],         │
        │         recommended_actions[]              │
        └────────────────────────────────────────────┘
                         │
                         ▼
Step 4: QA & SAFETY AGENT
        ┌────────────────────────────────────────────┐
        │ Input: Full opinion + all source chunks     │
        │                                            │
        │ Does:                                      │
        │  • Citation Check: Every legal statement   │
        │    must trace back to a retrieved chunk    │
        │  • Hallucination Check: Flag any claim     │
        │    that doesn't appear in source material  │
        │  • Confidence Score: Rate overall          │
        │    reliability (High / Medium / Low)       │
        │    based on:                               │
        │    - Number of supporting sources          │
        │    - Authority level of cited cases        │
        │    - How well facts match section elements │
        │  • Disclaimer Injection: Add appropriate   │
        │    legal disclaimers                       │
        │                                            │
        │ Output: verified_opinion, confidence_score,│
        │         flagged_claims[], disclaimers      │
        └────────────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────┴──────────────┐
          │                             │
    ┌─────▼──────┐              ┌───────▼──────┐
    │ FREE USER  │              │ PREMIUM USER │
    │            │              │              │
    │ Show:      │              │ Show:        │
    │ • Opinion  │              │ • Opinion    │
    │ • Cited    │              │ • Cited      │
    │   sections │              │   sections   │
    │ • Case     │              │ • Case refs  │
    │   refs     │              │ • Confidence │
    │ • Confid.  │              │              │
    │            │              │ PLUS:        │
    │ STOP HERE  │              │ Go to Step 5 │
    └────────────┘              └───────┬──────┘
                                        │
                                        ▼
Step 5: DRAFTING AGENT (Premium Only)
        ┌────────────────────────────────────────────┐
        │ Input: Verified opinion + facts + citations │
        │                                            │
        │ Does:                                      │
        │  • Selects appropriate document template   │
        │    (petition, affidavit, legal notice,     │
        │     consumer complaint, etc.)              │
        │  • Fills template with case-specific       │
        │    content, parties, facts, prayer         │
        │  • Applies court-specific formatting       │
        │    (cause title, court name, case style)   │
        │  • Attaches annexure references            │
        │                                            │
        │ Output: draft_document (structured text)   │
        └────────────────────────────────────────────┘
                         │
                         ▼
Step 6: GATEKEEPER AGENT
        ┌────────────────────────────────────────────┐
        │ Input: Everything produced so far           │
        │                                            │
        │ Does:                                      │
        │  • Presents full output to user for review │
        │  • "REQUIRES HUMAN APPROVAL" banner        │
        │  • User can:                               │
        │    - Approve → Download as PDF/DOCX        │
        │    - Request changes → Loop back           │
        │    - Ask follow-up questions               │
        │    - Start new query                       │
        │                                            │
        │ Output: Final approved document            │
        └────────────────────────────────────────────┘
```

---

## PHASE 5: OUTPUT DELIVERY

```
┌──────────────────────────────────────────────────────────┐
│                    RESPONSE FORMAT                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 📋 LEGAL OPINION                                   │  │
│  │                                                    │  │
│  │ Based on the facts you've shared, here's my       │  │
│  │ analysis of your situation...                      │  │
│  │                                                    │  │
│  │ [Natural language opinion in advocate's voice]     │  │
│  │                                                    │  │
│  │ ─────────────────────────────────────────────────  │  │
│  │                                                    │  │
│  │ 📖 APPLICABLE LAW                                  │  │
│  │                                                    │  │
│  │ • Section 108, Transfer of Property Act — ...      │  │
│  │ • Karnataka Rent Control Act, Section 22 — ...     │  │
│  │                                                    │  │
│  │ ─────────────────────────────────────────────────  │  │
│  │                                                    │  │
│  │ ⚖️ RELEVANT PRECEDENTS                             │  │
│  │                                                    │  │
│  │ • Ram Kumar v. Shyam Lal (2019) — Karnataka HC    │  │
│  │   "Security deposit must be returned within..."    │  │
│  │                                                    │  │
│  │ ─────────────────────────────────────────────────  │  │
│  │                                                    │  │
│  │ 💡 RECOMMENDED NEXT STEPS                          │  │
│  │                                                    │  │
│  │ 1. Send a legal notice demanding refund            │  │
│  │ 2. If no response in 15 days, file consumer        │  │
│  │    complaint or civil suit                         │  │
│  │                                                    │  │
│  │ ─────────────────────────────────────────────────  │  │
│  │                                                    │  │
│  │ ⚠️ CONFIDENCE: HIGH (3 bare acts, 2 SC judgments)  │  │
│  │                                                    │  │
│  │ ⚠️ DISCLAIMER: This is legal research assistance,  │  │
│  │ not legal advice. Please consult a practicing      │  │
│  │ advocate before taking legal action.               │  │
│  │                                                    │  │
│  │ ─────────────────────────────────────────────────  │  │
│  │                                                    │  │
│  │ [📄 Draft Legal Notice]  [📄 Draft Petition]       │  │
│  │    (Premium Only)           (Premium Only)         │  │
│  │                                                    │  │
│  │ [🔄 Ask Follow-up]  [🆕 New Query]                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## CROSS-CUTTING CONCERNS

### A. Rate Limiting & Tier Enforcement

```
Every API request:
        │
        ▼
┌─────────────────────────────────┐
│ MIDDLEWARE: Rate Limiter         │
│                                 │
│ Guest:   3 queries/day (by IP)  │
│ Free:    5 queries/day          │
│ Premium: 25 queries/day         │
│                                 │
│ Query = one complete research   │
│ cycle (Phase 2-5). Follow-up   │
│ questions within same case      │
│ do NOT count as new queries.    │
│                                 │
│ Exceeded? →                     │
│  Guest: "Sign up for more"      │
│  Free:  "Upgrade for 25/day"    │
│  Premium: "Limit reached, resets│
│            at midnight IST"     │
└─────────────────────────────────┘
```

### B. Chat History & Continuity

```
┌─────────────────────────────────┐
│ CHAT PERSISTENCE                │
│                                 │
│ Each conversation saved with:   │
│ • chat_id (UUID)                │
│ • user_id                       │
│ • title (auto-generated from    │
│   first legal issue identified) │
│ • messages[] (full transcript)  │
│ • structured_facts (JSON)       │
│ • retrieved_results (cached)    │
│ • opinion_generated             │
│ • documents_drafted[]           │
│ • created_at, updated_at        │
│                                 │
│ Users can:                      │
│ • Resume any past conversation  │
│ • Ask follow-ups on old cases   │
│ • Download past opinions/docs   │
│ • Delete their own chats        │
└─────────────────────────────────┘
```

### C. Safety & Compliance Layer

```
┌─────────────────────────────────────────────────────────┐
│ SAFETY RULES (Applied to EVERY response)                 │
│                                                         │
│ 1. CITATION MANDATORY                                   │
│    Every legal claim → must cite Act+Section or Case    │
│    Uncitable claims → removed or flagged                │
│                                                         │
│ 2. DISCLAIMER ALWAYS PRESENT                            │
│    "This is legal research assistance, not legal        │
│     advice. Consult a practicing advocate."             │
│                                                         │
│ 3. CONFIDENCE TRANSPARENCY                              │
│    Show confidence score visibly                        │
│    LOW confidence → extra warning to user               │
│                                                         │
│ 4. NO AUTOMATIC FILING                                  │
│    System NEVER files, submits, or sends anything       │
│    on behalf of user. Output is always a draft.         │
│                                                         │
│ 5. HARMFUL ADVICE GUARD                                 │
│    If query involves violence, fraud, or illegal acts   │
│    → Refuse and explain why                             │
│                                                         │
│ 6. AUDIT TRAIL                                          │
│    Every query logged: user_id, timestamp, query,       │
│    results retrieved, opinion generated, confidence     │
│                                                         │
│ 7. DATA PRIVACY                                         │
│    User case facts encrypted at rest                    │
│    No sharing of user data across accounts              │
│    User can delete all their data                       │
└─────────────────────────────────────────────────────────┘
```

### D. Error Handling & Fallbacks

```
┌─────────────────────────────────────────────────────────┐
│ GRACEFUL DEGRADATION                                     │
│                                                         │
│ Scenario                    → Fallback                  │
│ ─────────────────────────     ─────────────────────     │
│ FAISS returns 0 results     → Web search (DuckDuckGo)  │
│ Web search also fails       → "I couldn't find specific │
│                                law on this. Here's my   │
│                                general understanding,   │
│                                but please verify."      │
│ Ollama timeout/crash        → Retry once, then show     │
│                                "Server busy, try again" │
│ LLM produces no citations   → QA Agent blocks output,  │
│                                asks for re-generation   │
│ Confidence = LOW             → Extra warning + suggest  │
│                                user consult advocate    │
│ Rate limit hit               → Clear message + upgrade  │
│                                path for free users      │
└─────────────────────────────────────────────────────────┘
```

### E. Multi-Language Architecture (Future-Ready)

```
┌─────────────────────────────────────────────────────────┐
│ LANGUAGE LAYER (Designed now, implemented later)          │
│                                                         │
│ ┌─────────┐    ┌──────────────┐    ┌─────────────────┐ │
│ │ User    │───▶│ Language     │───▶│ Core Pipeline   │ │
│ │ Input   │    │ Detector     │    │ (always English │ │
│ │ (any    │    │              │    │  internally)    │ │
│ │ lang)   │    │ If Hindi →   │    │                 │ │
│ │         │    │ Translate to │    │                 │ │
│ │         │    │ English first│    │                 │ │
│ └─────────┘    └──────────────┘    └────────┬────────┘ │
│                                             │          │
│                                    ┌────────▼────────┐ │
│                                    │ Translate output │ │
│                                    │ back to user's   │ │
│                                    │ language          │ │
│                                    └─────────────────┘ │
│                                                        │
│ NOTE: Legal terms remain in English regardless.        │
│ Citations always in original form.                     │
└────────────────────────────────────────────────────────┘
```

---

## INFRASTRUCTURE (Production Deployment)

```
┌─────────────────────────────────────────────────────────┐
│ RECOMMENDED PRODUCTION SETUP                             │
│                                                         │
│ OPTION A: Single GPU Server (Budget)                    │
│ ─────────────────────────────────────                   │
│ • Cloud GPU: RunPod / Vast.ai (RTX A4000 or A5000)     │
│ • Ollama runs on GPU server                             │
│ • FastAPI + React served from same machine              │
│ • Nginx reverse proxy + Let's Encrypt SSL               │
│ • Cost: ~$50-100/month                                  │
│                                                         │
│ OPTION B: Split Architecture (Scalable)                 │
│ ─────────────────────────────────────                   │
│ • VPS (DigitalOcean/AWS Lightsail): FastAPI + Frontend  │
│ • Separate GPU server: Ollama inference only             │
│ • Internal API between VPS ↔ GPU                        │
│ • Can scale GPU up/down based on traffic                │
│ • Cost: ~$80-150/month                                  │
│                                                         │
│ OPTION C: Your Laptop (Development / MVP)               │
│ ─────────────────────────────────────                   │
│ • Cloudflare Tunnel exposes localhost                    │
│ • Your RTX 4060 handles inference                       │
│ • Free but: laptop must stay on, single user perf       │
│ • Good for beta testing with small user group           │
│                                                         │
│ RECOMMENDED: Start with OPTION C for beta,              │
│ move to OPTION A when you have paying users.            │
└─────────────────────────────────────────────────────────┘

DOMAIN SETUP:
─────────────
nyaymalaw.in          → React Frontend (static)
api.nyaymalaw.in      → FastAPI Backend
ollama.internal       → Ollama (not public-facing)
```

---

## IMPLEMENTATION ORDER

```
PRIORITY 1 (Week 1-2): Fix what's broken
─────────────────────────────────────────
□ Rewrite intent classifier (greetings, small talk, legal query)
□ Rewrite conversational prompts (natural advocate voice)
□ Restructure bare act indexing (section-level chunks + metadata)
□ Implement hybrid retrieval (FAISS + BM25 + cross-encoder re-rank)

PRIORITY 2 (Week 3-4): Consolidate & speed up
──────────────────────────────────────────────
□ Merge 16 agents → 8 consolidated agents
□ Optimize prompt chains (fewer LLM calls per query)
□ Add response caching for repeated legal queries
□ Improve fact collection interview flow

PRIORITY 3 (Week 5-6): Production features
───────────────────────────────────────────
□ Build freemium auth system (signup, tiers, rate limits)
□ Payment integration (Razorpay for India)
□ Chat history improvements (auto-title, resume, delete)
□ PDF/DOCX download for opinions and drafted documents

PRIORITY 4 (Week 7-8): Safety & deployment
──────────────────────────────────────────
□ QA & Safety agent (citation check + hallucination + confidence)
□ Disclaimer system
□ Audit logging
□ Deploy via Cloudflare Tunnel (beta)
□ Load testing and optimization
```
