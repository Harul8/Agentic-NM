# Nyaymalaw — Targeted Improvement Roadmap

**Principle:** Do not rebuild. The system already has 80–85% of a modern legal RAG architecture. Improvements are about **index quality**, **query coverage**, and **chunk importance** — not model size.

**Hardware:** Qwen-3-8B on RTX 4060 (8GB) is sufficient.

---

## Current strengths (keep as-is)

- Fact collection, intent routing (Gate 1 / Gate 2)
- Hybrid retrieval (FAISS + BM25)
- Cross-encoder reranking
- Statute → case two-phase pipeline
- Dispute decomposition, sufficiency detection, web fallback

---

## Remaining issues (in order of impact vs effort)

| # | Step | Effort | Impact | Status |
|---|------|--------|--------|--------|
| 1 | [Fix greeting latency](#step-1--fix-greeting-latency) | ~30 min | High (UX) | ✅ Static template |
| 2 | [Fix case law metadata](#step-2--fix-case-law-metadata) | Medium | Very high | ✅ Implemented (v4 smart_chunker) |
| 3 | [Remove bare-act editorial noise](#step-3--remove-bare-act-editorial-noise) | Medium | High | ✅ Implemented (v4 smart_chunker) |
| 4 | [Add paragraph-type metadata](#step-4--add-paragraph-type-metadata) | Medium | Very high | ✅ Implemented (v4 smart_chunker) |
| 5 | [Multi-query retrieval](#step-5--multi-query-retrieval) | Medium | High | ✅ Cap 3; expand returns 1–3; statute-anchored case |
| 6 | [Extract section refs from cases](#step-6--extract-section-refs-from-cases) | Medium | Medium | ✅ Implemented (v4 smart_chunker, sections_cited) |
| 7 | [Expand reranker candidate pool](#step-7--expand-reranker-candidate-pool) | Low | Medium | ✅ 40+40 |
| 8 | [Fix evaluation scoring](#step-8--fix-evaluation-scoring) | Medium | Medium | ✅ Batch + threshold_sweep use _rerank_score |
| 9 | [Clean OCR noise in case law](#step-9--clean-ocr-noise-in-case-law) | Low | Medium | ✅ Implemented (v4 smart_chunker) |
| 10 | [(Later) Citation graph](#step-10--later-citation-graph) | High | High (phase 2) | ✅ Implemented |

**After ingestion changes (steps 2, 3, 4, 6, 9):** Run a **full re-build** of the vector store so new chunk text and metadata (paragraph_type, sections_cited, cleaned case names, stripped bare-act noise, OCR-cleaned case text) are reflected. Use `python Ingestion/build_v2_index.py` (and ensure BARE_ACTS_DIR and CASELAW_DIR point to your source documents).

---

## Step 1 — Fix greeting latency

**Problem:** `is_greeting` → `generate_greeting_response` → `ask_llm()`. Even "hi" calls the LLM (3–5 s).

**Fix:** For messages in `GREETING_PHRASES` (or clear greeting set), return a **static template** instead of calling the LLM.

**Example template:**  
*"Hi! Tell me about your legal issue and I'll help you find relevant laws and cases."*

**Target:** Greeting response &lt;100 ms.

**Where:** `services/fact_collector.py` — before `generate_greeting_response()`, if exact/near-exact match to greeting set → return template.

---

## Step 2 — Fix case law metadata

**Problem:** Case index has parsing errors, e.g.  
`"case_name": "1959 near the Ahari Payin he saw the versus undergo RI for 5 years"`  
instead of e.g. *State of Bihar v Ramchandra*.

**Fix:** Improve `_detect_case_metadata` in `Ingestion/smart_chunker.py` so that extracted fields are:

- `case_name` — party names (e.g. "State of Bihar v Ramchandra")
- `court`
- `year`
- `citation`

Correct metadata improves reranking and evaluation matching.

---

## Step 3 — Remove bare-act editorial noise

**Problem:** Bare act chunks contain editorial content: "Subs. by Act 19 of 1986", gazette notifications, footnotes, page numbers — not legal content.

**Fix:** Strip these during ingestion (in `smart_chunker` or a dedicated cleaning step) so that only substantive section text is chunked and embedded.

**Result:** Cleaner embeddings, better vector similarity.

---

## Step 4 — Add paragraph-type metadata

**Problem:** All case-law paragraphs are indexed equally. Judgments mix facts, arguments, procedural history, **court reasoning**, and **ratio decidendi**. Retrieval often surfaces facts instead of the holding.

**Fix:**

1. **Ingestion:** Classify each case-law paragraph (e.g. with heuristics + optional LLM): `facts` | `arguments` | `reasoning` | `ratio` | `order`.
2. **Store** `paragraph_type` in chunk metadata.
3. **Retrieval/ranking:** Apply a boost by type, e.g. ratio = 1.0, reasoning = 0.8, facts = 0.3.

**Status:** Ingested in smart_chunker v4; retrieval applies boost in hybrid_retriever (ratio=0.5, reasoning=0.35, facts=0.1) so ratio/reasoning rank above facts.

**Note:** This is the “structural improvement that could double legal retrieval accuracy” — indexing by **what part** of the judgment the paragraph is, not just raw text.

---

## Step 5 — Multi-query retrieval

**Problem:** Single expanded query from `expand_legal_query()`; one phrasing can miss relevant docs.

**Fix:** Generate 3–5 legal search queries (LLM or templates), run hybrid retrieval for each, merge and dedupe, then run the existing reranker on the merged set. Take top-k after rerank.

**Example:** User says "government took my land" → queries: "land acquisition compensation India", "Article 300A property rights", "compulsory acquisition law India", "Land Acquisition Act compensation cases".

**Where:** In the path that builds the search query (e.g. `retrieve_bare_acts_phase` / `generate_response_v2`), add multi-query generation and merge before rerank.

---

## Step 6 — Extract section references from cases

**Problem:** Statute–case linkage is implicit; retrieval doesn’t use “this case cites Section X”.

**Fix:** During case-law ingestion, extract phrases like "Section 307 IPC", "Section 27 Arms Act" into a structured field, e.g. `sections_cited = ["IPC 307", "Arms Act 27"]`. Use this in retrieval (filter/boost by section or act) to connect statute → case law.

**Status:** ✅ **Ingestion** (smart_chunker v4) extracts and stores `sections_cited` per chunk. ✅ **Retrieval** (hybrid_retriever): when the query mentions an act/section (e.g. "Section 302 IPC") and a case-law chunk's `sections_cited` contains that reference, the chunk gets a **citation boost** (up to +0.5) so cases that cite the relevant provision rank higher.

---

## Step 7 — Expand reranker candidate pool

**Problem:** If the pipeline only reranks a small set (e.g. top 10 from vector), the reranker can’t recover missed recall.

**Fix:** Retrieve more candidates before rerank, e.g. vector top 40 + BM25 top 40 → merge → rerank → top 5. Ensure `hybrid_retriever` (and any caps) allow a larger candidate set into the cross-encoder step.

---

## Step 8 — Fix evaluation scoring

**Problem:** Eval (e.g. threshold sweep) uses scores that don’t match production: e.g. raw vector distances or a different scale than the reranker. Threshold and F1 become misleading.

**Fix:** Ensure evaluation uses the **same reranker score** (and scale) that drives final ranking in production. Write that score into batch results and use it in `eval/threshold_sweep` and related scripts. Align gold matching (or relax it) so that improvements in retrieval are visible in metrics.

---

## Step 9 — Clean OCR noise in case law

**Problem:** Case law text contains OCR artifacts: stray page numbers ("3", "4"), single letters ("D"), headers/footers. These pollute embeddings.

**Fix:** Add ingestion filters (e.g. in `smart_chunker` or pre-chunk step) to strip or mask these patterns before embedding.

---

## Step 10 — (Later) Citation graph

**Scope:** Phase 2, after 1–8 are in place.

**Idea:** Store relationships: Case A → cites → Case B; Case → interprets → Section. Use at retrieval to expand by precedent chain and statute–case links. Can be implemented with a simple edge store or graph DB.

**Status:** ✅ **Implemented.**  
- **Ingestion:** `Ingestion/smart_chunker.py` — `extract_cited_cases()` extracts "X v Y" from text; each case chunk now has `cited_cases` and existing `sections_cited`.  
- **Build:** `retrieval/citation_graph.py` — `build_citation_graph_from_chunks()` writes `citation_graph.json` (case_info, interprets, cites). Run `python scripts/build_citation_graph.py` or let `scripts/rebuild_vector_store.py` build it after case-law index.  
- **Retrieval:** `get_sections_interpreted_by_cases()`, `get_cases_cited_by()`, `get_cases_citing()`, `expand_case_names_by_precedent()`, `get_chunks_by_case_names()`. In `retrieval/hybrid_retriever.py`, `search_case_laws()` expands results by precedent: after hybrid search, adds up to 8 chunks from cited/citing cases (when graph exists).

### Citation graph — quality assessment and improvement plan

**Current assessment (legal-AI perspective):** Graph **concept and structure are correct** (case → cites → case, case → interprets → section). **Node quality is poor** (~3/10 overall), so the graph cannot yet provide reliable retrieval signals. Treat as experimental until parsing/normalization are improved.

| Component              | Quality        |
|------------------------|----------------|
| Case name extraction   | ❌ Poor         |
| Court extraction       | ⚠️ Weak        |
| Citation normalization | ❌ Missing     |
| Case IDs               | ❌ Unstable    |
| Citation edges         | ⚠️ Partially usable |
| Graph structure        | ⚠️ Conceptually correct |

**Root cause:** Case names are taken from body text, arguments, headings, counsel sections, and OCR noise — not from the **judgment header (top ~30 lines)**. That yields corrupted nodes (e.g. "Counsel for the versus Counsel for the", "6985 and 11988 of 2023", addresses), unstable IDs, and truncated cited-case strings ("rosy jacob v ja"), so edges often don’t resolve.

**Planned fixes (when improving the graph):**

1. ~~**Case name from header only**~~ — ✅ **Done.** Extract "A v B" / "vs" / "versus" only from the **top ~1800 chars** (`_CASE_NAME_HEADER_CHARS`); no fallback to body so "Counsel for the vs…" is avoided.
2. ~~**Reject OCR/non-case noise**~~ — ✅ **Done.** `_is_likely_bad_case_name()` rejects counsel, petitioner, respondent, scobserver, case numbers, addresses; used in metadata and in `extract_cited_cases`.
3. ~~**Normalize case names**~~ — ✅ **Done.** `_normalize_case_name_for_display()` strips S.C.R./AIR, unifies v./vs/versus to " v ", title-cases; applied when setting case name and when adding cited cases.
4. **Normalize courts** — "High Court" → full name (Supreme Court of India, Telangana High Court, etc.) where detectable. (Existing `_sanitise_court_name` and HC detection in header remain.)
5. ~~**Stable case IDs**~~ — ✅ **Done.** Citation graph uses `_stable_case_id(case_name, year)` = `hash(normalized_name + year)` so same case+year = one node; `name_to_ids` for lookup; bad nodes and truncated cited_cases filtered at build.

**Target node shape (example):**  
`case_id`, `case_name`, `court`, `year`, `citation` (e.g. "(2020) 2 SCC 498"). Edges: `from_case_id` → `to_case_id`. Once nodes are clean, precedent chains and section-interpretation clusters become usable for retrieval.

### Citation graph — structural quality assessment (from JSON review)

*From legal knowledge-graph perspective. No code changes in this subsection.*

**What is good:**  
- **Stable internal IDs** (hashed): edges remain valid when metadata changes; correct graph design.  
- **Separate `case_info` and edges** (nodes → case_info, edges → from_case_id / to_case_id): structure is conceptually correct.

**Biggest problems:**

| Issue | Observed | Impact |
|-------|----------|--------|
| **Bad case names** | e.g. "AOR Mr. P. v. Yogeswaran", "1996_President (legal) Mr. Ch.viswanath. versus New Delhi and Five Others" — from counsel, OCR, headings | Broken nodes, duplicate nodes, unusable precedent ranking |
| **Edge targets as fragments** | `to_case_id`: "cpc. but in a recent decision in samiulla v mohammed" — sentence fragment, not a case name | Edges don’t resolve to nodes; graph broken |
| **Citation extraction too permissive** | "in howarth v northcott", "of this court in rosy jacob v jacob" — context words captured with citation | Need strip prefixes ("in", "of this court in", "but in a recent decision") and normalize to "Howarth v Northcott", "Rosy Jacob v Jacob" |
| **Courts missing** | `"court": ""` for many cases | Precedent ranking by court hierarchy (SC > HC > Tribunal) impossible |
| **Case year errors** | Judgment document year used instead of **case year** (e.g. AIR 1966 SC 671 but year 2020) | Incorrect legal timelines |
| **Duplicate nodes** | Same case as "…versus New Delhi" and "…versus New Delhi_2" | Edges split across nodes; graph coherence lost |

**Good citation graph example:**  
Node: `id`, `case_name` (e.g. "A. Abhilasha v. Parkash"), `court`, `year`, `citation`. Edge: `from_case`, `to_case`, `relation: "cites"`. Clean IDs, no fragments.

**Quality score (from JSON review):**  
Graph structure: good. Node IDs: good. Case name extraction: poor. Citation extraction: poor. Normalization: poor. Court metadata: incomplete. **Overall ~3/10** — architecture correct, parsing/normalization need improvement.

**Most important fix:** **Citation extraction with regex + normalization.** Legal citations follow strong patterns (A v B, A vs B, A versus B). After regex extraction: strip prefixes ("in", "of this court in", "but in a recent decision"), strip trailing punctuation, normalize whitespace, produce canonical "Party1 v Party2" so edge targets are valid case names that can resolve to nodes. Reject any extracted string that is a long sentence fragment (e.g. over N words or containing "but", "however", "cpc." as non-party text).

**Why a clean graph matters for Nyaymalaw:**  
(1) **Precedent ranking** — frequently cited cases (e.g. Rosy Jacob v Jacob in custody) can be ranked higher. (2) **Case expansion** — if a retrieved case cites 5 landmark cases, surface them automatically. (3) **Authority weighting** — distinguish landmark / leading precedent from minor authority. Fixing extraction and normalization is mostly a **parsing problem**; once case name and citation parsing improve, the graph becomes much cleaner automatically.

### Landmark detection via PageRank (citation graph, no manual labeling)

*Technique used by legal research engines and academic legal networks. Documented for when the citation graph is clean enough.*

**Core idea:** Importance of a case is determined by **how often other cases cite it**. Examples: Rosy Jacob v Jacob, Kesavananda Bharati v State of Kerala, Maneka Gandhi v Union of India — cited hundreds or thousands of times. In a graph where Case A → cites → Case B, **Case B becomes more authoritative**.

**Algorithm:** Apply a **PageRank-style** score on the citation graph (out-edges = “cites”):

- `importance(case) ≈ sum( importance(citing_cases) / citations_from_that_case )`
- Being cited by **important** cases increases score; being cited **many times** increases authority.

**Example:** Maneka Gandhi v Union of India (1978) — many later cases cite it → PageRank becomes very high → Nyaymalaw can automatically label it as **landmark judgment** without manual annotation.

**What this enables:** (1) **Rank case results** by leading precedent, not “random similar case”. (2) **Detect leading precedents** (e.g. Rosy Jacob v Jacob, Mausami Moitra Ganguli, Thrity Hoshie Dolikuka). (3) **Filter weak authorities** — trial court or rarely cited HC judgments rank lower.

**Combined signals:** Citation graph + court level (SC > HC) + citation count + recency + citation depth. Example combined authority:

- `authority_score = citation_score + court_weight + recency_factor`
- Example: `score = 0.5 * pagerank + 0.3 * court_weight + 0.2 * recency` (e.g. SC=3, HC=2, Tribunal=1).

**Nyaymalaw retrieval:** Instead of ranking only by embedding similarity, combine with authority:

- `final_score = 0.7 * semantic_similarity + 0.3 * authority_score`
- Example: among three similar cases, the one cited 200 times ranks above those cited 10 and 2 times — **leading precedent** surfaces first.

**Why powerful:** Automatically identify landmark cases, prioritize binding precedents, reduce irrelevant case law — **all without manual labeling**. Answers the lawyer question: “What is the leading case on this issue?”

**Prerequisite:** Current graph can support this once case names are normalized and citation extraction improves; then run PageRank (or similar) over the citation graph and store per-case authority/citation_score for ranking.

---

### Citation Bucketing + Authority Scoring (recommended simpler alternative)

Many legal AI systems use a **simpler, more reliable structure first** instead of a full citation graph. It works very well for Indian case law and is easier to maintain and less sensitive to OCR/parsing errors.

**Idea:** Don’t store full graph edges (Case A → Case B → Case C). Store **per-case buckets** and **authority/section signals**:

| Component | What to store | Why it helps |
|-----------|----------------|--------------|
| **Cited cases** | `cited_cases[]` on each case (no edge resolution) | Precedent context without normalizing every citation to a global node. |
| **Authority score** | Numeric by court (e.g. SC +5, HC +3, Tribunal +1) | Rank by court hierarchy: `final_score = reranker_score + authority_weight`. |
| **Citation count** | How many other cases cite this case (if available) | Highly cited = important precedent; simple and effective. |
| **Sections interpreted** | `sections_interpreted[]` per case (e.g. "IPC 420", "Article 21") | **Section interpretation index** — retrieve cases that interpret a given section; very powerful for Indian law. |

**Ideal case index shape (per chunk or per case):**

```json
{
  "case_name": "A Abhilasha v Parkash",
  "court": "Supreme Court of India",
  "year": 2020,
  "citation": "(2020) 2 SCC 498",
  "authority_score": 5,
  "sections_interpreted": ["Hindu Adoptions and Maintenance Act Section 20"],
  "cited_cases": ["Rosy Jacob v Jacob", "Thrity Hoshie Dolikuka v Hoshiam"],
  "paragraph_type": "reasoning"
}
```

**Retrieval pipeline (recommended):**

```text
User query → fact extraction → statute retrieval → section identification
  → case retrieval by section (sections_interpreted) + hybrid
  → authority ranking (+ reranker)
  → LLM answer
```

Example: user asks "tenant threatening landlord" → statute retrieval finds IPC Section 503 → retrieve **cases interpreting IPC 503** → rank by authority + reranker. This mirrors how lawyers reason and is often more reliable than a full graph for Indian judgments.

**Current Nyaymalaw vs this design:**

| Feature | Current | Citation Bucketing + Authority |
|---------|---------|--------------------------------|
| Sections in case | ✅ `sections_cited` per chunk (ingestion) | Same idea; can expose as case-level `sections_interpreted` for section-first retrieval. |
| Cited cases | ✅ `cited_cases` per chunk | Same; use as bucket only (no graph edges). |
| Court / hierarchy | ✅ `binding_authority` (supreme_court, high_court, …) | Add numeric `authority_score` and use in ranking. |
| Paragraph type | ✅ `paragraph_type` (facts, reasoning, ratio, …) | Already used for retrieval boost. |
| Citation count | ❌ Not stored | Add if source data or graph-derived; use in ranking. |
| Section-first case retrieval | ⚠️ Partial (citation boost when query mentions section) | Strengthen: explicit "cases interpreting Section X" path when statute + section are identified. |

**Recommendation:** Treat the **full citation graph** as experimental until node quality is fixed. Prioritise **Citation Bucketing + Authority Scoring**: keep `sections_cited` / `cited_cases` as buckets, add numeric **authority_score** and use it in ranking, and strengthen **section-interpretation retrieval** (retrieve cases by `sections_interpreted` when a section is identified). This delivers most of the benefit with far less complexity and maintenance.

---

## Case metadata quality assessment and 6-stage cleaning pipeline

*From production legal research engine perspective. Expanded dataset assessment; no code changes in this section.*

### 1. Case metadata quality

**Good examples (usable nodes):**  
`Mohd. Abdul Samad v. The State of Telangana` (Supreme Court of India, 2024); `RAHUL GANDHI v. PURNESH ISHWERBHAI MODI` (High Court of Gujarat, 2023).

**Most nodes corrupted (OCR / header noise):**  
e.g. `"case_name": "HC_2025_6'ips, Nr, agundam'"`, `"HC_2025_Hisr.,yjil r'ji;b' '8,j,.1 ,tj#"`, `"HC_2025_District."` — from PDF headers, addresses, scanning noise, administrative pages. These **destroy graph quality**.

### 2. Structural errors

- **Duplicate cases:** e.g. `1996_President (legal) Mr. Ch.viswanath...` and `..._2`; `HC_2024_Occ- Business ... Hanmakonda` and `..._2`. Deduplicate by normalized key (e.g. `normalized_case_name + year`).
- **Impossible year:** e.g. `"year": "2094"` — parsing bug; reject.
- **Court misclassification:** e.g. case labeled **HC_** but `"court": "Supreme Court of India"` — header metadata mixed with inference. A case with HC_ prefix cannot be Supreme Court.

### 3. Statute–case linking weak

Observed: `"section_id": "the 6"`, `"section_id": "thesaid 4"` — **not valid** statutory references. Good link shape: `{"act": "Guardians and Wards Act 1890", "section": "Section 6"}`. Without **act name**, the reference is meaningless.

### 4. Citation graph empty

`"cites": []` — case citations not extracted or precedent relationships missing. Major lost opportunity for legal AI.

### 5. Root cause: noisy regions in judgments

Pipeline is effectively: PDF → OCR → regex extraction. Legal judgments contain:

| Region | Problem |
|--------|---------|
| Header | Case titles repeated, HC_/SC_ prefixes |
| Counsel section | Names misinterpreted as case names |
| Addresses | OCR noise |
| Footers | Page numbers |
| Statutory references | Partial phrases ("the 6", "thesaid 4") |

Without filtering these regions, the dataset stays messy.

### 6. Data quality estimate (sample)

| Category | Approx proportion |
|----------|--------------------|
| Clean cases | ~25–30% |
| Noisy cases | ~50% |
| Severely corrupted | ~20% |

Target for a legal search engine: **>90% clean metadata**.

### 7. Immediate cleaning rules to add

- **Rule 1 — Valid case pattern:** Only keep names matching `[A-Z].+ v\.? .+` or `[A-Z].+ versus .+`. Reject everything else.
- **Rule 2 — Remove administrative prefixes:** Strip `HC_`, `SC_`, `WP_`, `Crl_` (case identifiers, not case names).
- **Rule 3 — Reject OCR garbage:** Reject names containing `#`, `;`, `'`, digits mixed with punctuation (e.g. `HC_2025_6'ips, Nr, agundam'`).
- **Rule 4 — Year sanity:** Reject years outside `1850`–current year.

### 8. Clean node example

```json
{
  "case_id": "SC_2020_ABHILASHA_PARKASH",
  "case_name": "A Abhilasha v Parkash",
  "court": "Supreme Court of India",
  "year": 2020
}
```

### 9. Why cleaning matters for Nyaymalaw

Without cleaning: citation graphs break, precedent ranking fails, retrieval returns irrelevant cases, LLM reasoning becomes unreliable. **Legal data engineering is often harder than building the AI model.**

### 10. Overall assessment (expanded sample)

| Component | Score |
|-----------|-------|
| Architecture | 7/10 |
| Extraction pipeline | 6/10 |
| Metadata quality | 3/10 |
| Graph usability | 2/10 |

Good news: pipeline already has case extraction, statute references, case IDs, metadata structure — the hard parts. Main need is **data cleaning + normalization**.

---

### 6-stage legal document cleaning pipeline (design reference)

*A practical pipeline that can fix 70–90% of noisy metadata. Integrate before indexing so only clean data reaches the vector index.*

**Current flow:**  
`PDF → text extraction → chunking → embedding`

**Target flow:**  
`PDF → text extraction → **cleaning pipeline** → metadata extraction → deduplication → citation extraction → chunking → embedding`

---

**Stage 1 — Remove PDF noise (header / footer)**  
Judgment PDFs repeat headers on every page (e.g. "IN THE HIGH COURT OF TELANGANA", "WP NO. 1234 OF 2023", "Page 4"). If not removed, parser produces garbage like `HC_2025_Hayathnagar, Hyderabad`.  
*Logic:* Collect first 3 lines of each page, count frequency; if a line appears on >30% of pages, remove it. Removes page numbers, court headers, address blocks.

**Stage 2 — Case name extraction (strict pattern)**  
Extract case name **only from the beginning** of the document (e.g. first 30–50 lines). Accept only patterns: `A v B`, `A vs B`, `A versus B`. Example regex: `([A-Z][A-Za-z.& ]+) v(?:s\.?|ersus)? ([A-Z][A-Za-z.& ]+)`. Reject everything else. Removes junk like `HC_2025_District.`.

**Stage 3 — Normalize case names**  
e.g. `S.C.R. A ABHILASHA v. PARKASH` → `A Abhilasha v Parkash`. Remove SCR, AIR, SCC, extra punctuation, newlines; apply title-case normalization.

**Stage 4 — Metadata validation**  
- **Court:** Allowed values e.g. Supreme Court of India, High Court, Tribunal, District Court. Correct e.g. `HC_2020_Haryana` → High Court.  
- **Year:** Reject `year < 1850` or `year > current_year` (e.g. 2094 → discard).

**Stage 5 — Deduplicate cases**  
Duplicates occur when the same case appears in multiple sources or parsing varies. Deduplicate by normalized key: `normalized_case_name + year` (e.g. `a_abhilasha_v_parkash_2020`). When duplicates found: merge metadata, paragraphs, and citations.

**Stage 6 — Extract structured legal signals**  
- **Statutory references:** e.g. "Section 351 Bharatiya Nyaya Sanhita" → store `act_name`, `section_number`.  
- **Case citations:** e.g. "Rosy Jacob v Jacob" → create citation edge `case_A → cites → case_B` for the precedent graph.

**Example final clean record:**

```json
{
  "case_id": "SC_2020_ABHILASHA_PARKASH",
  "case_name": "A Abhilasha v Parkash",
  "court": "Supreme Court of India",
  "year": 2020,
  "legal_issues": ["child custody", "maintenance"],
  "citations": ["Rosy Jacob v Jacob", "Mausami Moitra Ganguli v Jayant Ganguli"],
  "statutes": [{"act": "Guardians and Wards Act 1890", "section": "6"}]
}
```

**Expected improvement after implementing:**

| Component | Before | After |
|-----------|--------|-------|
| Case name quality | ~30% clean | >90% clean |
| Duplicate cases | High | Minimal |
| Citation graph | Weak | Reliable |
| Retrieval quality | Inconsistent | Strong |

**Insight:** In legal AI systems, **data cleaning often takes more engineering effort than model training**. Once the dataset is clean, retrieval quality can improve dramatically even without changing the model.

---

### Follow-up audit: structural problems (post–cleaning sample)

*Second pass assessment: dataset is cleaner than before but still has serious structural problems. Documented for rule-based refinements.*

**Overall quality (estimated from sample):**

| Category | Estimated quality |
|----------|-------------------|
| Case names | ~40% correct |
| Court metadata | ~60% correct |
| Year metadata | ~50% correct |
| Section extraction | ~35% correct |
| Citation graph | ~0% (empty) |

Dataset is **still too noisy for production legal search**; structure is **salvageable with rule-based cleaning**.

**Case name problems:**  
- Reject: `HC_2025_Hyderabad- 500034.` (header/address), `HC_2025_District. versus District.` (parsing noise).  
- Accept only: **Party A v / vs / versus Party B**.  
- Good examples in data: `Punjab & Sind Bank v Baldev Singh`, `Rahul Gandhi v Purnesh Ishwerbhai Modi`, `Ch Kranti v G Ganesh Goud`.

**Court metadata:**  
- Wrong: `High Court of Hyderabad` (historically incorrect; use High Court of Andhra Pradesh / Telangana).  
- Inconsistent: `court: Supreme Court of India` with `case: HC_2007_K. Malla Reddy`.  
- **Rule:** Use a **controlled vocabulary**: Supreme Court of India; High Court of Telangana; High Court of Andhra Pradesh; High Court of Madras; High Court of Gujarat; etc. Reject or normalize everything else.

**Year problems:**  
- Example: `2009_Punjab & Sind Bank versus Baldev Singh` with `year: 1945` — clearly wrong.  
- **Rule:** Extract year from (1) case title, (2) judgment header; then validate `1850 ≤ year ≤ current_year`.

**Section extraction (OCR corruption):**  
- Reject: `theHumanHights 3` (OCR for Human Rights), `canIlotputupadefence 3` (broken OCR).  
- Keep: `CPC 151`, `Constitution 226`, `Constitution 32`, `HinduMarriage 26`.  
- **Rule:** Match **only known patterns** (e.g. `(Section|Article)\s+(\d+)\s+(Act|Code|Constitution)` and explicit act abbreviations). Avoid fuzzy OCR extraction for section refs.

**Citation graph empty:**  
- `"cites": []` is a huge missed opportunity. Legal research engines depend on citation networks (e.g. *Rosy Jacob v Jacob* cites *Saraswathibai Shripad*, *Mausami Moitra Ganguli*). Citation graphs improve retrieval **massively**.

**Duplicates:**  
- Example: `Malkajgiri Hyderabad` and `Malkajgiri Hyderabad_2`. Deduplicate by **normalized_case_name + year**.

**Target pipeline order (before indexing):**  
`PDF → header/footer removal → case title extraction → metadata validation → deduplication → section extraction → citation extraction → chunking → embedding`.

**Expected improvement after full cleaning:**

| Metric | Current | After cleaning |
|--------|--------|----------------|
| Case metadata accuracy | ~40% | >90% |
| Section extraction | ~35% | >85% |
| Citation graph | 0 | Strong |
| Retrieval precision | Low | High |

**Insight:** The biggest weakness of most legal AI systems is **dirty data**, not model size. Qwen 8B is sufficient. Nyaymalaw needs: **clean legal dataset + structured statute mapping + citation graph**.  

**Optional next step:** A full **Indian Supreme Court citation graph (1950–2025)** can improve case retrieval quality by **3–5×**; design and build can be added when citation extraction and node quality are stable.

---

## Log assessment — pipeline and legal reasoning (future fixes)

*Based on production log review. No code changes yet; this is the fix list for later.*

### What the system did correctly

- **Reduced query count:** Bare acts Round 1 uses 3 queries (improvement over earlier 7).
- **Hybrid retrieval:** FAISS + BM25 + cross-encoder is in place and healthy.
- **Dispute decomposition:** Correctly separates e.g. rent default vs criminal threat.

### Architectural problem: web search too aggressive

- **Current behaviour:** Web search (Tier 2) still runs even when local results exist, leading to unnecessary latency and noise.
- **Rule to implement:** `if local_results >= 3: skip_web_search` (or equivalent sufficiency threshold) so web is only used when local is insufficient.

### Critical legal errors observed

| Issue | Observed | Correct / desired |
|-------|----------|-------------------|
| Statute for threat | BNS Section 189 (weapon-related) | BNS Section 351 (criminal intimidation; earlier IPC 503) |
| Cause | Query "tenant threatens physical harm with metal rod" matched weapon sections via embedding instead of intimidation. | Map threat phrases → criminal intimidation → BNS 351. |
| Follow-up questions | "Is the metal rod a deadly weapon?", "Was the tenant part of an unlawful assembly?" | "Did the tenant threaten harm intentionally?", "Was the threat communicated directly?", "Do you fear the threat will be carried out?" |
| Web query error | "Indian Penal Code **1861** legal remedies for **rent default**" | IPC is 1860; rent default is not IPC — keep disputes separate (dispute 1 → rent law, dispute 2 → criminal law). |
| Recall | "1 relevant section across 1 dispute" | Rent dispute: Telangana Rent Act; criminal: BNS 351, 352 — aim for at least 2–3 statutes. |

### Root cause: missing legal concept / offence layer

- Statute retrieval currently relies on **semantic similarity** to raw user text. For criminal law, **offence mapping** works better.
- **Offence mapping examples:**  
  - "threat to break knees", "threat with weapon", "tenant threatening landlord" → **criminal intimidation** → BNS 351.  
  - "rent default", "tenant not paying" → rent/eviction law, not IPC/BNS.
- **Recommended pipeline:**  
  `facts → legal concept / offence detection → statute retrieval`  
  e.g. threat of injury → criminal intimidation → BNS 351. This is how most legal research engines improve statute accuracy.

### Other fixes to implement

1. **Legal concept layer before retrieval:** Extract concepts (e.g. rent default, criminal intimidation) from facts, then search statutes for those concepts.
2. **Keep disputes separate:** Do not merge rent + criminal into one query (e.g. no "IPC remedies for rent default"); each dispute drives its own act/section set.
3. **Act profile vocabulary:** Strengthen act profiles so rent default clearly maps to Rent Control / lease law, not IPC/BNS. Example Rent Act profile keywords: rent, eviction, tenant, landlord, lease, arrears, default.
4. **Follow-up questions:** Generate follow-ups from the **correct** statute (e.g. BNS 351 elements) so questions are legally relevant.

### Current diagnosis (summary)

| Area | Status |
|------|--------|
| Architecture (hybrid, reranker, decomposition) | Good |
| Vector retrieval | Good |
| Speed | Improving |
| Case index quality | Weak (improving with citation graph fixes) |
| Statute mapping | Weak — needs offence/concept layer |
| Legal concept detection | Missing — highest-impact addition |

### Single biggest improvement (for later)

Add a **legal concept / offence mapping layer** before statute retrieval (e.g. user phrases → criminal intimidation / rent default → BNS 351, Rent Act). This is likely to **significantly improve statute selection accuracy** and fix wrong-section and wrong-follow-up issues.

---

## Statute retrieval: structured lookup vs embeddings (how legal research engines do it)

*Many advanced legal research engines do **not** rely primarily on embeddings for statutes. They use a **structured statute retrieval system** instead. Embeddings are used mostly for **cases and commentary**, not bare acts.*

### Why embeddings work poorly for statutes

Statutes are **highly structured** (Act → Chapter → Section → Subsection / Explanation). Example: *Indian Penal Code — Section 503: Criminal intimidation*. User says *"tenant threatened to break my knees"* → lawyer thinks *criminal intimidation* — but embeddings can match unrelated sections because wording differs. Query *"threat of injury"* vs section text *"criminal intimidation"* may have weak semantic similarity. Statutes use **very specific legal terms** (criminal intimidation, culpable homicide, cheating, mischief) — these are **legal categories**, not just semantic phrases; embedding models often fail to map user language to them reliably.

### How professional systems retrieve statutes: concept index

Instead of embedding search, they use a **statute concept index**:

| Concept                 | Statute                         |
|-------------------------|---------------------------------|
| criminal intimidation   | IPC 503 / BNS 351               |
| rent default            | Rent Control Act eviction      |
| trespass                | IPC 441                         |

Retrieval becomes: **facts → legal concept → statute** (direct lookup). Example: *"tenant threatened to break my knees"* → concept *criminal intimidation* → lookup → BNS Section 351, IPC Section 503 → **direct section retrieval, no embedding search**.

### Professional pipeline

```text
User facts
  ↓
Issue / concept detection
  ↓
Statute lookup (concept → act/section)
  ↓
Section retrieval
  ↓
Case law retrieval (here embeddings + citation graph)
  ↓
LLM explanation
```

Embeddings may only help in **issue/concept detection**; statute retrieval is **dictionary lookup**.

### Speed and accuracy

| Method         | Time        |
|----------------|-------------|
| Vector search  | ~50–150 ms  |
| Lookup         | &lt;1 ms    |

Structured retrieval is **orders of magnitude faster**. Concept mapping fixes **accuracy**: it maps user language to legal categories instead of relying on semantic similarity to section text.

### Where embeddings still matter

| Retrieval type | Method                        |
|----------------|-------------------------------|
| **Statutes**   | Structured lookup (concept → section) |
| **Case law**   | Embeddings + citation graph   |
| **Commentary** | Embeddings                    |

Case law and commentary are narrative/unstructured — semantic similarity works well there.

### Ideal hybrid architecture for Nyaymalaw

```text
User query
  ↓
Dispute decomposition
  ↓
Legal concept extraction
  ↓
Statute lookup (concept index)
  ↓
Case law vector search (+ citation graph ranking)
  ↓
LLM explanation
```

### Benefits

| Improvement       | Effect           |
|-------------------|------------------|
| Statute accuracy  | Large increase   |
| Latency           | Much faster      |
| Case retrieval    | More precise (statute-anchored) |
| LLM reasoning     | Better grounded  |

### Why this matters for Nyaymalaw

Logs already show **act-first profile match** and **act refinement via LLM** — the system is **very close** to this architecture. The missing pieces are: (1) **legal concept extraction** (facts → legal concepts), and (2) **structured statute lookup** (concept → act/section table or index). Once these are in place, Nyaymalaw can evolve from a RAG chatbot toward a **legal research engine**.

### Updates to implement (from recent discussions)

Prioritized list tying the last hour’s discussions (citation graph quality, landmark/PageRank, structured statute retrieval) to concrete changes:

| Priority | Update | Source discussion | Effort | Impact |
|----------|--------|--------------------|--------|--------|
| **1** | **Legal concept extraction** | Statute retrieval: structured lookup vs embeddings | Medium | Very high — fixes wrong statute (e.g. weapon vs criminal intimidation). |
| **2** | **Structured statute lookup** | Same — concept → act/section index; use for statute retrieval instead of (or before) vector search. | Medium | Very high — faster, more accurate statute retrieval. |
| **3** | **Citation extraction + normalization** | Citation graph structural assessment | Medium | High — clean graph so edges resolve to real nodes; enables authority/PageRank later. |
| **4** | **Authority scoring in retrieval** | Citation Bucketing + Landmark/PageRank | Low–Medium | High — add numeric court_weight (SC/HC/Tribunal); optionally citation_count; combine with reranker: `final_score = reranker + authority_weight`. |
| **5** | **PageRank on citation graph** | Landmark detection via PageRank | Medium | High — run after graph is clean; store per-case authority/citation_score; use in case ranking (e.g. `0.7 * similarity + 0.3 * authority_score`). |

**Dependencies:** (3) and better case-name extraction must be done before (5) is meaningful. (1) and (2) can be done in parallel with citation-graph work. (4) can use existing court metadata immediately; (5) needs a clean graph.

**Already in place (no change):** Multi-query retrieval (cap 3), statute-anchored case retrieval, paragraph_type and sections_cited in ingestion, hybrid + reranker. Focus new work on concept layer + statute lookup + citation quality + authority/PageRank.

---

## Legal concept / offence detection layer (single biggest improvement)

*Design spec for the highest-impact change: identify legal issues before retrieval. No code in this section.*

### Why current pipeline is wrong for law

**Current (noisy):**  
`user facts → query expansion → vector search → statutes retrieved`

**How law actually works:**  
`facts → legal issue / offence → relevant statute → cases interpreting that statute`

Until the system **identifies the legal issue first**, retrieval will stay noisy (e.g. "tenant threatens with rod" matching weapon sections instead of criminal intimidation).

### Core idea: legal concept extraction

Insert a step that converts user facts into **legal concepts**.

**Example input:**
```
my tenant is not paying rent for 3 months
he threatened to break my knees with a metal rod
```

**Concept extraction output:**
```
legal_concepts = [
  "rent default",
  "eviction",
  "criminal intimidation",
  "threat of bodily harm"
]
```

These concepts then **drive** statute and case retrieval (not raw user text).

### New retrieval pipeline

```
User message
  ↓
Dispute decomposition (existing)
  ↓
Legal concept extraction (NEW)
  ↓
Statute retrieval (by concept)
  ↓
Case law retrieval (anchored to statute)
  ↓
LLM reasoning
  ↓
Answer
```

This mirrors **IRAC** (Issue → Rule → Application → Conclusion): the system currently jumps to Rule retrieval and skips **Issue detection**.

### Example: tenant case

| Step | Civil dispute | Criminal dispute |
|------|----------------|------------------|
| Concepts | rent default, eviction | criminal intimidation |
| Statute | Telangana Buildings (Lease, Rent and Eviction) Control Act | BNS Section 351 – Criminal intimidation |
| Case queries | cases interpreting Telangana Rent Control eviction | cases interpreting BNS 351 |

So instead of searching `tenant threatens physical harm` (→ weapon sections), search `criminal intimidation BNS` / `criminal intimidation statute`. Instead of `tenant not paying rent` only, search `rent default eviction law` / `rent control act eviction`. This massively improves statute recall and section accuracy.

### Implementation options

**Option A (recommended): small LLM prompt**

Use existing Qwen. Prompt pattern:

```
Extract legal concepts from the following dispute. Return concise legal issues only.

Dispute: {facts}

Examples:
Facts: "tenant has not paid rent for 6 months" → Concepts: rent default, eviction
Facts: "person threatened to kill me" → Concepts: criminal intimidation, threat of bodily harm
```

Output JSON: `{ "concepts": ["rent default", "eviction", "criminal intimidation"] }`. Latency ~0.2 s.

**Option B: rule + dictionary hybrid**

- Dictionary: e.g. `"threat" → criminal intimidation`, `"not paying rent" → rent default`, `"evict tenant" → eviction`.
- Combine rule detection + LLM fallback for unseen phrasings.

### How concepts drive retrieval

- **Statute:** Search with concept-based queries, e.g. `criminal intimidation BNS`, `rent default eviction law`, not only raw fact phrases.
- **Case law:** Search e.g. `criminal intimidation landlord tenant case law` or `BNS 351 case law` (statute-anchored), so case retrieval is concept- and statute-driven.

### Expected improvement

| Metric | Current | After concept layer |
|--------|--------|---------------------|
| Statute accuracy | ~60% | ~90% |
| Case relevance | medium | high |
| Wrong sections | frequent | rare |
| LLM hallucination | medium | low |

### Side benefit: better follow-up questions

Wrong statute (e.g. weapon section) led to irrelevant follow-ups: *"Is the metal rod a deadly weapon?"*  
With concept detection and correct statute (BNS 351), follow-ups become legally relevant: *"Did the tenant directly threaten you?"*, *"When did the threat occur?"*, *"Are there witnesses or recordings?"*  
Generate follow-ups from the **correct** statute (e.g. elements of BNS 351).

### Example final flow (Nyaymalaw)

**User:** *"my tenant hasn't paid rent for 3 months and threatened me"*

1. **Disputes:** (1) rent default, (2) criminal intimidation  
2. **Concepts:** rent default, eviction, criminal intimidation  
3. **Statutes:** Telangana Rent Control Act (eviction); BNS 351 (criminal intimidation)  
4. **Cases:** cases interpreting criminal intimidation; cases interpreting eviction for rent default  
5. **LLM** produces answer and **follow-ups** from the right provisions.

### Summary

- **Single change:** add a **Legal Concept Layer** between facts and retrieval.
- **Effects:** better statute accuracy, better case retrieval, fewer wrong sections, better follow-up questions, lower hallucination.
- **Place in pipeline:** after dispute decomposition, before statute retrieval; concepts feed both statute and case-law queries.

---

## Vector index / embedding context structure (retrieval quality)

*Design spec: how legal documents should be represented before embedding. No code in this section.*

### Problem: embedding only section/paragraph text

**Current bare-act chunk:** embedding is generated from something like `search_text`: section title + snippet (e.g. *"The Administrative Tribunals Act, 1985 Section 1 — Short title, extent and commencement"*). Many sections share generic titles: *Short title*, *Definitions*, *Preliminary*, *Interpretation*. Their embeddings become **very similar**, so semantic search often matches these generic sections for unrelated queries (e.g. "tenant threatens physical harm" matching *Definitions* / *Preliminary*). Legal meaning depends on **act name, chapter, legal domain, topic** — not just the section text.

**Current case-law chunk:** embedding is often **only the paragraph text**. That drops **case name, court, year, legal issues**, so case retrieval loses crucial legal signals.

### Fix: context-rich embedding text

**Do not** embed only the section or paragraph body. Build a **context-rich string** that is what gets embedded (and optionally shown in `search_text`).

**Bare acts — current vs better:**  
Current (weak): *Section 3 — Definitions*. Better: *Statute: Administrative Tribunals Act 1985 | Chapter: Preliminary | Section: Definitions | Legal topic: administrative tribunals jurisdiction and service matters | Text: ...*  
For rent: *Statute: Telangana Buildings (Lease, Rent and Eviction) Control Act | Legal domain: landlord tenant law | Section: Eviction of tenants for rent default | Text: ...*  
Then a query like *"tenant not paying rent"* matches **landlord tenant law / eviction** much more strongly than generic *Definitions*.

**Case law — current vs better:**  
Current (weak): paragraph text only. Better: *Case: Mohd Abdul Samad v State of Telangana | Court: Supreme Court of India | Year: 2024 | Legal issue: criminal intimidation and threat of bodily harm | Paragraph: ...*  
Include **case name, court, year, legal issue(s)** in the embedding text so case retrieval is signal-rich.

### Keywords field is underused

Many chunks have `"keywords": []`. Keywords are very effective for legal retrieval (e.g. eviction, rent default, tenant, landlord for an eviction section). They should be **included in the embedding text**. Populate keywords at ingestion (heuristic or LLM) for both bare acts and case law.

### Paragraph type for case law (already present; use in ranking)

Chunks already store **paragraph_type** (facts, arguments, reasoning, ratio, order). Use it in **ranking**: e.g. ratio decidendi = very high, reasoning = high, arguments = medium, facts = low. Ensure ranking/reranker applies this boost.

### Chunk size and unit of meaning

For statutes, **one section = one chunk** is usually the right legal unit. Splitting sections into sub-parts can dilute context. For case law, paragraph-level is fine, but each chunk’s **embedding text** should include the context above (case, court, year, legal issue, paragraph_type label).

### Ideal legal vector record (target shape)

**Bare act:** `doc_type`, `act_name`, `section_number`, `section_title`, `chapter`, `legal_domain`, `keywords` (populated), and **embedding_text** = *"Statute: ... | Legal topic: ... | Section: ... | Text: ..."* (and keywords line if desired).

**Case law:** embedding text = case name, court, year, legal issue(s), paragraph_type, then paragraph text. Keywords populated where possible.

### Why this matters for Nyaymalaw

Negative reranker scores (e.g. top -3.507) often indicate weak semantic match. **Richer embedding context** improves retrieval **without changing the model**; typical reported gain for legal RAG is **~30–40%** when metadata and legal topic/domain are injected into the embedded text. Metadata (`act_name`, `section_number`, `chapter`, `court`, `year`, `paragraph_type`) already exists; the main fix is **how the embedding input string is built** at index time.

### Suggested order of improvements (recap)

1. Add **legal concept extraction** layer.  
2. Improve **embedding context structure** (context-rich embedding text for bare acts and case law; include keywords).  
3. **Populate keywords** for statutes and cases (heuristic or LLM at ingestion).  
4. **Paragraph type** for case law — already stored; ensure **ranking** uses it (ratio > reasoning > facts).  
5. Reduce unnecessary **web search** when local results are sufficient.

---

## Legal query augmentation (section retrieval accuracy)

**Goal:** Improve bare-act and case-law **section matching** by bridging the gap between **user wording** and **statute wording**. Many legal AI systems rely on this; it is simple but very effective.

### Problem

Current flow: `User query → embed(query) → vector search → rerank`. Legal queries rarely match the exact wording of statutes.

| User phrase              | Statute phrase               |
|--------------------------|------------------------------|
| tenant has not paid rent | default in payment of rent   |
| threat                   | criminal intimidation        |
| steal                    | theft                        |
| land taken by govt       | compulsory acquisition       |
| rent not paid            | default in payment of rent   |

Embedding similarity helps but is not enough. **Legal query expansion** (augmentation) improves recall without hurting precision.

### Core idea: 3 legal variants instead of 1 query

Instead of searching with **one** query, generate **3 legal variants** and run hybrid retrieval for each, then **merge → deduplicate → rerank**.

**Example**

- Original: `tenant has defaulted on rent`
- Variants:
  1. `tenant default in payment of rent`
  2. `eviction for non payment of rent`
  3. `rent control act eviction tenant default`

This dramatically improves recall for sections that use formal phrasing.

### Where it fits in Nyaymalaw

In **`response_generator_v2.expand_legal_query()`** (or the caller that uses it): instead of returning **one** string, return **up to 3 queries** (e.g. a list). Downstream: for each query run hybrid retrieval, merge all results, deduplicate, then run the existing cross-encoder reranker and take top-k.

**Constraint:** Keep **max_queries = 3** so retrieval stays fast (aligned with the fast pipeline redesign).

### Two expansion mechanisms

1. **Legal synonym expansion**  
   Use a small **dictionary** mapping user terms to statute-style terms, e.g.  
   - rent → rent, lease payment, tenancy payment  
   - eviction → eviction, recovery of possession  
   - threat → criminal intimidation, threat of injury  
   - land acquisition → compulsory acquisition, state acquisition  

   Then expand the user query into one or two variants that substitute or add these phrases (e.g. “tenant threat” → “criminal intimidation by tenant”, “threat of injury by tenant”).

2. **Statute-style expansion**  
   Rewrite plain language into **formal statutory phrasing**.  
   - Example: “tenant did not pay rent” → “default in payment of rent by tenant”.  
   Can be done with templates, rules, or a small LLM call.

### Optional: section-number detection

If the query mentions **“Section 420”**, **“Article 21”**, etc., add an extra query for precision, e.g.  
- “IPC section 420”, “Indian Penal Code section 420”  
so that BM25/vector can hit the right act and section explicitly.

### Combine and rerank

For each of the 3 (or fewer) queries:

- Run **hybrid retrieval** (FAISS + BM25).
- Append to a combined list.
- **Deduplicate** by chunk/section id.
- Run **cross-encoder reranker** on the merged set.
- Take **top-k** (e.g. 8) for the LLM.

This increases recall without hurting precision, because the reranker keeps the best passages.

### Why it works especially well for law

Legal language is highly structured; the same concept appears in many phrasings. Query expansion aligns user language with statute language. Typical reported gain: **~30–40% improvement in legal section recall**, and **~20–30%** for case-law relevance when applied to case-law retrieval as well.

### Example improvement

- **Query:** “tenant threatening landlord”
- **Before (single query):** BNS/IPC intimidation sections sometimes missed.
- **After (3 variants):** e.g. “criminal intimidation tenant”, “threat of injury tenant”, “tenant threatening landlord criminal offence” → sections like BNS 351, IPC 503 are retrieved reliably.

### Resulting retrieval pipeline (with augmentation)

```
facts
  → query expansion (3 variants: base, synonym, statute-style; optional section-number query)
  → for each query: hybrid retrieval (FAISS + BM25)
  → merge results
  → deduplicate
  → cross-encoder rerank
  → top sections
  → bare acts + case laws
  → LLM answer
```

With **web enrichment** and **indexing** kept separate (not blocking the response), as in the fast pipeline redesign.

### Summary table (after all changes in this doc)

| Area             | Improvement              |
|------------------|--------------------------|
| Response speed   | minutes → 3–6 s          |
| Section matching | +30–40% (with expansion) |
| Case law relevance | +20–30%                |
| Greeting latency | instant (with shortcut)  |

No code written here; this section is the design reference for implementing legal query augmentation in Nyaymalaw.

---

## Statute-anchored retrieval (case-law relevance)

**Goal:** Make case-law retrieval follow **how lawyers reason**: facts → applicable statute → **cases interpreting that statute**. Top legal AI systems (Harvey, Casetext, Lexis AI) rely on this; it fits Nyaymalaw’s existing statute-then-case flow.

### Current vs target

**Current:** facts → retrieve bare acts → retrieve case laws. Case-law retrieval is still **mostly semantic (vector) search** on the facts, which often returns loosely related or random cases (e.g. generic eviction cases, property disputes, lease agreements).

**Target:** facts → retrieve statutes → **extract statute references** (act name + section number) → **retrieve cases using those references as queries** → rank by authority. So case retrieval is **anchored to the sections** already deemed relevant, not only to the raw facts.

### Why it works

Lawyers reason: **Facts → applicable statute → cases interpreting that statute.**  
Example: “Tenant has not paid rent for 3 months” → Telangana Rent Control Act, eviction-for-default section → cases interpreting that section. Without anchoring, vector search on facts alone can return random eviction or property cases. With anchoring, you get cases that **interpret the specific provision** you surfaced.

### What to change

1. **After** bare-act retrieval, **extract statute references** from the top retrieved sections: act name + section number (e.g. “Telangana Buildings (Lease, Rent and Eviction) Control Act”, “Section 10”).
2. **Build case-law search queries from those references**, not only from the original facts.  
   Example queries:  
   - `Section 10 Telangana Rent Control Act eviction`  
   - `case law Section 10 rent default tenant`  
   - `interpretation of rent default eviction section`
3. Run **hybrid retrieval** (FAISS + BM25) for case laws using these **statute-anchored queries** (and optionally merge with one facts-based query if desired).
4. **Authority ranking:** After retrieval, boost by court (e.g. Supreme Court +5, High Court +3, District Court +1). Case chunks already have `court` / `binding_authority`-style metadata, so this is a scoring layer on top of the reranker.

### Where it fits in Nyaymalaw

Inside **`generate_final_opinion_with_case_laws()`** (or the shared path that fetches case laws): once you have **retrieved bare-act sections**, build a list of statute-anchored queries, e.g. for each top section:  
`f"{section_number} {act_name} case law interpretation"` (and variants). Then run case-law hybrid retrieval on those queries instead of (or in addition to) a single facts-only query. Merge, dedupe, rerank, then apply authority boost.

### New case retrieval pipeline (conceptual)

```
facts
  → retrieve statutes (existing)
  → extract top sections (act + section number)
  → build statute-anchored queries
  → case retrieval using those queries (hybrid search)
  → merge + dedupe + rerank
  → authority ranking (court tier boost)
  → top cases to LLM
```

**Before:** facts → vector search cases → noisy mix.  
**After:** facts → statutes → section extraction → case retrieval by statute → authority ranking → precise, interpretative cases.

### Example

- **User:** “Government acquired my land without compensation.”
- **Statute retrieval:** Article 300A, Land Acquisition Act (existing).
- **Case queries:** e.g. “Article 300A right to property case law”, “Land Acquisition Act compensation case law”.
- **Result:** Cases like *K.T. Plantation v State of Karnataka* that **interpret** those provisions, instead of generic property disputes.

### Combined optimized architecture (after all improvements in this doc)

```
User query
  → fact collection
  → query expansion (3)
  → bare act retrieval
  → section extraction
  → case retrieval anchored to sections
  → authority ranking
  → LLM answer
```

Web search and indexing remain **asynchronous** (not blocking the response).

### Expected impact

| Metric           | Current | After (with statute anchoring) |
|------------------|---------|----------------------------------|
| Response time    | minutes | 3–6 s                            |
| Statute recall   | good    | excellent                        |
| Case relevance   | medium  | high                             |
| Legal reasoning  | medium  | strong                           |

No code written here; this section is the design reference for implementing statute-anchored case retrieval in Nyaymalaw.

---

## Expected outcomes (after 1–8)

| Metric                | Before | After (target) |
|-----------------------|--------|-----------------|
| Greeting latency      | 3–5 s  | &lt;0.1 s       |
| Statute recall        | good   | slightly better |
| Case law relevance    | medium | high            |
| Retrieval precision   | medium | high            |
| Overall response time | 6–12 s | 3–6 s           |

---

## Implementation order (recommended)

1. Greeting shortcut  
2. Fix case name extraction  
3. Remove bare-act editorial noise  
4. Add paragraph_type metadata  
5. Multi-query retrieval  
6. Extract section references from cases  
7. Expand reranker candidate pool  
8. Fix evaluation scoring  

Then consider: citation graph, knowledge graph, agentic retrieval.

---

## Debugging slow follow-ups (5–10 minutes)

If a **simple follow-up question** (e.g. answering "Telangana" or "yes") takes **minutes** instead of 1–3 s, the cause is usually **retrieval or web running when it shouldn’t**.

**Rule:** Follow-up questions must use only:

- `fact_collector.get_next_question_or_complete` → LLM (Gate 1 / Gate 2) → return next question.

They must **not** run: retrieval, tiered_search, auto_enricher, case_law_discovery, or indexing.

**How to find the bottleneck:**

1. **Enable pipeline timing** (env):
   ```bash
   set PIPELINE_TIMING=1
   ```
   Then send a slow request and check logs. You should see lines like:
   - `PIPELINE_TIMING process_chat START`
   - `PIPELINE_TIMING fact_collector.gate1: 2500 ms`
   - `PIPELINE_TIMING fact_collector.gate2: 3000 ms`
   - `PIPELINE_TIMING get_next_question_or_complete: 6000 ms`
   - If you see `FACT_COLLECTION → _run_bare_acts_phase` or `FACT_COLLECTION → retrieval (search/lookup)` **after a short message**, the LLM returned `action=complete` too early (e.g. treating "Telangana" as full facts). That triggers full retrieval + web and can take minutes.

2. **Check for the warning:**  
   When `action=complete` is returned with a very short `facts_summary` (< 80 chars), the code logs:
   ```text
   FACT_COLLECTION triggering RETRIEVAL ... facts_summary len=N preview='...'
   If this was a short follow-up (e.g. state name), Gate 2 may have misclassified.
   ```
   If you see this on a follow-up, fix Gate 2 / prompts so short answers yield `action=ask` with another question instead of `complete`.

3. **Temporary test:** Disable web and external APIs (tiered_search, auto_enricher, indian_kanoon_client) and re-test. If latency drops from minutes to seconds, the slowdown is in web/PDF fetch/indexing.

**Expected latency (with local model only):**

| Task               | Expected  |
|--------------------|-----------|
| Greeting           | &lt;0.2 s  |
| Follow-up question | 1–3 s     |
| Bare act retrieval | 2–4 s     |
| Full opinion       | 4–8 s     |

Anything above ~20 s for a follow-up indicates an unintended heavy path.

---

## Why a single response takes minutes (log-based diagnosis)

Real logs showed that **one** `/conversation/continue/stream` request was doing far too much work **synchronously** inside the request. The pipeline is behaving like a **research assistant** (fetch → download → chunk → then answer) instead of **answer from local store first, enrich later**.

### Where the time goes (from logs)

| Step | Approx. time | What’s happening |
|------|--------------|-------------------|
| Intent extraction | ~80 s | LLM (Gate 1 / Gate 2) |
| Dispute decomposition | ~30 s | LLM |
| Local retrieval | ~10 s | FAISS + BM25 + rerank |
| **Score check** | — | “Best local score &lt; 8.0 → **forcing web search**” even when local results exist |
| Web search | ~40 s | **Sequential** tiered_search (e.g. 6 queries × 5–10 s each) |
| **PDF download + parse + chunk** | ~40 s+ | **auto_enricher** runs **inside the request**: download PDF → parse → chunk (e.g. “Enriching: India Code PDF”, “Chunked bare act …”) — **10–40 s per PDF**, multiple acts |

Total for one request can reach **~200 s (~3 min)**; with retries or multiple disputes, **5–10 minutes**.

### Root causes (no code here — understanding only)

1. **auto_enricher is synchronous during the response**  
   New laws are downloaded, parsed, and chunked **before** the answer is sent. That should be: *return best local results → optionally add URLs to “pending indexing” → index in background later*.

2. **Web search is forced by a score threshold**  
   Logic like “best local score &lt; 8.0 → force web search” means the system **almost always** hits the internet even when the vector store has usable results. Prefer: *only go to web when local result count is low (e.g. &lt; 3)*, not a fixed score cutoff.

3. **Too many sequential web queries**  
   e.g. 7 local + 6 web queries, each tiered_search taking 5–10 s, run **one after another**. Fewer queries (e.g. 3 local, 2 web) and **parallel** web searches would cut tens of seconds.

4. **No PDF caching**  
   The same act (e.g. Telangana Rent Control Act) can be fetched and parsed again on every request. Caching fetched PDFs (or chunked output) would avoid repeated download+parse.

5. **Heavy LLM steps**  
   Intent extraction (~80 s) and dispute decomposition (~30 s) are slow; that’s model/GPU, but the **big** wins are from not doing web + enricher synchronously.

### Fixes to apply (in order of impact)

| Priority | Change | Effect |
|----------|--------|--------|
| 1 | **Disable auto_enricher during responses** | Stop download/parse/chunk inside the request. Only add discovered documents to “pending indexing”; user can index later. Removes 40+ s of PDF work per request. |
| 2 | **Stop forcing web search when local results exist** | Use “if local result count &lt; N then web” (e.g. N=3) instead of “if best score &lt; 8 then web”. Keeps most requests local-only. |
| 3 | **Run web searches in parallel** | e.g. `asyncio.gather` or equivalent for tiered_search calls. Can cut 30–40 s off web phase. |
| 4 | **Reduce number of queries** | e.g. 3 local queries, 2 web queries instead of 7+6. Fewer LLM and search steps. |
| 5 | **Cache act PDFs (or chunked output)** | Avoid re-downloading and re-parsing the same act on every request. |

**Fastest improvement:** Doing (1) and (2) alone can cut latency by **80–90%** (minutes → seconds for many requests). The architecture is fine; the issue is **how much automation runs synchronously in a single request**.

---

## Fast retrieval pipeline redesign (three tiers)

Goal: **3–6 s responses**, bare acts + case laws still returned, web enrichment **does not block** the response. Fully compatible with current architecture; no full rewrite.

### Core idea

**Current (slow):** User question → research engine → download laws → parse PDFs → answer.  
**Target (fast):** User question → retrieve from local index → answer → (optional) discover more sources in background.

Separate **chat response**, **research**, and **indexing** into different layers. Research and indexing must not run synchronously inside the user request.

---

### Tier 1 — Fast local retrieval (during request, 2–3 s)

Runs **only** during the user request. No web, no PDF download, no chunking.

```
User message
  → intent extraction
  → dispute decomposition
  → 3 query expansion (not 7)
  → hybrid retrieval (FAISS + BM25)
  → reranker
  → top statutes + cases
  → LLM answer
  → return response
```

**Hard limits for Tier 1:**

| Parameter | Value | Purpose |
|-----------|--------|---------|
| max_queries | 3 | Fewer LLM + retrieval steps |
| vector_candidates | 40 | Enough for reranker |
| bm25_candidates | 40 | Same |
| rerank_top | 8 | Final passages to LLM |

Target: **&lt;3 s** for this tier.

---

### Tier 2 — Optional web discovery (background)

If the system decides **coverage is weak** (e.g. local result count &lt; 3, not “score &lt; 8”), it can run **tiered_search** — but **asynchronously**.

- User gets the **local answer first**.
- A **background** process runs tiered_search.
- New results appear as **“additional sources”** (e.g. UI update or next turn), not by blocking the first response.

No waiting for web inside the request.

---

### Tier 3 — Index enrichment (offline / on demand)

Today: inside the user request the pipeline does “download PDF → chunk → embed → add to FAISS”. That must **never** run during a request.

**New flow:**

1. Web search (or Tier 2) finds a document → add to **pending_indexing** (metadata + URL only).
2. User or admin clicks **“Index”** in the UI.
3. A **background job** (or explicit indexing API) runs: download → chunk → embed → update FAISS/BM25.

So: **web finds doc → pending list only → index only when user triggers it.** No auto_enricher chunking during the chat response.

---

### New retrieval flow (summary)

**Synchronous (user request):**

1. process_chat → fact_collection → response_generation.
2. Expand query (**3** variants, not 7).
3. Hybrid retrieval (FAISS + BM25) with candidate limits above.
4. Reranker → top 8.
5. Statute selection + case law retrieval (local only).
6. LLM answer.
7. Return response.

**Asynchronous (after response, optional):**

- If coverage weak: run tiered_search in background; surface “additional sources” when ready.
- When user chooses “Index”: run auto_enricher (download → chunk → index) in background or via dedicated job.

---

### Code changes needed (when implementing)

| # | Change | Where / what |
|---|--------|----------------|
| 1 | **Disable synchronous enrichment** | In response_generator_v2: on web search, only **add to pending list**; do not call auto_enricher to chunk/embed during the request. |
| 2 | **Reduce query count** | Bare acts round 1: use **max_queries = 3** instead of 7. Cuts retrieval time ~50–60%. |
| 3 | **Remove forced web by score** | Replace “best_score &lt; 8 → web search” with “**if local result count &lt; 3** → (optionally) trigger web in background”. Score thresholds cause unnecessary blocking web. |
| 4 | **Parallelize web searches** | When web does run (e.g. in background), use **asyncio.gather** (or equivalent) for multiple tiered_search calls instead of sequential. 40 s → ~5 s for that phase. |

---

### Expected performance after redesign

| Operation | Current | After |
|-----------|---------|--------|
| Greeting | 5–10 s | &lt;0.2 s |
| Follow-up question | minutes | 2–3 s |
| Bare act retrieval | minutes | 2–4 s |
| Full legal opinion | minutes | 4–6 s |

Rough breakdown for full opinion: intent 1–2 s, dispute 1 s, local retrieval 1–2 s, LLM answer 1–2 s → **total 4–6 s**.

---

### Takeaway

The system is already sophisticated. The fix is to **stop doing research and indexing inside the user request**. Tier 1 = fast local answer. Tier 2 = optional background web discovery. Tier 3 = indexing only when user asks, in background. No code written here; this section is the design reference for implementation.

---

## References

- Architecture: `docs/Architecture.txt`
- Routing and intent: `docs/CREWAI_MULTI_AGENT_ARCHITECTURE.md`
- Eval and retrieval analysis: earlier thread (threshold sweep, gold vs retrieval, score scale).

---

## Consolidated: Architecture updates first, then code changes

Based on everything in this doc and the last hour of discussion, below are (A) **what to update in the Architecture** so the doc describes the target system, then (B) **code change suggestions** by area. No code is written here — only what to document and what to implement.

---

### A. Architecture doc updates (docs/Architecture.txt and related)

Do these **first** so the architecture document describes the **target** behaviour; then implement code to match.

1. **Three-tier retrieval model**  
   - Add a section that describes **Tier 1 (fast local)**, **Tier 2 (optional web, background)**, and **Tier 3 (index enrichment, offline/on demand)**.  
   - State clearly: *During the user request only Tier 1 runs; Tier 2 and Tier 3 never block the response.*

2. **Synchronous vs asynchronous boundaries**  
   - In the "Response generator v2" and "Retrieval" sections, spell out:  
     - **In-request:** intent, dispute decomposition, query expansion (3), hybrid retrieval (FAISS + BM25), reranker, section extraction, statute-anchored case retrieval, authority ranking, LLM answer.  
     - **Out-of-request:** tiered_search (when coverage weak), auto_enricher (download → chunk → index only when user triggers Index).  
   - Note that web search must not run synchronously inside the response path unless explicitly designed as a background step that does not block.

3. **Retrieval pipeline (target)**  
   - Replace or extend the current "Retrieval path (shared)" with the **target** flow:  
     - facts → **query expansion (3 variants:** base, legal synonym, statute-style; optional section-number)  
     - → hybrid retrieval per query → merge → dedupe → rerank  
     - → **section extraction** from top bare-act results  
     - → **statute-anchored case retrieval** (queries built from act + section)  
     - → **authority ranking** (court tier boost)  
     - → top passages to LLM.  
   - Mention that `expand_legal_query` returns up to 3 queries (not a single string) in the target design.

4. **Greeting path**  
   - In "Chat pipeline", update the greeting bullet to: *For messages in the greeting set, return a **static template** (no LLM call); only call generate_greeting_response when not in the set.* Target: greeting &lt;0.2 s.

5. **Web and enrichment in the diagram**  
   - In the high-level flow (section 10), add a note: *Web search and auto_enricher (PDF download, chunking, indexing) do not run inside the request; they run asynchronously or when user clicks Index.*

6. **Constants / limits**  
   - Add a small "Target limits" subsection (or refer to IMPROVEMENT_ROADMAP): max_queries = 3, vector_candidates = 40, bm25_candidates = 40, rerank_top = 8; web only when local result count &lt; 3 (not by score threshold).

7. **Cross-reference**  
   - At the end of Architecture.txt, add: *For latency diagnosis, three-tier design, legal query augmentation, and statute-anchored retrieval, see docs/IMPROVEMENT_ROADMAP.md.*

8. **CREWAI_MULTI_AGENT_ARCHITECTURE.md**  
   - Optionally add a short "Performance and retrieval" note: response path must not run web or indexing synchronously; see IMPROVEMENT_ROADMAP for Tier 1/2/3 and statute-anchored retrieval.

---

### B. Code change suggestions (by area)

Implement after (or in parallel with) the architecture doc updates. Order below matches impact and dependencies.

---

#### 1. Latency and pipeline boundaries (highest impact)

- **Disable synchronous enrichment during responses**  
  - In `response_generator_v2`, wherever web search finds a document and then calls auto_enricher to download/chunk/embed: remove the chunk/embed step from the request path.  
  - Only add the document to **pending_indexing** (metadata + URL). Chunking and indexing run when the user triggers Index (e.g. `/indexing/run`) or a background job.

- **Stop forcing web search by score**  
  - Find the condition that triggers web when "best local score &lt; 8" (or similar). Replace with: trigger web only when **local result count &lt; N** (e.g. N = 3). If you still run web, do it in background (Tier 2) so the response returns first.

- **Reduce query count**  
  - Where the pipeline builds 7 (or more) queries for bare-act round 1: cap at **max_queries = 3**. Use the same cap for any multi-query case-law path so Tier 1 stays under ~3 s.

- **Parallelize web searches**  
  - Where multiple `tiered_search` (or similar) calls run in a loop: run them in parallel (e.g. `asyncio.gather` or a thread pool). Do this for any path that still runs web (e.g. background Tier 2).

---

#### 2. Greeting latency

- **Static greeting for known phrases**  
  - In `services/fact_collector.py`, before calling `generate_greeting_response()`: if the message matches the greeting set (e.g. `GREETING_PHRASES` or a cleaned substring), return a fixed template string (e.g. "Hi! Tell me about your legal issue…") and do not call `ask_llm`.  
  - Keep `generate_greeting_response` for edge cases that are greeting-like but not in the set.

---

#### 3. Legal query expansion (section recall)

- **expand_legal_query returns multiple queries**  
  - In `response_generator_v2.expand_legal_query()` (or a wrapper): instead of returning a single string, return a **list of up to 3 queries** (e.g. base query, legal-synonym variant, statute-style variant).  
  - Add a small **legal synonym dictionary** (e.g. rent → lease payment, tenancy payment; eviction → recovery of possession; threat → criminal intimidation; land acquisition → compulsory acquisition) and a helper that expands the user text into one or two extra variants.  
  - Optionally add **statute-style** rewriting (e.g. "tenant did not pay rent" → "default in payment of rent by tenant") via templates or a short LLM call.  
  - If the query mentions "Section X" or "Article Y", add a dedicated query like "IPC section X" / "Indian Penal Code section X" for precision.

- **Callers of expand_legal_query**  
  - In `retrieve_bare_acts_phase`, `generate_response_v2`, and any path that builds the search string: loop over the 3 queries, run hybrid retrieval for each, merge results, deduplicate by chunk/section id, then run the existing reranker and take top-k (e.g. 8).

---

#### 4. Statute-anchored case retrieval

- **Section extraction**  
  - After bare-act retrieval, from the top retrieved sections extract **(act_name, section_number)** (and optionally section title). Keep a list of these "statute references" for the case-law step.

- **Case-law queries from sections**  
  - In `generate_final_opinion_with_case_laws()` (or the shared case-retrieval path): build case-law search queries from the extracted sections, e.g. for each top section: `f"{section_number} {act_name} case law interpretation"` and variants (e.g. "Section 10 Telangana Rent Control Act eviction").  
  - Run hybrid case-law retrieval on these statute-anchored queries (and optionally one facts-based query), merge, dedupe, rerank.

- **Authority ranking**  
  - After case-law rerank, apply a **court-tier boost** (e.g. Supreme Court +5, High Court +3, District +1) using existing `court` or binding metadata. Combine with reranker score so authoritative cases rise.

---

#### 5. Reranker candidate pool

- **Larger candidate set before rerank**  
  - In `hybrid_retriever` (or callers): ensure the pipeline retrieves more candidates before the cross-encoder (e.g. vector top 40, BM25 top 40, merge), then rerank and take top 5–8. Remove or relax any cap that forces rerank on only top 10.

---

#### 6. Ingestion and index quality

- **Case law metadata**  
  - In `Ingestion/smart_chunker.py`, improve `_detect_case_metadata` so that case_name, court, year, citation are parsed correctly (avoid garbage like "1959 near the Ahari Payin…"). This improves reranking and statute-anchored matching.

- **Bare-act editorial noise**  
  - In the bare-act chunking path, strip or mask "Subs. by Act…", gazette notifications, footnotes, page numbers before creating chunks so only substantive section text is embedded.

- **Paragraph-type metadata (case law)**  
  - During case-law ingestion, classify each paragraph (e.g. facts, arguments, reasoning, ratio, order) via heuristics or a small LLM call; store `paragraph_type` in chunk metadata. At retrieval, boost by type (e.g. ratio = 1.0, reasoning = 0.8, facts = 0.3).

- **Section refs in cases**  
  - During case-law ingestion, extract phrases like "Section 307 IPC", "Section 27 Arms Act" into a field such as `sections_cited`. Use this later for statute–case linking or filtering.

- **OCR noise**  
  - Add filters in the case-law ingestion path to strip stray page numbers, single letters, and header/footer patterns before embedding.

---

#### 7. Evaluation

- **Align eval with production scores**  
  - In `eval/batch_runner` and any script that writes batch results: ensure the **reranker score** (same scale as production) is written for each result so threshold_sweep and other evals use it.  
  - In `eval/threshold_sweep`, use that score for threshold curves; avoid raw vector distances or a different scale.  
  - Optionally relax or broaden gold matching (e.g. accept same act different section, or multiple valid acts) so improvements in retrieval show up in metrics.

---

#### 8. Optional (later)

- **PDF caching**  
  - When a PDF is fetched (e.g. from India Code), cache it by URL or content hash so the same act is not re-downloaded and re-parsed on every request.

- **Citation graph**  
  - Phase 2: store edges (case → cites → case, case → interprets → section); use at retrieval to expand by precedent chain. No change in the hot path until then.

---

### Vector store: what changes, and when to rebuild

**No index change (retrieval-time only)** — These work with the **existing** FAISS + BM25 + chunks; **no rebuild**:

- Greeting shortcut, disabling sync enrichment, web trigger (count vs score), 3-query cap, parallel web.
- Legal query expansion (3 variants): same index; we just run hybrid search multiple times and merge.
- Statute-anchored case retrieval: we use (act, section) from bare-act results to build case queries; same case index.
- Authority ranking: uses existing `court` (or similar) metadata on chunks.
- Reranker candidate pool (40+40): same index, just pass more candidates to the cross-encoder.
- Eval score alignment: no index change.

**Index content or metadata change — rebuild (or re-ingest) needed:**

| Change | Affects | Rebuild? |
|--------|--------|----------|
| **Fix case law metadata** (case_name, court, year, citation) | Case-law chunks JSON + metadata used at rerank/display | **Yes — re-run case-law ingestion** (or full `build_v2_index` for case laws). Existing chunks have wrong/missing metadata. |
| **Remove bare-act editorial noise** (strip "Subs. by Act…", footnotes) | Bare-act chunk **text** and embeddings | **Yes — re-run bare-act ingestion.** Current chunks contain noise; new chunks will have cleaner text and embeddings. |
| **Add paragraph_type** (facts, reasoning, ratio) to case laws | Case-law chunk **metadata** | **Yes — re-run case-law ingestion** with a pipeline that classifies paragraph type and stores it. Or a one-off migration over existing chunks to add the field. |
| **Add sections_cited** to case laws | Case-law chunk **metadata** | **Yes — re-run case-law ingestion** (or migration) that extracts "Section X Act" and populates sections_cited. |
| **Clean OCR noise** in case laws | Case-law chunk **text** and embeddings | **Yes — re-run case-law ingestion** so chunk text no longer contains page numbers, stray letters, etc. |

**Summary:**  
- **Bare acts:** Rebuild only if you implement **strip editorial noise**. Otherwise no rebuild.  
- **Case laws:** Rebuild (or re-ingest) when you implement **better case metadata**, **paragraph_type**, **sections_cited**, or **OCR cleaning**. You can do one full case-law re-ingestion after all ingestion changes are in place.  
- **Latency and retrieval-quality improvements** (three tiers, 3 queries, statute-anchored case retrieval, authority ranking, no sync enrichment) do **not** require a rebuild; they use the current index.

---

### Summary order

**Architecture (docs):**  
Update Architecture.txt (and optionally CREWAI doc) with three tiers, sync/async boundaries, target retrieval pipeline (query expansion, section extraction, statute-anchored case retrieval, authority ranking), greeting shortcut, and web/enrichment behaviour. Add cross-reference to IMPROVEMENT_ROADMAP.

**Code (suggested order):**  
(1) Disable sync enrichment + fix web trigger + reduce queries + parallelize web.  
(2) Greeting shortcut.  
(3) Legal query expansion (3 queries, synonym + statute-style).  
(4) Statute-anchored case retrieval + authority ranking.  
(5) Reranker candidate pool.  
(6) Ingestion: case metadata, bare-act noise, paragraph_type, sections_cited, OCR.  
(7) Eval score alignment.  
(8) Optional: PDF cache, citation graph later.

No code was written in this section; it is a checklist for architecture updates and implementation.
