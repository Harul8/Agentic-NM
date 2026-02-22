# CrewAI-Based Multi-Agent Architecture for Nyaymalaw

**Goal:** Replace the current single-pipeline flow with a **dynamic, query-driven multi-agent system** so that:
- A **parent/orchestrator agent** understands the query and decides **what** needs to be done.
- The **workflow is not fixed**: specific lookups (bare act / case law) go straight to search; legal-opinion requests trigger conversation to gather details first.
- The design is **modular** and **parallelizable**, and **cloud-ready** (e.g. RunPod) for scaling later.

For a high-level diagram of agents and data flow, see `docs/Architecture.txt`.

---

## 1. Routing: Two Gates + Orchestrator

Routing is **robust and vigilant**: the user message passes through **two gates** so that legal and non-legal (generalist) requests are clearly separated, and only legal requests are further classified.

### 1.1 Gate 1 (Legal vs Generalist vs Greeting)

- **Input:** User message + recent conversation.
- **Output:** One of **GREETING** | **LEGAL** | **GENERALIST**.
- **GREETING** — Hello, thanks, small talk with no substantive request → reply warmly and invite a legal query.
- **GENERALIST** — Any request that is **not** clearly one of: (1) find case laws, (2) find bare act sections, (3) legal advice on a situation, (4) fetch/index acts from a URL. This includes: politics, science, technology, general knowledge, how-to, trivia, and **any unclear or ambiguous** request. When in doubt, route to **GENERALIST**.
- **LEGAL** — Only when the user **clearly** wants one of the four legal types above → proceed to **Gate 2**.

### 1.2 Gate 2 (Legal intent only)

- **Input:** Only when Gate 1 returned **LEGAL**.
- **Output:** Exact intent and params: **search** | **lookup** | **legal_opinion** | **bulk_ingest** (with `result_count`, `source_url`, `facts_summary`, `reply_to_client` as needed). No generic_chat in Gate 2.
- The CrewAI orchestrator (or equivalent) can be used as fallback for Gate 2 if the dedicated Gate 2 classifier fails.

### 1.3 Agents after routing

- **search** / **lookup** / **legal_opinion** → Response generator (retrieval + summary).
- **bulk_ingest** → Bulk ingest service (crawl URL → save to Drive → index).
- **generic_chat** (Generalist) → Generalist agent (no legal retrieval; answer like ChatGPT/Perplexity).
- **greeting** → Greeting response only.

---

## 2. Behavior by User Intent

### 2.1 Routing Table

| User intent / phrasing | Behavior | Retrieval | Summary / output |
|------------------------|----------|-----------|------------------|
| **Bare act only** (e.g. "show me sections of IPC on X") | Route: `lookup` | Only bare act retrieval | Summary: **only relevant bare act sections** (no case laws) |
| **Case laws only** (e.g. "find case laws on X") | Route: `search` | Only case law retrieval | Summary: **dispute (facts) + court order/judgement in brief** per case |
| **Legal opinion** (user shares a problem, needs an opinion) | Route: `legal_opinion` | First relevant bare acts, then case laws **relevant to dispute + those bare sections** | Full legal opinion with bare acts + case laws |
| **Specific pull** (e.g. "pull 2 Supreme Court case laws for [dispute]") | Route: `search` with `result_count` | Exactly that many, most relevant | Same: dispute + order/judgement per case |
| **"Index all the bare act candidates"** (or similar) | User wants to index all candidates shown in the Pending indexing UI | No retrieval | Frontend: use **"Index all"** button to index all listed candidates (or only bare-act if specified). Backend indexes only what is sent. |
| **Generalist** (any query not covered by legal agents: politics, science, tech, general knowledge, how-to, unclear) | Route: `generic_chat` → **Generalist Agent** | No legal retrieval | **Generalist Agent** answers like ChatGPT/Perplexity; handles all non-legal and uncovered conversations. |
| **Bulk ingest** ("get all acts from [URL], get PDFs, store in drive and index") | Route: `bulk_ingest` | No search/lookup | **General-task agent**: fetch browse URL → discover act links → for each: get PDF, save to Drive, run existing ingestion (chunk + index). Return summary (discovered, saved, indexed). |

### 2.2 Intent → Backend

- **lookup** → Bare-act-only retrieval; bare-act-only summary.
- **search** → Case-law-only retrieval; dispute + order/judgement summary; respect `result_count` when specified.
- **legal_opinion** → Both bare acts and case laws (case laws aligned to dispute + retrieved bare sections).
- **generic_chat** → **Generalist Agent**: no legal retrieval; handles all general questions and any query not covered by search/lookup/legal_opinion/bulk_ingest. Single generalist agent for robustness.
- **bulk_ingest** → General-task agent: crawl the user-provided URL (e.g. India Code state acts browse page), discover act links, for each fetch PDF → save to Drive → trigger existing ingestion modules (chunk + index). Return a summary (discovered, saved, indexed).
- **"Index all"** → Trigger indexing of all (or all bare-act) candidates in the UI via the "Index all" button.

---

## 3. Indexing UI (Left Pane)

Indexing is **user-driven and explicit**. The response pipeline may return **indexing_candidates** (e.g. from web enrichment). These are shown in a **Pending indexing** section **below Chat history** in the left pane.

- **Per document:**
  - **Radio** — Select/unselect for indexing.
  - **Hyperlink** — Source URL.
  - **Dropdown** — **Category:** "Bare act" or "Case law" (user can change before indexing).

- **At the top of the Pending indexing section:** **"Index"** (sends only **selected** documents) and **"Index all"** (sends **all** listed candidates). When the user asks to "index all the bare act candidates" (or similar), they use the **Index all** button. After successful indexing, indexed items are removed from the list.

- **Backend:** The "Index" / "Index all" actions call the indexing API with `{ items: [ { title, source_url, category }, ... ] }`. The backend runs indexers only for the items received.

---

## 4. Dynamic Workflow (Per Route)

The workflow is **not** one fixed DAG; it is a **family of DAGs** selected by the orchestrator.

### 4.1 Route: greeting / chat

Orchestrator → route = greeting/chat. No retrieval; optional short reply.

### 4.2 Route: lookup (bare act only)

Orchestrator → complete, intent = lookup → **only** bare act retrieval → summary = relevant bare act sections only (no case laws in summary).

### 4.3 Route: search (case laws only)

Orchestrator → complete, intent = search → **only** case law retrieval → summary = for each case: dispute (facts) + court order/judgement in brief; respect `result_count` if present.

### 4.4 Route: legal_opinion

Orchestrator → ask (optional) or complete → fact collection as needed → **both** bare act and case law retrieval (case laws relevant to dispute + bare sections) → full legal opinion.

### 4.5 Route: generic_chat (Generalist Agent)

Gate 1 → **GENERALIST** (or equivalent) → **Generalist Agent**. Handles all conversations and queries that are **not** legal (search, lookup, legal_opinion, bulk_ingest): politics, science, technology, general knowledge, how-to, trivia, and any unclear request. Answers like ChatGPT/Perplexity; no legal retrieval.

### 4.6 Route: bulk_ingest (general-task agent)

Orchestrator → complete, intent = **bulk_ingest**, **source_url** = URL from user message. A **general-task agent** (bulk ingest service) runs: fetch the browse page → parse act links → for each act page: fetch (HTML/PDF) → save PDF to Drive → run **existing ingestion** (chunk + add to FAISS/BM25). No legal search or opinion; outcome is a summary (e.g. "Discovered N acts; saved M PDFs; indexed M documents").

### 4.7 Index-all instruction

No retrieval; user uses the **Index all** button in the Pending indexing section to send all (or all of a category) candidates to the indexing API.

---

## 5. Web / Enricher and Indexing Candidates

The **web enricher** runs tiered web search and fetches content **for the current response only**. It does **not** auto-index. When it finds documents suitable for the library, it outputs **indexing_candidates** (title, source_url, suggested_category, snippet). These are sent to the frontend and shown in the Indexing UI; the user chooses what to index and triggers **Index** (selected) or **Index all** (all listed).

---

## 6. Implementation Checklist (Summary)

1. **Two-gate routing** — **Gate 1**: Legal vs Generalist vs Greeting (strict; when in doubt → Generalist). **Gate 2**: Only for LEGAL; classify as search | lookup | legal_opinion | bulk_ingest. Ensures robust routing before any agent.
2. **Fact collector** — Runs Gate 1 first; if LEGAL runs Gate 2 (legal intent only); supports **Generalist Agent** (`generic_chat`) for all non-legal and uncovered queries, and **bulk_ingest** (with **source_url**).
3. **Response generator** — Intent-based retrieval (lookup = bare only; search = case only; legal_opinion = both); summary by intent (bare-only / dispute+order / full opinion).
4. **Interactive chat** — When intent = generic_chat, call **Generalist Agent** (_run_generic_chat: no retrieval, general reply for any non-legal query). When intent = **bulk_ingest**, call **bulk ingest service**; return summary as explanation.
5. **API** — Maps `response_type` (e.g. generic_chat) to UI; returns indexing_candidates when present.
6. **Bulk ingest (general-task agent)** — `services/bulk_ingest.run_bulk_ingest(browse_url)` fetches the page, discovers act links (e.g. India Code handle links), then for each calls existing `enrich_from_search_result` (fetch PDF → save to Drive → chunk → index). Allowed domains: indiacode.nic.in, legislative.gov.in.
7. **Indexing UI (left pane):** "Pending indexing" section below Chat history with per-document select, link, category dropdown, **Index** button (selected) and **Index all** button (all listed). "Index" / "Index all" call the indexing API with the chosen items; backend indexes only those items.

---

## 7. Flow: Bulk Ingest from URL (e.g. “Get all Telangana acts from indiacode, store in Drive, index”)

This section describes the **sequence of activities** and **end outcome** when the user asks the system to: *get all acts enacted by a state from a given URL (e.g. India Code browse page), download original PDFs, store them in Drive, and index them*. No code changes are specified here—only the flow from the architecture’s perspective.

### 7.1 Example user prompt

- *“Please get all the acts enacted by the state of Telangana hosted at [indiacode.nic.in browse URL] — get the original PDF documents, store them in the drive and index them.”*

### 7.2 How the model flows (sequence of activities)

1. **User sends the prompt** in chat (e.g. paste URL + “get PDFs, store in drive, index”).

2. **Orchestrator (query router)**  
   - Receives the message and conversation context.  
   - Classifies intent as a **bulk-ingest / crawl-from-URL** type request (e.g. explicit intent such as `bulk_ingest` or equivalent, when supported).  
   - Outputs: `action: "complete"`, intent indicating bulk ingest, and structured parameters: **source URL** (the indiacode browse page), **scope** (e.g. “all acts on this page” or “Telangana state acts”), **tasks**: get original PDFs, store in Drive, index.  
   - No fact-collection Q&A is needed; the URL and instructions are sufficient.

3. **Bulk ingest / crawl pipeline (downstream of orchestrator)**  
   - **Discover:** Fetch the browse URL (e.g. indiacode.nic.in handle/2508/…); parse the page to discover all act links (short titles, detail pages, or direct PDF links).  
   - **Resolve per act:** For each discovered act, resolve the **original PDF** URL (from act detail page or browse listing).  
   - **Download:** For each PDF URL, download the document (same “get original PDF” behavior as elsewhere in the system).  
   - **Store in Drive:** Save each PDF to the configured Google Drive (or equivalent) under a suitable structure (e.g. `BareActs/Telangana/` or `BareActs/State/Telangana/`), with a stable filename (e.g. short title + identifier).  
   - **Index:** For each saved PDF, run the existing **bare-act ingestion path**: extract text, chunk (e.g. smart chunker), embed, and add to the **vector store** (FAISS) as bare act content. Metadata (act name, state, source URL) is stored so the act can be retrieved in normal lookup/search flows.  
   - **Progress and errors:** Optionally report progress (e.g. “Fetched 45 of 120”) and record which items failed (e.g. PDF missing, timeout) for the final summary.

4. **Response generation**  
   - The pipeline produces a **structured summary** of what was done: number of acts discovered, number of PDFs downloaded, number saved to Drive, number indexed, and any failures or skipped items.  
   - This summary is returned to the user as the **reply** (e.g. in the chat as the main message, with optional breakdown or link to Drive folder).

5. **API / frontend**  
   - The chat API returns this reply as the **opinion_text** (or equivalent) and can optionally attach a short **progress or report** (e.g. “Ingested 98 acts; 2 failed”).  
   - No “Pending indexing” step is required for this flow because the user explicitly asked to **store and index** in the same request; the bulk pipeline performs both.

### 7.3 End outcome (what the user sees)

- **In chat:** A clear, concise message along the lines of:  
  - *“I’ve processed the Telangana acts from the India Code page you shared. Discovered **N** acts; downloaded **M** original PDFs, saved them to Google Drive under [BareActs/Telangana/], and indexed **M** documents. You can now search or ask questions about these acts in the app. [If any failed: **K** could not be fetched or indexed—see details below.]”*  
- **On Drive:** A folder (e.g. `BareActs/Telangana/`) containing the saved PDFs, one file per act.  
- **In the app:** The indexed acts are part of the vector store; subsequent **lookup** or **legal_opinion** queries that touch Telangana state law can retrieve these sections.

### 7.4 How this fits the rest of the architecture

- **Orchestrator:** Extends the intent set to include “bulk ingest from URL” so this request is not treated as a generic chat or a single-document lookup.  
- **No conflict with “Index” / “Index all”:** Those remain for **candidate lists** produced by the **web enricher** (e.g. after a legal opinion). Here, the user supplies the URL and asks for direct fetch + store + index in one go.  
- **Reuse:** “Get original PDF,” “save to Drive,” and “index as bare act” reuse the same concepts and components as the existing enrichment and indexing paths (e.g. `fetch_content_and_pdf`, save to Drive, chunk + embed + FAISS); the only new part is **discovery** from a browse page and **batch** execution over many acts.

---

## 8. Retrieval and similarity – validation notes

### 8.1 Why you might see only one case in "Case laws (search results)"

- **Chunks vs cases:** Retrieval returns **chunks** (paragraph-level). Multiple chunks can come from the **same judgment**. The pipeline groups by case and shows one row per case. If the top N chunks all belong to one judgment, you see **one case**.
- **Fix (implemented):** **Case-level diversity**: before formatting, we cap chunks per case (e.g. 4 per case) so more **distinct** cases appear. We also fetch more case-law chunks (top_k 45 for search) so there are more candidates to diversify from.

### 8.2 Indexing

- Internal case-law search uses the vector store under `DATA_ROOT` (e.g. `caselaws_v2_chunks.json`, `caselaws_v2.index`). Run `python scripts/list_indexed_cases.py` to list indexed case names.

### 8.3 Similarity pipeline

- **Local:** FAISS + BM25 → merge → cross-encoder re-rank → filter (MIN_RERANK_SCORE 3.0, quality) → **diversify by case** (max 4 chunks per case) → format (one row per case).
- **Web:** Gap-based tiered search → fetch PDFs → cross-encoder score → keep if ≥ WEB_MIN_SCORE (1.0). Same diversity and formatting after merge.
