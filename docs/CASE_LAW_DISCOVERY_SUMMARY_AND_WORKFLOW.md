# Case Law Discovery — Summary & Workflow

This document summarizes the **new case law discovery functionality** we discussed and presents it as a single workflow view. It covers two entry points (free-form statement vs bare-act–driven) and shared rules for sources, scoring, and selection.

---

## 1. Summary of What We Discussed

### 1.1 Two Ways to Trigger Case Law Discovery

| Trigger | What the user says (example) | What the system does |
|--------|------------------------------|----------------------|
| **Statement-based** | “Find case laws related to land encroachment” | Web search for that topic → filter by court/portal → score → shortlist → present for indexing. |
| **Bare-act–driven** | “For the first 10 bare acts in my vector store, find applicable case laws” | For each of the first 10 acts (alphabetically): go chunk-by-chunk → find case laws per chunk → score → shortlist per act → present for indexing; then next act. |

In both cases: **no auto-index**. All shortlisted documents are **proposed for indexing on the UI**; the user approves before anything is written to the Case Laws folder.

---

### 1.2 Where Things Live (Google Drive)

- The **vector store** is maintained on **Google Drive** mounted by the program (e.g. under `DATA_ROOT`).
- **Case Laws folder** lives in the **same directory (root)** as the vector store — i.e. same parent as `vector_store` (e.g. `DATA_ROOT/CaseLaws` alongside `DATA_ROOT/vector_store`).
- When you say “first 10 acts by acts in the vector store”, the system goes to this vector store, identifies the first 10 bare acts, finds case laws as described, and **stores** (after user approval) the retrieved case law documents in the **Case Laws** folder on the same Google Drive.

---

### 1.3 Summary Index (By Act Name)

- When storing retrieved case laws, maintain a **simple summary index table** keyed **by act name**.
- **Contents:** For each bare act, the index stores **which case laws** (the 25 or fewer) were retrieved and stored for that act. Each entry can be the document **signature** (court + parties from first two pages) and optionally title, filename, or URL.
- **When to update:** **Every time** you save (store) retrieved case laws for an act, **update this summary index** — add or update the row for that act with the list of case laws just stored.
- **Purpose:** This index is the single place to look when checking "already stored" and for deduplication (§1.4). It avoids scanning the whole Case Laws folder or opening PDFs.

---

### 1.4 Duplicate Check and Uniqueness (Top 25 Per Act)

- Before presenting the **top 25** (or shortlist) for a given act, the system must ensure the list is **unique** (no duplicates).
- **Duplicate check is done against the summary index only** (see §1.3; simpler than scanning the Case Laws folder):
  1. **Against the index:** For each candidate document, compute its **signature** (court + parties from first two pages). If that signature **already appears in the summary index** (for this act or for any act), treat it as a duplicate and remove it from the shortlist (do not count toward the 25).
  2. **Within the retrieved set:** Among the candidates for this act, treat two documents as **duplicates** if they have the **same signature** (court + parties). Keep only one (e.g. higher score).
- **How to decide “same document” (duplicate signature):** Use the **first two pages** of each document to derive:
  - **Court:** Whether it is a **Supreme Court** judgment or a **High Court** judgment (which court).
  - **Parties:** From patterns such as **“Party name … appellant versus Respondent name … respondent”** (or equivalent), extract **appellant** and **respondent** (and any other parties if present). Together, court + parties uniquely identify the judgment.
- Two documents are **duplicates** if they have the **same court** and the **same parties** (e.g. same appellant vs respondent). After this check, the **top 25** for this act must be **unique** (not already in the summary index, and not repeated within the 25). Whenever you **save** new case laws for an act, **update the summary index** so future duplicate checks stay correct.

---

### 1.5 Source Priority (Where to Find Documents)

1. **Primary:** Official court websites  
   - Supreme Court of India  
   - High Court of Telangana  
   - Only **official** judgments/orders (e.g. from main.sci.gov.in, tscourt.gov.in or equivalent).

2. **Fallback (only if not enough from courts):** Legal portals  
   - Bar and Bench, India Kanoon, Live Law (and similar).  
   - Use **only** when they host **official PDF copies** (same as court document).  
   - **Do not use** any **unofficial or unauthorized** content from these portals (no summaries, reprints, commentary, or non-official PDFs).

---

### 1.6 Similarity Scoring

- **Document score** for a given “query” (a user statement or a **bare act chunk**):  
  **Maximum** similarity between that query and **any part** of the case law document.  
- Implementation: split the document into segments (e.g. paragraphs or windows), compute similarity of each segment to the query, set **document score = max(segment scores)**.  
- Thresholds below use a **common scale** (e.g. 0–10 or same as your cross-encoder); we refer to them as **3** and **5** (or 0.3 and 0.5 if scale is 0–1).

---

### 1.7 Selection Rules (Shared Idea)

- **Minimum bar:** Drop any document with similarity **≤ 3** (or ≤ 0.3).  
- **Preferred:** Keep all documents with similarity **> 5** (or > 0.5).  
- **If not enough:** When the “preferred” set has fewer than the desired count (e.g. 10 or 25), **fill up** by taking the **top N by similarity** from documents **above the 3 threshold**, until we reach that count (or run out).

So in all cases: **threshold > 3**; prefer **> 5**; if fewer than target, take **top N** (N = 10 or 25 or 2 as below) above 3.

---

## 2. Workflow A — Statement-Based (“Find case laws related to …”)

**Input:** One statement, e.g. “Find case laws related to land encroachment.”

**Steps:**

1. **Web search** for court judgments/orders on that topic (Supreme Court of India, Telangana High Court).
2. **Filter by source:**  
   - Prefer results from **official court sites** (SCI, Telangana HC).  
   - If **insufficient** documents from courts → search **legal portals** (Bar and Bench, India Kanoon, Live Law) and keep **only** links/PDFs that are **official court PDF copies**; discard unofficial content.
3. **Fetch & classify:** For each candidate URL, fetch document; use **first two pages** to confirm it’s an **official** SC/THC judgment or order (or official copy); discard if not.
4. **Score:** For each retained document, compute **similarity** to the user statement (or to a representation of it). Use **max** over document segments if needed.
5. **Select:**  
   - Drop all with similarity **≤ 3**.  
   - Keep all with similarity **> 5**.  
   - If that gives **< 10** documents → take the **top 10** by similarity (still above 3).
6. **Output:** Present the shortlisted documents **on the UI for indexing** (user approval; no auto-index).

---

## 3. Workflow B — Bare-Act–Driven (“First 10 acts, find case laws”)

**Input:** Instruction like “For the first 10 bare acts in my vector store, find applicable case laws.”

**Steps:**

1. **Which acts**  
   - Go to the **vector store** on Google Drive (mounted by the program).  
   - Identify bare acts from the vector store; sort **alphabetically** by act name (deterministic).  
   - Take the **first 10** acts: \( A_1, A_2, \ldots, A_{10} \).

2. **For each act \( A_i \)** (process one act fully before moving to the next):

   **2.1 Get chunks**  
   - From the vector store, get all **chunks** for act \( A_i \): \( C_{i,1}, C_{i,2}, \ldots \).

   **2.2 For each chunk \( C_{i,j} \):**

   - **Search** for case laws relevant to **this chunk** (same source priority: courts first, then portals for official PDFs only).  
   - **Score:** For each case law document, set **document score = max** similarity between **this chunk** and **any part** of the document (segment document, compute chunk–segment similarity, take max).  
   - **Per-chunk selection:**  
     - Keep all documents with score **> 5**.  
     - If fewer than **2** with score > 5 → take the **top 2** by similarity with score **> 3**.  
     - So: **best 2** case laws per chunk (using >5, else top 2 above 3).

   **2.3 Merge for act \( A_i \):**

   - **Union** all documents selected for any chunk of this act.  
   - For each document, keep its **maximum** score over all chunks where it was selected.  
   - **Per-act selection:**  
     - Keep **all** with (act-level) score **> 5**.  
     - If that’s **fewer than 25** → take the **top 25** by score (minimum score still **> 3**).  
   - So: **up to 25** case laws per act (all >5, or top 25 above 3).

   **2.4 Duplicate check (so top 25 are unique)**

   - **Against the summary index:** For each document in the candidate set, use the **first two pages** to compute a **signature**: (court, parties). Court = Supreme Court vs High Court (and which High Court if needed). Parties = appellant vs respondent (from patterns like “Party name … appellant versus Respondent name … respondent”). If this signature **already appears in the summary index** (for this act or for any act), treat as duplicate and remove from the shortlist (do not count toward 25). No need to scan the Case Laws folder — the index is the source of truth.
   - **Within the retrieved set:** Among the remaining candidates treat two documents as **duplicates** if they have the **same signature** (court + parties). Keep only one (e.g. higher score) so each judgment appears at most once.
   - After both checks, the **top 25** for this act must be **unique** (no duplicate against Drive, no duplicate within the 25). If duplicates were removed, the list may have fewer than 25 items.

   **2.5 Store, update index, and present for indexing**

   - **Store:** Retrieved (and approved) case law documents are stored under the **Case Laws folder** in the **same directory as the vector store** on Google Drive (e.g. `DATA_ROOT/CaseLaws`).
   - **Update summary index:** **Every time** you save case laws for an act, **update the summary index** — add or update the row for that act with the list of case laws just stored (act name → list of signatures or doc identifiers).
   - **Present:** Show the **unique** shortlisted case laws for act \( A_i \) **on the UI for indexing** (user approval).  
   - Then move to the **next act** \( A_{i+1} \) and repeat from step 2.1.

3. **After all 10 acts**  
   - User can approve/refine per act; **no auto-index**.  
   - Approved documents are written to the **Case Laws** folder on Google Drive (same root as vector store).  
   - **Summary index** is updated each time case laws are saved for an act (act name → list of those case laws).  
   - Deduplicate across acts when writing (same document may appear for multiple acts; write once).

---

## 4. Workflow View (High Level)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CASE LAW DISCOVERY — HIGH-LEVEL FLOW                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │ User input            │
  │ (statement or        │
  │ “first 10 acts”)     │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐     ┌─────────────────────────────────────────────┐
  │ Which flow?          │────▶│ A. STATEMENT: one query → search → score   │
  └──────────┬───────────┘     │    → select (top 10, >5 / >3) → propose    │
             │                 └─────────────────────────────────────────────┘
             │                 ┌─────────────────────────────────────────────┐
             └────────────────▶│ B. BARE ACTS: for each of first 10 acts      │
                               │    → for each chunk: search → score (max)   │
                               │    → best 2 per chunk (>5 / top 2 above 3)  │
                               │    → per act: >5 or top 25 above 3          │
                               │    → propose per act → next act             │
                               └─────────────────────────────────────────────┘
             │
             ▼
  ┌──────────────────────┐
  │ SOURCES (both flows)│
  │ 1. Courts (SCI, THC)│
  │ 2. If needed:       │
  │    Portals (official │
  │    PDF copies only) │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ SCORING              │
  │ Doc score = max      │
  │ similarity of any    │
  │ part of doc to query │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ SELECTION            │
  │ • Min threshold > 3  │
  │ • Prefer all > 5     │
  │ • If < N: top N      │
  │   (N = 2 / 10 / 25)  │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐     (B only)
  │ DUPLICATE CHECK     │◀──── Against summary index only
  │ • vs summary index  │      (act name → list of case laws)
  │   (not full folder) │      • vs index (any act)
  │ • within retrieved  │      • within the 25
  │   set               │      Signature: court + parties
  │ Then update index   │        (first 2 pages)
  │ on every save       │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ OUTPUT / STORAGE     │
  │ • Propose for       │
  │   indexing on UI    │
  │ • Store in Case     │
  │   Laws folder (same  │
  │   root as vector    │
  │   store on Drive)   │
  │ • Update summary    │
  │   index (by act     │
  │   name) on every    │
  │   save              │
  │ No auto-index       │
  └──────────────────────┘
```

---

## 5. Quick Reference Table

| Item | Statement-based (A) | Bare-act–driven (B) |
|------|--------------------|----------------------|
| **Input** | One statement (e.g. “Find case laws related to land encroachment”) | “First 10 bare acts → find case laws” |
| **Scope** | One query | 10 acts; per act, all chunks |
| **Search** | Web for topic | Per chunk: web for that chunk’s topic |
| **Score** | Max similarity (query vs any part of doc) | Max similarity (chunk vs any part of doc) |
| **Per-unit selection** | — | Best **2** per chunk (>5, else top 2 above 3) |
| **Final selection** | All >5; if <10 → **top 10** above 3 | Per act: all >5; if <25 → **top 25** above 3 |
| **Min threshold** | > 3 | > 3 |
| **Output** | One shortlist → propose for indexing | One shortlist **per act** → propose for indexing |
| **Storage** | — | Case Laws folder in **same directory as vector store** on Google Drive |
| **Summary index** | — | (B) Simple table **by act name** → list of case laws (25 or fewer) stored for that act. **Update on every save.** Used for duplicate check only (no folder scan). |
| **Duplicate check** | — | (B) **Against summary index only:** (1) if signature (court + parties) already in index → duplicate; (2) **within** the 25, same signature → duplicate. Top 25 **unique**. Update index when saving. |

---

## 6. Duplicate Check (Summary) — Against Summary Index Only

| Check | What | How (signature) |
|-------|------|------------------|
| **Against summary index** | Case laws already in the **summary index** (by act name → list of case laws) | For each candidate: extract from **first two pages**: **court** (Supreme Court vs High Court), **parties** (appellant vs respondent from “Party … appellant versus Respondent … respondent”). If (court + parties) already in index (this act or any act) ⇒ duplicate; drop. No folder scan. |
| **Within retrieved set** | Among the 25 (or pool) for this act | Same signature (court + parties). Keep one (e.g. higher score); drop the rest so the final list has **no duplicates**. |

Result: The **top 25** (or fewer) are **unique** — not already in the summary index and not repeated within the list. **Update the summary index** every time case laws are saved for an act.

---

## 7. Document Status

- This file is a **summary and workflow view** of the case law discovery feature as discussed.  
- **Storage:** Vector store and Case Laws folder are on Google Drive (same root); retrieved case laws are stored in the Case Laws folder.  
- **Summary index:** Simple table by act name → list of case laws stored for that act; **update on every save**. Duplicate check is **against this index only** (no Case Laws folder scan). First-two-pages signature (court + parties); top 25 per act are unique.  
- Detailed algorithm (e.g. per-chunk “2” and per-act “25”) can be aligned with `SPEC_BARE_ACT_CASE_LAW_PROPOSAL.md` when implementing (that spec currently uses “up to 5 per chunk”; our discussion used **best 2 per chunk** and **top 25 per act**).

---

## 8. Persistence and Dynamic Workflow (Implementation)

### 8.1 Documents presented for indexing — persist until user action

- **Documents presented for indexing** are stored in a **persistent store** (e.g. `case_law_discovery_pending.json` under `DATA_ROOT`).
- They **remain there** across UI refresh, backend restart, and re-running the backend. **Nothing** (refresh, restart, or re-run) clears them automatically.
- They are cleared **only** when the user **completes indexing** (confirms Index for selected documents) or clicks the **Clear / Discard** button on the UI.
- The UI loads this list on mount and after Index/Clear so the same list is always shown until the user acts.

### 8.2 Dynamic workflow — only limits are hardcoded

- **Everything except the following is dynamic** (LLM-driven): similarity score limits (e.g. > 3, > 5) and document count limits (e.g. top 10 statement, best 2 per chunk, top 25 per act).
- **Dynamic parts:** What the user wants (statement vs first-N-acts), how to search, which sources, how to extract court/parties, etc. Additional LLM calls are acceptable.
- **Implementation:** A separate end-to-end workflow in `case_law_discovery/`; it does not modify existing agents or services and uses them without changing their behaviour.

### 8.3 Routing from chat to the separate workflow

- When the user sends a message in the **main chat** (e.g. "find case laws for the first bare act in vector store"), the backend **does not** run the usual pipeline (local FAISS/BM25 → web search → response generator). Instead it detects that the request is a **case law discovery** request and runs **only** the case law discovery workflow.
- Detection is a lightweight keyword check (no LLM): e.g. "first" + "bare act" or "vector store" + "case law". If it matches, the message is handled by `case_law_discovery.workflow.run()` and the response tells the user to check **Case law discovery – Pending** in the left sidebar. The main chat pipeline (fact collection, response generation, local-then-web search) is never invoked for these messages.
