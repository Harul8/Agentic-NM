
For the first 10 bare acts (by alphabetical order) in the local vector store:

- For each bare act chunk, search the web for court judgments/orders on that topic.
- Keep only **official Supreme Court of India** and/or **Telangana High Court** documents.
- Score each document by **maximum similarity** of any part of that document to the bare act chunk.
- Apply per-chunk and per-act selection rules.
- Propose the shortlisted case laws for indexing to the Case Laws folder (user approval required; no auto-index).

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **First 10 bare acts** | The first 10 bare acts when ordered **alphabetically** by act name (or by a stable identifier used in the vector store). Ordering must be deterministic and consistent across runs. |
| **Bare act chunk** | A single chunk from the vector store belonging to one of these 10 acts. Each act may have many chunks; process one chunk at a time. |
| **Document** | A single case law PDF/document (e.g. one judgment or order) fetched from the web. |
| **Similarity score (document)** | For a given bare act chunk and a given document: the **maximum** similarity between that bare act chunk and **any part** of the document. Implementation: split the document into segments (e.g. by paragraph or fixed-size sliding window), compute similarity of each segment to the bare act chunk using the existing filtering/scoring logic (e.g. cross-encoder), and set the document’s score = **max** over those segment scores. |
| **Official SC/THC document** | A document that is classified (from its **first two pages**) as an official Supreme Court of India and/or Telangana High Court judgment or order. Classification can use domain (e.g. main.sci.gov.in, tscourt.gov.in) and/or content (LLM or heuristics on first two pages). |

---

## 3. Inputs

- **Source:** Local vector store (Google Drive) — bare act index and chunks.
- **List of acts:** All bare acts present in the vector store; sort alphabetically by act name; take the **first 10**.
- **Per act:** All chunks belonging to that act (from the vector store chunk index).

No other user input is required for the core flow; the “first 10 acts” and “propose for Case Laws folder” are fixed by this spec.

---

## 4. Algorithm (step-by-step)

### 4.1 Enumerate acts and chunks

1. List all bare acts in the local vector store (from chunk metadata or manifest).
2. Sort by act name **alphabetically** (case-insensitive, stable sort).
3. Take the **first 10** acts: \( A_1, A_2, \ldots, A_{10} \).
4. For each act \( A_i \), get all chunks \( C_{i,1}, C_{i,2}, \ldots \) from the vector store.

### 4.2 Per bare act chunk: web search and document scoring

For each act \( A_i \) and each chunk \( C_{i,j} \):

1. **Search the web** for court judgments/orders related to the topic of \( C_{i,j} \) (and optionally the act name). Use queries targeting Supreme Court and Telangana High Court sources where possible.
2. **For each search result URL** (until a reasonable limit per chunk, e.g. top N results):
   - Fetch the document (PDF or content).
   - **First-two-pages check:** Extract text from the first two pages. Classify:
     - If **not** official Supreme Court and/or Telangana High Court → **discard** this document; do not score.
     - If **yes** → continue.
   - Extract full text (or sufficient text) of the document.
   - **Document similarity score:** Split the document into segments (paragraphs or fixed-size windows). For each segment, compute similarity to the bare act chunk \( C_{i,j} \) using the existing scoring/filtering logic (e.g. cross-encoder). Set **document score = max(segment scores)**.
   - Retain the document with this score for the current chunk.

### 4.3 Per-chunk selection (documents for this chunk)

From the documents retained for chunk \( C_{i,j} \) (each with one score):

- **Set H:** All documents with **score > 5.0**.
- If **|H| ≥ 5:** selected set for this chunk = H.
- If **|H| < 5:** selected set = H ∪ (up to 5 − |H| documents with the **top** similarity scores among the rest, **and** score **> 3.0**).  
  So: add documents in order of decreasing score until we have 5 total or no more with score > 3.0.

Call this selected set \( D_{i,j} \) (each document in \( D_{i,j} \) has a score for this chunk).

### 4.4 Merge across chunks of the same act

For act \( A_i \):

- **Union:** \( U_i = \bigcup_j D_{i,j} \) (all documents selected for any chunk of this act).
- **Per-document act-level score:** For each document \( d \in U_i \), set  
  **score_i(d) = max** over all chunks \( j \) for which \( d \in D_{i,j} \) of the score that \( d \) had for that chunk.
- **Final set for act \( A_i \):**
  - Let \( T = \{ d \in U_i : \text{score}_i(d) > 5.0 \} \).
  - Let \( Top25 \) = top 25 documents in \( U_i \) by \( \text{score}_i(d) \) (descending).
  - **If |U_i| ≤ 25:** shortlist for act \( A_i \) = \( U_i \).
  - **If |U_i| > 25:** shortlist for act \( A_i \) = **whichever is larger**: \( T \) or \( Top25 \).  
    (So if more than 25 documents have score > 5.0, keep all of those; otherwise keep top 25 by score.)

### 4.5 Deduplicate and propose for indexing

- **Union over acts:** \( \mathcal{D} = \bigcup_{i=1}^{10} \) (shortlist for act \( A_i \)).
- **Deduplicate** by document (e.g. by URL, or by title + year + court) so each document appears once.
- **Propose:** Add each unique document to the **proposed-for-indexing** list (Case Laws folder). This is the same “proposal” / “pending indexing” mechanism as elsewhere: user reviews and confirms; indexing then writes to the Case Laws folder on Google Drive. **No auto-index** in this spec.

---

## 5. Outputs

- **Proposed case laws:** A list of unique documents (metadata: title, URL, source, court type, and optionally act-level scores) to be presented to the user for approval.
- **On approval:** Indexing runs and writes approved documents to the **Case Laws** folder and into the case law vector store, per existing indexing pipeline.

---

## 6. Implementation notes

- **Similarity scale:** Use the same scoring as elsewhere (e.g. cross-encoder); thresholds 3.0 and 5.0 are on that scale.
- **“Any part” of the document:** Implement by segmenting the document (e.g. paragraphs or overlapping windows of fixed token length) and taking **max** over segment scores. Segment size/overlap can follow existing case-law chunking if available.
- **First-two-pages:** Reuse the same “first two pages” extraction and a **court classifier** (SC/THC, official only). Can be LLM-based or heuristic (e.g. domain allowlist: main.sci.gov.in, tscourt.gov.in, etc.).
- **Rate limiting:** Web search and fetch should be rate-limited; consider running as a **background job** with progress reporting.
- **Limits:** Optionally cap (e.g. max documents to fetch per chunk, max chunks per act) to control run time and API usage.

---

## 7. Edge cases

| Case | Handling |
|------|----------|
| Fewer than 10 acts in store | Use all available acts (still in alphabetical order). |
| Act has zero chunks | Skip that act. |
| No documents pass first-two-pages for a chunk | \( D_{i,j} = \emptyset \) for that chunk. |
| No document has score > 3.0 for a chunk | \( D_{i,j} \) = at most the set with score > 5.0 (possibly empty). |
| Tie in scores when taking top 25 | Break ties arbitrarily but consistently (e.g. by URL or title). |

---

## 8. Summary

- **10 acts:** Alphabetical order; first 10 from local vector store.
- **Document score:** Max similarity of any part of the document to the bare act chunk.
- **Per-chunk:** All with score > 5.0; if fewer than 5, add up to 5 with top scores and score > 3.0.
- **Per-act:** Union over chunks; assign each document its max score over chunks; if > 25 documents, keep max(top 25 by score, all with score > 5.0).
- **Output:** Deduplicated list of case laws **proposed** for indexing to the Case Laws folder (user approval required).

