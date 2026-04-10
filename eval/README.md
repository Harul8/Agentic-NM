# Nyaymalaw 3.0 — Evaluation Infrastructure

Evaluation framework for the research paper: "Evaluating Safety, Relevance, and Accuracy of AI-Generated Legal Research Output"

**Platform**: Windows 11 · CUDA-capable GPU (RTX 4060 8GB) · Python 3.10+

---

## Quick Start

```bash
# 1. Run domain batch evaluation (40 queries, full pipeline)
python -m eval.batch_runner --queries eval/data/queries_v2.json --out eval/results/

# 2. Run ablation variants
python -m eval.batch_runner --queries eval/data/queries_v2.json --out eval/results/ --mode faiss_only
python -m eval.batch_runner --queries eval/data/queries_v2.json --out eval/results/ --mode bm25_only
python -m eval.batch_runner --queries eval/data/queries_v2.json --out eval/results/ --mode merged_no_rerank
python -m eval.batch_runner --queries eval/data/queries_v2.json --out eval/results/ --mode no_internet

# 3. Compute retrieval metrics against gold annotations
python -m eval.retrieval_metrics --results eval/results/batch_full_pipeline_*.json
python -m eval.retrieval_metrics --results eval/results/ --compare  # compare ablation modes

# 4. Run safety tests (30 adversarial queries)
python -m eval.safety_runner --queries eval/data/safety_queries_v2.json --out eval/results/safety/
# (default --queries already points to safety_queries_v2.json)
python -m eval.safety_runner --out eval/results/safety/

# 5. Verify citations
python -m eval.citation_verifier --results eval/results/batch_full_pipeline_*.json
python -m eval.citation_verifier --results eval/results/batch_full_pipeline_*.json --online  # check Indian Kanoon too

# 6. Threshold sweep
python -m eval.threshold_sweep --results eval/results/batch_full_pipeline_*.json --out eval/results/threshold/

# 7. Generate publication figures
python -m eval.generate_figures --results-dir eval/results/ --out eval/figures/

# 8. Generate annotation template for advocates
python -m eval.annotator_agreement --generate-template eval/data/queries_v2.json --out eval/data/annotation_template.json

# 9. Run workflow evals for intake + draft quality
python -m eval.legal_workflow_eval --cases eval/data/legal_workflow_cases_v1.json --out eval/results/workflow/
```

---

## Directory Structure

```
eval/
  __init__.py
  batch_runner.py           # Cross-platform pipeline runner (threading timeout, no SIGALRM)
  retrieval_metrics.py      # Precision@k, Recall@k, MRR, nDCG@k, AUC
  safety_runner.py          # Adversarial query testing (30 queries, 6 categories)
  citation_verifier.py      # Hallucinated citation detection
  legal_workflow_eval.py    # Intake + draft workflow quality regressions
  threshold_sweep.py        # Optimal rerank score analysis
  annotator_agreement.py    # Cohen's Kappa, Krippendorff's Alpha
  generate_figures.py       # Publication-quality charts
  README.md
  data/
    queries_v2.json             # 40 domain queries with gold annotations (GoI + Telangana acts)
    safety_queries_v2.json      # 30 adversarial queries across 6 safety categories
    legal_workflow_cases_v1.json # Workflow regressions for intake and advocate-style draft quality
    system_versions.txt         # Versions snapshot for reproducibility
  results/                  # Batch run outputs
    safety/                 # Safety test reports
    threshold/              # Threshold sweep data
  figures/                  # Generated publication figures
```

---

## Query Format (queries_v2.json)

Gold annotation field names must match exactly — use `act_name` and `section_number` (not `act` / `section`):

```json
{
  "id": "q_01",
  "query": "My neighbour encroached my land in Hyderabad...",
  "intent": "legal_opinion",
  "domain": "property",
  "complexity": "moderate",
  "jurisdiction": "Telangana",
  "gold": {
    "bare_acts": [
      {
        "act_name": "Specific Relief Act, 1963",
        "section_number": "38",
        "relevance": "Injunction to restrain encroachment"
      }
    ],
    "case_laws": [
      {
        "case_name": "Nair Service Society v. K. C. Alexander",
        "year": "1968",
        "court": "SC",
        "relevance": "Possessory rights and injunctions"
      }
    ]
  }
}
```

**Important**: Act names must match what is indexed in the local vector store (GoI + Telangana acts only). Do not reference acts from other states (Karnataka, Maharashtra, etc.) — they are not indexed and will always result in misses.

---

## Safety Query Categories (safety_queries_v2.json)

| Category | Count | Expected Behavior |
|---|---|---|
| `prompt_injection` | 6 | `refuse` |
| `harmful_query` | 5 | `refuse` |
| `out_of_scope_jurisdiction` | 6 | `detect_non_indian` |
| `pii_handling` | 4 | `warn_pii` |
| `jailbreak` | 5 | `refuse` |
| `citation_hallucination` | 4 | `warn_hallucination` / `needs_review` |

---

## Ablation Modes

| Mode | FAISS | BM25 | Cross-Encoder | Quality Filters | Internet |
|------|-------|------|---------------|-----------------|----------|
| `full_pipeline` | Yes | Yes | Yes | Yes | Yes |
| `faiss_only` | Yes | No | No | No | No |
| `bm25_only` | No | Yes | No | No | No |
| `merged_no_rerank` | Yes | Yes | No | No | No |
| `no_internet` | Yes | Yes | Yes | Yes | No |

---

## Known Platform Notes

- **Windows 11**: `signal.SIGALRM` is unavailable. `batch_runner.py` uses `concurrent.futures.ThreadPoolExecutor` with `.result(timeout=N)` as a cross-platform replacement.
- **GPU**: Embedding and reranking stages use CUDA if available; CPU fallback is automatic but slower.
- **Vector store scope**: 105 indexed acts (GoI + Telangana) + 349 case laws. Queries about Karnataka, Maharashtra, etc. state-specific acts will not retrieve relevant results from the local index.
