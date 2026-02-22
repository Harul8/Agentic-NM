# Eval JSON rendering on UI

When adding or changing how eval JSON files are displayed in the app (Eval Results section in `Frontend/src/App.jsx`), follow these conventions:

## Conventions

1. **No collapsibles for eval data**  
   Prefer flat tables. Do not use `<details>`/collapse for eval JSON; show everything as tables with sub-columns where needed.

2. **Metrics objects** (e.g. `compare.json`: bm25, merged, full)  
   - **Rows:** metric names (e.g. `bare_act_mrr`, `bare_act_ndcg@10`).  
   - **Columns:** `mean`, `n`, `stdev`, `ci_95`.  
   - One table per run; run name as heading only.

3. **Aggregated-by-key objects** (e.g. `threshold_sweep.json` → `aggregated`)  
   - **Rows:** outer keys (e.g. threshold values `0`, `0.05`, `0.1`, …), sorted numerically.  
   - **Columns:** first column = key (e.g. `threshold`), then all inner keys (e.g. `mean_precision`, `mean_recall`, `mean_f1`, `mean_retrieved_count`, `n_queries`).

4. **Array of objects** (e.g. batch result files)  
   - Flatten each row: nested objects become **sub-columns** (e.g. `filter_results.bare_acts_after_score_filter`, `filter_results.case_laws_after_score_filter`, …).  
   - Arrays become a **single value**: length (e.g. `raw_scores.bare_act_rerank_scores` = count).  
   - One flat table; no nested tables or collapse in cells.

5. **New eval JSON shapes**  
   When introducing new eval outputs, add detection and rendering in `JsonToTable` so they follow the same style: tables with clear row/column roles, no unnecessary nesting or collapse.

## Where it lives

- **Component:** `JsonToTable` in `Frontend/src/App.jsx` (Eval Section).  
- **API:** `GET /eval/list`, `GET /eval/file?path=...` in `api_server.py`.  
- **Styles:** `.eval-table`, `.eval-table--metrics`, `.eval-table--flat`, etc. in `Frontend/src/App.css`.

Keep this rule in mind whenever touching eval UI or adding new eval JSON formats.
