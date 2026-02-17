# DEPRECATED — Use retrieval.tiered_search + retrieval.auto_enricher instead.
# This file kept only so old imports don't crash.
from retrieval.tiered_search import search_for_gaps as run_case_law_fallback  # noqa: F401
from retrieval.auto_enricher import enrich_from_gap_results as save_case_laws  # noqa: F401
