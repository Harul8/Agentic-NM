"""
Case law discovery — separate end-to-end workflow.

- Presents documents for indexing; persisted until user completes indexing or clicks Clear.
- Workflow is dynamic (LLM-driven) except hardcoded limits below.
- Does not modify existing modules; uses them where applicable.
"""

from case_law_discovery.limits import (
    SIMILARITY_THRESHOLD_LOW,
    SIMILARITY_THRESHOLD_HIGH,
    TOP_N_STATEMENT,
    TOP_N_PER_ACT,
    BEST_PER_CHUNK,
)

__all__ = [
    "SIMILARITY_THRESHOLD_LOW",
    "SIMILARITY_THRESHOLD_HIGH",
    "TOP_N_STATEMENT",
    "TOP_N_PER_ACT",
    "BEST_PER_CHUNK",
]
