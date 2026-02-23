"""
Build or refresh the bare act summary index (act_name -> summary text).

Can be run standalone (reads chunks from BARE_CHUNKS_V2) or called from
rebuild_vector_store after building the index. Uses get_or_create_act_summary
so existing summaries are reused and only missing acts get an LLM-generated summary.

Run from project root:
    python scripts/build_bare_act_summary_index.py

Or call build_bare_act_summary_index() from another module, optionally passing
bare_chunks to avoid re-reading from disk.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BARE_CHUNKS_V2
from case_law_discovery.workflow import get_or_create_act_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_bare_act_summary_index(bare_chunks=None):
    """
    Build or refresh the bare act summary index for all acts in the vector store.

    Args:
        bare_chunks: Optional list of chunk dicts (each with act_name, etc.).
                     If None, chunks are loaded from BARE_CHUNKS_V2.

    Returns:
        Number of acts for which the summary index was updated/ensured.
    """
    if bare_chunks is None:
        if not os.path.isfile(BARE_CHUNKS_V2):
            logger.warning("Bare chunks file not found: %s. Run rebuild_vector_store first.", BARE_CHUNKS_V2)
            return 0
        from retrieval.hybrid_retriever import load_chunks
        chunks_dict = load_chunks(BARE_CHUNKS_V2)
        if not chunks_dict:
            logger.warning("No chunks in %s", BARE_CHUNKS_V2)
            return 0
        bare_chunks = list(chunks_dict.values())

    chunks_by_act = {}
    for c in bare_chunks:
        name = (c.get("act_name") or "").strip()
        if name:
            chunks_by_act.setdefault(name, []).append(c)

    if not chunks_by_act:
        logger.info("No acts with act_name in chunks.")
        return 0

    logger.info("Building bare act summary index for %d acts...", len(chunks_by_act))
    for act_name, act_chunks in chunks_by_act.items():
        get_or_create_act_summary(act_name, act_chunks)
    logger.info("Bare act summary index: %d acts", len(chunks_by_act))
    return len(chunks_by_act)


def main():
    logger.info("Building bare act summary index (from %s)", BARE_CHUNKS_V2)
    n = build_bare_act_summary_index()
    logger.info("Done: %d acts in summary index.", n)


if __name__ == "__main__":
    main()
