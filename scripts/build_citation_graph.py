"""
Build the citation graph from case-law chunks (CASE_CHUNKS_V2).

Run after the v2 index exists. Writes citation_graph.json to the vector store.
Use at retrieval to expand by precedent (case → cites → case, case → interprets → section).

Usage:
    python scripts/build_citation_graph.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from config import CASE_CHUNKS_V2, CITATION_GRAPH_PATH
from retrieval.hybrid_retriever import load_chunks
from retrieval.citation_graph import build_citation_graph_from_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not os.path.isfile(CASE_CHUNKS_V2):
        logger.warning("Case chunks not found: %s. Run rebuild_vector_store.py first.", CASE_CHUNKS_V2)
        return 1
    chunks = load_chunks(CASE_CHUNKS_V2)
    if not chunks:
        logger.warning("No chunks loaded from %s", CASE_CHUNKS_V2)
        return 1
    build_citation_graph_from_chunks(chunks, CITATION_GRAPH_PATH)
    logger.info("Citation graph saved to %s", CITATION_GRAPH_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
