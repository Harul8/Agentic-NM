"""
Recreate the entire vector store from Google Drive: BareActs + CaseLaws.

Rebuilds all FAISS indexes and BM25 indexes (v2 + BM25) from PDFs in:
  - DATA_ROOT/BareActs
  - DATA_ROOT/CaseLaws

Uses GPU when available to speed up embedding. Run from project root:

    python scripts/rebuild_vector_store.py

Before rebuilding, removes existing v2 and BM25 files so the folder is fully recreated.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import (
    VECTOR_STORE,
    BARE_ACTS_DIR,
    CASELAW_DIR,
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
)
from Ingestion.build_v2_index import build_index, _get_embedder
from Ingestion.smart_chunker import process_bare_acts_directory, process_case_laws_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All v2 and BM25 files we will recreate
VECTOR_STORE_FILES_TO_REMOVE = [
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
]


def _remove_existing_index_files():
    """Remove existing FAISS and BM25 index files so we recreate from scratch."""
    for path in VECTOR_STORE_FILES_TO_REMOVE:
        if os.path.isfile(path):
            try:
                os.remove(path)
                logger.info("Removed %s", path)
            except OSError as e:
                logger.warning("Could not remove %s: %s", path, e)


def main():
    logger.info("=" * 60)
    logger.info("REBUILD VECTOR STORE (FAISS + BM25 for Bare Acts and Case Laws)")
    logger.info("=" * 60)
    logger.info("Vector store:  %s", VECTOR_STORE)
    logger.info("Bare Acts dir: %s", BARE_ACTS_DIR)
    logger.info("Case Laws dir: %s", CASELAW_DIR)

    os.makedirs(VECTOR_STORE, exist_ok=True)

    # Remove existing v2/BM25 files so we fully recreate
    logger.info("Removing existing v2 and BM25 index files...")
    _remove_existing_index_files()

    # Load embedder once (GPU if available)
    embedder = _get_embedder()
    device = getattr(embedder, "device", None)
    on_gpu = device is not None and "cuda" in str(device)
    batch_size = 128 if on_gpu else 32
    if on_gpu:
        logger.info("Using GPU for embeddings (batch_size=%d)", batch_size)
    else:
        logger.info("Using CPU for embeddings (batch_size=%d)", batch_size)

    # --- Bare Acts ---
    logger.info("\n--- BARE ACTS (FAISS + BM25) ---")
    if os.path.isdir(BARE_ACTS_DIR):
        bare_chunks = process_bare_acts_directory(BARE_ACTS_DIR)
        if bare_chunks:
            build_index(
                bare_chunks,
                BARE_INDEX_V2,
                BARE_CHUNKS_V2,
                BARE_BM25_INDEX,
                embedder,
                batch_size=batch_size,
            )
            logger.info("Bare acts: %d chunks -> FAISS + chunks JSON + BM25", len(bare_chunks))
            acts = {c.get("act_name", "") for c in bare_chunks if c.get("act_name")}
            logger.info("Acts covered: %d", len(acts))
        else:
            logger.warning("No bare act chunks produced. Check PDFs in BareActs/")
    else:
        logger.warning("Bare Acts directory not found: %s", BARE_ACTS_DIR)

    # --- Case Laws ---
    logger.info("\n--- CASE LAWS (FAISS + BM25) ---")
    if os.path.isdir(CASELAW_DIR):
        case_chunks = process_case_laws_directory(CASELAW_DIR)
        if case_chunks:
            build_index(
                case_chunks,
                CASE_INDEX_V2,
                CASE_CHUNKS_V2,
                CASE_BM25_INDEX,
                embedder,
                batch_size=batch_size,
            )
            logger.info("Case laws: %d chunks -> FAISS + chunks JSON + BM25", len(case_chunks))
            cases = {c.get("case_name", "") for c in case_chunks if c.get("case_name")}
            logger.info("Cases covered: %d", len(cases))
        else:
            logger.warning("No case law chunks produced. Check PDFs in CaseLaws/")
    else:
        logger.warning("Case Laws directory not found: %s", CASELAW_DIR)

    logger.info("\n" + "=" * 60)
    logger.info("Vector store rebuild complete. FAISS + BM25 recreated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
