"""
Rebuild the bare acts vector store from all PDFs in Google Drive (BareActs folder).

Uses GPU when available to speed up embedding. Run from project root:

    python scripts/rebuild_bare_acts_index.py

Or with GPU explicitly preferred (skip CPU fallback):

    set CUDA_VISIBLE_DEVICES=0
    python scripts/rebuild_bare_acts_index.py

Requires: NYAYMALAW_DATA_ROOT (or default data/) pointing at your Google Drive folder
          so that BareActs/ is at DATA_ROOT/BareActs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from config import BARE_ACTS_DIR, VECTOR_STORE, BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX
from Ingestion.build_v2_index import build_index, _get_embedder
from Ingestion.smart_chunker import process_bare_acts_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("REBUILD BARE ACTS VECTOR STORE (from Google Drive BareActs)")
    logger.info("=" * 60)
    logger.info("Bare Acts dir: %s", BARE_ACTS_DIR)
    logger.info("Vector store:  %s", VECTOR_STORE)

    if not os.path.isdir(BARE_ACTS_DIR):
        logger.error("Bare Acts directory not found: %s", BARE_ACTS_DIR)
        logger.info("Set NYAYMALAW_DATA_ROOT to your Google Drive path (e.g. G:\\My Drive\\Nyaymalaw)")
        sys.exit(1)

    # Load embedder (uses GPU if available)
    embedder = _get_embedder()
    device = getattr(embedder, "device", None)
    on_gpu = device is not None and "cuda" in str(device)
    if on_gpu:
        logger.info("Using GPU for embeddings (faster). Batch size will be 128.")
    else:
        logger.info("GPU not available; using CPU. Batch size 32.")

    # Chunk all bare act PDFs
    logger.info("Chunking all PDFs in BareActs...")
    bare_chunks = process_bare_acts_directory(BARE_ACTS_DIR)

    if not bare_chunks:
        logger.warning("No bare act chunks produced. Check that BareActs/ contains PDFs.")
        sys.exit(1)

    logger.info("Total chunks: %d", len(bare_chunks))
    acts = set(c.get("act_name", "") for c in bare_chunks if c.get("act_name"))
    logger.info("Acts covered: %d", len(acts))

    # Build index (batch_size 128 on GPU, 32 on CPU)
    batch_size = 128 if on_gpu else 32
    build_index(
        bare_chunks,
        BARE_INDEX_V2,
        BARE_CHUNKS_V2,
        BARE_BM25_INDEX,
        embedder,
        batch_size=batch_size,
    )

    logger.info("=" * 60)
    logger.info("Bare acts vector store rebuild complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
