"""
Recreate the entire vector store from Google Drive: BareActs + CaseLaws.

1. Bare act dedup: scans BareActs, groups by act title (from first 2 pages),
   keeps latest by "As on the ..." date per title, removes duplicate files.
2. Removes existing v2 and BM25 index files.
3. Rebuilds all FAISS + BM25 from PDFs in DATA_ROOT/BareActs and DATA_ROOT/CaseLaws.
4. Builds the bare act summary index (act_name -> summary) for all acts in the store.

Uses GPU when available. Run from project root:

    python scripts/rebuild_vector_store.py
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

# Import dedup from same scripts dir (works when run from project root)
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from bare_act_dedup import run_dedup  # noqa: E402
from build_bare_act_summary_index import build_bare_act_summary_index  # noqa: E402

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
    logger.info("REBUILD VECTOR STORE (Dedup BareActs + FAISS + BM25)")
    logger.info("=" * 60)
    logger.info("Vector store:  %s", VECTOR_STORE)
    logger.info("Bare Acts dir: %s", BARE_ACTS_DIR)
    logger.info("Case Laws dir: %s", CASELAW_DIR)

    os.makedirs(VECTOR_STORE, exist_ok=True)

    # 1) Bare act folder dedup: same title => keep latest by "As on" date, remove others
    if os.path.isdir(BARE_ACTS_DIR):
        logger.info("Step 1: Bare act duplicate check and removal...")
        run_dedup(BARE_ACTS_DIR)
    else:
        logger.warning("Bare Acts directory not found: %s (skipping dedup)", BARE_ACTS_DIR)

    # 2) Remove existing v2/BM25 files so we fully recreate
    logger.info("Step 2: Removing existing v2 and BM25 index files...")
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
            # Build bare act summary index (separate module; can be run standalone)
            build_bare_act_summary_index(bare_chunks)
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
