"""
Recreate the entire vector store from Google Drive: BareActs + CaseLaws.

1. Bare act dedup: scans BareActs, groups by act title (from first 2 pages),
   keeps latest by "As on the ..." date per title, removes duplicate files.
2. Removes existing v2 and BM25 index files.
3. Rebuilds all FAISS + BM25 from PDFs in DATA_ROOT/BareActs and DATA_ROOT/CaseLaws.
4. Builds the bare act summary index (act_name -> summary) for all acts in the store.
5. Builds the citation graph (case → interprets → section, case → cites → case) from
   case-law chunks so retrieval can expand by precedent.

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
    CITATION_GRAPH_PATH,
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

# All v2 and BM25 files we will recreate (citation graph depends on case chunks, so remove it too)
VECTOR_STORE_FILES_TO_REMOVE = [
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
    CITATION_GRAPH_PATH,
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


def _log_bare_act_stats(bare_chunks: list) -> None:
    """Log per-act chunk counts, avg chunk size, and quality warnings."""
    from collections import defaultdict

    act_chunks: dict = defaultdict(list)
    for c in bare_chunks:
        act_chunks[c.get("act_name", "Unknown")].append(c)

    logger.info("%-60s  %6s  %8s  %s", "Act", "Chunks", "Avg len", "Warnings")
    logger.info("-" * 90)

    total_warnings = 0
    for act_name in sorted(act_chunks):
        chunks = act_chunks[act_name]
        lengths = [len(c.get("full_text", "")) for c in chunks]
        avg_len = int(sum(lengths) / len(lengths)) if lengths else 0

        # Quality checks
        warnings = []
        very_short = sum(1 for l in lengths if l < 100)
        very_long  = sum(1 for l in lengths if l > 5000)
        no_secnum  = sum(1 for c in chunks if not c.get("section_number"))
        if very_short:
            warnings.append(f"{very_short} chunks <100 chars")
        if very_long:
            warnings.append(f"{very_long} chunks >5000 chars")
        if no_secnum > len(chunks) * 0.5:
            warnings.append("mostly fallback chunks (no section numbers)")

        warn_str = "; ".join(warnings) if warnings else "ok"
        total_warnings += len(warnings)

        logger.info("%-60s  %6d  %8d  %s", act_name[:60], len(chunks), avg_len, warn_str)

    logger.info("-" * 90)
    logger.info(
        "Total: %d chunks across %d acts | Quality warnings: %d",
        len(bare_chunks), len(act_chunks), total_warnings,
    )


def _log_case_law_stats(case_chunks: list) -> None:
    """Log per-case chunk counts."""
    from collections import defaultdict

    case_map: dict = defaultdict(int)
    for c in case_chunks:
        case_map[c.get("case_name", "Unknown")] += 1

    logger.info("Top 20 cases by chunk count:")
    for case_name, cnt in sorted(case_map.items(), key=lambda x: -x[1])[:20]:
        logger.info("  %-70s  %d chunks", case_name[:70], cnt)
    if len(case_map) > 20:
        logger.info("  ... and %d more cases", len(case_map) - 20)


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
        logger.info("Step 3: Chunking bare acts (section-level)...")
        bare_chunks = process_bare_acts_directory(BARE_ACTS_DIR)
        if bare_chunks:
            logger.info("\nStep 3 complete — chunk quality report:")
            _log_bare_act_stats(bare_chunks)

            logger.info("\nStep 4: Building FAISS + BM25 index for bare acts...")
            build_index(
                bare_chunks,
                BARE_INDEX_V2,
                BARE_CHUNKS_V2,
                BARE_BM25_INDEX,
                embedder,
                batch_size=batch_size,
            )
            logger.info(
                "Bare acts indexed: %d chunks from %d acts -> FAISS + chunks JSON + BM25",
                len(bare_chunks),
                len({c.get("act_name", "") for c in bare_chunks if c.get("act_name")}),
            )
            # Build bare act summary index (separate module; can be run standalone)
            logger.info("Step 5: Building bare act summary index...")
            build_bare_act_summary_index(bare_chunks)
        else:
            logger.warning("No bare act chunks produced. Check PDFs in BareActs/")
    else:
        logger.warning("Bare Acts directory not found: %s", BARE_ACTS_DIR)

    # --- Case Laws ---
    logger.info("\n--- CASE LAWS (FAISS + BM25) ---")
    if os.path.isdir(CASELAW_DIR):
        logger.info("Step 6: Chunking case laws (paragraph-level)...")
        case_chunks = process_case_laws_directory(CASELAW_DIR)
        if case_chunks:
            logger.info("\nStep 6 complete — case law stats:")
            _log_case_law_stats(case_chunks)

            logger.info("\nStep 7: Building FAISS + BM25 index for case laws...")
            build_index(
                case_chunks,
                CASE_INDEX_V2,
                CASE_CHUNKS_V2,
                CASE_BM25_INDEX,
                embedder,
                batch_size=batch_size,
            )
            cases = {c.get("case_name", "") for c in case_chunks if c.get("case_name")}
            logger.info(
                "Case laws indexed: %d chunks from %d cases -> FAISS + chunks JSON + BM25",
                len(case_chunks), len(cases),
            )
            # Citation graph: case → interprets → section, case → cites → case
            logger.info("Step 7b: Building citation graph from case chunks...")
            try:
                from retrieval.citation_graph import build_citation_graph_from_chunks
                from retrieval.hybrid_retriever import load_chunks
                chunks_dict = load_chunks(CASE_CHUNKS_V2)
                if chunks_dict:
                    build_citation_graph_from_chunks(chunks_dict, CITATION_GRAPH_PATH)
                    logger.info("Citation graph saved to %s", CITATION_GRAPH_PATH)
                else:
                    logger.warning("No chunks loaded for citation graph (skipped).")
            except Exception as e:
                logger.warning("Citation graph build failed (non-fatal): %s", e)
        else:
            logger.warning("No case law chunks produced. Check PDFs in CaseLaws/")
    else:
        logger.warning("Case Laws directory not found: %s", CASELAW_DIR)

    # --- Act Profile Index ---
    # Must run AFTER bare act FAISS + BM25 are written so BARE_CHUNKS_V2 is current.
    logger.info("\n--- ACT PROFILE INDEX ---")
    logger.info("Step 8: Building act-level profile index (for act-first search)...")
    try:
        from retrieval.act_profile_index import build_and_save_act_profiles
        ok = build_and_save_act_profiles()
        if ok:
            logger.info("Step 8 complete — act profile index saved.")
        else:
            logger.warning("Step 8: act profile index build returned False. "
                           "Act-first search will fall back to unfiltered at runtime.")
    except Exception as e:
        logger.error("Step 8: act profile index build failed: %s", e)
        logger.warning("Act-first search will fall back to unfiltered at runtime (non-fatal).")

    # Statute concept index: built from BARE_CHUNKS_V2 on first lookup. Invalidate cache
    # so the next lookup (this process or server) uses the updated store.
    try:
        from retrieval.statute_concept_index import invalidate_store_index
        invalidate_store_index()
        logger.info("Step 9: Statute concept index cache invalidated (will rebuild from store on next lookup).")
    except Exception as e:
        logger.debug("Statute concept index invalidate (non-fatal): %s", e)

    logger.info("\n" + "=" * 60)
    logger.info("Vector store rebuild complete. FAISS + BM25 + Act Profiles + Citation graph recreated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
