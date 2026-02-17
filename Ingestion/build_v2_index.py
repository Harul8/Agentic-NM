"""
Build V2 Index — Rebuild FAISS + BM25 indexes using smart section-level chunking.

Run this script to re-index all bare acts and case laws in Google Drive
with the new v2 chunking strategy (section-level for bare acts, paragraph-level
for case laws) plus BM25 keyword indexes for hybrid search.

Usage:
    python Ingestion/build_v2_index.py

This replaces the old blind 800-char chunking with legally-aware splitting.
The old v1 indexes are preserved — the system auto-detects and prefers v2 when available.
"""

import os
import sys
import json
import logging

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss

from config import (
    BARE_ACTS_DIR,
    CASELAW_DIR,
    VECTOR_STORE,
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
)
from Ingestion.smart_chunker import (
    process_bare_acts_directory,
    process_case_laws_directory,
)
from retrieval.hybrid_retriever import BM25, save_bm25_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get_embedder():
    """Load the embedding model."""
    import torch
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def build_index(chunks: list, faiss_path: str, chunks_path: str, bm25_path: str, embedder):
    """Build FAISS + BM25 indexes from a list of chunks."""
    if not chunks:
        logger.warning("No chunks to index!")
        return

    os.makedirs(VECTOR_STORE, exist_ok=True)

    # Prepare texts for embedding
    texts = []
    for chunk in chunks:
        text = (
            chunk.get("search_text")
            or chunk.get("full_text")
            or chunk.get("text")
            or ""
        ).strip()
        texts.append(text)

    # Embed all chunks
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index saved: {faiss_path} ({index.ntotal} vectors)")

    # Save chunks JSON
    chunk_store = {}
    for i, chunk in enumerate(chunks):
        chunk_store[str(i)] = chunk

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, indent=2, ensure_ascii=False)
    logger.info(f"Chunks JSON saved: {chunks_path}")

    # Build BM25 index
    bm25 = BM25()
    bm25.fit(texts)
    save_bm25_index(bm25, bm25_path)
    logger.info(f"BM25 index saved: {bm25_path}")


def main():
    logger.info("=" * 60)
    logger.info("NYAYMALAW V2 INDEX BUILDER")
    logger.info("=" * 60)
    logger.info(f"Bare Acts dir: {BARE_ACTS_DIR}")
    logger.info(f"Case Laws dir: {CASELAW_DIR}")
    logger.info(f"Vector Store:  {VECTOR_STORE}")

    embedder = _get_embedder()

    # --- Bare Acts ---
    logger.info("\n--- BARE ACTS (Section-Level Chunking) ---")
    if os.path.isdir(BARE_ACTS_DIR):
        bare_chunks = process_bare_acts_directory(BARE_ACTS_DIR)
        if bare_chunks:
            build_index(bare_chunks, BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX, embedder)
            logger.info(f"Bare acts: {len(bare_chunks)} section-level chunks indexed")

            # Stats
            acts = set(c.get("act_name", "") for c in bare_chunks if c.get("act_name"))
            logger.info(f"Acts covered: {len(acts)}")
            for act in sorted(acts):
                count = sum(1 for c in bare_chunks if c.get("act_name") == act)
                logger.info(f"  - {act}: {count} sections")
        else:
            logger.warning("No bare act chunks produced. Check PDF files in BareActs/")
    else:
        logger.warning(f"Bare Acts directory not found: {BARE_ACTS_DIR}")

    # --- Case Laws ---
    logger.info("\n--- CASE LAWS (Paragraph-Level Chunking) ---")
    if os.path.isdir(CASELAW_DIR):
        case_chunks = process_case_laws_directory(CASELAW_DIR)
        if case_chunks:
            build_index(case_chunks, CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX, embedder)
            logger.info(f"Case laws: {len(case_chunks)} paragraph-level chunks indexed")

            # Stats
            cases = set(c.get("case_name", "") for c in case_chunks if c.get("case_name"))
            logger.info(f"Cases covered: {len(cases)}")
            for case in sorted(cases)[:20]:
                count = sum(1 for c in case_chunks if c.get("case_name") == case)
                logger.info(f"  - {case}: {count} paragraphs")
        else:
            logger.warning("No case law chunks produced. Check files in CaseLaws/")
    else:
        logger.warning(f"Case Laws directory not found: {CASELAW_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("V2 INDEX BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
