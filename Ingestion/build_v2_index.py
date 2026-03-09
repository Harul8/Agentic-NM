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
    """
    Load the embedding model with explicit mean-pooling support.

    Mirrors hybrid_retriever._get_embedder() so that index-time and query-time
    embeddings use identical pooling — critical for FAISS distance to be valid.
    Native sentence-transformers models load via fast path; HuggingFace-only
    models (e.g. nlpaueb/legal-bert-base-uncased) fall back to explicit
    Transformer + mean-pooling layers.
    """
    import torch
    from sentence_transformers import SentenceTransformer, models
    from config import EMBEDDING_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        _ = embedder.encode("test", convert_to_numpy=True)  # smoke-test
        return embedder
    except Exception:
        logger.info(
            "Native SentenceTransformer load failed; building with explicit "
            "mean-pooling for '%s'", EMBEDDING_MODEL,
        )
        word_embedding_model = models.Transformer(EMBEDDING_MODEL)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        return SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )


def build_index(
    chunks: list,
    faiss_path: str,
    chunks_path: str,
    bm25_path: str,
    embedder,
    batch_size: int = None,
    append: bool = False,
):
    """Build (or extend) FAISS + BM25 indexes from a list of chunks.

    Parameters
    ----------
    chunks      : new chunks to embed and index
    faiss_path  : path to FAISS .index file
    chunks_path : path to chunks JSON store
    bm25_path   : path to BM25 index JSON
    embedder    : SentenceTransformer instance
    batch_size  : embedding batch size (default 32 CPU / 128 GPU)
    append      : if True AND existing index/chunks found, EXTEND them instead
                  of overwriting.  Use this for batched year-by-year indexing.
                  BM25 is always rebuilt from the full accumulated corpus so
                  IDF values remain correct across batches.

    Notes
    -----
    * ``search_text`` is stripped from stored chunks — it is only needed at
      embed time and its presence in the JSON would waste ~35 % of disk/RAM
      on the chunks store.  The retriever uses ``full_text`` for display.
    * FAISS index type is IndexFlatIP (exact cosine similarity after L2-norm).
      Supports incremental .add() without rebuilding the index structure.
    """
    if not chunks:
        logger.warning("No chunks to index!")
        return

    os.makedirs(VECTOR_STORE, exist_ok=True)

    # ── 1. Prepare embed texts (search_text preferred; fall back to full_text) ─
    embed_texts = [
        (c.get("search_text") or c.get("full_text") or c.get("text") or "").strip()
        for c in chunks
    ]

    # Strip search_text from stored chunks — not needed at query time
    stored_chunks = [{k: v for k, v in c.items() if k != "search_text"} for c in chunks]

    # ── 2. Batch-size selection ────────────────────────────────────────────────
    if batch_size is None:
        try:
            import torch
            on_gpu = (
                (getattr(embedder, "device", None) and "cuda" in str(embedder.device))
                or torch.cuda.is_available()
            )
            batch_size = 128 if on_gpu else 32
        except Exception:
            batch_size = 32

    # ── 3. Embed new chunks ────────────────────────────────────────────────────
    logger.info("Embedding %d chunks (batch_size=%d)...", len(embed_texts), batch_size)
    embeddings = embedder.encode(
        embed_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ── 4. FAISS: append or fresh build ───────────────────────────────────────
    if append and os.path.exists(faiss_path) and os.path.exists(chunks_path):
        logger.info("Append mode — loading existing FAISS index: %s", faiss_path)
        index = faiss.read_index(faiss_path)
        offset = index.ntotal          # new chunks start at this position
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, faiss_path)
        logger.info(
            "FAISS index extended: %s  (%d → %d vectors, +%d new)",
            faiss_path, offset, index.ntotal, len(embeddings),
        )

        # Load existing chunk store and extend it
        logger.info("Loading existing chunks store for merge (%s)...", chunks_path)
        with open(chunks_path, encoding="utf-8") as f:
            chunk_store = json.load(f)
        for i, chunk in enumerate(stored_chunks):
            chunk_store[str(offset + i)] = chunk
        logger.info(
            "Chunks store extended: %d → %d total chunks",
            offset, len(chunk_store),
        )

        # BM25: rebuild from the FULL accumulated corpus (correct IDF)
        # Use full_text (search_text was stripped from stored chunks)
        bm25_texts = [c.get("full_text") or c.get("text") or "" for c in chunk_store.values()]

    else:
        if append:
            logger.info(
                "Append mode requested but no existing index found at %s — "
                "building fresh index.", faiss_path,
            )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, faiss_path)
        logger.info("FAISS index saved: %s (%d vectors)", faiss_path, index.ntotal)

        chunk_store = {str(i): chunk for i, chunk in enumerate(stored_chunks)}
        # BM25 texts for fresh build: use embed_texts (search_text) for consistency
        bm25_texts = embed_texts

    # ── 5. Save / overwrite chunks JSON ───────────────────────────────────────
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, ensure_ascii=False)
    logger.info("Chunks JSON saved: %s (%d chunks)", chunks_path, len(chunk_store))

    # ── 6. Build BM25 from the full accumulated corpus ─────────────────────────
    logger.info("Fitting BM25 on %d documents...", len(bm25_texts))
    bm25 = BM25()
    bm25.fit(bm25_texts)
    save_bm25_index(bm25, bm25_path)
    logger.info("BM25 index saved: %s", bm25_path)


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
