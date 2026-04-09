"""
core/indexer.py — Build and extend FAISS + BM25 indexes from chunks.
"""

import os
import sys
import json
import logging
import time
import shutil
import math

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss

from config import (
    VECTOR_STORE,
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
    CASE_SUMMARY_INDEX_V2,
    CASE_SUMMARY_CHUNKS_V2,
    CASE_SUMMARY_BM25_INDEX,
    ACT_SUMMARY_INDEX_V2,
    ACT_SUMMARY_CHUNKS_V2,
    ACT_SUMMARY_BM25_INDEX,
)
from retrieval.retriever import BM25, save_bm25_index

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
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to build embeddings for index generation, but it "
            "is not installed in the current environment.\n\n"
            "Install the project dependencies first:\n"
            "  pip install -r requirements.txt\n\n"
            "If you are using a custom environment, make sure `torch`, "
            "`torchvision`, and `torchaudio` are installed there before rerunning "
            "the indexing script."
        ) from exc
    from sentence_transformers import SentenceTransformer, models
    from config import EMBEDDING_MODEL
    on_gpu = torch.cuda.is_available()
    device = "cuda" if on_gpu else "cpu"
    if on_gpu:
        torch.backends.cudnn.benchmark = True
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        _ = embedder.encode("test", convert_to_numpy=True)  # smoke-test
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
        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
    if on_gpu:
        embedder = embedder.half()
        logger.info("Embedding model loaded in FP16 on %s", torch.cuda.get_device_name(0))
    return embedder


def build_index(
    chunks: list,
    faiss_path: str,
    chunks_path: str,
    bm25_path: str,
    embedder,
    batch_size: int = None,
    append: bool = False,
    resume_embeddings: bool = False,
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
    on_gpu = False
    cuda_available = False
    embedder_device = str(getattr(embedder, "device", "unknown")).lower()
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        on_gpu = ("cuda" in embedder_device) or cuda_available
    except Exception as e:
        logger.warning("Could not evaluate CUDA availability cleanly: %s", e)

    if batch_size is None:
        try:
            # Tuned for higher throughput; can be overridden via env vars.
            default_gpu_bs = int(os.getenv("INDEX_EMBED_BATCH_GPU", "192"))
            default_cpu_bs = int(os.getenv("INDEX_EMBED_BATCH_CPU", "32"))
            batch_size = default_gpu_bs if on_gpu else default_cpu_bs
        except Exception as e:
            logger.warning(
                "Batch-size env parsing failed (%s); falling back to CPU-safe batch_size=32",
                e,
            )
            batch_size = 32

    logger.info(
        "Embedding backend check: embedder_device=%s, torch_cuda_available=%s, on_gpu=%s",
        embedder_device, cuda_available, on_gpu,
    )

    # ── 3. Embed new chunks ────────────────────────────────────────────────────
    logger.info("Embedding %d chunks (batch_size=%d)...", len(embed_texts), batch_size)
    t_embed_start = time.perf_counter()

    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    if on_gpu:
        # fp16 materially improves embedding throughput on most CUDA GPUs.
        encode_kwargs["precision"] = "float16"

    def _encode_once(texts: list, kwargs: dict):
        while True:
            try:
                return embedder.encode(texts, **kwargs)
            except ValueError as e:
                # sentence-transformers 2.7.0 does not support float16 via the
                # `precision` arg (it is for quantized output modes only).
                if "Precision float16 is not supported" in str(e) and "precision" in kwargs:
                    logger.warning(
                        "encode(precision='float16') unsupported in this "
                        "sentence-transformers version; retrying without precision override."
                    )
                    kwargs.pop("precision", None)
                    continue
                raise
            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()
                cur_bs = int(kwargs["batch_size"])
                if not (is_oom and cur_bs > 16):
                    raise
                new_bs = max(16, cur_bs // 2)
                logger.warning(
                    "Embedding OOM at batch_size=%d; retrying with batch_size=%d",
                    cur_bs, new_bs,
                )
                kwargs["batch_size"] = new_bs
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    if resume_embeddings:
        ckpt_dir = f"{faiss_path}.embed_ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        state_path = os.path.join(ckpt_dir, "state.json")
        # Avoid noisy per-window 25/25 bars; we log cumulative progress ourselves.
        encode_kwargs["show_progress_bar"] = False

        state = {"completed": 0, "batch_size": int(encode_kwargs["batch_size"])}
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
                logger.info(
                    "Resuming embeddings from checkpoint: %d/%d done",
                    int(state.get("completed", 0)), len(embed_texts),
                )
            except Exception:
                logger.warning("Checkpoint state unreadable; starting embedding from scratch.")
                state = {"completed": 0, "batch_size": int(encode_kwargs["batch_size"])}

        start_idx = int(state.get("completed", 0))
        all_parts = []
        if start_idx > 0:
            # Load already embedded shard files in order.
            for i in range(0, start_idx):
                shard = os.path.join(ckpt_dir, f"emb_{i:08d}.npy")
                if os.path.exists(shard):
                    all_parts.append(np.load(shard))
            loaded = sum(len(p) for p in all_parts) if all_parts else 0
            if loaded != start_idx:
                logger.warning("Checkpoint shards incomplete (%d/%d). Restarting embeddings.", loaded, start_idx)
                start_idx = 0
                all_parts = []

        cur = start_idx
        resume_window_batches = int(os.getenv("INDEX_EMBED_RESUME_WINDOW_BATCHES", "25"))
        resume_window_batches = max(1, resume_window_batches)
        batch_for_progress = int(encode_kwargs["batch_size"])
        total_batches = max(1, math.ceil(len(embed_texts) / batch_for_progress))
        logger.info(
            "Resume embedding progress will be reported as cumulative batches (window=%d, total=%d)",
            resume_window_batches, total_batches,
        )
        while cur < len(embed_texts):
            bs = int(encode_kwargs["batch_size"])
            # Process multiple mini-batches per encode() call so progress bars are meaningful.
            end = min(cur + (bs * resume_window_batches), len(embed_texts))
            part = _encode_once(embed_texts[cur:end], encode_kwargs)
            part = np.array(part, dtype="float32")
            np.save(os.path.join(ckpt_dir, f"emb_{cur:08d}.npy"), part)
            all_parts.append(part)
            cur = end
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"completed": cur, "batch_size": int(encode_kwargs["batch_size"])}, f)
            done_batches = min(total_batches, math.ceil(cur / batch_for_progress))
            logger.info(
                "Embedding progress: %d/%d batches (%.1f%%)",
                done_batches, total_batches, (done_batches * 100.0) / total_batches,
            )
        embeddings = np.vstack(all_parts) if all_parts else np.empty((0, 0), dtype="float32")
    else:
        embeddings = _encode_once(embed_texts, encode_kwargs)

    t_embed_end = time.perf_counter()
    logger.info("Embedding complete in %.2fs", t_embed_end - t_embed_start)
    if resume_embeddings:
        try:
            shutil.rmtree(f"{faiss_path}.embed_ckpt", ignore_errors=True)
        except Exception:
            logger.warning("Could not clean embedding checkpoint directory for %s", faiss_path)

    # ── 4. FAISS: append or fresh build ───────────────────────────────────────
    # GPU support: use faiss-gpu if available. GPU IndexFlatIP is used for
    # vector insertion (supports GPU .add()), then converted back to CPU before
    # saving (GPU indexes cannot be written to disk directly).
    # Note: IndexHNSWFlat has no GPU variant — on GPU we use IndexFlatIP which
    # gives exact cosine search and is fast enough at typical bare-act scales.
    _gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    _use_gpu   = _gpu_count > 0
    if _use_gpu:
        logger.info("FAISS GPU detected (%d device(s)) — building on GPU.", _gpu_count)
    else:
        logger.info("No FAISS GPU detected — building on CPU.")

    def _make_cpu_index(dim: int) -> faiss.Index:
        """Create a fresh CPU index. IndexFlatIP on GPU path, HNSW on CPU path."""
        if _use_gpu:
            # Exact inner-product search; GPU-compatible
            return faiss.IndexFlatIP(dim)
        else:
            idx = faiss.IndexHNSWFlat(dim, 32)
            idx.hnsw.efConstruction = 200
            return idx

    def _add_to_index(index: faiss.Index, vecs: np.ndarray) -> faiss.Index:
        """Add vectors, offloading to GPU when available."""
        if _use_gpu:
            _ = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            gpu_index.add(vecs)
            return faiss.index_gpu_to_cpu(gpu_index)
        else:
            index.add(vecs)
            return index

    vecs = np.array(embeddings, dtype="float32")

    t_faiss_start = time.perf_counter()
    if append and os.path.exists(faiss_path) and os.path.exists(chunks_path):
        logger.info("Append mode — loading existing FAISS index: %s", faiss_path)
        index = faiss.read_index(faiss_path)
        offset = index.ntotal          # new chunks start at this position
        index = _add_to_index(index, vecs)
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
        index = _make_cpu_index(dim)
        index = _add_to_index(index, vecs)
        faiss.write_index(index, faiss_path)
        logger.info(
            "FAISS index saved (%s): %s (%d vectors)",
            "GPU→FlatIP" if _use_gpu else "CPU→HNSW",
            faiss_path, index.ntotal,
        )

        chunk_store = {str(i): chunk for i, chunk in enumerate(stored_chunks)}
        # BM25 texts for fresh build: use full_text (raw paragraph text, not metadata-polluted search_text)
        bm25_texts = [c.get("full_text") or c.get("text") or "" for c in stored_chunks]
    t_faiss_end = time.perf_counter()
    logger.info("FAISS stage complete in %.2fs", t_faiss_end - t_faiss_start)

    # ── 5. Save / overwrite chunks JSON ───────────────────────────────────────
    t_chunks_start = time.perf_counter()
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, ensure_ascii=False)
    logger.info("Chunks JSON saved: %s (%d chunks)", chunks_path, len(chunk_store))
    t_chunks_end = time.perf_counter()
    logger.info("Chunk JSON write complete in %.2fs", t_chunks_end - t_chunks_start)

    # ── 6. Build BM25 from the full accumulated corpus ─────────────────────────
    t_bm25_start = time.perf_counter()
    logger.info("Fitting BM25 on %d documents...", len(bm25_texts))
    bm25 = BM25()
    bm25.fit(bm25_texts)
    save_bm25_index(bm25, bm25_path)
    logger.info("BM25 index saved: %s", bm25_path)
    t_bm25_end = time.perf_counter()
    logger.info("BM25 stage complete in %.2fs", t_bm25_end - t_bm25_start)


def main():
    logger.info("=" * 60)
    logger.info("NYAYMALAW V2 INDEX BUILDER")
    logger.info("=" * 60)
    logger.info(f"Vector Store: {VECTOR_STORE}")

    # Import pipeline chunk loaders here (deferred to avoid circular imports at module level).
    # These read from the pipeline's JSON output (json_output/caselaws/, json_output/BareActs/)
    # and produce chunks that are already enriched with paragraph_type, sections_cited,
    # cited_cases, and have header chunks filtered out.  This is the correct source for indexing.
    from legal_database.pipeline import _json_to_case_chunks, _json_to_statute_chunks

    embedder = _get_embedder()

    # --- Bare Acts (sections + act summaries) ---
    logger.info("\n--- BARE ACTS (Section-Level Chunking from JSON output) ---")
    bare_chunks, act_summary_chunks = _json_to_statute_chunks()
    if bare_chunks:
        build_index(bare_chunks, BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX, embedder)
        logger.info("Bare acts: %d section-level chunks indexed", len(bare_chunks))
        acts = set(c.get("act_name", "") for c in bare_chunks if c.get("act_name"))
        logger.info("Acts covered: %d", len(acts))
        for act in sorted(acts):
            count = sum(1 for c in bare_chunks if c.get("act_name") == act)
            logger.info("  - %s: %d sections", act, count)
    else:
        logger.warning(
            "No bare act chunks produced. Run the bare acts pipeline first to generate "
            "JSON output in json_output/BareActs/."
        )

    if act_summary_chunks:
        build_index(
            act_summary_chunks,
            ACT_SUMMARY_INDEX_V2, ACT_SUMMARY_CHUNKS_V2, ACT_SUMMARY_BM25_INDEX,
            embedder,
        )
        logger.info("Act summaries: %d act-level chunks indexed", len(act_summary_chunks))
    else:
        logger.warning("No act summary chunks produced.")

    # --- Case Laws (paragraphs + case summaries) ---
    logger.info("\n--- CASE LAWS (Paragraph-Level Chunking from JSON output) ---")
    case_chunks, case_summary_chunks = _json_to_case_chunks()
    if case_chunks:
        build_index(case_chunks, CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX, embedder)
        logger.info("Case laws: %d paragraph-level chunks indexed", len(case_chunks))

        # Paragraph type distribution
        ptypes: dict = {}
        for c in case_chunks:
            pt = c.get("paragraph_type", "unknown")
            ptypes[pt] = ptypes.get(pt, 0) + 1
        logger.info("Paragraph type distribution: %s", sorted(ptypes.items(), key=lambda x: -x[1]))

        cases = set(c.get("case_name", "") for c in case_chunks if c.get("case_name"))
        logger.info("Cases covered: %d", len(cases))
        for case in sorted(cases)[:20]:
            count = sum(1 for c in case_chunks if c.get("case_name") == case)
            logger.info("  - %s: %d paragraphs", case, count)
    else:
        logger.warning(
            "No case law chunks produced. Run the case law pipeline first to generate "
            "JSON output in json_output/caselaws/."
        )

    if case_summary_chunks:
        build_index(
            case_summary_chunks,
            CASE_SUMMARY_INDEX_V2, CASE_SUMMARY_CHUNKS_V2, CASE_SUMMARY_BM25_INDEX,
            embedder,
        )
        logger.info("Case summaries: %d case-level chunks indexed", len(case_summary_chunks))
    else:
        logger.warning("No case summary chunks produced.")

    logger.info("\n" + "=" * 60)
    logger.info("V2 INDEX BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
