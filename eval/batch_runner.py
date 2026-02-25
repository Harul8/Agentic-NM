"""
Batch Runner — Runs all evaluation queries through the Nyaymalaw pipeline and
logs every intermediate result (FAISS scores, BM25 scores, rerank scores,
sufficiency analysis, internet fallback, final response) as structured JSON.

Usage:
    python -m eval.batch_runner --queries eval/data/queries.json --out eval/results/
    python -m eval.batch_runner --queries eval/data/queries.json --out eval/results/ --mode faiss_only
    python -m eval.batch_runner --queries eval/data/queries.json --out eval/results/ --mode full_pipeline

Ablation modes:
    full_pipeline    — FAISS + BM25 + cross-encoder + quality filters + internet fallback (default)
    faiss_only       — FAISS vector search only, no BM25, no reranking
    bm25_only        — BM25 keyword search only, no FAISS
    merged_no_rerank — FAISS + BM25 merged, no cross-encoder re-ranking
    no_internet      — Full local pipeline but internet fallback disabled
"""

import argparse
import json
import logging
import os
import sys
import time
import signal
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Timeout helper — prevents a single hung LLM call from stalling the batch
# ---------------------------------------------------------------------------

LLM_TIMEOUT_SECONDS = 120  # 2 minutes max per LLM call


@contextmanager
def time_limit(seconds: int, label: str = "operation"):
    """Context manager that raises TimeoutError if block exceeds `seconds`."""
    def _handler(signum, frame):
        raise TimeoutError(f"{label} timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX
from config import CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX

logger = logging.getLogger("eval.batch_runner")


# ---------------------------------------------------------------------------
# Pre-flight validation — fail fast before wasting time on 100 queries
# ---------------------------------------------------------------------------

def validate_data_files(mode: str) -> bool:
    """
    Check that all required index files exist before running any queries.
    Prints a clear report and returns False if anything is missing.

    This prevents the silent failure where all 100 queries return empty results
    because the data root path is wrong or the index hasn't been built yet.
    """
    from config import (
        BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX,
        CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX,
    )

    # Which files are needed per mode
    always_needed = [
        ("Bare Acts FAISS index",   BARE_INDEX_V2),
        ("Bare Acts chunk store",   BARE_CHUNKS_V2),
        ("Case Laws FAISS index",   CASE_INDEX_V2),
        ("Case Laws chunk store",   CASE_CHUNKS_V2),
    ]
    bm25_needed = [
        ("Bare Acts BM25 index",    BARE_BM25_INDEX),
        ("Case Laws BM25 index",    CASE_BM25_INDEX),
    ]

    checks = list(always_needed)
    if mode not in ("faiss_only",):
        checks += bm25_needed

    missing = []
    print("\n=== PRE-FLIGHT DATA FILE CHECK ===")
    for label, path in checks:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0
        status = f"OK   ({size_kb:,.0f} KB)" if exists else "MISSING"
        print(f"  [{status:>20}]  {label}")
        print(f"                           {path}")
        if not exists:
            missing.append((label, path))

    if missing:
        print(f"\n  *** PREFLIGHT FAILED: {len(missing)} required file(s) are missing ***")
        print("  Fix: Run the ingestion pipeline to build the v2 index before evaluating.")
        print("  Hint: Check that NYAYMALAW_DATA_ROOT in .env points to your data folder.\n")
        return False

    print(f"\n  All {len(checks)} required files found. Proceeding with mode='{mode}'.\n")
    return True


# ---------------------------------------------------------------------------
# Ablation-aware retrieval
# ---------------------------------------------------------------------------

def _run_retrieval(query: str, mode: str, top_k: int = 30) -> dict:
    """
    Run retrieval in the specified ablation mode.
    Returns raw results with scores for analysis.
    """
    from retrieval.hybrid_retriever import (
        hybrid_search, search_bare_acts_auto, search_case_laws_auto,
        safe_read_faiss, load_chunks, load_bm25_index, _get_embedder, _get_cross_encoder,
    )
    import numpy as np

    results = {"bare_acts_raw": [], "case_laws_raw": [], "mode": mode}

    if mode == "full_pipeline":
        results["bare_acts_raw"] = search_bare_acts_auto(query, top_k)
        results["case_laws_raw"] = search_case_laws_auto(query, top_k)

    elif mode == "faiss_only":
        # FAISS only — no BM25, no cross-encoder
        for label, idx_path, chunks_path in [
            ("bare_acts", BARE_INDEX_V2, BARE_CHUNKS_V2),
            ("case_laws", CASE_INDEX_V2, CASE_CHUNKS_V2),
        ]:
            idx, ok = safe_read_faiss(idx_path)
            chunks = load_chunks(chunks_path)
            if not ok or not idx or not chunks:
                continue
            embedder = _get_embedder()
            qvec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            k = min(top_k, idx.ntotal)
            if k <= 0:
                continue
            distances, indices = idx.search(np.array([qvec], dtype="float32"), k)
            for rank, i in enumerate(indices[0]):
                if i >= 0 and str(i) in chunks:
                    chunk = dict(chunks[str(i)])
                    chunk["_rerank_score"] = float(distances[0][rank])
                    chunk["source_tag"] = "LOCAL_DB"
                    results[f"{label}_raw"].append(chunk)

    elif mode == "bm25_only":
        for label, bm25_path, chunks_path in [
            ("bare_acts", BARE_BM25_INDEX, BARE_CHUNKS_V2),
            ("case_laws", CASE_BM25_INDEX, CASE_CHUNKS_V2),
        ]:
            bm25 = load_bm25_index(bm25_path)
            chunks = load_chunks(chunks_path)
            if not bm25 or not chunks:
                continue
            bm25_results = bm25.score(query, top_k=top_k)
            for doc_idx, score in bm25_results:
                key = str(doc_idx)
                if key in chunks:
                    chunk = dict(chunks[key])
                    chunk["_rerank_score"] = score
                    chunk["source_tag"] = "LOCAL_DB"
                    results[f"{label}_raw"].append(chunk)

    elif mode == "merged_no_rerank":
        # Merge FAISS + BM25 but skip cross-encoder
        for label, idx_path, chunks_path, bm25_path in [
            ("bare_acts", BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX),
            ("case_laws", CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX),
        ]:
            idx, ok = safe_read_faiss(idx_path)
            chunks = load_chunks(chunks_path)
            if not ok or not idx or not chunks:
                continue
            # FAISS
            embedder = _get_embedder()
            qvec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            k = min(top_k, idx.ntotal)
            candidates = set()
            if k > 0:
                distances, indices = idx.search(np.array([qvec], dtype="float32"), k)
                for rank, i in enumerate(indices[0]):
                    if i >= 0 and str(i) in chunks:
                        candidates.add(str(i))
            # BM25
            bm25 = load_bm25_index(bm25_path)
            if bm25:
                for doc_idx, score in bm25.score(query, top_k=top_k):
                    if str(doc_idx) in chunks:
                        candidates.add(str(doc_idx))
            # Assign uniform score (no re-ranking)
            for key in candidates:
                chunk = dict(chunks[key])
                chunk["_rerank_score"] = 0.5  # uniform
                chunk["source_tag"] = "LOCAL_DB"
                results[f"{label}_raw"].append(chunk)

    elif mode == "no_internet":
        results["bare_acts_raw"] = search_bare_acts_auto(query, top_k)
        results["case_laws_raw"] = search_case_laws_auto(query, top_k)
        results["internet_disabled"] = True

    return results


def _run_quality_filters(raw_results: dict) -> dict:
    """Apply the same quality filters as response_generator_v2."""
    from services.response_generator_v2 import (
        MIN_RERANK_SCORE, _is_quality_bare_act, _is_quality_case_law, _case_year_for_sort, MIN_CASE_YEAR,
    )

    bare = raw_results.get("bare_acts_raw", [])
    cases = raw_results.get("case_laws_raw", [])

    # Score filter
    bare_scored = [b for b in bare if b.get("_rerank_score", 0) >= MIN_RERANK_SCORE]
    cases_scored = [c for c in cases if c.get("_rerank_score", 0) >= MIN_RERANK_SCORE]

    # Quality filter
    bare_quality = [b for b in bare_scored if _is_quality_bare_act(b)]
    cases_quality = [c for c in cases_scored if _is_quality_case_law(c)]

    return {
        "bare_acts_after_score_filter": len(bare_scored),
        "case_laws_after_score_filter": len(cases_scored),
        "bare_acts_after_quality_filter": len(bare_quality),
        "case_laws_after_quality_filter": len(cases_quality),
        "bare_acts_filtered": bare_quality,
        "case_laws_filtered": cases_quality,
    }


def _run_full_pipeline(query: str, intent: str, result_count: Optional[int], mode: str) -> dict:
    """Run the complete pipeline (or ablation variant) and capture all intermediate data."""
    from services.response_generator_v2 import generate_response_v2, expand_legal_query

    start_time = time.time()

    # Step 1: Query expansion
    legal_query = expand_legal_query(query)

    # Step 2: Retrieval (ablation-aware)
    search_query = f"{query} {legal_query}"[:500]
    raw_results = _run_retrieval(search_query, mode)

    # Step 3: Quality filtering
    filter_results = _run_quality_filters(raw_results)

    # Step 4: Full pipeline for final response (only in full_pipeline / no_internet modes)
    final_response = None
    if mode in ("full_pipeline", "no_internet"):
        try:
            with time_limit(LLM_TIMEOUT_SECONDS, label="generate_response_v2"):
                final_response = generate_response_v2(
                    query,
                    jurisdiction_state="",
                    intent=intent,
                    result_count=result_count,
                )
        except TimeoutError as e:
            logger.error(f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s: {e}")
            final_response = {"error": f"timeout: {e}"}
        except Exception as e:
            logger.error(f"Pipeline failed for query: {e}")
            final_response = {"error": str(e)}

    elapsed = time.time() - start_time

    return {
        "legal_query_expanded": legal_query,
        "search_query": search_query,
        "mode": mode,
        "raw_bare_acts_count": len(raw_results.get("bare_acts_raw", [])),
        "raw_case_laws_count": len(raw_results.get("case_laws_raw", [])),
        "filter_results": {
            k: v for k, v in filter_results.items()
            if not k.endswith("_filtered")  # Don't store full chunks in summary
        },
        "raw_scores": {
            "bare_act_rerank_scores": [
                {"score": b.get("_rerank_score", 0), "act_name": b.get("act_name", ""), "section": b.get("section_number", "")}
                for b in raw_results.get("bare_acts_raw", [])[:20]
            ],
            "case_law_rerank_scores": [
                {"score": c.get("_rerank_score", 0), "case_name": c.get("case_name", ""), "year": c.get("year", "")}
                for c in raw_results.get("case_laws_raw", [])[:20]
            ],
        },
        "final_response": _sanitize_response(final_response) if final_response else None,
        "elapsed_seconds": round(elapsed, 2),
    }


def _sanitize_response(resp: dict) -> dict:
    """Trim large text fields for JSON storage."""
    if not resp or "error" in resp:
        return resp
    sanitized = {}
    for key in ("bare_act_sections", "case_laws", "explanation", "sufficiency", "sources_used"):
        val = resp.get(key)
        if isinstance(val, str):
            sanitized[key] = val[:3000]
        elif isinstance(val, list):
            sanitized[key] = [
                {k: (v[:500] if isinstance(v, str) else v) for k, v in item.items()}
                if isinstance(item, dict) else item
                for item in val[:20]
            ]
        elif isinstance(val, dict):
            sanitized[key] = val
        else:
            sanitized[key] = val
    return sanitized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_batch(queries_path: str, output_dir: str, mode: str = "full_pipeline"):
    """Run all queries and save results."""
    # Pre-flight: abort immediately if required data files are missing
    if not validate_data_files(mode):
        sys.exit(1)

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"batch_{mode}_{timestamp}.json")

    all_results = []
    total = len(queries)

    for i, q in enumerate(queries):
        query_text = q.get("query", "")
        intent = q.get("intent", "legal_opinion")
        result_count = q.get("result_count")
        query_id = q.get("id", f"q_{i+1}")

        logger.info(f"[{i+1}/{total}] Running query: {query_text[:80]}...")

        try:
            result = _run_full_pipeline(query_text, intent, result_count, mode)
        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}")
            result = {"error": str(e)}

        result["query_id"] = query_id
        result["query_text"] = query_text
        result["intent"] = intent
        result["gold_annotations"] = q.get("gold", {})
        all_results.append(result)

        # Save incrementally
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Batch complete. {len(all_results)} results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Batch evaluation runner for Nyaymalaw 3.0")
    parser.add_argument("--queries", required=True, help="Path to queries JSON file")
    parser.add_argument("--out", default="eval/results/", help="Output directory")
    parser.add_argument("--mode", default="full_pipeline",
                        choices=["full_pipeline", "faiss_only", "bm25_only", "merged_no_rerank", "no_internet"],
                        help="Ablation mode")
    args = parser.parse_args()

    run_batch(args.queries, args.out, args.mode)
