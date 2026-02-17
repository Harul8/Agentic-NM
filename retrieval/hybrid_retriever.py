"""
Hybrid Retriever v2 — FAISS vector search + BM25 keyword search + Cross-Encoder re-ranking.

This replaces the old pure-FAISS retrieval with a three-stage pipeline:
1. FAISS (semantic similarity) — finds chunks with similar meaning
2. BM25 (keyword matching) — finds chunks with exact legal terms/section numbers
3. Cross-Encoder re-ranking — scores each candidate against the query for final relevance

The combination dramatically improves accuracy for legal queries where both
meaning AND specific terms (section numbers, act names) matter.
"""

import os
import json
import logging
import math
import numpy as np
import faiss
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded models (initialized on first use to save memory)
_embedder = None
_cross_encoder = None
_bm25_bare = None
_bm25_case = None


def _get_embedder():
    """Lazy-load the sentence-transformer embedding model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        from config import EMBEDDING_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _embedder


def _get_cross_encoder():
    """Lazy-load the cross-encoder re-ranker model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        from config import CROSS_ENCODER_MODEL
        logger.info(f"Loading cross-encoder '{CROSS_ENCODER_MODEL}'")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# BM25 Implementation (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class BM25:
    """
    Simple BM25 (Okapi BM25) implementation for keyword-based retrieval.
    Tokenizes on whitespace + punctuation, case-insensitive.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}       # term -> number of docs containing it
        self.term_freqs = []      # list of {term: count} per document
        self.idf_cache = {}

    @staticmethod
    def _tokenize(text: str) -> list:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        import re
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return tokens

    def fit(self, documents: list):
        """Build BM25 index from a list of text documents."""
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)

        # Pre-compute IDF
        self.idf_cache = {}
        for term, df in self.doc_freqs.items():
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = idf

    def score(self, query: str, top_k: int = 50) -> list:
        """Score all documents against query. Returns list of (doc_index, score)."""
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            doc_score = 0.0
            dl = self.doc_lengths[i]
            tf_dict = self.term_freqs[i]

            for token in query_tokens:
                if token not in self.idf_cache:
                    continue
                idf = self.idf_cache[token]
                tf = tf_dict.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                doc_score += idf * (numerator / max(denominator, 0.001))

            if doc_score > 0:
                scores.append((i, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        """Serialize BM25 index for saving to disk."""
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25":
        """Deserialize BM25 index from disk."""
        bm25 = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        bm25.doc_count = data["doc_count"]
        bm25.avg_dl = data["avg_dl"]
        bm25.doc_lengths = data["doc_lengths"]
        bm25.doc_freqs = data["doc_freqs"]
        bm25.term_freqs = data["term_freqs"]
        # Rebuild IDF cache
        bm25.idf_cache = {}
        for term, df in bm25.doc_freqs.items():
            bm25.idf_cache[term] = math.log(
                (bm25.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
        return bm25


def save_bm25_index(bm25: BM25, path: str):
    """Save BM25 index to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bm25.to_dict(), f)
    logger.info(f"BM25 index saved to {path}")


def load_bm25_index(path: str) -> Optional[BM25]:
    """Load BM25 index from JSON file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return BM25.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load BM25 index from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# FAISS Helpers
# ---------------------------------------------------------------------------

def safe_read_faiss(index_path: str):
    """Read FAISS index. Returns (index, True) or (None, False)."""
    try:
        if not os.path.exists(index_path):
            return None, False
        idx = faiss.read_index(index_path)
        return idx, True
    except Exception as e:
        logger.error(f"Failed to read FAISS index {index_path}: {e}")
        return None, False


def safe_write_faiss(index, path: str) -> bool:
    """Write FAISS index. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        return True
    except Exception as e:
        logger.error(f"Failed to write FAISS index to {path}: {e}")
        return False


def load_chunks(chunks_path: str) -> dict:
    """Load chunk store from JSON. Returns dict or empty dict on failure."""
    if not os.path.exists(chunks_path):
        return {}
    try:
        with open(chunks_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load chunks from {chunks_path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Hybrid Search: FAISS + BM25 + Cross-Encoder Re-Ranking
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    faiss_index_path: str,
    chunks_path: str,
    bm25_index_path: str,
    faiss_top_k: int = 50,
    bm25_top_k: int = 50,
    rerank_top_k: int = 20,
    min_rerank_score: float = 0.0,
) -> list:
    """
    Three-stage hybrid search:
    1. FAISS semantic search (top faiss_top_k)
    2. BM25 keyword search (top bm25_top_k)
    3. Merge, deduplicate, cross-encoder re-rank (top rerank_top_k)

    Returns list of chunk dicts, each with '_rerank_score' field, sorted by relevance.
    """
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning(f"No chunks found at {chunks_path}")
        return []

    # --- Stage 1: FAISS vector search ---
    faiss_candidates = set()
    faiss_index, ok = safe_read_faiss(faiss_index_path)
    if ok and faiss_index:
        try:
            embedder = _get_embedder()
            query_vec = embedder.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            k = min(faiss_top_k, faiss_index.ntotal)
            if k > 0:
                distances, indices = faiss_index.search(
                    np.array([query_vec], dtype="float32"), k
                )
                for rank, idx in enumerate(indices[0]):
                    if idx >= 0 and str(idx) in chunks:
                        faiss_candidates.add(str(idx))
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")

    # --- Stage 2: BM25 keyword search ---
    bm25_candidates = set()
    bm25 = load_bm25_index(bm25_index_path)
    if bm25:
        try:
            bm25_results = bm25.score(query, top_k=bm25_top_k)
            for doc_idx, score in bm25_results:
                key = str(doc_idx)
                if key in chunks:
                    bm25_candidates.add(key)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

    # --- Merge candidates ---
    all_candidate_keys = faiss_candidates | bm25_candidates
    if not all_candidate_keys:
        logger.info("No candidates from either FAISS or BM25")
        return []

    logger.info(
        f"Hybrid search: {len(faiss_candidates)} FAISS + "
        f"{len(bm25_candidates)} BM25 = {len(all_candidate_keys)} unique candidates"
    )

    # --- Stage 3: Cross-encoder re-ranking ---
    candidate_chunks = []
    candidate_texts = []
    for key in all_candidate_keys:
        chunk = chunks[key]
        # Use search_text if available (richer), else fall back to text/full_text
        text = (
            chunk.get("search_text")
            or chunk.get("full_text")
            or chunk.get("text")
            or ""
        ).strip()
        if len(text) < 30:
            continue
        candidate_chunks.append((key, chunk))
        # Truncate for cross-encoder (max ~512 tokens)
        candidate_texts.append(text[:1500])

    if not candidate_chunks:
        return []

    try:
        cross_encoder = _get_cross_encoder()
        pairs = [(query, text) for text in candidate_texts]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)

        # Combine with scores
        scored = []
        for i, (key, chunk) in enumerate(candidate_chunks):
            rerank_score = float(scores[i])
            if rerank_score >= min_rerank_score:
                result = dict(chunk)
                result["_chunk_key"] = key
                result["_rerank_score"] = rerank_score
                result["_in_faiss"] = key in faiss_candidates
                result["_in_bm25"] = key in bm25_candidates
                scored.append(result)

        # Sort by re-rank score
        scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
        results = scored[:rerank_top_k]

        logger.info(
            f"Re-ranked to {len(results)} results "
            f"(top score: {results[0]['_rerank_score']:.3f})" if results else "Re-ranked to 0 results"
        )
        return results

    except Exception as e:
        logger.error(f"Cross-encoder re-ranking failed: {e}")
        # Fallback: return FAISS candidates sorted by original score
        fallback = []
        for key in faiss_candidates:
            if key in chunks:
                chunk = dict(chunks[key])
                chunk["_chunk_key"] = key
                chunk["_rerank_score"] = chunk.get("_score", 0.5)
                fallback.append(chunk)
        return fallback[:rerank_top_k]


def search_bare_acts(query: str, top_k: int = 30) -> list:
    """Search bare acts using hybrid retrieval. Returns all relevant sections."""
    from config import BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX
    results = hybrid_search(
        query=query,
        faiss_index_path=BARE_INDEX_V2,
        chunks_path=BARE_CHUNKS_V2,
        bm25_index_path=BARE_BM25_INDEX,
        faiss_top_k=50,
        bm25_top_k=50,
        rerank_top_k=top_k,
        min_rerank_score=-5.0,  # keep generous; sufficiency analyzer decides
    )
    # Tag each result
    for r in results:
        r["source_tag"] = "LOCAL_DB"
    return results


def search_case_laws(query: str, top_k: int = 30) -> list:
    """Search case laws using hybrid retrieval."""
    from config import CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX
    results = hybrid_search(
        query=query,
        faiss_index_path=CASE_INDEX_V2,
        chunks_path=CASE_CHUNKS_V2,
        bm25_index_path=CASE_BM25_INDEX,
        faiss_top_k=50,
        bm25_top_k=50,
        rerank_top_k=top_k,
        min_rerank_score=-5.0,
    )
    for r in results:
        r["source_tag"] = "LOCAL_DB"
    return results


# ---------------------------------------------------------------------------
# Fallback: Search legacy v1 index if v2 not yet built
# ---------------------------------------------------------------------------

def search_bare_acts_legacy(query: str, top_k: int = 50, min_sim: float = 0.45) -> list:
    """Search bare acts from legacy v1 FAISS index (blind chunking)."""
    from config import BARE_INDEX, BARE_CHUNKS
    index, ok = safe_read_faiss(BARE_INDEX)
    if not ok or not index:
        return []
    chunks = load_chunks(BARE_CHUNKS)
    if not chunks:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            chunk["_rerank_score"] = score
            chunk["source_tag"] = "LOCAL_DB"
            results.append(chunk)
    return results


def search_case_laws_legacy(query: str, top_k: int = 15, min_sim: float = 0.40) -> list:
    """Search case laws from legacy v1 FAISS index."""
    from config import CASE_INDEX, CASE_CHUNKS
    index, ok = safe_read_faiss(CASE_INDEX)
    if not ok or not index:
        return []
    chunks = load_chunks(CASE_CHUNKS)
    if not chunks:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    distances, indices = index.search(np.array([query_vec], dtype="float32"), k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        score = float(distances[0][rank])
        if score < min_sim:
            continue
        key = str(idx)
        if key in chunks:
            chunk = dict(chunks[key])
            chunk["_rerank_score"] = score
            chunk["source_tag"] = "LOCAL_DB"
            results.append(chunk)
    return results


def search_bare_acts_auto(query: str, top_k: int = 30) -> list:
    """Auto-detect v2 or legacy index and search accordingly."""
    from config import BARE_INDEX_V2
    if os.path.exists(BARE_INDEX_V2):
        results = search_bare_acts(query, top_k)
        if results:
            return results
    return search_bare_acts_legacy(query, top_k)


def search_case_laws_auto(query: str, top_k: int = 30) -> list:
    """Auto-detect v2 or legacy index and search accordingly."""
    from config import CASE_INDEX_V2
    if os.path.exists(CASE_INDEX_V2):
        results = search_case_laws(query, top_k)
        if results:
            return results
    return search_case_laws_legacy(query, top_k)
