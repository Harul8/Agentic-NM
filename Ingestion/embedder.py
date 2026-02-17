# DEPRECATED — Embedding is handled by retrieval.hybrid_retriever.
# This file kept only so old imports don't crash.
def embed(texts):
    from retrieval.hybrid_retriever import _get_embedder
    return _get_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
