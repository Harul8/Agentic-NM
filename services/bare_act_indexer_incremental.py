"""
Incremental Bare Act Indexer - Add user-confirmed bare act sections to the vector store.
"""

import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import VECTOR_STORE, BARE_INDEX, BARE_CHUNKS, BARE_ACTS_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def chunk_text(text: str, size: int = 800) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)]


def index_new_bare_acts(bare_acts: list) -> dict:
    """
    Add new bare act sections to the existing vector store.
    bare_acts: list of {title, url, text, act_name, source}
    Returns: {success: bool, chunks_added: int, message: str}
    """
    if not bare_acts:
        return {"success": False, "chunks_added": 0, "message": "No bare acts to index"}

    os.makedirs(VECTOR_STORE, exist_ok=True)
    os.makedirs(BARE_ACTS_DIR, exist_ok=True)

    chunk_store = {}
    if os.path.exists(BARE_CHUNKS):
        with open(BARE_CHUNKS, encoding="utf-8") as f:
            chunk_store = json.load(f)

    chunk_id = len(chunk_store)
    all_embeddings = []
    new_chunks = {}

    for item in bare_acts:
        title = item.get("title") or item.get("act_name", "Unknown")
        url = item.get("url", "")
        text = item.get("text") or item.get("content", "")
        if not text:
            continue

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(title))[:80]
        filename = f"Indexed_{safe_title}.txt"
        filepath = os.path.join(BARE_ACTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"SOURCE: {url}\n\n")
            f.write(text)

        chunks = chunk_text(text)
        for chunk in chunks:
            emb = embedder.encode(
                chunk,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            key = str(chunk_id)
            new_chunks[key] = {"source": filename, "text": chunk}
            all_embeddings.append(emb)
            chunk_id += 1

    if not all_embeddings:
        return {"success": False, "chunks_added": 0, "message": "No valid content to index"}

    dim = len(all_embeddings[0])
    try:
        if os.path.exists(BARE_INDEX):
            index = faiss.read_index(BARE_INDEX)
            if index.d != dim:
                return {"success": False, "chunks_added": 0, "message": "Dimension mismatch with existing index"}
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(np.array(all_embeddings, dtype="float32"))
        faiss.write_index(index, BARE_INDEX)
    except Exception as e:
        return {"success": False, "chunks_added": 0, "message": f"Vector store path not writable (e.g. Drive path on Windows): {str(e)[:80]}"}

    chunk_store.update(new_chunks)
    try:
        with open(BARE_CHUNKS, "w", encoding="utf-8") as f:
            json.dump(chunk_store, f, indent=2)
    except Exception:
        pass

    return {
        "success": True,
        "chunks_added": len(new_chunks),
        "message": f"Successfully indexed {len(bare_acts)} bare act section(s) with {len(new_chunks)} chunks"
    }
