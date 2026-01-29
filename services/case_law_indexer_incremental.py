"""
Incremental Case Law Indexer - Add user-confirmed case laws to the vector store.
"""

import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE = os.path.join(BASE_DIR, "data", "vector_store")
CASE_INDEX = os.path.join(VECTOR_STORE, "caselaws.index")
CASE_CHUNKS = os.path.join(VECTOR_STORE, "caselaws_chunks.json")
CASELAW_DIR = os.path.join(BASE_DIR, "data", "CaseLaws")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def chunk_text(text: str, size: int = 800) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)]


def index_new_case_laws(case_laws: list) -> dict:
    """
    Add new case laws to the existing vector store.
    case_laws: list of {title, url, content, source}
    Returns: {success: bool, chunks_added: int, message: str}
    """
    if not case_laws:
        return {"success": False, "chunks_added": 0, "message": "No case laws to index"}

    os.makedirs(VECTOR_STORE, exist_ok=True)
    os.makedirs(CASELAW_DIR, exist_ok=True)

    # Load existing chunks and index
    chunk_store = {}
    if os.path.exists(CASE_CHUNKS):
        with open(CASE_CHUNKS, encoding="utf-8") as f:
            chunk_store = json.load(f)

    existing_count = len(chunk_store)
    chunk_id = existing_count

    all_embeddings = []
    new_chunks = {}

    for case in case_laws:
        title = case.get("title", "Unknown")
        url = case.get("url", "")
        # Prefer relevant_portion (LLM-extracted key portions) for better search
        content = (
            case.get("relevant_portion")
            or case.get("content")
            or case.get("snippet")
            or ""
        )
        if not content:
            continue

        # Save to CaseLaws folder
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
        filename = f"Indexed_{safe_title}.txt"
        filepath = os.path.join(CASELAW_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"SOURCE: {url}\n\n")
            f.write(content)

        # Chunk and embed
        chunks = chunk_text(content)
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

    # Update FAISS index
    dim = len(all_embeddings[0])
    if os.path.exists(CASE_INDEX):
        index = faiss.read_index(CASE_INDEX)
        if index.d != dim:
            return {"success": False, "chunks_added": 0, "message": "Dimension mismatch with existing index"}
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(np.array(all_embeddings, dtype="float32"))
    faiss.write_index(index, CASE_INDEX)

    # Merge and save chunks
    chunk_store.update(new_chunks)
    with open(CASE_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, indent=2)

    return {
        "success": True,
        "chunks_added": len(new_chunks),
        "message": f"Successfully indexed {len(case_laws)} case law(s) with {len(new_chunks)} chunks"
    }
