import os
import faiss
import numpy as np
import requests
from tqdm import tqdm
import os

# Absolute project root (Nyaymalaw 3.0)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CASELAW_DIR = os.path.join(DATA_DIR, "CaseLaws")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Vector store files
CASELAW_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "caselaws.index")
CASELAW_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "caselaws_chunks.json")

# Chunking
CHUNK_SIZE = 800


def embed(text):
    
    res = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return res.json()["embedding"]


def load_case_laws():
    texts = []

    for fname in os.listdir(CASELAW_DIR):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(CASELAW_DIR, fname), encoding="utf-8") as f:
            content = f.read()

        for i in range(0, len(content), CHUNK_SIZE):
            texts.append(content[i:i + CHUNK_SIZE])

    return texts


def main():
    print("📚 Reading case law files...")
    chunks = load_case_laws()

    if not chunks:
        print("❌ No case law text found.")
        return

    print(f"🔹 Total chunks: {len(chunks)}")

    embeddings = []
    for chunk in tqdm(chunks):
        embeddings.append(embed(chunk))

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Case law FAISS index saved at {INDEX_PATH}")


if __name__ == "__main__":
    main()
