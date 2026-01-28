import os
import json
import faiss
import numpy as np
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer

# ===============================
# PROJECT PATHS (ABSOLUTE)
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BARE_ACTS_DIR = os.path.join(DATA_DIR, "BareActs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

BAREACT_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "bareacts.index")
BAREACT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "bareacts_chunks.json")

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ===============================
# GPU EMBEDDING MODEL
# ===============================
print("🚀 Loading embedding model on GPU...")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

# ===============================
# HELPERS
# ===============================
def load_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, size: int = 800):
    return [text[i:i + size] for i in range(0, len(text), size)]


def embed_batch(texts):
    return embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

# ===============================
# MAIN INGESTION
# ===============================
def main():
    print("📁 Project root:", PROJECT_ROOT)
    print("📁 Bare Acts dir:", BARE_ACTS_DIR)

    files = [f for f in os.listdir(BARE_ACTS_DIR) if f.lower().endswith(".pdf")]
    print("📄 Files found:", files)

    all_embeddings = []
    chunk_store = {}
    chunk_id = 0

    for file in files:
        print(f"\n➡️ Processing: {file}")
        text = load_pdf(os.path.join(BARE_ACTS_DIR, file))
        chunks = chunk_text(text)

        print(f"   Chunks created: {len(chunks)}")

        for i in range(0, len(chunks), 32):
            batch = chunks[i:i + 32]
            embeddings = embed_batch(batch)

            for chunk, emb in zip(batch, embeddings):
                chunk_store[str(chunk_id)] = {
                    "source": file,
                    "text": chunk
                }
                all_embeddings.append(emb)
                chunk_id += 1

            print(f"      Embedded {min(i+32, len(chunks))}/{len(chunks)}")

    print("\n📊 Total chunks embedded:", len(all_embeddings))

    dim = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(all_embeddings, dtype="float32"))

    faiss.write_index(index, BAREACT_INDEX_PATH)

    with open(BAREACT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, indent=2)

    print("\n✅ Bare Acts indexing COMPLETE")
    print("📦 Index →", BAREACT_INDEX_PATH)
    print("📄 Chunks →", BAREACT_CHUNKS_PATH)


if __name__ == "__main__":
    main()
