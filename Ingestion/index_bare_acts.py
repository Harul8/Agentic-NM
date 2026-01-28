import os
import json
import faiss
import numpy as np
import pdfplumber
import requests

# =========================
# PROJECT PATHS (ABSOLUTE)
# =========================

# Nyaymalaw 3.0 (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BARE_ACTS_DIR = os.path.join(DATA_DIR, "BareActs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

BAREACT_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "bareacts.index")
BAREACT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "bareacts_chunks.json")

EMBEDDING_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800

# =========================
# UTILS
# =========================

def load_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE):
    """Split text into fixed-size chunks."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def embed(text: str):
    """Generate embedding using Ollama."""
    resp = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=60
    )

    resp.raise_for_status()
    data = resp.json()

    if "embedding" not in data:
        raise ValueError("❌ Embedding API returned no embedding")

    return data["embedding"]

# =========================
# MAIN INGESTION
# =========================

def main():
    print("📁 Project root:", PROJECT_ROOT)
    print("📁 Bare Acts dir:", BARE_ACTS_DIR)

    if not os.path.exists(BARE_ACTS_DIR):
        raise FileNotFoundError(f"❌ BareActs folder not found: {BARE_ACTS_DIR}")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    all_embeddings = []
    chunk_store = {}
    chunk_id = 0

    files = os.listdir(BARE_ACTS_DIR)
    print("📄 Files found:", files)

    for file in files:
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(BARE_ACTS_DIR, file)
        print(f"➡️ Processing: {file}")

        text = load_pdf(pdf_path)

        if not text.strip():
            print(f"⚠️ No extractable text in {file} (possibly scanned)")
            continue

        chunks = chunk_text(text)
        print(f"   Chunks created: {len(chunks)}")

        for chunk in chunks:
            embedding = embed(chunk)

            chunk_store[str(chunk_id)] = {
                "source": file,
                "text": chunk
            }

            all_embeddings.append(embedding)
            chunk_id += 1

    if not all_embeddings:
        raise RuntimeError("❌ No chunks indexed. Aborting.")

    # =========================
    # FAISS INDEX
    # =========================

    dim = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(all_embeddings, dtype="float32"))

    faiss.write_index(index, BAREACT_INDEX_PATH)

    with open(BAREACT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, ensure_ascii=False, indent=2)

    print("\n✅ Bare Acts indexing completed")
    print("📦 Index:", BAREACT_INDEX_PATH)
    print("📦 Chunks:", BAREACT_CHUNKS_PATH)
    print("🔢 Total chunks:", len(chunk_store))


if __name__ == "__main__":
    main()
