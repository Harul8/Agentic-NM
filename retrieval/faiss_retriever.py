import faiss
import requests
import numpy as np
import pdfplumber

INDEX_PATH = "data/vector_store/contract_act.index"
PDF_PATH = "data/BareActs/THE INDIAN CONTRACT ACT 1872.pdf"

def search_faiss(query_text, top_k=1):
    index = faiss.read_index(INDEX_PATH)

    emb = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": query_text}
    ).json()["embedding"]

    D, I = index.search(np.array([emb], dtype="float32"), top_k)

    full_text = "\n".join(
        [p.extract_text() for p in pdfplumber.open(PDF_PATH).pages if p.extract_text()]
    )
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    return [chunks[i] for i in I[0] if i >= 0]
