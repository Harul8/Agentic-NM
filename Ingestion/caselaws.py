import os
import faiss
import numpy as np
import requests
from ddgs import DDGS

VECTOR_PATH = "data/vector_store/caselaws.index"
CASELAW_DIR = "data/CaseLaws"


def embed(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
    )
    return np.array(response.json()["embedding"], dtype="float32")


def search_local_caselaws(query: str, top_k: int = 3):
    if not os.path.exists(VECTOR_PATH):
        return []

    index = faiss.read_index(VECTOR_PATH)
    query_vec = embed(query).reshape(1, -1)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(f"Matched CaseLaw Vector ID: {idx}")

    return results


def search_internet_caselaws(query: str, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(f"{query} Indian Supreme Court case", max_results=max_results):
            results.append({
                "title": r["title"],
                "url": r["href"]
            })
    return results


if __name__ == "__main__":
    query = input("Enter legal issue to search case laws: ")

    matches = search_local_caselaws(query)

    if matches:
        print("\n✅ Found relevant local case laws:")
        for m in matches:
            print("-", m)
    else:
        print("\n❌ No relevant case laws found locally.")
        print("🌐 Searching internet...\n")

        web_results = search_internet_caselaws(query)

        for i, r in enumerate(web_results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}\n")
