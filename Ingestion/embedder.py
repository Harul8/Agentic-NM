# ingestion/embedder.py
import requests

def embed(text):
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model":"nomic-embed-text", "prompt": text}
    )
    return resp.json()["embedding"]
