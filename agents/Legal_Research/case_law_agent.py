from crewai import Agent
from crewai.tools import tool
import faiss
import numpy as np
import requests
import os

INDEX_PATH = "data/vector_store/caselaws.index"
CASELAW_DIR = "data/CaseLaws"
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


def load_case_chunks():
    chunks = []
    sources = []

    for fname in os.listdir(CASELAW_DIR):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(CASELAW_DIR, fname)
        with open(path, encoding="utf-8") as f:
            content = f.read()

        for i in range(0, len(content), CHUNK_SIZE):
            chunks.append(content[i:i + CHUNK_SIZE])
            sources.append(fname)

    return chunks, sources


def retrieve_case_law(query):
    """
    Retrieve the most relevant Indian case law for a given legal issue.
    """
    index = faiss.read_index(INDEX_PATH)
    query_emb = embed(query)

    D, I = index.search(
        np.array([query_emb]).astype("float32"),
        3
    )

    chunks, sources = load_case_chunks()

    results = []
    for idx in I[0]:
        results.append({
            "source": sources[idx],
            "text": chunks[idx]
        })

    return results

@tool
def case_law_tool(query_text: str):
    """
    Retrieve relevant Case Laws for a legal query.
    """
    return retrieve_case_law(query_text)

case_law_agent = Agent(
    role="Case Law Researcher",
    goal="Retrieve relevant Indian case law for the given legal issue.",
    backstory="You are an expert legal researcher skilled in Indian case law analysis.",
    llm="ollama/llama3.1:8b",
    tools=[case_law_tool],
    verbose=True
)
