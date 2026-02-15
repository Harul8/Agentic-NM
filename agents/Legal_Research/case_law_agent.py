import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from crewai import Agent
from crewai.tools import tool

from llm.config import CREWAI_LLM

# ===============================
# PATHS (from config – DATA_ROOT e.g. Google Drive)
# ===============================
from config import VECTOR_STORE, CASE_INDEX, CASE_CHUNKS
INDEX_PATH = CASE_INDEX
CHUNKS_PATH = CASE_CHUNKS

# ===============================
# GPU EMBEDDING MODEL
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

# ===============================
# TOOL
# ===============================
@tool
def retrieve_case_law(query: str):
    """
    Retrieve relevant Indian case law using cosine similarity search.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return []

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    D, I = index.search(np.array([query_vec], dtype="float32"), 5)

    results = []
    for idx in I[0]:
        key = str(idx)
        if key in chunks:
            results.append(chunks[key])

    return results


# ===============================
# AGENT
# ===============================
case_law_agent = Agent(
    role="Case Law Researcher",
    goal="Retrieve relevant Indian case law accurately.",
    backstory="You retrieve judicial precedents without interpretation.",
    tools=[retrieve_case_law],
    llm=CREWAI_LLM,
    verbose=True
)