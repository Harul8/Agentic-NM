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
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VECTOR_STORE = os.path.join(BASE_DIR, "data", "vector_store")
INDEX_PATH = os.path.join(VECTOR_STORE, "bareacts.index")
CHUNKS_PATH = os.path.join(VECTOR_STORE, "bareacts_chunks.json")

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
def retrieve_bare_act_section(query: str):
    """
    Retrieve relevant Bare Act provisions using cosine similarity search.
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
bare_act_agent = Agent(
    role="Bare Act Researcher",
    goal="Retrieve exact statutory provisions from indexed Bare Acts.",
    backstory="You retrieve law exactly as written. You do not interpret.",
    tools=[retrieve_bare_act_section],
    llm=CREWAI_LLM,
    verbose=True
)
