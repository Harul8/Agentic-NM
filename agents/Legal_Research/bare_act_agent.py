from crewai import Agent
from crewai.tools import tool
import faiss, json, requests, numpy as np

def retrieve_bare_act_section(query_text):
    """Retrieve the most relevant Bare Act section for a given legal query."""

    # Load FAISS index
    index = faiss.read_index("data/vector_store/bareacts.index")

    # Load stored chunks
    with open("data/vector_store/bareacts_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Generate embedding
    emb = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": query_text}
    ).json()["embedding"]

    # Search
    D, I = index.search(np.array([emb], dtype="float32"), 3)

    # Return top matches
    results = []
    for idx in I[0]:
        results.append(chunks[str(idx)])

    return "\n\n---\n\n".join(results)

@tool
def bare_act_tool(query_text: str):
    """
    Retrieve relevant Bare Act provisions for a legal query.
    """
    return retrieve_bare_act_section(query_text)

bare_act_agent = Agent(
    role="Bare Act Researcher",
    goal="Retrieve exact statutory provisions from indexed Bare Acts.",
    backstory="You are a statutory retrieval system. You never invent law.",
    llm="ollama/llama3.1:8b",
    tools=[bare_act_tool],
    verbose=True
)
