from crewai import Agent
from crewai.tools import tool
import json
import faiss, requests, numpy as np, pdfplumber

@tool
def retrieve_bare_act_section(query_text):
    """Retrieve the most relevant Bare Act section for a given legal query."""
    
    # Load FAISS index
    index = faiss.read_index("data/vector_store/bareacts.index")

    # Generate embedding from Ollama
    emb = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": query_text}
    ).json()["embedding"]

    # Search FAISS
    D, I = index.search(np.array([emb], dtype="float32"), 1)

    # Load Bare Act PDF text
    full_text = "\n".join(
        [p.extract_text() for p in pdfplumber.open("data/BareActs/THE INDIAN CONTRACT ACT 1872.pdf").pages if p.extract_text()]
    )

    # Chunk the text
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    # Return best matching chunk
    return chunks[I[0][0]]


bare_act_agent = Agent(
    role="Bare Act Researcher",
    goal="Retrieve the most relevant Bare Act legal section.",
    backstory="You retrieve exact legal sections from indexed Bare Acts.",
    llm="ollama/llama3.1:8b",
    tools=[retrieve_bare_act_section],
    verbose=True
)
