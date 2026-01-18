from crewai import Agent
from crewai.tools import tool
import faiss, requests, numpy as np, pdfplumber

@tool
def retrieve_case_law(query_text):
    """Retrieve the most relevant case law paragraph for a given legal query."""
    
    index = faiss.read_index("data/vector_store/caselaw.index")

    emb = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": query_text}
    ).json()["embedding"]

    D, I = index.search(np.array([emb], dtype="float32"), 1)

    full_text = "\n".join(
        [p.extract_text() for p in pdfplumber.open("data/raw_pdfs/sample_judgment.pdf").pages if p.extract_text()]
    )

    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    return chunks[I[0][0]]


case_law_agent = Agent(
    role="Case Law Researcher",
    goal="Retrieve the most relevant legal precedents.",
    backstory="You search and retrieve relevant judgments from indexed case law.",
    llm="ollama/llama3.1:8b",
    tools=[retrieve_case_law],
    verbose=True
)
