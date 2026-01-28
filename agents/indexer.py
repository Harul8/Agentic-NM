# ingestion/indexer.py
import faiss
import numpy as np
from embedder import embed

def build_index(chunks, save_path):
    vectors = [embed(c) for c in chunks]
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))
    faiss.write_index(index, save_path)
