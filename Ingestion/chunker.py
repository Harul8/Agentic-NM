# ingestion/chunker.py
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]
