import requests

from llm.config import OLLAMA_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_llm(prompt, model=None):
    model = model or OLLAMA_MODEL
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]
