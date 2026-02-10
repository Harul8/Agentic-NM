import json
import requests

from llm.config import OLLAMA_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"


def _extract_text(data):
    """Get generated text from Ollama response. Handles different response shapes."""
    if not isinstance(data, dict):
        return ""
    for key in ("response", "message", "content", "text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def ask_llm(prompt, model=None):
    model = model or OLLAMA_MODEL
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    try:
        data = response.json() if (response.text and response.text.strip()) else {}
    except json.JSONDecodeError:
        data = {}

    if not response.ok:
        err = data.get("error", response.text or "Unknown error")
        raise RuntimeError(f"Ollama error: {err}")

    if data.get("error"):
        raise RuntimeError(f"Ollama error: {data.get('error')}")

    return _extract_text(data) or ""
