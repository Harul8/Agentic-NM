"""
Ollama model configuration — single source for all LLM operations.

Default: Qwen2.5 7B for normal prompts; auto-switch to Llama 3.1 8B for long context (128K).

Env vars:
  OLLAMA_MODEL              — default model (e.g. qwen2.5:7b-instruct)
  OLLAMA_MODEL_LONG_CONTEXT — model for long prompts (e.g. llama3.1:8b)
  LONG_CONTEXT_THRESHOLD    — switch when prompt length > this many characters (default 120000 ≈ 30K tokens)
"""

import os

# Default model — best for instruction/JSON/summaries (Qwen)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip() or "qwen2.5:7b-instruct"

# Long-context model — used automatically when prompt exceeds threshold (Llama 128K)
OLLAMA_MODEL_LONG_CONTEXT = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT", "llama3.1:8b"
).strip() or "llama3.1:8b"

# Switch to long-context model when prompt exceeds this many characters (~30K tokens at ≈4 chars/token)
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "120000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 120000

# CrewAI agents use default model (format: ollama/model_name)
CREWAI_LLM = f"ollama/{OLLAMA_MODEL}"
