"""
Ollama model configuration — single source for all LLM operations.

Model selection logic (see llm/ollama_client._get_model_for_prompt):
  1. If caller passes an explicit model name → use it.
  2. If task_hint is "long_context" → use OLLAMA_MODEL_LONG_CONTEXT.
  3. If prompt length > LONG_CONTEXT_THRESHOLD → use OLLAMA_MODEL_LONG_CONTEXT.
  4. Otherwise → use OLLAMA_MODEL (default).

Default: Qwen2.5 7B for normal prompts; auto-switch to Llama 3.1 8B for long context (128K).

Env vars:
  OLLAMA_MODEL              — default model (e.g. qwen2.5:7b-instruct)
  OLLAMA_MODEL_LONG_CONTEXT — model for long prompts (e.g. llama3.1:8b)
  LONG_CONTEXT_THRESHOLD    — switch when prompt length > this (default 120000 ≈ 30K tokens)
  OLLAMA_MODEL_DISPLAY      — UI label for default model (e.g. "Qwen 2.5 7B")
  OLLAMA_MODEL_LONG_CONTEXT_DISPLAY — UI label for long-context model (e.g. "Llama 3.1 8B")
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

# Display names for UI (e.g. "Qwen 2.5 7B", "Llama 3.1 8B")
OLLAMA_MODEL_DISPLAY = os.environ.get("OLLAMA_MODEL_DISPLAY", "Qwen 2.5 7B").strip() or "Qwen 2.5 7B"
OLLAMA_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT_DISPLAY", "Llama 3.1 8B"
).strip() or "Llama 3.1 8B"
