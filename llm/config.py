"""
Ollama model configuration — single source for all LLM operations.

Model selection logic (see llm/ollama_client._get_model_for_prompt):
  1. If caller passes an explicit model name → use it.
  2. If task_hint is "long_context" → use OLLAMA_MODEL_LONG_CONTEXT.
  3. If prompt length > LONG_CONTEXT_THRESHOLD → use OLLAMA_MODEL_LONG_CONTEXT.
  4. Otherwise → use OLLAMA_MODEL (default).

Default: Qwen 3.5 9B for normal and long-context prompts. Override via .env.

Env vars:
  OLLAMA_MODEL              — default model (e.g. qwen3.5:9b)
  OLLAMA_MODEL_LONG_CONTEXT — model for long prompts (e.g. qwen3.5:9b)
  LONG_CONTEXT_THRESHOLD    — switch when prompt length > this (default 120000 ≈ 30K tokens)
  OLLAMA_KEEP_ALIVE         — Ollama keep-alive window for loaded models (default 30m)
  OLLAMA_MODEL_DISPLAY      — UI label for default model (e.g. "Qwen 3.5 9B")
  OLLAMA_MODEL_LONG_CONTEXT_DISPLAY — UI label for long-context model
"""

import os

# Default model — Qwen 3.5 9B
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b").strip() or "qwen3.5:9b"

# Long-context model — same as default unless overridden in env
OLLAMA_MODEL_LONG_CONTEXT = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT", "qwen3.5:9b"
).strip() or "qwen3.5:9b"

# Switch to long-context model when prompt exceeds this many characters (~30K tokens at ≈4 chars/token)
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "120000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 120000

# CrewAI agents use default model (format: ollama/model_name)
CREWAI_LLM = f"ollama/{OLLAMA_MODEL}"

# Keep the model resident in memory between requests to reduce cold starts.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"

# Display names for UI (e.g. "Qwen 3.5 9B")
OLLAMA_MODEL_DISPLAY = os.environ.get("OLLAMA_MODEL_DISPLAY", "Qwen 3.5 9B").strip() or "Qwen 3.5 9B"
OLLAMA_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT_DISPLAY", "Qwen 3.5 9B"
).strip() or "Qwen 3.5 9B"
