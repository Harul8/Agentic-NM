"""
Ollama model configuration — single source for all LLM operations.

Model selection logic (see llm/ollama_client._get_model_for_prompt):
  1. If caller passes an explicit model name → use it.
  2. If task_hint is "long_context" → use OLLAMA_MODEL_LONG_CONTEXT.
  3. If prompt length > LONG_CONTEXT_THRESHOLD → use OLLAMA_MODEL_LONG_CONTEXT.
  4. Otherwise → use OLLAMA_MODEL (default).

Default: Qwen 3 8B for normal and long-context prompts (32K). Override via .env.

Env vars:
  OLLAMA_MODEL              — default model (e.g. qwen3:8b)
  OLLAMA_MODEL_LONG_CONTEXT — model for long prompts (e.g. qwen3:8b)
  LONG_CONTEXT_THRESHOLD    — switch when prompt length > this (default 120000 ≈ 30K tokens)
  OLLAMA_MODEL_DISPLAY      — UI label for default model (e.g. "Qwen 3 8B")
  OLLAMA_MODEL_LONG_CONTEXT_DISPLAY — UI label for long-context model
"""

import os

# Default model — Qwen 3 8B (good for instruction/JSON/summaries; fits 8GB GPU with quantization)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b").strip() or "qwen3:8b"

# Long-context model — same as default (Qwen 3 8B has 32K context)
OLLAMA_MODEL_LONG_CONTEXT = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT", "qwen3:8b"
).strip() or "qwen3:8b"

# Switch to long-context model when prompt exceeds this many characters (~30K tokens at ≈4 chars/token)
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "120000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 120000

# CrewAI agents use default model (format: ollama/model_name)
CREWAI_LLM = f"ollama/{OLLAMA_MODEL}"

# Display names for UI (e.g. "Qwen 3 8B")
OLLAMA_MODEL_DISPLAY = os.environ.get("OLLAMA_MODEL_DISPLAY", "Qwen 3 8B").strip() or "Qwen 3 8B"
OLLAMA_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT_DISPLAY", "Qwen 3 8B"
).strip() or "Qwen 3 8B"
