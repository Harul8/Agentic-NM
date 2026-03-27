"""
Ollama model configuration — single source for all LLM operations.

Model selection logic (see llm/ollama_client._get_model_for_prompt):
  1. If caller passes an explicit model name → use it.
  2. If task_hint is "long_context" → use OLLAMA_MODEL_LONG_CONTEXT.
  3. If prompt length > LONG_CONTEXT_THRESHOLD → use OLLAMA_MODEL_LONG_CONTEXT.
  4. Otherwise → use OLLAMA_MODEL (default).

Default: Qwen 3.5 9B for normal and long-context prompts. Override via .env.

Env vars:
  OLLAMA_MODEL              — default model (e.g. qwen3:8b)
  OLLAMA_MODEL_FAST         — fast model for intake / routing / lightweight chat
  OLLAMA_MODEL_LONG_CONTEXT — model for long prompts (e.g. qwen3.5:9b)
  LONG_CONTEXT_THRESHOLD    — switch when prompt length > this (default 120000 ≈ 30K tokens)
  OLLAMA_KEEP_ALIVE         — Ollama keep-alive window for loaded models (default 30m)
  OLLAMA_TIMEOUT_FAST_SEC   — timeout for fast intake/routing calls
  OLLAMA_TIMEOUT_DEFAULT_SEC — timeout for normal calls
  OLLAMA_TIMEOUT_LONG_SEC   — timeout for long-context calls
  OLLAMA_RETRIES_FAST       — retry count for fast calls
  OLLAMA_RETRIES_DEFAULT    — retry count for normal calls
  OLLAMA_RETRIES_LONG       — retry count for long-context calls
  OLLAMA_WARM_ANALYSIS_AT_STARTUP — warm the default analysis model on startup (default false)
  OLLAMA_MODEL_DISPLAY      — UI label for default model (e.g. "Qwen 3.5 9B")
  OLLAMA_MODEL_FAST_DISPLAY — UI label for fast model
  OLLAMA_MODEL_LONG_CONTEXT_DISPLAY — UI label for long-context model
"""

import os

# Default model — Llama 3.1 8B Instruct
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b").strip() or "llama3.1:8b"

# Fast model — defaults to Llama 3.2 3B for intake/routing unless overridden.
OLLAMA_MODEL_FAST = os.environ.get("OLLAMA_MODEL_FAST", "llama3.2:3b").strip() or "llama3.2:3b"

# Long-context model — Llama 3.2 27B for prompts exceeding the threshold
OLLAMA_MODEL_LONG_CONTEXT = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT", "llama3.2:27b"
).strip() or "llama3.2:27b"

# Switch to long-context model when prompt exceeds this many characters (~12.5K tokens at ≈4 chars/token)
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "50000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 50000

# CrewAI agents use default model (format: ollama/model_name)
CREWAI_LLM = f"ollama/{OLLAMA_MODEL}"

# Keep models resident for much longer so the first analysis after a quiet period
# does not pay the full load cost again. Override via env in tighter-memory setups.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "12h").strip() or "12h"

def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name, str(default)) or str(default)).strip())
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, str(default)) or str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")

OLLAMA_TIMEOUT_FAST_SEC = _int_env("OLLAMA_TIMEOUT_FAST_SEC", 25)
OLLAMA_TIMEOUT_DEFAULT_SEC = _int_env("OLLAMA_TIMEOUT_DEFAULT_SEC", 180)
OLLAMA_TIMEOUT_LONG_SEC = _int_env("OLLAMA_TIMEOUT_LONG_SEC", 240)
OLLAMA_RETRIES_FAST = _int_env("OLLAMA_RETRIES_FAST", 0)
OLLAMA_RETRIES_DEFAULT = _int_env("OLLAMA_RETRIES_DEFAULT", 1)
OLLAMA_RETRIES_LONG = _int_env("OLLAMA_RETRIES_LONG", 1)
OLLAMA_WARM_ANALYSIS_AT_STARTUP = _bool_env("OLLAMA_WARM_ANALYSIS_AT_STARTUP", False)

# Display names for UI (e.g. "Qwen 3.5 9B")
OLLAMA_MODEL_DISPLAY = os.environ.get("OLLAMA_MODEL_DISPLAY", "Llama 3.1 8B").strip() or "Llama 3.1 8B"
OLLAMA_MODEL_FAST_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_FAST_DISPLAY", "Llama 3.2 3B"
).strip() or "Llama 3.2 3B"
OLLAMA_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT_DISPLAY", "Llama 3.2 27B"
).strip() or "Llama 3.2 27B"
