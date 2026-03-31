"""
Ollama model configuration — single source for all LLM operations.
"""

import os

# LLM provider selection:
# - "ollama" (default): local Ollama models
# - "openai": API-backed models (e.g. gpt-5-mini)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower() or "ollama"

# Default model — legal analysis (thinking mode).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b").strip() or "qwen3:8b"

# Fast model — intake attempt1/attempt2 (no-think mode).
OLLAMA_MODEL_FAST = os.environ.get("OLLAMA_MODEL_FAST", "qwen3:4b").strip() or "qwen3:4b"

# Long-context placeholder (auto-switch disabled in ollama_client).
OLLAMA_MODEL_LONG_CONTEXT = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT", "qwen3:8b"
).strip() or "qwen3:8b"

# Trim prompts above this many characters before sending to Ollama.
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "60000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 60000

# CrewAI agents use default model (format: ollama/model_name)
CREWAI_LLM = f"ollama/{OLLAMA_MODEL}"

OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "12h").strip() or "12h"


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name, str(default)) or str(default)).strip())
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, str(default)) or str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


OLLAMA_TIMEOUT_FAST_SEC = _int_env("OLLAMA_TIMEOUT_FAST_SEC", 120)
OLLAMA_TIMEOUT_DEFAULT_SEC = _int_env("OLLAMA_TIMEOUT_DEFAULT_SEC", 300)
OLLAMA_TIMEOUT_LONG_SEC = _int_env("OLLAMA_TIMEOUT_LONG_SEC", 360)
OLLAMA_RETRIES_FAST = _int_env("OLLAMA_RETRIES_FAST", 0)
OLLAMA_RETRIES_DEFAULT = _int_env("OLLAMA_RETRIES_DEFAULT", 1)
OLLAMA_RETRIES_LONG = _int_env("OLLAMA_RETRIES_LONG", 1)
OLLAMA_WARM_ANALYSIS_AT_STARTUP = _bool_env("OLLAMA_WARM_ANALYSIS_AT_STARTUP", False)

# Qwen thinking mode flag.
# fast calls always pass think=False in ollama_client.
OLLAMA_THINK_ENABLED = _bool_env("OLLAMA_THINK_ENABLED", True)

# Display names for UI
OLLAMA_MODEL_DISPLAY = os.environ.get("OLLAMA_MODEL_DISPLAY", "Qwen 3 8B").strip() or "Qwen 3 8B"
OLLAMA_MODEL_FAST_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_FAST_DISPLAY", "Qwen 3 4B"
).strip() or "Qwen 3 4B"
OLLAMA_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OLLAMA_MODEL_LONG_CONTEXT_DISPLAY", "Qwen 3 8B"
).strip() or "Qwen 3 8B"

# OpenAI/API model mapping (used when LLM_PROVIDER=openai).
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-nano").strip() or "gpt-5-nano"
OPENAI_MODEL_LONG_CONTEXT = os.environ.get("OPENAI_MODEL_LONG_CONTEXT", OPENAI_MODEL).strip() or OPENAI_MODEL

OPENAI_MODEL_DISPLAY = os.environ.get("OPENAI_MODEL_DISPLAY", "GPT-5 mini").strip() or "GPT-5 mini"
OPENAI_MODEL_FAST_DISPLAY = os.environ.get(
    "OPENAI_MODEL_FAST_DISPLAY", "GPT-5 nano"
).strip() or "GPT-5 nano"
OPENAI_MODEL_LONG_CONTEXT_DISPLAY = os.environ.get(
    "OPENAI_MODEL_LONG_CONTEXT_DISPLAY", OPENAI_MODEL_DISPLAY
).strip() or OPENAI_MODEL_DISPLAY
