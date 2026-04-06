"""
OpenAI model configuration — single source of truth for all LLM operations.
"""

import os


def _int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name, str(default)) or str(default)).strip())
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, str(default)) or str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Three UI-selectable model tiers
# ---------------------------------------------------------------------------
MODEL_TIERS: dict[str, dict] = {
    "gpt5mini": {
        "fast":          "gpt-5-nano",
        "regular":       "gpt-5-mini",
        "large_context": "gpt-5-mini",
        "display":       "GPT-5 Mini",
    },
    "gpt51mini": {
        "fast":          "gpt-5-mini",
        "regular":       "gpt-5.1-mini",
        "large_context": "gpt-5.1-mini",
        "display":       "GPT-5.1 Mini",
    },
    "gpt54mini": {
        "fast":          "gpt-5.1-mini",
        "regular":       "gpt-5.4-mini",
        "large_context": "gpt-5.4-mini",
        "display":       "GPT-5.4 Mini",
    },
}

DEFAULT_TIER: str = os.environ.get("DEFAULT_MODEL_TIER", "gpt5mini").strip() or "gpt5mini"


def get_tier(tier_key: str | None) -> dict:
    """Return the tier dict for the given key, falling back to the default tier."""
    return MODEL_TIERS.get(tier_key or DEFAULT_TIER, MODEL_TIERS[DEFAULT_TIER])


# ---------------------------------------------------------------------------
# Backward-compatible constants (used widely in services/tests) — derived from
# the default tier so existing code continues to work unchanged.
# ---------------------------------------------------------------------------
_default_tier = get_tier(DEFAULT_TIER)

OPENAI_MODEL              = _default_tier["regular"]
OPENAI_MODEL_FAST         = _default_tier["fast"]
OPENAI_MODEL_LONG_CONTEXT = _default_tier["large_context"]
OPENAI_MODEL_DISPLAY      = _default_tier["display"]
OPENAI_MODEL_FAST_DISPLAY = _default_tier["display"]
OPENAI_MODEL_LONG_CONTEXT_DISPLAY = _default_tier["display"]

# CrewAI LLM string
CREWAI_LLM = f"openai/{OPENAI_MODEL}"

# Trim prompts above this many characters before sending to the API.
_threshold = os.environ.get("LONG_CONTEXT_THRESHOLD", "60000").strip()
try:
    LONG_CONTEXT_THRESHOLD = int(_threshold)
except ValueError:
    LONG_CONTEXT_THRESHOLD = 60000

# Timeout / retry settings (env-overridable)
OLLAMA_TIMEOUT_FAST_SEC    = _int_env("OPENAI_TIMEOUT_FAST_SEC",    120)
OLLAMA_TIMEOUT_DEFAULT_SEC = _int_env("OPENAI_TIMEOUT_DEFAULT_SEC", 300)
OLLAMA_TIMEOUT_LONG_SEC    = _int_env("OPENAI_TIMEOUT_LONG_SEC",    360)
OLLAMA_RETRIES_FAST        = _int_env("OPENAI_RETRIES_FAST",        0)
OLLAMA_RETRIES_DEFAULT     = _int_env("OPENAI_RETRIES_DEFAULT",     1)
OLLAMA_RETRIES_LONG        = _int_env("OPENAI_RETRIES_LONG",        1)

# Legacy stubs — kept so any remaining import doesn't crash
LLM_PROVIDER                    = "openai"
OLLAMA_WARM_ANALYSIS_AT_STARTUP = False
