"""
Ollama client — single interface for all LLM calls.

Features:
- Retry with exponential backoff (connection/timeout errors only)
- Configurable timeout
- Health check for /health endpoint and startup validation
"""

import json
import logging
import time

import requests

from llm.config import (
    OLLAMA_MODEL,
    OLLAMA_MODEL_LONG_CONTEXT,
    LONG_CONTEXT_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _get_model_for_prompt(prompt: str, explicit_model: str = None) -> str:
    """
    Choose model: use long-context model when prompt exceeds threshold and no model was explicitly passed.
    """
    if explicit_model:
        return explicit_model
    if len(prompt) > LONG_CONTEXT_THRESHOLD:
        return OLLAMA_MODEL_LONG_CONTEXT
    return OLLAMA_MODEL

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

# Retry config — only retries on transient network errors, NOT on model/prompt errors
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2  # seconds; doubles each retry (2s, 4s)
DEFAULT_TIMEOUT = 300  # 5 minutes — long prompts on 7B model need time


def _extract_text(data: dict) -> str:
    """Get generated text from Ollama response. Handles different response shapes."""
    if not isinstance(data, dict):
        return ""
    for key in ("response", "message", "content", "text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def ask_llm(prompt: str, model: str = None, timeout: int = None) -> str:
    """
    Send a prompt to Ollama and return the generated text.

    If model is not specified: uses OLLAMA_MODEL (e.g. Qwen) for normal prompts,
    and OLLAMA_MODEL_LONG_CONTEXT (e.g. Llama 3.1 8B) when prompt length exceeds
    LONG_CONTEXT_THRESHOLD (default 35K chars). Retries on connection/timeout only.
    """
    chosen = _get_model_for_prompt(prompt, model)
    if model is None and chosen == OLLAMA_MODEL_LONG_CONTEXT:
        logger.debug(
            "Long prompt (%d chars > %d), using %s",
            len(prompt), LONG_CONTEXT_THRESHOLD, OLLAMA_MODEL_LONG_CONTEXT,
        )
    model = chosen
    timeout = timeout or DEFAULT_TIMEOUT
    payload = {"model": model, "prompt": prompt, "stream": False}

    for attempt in range(1 + MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_GENERATE_URL, json=payload, timeout=timeout
            )
            break  # success — exit retry loop
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Ollama request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, 1 + MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Ollama request failed after %d attempts: %s",
                    1 + MAX_RETRIES, e,
                )
                raise RuntimeError(
                    f"Ollama unreachable after {1 + MAX_RETRIES} attempts: {e}"
                ) from e
        except requests.RequestException as e:
            # Non-transient error (e.g. invalid URL) — fail immediately
            raise RuntimeError(f"Ollama request failed: {e}") from e

    # Parse response
    try:
        data = response.json() if (response.text and response.text.strip()) else {}
    except json.JSONDecodeError:
        data = {}

    if not response.ok:
        err = data.get("error", response.text or "Unknown error")
        raise RuntimeError(f"Ollama error ({response.status_code}): {err}")

    if data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")

    return _extract_text(data) or ""


def _model_available(name: str, model_names: list) -> bool:
    """True if the given model name is in the list (exact or prefix match)."""
    return any(
        name == n or name in n or n.startswith(name)
        for n in model_names
    )


def check_ollama_health() -> dict:
    """
    Check Ollama connectivity and model availability.

    Returns:
        {
            "ollama_reachable": bool,
            "model_loaded": bool,
            "model": str,
            "long_context_model_loaded": bool,
            "long_context_model": str,
            "error": str (only if something failed),
        }
    """
    result = {
        "ollama_reachable": False,
        "model_loaded": False,
        "model": OLLAMA_MODEL,
        "long_context_model": OLLAMA_MODEL_LONG_CONTEXT,
        "long_context_model_loaded": False,
    }

    # 1. Check if Ollama is reachable
    try:
        resp = requests.get(OLLAMA_BASE_URL, timeout=5)
        result["ollama_reachable"] = resp.ok
    except requests.RequestException as e:
        result["error"] = f"Ollama not reachable: {e}"
        return result

    # 2. Check if both models are available
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.ok:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            result["model_loaded"] = _model_available(OLLAMA_MODEL, model_names)
            result["long_context_model_loaded"] = _model_available(
                OLLAMA_MODEL_LONG_CONTEXT, model_names
            )
            if not result["model_loaded"]:
                result["error"] = (
                    f"Model '{OLLAMA_MODEL}' not found. "
                    f"Available: {', '.join(model_names[:5]) or 'none'}. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                )
            elif not result["long_context_model_loaded"]:
                result["long_context_warning"] = (
                    f"Long-context model '{OLLAMA_MODEL_LONG_CONTEXT}' not found. "
                    f"Long prompts will use '{OLLAMA_MODEL}'. "
                    f"Run: ollama pull {OLLAMA_MODEL_LONG_CONTEXT} to enable auto-switch."
                )
        else:
            result["error"] = f"Failed to list models: HTTP {resp.status_code}"
    except requests.RequestException as e:
        result["error"] = f"Failed to check models: {e}"

    return result
