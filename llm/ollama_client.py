"""
Ollama client — single interface for all LLM calls.

Features:
- Retry with exponential backoff (connection/timeout errors only)
- Configurable timeout
- Health check for /health endpoint and startup validation
- GPU detection (nvidia-smi) and model display names for UI
"""

import json
import logging
import subprocess
import threading
import time

import requests

from llm.config import (
    OLLAMA_MODEL,
    OLLAMA_MODEL_LONG_CONTEXT,
    LONG_CONTEXT_THRESHOLD,
    OLLAMA_MODEL_DISPLAY,
    OLLAMA_MODEL_LONG_CONTEXT_DISPLAY,
)

logger = logging.getLogger(__name__)

# Thread-local: model name used by the last ask_llm call in this thread (for API to include in response)
_last_model_used = threading.local()


def _get_model_for_prompt(
    prompt: str,
    explicit_model: str = None,
    task_hint: str = None,
) -> str:
    """
    Decide which Ollama model to use for this request. Priority order:

    1. explicit_model — if the caller passed a model name, use it.
    2. task_hint == "long_context" — use long-context model (e.g. Llama 3.1 8B).
    3. Prompt length > LONG_CONTEXT_THRESHOLD — use long-context model so we don't
       truncate; default threshold is 120_000 characters (~30K tokens).
    4. Otherwise — use default model (e.g. Qwen 2.5 7B).

    So: short prompts and most tasks use the default model; long prompts (or
    explicit task_hint) use the long-context model. The UI shows which model
    was actually used via get_last_model_used() in the API response.
    """
    if explicit_model:
        return explicit_model
    if task_hint == "long_context" or (
        not task_hint and len(prompt) > LONG_CONTEXT_THRESHOLD
    ):
        return OLLAMA_MODEL_LONG_CONTEXT
    return OLLAMA_MODEL


def get_gpu_info() -> list:
    """Return list of GPU names (e.g. ['NVIDIA GeForce RTX 3080']). Uses nvidia-smi if available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            names = [n.strip() for n in out.stdout.strip().split("\n") if n.strip()]
            return names if names else []
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return []


def get_display_name_for_model(model_name: str) -> str:
    """Map internal model name to UI-friendly label (e.g. 'Qwen 2.5 7B')."""
    if not model_name:
        return ""
    if OLLAMA_MODEL in model_name or model_name in OLLAMA_MODEL:
        return OLLAMA_MODEL_DISPLAY
    if OLLAMA_MODEL_LONG_CONTEXT in model_name or model_name in OLLAMA_MODEL_LONG_CONTEXT:
        return OLLAMA_MODEL_LONG_CONTEXT_DISPLAY
    return model_name


def get_model_display_for_prompt(
    prompt: str,
    task_hint: str = None,
) -> tuple:
    """
    Return (display_name, is_switched) for the model that would be used for this prompt.
    display_name: e.g. 'Qwen 2.5 7B' or 'Llama 3.1 8B'.
    is_switched: True when long-context model is used (so UI can show "Switching to X model").
    """
    chosen = _get_model_for_prompt(prompt, None, task_hint)
    display = get_display_name_for_model(chosen)
    is_switched = chosen == OLLAMA_MODEL_LONG_CONTEXT
    return (display, is_switched)


def get_last_model_used() -> str:
    """Display name of the model used by the last ask_llm call in this thread. Empty if none."""
    try:
        name = getattr(_last_model_used, "value", None)
        return get_display_name_for_model(name) if name else ""
    except Exception:
        return ""

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


def ask_llm_stream(
    prompt: str,
    model: str = None,
    timeout: int = None,
    task_hint: str = None,
):
    """
    Stream tokens from Ollama one fragment at a time.

    Yields each text fragment (typically 1-4 words) as it is produced by the model.
    Use when you want to pipe each token to a UI callback for live display.

    Example:
        full = ""
        for token in ask_llm_stream(prompt):
            token_callback(token)
            full += token
        return full.strip()
    """
    chosen = _get_model_for_prompt(prompt, model, task_hint)
    try:
        _last_model_used.value = chosen
    except Exception:
        pass
    timeout = timeout or DEFAULT_TIMEOUT
    payload = {"model": chosen, "prompt": prompt, "stream": True}
    try:
        with requests.post(
            OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=timeout
        ) as response:
            if not response.ok:
                err = response.text or "Unknown error"
                raise RuntimeError(f"Ollama error ({response.status_code}): {err}")
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama streaming failed: {e}") from e


def ask_llm(
    prompt: str,
    model: str = None,
    timeout: int = None,
    task_hint: str = None,
) -> str:
    """
    Send a prompt to Ollama and return the generated text.

    If model is not specified: uses OLLAMA_MODEL (e.g. Qwen) for normal prompts,
    and OLLAMA_MODEL_LONG_CONTEXT (e.g. Llama 3.1 8B) when prompt length exceeds
    LONG_CONTEXT_THRESHOLD or task_hint is "long_context". Retries on connection/timeout only.
    """
    chosen = _get_model_for_prompt(prompt, model, task_hint)
    if model is None and chosen == OLLAMA_MODEL_LONG_CONTEXT:
        logger.debug(
            "Long prompt (%d chars > %d), using %s",
            len(prompt), LONG_CONTEXT_THRESHOLD, OLLAMA_MODEL_LONG_CONTEXT,
        )
    try:
        _last_model_used.value = chosen
    except Exception:
        pass
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
            "model_display": str,
            "long_context_model_loaded": bool,
            "long_context_model": str,
            "long_context_model_display": str,
            "gpu_names": list[str],
            "error": str (only if something failed),
        }
    """
    result = {
        "ollama_reachable": False,
        "model_loaded": False,
        "model": OLLAMA_MODEL,
        "model_display": OLLAMA_MODEL_DISPLAY,
        "long_context_model": OLLAMA_MODEL_LONG_CONTEXT,
        "long_context_model_display": OLLAMA_MODEL_LONG_CONTEXT_DISPLAY,
        "long_context_model_loaded": False,
        "gpu_names": get_gpu_info(),
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
