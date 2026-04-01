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
import os
import subprocess
import threading
import time
import contextvars
from contextlib import contextmanager
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from openai import OpenAI
import tiktoken

from llm.config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_MODEL_FAST,
    OLLAMA_MODEL_LONG_CONTEXT,
    LONG_CONTEXT_THRESHOLD,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT_FAST_SEC,
    OLLAMA_TIMEOUT_DEFAULT_SEC,
    OLLAMA_TIMEOUT_LONG_SEC,
    OLLAMA_RETRIES_FAST,
    OLLAMA_RETRIES_DEFAULT,
    OLLAMA_RETRIES_LONG,
    OLLAMA_THINK_ENABLED,
    OLLAMA_MODEL_DISPLAY,
    OLLAMA_MODEL_FAST_DISPLAY,
    OLLAMA_MODEL_LONG_CONTEXT_DISPLAY,
    OPENAI_MODEL,
    OPENAI_MODEL_FAST,
    OPENAI_MODEL_LONG_CONTEXT,
    OPENAI_MODEL_DISPLAY,
    OPENAI_MODEL_FAST_DISPLAY,
    OPENAI_MODEL_LONG_CONTEXT_DISPLAY,
)

logger = logging.getLogger(__name__)
OPENAI_ANALYSIS_SWITCH_INPUT_TOKENS = 6000
OPENAI_INPUT_TOKEN_LIMIT = 10000
OPENAI_OUTPUT_TOKEN_LIMIT = 5000
OPENAI_FAST_INPUT_TOKEN_LIMIT = 5000
OPENAI_FAST_OUTPUT_TOKEN_LIMIT = 2000
OPENAI_HOURLY_INPUT_TOKEN_LIMIT = 500000
OPENAI_HOURLY_OUTPUT_TOKEN_LIMIT = 5000

_session_lock = threading.Lock()
_session = None

# Thread-local: model name used by the last ask_llm call in this thread (for API to include in response)
_last_model_used = threading.local()
_openai_client = None
_openai_budget_lock = threading.Lock()
_openai_budget_window_start = datetime.utcnow()
_openai_hourly_input_used = 0
_openai_hourly_output_used = 0
_request_model_override = contextvars.ContextVar("request_model_override", default=None)


def _is_openai_provider() -> bool:
    return LLM_PROVIDER == "openai"


def _is_openai_request(explicit_model: str | None = None, chosen_model: str | None = None) -> bool:
    if _is_openai_provider():
        return True
    explicit = (explicit_model or "").strip().lower()
    if explicit in ("provider:openai", "openai"):
        return True
    return chosen_model in {OPENAI_MODEL, OPENAI_MODEL_FAST, OPENAI_MODEL_LONG_CONTEXT}


def _is_openai_forced(explicit_model: str | None = None, chosen_model: str | None = None) -> bool:
    """
    True when this request is explicitly pinned to OpenAI (via provider override
    or OpenAI model selection), so local-model fallback must never be used.
    """
    explicit = (explicit_model or "").strip().lower()
    if explicit in ("provider:openai", "openai"):
        return True
    return chosen_model in {OPENAI_MODEL, OPENAI_MODEL_FAST, OPENAI_MODEL_LONG_CONTEXT}


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set while LLM_PROVIDER=openai")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _trim_prompt_for_context(prompt: str) -> str:
    """
    Trim prompt to LONG_CONTEXT_THRESHOLD chars so we always stay on the
    default model instead of switching to a separate long-context model.
    """
    p = prompt or ""
    if len(p) <= LONG_CONTEXT_THRESHOLD:
        return p
    logger.info(
        "Prompt length %d exceeds %d; trimming for default model",
        len(p), LONG_CONTEXT_THRESHOLD,
    )
    return p[:LONG_CONTEXT_THRESHOLD]


def _count_tokens(text: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name or OPENAI_MODEL)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text or ""))


def _openai_limits(chosen_model: str, task_hint: str | None) -> tuple[int, int]:
    if task_hint == "fast" or chosen_model == OPENAI_MODEL_FAST:
        return OPENAI_FAST_INPUT_TOKEN_LIMIT, OPENAI_FAST_OUTPUT_TOKEN_LIMIT
    return OPENAI_INPUT_TOKEN_LIMIT, OPENAI_OUTPUT_TOKEN_LIMIT


def _truncate_openai_input(prompt: str, chosen_model: str, task_hint: str | None) -> str:
    p = prompt or ""
    input_limit, _ = _openai_limits(chosen_model, task_hint)
    try:
        enc = tiktoken.encoding_for_model(chosen_model or OPENAI_MODEL)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(p)
    if len(tokens) <= input_limit:
        return p
    logger.warning("OpenAI input exceeds hard limit (%d > %d); truncating", len(tokens), input_limit)
    return enc.decode(tokens[:input_limit])


def _reset_openai_budget_window_if_needed() -> None:
    global _openai_budget_window_start, _openai_hourly_input_used, _openai_hourly_output_used
    now = datetime.utcnow()
    if now - _openai_budget_window_start >= timedelta(hours=1):
        _openai_budget_window_start = now
        _openai_hourly_input_used = 0
        _openai_hourly_output_used = 0


def _allow_openai_budget(input_tokens: int, reserved_output_tokens: int) -> bool:
    global _openai_hourly_input_used, _openai_hourly_output_used
    with _openai_budget_lock:
        _reset_openai_budget_window_if_needed()
        next_input = _openai_hourly_input_used + max(0, input_tokens)
        next_output = _openai_hourly_output_used + max(0, reserved_output_tokens)
        if next_input > OPENAI_HOURLY_INPUT_TOKEN_LIMIT or next_output > OPENAI_HOURLY_OUTPUT_TOKEN_LIMIT:
            return False
        _openai_hourly_input_used = next_input
        _openai_hourly_output_used = next_output
        return True


def _refund_openai_output_budget(reserved: int, actual: int) -> None:
    global _openai_hourly_output_used
    refund = max(0, reserved - max(0, actual))
    if refund <= 0:
        return
    with _openai_budget_lock:
        _reset_openai_budget_window_if_needed()
        _openai_hourly_output_used = max(0, _openai_hourly_output_used - refund)


def _get_model_for_prompt(
    prompt: str,
    explicit_model: str = None,
    task_hint: str = None,
) -> str:
    """
    Decide which Ollama model to use for this request. Priority order:

    1. explicit_model — if the caller passed a model name, use it.
    2. task_hint == "fast" — use the fast intake model when configured.
    3. Otherwise — use default model (e.g. Qwen 3 8B).

    Long prompts are trimmed to LONG_CONTEXT_THRESHOLD before request dispatch,
    so we stay on a single default model for both normal and long requests.
    """
    # Request-scoped override (set by API flow) so helper calls without explicit_model
    # still follow the frontend-selected provider/model.
    if not explicit_model:
        explicit_model = _request_model_override.get()

    if explicit_model:
        normalized = explicit_model.strip().lower()
        if normalized in ("provider:openai", "openai"):
            if task_hint == "fast":
                return OPENAI_MODEL_FAST
            if task_hint == "long_context":
                return OPENAI_MODEL_LONG_CONTEXT
            input_tokens = _count_tokens(prompt, OPENAI_MODEL)
            return OPENAI_MODEL if input_tokens > OPENAI_ANALYSIS_SWITCH_INPUT_TOKENS else OPENAI_MODEL_FAST
        if normalized in ("provider:qwen", "qwen", "default"):
            if task_hint == "fast":
                return OLLAMA_MODEL_FAST
            return OLLAMA_MODEL
        return explicit_model
    if _is_openai_provider():
        if task_hint == "fast":
            return OPENAI_MODEL_FAST
        if task_hint == "long_context":
            return OPENAI_MODEL_LONG_CONTEXT
        input_tokens = _count_tokens(prompt, OPENAI_MODEL)
        return OPENAI_MODEL if input_tokens > OPENAI_ANALYSIS_SWITCH_INPUT_TOKENS else OPENAI_MODEL_FAST
    if task_hint == "fast":
        return OLLAMA_MODEL_FAST
    return OLLAMA_MODEL


@contextmanager
def use_request_model_override(model_override: str | None):
    """
    Request-scoped model override for all ask_llm() calls in the current context.
    Useful when helper functions don't pass explicit_model but should follow
    frontend-selected provider/model (e.g. OpenAI).
    """
    token = _request_model_override.set(model_override)
    try:
        yield
    finally:
        _request_model_override.reset(token)


def set_request_model_override(model_override: str | None) -> None:
    """
    Set request-scoped model override for current context.
    Next call can overwrite/clear by passing another value (including None).
    """
    _request_model_override.set(model_override)


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
    if _is_openai_provider():
        if OPENAI_MODEL in model_name or model_name in OPENAI_MODEL:
            return OPENAI_MODEL_DISPLAY
        if OPENAI_MODEL_FAST in model_name or model_name in OPENAI_MODEL_FAST:
            return OPENAI_MODEL_FAST_DISPLAY
        if OPENAI_MODEL_LONG_CONTEXT in model_name or model_name in OPENAI_MODEL_LONG_CONTEXT:
            return OPENAI_MODEL_LONG_CONTEXT_DISPLAY
        return model_name
    if OLLAMA_MODEL in model_name or model_name in OLLAMA_MODEL:
        return OLLAMA_MODEL_DISPLAY
    if OLLAMA_MODEL_FAST in model_name or model_name in OLLAMA_MODEL_FAST:
        return OLLAMA_MODEL_FAST_DISPLAY
    if OLLAMA_MODEL_LONG_CONTEXT in model_name or model_name in OLLAMA_MODEL_LONG_CONTEXT:
        return OLLAMA_MODEL_LONG_CONTEXT_DISPLAY
    return model_name


def get_model_display_for_prompt(
    prompt: str,
    task_hint: str = None,
    explicit_model: str = None,
) -> tuple:
    """
    Return (display_name, is_switched) for the model that would be used for this prompt.
    display_name: e.g. 'Qwen 2.5 7B' or 'Llama 3.1 8B'.
    is_switched: True when long-context model is used (so UI can show "Switching to X model").
    """
    chosen = _get_model_for_prompt(prompt, explicit_model, task_hint)
    display = get_display_name_for_model(chosen)
    is_switched = False
    return (display, is_switched)


def get_last_model_used() -> str:
    """Display name of the model used by the last ask_llm call in this thread. Empty if none."""
    try:
        name = getattr(_last_model_used, "value", None)
        return get_display_name_for_model(name) if name else ""
    except Exception:
        return ""

OLLAMA_BASE_URL = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

RETRY_BACKOFF_BASE = 2  # seconds; doubles each retry (2s, 4s)


def _get_http_session() -> requests.Session:
    """Reuse HTTP connections to Ollama so repeated local calls stay cheap."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                _session = session
    return _session


def _resolve_timeout_and_retries(
    prompt: str,
    chosen_model: str,
    task_hint: str = None,
    timeout: int = None,
) -> tuple[int, int]:
    """Choose task-appropriate timeout and retry policy."""
    if timeout is not None:
        # If caller overrides timeout, keep retries conservative for fast tasks.
        retries = OLLAMA_RETRIES_FAST if task_hint == "fast" else OLLAMA_RETRIES_DEFAULT
        return timeout, retries
    if _is_openai_provider():
        if task_hint == "fast" or chosen_model == OPENAI_MODEL_FAST:
            return OLLAMA_TIMEOUT_FAST_SEC, OLLAMA_RETRIES_FAST
        if task_hint == "long_context" or chosen_model == OPENAI_MODEL_LONG_CONTEXT:
            return OLLAMA_TIMEOUT_LONG_SEC, OLLAMA_RETRIES_LONG
        return OLLAMA_TIMEOUT_DEFAULT_SEC, OLLAMA_RETRIES_DEFAULT
    if task_hint == "fast" or chosen_model == OLLAMA_MODEL_FAST:
        return OLLAMA_TIMEOUT_FAST_SEC, OLLAMA_RETRIES_FAST
    return OLLAMA_TIMEOUT_DEFAULT_SEC, OLLAMA_RETRIES_DEFAULT


def _extract_text(data: dict) -> str:
    """Get generated text from Ollama response. Handles different response shapes."""
    if not isinstance(data, dict):
        return ""
    for key in ("response", "message", "content", "text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_openai_text(resp) -> str:
    """
    Robustly extract text from OpenAI Responses API objects.
    Handles both output_text convenience field and nested output content parts.
    """
    def _read(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _stringify_text_like(value) -> str:
        """
        Normalize SDK text payloads into a plain string.
        Handles plain strings and structured text objects/dicts
        that store actual text under keys like 'value' or 'text'.
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "value", "output_text", "content"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, list):
                    joined = " ".join(
                        str(item).strip()
                        for item in v
                        if isinstance(item, str) and item.strip()
                    ).strip()
                    if joined:
                        return joined
        return ""

    out = _stringify_text_like(_read(resp, "output_text", ""))
    if out:
        return out

    # Some SDK responses expose dict-like data via model_dump()
    payload = resp
    try:
        if hasattr(resp, "model_dump"):
            payload = resp.model_dump()
    except Exception:
        payload = resp

    parts: list[str] = []
    for item in (_read(payload, "output", []) or []):
        # Some responses may carry only reasoning summaries with no message/output_text.
        if (_read(item, "type", "") or "").strip().lower() == "reasoning":
            for s in (_read(item, "summary", []) or []):
                s_txt = _stringify_text_like(_read(s, "text", None))
                if s_txt:
                    parts.append(s_txt)

        for content in (_read(item, "content", []) or []):
            text_val = _stringify_text_like(_read(content, "text", None))
            if text_val:
                parts.append(text_val)
            # Chat-style content entries sometimes come as {"type":"output_text","text":"..."}
            if isinstance(content, dict):
                maybe_text = _stringify_text_like(content.get("text") or content.get("output_text"))
                if maybe_text:
                    parts.append(maybe_text)
                # Some shapes include refusal text instead of output_text
                refusal_text = _stringify_text_like(content.get("refusal"))
                if refusal_text:
                    parts.append(refusal_text)

    # Fallback for chat-completions-like shape
    if not parts:
        choices = _read(payload, "choices", []) or []
        for ch in choices:
            msg = _read(ch, "message", {}) or {}
            content = _read(msg, "content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            elif isinstance(content, list):
                for block in content:
                    txt = _stringify_text_like(_read(block, "text", None))
                    if txt:
                        parts.append(txt)

    if not parts:
        # Diagnostic metadata only; avoids leaking prompt/response body content.
        try:
            output_items = _read(payload, "output", []) or []
            output_types = []
            for item in output_items:
                item_type = _read(item, "type", None)
                if item_type:
                    output_types.append(str(item_type))
            logger.warning(
                "OpenAI response had no extractable text; top_keys=%s output_items=%d output_types=%s",
                list(payload.keys())[:12] if isinstance(payload, dict) else type(payload).__name__,
                len(output_items),
                output_types[:8],
            )
        except Exception:
            pass

    return "\n".join(parts).strip()


import re as _re
_THINK_BLOCK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """
    Remove Qwen 3 chain-of-thought blocks from the model output.

    Qwen 3 in thinking mode wraps its reasoning in <think>...</think> tags
    before the actual answer. These blocks are internal reasoning — they must
    be stripped before the text is used as a reply or passed to the quality gate.

    Ollama may strip these automatically when think=True is set, but we strip
    defensively in case the raw tokens leak through (e.g. on older Ollama builds).
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def _resolve_think(task_hint: str) -> bool | None:
    """
    Return the think flag to send to Ollama, or None to omit it entirely.

    Rules:
    - OLLAMA_THINK_ENABLED=false → None (never send think param; safe for non-Qwen3)
    - task_hint="fast"           → False (no-think: structured output, low latency)
    - everything else            → True  (thinking mode: chain-of-thought reasoning)
    """
    if not OLLAMA_THINK_ENABLED:
        return None
    return task_hint != "fast"


def warmup_ollama_model(
    model: str = None,
    timeout: int = 180,
) -> bool:
    """
    Warm the configured Ollama model so the first real request avoids model-load latency.
    Returns True on success, False on failure.
    """
    chosen = model or OLLAMA_MODEL
    payload = {
        "model": chosen,
        "prompt": "Reply with exactly: OK",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    try:
        response = _get_http_session().post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        if not response.ok:
            logger.warning(
                "Ollama warmup failed for %s: HTTP %s %s",
                chosen,
                response.status_code,
                (response.text or "").strip()[:200],
            )
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("Ollama warmup failed for %s: %s", chosen, exc)
        return False


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
    requested_model = model
    chosen = _get_model_for_prompt(prompt, requested_model, task_hint)
    use_openai = _is_openai_request(requested_model, chosen)
    try:
        _last_model_used.value = chosen
    except Exception:
        pass
    timeout, _ = _resolve_timeout_and_retries(prompt, chosen, task_hint, timeout)
    if use_openai:
        text = ask_llm(prompt, model=chosen, timeout=timeout, task_hint=task_hint)
        for i in range(0, len(text), 24):
            yield text[i:i + 24]
        return
    trimmed_prompt = _trim_prompt_for_context(prompt)
    payload = {
        "model": chosen,
        "prompt": trimmed_prompt,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    think = _resolve_think(task_hint)
    if think is not None:
        payload["think"] = think
    try:
        with _get_http_session().post(
            OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=timeout
        ) as response:
            if not response.ok:
                err = response.text or "Unknown error"
                raise RuntimeError(f"Ollama error ({response.status_code}): {err}")
            in_think_block = False
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            # Track <think>...</think> blocks inline so we never
                            # stream reasoning tokens to the UI.
                            if "<think>" in token:
                                in_think_block = True
                            if in_think_block:
                                if "</think>" in token:
                                    in_think_block = False
                            else:
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
    requested_model = model
    chosen = _get_model_for_prompt(prompt, requested_model, task_hint)
    use_openai = _is_openai_request(requested_model, chosen)
    openai_forced = _is_openai_forced(requested_model, chosen)
    trimmed_prompt = _trim_prompt_for_context(prompt)
    try:
        _last_model_used.value = chosen
    except Exception:
        pass
    model = chosen
    if use_openai:
        prompt = _truncate_openai_input(prompt, chosen, task_hint)
        _, max_output_tokens = _openai_limits(chosen, task_hint)
        input_tokens = _count_tokens(prompt, chosen)
        if not _allow_openai_budget(input_tokens, max_output_tokens):
            if openai_forced:
                logger.error("OpenAI hourly limit breached; OpenAI is forced for this request, skipping local fallback")
                raise RuntimeError("OpenAI hourly budget limit reached for this request")
            logger.warning("OpenAI hourly limit breached; falling back to Qwen")
            chosen = _get_model_for_prompt(prompt, "provider:qwen", task_hint)
            model = chosen
            use_openai = False
        else:
            timeout, max_retries = _resolve_timeout_and_retries(prompt, chosen, task_hint, timeout)
            last_exc = None
            for attempt in range(1 + max_retries):
                try:
                    request_kwargs = {
                        "model": model,
                        "input": prompt,
                        "timeout": timeout,
                        "max_output_tokens": max_output_tokens,
                        # Force plain-text response mode to avoid reasoning-only payloads.
                        "text": {"format": {"type": "text"}},
                    }
                    resp = _get_openai_client().responses.create(
                        **request_kwargs
                    )
                    out = _extract_openai_text(resp)
                    if not out:
                        # If no message text came back, make one immediate retry with
                        # explicit final-answer instruction before failing this attempt.
                        retry_kwargs = dict(request_kwargs)
                        retry_kwargs["input"] = (
                            f"{prompt}\n\n"
                            "IMPORTANT: Return only the final answer text. "
                            "Do not return reasoning metadata."
                        )
                        resp_retry = _get_openai_client().responses.create(**retry_kwargs)
                        out = _extract_openai_text(resp_retry)
                    _refund_openai_output_budget(max_output_tokens, _count_tokens(out, chosen) if out else 0)
                    if out:
                        return out
                    raise RuntimeError("Empty response from OpenAI API")
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "OpenAI request failed (attempt %d/%d), retrying in %ds: %s",
                            attempt + 1, 1 + max_retries, wait, exc,
                        )
                        time.sleep(wait)
                    else:
                        _refund_openai_output_budget(max_output_tokens, 0)
                        logger.error(
                            "OpenAI request failed after %d attempts: %s",
                            1 + max_retries, exc,
                        )
            raise RuntimeError(f"OpenAI request failed: {last_exc}")
    timeout, max_retries = _resolve_timeout_and_retries(trimmed_prompt, chosen, task_hint, timeout)
    payload = {
        "model": model,
        "prompt": trimmed_prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    think = _resolve_think(task_hint)
    if think is not None:
        payload["think"] = think

    for attempt in range(1 + max_retries):
        try:
            response = _get_http_session().post(
                OLLAMA_GENERATE_URL, json=payload, timeout=timeout
            )
            break  # success — exit retry loop
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Ollama request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, 1 + max_retries, wait, e,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Ollama request failed after %d attempts: %s",
                    1 + max_retries, e,
                )
                raise RuntimeError(
                    f"Ollama unreachable after {1 + max_retries} attempts: {e}"
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

    return _strip_think_blocks(_extract_text(data) or "")


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
    if _is_openai_provider():
        result.update(
            {
                "ollama_reachable": True,
                "model_loaded": True,
                "model": OPENAI_MODEL,
                "model_display": OPENAI_MODEL_DISPLAY,
                "long_context_model": OPENAI_MODEL_LONG_CONTEXT,
                "long_context_model_display": OPENAI_MODEL_LONG_CONTEXT_DISPLAY,
                "long_context_model_loaded": True,
                "provider": "openai",
            }
        )
        return result

    # 1. Check if Ollama is reachable
    try:
        resp = _get_http_session().get(OLLAMA_BASE_URL, timeout=5)
        result["ollama_reachable"] = resp.ok
    except requests.RequestException as e:
        result["error"] = f"Ollama not reachable: {e}"
        return result

    # 2. Check if both models are available
    try:
        resp = _get_http_session().get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
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
