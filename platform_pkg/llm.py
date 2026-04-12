"""
core/llm.py — LLM configuration + OpenAI client.

Single module for all LLM operations. Merged from llm/config.py + llm/ollama_client.py.
Import as:  from platform.llm import ask_llm, check_ollama_health, ...
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


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

import json
import logging
import os
import subprocess
import threading
import time
import contextvars
from contextlib import contextmanager
from datetime import datetime, timedelta

from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing — enabled when LANGCHAIN_TRACING_V2=true is set in .env
# Must be initialised before any LangChain / LangGraph objects are created.
# ---------------------------------------------------------------------------
def _init_langsmith() -> None:
    from config import (
        LANGSMITH_TRACING,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        LANGSMITH_ENDPOINT,
    )
    if not LANGSMITH_TRACING:
        return
    if not LANGSMITH_API_KEY:
        logger.warning("LangSmith tracing enabled but LANGCHAIN_API_KEY is not set — skipping.")
        return
    # Propagate env vars so the LangSmith SDK picks them up automatically.
    os.environ.setdefault("LANGCHAIN_TRACING_V2",  "true")
    os.environ.setdefault("LANGCHAIN_API_KEY",      LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT",      LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_ENDPOINT",     LANGSMITH_ENDPOINT)
    logger.info("LangSmith tracing enabled — project: %s", LANGSMITH_PROJECT)

_init_langsmith()
# Switch from fast→analysis model when input exceeds this (tokens).
# Sits above the fast-call ceiling so intake calls always stay on the fast model.
OPENAI_ANALYSIS_SWITCH_INPUT_TOKENS = 10000

# ---------------------------------------------------------------------------
# Per-call token limits — derived from worst-case content budget:
#
#   Quality calls (draft generation):
#     Retrieval content  : 3 bare-act sections × 5 disputes × ~430 tok  = 6 430 tok
#                        + 3 case laws         × 5 disputes × ~430 tok  = 6 430 tok
#     Prompt overhead    : system prompt + facts summary + labels        ≈ 1 500 tok
#     Total input ceiling                                               ≈ 14 360 tok
#     → OPENAI_INPUT_TOKEN_LIMIT = 25 000  (≈74 % buffer — generous headroom for
#       longer system prompts, full conversation history, and multi-stage intake state)
#
#     Output             : full structured draft across 5 disputes       ≈ 3 000–4 500 tok
#     → OPENAI_OUTPUT_TOKEN_LIMIT = 7 000  (≈56 % buffer)
#
#   Fast calls (intake / vetting / category detection):
#     Input              : system prompt + conv tail + intake-state JSON ≈ 1 500–2 000 tok
#     → OPENAI_FAST_INPUT_TOKEN_LIMIT = 5 000  (≈2.5× buffer — covers longer
#       structured fact objects and pre-draft summary prompts)
#
#     Output             : next question / JSON response                 ≈ 80–150 tok
#     → OPENAI_FAST_OUTPUT_TOKEN_LIMIT = 4 000
#       NOTE: for Chat Completions, max_completion_tokens is a COMBINED budget for
#       internal chain-of-thought reasoning tokens AND output text tokens.
#       Reasoning models (gpt-5-nano, gpt-5-mini, o4-mini, etc.) consume ~500–2 000
#       reasoning tokens before producing any output.  Setting this too low (e.g. 800)
#       leaves the model no budget to write the actual response and returns empty content.
#       4 000 gives ~2 500–3 000 tok of reasoning headroom plus 500–1 000 tok for output
#       which is sufficient for all fast tasks (query expansion, intake, intent extraction).
#
#   Section/case char cap (enforced in response_generator_v2.py):
#     MAX_SECTIONS_PER_DISPUTE_FOR_OPINION = 3, MAX_CASE_LAWS_PER_DISPUTE = 3
#     Each chunk truncated to 1 500 chars before being placed in prompt.
# ---------------------------------------------------------------------------
OPENAI_INPUT_TOKEN_LIMIT = 25000
OPENAI_OUTPUT_TOKEN_LIMIT = 7000
OPENAI_FAST_INPUT_TOKEN_LIMIT = 5000
OPENAI_FAST_OUTPUT_TOKEN_LIMIT = 4000
# Hourly budget — 500 k input / 500 k output supports ~30+ full sessions/hr
OPENAI_HOURLY_INPUT_TOKEN_LIMIT = 500000
OPENAI_HOURLY_OUTPUT_TOKEN_LIMIT = 100000

_openai_client = None
_openai_budget_lock = threading.Lock()
_openai_budget_window_start = datetime.utcnow()
_openai_hourly_input_used = 0
_openai_hourly_output_used = 0
_request_model_override = contextvars.ContextVar("request_model_override", default=None)

# Thread-local: model name used by the last ask_llm call in this thread (for API to include in response)
_last_model_used = threading.local()

RETRY_BACKOFF_BASE = 2  # seconds; doubles each retry (2s, 4s)


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


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


def _get_model_for_prompt(prompt: str, explicit_model: str = None, task_hint: str = None) -> str:
    """
    Decide which OpenAI model to use for this request. Priority order:

    1. explicit_model — if the caller passed a model/tier name, use it.
    2. task_hint == "fast" — use the fast tier model when configured.
    3. Otherwise — use default tier model.

    Resolves tier keys (e.g. "tier:gpt5mini") to actual model names.
    """
    if not explicit_model:
        explicit_model = _request_model_override.get()

    # Resolve tier key from override string (e.g. "tier:gpt5mini" → "gpt5mini")
    tier_key = None
    if explicit_model:
        s = explicit_model.strip().lower()
        if s.startswith("tier:"):
            tier_key = s[5:]
        elif s in MODEL_TIERS:
            tier_key = s
        # Legacy provider strings — default tier
        elif s in ("provider:openai", "openai"):
            tier_key = DEFAULT_TIER

    tier = get_tier(tier_key)

    if task_hint == "fast":
        return tier["fast"]
    if task_hint == "long_context":
        return tier["large_context"]

    # Auto-switch to regular model for larger inputs
    input_tokens = _count_tokens(prompt, tier["regular"])
    return tier["regular"] if input_tokens > OPENAI_ANALYSIS_SWITCH_INPUT_TOKENS else tier["fast"]


@contextmanager
def use_request_model_override(model_override: str | None):
    """
    Request-scoped model override for all ask_llm() calls in the current context.
    Useful when helper functions don't pass explicit_model but should follow
    frontend-selected tier (e.g. gpt51mini).
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
    """Map internal model name to UI-friendly tier display name."""
    if not model_name:
        return ""
    for tier in MODEL_TIERS.values():
        if model_name in (tier["fast"], tier["regular"], tier["large_context"]):
            return tier["display"]
    return model_name


def get_model_display_for_prompt(
    prompt: str,
    task_hint: str = None,
    explicit_model: str = None,
) -> tuple:
    """
    Return (display_name, is_switched) for the model that would be used for this prompt.
    display_name: e.g. 'GPT-5 Mini'
    is_switched: False (always, since we don't have a separate long-context model anymore)
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


def _resolve_timeout_and_retries(prompt: str, chosen_model: str, task_hint: str = None, timeout: int = None) -> tuple[int, int]:
    """Choose task-appropriate timeout and retry policy."""
    if timeout is not None:
        retries = OLLAMA_RETRIES_FAST if task_hint == "fast" else OLLAMA_RETRIES_DEFAULT
        return timeout, retries
    if task_hint == "fast":
        return OLLAMA_TIMEOUT_FAST_SEC, OLLAMA_RETRIES_FAST
    if task_hint == "long_context":
        return OLLAMA_TIMEOUT_LONG_SEC, OLLAMA_RETRIES_LONG
    return OLLAMA_TIMEOUT_DEFAULT_SEC, OLLAMA_RETRIES_DEFAULT


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
            # Check for content_filter finish_reason in choices — distinct from other empty responses
            choices_raw = _read(payload, "choices", []) or []
            for ch in choices_raw:
                fr = _read(ch, "finish_reason", None)
                if fr == "content_filter":
                    logger.warning(
                        "OpenAI content_filter: server-side safety policy blocked the response. "
                        "Consider adding professional framing to the system prompt or paraphrasing "
                        "sensitive client-reported language before sending."
                    )
                    return ""

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


def ocr_pages_with_vision(images_b64: list[tuple[str, str]]) -> str:
    """
    Send base64-encoded images to OpenAI vision API and return combined OCR text.

    Args:
        images_b64: List of (base64_data, media_type) tuples, where media_type is
                    e.g. "image/jpeg", "image/png".

    Returns:
        Combined OCR text from all images.
    """
    if not images_b64:
        return ""

    client = _get_openai_client()
    content = [
        {
            "type": "text",
            "text": "Extract and return all text from this image. Preserve formatting if possible.",
        }
    ]
    for b64, media_type in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }
        )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2000,
        )
        return _extract_openai_text(resp)
    except Exception as e:
        logger.error("OCR vision API call failed: %s", e)
        raise RuntimeError(f"OCR failed: {e}") from e


def ask_llm_stream(
    prompt: str,
    model: str = None,
    timeout: int = None,
    task_hint: str = None,
):
    """
    Stream tokens from OpenAI one fragment at a time.

    Yields each text fragment as it is produced by the model.
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
    try:
        _last_model_used.value = chosen
    except Exception:
        pass

    # For now, call ask_llm and chunk the output since OpenAI streaming
    # integration can be added later if needed.
    text = ask_llm(prompt, model=chosen, timeout=timeout, task_hint=task_hint)
    for i in range(0, len(text), 24):
        yield text[i:i + 24]


# ---------------------------------------------------------------------------
# Default system prompt — injected into every ask_llm call unless the caller
# provides its own system message.
#
# Purpose: give OpenAI's moderation layer the professional context it needs
# to correctly classify Indian legal queries.  Without this, queries that
# describe crimes (even from a victim's perspective) can trigger a silent
# content_filter refusal (empty choices[]).
# ---------------------------------------------------------------------------
_LEGAL_SYSTEM_PROMPT = (
    "You are an AI legal research assistant integrated into NyaymalaW, a professional "
    "Indian legal research platform used by advocates, law students, and litigants.\n\n"
    "All queries are processed in the context of Indian law (IPC, BNS, CrPC, BNSS, CPC, "
    "constitutional law, personal law, property law, etc.).  Users may be:\n"
    "  • Victims describing abuse, harassment, fraud, or violence they have experienced\n"
    "  • Advocates researching legal provisions, precedents, or penalties on behalf of clients\n"
    "  • Law students studying criminal or civil law\n"
    "  • Litigants seeking to understand their rights\n\n"
    "Descriptions of crimes, punishments, or sensitive legal scenarios are shared for the "
    "purpose of legal research and victim assistance — NOT to facilitate harm.  Respond "
    "fully and professionally to all Indian legal queries, including those involving "
    "criminal law, sexual offences, domestic violence, drug offences, or terrorism law, "
    "as these are standard topics in legal practice.  Do not refuse or truncate responses "
    "to legitimate legal research questions."
)


def ask_llm(
    prompt: str,
    model: str = None,
    timeout: int = None,
    task_hint: str = None,
    system: str = None,
) -> str:
    """
    Send a prompt to OpenAI and return the generated text.

    If model is not specified: uses the default tier's regular model for normal prompts,
    and the fast model when task_hint is "fast". Retries on connection/timeout only.

    If system is provided it is sent as a system-role message before the user message.
    If system is None, the default _LEGAL_SYSTEM_PROMPT is used to give OpenAI's
    moderation layer proper professional context for Indian legal queries.
    """
    # Use caller-supplied system prompt if provided; otherwise apply the default
    # legal context prompt to prevent silent content_filter refusals on legitimate
    # Indian legal queries (descriptions of crimes, penalties, victim scenarios, etc.)
    if system is None:
        system = _LEGAL_SYSTEM_PROMPT
    requested_model = model
    chosen = _get_model_for_prompt(prompt, requested_model, task_hint)
    try:
        _last_model_used.value = chosen
    except Exception:
        pass

    prompt = _truncate_openai_input(prompt, chosen, task_hint)
    _, max_output_tokens = _openai_limits(chosen, task_hint)
    input_tokens = _count_tokens(prompt, chosen)

    if not _allow_openai_budget(input_tokens, max_output_tokens):
        logger.error("OpenAI hourly budget limit reached")
        raise RuntimeError("OpenAI hourly budget limit reached for this request")

    timeout, max_retries = _resolve_timeout_and_retries(prompt, chosen, task_hint, timeout)
    last_exc = None

    # Build message list: optional system role + user message.
    _messages: list[dict] = []
    if system:
        _messages.append({"role": "system", "content": system})
    _messages.append({"role": "user", "content": prompt})

    for attempt in range(1 + max_retries):
        try:
            # Use Chat Completions API — works with both reasoning models
            # (gpt-5-mini, gpt-5-nano, o4-mini, etc.) and standard models.
            resp = _get_openai_client().chat.completions.create(
                model=chosen,
                messages=_messages,
                max_completion_tokens=max_output_tokens,
                timeout=timeout,
            )
            out = _extract_openai_text(resp)
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


def check_ollama_health() -> dict:
    """
    Stub — Ollama removed. Returns OpenAI status for backward compatibility.

    Returns:
        {
            "ollama_reachable": bool (always True),
            "model_loaded": bool (always True),
            "model": str,
            "model_display": str,
            "long_context_model": str,
            "long_context_model_display": str,
            "long_context_model_loaded": bool (always True),
            "gpu_names": list[str],
            "provider": "openai",
        }
    """
    tier = get_tier(None)
    return {
        "ollama_reachable": True,
        "model_loaded": True,
        "model": tier["regular"],
        "model_display": tier["display"],
        "long_context_model": tier["large_context"],
        "long_context_model_display": tier["display"],
        "long_context_model_loaded": True,
        "provider": "openai",
        "gpu_names": get_gpu_info(),
    }


def warmup_ollama_model(model: str = None, timeout: int = 180) -> bool:
    """
    Stub — no warmup needed for OpenAI.
    Returns True immediately for backward compatibility.
    """
    return True
