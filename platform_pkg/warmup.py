"""
platform/warmup.py — Runtime warmup orchestration.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_warmup_lock = threading.Lock()
_warmup_started = False
_warmup_completed = False
_warmup_error = ""
_warmup_started_at = 0.0
_warmup_finished_at = 0.0
_critical_warmup_completed = False
_ollama_warmup_started = False


def _set_status(started: bool | None = None, completed: bool | None = None, error: str | None = None):
    global _warmup_started, _warmup_completed, _warmup_error, _warmup_started_at, _warmup_finished_at
    if started is not None:
        _warmup_started = started
        if started and not _warmup_started_at:
            _warmup_started_at = time.perf_counter()
    if completed is not None:
        _warmup_completed = completed
        if completed:
            _warmup_finished_at = time.perf_counter()
    if error is not None:
        _warmup_error = error


def get_runtime_warmup_status() -> dict:
    elapsed = 0.0
    if _warmup_started_at:
        end = _warmup_finished_at or time.perf_counter()
        elapsed = max(0.0, end - _warmup_started_at)
    return {
        "started": _warmup_started,
        "completed": _warmup_completed,
        "error": _warmup_error,
        "elapsed_seconds": round(elapsed, 2),
        "critical_ready": _critical_warmup_completed,
    }


def run_critical_runtime_warmup(reason: str = "startup") -> dict:
    """
    Perform only small synchronous warmup steps that most directly affect the
    first live request.

    This keeps startup predictable while still shrinking first-turn latency.
    """
    global _critical_warmup_completed
    started_at = time.perf_counter()
    status = {
        "reason": reason,
        "fewshot_ready": False,
        "elapsed_seconds": 0.0,
    }

    with _warmup_lock:
        if _critical_warmup_completed:
            status["fewshot_ready"] = True
            status["elapsed_seconds"] = round(max(0.0, time.perf_counter() - started_at), 2)
            return status

        try:
            from platform_pkg.training.few_shot import preload_examples

            try:
                preload_examples()
                status["fewshot_ready"] = True
                logger.info("Critical runtime warmup: few-shot examples ready")
            except Exception as exc:
                logger.warning("Critical warmup could not preload few-shot examples: %s", exc)
            _critical_warmup_completed = bool(status["fewshot_ready"])
        finally:
            status["elapsed_seconds"] = round(max(0.0, time.perf_counter() - started_at), 2)

    return status


def _warmup_once():
    try:
        run_critical_runtime_warmup("background")
        from retrieval.retriever import (
            _get_cross_encoder_cpu,
            _get_cross_encoder_gpu,
            _get_embedder,
            preload_interactive_indexes,
            search_bare_acts_fast,
            search_case_summaries_fast,
        )
        try:
            preload_interactive_indexes()
        except Exception as e:
            logger.warning("Interactive index warmup failed: %s", e)

        try:
            _get_embedder()
            logger.info("Interactive warmup: embedding model ready")
        except Exception as e:
            logger.warning("Embedding warmup failed: %s", e)

        try:
            if _get_cross_encoder_gpu() is not None:
                logger.info("Interactive warmup: cross-encoder ready on CUDA")
            else:
                _get_cross_encoder_cpu()
                logger.info("Interactive warmup: cross-encoder ready on CPU")
        except Exception as e:
            logger.warning("Cross-encoder warmup failed: %s", e)

        try:
            search_bare_acts_fast("wrongful termination of employment", top_k=2)
            search_case_summaries_fast("disciplinary dismissal prejudice", top_k=2)
            logger.info("Interactive warmup: retrieval fast path primed")
        except Exception as e:
            logger.warning("Retrieval fast-path warmup failed: %s", e)

        _set_status(completed=True, error="")
        logger.info("Runtime warmup completed in %.1f s", time.perf_counter() - _warmup_started_at)
    except Exception as e:
        logger.exception("Runtime warmup crashed")
        _set_status(completed=False, error=str(e))


def kickoff_runtime_warmup(reason: str = "runtime") -> bool:
    """
    Start the interactive runtime warmup in a background thread once.

    Returns True when a new thread was started, False when warmup was already
    running or finished.
    """
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return False
        _set_status(started=True, completed=False, error="")
        thread = threading.Thread(
            target=_warmup_once,
            name=f"nyaymalaw-runtime-warmup-{reason}",
            daemon=True,
        )
        thread.start()
        logger.info("Started runtime warmup thread (%s)", reason)
        return True


def kickoff_ollama_warmup_if_qwen(model_override: str | None = None, reason: str = "") -> bool:
    """
    Stub — Ollama removed.
    Returns False immediately for backward compatibility.
    """
    return False
