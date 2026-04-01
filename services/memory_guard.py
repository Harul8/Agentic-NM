"""
Memory guard for runtime activity orchestration.

Goal:
- Log memory snapshots before heavy activities.
- Predict low-memory/OOM risk from host memory + process RSS.
- Serialize heavy activities under low-memory pressure until memory recovers.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_activity_lock = threading.Lock()

_MIN_AVAILABLE_MB = int(os.environ.get("MEM_GUARD_MIN_AVAILABLE_MB", "1400"))
_MIN_SWAP_FREE_MB = int(os.environ.get("MEM_GUARD_MIN_SWAP_FREE_MB", "256"))
_MAX_RSS_MB = int(os.environ.get("MEM_GUARD_MAX_RSS_MB", "9000"))
_WAIT_TIMEOUT_SEC = int(os.environ.get("MEM_GUARD_WAIT_TIMEOUT_SEC", "90"))
_WAIT_STEP_SEC = float(os.environ.get("MEM_GUARD_WAIT_STEP_SEC", "1.5"))


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                # meminfo values are in kB
                out[key.strip()] = int(parts[0])
    except Exception:
        return {}
    return out


def _read_proc_rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return 0
    return 0


def memory_snapshot() -> dict[str, float]:
    mem = _read_meminfo()
    rss_kb = _read_proc_rss_kb()
    avail_kb = mem.get("MemAvailable", 0)
    swap_free_kb = mem.get("SwapFree", 0)
    return {
        "rss_mb": round(rss_kb / 1024.0, 1),
        "available_mb": round(avail_kb / 1024.0, 1),
        "swap_free_mb": round(swap_free_kb / 1024.0, 1),
    }


def _oom_risk(snapshot: dict[str, float]) -> bool:
    return (
        snapshot.get("available_mb", 0.0) < _MIN_AVAILABLE_MB
        or snapshot.get("swap_free_mb", 0.0) < _MIN_SWAP_FREE_MB
        or snapshot.get("rss_mb", 0.0) > _MAX_RSS_MB
    )


@contextlib.contextmanager
def guard_activity(activity_name: str):
    """
    Guard a heavy activity.

    - Always logs a memory snapshot before activity.
    - If memory risk is high, serializes activity and waits for recovery.
    """
    start = time.perf_counter()
    snap = memory_snapshot()
    logger.info(
        "MEM_GUARD before %s | rss=%.1fMB avail=%.1fMB swap_free=%.1fMB",
        activity_name, snap["rss_mb"], snap["available_mb"], snap["swap_free_mb"],
    )

    lock_acquired = False
    if _oom_risk(snap):
        logger.warning(
            "MEM_GUARD risk detected before %s; serializing heavy activity "
            "(min_avail=%dMB min_swap=%dMB max_rss=%dMB)",
            activity_name, _MIN_AVAILABLE_MB, _MIN_SWAP_FREE_MB, _MAX_RSS_MB,
        )
        _activity_lock.acquire()
        lock_acquired = True
        deadline = time.perf_counter() + _WAIT_TIMEOUT_SEC
        while time.perf_counter() < deadline:
            cur = memory_snapshot()
            if not _oom_risk(cur):
                break
            time.sleep(_WAIT_STEP_SEC)
        post_wait = memory_snapshot()
        logger.info(
            "MEM_GUARD resume %s | rss=%.1fMB avail=%.1fMB swap_free=%.1fMB",
            activity_name, post_wait["rss_mb"], post_wait["available_mb"], post_wait["swap_free_mb"],
        )

    try:
        yield
    finally:
        if lock_acquired:
            _activity_lock.release()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        end = memory_snapshot()
        logger.info(
            "MEM_GUARD after %s | rss=%.1fMB avail=%.1fMB swap_free=%.1fMB elapsed=%.0fms",
            activity_name, end["rss_mb"], end["available_mb"], end["swap_free_mb"], elapsed_ms,
        )

