"""
Nyaymalaw API — top-level application entry point.

Route handlers live in api/ sub-modules:
  api/auth_routes.py       — /auth/*, /chats/*
  api/core_chat_routes.py  — /search, /upload-document, /submit_case, /interview_step,
                             /conversation/continue, /agent/stream
  api/library_routes.py    — /bareacts/*, /caselaws/*, /library/*
  api/admin_routes.py      — /user/tier, /admin/*, /propose_index, /eval/*,
                             /docs/*, /feedback/*, /feedback_log/*, /health
"""
import asyncio
import logging
import os
import time
import traceback
from collections import defaultdict
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nyaymalaw.api")

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Nyaymalaw API", version="3.0.0")

_cors_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-IP request rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------
from api.deps import _rate_store  # shared so /admin/reset-rate-limit can clear it

_RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
_RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "100"))
_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"

_RATE_LIMITED_PATHS = {
    "/submit_case", "/submit_case/stream",
    "/interview_step", "/interview_step/stream",
    "/conversation/continue", "/conversation/continue/stream",
    "/chat", "/search",
    "/agent/stream",
    "/upload-document",
}
_SKIP_RATE_LIMIT_IPS = {"127.0.0.1", "localhost", "::1"}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not _RATE_LIMIT_ENABLED:
        return await call_next(request)
    if request.method in ("POST", "PUT", "PATCH") and request.url.path in _RATE_LIMITED_PATHS:
        client_ip = request.client.host if request.client else "unknown"
        if client_ip not in _SKIP_RATE_LIMIT_IPS:
            now = time.time()
            window_start = now - _RATE_LIMIT_WINDOW
            _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window_start]
            if len(_rate_store[client_ip]) >= _RATE_LIMIT_MAX:
                logger.warning("Rate limit hit for IP %s on %s", client_ip, request.url.path)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please slow down and try again in a minute."},
                )
            _rate_store[client_ip].append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    method = request.method
    path = request.url.path
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s → %d (%.0fms)", method, path, response.status_code, elapsed_ms)
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("%s %s → 500 (%.0fms) %s", method, path, elapsed_ms, exc)
        raise


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# DB init + tier migration (run at import time, before any request)
# ---------------------------------------------------------------------------
from api.deps import _init_auth_db
_init_auth_db()

from platform_pkg.tiers import ensure_tier_columns
ensure_tier_columns()

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
from api.auth_routes import router as _auth_router
from api.core_chat_routes import router as _core_router
from api.library_routes import router as _library_router
from api.admin_routes import router as _admin_router

app.include_router(_auth_router)
app.include_router(_core_router)
app.include_router(_library_router)
app.include_router(_admin_router)


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_validation():
    logger.info("=" * 60)
    logger.info("Nyaymalaw API v3.0.0 starting up")
    logger.info("=" * 60)

    from api.deps import _get_db, _DB_PATH
    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        logger.info("Database accessible at %s", _DB_PATH)
    except Exception as e:
        logger.warning("Database error: %s", e)

    from config import DATA_ROOT, BARE_ACTS_DIR, VECTOR_STORE, BARE_INDEX_V2, CASE_SUMMARY_INDEX_V2
    logger.info("Data root: %s (BareActs: %s)", DATA_ROOT, BARE_ACTS_DIR)
    if os.path.isdir(VECTOR_STORE):
        bare_ok = os.path.isfile(BARE_INDEX_V2)
        case_summary_ok = os.path.isfile(CASE_SUMMARY_INDEX_V2)
        logger.info(
            "Vector store at %s (bare_acts: %s, case_summaries: %s)",
            VECTOR_STORE,
            "yes" if bare_ok else "no",
            "yes" if case_summary_ok else "no",
        )
    else:
        logger.warning("Vector store directory not found: %s", VECTOR_STORE)

    logger.info("CORS origins: %s", _cors_origins)

    try:
        from platform_pkg.warmup import kickoff_runtime_warmup
        kickoff_runtime_warmup("startup_post_ready")
        logger.info("Startup checks complete; remaining warmups launched in background")
    except Exception as e:
        logger.warning("Background warmups could not be started: %s", e)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
