import asyncio
import os
import json
import sqlite3
import hashlib
import bcrypt as _bcrypt
import logging
import secrets
import re
import time
import threading
import traceback
from html import escape as _html_escape
from queue import Queue, Empty
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Query, HTTPException, Depends, Header, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from retrieval.retriever import fuse_bare_act_and_case_law
from pipeline.chat import process_chat
try:
    from agents.intake.stage5_draft import build_legal_draft as _build_legal_draft
    _STAGE5_ENABLED = True
except Exception as _s5_err:
    _STAGE5_ENABLED = False
    _build_legal_draft = None
from retrieval.generator import generate_response_v2
from platform_pkg.feedback.store import (
    RESPONSE_FEEDBACK_TAGS,
    append_response_feedback,
    feedback_store_path,
)
from platform_pkg.llm import check_ollama_health, get_last_model_used

# Feedback logging (non-critical â€” import errors must not crash the server)
try:
    from platform_pkg.feedback.logger import log_interaction as _log_interaction
    from platform_pkg.feedback.reviewer import run_ai_review as _run_ai_review
    _FEEDBACK_ENABLED = True
except Exception as _fb_import_err:
    _FEEDBACK_ENABLED = False
    _log_interaction = None   # type: ignore[assignment]
    _run_ai_review   = None   # type: ignore[assignment]

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
_PIPELINE_TIMING = os.environ.get("PIPELINE_TIMING", "").lower() in ("1", "true", "yes")


def _log_pipeline_step(step_name: str, elapsed_ms: float, extra: str = "") -> None:
    """Log pipeline timing details when enabled via PIPELINE_TIMING=1."""
    if _PIPELINE_TIMING:
        msg = f"PIPELINE_TIMING api.{step_name}: {elapsed_ms:.0f} ms"
        if extra:
            msg += f" | {extra}"
        logger.info(msg)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="Nyaymalaw API", version="3.0.0")

# CORS: environment-aware â€” set ALLOWED_ORIGINS env var for production
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
from collections import defaultdict

_RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))   # seconds
_RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "100"))        # requests per window (increased from 30)
_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"  # Enabled by default; set to "false" for local dev
_rate_store: dict[str, list[float]] = defaultdict(list)

# Only rate-limit mutation / LLM-consuming endpoints (not health, static, etc.)
_RATE_LIMITED_PATHS = {
    "/submit_case", "/submit_case/stream",
    "/interview_step", "/interview_step/stream",
    "/conversation/continue", "/conversation/continue/stream",
    "/chat", "/search",
    "/agent/stream",          # H1 fix: orchestrator endpoint included
    "/upload-document",
}

# IPs to skip rate limiting (localhost for development)
_SKIP_RATE_LIMIT_IPS = {"127.0.0.1", "localhost", "::1"}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting if disabled or for localhost
    if not _RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    if request.method in ("POST", "PUT", "PATCH") and request.url.path in _RATE_LIMITED_PATHS:
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for localhost/development
        if client_ip in _SKIP_RATE_LIMIT_IPS:
            return await call_next(request)
        
        now = time.time()
        # Prune old entries
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
        logger.info(
            "%s %s â†’ %d (%.0fms)", method, path, response.status_code, elapsed_ms
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "%s %s â†’ 500 (%.0fms) %s", method, path, elapsed_ms, exc
        )
        raise


# ---------------------------------------------------------------------------
# Global error handler â€” catch unhandled exceptions, return clean JSON
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
        content={
            "detail": "An internal error occurred. Please try again.",
            # H3 fix: error_type removed — it was leaking internal class names to clients
        },
    )


# Paths and DB (must be before auth routes) â€“ use config for data root (e.g. Google Drive)
from config import (
    CHAT_HISTORY_DIR as _CHAT_HISTORY_DIR,
    DB_PATH as _DB_PATH,
    BARE_ACTS_DIR as _BARE_ACTS_DIR,
    CASELAW_DIR as _CASELAW_DIR,
    VECTOR_STORE as _VECTOR_STORE,
    USE_LEGAL_DATABASE as _USE_LEGAL_DATABASE,
    LEGAL_DB_JSON_OUTPUT as _LEGAL_DB_JSON_OUTPUT,
    LEGAL_DATABASE_DIR as _LEGAL_DATABASE_DIR,
)
_BASE_DIR = os.path.dirname(os.path.abspath(os.path.normpath(__file__)))

# Directories inside legal_database/json_output used by the pipeline.
# BareActs are mirrored under json_output/BareActs/<Jurisdiction>/..., and
# case-law JSONs are under json_output/caselaws/YYYY/MON/...
_LEGAL_DB_BAREACTS_DIR = os.path.join(_LEGAL_DB_JSON_OUTPUT, "BareActs")
_LEGAL_DB_CASELAWS_DIR = os.path.join(_LEGAL_DB_JSON_OUTPUT, "caselaws")

# â”€â”€ In-memory cache for library endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# _group_case_laws_by_court() opens every JSON file on disk; at 1000+ cases
# this takes ~2 minutes on first load.  Cache the result after the first call.
_caselaws_library_cache: dict[str, list[dict]] | None = None
_bareacts_library_cache: dict[str, list[dict]] | None = None


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt (salted, with work factor 12)."""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode("utf-8")


def _check_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.
    Also handles legacy SHA-256 hashes (hex strings) so existing accounts
    continue to work — they will be re-hashed to bcrypt on next successful login.
    """
    # Legacy SHA-256: hex string, 64 chars, no $2b$ prefix
    if not stored_hash.startswith("$2"):
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return secrets.compare_digest(legacy_hash, stored_hash)
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


def _get_db():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Anonymous user used when no auth token (auth disabled for direct access)
_ANONYMOUS_EMAIL = "anonymous@nyaymalaw.local"


def _init_auth_db():
    os.makedirs(_CHAT_HISTORY_DIR, exist_ok=True)
    conn = _get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                phone_number TEXT DEFAULT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL DEFAULT (datetime('now', '+30 days'))
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                opinion_text TEXT NOT NULL DEFAULT '',
                retrieved_json TEXT NOT NULL DEFAULT '[]',
                state_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        existing_chat_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(chats)").fetchall()
        }
        if "state_json" not in existing_chat_columns:
            conn.execute(
                "ALTER TABLE chats ADD COLUMN state_json TEXT NOT NULL DEFAULT '{}'"
            )
        # H2 migration: add expires_at to sessions if this is an existing DB
        existing_session_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "expires_at" not in existing_session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN expires_at TEXT NOT NULL "
                "DEFAULT (datetime('now', '+30 days'))"
            )
            conn.execute(
                "UPDATE sessions SET expires_at = datetime('now', '+30 days') "
                "WHERE expires_at IS NULL OR expires_at = ''"
            )
        # Phone migration: add phone_number to users if this is an existing DB
        existing_user_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "phone_number" not in existing_user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN phone_number TEXT DEFAULT NULL")
        conn.commit()
        # Ensure anonymous user exists for no-login access
        cur = conn.execute("SELECT id FROM users WHERE email = ?", (_ANONYMOUS_EMAIL,))
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (_ANONYMOUS_EMAIL, "", "Guest"),
            )
            conn.commit()
    finally:
        conn.close()


_init_auth_db()

# Phase 4: Ensure tier columns exist (idempotent migration)
from platform_pkg.tiers import (
    ensure_tier_columns,
    check_query_limit,
    increment_query_count,
    check_feature,
    get_tier_info,
    upgrade_user,
)
ensure_tier_columns()

security = HTTPBearer(auto_error=False)


def _user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Return user from token, or anonymous user when no token or invalid token (avoids 401 for stale tokens)."""
    def _anonymous_user():
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT id, email, name FROM users WHERE email = ?",
                (_ANONYMOUS_EMAIL,),
            ).fetchone()
            if row:
                return {"id": row["id"], "email": row["email"], "name": row["name"]}
        finally:
            conn.close()
        return {"id": 0, "email": _ANONYMOUS_EMAIL, "name": "Guest"}

    if not credentials or (credentials.credentials or "").strip() == "":
        return _anonymous_user()

    token = credentials.credentials.strip()
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT u.id, u.email, u.name FROM users u JOIN sessions s ON s.user_id = u.id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if not row:
            # Invalid or expired token: fall back to anonymous so app works without re-login
            logger.debug("Invalid or expired token; using anonymous user")
            return _anonymous_user()
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def _require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """Strict auth dependency — raises HTTP 401 if token is missing or invalid.
    Use on any endpoint that must NOT be accessible anonymously.
    """
    if not credentials or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials.strip()
    conn = _get_db()
    try:
        row = conn.execute(
            # H2: also check expires_at so stale sessions are rejected
            "SELECT u.id, u.email, u.name FROM users u "
            "JOIN sessions s ON s.user_id = u.id "
            "WHERE s.token = ? AND u.email != ? "
            "AND (s.expires_at IS NULL OR datetime(s.expires_at) > datetime('now'))",
            (token, _ANONYMOUS_EMAIL),
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def _enforce_query_limit(user: dict) -> None:
    """Raise 429 if the user has exceeded their daily query limit."""
    # Skip query limit enforcement for development (disabled by default)
    _QUERY_LIMIT_ENABLED = os.environ.get("QUERY_LIMIT_ENABLED", "false").lower() == "true"
    if not _QUERY_LIMIT_ENABLED:
        return  # Skip limit check in development
    
    # Also skip for anonymous/guest users (id 0 or anonymous email)
    if user.get("id") == 0 or user.get("email") == _ANONYMOUS_EMAIL:
        return
    
    result = check_query_limit(user["id"])
    if not result.get("allowed"):
        raise HTTPException(
            status_code=429,
            detail={
                "message": result.get("message", "Query limit exceeded"),
                "tier": result.get("tier", "free"),
                "queries_used": result.get("queries_used", 0),
                "queries_limit": result.get("queries_limit", 5),
            },
        )


def _enforce_feature(user: dict, feature: str) -> None:
    """Raise 403 if the user's tier doesn't include this feature."""
    result = check_feature(user["id"], feature)
    if not result.get("allowed"):
        raise HTTPException(
            status_code=403,
            detail={
                "message": result.get("message", "Feature not available"),
                "tier": result.get("tier", "free"),
                "feature": feature,
            },
        )


class RegisterRequest(BaseModel):
    email: str = ""
    name: str = ""
    password: str = ""
    phone_number: str = ""          # optional at registration


class LoginRequest(BaseModel):
    email: str = ""                 # email OR phone_number required
    phone_number: str = ""
    password: str = ""


class ChatPayload(BaseModel):
    id: Optional[int] = None
    title: str = ""
    messages: list = Field(default_factory=list)
    opinionText: str = ""
    retrieved: list = Field(default_factory=list)
    createdAt: str = ""
    workflowState: dict = Field(default_factory=dict)


def _normalize_workflow_state(raw_state: Optional[dict], messages: Optional[list] = None) -> dict:
    state = raw_state if isinstance(raw_state, dict) else {}
    msgs = messages if isinstance(messages, list) else []
    first_user = next(
        (
            m for m in msgs
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        ),
        None,
    )

    facts = state.get("facts")
    if not isinstance(facts, str):
        facts = first_user.get("content", "") if first_user else ""
    facts = facts.strip()

    stage = (state.get("stage") or "").strip().lower()
    if stage not in {"await_facts", "interview", "done"}:
        stage = "done" if msgs else "await_facts"

    current_question = state.get("currentQuestion")
    if not isinstance(current_question, str):
        current_question = ""
    current_question = current_question.strip()

    qa_history = []
    for qa in (state.get("qaHistory") or []):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if question or answer:
            qa_history.append({"question": question, "answer": answer})

    analysis_stage = str(state.get("analysisStage") or "").strip().lower()
    if not analysis_stage:
        analysis_stage = "intake" if stage != "done" else ""

    facts_summary = state.get("factsSummary")
    if not isinstance(facts_summary, str):
        facts_summary = ""
    facts_summary = facts_summary.strip()

    last_response_type = state.get("lastResponseType")
    if not isinstance(last_response_type, str):
        last_response_type = ""
    last_response_type = last_response_type.strip()

    # Preserve Stage 1 structured intake state (opaque blob — pass through as-is)
    intake_state = state.get("intakeState")
    if not isinstance(intake_state, dict):
        intake_state = None

    return {
        "stage": stage,
        "facts": facts,
        "currentQuestion": current_question,
        "qaHistory": qa_history,
        "analysisStage": analysis_stage,
        "factsSummary": facts_summary,
        "lastResponseType": last_response_type,
        "intakeState": intake_state,
    }


def _safe_json_loads(raw_text: Optional[str], fallback):
    try:
        return json.loads(raw_text or "")
    except Exception:
        return fallback


# ---------- Auth & Chat history (backend storage) ----------

@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    email = (req.email or "").strip().lower()
    name = (req.name or "").strip()
    password = (req.password or "").strip()
    phone = (req.phone_number or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not name:
        raise HTTPException(status_code=400, detail="Name required")
    password_hash = _hash_password(password)
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, name, phone_number) VALUES (?, ?, ?, ?)",
            (email, password_hash, name, phone or None),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+30 days'))",
            (user_id, token),
        )
        conn.commit()
        return {
            "success": True,
            "token": token,
            "user": {"email": email, "name": name},
        }
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="An account with this email already exists")
    finally:
        conn.close()


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    phone = (req.phone_number or "").strip()
    password = (req.password or "").strip()
    if not (email or phone) or not password:
        raise HTTPException(status_code=400, detail="Email (or phone number) and password required")
    conn = _get_db()
    try:
        # Support login via email OR phone number
        if email:
            row = conn.execute(
                "SELECT id, email, name, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, email, name, password_hash FROM users WHERE phone_number = ?",
                (phone,),
            ).fetchone()
        # Use constant-time check to prevent timing attacks; also handles legacy SHA-256
        if not row or not _check_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        # Transparently upgrade legacy SHA-256 hash to bcrypt on successful login
        if not row["password_hash"].startswith("$2"):
            new_hash = _hash_password(password)
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, row["id"]),
            )
            logger.info("Upgraded password hash to bcrypt for user id=%s", row["id"])
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) "
            "VALUES (?, ?, datetime('now', '+30 days'))",
            (row["id"], token),
        )
        conn.commit()
        return {
            "success": True,
            "token": token,
            "user": {"email": row["email"], "name": row["name"]},
        }
    finally:
        conn.close()


@app.post("/auth/logout")
def auth_logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """H2: Invalidate the current session token. Safe to call even if already logged out."""
    if not credentials or not (credentials.credentials or "").strip():
        return {"success": True, "detail": "No active session."}
    token = credentials.credentials.strip()
    conn = _get_db()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        logger.info("Session token invalidated via /auth/logout")
    finally:
        conn.close()
    return {"success": True, "detail": "Logged out successfully."}


@app.get("/chats")
def chats_list(user: dict = Depends(_require_auth)):
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, state_json, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
        out = []
        for r in rows:
            messages = _safe_json_loads(r["messages_json"], [])
            out.append({
                "id": r["id"],
                "title": r["title"],
                "messages": messages,
                "opinionText": r["opinion_text"] or "",
                "retrieved": _safe_json_loads(r["retrieved_json"], []),
                "workflowState": _normalize_workflow_state(
                    _safe_json_loads(r["state_json"], {}),
                    messages,
                ),
                "createdAt": r["created_at"],
            })
        return {"chats": out}
    finally:
        conn.close()


def _coerce_chat_id(v):
    """Ensure chat id is int for DB (handles int or string from JSON)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@app.post("/chats")
def chats_upsert(payload: ChatPayload, user: dict = Depends(_require_auth)):
    """Create or update a chat. If id is provided and exists for this user, update; else create with id or new id."""
    title = (payload.title or "").strip() or "Untitled chat"
    messages = payload.messages if isinstance(payload.messages, list) else []
    opinion_text = payload.opinionText or ""
    retrieved = payload.retrieved if isinstance(payload.retrieved, list) else []
    workflow_state = _normalize_workflow_state(payload.workflowState, messages)
    created_at = payload.createdAt or __import__("datetime").datetime.utcnow().isoformat() + "Z"
    messages_json = json.dumps(messages)
    retrieved_json = json.dumps(retrieved)
    state_json = json.dumps(workflow_state)
    payload_id = _coerce_chat_id(payload.id)
    conn = _get_db()
    try:
        if payload_id is not None:
            cur = conn.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?",
                (payload_id, user["id"]),
            )
            existing = cur.fetchone()
            if existing:
                conn.execute(
                    "UPDATE chats SET title = ?, messages_json = ?, opinion_text = ?, retrieved_json = ?, state_json = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
                    (title, messages_json, opinion_text, retrieved_json, state_json, payload_id, user["id"]),
                )
                conn.commit()
                return {"id": payload_id, "title": title}
        chat_id = payload_id if payload_id is not None else int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM chats").fetchone()[0])
        conn.execute(
            "INSERT OR REPLACE INTO chats (id, user_id, title, messages_json, opinion_text, retrieved_json, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user["id"], title, messages_json, opinion_text, retrieved_json, state_json, created_at),
        )
        conn.commit()
        return {"id": chat_id, "title": title}
    finally:
        conn.close()


@app.get("/chats/{chat_id}")
def chat_get(chat_id: int, user: dict = Depends(_require_auth)):
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, state_json, created_at FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chat not found")
        messages = _safe_json_loads(row["messages_json"], [])
        return {
            "id": row["id"],
            "title": row["title"],
            "messages": messages,
            "opinionText": row["opinion_text"] or "",
            "retrieved": _safe_json_loads(row["retrieved_json"], []),
            "workflowState": _normalize_workflow_state(
                _safe_json_loads(row["state_json"], {}),
                messages,
            ),
            "createdAt": row["created_at"],
        }
    finally:
        conn.close()


@app.delete("/chats/{chat_id}")
def chat_delete(chat_id: str, user: dict = Depends(_require_auth)):
    """Remove a chat from the user's history."""
    try:
        id_val = int(chat_id.strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid chat id")
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (id_val, user["id"]))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"success": True}
    finally:
        conn.close()


# ---------- End Auth & Chat history ----------


class SearchQuery(BaseModel):
    issue: str


class ChatMessage(BaseModel):
    role: str
    content: str | dict  # str for API, dict if frontend sends object


class ChatRequest(BaseModel):
    conversation: list[ChatMessage]
    message: str
    phase: str = "fact_collection"
    facts_summary: str | None = None




class SubmitCaseRequest(BaseModel):
    """Initial case submission - frontend sends { text, mode }."""
    text: str = ""
    # chat_mode: "legal_opinion" | "legal_research" | "general" (optional; default router when empty)
    mode: str | None = None
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


class QAPair(BaseModel):
    question: str = ""
    answer: str = ""


class InterviewStepRequest(BaseModel):
    """Follow-up answer in interview - frontend sends { facts, qa_history, mode }."""
    facts: str = ""
    qa_history: list[QAPair] = []
    mode: str | None = None      # "legal_opinion" | "legal_research" | "general"
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


class ContinueChatRequest(BaseModel):
    """Continue a loaded chat - full conversation history + new user message."""
    conversation: list[ChatMessage] = []
    message: str = ""
    mode: str | None = None  # "legal_opinion" | "legal_research" | "general"
    model_override: str | None = None
    workflowState: dict = Field(default_factory=dict)


def _build_conv(messages: list[ChatMessage] | None) -> list[dict]:
    if not messages:
        return []
    return [{"role": m.role, "content": _normalize_content(m.content)} for m in messages]


# Path to vector store and BareActs directory â€“ resolve from this fileâ€™s location


def _iter_bareacts_json_files():
    """
    Yield (rel_dir, base_name) for each BareActs JSON file under the legal
    database. Mirrors the json_output/BareActs/<Jurisdiction>/ tree produced
    by legal_database/pipeline.py.
    """
    if not _USE_LEGAL_DATABASE:
        return

    # Preferred structure: json_output/BareActs/<Jurisdiction>/YYYY_idx_ActName.json
    if os.path.isdir(_LEGAL_DB_BAREACTS_DIR):
        for root, _, files in os.walk(_LEGAL_DB_BAREACTS_DIR):
            rel = os.path.relpath(root, _LEGAL_DB_BAREACTS_DIR)
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                base = os.path.splitext(name)[0]
                if base:
                    yield rel, base
        return

    # Backwards-compatible fallback: flat json_output/ directory.
    if os.path.isdir(_LEGAL_DB_JSON_OUTPUT):
        for name in os.listdir(_LEGAL_DB_JSON_OUTPUT):
            if not name.lower().endswith(".json"):
                continue
            base = os.path.splitext(name)[0]
            if base:
                # Treat flat layout as having no jurisdiction-specific folder.
                yield "", base


def _list_bare_acts_from_json_output() -> list[str]:
    """List act names (JSON base names) from legal_database/json_output (statute schema)."""
    if not _USE_LEGAL_DATABASE:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for _rel, base in _iter_bareacts_json_files():
        if base in seen:
            continue
        # Optionally validate statute schema by peeking into one file per base,
        # but skip heavy JSON reads here for performance.
        seen.add(base)
        out.append(base)
    return sorted(out)


def _iter_caselaws_json_files():
    """
    Yield (path, base_name) for each case-law JSON file under the legal
    database. Matches the json_output/caselaws/YYYY/MON/ tree produced
    by legal_database/pipeline.py, with a flat json_output/ fallback.
    """
    if not _USE_LEGAL_DATABASE:
        return

    # Preferred structure: json_output/caselaws/YYYY/MON/CaseName.json
    if os.path.isdir(_LEGAL_DB_CASELAWS_DIR):
        for root, _, files in os.walk(_LEGAL_DB_CASELAWS_DIR):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                base = os.path.splitext(name)[0]
                if base:
                    yield os.path.join(root, name), base
        return

    # Backwards-compatible fallback: flat json_output/ directory.
    if os.path.isdir(_LEGAL_DB_JSON_OUTPUT):
        for name in os.listdir(_LEGAL_DB_JSON_OUTPUT):
            if not name.lower().endswith(".json"):
                continue
            base = os.path.splitext(name)[0]
            if base:
                yield os.path.join(_LEGAL_DB_JSON_OUTPUT, name), base


def _list_case_laws_from_json_output() -> list[str]:
    """
    List case names (JSON base names) from legal_database/json_output
    (case schema).
    """
    if not _USE_LEGAL_DATABASE:
        return []

    bases: set[str] = set()
    for _path, base in _iter_caselaws_json_files():
        bases.add(base)
    return sorted(bases)


def _group_case_laws_by_court() -> dict[str, list[dict]]:
    """
    Group case laws by court using raw_data/CaseLaws/<Court>/ folder names.
    Returns {"Court": [{"name": "Case Title", "file": "CaseLaws/Court/case.txt"}, ...]}
    where "file" is relative to raw_data/ and used directly in the view URL.
    """
    groups: dict[str, list[dict]] = {}
    if not _USE_LEGAL_DATABASE or not os.path.isdir(_CASELAW_DIR):
        return groups

    vs_path = os.path.join(_VECTOR_STORE, "case_summaries_v2_chunks.json")
    case_title_map: dict[str, str] = {}
    if os.path.exists(vs_path):
        try:
            import json as _json
            with open(vs_path, encoding="utf-8") as f:
                summaries = _json.load(f)
            for chunk in summaries.values():
                sf = chunk.get("source_file", "").replace("\\", "/")
                title = (chunk.get("title") or chunk.get("case_name") or "").strip()
                if sf and title:
                    case_title_map[sf] = title
        except Exception:
            pass

    raw_data_dir = os.path.dirname(_CASELAW_DIR)

    for entry in os.scandir(_CASELAW_DIR):
        if not entry.is_dir():
            continue
        court_label = entry.name
        cases: list[dict] = []
        for root, _, files in os.walk(entry.path):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, raw_data_dir).replace("\\", "/")
                title = case_title_map.get(rel, "") or os.path.splitext(fname)[0]
                cases.append({"name": title, "file": rel})
        if cases:
            cases.sort(key=lambda c: c["name"])
            groups[court_label] = cases

    return groups


def _list_bare_acts_from_vector_store() -> list[str]:
    """Return bare act names; from json_output when USE_LEGAL_DATABASE, else from disk."""
    if _USE_LEGAL_DATABASE:
        return _list_bare_acts_from_json_output()
    return _list_bare_acts_from_disk()


def _bare_act_file_exists(name: str) -> bool:
    """True if a file with this name exists in BareActs (case-insensitive on Windows)."""
    base = os.path.basename(name).strip()
    if not base:
        return False
    path = os.path.join(_BARE_ACTS_DIR, base)
    if os.path.isfile(path):
        return True
    # Case-insensitive fallback (e.g. Windows)
    if not os.path.isdir(_BARE_ACTS_DIR):
        return False
    for f in os.listdir(_BARE_ACTS_DIR):
        if f and os.path.isfile(os.path.join(_BARE_ACTS_DIR, f)) and f.lower() == base.lower():
            return True
    return False


def _list_bare_acts_from_disk() -> list[str]:
    """Fallback: list PDF/text files directly from BareActs directory."""
    if not os.path.isdir(_BARE_ACTS_DIR):
        return []
    out = []
    for f in os.listdir(_BARE_ACTS_DIR):
        path = os.path.join(_BARE_ACTS_DIR, f)
        if os.path.isfile(path) and (f.lower().endswith(".pdf") or f.lower().endswith(".txt")):
            out.append(f)
    return sorted(out)


def _load_act_names_from_summaries() -> dict[str, str]:
    """
    Load source_file -> act_name mapping from act_summaries vector store chunks.
    Keys are normalised forward-slash paths relative to raw_data/, e.g.
    "BareActs/Telangana/1948_0.txt".
    """
    vs_path = os.path.join(_VECTOR_STORE, "act_summaries_v2_chunks.json")
    if not os.path.exists(vs_path):
        return {}
    try:
        import json as _json
        with open(vs_path, encoding="utf-8") as f:
            chunks = _json.load(f)
        mapping: dict[str, str] = {}
        for chunk in chunks.values():
            sf = chunk.get("source_file", "").replace("\\", "/")
            name = chunk.get("act_name", "").strip()
            if sf and name:
                mapping[sf] = name
        return mapping
    except Exception:
        return {}


def _group_bare_acts_by_jurisdiction() -> dict[str, list[dict]]:
    """
    Group Bare Acts by jurisdiction folder under raw_data/BareActs.
    Returns {"Jurisdiction": [{"name": "Act Name", "file": "BareActs/Jurisdiction/1948_0.txt"}, ...]}
    where "file" is relative to raw_data/ and used directly in the view URL.
    """
    groups: dict[str, list[dict]] = {}

    if not _USE_LEGAL_DATABASE or not os.path.isdir(_BARE_ACTS_DIR):
        return groups

    act_name_map = _load_act_names_from_summaries()
    raw_data_dir = os.path.dirname(_BARE_ACTS_DIR)

    for entry in os.scandir(_BARE_ACTS_DIR):
        if not entry.is_dir():
            continue
        jurisdiction = entry.name
        acts: list[dict] = []
        for root, _, files in os.walk(entry.path):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, raw_data_dir).replace("\\", "/")
                act_name = act_name_map.get(rel, "") or os.path.splitext(fname)[0]
                acts.append({"name": act_name, "file": rel})
        if acts:
            acts.sort(key=lambda a: a["name"])
            groups[jurisdiction] = acts

    return groups


def _chat_error_fallback(detail: str = "") -> dict:
    """Return a safe 200 response when chat processing fails so frontend does not see 500.
    We try to generate a message from the LLM; if that also fails we use a minimal technical note."""
    try:
        from platform_pkg.llm import ask_llm
        error_context = f" (Technical detail: {detail})" if detail else ""
        msg = ask_llm(
            f"You are a legal assistant. Something went wrong while processing the user's request.{error_context} "
            "Write a short, friendly one-sentence apology to the user asking them to try again. Do not mention technical details."
        ).strip()
        if msg:
            return {"status": "question", "next_question": msg, "retrieved": []}
    except Exception:
        pass
    # Absolute last resort (LLM itself is down)
    return {
        "status": "question",
        "next_question": "Something went wrong. Please try again.",
        "retrieved": [],
    }


def _fire_feedback_log(result: dict, facts: str, session_ref: str = "") -> None:
    """
    Background helper: extract model output fields from a 'done'-phase result and
    write one row to the Feedback Log workbook. Runs in a daemon thread so it never
    blocks the HTTP response. Silently swallows all errors.
    """
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return
    if result.get("phase") != "done" or not result.get("response"):
        return
    import threading
    resp = result.get("response") or {}

    # Router classification for this interaction: one of
    # "Legal Opinion", "Direct search/lookup", "Non Legal".
    response_type = (result.get("response_type") or "").strip().lower()
    if response_type in {"search_results", "lookup_results"}:
        router_classification = "Direct search/lookup"
    elif response_type == "generic_chat":
        router_classification = "Non Legal"
    else:
        # Default route is legal opinion (includes bare-acts-only rows)
        router_classification = "Legal Opinion"

    disputes_list = [d.get("dispute", "") for d in (resp.get("disputes") or []) if d.get("dispute")]
    if not disputes_list:
        disputes_list = [str(result.get("facts_summary", "")[:80])]

    bare_acts = resp.get("bare_act_sections") or []
    sections_list = [
        f"{ba.get('act_name','?')} Â§ {ba.get('section_number','?')}"
        for ba in bare_acts
    ]

    case_laws_all = (resp.get("case_laws") or []) + (resp.get("internet_case_laws") or [])
    cl_list = [
        (cl.get("case_name") or cl.get("title") or cl.get("citation") or "?")
        for cl in case_laws_all
    ]

    explanation = (resp.get("explanation") or "").strip()
    followup = (resp.get("followup_question") or result.get("followup_question") or "").strip()

    def _do_log():
        try:
            _log_interaction(
                facts=facts,
                followup_question=followup,
                disputes=disputes_list,
                sections=sections_list,
                case_laws=cl_list,
                legal_opinion=explanation,
                router_classification=router_classification,
                session_ref=session_ref,
            )
        except Exception as _le:
            logger.warning("Feedback log failed (non-critical): %s", _le)

    t = threading.Thread(target=_do_log, daemon=True, name="feedback-log")
    t.start()




def _fire_feedback_log_research(result: dict, query: str, session_ref: str = "") -> None:
    """
    Log one feedback row for a raw /search (fusion research) call.

    The research result has no 'phase' or AI-generated opinion â€” it is a pure
    retrieval result from fuse_bare_act_and_case_law.  We map its fields directly:

      facts          â†’ the search query string
      disputes       â†’ [query[:80]]  (no dispute extraction on raw search)
      sections       â†’ act_name Â§ section_number  from bare_act_sections
      case_laws      â†’ case_name / citation  from case_laws list
      legal_opinion  â†’ ""  (raw retrieval, no opinion generated)
      followup       â†’ ""
    """
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return
    import threading

    bare_acts = result.get("bare_act_sections") or []
    sections_list = [
        f"{ba.get('act_name', '?')} Â§ {ba.get('section_number', '?')}"
        for ba in bare_acts
        if isinstance(ba, dict)
    ]

    case_laws_raw = result.get("case_laws") or []
    cl_list = [
        (cl.get("case_name") or cl.get("title") or cl.get("citation") or "?")
        for cl in case_laws_raw
        if isinstance(cl, dict)
    ]

    def _do_log():
        try:
            _log_interaction(
                facts=query,
                followup_question="",
                disputes=[query[:80]] if query else [],
                sections=sections_list,
                case_laws=cl_list,
                legal_opinion="",
                 router_classification="Direct search/lookup",
                session_ref=session_ref,
            )
        except Exception as _le:
            logger.warning("Feedback log (research) failed (non-critical): %s", _le)

    t = threading.Thread(target=_do_log, daemon=True, name="feedback-log-research")
    t.start()


def _safe_build_advocate_review(
    bare_act_sections: list,
    facts_summary: str,
    intake_state: dict | None = None,
) -> dict | None:
    """
    Build the advocate-review JSON brief from Stage 5.
    Returns None if Stage 5 is disabled or if it raises.
    Called only when phase == "done" with a legal_opinion response.
    """
    if not _STAGE5_ENABLED or not _build_legal_draft:
        return None
    if not bare_act_sections:
        return None
    try:
        result = _build_legal_draft(
            intake_state=intake_state,
            bare_act_sections=bare_act_sections,
            facts_summary=facts_summary or "",
        )
        return result.get("advocate_review")
    except Exception as _ar_err:
        logger.warning("Stage5 advocate_review build failed: %s", _ar_err)
        return None


def _map_chat_result_to_ui(result: dict, pre_draft_msg: str = "") -> dict:
    """Map process_chat result to the shape the frontend expects (status, next_question, etc.)."""
    phase = result.get("phase")
    response_type = result.get("response_type")  # "search_results", "lookup_results", "legal_opinion"
    analysis_stage = result.get("analysis_stage") or ""
    facts_summary = (result.get("facts_summary") or "").strip()

    def _safe_next_question(text: str) -> str:
        candidate = (text or "").strip()
        normalized = re.sub(r"[\s\W_]+", "", candidate)
        if len(normalized) < 6:
            return "Please share one more important detail, or say 'proceed' if you want me to identify the applicable bare act sections."
        return candidate

    if phase == "fact_collection":
        next_q = _safe_next_question(result.get("message") or "")
        return {
            "status": "question",
            "next_question": next_q,
            "retrieved": result.get("response") or [],
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
            "intake_state": result.get("intake_state") or None,
        }
    if phase == "done" and result.get("response"):
        resp = result["response"]
        bare_acts = resp.get("bare_act_sections") or []
        case_laws = resp.get("case_laws") or []
        internet_case_laws = resp.get("internet_case_laws") or []
        all_case_laws = case_laws + internet_case_laws
        # Item 19: merge pre-draft summary with explanation.
        # pre_draft_msg is the Stage 1 closing / Stage 2 opening summary generated
        # before response_generation ran. Combine: pre_draft → explanation.
        effective_greeting = pre_draft_msg.strip() or (result.get("message") or "").strip()
        explanation = (resp.get("explanation") or "").strip()
        if effective_greeting and explanation:
            if effective_greeting == explanation or effective_greeting in explanation:
                combined_text = explanation
            elif explanation in effective_greeting:
                combined_text = effective_greeting
            else:
                combined_text = f"{effective_greeting}\n\n{explanation}"
        else:
            combined_text = effective_greeting or explanation
        # Ensure we never send an empty or trivial intro (e.g. just "âš–")
        if not combined_text or len(combined_text.strip()) < 20:
            combined_text = "I've prepared an initial response based on the information currently available."
        # If bare acts have nested case laws, don't return separate case_laws array to avoid duplicates
        # Case laws are now nested under bare_acts[].related_case_laws
        separate_case_laws = []
        if bare_acts and len(bare_acts) > 0:
            has_nested_case_laws = any(ba.get("related_case_laws") for ba in bare_acts)
            if not has_nested_case_laws:
                separate_case_laws = all_case_laws

        # Item 20: build advocate-review JSON brief from Stage 5
        advocate_review = _safe_build_advocate_review(
            bare_act_sections=bare_acts,
            facts_summary=facts_summary,
        )

        out = {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": combined_text,
            "bare_acts": bare_acts,
            "case_laws": separate_case_laws,
            "next_steps": resp.get("next_steps") or [],
            "next_steps_summary": (resp.get("next_steps_summary") or "").strip(),
            "retrieved": all_case_laws + bare_acts,
            "progress": resp.get("progress"),
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
            "advocate_review": advocate_review,  # Item 20: structured brief for UI panel
        }
        return out
    if phase == "done":
        return {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": (result.get("message") or "").strip() or "Your request has been processed.",
            "bare_acts": [],
            "case_laws": [],
            "next_steps": [],
            "next_steps_summary": "",
            "retrieved": [],
            "model_used": get_last_model_used(),
            "analysis_stage": analysis_stage,
            "facts_summary": facts_summary,
        }
    next_q = _safe_next_question(result.get("message") or "")
    return {
        "status": "question",
        "next_question": next_q,
        "retrieved": result.get("response") or [],
        "model_used": get_last_model_used(),
        "analysis_stage": analysis_stage,
        "facts_summary": facts_summary,
    }


@app.post("/search")
def search_law(query: SearchQuery, background_tasks: BackgroundTasks):
    """Legacy search endpoint - single query, returns bare acts + case laws."""
    result = fuse_bare_act_and_case_law.run(issue=query.issue)
    # Log the research query to the feedback log (non-critical, runs in background)
    background_tasks.add_task(
        _fire_feedback_log_research,
        result if isinstance(result, dict) else {},
        query.issue,
        "",  # no session_ref available on raw search
    )
    return result


@app.get("/bareacts/library")
def bareacts_library():
    """
    Return Bare Acts grouped by jurisdiction, mirroring the
    legal_database/json_output/BareActs/<Jurisdiction>/ structure.

    Response shape:
      {
        "jurisdictions": [
          {"name": "Telangana", "acts": ["YYYY_idx_ActName", ...]},
          {"name": "Union of India", "acts": ["YYYY_idx_ActName", ...]},
          ...
        ]
      }
    """
    global _bareacts_library_cache

    if not _USE_LEGAL_DATABASE:
        acts = _list_bare_acts_from_vector_store()
        return {"jurisdictions": [{"name": "All", "acts": acts}]}

    if _bareacts_library_cache is None:
        _bareacts_library_cache = _group_bare_acts_by_jurisdiction()

    jurisdictions = [
        {"name": name, "acts": acts}
        for name, acts in sorted(_bareacts_library_cache.items(), key=lambda kv: kv[0].lower())
    ]
    return {"jurisdictions": jurisdictions}


@app.post("/library/refresh-cache")
def library_refresh_cache():
    """Force-refresh the in-memory library caches (call after ingesting new documents)."""
    global _caselaws_library_cache, _bareacts_library_cache
    _caselaws_library_cache = None
    _bareacts_library_cache = None
    return {"status": "ok", "message": "Library caches cleared â€” will reload on next request"}


@app.get("/bareacts/list")
def bareacts_list():
    """Return flat list of Bare Acts for backwards compatibility."""
    acts = _list_bare_acts_from_vector_store()
    return {"acts": acts}


@app.get("/bareacts/debug")
def bareacts_debug():
    """Help debug 404s: returns the path the server uses and whether it exists."""
    return {
        "base_dir": _BASE_DIR,
        "bare_acts_dir": _BARE_ACTS_DIR,
        "dir_exists": os.path.isdir(_BARE_ACTS_DIR),
        "files": sorted(os.listdir(_BARE_ACTS_DIR)) if os.path.isdir(_BARE_ACTS_DIR) else [],
    }


def _resolve_bare_act_path(base: str):
    """Return absolute path to file in BareActs if it exists, else None (tries case-insensitive)."""
    base = (base or "").strip()
    if not base:
        return None
    lookup = base if base.lower().endswith((".pdf", ".txt")) else base + ".pdf"
    path = os.path.join(_BARE_ACTS_DIR, lookup)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_BARE_ACTS_DIR):
        for f in os.listdir(_BARE_ACTS_DIR):
            if f.lower() == lookup.lower():
                return os.path.abspath(os.path.join(_BARE_ACTS_DIR, f))
    return None


def _get_json_from_legal_db(base: str, is_statute: bool) -> dict | None:
    """Return parsed JSON for a statute or case from legal_database/json_output."""
    if not _USE_LEGAL_DATABASE:
        return None
    base = (base or "").strip()
    if not base:
        return None
    if base.lower().endswith(".json"):
        base = base[:-5]

    # Choose the correct root based on whether we are looking for a statute
    # (BareActs) or a case law (caselaws). The pipeline mirrors:
    #   json_output/BareActs/<Jurisdiction>/YYYY_idx_ActName.json
    #   json_output/caselaws/YYYY/MON/CaseName.json
    if is_statute:
        search_root = _LEGAL_DB_BAREACTS_DIR if os.path.isdir(_LEGAL_DB_BAREACTS_DIR) else _LEGAL_DB_JSON_OUTPUT
    else:
        search_root = _LEGAL_DB_CASELAWS_DIR if os.path.isdir(_LEGAL_DB_CASELAWS_DIR) else _LEGAL_DB_JSON_OUTPUT

    if not os.path.isdir(search_root):
        return None

    # First try an exact filename under the chosen root (non-recursive).
    candidate = os.path.join(search_root, base + ".json")
    path = candidate if os.path.isfile(candidate) else ""

    # Fallback: walk the tree and match by base name, case-insensitive.
    if not path:
        target = base.lower()
        for root, _, files in os.walk(search_root):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                if os.path.splitext(name)[0].lower() == target:
                    path = os.path.join(root, name)
                    break
            if path:
                break
        if not path:
            return None
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if is_statute and not (data.get("act_id") or data.get("sections")):
            return None
        if not is_statute and data.get("act_id"):
            return None
        return data
    except Exception:
        return None


@app.get("/bareacts/json")
def bareacts_json(name: str = Query(..., description="Base name of the act JSON (e.g. ADVOCATES_Act)")):
    """Return full statute JSON from legal_database/json_output. Requires NYAYMALAW_DATA_SOURCE=legal_database."""
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=True)
    if not data:
        raise HTTPException(status_code=404, detail="Act not found")
    return data


@app.get("/caselaws/json")
def caselaws_json(name: str = Query(..., description="Base name of the case JSON (e.g. ABHILASHA_V_PARKASH)")):
    """Return full case law JSON from legal_database/json_output. Requires NYAYMALAW_DATA_SOURCE=legal_database."""
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=False)
    if not data:
        raise HTTPException(status_code=404, detail="Case not found")
    return data


def _serve_raw_txt_as_html(file_rel: str, title: str) -> HTMLResponse:
    """Serve a raw_data/ txt file as a simple readable HTML page."""
    raw_data_dir = os.path.dirname(_BARE_ACTS_DIR)
    raw_path = os.path.abspath(os.path.join(raw_data_dir, file_rel.replace("/", os.sep)))
    if not raw_path.startswith(os.path.abspath(raw_data_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.isfile(raw_path):
        raise HTTPException(status_code=404, detail="File not found")
    content = open(raw_path, encoding="utf-8", errors="replace").read()
    is_case = file_rel.startswith("CaseLaws/") or "/CaseLaws/" in file_rel
    return HTMLResponse(content=_format_legal_html(content, title, is_case=is_case))


def _format_legal_html(text: str, title: str, is_case: bool = False) -> str:
    """
    Convert a raw bare-act / case-law TXT into a clean, readable HTML page.
    Section and subsection numbers are rendered inline with their text.
    A Print/PDF button is included at the top.
    For case laws (is_case=True), numbered paragraphs are merged across line
    breaks so the text flows naturally.
    """
    import html as _h
    import re as _re

    body_parts: list[str] = []

    if is_case:
        # â”€â”€ Case law: render raw text as-is â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        body_parts.append(f'<pre class="case-raw">{_h.escape(text)}</pre>')

    else:
        # â”€â”€ Bare act: section-aware line-by-line parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lines = [l.rstrip() for l in text.splitlines()]

        # Patterns
        SEC_RE   = _re.compile(r'^(\d+[A-Z]?)\.\s*$')          # "14." alone
        SUB_RE   = _re.compile(r'^(\([0-9a-zA-Z]+\))\s*$')      # "(1)" or "(a)" alone
        HEAD_RE  = _re.compile(r'^(PART|CHAPTER|SCHEDULE)\b', _re.I)
        BRACKET_LINE_RE = _re.compile(r'^\[.*\]\s*$')            # editorial notes "[...]"

        def next_nonempty(idx):
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            return j

        i = 0
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()

            if not line:
                i += 1
                continue

            # â”€â”€ Section heading: "14." on its own line â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            m = SEC_RE.match(line)
            if m:
                num = m.group(1)
                j = next_nonempty(i)
                heading = _h.escape(lines[j].strip()) if j < len(lines) else ""
                body_parts.append(
                    f'<div class="section">'
                    f'<span class="sec-num">{_h.escape(num)}.</span> '
                    f'<span class="sec-title">{heading}</span>'
                    f'</div>'
                )
                i = j + 1 if j < len(lines) else i + 1
                continue

            # â”€â”€ Subsection: "(1)" or "(a)" on its own line â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            m = SUB_RE.match(line)
            if m:
                marker = m.group(1)
                j = next_nonempty(i)
                sub_text = _h.escape(lines[j].strip()) if j < len(lines) else ""
                body_parts.append(
                    f'<p class="subsection">'
                    f'<span class="sub-marker">{_h.escape(marker)}</span> {sub_text}'
                    f'</p>'
                )
                i = j + 1 if j < len(lines) else i + 1
                continue

            # â”€â”€ Part / Chapter / Schedule heading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if HEAD_RE.match(line) or (line.isupper() and 4 < len(line) < 80 and not BRACKET_LINE_RE.match(line)):
                body_parts.append(f'<h2 class="chapter">{_h.escape(line)}</h2>')
                i += 1
                continue

            # â”€â”€ Editorial note in brackets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if BRACKET_LINE_RE.match(line):
                body_parts.append(f'<p class="editorial">{_h.escape(line)}</p>')
                i += 1
                continue

            # â”€â”€ Plain paragraph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            body_parts.append(f'<p>{_h.escape(line)}</p>')
            i += 1

    body_html = "\n".join(body_parts)
    t = _h.escape(title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{t}</title>
<style>
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 860px;
    margin: 36px auto;
    padding: 0 28px 60px;
    line-height: 1.8;
    color: #1a1a1a;
    font-size: 15px;
  }}
  h1 {{
    font-size: 1.25em;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
    margin-bottom: 24px;
  }}
  h2.chapter {{
    font-size: 1em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 28px;
    margin-bottom: 4px;
    color: #444;
  }}
  div.section {{
    margin-top: 20px;
    margin-bottom: 4px;
  }}
  .sec-num {{
    font-weight: bold;
    font-size: 1em;
    color: #111;
  }}
  .sec-title {{
    font-weight: bold;
  }}
  p {{
    margin: 4px 0 4px 0;
  }}
  p.subsection {{
    margin: 4px 0 4px 1.6em;
  }}
  .sub-marker {{
    font-weight: 600;
    min-width: 2em;
    display: inline-block;
  }}
  p.editorial {{
    color: #666;
    font-style: italic;
    font-size: 0.9em;
    margin: 2px 0 2px 1.6em;
  }}
  pre.case-raw {{
    white-space: pre-wrap;
    word-break: break-word;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 15px;
    line-height: 1.8;
    margin: 0;
  }}
  /* Download / print button */
  #dl-bar {{
    position: sticky;
    top: 0;
    background: #f8f8f8;
    border-bottom: 1px solid #ddd;
    padding: 8px 0 8px 0;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    z-index: 100;
  }}
  #dl-bar h1 {{
    margin: 0;
    border: none;
    padding: 0;
    font-size: 1em;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  #pdf-btn {{
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 6px 16px;
    font-size: 0.88em;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  #pdf-btn:hover {{ background: #1648c0; }}
  @media print {{
    #dl-bar {{ display: none; }}
    body {{ margin: 0; padding: 16px; }}
  }}
</style>
</head>
<body>
<div id="dl-bar">
  <h1>{t}</h1>
  <button id="pdf-btn" onclick="window.print()">â¬‡ Download PDF</button>
</div>
{body_html}
</body>
</html>"""


@app.get("/bareacts/view", response_class=HTMLResponse)
def bareacts_view(
    file: str = Query(None, description="Path relative to raw_data/ e.g. BareActs/Telangana/1948_0.txt"),
    name: str = Query(None, description="Legacy: act name (fallback to JSON output)"),
):
    """Render a bare act as HTML. Prefers file= (raw txt); falls back to name= (JSON output)."""
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="Requires NYAYMALAW_DATA_SOURCE=legal_database")

    if file:
        return _serve_raw_txt_as_html(file, name or file.split("/")[-1])

    # Legacy fallback: look up by name in json_output
    data = _get_json_from_legal_db(name or "", is_statute=True)
    if not data:
        raise HTTPException(status_code=404, detail="Act not found")

    title = (data.get("act_name") or name or "").strip() or "Bare Act"
    year = data.get("year") or (data.get("act_summary") or {}).get("year")
    sections = data.get("sections") or []

    # Group sections by section_number and sort numerically.
    sections_by_num: dict[str, list[dict]] = {}
    for sec in sections:
        num = str(sec.get("section_number") or "").strip()
        if not num:
            continue
        sections_by_num.setdefault(num, []).append(sec)

    def _sec_sort_key(num_str: str) -> tuple[int, str]:
        import re as _re
        m = _re.match(r"(\d+)", num_str)
        if m:
            base = int(m.group(1))
            suffix = num_str[m.end() :].strip()
        else:
            base = 10**9
            suffix = num_str
        return (base, suffix)

    ordered_nums = sorted(sections_by_num.keys(), key=_sec_sort_key)

    def _sub_sort_key(sec: dict) -> tuple[int, int, str]:
        import re as _re
        sub = (sec.get("sub_section") or "").strip()
        if not sub:
            return (0, 0, "")
        m = _re.match(r"\(?(\d+)\)?\s*([A-Za-z]*)", sub)
        if m:
            num = int(m.group(1)) if m.group(1) else 0
            suf = m.group(2) or ""
            return (1, num, suf)
        return (1, 0, sub)

    def _render_text_block(text: str) -> str:
        """
        Render a section's text into HTML paragraphs.

        Normalises patterns where section / sub-section markers are on their own
        line and the substantive text starts on the next line, e.g.:
          "11.\nGrant of probate..."   â†’ "11. Grant of probate..."
          "(a)\nany person appears..." â†’ "(a) any person appears..."
        """
        import re as _re
        t = text or ""

        # First, join marker-only lines with the following content line.
        lines = t.splitlines()
        joined_lines: list[str] = []
        i = 0
        sec_pat = _re.compile(r"^\s*\d+[A-Za-z]*\.\s*$")              # 11. / 11A.
        sub_pat = _re.compile(r"^\s*\([A-Za-z0-9ivxIVX]+\)\s*$")      # (a), (b), (i), (ii), etc.
        while i < len(lines):
            line = lines[i]
            if (sec_pat.match(line) or sub_pat.match(line)) and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Only join when the next line has substantive text.
                if next_line.strip():
                    joined_lines.append(line.strip() + " " + next_line.lstrip())
                    i += 2
                    continue
            joined_lines.append(line)
            i += 1

        normalised = "\n".join(joined_lines)

        # Split into paragraphs on double newlines; preserve remaining single
        # newlines as <br>.
        paras = [s for s in normalised.split("\n\n") if s.strip()]
        if not paras:
            return ""
        html_parts: list[str] = []
        for para in paras:
            safe = _html_escape(para)
            safe = safe.replace("\n", "<br />")
            html_parts.append(f"<p>{safe}</p>")
        return "\n".join(html_parts)

    sections_html: list[str] = []
    for num in ordered_nums:
        group = sections_by_num[num]
        group_sorted = sorted(group, key=_sub_sort_key)
        first = group_sorted[0]
        sec_title = (
            (first.get("section_title") or first.get("title") or "").strip()
            or (first.get("text") or "").split("\n", 1)[0].strip()
        )
        header = f"Section {num}"
        if sec_title:
            header = f"{header}. {sec_title}"
        body_text = "\n\n".join((sec.get("text") or "").strip() for sec in group_sorted if (sec.get("text") or "").strip())
        body_html = _render_text_block(body_text)
        sections_html.append(
            f"<section class='bare-section'>"
            f"<h2>{_html_escape(header)}</h2>"
            f"{body_html}"
            f"</section>"
        )

    sections_joined = "\n".join(sections_html) if sections_html else "<p>No sections found in this act.</p>"

    html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{_html_escape(title)}</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 16px;
        background: #f7f7f8;
        color: #222;
      }}
      h1 {{
        font-size: 1.4rem;
        margin-bottom: 12px;
      }}
      h2 {{
        font-size: 1.05rem;
        margin-top: 18px;
        margin-bottom: 6px;
      }}
      .meta {{
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #555;
      }}
      .bare-section {{
        padding-bottom: 12px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 12px;
      }}
      .bare-section:last-of-type {{
        border-bottom: none;
      }}
      p {{
        font-size: 0.92rem;
        line-height: 1.5;
      }}
    </style>
  </head>
  <body>
    <h1>{_html_escape(title)}</h1>
    <div class="meta">
      Source: legal_database/json_output (bare act JSON)
      {"&nbsp;â€¢&nbsp;Year: " + _html_escape(str(year)) if year else ""}
    </div>
    {sections_joined}
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/caselaws/view", response_class=HTMLResponse)
def caselaws_view(
    file: str = Query(None, description="Path relative to raw_data/ e.g. CaseLaws/Supreme Court/case.txt"),
    name: str = Query(None, description="Legacy: case name (fallback to JSON output)"),
):
    """Render a case law as HTML. Prefers file= (raw txt); falls back to name= (JSON output)."""
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="Requires NYAYMALAW_DATA_SOURCE=legal_database")

    if file:
        return _serve_raw_txt_as_html(file, name or file.split("/")[-1])

    data = _get_json_from_legal_db(name or "", is_statute=False)
    if not data:
        raise HTTPException(status_code=404, detail="Case not found")

    title = (data.get("case_name") or name or "").strip() or "Case law"
    court = (data.get("court") or "").strip()
    date = (data.get("date_of_judgment") or "").strip()
    judges = data.get("judges") or []
    bench_type = (data.get("bench_type") or "").strip()
    citations = data.get("equivalent_citations") or data.get("reporter_citations") or []
    paragraphs = data.get("paragraphs") or []

    paragraphs_sorted = sorted(
        paragraphs,
        key=lambda p: int(p.get("paragraph_id") or 0),
    )

    def _render_para_text(text: str) -> str:
        parts = [t for t in (text or "").split("\n\n") if t.strip()]
        if not parts:
            return ""
        html_parts: list[str] = []
        for part in parts:
            safe = _html_escape(part)
            safe = safe.replace("\n", "<br />")
            html_parts.append(f"<p>{safe}</p>")
        return "\n".join(html_parts)

    paras_html: list[str] = []
    for p in paragraphs_sorted:
        pid = p.get("paragraph_id")
        text = p.get("text") or ""
        body_html = _render_para_text(text)
        if not body_html:
            continue
        label = f"Â¶ {pid}" if pid is not None else "Â¶"
        paras_html.append(
            f"<article class='case-paragraph'>"
            f"<div class='para-label'>{_html_escape(label)}</div>"
            f"<div class='para-body'>{body_html}</div>"
            f"</article>"
        )

    paras_joined = "\n".join(paras_html) if paras_html else "<p>No paragraphs available for this judgment.</p>"

    judges_str = ", ".join(judges) if judges else ""
    citations_str = "; ".join(citations) if citations else ""

    html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{_html_escape(title)}</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 16px;
        background: #f7f7f8;
        color: #222;
      }}
      h1 {{
        font-size: 1.4rem;
        margin-bottom: 12px;
      }}
      .meta {{
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #555;
      }}
      .meta b {{
        font-weight: 600;
      }}
      .case-paragraph {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 8px 12px;
        padding: 8px 0;
        border-bottom: 1px solid #e0e0e0;
      }}
      .case-paragraph:last-of-type {{
        border-bottom: none;
      }}
      .para-label {{
        font-size: 0.8rem;
        color: #777;
        min-width: 48px;
      }}
      .para-body p {{
        margin: 0 0 6px 0;
        font-size: 0.95rem;
        line-height: 1.5;
      }}
      .para-body p:last-child {{
        margin-bottom: 0;
      }}
    </style>
  </head>
  <body>
    <h1>{_html_escape(title)}</h1>
    <div class="meta">
      {f"<b>Court:</b> {_html_escape(court)}<br />" if court else ""}
      {f"<b>Date:</b> {_html_escape(date)}<br />" if date else ""}
      {f"<b>Bench:</b> {_html_escape(judges_str)}" if judges_str else ""}
      {f" &nbsp;&nbsp;({_html_escape(bench_type)})" if bench_type else ""}
      {f"<br /><b>Citations:</b> {_html_escape(citations_str)}" if citations_str else ""}
    </div>
    {paras_joined}
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/bareacts/download")
def bareacts_download(
    name: str = Query(..., description="Filename of the bare act to download"),
    inline: bool = Query(False, description="If true, open in browser (inline) instead of download"),
):
    """Serve a bare act file for download or inline view. Name must be a safe filename (no path traversal)."""
    # Normalize: strip and take basename so we accept names with or without path/whitespace
    base = os.path.basename(name).strip() if name else ""
    if not base or ".." in base or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_bare_act_path(base)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=filename, media_type=media_type)
        # inline: open in new tab (browser displays PDF); attachment: download
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not serve file: {e!s}")


# ---------------------------------------------------------------------------
# Case laws list and download (same pattern as bare acts)
# ---------------------------------------------------------------------------

def _list_case_laws_from_disk() -> list[str]:
    """List PDF/text filenames from CaseLaws directory."""
    if not os.path.isdir(_CASELAW_DIR):
        return []
    out = []
    for f in os.listdir(_CASELAW_DIR):
        path = os.path.join(_CASELAW_DIR, f)
        if os.path.isfile(path) and (f.lower().endswith(".pdf") or f.lower().endswith(".txt")):
            out.append(f)
    return sorted(out)


def _resolve_case_law_path(base: str):
    """Return absolute path to file in CaseLaws if it exists, else None (case-insensitive)."""
    base = (base or "").strip()
    if not base:
        return None
    lookup = base if base.lower().endswith((".pdf", ".txt")) else base + ".pdf"
    path = os.path.join(_CASELAW_DIR, lookup)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_CASELAW_DIR):
        for f in os.listdir(_CASELAW_DIR):
            if f.lower() == lookup.lower():
                return os.path.abspath(os.path.join(_CASELAW_DIR, f))
    return None


def _list_case_laws_from_disk_or_json() -> list[str]:
    """Return case names; from json_output when USE_LEGAL_DATABASE, else from disk."""
    if _USE_LEGAL_DATABASE:
        return _list_case_laws_from_json_output()
    return _list_case_laws_from_disk()


@app.get("/caselaws/list")
def caselaws_list():
    """Return list of case law names (from legal_database/json_output or data/CaseLaws)."""
    files = _list_case_laws_from_disk_or_json()
    return {"cases": files}


@app.get("/caselaws/library")
def caselaws_library():
    """
    Return case laws grouped by court, using the legal_database/json_output
    case-law JSONs when USE_LEGAL_DATABASE is enabled.

    Response shape:
      {
        "courts": [
          {"name": "Supreme Court", "cases": [...]},
          {"name": "Telangana HC", "cases": [...]},
          {"name": "TG HC", "cases": [...]}
        ]
      }
    """
    global _caselaws_library_cache

    if not _USE_LEGAL_DATABASE:
        files = _list_case_laws_from_disk_or_json()
        return {"courts": [{"name": "TG HC", "cases": files}]}

    if _caselaws_library_cache is None:
        _caselaws_library_cache = _group_case_laws_by_court()

    courts = [
        {"name": name, "cases": cases}
        for name, cases in sorted(_caselaws_library_cache.items(), key=lambda kv: kv[0].lower())
    ]
    return {"courts": courts}


@app.get("/caselaws/download")
def caselaws_download(
    name: str = Query(..., description="Filename of the case law to download"),
    inline: bool = Query(False, description="If true, open in browser (inline) instead of download"),
):
    """Serve a case law file for download or inline view."""
    base = os.path.basename(name).strip() if name else ""
    if not base or ".." in base or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_case_law_path(base)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=filename, media_type=media_type)
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/caselaws/most_cited", response_class=HTMLResponse)
def caselaws_most_cited():
    """
    Render a single HTML table of the 200 most-cited cases from
    legal_database/top_200_cited_cases.jsonl.

    Each JSON field becomes a column; each case is a row. The case_name
    column is hyperlinked to the underlying case-law HTML view when a
    json_base identifier is available.
    """
    path = os.path.join(_LEGAL_DATABASE_DIR, "top_200_cited_cases.jsonl")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Most-cited cases file not found")

    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(status_code=404, detail="No most-cited cases available")

    # Use keys from first row as canonical columns (keeps table manageable).
    first = rows[0]
    columns = list(first.keys())

    # Build HTML table (compact, with expandable text cells)
    def esc(val: str) -> str:
        return _html_escape(str(val)) if val is not None else ""

    header_cells_parts = []
    for col in columns:
        col_label = esc(col)
        if col == "case_name":
            header_cells_parts.append(f"<th class='col-case-name'>{col_label}</th>")
        elif col in ("disputes", "key_arguments", "reasoning"):
            header_cells_parts.append(f"<th class='col-long'>{col_label}</th>")
        else:
            header_cells_parts.append(f"<th class='col-meta'>{col_label}</th>")
    header_cells = "".join(header_cells_parts)

    body_rows: list[str] = []
    for idx, r in enumerate(rows):
        cells: list[str] = []
        for col in columns:
            val = r.get(col)
            if col == "case_name":
                base = r.get("json_base") or r.get("case_id") or ""
                base = str(base).strip()
                if base:
                    href = f"/caselaws/view?name={base}"
                    cells.append(
                        f"<td class='col-case-name'><a href='{esc(href)}' target='_blank' rel='noopener noreferrer'>{esc(val)}</a></td>"
                    )
                else:
                    cells.append(f"<td class='col-case-name'>{esc(val)}</td>")
            elif col in ("disputes", "key_arguments", "reasoning"):
                text = esc(val) if val is not None else ""
                cells.append(
                    "<td class='col-long'>"
                    f"<div class='cell-text truncated' data-row='{idx}' data-col='{esc(col)}'>{text}</div>"
                    f"<button type='button' class='expand-btn' data-row='{idx}' aria-label='Expand row'>â¤¢</button>"
                    "</td>"
                )
            else:
                cells.append(f"<td class='col-meta'>{esc(val)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
<table>
  <thead>
    <tr>{header_cells}</tr>
  </thead>
  <tbody>
    {''.join(body_rows)}
  </tbody>
</table>
"""

    html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Most cited case laws</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 16px;
        background: #f7f7f8;
        color: #222;
      }}
      h1 {{
        font-size: 1.4rem;
        margin-bottom: 12px;
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.8rem;
        table-layout: fixed;
      }}
      th, td {{
        border: 1px solid #ddd;
        padding: 4px 6px;
        vertical-align: top;
      }}
      th {{
        background: #f0f0f3;
        position: sticky;
        top: 0;
        z-index: 1;
        text-align: left;
      }}
      tr:nth-child(even) td {{
        background: #fafafa;
      }}
      .col-meta {{
        width: 90px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .col-case-name {{
        width: 220px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .col-long {{
        width: 28%;
      }}
      .cell-text {{
        display: block;
        line-height: 1.35;
      }}
      .cell-text.truncated {{
        max-height: 4.05em; /* ~3 lines */
        overflow: hidden;
      }}
      .expand-btn {{
        margin-top: 4px;
        padding: 0 4px;
        font-size: 0.7rem;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: #fff;
        cursor: pointer;
      }}
      .expand-btn:hover {{
        background: #f0f0f3;
      }}
      a {{
        color: #0b5fff;
        text-decoration: none;
      }}
      a:hover {{
        text-decoration: underline;
      }}
    </style>
  </head>
  <body>
    <h1>Most cited case laws (Top 200)</h1>
    {table_html}
    <script>
      (function () {{
        let expandedRow = null;
        function setRowState(rowId, expand) {{
          const cells = document.querySelectorAll(".cell-text[data-row='" + rowId + "']");
          cells.forEach((el) => {{
            if (expand) {{
              el.classList.remove("truncated");
            }} else {{
              el.classList.add("truncated");
            }}
          }});
          const buttons = document.querySelectorAll(".expand-btn[data-row='" + rowId + "']");
          buttons.forEach((btn) => {{
            btn.textContent = expand ? "â¤¡" : "â¤¢";
          }});
        }}
        document.addEventListener("click", function (e) {{
          const btn = e.target.closest(".expand-btn");
          if (!btn) return;
          const rowId = btn.getAttribute("data-row");
          if (!rowId) return;
          if (expandedRow !== null && expandedRow !== rowId) {{
            setRowState(expandedRow, false);
          }}
          if (expandedRow === rowId) {{
            setRowState(rowId, false);
            expandedRow = null;
          }} else {{
            setRowState(rowId, true);
            expandedRow = rowId;
          }}
        }});
      }})();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


def _normalize_content(c):
    """Ensure content is string for backend processing."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "text" in c:
        return c["text"]
    if isinstance(c, dict) and c.get("type") == "final_opinion":
        return c.get("opinionText") or ""
    if isinstance(c, dict) and c.get("type") == "results":
        for p in c.get("parts", []):
            if p.get("type") == "explanation":
                return p.get("text", "[Response]")
        return "[Legal research response]"
    return str(c) if c else ""


async def _stream_sse_queue(queue: Queue, loop):
    """
    Drain a worker queue into SSE events, with heartbeats so the connection feels
    live even while retrieval is still in flight.
    """
    while True:
        try:
            kind, payload = await loop.run_in_executor(None, lambda: queue.get(timeout=1.0))
        except Empty:
            yield ": ping\n\n"
            continue
        except Exception:
            break

        if kind == "result":
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
            break
        if kind == "progress":
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            continue
        if kind == "step":
            yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            continue
        if kind == "token":
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"


_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB hard cap


@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(_require_auth),
):
    """
    Extract plain text from an uploaded document or image.
    Supported formats:
      - Text-based PDF (.pdf)          → pypdf text extraction
      - Scanned/image-based PDF (.pdf) → pymupdf renders pages → OpenAI Vision OCR
      - Word document (.docx, .doc)    → python-docx paragraph extraction
      - Images (.jpg, .jpeg, .png,
                .webp, .tiff, .bmp)   → OpenAI Vision OCR directly

    Returns {"text": str, "filename": str, "char_count": int, "method": str}.
    The caller injects the text into the chat composer for review before submitting.
    Requires authentication. Maximum upload size: 20 MB.
    """
    import io, base64
    from platform_pkg.llm import ocr_pages_with_vision

    _IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp"}
    _IMAGE_MIME = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff",
        "bmp": "image/bmp",
    }
    # Characters per page below this threshold → treat PDF as scanned
    _SCANNED_CHARS_THRESHOLD = 80

    filename = (file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {"pdf", "docx", "doc"} | _IMAGE_EXTS
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Accepted: PDF (.pdf), Word (.docx), "
                "or image (.jpg, .jpeg, .png, .webp, .tiff, .bmp)."
            ),
        )

    content = await file.read()

    # Enforce 20 MB size cap — reject before any parsing
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(content) // (1024 * 1024)} MB). "
                "Maximum upload size is 20 MB."
            ),
        )
    text = ""
    method = "text"

    try:
        # ── Image files ───────────────────────────────────────────────────────
        if ext in _IMAGE_EXTS:
            mime = _IMAGE_MIME.get(ext, "image/png")
            b64 = base64.b64encode(content).decode()
            text = ocr_pages_with_vision([(b64, mime)])
            method = "vision_ocr"

        # ── PDF ───────────────────────────────────────────────────────────────
        elif ext == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            extracted = "\n\n".join(p.strip() for p in pages_text if p.strip())

            # Decide if scanned: average chars per page is very low
            num_pages = max(len(reader.pages), 1)
            avg_chars = len(extracted) / num_pages
            if avg_chars >= _SCANNED_CHARS_THRESHOLD:
                text = extracted
                method = "text"
            else:
                # Render each page to an image and OCR via vision
                import fitz  # pymupdf
                doc = fitz.open(stream=content, filetype="pdf")
                images_b64: list[tuple[str, str]] = []
                for page in doc:
                    mat = fitz.Matrix(2.0, 2.0)  # 2× zoom → better OCR accuracy
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    images_b64.append((base64.b64encode(img_bytes).decode(), "image/png"))
                doc.close()
                text = ocr_pages_with_vision(images_b64)
                method = "vision_ocr"

        # ── Word document ─────────────────────────────────────────────────────
        else:
            import docx as _docx
            doc = _docx.Document(io.BytesIO(content))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            method = "text"

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Document extraction failed for %s: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the document. The file may be corrupted or unsupported.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text found. For scanned documents this usually means the "
                "OpenAI Vision API call failed — check your OPENAI_API_KEY and try again."
            ),
        )
    return {"text": text.strip(), "filename": filename, "char_count": len(text), "method": method}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Interactive chat endpoint.
    Phase: fact_collection | response_generation
    """
    message = (request.message or "").strip()
    conv = [
        {"role": m.role, "content": _normalize_content(m.content)}
        for m in request.conversation
    ]
    result = process_chat(
        conversation=conv,
        current_message=request.message,
        phase=request.phase,
        facts_summary=request.facts_summary,
    )

    # If fact collection complete, trigger response generation
    if result["phase"] == "response_generation" and result.get("facts_summary"):
        resp_result = process_chat(
            conversation=conv,
            current_message=result["facts_summary"],
            phase="response_generation",
            facts_summary=result["facts_summary"],
            intent=result.get("intent", "legal_opinion"),
            document_types=result.get("document_types", "both"),
            search_strategy=result.get("search_strategy", "local_then_web"),
            result_count=result.get("result_count"),
            analysis_mode=result.get("analysis_mode"),
        )
        return resp_result

    return result


@app.post("/submit_case")
def submit_case(request: SubmitCaseRequest, user: dict = Depends(_require_auth)):
    """
    Initial case submission (await_facts). Frontend sends { text }.
    Returns status + next_question | opinion_text so the UI can continue the flow.
    """
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not text:
        return {
            "status": "question",
            "next_question": "",
            "retrieved": [],
        }
    try:
        workflow_state = _normalize_workflow_state(request.workflowState)
        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv,
            current_message=text,
            phase="fact_collection",
            facts_summary=None,
            chat_mode=mode,
            model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19: preserve pre-draft summary
            conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                chat_mode=mode,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=text, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@app.post("/interview_step")
def interview_step(request: InterviewStepRequest, user: dict = Depends(_require_auth)):
    """
    Follow-up answer in interview. Frontend sends { facts, qa_history } (qa_history includes the latest answer).
    Returns same shape as submit_case for consistent UI handling.
    """
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not qa_history:
        return {
            "status": "question",
            "next_question": "",
            "retrieved": [],
        }

    try:
        workflow_state = _normalize_workflow_state(request.workflowState)
        # Build conversation from interview transcript:
        # - user: overall facts
        # - assistant/user alternation for each Q/A turn
        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})

        current_message = qa_history[-1].answer if qa_history else ""
        result = process_chat(
            conversation=conv,
            current_message=current_message,
            phase="fact_collection",
            facts_summary=None,
            chat_mode=mode,
            model_override=model_override,
            workflow_state=workflow_state,
        )

        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19
            conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                chat_mode=mode,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )

        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=facts, session_ref=str(user.get("id", "")))

        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@app.post("/conversation/continue")
def continue_chat(request: ContinueChatRequest, user: dict = Depends(_require_auth)):
    """
    Continue a conversation from chat history. Sends full conversation + new message
    so the LLM has full context. Returns same shape as submit_case / interview_step.
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not message:
        return {
            "status": "question",
            "next_question": "",
            "retrieved": [],
        }
    try:
        conv = [
            {"role": m.role, "content": _normalize_content(m.content)}
            for m in (request.conversation or [])
        ]
        workflow_state = _normalize_workflow_state(request.workflowState, request.conversation)
        result = process_chat(
            conversation=conv,
            current_message=message,
            phase="fact_collection",
            facts_summary=None,
            chat_mode=mode,
            model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19
            conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": pre_draft_msg}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                chat_mode=mode,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=message, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


def _run_continue_chat_with_progress(conv: list, message: str, queue: Queue, user_id: str, mode: str | None = None, model_override: str | None = None, workflow_state: dict | None = None) -> None:
    """Run the same logic as continue_chat, pushing progress to queue and finally the result."""
    try:
        queue.put(("step", {"message": "Starting analysis of your latest message", "icon": ""}))
        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))
        def step_callback(step_data: dict):
            queue.put(("step", step_data))
        def token_callback(token: str):
            queue.put(("token", {"content": token}))

        result = process_chat(
            conversation=conv,
            current_message=message,
            phase="fact_collection",
            facts_summary=None,
            progress_callback=progress_callback,
            chat_mode=mode,
            step_callback=step_callback,
            token_callback=token_callback,
            model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19
            conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": pre_draft_msg}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback,
                chat_mode=mode,
                step_callback=step_callback,
                token_callback=token_callback,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=message, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
    except Exception as e:
        logger.exception("Stream continue_chat failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_submit_case_with_progress(text: str, queue: Queue, user_id: str, mode: str | None = None, model_override: str | None = None, workflow_state: dict | None = None) -> None:
    """Run submit_case logic with progress streaming."""
    try:
        queue.put(("step", {"message": "Reviewing the facts you shared", "icon": ""}))
        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))
        def step_callback(step_data: dict):
            queue.put(("step", step_data))
        def token_callback(token: str):
            queue.put(("token", {"content": token}))

        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv,
            current_message=text,
            phase="fact_collection",
            facts_summary=None,
            progress_callback=progress_callback,
            chat_mode=mode,
            step_callback=step_callback,
            token_callback=token_callback,
            model_override=model_override,
            workflow_state=workflow_state,
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19
            conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback,
                chat_mode=mode,
                step_callback=step_callback,
                token_callback=token_callback,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=text, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
    except Exception as e:
        logger.exception("Stream submit_case failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_interview_step_with_progress(facts: str, qa_history: list, queue: Queue, user_id: str, mode: str | None = None, model_override: str | None = None, workflow_state: dict | None = None) -> None:
    """Run interview_step logic with progress streaming."""
    try:
        t_total = time.perf_counter()
        queue.put(("step", {"message": "Reviewing your latest answer", "icon": ""}))

        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))
        def step_callback(step_data: dict):
            queue.put(("step", step_data))
        def token_callback(token: str):
            queue.put(("token", {"content": token}))

        t_conv = time.perf_counter()
        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})
        current_message = qa_history[-1].answer if qa_history else ""
        _log_pipeline_step(
            "interview_step.build_conversation",
            (time.perf_counter() - t_conv) * 1000,
            f"qa_turns={len(qa_history)}",
        )
        t_phase = time.perf_counter()
        result = process_chat(
            conversation=conv,
            current_message=current_message,
            phase="fact_collection",
            facts_summary=None,
            progress_callback=progress_callback,
            chat_mode=mode,
            step_callback=step_callback,
            token_callback=token_callback,
            model_override=model_override,
            workflow_state=workflow_state,
        )
        _log_pipeline_step(
            "interview_step.process_chat.fact_collection",
            (time.perf_counter() - t_phase) * 1000,
            f"phase={result.get('phase')}",
        )
        pre_draft_msg = ""
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            pre_draft_msg = result.get("message", "")  # Item 19
            conv = conv + [{"role": "assistant", "content": pre_draft_msg}]
            t_phase = time.perf_counter()
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
                progress_callback=progress_callback,
                chat_mode=mode,
                step_callback=step_callback,
                token_callback=token_callback,
                model_override=model_override,
                workflow_state=workflow_state,
                analysis_mode=result.get("analysis_mode"),
            )
            _log_pipeline_step(
                "interview_step.process_chat.response_generation",
                (time.perf_counter() - t_phase) * 1000,
                f"phase={result.get('phase')}",
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=facts, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result, pre_draft_msg=pre_draft_msg)))
        _log_pipeline_step(
            "interview_step.total",
            (time.perf_counter() - t_total) * 1000,
            f"final_phase={result.get('phase')}",
        )
    except Exception as e:
        logger.exception("Stream interview_step failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


@app.post("/submit_case/stream")
async def submit_case_stream(request: SubmitCaseRequest, user: dict = Depends(_require_auth)):
    """Same as /submit_case but streams progress via Server-Sent Events."""
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not text:
        return JSONResponse(
            status_code=400,
            content={"detail": "text is required"},
        )
    queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id", _ANONYMOUS_EMAIL)
    workflow_state = _normalize_workflow_state(request.workflowState)

    def run_in_thread():
        _run_submit_case_with_progress(text, queue, user_id, mode, model_override, workflow_state=workflow_state)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/interview_step/stream")
async def interview_step_stream(request: InterviewStepRequest, user: dict = Depends(_require_auth)):
    """Same as /interview_step but streams progress via Server-Sent Events."""
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not qa_history:
        return JSONResponse(
            status_code=400,
            content={"detail": "qa_history is required"},
        )
    queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id", _ANONYMOUS_EMAIL)
    workflow_state = _normalize_workflow_state(request.workflowState)

    def run_in_thread():
        _run_interview_step_with_progress(facts, qa_history, queue, user_id, mode=mode, model_override=model_override, workflow_state=workflow_state)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/conversation/continue/stream")
async def continue_chat_stream(request: ContinueChatRequest, user: dict = Depends(_require_auth)):
    """
    Same as /conversation/continue but streams progress via Server-Sent Events.
    Events: "progress" (progress snapshot JSON), "done" (final UI result JSON).
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    mode = (request.mode or "").strip().lower() or None
    model_override = (request.model_override or "").strip() or None
    if not message:
        return JSONResponse(
            status_code=400,
            content={"detail": "message is required"},
        )
    conv = [
        {"role": m.role, "content": _normalize_content(m.content)}
        for m in (request.conversation or [])
    ]
    queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id", _ANONYMOUS_EMAIL)

    workflow_state = _normalize_workflow_state(request.workflowState, request.conversation)

    def run_in_thread():
        _run_continue_chat_with_progress(conv, message, queue, user_id, mode, model_override, workflow_state)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator Agent endpoint — /agent/stream
# ---------------------------------------------------------------------------
# This replaces the hardcoded /conversation/continue/stream two-phase loop.
# The orchestrator holds all 8 tools and decides dynamically what to do.
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    message: str
    conversation: Optional[List] = Field(default_factory=list)
    workflowState: Optional[dict] = None


@app.post("/agent/stream")
async def agent_stream(request: AgentRequest, user: dict = Depends(_require_auth)):
    """
    Agentic endpoint. Streams SSE events from the OrchestratorAgent.

    Events:
      step  — progress label (tool being called)
      token — streamed text fragment
      done  — final result payload {reply, workflow_state, intake_state, session_id}
      error — non-fatal error message

    Replaces /conversation/continue/stream for agentic-mode clients.
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "message is required"})

    conv = [
        {"role": m["role"] if isinstance(m, dict) else m.role,
         "content": _normalize_content(m["content"] if isinstance(m, dict) else m.content)}
        for m in (request.conversation or [])
    ]
    workflow_state = dict(request.workflowState or {})
    queue: Queue = Queue()
    loop = asyncio.get_event_loop()

    def run_in_thread():
        try:
            from agents.orchestrator import OrchestratorAgent
            agent = OrchestratorAgent()
            for event in agent.run_stream(message, conv, workflow_state):
                ev_type = event.get("type", "step")
                if ev_type == "step":
                    queue.put(("step", {"message": event.get("message", "")}))
                elif ev_type == "token":
                    queue.put(("token", {"text": event.get("text", "")}))
                elif ev_type == "done":
                    queue.put(("result", event.get("payload", {})))
                    return
                elif ev_type == "error":
                    queue.put(("step", {"message": f"⚠ {event.get('message', 'Error')}"}))
            # If run_stream ended without a done event
            queue.put(("result", {"reply": "", "workflow_state": workflow_state}))
        except Exception as exc:
            logger.exception("agent_stream worker failed: %s", exc)
            queue.put(("result", {"reply": "An error occurred. Please try again.", "error": str(exc)}))

    import threading as _threading
    thread = _threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def event_generator():
        async for event in _stream_sse_queue(queue, loop):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- Tier / Freemium endpoints ----------

class UpgradeRequest(BaseModel):
    user_id: int
    tier: str = "premium"


@app.get("/user/tier")
def user_tier(user: dict = Depends(_user_from_token)):
    """Return the user's current tier, query usage, and feature access."""
    info = get_tier_info(user["id"])
    return info


@app.post("/admin/upgrade")
def admin_upgrade(request: UpgradeRequest):
    """
    Upgrade a user's tier. Placeholder for Razorpay webhook integration.
    In production, this should be authenticated with an admin token.
    """
    result = upgrade_user(request.user_id, request.tier)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Upgrade failed"))
    return result


# ---------- Health check endpoint ----------

@app.post("/admin/reset-rate-limit")
def reset_rate_limit():
    """Reset rate limit store (development only)."""
    global _rate_store
    _rate_store.clear()
    logger.info("Rate limit store cleared")
    return {"status": "ok", "message": "Rate limit store cleared"}


# ---------------------------------------------------------------------------
# Propose for Index â€” save web-sourced bare act sections to proposed_sections.json
# The user can then run scripts/index_proposed.py to add them to the local index.
# ---------------------------------------------------------------------------

class ProposeIndexRequest(BaseModel):
    section: dict   # Full bare-act section dict (act_name, section_number, full_text, url, â€¦)

@app.post("/propose_index")
async def propose_index(req: ProposeIndexRequest):
    """
    Save a web-sourced bare act section to proposed_sections.json so the user can
    add it to the local vector index via scripts/index_proposed.py.

    The file lives alongside the vector store:
        <VECTOR_STORE_DIR>/proposed_sections.json
    """
    from pathlib import Path as _Path
    import datetime as _dt

    vector_store_dir = _Path(_VECTOR_STORE).parent if _Path(_VECTOR_STORE).suffix else _Path(_VECTOR_STORE)
    proposed_path = vector_store_dir / "proposed_sections.json"

    # Load existing proposals (or start fresh)
    try:
        if proposed_path.exists():
            with open(proposed_path, encoding="utf-8") as f:
                proposals = json.load(f)
        else:
            proposals = []
    except Exception:
        proposals = []

    section = req.section
    # Deduplicate by (act_name, section_number) â€” don't add the same section twice
    act  = (section.get("act_name")     or "").strip().lower()
    sec  = (section.get("section_number") or "").strip().lower()
    already = any(
        (p.get("act_name","").strip().lower() == act and
         p.get("section_number","").strip().lower() == sec)
        for p in proposals
    )
    if already:
        return {"status": "already_proposed", "message": f"{section.get('act_name')} Â§{section.get('section_number')} already in proposal list"}

    # Strip internal pipeline keys before saving
    clean = {k: v for k, v in section.items() if not k.startswith("_")}
    clean["_proposed_at"] = _dt.datetime.utcnow().isoformat()

    proposals.append(clean)
    try:
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proposed_path, "w", encoding="utf-8") as f:
            json.dump(proposals, f, ensure_ascii=False, indent=2)
        logger.info("Proposed for index: %s Â§%s â†’ %s", section.get("act_name"), section.get("section_number"), proposed_path)
        return {
            "status": "proposed",
            "message": f"Saved to {proposed_path.name}. Run scripts/index_proposed.py to add to local index.",
            "total_proposed": len(proposals),
        }
    except Exception as e:
        logger.error("Failed to write proposed_sections.json: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not save proposal: {e}")


# ---------------------------------------------------------------------------
# Eval Files â€” serve eval JSON results for UI display
# ---------------------------------------------------------------------------

EVAL_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "results")
EVAL_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "figures")


@app.get("/eval/list")
def eval_list_files():
    """List available eval JSON files (batch results, compare, threshold sweep)."""
    files = []
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return {"files": []}
    for root, _dirs, fnames in os.walk(EVAL_RESULTS_DIR):
        rel_root = os.path.relpath(root, EVAL_RESULTS_DIR)
        if rel_root == ".":
            rel_root = ""
        for name in fnames:
            if name.endswith(".json"):
                path = os.path.join(rel_root, name) if rel_root else name
                path = path.replace("\\", "/")
                files.append({"path": path, "name": name})
    return {"files": sorted(files, key=lambda x: x["path"])}


@app.get("/eval/file")
def eval_get_file(path: str = Query(..., description="Relative path to eval JSON file")):
    """Serve a single eval JSON file by path."""
    path = path.lstrip("/").replace("..", "")
    full = os.path.normpath(os.path.join(EVAL_RESULTS_DIR, path))
    base = os.path.realpath(EVAL_RESULTS_DIR)
    if not os.path.isfile(full) or not os.path.realpath(full).startswith(base):
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    try:
        with open(full, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.exception("Eval file read failed: %s", path)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/eval/figures/list")
def eval_list_figures():
    """List available eval figure files (PNG, JPG, etc.) in eval/figures."""
    figures = []
    if not os.path.isdir(EVAL_FIGURES_DIR):
        return {"figures": []}
    for name in os.listdir(EVAL_FIGURES_DIR):
        full = os.path.join(EVAL_FIGURES_DIR, name)
        if os.path.isfile(full) and name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            figures.append({"path": name, "name": name})
    return {"figures": sorted(figures, key=lambda x: x["name"])}


@app.get("/eval/figures/file")
def eval_get_figure(path: str = Query(..., description="Figure filename (e.g. fig_threshold_curve.png)")):
    """Serve a single eval figure file by name."""
    path = path.lstrip("/").replace("..", "").replace("\\", "/")
    if "/" in path:
        return JSONResponse(status_code=400, content={"detail": "path must be a filename"})
    full = os.path.normpath(os.path.join(EVAL_FIGURES_DIR, path))
    base = os.path.realpath(EVAL_FIGURES_DIR)
    if not os.path.isfile(full) or not os.path.realpath(full).startswith(base):
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    ext = os.path.splitext(path)[1].lower()
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(full, filename=path, media_type=media_type)


# ---------------------------------------------------------------------------
# Docs â€” serve architecture markdown for UI
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
ARCHITECTURE_MD = os.path.join(DOCS_DIR, "CREWAI_MULTI_AGENT_ARCHITECTURE.md")


@app.get("/docs/architecture")
def docs_architecture():
    """Serve the CrewAI multi-agent architecture doc as markdown text for the UI."""
    if not os.path.isfile(ARCHITECTURE_MD):
        return JSONResponse(status_code=404, content={"detail": "Architecture doc not found"})
    try:
        with open(ARCHITECTURE_MD, encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        logger.exception("Docs architecture read failed: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ---------------------------------------------------------------------------
# Feedback / AI Gate endpoints
# ---------------------------------------------------------------------------

@app.post("/feedback/review/{case_id}")
def feedback_review(case_id: str, user: dict = Depends(_require_auth)):
    """
    Trigger AI Gate review for a logged interaction.

    POST /feedback/review/NM-20260301-004

    Reads the row for case_id from the Feedback Log workbook, sends the
    model output to the configured LLM for review against the Golden Rules,
    and writes results back into columns Kâ€“R and Vâ€“W.

    Returns the review result as JSON.
    Requires authentication (any valid user token).
    """
    if not _FEEDBACK_ENABLED or not _run_ai_review:
        raise HTTPException(
            status_code=503,
            detail="Feedback system not available (feedback_logger / ai_reviewer import failed at startup).",
        )
    try:
        review = _run_ai_review(case_id)
        return {"success": True, "case_id": case_id, "review": review}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Feedback log not found: {e}")
    except Exception as e:
        logger.error("AI Gate review failed for %s: %s", case_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI review failed: {e}")


@app.get("/feedback/status")
def feedback_status():
    """Return whether the feedback logging system is active."""
    return {
        "feedback_enabled": _FEEDBACK_ENABLED,
        "log_path": (
            os.environ.get("FEEDBACK_LOG_PATH", "")
            or getattr(__import__("config"), "FEEDBACK_LOG_PATH", "not configured")
        ),
    }


# ---------------------------------------------------------------------------
# Feedback Log Excel â†” HTML sync
# ---------------------------------------------------------------------------

def _feedback_log_path():
    from config import FEEDBACK_LOG_PATH
    return FEEDBACK_LOG_PATH


@app.get("/feedback_log/data")
def feedback_log_get_data():
    """
    Return the Feedback Log Excel as JSON for HTML to load (Excel â†’ HTML sync).
    Rows are returned as arrays of cell values; first 3 rows are header/notes.
    """
    import pandas as pd
    path = _feedback_log_path()
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Feedback log Excel file not found.")
    try:
        df = pd.read_excel(path, sheet_name=0, header=None)
        rows = []
        for i in range(len(df)):
            row = df.iloc[i].tolist()
            rows.append(["" if (x != x or x is None) else str(x).strip() for x in row])
        return {"rows": rows}
    except Exception as e:
        logger.exception("feedback_log GET data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class FeedbackLogSaveRequest(BaseModel):
    rows: List[List[str]] = []  # array of rows, each row array of cell values


class ResponseFeedbackRequest(BaseModel):
    chat_id: str = ""
    message_id: str = ""
    rating: str = ""
    reason_tags: List[str] = []
    free_text: str = ""
    assistant_text: str = ""
    user_message: str = ""
    stage: str = ""
    response_type: str = ""
    model_used: str = ""
    latency_ms: Optional[float] = None
    metadata: dict = {}


@app.post("/feedback_log/save")
def feedback_log_save(request: FeedbackLogSaveRequest):
    """
    Save table data from HTML to the Feedback Log Excel (HTML â†’ Excel sync).
    Expects rows: [ notes_row, data_row_1, ... ] (tbody only). Preserves Excel header rows 0â€“1.
    """
    import pandas as pd
    path = _feedback_log_path()
    if not path:
        raise HTTPException(status_code=500, detail="FEEDBACK_LOG_PATH not configured.")
    new_rows = getattr(request, "rows", []) or []
    if not new_rows:
        raise HTTPException(status_code=400, detail="No rows provided.")
    try:
        existing = pd.read_excel(path, sheet_name=0, header=None)
        # Keep first 2 rows (section headers + column names), replace from row 2 with payload
        header = existing.iloc[:2] if len(existing) >= 2 else existing
        new_df = pd.DataFrame(new_rows)
        out = pd.concat([header, new_df], ignore_index=True)
        out.to_excel(path, index=False, header=False)
        return {"message": "Saved", "rows": len(new_rows)}
    except Exception as e:
        logger.exception("feedback_log POST save: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/response/taxonomy")
def response_feedback_taxonomy():
    return {
        "ratings": ["good", "okay", "bad"],
        "reason_tags": list(RESPONSE_FEEDBACK_TAGS),
        "store_path": str(feedback_store_path()),
    }


@app.post("/feedback/response")
def submit_response_feedback(request: ResponseFeedbackRequest, user: dict = Depends(_require_auth)):
    message_id = (request.message_id or "").strip()
    rating = (request.rating or "").strip().lower()
    assistant_text = (request.assistant_text or "").strip()
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")
    if not rating:
        raise HTTPException(status_code=400, detail="rating is required")

    try:
        append_response_feedback(
            {
                "user_id": user.get("id"),
                "chat_id": (request.chat_id or "").strip(),
                "message_id": message_id,
                "rating": rating,
                "reason_tags": request.reason_tags or [],
                "free_text": (request.free_text or "").strip(),
                "assistant_text": assistant_text,
                "user_message": (request.user_message or "").strip(),
                "stage": (request.stage or "").strip(),
                "response_type": (request.response_type or "").strip(),
                "model_used": (request.model_used or "").strip(),
                "latency_ms": request.latency_ms,
                "metadata": request.metadata or {},
            }
        )
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("submit_response_feedback failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """
    Production health check â€” verifies Ollama, DB, and vector store.
    Returns 200 if all healthy, 503 if any critical service is down.
    """
    health = {"status": "healthy", "checks": {}}

    # 1. Ollama + model
    ollama = check_ollama_health()
    health["checks"]["ollama"] = ollama
    if not ollama.get("ollama_reachable"):
        health["status"] = "unhealthy"
    if not ollama.get("model_loaded"):
        health["status"] = "degraded" if health["status"] == "healthy" else health["status"]

    # 2. Database
    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        health["checks"]["database"] = {"reachable": True}
    except Exception as e:
        health["checks"]["database"] = {"reachable": False, "error": str(e)}
        health["status"] = "unhealthy"

    # 3. Vector store
    from config import VECTOR_STORE, BARE_INDEX_V2, CASE_INDEX_V2
    vs_exists = os.path.isdir(VECTOR_STORE)
    bare_index_exists = os.path.isfile(BARE_INDEX_V2)
    case_index_exists = os.path.isfile(CASE_INDEX_V2)
    health["checks"]["vector_store"] = {
        "directory_exists": vs_exists,
        "bare_acts_index": bare_index_exists,
        "case_laws_index": case_index_exists,
    }
    if not vs_exists:
        health["status"] = "degraded" if health["status"] == "healthy" else health["status"]

    status_code = 200 if health["status"] != "unhealthy" else 503
    return JSONResponse(content=health, status_code=status_code)


# ---------- Startup validation ----------

@app.on_event("startup")
async def startup_validation():
    """Run startup checks, perform critical warmup, then launch background warmup."""
    logger.info("=" * 60)
    logger.info("Nyaymalaw API v3.0.0 starting up")
    logger.info("=" * 60)

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
    return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
