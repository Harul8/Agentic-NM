import asyncio
import os
import json
import sqlite3
import hashlib
import logging
import secrets
import time
import traceback
from html import escape as _html_escape
from queue import Queue
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Query, HTTPException, Depends, Header, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law
from services.interactive_chat import process_chat
from services.response_generator_v2 import generate_response_v2
from llm.ollama_client import check_ollama_health, get_last_model_used

# Feedback logging (non-critical — import errors must not crash the server)
try:
    from services.feedback_logger import log_interaction as _log_interaction
    from services.ai_reviewer import run_ai_review as _run_ai_review
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

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="Nyaymalaw API", version="3.0.0")

# CORS: environment-aware — set ALLOWED_ORIGINS env var for production
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
_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "false").lower() == "true"  # Disabled by default for development
_rate_store: dict[str, list[float]] = defaultdict(list)

# Only rate-limit mutation endpoints (not health, static, etc.)
_RATE_LIMITED_PATHS = {"/submit_case", "/submit_case/stream", "/interview_step", "/interview_step/stream", "/conversation/continue", "/conversation/continue/stream", "/chat", "/chat/confirm-index", "/search"}

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
            "%s %s → %d (%.0fms)", method, path, response.status_code, elapsed_ms
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "%s %s → 500 (%.0fms) %s", method, path, elapsed_ms, exc
        )
        raise


# ---------------------------------------------------------------------------
# Global error handler — catch unhandled exceptions, return clean JSON
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
            "error_type": type(exc).__name__,
        },
    )


# Paths and DB (must be before auth routes) – use config for data root (e.g. Google Drive)
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

# ── In-memory cache for library endpoints ────────────────────────────────────
# _group_case_laws_by_court() opens every JSON file on disk; at 1000+ cases
# this takes ~2 minutes on first load.  Cache the result after the first call.
_caselaws_library_cache: dict[str, list[str]] | None = None
_bareacts_library_cache: dict[str, list[str]] | None = None


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


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
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                opinion_text TEXT NOT NULL DEFAULT '',
                retrieved_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
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
from services.tier_manager import (
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


class LoginRequest(BaseModel):
    email: str = ""
    password: str = ""


class ChatPayload(BaseModel):
    id: Optional[int] = None
    title: str = ""
    messages: list = []
    opinionText: str = ""
    retrieved: list = []
    createdAt: str = ""


# ---------- Auth & Chat history (backend storage) ----------

@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    email = (req.email or "").strip().lower()
    name = (req.name or "").strip()
    password = (req.password or "").strip()
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
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email, password_hash, name),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO sessions (user_id, token) VALUES (?, ?)", (user_id, token))
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
    password = (req.password or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    password_hash = _hash_password(password)
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, email, name FROM users WHERE email = ? AND password_hash = ?",
            (email, password_hash),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO sessions (user_id, token) VALUES (?, ?)", (row["id"], token))
        conn.commit()
        return {
            "success": True,
            "token": token,
            "user": {"email": row["email"], "name": row["name"]},
        }
    finally:
        conn.close()


@app.get("/chats")
def chats_list(user: dict = Depends(_user_from_token)):
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "title": r["title"],
                "messages": json.loads(r["messages_json"] or "[]"),
                "opinionText": r["opinion_text"] or "",
                "retrieved": json.loads(r["retrieved_json"] or "[]"),
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
def chats_upsert(payload: ChatPayload, user: dict = Depends(_user_from_token)):
    """Create or update a chat. If id is provided and exists for this user, update; else create with id or new id."""
    title = (payload.title or "").strip() or "Untitled chat"
    messages = payload.messages if isinstance(payload.messages, list) else []
    opinion_text = payload.opinionText or ""
    retrieved = payload.retrieved if isinstance(payload.retrieved, list) else []
    created_at = payload.createdAt or __import__("datetime").datetime.utcnow().isoformat() + "Z"
    messages_json = json.dumps(messages)
    retrieved_json = json.dumps(retrieved)
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
                    "UPDATE chats SET title = ?, messages_json = ?, opinion_text = ?, retrieved_json = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
                    (title, messages_json, opinion_text, retrieved_json, payload_id, user["id"]),
                )
                conn.commit()
                return {"id": payload_id, "title": title}
        chat_id = payload_id if payload_id is not None else int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM chats").fetchone()[0])
        conn.execute(
            "INSERT OR REPLACE INTO chats (id, user_id, title, messages_json, opinion_text, retrieved_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user["id"], title, messages_json, opinion_text, retrieved_json, created_at),
        )
        conn.commit()
        return {"id": chat_id, "title": title}
    finally:
        conn.close()


@app.get("/chats/{chat_id}")
def chat_get(chat_id: int, user: dict = Depends(_user_from_token)):
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, created_at FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {
            "id": row["id"],
            "title": row["title"],
            "messages": json.loads(row["messages_json"] or "[]"),
            "opinionText": row["opinion_text"] or "",
            "retrieved": json.loads(row["retrieved_json"] or "[]"),
            "createdAt": row["created_at"],
        }
    finally:
        conn.close()


@app.delete("/chats/{chat_id}")
def chat_delete(chat_id: str, user: dict = Depends(_user_from_token)):
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


class ConfirmIndexRequest(BaseModel):
    bare_acts: list[dict] = []
    case_laws: list[dict] = []
    facts_summary: str = ""


class SubmitCaseRequest(BaseModel):
    """Initial case submission - frontend sends { text, mode }."""
    text: str = ""
    # chat_mode: "legal_opinion" | "legal_research" | "general" (optional; default router when empty)
    mode: str | None = None


class QAPair(BaseModel):
    question: str = ""
    answer: str = ""


class InterviewStepRequest(BaseModel):
    """Follow-up answer in interview - frontend sends { facts, qa_history, bare_acts?, mode }."""
    facts: str = ""
    qa_history: list[QAPair] = []
    bare_acts: list[dict] = []   # populated when answering a bare-acts-phase follow-up question
    mode: str | None = None      # "legal_opinion" | "legal_research" | "general"


class ContinueChatRequest(BaseModel):
    """Continue a loaded chat - full conversation history + new user message."""
    conversation: list[ChatMessage] = []
    message: str = ""
    mode: str | None = None  # "legal_opinion" | "legal_research" | "general"


def _build_conv(messages: list[ChatMessage] | None) -> list[dict]:
    if not messages:
        return []
    return [{"role": m.role, "content": _normalize_content(m.content)} for m in messages]


# Path to vector store and BareActs directory – resolve from this file’s location


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


def _group_case_laws_by_court() -> dict[str, list[str]]:
    """
    Group case laws by court, using the "court" field from the case-law JSON.

    Returns a mapping like:
      {
        "Supreme Court": [...],
        "Telangana HC": [...],
        "TG HC": [...]
      }
    """
    groups: dict[str, list[str]] = {}
    if not _USE_LEGAL_DATABASE:
        return groups

    for path, base in _iter_caselaws_json_files():
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        court = (data.get("court") or "").lower()
        if "supreme" in court:
            label = "Supreme Court"
        elif "telangana" in court:
            label = "Telangana HC"
        else:
            # Bucket all remaining courts under a concise label for the UI
            label = "TG HC"
        groups.setdefault(label, []).append(base)

    for cases in groups.values():
        cases.sort()
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


def _group_bare_acts_by_jurisdiction() -> dict[str, list[str]]:
    """
    Group Bare Acts by top-level jurisdiction folder under json_output/BareActs.

    Example layout (mirrors legal_database/pipeline.py):
      json_output/BareActs/Telangana/1987_15_SomeAct.json
      json_output/BareActs/Union of India/2023_01_AnotherAct.json

    Returns:
      {"Telangana": [...], "Union of India": [...], ...}
    """
    groups: dict[str, list[str]] = {}
    if not _USE_LEGAL_DATABASE:
        return groups

    for rel, base in _iter_bareacts_json_files():
        # rel example: ".", "Telangana", "Union of India", "Telangana/Subdir"
        jurisdiction = ""
        if rel and rel != ".":
            jurisdiction = rel.split(os.sep, 1)[0]
        if not jurisdiction:
            jurisdiction = "Union of India"
        groups.setdefault(jurisdiction, []).append(base)

    for acts in groups.values():
        acts.sort()
    return groups


def _chat_error_fallback(detail: str = "") -> dict:
    """Return a safe 200 response when chat processing fails so frontend does not see 500.
    We try to generate a message from the LLM; if that also fails we use a minimal technical note."""
    try:
        from llm.ollama_client import ask_llm
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
        f"{ba.get('act_name','?')} § {ba.get('section_number','?')}"
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


def _fire_feedback_log_bare_acts(result: dict, facts: str, session_ref: str = "") -> None:
    """
    When phase is 'bare_acts_presented', log one row with timestamp, user input, and
    model output (disputes, sections, intro text as opinion, follow-up as additional info).
    So if the user closes the chat after only bare acts, the feedback log already has the row.
    """
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return
    if result.get("phase") != "bare_acts_presented":
        return
    import threading
    bare_acts = result.get("bare_acts") or []
    disputes_raw = result.get("disputes") or []
    disputes_list = [d.get("dispute", "") for d in disputes_raw if isinstance(d, dict) and d.get("dispute")]
    if not disputes_list:
        disputes_list = [str((result.get("facts_summary") or "")[:80])]
    sections_list = [
        f"{ba.get('act_name', '?')} § {ba.get('section_number', '?')}"
        for ba in bare_acts
    ]
    intro = (result.get("message") or "").strip() or "Here are the relevant bare act sections I found."
    followup = (result.get("followup_question") or "").strip()

    def _do_log():
        try:
            _log_interaction(
                facts=facts,
                followup_question=followup,
                disputes=disputes_list,
                sections=sections_list,
                case_laws=[],
                legal_opinion=intro,
                router_classification="Legal Opinion",
                session_ref=session_ref,
            )
        except Exception as _le:
            logger.warning("Feedback log (bare acts) failed (non-critical): %s", _le)

    t = threading.Thread(target=_do_log, daemon=True, name="feedback-log-bare-acts")
    t.start()


def _fire_feedback_log_research(result: dict, query: str, session_ref: str = "") -> None:
    """
    Log one feedback row for a raw /search (fusion research) call.

    The research result has no 'phase' or AI-generated opinion — it is a pure
    retrieval result from fuse_bare_act_and_case_law.  We map its fields directly:

      facts          → the search query string
      disputes       → [query[:80]]  (no dispute extraction on raw search)
      sections       → act_name § section_number  from bare_act_sections
      case_laws      → case_name / citation  from case_laws list
      legal_opinion  → ""  (raw retrieval, no opinion generated)
      followup       → ""
    """
    if not _FEEDBACK_ENABLED or not _log_interaction:
        return
    import threading

    bare_acts = result.get("bare_act_sections") or []
    sections_list = [
        f"{ba.get('act_name', '?')} § {ba.get('section_number', '?')}"
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


def _map_chat_result_to_ui(result: dict) -> dict:
    """Map process_chat result to the shape the frontend expects (status, next_question, etc.)."""
    phase = result.get("phase")
    response_type = result.get("response_type")  # "search_results", "lookup_results", "legal_opinion"

    if phase == "fact_collection":
        next_q = (result.get("message") or "").strip()
        if not next_q:
            next_q = "Could you tell me more about your legal query?"
        return {
            "status": "question",
            "next_question": next_q,
            "retrieved": result.get("response") or [],
        }
    if phase == "bare_acts_presented":
        # Intermediate phase: show retrieved bare acts with explanations + optional follow-up question
        bare_acts = result.get("bare_acts") or []
        disputes  = result.get("disputes") or []   # grouped for new per-dispute UI
        followup = (result.get("followup_question") or "").strip() or None
        intro = (result.get("message") or "").strip() or "Here are the relevant bare act sections I found."
        return {
            "status": "bare_acts_presented",
            "opinion_text": intro,
            "disputes": disputes,             # new: grouped [{id, dispute, sections}]
            "bare_acts": bare_acts,           # kept: flat list for Phase B backward compat
            "followup_question": followup,
            # Pass facts_summary so the frontend can echo it back in the next request
            "facts_summary": result.get("facts_summary") or "",
        }
    if phase == "confirm_materials":
        return {
            "needs_confirmation": True,
            "materials_to_confirm": result.get("materials_to_confirm"),
            "summary": result.get("message", ""),
        }
    if phase == "done" and result.get("response"):
        resp = result["response"]
        bare_acts = resp.get("bare_act_sections") or []
        case_laws = resp.get("case_laws") or []
        internet_case_laws = resp.get("internet_case_laws") or []
        all_case_laws = case_laws + internet_case_laws
        # Combine greeting/acknowledgment with explanation; avoid showing the same content twice
        greeting = (result.get("message") or "").strip()
        explanation = (resp.get("explanation") or "").strip()
        if greeting and explanation:
            # If they are the same or one contains the other, show only once (the longer)
            if greeting == explanation:
                combined_text = explanation
            elif greeting in explanation:
                combined_text = explanation
            elif explanation in greeting:
                combined_text = greeting
            else:
                combined_text = f"{greeting}\n\n{explanation}"
        else:
            combined_text = greeting or explanation
        # Ensure we never send an empty or trivial intro (e.g. just "⚖")
        if not combined_text or len(combined_text.strip()) < 20:
            combined_text = "Here’s what I found for your query. Below are the Supreme Court judgments and any relevant provisions."
        # If bare acts have nested case laws, don't return separate case_laws array to avoid duplicates
        # Case laws are now nested under bare_acts[].related_case_laws
        separate_case_laws = []
        if bare_acts and len(bare_acts) > 0:
            # Check if any bare act has nested case laws
            has_nested_case_laws = any(ba.get("related_case_laws") for ba in bare_acts)
            if not has_nested_case_laws:
                # Only return separate case_laws if no nested case laws exist
                separate_case_laws = all_case_laws
        
        out = {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": combined_text,
            "bare_acts": bare_acts,
            "case_laws": separate_case_laws,  # Empty if case laws are nested under bare acts
            "retrieved": all_case_laws + bare_acts,
            "progress": resp.get("progress"),  # Include progress tracking data
            "model_used": get_last_model_used(),
        }
        return out
    if phase == "done":
        return {
            "status": "done",
            "response_type": response_type or "legal_opinion",
            "opinion_text": (result.get("message") or "").strip() or "Your request has been processed.",
            "bare_acts": [],
            "case_laws": [],
            "retrieved": [],
            "model_used": get_last_model_used(),
        }
    next_q = (result.get("message") or "").strip()
    if not next_q:
        next_q = "Please continue or rephrase your question."
    return {
        "status": "question",
        "next_question": next_q,
        "retrieved": result.get("response") or [],
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
    return {"status": "ok", "message": "Library caches cleared — will reload on next request"}


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


@app.get("/bareacts/view", response_class=HTMLResponse)
def bareacts_view(name: str = Query(..., description="Base name of the act JSON (e.g. ADVOCATES_Act)")):
    """
    Render a statute JSON from legal_database/json_output as a readable HTML document.
    Sections are ordered numerically and long sections that were split into
    sub-chunks are recombined in display order.
    """
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=True)
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
          "11.\nGrant of probate..."   → "11. Grant of probate..."
          "(a)\nany person appears..." → "(a) any person appears..."
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
      {"&nbsp;•&nbsp;Year: " + _html_escape(str(year)) if year else ""}
    </div>
    {sections_joined}
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/caselaws/view", response_class=HTMLResponse)
def caselaws_view(name: str = Query(..., description="Base name of the case JSON (e.g. ABHILASHA_V_PARKASH)")):
    """
    Render a case-law JSON from legal_database/json_output as a readable HTML document.
    Paragraphs are shown in logical order (paragraph_id) so that any chunking
    for indexing does not affect the reading flow.
    """
    if not _USE_LEGAL_DATABASE:
        raise HTTPException(status_code=400, detail="JSON endpoint requires NYAYMALAW_DATA_SOURCE=legal_database")
    data = _get_json_from_legal_db(name, is_statute=False)
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
        label = f"¶ {pid}" if pid is not None else "¶"
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
                    f"<button type='button' class='expand-btn' data-row='{idx}' aria-label='Expand row'>⤢</button>"
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
            btn.textContent = expand ? "⤡" : "⤢";
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


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Interactive chat endpoint.
    Phase: fact_collection | response_generation | confirm_index
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
        )
        return resp_result

    return result


@app.post("/submit_case")
def submit_case(request: SubmitCaseRequest, user: dict = Depends(_user_from_token)):
    """
    Initial case submission (await_facts). Frontend sends { text }.
    Returns status + next_question | opinion_text | needs_confirmation so the UI can continue the flow.
    """
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    if not text:
        return {
            "status": "question",
            "next_question": "",
            "retrieved": [],
        }
    try:
        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv,
            current_message=text,
            phase="fact_collection",
            facts_summary=None,
            chat_mode=mode,
        )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
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
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=text, session_ref=str(user.get("id", "")))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=text, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@app.post("/interview_step")
def interview_step(request: InterviewStepRequest, user: dict = Depends(_user_from_token)):
    """
    Follow-up answer in interview. Frontend sends { facts, qa_history } (qa_history includes the latest answer).
    Returns same shape as submit_case for consistent UI handling.
    """
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    mode = (request.mode or "").strip().lower() or None
    if not qa_history:
        return {
            "status": "question",
            "next_question": "",
            "retrieved": [],
        }
    bare_acts_from_client = request.bare_acts or []
    try:
        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})
        current_message = qa_history[-1].answer

        # If the frontend passed bare_acts, the user is answering the bare-acts follow-up question
        if bare_acts_from_client:
            result = process_chat(
                conversation=conv,
                current_message=current_message,
                phase="bare_acts_review",
                facts_summary=facts,
                bare_acts=bare_acts_from_client,
                chat_mode=mode,
            )
        else:
            result = process_chat(
                conversation=conv,
                current_message=current_message,
                phase="fact_collection",
                facts_summary=None,
                chat_mode=mode,
            )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
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
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=facts, session_ref=str(user.get("id", "")))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=facts, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


@app.post("/conversation/continue")
def continue_chat(request: ContinueChatRequest, user: dict = Depends(_user_from_token)):
    """
    Continue a conversation from chat history. Sends full conversation + new message
    so the LLM has full context. Returns same shape as submit_case / interview_step.
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
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
        result = process_chat(
            conversation=conv,
            current_message=message,
            phase="fact_collection",
            facts_summary=None,
        )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": result.get("message", "")}]
            result = process_chat(
                conversation=conv,
                current_message=result["facts_summary"],
                phase="response_generation",
                facts_summary=result["facts_summary"],
                intent=result.get("intent", "legal_opinion"),
                document_types=result.get("document_types", "both"),
                search_strategy=result.get("search_strategy", "local_then_web"),
                result_count=result.get("result_count"),
            )
        if result.get("phase") == "done":
            increment_query_count(user["id"])
            _fire_feedback_log(result, facts=message, session_ref=str(user.get("id", "")))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=message, session_ref=str(user.get("id", "")))
        return _map_chat_result_to_ui(result)
    except Exception as e:
        return _chat_error_fallback(str(e)[:200])


def _run_continue_chat_with_progress(conv: list, message: str, queue: Queue, user_id: str, mode: str | None = None) -> None:
    """Run the same logic as continue_chat, pushing progress to queue and finally the result."""
    try:
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
        )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "user", "content": message}, {"role": "assistant", "content": result.get("message", "")}]
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
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=message, session_ref=str(user_id))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=message, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result)))
    except Exception as e:
        logger.exception("Stream continue_chat failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_submit_case_with_progress(text: str, queue: Queue, user_id: str, mode: str | None = None) -> None:
    """Run submit_case logic with progress streaming."""
    try:
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
        )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
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
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=text, session_ref=str(user_id))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=text, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result)))
    except Exception as e:
        logger.exception("Stream submit_case failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


def _run_interview_step_with_progress(facts: str, qa_history: list, queue: Queue, user_id: str, bare_acts: list = None, mode: str | None = None) -> None:
    """Run interview_step logic with progress streaming."""
    try:
        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))
        def step_callback(step_data: dict):
            queue.put(("step", step_data))
        def token_callback(token: str):
            queue.put(("token", {"content": token}))

        conv = [{"role": "user", "content": facts}]
        for qa in qa_history:
            conv.append({"role": "assistant", "content": qa.question})
            conv.append({"role": "user", "content": qa.answer})
        current_message = qa_history[-1].answer if qa_history else ""

        if bare_acts:
            # User is answering the bare-acts follow-up question — go straight to final opinion
            result = process_chat(
                conversation=conv,
                current_message=current_message,
                phase="bare_acts_review",
                facts_summary=facts,
                bare_acts=bare_acts,
                progress_callback=progress_callback,
                chat_mode=mode,
                step_callback=step_callback,
                token_callback=token_callback,
            )
        else:
            result = process_chat(
                conversation=conv,
                current_message=current_message,
                phase="fact_collection",
                facts_summary=None,
                progress_callback=progress_callback,
                chat_mode=mode,
                step_callback=step_callback,
                token_callback=token_callback,
            )
        if result.get("phase") == "response_generation" and result.get("facts_summary"):
            conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
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
            )
        if result.get("phase") == "done":
            increment_query_count(user_id)
            _fire_feedback_log(result, facts=facts, session_ref=str(user_id))
        if result.get("phase") == "bare_acts_presented":
            _fire_feedback_log_bare_acts(result, facts=facts, session_ref=str(user_id))
        queue.put(("result", _map_chat_result_to_ui(result)))
    except Exception as e:
        logger.exception("Stream interview_step failed")
        queue.put(("result", _chat_error_fallback(str(e)[:200])))


@app.post("/submit_case/stream")
async def submit_case_stream(request: SubmitCaseRequest, user: dict = Depends(_user_from_token)):
    """Same as /submit_case but streams progress via Server-Sent Events."""
    _enforce_query_limit(user)
    text = (request.text or "").strip()
    mode = (request.mode or "").strip().lower() or None
    if not text:
        return JSONResponse(
            status_code=400,
            content={"detail": "text is required"},
        )
    queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id", _ANONYMOUS_EMAIL)

    def run_in_thread():
        _run_submit_case_with_progress(text, queue, user_id, mode)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        while True:
            try:
                kind, payload = await loop.run_in_executor(None, queue.get)
            except Exception:
                break
            if kind == "result":
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            elif kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            elif kind == "step":
                yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            elif kind == "token":
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"

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
async def interview_step_stream(request: InterviewStepRequest, user: dict = Depends(_user_from_token)):
    """Same as /interview_step but streams progress via Server-Sent Events."""
    _enforce_query_limit(user)
    facts = request.facts or ""
    qa_history = request.qa_history or []
    bare_acts_from_client = request.bare_acts or []
    mode = (request.mode or "").strip().lower() or None
    if not qa_history:
        return JSONResponse(
            status_code=400,
            content={"detail": "qa_history is required"},
        )
    queue = Queue()
    loop = asyncio.get_event_loop()
    user_id = user.get("id", _ANONYMOUS_EMAIL)

    def run_in_thread():
        _run_interview_step_with_progress(facts, qa_history, queue, user_id, bare_acts=bare_acts_from_client, mode=mode)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        while True:
            try:
                kind, payload = await loop.run_in_executor(None, queue.get)
            except Exception:
                break
            if kind == "result":
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            elif kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            elif kind == "step":
                yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            elif kind == "token":
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"

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
async def continue_chat_stream(request: ContinueChatRequest, user: dict = Depends(_user_from_token)):
    """
    Same as /conversation/continue but streams progress via Server-Sent Events.
    Events: "progress" (progress snapshot JSON), "done" (final UI result JSON).
    """
    _enforce_query_limit(user)
    message = (request.message or "").strip()
    mode = (request.mode or "").strip().lower() or None
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

    def run_in_thread():
        _run_continue_chat_with_progress(conv, message, queue, user_id, mode)

    thread = __import__("threading").Thread(target=run_in_thread)
    thread.start()

    async def event_generator():
        while True:
            try:
                kind, payload = await loop.run_in_executor(None, queue.get)
            except Exception:
                break
            if kind == "result":
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            elif kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            elif kind == "step":
                yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            elif kind == "token":
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"

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
# Propose for Index — save web-sourced bare act sections to proposed_sections.json
# The user can then run scripts/index_proposed.py to add them to the local index.
# ---------------------------------------------------------------------------

class ProposeIndexRequest(BaseModel):
    section: dict   # Full bare-act section dict (act_name, section_number, full_text, url, …)

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
    # Deduplicate by (act_name, section_number) — don't add the same section twice
    act  = (section.get("act_name")     or "").strip().lower()
    sec  = (section.get("section_number") or "").strip().lower()
    already = any(
        (p.get("act_name","").strip().lower() == act and
         p.get("section_number","").strip().lower() == sec)
        for p in proposals
    )
    if already:
        return {"status": "already_proposed", "message": f"{section.get('act_name')} §{section.get('section_number')} already in proposal list"}

    # Strip internal pipeline keys before saving
    clean = {k: v for k, v in section.items() if not k.startswith("_")}
    clean["_proposed_at"] = _dt.datetime.utcnow().isoformat()

    proposals.append(clean)
    try:
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proposed_path, "w", encoding="utf-8") as f:
            json.dump(proposals, f, ensure_ascii=False, indent=2)
        logger.info("Proposed for index: %s §%s → %s", section.get("act_name"), section.get("section_number"), proposed_path)
        return {
            "status": "proposed",
            "message": f"Saved to {proposed_path.name}. Run scripts/index_proposed.py to add to local index.",
            "total_proposed": len(proposals),
        }
    except Exception as e:
        logger.error("Failed to write proposed_sections.json: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not save proposal: {e}")


# ---------------------------------------------------------------------------
# Eval Files — serve eval JSON results for UI display
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
# Docs — serve architecture markdown for UI
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
def feedback_review(case_id: str, user: dict = Depends(_user_from_token)):
    """
    Trigger AI Gate review for a logged interaction.

    POST /feedback/review/NM-20260301-004

    Reads the row for case_id from the Feedback Log workbook, sends the
    model output to the configured LLM for review against the Golden Rules,
    and writes results back into columns K–R and V–W.

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
# Feedback Log Excel ↔ HTML sync
# ---------------------------------------------------------------------------

def _feedback_log_path():
    from config import FEEDBACK_LOG_PATH
    return FEEDBACK_LOG_PATH


@app.get("/feedback_log/data")
def feedback_log_get_data():
    """
    Return the Feedback Log Excel as JSON for HTML to load (Excel → HTML sync).
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


@app.post("/feedback_log/save")
def feedback_log_save(request: FeedbackLogSaveRequest):
    """
    Save table data from HTML to the Feedback Log Excel (HTML → Excel sync).
    Expects rows: [ notes_row, data_row_1, ... ] (tbody only). Preserves Excel header rows 0–1.
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


@app.get("/health")
def health_check():
    """
    Production health check — verifies Ollama, DB, and vector store.
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
    """Log system status on startup — warns but does NOT block if services are down."""
    logger.info("=" * 60)
    logger.info("Nyaymalaw API v3.0.0 starting up")
    logger.info("=" * 60)

    # Check Ollama
    ollama = check_ollama_health()
    if ollama.get("ollama_reachable") and ollama.get("model_loaded"):
        logger.info("✓ Ollama reachable, model '%s' loaded", ollama["model"])
    elif ollama.get("ollama_reachable"):
        logger.warning("⚠ Ollama reachable but model '%s' NOT found. Run: ollama pull %s", ollama["model"], ollama["model"])
    else:
        logger.warning("⚠ Ollama NOT reachable at localhost:11434. Start Ollama first.")

    # Check DB
    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        logger.info("✓ Database accessible at %s", _DB_PATH)
    except Exception as e:
        logger.warning("⚠ Database error: %s", e)

    # Data root (PDFs and indexes go here; must match NYAYMALAW_DATA_ROOT in .env)
    from config import DATA_ROOT, BARE_ACTS_DIR
    logger.info("Data root: %s (BareActs: %s)", DATA_ROOT, BARE_ACTS_DIR)

    # Check vector store
    from config import VECTOR_STORE, BARE_INDEX_V2, CASE_INDEX_V2
    if os.path.isdir(VECTOR_STORE):
        bare_ok = os.path.isfile(BARE_INDEX_V2)
        case_ok = os.path.isfile(CASE_INDEX_V2)
        logger.info(
            "✓ Vector store at %s (bare_acts: %s, case_laws: %s)",
            VECTOR_STORE, "✓" if bare_ok else "✗", "✓" if case_ok else "✗",
        )
    else:
        logger.warning("⚠ Vector store directory not found: %s", VECTOR_STORE)

    logger.info("CORS origins: %s", _cors_origins)

    # Pre-load ML models to eliminate cold-start latency on the first real request.
    # The embedder and cross-encoder are lazy-loaded on first use; calling them here
    # during startup ensures they are in memory before any user query arrives.
    # Failure is non-fatal — models will still load on demand.
    try:
        from retrieval.hybrid_retriever import _get_embedder, _get_cross_encoder
        _get_embedder()
        logger.info("✓ Embedding model pre-loaded (warm)")
        _get_cross_encoder()
        logger.info("✓ Cross-encoder model pre-loaded (warm)")
    except Exception as _warmup_err:
        logger.warning("⚠ Model pre-load failed (will load on first request): %s", _warmup_err)

    # Pre-load all FAISS indexes, BM25 indexes, and chunk stores into RAM.
    # Without this, each query loads 5+ GB of data from disk, causing seconds of
    # I/O latency per request.  Preloading at startup means all queries serve from
    # in-memory cache.  Failure is non-fatal — indexes will still load on demand.
    try:
        from retrieval.hybrid_retriever import preload_all_indexes
        preload_all_indexes()
        logger.info("✓ All indexes pre-loaded into RAM (queries will serve from cache)")
    except Exception as _preload_err:
        logger.warning("⚠ Index pre-load failed (will load on first request): %s", _preload_err)

    logger.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
