import asyncio
import os
import json
import sqlite3
import hashlib
import logging
import secrets
import time
import traceback
from queue import Queue
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Query, HTTPException, Depends, Header, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law
from services.interactive_chat import process_chat
from services.case_law_indexer_incremental import index_new_case_laws
from services.bare_act_indexer_incremental import index_new_bare_acts
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

# Case law discovery: separate workflow; documents persist until user Index or Clear
try:
    from case_law_discovery.routes import router as case_law_discovery_router
    app.include_router(case_law_discovery_router)
except Exception as e:
    logger.warning("Case law discovery routes not loaded: %s", e)

# ---------------------------------------------------------------------------
# Per-IP request rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------
from collections import defaultdict

_RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))   # seconds
_RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "100"))        # requests per window (increased from 30)
_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "false").lower() == "true"  # Disabled by default for development
_rate_store: dict[str, list[float]] = defaultdict(list)

# Only rate-limit mutation endpoints (not health, static, etc.)
_RATE_LIMITED_PATHS = {"/submit_case", "/submit_case/stream", "/interview_step", "/interview_step/stream", "/conversation/continue", "/conversation/continue/stream", "/chat", "/chat/confirm-index", "/search", "/indexing/run"}

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
)
_BASE_DIR = os.path.dirname(os.path.abspath(os.path.normpath(__file__)))


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


class IndexingItem(BaseModel):
    """One document to index (from Pending indexing UI)."""
    url: str = ""
    title: str = ""
    category: str = "case_law"  # "bare_act" | "case_law"


class IndexingRunRequest(BaseModel):
    """Request to index selected documents (user-triggered from left pane)."""
    items: list[IndexingItem] = []


class PendingIndexingUpdateRequest(BaseModel):
    """Request to persist pending indexing candidates (survives refresh)."""
    items: list[dict] = []


def _build_conv(messages: list[ChatMessage] | None) -> list[dict]:
    if not messages:
        return []
    return [{"role": m.role, "content": _normalize_content(m.content)} for m in messages]


# Path to vector store and BareActs directory – resolve from this file’s location


def _list_bare_acts_from_vector_store() -> list[str]:
    """Return bare act filenames from disk only, so list and download always use the same source."""
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
        if resp.get("indexing_candidates"):
            out["indexing_candidates"] = resp["indexing_candidates"]
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


@app.get("/bareacts/list")
def bareacts_list():
    """Return list of bare act filenames from data/BareActs (same source as download)."""
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
    path = os.path.join(_BARE_ACTS_DIR, base)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_BARE_ACTS_DIR):
        for f in os.listdir(_BARE_ACTS_DIR):
            if f.lower() == base.lower():
                return os.path.abspath(os.path.join(_BARE_ACTS_DIR, f))
    return None


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
    media_type = "application/pdf" if base.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=base, media_type=media_type)
        # inline: open in new tab (browser displays PDF); attachment: download
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{base}"'
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
    """Return absolute path to file in CaseLaws if it exists, else None (case-insensitive on Windows)."""
    path = os.path.join(_CASELAW_DIR, base)
    if os.path.isfile(path):
        return os.path.abspath(path)
    if os.path.isdir(_CASELAW_DIR):
        for f in os.listdir(_CASELAW_DIR):
            if f.lower() == base.lower():
                return os.path.abspath(os.path.join(_CASELAW_DIR, f))
    return None


@app.get("/caselaws/list")
def caselaws_list():
    """Return list of case law filenames from data/CaseLaws."""
    files = _list_case_laws_from_disk()
    return {"cases": files}


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
    media_type = "application/pdf" if base.lower().endswith(".pdf") else "text/plain"
    try:
        response = FileResponse(path, filename=base, media_type=media_type)
        disposition = "inline" if inline else "attachment"
        response.headers["Content-Disposition"] = f'{disposition}; filename="{base}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    Case-law-discovery requests (e.g. user says "case law discovery" or first N bare acts) are
    routed to the case law discovery workflow and do not go through the main pipeline.
    """
    message = (request.message or "").strip()
    if message:
        from case_law_discovery.workflow import is_case_law_discovery_request, run as run_case_law_discovery
        if is_case_law_discovery_request(message):
            workflow_result = run_case_law_discovery(message)
            return _case_law_discovery_ui_result(workflow_result)

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
    Case-law-discovery requests (e.g. first N bare acts, find case laws in vector store) are
    routed to the separate case law discovery workflow and do not go through the main pipeline.
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
        from case_law_discovery.workflow import is_case_law_discovery_request, run as run_case_law_discovery
        if is_case_law_discovery_request(text):
            workflow_result = run_case_law_discovery(text)
            increment_query_count(user["id"])
            return _case_law_discovery_ui_result(workflow_result)
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


def _case_law_discovery_ui_result(workflow_result: dict) -> dict:
    """Build UI result when the request was handled by the separate case law discovery workflow.
    Use response_type generic_chat so the UI does not show 'Legal Opinion' or 'Download as PDF'.
    """
    msg = (workflow_result.get("message") or "").strip()
    if not msg:
        msg = "Case law discovery ran. Check **Case law discovery – Pending** in the left sidebar for documents to index or clear."
    else:
        msg = f"{msg} Check **Case law discovery – Pending** in the left sidebar to index or clear."
    return {
        "status": "done",
        "response_type": "generic_chat",
        "case_law_discovery": True,
        "opinion_text": msg,
        "bare_acts": [],
        "case_laws": [],
        "retrieved": [],
        "model_used": get_last_model_used(),
    }


@app.post("/conversation/continue")
def continue_chat(request: ContinueChatRequest, user: dict = Depends(_user_from_token)):
    """
    Continue a conversation from chat history. Sends full conversation + new message
    so the LLM has full context. Returns same shape as submit_case / interview_step.
    Case-law-discovery requests (e.g. first N bare acts, find case laws in vector store) are
    routed to the separate case law discovery workflow and do not go through the main pipeline.
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
        from case_law_discovery.workflow import is_case_law_discovery_request, run as run_case_law_discovery
        if is_case_law_discovery_request(message):
            workflow_result = run_case_law_discovery(message)
            increment_query_count(user.get("id", 0))
            return _case_law_discovery_ui_result(workflow_result)
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
    """Run the same logic as continue_chat, pushing progress to queue and finally the result.
    Case-law-discovery requests are handled by the separate workflow and do not use process_chat.
    """
    try:
        from case_law_discovery.workflow import is_case_law_discovery_request, run as run_case_law_discovery
        if is_case_law_discovery_request(message):
            queue.put(("progress", {"groups": [{"name": "Case law discovery", "steps": [{"name": "Running case law discovery workflow…", "status": "running", "duration_seconds": 0}]}]}))
            workflow_result = run_case_law_discovery(message)
            increment_query_count(user_id)
            queue.put(("result", _case_law_discovery_ui_result(workflow_result)))
            return
        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))

        result = process_chat(
            conversation=conv,
            current_message=message,
            phase="fact_collection",
            facts_summary=None,
            progress_callback=progress_callback,
            chat_mode=mode,
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
    """Run submit_case logic with progress streaming.
    Case-law-discovery requests are handled by the separate workflow and do not use process_chat.
    """
    try:
        from case_law_discovery.workflow import is_case_law_discovery_request, run as run_case_law_discovery
        if is_case_law_discovery_request(text):
            queue.put(("progress", {"groups": [{"name": "Case law discovery", "steps": [{"name": "Running case law discovery workflow…", "status": "running", "duration_seconds": 0}]}]}))
            workflow_result = run_case_law_discovery(text)
            increment_query_count(user_id)
            queue.put(("result", _case_law_discovery_ui_result(workflow_result)))
            return
        def progress_callback(progress_snapshot: dict):
            queue.put(("progress", progress_snapshot))

        conv = [{"role": "user", "content": text}]
        result = process_chat(
            conversation=conv,
            current_message=text,
            phase="fact_collection",
            facts_summary=None,
            progress_callback=progress_callback,
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
                progress_callback=progress_callback,
                chat_mode=mode,
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
            )
        else:
            result = process_chat(
                conversation=conv,
                current_message=current_message,
                phase="fact_collection",
                facts_summary=None,
                progress_callback=progress_callback,
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
                progress_callback=progress_callback,
                chat_mode=mode,
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
            if kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

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
            if kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

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
            if kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/confirm-index")
def confirm_index(request: ConfirmIndexRequest, user: dict = Depends(_user_from_token)):
    """
    Index user-confirmed bare acts and case laws into the vector store,
    then generate and return the full legal research response.
    """
    bare_result = {"success": True, "chunks_added": 0}
    case_result = {"success": True, "chunks_added": 0}

    if request.bare_acts:
        bare_result = index_new_bare_acts(request.bare_acts)
    if request.case_laws:
        case_result = index_new_case_laws(request.case_laws)

    if not bare_result.get("success") and not case_result.get("success"):
        return {
            "success": False,
            "message": bare_result.get("message", "") or case_result.get("message", ""),
            "response": None,
        }

    # Generate full response with confirmed materials
    confirmed = {
        "bare_acts": request.bare_acts,
        "case_laws": request.case_laws,
    }
    resp = generate_response_v2(request.facts_summary, confirmed_materials=confirmed)

    return {
        "success": True,
        "message": f"Indexed: {bare_result.get('chunks_added', 0)} bare act chunks, {case_result.get('chunks_added', 0)} case law chunks",
        "response": resp,
    }


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


def _load_pending_indexing() -> list:
    """Load persisted pending indexing candidates from disk."""
    from config import PENDING_INDEXING_PATH, DATA_ROOT
    try:
        if os.path.isfile(PENDING_INDEXING_PATH):
            with open(PENDING_INDEXING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    except Exception as e:
        logger.warning("Could not load pending indexing: %s", e)
    return []


def _save_pending_indexing(items: list) -> None:
    """Persist pending indexing candidates to disk."""
    from config import PENDING_INDEXING_PATH, DATA_ROOT
    try:
        os.makedirs(DATA_ROOT, exist_ok=True)
        with open(PENDING_INDEXING_PATH, "w", encoding="utf-8") as f:
            json.dump({"items": items, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    except Exception as e:
        logger.warning("Could not save pending indexing: %s", e)


def _remove_from_pending_indexing(url_title_pairs: list[tuple]) -> None:
    """Remove indexed items from persisted pending list by (url, title) pairs."""
    items = _load_pending_indexing()
    if not items or not url_title_pairs:
        return
    seen = {(str(u).strip(), str(t).strip()) for u, t in url_title_pairs}
    kept = [c for c in items if (str(c.get("source_url", "") or "").strip(), str(c.get("title", "") or "").strip()) not in seen]
    if len(kept) != len(items):
        _save_pending_indexing(kept)


@app.get("/indexing/pending")
def indexing_pending_get(user: dict = Depends(_user_from_token)):
    """Return persisted pending indexing candidates (survives refresh)."""
    items = _load_pending_indexing()
    return {"items": items}


@app.post("/indexing/pending")
def indexing_pending_save(request: PendingIndexingUpdateRequest, user: dict = Depends(_user_from_token)):
    """Persist pending indexing candidates (replace full list)."""
    items = request.items or []
    _save_pending_indexing(items)
    return {"items": items, "message": "Saved"}


@app.delete("/indexing/pending")
def indexing_pending_clear(user: dict = Depends(_user_from_token)):
    """Clear persisted pending indexing candidates."""
    _save_pending_indexing([])
    return {"items": [], "message": "Cleared"}


@app.post("/indexing/run")
def indexing_run(request: IndexingRunRequest, user: dict = Depends(_user_from_token)):
    """
    Index selected documents from the Pending indexing list.
    For each item: fetch from source_url, then chunk and add to the appropriate index (bare_act or case_law).
    On success, removes indexed items from persisted pending list.

    P0: BM25 rebuilds deferred to end of batch (one rebuild per category instead of N).
    P1: Up to 3 documents fetched + embedded in parallel; writes serialised via _index_write_lock.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from retrieval.auto_enricher import enrich_from_search_result, begin_batch_indexing, end_batch_indexing

    _enforce_query_limit(user)
    items = request.items or []
    if not items:
        return {"indexed": 0, "errors": [], "message": "No items to index."}

    indexed = 0
    errors = []
    _results_lock = threading.Lock()

    def _index_one(i: int, item) -> None:
        nonlocal indexed
        url = (item.url or "").strip()
        title = (item.title or "").strip()
        category = (item.category or "case_law").strip().lower()
        if category not in ("bare_act", "case_law"):
            category = "case_law"
        if not url or not title:
            with _results_lock:
                errors.append({"index": i, "error": "Missing url or title"})
            return
        try:
            result = {"url": url, "title": title, "snippet": "", "source_tag": "USER_INDEX"}
            enrichment = enrich_from_search_result(result, category, original_query="", skip_index=False)
            if enrichment.get("chunks_added", 0) > 0 or enrichment.get("indexed"):
                with _results_lock:
                    indexed += 1
                _remove_from_pending_indexing([(url, title)])
        except Exception as e:
            logger.exception("Indexing failed for %s: %s", url[:80], e)
            with _results_lock:
                errors.append({"index": i, "url": url[:80], "error": str(e)[:200]})

    # P0: enter batch mode so BM25 is only rebuilt once at the end
    begin_batch_indexing()
    try:
        if len(items) == 1:
            _index_one(0, items[0])
        else:
            # P1: parallel fetch + embed (max 3 workers); writes are serialised inside add_chunks_to_index
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_index_one, i, item): i for i, item in enumerate(items)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Unexpected error in indexing worker: %s", e)
    finally:
        # P0: flush deferred BM25 rebuilds regardless of errors
        end_batch_indexing()

    return {
        "indexed": indexed,
        "errors": errors,
        "message": f"Indexed {indexed} of {len(items)} document(s)." if items else "No items to index.",
    }


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

    logger.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
