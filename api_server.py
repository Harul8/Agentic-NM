import os
import json
import sqlite3
import hashlib
import secrets
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law
from services.interactive_chat import process_chat
from services.case_law_indexer_incremental import index_new_case_laws
from services.bare_act_indexer_incremental import index_new_bare_acts
from services.response_generator import generate_response

app = FastAPI(title="Nyaymalaw API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and DB (must be before auth routes)
_THIS_FILE = os.path.abspath(os.path.normpath(__file__))
_BASE_DIR = os.path.dirname(_THIS_FILE)
_CHAT_HISTORY_DIR = os.path.join(_BASE_DIR, "data", "chat_history")
_DB_PATH = os.path.join(_CHAT_HISTORY_DIR, "app.db")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _get_db():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
    finally:
        conn.close()


_init_auth_db()

security = HTTPBearer(auto_error=False)


def _user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials or (credentials.credentials or "").strip() == "":
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials.strip()
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT u.id, u.email, u.name FROM users u JOIN sessions s ON s.user_id = u.id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


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
    password = req.password or ""
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
    password = req.password or ""
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
    conn = _get_db()
    try:
        if payload.id is not None:
            cur = conn.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?",
                (payload.id, user["id"]),
            )
            existing = cur.fetchone()
            if existing:
                conn.execute(
                    "UPDATE chats SET title = ?, messages_json = ?, opinion_text = ?, retrieved_json = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
                    (title, messages_json, opinion_text, retrieved_json, payload.id, user["id"]),
                )
                conn.commit()
                return {"id": payload.id, "title": title}
        chat_id = payload.id if payload.id is not None else int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM chats").fetchone()[0])
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
    """Initial case submission - frontend sends { text }."""
    text: str = ""


class QAPair(BaseModel):
    question: str = ""
    answer: str = ""


class InterviewStepRequest(BaseModel):
    """Follow-up answer in interview - frontend sends { facts, qa_history }."""
    facts: str = ""
    qa_history: list[QAPair] = []


def _build_conv(messages: list[ChatMessage] | None) -> list[dict]:
    if not messages:
        return []
    return [{"role": m.role, "content": _normalize_content(m.content)} for m in messages]


# Path to vector store and BareActs directory – resolve from this file’s location
_BARE_CHUNKS_PATH = os.path.join(_BASE_DIR, "data", "vector_store", "bareacts_chunks.json")
_BARE_ACTS_DIR = os.path.normpath(os.path.join(_BASE_DIR, "data", "BareActs"))


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


def _map_chat_result_to_ui(result: dict) -> dict:
    """Map process_chat result to the shape the frontend expects (status, next_question, etc.)."""
    phase = result.get("phase")
    if phase == "fact_collection":
        return {
            "status": "question",
            "next_question": result.get("message", "Could you share more details?"),
            "retrieved": result.get("response") or [],
        }
    if phase == "confirm_materials":
        return {
            "needs_confirmation": True,
            "materials_to_confirm": result.get("materials_to_confirm"),
            "summary": result.get("message", ""),
        }
    if phase == "done" and result.get("response"):
        resp = result["response"]
        retrieved = (resp.get("case_laws") or []) + (resp.get("bare_act_sections") or [])
        return {
            "status": "done",
            "opinion_text": resp.get("explanation", ""),
            "retrieved": retrieved,
        }
    # Fallback: treat as next question
    return {
        "status": "question",
        "next_question": result.get("message", "Could you share more details?"),
        "retrieved": result.get("response") or [],
    }


@app.post("/search")
def search_law(query: SearchQuery):
    """Legacy search endpoint - single query, returns bare acts + case laws."""
    result = fuse_bare_act_and_case_law.run(issue=query.issue)
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
def bareacts_download(name: str = Query(..., description="Filename of the bare act to download")):
    """Serve a bare act file for download. Name must be a safe filename (no path traversal)."""
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
        response.headers["Content-Disposition"] = f'attachment; filename="{base}"'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not serve file: {e!s}")


def _normalize_content(c):
    """Ensure content is string for backend processing."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "text" in c:
        return c["text"]
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
        )
        return resp_result

    return result


@app.post("/submit_case")
def submit_case(request: SubmitCaseRequest):
    """
    Initial case submission (await_facts). Frontend sends { text }.
    Returns status + next_question | opinion_text | needs_confirmation so the UI can continue the flow.
    """
    text = (request.text or "").strip()
    if not text:
        return {
            "status": "question",
            "next_question": "Please describe your legal issue or the facts of your case in a few sentences.",
            "retrieved": [],
        }
    conv = [{"role": "user", "content": text}]
    result = process_chat(
        conversation=conv,
        current_message=text,
        phase="fact_collection",
        facts_summary=None,
    )
    if result.get("phase") == "response_generation" and result.get("facts_summary"):
        conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
        result = process_chat(
            conversation=conv,
            current_message=result["facts_summary"],
            phase="response_generation",
            facts_summary=result["facts_summary"],
        )
    return _map_chat_result_to_ui(result)


@app.post("/interview_step")
def interview_step(request: InterviewStepRequest):
    """
    Follow-up answer in interview. Frontend sends { facts, qa_history } (qa_history includes the latest answer).
    Returns same shape as submit_case for consistent UI handling.
    """
    facts = request.facts or ""
    qa_history = request.qa_history or []
    if not qa_history:
        return {
            "status": "question",
            "next_question": "Please share more details about your case.",
            "retrieved": [],
        }
    conv = [{"role": "user", "content": facts}]
    for qa in qa_history:
        conv.append({"role": "assistant", "content": qa.question})
        conv.append({"role": "user", "content": qa.answer})
    current_message = qa_history[-1].answer
    result = process_chat(
        conversation=conv,
        current_message=current_message,
        phase="fact_collection",
        facts_summary=None,
    )
    if result.get("phase") == "response_generation" and result.get("facts_summary"):
        conv = conv + [{"role": "assistant", "content": result.get("message", "")}]
        result = process_chat(
            conversation=conv,
            current_message=result["facts_summary"],
            phase="response_generation",
            facts_summary=result["facts_summary"],
        )
    return _map_chat_result_to_ui(result)


@app.post("/chat/confirm-index")
def confirm_index(request: ConfirmIndexRequest):
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
    resp = generate_response(request.facts_summary, confirmed_materials=confirmed)

    return {
        "success": True,
        "message": f"Indexed: {bare_result.get('chunks_added', 0)} bare act chunks, {case_result.get('chunks_added', 0)} case law chunks",
        "response": resp,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
