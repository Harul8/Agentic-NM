"""
Auth and chat-history CRUD routes.

/auth/register  POST
/auth/login     POST
/auth/logout    POST
/chats          GET / POST
/chats/{id}     GET / DELETE
"""
import json
import sqlite3
import secrets
import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from api.deps import (
    _get_db,
    _hash_password,
    _check_password,
    _require_auth,
    security,
    logger,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = ""
    name: str = ""
    password: str = ""
    phone_number: str = ""


class LoginRequest(BaseModel):
    email: str = ""
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


# ---------------------------------------------------------------------------
# Helpers (used by both this module and core_chat_routes)
# ---------------------------------------------------------------------------

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

    intake_state = state.get("intakeState")
    if not isinstance(intake_state, dict):
        intake_state = None

    prelim_retrieval_context = state.get("preliminaryRetrievalContext")
    if not isinstance(prelim_retrieval_context, dict):
        prelim_retrieval_context = None

    return {
        "stage": stage,
        "facts": facts,
        "currentQuestion": current_question,
        "qaHistory": qa_history,
        "analysisStage": analysis_stage,
        "factsSummary": facts_summary,
        "lastResponseType": last_response_type,
        "intakeState": intake_state,
        "preliminaryRetrievalContext": prelim_retrieval_context,
    }


def _safe_json_loads(raw_text: Optional[str], fallback):
    try:
        return json.loads(raw_text or "")
    except Exception:
        return fallback


def _coerce_chat_id(v):
    """Ensure chat id is int for DB (handles int or string from JSON)."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@router.post("/auth/register")
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
        return {"success": True, "token": token, "user": {"email": email, "name": name}}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="An account with this email already exists")
    finally:
        conn.close()


@router.post("/auth/login")
def auth_login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    phone = (req.phone_number or "").strip()
    password = (req.password or "").strip()
    if not (email or phone) or not password:
        raise HTTPException(status_code=400, detail="Email (or phone number) and password required")
    conn = _get_db()
    try:
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
        if not row or not _check_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        if not row["password_hash"].startswith("$2"):
            new_hash = _hash_password(password)
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, row["id"]))
            logger.info("Upgraded password hash to bcrypt for user id=%s", row["id"])
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+30 days'))",
            (row["id"], token),
        )
        conn.commit()
        return {"success": True, "token": token, "user": {"email": row["email"], "name": row["name"]}}
    finally:
        conn.close()


@router.post("/auth/logout")
def auth_logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Invalidate the current session token. Safe to call even if already logged out."""
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


# ---------------------------------------------------------------------------
# Chat history CRUD routes
# ---------------------------------------------------------------------------

@router.get("/chats")
def chats_list(user: dict = Depends(_require_auth)):
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, state_json, created_at "
            "FROM chats WHERE user_id = ? ORDER BY created_at DESC",
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


@router.post("/chats")
def chats_upsert(payload: ChatPayload, user: dict = Depends(_require_auth)):
    """Create or update a chat."""
    title = (payload.title or "").strip() or "Untitled chat"
    messages = payload.messages if isinstance(payload.messages, list) else []
    opinion_text = payload.opinionText or ""
    retrieved = payload.retrieved if isinstance(payload.retrieved, list) else []
    workflow_state = _normalize_workflow_state(payload.workflowState, messages)
    created_at = payload.createdAt or datetime.datetime.utcnow().isoformat() + "Z"
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
            if cur.fetchone():
                conn.execute(
                    "UPDATE chats SET title=?, messages_json=?, opinion_text=?, retrieved_json=?, "
                    "state_json=?, updated_at=datetime('now') WHERE id=? AND user_id=?",
                    (title, messages_json, opinion_text, retrieved_json, state_json, payload_id, user["id"]),
                )
                conn.commit()
                return {"id": payload_id, "title": title}
        chat_id = (
            payload_id
            if payload_id is not None
            else int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM chats").fetchone()[0])
        )
        conn.execute(
            "INSERT OR REPLACE INTO chats (id, user_id, title, messages_json, opinion_text, "
            "retrieved_json, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user["id"], title, messages_json, opinion_text, retrieved_json, state_json, created_at),
        )
        conn.commit()
        return {"id": chat_id, "title": title}
    finally:
        conn.close()


@router.get("/chats/{chat_id}")
def chat_get(chat_id: int, user: dict = Depends(_require_auth)):
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, title, messages_json, opinion_text, retrieved_json, state_json, created_at "
            "FROM chats WHERE id = ? AND user_id = ?",
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


@router.delete("/chats/{chat_id}")
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
