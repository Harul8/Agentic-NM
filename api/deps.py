"""
Shared FastAPI dependencies — auth, DB, security, tier enforcement.

Imported by all route modules.  No route handlers here — only infrastructure.
"""
import os
import sqlite3
import hashlib
import secrets
import logging
from typing import Optional

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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
from platform_pkg.tiers import (
    check_query_limit,
    increment_query_count,
    check_feature,
    get_tier_info,
    upgrade_user,
)

logger = logging.getLogger("nyaymalaw.api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(os.path.normpath(__file__ + "/..")))
_LEGAL_DB_BAREACTS_DIR = os.path.join(_LEGAL_DB_JSON_OUTPUT, "BareActs")
_LEGAL_DB_CASELAWS_DIR = os.path.join(_LEGAL_DB_JSON_OUTPUT, "caselaws")

# In-memory cache for library endpoints (None = not yet populated)
_caselaws_library_cache: dict | None = None
_bareacts_library_cache: dict | None = None

# Anonymous user used when no auth token
_ANONYMOUS_EMAIL = "anonymous@nyaymalaw.local"

# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash a password using bcrypt (salted, with work factor 12)."""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode("utf-8")


def _check_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.
    Also handles legacy SHA-256 hashes (hex strings) so existing accounts
    continue to work — they will be re-hashed to bcrypt on next successful login.
    """
    if not stored_hash.startswith("$2"):
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return secrets.compare_digest(legacy_hash, stored_hash)
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

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
            row[1] for row in conn.execute("PRAGMA table_info(chats)").fetchall()
        }
        if "state_json" not in existing_chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN state_json TEXT NOT NULL DEFAULT '{}'")
        existing_session_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
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
        existing_user_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "phone_number" not in existing_user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN phone_number TEXT DEFAULT NULL")
        conn.commit()
        cur = conn.execute("SELECT id FROM users WHERE email = ?", (_ANONYMOUS_EMAIL,))
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (_ANONYMOUS_EMAIL, "", "Guest"),
            )
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Security + auth dependencies
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def _user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Return user from token, or anonymous user when no/invalid token."""
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
            "SELECT u.id, u.email, u.name FROM users u "
            "JOIN sessions s ON s.user_id = u.id "
            "WHERE s.token = ? AND u.email != ? "
            "AND (s.expires_at IS NULL OR datetime(s.expires_at) > datetime('now'))",
            (token, _ANONYMOUS_EMAIL),
        ).fetchone()
        if not row:
            logger.debug("Invalid or expired token; using anonymous user")
            return _anonymous_user()
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def _require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """Strict auth dependency — raises HTTP 401 if token is missing or invalid."""
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
    _QUERY_LIMIT_ENABLED = os.environ.get("QUERY_LIMIT_ENABLED", "false").lower() == "true"
    if not _QUERY_LIMIT_ENABLED:
        return
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


# ---------------------------------------------------------------------------
# Shared mutable state (importable so admin routes can reset)
# ---------------------------------------------------------------------------
from collections import defaultdict
_rate_store: dict = defaultdict(list)
