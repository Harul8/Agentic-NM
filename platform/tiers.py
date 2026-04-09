"""
platform/tiers.py — Freemium tier management.
"""
import sqlite3
import logging
from datetime import date

from config import (
    DB_PATH,
    TIER_FREE,
    TIER_PREMIUM,
    TIER_QUERY_LIMITS,
    TIER_FEATURES,
)

logger = logging.getLogger(__name__)


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# DB migration: ensure tier columns exist
# ---------------------------------------------------------------------------

def ensure_tier_columns():
    """
    Add tier-related columns to users table if they don't exist.
    Safe to call multiple times (idempotent).
    """
    conn = _get_db()
    try:
        # Check if columns already exist
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "tier" not in columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN tier TEXT NOT NULL DEFAULT '{TIER_FREE}'")
            logger.info("Added 'tier' column to users table")

        if "queries_today" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN queries_today INTEGER NOT NULL DEFAULT 0")
            logger.info("Added 'queries_today' column to users table")

        if "queries_reset_date" not in columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN queries_reset_date TEXT NOT NULL DEFAULT '{date.today().isoformat()}'")
            logger.info("Added 'queries_reset_date' column to users table")

        conn.commit()
    except Exception as e:
        logger.error("Failed to migrate tier columns: %s", e)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query counting and limit enforcement
# ---------------------------------------------------------------------------

def _reset_if_new_day(conn, user_id: int) -> None:
    """Reset daily query count if the date has changed."""
    row = conn.execute(
        "SELECT queries_today, queries_reset_date FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        return

    today = date.today().isoformat()
    if row["queries_reset_date"] != today:
        conn.execute(
            "UPDATE users SET queries_today = 0, queries_reset_date = ? WHERE id = ?",
            (today, user_id),
        )
        conn.commit()


def check_query_limit(user_id: int) -> dict:
    """
    Check if the user can make another query.

    Returns:
        {
            "allowed": bool,
            "tier": str,
            "queries_used": int,
            "queries_limit": int,
            "queries_remaining": int,
            "message": str (only if not allowed),
        }
    """
    conn = _get_db()
    try:
        _reset_if_new_day(conn, user_id)

        row = conn.execute(
            "SELECT tier, queries_today FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"allowed": False, "message": "User not found"}

        tier = row["tier"] or TIER_FREE
        used = row["queries_today"] or 0
        limit = TIER_QUERY_LIMITS.get(tier, TIER_QUERY_LIMITS[TIER_FREE])
        remaining = max(0, limit - used)

        if used >= limit:
            if tier == TIER_FREE:
                msg = (
                    f"You've used all {limit} free queries for today. "
                    "Upgrade to Premium for 25 queries per day, advanced research, "
                    "and document drafting. Your limit resets tomorrow."
                )
            else:
                msg = (
                    f"You've reached your daily limit of {limit} queries. "
                    "Your limit resets tomorrow."
                )
            return {
                "allowed": False,
                "tier": tier,
                "queries_used": used,
                "queries_limit": limit,
                "queries_remaining": 0,
                "message": msg,
            }

        return {
            "allowed": True,
            "tier": tier,
            "queries_used": used,
            "queries_limit": limit,
            "queries_remaining": remaining,
        }
    finally:
        conn.close()


def increment_query_count(user_id: int) -> None:
    """Increment the user's daily query count. Call AFTER a successful query."""
    conn = _get_db()
    try:
        _reset_if_new_day(conn, user_id)
        conn.execute(
            "UPDATE users SET queries_today = queries_today + 1 WHERE id = ?",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feature access checks
# ---------------------------------------------------------------------------

def check_feature(user_id: int, feature: str) -> dict:
    """
    Check if a user has access to a specific feature.

    Returns:
        {
            "allowed": bool,
            "tier": str,
            "message": str (only if not allowed),
        }
    """
    conn = _get_db()
    try:
        row = conn.execute("SELECT tier FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return {"allowed": False, "tier": TIER_FREE, "message": "User not found"}

        tier = row["tier"] or TIER_FREE
        features = TIER_FEATURES.get(tier, TIER_FEATURES[TIER_FREE])
        allowed = features.get(feature, False)

        if not allowed:
            return {
                "allowed": False,
                "tier": tier,
                "message": f"'{feature}' is a Premium feature. Upgrade to access advanced research, document drafting, and more.",
            }

        return {"allowed": True, "tier": tier}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tier info (for frontend display)
# ---------------------------------------------------------------------------

def get_tier_info(user_id: int) -> dict:
    """
    Get complete tier info for the frontend.

    Returns:
        {
            "tier": str,
            "queries_used": int,
            "queries_limit": int,
            "queries_remaining": int,
            "features": dict,
            "reset_date": str,
        }
    """
    conn = _get_db()
    try:
        _reset_if_new_day(conn, user_id)

        row = conn.execute(
            "SELECT tier, queries_today, queries_reset_date FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {
                "tier": TIER_FREE,
                "queries_used": 0,
                "queries_limit": TIER_QUERY_LIMITS[TIER_FREE],
                "queries_remaining": TIER_QUERY_LIMITS[TIER_FREE],
                "features": TIER_FEATURES[TIER_FREE],
                "reset_date": date.today().isoformat(),
            }

        tier = row["tier"] or TIER_FREE
        used = row["queries_today"] or 0
        limit = TIER_QUERY_LIMITS.get(tier, TIER_QUERY_LIMITS[TIER_FREE])

        return {
            "tier": tier,
            "queries_used": used,
            "queries_limit": limit,
            "queries_remaining": max(0, limit - used),
            "features": TIER_FEATURES.get(tier, TIER_FEATURES[TIER_FREE]),
            "reset_date": row["queries_reset_date"] or date.today().isoformat(),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tier management
# ---------------------------------------------------------------------------

def upgrade_user(user_id: int, tier: str = TIER_PREMIUM) -> dict:
    """
    Upgrade a user's tier. Used by admin endpoint (and later by Razorpay webhook).

    Returns: {"success": bool, "tier": str, "message": str}
    """
    if tier not in (TIER_FREE, TIER_PREMIUM):
        return {"success": False, "tier": "", "message": f"Invalid tier: {tier}"}

    conn = _get_db()
    try:
        row = conn.execute("SELECT id, tier FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return {"success": False, "tier": "", "message": "User not found"}

        conn.execute("UPDATE users SET tier = ? WHERE id = ?", (tier, user_id))
        conn.commit()

        return {
            "success": True,
            "tier": tier,
            "message": f"User upgraded to {tier}",
        }
    finally:
        conn.close()
