"""
Admin, tier, eval, docs, feedback, and health routes.

/user/tier                  GET
/admin/upgrade              POST
/admin/reset-rate-limit     POST
/propose_index              POST
/eval/list                  GET
/eval/file                  GET
/eval/figures/list          GET
/eval/figures/file          GET
/docs/architecture          GET
/feedback/review/{case_id}  POST
/feedback/status            GET
/feedback_log/data          GET
/feedback_log/save          POST
/feedback/response/taxonomy GET
/feedback/response          POST
/health                     GET
startup event
"""
import json
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.deps import (
    _get_db,
    _user_from_token,
    _require_auth,
    _rate_store,
    logger,
)
from platform_pkg.tiers import get_tier_info, upgrade_user
from platform_pkg.feedback.store import (
    RESPONSE_FEEDBACK_TAGS,
    append_response_feedback,
    feedback_store_path,
)

# Feedback AI reviewer (optional)
try:
    from platform_pkg.feedback.reviewer import run_ai_review as _run_ai_review
    _FEEDBACK_REVIEWER_ENABLED = True
except Exception:
    _run_ai_review = None  # type: ignore[assignment]
    _FEEDBACK_REVIEWER_ENABLED = False

# Feedback logger (for status endpoint)
try:
    from platform_pkg.feedback.logger import log_interaction as _log_interaction
    _FEEDBACK_ENABLED = True
except Exception:
    _FEEDBACK_ENABLED = False

router = APIRouter()

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_RESULTS_DIR = os.path.join(_BASE_DIR, "eval", "results")
EVAL_FIGURES_DIR = os.path.join(_BASE_DIR, "eval", "figures")
DOCS_DIR = os.path.join(_BASE_DIR, "docs")
ARCHITECTURE_MD = os.path.join(DOCS_DIR, "CREWAI_MULTI_AGENT_ARCHITECTURE.md")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UpgradeRequest(BaseModel):
    user_id: int
    tier: str = "premium"


class ProposeIndexRequest(BaseModel):
    section: dict


class FeedbackLogSaveRequest(BaseModel):
    rows: List[List[str]] = []


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


# ---------------------------------------------------------------------------
# Tier / freemium
# ---------------------------------------------------------------------------

@router.get("/user/tier")
def user_tier(user: dict = Depends(_user_from_token)):
    return get_tier_info(user["id"])


@router.post("/admin/upgrade")
def admin_upgrade(request: UpgradeRequest):
    result = upgrade_user(request.user_id, request.tier)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Upgrade failed"))
    return result


@router.post("/admin/reset-rate-limit")
def reset_rate_limit():
    """Reset rate limit store (development only)."""
    _rate_store.clear()
    logger.info("Rate limit store cleared")
    return {"status": "ok", "message": "Rate limit store cleared"}


# ---------------------------------------------------------------------------
# Propose for index
# ---------------------------------------------------------------------------

@router.post("/propose_index")
async def propose_index(req: ProposeIndexRequest):
    from pathlib import Path as _Path
    import datetime as _dt
    from config import VECTOR_STORE as _VECTOR_STORE

    vector_store_dir = _Path(_VECTOR_STORE).parent if _Path(_VECTOR_STORE).suffix else _Path(_VECTOR_STORE)
    proposed_path = vector_store_dir / "proposed_sections.json"

    try:
        proposals = json.loads(proposed_path.read_text(encoding="utf-8")) if proposed_path.exists() else []
    except Exception:
        proposals = []

    section = req.section
    act = (section.get("act_name") or "").strip().lower()
    sec = (section.get("section_number") or "").strip().lower()
    already = any(
        p.get("act_name", "").strip().lower() == act
        and p.get("section_number", "").strip().lower() == sec
        for p in proposals
    )
    if already:
        return {
            "status": "already_proposed",
            "message": f"{section.get('act_name')} §{section.get('section_number')} already in proposal list",
        }

    clean = {k: v for k, v in section.items() if not k.startswith("_")}
    clean["_proposed_at"] = _dt.datetime.utcnow().isoformat()
    proposals.append(clean)
    try:
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        proposed_path.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
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
# Eval files
# ---------------------------------------------------------------------------

@router.get("/eval/list")
def eval_list_files():
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
                files.append({"path": path.replace("\\", "/"), "name": name})
    return {"files": sorted(files, key=lambda x: x["path"])}


@router.get("/eval/file")
def eval_get_file(path: str = Query(...)):
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


@router.get("/eval/figures/list")
def eval_list_figures():
    figures = []
    if not os.path.isdir(EVAL_FIGURES_DIR):
        return {"figures": []}
    for name in os.listdir(EVAL_FIGURES_DIR):
        full = os.path.join(EVAL_FIGURES_DIR, name)
        if os.path.isfile(full) and name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            figures.append({"path": name, "name": name})
    return {"figures": sorted(figures, key=lambda x: x["name"])}


@router.get("/eval/figures/file")
def eval_get_figure(path: str = Query(...)):
    path = path.lstrip("/").replace("..", "").replace("\\", "/")
    if "/" in path:
        return JSONResponse(status_code=400, content={"detail": "path must be a filename"})
    full = os.path.normpath(os.path.join(EVAL_FIGURES_DIR, path))
    base = os.path.realpath(EVAL_FIGURES_DIR)
    if not os.path.isfile(full) or not os.path.realpath(full).startswith(base):
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    ext = os.path.splitext(path)[1].lower()
    return FileResponse(full, filename=path, media_type=media_types.get(ext, "application/octet-stream"))


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

@router.get("/docs/architecture")
def docs_architecture():
    if not os.path.isfile(ARCHITECTURE_MD):
        return JSONResponse(status_code=404, content={"detail": "Architecture doc not found"})
    try:
        with open(ARCHITECTURE_MD, encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        logger.exception("Docs architecture read failed")
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ---------------------------------------------------------------------------
# Feedback / AI Gate
# ---------------------------------------------------------------------------

@router.post("/feedback/review/{case_id}")
def feedback_review(case_id: str, user: dict = Depends(_require_auth)):
    if not _FEEDBACK_REVIEWER_ENABLED or not _run_ai_review:
        raise HTTPException(
            status_code=503,
            detail="Feedback system not available (feedback_reviewer import failed at startup).",
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


@router.get("/feedback/status")
def feedback_status():
    return {
        "feedback_enabled": _FEEDBACK_ENABLED,
        "log_path": (
            os.environ.get("FEEDBACK_LOG_PATH", "")
            or getattr(__import__("config"), "FEEDBACK_LOG_PATH", "not configured")
        ),
    }


def _feedback_log_path():
    from config import FEEDBACK_LOG_PATH
    return FEEDBACK_LOG_PATH


@router.get("/feedback_log/data")
def feedback_log_get_data():
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


@router.post("/feedback_log/save")
def feedback_log_save(request: FeedbackLogSaveRequest):
    import pandas as pd
    path = _feedback_log_path()
    if not path:
        raise HTTPException(status_code=500, detail="FEEDBACK_LOG_PATH not configured.")
    new_rows = request.rows or []
    if not new_rows:
        raise HTTPException(status_code=400, detail="No rows provided.")
    try:
        existing = pd.read_excel(path, sheet_name=0, header=None)
        header = existing.iloc[:2] if len(existing) >= 2 else existing
        new_df = pd.DataFrame(new_rows)
        out = pd.concat([header, new_df], ignore_index=True)
        out.to_excel(path, index=False, header=False)
        return {"message": "Saved", "rows": len(new_rows)}
    except Exception as e:
        logger.exception("feedback_log POST save: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/response/taxonomy")
def response_feedback_taxonomy():
    return {
        "ratings": ["good", "okay", "bad"],
        "reason_tags": list(RESPONSE_FEEDBACK_TAGS),
        "store_path": str(feedback_store_path()),
    }


@router.post("/feedback/response")
def submit_response_feedback(request: ResponseFeedbackRequest, user: dict = Depends(_require_auth)):
    message_id = (request.message_id or "").strip()
    rating = (request.rating or "").strip().lower()
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")
    if not rating:
        raise HTTPException(status_code=400, detail="rating is required")
    try:
        append_response_feedback({
            "user_id": user.get("id"),
            "chat_id": (request.chat_id or "").strip(),
            "message_id": message_id,
            "rating": rating,
            "reason_tags": request.reason_tags or [],
            "free_text": (request.free_text or "").strip(),
            "assistant_text": (request.assistant_text or "").strip(),
            "user_message": (request.user_message or "").strip(),
            "stage": (request.stage or "").strip(),
            "response_type": (request.response_type or "").strip(),
            "model_used": (request.model_used or "").strip(),
            "latency_ms": request.latency_ms,
            "metadata": request.metadata or {},
        })
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("submit_response_feedback failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check():
    from platform_pkg.llm import check_ollama_health
    health = {"status": "healthy", "checks": {}}

    ollama = check_ollama_health()
    health["checks"]["ollama"] = ollama
    if not ollama.get("ollama_reachable"):
        health["status"] = "unhealthy"
    if not ollama.get("model_loaded"):
        health["status"] = "degraded" if health["status"] == "healthy" else health["status"]

    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        health["checks"]["database"] = {"reachable": True}
    except Exception as e:
        health["checks"]["database"] = {"reachable": False, "error": str(e)}
        health["status"] = "unhealthy"

    from config import VECTOR_STORE, BARE_INDEX_V2, CASE_INDEX_V2
    vs_exists = os.path.isdir(VECTOR_STORE)
    health["checks"]["vector_store"] = {
        "directory_exists": vs_exists,
        "bare_acts_index": os.path.isfile(BARE_INDEX_V2),
        "case_laws_index": os.path.isfile(CASE_INDEX_V2),
    }
    if not vs_exists:
        health["status"] = "degraded" if health["status"] == "healthy" else health["status"]

    status_code = 200 if health["status"] != "unhealthy" else 503
    return JSONResponse(content=health, status_code=status_code)
