"""
API routes for case law discovery. Mount under /case-law-discovery.
Documents presented for indexing persist until user completes indexing or clicks Clear.
"""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/case-law-discovery", tags=["case-law-discovery"])


class RunRequest(BaseModel):
    message: str = ""


class ConfirmIndexRequest(BaseModel):
    items: list[dict] = []  # same shape as indexing: url, title, category


@router.get("/pending")
def get_pending():
    """Return persisted documents presented for indexing (survives refresh/restart)."""
    from case_law_discovery.store import load_pending
    items = load_pending()
    return {"items": items}


@router.delete("/pending")
def clear_pending():
    """Clear persisted list (user clicked Clear)."""
    from case_law_discovery.store import clear_pending as do_clear
    do_clear()
    return {"items": [], "message": "Cleared"}


@router.get("/retrieve")
def first_gate_retrieve(query: str = ""):
    """
    First-gate retrieval: resolve act(s) from query (act name or text), return
    act summary + case law summaries from indexes (no vector store). Use before
    fetching full docs from vector store.
    """
    from case_law_discovery.workflow import retrieve_first_gate
    try:
        return retrieve_first_gate(query=query or "")
    except Exception as e:
        logger.exception("First-gate retrieve failed: %s", e)
        return {"acts": [], "query": query, "error": str(e)[:200]}


@router.post("/run")
def run_workflow(request: RunRequest):
    """Run case law discovery workflow; appends results to pending store."""
    from case_law_discovery.workflow import run
    msg = (request.message or "").strip()
    if not msg:
        return {"flow_type": None, "added": 0, "pending_total": 0, "message": "No message provided."}
    try:
        result = run(msg)
        return result
    except Exception as e:
        logger.exception("Case law discovery run failed: %s", e)
        return {"flow_type": None, "added": 0, "pending_total": 0, "message": str(e)[:200]}


@router.post("/confirm-index")
def confirm_index(request: ConfirmIndexRequest):
    """
    Index selected documents from case law discovery pending (same as main indexing),
    then remove them from case law discovery pending store.
    Updates summary index (by act name) when a document has act_name and signature.
    """
    from case_law_discovery.store import load_pending, save_pending, add_signature_to_summary_index, add_case_law_summary
    from retrieval.auto_enricher import enrich_from_search_result

    items = request.items or []
    if not items:
        return {"indexed": 0, "errors": [], "message": "No items to index.", "remaining": 0}

    pending = load_pending()
    indexed = 0
    errors = []
    removed_url_title = []

    for i, it in enumerate(items):
        url = (it.get("source_url") or it.get("url") or "").strip()
        title = (it.get("title") or "").strip()
        category = (it.get("suggested_category") or it.get("category") or "case_law").strip().lower()
        if category not in ("bare_act", "case_law"):
            category = "case_law"
        if not url and not title:
            errors.append({"index": i, "error": "Missing url or title"})
            continue
        try:
            result = {"url": url, "title": title, "snippet": "", "source_tag": "CASE_LAW_DISCOVERY"}
            enrichment = enrich_from_search_result(result, category, original_query="", skip_index=False)
            if enrichment.get("chunks_added", 0) > 0 or enrichment.get("indexed") or enrichment.get("pdf_saved"):
                indexed += 1
                removed_url_title.append((url, title))
                act_name = (it.get("act_name") or "").strip()
                signature = (it.get("signature") or "").strip()
                summary = (it.get("summary") or "").strip()
                if act_name and signature:
                    add_signature_to_summary_index(act_name, signature)
                if signature and summary:
                    add_case_law_summary(signature, summary)
        except Exception as e:
            logger.exception("Indexing failed for %s: %s", (url or title)[:80], e)
            errors.append({"index": i, "url": (url or title)[:80], "error": str(e)[:200]})

    if removed_url_title:
        seen = {(str(u).strip(), str(t).strip()) for u, t in removed_url_title}
        pending = [p for p in pending if (str(p.get("source_url") or "").strip(), str(p.get("title") or "").strip()) not in seen]
        save_pending(pending)

    return {
        "indexed": indexed,
        "errors": errors,
        "message": f"Indexed {indexed} of {len(items)} document(s).",
        "remaining": len(pending),
    }
