"""
Incremental Case Law Indexer — Thin wrapper around v2 modules.

All chunking → Ingestion.smart_chunker
All indexing → retrieval.auto_enricher (FAISS v2 + BM25)
"""

import os
import logging

from config import CASELAW_DIR, VECTOR_STORE
from Ingestion.smart_chunker import chunk_case_law
from retrieval.auto_enricher import enrich_from_search_result

logger = logging.getLogger(__name__)


def index_new_case_laws(case_laws: list) -> dict:
    """
    Add new case laws to the v2 vector store.

    case_laws: list of {title, url, content, relevant_portion, source}
    Returns: {success: bool, chunks_added: int, message: str}
    """
    if not case_laws:
        return {"success": False, "chunks_added": 0, "message": "No case laws to index"}

    os.makedirs(VECTOR_STORE, exist_ok=True)
    os.makedirs(CASELAW_DIR, exist_ok=True)

    total_enriched = 0
    for case in case_laws:
        title = case.get("title", "Unknown")
        url = case.get("url", "")
        content = (
            case.get("relevant_portion")
            or case.get("content")
            or case.get("snippet")
            or ""
        )
        if not content:
            continue

        # Use auto_enricher which handles: save to Drive + chunk via smart_chunker + index to FAISS+BM25
        result = enrich_from_search_result(
            result={
                "title": title,
                "url": url,
                "snippet": content[:400],
                "content_text": content,
                "pdf_bytes": None,
                "source_tier": "tier2_official",
            },
            search_type="case_law",
        )
        if result.get("indexed"):
            total_enriched += 1

    if total_enriched == 0:
        # Fallback: save as text files so data is not lost
        for case in case_laws:
            title = case.get("title", "Unknown")
            content = case.get("relevant_portion") or case.get("content") or ""
            if not content:
                continue
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
            filepath = os.path.join(CASELAW_DIR, f"Indexed_{safe_title}.txt")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"TITLE: {title}\nSOURCE: {case.get('url', '')}\n\n{content}")
            except Exception:
                pass

        return {
            "success": True,
            "chunks_added": 0,
            "message": f"Saved {len(case_laws)} case law(s) as text (enricher unavailable)",
        }

    return {
        "success": True,
        "chunks_added": total_enriched,
        "message": f"Successfully indexed {total_enriched} case law(s) via v2 pipeline",
    }
