"""
Incremental Bare Act Indexer — Thin wrapper around v2 modules.

All chunking → Ingestion.smart_chunker
All indexing → retrieval.auto_enricher (FAISS v2 + BM25)
"""

import os
import logging

from config import BARE_ACTS_DIR, VECTOR_STORE
from Ingestion.smart_chunker import chunk_bare_act
from retrieval.auto_enricher import enrich_from_search_result

logger = logging.getLogger(__name__)


def index_new_bare_acts(bare_acts: list) -> dict:
    """
    Add new bare act sections to the v2 vector store.

    bare_acts: list of {title, url, text, act_name, source}
    Returns: {success: bool, chunks_added: int, message: str}
    """
    if not bare_acts:
        return {"success": False, "chunks_added": 0, "message": "No bare acts to index"}

    os.makedirs(VECTOR_STORE, exist_ok=True)
    os.makedirs(BARE_ACTS_DIR, exist_ok=True)

    total_enriched = 0
    for item in bare_acts:
        title = item.get("title") or item.get("act_name", "Unknown")
        url = item.get("url", "")
        text = item.get("text") or item.get("content", "")
        if not text:
            continue

        # Use auto_enricher which handles: save to Drive + chunk via smart_chunker + index to FAISS+BM25
        result = enrich_from_search_result(
            result={
                "title": title,
                "url": url,
                "snippet": text[:400],
                "content_text": text,
                "pdf_bytes": None,  # No PDF bytes from confirmed materials
                "source_tier": "tier2_official",
            },
            search_type="bare_act",
        )
        if result.get("indexed"):
            total_enriched += 1

    if total_enriched == 0:
        # Fallback: save as text files so data is not lost
        for item in bare_acts:
            title = item.get("title") or item.get("act_name", "Unknown")
            text = item.get("text") or item.get("content", "")
            if not text:
                continue
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(title))[:80]
            filepath = os.path.join(BARE_ACTS_DIR, f"Indexed_{safe_title}.txt")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"TITLE: {title}\nSOURCE: {item.get('url', '')}\n\n{text}")
            except Exception:
                pass

        return {
            "success": True,
            "chunks_added": 0,
            "message": f"Saved {len(bare_acts)} bare act(s) as text (enricher unavailable)",
        }

    return {
        "success": True,
        "chunks_added": total_enriched,
        "message": f"Successfully indexed {total_enriched} bare act section(s) via v2 pipeline",
    }
