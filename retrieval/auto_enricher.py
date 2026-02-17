"""
Auto-Enricher — Downloads PDFs from internet, saves to Google Drive, indexes to vector store.

This creates a self-growing legal database: every time the system fetches a judgment
or bare act from the internet, it saves the original PDF to Google Drive and adds
it to the FAISS + BM25 vector store. Next time the same topic is queried, it will
be found locally (Tier 1) without needing internet.

For articles/news (no PDF), stores metadata in a web_references table but does NOT
index them as if they were primary legal sources.
"""

import os
import re
import json
import logging
from typing import Optional

from config import (
    BARE_ACTS_DIR,
    CASELAW_DIR,
    VECTOR_STORE,
    BARE_INDEX_V2,
    BARE_CHUNKS_V2,
    BARE_BM25_INDEX,
    CASE_INDEX_V2,
    CASE_CHUNKS_V2,
    CASE_BM25_INDEX,
    WEB_REFERENCES_DB,
)
from retrieval.tiered_search import classify_source, fetch_content_and_pdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF Saving to Google Drive
# ---------------------------------------------------------------------------

def _safe_filename(title: str) -> str:
    """Convert a title to a safe filename."""
    name = re.sub(r"[^\w\s\-\.]", "", title.strip())
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "document"


def save_pdf_to_drive(pdf_bytes: bytes, filename: str, subfolder: str) -> str:
    """
    Save PDF bytes to Google Drive (BareActs/ or CaseLaws/ directory).

    Args:
        pdf_bytes: Raw PDF content
        filename: Desired filename (without extension)
        subfolder: "BareActs" or "CaseLaws"

    Returns:
        Full path to saved file, or empty string on failure.
    """
    target_dir = BARE_ACTS_DIR if subfolder == "BareActs" else CASELAW_DIR
    os.makedirs(target_dir, exist_ok=True)

    safe_name = _safe_filename(filename)
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    filepath = os.path.join(target_dir, safe_name)

    # Don't overwrite existing files
    if os.path.exists(filepath):
        logger.info(f"File already exists, skipping: {filepath}")
        return filepath

    try:
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"Saved PDF to Google Drive: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save PDF to {filepath}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Incremental Indexing (add to existing v2 index)
# ---------------------------------------------------------------------------

def _get_embedder():
    """Get the embedding model (reuse from hybrid_retriever)."""
    from retrieval.hybrid_retriever import _get_embedder as get_emb
    return get_emb()


def add_chunks_to_index(
    new_chunks: list,
    faiss_index_path: str,
    chunks_json_path: str,
    bm25_json_path: str,
) -> int:
    """
    Add new chunks to an existing FAISS + BM25 index.

    Args:
        new_chunks: List of chunk dicts (must have 'search_text' or 'full_text')
        faiss_index_path: Path to FAISS index file
        chunks_json_path: Path to chunks JSON file
        bm25_json_path: Path to BM25 index JSON file

    Returns:
        Number of chunks actually added.
    """
    import faiss
    import numpy as np
    from retrieval.hybrid_retriever import (
        safe_read_faiss,
        safe_write_faiss,
        load_chunks,
        load_bm25_index,
        save_bm25_index,
        BM25,
    )

    if not new_chunks:
        return 0

    embedder = _get_embedder()

    # Load existing index and chunks
    existing_chunks = load_chunks(chunks_json_path)
    faiss_index, ok = safe_read_faiss(faiss_index_path)

    # Determine starting chunk ID
    if existing_chunks:
        max_key = max(int(k) for k in existing_chunks.keys())
        next_id = max_key + 1
    else:
        next_id = 0

    # Embed new chunks
    texts_to_embed = []
    valid_chunks = []
    for chunk in new_chunks:
        text = (
            chunk.get("search_text")
            or chunk.get("full_text")
            or chunk.get("text")
            or ""
        ).strip()
        if len(text) < 30:
            continue
        texts_to_embed.append(text)
        valid_chunks.append(chunk)

    if not valid_chunks:
        return 0

    logger.info(f"Embedding {len(valid_chunks)} new chunks...")
    embeddings = embedder.encode(
        texts_to_embed,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Add to FAISS index
    dim = embeddings.shape[1]
    if faiss_index is None:
        faiss_index = faiss.IndexFlatIP(dim)

    faiss_index.add(np.array(embeddings, dtype="float32"))

    # Add to chunks JSON
    for i, chunk in enumerate(valid_chunks):
        key = str(next_id + i)
        existing_chunks[key] = chunk

    # Save FAISS
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    safe_write_faiss(faiss_index, faiss_index_path)

    # Save chunks JSON
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)

    # Rebuild BM25 index (full rebuild since BM25 doesn't support incremental)
    all_texts = []
    for key in sorted(existing_chunks.keys(), key=int):
        chunk = existing_chunks[key]
        text = (
            chunk.get("search_text")
            or chunk.get("full_text")
            or chunk.get("text")
            or ""
        )
        all_texts.append(text)

    bm25 = BM25()
    bm25.fit(all_texts)
    save_bm25_index(bm25, bm25_json_path)

    logger.info(f"Added {len(valid_chunks)} chunks. Total now: {len(existing_chunks)}")
    return len(valid_chunks)


def add_bare_act_chunks(new_chunks: list) -> int:
    """Add new bare act chunks to the v2 index."""
    return add_chunks_to_index(
        new_chunks, BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX
    )


def add_case_law_chunks(new_chunks: list) -> int:
    """Add new case law chunks to the v2 index."""
    return add_chunks_to_index(
        new_chunks, CASE_INDEX_V2, CASE_CHUNKS_V2, CASE_BM25_INDEX
    )


# ---------------------------------------------------------------------------
# Full Auto-Enrich Pipeline
# ---------------------------------------------------------------------------

def enrich_from_search_result(
    result: dict,
    search_type: str = "both",
) -> dict:
    """
    Process a single internet search result:
    1. Fetch content (+ check for PDF)
    2. If PDF found → save to Google Drive, chunk, index to vector store
    3. If no PDF → store as web reference (not indexed as primary source)

    Args:
        result: Dict with url, title, snippet, source_tag, tier
        search_type: "bare_act", "case_law", or "both"

    Returns:
        Dict with enrichment result:
        {
            "url": ...,
            "title": ...,
            "content": ...,
            "pdf_saved": bool,
            "indexed": bool,
            "source_tag": ...,
            "chunks_added": int,
        }
    """
    from Ingestion.smart_chunker import chunk_bare_act, chunk_case_law

    url = result.get("url", "")
    title = result.get("title", "Unknown")
    source_tag = result.get("source_tag", "UNKNOWN")

    logger.info(f"Enriching: [{source_tag}] {title} — {url}")

    # Fetch content
    text_content, pdf_bytes = fetch_content_and_pdf(url)

    enrichment = {
        "url": url,
        "title": title,
        "content": text_content[:3000] if text_content else "",
        "pdf_saved": False,
        "indexed": False,
        "source_tag": source_tag,
        "chunks_added": 0,
    }

    if not text_content and not pdf_bytes:
        logger.warning(f"No content fetched from {url}")
        _save_web_reference(result, "")
        return enrichment

    # Determine if this is a bare act or case law
    is_case_law = _looks_like_case_law(url, title, text_content)
    doc_type = "case_law" if is_case_law else "bare_act"

    if search_type == "bare_act":
        doc_type = "bare_act"
    elif search_type == "case_law":
        doc_type = "case_law"

    # If we have a PDF, save it and index it
    if pdf_bytes and len(pdf_bytes) > 1000:
        subfolder = "CaseLaws" if doc_type == "case_law" else "BareActs"
        saved_path = save_pdf_to_drive(pdf_bytes, title, subfolder)
        enrichment["pdf_saved"] = bool(saved_path)

        if saved_path and text_content:
            # Chunk and index
            if doc_type == "case_law":
                chunks = chunk_case_law(text_content, saved_path)
                added = add_case_law_chunks(chunks)
            else:
                chunks = chunk_bare_act(text_content, saved_path)
                added = add_bare_act_chunks(chunks)

            enrichment["indexed"] = added > 0
            enrichment["chunks_added"] = added
            logger.info(f"Indexed {added} chunks from PDF: {title}")
    else:
        # No PDF — save as web reference only (don't index as primary source)
        _save_web_reference(result, text_content)
        logger.info(f"No PDF available, saved as web reference: {title}")

    return enrichment


def enrich_from_gap_results(gap_results: dict, search_type: str = "both") -> dict:
    """
    Process all search results from gap filling.

    Args:
        gap_results: Output from tiered_search.search_for_gaps()
        search_type: Default doc type if unclear

    Returns:
        Summary of enrichment.
    """
    summary = {
        "pdfs_saved": 0,
        "chunks_indexed": 0,
        "web_references_saved": 0,
        "enriched_bare_acts": [],
        "enriched_case_laws": [],
        "enriched_news": [],
    }

    for result in gap_results.get("bare_act_results", []):
        enrichment = enrich_from_search_result(result, "bare_act")
        if enrichment["pdf_saved"]:
            summary["pdfs_saved"] += 1
        summary["chunks_indexed"] += enrichment["chunks_added"]
        if enrichment["content"]:
            summary["enriched_bare_acts"].append(enrichment)

    for result in gap_results.get("case_law_results", []):
        enrichment = enrich_from_search_result(result, "case_law")
        if enrichment["pdf_saved"]:
            summary["pdfs_saved"] += 1
        summary["chunks_indexed"] += enrichment["chunks_added"]
        if enrichment["content"]:
            summary["enriched_case_laws"].append(enrichment)

    for result in gap_results.get("news_results", []):
        _save_web_reference(result, "")
        summary["web_references_saved"] += 1
        summary["enriched_news"].append({
            "url": result.get("url"),
            "title": result.get("title"),
            "snippet": result.get("snippet", ""),
            "source_tag": "NEWS_REFERENCE",
        })

    logger.info(
        f"Enrichment complete: {summary['pdfs_saved']} PDFs saved, "
        f"{summary['chunks_indexed']} chunks indexed, "
        f"{summary['web_references_saved']} web references saved"
    )
    return summary


# ---------------------------------------------------------------------------
# Web References Storage (for articles/news — NOT primary legal sources)
# ---------------------------------------------------------------------------

def _save_web_reference(result: dict, content: str):
    """Save a web reference (article/news) to the references table."""
    try:
        refs = []
        if os.path.exists(WEB_REFERENCES_DB):
            with open(WEB_REFERENCES_DB, encoding="utf-8") as f:
                refs = json.load(f)

        # Don't duplicate
        existing_urls = {r.get("url") for r in refs}
        url = result.get("url", "")
        if url in existing_urls:
            return

        from datetime import datetime
        refs.append({
            "url": url,
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "source_tag": result.get("source_tag", "UNKNOWN"),
            "tier": result.get("tier", 99),
            "content_preview": content[:500] if content else "",
            "fetched_at": datetime.now().isoformat(),
        })

        os.makedirs(os.path.dirname(WEB_REFERENCES_DB), exist_ok=True)
        with open(WEB_REFERENCES_DB, "w", encoding="utf-8") as f:
            json.dump(refs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save web reference: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_case_law(url: str, title: str, content: str) -> bool:
    """Determine if a document is a case law (vs bare act) based on content."""
    combined = f"{url} {title} {(content or '')[:2000]}".lower()
    case_indicators = [
        " v. ", " v/s ", " vs ", "appellant", "respondent",
        "petitioner", "judgment", "judgement", "hon'ble",
        "supreme court", "high court", "bench", "coram",
    ]
    bare_act_indicators = [
        "bare act", "act,", "code,", "ordinance", "regulation",
        "section", "chapter", "schedule", "notification",
        "indiacode", "legislative", "gazette",
    ]

    case_score = sum(1 for i in case_indicators if i in combined)
    act_score = sum(1 for i in bare_act_indicators if i in combined)

    return case_score > act_score
