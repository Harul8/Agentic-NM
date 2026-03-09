"""
Auto-Enricher — Fetches content from internet for display; indexing is manual via the UI.

When called from the response pipeline (skip_index=True), we only fetch and extract
content for the current response. No PDFs are saved and no chunks are indexed here.
Indexing is done manually by the user via the Pending indexing UI (official PDFs only).

When skip_index=False (legacy/custom use), the enricher can save PDFs to Google Drive
and index to the vector store. Only official PDF documents are proposed as indexing
candidates; legal portals and news are never proposed for indexing.
"""

import os
import re
import json
import logging
import hashlib
import threading
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
from retrieval.case_law_filename import suggest_case_law_basename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P0: Batch-indexing mode — defer BM25 rebuilds until end of batch
# ---------------------------------------------------------------------------
_batch_lock = threading.Lock()
_batch_mode: bool = False
_bm25_dirty: dict = {}  # bm25_json_path -> chunks_json_path


def _is_likely_pdf_url(url: str) -> bool:
    """
    True if URL is likely a PDF or official document we can index.
    Used so only relevant PDFs (bare acts, case laws) are presented for indexing on the UI,
    not generic HTML pages or non-document links.
    """
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    if ".pdf" in u or u.endswith(".pdf"):
        return True
    if "/pdf/" in u or "/bitstream/" in u:
        return True
    # Official doc paths that often serve or redirect to PDFs
    if "indiacode.nic.in" in u and ("/show-data" in u or "actpdf" in u or "bareact" in u):
        return True
    if "main.sci.gov.in" in u and ("judgment" in u or "jonew" in u):
        return True
    if "judgments.ecourts.gov.in" in u or "ecourts.gov.in" in u:
        return True
    return False


def begin_batch_indexing() -> None:
    """Enter batch mode: BM25 rebuilds are deferred until end_batch_indexing()."""
    global _batch_mode
    with _batch_lock:
        _batch_mode = True
        _bm25_dirty.clear()
    logger.debug("Batch indexing mode started")


def end_batch_indexing() -> None:
    """Exit batch mode and flush all deferred BM25 rebuilds."""
    global _batch_mode
    with _batch_lock:
        _batch_mode = False
        dirty_snapshot = dict(_bm25_dirty)
        _bm25_dirty.clear()
    if not dirty_snapshot:
        return
    from retrieval.hybrid_retriever import load_chunks, save_bm25_index, BM25
    for bm25_path, chunks_path in dirty_snapshot.items():
        try:
            existing_chunks = load_chunks(chunks_path)
            all_texts = [
                existing_chunks[k].get("search_text") or existing_chunks[k].get("full_text") or existing_chunks[k].get("text") or ""
                for k in sorted(existing_chunks.keys(), key=int)
            ]
            bm25 = BM25()
            bm25.fit(all_texts)
            save_bm25_index(bm25, bm25_path)
            logger.info("Flushed BM25 for %s (%d docs)", bm25_path, len(all_texts))
        except Exception as e:
            logger.warning("BM25 flush failed for %s: %s", bm25_path, e)
    logger.debug("Batch indexing ended; %d BM25 index(es) rebuilt", len(dirty_snapshot))


# ---------------------------------------------------------------------------
# P1: Index write lock — serialises FAISS + JSON writes for parallel indexing
# ---------------------------------------------------------------------------
_index_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# P2: LLM classification cache — avoid re-calling LLM for same document text
# ---------------------------------------------------------------------------
_act_classify_cache: dict = {}  # sha256(text[:3000]) -> bool


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes. Tries pdfplumber first, then pypdf as fallback.
    Returns empty string if both fail.
    """
    import io
    if not pdf_bytes or len(pdf_bytes) < 100:
        return ""
    # Try pdfplumber first (better for legal PDFs)
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug("pdfplumber extraction failed: %s", e)
    # Fallback: pypdf (already in requirements)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug("pypdf extraction failed: %s", e)
    return ""


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

    # P1: Serialise FAISS + JSON writes so parallel indexing threads don't corrupt the index
    with _index_write_lock:
        # Save FAISS
        os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
        safe_write_faiss(faiss_index, faiss_index_path)

        # P3: Save chunks JSON — compact (no indent) for faster writes on large corpora
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, separators=(",", ":"), ensure_ascii=False)

        # P0: Rebuild BM25 — deferred in batch mode; immediate otherwise
        with _batch_lock:
            if _batch_mode:
                _bm25_dirty[bm25_json_path] = chunks_json_path
                logger.debug("BM25 rebuild deferred (batch mode) for %s", bm25_json_path)
            else:
                all_texts = [
                    existing_chunks[k].get("search_text") or existing_chunks[k].get("full_text") or existing_chunks[k].get("text") or ""
                    for k in sorted(existing_chunks.keys(), key=int)
                ]
                bm25 = BM25()
                bm25.fit(all_texts)
                save_bm25_index(bm25, bm25_json_path)

    logger.info(f"Added {len(valid_chunks)} chunks. Total now: {len(existing_chunks)}")

    # Invalidate the signatures cache so the next duplicate check sees fresh data
    try:
        from services.indexing_duplicate_check import invalidate_signatures_cache
        invalidate_signatures_cache()
    except Exception:
        pass  # Non-critical; cache will expire naturally via TTL

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
    original_query: str = "",
    skip_index: bool = False,
) -> dict:
    """
    Process a single internet search result:
    1. Fetch content (+ check for PDF)
    2. If PDF found and not skip_index → save to Google Drive, chunk, index to vector store
    3. If no PDF and not skip_index → store as web reference (not indexed as primary source)

    When skip_index=True: only fetch and score; do not save PDF or index. Used to surface
    indexing_candidates to the UI for user-triggered indexing. Only official PDFs (acts, judgments)
    are ever proposed as indexing candidates; news articles are never proposed for indexing.

    Args:
        result: Dict with url, title, snippet, source_tag, tier
        search_type: "bare_act", "case_law", or "both"
        original_query: Original user query for relevance filtering (prevents downloading irrelevant docs)
        skip_index: If True, do not save PDF or index; still return enrichment with content for display.

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
    snippet = result.get("snippet", "")
    source_tag = result.get("source_tag", "UNKNOWN")

    # Relevance check BEFORE downloading
    if not _is_relevant_to_query(title, snippet, original_query):
        logger.info(f"Skipping irrelevant result: '{title}' (not relevant to query)")
        return {
            "url": url,
            "title": title,
            "content": "",
            "pdf_saved": False,
            "indexed": False,
            "source_tag": source_tag,
            "chunks_added": 0,
        }

    logger.info(f"Enriching: [{source_tag}] {title} — {url}")

    # Fetch content
    text_content, pdf_bytes = fetch_content_and_pdf(url)

    # If we have PDF but no/short text (e.g. fetch extractor failed), try extraction here
    if pdf_bytes and len(pdf_bytes) > 1000 and (not text_content or len((text_content or "").strip()) < 200):
        extracted = _extract_text_from_pdf_bytes(pdf_bytes)
        if extracted:
            text_content = extracted
            logger.info("Extracted text from PDF in enricher (fetch had returned empty)")

    # For bare_act candidates: decide from first two pages if this is actually an Act/Law (legislation) or other (notification, circular, etc.)
    is_legislation = True
    if search_type == "bare_act" and (text_content or "").strip() and len((text_content or "").strip()) >= 200:
        first_two_pages = (text_content or "").strip()[:_FIRST_TWO_PAGES_CHARS]
        is_legislation = _is_act_or_law(first_two_pages)
        if not is_legislation:
            logger.info("Document classified as NOT an Act/Law (notification/circular/other): %s", title[:60])

    enrichment = {
        "url": url,
        "title": title,
        "content": text_content[:10000] if text_content else "",  # Store more content (up to 10K) for better summaries
        "pdf_saved": False,
        "indexed": False,
        "source_tag": source_tag,
        "chunks_added": 0,
        "is_legislation": is_legislation,
    }

    if not text_content and not pdf_bytes:
        logger.warning(f"No content fetched from {url}")
        if not skip_index:
            _save_web_reference(result, "")
        return enrichment

    # Determine if this is a bare act or case law (content overrides search_type to prevent misclassification)
    is_case_law = _looks_like_case_law(url, title, text_content)
    doc_type = "case_law" if is_case_law else "bare_act"

    if search_type == "case_law":
        doc_type = "case_law"
    elif search_type == "bare_act" and not is_case_law:
        doc_type = "bare_act"
    # If search_type was "bare_act" but content looks like a judgment, keep doc_type = "case_law"
    # so we do not index judgment PDFs as bare acts (saves to CaseLaws, chunks as case_law)

    if skip_index:
        # Only fetch and score; no save/index. Content still used for current response.
        # Record if we got PDF bytes (for indexing_candidates: official PDFs only).
        enrichment["content"] = (text_content or "")[:10000]
        enrichment["pdf_saved"] = bool(pdf_bytes and len(pdf_bytes) > 1000)
        if doc_type == "case_law" and original_query:
            from retrieval.hybrid_retriever import score_query_document
            doc_text = (text_content or "").strip()
            if doc_text and len(doc_text) >= 100:
                enrichment["_rerank_score"] = score_query_document(original_query, doc_text)
            else:
                enrichment["_rerank_score"] = 0.0
                logger.warning(
                    "Cannot score case law '%s': PDF text extraction failed or too short (%s chars)",
                    title[:50], len(doc_text) if doc_text else 0
                )
        return enrichment

    # For case laws from web: score with cross-encoder using FULL PDF content; index only if score > 5.0
    HIGH_QUALITY_SCORE = 5.0
    if doc_type == "case_law" and original_query:
        from retrieval.hybrid_retriever import score_query_document

        # Require full PDF text extraction - no fallback to title+snippet for scoring
        # If extraction failed, we can't score accurately, so skip or use low score
        doc_text = (text_content or "").strip()
        if not doc_text or len(doc_text) < 100:
            # PDF extraction failed - cannot score accurately without content
            logger.warning(f"Cannot score case law '{title}': PDF text extraction failed or too short ({len(doc_text)} chars)")
            enrichment["_rerank_score"] = 0.0
            enrichment["content"] = f"{title}\n\n{snippet}" if snippet else title  # Use title+snippet only for display
            # Still save PDF if we have bytes, but don't index
            if pdf_bytes and len(pdf_bytes) > 1000:
                filename = suggest_case_law_basename((text_content or "")[:3000], title)
                saved_path = save_pdf_to_drive(pdf_bytes, filename, "CaseLaws")
                enrichment["pdf_saved"] = bool(saved_path)
            return enrichment

        # Score using FULL extracted PDF text (up to 15K chars handled by score_query_document)
        rerank_score = score_query_document(original_query, doc_text)
        enrichment["_rerank_score"] = rerank_score

        # Store full text for downstream (up to 10K chars for better summaries and extraction)
        enrichment["content"] = doc_text[:10000]  # Use full extracted PDF text (up to 10K)

        if pdf_bytes and len(pdf_bytes) > 1000 and rerank_score > HIGH_QUALITY_SCORE:
            subfolder = "CaseLaws"
            filename = suggest_case_law_basename((text_content or "")[:3000], title)
            saved_path = save_pdf_to_drive(pdf_bytes, filename, subfolder)
            enrichment["pdf_saved"] = bool(saved_path)
            if saved_path and text_content:
                chunks = chunk_case_law(text_content, saved_path)
                added = add_case_law_chunks(chunks)
                enrichment["indexed"] = added > 0
                enrichment["chunks_added"] = added
                logger.info(f"Indexed {added} chunks from PDF (score {rerank_score:.2f} > {HIGH_QUALITY_SCORE}): {title}")
        elif pdf_bytes and len(pdf_bytes) > 1000:
            # Save PDF to Drive but do NOT index (score <= 5.0)
            filename = suggest_case_law_basename((text_content or "")[:3000], title)
            saved_path = save_pdf_to_drive(pdf_bytes, filename, "CaseLaws")
            enrichment["pdf_saved"] = bool(saved_path)
            if not saved_path:
                _save_web_reference(result, text_content)
        else:
            _save_web_reference(result, text_content)
        return enrichment

    # Bare acts or unscored: save and index as before (only as bare act if classified as legislation)
    if pdf_bytes and len(pdf_bytes) > 1000:
        if doc_type == "bare_act" and not is_legislation:
            # Document is not an Act/Law (e.g. notification, circular); do not save as BareAct or index
            _save_web_reference(result, text_content)
            logger.info(f"Not an Act/Law; saved as web reference only: {title[:60]}")
        else:
            subfolder = "CaseLaws" if doc_type == "case_law" else "BareActs"
            filename = suggest_case_law_basename((text_content or "")[:3000], title) if doc_type == "case_law" else title
            saved_path = save_pdf_to_drive(pdf_bytes, filename, subfolder)
            enrichment["pdf_saved"] = bool(saved_path)

            if saved_path and text_content:
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
        _save_web_reference(result, text_content)
        logger.info(f"No PDF available, saved as web reference: {title}")

    return enrichment


def enrich_from_gap_results(
    gap_results: dict,
    search_type: str = "both",
    original_query: str = "",
    local_high_quality_count: int = 0,
    target_high_quality: int = 5,
    skip_index: bool = False,
    pending_only: bool = False,
    max_bare_acts_to_enrich: int = None,
    max_case_laws_to_enrich: int = None,
    progress_callback=None,
) -> dict:
    """
    Process all search results from gap filling.
    Case laws: only official PDFs; extract FULL PDF content, score with cross-encoder;
    process top-ranked PDFs first, stop when we have enough results (score >= WEB_MIN_SCORE).
    Index only if score > 5.0 and skip_index is False.

    When skip_index=True: fetch and score only; do not save or index. Caller uses
    enriched_bare_acts/enriched_case_laws to build indexing_candidates for the UI.
    Indexing is done manually via the Pending indexing UI (official PDFs only).

    When pending_only=True (response path): do NOT fetch or download. Only build
    enriched_bare_acts/enriched_case_laws from raw result metadata (title, url, snippet).
    No PDF download, no chunking — removes 40+ s per request. User can index later via UI.
    """
    HIGH_QUALITY_SCORE = 5.0
    WEB_MIN_SCORE = 1.0
    TARGET_RESULTS = 5
    DEFAULT_MAX_BARE_ACTS = 3   # Fetch at most 3 PDFs per web round (was 15); each PDF = 100+ sections
    DEFAULT_MAX_CASE_LAWS = 10
    MAX_BARE_ACTS_TO_ENRICH = max_bare_acts_to_enrich if max_bare_acts_to_enrich is not None else DEFAULT_MAX_BARE_ACTS
    MAX_CASE_LAWS_TO_ENRICH = max_case_laws_to_enrich if max_case_laws_to_enrich is not None else DEFAULT_MAX_CASE_LAWS

    # Old acts superseded in 2024 — IPC→BNS, CrPC→BNSS, IEA→BSA.
    # Fetching their full PDFs wastes time and injects obsolete law.
    _OLD_ACT_PDF_PATTERNS = (
        "the-indian-penal-code",
        "ipc_act",
        "ipc-act",
        "1860",          # IPC 1860 PDFs
        "code-of-criminal-procedure",
        "crpc",
        "indian-evidence-act",
        "a1872",         # Evidence Act
        "a1860",         # IPC
        "a1973",         # CrPC
    )

    def _skip_low_value(result: dict) -> bool:
        """Filter out search pages, generic home pages, old IPC/CrPC act PDFs, and similar."""
        url = (result.get("url") or "").lower()
        title = (result.get("title") or "").strip()
        if "/search/?forminput=" in url or "forminput=" in url:
            return True
        if title.startswith("Home | Legislative Department") or title == "Home | Legislative Department":
            return True
        if "indiankanoon.org/search/" in url:
            return True
        # Skip old-law PDFs (IPC 1860, CrPC 1973, IEA 1872) — superseded by BNS/BNSS/BSA
        if any(pat in url for pat in _OLD_ACT_PDF_PATTERNS):
            return True
        return False

    def _sort_key_bare_act(r: dict) -> tuple:
        """
        Prefer: section-level URLs > official PDFs > other.
        Section-level URLs (indiacode.nic.in/show-data?...&orderno=X) fetch ONE section fast.
        Whole-act PDFs (indiacode.nic.in/bitstream/...) fetch 100+ sections and are very slow.
        """
        tag = (r.get("source_tag") or "").upper()
        url = (r.get("url") or "").lower()
        is_off = 0 if tag in ("OFFICIAL", "OFFICIAL_COURT") else 1
        # section-level URL = best; whole-act PDF = worst
        is_section_level = 0 if ("show-data" in url or "orderno=" in url) else 1
        is_whole_pdf = 0 if (".pdf" in url or "/bitstream/" in url) else 1
        tier = r.get("tier", 99)
        return (is_off, is_section_level, is_whole_pdf, tier)

    summary = {
        "pdfs_saved": 0,
        "chunks_indexed": 0,
        "web_references_saved": 0,
        "enriched_bare_acts": [],
        "enriched_case_laws": [],
        "enriched_news": [],
    }

    bare_act_results = [r for r in gap_results.get("bare_act_results", []) if not _skip_low_value(r)]
    bare_act_results.sort(key=_sort_key_bare_act)
    bare_act_results = bare_act_results[:MAX_BARE_ACTS_TO_ENRICH]

    case_law_results_pre = [
        r for r in gap_results.get("case_law_results", [])
        if (r.get("source_tag") or "").upper() in ("OFFICIAL", "OFFICIAL_COURT") and not _skip_low_value(r)
    ]
    case_law_results_pre = case_law_results_pre[:MAX_CASE_LAWS_TO_ENRICH]
    total_download = len(bare_act_results) + len(case_law_results_pre)
    download_n = 0

    # Pending-only path: no fetch, no download, no chunking. Only PDF/official-doc URLs for indexing UI.
    if pending_only:
        for r in bare_act_results:
            url = r.get("url", "")
            if not _is_likely_pdf_url(url):
                continue
            summary["enriched_bare_acts"].append({
                "url": url,
                "title": r.get("title", "Unknown"),
                "snippet": r.get("snippet", ""),
                "content": "",
                "source_tag": r.get("source_tag", "UNKNOWN"),
                "suggested_category": "bare_act",
                "_rerank_score": 0.0,
            })
        for r in case_law_results_pre:
            url = r.get("url", "")
            if not _is_likely_pdf_url(url):
                continue
            summary["enriched_case_laws"].append({
                "url": url,
                "title": r.get("title", "Unknown"),
                "snippet": r.get("snippet", ""),
                "content": "",
                "source_tag": r.get("source_tag", "UNKNOWN"),
                "suggested_category": "case_law",
                "_rerank_score": 0.0,
            })
        # News is never shown for indexing (only bare_act/case_law go to pending_indexing_list)
        for result in gap_results.get("news_results", []):
            summary["enriched_news"].append({
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet", ""),
                "source_tag": "NEWS_REFERENCE",
            })
        logger.info(
            "Enrichment (pending_only): %d bare-act + %d case-law PDF candidates for Pending indexing UI (no fetch).",
            len(summary["enriched_bare_acts"]), len(summary["enriched_case_laws"]),
        )
        return summary

    # P0: Batch mode
    # Guard against nesting: if the caller (e.g. api_server.py /indexing/run) already entered
    # batch mode we must NOT enter or exit it here — that would clobber the caller's dirty-set.
    _already_in_batch = _batch_mode
    if not skip_index and not _already_in_batch:
        begin_batch_indexing()
        logger.debug("enrich_from_gap_results: batch indexing mode entered (%d docs)", total_download)

    # P2 (dedup cache): Pre-load existing-signatures cache once for the whole batch so that
    # every enrich_from_search_result() call does NOT hit the chunk JSON files from disk
    # repeatedly. get_existing_signatures() has a 5-minute TTL and invalidate_signatures_cache()
    # is called after actual indexing, so the next batch always sees fresh data.
    # We also store the function reference so we can use it inside the loops without
    # repeated imports (Python caches the module, but the name lookup is cleaner this way).
    _existing_sigs = None
    _dup_check = None
    if not skip_index:
        try:
            from services.indexing_duplicate_check import (
                get_existing_signatures,
                is_duplicate_of_existing as _is_dup_fn,
            )
            _existing_sigs = get_existing_signatures()
            _dup_check = _is_dup_fn
            logger.debug(
                "Dedup cache pre-loaded: %d bare-act, %d case-law signatures",
                len(_existing_sigs.get("bare_act", [])),
                len(_existing_sigs.get("case_law", [])),
            )
        except Exception as _pre_e:
            logger.warning("Could not pre-load dedup signatures: %s", _pre_e)

    try:
        for result in bare_act_results:
            # Dedup pre-check: skip title-matched duplicates before the expensive HTTP/PDF download.
            if _dup_check is not None and _existing_sigs is not None:
                _title = result.get("title", "")
                if _dup_check(_title, existing=_existing_sigs, suggested_category="bare_act"):
                    logger.info("Dedup: bare act already in store, skipping download: %s", _title[:60])
                    continue
            enrichment = enrich_from_search_result(result, "bare_act", original_query, skip_index=skip_index)
            download_n += 1
            if progress_callback:
                progress_callback(download_n, total_download)
            if enrichment["pdf_saved"]:
                summary["pdfs_saved"] += 1
            summary["chunks_indexed"] += enrichment["chunks_added"]
            if enrichment["content"] and enrichment.get("is_legislation", True):
                summary["enriched_bare_acts"].append(enrichment)

        # Case laws: only official PDFs; process in ranking order, stop when we have enough
        web_high_quality_count = 0
        needed_high_quality = max(0, target_high_quality - local_high_quality_count)
        case_law_results = case_law_results_pre
        # When pull_all (max_case_laws_to_enrich > default), allow more results; otherwise cap at TARGET_RESULTS.
        case_law_cap = max(TARGET_RESULTS, MAX_CASE_LAWS_TO_ENRICH)
        for result in case_law_results:
            # Early exit: stop if we have enough high-quality (score > 5.0) OR enough total results (score >= WEB_MIN_SCORE)
            if web_high_quality_count >= needed_high_quality and MAX_CASE_LAWS_TO_ENRICH <= DEFAULT_MAX_CASE_LAWS:
                logger.info(f"Reached {target_high_quality} high-quality case laws (score > {HIGH_QUALITY_SCORE}), stopping")
                break
            if len(summary["enriched_case_laws"]) >= case_law_cap:
                logger.info(f"Reached {case_law_cap} case laws with score >= {WEB_MIN_SCORE}, stopping enrichment")
                break

            # Dedup pre-check: skip case laws whose title is already in the store.
            if _dup_check is not None and _existing_sigs is not None:
                _title = result.get("title", "")
                if _dup_check(_title, existing=_existing_sigs, suggested_category="case_law"):
                    logger.info("Dedup: case law already in store, skipping download: %s", _title[:60])
                    continue

            enrichment = enrich_from_search_result(result, "case_law", original_query, skip_index=skip_index)
            download_n += 1
            if progress_callback:
                progress_callback(download_n, total_download)
            source_tag = result.get("source_tag", "UNKNOWN")
            if enrichment["pdf_saved"]:
                summary["pdfs_saved"] += 1
            summary["chunks_indexed"] += enrichment["chunks_added"]
            score = enrichment.get("_rerank_score", 0)
            title_short = (enrichment.get("title") or "Unknown")[:60]
            passed = enrichment.get("content") and score >= WEB_MIN_SCORE
            logger.info(
                "Web case law: '%s' score=%.2f (threshold=%.1f) %s",
                title_short, score, WEB_MIN_SCORE, "included" if passed else "excluded"
            )

            if score > HIGH_QUALITY_SCORE:
                web_high_quality_count += 1

            if passed:
                summary["enriched_case_laws"].append(enrichment)

        for result in gap_results.get("news_results", []):
            if not skip_index:
                _save_web_reference(result, "")
                summary["web_references_saved"] += 1
            summary["enriched_news"].append({
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet", ""),
                "source_tag": "NEWS_REFERENCE",
            })

        total_fetched = len(summary["enriched_bare_acts"]) + len(summary["enriched_case_laws"])
        if skip_index:
            logger.info(
                "Enrichment complete: %d docs fetched for display. "
                "Indexing available via the Pending indexing UI (official PDFs only).",
                total_fetched,
            )
        else:
            logger.info(
                "Enrichment complete: %d docs fetched, %d chunks indexed "
                "(BM25 rebuild deferred to batch flush).",
                total_fetched, summary["chunks_indexed"],
            )
        return summary

    finally:
        # P0: Always flush deferred BM25 rebuilds when we own the batch context.
        # The 'finally' block runs even on 'return', so BM25 is always flushed.
        if not skip_index and not _already_in_batch:
            end_batch_indexing()
            logger.debug("enrich_from_gap_results: batch indexing mode exited")


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

def _is_relevant_to_query(title: str, snippet: str, original_query: str) -> bool:
    """
    Quick relevance check: does the title/snippet contain keywords from the original query?
    Prevents downloading completely unrelated documents (e.g. Income Tax Act when query is about land acquisition).
    """
    if not original_query or not original_query.strip():
        return True  # No query to check against, allow it
    
    query_lower = original_query.lower()
    title_lower = (title or "").lower()
    snippet_lower = (snippet or "")[:500].lower()
    combined = f"{title_lower} {snippet_lower}"
    
    # Extract key terms from query (2+ character words, excluding common stopwords)
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "can", "must", "shall"}
    query_words = [w for w in query_lower.split() if len(w) >= 3 and w not in stopwords]
    
    if not query_words:
        return True  # No meaningful keywords, allow it
    
    # Check if at least 2 key terms from query appear in title/snippet
    matches = sum(1 for word in query_words[:10] if word in combined)  # Check top 10 query words
    relevance_threshold = min(2, len(query_words) // 2)  # At least 2 matches, or half of query words
    
    if matches >= relevance_threshold:
        return True
    
    # Special case: if title contains completely unrelated act names (Income Tax, Gratuity, Court Fees, etc.)
    # and query is about something else, reject it
    unrelated_acts = ["income tax", "gratuity", "court fee", "court-fee", "motor vehicle", "companies act", "contract act", "sale of goods"]
    if any(act in title_lower for act in unrelated_acts):
        # Check if query is about these acts
        if not any(act in query_lower for act in unrelated_acts):
            logger.info(f"Skipping irrelevant document: '{title}' (query: '{original_query[:80]}')")
            return False
    
    return True  # Default: allow if unsure


# Approximate "first two pages" for act vs judgment when we only have plain text (no page boundaries)
_FIRST_TWO_PAGES_CHARS = 3000


def _is_act_or_law(first_two_pages_text: str) -> bool:
    """
    Use the LLM to decide if the document (from first two pages) is an Act/Law (legislation)
    or something else (notification, circular, order, gazette notice, etc.).
    Returns True only when the model says it is an Act or Law; otherwise False so we don't use it as bare act.
    """
    if not first_two_pages_text or len(first_two_pages_text.strip()) < 150:
        return True  # Too short to classify; allow (backward compat)
    text = first_two_pages_text.strip()[:3000]
    # P2: Check in-memory cache before calling LLM
    cache_key = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if cache_key in _act_classify_cache:
        logger.debug("Act/law classification cache hit")
        return _act_classify_cache[cache_key]
    try:
        from llm.ollama_client import ask_llm
        prompt = f"""You are classifying a legal document from India (e.g. legislative.gov.in). Below is text from the first two pages.

Is this document an **Act or Law** (legislation: Central/State Act, Ordinance, Code, Regulation) that creates or amends law?
Or is it something **other** (e.g. notification, circular, order, gazette notice, clarification, rules notification, amendment notification that only notifies without being the Act text)?

Reply with exactly one of these two words:
ACT_OR_LAW
OTHER

Document excerpt:
{text}

Your one-word reply:"""
        reply = (ask_llm(prompt, timeout=60) or "").strip().upper()
        result = "ACT_OR_LAW" in reply or reply == "ACT_OR_LAW"
        _act_classify_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning("LLM act/law classification failed: %s; treating as legislation", e)
        return True  # On failure, allow (backward compat)


def _looks_like_case_law(url: str, title: str, content: str) -> bool:
    """Determine if a document is a case law (vs bare act) based on first two pages of content."""
    first_two_pages = (content or "")[:_FIRST_TWO_PAGES_CHARS]
    combined = f"{url} {title} {first_two_pages}".lower()
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
