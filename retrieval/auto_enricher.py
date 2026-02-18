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
    original_query: str = "",
) -> dict:
    """
    Process a single internet search result:
    1. Fetch content (+ check for PDF)
    2. If PDF found → save to Google Drive, chunk, index to vector store
    3. If no PDF → store as web reference (not indexed as primary source)

    Args:
        result: Dict with url, title, snippet, source_tag, tier
        search_type: "bare_act", "case_law", or "both"
        original_query: Original user query for relevance filtering (prevents downloading irrelevant docs)

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

    enrichment = {
        "url": url,
        "title": title,
        "content": text_content[:10000] if text_content else "",  # Store more content (up to 10K) for better summaries
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
                saved_path = save_pdf_to_drive(pdf_bytes, title, "CaseLaws")
                enrichment["pdf_saved"] = bool(saved_path)
            return enrichment

        # Score using FULL extracted PDF text (up to 15K chars handled by score_query_document)
        rerank_score = score_query_document(original_query, doc_text)
        enrichment["_rerank_score"] = rerank_score

        # Store full text for downstream (up to 10K chars for better summaries and extraction)
        enrichment["content"] = doc_text[:10000]  # Use full extracted PDF text (up to 10K)

        if pdf_bytes and len(pdf_bytes) > 1000 and rerank_score > HIGH_QUALITY_SCORE:
            subfolder = "CaseLaws"
            saved_path = save_pdf_to_drive(pdf_bytes, title, subfolder)
            enrichment["pdf_saved"] = bool(saved_path)
            if saved_path and text_content:
                chunks = chunk_case_law(text_content, saved_path)
                added = add_case_law_chunks(chunks)
                enrichment["indexed"] = added > 0
                enrichment["chunks_added"] = added
                logger.info(f"Indexed {added} chunks from PDF (score {rerank_score:.2f} > {HIGH_QUALITY_SCORE}): {title}")
        elif pdf_bytes and len(pdf_bytes) > 1000:
            # Save PDF to Drive but do NOT index (score <= 5.0)
            saved_path = save_pdf_to_drive(pdf_bytes, title, "CaseLaws")
            enrichment["pdf_saved"] = bool(saved_path)
            if not saved_path:
                _save_web_reference(result, text_content)
        else:
            _save_web_reference(result, text_content)
        return enrichment

    # Bare acts or unscored: save and index as before
    if pdf_bytes and len(pdf_bytes) > 1000:
        subfolder = "CaseLaws" if doc_type == "case_law" else "BareActs"
        saved_path = save_pdf_to_drive(pdf_bytes, title, subfolder)
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
) -> dict:
    """
    Process all search results from gap filling.
    Case laws: only official PDFs; extract FULL PDF content, score with cross-encoder;
    process top-ranked PDFs first, stop when we have enough results (score > 2.0).
    Index only if score > 5.0.
    """
    HIGH_QUALITY_SCORE = 5.0
    WEB_MIN_SCORE = 2.0  # Lower threshold - filter unrelated ones, but include more relevant results
    TARGET_RESULTS = 5  # Stop when we have this many case laws with score > WEB_MIN_SCORE
    
    summary = {
        "pdfs_saved": 0,
        "chunks_indexed": 0,
        "web_references_saved": 0,
        "enriched_bare_acts": [],
        "enriched_case_laws": [],
        "enriched_news": [],
    }

    for result in gap_results.get("bare_act_results", []):
        enrichment = enrich_from_search_result(result, "bare_act", original_query)
        if enrichment["pdf_saved"]:
            summary["pdfs_saved"] += 1
        summary["chunks_indexed"] += enrichment["chunks_added"]
        if enrichment["content"]:
            summary["enriched_bare_acts"].append(enrichment)

    # Case laws: only official PDFs; process in ranking order, stop when we have enough
    web_high_quality_count = 0
    needed_high_quality = max(0, target_high_quality - local_high_quality_count)
    case_law_results = [
        r for r in gap_results.get("case_law_results", [])
        if r.get("source_tag") == "OFFICIAL_COURT"
    ]
    # Results should already be ranked by tiered_search, process top to bottom
    for result in case_law_results:
        # Early exit: stop if we have enough high-quality (score > 5.0) OR enough total results (score > 2.0)
        if web_high_quality_count >= needed_high_quality:
            logger.info(f"Reached {target_high_quality} high-quality case laws (score > {HIGH_QUALITY_SCORE}), stopping")
            break
        if len(summary["enriched_case_laws"]) >= TARGET_RESULTS:
            logger.info(f"Reached {TARGET_RESULTS} case laws with score > {WEB_MIN_SCORE}, stopping enrichment")
            break
            
        enrichment = enrich_from_search_result(result, "case_law", original_query)
        source_tag = result.get("source_tag", "UNKNOWN")
        if enrichment["pdf_saved"]:
            summary["pdfs_saved"] += 1
        summary["chunks_indexed"] += enrichment["chunks_added"]
        score = enrichment.get("_rerank_score", 0)
        
        if score > HIGH_QUALITY_SCORE:
            web_high_quality_count += 1
        
        # Include case laws if they have content and score >= 2.0 (full PDF scoring)
        if enrichment.get("content") and score >= WEB_MIN_SCORE:
            summary["enriched_case_laws"].append(enrichment)
            logger.debug(f"Added case law '{enrichment.get('title', 'Unknown')}' with score {score:.2f}")

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
