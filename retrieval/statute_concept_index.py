"""
Structured statute lookup: legal concept → act/section.

The statute concept index is built only from the vector store (BARE_CHUNKS_V2).
Every bare-act section in the store is indexed by its section_title and by 2–3 word
phrases from the title, so concepts like "criminal intimidation" or "grievous hurt"
resolve to the matching sections. No static index: whenever the vector store is
rebuilt or updated, the next lookup rebuilds the index from the updated chunks file.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-built index from vector store (concept_key -> list of refs). Invalidated when store is rebuilt.
_store_index: Optional[dict] = None
_store_index_mtime: Optional[float] = None

# Stopwords to skip when building phrase keys from section titles
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "for", "on", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "may", "might", "must", "shall", "should",
    "can", "could", "or", "and", "if", "as", "into", "upon", "any", "such",
})


def _normalize_concept(concept: str) -> str:
    """Lowercase, strip, collapse spaces for lookup."""
    if not concept or not isinstance(concept, str):
        return ""
    return " ".join(concept.strip().lower().split())


def _concept_keys_from_title(section_title: str) -> list[str]:
    """
    Generate concept lookup keys from a section title for store-backed index.
    Returns: full normalized title + 2-word and 3-word phrases (skip stopwords).
    """
    if not (section_title or "").strip():
        return []
    raw = " ".join((section_title or "").strip().lower().split())
    if not raw or len(raw) < 3:
        return []
    words = [w for w in raw.split() if len(w) >= 2 and w not in _TITLE_STOPWORDS]
    if not words:
        keys = [raw]
    else:
        keys = [raw]
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if len(phrase) >= 4 and phrase not in keys:
                    keys.append(phrase)
    return keys


def _build_index_from_bare_chunks(chunks_dict: dict) -> dict:
    """
    Build concept_key -> [refs] from all bare_act chunks in the store.
    Dedupes by (act_name, section_number). Each section gets keys from its section_title.
    """
    index = {}
    seen_sections = set()
    it = chunks_dict.items() if isinstance(chunks_dict, dict) else enumerate(chunks_dict)
    for _chunk_id, chunk in it:
        if not isinstance(chunk, dict) or chunk.get("doc_type") != "bare_act":
            continue
        act_name = (chunk.get("act_name") or "").strip()
        section_number = str(chunk.get("section_number") or "").strip()
        section_title = (chunk.get("section_title") or "").strip()
        if not act_name or not section_number:
            continue
        key_sec = (act_name.lower(), section_number.lower())
        if key_sec in seen_sections:
            continue
        seen_sections.add(key_sec)
        ref = {"act_name": act_name, "section_number": section_number, "section_title": section_title}
        for key in _concept_keys_from_title(section_title):
            if not key or len(key) < 2:
                continue
            index.setdefault(key, []).append(ref)
    return index


def _get_store_index() -> dict:
    """
    Lazy-load and return the concept index built from BARE_CHUNKS_V2.
    Built once per process and cached; returned from cache on every subsequent lookup
    until the chunks file mtime changes (e.g. after vector store rebuild).
    """
    global _store_index, _store_index_mtime
    try:
        from config import BARE_CHUNKS_V2
        path = BARE_CHUNKS_V2
    except Exception:
        return {}
    if not path or not os.path.isfile(path):
        return {}
    mtime = os.path.getmtime(path)
    # Cache hit: same file as last build — return cached index (no rebuild on every lookup)
    if _store_index is not None and _store_index_mtime == mtime:
        return _store_index
    try:
        from retrieval.hybrid_retriever import load_chunks
        chunks = load_chunks(path)
        _store_index = _build_index_from_bare_chunks(chunks) if chunks else {}
        _store_index_mtime = mtime
        logger.debug("Statute concept index from store: %d keys from bare acts", len(_store_index))
    except Exception as e:
        logger.debug("Could not build statute concept index from store: %s", e)
        _store_index = {}
        _store_index_mtime = None
    return _store_index


def get_full_concept_index() -> dict:
    """
    Return the statute concept index built from BARE_CHUNKS_V2.
    Rebuilt automatically when the chunks file changes (mtime). Use invalidate_store_index()
    after a vector store rebuild so the next lookup uses the updated store.
    """
    return _get_store_index()


def invalidate_store_index() -> None:
    """Clear the cached store-derived index so next lookup rebuilds from disk (e.g. after re-indexing)."""
    global _store_index, _store_index_mtime
    _store_index = None
    _store_index_mtime = None


def lookup_sections_for_concepts(concepts: list) -> list[dict]:
    """
    Return list of section refs (act_name, section_number, section_title) for the given legal concepts.
    Uses the concept index built from BARE_CHUNKS_V2. Deduplicated by (act_name, section_number).
    """
    if not concepts:
        return []
    index = get_full_concept_index()
    seen = set()
    result = []
    for c in concepts:
        key = _normalize_concept(c)
        if not key:
            continue
        refs = index.get(key)
        if refs:
            for r in refs:
                t = (r.get("act_name") or "", r.get("section_number") or "")
                if t not in seen:
                    seen.add(t)
                    result.append(dict(r))
        for index_key, refs in index.items():
            if index_key in key or key in index_key:
                for r in refs:
                    t = (r.get("act_name") or "", r.get("section_number") or "")
                    if t not in seen:
                        seen.add(t)
                        result.append(dict(r))
    return result


def get_act_names_for_concepts(concepts: list) -> frozenset:
    """Return set of act names that apply to the given concepts (for act-filtering in retrieval)."""
    refs = lookup_sections_for_concepts(concepts)
    return frozenset((r.get("act_name") or "").strip() for r in refs if (r.get("act_name") or "").strip())
