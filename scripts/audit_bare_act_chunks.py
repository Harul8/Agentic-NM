"""
Audit Bare Act Chunks — Find and optionally remove chunks that are actually court judgments
misclassified as bare acts (e.g. judgment PDFs in BareActs folder or mis-tagged during indexing).

Usage:
  python scripts/audit_bare_act_chunks.py              # Report only
  python scripts/audit_bare_act_chunks.py --remove     # Remove bad chunks and rebuild bare act index

After --remove, the bare act FAISS and BM25 indexes are rebuilt from the cleaned chunk list.
Move any misclassified PDFs from data/BareActs to data/CaseLaws to prevent re-adding on next full build.
"""

import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BARE_CHUNKS_V2, BARE_INDEX_V2, BARE_BM25_INDEX, VECTOR_STORE
from Ingestion.smart_chunker import _looks_like_judgment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_bare_chunks_list():
    """Load bare act chunks from JSON (keys '0','1',...) into a list preserving order."""
    if not os.path.isfile(BARE_CHUNKS_V2):
        logger.warning("Bare chunks file not found: %s", BARE_CHUNKS_V2)
        return []
    with open(BARE_CHUNKS_V2, "r", encoding="utf-8") as f:
        store = json.load(f)
    # Preserve order by numeric key
    keys = sorted(store.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    return [store[k] for k in keys]


def audit(chunks):
    """Return (bad_indices, bad_sources) for chunks that look like judgments."""
    bad_indices = []
    seen_source = set()
    for i, ch in enumerate(chunks):
        text = (ch.get("full_text") or ch.get("search_text") or ch.get("text") or "").strip()
        src = ch.get("source_file") or ch.get("act_name") or ""
        if not text:
            continue
        if _looks_like_judgment(text, src):
            bad_indices.append(i)
            seen_source.add(src)
    return bad_indices, list(seen_source)


def main():
    parser = argparse.ArgumentParser(description="Audit bare act chunks for misclassified judgments")
    parser.add_argument("--remove", action="store_true", help="Remove bad chunks and rebuild bare act index")
    args = parser.parse_args()

    chunks = load_bare_chunks_list()
    bad_indices, bad_sources = audit(chunks)

    if not bad_indices:
        logger.info("No judgment-like chunks found in bare act store. OK.")
        return

    logger.warning(
        "Found %d chunk(s) that look like court judgments (not bare acts) in %s",
        len(bad_indices),
        BARE_CHUNKS_V2,
    )
    for i in bad_indices:
        ch = chunks[i]
        logger.warning(
            "  - index %s | source_file=%s | act_name=%s",
            i,
            ch.get("source_file", ""),
            (ch.get("act_name") or "")[:60],
        )
    logger.warning("Affected source files: %s", bad_sources)

    if not args.remove:
        logger.info("Run with --remove to drop these chunks and rebuild the bare act index.")
        logger.info("Then move the listed PDFs from data/BareActs to data/CaseLaws and re-run build_v2_index if needed.")
        return

    # Remove bad chunks and rebuild index
    good_chunks = [c for i, c in enumerate(chunks) if i not in bad_indices]
    logger.info("Keeping %d chunks, removing %d.", len(good_chunks), len(bad_indices))

    if not good_chunks:
        logger.warning("No chunks left after removal. Delete index/chunk files manually or re-run build_v2_index.")
        return

    from Ingestion.build_v2_index import build_index, _get_embedder
    embedder = _get_embedder()
    os.makedirs(VECTOR_STORE, exist_ok=True)
    build_index(good_chunks, BARE_INDEX_V2, BARE_CHUNKS_V2, BARE_BM25_INDEX, embedder)
    logger.info("Bare act index rebuilt. Move misclassified PDFs from BareActs to CaseLaws to avoid re-adding.")


if __name__ == "__main__":
    main()
