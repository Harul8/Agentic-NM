"""
Remove named Bare Act source files from the indexed chunk stores and rebuild indexes.

This script removes all chunks whose ``source_file`` matches a provided file name
or relative source path (for example ``1992_9.txt`` or
``BareActs/Union of India/1992_9.txt``), then rebuilds:

  - bareacts_v2.index
  - bareacts_v2_chunks.json
  - bareacts_bm25.json
  - act_summaries_v2.index
  - act_summaries_v2_chunks.json
  - act_summaries_bm25.json

Usage examples:

    python scripts/remove_bareact_files_from_index.py --dry-run --remove-file 1992_9.txt

    python scripts/remove_bareact_files_from_index.py ^
        --remove-list legal_database/Others/remove_ids.txt

The script creates backups of the six target vector-store files before replacing
them, unless ``--skip-backup`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ingestion.build_v2_index import _get_embedder, build_index
from config import (
    ACT_SUMMARY_BM25_INDEX,
    ACT_SUMMARY_CHUNKS_V2,
    ACT_SUMMARY_INDEX_V2,
    BARE_BM25_INDEX,
    BARE_CHUNKS_V2,
    BARE_INDEX_V2,
    VECTOR_STORE,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_ID_PREFIX_RE = re.compile(r"^(\d{4}_\d+)(?:_.+)?(\.[A-Za-z0-9]+)?$")


def _normalise_source_token(value: str) -> str:
    """Normalise a source path or filename for case-insensitive matching."""
    return value.replace("/", "\\").strip().lower()


def _extract_id_filename(value: str) -> str | None:
    """
    Extract an index-style source filename like ``1992_9.txt`` from a renamed file.
    """
    basename = os.path.basename(value.strip())
    match = _ID_PREFIX_RE.match(basename)
    if not match:
        return None
    stem = match.group(1)
    ext = match.group(2) or ".txt"
    return f"{stem}{ext}".lower()


def _source_variants(value: str) -> set[str]:
    """Return normalised full-path and basename variants for matching."""
    normalised = _normalise_source_token(value)
    variants = {normalised}
    basename = os.path.basename(normalised)
    if basename:
        variants.add(basename)
        id_filename = _extract_id_filename(basename)
        if id_filename:
            variants.add(id_filename)
    return variants


def load_removal_targets(remove_files: list[str], remove_list_path: str | None) -> set[str]:
    """Load and normalise all removal targets from args and optional text file."""
    targets: set[str] = set()

    for item in remove_files:
        targets.update(_source_variants(item))

    if remove_list_path:
        with open(remove_list_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                targets.update(_source_variants(line))

    return targets


def load_chunks_list(chunks_path: str) -> list[dict]:
    """
    Load chunk JSON and return a stable list ordered by numeric key where possible.
    """
    with open(chunks_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        def _sort_key(item: tuple[str, dict]):
            key = item[0]
            return (0, int(key)) if str(key).isdigit() else (1, str(key))

        return [chunk for _, chunk in sorted(raw.items(), key=_sort_key)]

    if isinstance(raw, list):
        return raw

    raise ValueError(f"Unsupported chunk store format in {chunks_path}")


def filter_chunks(chunks: list[dict], removal_targets: set[str]) -> tuple[list[dict], list[dict], Counter]:
    """
    Split chunks into kept and removed lists based on source_file matching.
    """
    kept: list[dict] = []
    removed: list[dict] = []
    removed_counts: Counter = Counter()

    for chunk in chunks:
        source_file = str(chunk.get("source_file") or chunk.get("source") or "").strip()
        variants = _source_variants(source_file)
        if variants & removal_targets:
            removed.append(chunk)
            removed_counts[source_file] += 1
        else:
            kept.append(chunk)

    return kept, removed, removed_counts


def filter_chunks_with_jurisdiction(
    chunks: list[dict],
    removal_targets: set[str],
    jurisdiction: str | None,
) -> tuple[list[dict], list[dict], Counter]:
    """
    Split chunks into kept and removed lists, optionally restricting matches to one jurisdiction.
    """
    kept: list[dict] = []
    removed: list[dict] = []
    removed_counts: Counter = Counter()
    jurisdiction_token = None
    if jurisdiction:
        jurisdiction_token = _normalise_source_token(f"BareActs/{jurisdiction}/")

    for chunk in chunks:
        source_file = str(chunk.get("source_file") or chunk.get("source") or "").strip()
        source_norm = _normalise_source_token(source_file)
        if jurisdiction_token and jurisdiction_token not in source_norm:
            kept.append(chunk)
            continue
        variants = _source_variants(source_file)
        if variants & removal_targets:
            removed.append(chunk)
            removed_counts[source_file] += 1
        else:
            kept.append(chunk)

    return kept, removed, removed_counts


def backup_targets(paths: list[str], backup_root: str) -> None:
    """Copy target files into a timestamped backup folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, f"remove_bareacts_backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    for path in paths:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))

    logger.info("Backed up vector-store files to %s", backup_dir)


def rebuild_store(
    *,
    kept_chunks: list[dict],
    faiss_path: str,
    chunks_path: str,
    bm25_path: str,
    embedder,
) -> None:
    """Rebuild one FAISS + chunks + BM25 triplet from the kept chunks."""
    if not kept_chunks:
        raise ValueError(f"Refusing to rebuild empty index for {chunks_path}")

    build_index(
        kept_chunks,
        faiss_path=faiss_path,
        chunks_path=chunks_path,
        bm25_path=bm25_path,
        embedder=embedder,
        append=False,
        resume_embeddings=False,
    )


def replace_with_rebuilt_outputs(temp_dir: str) -> None:
    """Replace production vector-store files with rebuilt temporary outputs."""
    targets = [
        (os.path.join(temp_dir, os.path.basename(BARE_INDEX_V2)), BARE_INDEX_V2),
        (os.path.join(temp_dir, os.path.basename(BARE_CHUNKS_V2)), BARE_CHUNKS_V2),
        (os.path.join(temp_dir, os.path.basename(BARE_BM25_INDEX)), BARE_BM25_INDEX),
        (os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_INDEX_V2)), ACT_SUMMARY_INDEX_V2),
        (os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_CHUNKS_V2)), ACT_SUMMARY_CHUNKS_V2),
        (os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_BM25_INDEX)), ACT_SUMMARY_BM25_INDEX),
    ]

    for src, dst in targets:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected rebuilt file not found: {src}")
        shutil.copy2(src, dst)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove named Bare Act files from chunk stores and rebuild indexes."
    )
    parser.add_argument(
        "--remove-file",
        action="append",
        default=[],
        help="Bare Act source filename or relative source path to remove. Can be repeated.",
    )
    parser.add_argument(
        "--remove-list",
        help="Path to a UTF-8 text file listing one source filename/path per line.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without rebuilding the indexes.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip copying backup files before replacing the vector store.",
    )
    parser.add_argument(
        "--backup-dir",
        default=VECTOR_STORE,
        help="Directory where timestamped backups should be stored.",
    )
    parser.add_argument(
        "--jurisdiction",
        default=None,
        help="Optional BareActs jurisdiction folder name to restrict removals, for example 'Union of India'.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    removal_targets = load_removal_targets(args.remove_file, args.remove_list)
    if not removal_targets:
        parser.error("Provide at least one --remove-file or --remove-list.")

    bare_chunks = load_chunks_list(BARE_CHUNKS_V2)
    act_summary_chunks = load_chunks_list(ACT_SUMMARY_CHUNKS_V2)

    kept_bare, removed_bare, removed_bare_counts = filter_chunks_with_jurisdiction(
        bare_chunks,
        removal_targets,
        args.jurisdiction,
    )
    kept_summary, removed_summary, removed_summary_counts = filter_chunks_with_jurisdiction(
        act_summary_chunks,
        removal_targets,
        args.jurisdiction,
    )

    removed_sources = sorted(set(removed_bare_counts) | set(removed_summary_counts))
    logger.info("Removal targets supplied: %d", len(removal_targets))
    logger.info("Matched source files in indexes: %d", len(removed_sources))
    logger.info("Bare Act chunks: kept=%d removed=%d", len(kept_bare), len(removed_bare))
    logger.info("Act summary chunks: kept=%d removed=%d", len(kept_summary), len(removed_summary))

    for source in removed_sources[:50]:
        logger.info(
            "Matched: %s (bare=%d, summary=%d)",
            source,
            removed_bare_counts.get(source, 0),
            removed_summary_counts.get(source, 0),
        )

    if not removed_sources:
        logger.warning("No indexed chunks matched the provided file names.")
        return 0

    if args.dry_run:
        logger.info("Dry run only. No vector-store files were changed.")
        return 0

    if not args.skip_backup:
        backup_targets(
            [
                BARE_INDEX_V2,
                BARE_CHUNKS_V2,
                BARE_BM25_INDEX,
                ACT_SUMMARY_INDEX_V2,
                ACT_SUMMARY_CHUNKS_V2,
                ACT_SUMMARY_BM25_INDEX,
            ],
            args.backup_dir,
        )

    embedder = _get_embedder()

    with TemporaryDirectory(prefix="bareact_index_rebuild_", dir=VECTOR_STORE) as temp_dir:
        temp_bare_index = os.path.join(temp_dir, os.path.basename(BARE_INDEX_V2))
        temp_bare_chunks = os.path.join(temp_dir, os.path.basename(BARE_CHUNKS_V2))
        temp_bare_bm25 = os.path.join(temp_dir, os.path.basename(BARE_BM25_INDEX))
        temp_summary_index = os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_INDEX_V2))
        temp_summary_chunks = os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_CHUNKS_V2))
        temp_summary_bm25 = os.path.join(temp_dir, os.path.basename(ACT_SUMMARY_BM25_INDEX))

        logger.info("Rebuilding Bare Acts index from filtered chunks...")
        rebuild_store(
            kept_chunks=kept_bare,
            faiss_path=temp_bare_index,
            chunks_path=temp_bare_chunks,
            bm25_path=temp_bare_bm25,
            embedder=embedder,
        )

        logger.info("Rebuilding act summaries index from filtered chunks...")
        rebuild_store(
            kept_chunks=kept_summary,
            faiss_path=temp_summary_index,
            chunks_path=temp_summary_chunks,
            bm25_path=temp_summary_bm25,
            embedder=embedder,
        )

        replace_with_rebuilt_outputs(temp_dir)

    logger.info("Vector-store replacement complete.")
    logger.info("Restart the app so preloaded indexes are reloaded into memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
