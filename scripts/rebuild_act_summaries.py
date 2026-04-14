"""
scripts/rebuild_act_summaries.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regenerate the act-summary index with LLM-generated descriptive summaries.

Touches ONLY:
    act_summaries_v2.index
    act_summaries_v2_chunks.json
    act_summaries_bm25.json

Does NOT touch:
    bare_acts_v2.*        (section-level index)
    case_laws_v2.*        (case-law index)
    Any other vector store file

Usage
-----
    cd "/mnt/c/Users/rahul/Agentic NM"
    source .venv/bin/activate
    python scripts/rebuild_act_summaries.py [--dry-run] [--limit N] [--resume]

Flags
-----
    --dry-run     Generate summaries but do NOT write index files.
    --limit N     Process only the first N acts (for testing).
    --resume      Skip acts whose summary is already in the progress cache.
    --workers N   Parallel LLM workers (default 4). Each worker makes one
                  ask_llm call; stay within your OpenAI rate limits.

Progress
--------
The script writes a progress file:
    <VECTOR_STORE>/act_summary_progress.json

Each completed act is saved there immediately. If the run is interrupted,
re-run with --resume to pick up from where it left off. The final index
rebuild only runs after all acts are processed (or --dry-run is set).
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ACT_SUMMARY_BM25_INDEX,
    ACT_SUMMARY_CHUNKS_V2,
    ACT_SUMMARY_INDEX_V2,
    BARE_CHUNKS_V2,
    VECTOR_STORE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rebuild_act_summaries")

PROGRESS_FILE = os.path.join(VECTOR_STORE, "act_summary_progress.json")

# ── Prompt ────────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM = (
    "You are a senior Indian advocate summarizing legislation for a legal retrieval system. "
    "Your summary will be used as a search index entry — it must be dense with legally "
    "meaningful terms so that queries about real disputes can match this Act correctly."
)

SUMMARY_PROMPT_TEMPLATE = """Summarize the following Indian statute for a legal search index.

ACT: {act_name} ({year})
TOTAL SECTIONS: {total_sections}

SECTION EXCERPTS (representative sample):
{section_excerpts}

Write a single paragraph of 120-180 words covering:
1. What this Act governs and its purpose
2. Who it protects or regulates (parties, relationships, situations)
3. Key offences, duties, or rights it creates
4. Typical remedies, penalties, or relief it provides
5. The fact patterns and disputes where this Act would apply

Use plain legal English. Include legally significant terms that a practitioner or litigant
would use when describing a dispute — e.g. "domestic violence", "protection order",
"dowry harassment", "cruelty", "maintenance", "eviction", "wrongful termination", etc.
Do NOT include section numbers or Act citation in the summary text.

Output ONLY the summary paragraph. Nothing else."""


def _build_section_excerpts(sections: list[dict], max_chars: int = 2000) -> str:
    """Pick a representative spread of sections and truncate to max_chars."""
    if not sections:
        return "(no sections available)"

    # Sort by section number (numeric where possible)
    def _sec_key(s):
        try:
            return int(str(s.get("section_number") or "0").split(".")[0])
        except Exception:
            return 0

    sorted_secs = sorted(sections, key=_sec_key)

    # Take first 3, middle 3, last 3 to give spread across the Act
    n = len(sorted_secs)
    indices = list(dict.fromkeys(
        [0, 1, 2, n // 4, n // 2, 3 * n // 4, n - 3, n - 2, n - 1]
    ))
    chosen = [sorted_secs[i] for i in sorted(set(indices)) if 0 <= i < n]

    parts = []
    total = 0
    for sec in chosen:
        title = sec.get("section_title") or ""
        text  = (sec.get("full_text") or sec.get("text") or "").strip()
        if not text:
            continue
        snippet = f"§{sec.get('section_number', '?')} {title}: {text[:300]}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)

    return "\n\n".join(parts) if parts else "(no text available)"


def generate_summary(act_name: str, year: str, total_sections: int,
                     sections: list[dict]) -> str:
    """Call ask_llm to generate a descriptive summary for one Act."""
    from platform_pkg.llm import ask_llm

    excerpts = _build_section_excerpts(sections)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        act_name=act_name,
        year=year or "unknown",
        total_sections=total_sections,
        section_excerpts=excerpts,
    )
    try:
        result = ask_llm(prompt, task_hint="fast", system=SUMMARY_SYSTEM)
        return (result or "").strip()
    except Exception as exc:
        logger.warning("LLM failed for %r: %s", act_name, exc)
        # Fallback: use first 300 chars of first non-boilerplate section text
        for sec in sections:
            t = (sec.get("full_text") or "").strip()
            if len(t) > 80 and "short title" not in t[:60].lower():
                return t[:300]
        return excerpts[:300]


def _worker(task: tuple) -> tuple[str, str]:
    """Worker function for thread pool: returns (act_id, summary)."""
    act_id, act_name, year, total_sections, sections = task
    summary = generate_summary(act_name, year, total_sections, sections)
    return act_id, summary


def main():
    parser = argparse.ArgumentParser(description="Rebuild act summary index with LLM summaries")
    parser.add_argument("--dry-run",  action="store_true", help="Generate summaries but skip index write")
    parser.add_argument("--limit",    type=int, default=0,  help="Process only first N acts")
    parser.add_argument("--resume",   action="store_true", help="Skip acts already in progress file")
    parser.add_argument("--workers",  type=int, default=4,  help="Parallel LLM workers")
    args = parser.parse_args()

    logger.info("Loading act summary chunks from %s", ACT_SUMMARY_CHUNKS_V2)
    act_summary_chunks: dict = json.load(open(ACT_SUMMARY_CHUNKS_V2, encoding="utf-8"))

    logger.info("Loading bare act section chunks from %s", BARE_CHUNKS_V2)
    bare_chunks: dict = json.load(open(BARE_CHUNKS_V2, encoding="utf-8"))

    # Group bare sections by act_id
    from collections import defaultdict
    sections_by_act: dict[str, list] = defaultdict(list)
    for chunk in bare_chunks.values():
        aid = (chunk.get("act_id") or "").strip()
        if aid:
            sections_by_act[aid].append(chunk)

    # Load progress cache
    progress: dict[str, str] = {}
    if args.resume and os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE, encoding="utf-8"))
        logger.info("Resuming: %d acts already done", len(progress))

    # Build work list
    all_acts = list(act_summary_chunks.values())
    if args.limit:
        all_acts = all_acts[:args.limit]

    todo = []
    for chunk in all_acts:
        act_id   = (chunk.get("act_id") or "").strip()
        act_name = (chunk.get("act_name") or "").strip()
        if not act_id or not act_name:
            continue
        if args.resume and act_id in progress:
            continue
        year           = str(chunk.get("year") or "")
        total_sections = int(chunk.get("total_sections") or 0)
        sections       = sections_by_act.get(act_id, [])
        todo.append((act_id, act_name, year, total_sections, sections))

    logger.info("%d acts to process (%d workers)", len(todo), args.workers)

    # ── Generate summaries ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, task): task for task in todo}
        for future in as_completed(futures):
            try:
                act_id, summary = future.result()
                progress[act_id] = summary
                done += 1
                if done % 10 == 0 or done == len(todo):
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (len(todo) - done) / rate if rate > 0 else 0
                    logger.info(
                        "[%d/%d] %.1f acts/min | ETA %.0f min",
                        done, len(todo), rate * 60, remaining / 60,
                    )
                    # Flush progress to disk every 10 acts
                    if not args.dry_run:
                        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                            json.dump(progress, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                errors += 1
                task = futures[future]
                logger.error("Worker error for %r: %s", task[1], exc)

    logger.info("Summary generation done: %d completed, %d errors", done, errors)

    if args.dry_run:
        logger.info("--dry-run: skipping index write. Sample summaries:")
        for act_id, summary in list(progress.items())[:3]:
            # Find act_name
            for chunk in act_summary_chunks.values():
                if chunk.get("act_id") == act_id:
                    logger.info("  [%s]\n    %s\n", chunk.get("act_name"), summary[:200])
                    break
        return

    # ── Merge summaries back into chunks ──────────────────────────────────────
    logger.info("Merging summaries into chunk store...")
    enriched_chunks = []
    skipped = 0
    for chunk in act_summary_chunks.values():
        act_id = (chunk.get("act_id") or "").strip()
        summary = progress.get(act_id, "")
        if not summary:
            skipped += 1
            summary = (chunk.get("full_text") or "").strip()[:400]  # fallback

        new_chunk = dict(chunk)
        new_chunk["text"]        = summary          # used by BM25 at query time
        new_chunk["search_text"] = summary          # used for embedding
        new_chunk["full_text"]   = summary          # used for display
        enriched_chunks.append(new_chunk)

    logger.info(
        "Enriched %d act summary chunks (%d used LLM, %d used fallback text)",
        len(enriched_chunks), len(enriched_chunks) - skipped, skipped,
    )

    # ── Rebuild ONLY the act summary index ───────────────────────────────────
    logger.info("Rebuilding act summary index (leaving all other indexes untouched)...")
    from retrieval.indexer import build_index, _get_embedder

    embedder = _get_embedder()
    build_index(
        enriched_chunks,
        ACT_SUMMARY_INDEX_V2,
        ACT_SUMMARY_CHUNKS_V2,
        ACT_SUMMARY_BM25_INDEX,
        embedder,
    )

    logger.info("✓ Act summary index rebuilt successfully.")
    logger.info("  Index : %s", ACT_SUMMARY_INDEX_V2)
    logger.info("  Chunks: %s", ACT_SUMMARY_CHUNKS_V2)
    logger.info("  BM25  : %s", ACT_SUMMARY_BM25_INDEX)
    logger.info("  Bare acts / case law indexes: UNTOUCHED")


if __name__ == "__main__":
    main()
