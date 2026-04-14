"""
scripts/rebuild_case_summaries_fast.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate AI summaries for case_summaries_v2_chunks.json using async OpenAI.

Currently case summaries contain verbatim judgment text.  This script
generates a concise, retrieval-optimised paragraph per case and writes it
back into the chunks store, then rebuilds FAISS + BM25.

Touches ONLY:
    case_summaries_v2.index
    case_summaries_v2_chunks.json
    case_summaries_bm25.json

Does NOT touch:
    caselaws_v2.*   bareacts_v2.*   act_summaries_v2.*

Usage
─────
    cd "/mnt/c/Users/rahul/Agentic NM"
    source .venv/bin/activate
    python scripts/rebuild_case_summaries_fast.py [options]

Options
───────
    --dry-run          Generate summaries but skip index write.
    --limit N          Process only the first N cases (for testing).
    --resume           Skip cases whose summary is already in the progress file.
    --concurrency N    Max simultaneous OpenAI requests (default 30).
    --model MODEL      Override model name (default: gpt-4.1-nano).
    --max-output N     max_completion_tokens per call (default 400).
    --retries N        Max retries on 429 / 5xx (default 6).

Progress file
─────────────
    <VECTOR_STORE>/case_summary_progress.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    CASE_CHUNKS_V2,
    CASE_SUMMARY_BM25_INDEX,
    CASE_SUMMARY_CHUNKS_V2,
    CASE_SUMMARY_INDEX_V2,
    VECTOR_STORE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rebuild_case_summaries_fast")

# Silence noisy HTTP/OpenAI loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

PROGRESS_FILE = os.path.join(VECTOR_STORE, "case_summary_progress.json")

DEFAULT_MODEL = "gpt-4.1-nano"

# ── Prompts ───────────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = (
    "You are a senior Indian advocate summarizing case law for a legal retrieval system. "
    "Your summary will be used as a search index entry — it must be dense with legally "
    "meaningful terms so that queries about real disputes can match this case correctly."
)

SUMMARY_PROMPT_TEMPLATE = """Summarize the following Indian court case for a legal search index.

CASE: {case_name}
COURT: {court}
YEAR: {year}
CITATION: {citation}
CITED BY: {cited_by} cases

CASE CONTENT:
{case_content}

Write a single paragraph of 120-180 words covering:
1. The core legal dispute and key facts
2. The legal issues decided (acts, sections, constitutional provisions involved)
3. The court's holding and ratio decidendi
4. The practical outcome (relief granted, conviction upheld/quashed, etc.)
5. The fact patterns and disputes where this precedent would apply

Use plain legal English. Include legally significant terms a practitioner would use —
e.g. "wrongful termination", "writ of mandamus", "specific performance", "anticipatory bail",
"contempt of court", "res judicata", "natural justice", "fundamental rights", etc.
Do NOT include case citation or court name in the summary text.

Output ONLY the summary paragraph. Nothing else."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_case_content(full_text: str, max_chars: int = 3000) -> str:
    """
    Extract the substantive content from the structured full_text field,
    stripping the redundant metadata header (Case/Court/Year/Cited by lines).
    """
    if not full_text:
        return "(no content available)"

    # Strip the header block that starts with "Case: ..."
    lines = full_text.split("\n")
    content_lines = []
    header_done = False
    skip_prefixes = ("case:", "court:", "year:", "cited by:", "citation:")

    for line in lines:
        stripped = line.strip()
        if not header_done:
            if stripped.lower().startswith(skip_prefixes) or stripped == "":
                continue
            header_done = True
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    if not content:
        content = full_text.strip()

    return content[:max_chars]


# ── Async summary generation ──────────────────────────────────────────────────

async def _generate_summary_async(
    client,
    sem: asyncio.Semaphore,
    case_id: str,
    case_name: str,
    court: str,
    year: str,
    citation: str,
    cited_by: int,
    full_text: str,
    model: str,
    max_output: int,
    max_retries: int,
) -> tuple[str, str]:
    """Async worker: generate one case summary with retry + back-off."""
    content = _extract_case_content(full_text)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        case_name=case_name,
        court=court or "Indian Court",
        year=year or "unknown",
        citation=citation or "N/A",
        cited_by=cited_by or 0,
        case_content=content,
    )
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    for attempt in range(max_retries + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_output,
                )
            choice  = resp.choices[0]
            summary = (choice.message.content or "").strip()
            if not summary:
                logger.warning(
                    "Empty response for %r — finish_reason=%r usage=%s",
                    case_name, choice.finish_reason, resp.usage,
                )
            return case_id, summary

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate   = "429" in err_str or "rate limit" in err_str
            is_server = any(c in err_str for c in ("500", "502", "503", "529"))
            if attempt >= max_retries or not (is_rate or is_server):
                logger.warning("Failed for %r after %d attempts: %s", case_name, attempt + 1, exc)
                return case_id, ""

            delay = min(60, (2 ** attempt) + random.uniform(0, 1))
            logger.debug("Retry %d/%d for %r in %.1fs (err: %s)",
                         attempt + 1, max_retries, case_name, delay, exc)
            await asyncio.sleep(delay)

    return case_id, ""


async def _run_async(
    todo: list[tuple],
    progress: dict[str, str],
    model: str,
    concurrency: int,
    max_output: int,
    max_retries: int,
    dry_run: bool,
) -> None:
    from openai import AsyncOpenAI

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    total  = len(todo)
    done   = 0
    errors = 0
    t0     = time.perf_counter()
    FLUSH_EVERY = 500

    tasks = [
        asyncio.create_task(
            _generate_summary_async(
                client, sem,
                case_id, case_name, court, year, citation, cited_by, full_text,
                model, max_output, max_retries,
            )
        )
        for case_id, case_name, court, year, citation, cited_by, full_text in todo
    ]

    for coro in asyncio.as_completed(tasks):
        try:
            case_id, summary = await coro
            progress[case_id] = summary
            done += 1

            if done % FLUSH_EVERY == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate / 60 if rate > 0 else 0
                print(f"{done}/{total} done  |  {rate * 60:.0f} cases/min  |  ETA {eta:.1f} min", flush=True)
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)

        except Exception as exc:
            errors += 1
            logger.error("Unhandled error: %s", exc)

    await client.close()
    logger.info("Generation done — %d completed, %d errors in %.1fs",
                done, errors, time.perf_counter() - t0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate AI summaries for case_summaries_v2 index (fast async)"
    )
    parser.add_argument("--dry-run",     action="store_true", help="Generate but skip index write")
    parser.add_argument("--limit",       type=int, default=0,   help="Process only first N cases")
    parser.add_argument("--resume",      action="store_true",   help="Skip cases in progress file")
    parser.add_argument("--concurrency", type=int, default=20,  help="Max concurrent OpenAI requests")
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-output",  type=int, default=2000, help="max_completion_tokens per call")
    parser.add_argument("--retries",     type=int, default=6,   help="Max retries on 429/5xx")
    args = parser.parse_args()

    logger.info("Model          : %s", args.model)
    logger.info("Concurrency    : %d async workers", args.concurrency)
    logger.info("Max output tok : %d", args.max_output)

    # ── Load case summary chunks (output target) ──────────────────────────────
    logger.info("Loading case summary chunks from %s", CASE_SUMMARY_CHUNKS_V2)
    with open(CASE_SUMMARY_CHUNKS_V2, encoding="utf-8") as f:
        chunks: dict = json.load(f)
    logger.info("Loaded %d case summary chunks", len(chunks))

    # ── Progress cache ────────────────────────────────────────────────────────
    progress: dict[str, str] = {}
    if args.resume and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                progress = json.load(f)
        except Exception as e:
            logger.warning("Progress file unreadable (%s) — starting fresh", e)
        logger.info("Resuming: %d cases already done", len(progress))

    # ── Build work list ───────────────────────────────────────────────────────
    # Primary source: caselaws_v2_chunks.json — contains ALL cases including
    # those excluded from a prior rebuild due to empty summaries.
    # This ensures retries always have access to the full 34k case set.
    seen_ids: set[str] = set()
    all_cases: list[tuple] = []
    meta_by_case: dict[str, dict] = {}  # case_id → best metadata chunk

    logger.info("Loading source case chunks from %s", CASE_CHUNKS_V2)
    with open(CASE_CHUNKS_V2, encoding="utf-8") as f:
        source_chunks: dict = json.load(f)
    logger.info("Loaded %d source chunks", len(source_chunks))

    for v in source_chunks.values():
        case_id = (v.get("case_id") or "").strip()
        if not case_id:
            continue
        if case_id not in meta_by_case:
            meta_by_case[case_id] = v
        else:
            # Prefer chunk with more content
            existing = meta_by_case[case_id]
            if len(v.get("full_text") or "") > len(existing.get("full_text") or ""):
                meta_by_case[case_id] = v

    for case_id, v in meta_by_case.items():
        case_name = (v.get("case_name") or "").strip()
        if not case_name or case_id in seen_ids:
            continue
        seen_ids.add(case_id)
        all_cases.append((
            case_id,
            case_name,
            (v.get("court") or "").strip(),
            str(v.get("year") or ""),
            (v.get("citation") or "").strip(),
            int(v.get("cited_by_count") or 0),
            (v.get("full_text") or "").strip(),
        ))

    logger.info("Total unique cases found: %d", len(all_cases))

    if args.limit:
        all_cases = all_cases[:args.limit]

    todo = [
        c for c in all_cases
        if not (args.resume and c[0] in progress and progress[c[0]])
    ]

    logger.info("%d cases to summarise (concurrency=%d)", len(todo), args.concurrency)

    if not todo:
        logger.info("Nothing to do — all cases already in progress file.")
    else:
        asyncio.run(_run_async(
            todo, progress, args.model,
            args.concurrency, args.max_output, args.retries,
            args.dry_run,
        ))

    if args.dry_run:
        logger.info("--dry-run: skipping index write. Sample summaries:")
        sample = list(progress.items())[:3]
        for case_id, summary in sample:
            logger.info("  [%s]\n    %s\n", case_id, summary[:200])
        return

    # ── Build fresh chunks with LLM content only ─────────────────────────────
    # One chunk per case_id. Strip verbatim full_text — only AI summary is kept.
    # Cases with no generated summary are excluded entirely.
    logger.info("Building fresh chunks from %d source entries...", len(chunks))

    enriched = []
    merged = 0
    skipped = 0

    for case_id, base in meta_by_case.items():
        summary = progress.get(case_id, "").strip()
        if not summary:
            skipped += 1
            continue
        fresh = {
            "chunk_id":      base.get("chunk_id", case_id + "_SUMMARY"),
            "case_id":       case_id,
            "case_name":     base.get("case_name", ""),
            "court":         base.get("court", ""),
            "year":          base.get("year", ""),
            "citation":      base.get("citation", ""),
            "cites_count":   base.get("cites_count", 0),
            "cited_by_count": base.get("cited_by_count", 0),
            "source_file":   base.get("source_file", ""),
            "doc_type":      "case_summary",
            "text":          summary,
            "search_text":   summary,
        }
        enriched.append(fresh)
        merged += 1

    logger.info("Fresh chunks built: %d  |  Excluded (no summary): %d", merged, skipped)

    # ── Rebuild index ─────────────────────────────────────────────────────────
    logger.info("Rebuilding case summary index (FAISS + BM25)...")
    from retrieval.indexer import _get_embedder, build_index
    embedder = _get_embedder()
    build_index(
        enriched,
        CASE_SUMMARY_INDEX_V2,
        CASE_SUMMARY_CHUNKS_V2,
        CASE_SUMMARY_BM25_INDEX,
        embedder,
    )

    logger.info("✓ Case summary index rebuilt.")
    logger.info("  Index : %s", CASE_SUMMARY_INDEX_V2)
    logger.info("  Chunks: %s", CASE_SUMMARY_CHUNKS_V2)
    logger.info("  BM25  : %s", CASE_SUMMARY_BM25_INDEX)


if __name__ == "__main__":
    main()
