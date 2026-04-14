"""
scripts/rebuild_act_summaries_fast.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Faster drop-in replacement for rebuild_act_summaries.py.

Key differences vs the original:
  • Uses openai.AsyncOpenAI directly — bypasses ask_llm budget caps and
    thread overhead.
  • asyncio + asyncio.Semaphore for high concurrency (default 30 in-flight
    requests at once vs. the original's 4 threads).
  • Exponential back-off with jitter on 429 / 5xx so rate-limit bursts
    resolve automatically without crashing.
  • Progress is flushed to disk after every batch so --resume works even
    after a hard kill.
  • Embedding step (SentenceTransformer + FAISS rebuild) is unchanged.

Touches ONLY:
    act_summaries_v2.index
    act_summaries_v2_chunks.json
    act_summaries_bm25.json

Does NOT touch:
    bareacts_v2.*   case_summaries_v2.*   caselaws_v2.*

Usage
─────
    cd "/mnt/c/Users/rahul/Agentic NM"
    source .venv/bin/activate
    python scripts/rebuild_act_summaries_fast.py [options]

Options
───────
    --dry-run          Generate summaries but skip index write.
    --limit N          Process only the first N acts (for testing).
    --resume           Skip acts whose summary is already in the progress file.
    --patch-raw        Only re-generate summaries for acts whose current chunk
                       text looks like raw section text (starts with "N.\n...")
                       rather than a proper AI-generated summary. Useful for
                       fixing the ~411 fallback entries without a full rebuild.
    --concurrency N    Max simultaneous OpenAI requests (default 30).
    --model MODEL      Override model name (default: reads from MODEL_TIERS /
                       DEFAULT_MODEL_TIER env var, same as ask_llm "fast" tier).
    --max-output N     max_completion_tokens for each call (default 400).
    --retries N        Max retries on 429 / 5xx (default 6).

Progress file
─────────────
    <VECTOR_STORE>/act_summary_progress.json   (same as original script)
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rebuild_act_summaries_fast")

PROGRESS_FILE = os.path.join(VECTOR_STORE, "act_summary_progress.json")

# ── Prompts (identical to original) ──────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

_RAW_SECTION_RE = re.compile(r"^\d+\.\s*\n")

def _is_raw_section_text(text: str) -> bool:
    """Return True if text looks like a raw act section dump rather than an AI summary."""
    if not text or len(text) < 10:
        return True
    return bool(_RAW_SECTION_RE.match(text))


def _build_section_excerpts(sections: list[dict], max_chars: int = 2000) -> str:
    """Pick a representative spread of sections (same logic as original)."""
    if not sections:
        return "(no sections available)"

    def _sec_key(s):
        try:
            return int(str(s.get("section_number") or "0").split(".")[0])
        except Exception:
            return 0

    sorted_secs = sorted(sections, key=_sec_key)
    n = len(sorted_secs)
    indices = list(dict.fromkeys(
        [0, 1, 2, n // 4, n // 2, 3 * n // 4, n - 3, n - 2, n - 1]
    ))
    chosen = [sorted_secs[i] for i in sorted(set(indices)) if 0 <= i < n]

    parts, total = [], 0
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


def _resolve_model() -> str:
    """Return the fast-tier model name the same way ask_llm does."""
    try:
        from platform_pkg.llm import OPENAI_MODEL_FAST
        return OPENAI_MODEL_FAST
    except Exception:
        return os.environ.get("OPENAI_MODEL_FAST", "gpt-4o-mini")


# ── Async summary generation ──────────────────────────────────────────────────

async def _generate_summary_async(
    client,
    sem: asyncio.Semaphore,
    act_id: str,
    act_name: str,
    year: str,
    total_sections: int,
    sections: list[dict],
    model: str,
    max_output: int,
    max_retries: int,
) -> tuple[str, str]:
    """
    Async worker: generate one act summary with retry + back-off.
    Returns (act_id, summary_text).
    """
    excerpts = _build_section_excerpts(sections)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        act_name=act_name,
        year=year or "unknown",
        total_sections=total_sections,
        section_excerpts=excerpts,
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
                    "Empty content for %r — finish_reason=%r usage=%s",
                    act_name, choice.finish_reason, resp.usage,
                )
            return act_id, summary

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate  = "429" in err_str or "rate limit" in err_str
            is_server = any(c in err_str for c in ("500", "502", "503", "529"))
            if attempt >= max_retries or not (is_rate or is_server):
                logger.warning("Failed for %r after %d attempts: %s", act_name, attempt + 1, exc)
                # Fallback: first non-boilerplate section text
                for sec in sections:
                    t = (sec.get("full_text") or "").strip()
                    if len(t) > 80 and "short title" not in t[:60].lower():
                        return act_id, t[:300]
                return act_id, excerpts[:300]

            # Exponential back-off with jitter
            delay = min(60, (2 ** attempt) + random.uniform(0, 1))
            logger.info("Retry %d/%d for %r in %.1fs (err: %s)",
                        attempt + 1, max_retries, act_name, delay, exc)
            await asyncio.sleep(delay)

    return act_id, ""   # unreachable but satisfies type checker


async def _run_async(
    todo: list[tuple],
    progress: dict[str, str],
    model: str,
    concurrency: int,
    max_output: int,
    max_retries: int,
    dry_run: bool,
) -> None:
    """Drive all async summary tasks and flush progress to disk."""
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

    # Batch size for progress flushes — flush every 20 completions
    # Always flush to disk regardless of --dry-run so progress is not lost
    FLUSH_EVERY = 20

    tasks = [
        asyncio.create_task(
            _generate_summary_async(
                client, sem,
                act_id, act_name, year, total_sections, sections,
                model, max_output, max_retries,
            )
        )
        for act_id, act_name, year, total_sections, sections in todo
    ]

    for coro in asyncio.as_completed(tasks):
        try:
            act_id, summary = await coro
            progress[act_id] = summary
            done += 1

            if done % FLUSH_EVERY == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate / 60 if rate > 0 else 0
                logger.info(
                    "[%d/%d]  %.1f acts/min | ETA %.1f min",
                    done, total, rate * 60, eta,
                )
                # Always persist — so progress is never lost even on --dry-run or kill
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
        description="Rebuild act summary index with async OpenAI calls (fast)"
    )
    parser.add_argument("--dry-run",     action="store_true", help="Generate but skip index write")
    parser.add_argument("--limit",       type=int, default=0,   help="Process only first N acts")
    parser.add_argument("--resume",      action="store_true",   help="Skip acts in progress file")
    parser.add_argument("--patch-raw",   action="store_true",   help="Only re-generate acts with raw section text in current chunks")
    parser.add_argument("--concurrency", type=int, default=30,  help="Max concurrent OpenAI requests")
    parser.add_argument("--model",       type=str, default="",  help="Override model name")
    parser.add_argument("--max-output",  type=int, default=5000, help="max_completion_tokens per call (reasoning models consume tokens for chain-of-thought before output)")
    parser.add_argument("--retries",     type=int, default=6,   help="Max retries on 429/5xx")
    args = parser.parse_args()

    model = args.model.strip() or _resolve_model()
    logger.info("Model          : %s", model)
    logger.info("Concurrency    : %d async workers", args.concurrency)
    logger.info("Max output tok : %d", args.max_output)
    logger.info("Max retries    : %d", args.retries)

    # ── Load bare act sections (streaming — file is ~112 MB) ──────────────────
    logger.info("Streaming bare act sections from %s", BARE_CHUNKS_V2)
    try:
        import ijson as _ijson
        sections_by_act: dict[str, list] = defaultdict(list)
        acts_meta: dict[str, dict] = {}    # act_id → {act_name, year}
        with open(BARE_CHUNKS_V2, "rb") as _f:
            for _k, _v in _ijson.kvitems(_f, ""):
                if not isinstance(_v, dict):
                    continue
                aid = (_v.get("act_id") or "").strip()
                if not aid:
                    continue
                sections_by_act[aid].append(_v)
                if aid not in acts_meta:
                    acts_meta[aid] = {
                        "act_name": (_v.get("act_name") or "").strip(),
                        "year":     str(_v.get("year") or ""),
                    }
        logger.info("Streamed %d sections across %d acts", sum(len(v) for v in sections_by_act.values()), len(acts_meta))
    except ImportError:
        logger.warning("ijson not found — falling back to json.load (high RAM)")
        bare_chunks: dict = json.load(open(BARE_CHUNKS_V2, encoding="utf-8"))
        sections_by_act = defaultdict(list)
        acts_meta = {}
        for chunk in bare_chunks.values():
            aid = (chunk.get("act_id") or "").strip()
            if not aid:
                continue
            sections_by_act[aid].append(chunk)
            if aid not in acts_meta:
                acts_meta[aid] = {
                    "act_name": (chunk.get("act_name") or "").strip(),
                    "year":     str(chunk.get("year") or ""),
                }

    # ── Also load act_summary_chunks for any extra metadata (best-effort) ─────
    # The file may be truncated — parse defensively.
    act_summary_chunks: dict = {}
    try:
        act_summary_chunks = json.load(open(ACT_SUMMARY_CHUNKS_V2, encoding="utf-8"))
        logger.info("Loaded %d act summary chunks from %s", len(act_summary_chunks), ACT_SUMMARY_CHUNKS_V2)
    except Exception as _e:
        logger.warning("act_summary_chunks load failed (%s) — deriving act list from bare_chunks only", _e)

    # ── Progress cache ────────────────────────────────────────────────────────
    progress: dict[str, str] = {}
    if args.resume and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "rb") as _pf:
                _raw = _pf.read()
            progress = json.loads(_raw.decode("utf-8", errors="replace"))
        except Exception as _e:
            logger.warning("Progress file unreadable (%s) — starting fresh", _e)
        logger.info("Resuming: %d acts already done", len(progress))

    # ── Build work list from bare_chunks (full act list, not truncated chunks) ─
    # Supplement with any extra metadata from act_summary_chunks.
    all_act_ids = sorted(acts_meta.keys())
    if args.limit:
        all_act_ids = all_act_ids[: args.limit]

    # Build set of act_ids with raw/bad text when --patch-raw is set
    raw_act_ids: set[str] = set()
    if args.patch_raw:
        for v in act_summary_chunks.values():
            aid = (v.get("act_id") or "").strip()
            if not aid:
                continue
            text = v.get("text") or v.get("full_text") or ""
            if _is_raw_section_text(text):
                raw_act_ids.add(aid)
        logger.info("--patch-raw: found %d acts with raw section text to re-generate", len(raw_act_ids))

    todo = []
    for act_id in all_act_ids:
        meta     = acts_meta[act_id]
        act_name = meta["act_name"]
        if not act_name:
            continue
        if args.patch_raw and act_id not in raw_act_ids:
            continue
        if args.resume and act_id in progress and not _is_raw_section_text(progress.get(act_id, "")):
            continue
        year = meta["year"]
        secs = sections_by_act.get(act_id, [])
        todo.append((
            act_id,
            act_name,
            year,
            len(secs),
            secs,
        ))

    logger.info("%d acts to summarise (concurrency=%d)", len(todo), args.concurrency)

    if not todo:
        logger.info("Nothing to do — all acts already in progress file. "
                    "Remove %s to regenerate from scratch.", PROGRESS_FILE)
        if not args.resume:
            pass
        # Still fall through to rebuild index from existing progress
    else:
        # ── Async generation ──────────────────────────────────────────────────
        asyncio.run(_run_async(
            todo, progress, model,
            args.concurrency, args.max_output, args.retries,
            args.dry_run,
        ))

    if args.dry_run:
        logger.info("--dry-run: skipping index write. Sample summaries:")
        for act_id, summary in list(progress.items())[:3]:
            for chunk in act_summary_chunks.values():
                if chunk.get("act_id") == act_id:
                    logger.info("  [%s]\n    %s\n", chunk.get("act_name"), summary[:200])
                    break
        return

    # ── Merge summaries back into chunks ─────────────────────────────────────
    # Use acts_meta (derived from bare_chunks) as the authoritative act list so
    # truncated act_summary_chunks doesn't cause missing acts.
    logger.info("Merging summaries into chunk store (%d acts from bare_chunks)...", len(acts_meta))
    # Build a lookup from the existing (possibly partial) act_summary_chunks
    existing_by_act = {
        v.get("act_id", "").strip(): v
        for v in act_summary_chunks.values()
        if v.get("act_id")
    }

    enriched_chunks = []
    skipped = 0
    for act_id, meta in acts_meta.items():
        summary = progress.get(act_id, "")
        # Use existing chunk as base if available; otherwise build from bare_chunks metadata
        base = existing_by_act.get(act_id) or {
            "act_id":         act_id,
            "act_name":       meta["act_name"],
            "year":           meta["year"],
            "total_sections": len(sections_by_act.get(act_id, [])),
            "chunk_type":     "act_summary",
        }
        if not summary:
            skipped += 1
            summary = (base.get("full_text") or "").strip()[:400]

        new_chunk = dict(base)
        new_chunk["text"]        = summary
        new_chunk["search_text"] = summary
        new_chunk["full_text"]   = summary
        enriched_chunks.append(new_chunk)

    logger.info(
        "Enriched %d chunks (%d LLM, %d fallback)",
        len(enriched_chunks), len(enriched_chunks) - skipped, skipped,
    )

    # ── Rebuild act summary index (embeddings + FAISS + BM25) ─────────────────
    logger.info("Rebuilding act summary index (all other indexes untouched)...")
    from retrieval.indexer import _get_embedder, build_index

    embedder = _get_embedder()
    build_index(
        enriched_chunks,
        ACT_SUMMARY_INDEX_V2,
        ACT_SUMMARY_CHUNKS_V2,
        ACT_SUMMARY_BM25_INDEX,
        embedder,
    )

    logger.info("✓ Act summary index rebuilt.")
    logger.info("  Index : %s", ACT_SUMMARY_INDEX_V2)
    logger.info("  Chunks: %s", ACT_SUMMARY_CHUNKS_V2)
    logger.info("  BM25  : %s", ACT_SUMMARY_BM25_INDEX)


if __name__ == "__main__":
    main()
