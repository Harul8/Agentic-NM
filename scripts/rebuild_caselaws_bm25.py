"""
rebuild_caselaws_bm25.py
------------------------
Rebuilds caselaws_bm25.json from the existing caselaws_v2_chunks.json
WITHOUT re-embedding. Safe to run after a WSL kill mid-indexing.

Run from the project root:
    python scripts/rebuild_caselaws_bm25.py

Requirements: ijson  (pip install ijson)
              Everything else is stdlib.
"""

import sys
import os
import json
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ijson
except ImportError:
    sys.exit("ERROR: ijson is required.  Install it with:  pip install ijson")

from config import CASE_CHUNKS_V2, CASE_BM25_INDEX

# ── Tokeniser (must match retriever.BM25._tokenize exactly) ──────────────────

def _tokenize(text: str) -> list:
    text = re.sub(r'(\d+)-([a-z])\b', r'\1\2', text.lower())
    return re.findall(r"[a-z0-9]+", text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chunks_path = CASE_CHUNKS_V2
    bm25_path   = CASE_BM25_INDEX
    bm25_tmp    = bm25_path + ".rebuilding"

    print(f"Chunks source : {chunks_path}")
    print(f"BM25 output   : {bm25_path}")
    print(f"File size     : {os.path.getsize(chunks_path) / 1024 / 1024:.0f} MB\n")

    doc_lengths = []
    term_freqs  = []
    doc_freqs   = {}
    doc_count   = 0
    t0 = time.perf_counter()

    print("Pass 1/1 — streaming chunks + fitting BM25 ...")
    with open(chunks_path, "rb") as f:
        for _k, v in ijson.kvitems(f, ""):
            text   = (v.get("full_text") or v.get("text") or "") if isinstance(v, dict) else ""
            tokens = _tokenize(text)
            tfreq  = {}
            for t in tokens:
                tfreq[t] = tfreq.get(t, 0) + 1
            doc_lengths.append(len(tokens))
            term_freqs.append(tfreq)
            for t in tfreq:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
            doc_count += 1
            if doc_count % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {doc_count:>10,} docs processed  ({elapsed:.0f}s)")

    avgdl   = sum(doc_lengths) / doc_count if doc_count else 1.0
    elapsed = time.perf_counter() - t0
    print(f"\nFitted BM25: {doc_count:,} docs, avgdl={avgdl:.1f}  [{elapsed:.1f}s]")

    print("\nSerialising to JSON (this may take a minute) ...")
    t1 = time.perf_counter()
    bm25_dict = {
        "doc_count":   doc_count,
        "avgdl":       avgdl,
        "doc_lengths": doc_lengths,
        "term_freqs":  term_freqs,
        "doc_freqs":   doc_freqs,
    }
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_tmp, "w", encoding="utf-8") as f:
        json.dump(bm25_dict, f)
    os.replace(bm25_tmp, bm25_path)        # atomic swap — safe on same filesystem

    size_mb = os.path.getsize(bm25_path) / 1024 / 1024
    print(f"Done → {bm25_path}  ({size_mb:.0f} MB)  [{time.perf_counter()-t1:.1f}s]")
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
