"""
Validate local bare acts index quality.

Run from the project root on Windows:
    python scripts/validate_index.py

Checks:
  1. Which acts are indexed and how many sections each has
  2. Average text length per act (short = stub-only indexing)
  3. Live cross-encoder scores for 5 key legal queries
     (shows whether the right sections are actually findable)
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ── locate the index ─────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import BARE_CHUNKS_V2, BARE_INDEX_V2, BARE_BM25_INDEX
except Exception as e:
    print(f"Could not import config: {e}")
    sys.exit(1)

chunks_path = Path(BARE_CHUNKS_V2)
if not chunks_path.exists():
    print(f"ERROR: Index not found at {chunks_path}")
    sys.exit(1)

print(f"Loading index from {chunks_path} …")
with open(chunks_path, encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\nTotal chunks in index: {len(chunks)}\n")

# ── 1. Act inventory ──────────────────────────────────────────────────────────
act_counts    = Counter()
act_text_lens = defaultdict(list)
act_short     = defaultdict(int)   # chunks with text < 100 chars

for k, v in chunks.items():
    act  = (v.get("act_name") or "Unknown").strip()
    text = (v.get("full_text") or v.get("search_text") or v.get("text") or "")
    act_counts[act]  += 1
    act_text_lens[act].append(len(text))
    if len(text) < 100:
        act_short[act] += 1

print(f"{'Act Name':<58} {'Secs':>5}  {'Avg chars':>9}  {'Short(<100)':>11}")
print("─" * 90)
for act, count in act_counts.most_common():
    avg  = sum(act_text_lens[act]) / len(act_text_lens[act])
    stub = act_short[act]
    flag = " ⚠ STUB" if avg < 150 else ""
    print(f"{act[:57]:<58} {count:>5}  {avg:>9.0f}  {stub:>11}{flag}")

# ── 2. Key acts presence check ────────────────────────────────────────────────
KEY_ACTS = [
    "Bharatiya Nyaya Sanhita",          # BNS — criminal
    "Bharatiya Nagarik Suraksha Sanhita",# BNSS — procedural
    "Bharatiya Sakshya Adhiniyam",       # BSA — evidence
    "Transfer of Property Act",
    "Specific Relief Act",
    "Code of Civil Procedure",
    "Indian Evidence Act",               # old name
    "Indian Penal Code",                 # old name — should NOT be indexed
]

print("\n\n── Key acts check ──────────────────────────────────────────────────────────")
indexed_acts_lower = {a.lower() for a in act_counts}
for key in KEY_ACTS:
    found = any(key.lower() in a for a in indexed_acts_lower)
    status = "✅ present" if found else "❌ MISSING"
    count  = sum(v for a, v in act_counts.items() if key.lower() in a.lower())
    print(f"  {status}  {key}  ({count} sections)")

# ── 3. Sample sections for BNS to check text quality ─────────────────────────
print("\n\n── Sample BNS sections (first 3, to check text quality) ────────────────────")
bns_sections = [
    v for v in chunks.values()
    if "bharatiya nyaya sanhita" in (v.get("act_name") or "").lower()
][:3]
if not bns_sections:
    print("  ❌ No BNS sections found in index!")
else:
    for s in bns_sections:
        print(f"\n  Section {s.get('section_number','?')} — {s.get('section_title','')}")
        text = (s.get("full_text") or s.get("text") or "")
        print(f"  Text ({len(text)} chars): {text[:300]!r}")

# ── 4. Live query test ────────────────────────────────────────────────────────
print("\n\n── Live cross-encoder query test ───────────────────────────────────────────")
TEST_QUERIES = [
    ("Assault — grievous hurt",     "punishment for voluntarily causing grievous hurt with weapon"),
    ("Eviction — property rights",  "civil remedy for unlawful forced eviction from property"),
    ("Encroachment — trespass",     "criminal liability for illegal encroachment trespass land"),
    ("Evidence — witness",          "admissibility of video recording as evidence in court"),
    ("Property — transfer",         "transfer of immovable property rights possession"),
]

try:
    from retrieval.hybrid_retriever import search_bare_acts_auto
    print("  Running 5 test queries …\n")
    for label, query in TEST_QUERIES:
        results = search_bare_acts_auto(query, top_k=3)
        print(f"  [{label}]")
        print(f"  Query: {query}")
        if not results:
            print("  ❌ No results returned")
        else:
            for r in results:
                act  = r.get("act_name", "?")
                sec  = r.get("section_number", "?")
                sc   = r.get("_rerank_score", 0)
                text = (r.get("full_text") or r.get("text") or "")[:80]
                print(f"    score={sc:6.2f}  {act}, §{sec}  |  {text!r}")
        print()
except Exception as e:
    print(f"  Could not run live query test: {e}")
    print("  (Make sure sentence-transformers and faiss are installed)")

print("\nDone.")
