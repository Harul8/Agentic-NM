"""
Vector Store Independent Analysis Script
=========================================
Run from the project root:
    python scripts/analyse_vector_store.py

Analyses bareacts_v2_chunks.json, caselaws_v2_chunks.json,
bareacts_v2.index and caselaws_v2.index from the configured DATA_ROOT.

Produces a structured report covering:
  1. Basic corpus stats
  2. Chunk size distribution (bare acts & case laws)
  3. Oversized chunks — full list
  4. Section boundary quality (bare acts)
  5. Act name quality
  6. Case law duplicate detection
  7. FAISS dimension vs current embedding model
  8. BM25 index health
  9. Act-specific deep dive (TPA, CPC, Companies Act)
 10. Search_text / full_text field presence
 11. Prioritised recommendation table
"""

import os, sys, json, re, hashlib, collections, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BARE_CHUNKS_V2, BARE_INDEX_V2, BARE_BM25_INDEX,
    CASE_CHUNKS_V2, CASE_INDEX_V2, CASE_BM25_INDEX,
    EMBEDDING_MODEL,
)

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def embed_text(chunk):
    """Mirror what build_v2_index sends to the embedder."""
    return (chunk.get("search_text") or chunk.get("full_text") or chunk.get("text") or "").strip()

def size_bucket(n):
    for lo, hi, label in [
        (0,     500,   "<500"),
        (500,   2000,  "500–2k"),
        (2000,  4000,  "2k–4k"),
        (4000,  8000,  "4k–8k"),
        (8000,  20000, "8k–20k"),
        (20000, 50000, "20k–50k"),
        (50000, 10**9, ">50k"),
    ]:
        if lo <= n < hi:
            return label
    return ">50k"

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "—"

def section_num_in_text(sec_num, text):
    """Check if the claimed section_number actually appears near the start of the chunk text."""
    probe = text[:300].lower()
    sn = sec_num.lower().replace("-", "").replace(" ", "")
    return sn in probe.replace("-", "").replace(" ", "")


def expected_embedding_dim(model_name):
    model = (model_name or "").strip().lower()
    if not model:
        return None
    explicit_dims = {
        "baai/bge-large-en-v1.5": 1024,
        "baai/bge-large-zh-v1.5": 1024,
        "baai/bge-m3": 1024,
        "baai/bge-base-en-v1.5": 768,
        "baai/bge-base-zh-v1.5": 768,
        "thenlper/gte-base": 768,
        "intfloat/e5-base-v2": 768,
        "law-ai/inlegalbert": 768,
        "nlpaueb/legal-bert-base-uncased": 768,
        "baai/bge-small-en-v1.5": 384,
        "sentence-transformers/all-minilm-l6-v2": 384,
    }
    if model in explicit_dims:
        return explicit_dims[model]
    if "bge-large" in model or "bge-m3" in model:
        return 1024
    if (
        "bge-base" in model
        or "gte-base" in model
        or "e5-base" in model
        or "legal-bert" in model
        or "bert-base" in model
        or "inlegalbert" in model
    ):
        return 768
    if "bge-small" in model or "minilm" in model or "all-mini" in model:
        return 384
    return None


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(SEP)
print("NYAYMALAW 3.0 — VECTOR STORE INDEPENDENT ANALYSIS")
print(SEP)

bare_chunks_raw = load_json(BARE_CHUNKS_V2)
case_chunks_raw = load_json(CASE_CHUNKS_V2)

if bare_chunks_raw is None:
    print(f"[ERROR] Cannot find bare-act chunks at: {BARE_CHUNKS_V2}")
    sys.exit(1)
if case_chunks_raw is None:
    print(f"[ERROR] Cannot find case-law chunks at: {CASE_CHUNKS_V2}")
    sys.exit(1)

# Normalise: dict → list
if isinstance(bare_chunks_raw, dict):
    bare_chunks = list(bare_chunks_raw.values())
else:
    bare_chunks = bare_chunks_raw

if isinstance(case_chunks_raw, dict):
    case_chunks = list(case_chunks_raw.values())
else:
    case_chunks = case_chunks_raw

print(f"\nData root   : {os.path.dirname(BARE_CHUNKS_V2)}")
print(f"Embedding   : {EMBEDDING_MODEL}")
print(f"Bare chunks : {len(bare_chunks):,}")
print(f"Case chunks : {len(case_chunks):,}")


# ---------------------------------------------------------------------------
# 2. FAISS dimension check
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("A. FAISS DIMENSION CHECK")
print(SEP2)

try:
    import faiss
    import numpy as np

    def check_faiss(path, label):
        if not os.path.exists(path):
            print(f"  {label}: FILE NOT FOUND at {path}")
            return
        idx = faiss.read_index(path)
        d = idx.d
        n = idx.ntotal
        print(f"  {label}: {n:,} vectors, dimension={d}")
        expected = expected_embedding_dim(EMBEDDING_MODEL)
        if expected:
            match = "✓ MATCH" if d == expected else f"✗ MISMATCH — model expects {expected}-dim"
            print(f"           → vs model ({EMBEDDING_MODEL}): {match}")
        else:
            print(f"           → vs model ({EMBEDDING_MODEL}): unknown expected dimension")

    check_faiss(BARE_INDEX_V2,  "Bare acts FAISS ")
    check_faiss(CASE_INDEX_V2,  "Case laws FAISS ")
except ImportError:
    print("  faiss not importable in this environment — skipping dimension check")
except Exception as e:
    print(f"  FAISS check error: {e}")


# ---------------------------------------------------------------------------
# 3. Bare act — basic stats + size distribution
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("B. BARE ACT CHUNKS — SIZE DISTRIBUTION")
print(SEP2)

bare_embed_lens = [len(embed_text(c)) for c in bare_chunks]
bare_full_lens  = [len(c.get("full_text","")) for c in bare_chunks]

buckets_e = collections.Counter(size_bucket(n) for n in bare_embed_lens)
bucket_order = ["<500","500–2k","2k–4k","4k–8k","8k–20k","20k–50k",">50k"]
print(f"  {'Bucket':<12}  {'Count':>7}  {'%':>6}  (embed_text length)")
for b in bucket_order:
    c = buckets_e[b]
    if c:
        print(f"  {b:<12}  {c:>7,}  {pct(c, len(bare_chunks)):>6}")

if bare_embed_lens:
    print(f"\n  Min   : {min(bare_embed_lens):,} chars")
    print(f"  Median: {sorted(bare_embed_lens)[len(bare_embed_lens)//2]:,} chars")
    print(f"  Mean  : {int(sum(bare_embed_lens)/len(bare_embed_lens)):,} chars")
    print(f"  Max   : {max(bare_embed_lens):,} chars")
    over_2k  = sum(1 for n in bare_embed_lens if n > 2000)
    over_8k  = sum(1 for n in bare_embed_lens if n > 8000)
    over_50k = sum(1 for n in bare_embed_lens if n > 50000)
    print(f"\n  Chunks > 2,000 chars (beyond ~512-token window) : {over_2k:,}  ({pct(over_2k,len(bare_chunks))})")
    print(f"  Chunks > 8,000 chars (severe truncation)        : {over_8k:,}  ({pct(over_8k,len(bare_chunks))})")
    print(f"  Chunks > 50,000 chars (catastrophic truncation) : {over_50k:,}  ({pct(over_50k,len(bare_chunks))})")


# ---------------------------------------------------------------------------
# 4. Bare act — oversized chunks, sorted desc
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("C. BARE ACT — TOP 30 OVERSIZED CHUNKS (by embed_text length)")
print(SEP2)

oversized = sorted(
    [(len(embed_text(c)), c) for c in bare_chunks if len(embed_text(c)) > 4000],
    key=lambda x: x[0],
    reverse=True,
)[:30]

if oversized:
    print(f"  {'Act':<45}  {'Section':<10}  {'Chars':>10}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*10}")
    for length, c in oversized:
        act  = (c.get("act_name","?") or "?")[:44]
        sec  = str(c.get("section_number","?") or "?")[:9]
        print(f"  {act:<45}  {sec:<10}  {length:>10,}")
else:
    print("  No chunks > 4,000 chars — good!")


# ---------------------------------------------------------------------------
# 5. Bare act — section number / metadata quality
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("D. BARE ACT — SECTION METADATA QUALITY")
print(SEP2)

missing_secnum   = [c for c in bare_chunks if not (c.get("section_number") or "").strip()]
fragment_titles  = []
_FRAG = re.compile(r"^\d+\.|^of \d{4}|^and [A-Z]|^Subs\. by|^Ins\. by|^Omitted|^Schedule", re.I)
for c in bare_chunks:
    t = (c.get("section_title") or "").strip()
    if t and _FRAG.match(t):
        fragment_titles.append(c)

# Section number not found in chunk text
mismatched_secnum = []
for c in bare_chunks:
    sn = (c.get("section_number") or "").strip()
    ft = embed_text(c)
    if sn and not section_num_in_text(sn, ft):
        mismatched_secnum.append(c)

print(f"  Chunks with no section_number          : {len(missing_secnum):,}  ({pct(len(missing_secnum),len(bare_chunks))})")
print(f"  Chunks with fragment section_title     : {len(fragment_titles):,}  ({pct(len(fragment_titles),len(bare_chunks))})")
print(f"  Chunks where section# not in text head : {len(mismatched_secnum):,}  ({pct(len(mismatched_secnum),len(bare_chunks))})")

if fragment_titles[:10]:
    print(f"\n  Sample fragment titles:")
    for c in fragment_titles[:10]:
        print(f"    [{c.get('act_name','?')[:35]}] S:{c.get('section_number','?')} → title: \"{c.get('section_title','')[:60]}\"")

if mismatched_secnum[:8]:
    print(f"\n  Sample section# mismatches (claimed vs text start):")
    for c in mismatched_secnum[:8]:
        preview = embed_text(c)[:80].replace("\n"," ")
        print(f"    [{c.get('act_name','?')[:30]}] claimed S:{c.get('section_number','?'):<6}  text: \"{preview}\"")


# ---------------------------------------------------------------------------
# 6. Bare act — act name quality
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("E. BARE ACT — ACT NAMES (all unique, with chunk counts)")
print(SEP2)

act_counts = collections.Counter(c.get("act_name","(missing)") or "(missing)" for c in bare_chunks)
suspicious = []
for act, cnt in sorted(act_counts.items(), key=lambda x: -x[1]):
    flag = ""
    alow = act.lower()
    if len(act) < 10:
        flag = "  ← TOO SHORT"
        suspicious.append((act, cnt, flag))
    elif re.search(r"\bof\s+act\b|\bthe act\b|^act$|^code$", alow):
        flag = "  ← LIKELY TRUNCATED"
        suspicious.append((act, cnt, flag))
    elif act.count(" ") < 1:
        flag = "  ← SINGLE WORD"
        suspicious.append((act, cnt, flag))
    print(f"  {cnt:>5}  {act}{flag}")

if suspicious:
    print(f"\n  ⚠  {len(suspicious)} suspicious act name(s) — see flags above")


# ---------------------------------------------------------------------------
# 7. Act-specific deep dive: TPA, CPC, Companies Act, BNSS
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("F. ACT-SPECIFIC SECTION BOUNDARY AUDIT")
print(SEP2)

TARGET_ACTS = {
    "tpa": "Transfer of Property",
    "cpc": "Code of Civil Procedure",
    "companies": "Companies Act",
    "bnss": "Bharatiya Nagarik Suraksha",
    "bns":  "Bharatiya Nyaya Sanhita",
}

for key, label in TARGET_ACTS.items():
    act_chunks = [c for c in bare_chunks
                  if label.lower() in (c.get("act_name","") or "").lower()]
    if not act_chunks:
        print(f"\n  [{label}] — no chunks found")
        continue

    act_name_sample = act_chunks[0].get("act_name","?")
    total_chars = sum(len(embed_text(c)) for c in act_chunks)
    max_chunk   = max(act_chunks, key=lambda c: len(embed_text(c)))
    max_len     = len(embed_text(max_chunk))
    over_2k     = sum(1 for c in act_chunks if len(embed_text(c)) > 2000)
    over_8k     = sum(1 for c in act_chunks if len(embed_text(c)) > 8000)

    print(f"\n  [{act_name_sample}]")
    print(f"    Chunks: {len(act_chunks)}  |  Total chars: {total_chars:,}  |  >2k: {over_2k}  |  >8k: {over_8k}")
    print(f"    Largest chunk: S:{max_chunk.get('section_number','?')}  ({max_len:,} chars)")

    # Show all section numbers in order
    sec_nums = [(c.get("section_number","?"), len(embed_text(c))) for c in act_chunks]
    over8k_secs = [(sn, sz) for sn, sz in sec_nums if sz > 8000]
    if over8k_secs:
        print(f"    Sections >8k chars:")
        for sn, sz in over8k_secs:
            print(f"      S:{sn:<10} {sz:>10,} chars")

    # For TPA: specifically check if S.106 is reachable
    if key == "tpa":
        s106 = [c for c in act_chunks if str(c.get("section_number","")) == "106"]
        if s106:
            sz = len(embed_text(s106[0]))
            print(f"    TPA S.106 found as own chunk: {sz:,} chars  ✓")
        else:
            # Find which chunk contains "section 106" text
            containing = []
            for c in act_chunks:
                if re.search(r'\bsection\s+106\b', embed_text(c), re.I):
                    containing.append(c)
            if containing:
                for cc in containing:
                    et = embed_text(cc)
                    pos = re.search(r'\bsection\s+106\b', et, re.I)
                    print(f"    TPA S.106 NOT its own chunk — buried inside S:{cc.get('section_number','?')} "
                          f"at char {pos.start():,} of {len(et):,}  ✗")
            else:
                print(f"    TPA S.106: NOT found anywhere in bare-act chunks  ✗")


# ---------------------------------------------------------------------------
# 8. Case law — size distribution
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("G. CASE LAW CHUNKS — SIZE DISTRIBUTION")
print(SEP2)

case_embed_lens = [len(embed_text(c)) for c in case_chunks]
buckets_c = collections.Counter(size_bucket(n) for n in case_embed_lens)
print(f"  {'Bucket':<12}  {'Count':>7}  {'%':>6}")
for b in bucket_order:
    cnt = buckets_c[b]
    if cnt:
        print(f"  {b:<12}  {cnt:>7,}  {pct(cnt, len(case_chunks)):>6}")

if case_embed_lens:
    print(f"\n  Min   : {min(case_embed_lens):,} chars")
    print(f"  Median: {sorted(case_embed_lens)[len(case_embed_lens)//2]:,} chars")
    print(f"  Mean  : {int(sum(case_embed_lens)/len(case_embed_lens)):,} chars")
    print(f"  Max   : {max(case_embed_lens):,} chars")
    c_over_2k  = sum(1 for n in case_embed_lens if n > 2000)
    c_over_8k  = sum(1 for n in case_embed_lens if n > 8000)
    print(f"\n  Chunks > 2,000 chars : {c_over_2k:,}  ({pct(c_over_2k,len(case_chunks))})")
    print(f"  Chunks > 8,000 chars : {c_over_8k:,}  ({pct(c_over_8k,len(case_chunks))})")


# ---------------------------------------------------------------------------
# 9. Case law — duplicates
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("H. CASE LAW — DUPLICATE EMBED-TEXT DETECTION")
print(SEP2)

hash_to_chunks = collections.defaultdict(list)
for i, c in enumerate(case_chunks):
    h = hashlib.md5(embed_text(c).encode("utf-8")).hexdigest()
    hash_to_chunks[h].append(i)

dup_groups  = {h: idxs for h, idxs in hash_to_chunks.items() if len(idxs) > 1}
dup_chunks  = sum(len(v) - 1 for v in dup_groups.values())  # extra copies
total_dups  = sum(len(v) for v in dup_groups.values())

print(f"  Unique embed texts    : {len(hash_to_chunks) - len(dup_groups):,}")
print(f"  Duplicate groups      : {len(dup_groups):,}")
print(f"  Wasted (extra) chunks : {dup_chunks:,}  ({pct(dup_chunks, len(case_chunks))})")

# Show top duplicate groups
top_dups = sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:10]
if top_dups:
    print(f"\n  Top duplicate groups:")
    print(f"  {'Copies':>6}  {'Case name':<50}  {'Text preview'}")
    for h, idxs in top_dups:
        c = case_chunks[idxs[0]]
        case_name = (c.get("case_name") or "?")[:49]
        preview   = embed_text(c)[:60].replace("\n"," ")
        print(f"  {len(idxs):>6}  {case_name:<50}  \"{preview}\"")


# ---------------------------------------------------------------------------
# 10. Case law — metadata quality
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("I. CASE LAW — METADATA QUALITY")
print(SEP2)

no_case_name  = sum(1 for c in case_chunks if not (c.get("case_name","") or "").strip())
no_citation   = sum(1 for c in case_chunks if not (c.get("citation","") or "").strip())
no_court      = sum(1 for c in case_chunks if not (c.get("court","") or "").strip())
no_year       = sum(1 for c in case_chunks if not (c.get("year","") or "").strip())

print(f"  Missing case_name  : {no_case_name:,}  ({pct(no_case_name, len(case_chunks))})")
print(f"  Missing citation   : {no_citation:,}  ({pct(no_citation, len(case_chunks))})")
print(f"  Missing court      : {no_court:,}  ({pct(no_court, len(case_chunks))})")
print(f"  Missing year       : {no_year:,}  ({pct(no_year, len(case_chunks))})")

# Court distribution
court_dist = collections.Counter((c.get("court","") or "(none)") for c in case_chunks)
print(f"\n  Court distribution (top 10):")
for court, cnt in court_dist.most_common(10):
    print(f"    {cnt:>7,}  {court}")


# ---------------------------------------------------------------------------
# 11. BM25 index health
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("J. BM25 INDEX HEALTH")
print(SEP2)

def check_bm25(path, chunk_count, label):
    if not os.path.exists(path):
        print(f"  {label}: FILE NOT FOUND at {path}")
        return
    size_mb = os.path.getsize(path) / 1024 / 1024
    data = load_json(path)
    if data is None:
        print(f"  {label}: Could not load JSON")
        return
    bm25_doc_count = data.get("doc_count", "?")
    avg_dl         = data.get("avg_dl", "?")
    vocab_size     = len(data.get("doc_freqs", {}))
    match = "✓" if bm25_doc_count == chunk_count else f"✗ MISMATCH (chunks={chunk_count})"
    print(f"  {label}:")
    print(f"    File size   : {size_mb:.1f} MB")
    print(f"    doc_count   : {bm25_doc_count}  {match}")
    print(f"    avg_dl      : {avg_dl:.0f} tokens" if isinstance(avg_dl, float) else f"    avg_dl: {avg_dl}")
    print(f"    vocab size  : {vocab_size:,} terms")

check_bm25(BARE_BM25_INDEX, len(bare_chunks), "Bare acts BM25")
print()
check_bm25(CASE_BM25_INDEX, len(case_chunks), "Case laws BM25")


# ---------------------------------------------------------------------------
# 12. field presence audit
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("K. FIELD PRESENCE AUDIT")
print(SEP2)

def field_audit(chunks, label, fields):
    print(f"  {label} ({len(chunks):,} chunks):")
    for f in fields:
        present = sum(1 for c in chunks if (c.get(f) or ""))
        if present < len(chunks):
            print(f"    {f:<18}: {present:>7,} present  ({pct(present,len(chunks))} — {len(chunks)-present} missing)")
        else:
            print(f"    {f:<18}: {present:>7,} present  (100.0%)")
    print()

field_audit(bare_chunks, "Bare acts", ["section_number", "act_name", "citation", "keywords"])
field_audit(case_chunks, "Case laws", ["case_name", "citation", "court", "year", "keywords"])


# ---------------------------------------------------------------------------
# 13. Prioritised recommendation table
# ---------------------------------------------------------------------------
print(SEP)
print("L. PRIORITISED RECOMMENDATIONS")
print(SEP2)

recs = [
    ("CRITICAL", "Split oversized bare-act sections into sub-chunks",
     "MAX_SECTION_CHARS cap (~4k) in smart_chunker.chunk_bare_act(); sub-chunks share act_name+section_number but get distinct chunk_ids"),
    ("CRITICAL", "FAISS dim mismatch — rebuild vector store if index dim != current model dim",
     f"Current model is {EMBEDDING_MODEL}; rebuild whenever the FAISS index dimension differs from that model"),
    ("HIGH",     "Deduplicate case-law chunks before indexing",
     "Hash embed_text in process_case_laws_directory(); skip duplicate; saves ~8-10% FAISS/BM25 slots"),
    ("HIGH",     "Fix 'Protection Of Act' and other truncated act names",
     "Improve _detect_act_name_from_text() regex; also fixes allowed_acts filter failures"),
    ("HIGH",     "Fragment section_titles — fix section boundary detection",
     "Tighten Pattern 2 to reject lines starting with digits that look like footnote/amendment refs"),
    ("MEDIUM",   "Section# not in chunk text (mismatched metadata)",
     "Post-process chunks: verify section_num appears in first 300 chars; flag or re-detect if not"),
    ("MEDIUM",   "Case law missing citation/court (~high %)",
     "Improve _detect_case_metadata() to catch more citation formats (ILR, Manu, SCR, SCC Online)"),
    ("LOW",      "Illustrations as separate sub-chunk for sections like TPA S.106",
     "Detect 'ILLUSTRATION' block within section text and split it off as _illus sub-chunk"),
    ("LOW",      "Skip/flag embed_text < 50 chars",
     "Add guard in build_index() to skip degenerate chunks"),
]

print(f"  {'Priority':<10}  {'Issue':<48}  {'Action'}")
print(f"  {'-'*10}  {'-'*48}  {'-'*30}")
for pri, issue, action in recs:
    print(f"  {pri:<10}  {issue:<48}")
    print(f"             → {action}")
    print()

print(SEP)
print("Analysis complete.")
print(SEP)
