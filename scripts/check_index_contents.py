"""
Diagnostic: show exactly what acts and sections are in the local bare_acts index.
Run from the project root:   python scripts/check_index_contents.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BARE_CHUNKS_V2, CASE_CHUNKS_V2

def show_index_contents(chunks_path: str, label: str) -> None:
    if not os.path.isfile(chunks_path):
        print(f"  ✗ {label} index NOT FOUND at: {chunks_path}")
        return

    size_mb = os.path.getsize(chunks_path) / (1024 * 1024)
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    # Normalise: dict-of-chunks or list-of-chunks
    if isinstance(data, dict):
        chunks = list(data.values())
    else:
        chunks = data

    print(f"  ✓ {label} index: {len(chunks)} chunks ({size_mb:.1f} MB)")
    print()

    act_counts: Counter = Counter()
    act_sections: dict = defaultdict(set)   # act_name → set of section numbers

    for c in chunks:
        meta = c.get("metadata") or c
        act  = (meta.get("act_name") or "UNKNOWN").strip()
        sec  = str(meta.get("section_number") or "?").strip()
        act_counts[act] += 1
        act_sections[act].add(sec)

    print(f"  {'CHUNKS':>6}  {'UNIQUE §':>8}  ACT NAME")
    print(f"  {'-'*6}  {'-'*8}  {'-'*50}")
    for act, cnt in sorted(act_counts.items(), key=lambda x: -x[1]):
        unique_secs = len(act_sections[act])
        print(f"  {cnt:6d}  {unique_secs:8d}  {act}")
    print()


def test_query(query: str, top_k: int = 5) -> None:
    """Run a single query against the local bare_acts index and show top results."""
    from retrieval.hybrid_retriever import search_bare_acts_auto
    print(f"  Query: \"{query}\"")
    results = search_bare_acts_auto(query, top_k=top_k)
    if not results:
        print("  → 0 results (index may be empty or query has no matching sections)")
    for i, r in enumerate(results, 1):
        act   = (r.get("act_name") or "?").strip()
        sec   = (r.get("section_number") or "?").strip()
        title = (r.get("section_title") or r.get("title") or "").strip()[:60]
        score = r.get("_rerank_score", 0.0)
        print(f"  {i}. [{score:+.3f}]  {act} § {sec}  —  {title}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("NYAYMALAW LOCAL INDEX DIAGNOSTIC")
    print("=" * 70)
    print()

    print("── BARE ACTS INDEX ─────────────────────────────────────────────────")
    show_index_contents(BARE_CHUNKS_V2, "Bare Acts")

    print("── CASE LAWS INDEX ─────────────────────────────────────────────────")
    show_index_contents(CASE_CHUNKS_V2, "Case Laws")

    # Run a few test queries to see if the right acts surface
    print("── RETRIEVAL TEST QUERIES ──────────────────────────────────────────")
    print("  (Should see BNS/BNSS/BSA sections for criminal queries,")
    print("   Transfer of Property Act for property queries)\n")
    test_queries = [
        "voluntarily causing hurt assault injury",
        "unlawful eviction from property possession",
        "encroachment on land trespass property",
        "admissibility of evidence in court",
        "arrest without warrant police powers",
    ]
    for q in test_queries:
        test_query(q, top_k=3)

    print("=" * 70)
    print("INSTRUCTIONS:")
    print("  If 'Bare Acts' shows 0 chunks → the index has never been built.")
    print("  Run:  python scripts/index_bare_acts.py  (or the admin UI → Index)")
    print()
    print("  If BNS/BNSS/BSA/TPA are MISSING from the act list above:")
    print("  → Place their PDFs in the BareActs/ folder and re-index.")
    print("  → Key PDFs needed:")
    print("      Bharatiya_Nyaya_Sanhita_2023.pdf   (BNS - replaces IPC)")
    print("      Bharatiya_Nagarik_Suraksha_Sanhita_2023.pdf  (BNSS - replaces CrPC)")
    print("      Bharatiya_Sakshya_Adhiniyam_2023.pdf  (BSA - replaces IEA)")
    print("      Transfer_of_Property_Act_1882.pdf")
    print()
    print("  If the test queries return IPC/IEA sections instead of BNS/BNSS:")
    print("  → Both are indexed; IPC is winning the score race.")
    print("  → Add a section-level re-ranking preference for new acts in config.")
    print("=" * 70)
