"""
List case names in the current case law index (internal search).
Run from project root: python scripts/list_indexed_cases.py

Use this to verify whether a case (e.g. "APPELLANTS v. RELIANCE INDUSTRIES LTD.") 
exists in your local vector store.
"""
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CASE_CHUNKS_V2, CASE_CHUNKS, VECTOR_STORE


def main():
    # Prefer v2 chunks
    for path in (CASE_CHUNKS_V2, CASE_CHUNKS):
        if not os.path.exists(path):
            continue
        print(f"Loading: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error: {e}")
            return

        # data is either dict (key -> chunk) or list of chunks
        if isinstance(data, dict):
            chunks = list(data.values())
        else:
            chunks = data

        case_names = set()
        source_files = set()
        for c in chunks:
            name = (c.get("case_name") or "").strip()
            src = (c.get("source") or c.get("source_file") or "").strip()
            if name:
                case_names.add(name)
            if src:
                source_files.add(src)

        print(f"Total chunks: {len(chunks)}")
        print(f"Unique case_name values: {len(case_names)}")
        print()

        target = "APPELLANTS v. RELIANCE INDUSTRIES LTD."
        if target in case_names:
            print(f"YES — '{target}' EXISTS in the internal case law index.")
        else:
            print(f"NO — '{target}' NOT FOUND in the internal case law index.")
        print()

        print("All indexed case names (first 80):")
        for i, name in enumerate(sorted(case_names)[:80]):
            print(f"  {i+1}. {name[:80]}{'...' if len(name) > 80 else ''}")
        if len(case_names) > 80:
            print(f"  ... and {len(case_names) - 80} more.")

        return

    print(f"Neither {CASE_CHUNKS_V2} nor {CASE_CHUNKS} found.")
    print(f"Vector store dir: {VECTOR_STORE}")
    print("No case law index present — internal search has no case laws.")


if __name__ == "__main__":
    main()
