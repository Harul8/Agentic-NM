"""
Download specific Union of India bare acts from Indian Kanoon and save them
under legal_database/raw_data/BareActs/Union of India with year_index.txt names.

Usage:
  python scripts/add_union_bareacts_manual.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _scripts_dir.parent
UNION_DIR = PROJECT_ROOT / "legal_database" / "raw_data" / "BareActs" / "Union of India"

# URLs to download
URLS = [
    "https://indiankanoon.org/doc/149679501/",
    "https://indiankanoon.org/doc/91117739/",
    "https://indiankanoon.org/doc/70224818/",
]


def _ensure_sys_path() -> None:
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))


def _guess_year(text: str) -> str:
    """Guess first 4-digit year (>=1900) from the text; default '0000' if none."""
    if not text:
        return "0000"
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if not m:
        return "0000"
    return m.group(0)


def _next_index_for_year(year: str) -> int:
    """Return next sequential index for files named year_*.txt in UNION_DIR."""
    if not UNION_DIR.is_dir():
        return 0
    max_idx = -1
    for p in UNION_DIR.glob(f"{year}_*.txt"):
        stem = p.stem  # e.g. 2019_8
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def main() -> None:
    _ensure_sys_path()
    try:
        from download_indiankanoon_data import get_doc_text_any  # type: ignore
    except Exception as e:  # pragma: no cover - defensive
        print(f"Could not import get_doc_text_any: {e}")
        return

    UNION_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        print(f"\nFetching: {url}")
        text = get_doc_text_any(url)
        if not text:
            print("  -> Skipped (no text or fetch error).")
            continue
        year = _guess_year(text)
        idx = _next_index_for_year(year)
        out_path = UNION_DIR / f"{year}_{idx}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> Saved to {out_path}")


if __name__ == "__main__":
    main()

