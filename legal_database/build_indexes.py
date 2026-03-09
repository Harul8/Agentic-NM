#!/usr/bin/env python
"""
Build vector indexes from existing json_output — FAISS + BM25 + citation graph.

Extraction is in pipeline.py (PDF → json_output). Indexing is here: run this
after pipeline.py when json_output exists. No PDF processing.

Usage:
  python build_indexes.py
  # or from project root:
  python legal_database/build_indexes.py
"""
import os
import sys

# Ensure project root on path (for retrieval, Ingestion, config imports)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Run index build (no PDF extraction)
from pipeline import run_build_indexes_only

if __name__ == "__main__":
    run_build_indexes_only()
