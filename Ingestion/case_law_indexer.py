# DEPRECATED — Use Ingestion.build_v2_index instead.
# Run: python Ingestion/build_v2_index.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("case_law_indexer.py is DEPRECATED. Redirecting to build_v2_index.py...")
    from Ingestion.build_v2_index import main as build_main
    build_main()


if __name__ == "__main__":
    main()
