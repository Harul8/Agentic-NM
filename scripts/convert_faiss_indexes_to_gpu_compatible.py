import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ACT_SUMMARY_INDEX_V2, BARE_INDEX_V2, CASE_INDEX_V2, CASE_SUMMARY_INDEX_V2


TARGETS = {
    "bareacts_v2": BARE_INDEX_V2,
    "caselaws_v2": CASE_INDEX_V2,
    "case_summaries_v2": CASE_SUMMARY_INDEX_V2,
    "act_summaries_v2": ACT_SUMMARY_INDEX_V2,
}

METRIC_LABELS = {
    faiss.METRIC_INNER_PRODUCT: "ip",
    faiss.METRIC_L2: "l2",
}


def _make_flat_index(dim: int, metric_type: int):
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexFlatIP(dim)
    if metric_type == faiss.METRIC_L2:
        return faiss.IndexFlatL2(dim)
    raise RuntimeError(f"Unsupported metric_type={metric_type}")


def _convert_index(path_str: str, batch_size: int, keep_backup: bool):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    src = faiss.read_index(str(path))
    metric_type = getattr(src, "metric_type", faiss.METRIC_L2)
    metric_label = METRIC_LABELS.get(metric_type, str(metric_type))
    src_type = type(src).__name__
    if src_type in {"IndexFlatIP", "IndexFlatL2"}:
        print(f"SKIP {path.name}: already {src_type}")
        return

    dst = _make_flat_index(src.d, metric_type)
    total = src.ntotal
    start_time = time.perf_counter()
    print(f"CONVERT {path.name}: {src_type} -> {type(dst).__name__} | ntotal={total} d={src.d} metric={metric_label}")

    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        xb = src.reconstruct_n(start, count)
        dst.add(np.ascontiguousarray(xb, dtype="float32"))
        done = start + count
        print(f"  progress {done}/{total}")

    tmp_path = path.with_suffix(path.suffix + ".flat.tmp")
    backup_path = path.with_suffix(path.suffix + ".hnsw.bak")
    faiss.write_index(dst, str(tmp_path))
    if keep_backup:
        if backup_path.exists():
            backup_path.unlink()
        path.replace(backup_path)
    else:
        path.unlink()
    tmp_path.replace(path)

    elapsed = time.perf_counter() - start_time
    print(f"DONE {path.name}: wrote {type(dst).__name__} in {elapsed:.1f}s")


def main():
    args = sys.argv[1:]
    if not args:
        names = ["case_summaries_v2", "act_summaries_v2", "bareacts_v2"]
    else:
        names = args

    batch_size = int(os.environ.get("NYAYMALAW_FAISS_CONVERT_BATCH", "8192"))
    keep_backup = os.environ.get("NYAYMALAW_FAISS_KEEP_BACKUP", "1").strip().lower() in ("1", "true", "yes")

    for name in names:
        path = TARGETS.get(name, name)
        _convert_index(path, batch_size=batch_size, keep_backup=keep_backup)


if __name__ == "__main__":
    main()
