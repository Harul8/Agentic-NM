"""
Threshold Sweep — Sweeps MIN_RERANK_SCORE from 0.0 to 1.0 and plots
Precision-Recall trade-off curves to find the optimal threshold.

This is one of the most important analyses: it tells you whether 0.55 is
actually the right threshold, or if a different value gives better results.

Usage:
    python -m eval.threshold_sweep --queries eval/data/queries.json --out eval/results/threshold/
    python -m eval.threshold_sweep --results eval/results/batch_full_pipeline_*.json --out eval/results/threshold/

The sweep can run in two modes:
1. Live mode (--queries): runs retrieval for each query and tests all thresholds
2. Offline mode (--results): uses pre-computed batch results with raw scores
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("eval.threshold_sweep")


def _get_gold_case_names(gold: dict) -> set:
    """Extract normalized gold case names for matching."""
    names = set()
    for cl in gold.get("case_laws", []):
        name = (cl.get("case_name") or "").lower().strip()
        if name:
            names.add(name)
    return names


def _get_gold_sections(gold: dict) -> set:
    """Extract normalized gold sections for matching."""
    sections = set()
    for ba in gold.get("bare_acts", []):
        act = (ba.get("act_name") or "").lower().strip()
        sec = (ba.get("section_number") or "").lower().strip()
        if act and sec:
            sections.add(f"{act}|{sec}")
    return sections


def _fuzzy_match_case(name: str, gold_names: set) -> bool:
    """Check if a case name fuzzy-matches any gold case name."""
    n = name.lower().strip()
    for g in gold_names:
        if g in n or n in g:
            return True
        # Party overlap
        g_parts = set(p.strip() for p in g.replace(" v ", "|").replace(" v. ", "|").split("|"))
        n_parts = set(p.strip() for p in n.replace(" v ", "|").replace(" v. ", "|").split("|"))
        overlap = g_parts & n_parts
        if overlap and any(len(p) > 3 for p in overlap):
            return True
    return False


def sweep_single_query(raw_scores: dict, gold: dict, thresholds: list) -> dict:
    """
    For a single query, compute metrics at each threshold.

    Args:
        raw_scores: {"bare_act_rerank_scores": [...], "case_law_rerank_scores": [...]}
        gold: {"bare_acts": [...], "case_laws": [...]}
        thresholds: list of float thresholds to test

    Returns:
        {threshold: {"precision": ..., "recall": ..., "f1": ..., "count": ...}}
    """
    gold_case_names = _get_gold_case_names(gold)
    gold_sections = _get_gold_sections(gold)

    ba_scores = raw_scores.get("bare_act_rerank_scores", [])
    cl_scores = raw_scores.get("case_law_rerank_scores", [])

    results = {}
    for threshold in thresholds:
        # Filter by threshold
        ba_above = [s for s in ba_scores if s.get("score", 0) >= threshold]
        cl_above = [s for s in cl_scores if s.get("score", 0) >= threshold]

        # Count matches
        ba_matches = sum(
            1 for s in ba_above
            if f"{(s.get('act_name') or '').lower()}|{(s.get('section') or '').lower()}" in gold_sections
        ) if gold_sections else 0

        cl_matches = sum(
            1 for s in cl_above
            if _fuzzy_match_case(s.get("case_name", ""), gold_case_names)
        ) if gold_case_names else 0

        total_retrieved = len(ba_above) + len(cl_above)
        total_relevant_found = ba_matches + cl_matches
        total_gold = len(gold_sections) + len(gold_case_names)

        precision = total_relevant_found / max(total_retrieved, 1)
        recall = total_relevant_found / max(total_gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        results[threshold] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "retrieved_count": total_retrieved,
            "relevant_found": total_relevant_found,
            "total_gold": total_gold,
        }

    return results


def sweep_batch(results_path: str, thresholds: Optional[list] = None) -> dict:
    """Run threshold sweep over a batch of pre-computed results."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(0, 21)]  # 0.0 to 1.0 in 0.05 steps

    per_query_sweeps = []
    for r in results:
        gold = r.get("gold_annotations", {})
        raw = r.get("raw_scores", {})
        if not gold or not raw:
            continue
        sweep = sweep_single_query(raw, gold, thresholds)
        per_query_sweeps.append({
            "query_id": r.get("query_id", ""),
            "sweep": sweep,
        })

    # Aggregate: mean metrics at each threshold
    aggregated = {}
    for t in thresholds:
        t_key = str(t)
        precisions = [q["sweep"][t]["precision"] for q in per_query_sweeps if t in q["sweep"]]
        recalls = [q["sweep"][t]["recall"] for q in per_query_sweeps if t in q["sweep"]]
        f1s = [q["sweep"][t]["f1"] for q in per_query_sweeps if t in q["sweep"]]
        counts = [q["sweep"][t]["retrieved_count"] for q in per_query_sweeps if t in q["sweep"]]

        if precisions:
            aggregated[t_key] = {
                "mean_precision": round(float(np.mean(precisions)), 4),
                "mean_recall": round(float(np.mean(recalls)), 4),
                "mean_f1": round(float(np.mean(f1s)), 4),
                "mean_retrieved_count": round(float(np.mean(counts)), 1),
                "n_queries": len(precisions),
            }

    # Find optimal threshold (max F1)
    best_threshold = max(aggregated.items(), key=lambda x: x[1]["mean_f1"])[0] if aggregated else "0.5"

    return {
        "thresholds_tested": thresholds,
        "aggregated": aggregated,
        "optimal_threshold": float(best_threshold),
        "optimal_f1": aggregated.get(best_threshold, {}).get("mean_f1", 0),
        "per_query": per_query_sweeps,
    }


def sweep_live(queries_path: str, thresholds: Optional[list] = None) -> dict:
    """
    Run retrieval for each query and compute threshold sweep metrics.
    This is slower but gives raw score data.
    """
    from eval.batch_runner import _run_retrieval, _run_quality_filters
    from pipeline.generator import expand_legal_query

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(0, 21)]

    per_query_sweeps = []
    for i, q in enumerate(queries):
        query_text = q.get("query", "")
        gold = q.get("gold", {})
        if not gold:
            continue

        logger.info(f"[{i+1}/{len(queries)}] Sweeping: {query_text[:60]}...")

        legal_query = expand_legal_query(query_text)
        search_query = f"{query_text} {legal_query}"[:500]
        raw = _run_retrieval(search_query, "full_pipeline")

        # Build raw scores
        raw_scores = {
            "bare_act_rerank_scores": [
                {"score": b.get("_rerank_score", 0), "act_name": b.get("act_name", ""), "section": b.get("section_number", "")}
                for b in raw.get("bare_acts_raw", [])
            ],
            "case_law_rerank_scores": [
                {"score": c.get("_rerank_score", 0), "case_name": c.get("case_name", ""), "year": c.get("year", "")}
                for c in raw.get("case_laws_raw", [])
            ],
        }

        sweep = sweep_single_query(raw_scores, gold, thresholds)
        per_query_sweeps.append({
            "query_id": q.get("id", f"q_{i+1}"),
            "sweep": sweep,
            "raw_score_distribution": {
                "bare_act_scores": sorted([s["score"] for s in raw_scores["bare_act_rerank_scores"]], reverse=True),
                "case_law_scores": sorted([s["score"] for s in raw_scores["case_law_rerank_scores"]], reverse=True),
            },
        })

    # Aggregate
    aggregated = {}
    for t in thresholds:
        precisions = [q["sweep"][t]["precision"] for q in per_query_sweeps if t in q["sweep"]]
        recalls = [q["sweep"][t]["recall"] for q in per_query_sweeps if t in q["sweep"]]
        f1s = [q["sweep"][t]["f1"] for q in per_query_sweeps if t in q["sweep"]]
        counts = [q["sweep"][t]["retrieved_count"] for q in per_query_sweeps if t in q["sweep"]]

        if precisions:
            aggregated[str(t)] = {
                "mean_precision": round(float(np.mean(precisions)), 4),
                "mean_recall": round(float(np.mean(recalls)), 4),
                "mean_f1": round(float(np.mean(f1s)), 4),
                "mean_retrieved_count": round(float(np.mean(counts)), 1),
            }

    best = max(aggregated.items(), key=lambda x: x[1]["mean_f1"])[0] if aggregated else "0.5"

    return {
        "thresholds_tested": thresholds,
        "aggregated": aggregated,
        "optimal_threshold": float(best),
        "optimal_f1": aggregated.get(best, {}).get("mean_f1", 0),
        "per_query": per_query_sweeps,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Threshold sweep for optimal rerank score")
    parser.add_argument("--queries", help="Queries JSON for live sweep")
    parser.add_argument("--results", help="Pre-computed batch results for offline sweep")
    parser.add_argument("--out", default="eval/results/threshold/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.results:
        results_path = args.results
        # Expand glob on Windows (PowerShell doesn't expand batch_full_pipeline_*.json)
        if "*" in results_path:
            matches = sorted(glob.glob(results_path), key=lambda p: (os.path.getmtime(p), p), reverse=True)
            if not matches:
                logger.error(f"No files match: {results_path}")
                sys.exit(1)
            results_path = matches[0]
            logger.info(f"Using: {results_path}")
        report = sweep_batch(results_path)
    elif args.queries:
        report = sweep_live(args.queries)
    else:
        parser.print_help()
        sys.exit(1)

    output_file = os.path.join(args.out, "threshold_sweep.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Optimal threshold: {report['optimal_threshold']} (F1={report['optimal_f1']:.4f})")
    logger.info(f"Report saved to {output_file}")
