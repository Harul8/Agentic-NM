"""
Retrieval Metrics — Computes Precision@k, Recall@k, MRR, nDCG@k, and
Rerank-Score AUC by comparing system results against gold-standard annotations.

Usage:
    python -m eval.retrieval_metrics --results eval/results/batch_full_pipeline_*.json --gold eval/data/queries.json
    python -m eval.retrieval_metrics --results eval/results/ --compare  # compare all ablation modes

Input format (queries.json):
    [
        {
            "id": "q_1",
            "query": "...",
            "gold": {
                "bare_acts": [
                    {"act_name": "Transfer of Property Act", "section_number": "54"}
                ],
                "case_laws": [
                    {"case_name": "State of Bihar v. Kameshwar Singh", "year": "1952"}
                ]
            }
        }
    ]
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("eval.retrieval_metrics")


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if not relevant:
        return 1.0  # No relevant items = trivially complete
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list, relevant: set) -> float:
    """MRR: 1/rank of the first relevant result."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    if k <= 0 or not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed

    # Ideal DCG (all relevant items ranked first)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def rerank_score_auc(scores_and_labels: list) -> float:
    """
    Compute AUC-ROC using rerank scores as predictor of relevance.
    Input: list of (rerank_score, is_relevant_bool)
    """
    if not scores_and_labels:
        return 0.5

    # Sort by score descending
    sorted_items = sorted(scores_and_labels, key=lambda x: -x[0])

    positives = sum(1 for _, label in sorted_items if label)
    negatives = len(sorted_items) - positives
    if positives == 0 or negatives == 0:
        return 0.5  # Undefined

    # Wilcoxon-Mann-Whitney statistic
    tp = 0
    auc_sum = 0.0
    for score, label in sorted_items:
        if label:
            tp += 1
        else:
            auc_sum += tp

    return auc_sum / (positives * negatives)


# ---------------------------------------------------------------------------
# Gold-Standard Matching
# ---------------------------------------------------------------------------

def _normalize_case_name(name: str) -> str:
    """Normalize case name for fuzzy matching."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common prefixes/suffixes
    for remove in ("the ", "union of india", "state of ", "m/s ", "shri ", "smt "):
        n = n.replace(remove, "")
    # Normalize separators
    for sep in (" v. ", " vs. ", " v/s ", " versus "):
        n = n.replace(sep, " v ")
    # Remove extra whitespace
    return " ".join(n.split())


def _normalize_section(act: str, section: str) -> str:
    """Normalize bare act + section for matching."""
    a = (act or "").lower().strip()
    s = (section or "").lower().strip().lstrip("section ").lstrip("s. ").lstrip("s ")
    return f"{a}|{s}"


def _match_case_law(retrieved_case: dict, gold_cases: list) -> bool:
    """Check if a retrieved case matches any gold-standard case."""
    ret_name = _normalize_case_name(
        retrieved_case.get("case_name", retrieved_case.get("title", ""))
    )
    if not ret_name:
        return False

    for gold in gold_cases:
        gold_name = _normalize_case_name(gold.get("case_name", ""))
        if not gold_name:
            continue
        # Substring match (case names often vary in citation format)
        if gold_name in ret_name or ret_name in gold_name:
            return True
        # Check if key parties match
        gold_parties = set(gold_name.replace(" v ", "|").split("|"))
        ret_parties = set(ret_name.replace(" v ", "|").split("|"))
        if gold_parties and ret_parties:
            overlap = gold_parties & ret_parties
            if len(overlap) >= 1 and any(len(p) > 3 for p in overlap):
                return True
    return False


def _match_bare_act(retrieved_ba: dict, gold_bare_acts: list) -> bool:
    """Check if a retrieved bare act section matches any gold-standard entry."""
    ret_key = _normalize_section(
        retrieved_ba.get("act_name", ""),
        retrieved_ba.get("section_number", ""),
    )
    for gold in gold_bare_acts:
        gold_key = _normalize_section(
            gold.get("act_name", ""),
            gold.get("section_number", ""),
        )
        if gold_key and ret_key and gold_key == ret_key:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-Query Evaluation
# ---------------------------------------------------------------------------

def evaluate_query(result: dict) -> dict:
    """
    Evaluate a single query's results against gold annotations.
    Returns metrics dict.
    """
    gold = result.get("gold_annotations", {})
    gold_bare_acts = gold.get("bare_acts", [])
    gold_case_laws = gold.get("case_laws", [])

    # Extract system results
    final = result.get("final_response") or {}
    sys_bare_acts = final.get("bare_act_sections", [])
    sys_case_laws = final.get("case_laws", [])

    metrics = {"query_id": result.get("query_id", "")}

    # Bare act metrics
    if gold_bare_acts:
        gold_ba_set = set(
            _normalize_section(g.get("act_name", ""), g.get("section_number", ""))
            for g in gold_bare_acts
        )
        retrieved_ba = [
            _normalize_section(b.get("act_name", ""), b.get("section_number", ""))
            for b in sys_bare_acts
        ]
        for k in (3, 5, 10):
            metrics[f"bare_act_precision@{k}"] = precision_at_k(retrieved_ba, gold_ba_set, k)
            metrics[f"bare_act_recall@{k}"] = recall_at_k(retrieved_ba, gold_ba_set, k)
            metrics[f"bare_act_ndcg@{k}"] = ndcg_at_k(retrieved_ba, gold_ba_set, k)
        metrics["bare_act_mrr"] = mean_reciprocal_rank(retrieved_ba, gold_ba_set)

    # Case law metrics
    if gold_case_laws:
        gold_cl_names = gold_case_laws  # Keep as list for fuzzy matching
        retrieved_cl_matched = [
            _match_case_law(c, gold_cl_names) for c in sys_case_laws
        ]
        # Convert to positional list for standard metrics
        matched_indices = [i for i, m in enumerate(retrieved_cl_matched) if m]
        relevant_set = set(range(len(gold_cl_names)))  # Ideal: all gold items found

        # Build a retrieved list where matched items map to gold indices
        retrieved_as_indices = []
        gold_matched = 0
        for i, is_match in enumerate(retrieved_cl_matched):
            if is_match and gold_matched < len(gold_cl_names):
                retrieved_as_indices.append(gold_matched)
                gold_matched += 1
            else:
                retrieved_as_indices.append(i + 1000)  # Non-matching sentinel

        for k in (3, 5, 10):
            metrics[f"case_law_precision@{k}"] = precision_at_k(retrieved_as_indices, relevant_set, k)
            metrics[f"case_law_recall@{k}"] = recall_at_k(retrieved_as_indices, relevant_set, k)
            metrics[f"case_law_ndcg@{k}"] = ndcg_at_k(retrieved_as_indices, relevant_set, k)
        metrics["case_law_mrr"] = mean_reciprocal_rank(retrieved_as_indices, relevant_set)

    # Rerank score AUC (from raw scores)
    raw_scores = result.get("raw_scores", {})
    for category, gold_list, match_fn in [
        ("bare_act", gold_bare_acts, lambda s: any(
            _normalize_section(s.get("act_name", ""), s.get("section", "")) ==
            _normalize_section(g.get("act_name", ""), g.get("section_number", ""))
            for g in gold_bare_acts
        )),
        ("case_law", gold_case_laws, lambda s: _match_case_law(
            {"case_name": s.get("case_name", "")}, gold_case_laws
        )),
    ]:
        scores_list = raw_scores.get(f"{category}_rerank_scores", [])
        if scores_list and gold_list:
            pairs = [(s.get("score", 0), match_fn(s)) for s in scores_list]
            metrics[f"{category}_rerank_auc"] = round(rerank_score_auc(pairs), 4)

    # Metadata
    metrics["elapsed_seconds"] = result.get("elapsed_seconds", 0)
    metrics["mode"] = result.get("mode", "unknown")

    return metrics


# ---------------------------------------------------------------------------
# Aggregate Metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(per_query_metrics: list) -> dict:
    """Compute mean and 95% CI for all metrics across queries."""
    import statistics

    # Collect all metric keys (excluding non-numeric)
    skip_keys = {"query_id", "mode"}
    all_keys = set()
    for m in per_query_metrics:
        all_keys.update(k for k in m.keys() if k not in skip_keys)

    agg = {}
    for key in sorted(all_keys):
        values = [m[key] for m in per_query_metrics if key in m and isinstance(m[key], (int, float))]
        if not values:
            continue
        mean_val = statistics.mean(values)
        agg[key] = {"mean": round(mean_val, 4), "n": len(values)}
        if len(values) >= 2:
            stdev = statistics.stdev(values)
            ci_95 = 1.96 * stdev / (len(values) ** 0.5)
            agg[key]["stdev"] = round(stdev, 4)
            agg[key]["ci_95"] = round(ci_95, 4)

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_results_file(results_path: str) -> dict:
    """Evaluate all queries in a batch results file."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    per_query = [evaluate_query(r) for r in results]
    aggregated = aggregate_metrics(per_query)

    return {
        "source_file": results_path,
        "num_queries": len(results),
        "per_query": per_query,
        "aggregated": aggregated,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute retrieval metrics against gold annotations")
    parser.add_argument("--results", required=True, help="Batch results JSON file or directory")
    parser.add_argument("--out", default=None, help="Output file (default: stdout)")
    parser.add_argument("--compare", action="store_true", help="Compare all result files in directory")
    args = parser.parse_args()

    if args.compare and os.path.isdir(args.results):
        # Compare all ablation modes
        comparisons = {}
        for fname in sorted(os.listdir(args.results)):
            if fname.startswith("batch_") and fname.endswith(".json"):
                fpath = os.path.join(args.results, fname)
                mode = fname.split("_")[1]
                logger.info(f"Evaluating {fname} (mode: {mode})...")
                evaluation = evaluate_results_file(fpath)
                comparisons[mode] = evaluation["aggregated"]

        output = json.dumps(comparisons, indent=2)
    else:
        evaluation = evaluate_results_file(args.results)
        output = json.dumps(evaluation, indent=2, default=str)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"Results written to {args.out}")
    else:
        print(output)
