"""
Annotator Agreement — Computes inter-annotator reliability metrics:
- Cohen's Kappa (2 annotators, binary relevance)
- Krippendorff's Alpha (2+ annotators, ordinal/nominal scales)

Used to validate the quality of gold-standard annotations.

Usage:
    python -m eval.annotator_agreement --annotations eval/data/annotations.json

Input format (annotations.json):
    {
        "binary_relevance": {
            "annotator_1": {"q_1_case_1": 1, "q_1_case_2": 0, ...},
            "annotator_2": {"q_1_case_1": 1, "q_1_case_2": 1, ...}
        },
        "opinion_quality": {
            "annotator_1": {
                "q_1": {"correctness": 4, "completeness": 3, "relevance": 5, "actionability": 4, "recency": 3}
            },
            "annotator_2": {
                "q_1": {"correctness": 5, "completeness": 3, "relevance": 4, "actionability": 4, "recency": 4}
            }
        }
    }
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional
from collections import Counter

logger = logging.getLogger("eval.annotator_agreement")


# ---------------------------------------------------------------------------
# Cohen's Kappa (2 annotators, binary)
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list, labels_b: list) -> dict:
    """
    Compute Cohen's Kappa for two annotators.

    Args:
        labels_a: list of binary labels (0/1) from annotator A
        labels_b: list of binary labels (0/1) from annotator B

    Returns:
        {"kappa": float, "po": float, "pe": float, "n": int, "interpretation": str}
    """
    assert len(labels_a) == len(labels_b), "Label lists must be same length"
    n = len(labels_a)
    if n == 0:
        return {"kappa": 0, "po": 0, "pe": 0, "n": 0, "interpretation": "No data"}

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    # Expected agreement (by chance)
    a_pos = sum(labels_a)
    b_pos = sum(labels_b)
    a_neg = n - a_pos
    b_neg = n - b_pos

    pe = (a_pos * b_pos + a_neg * b_neg) / (n * n)

    # Kappa
    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    # Interpretation (Landis & Koch 1977)
    if kappa < 0:
        interp = "Poor (less than chance)"
    elif kappa < 0.21:
        interp = "Slight"
    elif kappa < 0.41:
        interp = "Fair"
    elif kappa < 0.61:
        interp = "Moderate"
    elif kappa < 0.81:
        interp = "Substantial"
    else:
        interp = "Almost perfect"

    return {
        "kappa": round(kappa, 4),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "n": n,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Krippendorff's Alpha (2+ annotators, ordinal/nominal)
# ---------------------------------------------------------------------------

def krippendorffs_alpha(data: dict, level: str = "ordinal") -> dict:
    """
    Compute Krippendorff's Alpha for multiple annotators.

    Args:
        data: {annotator_id: {item_id: value, ...}, ...}
        level: "nominal", "ordinal", or "interval"

    Returns:
        {"alpha": float, "n_items": int, "n_annotators": int, "interpretation": str}
    """
    # Build reliability matrix: rows = items, columns = annotator values
    all_items = set()
    for annotator_data in data.values():
        all_items.update(annotator_data.keys())

    items = sorted(all_items)
    annotators = sorted(data.keys())
    n_annotators = len(annotators)

    if n_annotators < 2:
        return {"alpha": 0, "n_items": len(items), "n_annotators": n_annotators, "interpretation": "Need 2+ annotators"}

    # Build value matrix (items x annotators), None for missing
    matrix = []
    for item in items:
        row = []
        for ann in annotators:
            val = data[ann].get(item)
            row.append(val)
        matrix.append(row)

    # Collect all non-None values
    all_values = []
    for row in matrix:
        for val in row:
            if val is not None:
                all_values.append(val)

    if not all_values:
        return {"alpha": 0, "n_items": 0, "n_annotators": n_annotators, "interpretation": "No data"}

    # Observed disagreement
    def distance(v1, v2, level):
        if level == "nominal":
            return 0.0 if v1 == v2 else 1.0
        elif level == "ordinal":
            return (v1 - v2) ** 2
        else:  # interval
            return (v1 - v2) ** 2

    Do = 0.0  # Observed disagreement
    n_pairs = 0

    for row in matrix:
        vals = [v for v in row if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                Do += distance(vals[i], vals[j], level)
                n_pairs += 1

    if n_pairs == 0:
        return {"alpha": 1.0, "n_items": len(items), "n_annotators": n_annotators, "interpretation": "Perfect (trivial)"}

    Do /= n_pairs

    # Expected disagreement
    n_total = len(all_values)
    De = 0.0
    n_expected_pairs = 0

    for i in range(n_total):
        for j in range(i + 1, n_total):
            De += distance(all_values[i], all_values[j], level)
            n_expected_pairs += 1

    De /= max(n_expected_pairs, 1)

    # Alpha
    if De == 0:
        alpha = 1.0
    else:
        alpha = 1.0 - Do / De

    # Interpretation
    if alpha >= 0.8:
        interp = "Good reliability"
    elif alpha >= 0.667:
        interp = "Acceptable for tentative conclusions"
    else:
        interp = "Low reliability — data should not be trusted"

    return {
        "alpha": round(alpha, 4),
        "Do": round(Do, 4),
        "De": round(De, 4),
        "n_items": len(items),
        "n_annotators": n_annotators,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Compute All Agreement Metrics
# ---------------------------------------------------------------------------

def compute_all_agreement(annotations: dict) -> dict:
    """
    Compute agreement metrics for all annotation types.

    Args:
        annotations: {
            "binary_relevance": {annotator: {item: 0/1, ...}},
            "opinion_quality": {annotator: {query: {dim: score}, ...}}
        }
    """
    results = {}

    # Binary relevance: Cohen's Kappa (pairwise for 2 annotators)
    binary = annotations.get("binary_relevance", {})
    if len(binary) >= 2:
        annotators = sorted(binary.keys())
        # Pairwise Kappa
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a_name = annotators[i]
                b_name = annotators[j]
                # Align items
                common_items = sorted(set(binary[a_name].keys()) & set(binary[b_name].keys()))
                if common_items:
                    labels_a = [binary[a_name][item] for item in common_items]
                    labels_b = [binary[b_name][item] for item in common_items]
                    kappa = cohens_kappa(labels_a, labels_b)
                    results[f"kappa_{a_name}_vs_{b_name}"] = kappa

        # Krippendorff's Alpha (all annotators)
        kalpha = krippendorffs_alpha(binary, level="nominal")
        results["binary_relevance_krippendorff"] = kalpha

    # Opinion quality: Krippendorff's Alpha per dimension
    opinion = annotations.get("opinion_quality", {})
    if len(opinion) >= 2:
        dimensions = ["correctness", "completeness", "relevance", "actionability", "recency"]
        for dim in dimensions:
            dim_data = {}
            for ann, queries in opinion.items():
                dim_data[ann] = {}
                for q_id, scores in queries.items():
                    if dim in scores:
                        dim_data[ann][q_id] = scores[dim]

            if all(dim_data.values()):
                kalpha = krippendorffs_alpha(dim_data, level="ordinal")
                results[f"opinion_{dim}_krippendorff"] = kalpha

    return results


def generate_sample_annotation_template(queries_path: str, output_path: str):
    """Generate an annotation template from a queries file."""
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    template = {
        "instructions": (
            "For each query, annotate the gold-standard answers. "
            "Binary relevance: 1 = relevant, 0 = not relevant. "
            "Opinion quality: score 1-5 on each dimension."
        ),
        "binary_relevance": {
            "annotator_1": {},
            "annotator_2": {},
        },
        "opinion_quality": {
            "annotator_1": {},
            "annotator_2": {},
        },
    }

    for q in queries:
        q_id = q.get("id", "")
        template["opinion_quality"]["annotator_1"][q_id] = {
            "correctness": 0, "completeness": 0, "relevance": 0, "actionability": 0, "recency": 0
        }
        template["opinion_quality"]["annotator_2"][q_id] = {
            "correctness": 0, "completeness": 0, "relevance": 0, "actionability": 0, "recency": 0
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    logger.info(f"Annotation template saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Inter-annotator agreement calculator")
    parser.add_argument("--annotations", help="Annotations JSON file")
    parser.add_argument("--generate-template", help="Queries JSON to generate template from")
    parser.add_argument("--out", help="Output file")
    args = parser.parse_args()

    if args.generate_template:
        out = args.out or "eval/data/annotation_template.json"
        generate_sample_annotation_template(args.generate_template, out)
    elif args.annotations:
        with open(args.annotations, encoding="utf-8") as f:
            annotations = json.load(f)
        results = compute_all_agreement(annotations)
        output = json.dumps(results, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)
    else:
        parser.print_help()
