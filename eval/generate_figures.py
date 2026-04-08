"""
Generate Figures — Creates publication-quality charts for the research paper.

Generates:
1. Ablation bar chart (retrieval metrics across pipeline stages)
2. Threshold sweep P-R curve
3. Score distribution histograms
4. Safety category breakdown
5. Opinion quality radar chart
6. Source distribution pie chart

Usage:
    python -m eval.generate_figures --results-dir eval/results/ --out eval/figures/

Dependencies: matplotlib, numpy (pip install matplotlib numpy)
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("eval.generate_figures")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")


# Publication style defaults
def _setup_style():
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# 1. Ablation Bar Chart
# ---------------------------------------------------------------------------

def plot_ablation_bars(comparison_data: dict, output_path: str):
    """
    Bar chart comparing retrieval metrics across ablation modes.
    Input: {mode: {metric: {mean: float, ...}}}
    """
    _setup_style()
    modes = list(comparison_data.keys())
    metrics = ["bare_act_precision@5", "bare_act_recall@5", "case_law_precision@5", "case_law_recall@5"]
    metric_labels = ["BA Prec@5", "BA Recall@5", "CL Prec@5", "CL Recall@5"]

    x = np.arange(len(metrics))
    width = 0.15
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, mode in enumerate(modes):
        values = [comparison_data[mode].get(m, {}).get("mean", 0) for m in metrics]
        errors = [comparison_data[mode].get(m, {}).get("ci_95", 0) for m in metrics]
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mode.replace("_", " ").title(),
                       color=colors[i % len(colors)], yerr=errors, capsize=3)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality: Ablation Study")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Ablation chart saved to {output_path}")


# ---------------------------------------------------------------------------
# 2. Threshold Sweep P-R Curve
# ---------------------------------------------------------------------------

def plot_threshold_curve(sweep_data: dict, output_path: str):
    """
    Precision-Recall-F1 curves as a function of rerank score threshold.
    Input: {"aggregated": {threshold_str: {mean_precision, mean_recall, mean_f1}}}
    """
    _setup_style()
    agg = sweep_data.get("aggregated", {})
    thresholds = sorted([float(t) for t in agg.keys()])
    precisions = [agg[str(t)]["mean_precision"] for t in thresholds]
    recalls = [agg[str(t)]["mean_recall"] for t in thresholds]
    f1s = [agg[str(t)]["mean_f1"] for t in thresholds]

    optimal = sweep_data.get("optimal_threshold", 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, "o-", color="#2E86AB", label="Precision", linewidth=2, markersize=4)
    ax.plot(thresholds, recalls, "s-", color="#A23B72", label="Recall", linewidth=2, markersize=4)
    ax.plot(thresholds, f1s, "^-", color="#F18F01", label="F1", linewidth=2, markersize=4)
    ax.axvline(x=optimal, color="#C73E1D", linestyle="--", alpha=0.7, label=f"Optimal ({optimal})")

    ax.set_xlabel("Cross-Encoder Rerank Score Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision-Recall-F1 Trade-off")
    ax.legend(loc="center left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Threshold curve saved to {output_path}")


# ---------------------------------------------------------------------------
# 3. Score Distribution Histograms
# ---------------------------------------------------------------------------

def plot_score_distributions(sweep_data: dict, output_path: str):
    """
    Histogram of rerank scores, colored by relevant vs. not relevant.
    Input: sweep_data with per_query raw_score_distribution
    """
    _setup_style()

    all_ba_scores = []
    all_cl_scores = []
    for q in sweep_data.get("per_query", []):
        dist = q.get("raw_score_distribution", {})
        all_ba_scores.extend(dist.get("bare_act_scores", []))
        all_cl_scores.extend(dist.get("case_law_scores", []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if all_ba_scores:
        ax1.hist(all_ba_scores, bins=30, color="#2E86AB", alpha=0.8, edgecolor="white")
        ax1.axvline(x=0.55, color="#C73E1D", linestyle="--", label="Current threshold (0.55)")
        ax1.set_xlabel("Cross-Encoder Rerank Score")
        ax1.set_ylabel("Count")
        ax1.set_title("Bare Act Score Distribution")
        ax1.legend()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

    if all_cl_scores:
        ax2.hist(all_cl_scores, bins=30, color="#A23B72", alpha=0.8, edgecolor="white")
        ax2.axvline(x=0.55, color="#C73E1D", linestyle="--", label="Current threshold (0.55)")
        ax2.set_xlabel("Cross-Encoder Rerank Score")
        ax2.set_ylabel("Count")
        ax2.set_title("Case Law Score Distribution")
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Score distributions saved to {output_path}")


# ---------------------------------------------------------------------------
# 4. Safety Category Breakdown
# ---------------------------------------------------------------------------

def plot_safety_breakdown(safety_report: dict, output_path: str):
    """Stacked bar chart showing safety test results by category."""
    _setup_style()
    categories = safety_report.get("by_category", {})
    if not categories:
        logger.warning("No safety data to plot")
        return

    cats = sorted(categories.keys())
    refused = [categories[c].get("refused", 0) for c in cats]
    complied = [categories[c].get("complied", 0) for c in cats]
    partial = [categories[c].get("partial", 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, refused, width, label="Correctly Refused", color="#2E86AB")
    ax.bar(x, partial, width, bottom=refused, label="Partial", color="#F18F01")
    ax.bar(x, complied, width, bottom=[r + p for r, p in zip(refused, partial)],
           label="Incorrectly Complied", color="#C73E1D")

    ax.set_xlabel("Safety Category")
    ax.set_ylabel("Count")
    ax.set_title("Safety Evaluation: Results by Category")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Safety breakdown saved to {output_path}")


# ---------------------------------------------------------------------------
# 5. Opinion Quality Radar Chart
# ---------------------------------------------------------------------------

def plot_opinion_radar(mean_scores: dict, output_path: str):
    """
    Radar/spider chart for opinion quality dimensions.
    Input: {"correctness": 3.8, "completeness": 3.2, "relevance": 4.1, "actionability": 3.5, "recency": 3.9}
    """
    _setup_style()
    dims = ["correctness", "completeness", "relevance", "actionability", "recency"]
    labels = ["Legal\nCorrectness", "Completeness", "Relevance", "Actionability", "Recency"]
    values = [mean_scores.get(d, 0) for d in dims]
    values += values[:1]  # Close the polygon

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#2E86AB", alpha=0.25)
    ax.plot(angles, values, "o-", color="#2E86AB", linewidth=2, markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_title("Legal Opinion Quality (Mean Advocate Scores)", pad=20)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Opinion radar saved to {output_path}")


# ---------------------------------------------------------------------------
# 6. Source Distribution Pie Chart
# ---------------------------------------------------------------------------

def plot_source_distribution(results: list, output_path: str):
    """Pie chart showing where final results come from (LOCAL_DB, OFFICIAL, LEGAL_PORTAL, etc.)."""
    _setup_style()
    source_counts = {}
    for r in results:
        final = r.get("final_response") or {}
        for item_list in ("bare_act_sections", "case_laws"):
            for item in final.get(item_list, []):
                tag = item.get("source_tag", "LOCAL_DB")
                source_counts[tag] = source_counts.get(tag, 0) + 1

    if not source_counts:
        logger.warning("No source data to plot")
        return

    labels = list(source_counts.keys())
    sizes = list(source_counts.values())
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                                       autopct="%1.1f%%", startangle=140, textprops={"fontsize": 10})
    ax.set_title("Source Distribution of Retrieved Materials")

    plt.savefig(output_path)
    plt.close()
    logger.info(f"Source distribution saved to {output_path}")


# ---------------------------------------------------------------------------
# Main: Generate All Figures
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all publication figures from evaluation results."""
    if not HAS_MPL:
        logger.error("matplotlib is required. Install: pip install matplotlib --break-system-packages")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ablation comparison (if multiple modes exist)
    comparison = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("batch_") and fname.endswith(".json"):
            mode = fname.split("_")[1]
            fpath = os.path.join(results_dir, fname)
            try:
                from eval.retrieval_metrics import evaluate_results_file
                evaluation = evaluate_results_file(fpath)
                comparison[mode] = evaluation.get("aggregated", {})
            except Exception as e:
                logger.warning(f"Could not evaluate {fname}: {e}")

    if len(comparison) >= 2:
        plot_ablation_bars(comparison, os.path.join(output_dir, "fig_ablation.png"))

    # 2. Threshold sweep
    threshold_dir = os.path.join(results_dir, "threshold")
    threshold_file = os.path.join(threshold_dir, "threshold_sweep.json")
    if os.path.exists(threshold_file):
        with open(threshold_file, encoding="utf-8") as f:
            sweep_data = json.load(f)
        plot_threshold_curve(sweep_data, os.path.join(output_dir, "fig_threshold_curve.png"))
        plot_score_distributions(sweep_data, os.path.join(output_dir, "fig_score_distributions.png"))

    # 3. Safety report
    safety_dir = os.path.join(results_dir, "safety")
    if os.path.isdir(safety_dir):
        for fname in os.listdir(safety_dir):
            if fname.startswith("safety_report") and fname.endswith(".json"):
                with open(os.path.join(safety_dir, fname), encoding="utf-8") as f:
                    safety_data = json.load(f)
                plot_safety_breakdown(safety_data, os.path.join(output_dir, "fig_safety_breakdown.png"))
                break

    # 4. Source distribution (from any batch result)
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("batch_full") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
                batch_data = json.load(f)
            plot_source_distribution(batch_data, os.path.join(output_dir, "fig_source_distribution.png"))
            break

    logger.info(f"All figures generated in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate publication-quality figures")
    parser.add_argument("--results-dir", default="eval/results/", help="Directory with evaluation results")
    parser.add_argument("--out", default="eval/figures/", help="Output directory for figures")
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.out)
