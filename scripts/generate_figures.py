"""Generate publication-ready figures from a run artifact.

Reads a structured JSON artifact produced by scripts/run_pipeline.py
(config + git commit + full results) and renders a fixed set of static
figures used in the paper: forecast accuracy comparison, actual-vs-predicted
per region, global SHAP feature importance, faithfulness precision/recall
per region, and stockout risk tier distribution.

Figures are derived data, not source data -- they are not committed to git
and are always regenerated from a run artifact:
    python -m scripts.generate_figures --run results/run_<timestamp>.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DPI = 300
FONT_FAMILY = "DejaVu Sans"
COLOR_BASELINE = "#4C72B0"
COLOR_AUGMENTED = "#DD8452"
COLOR_NEUTRAL = "#55A868"
COLOR_RISK = {"low": "#55A868", "medium": "#DD8452", "high": "#C44E52"}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 100,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, output_dir: str, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_mape_comparison(result: dict, output_dir: str) -> list[str]:
    baseline_mape = result["baseline_mape"]
    augmented_mape = result["augmented_mape"]
    p_value = result["significance"]["p_value"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(
        ["Baseline\n(sales lags + seasonality)", "Health-augmented\n(+ ILI features)"],
        [baseline_mape, augmented_mape],
        color=[COLOR_BASELINE, COLOR_AUGMENTED],
        width=0.55,
    )
    for bar, value in zip(bars, [baseline_mape, augmented_mape]):
        ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast Accuracy: Baseline vs. Health-Augmented Model")
    significance_label = "significant" if p_value < 0.05 else "not significant"
    ax.text(
        0.5,
        -0.22,
        f"Wilcoxon signed-rank test, p = {p_value:.3f} ({significance_label} at α = 0.05)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    ax.set_ylim(0, max(baseline_mape, augmented_mape) * 1.25)
    fig.tight_layout()
    return _save(fig, output_dir, "mape_comparison")


def plot_actual_vs_predicted(result: dict, output_dir: str) -> list[str]:
    predictions = result["predictions"]
    by_region = defaultdict(list)
    for row in predictions:
        by_region[row["region"]].append(row)
    regions = sorted(by_region)

    fig, axes = plt.subplots(len(regions), 1, figsize=(5.2, 1.9 * len(regions)), sharey=False)
    if len(regions) == 1:
        axes = [axes]

    lines = None
    for ax, region in zip(axes, regions):
        rows = sorted(by_region[region], key=lambda r: (r["year"], r["week"]))
        x = np.arange(len(rows))
        actual = [r["actual_sales"] for r in rows]
        baseline_pred = [r["baseline_pred"] for r in rows]
        augmented_pred = [r["augmented_pred"] for r in rows]

        ax.plot(x, actual, label="Actual", color="black", marker="o", linewidth=2, zorder=3)
        ax.plot(
            x, baseline_pred, label="Baseline", color=COLOR_BASELINE, marker="s", linestyle="--"
        )
        ax.plot(
            x,
            augmented_pred,
            label="Health-augmented",
            color=COLOR_AUGMENTED,
            marker="^",
            linestyle="--",
        )
        ax.set_title(f"HHS Region {region}")
        ax.set_xlabel("Test-period week index")
        ax.set_ylabel("Weekly unit sales")
        lines = ax.get_lines()

    fig.suptitle(
        "Actual vs. Predicted Demand on the Held-Out Test Window", fontweight="bold", y=0.995
    )
    fig.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, output_dir, "actual_vs_predicted")


def plot_shap_importance(result: dict, output_dir: str) -> list[str]:
    importance = result["shap_importance"]
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [name for name, _ in items]
    values = [value for _, value in items]

    is_health_feature = [name.startswith("ili_rate") for name in names]
    colors = [COLOR_AUGMENTED if is_health else COLOR_BASELINE for is_health in is_health_feature]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(names))))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (Health-Augmented Model)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE, label="Sales / seasonality feature"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AUGMENTED, label="ILI (health) feature"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, "shap_importance")


def plot_faithfulness_by_region(result: dict, output_dir: str) -> list[str]:
    faithfulness = result["faithfulness"]
    by_region = defaultdict(lambda: {"precision": [], "recall": []})
    for row in faithfulness:
        by_region[row["region"]]["precision"].append(row["precision"])
        by_region[row["region"]]["recall"].append(row["recall"])

    regions = sorted(by_region)
    precision_means = [np.mean(by_region[r]["precision"]) for r in regions]
    recall_means = [np.mean(by_region[r]["recall"]) for r in regions]

    x = np.arange(len(regions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, precision_means, width, label="Precision", color=COLOR_BASELINE)
    ax.bar(x + width / 2, recall_means, width, label="Recall", color=COLOR_AUGMENTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Region {r}" for r in regions])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("LLM Advisory Faithfulness vs. SHAP Ground Truth")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_dir, "faithfulness_by_region")


def plot_risk_tier_distribution(result: dict, output_dir: str) -> list[str]:
    tiers = [row["risk_tier"] for row in result["faithfulness"]]
    counts = Counter(tiers)
    ordered_tiers = [t for t in ("low", "medium", "high") if t in counts]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(
        ordered_tiers,
        [counts[t] for t in ordered_tiers],
        color=[COLOR_RISK[t] for t in ordered_tiers],
    )
    ax.set_ylabel("Number of test-window forecasts")
    ax.set_title("Stockout Risk Tier Distribution")
    ax.set_ylim(0, max(counts.values()) * 1.25)
    for i, tier in enumerate(ordered_tiers):
        ax.annotate(
            str(counts[tier]),
            xy=(i, counts[tier]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    return _save(fig, output_dir, "risk_tier_distribution")


FIGURES = [
    plot_mape_comparison,
    plot_actual_vs_predicted,
    plot_shap_importance,
    plot_faithfulness_by_region,
    plot_risk_tier_distribution,
]


def generate_all_figures(artifact: dict, output_dir: str) -> list[str]:
    _apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    result = artifact["result"]

    written = []
    for plot_fn in FIGURES:
        written.extend(plot_fn(result, output_dir))
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures from a run artifact"
    )
    parser.add_argument("--run", required=True, help="Path to a results/run_*.json artifact")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    with open(args.run) as f:
        artifact = json.load(f)

    written = generate_all_figures(artifact, args.output_dir)
    print(f"Wrote {len(written)} figure files to {args.output_dir}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
