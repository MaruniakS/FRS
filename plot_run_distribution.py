#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev


CLASSES = ("0", "1", "2")
METRICS = ("precision", "recall", "f1-score")
CLASS_LABELS = {
    "0": "Direct",
    "1": "Indirect",
    "2": "Outage",
}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1-score",
}


def load_reports(input_dir):
    reports = []
    for path in sorted(input_dir.glob("run_*.json"), key=run_number):
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    if not reports:
        raise ValueError(f"No run_*.json reports found in {input_dir}")

    return reports


def run_number(path):
    return int(path.stem[len("run_") :])


def summarize(reports):
    rows = []
    for class_name in CLASSES:
        for metric in METRICS:
            values = [report[class_name][metric] for report in reports]
            rows.append(
                {
                    "class": class_name,
                    "metric": metric,
                    "mean": mean(values),
                    "median": median(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                }
            )
    return rows


def metric_values(reports):
    rows = []
    for class_name in CLASSES:
        for metric in METRICS:
            rows.append(
                {
                    "class": class_name,
                    "metric": metric,
                    "values": [report[class_name][metric] for report in reports],
                }
            )
    return rows


def import_plotting():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib/numpy are required to create the PNG. "
            "Install project dependencies first, for example with `pipenv install`."
        ) from exc

    return plt, np


def plot_summary(rows, output_path, title):
    plt, np = import_plotting()

    labels = [METRIC_LABELS[row["metric"]] for row in rows]
    means = [row["mean"] for row in rows]
    medians = [row["median"] for row in rows]
    stds = [row["std"] for row in rows]

    class_colors = {
        "0": "#4C78A8",
        "1": "#F58518",
        "2": "#54A24B",
    }
    colors = [class_colors[row["class"]] for row in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.9,
        label="Mean +/- std",
    )
    ax.scatter(
        x,
        medians,
        color="#111111",
        s=28,
        marker="D",
        zorder=3,
        label="Median",
    )

    for i, value in enumerate(means):
        ax.text(
            i,
            min(value + stds[i] + 0.025, 1.06),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if title:
        ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="#BBBBBB", linewidth=0.8, alpha=0.8)

    for center, class_name in zip((1, 4, 7), CLASSES):
        ax.text(
            center,
            -0.08,
            CLASS_LABELS[class_name],
            ha="center",
            va="top",
            fontsize=13,
            transform=ax.get_xaxis_transform(),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(rows, output_path, title):
    plt, np = import_plotting()

    labels = [METRIC_LABELS[row["metric"]] for row in rows]
    values = [row["values"] for row in rows]
    class_colors = {
        "0": "#4C78A8",
        "1": "#F58518",
        "2": "#54A24B",
    }
    colors = [class_colors[row["class"]] for row in rows]

    rng = np.random.default_rng(7)
    x = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    box = ax.boxplot(
        values,
        positions=x,
        widths=0.62,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111111",
            "markeredgecolor": "#111111",
            "markersize": 4,
        },
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": "#333333",
            "markersize": 4,
            "alpha": 0.8,
        },
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
        patch.set_alpha(0.38)

    for i, vals in enumerate(values, start=1):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=15,
            color=colors[i - 1],
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.72,
            zorder=3,
        )

    if title:
        ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for boundary in (3.5, 6.5):
        ax.axvline(boundary, color="#BBBBBB", linewidth=0.8, alpha=0.8)

    for center, class_name in zip((2, 5, 8), CLASSES):
        ax.text(
            center,
            -0.08,
            CLASS_LABELS[class_name],
            ha="center",
            va="top",
            fontsize=13,
            transform=ax.get_xaxis_transform(),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot metric distributions for classification run JSON reports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/classification_runs/baseline"),
        help="Directory containing run_*.json reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/classification_runs/baseline/baseline_distribution_boxplot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Baseline run distribution: test_unseen summary",
        help="Chart title.",
    )
    parser.add_argument(
        "--kind",
        choices=("bar", "box"),
        default="box",
        help="Chart type to create.",
    )
    args = parser.parse_args()

    reports = load_reports(args.input_dir)
    if args.kind == "bar":
        plot_summary(summarize(reports), args.output, args.title)
    else:
        plot_boxplot(metric_values(reports), args.output, args.title)
    print(f"Saved chart to {args.output}")


if __name__ == "__main__":
    main()
