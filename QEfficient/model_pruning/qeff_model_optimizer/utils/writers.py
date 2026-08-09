"""Opt-in writers that reproduce the legacy CSV/PNG artifacts.

The typed :class:`WeakLayerReport` is the source of truth for analysis output.
These writers exist so downstream pipelines that already read legacy files keep
working when callers opt in via ``analyze_weak_layers(..., output_dir=...)``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


def write_legacy_csv(
    output_dir: Path,
    dataset: str,
    metric: str,
    per_layer_stats: list[Mapping[str, float]],
) -> Path:
    """Write one legacy-format CSV for a single dataset/metric pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer_contributions_{dataset}_{metric}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer_index", "avg_delta", "std_delta", "min_delta", "max_delta"])
        for idx, stats in enumerate(per_layer_stats, start=1):
            writer.writerow([idx, stats["avg"], stats["std"], stats["min"], stats["max"]])
    logger.info("[analyze] wrote CSV artifact %s", path)
    return path


def write_legacy_png(
    output_dir: Path,
    dataset: str,
    metric: str,
    per_layer_stats: list[Mapping[str, float]],
) -> Path | None:
    """Write one legacy-format line plot if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(
            "[analyze] skipping PNG artifact for dataset=%s metric=%s because matplotlib is unavailable: %s",
            dataset,
            metric,
            exc,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer_contributions_{dataset}_{metric}.png"

    x = list(range(1, len(per_layer_stats) + 1))
    y = [s["avg"] for s in per_layer_stats]
    yerr = [s["std"] for s in per_layer_stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=2,
        markersize=5,
        capsize=4,
        capthick=1.5,
        alpha=0.85,
    )
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel(f"Average {metric.capitalize()} Delta", fontsize=12)
    ax.set_title(f"{dataset.upper()}: Per-layer {metric} contribution", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[analyze] wrote PNG artifact %s", path)
    return path


def write_combined_png(
    output_dir: Path,
    metric: str,
    per_dataset_stats: dict[str, list[Mapping[str, float]]],
    model_id: str | None = None,
) -> Path | None:
    """Overlay all datasets in one chart, or write the single-dataset chart."""
    if len(per_dataset_stats) < 2:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        logger.warning(
            "[analyze] skipping combined PNG artifact for metric=%s because matplotlib/numpy is unavailable: %s",
            metric,
            exc,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"multi_dataset_{metric}.png"

    datasets = list(per_dataset_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(datasets), 2)))

    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, dataset in enumerate(datasets):
        stats = per_dataset_stats[dataset]
        x = list(range(1, len(stats) + 1))
        y = [s["avg"] for s in stats]
        yerr = [s["std"] for s in stats]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=4,
            capthick=1.5,
            label=dataset.upper(),
            color=colors[idx % len(colors)],
            alpha=0.8,
        )

    metric_label = "Cosine Distance" if metric == "cosine" else "L2 Distance"
    title_parts = ["Multi-Dataset Layer Contribution Comparison"]
    if model_id:
        title_parts.append(model_id)
    title_parts.append(f"Metric: {metric_label}")
    ax.set_title("\n".join(title_parts), fontsize=14, fontweight="bold")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel(f"Average {metric_label}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=max(1, len(datasets) // 8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[analyze] wrote combined PNG artifact %s", path)
    return path
