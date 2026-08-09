"""Reusable chart generators for NAS evaluation results.

These helpers produce the same kinds of charts used by the scripts in
``scripts/`` but in a packaged, reusable form. Each function:

- Imports matplotlib lazily (so the rest of ``nas.evaluation`` works without
  matplotlib installed).
- Uses the ``Agg`` backend (no display required — works headless).
- Returns the output path for easy chaining / logging.

Functions:
- :func:`chart_weak_layers` — per-layer contribution bar chart
- :func:`chart_head_importance_heatmap` — 2D heatmap of head scores
- :func:`chart_perplexity_comparison` — PPL and delta-% bars side-by-side
- :func:`chart_per_dataset_perplexity` — grouped bars per dataset per plan
- :func:`chart_qaic_performance` — TTFT / decode / E2E bars across plans
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# --- Lazy matplotlib loader ---

def _get_plt():
    """Import matplotlib lazily; raises ImportError with a helpful message."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for nas.evaluation.charts. "
            "Install with: pip install matplotlib"
        ) from e


# --- Chart functions ---

def chart_weak_layers(
    weak_report,
    output_path: str | Path,
    *,
    title_prefix: str = "",
    highlight_weakest_n: int = 5,
) -> Path:
    """Bar chart of per-layer contribution scores, highlighting weakest N.

    Accepts a ``WeakLayerReport`` (or its ``.to_dict()`` payload).
    Weakest ``highlight_weakest_n`` layers are colored red (skip candidates);
    the rest are blue.
    """
    plt = _get_plt()
    output_path = Path(output_path)

    # Support both WeakLayerReport objects and raw dicts.
    if hasattr(weak_report, "ranked_layers"):
        ranked = weak_report.ranked_layers
        get_layer = lambda r: r.layer
        get_score = lambda r: r.aggregate_score
    else:
        ranked = weak_report["ranked_layers"]
        get_layer = lambda r: r["layer"]
        get_score = lambda r: r["aggregate_score"]

    num_layers = len(ranked)
    scores = [0.0] * num_layers
    for r in ranked:
        scores[get_layer(r)] = get_score(r)

    # Find threshold for the N weakest layers.
    threshold = sorted(scores)[highlight_weakest_n - 1] if scores else 0.0
    colors = [
        "#d32f2f" if scores[i] <= threshold else "#1976d2"
        for i in range(num_layers)
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(num_layers), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Contribution Score")
    ax.set_title(
        f"Per-Layer Contribution{' - ' + title_prefix if title_prefix else ''}\n"
        f"(red = weakest {highlight_weakest_n} layers, best skip candidates)"
    )
    ax.set_xticks(range(0, num_layers, 2))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def chart_head_importance_heatmap(
    head_report,
    output_path: str | Path,
    *,
    title_prefix: str = "",
) -> Path:
    """2D heatmap: layers (rows) x heads (columns) with importance score.

    Accepts a ``HeadImportanceReport`` or its ``.to_dict()`` payload.
    """
    import numpy as np
    plt = _get_plt()
    output_path = Path(output_path)

    if hasattr(head_report, "per_layer_scores"):
        num_layers = head_report.num_layers
        num_heads = head_report.num_heads
        per_layer = {k: v for k, v in head_report.per_layer_scores.items()}
    else:
        num_layers = head_report["num_layers"]
        num_heads = head_report["num_heads"]
        per_layer = {int(k): v for k, v in head_report["per_layer_scores"].items()}

    grid = [[0.0] * num_heads for _ in range(num_layers)]
    for layer, heads in per_layer.items():
        for head_idx, score in heads:
            if layer < num_layers and head_idx < num_heads:
                grid[layer][head_idx] = score

    arr = np.array(grid)
    fig, ax = plt.subplots(figsize=(max(12, num_heads * 0.5), max(6, num_layers * 0.25)))
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(
        f"Head Importance Heatmap{' - ' + title_prefix if title_prefix else ''}\n"
        f"(darker = more important)"
    )
    fig.colorbar(im, ax=ax, label="Mean L2 Norm")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def chart_perplexity_comparison(
    plan_results: list[Any],
    output_path: str | Path,
    *,
    title_prefix: str = "",
    baseline_name: str = "baseline",
) -> Path:
    """Side-by-side: absolute PPL per plan, and % degradation vs baseline.

    Accepts ``list[PlanResult]`` or ``list[dict]`` (with overall_perplexity
    and plan_name keys). Plans that errored are skipped.
    """
    plt = _get_plt()
    output_path = Path(output_path)

    # Normalise to dicts for consistent access.
    items = []
    for r in plan_results:
        if hasattr(r, "to_dict"):
            d = r.to_dict()
        else:
            d = dict(r)
        if d.get("error"):
            continue
        ppl = d.get("overall_perplexity")
        if ppl is None or ppl == float("inf") or ppl > 1e5:
            continue
        items.append({"plan_name": d["plan_name"], "perplexity": ppl})

    if not items:
        raise ValueError("No valid plan results to chart")

    # Find baseline for delta computation.
    baseline_ppl = next(
        (it["perplexity"] for it in items if it["plan_name"] == baseline_name),
        items[0]["perplexity"],
    )

    names = [it["plan_name"] for it in items]
    ppls = [it["perplexity"] for it in items]
    deltas = [(p - baseline_ppl) / baseline_ppl * 100 for p in ppls]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: absolute PPL.
    colors_abs = [
        "#4caf50" if n == baseline_name else "#ff9800"
        for n in names
    ]
    ax1.barh(range(len(names)), ppls, color=colors_abs, edgecolor="white")
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Perplexity (lower=better)")
    ax1.set_title("Absolute Perplexity")
    ax1.invert_yaxis()

    # Right panel: % delta vs baseline, with budget lines.
    colors_delta = [
        "#4caf50" if d <= 5 else
        "#ff9800" if d <= 15 else
        "#d32f2f"
        for d in deltas
    ]
    ax2.barh(range(len(names)), deltas, color=colors_delta, edgecolor="white")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("PPL Increase (%)")
    ax2.set_title("Quality Degradation vs Baseline")
    ax2.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="5% budget")
    ax2.axvline(x=10, color="gray", linestyle=":", alpha=0.5, label="10% budget")
    ax2.legend(fontsize=7)
    ax2.invert_yaxis()

    fig.suptitle(
        f"Plan Comparison{' - ' + title_prefix if title_prefix else ''}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def chart_per_dataset_perplexity(
    plan_results: list[Any],
    output_path: str | Path,
    *,
    title_prefix: str = "",
    max_ppl: float = 100.0,
) -> Path:
    """Grouped horizontal bars: per-dataset PPL for each plan.

    Each plan gets a cluster of bars, one per dataset. Plans exceeding
    ``max_ppl`` are filtered out to keep the chart readable.
    """
    plt = _get_plt()
    output_path = Path(output_path)

    # Collect plan results as dicts with per-dataset breakdowns.
    items = []
    for r in plan_results:
        if hasattr(r, "perplexity_report") and r.perplexity_report:
            per_ds = {
                name: dp.perplexity
                for name, dp in r.perplexity_report.per_dataset.items()
            }
            overall = r.perplexity_report.overall_perplexity
            name = r.plan_name
            errored = r.error is not None
        else:
            d = dict(r)
            if d.get("error"):
                continue
            pr = d.get("perplexity_report") or {}
            per_ds = {
                k: v.get("perplexity", 0)
                for k, v in pr.get("per_dataset", {}).items()
            }
            overall = pr.get("overall_perplexity", d.get("overall_perplexity", 0))
            name = d["plan_name"]
            errored = False

        if errored or overall > max_ppl:
            continue
        items.append({"name": name, "per_ds": per_ds, "overall": overall})

    if not items:
        raise ValueError("No valid plan results to chart")

    # Union of all dataset names seen.
    dataset_names: list[str] = []
    for it in items:
        for ds in it["per_ds"]:
            if ds not in dataset_names:
                dataset_names.append(ds)

    plan_names = [it["name"] for it in items]
    y_pos = range(len(plan_names))
    width = 0.8 / len(dataset_names) if dataset_names else 0.8

    colors = ["#1976d2", "#388e3c", "#f57c00", "#7b1fa2", "#c62828", "#0097a7",
              "#5d4037", "#455a64"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(plan_names) * 0.5)))
    for i, ds in enumerate(dataset_names):
        ppls = [it["per_ds"].get(ds, 0) for it in items]
        offsets = [y + (i - len(dataset_names) / 2) * width for y in y_pos]
        ax.barh(offsets, ppls, width, label=ds, color=colors[i % len(colors)])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plan_names, fontsize=9)
    ax.set_xlabel("Perplexity (lower=better)")
    ax.set_title(f"Per-Dataset Perplexity{' - ' + title_prefix if title_prefix else ''}")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def chart_qaic_performance(
    qaic_results: list[Any],
    output_path: str | Path,
    *,
    title_prefix: str = "",
) -> Path:
    """Three-panel chart: TTFT, Decode Throughput, E2E Latency per plan.

    Accepts ``list[QAICRunResult]`` or their ``to_dict()`` payloads. Errored
    runs are skipped.
    """
    plt = _get_plt()
    output_path = Path(output_path)

    # Normalise.
    items = []
    for r in qaic_results:
        if hasattr(r, "to_dict"):
            d = r.to_dict()
        else:
            d = dict(r)
        if d.get("error"):
            continue
        items.append(d)

    if not items:
        raise ValueError("No valid QAIC results to chart")

    names = [r["plan_name"] for r in items]
    ttfts = [r["avg_stats"].get("ttft", 0) for r in items]
    decodes = [r["avg_stats"].get("decode_tps", 0) for r in items]
    e2es = [r["avg_stats"].get("e2e", 0) for r in items]
    colors = [
        "#4caf50" if "baseline" in n else "#2196f3"
        for n in names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, values, ylabel, title in [
        (axes[0], ttfts, "Seconds", "TTFT (lower=better)"),
        (axes[1], decodes, "Tokens/sec", "Decode (higher=better)"),
        (axes[2], e2es, "Seconds", "E2E Latency (lower=better)"),
    ]:
        ax.bar(range(len(names)), values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, fontsize=8, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(
        f"QAIC Performance{' - ' + title_prefix if title_prefix else ''}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = [
    "chart_weak_layers",
    "chart_head_importance_heatmap",
    "chart_perplexity_comparison",
    "chart_per_dataset_perplexity",
    "chart_qaic_performance",
]
