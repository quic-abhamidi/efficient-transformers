#!/usr/bin/env python3
"""
Generate comparison CSV + visualization for baseline vs skip vs skip+comp runs.

Expected directory layout:
  <run_dir>/baseline/results.csv
  <run_dir>/skip_only/results.csv
  <run_dir>/skip_compensated/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build comparison artifacts for skip-layer compensation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Run directory with baseline/skip outputs")
    parser.add_argument("--model-name", default=None, help="Model label for chart/summary title")
    parser.add_argument("--skip-layers", default=None, help="Optional text description of skipped layers")
    parser.add_argument("--output-csv", default=None, help="Path to comparison_results.csv")
    parser.add_argument("--output-chart", default=None, help="Path to comparison_visualization.png")
    parser.add_argument("--output-summary", default=None, help="Path to experiment_summary.txt")
    return parser.parse_args()


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    df = pd.read_csv(path)
    if "benchmark" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Invalid results format in {path}: expected benchmark,score columns")
    return df[["benchmark", "score"]].copy()


def build_table(base: pd.DataFrame, skip: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(skip, on="benchmark", suffixes=("_baseline", "_skip"))
    merged = merged.merge(comp, on="benchmark")
    merged = merged.rename(columns={"score": "score_comp"})

    merged["baseline"] = merged["score_baseline"].astype(float)
    merged["skip_only"] = merged["score_skip"].astype(float)
    merged["skip_compensated"] = merged["score_comp"].astype(float)
    merged["skip_degradation"] = merged["baseline"] - merged["skip_only"]
    merged["compensation_recovery"] = merged["skip_compensated"] - merged["skip_only"]

    merged["recovery_percentage"] = np.where(
        merged["skip_degradation"].abs() > 1e-9,
        (merged["compensation_recovery"] / merged["skip_degradation"]) * 100.0,
        0.0,
    )
    merged["accuracy delta with mainline"] = np.where(
        merged["baseline"].abs() > 1e-12,
        ((merged["skip_only"] - merged["baseline"]) / merged["baseline"]) * 100.0,
        0.0,
    )
    merged["Accuracy recover with skip + compensation"] = np.where(
        merged["skip_only"].abs() > 1e-12,
        ((merged["skip_compensated"] - merged["skip_only"]) / merged["skip_only"]) * 100.0,
        0.0,
    )

    keep = [
        "benchmark",
        "baseline",
        "skip_only",
        "skip_compensated",
        "skip_degradation",
        "compensation_recovery",
        "recovery_percentage",
        "accuracy delta with mainline",
        "Accuracy recover with skip + compensation",
    ]
    return merged[keep]


def format_benchmark_names(raw_names: List[str]) -> List[str]:
    out = []
    for name in raw_names:
        out.append(name.replace("_", " ").upper())
    return out


def add_value_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}", dy: float = 0.006) -> None:
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + (bar.get_width() / 2.0)
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=8)


def draw_chart(table: pd.DataFrame, out_file: Path, model_name: str, skip_layers: str) -> None:
    names = table["benchmark"].tolist()
    labels = format_benchmark_names(names)
    x = np.arange(len(names))

    baseline = table["baseline"].to_numpy()
    skip = table["skip_only"].to_numpy()
    comp = table["skip_compensated"].to_numpy()

    drop = np.clip(table["skip_degradation"].to_numpy(), 0.0, None)
    recovery_raw = table["compensation_recovery"].to_numpy()
    recovered = np.clip(recovery_raw, 0.0, None)
    unrecovered = np.clip(drop - recovered, 0.0, None)
    rec_pct = np.where(drop > 1e-9, (recovered / drop) * 100.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2.0, 1.5]})
    fig.suptitle(
        f"{model_name}: Baseline vs Skip vs Skip + Phase-Mag Compensation\nSkipped layers: {skip_layers}",
        fontsize=14,
        fontweight="bold",
    )

    width = 0.24
    b1 = ax1.bar(x - width, baseline, width, color="#2f4858", label="Baseline")
    b2 = ax1.bar(x, skip, width, color="#e74c3c", label="Skip Only")
    b3 = ax1.bar(x + width, comp, width, color="#2ca25f", label="Skip + Compensation")
    add_value_labels(ax1, b1)
    add_value_labels(ax1, b2)
    add_value_labels(ax1, b3)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="lower left")
    ax1.set_title("Per-Benchmark Accuracy (numbers shown on bars)")
    y_max = max(float(np.max(baseline)), float(np.max(skip)), float(np.max(comp)))
    ax1.set_ylim(0.0, min(1.05, y_max + 0.14))

    r1 = ax2.bar(x, recovered, color="#2ca25f", label="Recovered by compensation")
    r2 = ax2.bar(x, unrecovered, bottom=recovered, color="#f4cccc", label="Still missing vs baseline")
    ax2.set_ylabel("Accuracy Points")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right")
    ax2.set_title("How Much Skip Loss Is Recovered")

    for idx, (bar_rec, bar_unrec) in enumerate(zip(r1, r2)):
        total = drop[idx]
        rec = recovered[idx]
        if total <= 1e-9:
            text = "No drop"
        else:
            text = f"{rec_pct[idx]:.1f}% recovered"
            if recovery_raw[idx] < 0:
                text = f"Worse ({recovery_raw[idx]:.3f})"
        top = bar_rec.get_height() + bar_unrec.get_height()
        ax2.text(
            bar_rec.get_x() + (bar_rec.get_width() / 2.0),
            top + 0.004,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def build_summary(table: pd.DataFrame, model_name: str, skip_layers: str) -> str:
    baseline_mean = table["baseline"].mean()
    skip_mean = table["skip_only"].mean()
    comp_mean = table["skip_compensated"].mean()
    drop = baseline_mean - skip_mean
    rec = comp_mean - skip_mean
    rec_pct = (100.0 * rec / drop) if abs(drop) > 1e-9 else 0.0
    return (
        f"Model: {model_name}\n"
        f"Skipped Layers: {skip_layers}\n"
        f"Benchmarks: {len(table)}\n\n"
        f"Average Baseline Accuracy:        {baseline_mean:.6f}\n"
        f"Average Skip-Only Accuracy:       {skip_mean:.6f}\n"
        f"Average Skip+Comp Accuracy:       {comp_mean:.6f}\n"
        f"Average Accuracy Drop (skip):     {drop:.6f}\n"
        f"Average Accuracy Recovery:        {rec:.6f}\n"
        f"Average Recovery Percentage:      {rec_pct:.2f}%\n"
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    output_csv = Path(args.output_csv).resolve() if args.output_csv else run_dir / "comparison_results.csv"
    output_chart = Path(args.output_chart).resolve() if args.output_chart else run_dir / "comparison_visualization.png"
    output_summary = Path(args.output_summary).resolve() if args.output_summary else run_dir / "experiment_summary.txt"

    base = load_scores(run_dir / "baseline" / "results.csv")
    skip = load_scores(run_dir / "skip_only" / "results.csv")
    comp = load_scores(run_dir / "skip_compensated" / "results.csv")

    model_name = args.model_name or run_dir.name
    skip_layers = args.skip_layers or "N/A"

    table = build_table(base, skip, comp)
    table.to_csv(output_csv, index=False)
    draw_chart(table, output_chart, model_name=model_name, skip_layers=skip_layers)
    output_summary.write_text(build_summary(table, model_name=model_name, skip_layers=skip_layers), encoding="utf-8")

    print(f"Saved {output_csv}")
    print(f"Saved {output_chart}")
    print(f"Saved {output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

