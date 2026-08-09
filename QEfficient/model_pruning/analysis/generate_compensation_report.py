#!/usr/bin/env python3
"""
Generate plots + report for Qwen2.5 compensation experiments.

Outputs:
  - results/compensation_runs/cross_method/Qwen2.5-7b-Instruct_compensation_experiment/plots/*.png
  - results/compensation_runs/cross_method/Qwen2.5-7b-Instruct_compensation_experiment/COMPREHENSIVE_DATA_REPORT.md
  - results/compensation_runs/cross_method/Qwen2.5-7b-Instruct_compensation_experiment/COMPREHENSIVE_DATA_REPORT.txt
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np



def load_results_csv(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    data: Dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("benchmark") or row.get("task") or ""
            score = row.get("score")
            if task and score:
                data[task] = float(score)
    if not data:
        raise ValueError(f"No rows found in {path}")
    return data


def compute_recovery(
    baseline: Dict[str, float],
    skip_only: Dict[str, float],
    compensated: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    recovery = {}
    recovery_pct = {}
    for task, base_val in baseline.items():
        skip_val = skip_only.get(task, np.nan)
        comp_val = compensated.get(task, np.nan)
        recovery_val = comp_val - skip_val
        denom = base_val - skip_val
        if abs(denom) < 1e-8:
            recovery_pct_val = 0.0
        else:
            recovery_pct_val = (recovery_val / denom) * 100.0
        recovery[task] = recovery_val
        recovery_pct[task] = recovery_pct_val
    return recovery, recovery_pct


def avg(scores: Dict[str, float]) -> float:
    return float(np.mean(list(scores.values())))


def plot_per_task_accuracy(model_name, baseline, skip_only, phase_mag, mlp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(baseline.keys())
    labels = ["Baseline", "Skip-only", "Phase-Mag", "MLP Small"]
    values = {
        "Baseline": baseline,
        "Skip-only": skip_only,
        "Phase-Mag": phase_mag,
        "MLP Small": mlp,
    }
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 6), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        vals = [values[label].get(task, 0.0) for label in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))
        ax.axhline(baseline.get(task, 0.0), color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel("Accuracy")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    plt.suptitle(f"Per-Task Accuracy: {model_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_per_task_accuracy_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_pct(model_name, recovery_phase, recovery_mlp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(recovery_phase.keys())
    x = np.arange(len(tasks))
    width = 0.35
    phase_vals = [recovery_phase[t] for t in tasks]
    mlp_vals = [recovery_mlp[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, phase_vals, width, label="Phase-Mag", color="#4CAF50")
    ax.bar(x + width / 2, mlp_vals, width, label="MLP Small", color="#FF9800")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Recovery %")
    ax.set_title(f"Recovery % vs Skip-only: {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_recovery_pct_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_avg_accuracy(model_name, baseline, skip_only, phase_mag, mlp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Baseline", "Skip-only", "Phase-Mag", "MLP Small"]
    vals = [avg(baseline), avg(skip_only), avg(phase_mag), avg(mlp)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=["#2196F3", "#F44336", "#4CAF50", "#FF9800"], alpha=0.85)
    ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))
    ax.set_ylabel("Average Accuracy")
    ax.set_title(f"Average Accuracy: {model_name}")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_avg_accuracy_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_heatmap(model_name, recovery_phase, recovery_mlp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(recovery_phase.keys())
    data = np.array([
        [recovery_phase[t] for t in tasks],
        [recovery_mlp[t] for t in tasks],
    ])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(data, cmap="coolwarm", aspect="auto")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Phase-Mag", "MLP Small"])
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_title(f"Recovery % Heatmap: {model_name}")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Recovery %")
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_recovery_heatmap_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(data, out_md, out_txt):
    lines = []
    lines.append("# Qwen2.5 Compensation Experiment Report\n")
    lines.append("Models: Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct\n")
    lines.append("Strategies: Phase-Mag, MLP Small\n")

    for model_name, info in data.items():
        baseline = info["baseline"]
        skip_only = info["skip_only"]
        phase_mag = info["phase_mag"]
        mlp = info["mlp_small"]
        rec_phase = info["recovery_phase"]
        recp_phase = info["recovery_phase_pct"]
        rec_mlp = info["recovery_mlp"]
        recp_mlp = info["recovery_mlp_pct"]

        lines.append(f"\n## {model_name}\n")
        lines.append("| Task | Baseline | Skip-only | Phase-Mag | MLP Small |")
        lines.append("| --- | --- | --- | --- | --- |")
        for task in sorted(baseline.keys()):
            lines.append(
                f"| {task} | {baseline[task]:.3f} | {skip_only[task]:.3f} | "
                f"{phase_mag[task]:.3f} | {mlp[task]:.3f} |"
            )
        lines.append("\n**Recovery vs Skip-only**\n")
        lines.append("| Task | Phase-Mag Recovery | Phase-Mag % | MLP Recovery | MLP % |")
        lines.append("| --- | --- | --- | --- | --- |")
        for task in sorted(baseline.keys()):
            lines.append(
                f"| {task} | {rec_phase[task]:+.3f} | {recp_phase[task]:+.1f}% | "
                f"{rec_mlp[task]:+.3f} | {recp_mlp[task]:+.1f}% |"
            )
        lines.append("\n**Averages**\n")
        lines.append(
            f"- Baseline avg: {avg(baseline):.4f}\n"
            f"- Skip-only avg: {avg(skip_only):.4f}\n"
            f"- Phase-Mag avg: {avg(phase_mag):.4f}\n"
            f"- MLP Small avg: {avg(mlp):.4f}\n"
        )

    text = "\n".join(lines)
    out_md.write_text(text)
    out_txt.write_text(text.replace("|", "\t"))


def main(results_root: Path) -> None:
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "Qwen2.5-7B-Instruct": {
            "baseline": results_root / "7b" / "mlp_small" / "baseline" / "results.csv",
            "skip_only": results_root / "7b" / "mlp_small" / "skip_only" / "results.csv",
            "phase_mag": results_root / "7b" / "phase_mag" / "phase_mag_rescaling" / "results.csv",
            "mlp_small": results_root / "7b" / "mlp_small" / "compensated_mlp" / "results.csv",
        },
        "Qwen2.5-32B-Instruct": {
            "baseline": results_root / "32b" / "mlp_small" / "baseline" / "results.csv",
            "skip_only": results_root / "32b" / "mlp_small" / "skip_only" / "results.csv",
            "phase_mag": results_root / "32b" / "phase_mag" / "phase_mag_rescaling" / "results.csv",
            "mlp_small": results_root / "32b" / "mlp_small" / "compensated_mlp" / "results.csv",
        },
    }

    data = {}
    for model_name, paths in configs.items():
        baseline = load_results_csv(paths["baseline"])
        skip_only = load_results_csv(paths["skip_only"])
        phase_mag = load_results_csv(paths["phase_mag"])
        mlp_small = load_results_csv(paths["mlp_small"])

        recovery_phase, recovery_phase_pct = compute_recovery(baseline, skip_only, phase_mag)
        recovery_mlp, recovery_mlp_pct = compute_recovery(baseline, skip_only, mlp_small)

        data[model_name] = {
            "baseline": baseline,
            "skip_only": skip_only,
            "phase_mag": phase_mag,
            "mlp_small": mlp_small,
            "recovery_phase": recovery_phase,
            "recovery_phase_pct": recovery_phase_pct,
            "recovery_mlp": recovery_mlp,
            "recovery_mlp_pct": recovery_mlp_pct,
        }

        plot_per_task_accuracy(model_name, baseline, skip_only, phase_mag, mlp_small, plots_dir)
        plot_recovery_pct(model_name, recovery_phase_pct, recovery_mlp_pct, plots_dir)
        plot_avg_accuracy(model_name, baseline, skip_only, phase_mag, mlp_small, plots_dir)
        plot_recovery_heatmap(model_name, recovery_phase_pct, recovery_mlp_pct, plots_dir)

    report_md = results_root / "COMPREHENSIVE_DATA_REPORT.md"
    report_txt = results_root / "COMPREHENSIVE_DATA_REPORT.txt"
    build_report(data, report_md, report_txt)

    with (results_root / "summary_metrics.json").open("w") as f:
        json.dump(data, f, indent=2)

    print("Plots + reports written to:", results_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Directory containing 7b/ and 32b/ compensation result CSVs.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args().results_root)
