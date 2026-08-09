#!/usr/bin/env python3
"""Analyze MoE expert-routing matrices and summarize expert importance.

# START: MoE expert-importance CSV analysis module
# This module is adapted from the reference MoE expert-importance analysis
# logic.  It is intentionally small and file-oriented: it reads the routing
# matrix CSV files produced by ``measure_moe_expert_importance.py``, ranks
# experts inside each layer, writes human-readable CSV summaries, and returns
# a JSON-serializable summary dictionary to the pipeline runner.
# END: MoE expert-importance CSV analysis module
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


METRIC_FILE_SUFFIXES = {
    "combined_score": "combined_score",
    "freq_fraction": "freq_fraction",
    "freq_counts": "freq_counts",
}


def _layer_label_to_index(label: str) -> int:
    return int(str(label).replace("layer_", ""))


def load_metric_matrices(routing_dir: str | Path, metric: str = "combined_score") -> dict[str, pd.DataFrame]:
    """Load ``{dataset}_{metric}.csv`` matrices from a routing output directory."""

    routing_dir = Path(routing_dir)
    if metric not in METRIC_FILE_SUFFIXES:
        raise ValueError(f"Unsupported metric `{metric}`. Choose from {sorted(METRIC_FILE_SUFFIXES)}")

    suffix = METRIC_FILE_SUFFIXES[metric]
    matrices: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(routing_dir.glob(f"*_{suffix}.csv")):
        dataset = csv_path.name[: -len(f"_{suffix}.csv")]
        df = pd.read_csv(csv_path)
        if "layer" not in df.columns:
            raise ValueError(f"Missing `layer` column in {csv_path}")
        matrices[dataset] = df

    if not matrices:
        raise ValueError(f"No `*_{suffix}.csv` files found in {routing_dir}")
    return matrices


def rank_dataset_experts(df: pd.DataFrame, dataset: str, metric: str, top_k: int) -> list[dict]:
    """Rank experts within each layer for one dataset matrix."""

    expert_columns = [column for column in df.columns if column.startswith("expert_")]
    rows: list[dict] = []

    for _, row in df.iterrows():
        layer_index = _layer_label_to_index(row["layer"])
        values = row[expert_columns].astype(float)
        ascending = values.sort_values(ascending=True).head(top_k)
        descending = values.sort_values(ascending=False).head(top_k)

        for rank, (expert_name, value) in enumerate(ascending.items(), start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "layer_index": layer_index,
                    "expert_index": int(expert_name.replace("expert_", "")),
                    "rank_type": "least_important",
                    "rank": rank,
                    "metric": metric,
                    "score": float(value),
                }
            )

        for rank, (expert_name, value) in enumerate(descending.items(), start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "layer_index": layer_index,
                    "expert_index": int(expert_name.replace("expert_", "")),
                    "rank_type": "most_important",
                    "rank": rank,
                    "metric": metric,
                    "score": float(value),
                }
            )

    return rows


def build_consensus(ranking_df: pd.DataFrame, rank_type: str = "least_important") -> list[dict]:
    """Aggregate per-dataset expert rankings into layer/expert consensus rows."""

    subset = ranking_df[ranking_df["rank_type"] == rank_type]
    if subset.empty:
        return []

    consensus_rows: list[dict] = []
    group_columns = ["layer_index", "expert_index"]
    for (layer_index, expert_index), group in subset.groupby(group_columns):
        datasets = sorted(group["dataset"].unique().tolist())
        consensus_rows.append(
            {
                "layer_index": int(layer_index),
                "expert_index": int(expert_index),
                "rank_type": rank_type,
                "dataset_frequency": int(len(datasets)),
                "datasets": datasets,
                "mean_rank": float(group["rank"].mean()),
                "mean_score": float(group["score"].mean()),
                "min_score": float(group["score"].min()),
                "max_score": float(group["score"].max()),
            }
        )

    return sorted(consensus_rows, key=lambda row: (-row["dataset_frequency"], row["mean_rank"], row["mean_score"]))


def summarize_layer_recommendations(consensus_rows: list[dict], top_k: int = 3) -> dict[str, list[dict]]:
    """Keep top consensus experts per layer."""

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in consensus_rows:
        by_layer[row["layer_index"]].append(row)

    return {str(layer): rows[:top_k] for layer, rows in sorted(by_layer.items())}


def build_full_expert_importance(matrices: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Build one aggregated least-to-most importance row per layer/expert."""

    # START: Full aggregated expert-importance table
    # Unlike the top-k ranking/consensus CSVs above, this table keeps every
    # expert in every layer.  Scores are aggregated across datasets first, then
    # ranked inside each layer from least important to most important.  The row
    # count is therefore exactly ``num_layers * num_experts_per_layer`` when all
    # datasets have the same routing matrix shape.
    rows: list[dict] = []
    for dataset, df in matrices.items():
        expert_columns = [column for column in df.columns if column.startswith("expert_")]
        for _, row in df.iterrows():
            layer_index = _layer_label_to_index(row["layer"])
            for expert_column in expert_columns:
                rows.append(
                    {
                        "dataset": dataset,
                        "layer_index": layer_index,
                        "expert_index": int(expert_column.replace("expert_", "")),
                        "metric": metric,
                        "score": float(row[expert_column]),
                    }
                )

    raw_df = pd.DataFrame(rows)
    full_rows: list[dict] = []
    for (layer_index, expert_index), group in raw_df.groupby(["layer_index", "expert_index"]):
        datasets = sorted(group["dataset"].unique().tolist())
        full_rows.append(
            {
                "layer_index": int(layer_index),
                "expert_index": int(expert_index),
                "metric": metric,
                "dataset_frequency": int(len(datasets)),
                "datasets": datasets,
                "mean_score": float(group["score"].mean()),
                "min_score": float(group["score"].min()),
                "max_score": float(group["score"].max()),
            }
        )

    full_df = pd.DataFrame(full_rows)
    full_df = full_df.sort_values(
        by=["layer_index", "mean_score", "expert_index"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    full_df["importance_rank"] = full_df.groupby("layer_index").cumcount() + 1
    full_df = full_df[
        [
            "layer_index",
            "expert_index",
            "importance_rank",
            "metric",
            "dataset_frequency",
            "datasets",
            "mean_score",
            "min_score",
            "max_score",
        ]
    ]
    return full_df
    # END: Full aggregated expert-importance table


def analyze_moe_expert_importance(
    routing_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    metric: str = "combined_score",
    top_k: int = 3,
) -> dict:
    """Analyze MoE routing matrices and write CSV/JSON summaries."""

    # START: CSV-to-importance summary flow
    # The profiler writes one matrix per dataset and metric.  This function
    # loads the requested metric, ranks each layer's experts, emits the original
    # reference top-k ranking/consensus CSVs, and additionally writes a full
    # aggregated all-experts CSV used by the pruning JSON artifact.
    routing_dir = Path(routing_dir)
    output_dir = Path(output_dir) if output_dir is not None else routing_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices = load_metric_matrices(routing_dir, metric=metric)
    ranking_rows: list[dict] = []
    for dataset, matrix_df in matrices.items():
        ranking_rows.extend(rank_dataset_experts(matrix_df, dataset=dataset, metric=metric, top_k=top_k))

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_csv = output_dir / "expert_importance_rankings.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    least_consensus = build_consensus(ranking_df, rank_type="least_important")
    most_consensus = build_consensus(ranking_df, rank_type="most_important")

    consensus_df = pd.DataFrame(least_consensus + most_consensus)
    consensus_csv = output_dir / "expert_importance_consensus.csv"
    consensus_df.to_csv(consensus_csv, index=False)

    full_importance_df = build_full_expert_importance(matrices, metric=metric)
    full_importance_csv = output_dir / "expert_importance_full.csv"
    full_importance_df.to_csv(full_importance_csv, index=False)

    summary = {
        "routing_dir": str(routing_dir),
        "metric": metric,
        "top_k": top_k,
        "datasets_analyzed": sorted(matrices),
        "num_datasets": len(matrices),
        "ranking_csv": str(ranking_csv),
        "consensus_csv": str(consensus_csv),
        "full_importance_csv": str(full_importance_csv),
        "full_importance_rows": int(len(full_importance_df)),
        "least_important_consensus": least_consensus,
        "most_important_consensus": most_consensus,
        "least_important_by_layer": summarize_layer_recommendations(least_consensus, top_k=top_k),
        "most_important_by_layer": summarize_layer_recommendations(most_consensus, top_k=top_k),
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = output_dir / "expert_importance_summary.json"
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    summary["summary_json"] = str(summary_path)

    return summary
    # END: CSV-to-importance summary flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MoE expert-routing matrices.")
    parser.add_argument("--routing-dir", required=True, help="Directory containing `{dataset}_{metric}.csv` files.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary outputs. Defaults to routing dir parent.")
    parser.add_argument("--metric", default="combined_score", choices=sorted(METRIC_FILE_SUFFIXES))
    parser.add_argument("--top-k", type=int, default=3, help="Experts to keep per layer for least/most rankings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_moe_expert_importance(
        routing_dir=args.routing_dir,
        output_dir=args.output_dir,
        metric=args.metric,
        top_k=args.top_k,
    )
    print(f"Wrote expert importance summary to {summary['summary_json']}")


if __name__ == "__main__":
    main()
