#!/usr/bin/env python3
"""Pipeline orchestrator for GPT-OSS MoE expert-importance analysis.

# START: Pipeline 1 overview
# This runner implements the MoE expert-importance pipeline using the reference
# logic adapted into the current NAS repository.  The flow is intentionally
# simple and stage-based:
#   1. profile GPT-OSS router choices and write layer-by-expert metric CSVs,
#   2. analyze those CSVs into expert-importance ranking/consensus CSVs,
#   3. convert the consensus CSV into a pruned-experts JSON artifact,
#   4. write a pipeline summary and resumable checkpoint.
#
# The final pruning JSON has the exact top-level shape expected by the pruning
# pipeline: {"pruned_experts": {"<layer_index>": [<expert_id>, ...]}}.
# Experts are grouped layer-wise and ordered by least-importance ranking from
# the generated expert-importance consensus CSV.
# END: Pipeline 1 overview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


from QEfficient.model_pruning.analysis.analyze_moe_expert_importance import analyze_moe_expert_importance
from QEfficient.model_pruning.analysis.measure_moe_expert_importance import DATASET_REGISTRY, parse_dataset_aliases, profile_moe_expert_importance


from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


class MoePipelineCheckpoint:
    """Small checkpoint manager for the MoE expert-importance pipeline."""

    CHECKPOINT_FILE = "pipeline_checkpoint.json"
    STAGES = {
        "moe_routing_profile": {
            "name": "MoE Routing Profile",
            "outputs": ["moe_routing/"],
        },
        "expert_importance_analysis": {
            "name": "Expert Importance Analysis",
            "outputs": ["expert_importance_summary.json", "expert_importance_consensus.csv", "expert_importance_full.csv"],
        },
        "pruned_experts_json": {
            "name": "Pruned Experts JSON",
            "outputs": ["pruned_experts.json"],
        },
        "summary": {
            "name": "Pipeline Summary",
            "outputs": ["pipeline_summary.json"],
        },
    }

    def __init__(self, output_dir: Path, model: str) -> None:
        self.output_dir = output_dir
        self.model = model
        self.checkpoint_path = output_dir / self.CHECKPOINT_FILE
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self.checkpoint_path.exists():
            try:
                with self.checkpoint_path.open() as checkpoint_file:
                    return json.load(checkpoint_file)
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint: {exc}. Creating a new one.")

        return {
            "version": "1.0",
            "pipeline": "moe_expert_importance",
            "model": self.model,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stages": {},
        }

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with temp_path.open("w") as checkpoint_file:
            json.dump(self.data, checkpoint_file, indent=2)
        temp_path.replace(self.checkpoint_path)

    def _outputs_exist(self, stage_name: str) -> bool:
        outputs = self.STAGES[stage_name].get("outputs", [])
        for output in outputs:
            output_path = self.output_dir / output.rstrip("/")
            if not output_path.exists():
                return False
        return True

    def is_complete(self, stage_name: str) -> bool:
        return self.data.get("stages", {}).get(stage_name, {}).get("status") == "complete" and self._outputs_exist(stage_name)

    def should_skip(self, stage_name: str, force: bool = False) -> bool:
        return not force and self.is_complete(stage_name)

    def mark_started(self, stage_name: str, metadata: Optional[dict] = None) -> None:
        stage = {
            "status": "in_progress",
            "display_name": self.STAGES[stage_name]["name"],
            "started_at": datetime.now().isoformat(),
        }
        if metadata:
            stage["metadata"] = metadata
        self.data["stages"][stage_name] = stage
        self.save()

    def mark_complete(self, stage_name: str, metadata: Optional[dict] = None) -> None:
        stage = self.data["stages"].get(stage_name, {})
        stage.update(
            {
                "status": "complete",
                "display_name": self.STAGES[stage_name]["name"],
                "completed_at": datetime.now().isoformat(),
            }
        )
        if metadata:
            stage["metadata"] = metadata
        self.data["stages"][stage_name] = stage
        self.save()
        logger.info(f"✓ Stage '{self.STAGES[stage_name]['name']}' complete")

    def mark_failed(self, stage_name: str, error: Exception | str) -> None:
        stage = self.data["stages"].get(stage_name, {})
        stage.update(
            {
                "status": "failed",
                "display_name": self.STAGES[stage_name]["name"],
                "failed_at": datetime.now().isoformat(),
                "error": str(error),
            }
        )
        self.data["stages"][stage_name] = stage
        self.save()
        logger.error(f"✗ Stage '{self.STAGES[stage_name]['name']}' failed: {error}")

    def summary(self) -> dict:
        return {stage: self.data.get("stages", {}).get(stage, {}).get("status", "pending") for stage in self.STAGES}


def extract_clean_model_name(model_id: str) -> str:
    import re

    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    return re.sub(r"[^\w\-.]", "_", model_name)


def build_pruned_experts_json_from_csv(
    expert_importance_csv: str | Path,
    output_json: str | Path,
) -> dict:
    """Convert full expert-importance CSV into the pruning JSON shape."""

    # START: Expert-importance CSV to pruning JSON conversion
    # ``expert_importance_full.csv`` contains exactly one row per layer/expert
    # after aggregating metric scores across datasets.  For the pruning JSON we
    # group by layer and preserve the CSV's least-to-most importance ordering:
    # lower ``importance_rank`` means less important and appears earlier.
    expert_importance_csv = Path(expert_importance_csv)
    output_json = Path(output_json)
    df = pd.read_csv(expert_importance_csv)

    required_columns = {"layer_index", "expert_index", "importance_rank", "mean_score"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {expert_importance_csv}: {missing_columns}")

    df = df.sort_values(
        by=["layer_index", "importance_rank", "mean_score", "expert_index"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    pruned_experts: dict[str, list[int]] = {}
    for layer_index, layer_df in df.groupby("layer_index", sort=True):
        pruned_experts[str(int(layer_index))] = [int(expert_id) for expert_id in layer_df["expert_index"].tolist()]

    payload = {"pruned_experts": pruned_experts}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as json_file:
        json.dump(payload, json_file, indent=2)

    return payload
    # END: Expert-importance CSV to pruning JSON conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline for GPT-OSS MoE expert-routing importance analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Hugging Face model id or local path.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hellaswag"],
        help=f"Dataset aliases, comma-separated aliases, or all. Available: {', '.join(DATASET_REGISTRY)}",
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum non-empty samples per dataset.")
    parser.add_argument("--num-samples", type=int, default=None, help="Alias for --max-samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Texts per forward pass.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument("--output-dir", default=None, help="Pipeline output directory.")
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda, cuda:0, etc. Ignored when --device-map is set.")
    parser.add_argument("--device-map", default=None, help="Optional Transformers device_map, e.g. auto.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--write-importance-debug", action="store_true")
    parser.add_argument(
        "--importance-metric",
        default="combined_score",
        choices=["combined_score", "freq_fraction", "freq_counts"],
        help="Matrix used for least/most expert ranking.",
    )
    parser.add_argument("--top-k-experts", type=int, default=3, help="Experts to keep per layer for top-k ranking summaries.")
    parser.add_argument("--force-rerun", action="store_true", help="Run all stages even when checkpoint is complete.")
    parser.add_argument(
        "--resume-from",
        choices=["moe_routing_profile", "expert_importance_analysis", "pruned_experts_json", "summary"],
        default=None,
        help="Force rerun from this stage onward.",
    )
    parser.add_argument("--clean-checkpoint", action="store_true", help="Delete existing checkpoint before starting.")
    return parser.parse_args()


def should_force_stage(args: argparse.Namespace, stage_name: str, stage_order: dict[str, int]) -> bool:
    if args.force_rerun:
        return True
    if args.resume_from is None:
        return False
    return stage_order[stage_name] >= stage_order[args.resume_from]


def main() -> None:
    # START: End-to-end MoE expert-importance pipeline execution
    # Parse runtime options, prepare output/checkpoint state, execute the three
    # data-producing stages in order, and finish with a compact summary.  Each
    # stage is independently resumable through the checkpoint file.
    args = parse_args()
    start_time = time.time()
    max_samples = args.num_samples if args.num_samples is not None else args.max_samples
    dataset_aliases = parse_dataset_aliases(args.datasets)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"{extract_clean_model_name(args.model)}_MoE_Expert_Importance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_checkpoint:
        checkpoint_path = output_dir / MoePipelineCheckpoint.CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Deleted existing checkpoint")

    checkpoint = MoePipelineCheckpoint(output_dir, args.model)
    stage_order = {stage: index for index, stage in enumerate(MoePipelineCheckpoint.STAGES)}
    routing_dir = output_dir / "moe_routing"
    pruned_json_path = output_dir / "pruned_experts.json"

    logger.info("=" * 80)
    logger.info("MOE EXPERT IMPORTANCE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Datasets: {dataset_aliases}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pruned experts JSON: {pruned_json_path}")
    logger.info("=" * 80)

    profile_metadata = None
    analysis_summary = None
    pruned_json_payload = None

    stage_name = "moe_routing_profile"
    force_stage = should_force_stage(args, stage_name, stage_order)
    if checkpoint.should_skip(stage_name, force=force_stage):
        logger.info(f"\n[STEP 1/4] Skipping '{MoePipelineCheckpoint.STAGES[stage_name]['name']}'")
    else:
        logger.info(f"\n[STEP 1/4] {MoePipelineCheckpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_started(stage_name, {"datasets": dataset_aliases, "max_samples": max_samples})
        try:
            profile_metadata = profile_moe_expert_importance(
                model_name=args.model,
                datasets=dataset_aliases,
                max_samples=max_samples,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_dir=routing_dir,
                device=args.device,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                write_importance_debug=args.write_importance_debug,
            )
            checkpoint.mark_complete(stage_name, profile_metadata)

            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("✓ GPU memory cleared after MoE routing profile")
        except Exception as exc:
            checkpoint.mark_failed(stage_name, exc)
            return

    stage_name = "expert_importance_analysis"
    force_stage = should_force_stage(args, stage_name, stage_order)
    if checkpoint.should_skip(stage_name, force=force_stage):
        logger.info(f"\n[STEP 2/4] Skipping '{MoePipelineCheckpoint.STAGES[stage_name]['name']}'")
    else:
        logger.info(f"\n[STEP 2/4] {MoePipelineCheckpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_started(stage_name, {"metric": args.importance_metric, "top_k": args.top_k_experts})
        try:
            analysis_summary = analyze_moe_expert_importance(
                routing_dir=routing_dir,
                output_dir=output_dir,
                metric=args.importance_metric,
                top_k=args.top_k_experts,
            )
            checkpoint.mark_complete(
                stage_name,
                {
                    "summary_json": analysis_summary["summary_json"],
                    "ranking_csv": analysis_summary["ranking_csv"],
                    "consensus_csv": analysis_summary["consensus_csv"],
                    "full_importance_csv": analysis_summary["full_importance_csv"],
                },
            )
        except Exception as exc:
            checkpoint.mark_failed(stage_name, exc)
            return

    stage_name = "pruned_experts_json"
    force_stage = should_force_stage(args, stage_name, stage_order)
    if checkpoint.should_skip(stage_name, force=force_stage):
        logger.info(f"\n[STEP 3/4] Skipping '{MoePipelineCheckpoint.STAGES[stage_name]['name']}'")
    else:
        logger.info(f"\n[STEP 3/4] {MoePipelineCheckpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_started(stage_name, {"source_csv": str(output_dir / "expert_importance_full.csv")})
        try:
            pruned_json_payload = build_pruned_experts_json_from_csv(
                expert_importance_csv=output_dir / "expert_importance_full.csv",
                output_json=pruned_json_path,
            )
            checkpoint.mark_complete(
                stage_name,
                {
                    "pruned_experts_json": str(pruned_json_path),
                    "num_layers": len(pruned_json_payload["pruned_experts"]),
                },
            )
        except Exception as exc:
            checkpoint.mark_failed(stage_name, exc)
            return

    stage_name = "summary"
    force_stage = should_force_stage(args, stage_name, stage_order)
    if checkpoint.should_skip(stage_name, force=force_stage):
        logger.info(f"\n[STEP 4/4] Skipping '{MoePipelineCheckpoint.STAGES[stage_name]['name']}'")
    else:
        logger.info(f"\n[STEP 4/4] {MoePipelineCheckpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_started(stage_name)
        try:
            summary = {
                "pipeline": "moe_expert_importance",
                "model": args.model,
                "datasets": dataset_aliases,
                "max_samples": max_samples,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "routing_dir": str(routing_dir),
                "importance_metric": args.importance_metric,
                "top_k_experts": args.top_k_experts,
                "profile_metadata": profile_metadata,
                "analysis_summary_path": str(output_dir / "expert_importance_summary.json"),
                "expert_importance_consensus_csv": str(output_dir / "expert_importance_consensus.csv"),
                "expert_importance_full_csv": str(output_dir / "expert_importance_full.csv"),
                "pruned_experts_json": str(pruned_json_path),
                "execution_time_seconds": time.time() - start_time,
                "output_directory": str(output_dir),
                "checkpoint_status": checkpoint.summary(),
                "timestamp": datetime.now().isoformat(),
            }
            with (output_dir / "pipeline_summary.json").open("w") as summary_file:
                json.dump(summary, summary_file, indent=2)
            checkpoint.mark_complete(stage_name, {"summary_json": str(output_dir / "pipeline_summary.json")})
        except Exception as exc:
            checkpoint.mark_failed(stage_name, exc)
            return

    logger.info("\n" + "=" * 80)
    logger.info("MOE PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Execution time: {(time.time() - start_time) / 60:.1f} minutes")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Pruned experts JSON: {pruned_json_path}")
    logger.info("=" * 80)
    # END: End-to-end MoE expert-importance pipeline execution


if __name__ == "__main__":
    main()
