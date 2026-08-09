#!/usr/bin/env python3
"""GPU MoE expert-pruning benchmark pipeline for GPT-OSS models.

# START: Pipeline 2 overview
# This runner is adapted from the reference MoE pruning benchmark logic.  It
# consumes the ranked expert JSON emitted by Pipeline 1, selects the first N
# least-important experts per layer via ``--experts-per-layer``, benchmarks the
# original model, benchmarks a freshly loaded model with those experts masked at
# the router, then writes score/report/summary artifacts.
#
# The pruning mechanism is router-logit masking only: selected expert logits are
# set to the dtype minimum before top-k expert selection.  Native GPT-OSS
# ``set_pruned_experts`` support is used when available; otherwise the runtime
# fallback below patches compatible router modules in memory.
# END: Pipeline 2 overview
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


# START: Benchmark dataset mapping
# Keep this mapping local to the pipeline so the runner remains a direct,
# self-contained adaptation of the reference pruning pipeline.
BENCHMARK_MAPPING = {
    "gsm8k": "gsm8k",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "truthfulqa": "truthfulqa_mc2",
    "piqa": "piqa",
    "boolq": "boolq",
    "openbookqa": "openbookqa",
}
ALL_DATASETS = list(BENCHMARK_MAPPING)
# END: Benchmark dataset mapping


# START: JSON serialization helpers
def make_json_serializable(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(value) for value in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if np is not None and isinstance(obj, np.integer):
        return int(obj)
    if np is not None and isinstance(obj, np.floating):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    return str(obj)
# END: JSON serialization helpers


# START: Checkpoint manager
class PipelineCheckpoint:
    """Small stage-level checkpoint manager for the MoE pruning pipeline."""

    FILENAME = "pipeline_checkpoint.json"
    STAGES = {
        "baseline_benchmark": "Baseline Benchmark",
        "pruned_benchmark": "Pruned Model Benchmark",
        "report_generation": "Comparison Report",
        "summary": "Pipeline Summary",
    }

    def __init__(self, output_dir: Path, model_name: str) -> None:
        self.output_dir = output_dir
        self.path = output_dir / self.FILENAME
        self.model_name = model_name
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                with self.path.open() as checkpoint_file:
                    data = json.load(checkpoint_file)
                if data.get("pipeline") == "moe_pruning_gpu":
                    return data
            except Exception as exc:
                logger.warning(f"Could not load checkpoint: {exc}. Starting fresh.")

        return {
            "version": "1.0",
            "pipeline": "moe_pruning_gpu",
            "model": self.model_name,
            "stages": {},
        }

    def save(self) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w") as checkpoint_file:
            json.dump(make_json_serializable(self.data), checkpoint_file, indent=2)
        temp_path.replace(self.path)

    def mark_started(self, stage: str) -> None:
        self.data["stages"][stage] = {"status": "in_progress"}
        self.save()

    def mark_complete(self, stage: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self.data["stages"][stage] = {"status": "complete", **(metadata or {})}
        self.save()
        logger.info(f"✓ Stage '{self.STAGES[stage]}' complete")

    def mark_failed(self, stage: str, error: Exception | str) -> None:
        self.data["stages"][stage] = {"status": "failed", "error": str(error)}
        self.save()
        logger.error(f"✗ Stage '{self.STAGES[stage]}' failed: {error}")

    def is_complete(self, stage: str) -> bool:
        return self.data.get("stages", {}).get(stage, {}).get("status") == "complete"

    def should_skip(self, stage: str, force_rerun: bool) -> bool:
        return not force_rerun and self.is_complete(stage)

    def get_summary(self) -> dict[str, Any]:
        return {
            "stages_complete": sum(
                1 for stage in self.data.get("stages", {}).values() if stage.get("status") == "complete"
            ),
            "total_stages": len(self.STAGES),
        }
# END: Checkpoint manager


# START: Model loading and pruning helpers
def parse_torch_dtype(raw_dtype: str):
    if raw_dtype == "auto":
        return "auto"
    if raw_dtype == "float32":
        return torch.float32
    if raw_dtype == "float16":
        return torch.float16
    if raw_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {raw_dtype}")


def normalize_device_map(device: str, device_map: Optional[str]) -> Optional[str]:
    if device_map is not None and device_map.lower() in {"none", "null"}:
        return None
    if device_map is not None:
        return device_map
    if device.startswith("cuda") and torch.cuda.is_available():
        return "auto"
    return None


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    device_map: Optional[str],
    torch_dtype: str,
) -> tuple[Any, Any]:
    cleanup_cuda_memory()
    resolved_device_map = normalize_device_map(device, device_map)
    dtype = parse_torch_dtype(torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map

    logger.info(
        "Loading model with torch_dtype=%s, device=%s, device_map=%s",
        torch_dtype,
        device,
        resolved_device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if resolved_device_map is None:
        model.to(device)

    return model, tokenizer


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def cleanup_objects(*objects: Any) -> None:
    for obj in objects:
        del obj
    cleanup_cuda_memory()


def apply_moe_pruning(model: Any, prune_config: dict[str, Any]) -> dict[str, Any]:
    """Apply layer-wise MoE expert pruning through router masking."""

    pruned_experts_map = prune_config.get("pruned_experts", {})
    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find `model.model.layers`; expected a GPT-OSS decoder model.")

    applied_by_layer: dict[str, list[int]] = {}
    total_pruned = 0
    for layer_idx, decoder_layer in enumerate(layers):
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue

        expert_ids = [int(expert_id) for expert_id in pruned_experts_map.get(str(layer_idx), [])]
        mlp._layer_idx = layer_idx
        set_mlp_pruned_experts(mlp, expert_ids)
        if expert_ids:
            applied_by_layer[str(layer_idx)] = expert_ids
            total_pruned += len(expert_ids)
            logger.info("Layer %s: pruning experts %s", layer_idx, expert_ids)

    return {
        "layers_pruned": len(applied_by_layer),
        "experts_pruned": total_pruned,
        "pruned_experts": applied_by_layer,
    }


# START: Runtime fallback for unpatched GPT-OSS model classes
def set_mlp_pruned_experts(mlp: Any, expert_ids: list[int]) -> None:
    if hasattr(mlp, "set_pruned_experts"):
        mlp.set_pruned_experts(expert_ids)
        return

    router = getattr(mlp, "router", None)
    if router is None:
        raise RuntimeError("Could not find `mlp.router`; expected a GPT-OSS MoE MLP.")

    ensure_router_pruning_support(router)
    mlp.pruned_expert_ids = {int(expert_id) for expert_id in expert_ids}
    router.set_pruned_experts(mlp.pruned_expert_ids)


def ensure_router_pruning_support(router: Any) -> None:
    if not hasattr(router, "_pruned_expert_mask"):
        num_experts = int(getattr(router, "num_experts"))
        router.register_buffer(
            "_pruned_expert_mask",
            torch.zeros(num_experts, dtype=torch.bool, device=router.weight.device),
            persistent=False,
        )

    if not hasattr(router, "set_pruned_experts"):
        router.set_pruned_experts = types.MethodType(runtime_set_pruned_experts, router)

    if not getattr(router, "_runtime_moe_pruning_forward_patched", False):
        router.forward = types.MethodType(runtime_pruned_router_forward, router)
        router._runtime_moe_pruning_forward_patched = True


def runtime_set_pruned_experts(self: Any, expert_ids: set[int] | list[int]) -> None:
    valid_expert_ids = sorted(
        {int(expert_id) for expert_id in expert_ids if 0 <= int(expert_id) < int(self.num_experts)}
    )
    if len(valid_expert_ids) > int(self.num_experts) - int(self.top_k):
        raise ValueError(
            f"Cannot prune {len(valid_expert_ids)} experts when "
            f"num_experts={self.num_experts} and top_k={self.top_k}."
        )

    self._pruned_expert_mask.zero_()
    if valid_expert_ids:
        expert_ids_tensor = torch.tensor(valid_expert_ids, device=self._pruned_expert_mask.device, dtype=torch.long)
        self._pruned_expert_mask[expert_ids_tensor] = True


def runtime_pruned_router_forward(self: Any, hidden_states: torch.Tensor):
    router_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
    if self._pruned_expert_mask.any():
        router_logits = router_logits.masked_fill(self._pruned_expert_mask, torch.finfo(router_logits.dtype).min)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    return router_logits, router_scores, router_indices
# END: Runtime fallback for unpatched GPT-OSS model classes


def load_baseline_model(model_name: str, device: str, device_map: Optional[str], torch_dtype: str) -> tuple[Any, Any]:
    logger.info("Loading BASELINE model with no MoE pruning")
    return load_model_and_tokenizer(model_name, device, device_map, torch_dtype)


def load_pruned_model(
    model_name: str,
    prune_config: dict[str, Any],
    device: str,
    device_map: Optional[str],
    torch_dtype: str,
) -> tuple[Any, Any, dict[str, Any]]:
    logger.info("Loading PRUNED model with native MoE expert masking")
    model, tokenizer = load_model_and_tokenizer(model_name, device, device_map, torch_dtype)
    pruning_summary = apply_moe_pruning(model, prune_config)
    logger.info(
        "Applied pruning: %s experts across %s layers",
        pruning_summary["experts_pruned"],
        pruning_summary["layers_pruned"],
    )
    return model, tokenizer, pruning_summary


def cleanup_model(model: Any, tokenizer: Any) -> None:
    cleanup_objects(model, tokenizer)
# END: Model loading and pruning helpers


# START: lm-eval helpers
def run_lm_eval(
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    batch_size: int,
    device: str,
    limit: Optional[int],
    verbosity: str,
) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError("lm-eval is not installed. Install with: pip install lm-eval>=0.4.0") from exc

    lm_model = None
    try:
        lm_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            trust_remote_code=True,
        )

        results = simple_evaluate(
            model=lm_model,
            tasks=tasks,
            limit=limit,
            bootstrap_iters=100000,
            write_out=False,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            verbosity=verbosity,
        )
        if not results or "results" not in results:
            raise RuntimeError("lm-eval returned an invalid results structure.")
        return results
    finally:
        cleanup_objects(lm_model)


METRIC_PRIORITY = [
    "acc_norm,none",
    "acc,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc_norm",
    "acc",
]


def extract_score(task_results: dict[str, Any]) -> tuple[Optional[float], str]:
    for metric in METRIC_PRIORITY:
        if metric in task_results:
            return float(task_results[metric]), metric
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr"):
            return float(value), key
    return None, "missing"


def extract_all_scores(lm_results: dict[str, Any], datasets: list[str]) -> dict[str, dict[str, Any]]:
    scores = {}
    raw_results = lm_results.get("results", {})
    for dataset in datasets:
        task_name = BENCHMARK_MAPPING[dataset]
        if task_name in raw_results:
            score, metric = extract_score(raw_results[task_name])
            scores[dataset] = {"score": score, "metric": metric}
        else:
            logger.warning("Task result missing for dataset=%s task=%s", dataset, task_name)
            scores[dataset] = {"score": None, "metric": "missing"}
    return scores
# END: lm-eval helpers


# START: Report helpers
def build_comparison_report(
    baseline_scores: dict[str, dict[str, Any]],
    pruned_scores: dict[str, dict[str, Any]],
    datasets: list[str],
    pruning_summary: dict[str, Any],
    accuracy_threshold: float,
) -> dict[str, Any]:
    per_dataset = {}
    pct_deltas = []

    for dataset in datasets:
        baseline_score = baseline_scores.get(dataset, {}).get("score")
        pruned_score = pruned_scores.get(dataset, {}).get("score")
        metric = baseline_scores.get(dataset, {}).get("metric", "unknown")

        if baseline_score is None or pruned_score is None:
            per_dataset[dataset] = {
                "baseline_score": baseline_score,
                "pruned_score": pruned_score,
                "abs_delta": None,
                "pct_delta": None,
                "within_threshold": None,
                "metric": metric,
            }
            continue

        abs_delta = pruned_score - baseline_score
        pct_delta = (abs_delta / baseline_score * 100.0) if baseline_score != 0 else 0.0
        within_threshold = abs(pct_delta) <= accuracy_threshold
        pct_deltas.append(pct_delta)

        per_dataset[dataset] = {
            "baseline_score": round(baseline_score, 6),
            "pruned_score": round(pruned_score, 6),
            "abs_delta": round(abs_delta, 6),
            "pct_delta": round(pct_delta, 4),
            "within_threshold": within_threshold,
            "metric": metric,
        }

    avg_accuracy_delta = round(sum(pct_deltas) / len(pct_deltas), 4) if pct_deltas else None
    datasets_within = sum(1 for values in per_dataset.values() if values.get("within_threshold"))
    datasets_total = sum(1 for values in per_dataset.values() if values.get("within_threshold") is not None)

    if avg_accuracy_delta is None:
        recommendation = "UNKNOWN — could not compute accuracy delta"
    elif avg_accuracy_delta >= 0:
        recommendation = "SAFE — pruned model is on par with or better than baseline"
    elif abs(avg_accuracy_delta) <= accuracy_threshold:
        recommendation = (
            f"SAFE — average accuracy drop {abs(avg_accuracy_delta):.2f}% is within threshold {accuracy_threshold}%"
        )
    else:
        recommendation = (
            f"RISKY — average accuracy drop {abs(avg_accuracy_delta):.2f}% exceeds threshold {accuracy_threshold}%"
        )

    return {
        "pruning_summary": pruning_summary,
        "per_dataset": per_dataset,
        "overall": {
            "avg_accuracy_delta_pct": avg_accuracy_delta,
            "datasets_within_threshold": f"{datasets_within}/{datasets_total}",
            "accuracy_threshold_pct": accuracy_threshold,
            "recommendation": recommendation,
        },
    }


def save_comparison_report(report: dict[str, Any], output_dir: Path) -> None:
    report_path = output_dir / "comparison_report.json"
    with report_path.open("w") as report_file:
        json.dump(make_json_serializable(report), report_file, indent=2)
    logger.info("Comparison report JSON: %s", report_path)

    try:
        import csv

        csv_path = output_dir / "comparison_report.csv"
        rows = []
        for dataset, values in report["per_dataset"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "metric": values.get("metric", ""),
                    "baseline_score": values.get("baseline_score", ""),
                    "pruned_score": values.get("pruned_score", ""),
                    "abs_delta": values.get("abs_delta", ""),
                    "pct_delta": values.get("pct_delta", ""),
                    "within_threshold": values.get("within_threshold", ""),
                }
            )
        if rows:
            with csv_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Comparison report CSV: %s", csv_path)
    except Exception as exc:
        logger.warning("Could not write comparison CSV: %s", exc)


def log_comparison_table(report: dict[str, Any]) -> None:
    logger.info("=" * 80)
    logger.info("PER-DATASET ACCURACY COMPARISON")
    logger.info("  %-20s %10s %10s %10s %5s", "Dataset", "Baseline", "Pruned", "Delta%", "OK")
    logger.info("  %s", "-" * 62)
    for dataset, values in report["per_dataset"].items():
        baseline = values.get("baseline_score")
        pruned = values.get("pruned_score")
        delta = values.get("pct_delta")
        ok_marker = "✓" if values.get("within_threshold") else "✗"
        baseline_text = f"{baseline:.4f}" if baseline is not None else "N/A"
        pruned_text = f"{pruned:.4f}" if pruned is not None else "N/A"
        delta_text = f"{delta:+.2f}%" if delta is not None else "N/A"
        logger.info("  %-20s %10s %10s %10s %5s", dataset, baseline_text, pruned_text, delta_text, ok_marker)
    logger.info("Recommendation: %s", report["overall"]["recommendation"])
    logger.info("=" * 80)
# END: Report helpers


# START: Prune-config selection helpers
def select_pruned_experts_per_layer(prune_config: dict[str, Any], experts_per_layer: int) -> dict[str, Any]:
    """Select the first N least-important experts per layer from Pipeline 1 JSON."""

    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")

    source = prune_config.get("pruned_experts")
    if not isinstance(source, dict):
        raise ValueError('Pruning config must contain object key "pruned_experts".')

    selected: dict[str, list[int]] = {}
    for layer, expert_ids in source.items():
        if not isinstance(expert_ids, list):
            raise ValueError(f"Layer {layer!r} must map to a list of expert ids.")
        selected[str(int(layer))] = [int(expert_id) for expert_id in expert_ids[:experts_per_layer]]

    return {"pruned_experts": selected}


def write_selected_prune_config(prune_config: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "selected_prune_config.json"
    with path.open("w") as config_file:
        json.dump(make_json_serializable(prune_config), config_file, indent=2)
    return path
# END: Prune-config selection helpers


# START: CLI and pipeline orchestration
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU GPT-OSS MoE expert-pruning pipeline without external transforms or latency tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument(
        "--prune-config",
        required=True,
        help='JSON file with format: {"pruned_experts": {"3": [2, 5], "7": [0]}}',
    )
    parser.add_argument(
        "--experts-per-layer",
        type=int,
        required=True,
        help="Prune the first N least-important experts from each layer list in --prune-config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hellaswag"],
        choices=ALL_DATASETS + ["all"],
        help="Benchmark datasets to evaluate.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="lm-eval limit per dataset. None means all samples.")
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=3.0,
        help="Maximum acceptable relative accuracy drop percentage.",
    )
    parser.add_argument("--device", default="cuda", help="Device for lm-eval wrapper, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to disable and call model.to(device).",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size passed to lm-eval HFLM.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <model>_moe_pruning_gpu.")
    parser.add_argument("--lm-eval-verbosity", default="WARNING", help="Verbosity passed to lm-eval simple_evaluate.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore completed checkpoint stages.")
    parser.add_argument("--clean-checkpoint", action="store_true", help="Delete existing checkpoint before starting.")
    return parser.parse_args()


def resolve_datasets(raw_datasets: list[str]) -> list[str]:
    if "all" in raw_datasets:
        return ALL_DATASETS
    return raw_datasets


def clean_model_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")


def load_prune_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pruning config not found: {path}")
    with path.open() as config_file:
        prune_config = json.load(config_file)
    if "pruned_experts" not in prune_config:
        raise ValueError('Pruning config must contain top-level key "pruned_experts".')
    return prune_config


def main() -> None:
    # START: End-to-end MoE pruning benchmark execution
    # Load the ranked Pipeline 1 JSON, select first N least-important experts
    # per layer, then run baseline and pruned lm-eval stages with checkpointed
    # report generation.  Baseline and pruned models are loaded separately so
    # router masking can never leak into the baseline measurement.
    args = parse_args()
    start_time = time.time()

    datasets = resolve_datasets(args.datasets)
    tasks = [BENCHMARK_MAPPING[dataset] for dataset in datasets]
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{clean_model_name(args.model)}_moe_pruning_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_checkpoint:
        checkpoint_path = output_dir / PipelineCheckpoint.FILENAME
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Deleted existing checkpoint: %s", checkpoint_path)

    source_prune_config = load_prune_config(Path(args.prune_config))
    prune_config = select_pruned_experts_per_layer(source_prune_config, args.experts_per_layer)
    selected_prune_config_path = write_selected_prune_config(prune_config, output_dir)

    checkpoint = PipelineCheckpoint(output_dir, args.model)

    baseline_results_path = output_dir / "baseline_results.json"
    baseline_scores_path = output_dir / "baseline_scores.json"
    pruned_results_path = output_dir / "pruned_results.json"
    pruned_scores_path = output_dir / "pruned_scores.json"
    pruning_summary_path = output_dir / "pruning_summary.json"

    logger.info("=" * 80)
    logger.info("MOE PRUNING GPU PIPELINE")
    logger.info("=" * 80)
    logger.info("Model: %s", args.model)
    logger.info("Datasets: %s", datasets)
    logger.info("Tasks: %s", tasks)
    logger.info("Source prune config: %s", args.prune_config)
    logger.info("Selected prune config: %s", selected_prune_config_path)
    logger.info("Experts per layer: %s", args.experts_per_layer)
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 80)

    stage = "baseline_benchmark"
    if checkpoint.should_skip(stage, args.force_rerun):
        logger.info("[STEP 1/4] Skipping baseline benchmark from checkpoint")
    else:
        logger.info("[STEP 1/4] Running baseline benchmark")
        checkpoint.mark_started(stage)
        try:
            model, tokenizer = load_baseline_model(args.model, args.device, args.device_map, args.torch_dtype)
            baseline_results = run_lm_eval(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.num_samples,
                verbosity=args.lm_eval_verbosity,
            )
            baseline_scores = extract_all_scores(baseline_results, datasets)
            with baseline_results_path.open("w") as results_file:
                json.dump(make_json_serializable(baseline_results), results_file, indent=2)
            with baseline_scores_path.open("w") as scores_file:
                json.dump(make_json_serializable(baseline_scores), scores_file, indent=2)
            del model, tokenizer
            cleanup_cuda_memory()
            checkpoint.mark_complete(stage, {"scores_json": str(baseline_scores_path)})
        except Exception as exc:
            checkpoint.mark_failed(stage, exc)
            sys.exit(1)

    stage = "pruned_benchmark"
    if checkpoint.should_skip(stage, args.force_rerun):
        logger.info("[STEP 2/4] Skipping pruned benchmark from checkpoint")
    else:
        logger.info("[STEP 2/4] Running pruned benchmark")
        checkpoint.mark_started(stage)
        try:
            model, tokenizer, pruning_summary = load_pruned_model(
                args.model,
                prune_config,
                args.device,
                args.device_map,
                args.torch_dtype,
            )
            pruned_results = run_lm_eval(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.num_samples,
                verbosity=args.lm_eval_verbosity,
            )
            pruned_scores = extract_all_scores(pruned_results, datasets)
            with pruned_results_path.open("w") as results_file:
                json.dump(make_json_serializable(pruned_results), results_file, indent=2)
            with pruned_scores_path.open("w") as scores_file:
                json.dump(make_json_serializable(pruned_scores), scores_file, indent=2)
            with pruning_summary_path.open("w") as summary_file:
                json.dump(make_json_serializable(pruning_summary), summary_file, indent=2)
            del model, tokenizer
            cleanup_cuda_memory()
            checkpoint.mark_complete(
                stage,
                {
                    "scores_json": str(pruned_scores_path),
                    "pruning_summary_json": str(pruning_summary_path),
                },
            )
        except Exception as exc:
            checkpoint.mark_failed(stage, exc)
            sys.exit(1)

    stage = "report_generation"
    if checkpoint.should_skip(stage, args.force_rerun):
        logger.info("[STEP 3/4] Skipping report generation from checkpoint")
    else:
        logger.info("[STEP 3/4] Generating comparison report")
        checkpoint.mark_started(stage)
        try:
            with baseline_scores_path.open() as scores_file:
                baseline_scores = json.load(scores_file)
            with pruned_scores_path.open() as scores_file:
                pruned_scores = json.load(scores_file)
            with pruning_summary_path.open() as summary_file:
                pruning_summary = json.load(summary_file)

            report = build_comparison_report(
                baseline_scores=baseline_scores,
                pruned_scores=pruned_scores,
                datasets=datasets,
                pruning_summary=pruning_summary,
                accuracy_threshold=args.accuracy_threshold,
            )
            save_comparison_report(report, output_dir)
            log_comparison_table(report)
            checkpoint.mark_complete(stage, {"comparison_report_json": str(output_dir / "comparison_report.json")})
        except Exception as exc:
            checkpoint.mark_failed(stage, exc)
            sys.exit(1)

    stage = "summary"
    logger.info("[STEP 4/4] Writing pipeline summary")
    checkpoint.mark_started(stage)
    try:
        summary = {
            "pipeline": "moe_pruning_gpu",
            "model": args.model,
            "source_prune_config": args.prune_config,
            "selected_prune_config": str(selected_prune_config_path),
            "experts_per_layer": args.experts_per_layer,
            "datasets": datasets,
            "num_samples": args.num_samples,
            "accuracy_threshold": args.accuracy_threshold,
            "device": args.device,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "output_dir": str(output_dir),
            "execution_time_seconds": round(time.time() - start_time, 2),
            "checkpoint": checkpoint.get_summary(),
            "notes": "Native GPT-OSS router masking only; latency tracking is not run.",
        }
        summary_path = output_dir / "pipeline_summary.json"
        with summary_path.open("w") as summary_file:
            json.dump(make_json_serializable(summary), summary_file, indent=2)
        checkpoint.mark_complete(stage, {"summary_json": str(summary_path)})
    except Exception as exc:
        checkpoint.mark_failed(stage, exc)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("Output dir: %s", output_dir)
    logger.info("Key outputs:")
    logger.info("  %s", output_dir / "selected_prune_config.json")
    logger.info("  %s", output_dir / "comparison_report.json")
    logger.info("  %s", output_dir / "comparison_report.csv")
    logger.info("  %s", output_dir / "pipeline_summary.json")
    logger.info("=" * 80)
    # END: End-to-end MoE pruning benchmark execution


if __name__ == "__main__":
    main()
# END: CLI and pipeline orchestration
