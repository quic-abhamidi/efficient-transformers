#!/usr/bin/env python3
"""Reproducible QEff model optimization workflow.

Stages:
  analyze  -> weak-layer report + candidate plans
  evaluate -> quality ranking + best_plan.json
  qaic     -> QEfficient compile/run benchmark for baseline and best plan

The script intentionally writes JSON after every stage so another team can
re-run, inspect, or validate the exact candidate without importing Python
objects from an interactive session.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from QEfficient.model_pruning.logging_utils import configure_file_logging, get_logger, set_verbose_logging

logger = get_logger("nas_pipeline")

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_device_group(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _manual_candidate_payload(skip_layers: list[int]) -> list[dict[str, Any]]:
    layers = sorted({int(layer) for layer in skip_layers})
    if not layers:
        raise ValueError("--skip-layers must contain at least one layer index")
    if layers[0] < 0:
        raise ValueError("--skip-layers must be non-negative")
    return [
        {
            "plan": {"transforms": [], "compatibility_mode": "strict", "metadata": {}},
            "priority": 0.0,
            "rationale": "baseline: no transform applied",
            "metadata": {"kind": "baseline"},
        },
        {
            "plan": {
                "transforms": [{"kind": "skip_layers", "layers": layers}],
                "compatibility_mode": "strict",
                "metadata": {},
            },
            "priority": 1.0,
            "rationale": f"manual skip layers {layers}",
            "metadata": {"kind": "manual", "layers": layers},
        },
    ]


def _manual_best_plan_payload(skip_layers: list[int]) -> dict[str, Any]:
    layers = sorted({int(layer) for layer in skip_layers})
    if not layers:
        raise ValueError("--skip-layers must contain at least one layer index")
    if layers[0] < 0:
        raise ValueError("--skip-layers must be non-negative")
    return {
        "plan_name": "manual_skip_layers_" + "_".join(str(layer) for layer in layers),
        "plan": {
            "transforms": [{"kind": "skip_layers", "layers": layers}],
            "compatibility_mode": "strict",
            "metadata": {},
        },
        "skip_layers": layers,
        "transform_kinds": ["skip_layers"],
        "selection_mode": "manual_skip_layers_qaic_only",
    }


def _accuracy_regression_pct(baseline_score: float | None, candidate_score: float | None) -> float | None:
    if baseline_score is None or candidate_score is None:
        return None
    if baseline_score == 0:
        return None
    return (baseline_score - candidate_score) / baseline_score * 100.0


def _perplexity_increase_pct(baseline_ppl: float, candidate_ppl: float) -> float | None:
    if baseline_ppl in (0, float("inf")) or candidate_ppl == float("inf"):
        return None
    return (candidate_ppl - baseline_ppl) / baseline_ppl * 100.0


def _pct_lower_is_better(baseline: float | None, optimized: float | None) -> float | None:
    if baseline is None or optimized is None or baseline == 0:
        return None
    return (baseline - optimized) / baseline * 100.0


def _pct_higher_is_better(baseline: float | None, optimized: float | None) -> float | None:
    if baseline is None or optimized is None or baseline == 0:
        return None
    return (optimized - baseline) / baseline * 100.0


def _qaic_metric_comparison(baseline: Any, optimized: Any) -> dict[str, Any]:
    baseline_stats = getattr(baseline, "avg_stats", {}) or {}
    optimized_stats = getattr(optimized, "avg_stats", {}) or {}

    baseline_compile = getattr(baseline, "compile_time_s", None)
    optimized_compile = getattr(optimized, "compile_time_s", None)
    baseline_ttft = baseline_stats.get("ttft")
    optimized_ttft = optimized_stats.get("ttft")
    baseline_decode_tps = baseline_stats.get("decode_tps")
    optimized_decode_tps = optimized_stats.get("decode_tps")
    baseline_total_tps = baseline_stats.get("total_tps")
    optimized_total_tps = optimized_stats.get("total_tps")
    baseline_e2e = baseline_stats.get("e2e")
    optimized_e2e = optimized_stats.get("e2e")

    return {
        "baseline_compile_time_s": baseline_compile,
        "optimized_compile_time_s": optimized_compile,
        "compile_time_delta_s": (
            optimized_compile - baseline_compile
            if baseline_compile is not None and optimized_compile is not None
            else None
        ),
        "compile_time_improvement_pct": _pct_lower_is_better(baseline_compile, optimized_compile),
        "baseline_ttft_s": baseline_ttft,
        "optimized_ttft_s": optimized_ttft,
        "ttft_delta_s": (
            optimized_ttft - baseline_ttft
            if baseline_ttft is not None and optimized_ttft is not None
            else None
        ),
        "ttft_improvement_pct": _pct_lower_is_better(baseline_ttft, optimized_ttft),
        "baseline_decode_tokens_per_sec": baseline_decode_tps,
        "optimized_decode_tokens_per_sec": optimized_decode_tps,
        "decode_tokens_per_sec_delta": (
            optimized_decode_tps - baseline_decode_tps
            if baseline_decode_tps is not None and optimized_decode_tps is not None
            else None
        ),
        "decode_tokens_per_sec_improvement_pct": _pct_higher_is_better(baseline_decode_tps, optimized_decode_tps),
        "baseline_total_tokens_per_sec": baseline_total_tps,
        "optimized_total_tokens_per_sec": optimized_total_tps,
        "total_tokens_per_sec_delta": (
            optimized_total_tps - baseline_total_tps
            if baseline_total_tps is not None and optimized_total_tps is not None
            else None
        ),
        "total_tokens_per_sec_improvement_pct": _pct_higher_is_better(baseline_total_tps, optimized_total_tps),
        "baseline_e2e_s": baseline_e2e,
        "optimized_e2e_s": optimized_e2e,
        "e2e_delta_s": (
            optimized_e2e - baseline_e2e
            if baseline_e2e is not None and optimized_e2e is not None
            else None
        ),
        "e2e_improvement_pct": _pct_lower_is_better(baseline_e2e, optimized_e2e),
    }


def _evaluation_comparison(results: list[Any], best: Any | None, baseline_name: str = "baseline") -> dict[str, Any]:
    baseline = next((r for r in results if r.plan_name == baseline_name and r.error is None), None)
    if baseline is None or best is None:
        return {"baseline_plan_name": baseline_name, "selected_plan_name": best.plan_name if best else None}
    return {
        "baseline_plan_name": baseline.plan_name,
        "selected_plan_name": best.plan_name,
        "baseline_accuracy_score": baseline.accuracy_score,
        "selected_accuracy_score": best.accuracy_score,
        "accuracy_metric": best.accuracy_metric or baseline.accuracy_metric,
        "accuracy_regression_pct": _accuracy_regression_pct(baseline.accuracy_score, best.accuracy_score),
        "baseline_perplexity": baseline.overall_perplexity,
        "selected_perplexity": best.overall_perplexity,
        "perplexity_increase_pct": _perplexity_increase_pct(baseline.overall_perplexity, best.overall_perplexity),
    }


def _format_plan_errors(results: list[Any]) -> str:
    errors = [f"{r.plan_name}: {r.error}" for r in results if getattr(r, "error", None)]
    if not errors:
        return ""
    return " Plan errors: " + "; ".join(errors)


def _normalize_scores_by_layer(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return {layer: 0.0 for layer in scores}
    return {layer: (value - lo) / (hi - lo) for layer, value in scores.items()}


def _combine_weak_layer_reports(reports: dict[str, Any]) -> Any:
    """Build one normalized weak-layer report from cosine and L2 reports."""
    if set(reports) != {"cosine", "l2"}:
        raise ValueError("combined weak-layer report requires cosine and l2 reports")

    from QEfficient.model_pruning.qeff_model_optimizer.analysis import RankedLayer, WeakLayerReport

    first = reports["cosine"]
    datasets = list(first.datasets)
    layer_ids = {entry.layer for entry in first.ranked_layers}
    for metric, report in reports.items():
        if list(report.datasets) != datasets:
            raise ValueError(f"{metric} report datasets differ from cosine report")
        if {entry.layer for entry in report.ranked_layers} != layer_ids:
            raise ValueError(f"{metric} report layers differ from cosine report")

    by_metric = {
        metric: {entry.layer: entry for entry in report.ranked_layers}
        for metric, report in reports.items()
    }
    per_dataset_combined: dict[str, dict[int, float]] = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        normalized_by_metric = {}
        for metric, entries_by_layer in by_metric.items():
            raw = {
                layer: entries_by_layer[layer].per_dataset_scores[dataset]
                for layer in layer_ids
                if dataset in entries_by_layer[layer].per_dataset_scores
            }
            normalized_by_metric[metric] = _normalize_scores_by_layer(raw)
        for layer in layer_ids:
            values = [
                normalized_by_metric[metric][layer]
                for metric in ("cosine", "l2")
                if layer in normalized_by_metric[metric]
            ]
            if values:
                per_dataset_combined[dataset][layer] = sum(values) / len(values)

    rows = []
    for layer in sorted(layer_ids):
        per_dataset_scores = {
            dataset: per_dataset_combined[dataset][layer]
            for dataset in datasets
            if layer in per_dataset_combined[dataset]
        }
        aggregate = sum(per_dataset_scores.values()) / len(per_dataset_scores) if per_dataset_scores else 0.0
        rows.append((layer, aggregate, per_dataset_scores))
    rows.sort(key=lambda row: (row[1], row[0]))

    ranked_layers = [
        RankedLayer(
            layer=layer,
            aggregate_score=aggregate,
            rank=rank,
            per_dataset_scores=per_dataset_scores,
        )
        for rank, (layer, aggregate, per_dataset_scores) in enumerate(rows, start=1)
    ]
    metadata = dict(getattr(first, "metadata", {}))
    metadata.update({"metric": "both", "source_metrics": ["cosine", "l2"], "combination": "normalized_average"})
    return WeakLayerReport(
        model_spec=first.model_spec,
        datasets=datasets,
        ranked_layers=ranked_layers,
        metadata=metadata,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    set_verbose_logging(getattr(args, "verbose", False))
    if getattr(args, "verbose", False):
        configure_file_logging(args.output_dir)
    from QEfficient.model_pruning.qeff_model_optimizer.analysis import WeakLayerReport, analyze_weak_layers
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.search import generate_candidate_plans

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    spec = ModelSpec(
        model_id=args.model,
        revision=args.revision,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )

    logger.info(f"[analyze] model={args.model}")
    logger.info(f"[analyze] datasets={args.datasets} samples={args.num_samples} batch={args.batch_size}")
    t0 = time.time()
    analysis_metrics = ["cosine", "l2"] if args.metric == "both" else [args.metric]
    reports: dict[str, WeakLayerReport] = {}
    for metric in analysis_metrics:
        reports[metric] = analyze_weak_layers(
            spec,
            datasets=args.datasets,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            metric=metric,
            max_length=args.max_length,
            output_dir=output_dir / "analysis_artifacts",
            verbose=args.verbose,
        )

    if args.metric == "both":
        report = _combine_weak_layer_reports(reports)
        candidates = generate_candidate_plans(
            report,
            max_skip_layers=args.max_skip_layers,
            top_k=args.top_k,
            include_baseline=True,
            metric="both",
        )
        for metric, metric_report in reports.items():
            candidates.extend(
                generate_candidate_plans(
                    metric_report,
                    max_skip_layers=args.max_skip_layers,
                    top_k=args.top_k,
                    include_baseline=False,
                    metric=metric,
                )
            )
            _write_json(output_dir / f"weak_layer_report_{metric}.json", metric_report.to_dict())
    else:
        report = reports[args.metric]
        candidates = generate_candidate_plans(
            report,
            max_skip_layers=args.max_skip_layers,
            top_k=args.top_k,
            include_baseline=True,
            metric=args.metric,
        )

    _write_json(output_dir / "weak_layer_report.json", report.to_dict())
    _write_json(output_dir / "candidate_plans.json", [c.to_dict() for c in candidates])
    _write_json(
        output_dir / "run_metadata.json",
        {
            "stage": "analyze",
            "model": args.model,
            "revision": args.revision,
            "datasets": args.datasets,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "metric": args.metric,
            "max_length": args.max_length,
            "elapsed_s": round(time.time() - t0, 2),
        },
    )

    logger.info(f"[analyze] wrote {output_dir / 'weak_layer_report.json'}")
    logger.info(f"[analyze] wrote {output_dir / 'candidate_plans.json'}")
    logger.info("[analyze] weakest layers: %s", [r.layer for r in report.weakest(min(10, len(report.ranked_layers)))])


def cmd_evaluate(args: argparse.Namespace) -> None:
    set_verbose_logging(getattr(args, "verbose", False))
    if getattr(args, "verbose", False):
        configure_file_logging(args.output_dir)
    import torch

    from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import plan_to_dict
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import PlanEvaluator
    from QEfficient.model_pruning.qeff_model_optimizer.search import CandidatePlan

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_layers and args.candidate_plans:
        raise ValueError("Use either --candidate-plans or --skip-layers, not both")
    if args.skip_layers:
        raw_candidates = _manual_candidate_payload(args.skip_layers)
        manual_candidates_path = output_dir / "manual_candidate_plans.json"
        _write_json(manual_candidates_path, raw_candidates)
        logger.info("[evaluate] using manual skip layers=%s", sorted({int(layer) for layer in args.skip_layers}))
        logger.info("[evaluate] wrote %s", manual_candidates_path)
    elif args.candidate_plans:
        raw_candidates = _load_json(Path(args.candidate_plans))
    else:
        raise ValueError("Either --candidate-plans or --skip-layers must be provided")

    candidates = [CandidatePlan.from_dict(item) for item in raw_candidates]
    if args.skip_layers:
        selected_entries = list(enumerate(candidates))
    else:
        baseline_entry = next(
            (
                (idx, candidate)
                for idx, candidate in enumerate(candidates)
                if candidate.metadata.get("kind") == "baseline"
            ),
            None,
        )
        selected_entries = []
        if baseline_entry is not None:
            selected_entries.append(baseline_entry)
        selected_entries.extend(
            (idx, candidate)
            for idx, candidate in enumerate(candidates)
            if candidate.metadata.get("kind") != "baseline"
        )
        selection_limit = (1 if baseline_entry is not None else 0) + args.max_candidates
        selected_entries = selected_entries[:selection_limit]

    plans = {
        _candidate_name(idx, candidate): candidate.plan
        for idx, candidate in selected_entries
    }

    spec = ModelSpec(
        model_id=args.model,
        revision=args.revision,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )

    logger.info(f"[evaluate] loading model={args.model}")
    loader = TransformersModelLoader()
    model, tokenizer = loader.load(spec)
    if torch.cuda.is_available():
        logger.info(f"[evaluate] cuda memory allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    evaluator = PlanEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_spec=spec,
        datasets=args.datasets,
        num_samples=args.num_samples,
        max_length=args.max_length,
        generation_len=args.generation_len,
        eval_method=args.eval_method,
        accuracy_metric=args.accuracy_metric,
        lm_eval_batch_size=args.lm_eval_batch_size,
        lm_eval_limit=args.lm_eval_limit,
        videomme_dataset_path=args.videomme_dataset_path,
        videomme_video_root=args.videomme_video_root,
        videomme_split=args.videomme_split,
        videomme_num_frames=args.videomme_num_frames,
        videomme_fps=args.videomme_fps,
        videomme_use_subtitles=args.videomme_use_subtitles,
    )

    logger.info(f"[evaluate] evaluating {len(plans)} plans on datasets={args.datasets} method={args.eval_method}")
    t0 = time.time()
    results = evaluator.evaluate_all(plans)
    if args.skip_layers:
        best = next((r for r in results if r.plan_name != "baseline" and r.error is None), None)
    else:
        best = evaluator.select_best(results, accuracy_threshold=args.accuracy_threshold)
    comparison = _evaluation_comparison(results, best)

    _write_json(output_dir / "plan_results.json", [r.to_dict() for r in results])
    _write_json(output_dir / "comparison_report.json", comparison)
    _write_json(output_dir / "accuracy_regression_report.json", comparison)
    _write_json(
        output_dir / "evaluation_summary.json",
        {
            "stage": "evaluate",
            "model": args.model,
            "datasets": args.datasets,
            "num_samples": args.num_samples,
            "accuracy_threshold": None if args.skip_layers else args.accuracy_threshold,
            "selection_mode": "manual_skip_layers" if args.skip_layers else "threshold",
            "eval_method": args.eval_method,
            "accuracy_metric": args.accuracy_metric,
            "videomme_dataset_path": args.videomme_dataset_path if args.eval_method == "videomme" else None,
            "videomme_video_root": args.videomme_video_root if args.eval_method == "videomme" else None,
            "videomme_use_subtitles": args.videomme_use_subtitles if args.eval_method == "videomme" else None,
            "elapsed_s": round(time.time() - t0, 2),
            "best_plan_name": best.plan_name if best else None,
            "comparison": comparison,
        },
    )

    if best is not None:
        payload = {
            "plan_name": best.plan_name,
            "plan": plan_to_dict(best.plan),
            "skip_layers": best.skip_layers,
            "transform_kinds": best.transform_kinds,
            "overall_perplexity": best.overall_perplexity,
            "accuracy_score": best.accuracy_score,
            "accuracy_metric": best.accuracy_metric,
            "videomme_report": best.videomme_report.to_dict() if best.videomme_report else None,
            "eval_method": args.eval_method,
            "accuracy_threshold": None if args.skip_layers else args.accuracy_threshold,
            "selection_mode": "manual_skip_layers" if args.skip_layers else "threshold",
            "comparison": comparison,
        }
        _write_json(output_dir / "best_plan.json", payload)
        logger.info(f"[evaluate] selected best plan={best.plan_name} skip_layers={best.skip_layers}")
        logger.info(f"[evaluate] wrote {output_dir / 'best_plan.json'}")
    else:
        if args.skip_layers:
            message = "Manual skip-layer evaluation failed; best_plan.json was not written"
        else:
            message = "No non-baseline plan fit the accuracy threshold; best_plan.json was not written"
        message += _format_plan_errors(results)
        logger.error("[evaluate] %s", message)
        logger.info(f"[evaluate] wrote {output_dir / 'plan_results.json'}")
        raise RuntimeError(message)

    logger.info(f"[evaluate] wrote {output_dir / 'plan_results.json'}")


def _candidate_name(idx: int, candidate: Any) -> str:
    kind = candidate.metadata.get("kind", "candidate")
    if kind == "baseline":
        return "baseline"
    metric = candidate.metadata.get("metric")
    metric_prefix = f"{metric}_" if metric else ""
    layers = candidate.metadata.get("layers")
    layer_suffix = "_".join(str(layer) for layer in layers) if layers else str(idx)
    return f"{idx:02d}_{metric_prefix}{kind}_{layer_suffix}"


def cmd_qaic(args: argparse.Namespace) -> None:
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan, plan_from_dict
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import QAICBenchmarkRunner

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_layers:
        plan_payload = _manual_best_plan_payload(args.skip_layers)
        _write_json(output_dir / "manual_best_plan.json", plan_payload)
        plan_file = None
    elif args.plan:
        plan_payload = _load_json(Path(args.plan))
        plan_file = str(Path(args.plan))
    else:
        raise ValueError("qaic requires either --plan or --skip-layers")

    plan = plan_from_dict(plan_payload["plan"])
    plan_name = str(plan_payload.get("plan_name", "optimized"))
    device_group = _parse_device_group(args.device_group)

    runner = QAICBenchmarkRunner(
        model_id=args.model,
        prompts=args.prompts,
        generation_len=args.generation_len,
        ctx_len=args.ctx_len,
        prefill_seq_len=args.prefill_seq_len,
        num_cores=args.num_cores,
        compile_dir_base=args.compile_dir_base,
        mxfp6_matmul=not args.no_mxfp6_matmul,
        mxint8_kv_cache=not args.no_mxint8_kv_cache,
        videomme_dataset_path=args.videomme_dataset_path,
        videomme_video_root=args.videomme_video_root,
        videomme_split=args.videomme_split,
        videomme_num_samples=args.videomme_num_samples,
        videomme_num_frames=args.videomme_num_frames,
        videomme_fps=args.videomme_fps,
        videomme_use_subtitles=args.videomme_use_subtitles,
    )

    logger.info(f"[qaic] compiling baseline on devices={device_group} batch={args.batch_size}")
    baseline = runner.run(
        name="baseline",
        plan=TransformationPlan(),
        device_group=device_group,
        batch_size=args.batch_size,
    )
    _write_json(output_dir / "baseline.json", baseline.to_dict())

    logger.info(f"[qaic] compiling optimized plan={plan_name} on devices={device_group} batch={args.batch_size}")
    optimized = runner.run(
        name=plan_name,
        plan=plan,
        device_group=device_group,
        batch_size=args.batch_size,
    )
    _write_json(output_dir / f"{plan_name}.json", optimized.to_dict())

    speedups = runner.compute_speedups([baseline, optimized], baseline_name="baseline")
    benchmark_comparison = {
        "baseline_plan_name": "baseline",
        "optimized_plan_name": plan_name,
        "skip_layers": list(plan_payload.get("skip_layers", [])),
        "qaic_stage": "baseline_qeff_qaic_vs_optimized_qeff_qaic",
        "speedups": speedups.get(plan_name, {}),
        "metrics": _qaic_metric_comparison(baseline, optimized),
        "baseline_avg_stats": getattr(baseline, "avg_stats", {}),
        "optimized_avg_stats": getattr(optimized, "avg_stats", {}),
        "baseline_accuracy_score": getattr(baseline, "accuracy_score", None),
        "optimized_accuracy_score": getattr(optimized, "accuracy_score", None),
        "accuracy_metric": getattr(optimized, "accuracy_metric", None) or getattr(baseline, "accuracy_metric", None),
        "baseline_videomme_report": getattr(baseline, "videomme_report", None),
        "optimized_videomme_report": getattr(optimized, "videomme_report", None),
    }
    _write_json(output_dir / "benchmark_comparison.json", benchmark_comparison)
    _write_json(
        output_dir / "all_results.json",
        {
            "model": args.model,
            "plan_file": plan_file,
            "selection_mode": plan_payload.get("selection_mode"),
            "device_group": device_group,
            "batch_size": args.batch_size,
            "results": [baseline.to_dict(), optimized.to_dict()],
            "speedups": speedups,
            "comparison": benchmark_comparison,
        },
    )

    if baseline.error:
        logger.error(f"[qaic] baseline error={baseline.error}")
    if optimized.error:
        logger.error(f"[qaic] optimized error={optimized.error}")
    logger.info(f"[qaic] wrote {output_dir / 'all_results.json'}")
    logger.info(f"[qaic] optimized qpc_path={optimized.qpc_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Generate weak-layer report and candidate plans", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--revision", default=None)
    analyze.add_argument("--datasets", nargs="+", default=["gsm8k", "hellaswag"])
    analyze.add_argument("--num-samples", type=int, default=64)
    analyze.add_argument("--batch-size", type=int, default=4)
    analyze.add_argument("--max-length", type=int, default=512)
    analyze.add_argument("--metric", choices=["cosine", "l2", "both"], default="cosine")
    analyze.add_argument("--dtype", default="bfloat16")
    analyze.add_argument("--device-map", default="auto")
    analyze.add_argument("--no-trust-remote-code", action="store_true")
    analyze.add_argument("--max-skip-layers", type=int, default=3)
    analyze.add_argument("--top-k", type=int, default=8)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--verbose", action="store_true", help="Log dataset and batch progress during analysis")
    analyze.set_defaults(func=cmd_analyze)

    evaluate = sub.add_parser("evaluate", help="Evaluate candidate plans and save best_plan.json", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--revision", default=None, help="Optional Hugging Face revision, branch, tag, or commit SHA. Leave unset for repo default.")
    evaluate.add_argument("--candidate-plans", default=None, help="Path to candidate_plans.json generated by the analyze stage. Mutually exclusive with --skip-layers.")
    evaluate.add_argument("--skip-layers", nargs="+", type=int, default=None, help="Manual decoder layer indices to skip. Generates a baseline plus manual skip candidate internally.")
    evaluate.add_argument("--datasets", nargs="+", default=["gsm8k", "hellaswag"], help="Evaluation datasets/tasks. For lm_eval these are resolved to lm_eval task names.")
    evaluate.add_argument("--num-samples", type=int, default=50, help="Number of samples per evaluation dataset. Also used as lm_eval limit when --lm-eval-limit is unset.")
    evaluate.add_argument("--max-candidates", type=int, default=5, help="Number of non-baseline candidate plans to evaluate.")
    evaluate.add_argument("--accuracy-threshold", type=float, default=5.0, help="Allowed quality loss in percent for candidate-plan selection. Ignored when --skip-layers is used because manual layers are always evaluated and reported.")
    evaluate.add_argument("--max-length", type=int, default=512)
    evaluate.add_argument("--generation-len", type=int, default=40)
    evaluate.add_argument("--eval-method", choices=["perplexity", "lm_eval", "videomme"], default="lm_eval", help="Evaluation backend. Use lm_eval for text accuracy, videomme for video QA accuracy, or perplexity for PPL checks.")
    evaluate.add_argument("--accuracy-metric", default="auto", help="Metric extracted from lm_eval results. auto prefers acc_norm, acc, exact_match, em, then mc2; set acc for plain accuracy.")
    evaluate.add_argument("--lm-eval-batch-size", type=int, default=1, help="Batch size passed to lm_eval. Keep 1 for large models or memory-limited runs.")
    evaluate.add_argument("--lm-eval-limit", type=int, default=None, help="Optional lm_eval sample limit. Leave unset to use --num-samples.")
    evaluate.add_argument("--videomme-dataset-path", default=None, help="Local Video-MME JSON/JSONL file or directory. If omitted, loads lmms-lab/Video-MME from Hugging Face.")
    evaluate.add_argument("--videomme-video-root", default=None, help="Directory containing Video-MME video files referenced by the dataset rows.")
    evaluate.add_argument("--videomme-split", default="test", help="Hugging Face split for Video-MME when --videomme-dataset-path is omitted.")
    evaluate.add_argument("--videomme-num-frames", type=int, default=8, help="Number of uniformly sampled frames per video.")
    evaluate.add_argument("--videomme-fps", type=float, default=None, help="Optional fixed FPS sampling before frame-count downsampling.")
    evaluate.add_argument("--videomme-use-subtitles", action="store_true", help="Inject subtitles/captions into the Video-MME prompt when available.")
    evaluate.add_argument("--dtype", default="bfloat16", help="Torch dtype used when loading the Hugging Face model.")
    evaluate.add_argument("--device-map", default="auto", help="Hugging Face device_map for model loading, for example auto or cpu.")
    evaluate.add_argument("--no-trust-remote-code", action="store_true")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    qaic = sub.add_parser("qaic", help="Compile and run baseline plus best plan on QAIC", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    qaic.add_argument("--model", required=True)
    qaic_plan = qaic.add_mutually_exclusive_group(required=True)
    qaic_plan.add_argument("--plan", default=None, help="Path to best_plan.json")
    qaic_plan.add_argument("--skip-layers", nargs="+", type=int, default=None, help="Manual decoder layer indices to skip for a QAIC-only comparison. Skips HF evaluate and writes manual_best_plan.json in --output-dir.")
    qaic.add_argument("--device-group", default="0", help="Comma-separated QAIC device IDs, e.g. 0 or 0,1,2,3")
    qaic.add_argument("--batch-size", type=int, default=1)
    qaic.add_argument("--ctx-len", type=int, default=4096)
    qaic.add_argument("--prefill-seq-len", type=int, default=128)
    qaic.add_argument("--num-cores", type=int, default=16)
    qaic.add_argument("--generation-len", type=int, default=60)
    qaic.add_argument("--compile-dir-base", default=None)
    qaic.add_argument("--prompts", nargs="+", default=["The capital of France is"])
    qaic.add_argument("--no-mxfp6-matmul", action="store_true")
    qaic.add_argument("--no-mxint8-kv-cache", action="store_true")
    qaic.add_argument("--videomme-dataset-path", default=None, help="Local Video-MME JSON/JSONL file or directory for QAIC VLM accuracy.")
    qaic.add_argument("--videomme-video-root", default=None, help="Directory containing Video-MME video files referenced by dataset rows.")
    qaic.add_argument("--videomme-split", default="test")
    qaic.add_argument("--videomme-num-samples", type=int, default=None, help="Number of Video-MME samples to run on QAIC. Defaults to all loaded rows.")
    qaic.add_argument("--videomme-num-frames", type=int, default=8)
    qaic.add_argument("--videomme-fps", type=float, default=None)
    qaic.add_argument("--videomme-use-subtitles", action="store_true")
    qaic.add_argument("--output-dir", required=True)
    qaic.set_defaults(func=cmd_qaic)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
