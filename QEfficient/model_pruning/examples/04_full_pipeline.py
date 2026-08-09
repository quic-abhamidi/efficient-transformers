#!/usr/bin/env python3
"""Example 04: Complete end-to-end NAS optimization pipeline.

Ties examples 01-03 into a single script: analyze → evaluate plans → select
best → compile on QAIC. This is the script to copy when building your own
production optimization pipeline.

Phases:
  1. GPU Analysis      — find weak layers/heads/channels
  2. GPU Evaluation    — score candidate plans by perplexity
  3. QAIC Deployment   — compile best plan + baseline, measure speedup
  4. Final Report      — aggregated results, charts, JSON

Usage:
    python examples/04_full_pipeline.py [--model MODEL_ID]
    python examples/04_full_pipeline.py --skip-qaic  # skip phase 3

Requires: GPU + QAIC hardware (for all phases) or GPU only (with --skip-qaic).
"""

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from uuid import uuid4

warnings.filterwarnings("ignore")

import torch

# ── NAS imports ──────────────────────────────────────────────────────────────
from QEfficient.model_pruning.qeff_model_optimizer.analysis import (
    compute_channel_importance,
    compute_head_importance,
    compute_kv_head_similarity,
    compute_weak_layer_report,
)
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    HeadPruningSpec, LayerHeadSelection, MlpPruningSpec,
    SkipLayersSpec, TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.evaluation import PlanEvaluator, QAICBenchmarkRunner, charts
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry


# ── Configuration ────────────────────────────────────────────────────────────

# Datasets used for BOTH analysis (Phase 1) and evaluation (Phase 2).
# Using the same datasets for both phases gives a coherent story: "we
# identified weak layers on X, we measured degradation on X, we deployed
# the winner on hardware."
EVAL_DATASETS = ["wikitext", "mmlu_pro", "bbh_causal", "ifeval", "gsm_hard"]

# Small numbers make the demo runnable in 30-60 min. For production analysis,
# bump analysis_samples to 50-100 and eval_samples to 200-500.
ANALYSIS_SAMPLES = 16
EVAL_SAMPLES = 50
MAX_LENGTH = 512

# Accuracy threshold for plan selection. 10% means: pick the most aggressive plan
# whose overall PPL is within 10% of baseline. Tighten for quality-critical
# use cases (5%), loosen for aggressive throughput (20%+).
ACCURACY_THRESHOLD_PCT = 10.0


def section(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ── Phase 1: GPU Analysis ────────────────────────────────────────────────────

def phase1_analysis(model_id: str, output_dir: Path):
    """Run the 4-part NAS analysis suite.

    Returns all 4 reports + the model spec. Model is loaded, analyzed, then
    left in memory for phase 2 (which reuses the same instance).
    """
    section("PHASE 1: GPU Analysis")

    # Load model.
    spec = ModelSpec(model_id=model_id, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()
    t0 = time.time()
    model, tokenizer = loader.load(spec)
    print(f"  Model loaded in {time.time()-t0:.1f}s | "
          f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    print(f"  Layers={model.config.num_hidden_layers}, "
          f"Heads={model.config.num_attention_heads}, "
          f"KV={getattr(model.config, 'num_key_value_heads', '?')}")

    # Wrap in session + artifact so analysis APIs can use it.
    applier = TransformApplier(default_transform_registry())
    session = NASSession(loader=loader, transform_applier=applier)
    artifact = ModelArtifact(
        artifact_id=uuid4().hex,
        model=model, tokenizer=tokenizer,
        model_spec=spec, plan=TransformationPlan(),
    )
    session.artifacts[artifact.artifact_id] = artifact

    # Run the 4 analyses.
    t0 = time.time()
    print(f"\n  [1/4] Weak layer analysis...")
    weak = compute_weak_layer_report(
        artifact, datasets=EVAL_DATASETS[:3],  # use a subset for speed
        num_samples=ANALYSIS_SAMPLES, batch_size=2, max_length=256,
    )
    print(f"    Weakest 5: {[(r.layer, round(r.aggregate_score, 3)) for r in weak.ranked_layers[:5]]}")

    print(f"  [2/4] Head importance...")
    head = compute_head_importance(
        artifact, datasets=EVAL_DATASETS[:3],
        num_samples=ANALYSIS_SAMPLES, batch_size=2, max_length=256,
    )

    print(f"  [3/4] Channel importance...")
    channel = compute_channel_importance(
        artifact, datasets=EVAL_DATASETS[:3],
        num_samples=ANALYSIS_SAMPLES, batch_size=2, max_length=256,
    )

    print(f"  [4/4] KV similarity (weight-only, instant)...")
    kv = compute_kv_head_similarity(artifact)

    print(f"\n  Analysis done in {time.time()-t0:.1f}s")

    # Save reports + charts.
    analysis_dir = output_dir / "phase1_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    save_json({
        "ranked_layers": [r.to_dict() for r in weak.ranked_layers],
        "metadata": weak.metadata,
    }, analysis_dir / "weak_layer_report.json")
    save_json(head.to_dict(), analysis_dir / "head_importance_report.json")
    save_json(channel.to_dict(), analysis_dir / "channel_importance_report.json")
    save_json(kv.to_dict(), analysis_dir / "kv_similarity_report.json")

    try:
        charts.chart_weak_layers(
            weak, analysis_dir / "weak_layers.png",
            title_prefix=model_id, highlight_weakest_n=5,
        )
        charts.chart_head_importance_heatmap(
            head, analysis_dir / "head_importance_heatmap.png",
            title_prefix=model_id,
        )
    except Exception as e:
        print(f"    [warn] chart failed: {e}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_spec": spec,
        "session": session,
        "artifact": artifact,
        "weak_report": weak,
        "head_report": head,
        "channel_report": channel,
        "kv_report": kv,
    }


# ── Phase 2: GPU Plan Evaluation ─────────────────────────────────────────────

def phase2_plan_evaluation(analysis_result: dict, output_dir: Path):
    """Build candidate plans, evaluate each, select the best."""
    section("PHASE 2: Plan Evaluation")

    model = analysis_result["model"]
    tokenizer = analysis_result["tokenizer"]
    spec = analysis_result["model_spec"]
    weak = analysis_result["weak_report"]
    head = analysis_result["head_report"]

    weakest = [r.layer for r in weak.ranked_layers]
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    # Build plans covering conservative → aggressive.
    plans: dict[str, TransformationPlan] = {
        "baseline": TransformationPlan(),
    }

    # Skip variants (single layer and spread).
    plans["skip_1"] = TransformationPlan(
        transforms=[SkipLayersSpec(layers=[weakest[0]])]
    )
    if len(weakest) >= 3:
        # Non-contiguous picks: spread across the model for robustness.
        third = num_layers // 3
        early = next((l for l in weakest if l < third), weakest[0])
        middle = next((l for l in weakest if third <= l < 2*third), weakest[1])
        late = next((l for l in weakest if l >= 2*third), weakest[2])
        spread = sorted(set([early, middle, late]))
        if len(spread) == 3:
            plans["skip_3_spread"] = TransformationPlan(
                transforms=[SkipLayersSpec(layers=spread)]
            )
    plans["skip_3_topN"] = TransformationPlan(
        transforms=[SkipLayersSpec(layers=sorted(weakest[:3]))]
    )

    # Head pruning at 15%.
    prune_count = max(1, int(num_heads * 0.15))
    plans["head_prune_15"] = TransformationPlan(
        transforms=[HeadPruningSpec(selections=[
            LayerHeadSelection(
                layer=layer_idx,
                heads=[h for h, _s in scores[:prune_count]],
            )
            for layer_idx, scores in head.per_layer_scores.items()
        ])]
    )

    # Combined.
    plans["skip_1_mlp_15"] = TransformationPlan(
        transforms=[
            SkipLayersSpec(layers=[weakest[0]]),
            MlpPruningSpec(target_layers=[], pruning_ratio=0.15),
        ]
    )

    print(f"\n  Plans: {len(plans)}")
    for name, plan in plans.items():
        print(f"    {name:20s} -> {[t.kind for t in plan.transforms] or ['none']}")

    # Evaluate.
    evaluator = PlanEvaluator(
        model=model, tokenizer=tokenizer, model_spec=spec,
        datasets=EVAL_DATASETS,
        num_samples=EVAL_SAMPLES, max_length=MAX_LENGTH,
    )

    t0 = time.time()
    results = evaluator.evaluate_all(plans)
    print(f"\n  Evaluated {len(results)} plans in {(time.time()-t0)/60:.1f} min")

    # Report.
    baseline = next(r for r in results if r.plan_name == "baseline")
    print(f"\n  Plan               PPL       Delta    Transforms")
    print(f"  ---------------- -------  --------   ----------")
    for r in results:
        if r.error:
            print(f"  {r.plan_name:<16s} ERROR: {r.error[:40]}")
            continue
        delta = (r.overall_perplexity - baseline.overall_perplexity) / baseline.overall_perplexity * 100
        print(f"  {r.plan_name:<16s} {r.overall_perplexity:>7.2f}   {delta:>+7.1f}%   "
              f"{r.transform_kinds}")

    # Select best within threshold.
    best = evaluator.select_best(results, accuracy_threshold=ACCURACY_THRESHOLD_PCT)
    if best is None:
        # If nothing fits threshold, fall back to skip_1 (usually safe).
        best = next((r for r in results if r.plan_name == "skip_1"), baseline)

    print(f"\n  Best plan within {ACCURACY_THRESHOLD_PCT}% budget: {best.plan_name}")
    print(f"    PPL: {baseline.overall_perplexity:.2f} -> {best.overall_perplexity:.2f}")

    # Save phase 2 outputs.
    eval_dir = output_dir / "phase2_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json([r.to_dict() for r in results], eval_dir / "plan_results.json")
    save_json(best.to_dict(), eval_dir / "best_plan.json")

    try:
        charts.chart_perplexity_comparison(
            results, eval_dir / "perplexity_comparison.png",
            title_prefix=spec.model_id,
        )
        charts.chart_per_dataset_perplexity(
            results, eval_dir / "per_dataset_ppl.png",
            title_prefix=spec.model_id,
        )
    except Exception as e:
        print(f"    [warn] chart failed: {e}")

    # Cleanup GPU so phase 3 has room.
    analysis_result["session"].close()
    del analysis_result["model"], analysis_result["artifact"]
    gc.collect()
    torch.cuda.empty_cache()

    return best, results


# ── Phase 3: QAIC Deployment ─────────────────────────────────────────────────

def phase3_qaic(model_id: str, best_plan_result, output_dir: Path):
    """Compile baseline + best plan on QAIC, measure hardware speedup."""
    section("PHASE 3: QAIC Deployment")

    # Rebuild the plan (best_plan_result.plan is still a TransformationPlan).
    best_plan = best_plan_result.plan

    # Single QAIC runner handles everything.
    runner = QAICBenchmarkRunner(
        model_id=model_id,
        compile_dir_base="results/model_pruning/examples/nas_example_04",
    )

    results = []

    # Baseline: full model, 1 device, BS=1 (low-latency reference).
    print(f"\n  Compiling BASELINE (1 device, BS=1)...")
    t0 = time.time()
    baseline = runner.run(
        name="baseline",
        plan=TransformationPlan(),
        device_group=[0], batch_size=1,
    )
    print(f"    Done in {(time.time()-t0)/60:.1f} min")
    if baseline.error:
        print(f"    ERROR: {baseline.error}")
    else:
        s = baseline.avg_stats
        print(f"    TTFT={s.get('ttft', 0):.3f}s  "
              f"Decode={s.get('decode_tps', 0):.1f}/s  "
              f"E2E={s.get('e2e', 0):.2f}s")
    results.append(baseline)

    # Optimized: the plan from phase 2.
    print(f"\n  Compiling OPTIMIZED ({best_plan_result.plan_name}) on 1 device...")
    t0 = time.time()
    optimized = runner.run(
        name=f"{best_plan_result.plan_name}_1dev",
        plan=best_plan,
        device_group=[0], batch_size=1,
    )
    print(f"    Done in {(time.time()-t0)/60:.1f} min")
    if optimized.error:
        print(f"    ERROR: {optimized.error}")
    else:
        s = optimized.avg_stats
        print(f"    TTFT={s.get('ttft', 0):.3f}s  "
              f"Decode={s.get('decode_tps', 0):.1f}/s  "
              f"E2E={s.get('e2e', 0):.2f}s")
    results.append(optimized)

    # Compute speedups.
    speedups = runner.compute_speedups(results, baseline_name="baseline")

    print(f"\n  Speedup vs baseline:")
    for name, sp in speedups.items():
        print(f"    {name:<25s} TTFT {sp['ttft_pct']:+.1f}%  "
              f"Decode {sp['decode_pct']:+.1f}%  E2E {sp['e2e_pct']:+.1f}%")

    # Save.
    qaic_dir = output_dir / "phase3_qaic"
    qaic_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        save_json(r.to_dict(), qaic_dir / f"{r.plan_name}.json")
    save_json(speedups, qaic_dir / "speedups.json")

    try:
        charts.chart_qaic_performance(
            results, qaic_dir / "qaic_comparison.png",
            title_prefix=model_id,
        )
    except Exception as e:
        print(f"    [warn] chart failed: {e}")

    return results, speedups


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model id")
    parser.add_argument("--output-dir", default="results/example_04_full",
                        help="Where to save all phase outputs")
    parser.add_argument("--skip-qaic", action="store_true",
                        help="Skip phase 3 (QAIC compilation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*72}")
    print(f"  NAS Full Pipeline")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {output_dir}")
    print(f"  Budget:      {ACCURACY_THRESHOLD_PCT}% PPL increase")
    print(f"{'='*72}")

    t_start = time.time()

    # Phase 1: Analysis.
    analysis = phase1_analysis(args.model, output_dir)

    # Phase 2: Plan evaluation (reuses model from phase 1).
    best, all_plan_results = phase2_plan_evaluation(analysis, output_dir)

    # Phase 3: QAIC deployment (loads fresh CPU model).
    qaic_results = []
    speedups = {}
    if not args.skip_qaic:
        qaic_results, speedups = phase3_qaic(args.model, best, output_dir)
    else:
        print("\n[skip] Phase 3 QAIC compilation disabled (--skip-qaic)")

    # ── Phase 4: Final summary ──────────────────────────────────────────────
    section("PHASE 4: Final Report")

    baseline_ppl = next(
        r.overall_perplexity for r in all_plan_results if r.plan_name == "baseline"
    )

    report = {
        "model": args.model,
        "total_time_min": round((time.time() - t_start) / 60, 1),
        "best_plan": best.plan_name,
        "best_plan_ppl": best.overall_perplexity,
        "baseline_ppl": baseline_ppl,
        "ppl_degradation_pct": round(
            (best.overall_perplexity - baseline_ppl) / baseline_ppl * 100, 2
        ),
        "skip_layers": best.skip_layers,
        "transforms": best.transform_kinds,
        "accuracy_threshold": ACCURACY_THRESHOLD_PCT,
        "qaic_speedups": speedups,
    }
    save_json(report, output_dir / "final_report.json")

    print(f"\n  Best plan:      {best.plan_name}")
    print(f"  Transforms:     {best.transform_kinds}")
    print(f"  Skip layers:    {best.skip_layers}")
    print(f"  PPL change:     {baseline_ppl:.2f} -> {best.overall_perplexity:.2f} "
          f"({report['ppl_degradation_pct']:+.1f}%)")

    if speedups:
        any_sp = next(iter(speedups.values()))
        print(f"\n  QAIC Speedup:")
        print(f"    Decode: {any_sp['decode_pct']:+.1f}%")
        print(f"    E2E:    {any_sp['e2e_pct']:+.1f}%")

    print(f"\n  Total time:     {report['total_time_min']:.1f} minutes")
    print(f"  Results:        {output_dir}/")


if __name__ == "__main__":
    main()
