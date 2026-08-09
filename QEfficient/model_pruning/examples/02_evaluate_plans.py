#!/usr/bin/env python3
"""Example 02: Evaluate multiple optimization plans by perplexity.

Given the weak-layer rankings from example 01, this script builds several
candidate transformation plans (baseline, skip 1 layer, skip 3 spread, head
prune, combined), applies each one in turn to a shared model, and measures
perplexity on real datasets.

Output: a ranked list of plans and a chart comparing their quality. The best
plan within your accuracy threshold becomes the input to example 03 (QAIC deploy).

Usage:
    python examples/02_evaluate_plans.py [--model MODEL_ID]

Prereq: run example 01 first so ``weak_layer_report.json`` exists.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

# ── NAS imports ──────────────────────────────────────────────────────────────
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    HeadPruningSpec, LayerHeadSelection, MlpPruningSpec,
    SkipLayersSpec, TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.evaluation import PlanEvaluator, charts


# ── Configuration ────────────────────────────────────────────────────────────

# The key difference vs example 01: we use MORE samples here, because plan
# evaluation needs reliable PPL numbers. 300 samples across 6 datasets gives
# a reasonable signal; for production comparisons, bump to 500-1000.
EVAL_DATASETS = ["wikitext", "mmlu_pro", "bbh_causal", "ifeval", "gsm_hard"]
NUM_SAMPLES = 50
MAX_LENGTH = 512


def load_weak_layers(analysis_dir: Path) -> list[int]:
    """Load the weak-layer ranking from example 01's output.

    Falls back to a sensible default if the file is missing (lets this
    example run standalone, though the results will be less meaningful).
    """
    report_path = analysis_dir / "weak_layer_report.json"
    if not report_path.exists():
        print(f"  [warn] {report_path} not found. "
              f"Run example 01 first for real analysis data.")
        print(f"         Using placeholder weak layers.")
        return list(range(10, 20))  # arbitrary middle layers

    data = json.loads(report_path.read_text())
    # ranked_layers is sorted weakest-first.
    return [r["layer"] for r in data["ranked_layers"]]


def load_head_scores(analysis_dir: Path) -> dict | None:
    """Load head importance scores from example 01. Returns None if missing."""
    path = analysis_dir / "head_importance_report.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # per_layer_scores has string keys in JSON; convert back to int.
    return {int(k): v for k, v in data["per_layer_scores"].items()}


def build_candidate_plans(
    weakest_layers: list[int],
    head_scores: dict | None,
    num_heads: int,
    num_layers: int,
) -> dict[str, TransformationPlan]:
    """Construct a set of plans spanning conservative to aggressive.

    Each plan tests a different hypothesis:
    - skip_N: is the Nth weakest layer actually skippable?
    - skip_3_spread: are non-adjacent layers more robust than contiguous?
    - head_prune: how much head pruning is tolerable?
    - combined: do combined transforms degrade more than the sum?
    """
    plans: dict[str, TransformationPlan] = {}

    # Baseline: no transforms applied. Gives us the reference PPL.
    plans["baseline"] = TransformationPlan()

    # Granular skip variants (1, 2, 3 weakest layers).
    for n in (1, 2, 3):
        plans[f"skip_{n}_topN"] = TransformationPlan(
            transforms=[SkipLayersSpec(layers=sorted(weakest_layers[:n]))]
        )

    # Spread variant: skip layers from DIFFERENT parts of the model. Often
    # much better than contiguous skipping because the residual stream has
    # more room to recover between skips.
    if len(weakest_layers) >= 3:
        # Pick the weakest layer from beginning, middle, and end thirds.
        third = num_layers // 3
        early = next((l for l in weakest_layers if l < third), weakest_layers[0])
        middle = next((l for l in weakest_layers if third <= l < 2 * third), weakest_layers[1])
        late = next((l for l in weakest_layers if l >= 2 * third), weakest_layers[2])
        spread_layers = sorted(set([early, middle, late]))
        if len(spread_layers) == 3:
            plans["skip_3_spread"] = TransformationPlan(
                transforms=[SkipLayersSpec(layers=spread_layers)]
            )

    # Head pruning (only if we loaded the head importance report).
    if head_scores:
        # Prune the weakest 15% of heads in each layer.
        prune_count = max(1, int(num_heads * 0.15))
        selections = [
            LayerHeadSelection(
                layer=layer_idx,
                heads=[h for h, _score in scores[:prune_count]],
            )
            for layer_idx, scores in head_scores.items()
        ]
        plans["head_prune_15"] = TransformationPlan(
            transforms=[HeadPruningSpec(selections=selections)]
        )

    # Combined: skip 1 layer + prune 15% MLP channels. Tests composability.
    plans["skip_1_mlp_15"] = TransformationPlan(
        transforms=[
            SkipLayersSpec(layers=[weakest_layers[0]]),
            MlpPruningSpec(target_layers=[], pruning_ratio=0.15),
        ]
    )

    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model id")
    parser.add_argument("--analysis-dir", default="results/example_01_analysis",
                        help="Where example 01 saved its reports")
    parser.add_argument("--output-dir", default="results/example_02_plans",
                        help="Where to save plan evaluation results")
    parser.add_argument("--accuracy-threshold", type=float, default=10.0,
                        help="Max acceptable PPL increase over baseline (%%)")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Example 02: Plan Evaluation")
    print(f"  Model:        {args.model}")
    print(f"  Datasets:     {EVAL_DATASETS}")
    print(f"  Samples/ds:   {NUM_SAMPLES}")
    print(f"  Accuracy threshold: ≤{args.accuracy_threshold}% PPL increase")
    print(f"{'='*70}")

    # ── Step 1: Load prior analysis results ─────────────────────────────────
    # We inherit the weak-layer ranking from example 01 rather than re-running
    # the analysis, which is expensive.
    print(f"\n[1/4] Loading analysis results from {analysis_dir}...")
    weakest_layers = load_weak_layers(analysis_dir)
    head_scores = load_head_scores(analysis_dir)
    print(f"  Weakest layers (top 10): {weakest_layers[:10]}")
    print(f"  Head scores: {'loaded' if head_scores else 'not available'}")

    # ── Step 2: Load model on GPU ───────────────────────────────────────────
    print(f"\n[2/4] Loading model...")
    t0 = time.time()
    spec = ModelSpec(model_id=args.model, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()
    model, tokenizer = loader.load(spec)
    print(f"  Loaded in {time.time()-t0:.1f}s | "
          f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Step 3: Build candidate plans ───────────────────────────────────────
    plans = build_candidate_plans(
        weakest_layers=weakest_layers,
        head_scores=head_scores,
        num_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers,
    )
    print(f"\n[3/4] Evaluating {len(plans)} plans:")
    for name, plan in plans.items():
        kinds = [t.kind for t in plan.transforms]
        print(f"    {name:20s} -> {kinds or ['none']}")

    # ── Step 4: Run PlanEvaluator ───────────────────────────────────────────
    # PlanEvaluator applies each plan, scores it, then rolls it back. The
    # same model instance is reused across all plans — critical for speed
    # since loading the model takes minutes for a 14B.
    print(f"\n[4/4] Running evaluator...")
    evaluator = PlanEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_spec=spec,
        datasets=EVAL_DATASETS,
        num_samples=NUM_SAMPLES,
        max_length=MAX_LENGTH,
    )

    t0 = time.time()
    results = evaluator.evaluate_all(plans)
    print(f"\n  All plans evaluated in {(time.time()-t0)/60:.1f} min")

    # ── Report: summary table ───────────────────────────────────────────────
    baseline = next(r for r in results if r.plan_name == "baseline")
    baseline_ppl = baseline.overall_perplexity

    print(f"\n{'='*70}")
    print(f"  RESULTS (sorted by perplexity)")
    print(f"{'='*70}")
    print(f"\n  {'Plan':<22s} {'Transforms':>12s} {'PPL':>8s} {'Delta':>8s}  Transforms")
    print(f"  {'-'*22} {'-'*12} {'-'*8} {'-'*8}")
    for r in results:
        if r.error:
            print(f"  {r.plan_name:<22s} ERROR: {r.error[:50]}")
            continue
        delta = (r.overall_perplexity - baseline_ppl) / baseline_ppl * 100 if baseline_ppl else 0
        marker = "  <-- baseline" if r.plan_name == "baseline" else ""
        print(f"  {r.plan_name:<22s} {len(r.transform_kinds):>12d} "
              f"{r.overall_perplexity:>8.2f} {delta:>+7.1f}%{marker}")

    # ── Report: select best plan within threshold ──────────────────────────────
    best = evaluator.select_best(results, accuracy_threshold=args.accuracy_threshold)
    print(f"\n  Best plan within {args.accuracy_threshold}% accuracy threshold: ", end="")
    if best is None:
        print(f"NONE — no optimization fits this accuracy threshold.")
    else:
        delta = (best.overall_perplexity - baseline_ppl) / baseline_ppl * 100
        print(f"{best.plan_name}")
        print(f"    PPL: {baseline_ppl:.2f} -> {best.overall_perplexity:.2f} "
              f"({delta:+.1f}%)")
        print(f"    Transforms: {best.transform_kinds}")
        if best.skip_layers:
            print(f"    Skips layers: {best.skip_layers}")

    # ── Save results ────────────────────────────────────────────────────────
    print(f"\n  Saving results to {output_dir}/...")
    all_results = [r.to_dict() for r in results]
    (output_dir / "plan_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    if best is not None:
        (output_dir / "best_plan.json").write_text(
            json.dumps(best.to_dict(), indent=2, default=str)
        )

    # ── Generate comparison chart ───────────────────────────────────────────
    try:
        charts.chart_perplexity_comparison(
            results, output_dir / "perplexity_comparison.png",
            title_prefix=args.model,
        )
        charts.chart_per_dataset_perplexity(
            results, output_dir / "per_dataset_ppl.png",
            title_prefix=args.model,
        )
        print(f"    Charts saved.")
    except Exception as e:
        print(f"    [warn] chart generation failed: {e}")

    print(f"\n{'='*70}")
    print(f"  Next step: run example 03 to compile the best plan on QAIC")
    print(f"  Best plan: {best.plan_name if best else 'none'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
