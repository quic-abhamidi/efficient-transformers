#!/usr/bin/env python3
"""Example 03: Compile to QAIC hardware and benchmark performance.

Takes the optimal plan selected in example 02 and compiles it onto QAIC
(Qualcomm AI Cloud) hardware using QAICBenchmarkRunner. Also compiles a
baseline for comparison and reports the measured speedup.

This example demonstrates both:
- Single-device, BS=1 compilation — minimum latency config
- Multi-device, BS=8 compilation — maximum throughput config

Usage:
    python examples/03_qaic_deployment.py [--model MODEL_ID]

Prereq: run example 02 first to generate best_plan.json.
Requires: ``QEfficient`` installed and QAIC hardware available.

Notes on hook-based skip_layers and QAIC:
- This example uses SkipLayersSpec, which PATCHES the forward() method to
  be a no-op. The QAIC compiler traces the patched forward and includes the
  "no-op" as a pass-through in the compiled graph. The KV cache still
  reserves storage for that layer, so you get a modest speedup (2-5%) but
  not the full "layer removed" speedup.
- For maximum QAIC speedup, structural layer removal is needed: use
  ``QEFFAutoModelForCausalLM.from_pretrained(..., remove_layers=[34])``.
  That path requires a custom QEff fork (see nas_backup/efficient-transformers
  in this repo) until upstream support lands.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    SkipLayersSpec, TransformationPlan, transform_spec_from_dict,
)
from QEfficient.model_pruning.qeff_model_optimizer.evaluation import QAICBenchmarkRunner, charts


# ── Configuration ────────────────────────────────────────────────────────────

# QAIC compile parameters. ctx_len=4096 fits most chat use cases; prefill=128
# balances compile time and max prompt length.
CTX_LEN = 4096
PREFILL_SEQ_LEN = 128
NUM_CORES = 16  # cores per QAIC device

# Prompts used for performance measurement. Keep short — each prompt adds
# ~generation_len seconds to the run. Results are averaged across prompts.
PROMPTS = [
    "The capital of France is",
    "Write a Python function to compute fibonacci numbers:",
    "Explain quantum computing in simple terms:",
]
GENERATION_LEN = 60


def load_best_plan(plans_dir: Path) -> tuple[str, TransformationPlan, list[int]]:
    """Reconstruct the best plan saved by example 02.

    Returns ``(plan_name, TransformationPlan, skip_layers)``.
    Falls back to skip_1 of layer 34 if not found.
    """
    path = plans_dir / "best_plan.json"
    if not path.exists():
        print(f"  [warn] {path} not found — using fallback plan (skip layer 34)")
        return "skip_34_fallback", TransformationPlan(
            transforms=[SkipLayersSpec(layers=[34])]
        ), [34]

    data = json.loads(path.read_text())
    # Re-hydrate TransformSpec objects from their dict form.
    transforms = []
    for t_dict in data.get("transforms", []):
        # The saved dict only has 'kind' + the layers; we need to reconstruct.
        # For this example we only support skip_layers; extend as needed.
        if t_dict == "skip_layers" and data.get("skip_layers"):
            transforms.append(SkipLayersSpec(layers=data["skip_layers"]))

    # If we couldn't reconstruct transforms from the saved data, fall back.
    if not transforms and data.get("skip_layers"):
        transforms = [SkipLayersSpec(layers=data["skip_layers"])]

    plan = TransformationPlan(transforms=transforms)
    return data["plan_name"], plan, data.get("skip_layers", [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model id")
    parser.add_argument("--plans-dir", default="results/example_02_plans",
                        help="Where example 02 saved best_plan.json")
    parser.add_argument("--output-dir", default="results/example_03_qaic",
                        help="Where to save QAIC benchmark results")
    parser.add_argument("--single-device-only", action="store_true",
                        help="Skip the 4-device compilation")
    args = parser.parse_args()

    plans_dir = Path(args.plans_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Example 03: QAIC Deployment")
    print(f"  Model:    {args.model}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*70}")

    # ── Step 1: Load the best plan from example 02 ──────────────────────────
    print(f"\n[1/4] Loading best plan...")
    plan_name, optimal_plan, skip_layers = load_best_plan(plans_dir)
    print(f"  Plan: {plan_name}")
    print(f"  Transforms: {[t.kind for t in optimal_plan.transforms]}")
    print(f"  Skip layers: {skip_layers}")

    # ── Step 2: Instantiate the QAIC runner ─────────────────────────────────
    # The runner handles the load-fresh-model → apply-plan → QEff-wrap →
    # compile-to-QPC → run-prompts → parse-perf cycle. Each call to run() is
    # independent (loads its own fresh model).
    print(f"\n[2/4] Setting up QAIC runner...")
    runner = QAICBenchmarkRunner(
        model_id=args.model,
        prompts=PROMPTS,
        generation_len=GENERATION_LEN,
        ctx_len=CTX_LEN,
        prefill_seq_len=PREFILL_SEQ_LEN,
        num_cores=NUM_CORES,
        compile_dir_base="results/model_pruning/examples/nas_example_03",
        mxfp6_matmul=True,   # 6-bit weights
        mxint8_kv_cache=True,  # 8-bit KV cache
    )

    results = []

    # ── Step 3a: Baseline on 1 device ───────────────────────────────────────
    # Compile the full model (no transforms) for low-latency single-device
    # inference. This is our reference point for measuring speedup.
    print(f"\n[3/4] Compiling BASELINE on 1 QAIC device (BS=1)...")
    t0 = time.time()
    baseline_1dev = runner.run(
        name="baseline_1dev",
        plan=TransformationPlan(),
        device_group=[0],
        batch_size=1,
    )
    print(f"  Completed in {(time.time()-t0)/60:.1f} min")
    if baseline_1dev.error:
        print(f"  ERROR: {baseline_1dev.error}")
    else:
        s = baseline_1dev.avg_stats
        print(f"  TTFT: {s.get('ttft', 0):.3f}s | "
              f"Decode: {s.get('decode_tps', 0):.1f} tok/s | "
              f"E2E: {s.get('e2e', 0):.2f}s")
    results.append(baseline_1dev)
    (output_dir / "baseline_1dev.json").write_text(
        json.dumps(baseline_1dev.to_dict(), indent=2, default=str)
    )

    # ── Step 3b: Optimized on 1 device ──────────────────────────────────────
    # Apply the plan from example 02 and compile for comparison.
    print(f"\n  Compiling OPTIMIZED ({plan_name}) on 1 device (BS=1)...")
    t0 = time.time()
    optimized_1dev = runner.run(
        name=f"{plan_name}_1dev",
        plan=optimal_plan,
        device_group=[0],
        batch_size=1,
    )
    print(f"  Completed in {(time.time()-t0)/60:.1f} min")
    if optimized_1dev.error:
        print(f"  ERROR: {optimized_1dev.error}")
    else:
        s = optimized_1dev.avg_stats
        print(f"  TTFT: {s.get('ttft', 0):.3f}s | "
              f"Decode: {s.get('decode_tps', 0):.1f} tok/s | "
              f"E2E: {s.get('e2e', 0):.2f}s")
    results.append(optimized_1dev)
    (output_dir / f"{plan_name}_1dev.json").write_text(
        json.dumps(optimized_1dev.to_dict(), indent=2, default=str)
    )

    # ── Step 3c: (optional) 4-device, high batch size ───────────────────────
    if not args.single_device_only:
        print(f"\n[4/4] Compiling OPTIMIZED on 4 devices (BS=8, throughput mode)...")
        t0 = time.time()
        optimized_4dev = runner.run(
            name=f"{plan_name}_4dev_bs8",
            plan=optimal_plan,
            device_group=[0, 1, 2, 3],
            batch_size=8,
        )
        print(f"  Completed in {(time.time()-t0)/60:.1f} min")
        if optimized_4dev.error:
            print(f"  ERROR: {optimized_4dev.error}")
        else:
            s = optimized_4dev.avg_stats
            # Note: 4-device BS=8 gives PER-REQUEST stats. To get aggregate
            # throughput, multiply decode_tps by batch_size.
            print(f"  TTFT: {s.get('ttft', 0):.3f}s | "
                  f"Decode: {s.get('decode_tps', 0):.1f} tok/s (per request) | "
                  f"Aggregate: {s.get('decode_tps', 0) * 8:.1f} tok/s (8 parallel) | "
                  f"E2E: {s.get('e2e', 0):.2f}s")
        results.append(optimized_4dev)
        (output_dir / f"{plan_name}_4dev_bs8.json").write_text(
            json.dumps(optimized_4dev.to_dict(), indent=2, default=str)
        )
    else:
        print(f"\n[4/4] Skipping 4-device compilation (--single-device-only)")

    # ── Report: speedup table ───────────────────────────────────────────────
    speedups = runner.compute_speedups(results, baseline_name="baseline_1dev")

    print(f"\n{'='*70}")
    print(f"  SPEEDUP vs baseline_1dev")
    print(f"{'='*70}")
    for name, speedup in speedups.items():
        print(f"\n  {name}:")
        print(f"    TTFT:   {speedup['ttft_pct']:+.1f}%")
        print(f"    Decode: {speedup['decode_pct']:+.1f}%")
        print(f"    E2E:    {speedup['e2e_pct']:+.1f}%")

    # ── Chart performance comparison ────────────────────────────────────────
    try:
        charts.chart_qaic_performance(
            results, output_dir / "qaic_comparison.png",
            title_prefix=args.model,
        )
        print(f"\n  Chart saved: {output_dir / 'qaic_comparison.png'}")
    except Exception as e:
        print(f"\n  [warn] chart generation failed: {e}")

    # ── Save aggregated results ─────────────────────────────────────────────
    (output_dir / "all_results.json").write_text(
        json.dumps({
            "model": args.model,
            "results": [r.to_dict() for r in results],
            "speedups": speedups,
        }, indent=2, default=str)
    )

    print(f"\n{'='*70}")
    print(f"  Deployment done. Outputs in {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
