#!/usr/bin/env python3
"""Comprehensive linear attention evaluation on Qwen3-32B.

1. Full perplexity across all 5 standard datasets (50 samples each).
2. Prefill latency (time to process a prompt).
3. Decode throughput (tokens/sec for autoregressive generation).
"""

import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.cuda import Event as CudaEvent

from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    LinearAttentionSpec,
    SkipLayersSpec,
    TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.evaluation import PlanEvaluator
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import run_model_cleanup

MODEL_ID = "Qwen/Qwen3-32B"
EVAL_DATASETS = ["wikitext", "mmlu_pro", "bbh_causal", "ifeval", "gsm_hard"]
NUM_SAMPLES = 50
MAX_LENGTH = 512

WEAK_LAYERS_FILE = "results/Qwen_Qwen3-32B_optimization_results.json"

PREFILL_LENGTHS = [128, 512, 1024]
DECODE_TOKENS = 128
WARMUP_ITERS = 2
BENCH_ITERS = 5


def load_weak_layers() -> list[int]:
    data = json.load(open(WEAK_LAYERS_FILE))
    ranked = data["analysis"]["weakest_layers"]
    return [layer_idx for layer_idx, _score in ranked]


def build_plans(weak_layers: list[int]) -> dict[str, TransformationPlan]:
    plans: dict[str, TransformationPlan] = {}
    plans["baseline"] = TransformationPlan()

    for n in (1, 3, 5):
        if n > len(weak_layers):
            continue
        target = sorted(weak_layers[:n])
        plans[f"elu_weak_{n}"] = TransformationPlan(
            transforms=[
                LinearAttentionSpec(
                    implementation="elu",
                    target_layers=target,
                )
            ]
        )

    target_3 = sorted(weak_layers[:3])
    plans["elu_weak_3_decode_only"] = TransformationPlan(
        transforms=[
            LinearAttentionSpec(
                implementation="elu",
                target_layers=target_3,
                mode="decode_only",
            )
        ]
    )

    plans["skip_weak_3"] = TransformationPlan(
        transforms=[
            SkipLayersSpec(layers=sorted(weak_layers[:3]))
        ]
    )

    return plans


def measure_prefill_latency(model, tokenizer, seq_len, warmup=2, iters=5):
    """Measure prefill latency (forward pass on a prompt of seq_len tokens)."""
    dummy_ids = torch.randint(100, 30000, (1, seq_len), device=model.device)

    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_ids)
    torch.cuda.synchronize()

    start = CudaEvent(enable_timing=True)
    end = CudaEvent(enable_timing=True)

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            model(dummy_ids)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    return avg_ms


def measure_decode_throughput(model, tokenizer, prompt_len, gen_tokens, warmup=1, iters=3):
    """Measure decode throughput (tokens/sec) via model.generate."""
    dummy_ids = torch.randint(100, 30000, (1, prompt_len), device=model.device)
    attn_mask = torch.ones_like(dummy_ids)

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(
                dummy_ids,
                attention_mask=attn_mask,
                max_new_tokens=gen_tokens,
                do_sample=False,
            )
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                dummy_ids,
                attention_mask=attn_mask,
                max_new_tokens=gen_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        actual_new = out.shape[1] - prompt_len
        times.append((t1 - t0, actual_new))

    total_time = sum(t for t, _ in times)
    total_tokens = sum(n for _, n in times)
    tps = total_tokens / total_time if total_time > 0 else 0
    avg_latency = total_time / len(times)
    return tps, avg_latency


def run_latency_benchmarks(model, tokenizer, plans, applier):
    """Run prefill and decode latency benchmarks for each plan."""
    results = {}

    for plan_name, plan in plans.items():
        print(f"\n  Benchmarking: {plan_name}")
        run_model_cleanup(model)
        if plan.transforms:
            from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
            from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec as MS
            artifact = ModelArtifact(
                artifact_id="bench",
                model=model,
                tokenizer=tokenizer,
                model_spec=MS(model_id=MODEL_ID),
                plan=TransformationPlan(),
                applied_transforms=[],
            )
            try:
                applier.apply(artifact, plan)
            except Exception as e:
                print(f"    ERROR applying plan: {e}")
                results[plan_name] = {"error": str(e)}
                continue

        plan_results = {"prefill": {}, "decode": {}}

        for seq_len in PREFILL_LENGTHS:
            ms = measure_prefill_latency(
                model, tokenizer, seq_len,
                warmup=WARMUP_ITERS, iters=BENCH_ITERS,
            )
            plan_results["prefill"][seq_len] = ms
            print(f"    Prefill {seq_len:>5d} tokens: {ms:>8.1f} ms")

        tps, avg_s = measure_decode_throughput(
            model, tokenizer,
            prompt_len=128, gen_tokens=DECODE_TOKENS,
            warmup=1, iters=3,
        )
        plan_results["decode"]["tokens_per_sec"] = tps
        plan_results["decode"]["avg_latency_s"] = avg_s
        print(f"    Decode {DECODE_TOKENS} tokens:  {tps:>8.1f} tok/s  ({avg_s:.2f}s avg)")

        results[plan_name] = plan_results

    run_model_cleanup(model)
    return results


def main():
    weak_layers = load_weak_layers()
    print(f"{'=' * 78}")
    print(f"  Comprehensive Linear Attention Evaluation — {MODEL_ID}")
    print(f"  Weak layers (ranked): {weak_layers}")
    print(f"  Datasets:   {EVAL_DATASETS} ({NUM_SAMPLES} samples each)")
    print(f"  Prefill lengths: {PREFILL_LENGTHS}")
    print(f"  Decode tokens:   {DECODE_TOKENS}")
    print(f"{'=' * 78}")

    print(f"\n[1/4] Loading model...")
    t0 = time.time()
    spec = ModelSpec(model_id=MODEL_ID, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()
    model, tokenizer = loader.load(spec)
    num_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time() - t0:.1f}s | "
          f"Layers: {num_layers} | "
          f"GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    plans = build_plans(weak_layers)

    print(f"\n[2/4] Plans to evaluate:")
    for name, plan in plans.items():
        if plan.transforms:
            kinds = [t.kind for t in plan.transforms]
            s = plan.transforms[0]
            if hasattr(s, "target_layers") and s.target_layers:
                detail = f"{kinds}, layers={s.target_layers}"
            elif hasattr(s, "layers"):
                detail = f"{kinds}, layers={s.layers}"
            else:
                detail = str(kinds)
            if hasattr(s, "mode"):
                detail += f", mode={s.mode}"
        else:
            detail = "no transforms"
        print(f"    {name:30s} -> {detail}")

    # -- Part A: Full perplexity evaluation --
    print(f"\n[3/4] Perplexity evaluation ({len(EVAL_DATASETS)} datasets, "
          f"{NUM_SAMPLES} samples each)...")
    evaluator = PlanEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_spec=spec,
        datasets=EVAL_DATASETS,
        num_samples=NUM_SAMPLES,
        max_length=MAX_LENGTH,
    )

    t0 = time.time()
    ppl_results = evaluator.evaluate_all(plans)
    ppl_elapsed = (time.time() - t0) / 60
    print(f"  Perplexity evaluation done in {ppl_elapsed:.1f} min")

    # -- Part B: Latency benchmarks --
    print(f"\n[4/4] Latency benchmarks...")
    applier = TransformApplier(default_transform_registry())
    latency_results = run_latency_benchmarks(model, tokenizer, plans, applier)

    # =====================================================================
    # REPORT
    # =====================================================================
    baseline_r = next(r for r in ppl_results if r.plan_name == "baseline")
    baseline_ppl = baseline_r.overall_perplexity
    baseline_lat = latency_results.get("baseline", {})

    print(f"\n{'=' * 78}")
    print(f"  PERPLEXITY RESULTS (overall, sorted)")
    print(f"{'=' * 78}")
    print(f"\n  {'Plan':<30s} {'PPL':>8s} {'Delta':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")
    for r in ppl_results:
        if r.error:
            print(f"  {r.plan_name:<30s} ERROR: {r.error[:50]}")
            continue
        delta = (r.overall_perplexity - baseline_ppl) / baseline_ppl * 100
        marker = "  <-- baseline" if r.plan_name == "baseline" else ""
        print(f"  {r.plan_name:<30s} {r.overall_perplexity:>8.2f} {delta:>+7.1f}%{marker}")

    # Per-dataset breakdown
    print(f"\n{'=' * 78}")
    print(f"  PER-DATASET PERPLEXITY")
    print(f"{'=' * 78}")
    ds_names = EVAL_DATASETS
    header = f"  {'Plan':<28s}" + "".join(f" {d:>12s}" for d in ds_names)
    print(f"\n{header}")
    print(f"  {'-' * 28}" + "".join(f" {'-' * 12}" for _ in ds_names))
    for r in ppl_results:
        if r.error or r.perplexity_report is None:
            continue
        row = f"  {r.plan_name:<28s}"
        for ds in ds_names:
            dp = r.perplexity_report.per_dataset.get(ds)
            if dp:
                row += f" {dp.perplexity:>12.2f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # Per-dataset delta from baseline
    baseline_pd = baseline_r.perplexity_report.per_dataset if baseline_r.perplexity_report else {}
    print(f"\n  Per-dataset delta from baseline (%):")
    header2 = f"  {'Plan':<28s}" + "".join(f" {d:>12s}" for d in ds_names)
    print(f"\n{header2}")
    print(f"  {'-' * 28}" + "".join(f" {'-' * 12}" for _ in ds_names))
    for r in ppl_results:
        if r.error or r.perplexity_report is None or r.plan_name == "baseline":
            continue
        row = f"  {r.plan_name:<28s}"
        for ds in ds_names:
            dp = r.perplexity_report.per_dataset.get(ds)
            bp = baseline_pd.get(ds)
            if dp and bp and bp.perplexity > 0:
                d = (dp.perplexity - bp.perplexity) / bp.perplexity * 100
                row += f" {d:>+11.1f}%"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # Latency results
    print(f"\n{'=' * 78}")
    print(f"  LATENCY RESULTS")
    print(f"{'=' * 78}")

    # Prefill
    print(f"\n  Prefill latency (ms):")
    header3 = f"  {'Plan':<28s}" + "".join(f" {f'{sl} tok':>10s}" for sl in PREFILL_LENGTHS)
    print(f"\n{header3}")
    print(f"  {'-' * 28}" + "".join(f" {'-' * 10}" for _ in PREFILL_LENGTHS))
    for name in plans:
        lr = latency_results.get(name, {})
        if "error" in lr:
            print(f"  {name:<28s} ERROR")
            continue
        prefill = lr.get("prefill", {})
        row = f"  {name:<28s}"
        for sl in PREFILL_LENGTHS:
            ms = prefill.get(sl, 0)
            row += f" {ms:>10.1f}"
        print(row)

    # Prefill speedup
    if baseline_lat and "prefill" in baseline_lat:
        print(f"\n  Prefill speedup vs baseline (%):")
        header4 = f"  {'Plan':<28s}" + "".join(f" {f'{sl} tok':>10s}" for sl in PREFILL_LENGTHS)
        print(f"\n{header4}")
        print(f"  {'-' * 28}" + "".join(f" {'-' * 10}" for _ in PREFILL_LENGTHS))
        for name in plans:
            if name == "baseline":
                continue
            lr = latency_results.get(name, {})
            if "error" in lr:
                continue
            prefill = lr.get("prefill", {})
            row = f"  {name:<28s}"
            for sl in PREFILL_LENGTHS:
                ms = prefill.get(sl, 0)
                bms = baseline_lat["prefill"].get(sl, 1)
                speedup = (bms - ms) / bms * 100 if bms > 0 else 0
                row += f" {speedup:>+9.1f}%"
            print(row)

    # Decode
    print(f"\n  Decode throughput:")
    print(f"\n  {'Plan':<28s} {'tok/s':>10s} {'latency':>10s} {'speedup':>10s}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}")
    baseline_tps = baseline_lat.get("decode", {}).get("tokens_per_sec", 1)
    for name in plans:
        lr = latency_results.get(name, {})
        if "error" in lr:
            print(f"  {name:<28s} ERROR")
            continue
        dec = lr.get("decode", {})
        tps = dec.get("tokens_per_sec", 0)
        lat = dec.get("avg_latency_s", 0)
        speedup = (tps - baseline_tps) / baseline_tps * 100 if baseline_tps > 0 else 0
        su_str = f"{speedup:>+9.1f}%" if name != "baseline" else "—"
        print(f"  {name:<28s} {tps:>10.1f} {lat:>9.2f}s {su_str:>10s}")

    # Generation comparison
    print(f"\n{'=' * 78}")
    print(f"  GENERATION COMPARISON")
    print(f"{'=' * 78}")
    for r in ppl_results:
        if r.error or not r.completions:
            continue
        print(f"\n  --- {r.plan_name} ---")
        for prompt, completion in list(r.completions.items())[:2]:
            print(f"  Prompt: {prompt[:80]}...")
            print(f"  Output: {completion[:200]}...")
            print()

    print(f"\n{'=' * 78}")
    print(f"  Done. Total eval time: {ppl_elapsed:.1f} min (PPL) + latency benchmarks")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
