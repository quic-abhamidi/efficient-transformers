"""Detailed performance and accuracy comparison: baseline vs optimized plans.

Measures:
  1. Perplexity (cross-entropy loss) per dataset
  2. Generation speed (tokens/sec)
  3. Prefill latency (time to first token)
  4. Memory usage
  5. Text quality (side-by-side generation samples)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.channel_importance import compute_channel_importance
from QEfficient.model_pruning.qeff_model_optimizer.analysis.head_importance import compute_head_importance
from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import compute_kv_head_similarity
from QEfficient.model_pruning.qeff_model_optimizer.analysis.weak_layers import compute_weak_layer_report
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    HeadPruningSpec, LayerHeadSelection,
    MlpPruningSpec, SkipLayersSpec, StructuredSparsitySpec,
    TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.search.optimization import generate_optimization_plans
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry
from uuid import uuid4


EVAL_DATASETS = ["mmlu_pro", "bbh_causal", "bbh_logical_deduction", "ifeval", "gsm_hard", "humanevalpack", "orca_math"]
ANALYSIS_DATASETS = EVAL_DATASETS
NUM_ANALYSIS_SAMPLES = 30
NUM_EVAL_SAMPLES = 50
BATCH_SIZE = 4
MAX_LENGTH = 512

QUALITY_PROMPTS = [
    "Explain the difference between TCP and UDP in networking.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "A train leaves station A at 9:00 AM traveling at 60 mph. Another train leaves station B (300 miles away) at 10:00 AM traveling at 80 mph toward station A. At what time do they meet?",
    "Summarize the key principles of object-oriented programming.",
    "What are the main causes and effects of climate change?",
    "Debug this code: def fib(n): return fib(n-1) + fib(n-2)",
]

SPEED_PROMPTS = [
    "Write a detailed essay about the history of artificial intelligence, covering its origins, major milestones, and future directions.",
    "Explain quantum computing from first principles. Start with quantum mechanics basics and build up to quantum gates and algorithms.",
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
]


def _load_eval_prompts():
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import load_dataset_samples
    all_prompts = {}
    for ds in EVAL_DATASETS:
        try:
            all_prompts[ds] = load_dataset_samples(ds, NUM_EVAL_SAMPLES)
        except Exception as e:
            print(f"  WARNING: {ds} failed: {e}")
    return all_prompts


def measure_perplexity(model, tokenizer, prompts, max_length=512):
    total_loss = 0.0
    total_tokens = 0
    count = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        if seq_len < 2:
            continue
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * seq_len
        total_tokens += seq_len
        count += 1
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return {"loss": round(avg_loss, 4), "perplexity": round(ppl, 2), "samples": count}


def measure_generation_speed(model, tokenizer, prompts, max_new_tokens=128):
    total_gen_tokens = 0
    total_time = 0.0
    prefill_times = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        gen_tokens = out.shape[1] - input_len
        total_gen_tokens += gen_tokens
        total_time += elapsed

        torch.cuda.synchronize()
        tp = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        prefill_times.append(time.time() - tp)

    tokens_per_sec = total_gen_tokens / total_time if total_time > 0 else 0
    avg_prefill = sum(prefill_times) / len(prefill_times) if prefill_times else 0

    return {
        "tokens_per_sec": round(tokens_per_sec, 1),
        "total_tokens": total_gen_tokens,
        "total_time_sec": round(total_time, 2),
        "avg_prefill_ms": round(avg_prefill * 1000, 1),
    }


def measure_memory():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2)}


def generate_quality_samples(model, tokenizer, prompts, max_new_tokens=150):
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def evaluate_plan_full(model, tokenizer, plan_name, eval_prompts):
    print(f"\n  Measuring perplexity...")
    ppl_results = {}
    for ds, prompts in eval_prompts.items():
        ppl_results[ds] = measure_perplexity(model, tokenizer, prompts[:NUM_EVAL_SAMPLES])

    avg_loss = sum(r["loss"] for r in ppl_results.values()) / len(ppl_results)
    avg_ppl = sum(r["perplexity"] for r in ppl_results.values()) / len(ppl_results)

    print(f"  Measuring generation speed...")
    speed = measure_generation_speed(model, tokenizer, SPEED_PROMPTS, max_new_tokens=128)

    memory = measure_memory()

    print(f"  Generating quality samples...")
    quality = generate_quality_samples(model, tokenizer, QUALITY_PROMPTS, max_new_tokens=150)

    return {
        "perplexity": ppl_results,
        "avg_loss": round(avg_loss, 4),
        "avg_perplexity": round(avg_ppl, 2),
        "speed": speed,
        "memory": memory,
        "quality_samples": quality,
    }


def print_comparison_table(results: dict, model_id: str):
    plans = list(results.keys())
    baseline = results.get("baseline", {})
    bl_loss = baseline.get("avg_loss", 0)
    bl_ppl = baseline.get("avg_perplexity", 0)
    bl_speed = baseline.get("speed", {}).get("tokens_per_sec", 0)
    bl_prefill = baseline.get("speed", {}).get("avg_prefill_ms", 0)

    print(f"\n{'='*100}")
    print(f"  PERFORMANCE & ACCURACY COMPARISON — {model_id}")
    print(f"{'='*100}")

    print(f"\n  {'Plan':<24s} {'Avg Loss':>9s} {'PPL':>8s} {'Tok/s':>8s} {'Prefill':>9s} {'GPU Mem':>8s} {'Loss Δ':>8s} {'Speed Δ':>8s}")
    print(f"  {'-'*24} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

    for plan_name in plans:
        r = results[plan_name]
        loss = r["avg_loss"]
        ppl = r["avg_perplexity"]
        tps = r["speed"]["tokens_per_sec"]
        prefill = r["speed"]["avg_prefill_ms"]
        mem = r["memory"]["allocated_gb"]

        loss_delta = ((loss - bl_loss) / bl_loss * 100) if bl_loss > 0 else 0
        speed_delta = ((tps - bl_speed) / bl_speed * 100) if bl_speed > 0 else 0

        loss_str = f"{loss_delta:+.1f}%" if plan_name != "baseline" else "—"
        speed_str = f"{speed_delta:+.1f}%" if plan_name != "baseline" else "—"

        print(f"  {plan_name:<24s} {loss:>9.4f} {ppl:>8.1f} {tps:>8.1f} {prefill:>8.1f}ms {mem:>7.2f}G {loss_str:>8s} {speed_str:>8s}")

    print(f"\n  PER-DATASET PERPLEXITY BREAKDOWN:")
    print(f"  {'Dataset':<30s}", end="")
    for plan_name in plans:
        print(f" {plan_name:>14s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in plans:
        print(f" {'-'*14}", end="")
    print()

    all_datasets = set()
    for r in results.values():
        all_datasets.update(r.get("perplexity", {}).keys())
    for ds in sorted(all_datasets):
        print(f"  {ds:<30s}", end="")
        for plan_name in plans:
            ppl_data = results[plan_name].get("perplexity", {}).get(ds, {})
            ppl_val = ppl_data.get("perplexity", 0) if ppl_data else 0
            print(f" {ppl_val:>14.1f}", end="")
        print()

    print(f"\n  GENERATION QUALITY COMPARISON:")
    for i, prompt in enumerate(QUALITY_PROMPTS):
        print(f"\n  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        for plan_name in plans:
            samples = results[plan_name].get("quality_samples", [])
            if i < len(samples):
                text = samples[i][:120].replace("\n", " ")
                print(f"    {plan_name:<20s}: {text}")


def run_benchmark(model_id: str):
    print(f"\n{'='*100}")
    print(f"  FULL BENCHMARK: {model_id}")
    print(f"{'='*100}")

    spec = ModelSpec(model_id=model_id, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()

    print("\n  Loading model...")
    t0 = time.time()
    model, tokenizer = loader.load(spec)
    print(f"  Loaded in {time.time()-t0:.1f}s — {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads, {getattr(model.config, 'num_key_value_heads', '?')} KV heads")

    applier = TransformApplier(default_transform_registry())
    session = NASSession(loader=loader, transform_applier=applier)
    artifact = ModelArtifact(
        artifact_id=uuid4().hex, model=model, tokenizer=tokenizer,
        model_spec=spec, plan=TransformationPlan(),
    )
    session.artifacts[artifact.artifact_id] = artifact

    print("\n  Loading evaluation prompts...")
    eval_prompts = _load_eval_prompts()
    print(f"  {sum(len(v) for v in eval_prompts.values())} total prompts across {len(eval_prompts)} datasets")

    print("\n  Running analyses...")
    t0 = time.time()
    weak_report = compute_weak_layer_report(artifact, ANALYSIS_DATASETS, NUM_ANALYSIS_SAMPLES, BATCH_SIZE, max_length=MAX_LENGTH)
    head_report = compute_head_importance(artifact, ANALYSIS_DATASETS[:4], NUM_ANALYSIS_SAMPLES, BATCH_SIZE, MAX_LENGTH)
    channel_report = compute_channel_importance(artifact, ANALYSIS_DATASETS[:4], NUM_ANALYSIS_SAMPLES, BATCH_SIZE, MAX_LENGTH)
    kv_report = compute_kv_head_similarity(artifact)
    print(f"  Analyses complete in {time.time()-t0:.1f}s")

    weakest = weak_report.ranked_layers[:3]
    print(f"  Weakest layers: {[(r.layer, round(r.aggregate_score, 4)) for r in weakest]}")

    candidates = generate_optimization_plans(
        weak_report, head_report, channel_report, kv_report,
        accuracy_budget=0.05, enable_sparsity=False,
        head_prune_ratio=0.25, mlp_prune_ratio=0.2, kv_merge_ratio=0.5,
    )

    plans_to_eval = {}

    plans_to_eval["baseline"] = TransformationPlan()

    for c in candidates:
        kind = c.metadata.get("kind", "")
        if kind in ("skip_only", "conservative", "head_prune_only"):
            plans_to_eval[kind] = c.plan

    weakest_layer = weak_report.ranked_layers[0].layer
    second_weakest = weak_report.ranked_layers[1].layer
    num_heads = model.config.num_attention_heads

    weakest_heads_per_layer = {}
    for layer_idx, scores in head_report.per_layer_scores.items():
        weakest_heads_per_layer[layer_idx] = [h for h, _ in scores[:max(1, num_heads // 8)]]

    plans_to_eval["skip_2_layers"] = TransformationPlan(transforms=[
        SkipLayersSpec(layers=[weakest_layer, second_weakest]),
    ])

    plans_to_eval["skip+light_head"] = TransformationPlan(transforms=[
        SkipLayersSpec(layers=[weakest_layer]),
        HeadPruningSpec(selections=[
            LayerHeadSelection(layer=l, heads=weakest_heads_per_layer[l])
            for l in sorted(weakest_heads_per_layer.keys())[:5]
        ]),
    ])

    plans_to_eval["skip+moderate_head"] = TransformationPlan(transforms=[
        SkipLayersSpec(layers=[weakest_layer]),
        HeadPruningSpec(selections=[
            LayerHeadSelection(layer=l, heads=[h for h, _ in head_report.per_layer_scores[l][:max(1, num_heads // 4)]])
            for l in sorted(head_report.per_layer_scores.keys())
        ]),
    ])

    all_results = {}

    for plan_name, plan in plans_to_eval.items():
        transforms = [t.kind for t in plan.transforms]
        print(f"\n{'='*70}")
        print(f"  Evaluating: {plan_name}")
        print(f"  Transforms: {transforms or ['none']}")
        print(f"{'='*70}")

        t0 = time.time()
        session.apply_plan(artifact, plan)
        r = evaluate_plan_full(model, tokenizer, plan_name, eval_prompts)
        r["elapsed"] = round(time.time() - t0, 1)
        r["transforms"] = transforms
        all_results[plan_name] = r

        print(f"  Done in {r['elapsed']}s — loss={r['avg_loss']}, ppl={r['avg_perplexity']}, {r['speed']['tokens_per_sec']} tok/s")

    session.close()

    print_comparison_table(all_results, model_id)

    output_path = Path(f"results/{model_id.replace('/', '_')}_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: vv for kk, vv in v.items() if kk != "quality_samples"}
        serializable[k]["quality_samples_preview"] = [s[:200] for s in v.get("quality_samples", [])]

    output_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n  Full results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    try:
        run_benchmark(model_id)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
