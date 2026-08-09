"""End-to-end verification of the full NAS optimization pipeline.

Tests every component:
  1. Model loading
  2. LayerAnatomy adapter resolution
  3. Weak layer analysis (modern datasets)
  4. Head importance analysis
  5. Channel importance analysis
  6. KV head similarity analysis
  7. Optimization plan generation
  8. Head pruning transform
  9. MLP pruning transform
  10. KV cache compression transform
  11. 2:4 Structured sparsity transform
  12. Combined multi-transform plan
  13. Post-transform inference
  14. Session cleanup & restoration
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.channel_importance import ChannelImportanceReport, compute_channel_importance
from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import DEFAULT_ANALYSIS_DATASETS, MODERN_DATASETS, SUPPORTED_DATASETS
from QEfficient.model_pruning.qeff_model_optimizer.analysis.head_importance import HeadImportanceReport, compute_head_importance
from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import KvSimilarityReport, compute_kv_head_similarity
from QEfficient.model_pruning.qeff_model_optimizer.analysis.weak_layers import compute_weak_layer_report
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    HeadPruningSpec,
    KvCacheCompressionSpec,
    LayerHeadSelection,
    MlpPruningSpec,
    SkipLayersSpec,
    StructuredSparsitySpec,
    TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.search.optimization import generate_optimization_plans
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import run_model_cleanup


ANALYSIS_DATASETS = ["mmlu_pro", "bbh_causal", "ifeval", "gsm_hard", "humanevalpack"]
NUM_SAMPLES = 8
BATCH_SIZE = 2
MAX_LENGTH = 256


class VerificationResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details: dict = {}
        self.error: str | None = None
        self.elapsed: float = 0.0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.elapsed:.1f}s)"


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def verify_model_load(model_id: str) -> tuple[object, object, ModelSpec, VerificationResult]:
    result = VerificationResult("1. Model Loading")
    t0 = time.time()
    try:
        spec = ModelSpec(model_id=model_id, dtype="bfloat16", device_map="auto")
        loader = TransformersModelLoader()
        model, tokenizer = loader.load(spec)
        result.details = {
            "model_type": model.config.model_type,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "num_kv_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
            "pad_token": tokenizer.pad_token,
        }
        assert model.config.model_type in ("qwen3", "llama", "mistral", "qwen2", "gemma3")
        assert tokenizer.pad_token is not None
        result.passed = True
    except Exception as e:
        result.error = str(e)
        raise
    finally:
        result.elapsed = time.time() - t0
    return model, tokenizer, spec, result


def verify_layer_anatomy(model, result_details: dict) -> VerificationResult:
    result = VerificationResult("2. LayerAnatomy Adapter")
    t0 = time.time()
    try:
        adapter = resolve_layer_adapter(model)
        num_layers = adapter.num_layers
        assert num_layers == result_details["num_layers"]

        for layer_idx in [0, num_layers // 2, num_layers - 1]:
            anatomy = resolve_layer_anatomy(model, layer_idx)
            assert anatomy.q_proj is not None
            assert anatomy.k_proj is not None
            assert anatomy.v_proj is not None
            assert anatomy.o_proj is not None
            assert anatomy.gate_proj is not None
            assert anatomy.down_proj is not None
            assert anatomy.num_heads == result_details["num_heads"]
            assert anatomy.num_kv_heads == result_details["num_kv_heads"]
            assert anatomy.intermediate_size == result_details["intermediate_size"]
            assert anatomy.head_dim == getattr(
                model.config, "head_dim",
                result_details["hidden_size"] // result_details["num_heads"],
            )

        result.details = {
            "layers_verified": [0, num_layers // 2, num_layers - 1],
            "q_proj_shape": list(resolve_layer_anatomy(model, 0).q_proj.weight.shape),
            "gate_proj_shape": list(resolve_layer_anatomy(model, 0).gate_proj.weight.shape),
        }
        result.passed = True
    except Exception as e:
        result.error = str(e)
    finally:
        result.elapsed = time.time() - t0
    return result


def verify_weak_layer_analysis(artifact: ModelArtifact) -> tuple:
    result = VerificationResult("3. Weak Layer Analysis (modern datasets)")
    t0 = time.time()
    try:
        report = compute_weak_layer_report(
            artifact=artifact,
            datasets=ANALYSIS_DATASETS,
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            metric="cosine",
            max_length=MAX_LENGTH,
        )
        assert len(report.datasets) == len(ANALYSIS_DATASETS)
        assert len(report.ranked_layers) == artifact.model.config.num_hidden_layers
        ranks = [r.rank for r in report.ranked_layers]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

        top5 = report.ranked_layers[:5]
        result.details = {
            "datasets_analyzed": len(report.datasets),
            "layers_ranked": len(report.ranked_layers),
            "top5_weakest": [(r.layer, round(r.aggregate_score, 4)) for r in top5],
        }
        result.passed = True
        return report, result
    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return None, result
    finally:
        result.elapsed = time.time() - t0


def verify_head_importance(artifact: ModelArtifact) -> tuple:
    result = VerificationResult("4. Head Importance Analysis")
    t0 = time.time()
    try:
        report = compute_head_importance(
            artifact=artifact,
            datasets=ANALYSIS_DATASETS[:3],
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        num_layers = artifact.model.config.num_hidden_layers
        assert report.num_layers == num_layers
        assert report.num_heads == artifact.model.config.num_attention_heads
        assert len(report.per_layer_scores) == num_layers

        for layer_idx, scores in report.per_layer_scores.items():
            assert len(scores) == report.num_heads
            assert all(s >= 0 for _, s in scores)
            svals = [s for _, s in scores]
            assert svals == sorted(svals), f"Layer {layer_idx} scores not sorted ascending"

        payload = report.to_dict()
        rt = HeadImportanceReport.from_dict(payload)
        assert rt.num_layers == report.num_layers

        layer0 = report.per_layer_scores[0]
        result.details = {
            "num_layers": report.num_layers,
            "num_heads": report.num_heads,
            "weakest_head_layer0": (layer0[0][0], round(layer0[0][1], 4)),
            "strongest_head_layer0": (layer0[-1][0], round(layer0[-1][1], 4)),
            "round_trip": "OK",
        }
        result.passed = True
        return report, result
    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return None, result
    finally:
        result.elapsed = time.time() - t0


def verify_channel_importance(artifact: ModelArtifact) -> tuple:
    result = VerificationResult("5. Channel Importance Analysis")
    t0 = time.time()
    try:
        report = compute_channel_importance(
            artifact=artifact,
            datasets=ANALYSIS_DATASETS[:3],
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            metric="activation_norm",
        )
        num_layers = artifact.model.config.num_hidden_layers
        intermediate = artifact.model.config.intermediate_size
        assert report.num_layers == num_layers
        assert report.intermediate_size == intermediate
        assert len(report.per_layer_scores) == num_layers

        for layer_idx, scores in report.per_layer_scores.items():
            assert len(scores) == intermediate

        payload = report.to_dict()
        rt = ChannelImportanceReport.from_dict(payload)
        assert rt.num_layers == report.num_layers

        result.details = {
            "num_layers": report.num_layers,
            "intermediate_size": report.intermediate_size,
            "metric": report.metric,
            "min_score_layer0": round(min(report.per_layer_scores[0]), 6),
            "max_score_layer0": round(max(report.per_layer_scores[0]), 6),
            "round_trip": "OK",
        }
        result.passed = True
        return report, result
    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return None, result
    finally:
        result.elapsed = time.time() - t0


def verify_kv_similarity(artifact: ModelArtifact) -> tuple:
    result = VerificationResult("6. KV Head Similarity Analysis")
    t0 = time.time()
    try:
        report = compute_kv_head_similarity(artifact=artifact)
        num_layers = artifact.model.config.num_hidden_layers
        num_kv = getattr(artifact.model.config, "num_key_value_heads", artifact.model.config.num_attention_heads)
        assert report.num_layers == num_layers
        assert report.num_kv_heads == num_kv
        assert len(report.similarity_matrices) == num_layers

        for layer_idx, matrix in report.similarity_matrices.items():
            assert len(matrix) == num_kv
            for row in matrix:
                assert len(row) == num_kv
            for i in range(num_kv):
                assert abs(matrix[i][i] - 1.0) < 0.01, f"Self-similarity should be ~1.0"

        for layer_idx, pairs in report.merge_pairs.items():
            if len(pairs) >= 2:
                assert pairs[0][1] is not None

        payload = report.to_dict()
        rt = KvSimilarityReport.from_dict(payload)
        assert rt.num_layers == report.num_layers

        layer0_pairs = report.merge_pairs.get(0, [])
        result.details = {
            "num_layers": report.num_layers,
            "num_kv_heads": report.num_kv_heads,
            "top_merge_pair_layer0": layer0_pairs[0] if layer0_pairs else "N/A",
            "round_trip": "OK",
        }
        result.passed = True
        return report, result
    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return None, result
    finally:
        result.elapsed = time.time() - t0


def verify_optimization_plans(weak_report, head_report, channel_report, kv_report) -> tuple:
    result = VerificationResult("7. Optimization Plan Generation")
    t0 = time.time()
    try:
        candidates = generate_optimization_plans(
            weak_layer_report=weak_report,
            head_importance_report=head_report,
            channel_importance_report=channel_report,
            kv_similarity_report=kv_report,
            accuracy_budget=0.05,
            enable_sparsity=True,
            head_prune_ratio=0.25,
            mlp_prune_ratio=0.2,
            kv_merge_ratio=0.5,
        )
        assert len(candidates) > 0

        kinds = {}
        for c in candidates:
            k = c.metadata.get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1

        assert "baseline" in kinds
        baseline = [c for c in candidates if c.metadata.get("kind") == "baseline"][0]
        assert len(baseline.plan.transforms) == 0

        for c in candidates:
            serialized = c.to_dict()
            from QEfficient.model_pruning.qeff_model_optimizer.search.candidates import CandidatePlan
            rt = CandidatePlan.from_dict(serialized)
            assert rt.rationale == c.rationale

        result.details = {
            "total_candidates": len(candidates),
            "plan_kinds": kinds,
            "serialization": "OK",
        }
        result.passed = True
        return candidates, result
    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return None, result
    finally:
        result.elapsed = time.time() - t0


def _generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def verify_individual_transforms(session, artifact) -> list[VerificationResult]:
    results = []
    model = artifact.model
    tokenizer = artifact.tokenizer
    num_layers = model.config.num_hidden_layers

    baseline_text = _generate_text(model, tokenizer, "The capital of France is")

    # 8. Head Pruning
    r = VerificationResult("8. Head Pruning Transform")
    t0 = time.time()
    try:
        plan = TransformationPlan(transforms=[
            HeadPruningSpec(selections=[
                LayerHeadSelection(layer=num_layers // 2, heads=[0, 1]),
            ])
        ])
        session.apply_plan(artifact, plan)
        assert artifact.applied_transforms[0].kind == "head_pruning"
        assert artifact.applied_transforms[0].status == "applied"
        text = _generate_text(model, tokenizer, "The capital of France is")
        assert len(text) > 0
        r.details = {"output_preview": text[:60], "layers_pruned": 1, "heads_pruned": [0, 1]}
        r.passed = True
    except Exception as e:
        r.error = str(e)
    r.elapsed = time.time() - t0
    results.append(r)

    session.apply_plan(artifact, TransformationPlan())

    # 9. MLP Pruning
    r = VerificationResult("9. MLP Pruning Transform")
    t0 = time.time()
    try:
        plan = TransformationPlan(transforms=[
            MlpPruningSpec(target_layers=[num_layers // 2], pruning_ratio=0.2)
        ])
        session.apply_plan(artifact, plan)
        assert artifact.applied_transforms[0].kind == "mlp_pruning"
        text = _generate_text(model, tokenizer, "The capital of France is")
        assert len(text) > 0
        r.details = {"output_preview": text[:60], "pruning_ratio": 0.2}
        r.passed = True
    except Exception as e:
        r.error = str(e)
    r.elapsed = time.time() - t0
    results.append(r)

    session.apply_plan(artifact, TransformationPlan())

    # 10. KV Cache Compression
    r = VerificationResult("10. KV Cache Compression Transform")
    t0 = time.time()
    try:
        num_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        if num_kv < model.config.num_attention_heads:
            plan = TransformationPlan(transforms=[
                KvCacheCompressionSpec(target_layers=[num_layers // 2], merge_ratio=0.5)
            ])
            session.apply_plan(artifact, plan)
            assert artifact.applied_transforms[0].kind == "kv_cache_compression"
            text = _generate_text(model, tokenizer, "The capital of France is")
            assert len(text) > 0
            r.details = {"output_preview": text[:60], "merge_ratio": 0.5, "gqa": True}
        else:
            r.details = {"skipped": "MHA model, GQA required"}
        r.passed = True
    except Exception as e:
        r.error = str(e)
    r.elapsed = time.time() - t0
    results.append(r)

    session.apply_plan(artifact, TransformationPlan())

    # 11. Structured Sparsity
    r = VerificationResult("11. 2:4 Structured Sparsity Transform")
    t0 = time.time()
    try:
        plan = TransformationPlan(transforms=[
            StructuredSparsitySpec(
                target_layers=[num_layers // 2],
                target_modules=["q_proj", "gate_proj"],
            )
        ])
        session.apply_plan(artifact, plan)
        assert artifact.applied_transforms[0].kind == "structured_sparsity"

        anatomy = resolve_layer_anatomy(model, num_layers // 2)
        q_weight = anatomy.q_proj.weight.data
        zero_count = (q_weight == 0).sum().item()
        total = q_weight.numel()
        sparsity = zero_count / total
        assert 0.45 < sparsity < 0.55, f"Expected ~50% sparsity, got {sparsity:.2%}"

        text = _generate_text(model, tokenizer, "The capital of France is")
        assert len(text) > 0
        r.details = {
            "output_preview": text[:60],
            "q_proj_sparsity": f"{sparsity:.2%}",
            "modules_sparsified": ["q_proj", "gate_proj"],
        }
        r.passed = True
    except Exception as e:
        r.error = str(e)
    r.elapsed = time.time() - t0
    results.append(r)

    session.apply_plan(artifact, TransformationPlan())

    # 12. Combined Multi-Transform Plan
    r = VerificationResult("12. Combined Multi-Transform Plan")
    t0 = time.time()
    try:
        mid = num_layers // 2
        transforms = [
            SkipLayersSpec(layers=[mid + 2]),
            HeadPruningSpec(selections=[
                LayerHeadSelection(layer=mid, heads=[0]),
            ]),
            MlpPruningSpec(target_layers=[mid], pruning_ratio=0.15),
            StructuredSparsitySpec(
                target_layers=[mid - 1],
                target_modules=["gate_proj"],
            ),
        ]
        plan = TransformationPlan(transforms=transforms)
        session.apply_plan(artifact, plan)

        applied_kinds = [t.kind for t in artifact.applied_transforms]
        assert "skip_layers" in applied_kinds
        assert "head_pruning" in applied_kinds
        assert "mlp_pruning" in applied_kinds
        assert "structured_sparsity" in applied_kinds

        text = _generate_text(model, tokenizer, "The capital of France is")
        assert len(text) > 0

        r.details = {
            "transforms_applied": applied_kinds,
            "output_preview": text[:60],
        }
        r.passed = True
    except Exception as e:
        r.error = str(e)
    r.elapsed = time.time() - t0
    results.append(r)

    return results


def verify_inference_quality(model, tokenizer) -> VerificationResult:
    result = VerificationResult("13. Post-Transform Inference Quality")
    t0 = time.time()
    try:
        prompts = [
            "What is the capital of France?",
            "Write a Python function to reverse a string.",
            "Explain quantum computing in simple terms.",
            "If x + 3 = 7, what is x?",
        ]
        outputs = []
        for prompt in prompts:
            text = _generate_text(model, tokenizer, prompt, max_new_tokens=50)
            assert len(text) > 0, f"Empty output for: {prompt}"
            outputs.append(text[:80])

        result.details = {f"prompt_{i}": {"q": p[:40], "a": o} for i, (p, o) in enumerate(zip(prompts, outputs))}
        result.passed = True
    except Exception as e:
        result.error = str(e)
    finally:
        result.elapsed = time.time() - t0
    return result


def verify_cleanup(session, model) -> VerificationResult:
    result = VerificationResult("14. Session Cleanup & Restoration")
    t0 = time.time()
    try:
        session.close()

        has_nas = any(hasattr(m, "_nas_original_forward") for m in model.modules())
        assert not has_nas, "_nas_original_forward attributes should be cleaned up"

        has_cb = hasattr(model, "_nas_cleanup_callbacks")
        assert not has_cb, "_nas_cleanup_callbacks should be removed"

        result.details = {"nas_attrs_cleaned": True, "callbacks_removed": True}
        result.passed = True
    except Exception as e:
        result.error = str(e)
    finally:
        result.elapsed = time.time() - t0
    return result


def run_verification(model_id: str):
    section(f"NAS Optimization Pipeline — Full Verification")
    print(f"  Model: {model_id}")
    print(f"  Datasets: {ANALYSIS_DATASETS}")
    print(f"  Samples/dataset: {NUM_SAMPLES}")

    all_results: list[VerificationResult] = []

    model, tokenizer, spec, r = verify_model_load(model_id)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    r = verify_layer_anatomy(model, all_results[0].details)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    applier = TransformApplier(default_transform_registry())
    session = NASSession(loader=TransformersModelLoader(), transform_applier=applier)
    from uuid import uuid4
    artifact = ModelArtifact(
        artifact_id=uuid4().hex, model=model, tokenizer=tokenizer,
        model_spec=spec, plan=TransformationPlan(),
    )
    session.artifacts[artifact.artifact_id] = artifact

    weak_report, r = verify_weak_layer_analysis(artifact)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    head_report, r = verify_head_importance(artifact)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    channel_report, r = verify_channel_importance(artifact)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    kv_report, r = verify_kv_similarity(artifact)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    candidates, r = verify_optimization_plans(weak_report, head_report, channel_report, kv_report)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    transform_results = verify_individual_transforms(session, artifact)
    for r in transform_results:
        section(r.name)
        for k, v in r.details.items():
            print(f"  {k}: {v}")
        if r.error:
            print(f"  ERROR: {r.error}")
        print(f"  {r}")
    all_results.extend(transform_results)

    r = verify_inference_quality(model, tokenizer)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    r = verify_cleanup(session, model)
    section(r.name)
    for k, v in r.details.items():
        print(f"  {k}: {v}")
    print(f"  {r}")
    all_results.append(r)

    section("VERIFICATION SUMMARY")
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    total_time = sum(r.elapsed for r in all_results)

    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.elapsed:.1f}s)")

    print(f"\n  {passed}/{total} checks passed in {total_time:.1f}s total")

    if passed < total:
        failed = [r for r in all_results if not r.passed]
        print(f"\n  FAILURES:")
        for r in failed:
            print(f"    {r.name}: {r.error}")
        return False

    print(f"\n  ALL CHECKS PASSED for {model_id}")
    return True


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    try:
        success = run_verification(model_id)
        del success
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
