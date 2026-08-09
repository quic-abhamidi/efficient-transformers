"""End-to-end verification of the NAS pipeline with Qwen3-4B across all datasets."""

from __future__ import annotations

import gc
import sys
import time
import traceback

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import SUPPORTED_DATASETS
from QEfficient.model_pruning.qeff_model_optimizer.analysis.weak_layers import compute_weak_layer_report
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec, TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.search.candidates import generate_candidate_plans
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry


MODEL_ID = "Qwen/Qwen3-4B"
ALL_DATASETS = list(SUPPORTED_DATASETS.keys())
NUM_SAMPLES = 10
BATCH_SIZE = 4
MAX_LENGTH = 256


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def verify_model_load():
    section("1. Model Loading")
    spec = ModelSpec(model_id=MODEL_ID, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()
    model, tokenizer = loader.load(spec)
    print(f"  Model type   : {model.config.model_type}")
    print(f"  Num layers   : {model.config.num_hidden_layers}")
    print(f"  Hidden size  : {model.config.hidden_size}")
    print(f"  Dtype        : {next(model.parameters()).dtype}")
    print(f"  Device       : {next(model.parameters()).device}")
    print(f"  Pad token    : {tokenizer.pad_token!r}")
    assert model.config.model_type == "qwen3"
    assert tokenizer.pad_token is not None
    print("  PASS: model loaded successfully")
    return model, tokenizer, spec


def verify_weak_layer_analysis(session, artifact):
    section("2. Weak Layer Analysis (all 13 datasets)")
    t0 = time.time()
    report = compute_weak_layer_report(
        artifact=artifact,
        datasets=ALL_DATASETS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        metric="cosine",
        max_length=MAX_LENGTH,
    )
    elapsed = time.time() - t0
    print(f"  Datasets analyzed: {len(report.datasets)}")
    print(f"  Layers ranked   : {len(report.ranked_layers)}")
    print(f"  Elapsed         : {elapsed:.1f}s")

    assert len(report.datasets) == len(ALL_DATASETS), (
        f"Expected {len(ALL_DATASETS)} datasets, got {len(report.datasets)}"
    )
    assert len(report.ranked_layers) == artifact.model.config.num_hidden_layers

    print(f"\n  Top 5 weakest layers (rank 1 = weakest):")
    for r in report.ranked_layers[:5]:
        scores_str = ", ".join(
            f"{ds}={r.per_dataset_scores.get(ds, 0):.4f}" for ds in ALL_DATASETS[:4]
        )
        print(f"    Rank {r.rank:2d}: layer {r.layer:2d}  agg={r.aggregate_score:.4f}  ({scores_str}, ...)")

    ranks = [r.rank for r in report.ranked_layers]
    assert sorted(ranks) == list(range(1, len(ranks) + 1)), "Ranks must be 1..N"
    print("  PASS: weak layer analysis completed for all datasets")
    return report


def verify_candidate_generation(report):
    section("3. Candidate Plan Generation")
    candidates = generate_candidate_plans(report, max_skip_layers=3, top_k=5)
    kinds = {}
    for c in candidates:
        k = c.metadata.get("kind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1

    print(f"  Total candidates: {len(candidates)}")
    for k, v in sorted(kinds.items()):
        print(f"    {k:12s}: {v}")

    assert len(candidates) > 0
    assert kinds.get("baseline", 0) == 1
    assert kinds.get("single", 0) > 0

    for c in candidates:
        for t in c.plan.transforms:
            if isinstance(t, SkipLayersSpec):
                assert len(t.layers) >= 1
                for layer in t.layers:
                    assert 0 <= layer < report.ranked_layers[0].layer + len(report.ranked_layers)
    print("  PASS: candidate generation works correctly")
    return candidates


def verify_skip_transform(session, artifact, candidates):
    section("4. Skip Layer Transform")
    non_baseline = [c for c in candidates if c.metadata.get("kind") != "baseline"]
    if not non_baseline:
        print("  SKIP: no non-baseline candidates")
        return

    best = non_baseline[0]
    skip_spec = best.plan.transforms[0]
    assert isinstance(skip_spec, SkipLayersSpec)
    print(f"  Applying: skip layers {skip_spec.layers}")
    print(f"  Rationale: {best.rationale}")

    session.apply_plan(artifact, best.plan)
    assert len(artifact.applied_transforms) >= 1
    assert artifact.applied_transforms[0].status == "applied"
    assert artifact.applied_transforms[0].kind == "skip_layers"
    print(f"  Applied transforms: {[t.kind for t in artifact.applied_transforms]}")
    print("  PASS: skip transform applied successfully")


def verify_inference(artifact):
    section("5. Post-Transform Inference")
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a Python function to compute fibonacci numbers.",
    ]
    tokenizer = artifact.tokenizer
    model = artifact.model

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  Q: {prompt[:50]}")
        print(f"  A: {response[:100]}")
        assert len(response) > 0, "Model should generate non-empty response"

    print("  PASS: inference works after transform")


def verify_cleanup(session, model):
    section("6. Session Cleanup & Restoration")
    session.close()

    has_nas_attrs = any(
        hasattr(m, "_nas_original_forward")
        for m in model.modules()
    )
    assert not has_nas_attrs, "cleanup should remove all _nas_original_forward attributes"

    has_cleanup_callbacks = hasattr(model, "_nas_cleanup_callbacks")
    assert not has_cleanup_callbacks, "cleanup should remove _nas_cleanup_callbacks"
    print("  PASS: session cleanup restored model correctly")


def main():
    print("NAS Pipeline End-to-End Verification")
    print(f"Model: {MODEL_ID}")
    print(f"Datasets: {len(ALL_DATASETS)} ({', '.join(ALL_DATASETS)})")

    model, tokenizer, spec = verify_model_load()

    applier = TransformApplier(default_transform_registry())
    session = NASSession(loader=TransformersModelLoader(), transform_applier=applier)

    from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
    from uuid import uuid4

    artifact = ModelArtifact(
        artifact_id=uuid4().hex,
        model=model,
        tokenizer=tokenizer,
        model_spec=spec,
        plan=TransformationPlan(),
    )
    session.artifacts[artifact.artifact_id] = artifact

    report = verify_weak_layer_analysis(session, artifact)
    candidates = verify_candidate_generation(report)
    verify_skip_transform(session, artifact, candidates)
    verify_inference(artifact)
    verify_cleanup(session, model)

    section("ALL CHECKS PASSED")
    print("  The NAS pipeline is fully functional with Qwen3-4B.")
    print(f"  Verified: model load, {len(ALL_DATASETS)}-dataset analysis,")
    print("  candidate generation, skip transform, inference, cleanup.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
