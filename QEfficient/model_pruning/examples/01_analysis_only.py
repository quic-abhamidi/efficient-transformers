#!/usr/bin/env python3
"""Example 01: Analyze model weak layers and heads (no transforms applied).

The simplest possible NAS workflow. Loads a model, runs the 4-part analysis
suite (weak layers, head importance, channel importance, KV similarity), and
saves charts + JSON reports.

Use this when you want to understand what's in the model before committing
to any optimization. The output tells you which layers, heads, and channels
are candidates for pruning.

Usage:
    python examples/01_analysis_only.py [--model MODEL_ID]

Defaults to Qwen/Qwen3-4B (smaller than 14B, faster to demo).
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from uuid import uuid4

warnings.filterwarnings("ignore")

import torch

# ── NAS imports ──────────────────────────────────────────────────────────────
# The analysis functions all live in ``nas.analysis`` and return typed reports.
# Each report has ``.to_dict()`` for easy JSON serialisation.
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
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.evaluation import charts  # reusable chart helpers
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry


# ── Configuration ────────────────────────────────────────────────────────────

# Datasets used for calibration. These are defined in ``nas.analysis.datasets``.
# The analysis measures how each layer transforms the hidden state for samples
# drawn from these datasets, so use datasets representative of your workload.
ANALYSIS_DATASETS = ["mmlu_pro", "bbh_causal", "ifeval"]

# Small numbers keep this example runnable in a few minutes. For production
# analysis, use 50-100 samples per dataset.
NUM_SAMPLES = 16
BATCH_SIZE = 2
MAX_LENGTH = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model id")
    parser.add_argument("--output-dir", default="results/example_01_analysis",
                        help="Where to save reports and charts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Example 01: NAS Analysis")
    print(f"  Model:   {args.model}")
    print(f"  Outputs: {output_dir}")
    print(f"{'='*70}")

    # ── Step 1: Load the model ──────────────────────────────────────────────
    # ModelSpec is a typed descriptor for model loading. device_map="auto"
    # lets accelerate figure out the best placement; dtype="bfloat16" halves
    # memory vs float32 with negligible accuracy impact.
    print(f"\n[1/5] Loading model on GPU...")
    t0 = time.time()
    spec = ModelSpec(model_id=args.model, dtype="bfloat16", device_map="auto")
    loader = TransformersModelLoader()
    model, tokenizer = loader.load(spec)
    print(f"  Loaded in {time.time()-t0:.1f}s | "
          f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Step 2: Wrap in NASSession + ModelArtifact ──────────────────────────
    # NASSession is the top-level context manager. ModelArtifact bundles the
    # model, tokenizer, and current transform plan. All analysis APIs operate
    # on artifacts (not raw models) so they can access the loader/spec.
    applier = TransformApplier(default_transform_registry())
    session = NASSession(loader=loader, transform_applier=applier)
    artifact = ModelArtifact(
        artifact_id=uuid4().hex,
        model=model, tokenizer=tokenizer,
        model_spec=spec, plan=TransformationPlan(),
    )
    session.artifacts[artifact.artifact_id] = artifact

    # ── Step 3: Run the 4 analysis passes ───────────────────────────────────
    # Each analysis does forward passes through the calibration samples and
    # measures a different quantity:

    # 3a. Weak layer analysis: per-layer cosine delta between input and output
    #     hidden states. Layers with small deltas barely transform the signal.
    print(f"\n[2/5] Weak layer analysis...")
    t0 = time.time()
    weak_report = compute_weak_layer_report(
        artifact,
        datasets=ANALYSIS_DATASETS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        output_dir=output_dir,  # auto-generates per-dataset CSVs and PNGs
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Top 5 weakest layers: "
          f"{[(r.layer, round(r.aggregate_score, 4)) for r in weak_report.ranked_layers[:5]]}")

    # 3b. Head importance: per-head L2 norm of attention output (pre-o_proj).
    #     Low-norm heads contribute little to the residual stream.
    print(f"\n[3/5] Head importance...")
    t0 = time.time()
    head_report = compute_head_importance(
        artifact,
        datasets=ANALYSIS_DATASETS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    print(f"  Done in {time.time()-t0:.1f}s | "
          f"{head_report.num_layers} layers × {head_report.num_heads} heads")

    # 3c. Channel importance: per-MLP-channel activation magnitude.
    #     Channels with low mean |activation| can be zeroed with little impact.
    print(f"\n[4/5] Channel importance...")
    t0 = time.time()
    channel_report = compute_channel_importance(
        artifact,
        datasets=ANALYSIS_DATASETS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    print(f"  Done in {time.time()-t0:.1f}s | "
          f"intermediate_size={channel_report.intermediate_size}")

    # 3d. KV head similarity: weight-only analysis (no forward passes needed).
    #     Identifies pairs of KV heads whose projection weights are similar and
    #     could be merged (only useful on GQA models where num_kv < num_heads).
    print(f"\n[5/5] KV head similarity...")
    t0 = time.time()
    kv_report = compute_kv_head_similarity(artifact)
    print(f"  Done in {time.time()-t0:.1f}s | "
          f"num_kv_heads={kv_report.num_kv_heads}")

    # ── Step 4: Save reports as JSON ────────────────────────────────────────
    # Every NAS report type has a .to_dict() / .from_dict() pair for round-trip
    # serialisation. Load them back later with the from_dict classmethod.
    print(f"\n  Saving reports...")

    def save_json(data, filename):
        path = output_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"    {path}")

    save_json(
        {
            "model_id": args.model,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "num_kv_heads": getattr(model.config, "num_key_value_heads", None),
            "hidden_size": model.config.hidden_size,
        },
        "model_info.json",
    )
    save_json(
        {
            "ranked_layers": [r.to_dict() for r in weak_report.ranked_layers],
            "metadata": weak_report.metadata,
        },
        "weak_layer_report.json",
    )
    save_json(head_report.to_dict(), "head_importance_report.json")
    save_json(channel_report.to_dict(), "channel_importance_report.json")
    save_json(kv_report.to_dict(), "kv_similarity_report.json")

    # ── Step 5: Generate charts ─────────────────────────────────────────────
    # nas.evaluation.charts provides reusable matplotlib helpers for each
    # analysis type. They accept either the dataclass or its .to_dict().
    print(f"\n  Generating charts...")
    try:
        charts.chart_weak_layers(
            weak_report, output_dir / "weak_layers.png",
            title_prefix=args.model, highlight_weakest_n=5,
        )
        charts.chart_head_importance_heatmap(
            head_report, output_dir / "head_importance_heatmap.png",
            title_prefix=args.model,
        )
        print(f"    Charts saved to {output_dir}/")
    except ImportError:
        print(f"    Skipping charts (install matplotlib for visualization)")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    # session.close() runs any registered cleanup callbacks. In this example
    # no transforms were applied so there's nothing to clean up, but it's
    # good practice.
    session.close()

    print(f"\n{'='*70}")
    print(f"  Analysis complete. Next steps:")
    print(f"  - Review weak_layers.png to see which layers to target")
    print(f"  - Look at head_importance_heatmap.png for head pruning candidates")
    print(f"  - Load these reports in example 02 to evaluate optimization plans")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
