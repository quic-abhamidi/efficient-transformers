# NAS Examples

End-to-end examples for the NAS (Neural Architecture Search) toolkit. Each
example is a self-contained Python script that exercises a different slice of
the framework, from simple analysis to full-pipeline optimization and QAIC
deployment.

## What is NAS?

NAS is an API-first toolkit for analyzing and optimizing decoder-only
transformer models (Llama, Qwen, Mistral, Gemma...). Given a loaded model, it
can:

1. **Analyze** — find layers, heads, and MLP channels that contribute least
2. **Transform** — apply reversible structural changes (layer skipping, head
   pruning, MLP pruning, KV compression, sparsity)
3. **Evaluate** — measure how each transformation affects perplexity and
   task accuracy
4. **Deploy** — compile optimized models to Qualcomm AI Cloud (QAIC)
   hardware for production inference

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (for GPU analysis and evaluation)
- HuggingFace Transformers
- ``QEfficient`` (for QAIC compilation examples only — 03/04)
- QAIC hardware or simulator (for QAIC compile/run — 03/04)
- ``matplotlib`` (for chart examples)

Install NAS (from the repo root):

```bash
pip install -e .
```

## Example guide

### [01_analysis_only.py](01_analysis_only.py) - Analyze model weak layers and heads
**Requirements**: 1x GPU (or CPU with patience)

The simplest starting point. Loads a model, runs the NAS analysis suite, and
saves per-layer contribution charts. No transforms are applied. Good for
answering "which layers are weakest and which heads can I prune?" without
committing to any optimization yet.

**Runs in**: ~2 minutes for Qwen3-4B on A100, ~1 minute for smaller models.

### [02_evaluate_plans.py](02_evaluate_plans.py) - Rank optimization plans by perplexity
**Requirements**: 1x GPU

Shows the `PlanEvaluator` in action. Builds several transformation plans
(baseline, skip 1 layer, skip 3 layers, head prune, etc.), applies each in
turn to the same model, measures perplexity on real datasets, and ranks them.
Output: a table of plans sorted by quality and a chart comparing them.

**Runs in**: ~5-10 minutes depending on plan count and sample size.

### [03_qaic_deployment.py](03_qaic_deployment.py) - Compile to QAIC and benchmark
**Requirements**: 1x QAIC device (minimum)

Takes the optimal plan from example 02 and compiles it onto QAIC hardware
using `QAICBenchmarkRunner`. Runs inference against a baseline to measure
real hardware speedup. Demonstrates both single-device (low-latency) and
multi-device (high-throughput) compilation patterns.

**Runs in**: ~10-30 minutes per configuration (compilation is slow).

### [04_full_pipeline.py](04_full_pipeline.py) - Complete end-to-end workflow
**Requirements**: 1x GPU + 1-4x QAIC devices

Ties everything together: analysis → plan generation → GPU evaluation →
optimal plan selection → QAIC compilation → performance comparison. This is
the script to copy if you're building a production optimization pipeline.

**Runs in**: ~30-60 minutes for a 14B model.

## Common patterns

### Reuse a single model across many plans

Loading a 14B model takes 5+ seconds and ~28GB of GPU memory. Example 02
shows how to load once and evaluate many plans by applying and rolling back
transforms between runs. Do this rather than reloading.

### Match calibration datasets to evaluation datasets

Phase 1 analysis uses calibration samples to identify weak layers. Phase 2
evaluation measures how much quality drops when you actually skip those
layers. Use the same datasets for both — it gives you a consistent signal
and lets you trust the relative ranking.

### Accuracy-threshold-based plan selection

`PlanEvaluator.select_best(results, accuracy_threshold=10)` picks the most
aggressive plan whose PPL is within 10% of baseline. This is usually what
you want — maximum optimization that still preserves usable quality.

### Structural vs hook-based transforms

The NAS framework transforms are **hook-based** — they modify the forward
pass via torch hooks, leaving weights untouched. This is great for GPU
evaluation (reversible, composable) but QAIC compilers sometimes need
**structural** removal (actual weight deletion). For production QAIC
deployment you may need to switch to the `remove_layers=` path of
`QEFFAutoModelForCausalLM` (see notes in example 03).

## Output artifacts

Each example writes its outputs to `results/<example-name>/`:

- `*.json` — analysis reports, plan results, benchmark data
- `*.png` — charts (requires matplotlib)
- Per-plan sub-files for easy selective loading

These files are consumed by the NAS reporting tooling and can be loaded back
into typed dataclasses via `.from_dict()` classmethods on each report type.

## Common problems

**Model download is slow** — Qwen3-14B is ~28GB. Pre-download with:
```bash
huggingface-cli download Qwen/Qwen3-14B
```

**Out of GPU memory** — reduce `num_samples` or use a smaller model
(Qwen3-4B instead of Qwen3-14B). For pure QAIC deployment you only need
CPU memory, not GPU.

**QAIC compilation fails** — check `ls /dev/accel/` for device presence,
verify QEfficient install with `python -c "import QEfficient"`. Note that
4-device compilation requires all 4 devices to be idle.

**Hook-based transforms don't give QAIC speedup** — if you apply skip_layers
hooks and see no perf change on QAIC, the compiler is still allocating
compute for the no-op layer. For real QAIC speedup, use structural removal
(see example 03 notes).

## Reading order

If you're new, follow the examples in order (01 → 02 → 03 → 04). Each
builds on concepts from the previous ones.

If you're porting an existing optimization pipeline, start with
`04_full_pipeline.py` and strip out the pieces you don't need.

## Getting help

Each example has extensive inline comments. The NAS API itself is documented
via docstrings — use `help(nas.evaluation.PlanEvaluator)` to explore.

For architectural questions about the framework, read the module docstrings
in `nas/analysis/__init__.py`, `nas/evaluation/__init__.py`, and
`nas/transforms/__init__.py`.
