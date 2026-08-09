# NAS API — Usage Guide

A practical reference for the new API-first NAS package at `nas/`. Examples below show the intended API shape after integration into QEfficient; run the model_pruning test suite and hardware validation in your target environment before relying on numeric results.

For the architectural rationale see [`api_first_nas_design.md`](./api_first_nas_design.md); for the migration roadmap see [`api_first_nas_implementation_plan.md`](./api_first_nas_implementation_plan.md).

---

## 1. What you get

Instead of flipping flags on a script, you compose a run out of typed objects:

```
ModelSpec + TransformationPlan + Runtime
                │
                ▼
        nas.api.run(...)   ◄── one call: load → apply → evaluate → cleanup
                │
                ▼
             results
```

Under the hood `run()` uses an `NASSession`, which is also available directly for multi-step workflows (see §4.5). The pieces — `TransformersModelLoader`, `TransformApplier`, `HuggingFaceRuntime`, `QEffRuntime`, `ArtifactManifest` — are all independently usable.

Key traits:
- **No patched `transformers/` required.** Uses stock HF `AutoModelForCausalLM`.
- **Reversible mutations.** Every transform registers a cleanup callback; cleanup runs automatically on `run()` completion or `NASSession.close()`.
- **Serializable plans.** Any plan (skip + compensation + head-pruning + linear-attention) round-trips through JSON.
- **Two execution backends.** `HuggingFaceRuntime` (GPU/CPU via `lm_eval`) and `QEffRuntime` (QAIC via `QEfficient`).
- **The legacy CLI still works.** `benchmarking/run_benchmark.py` now delegates to this API internally when the flag combination is supported. `run_pipeline.py` still drives the legacy `analysis/` and `optimization/` modules (see §10).

---

## 2. Environment

There is no `setup.py` / `pyproject.toml` yet, so the package is used in-place via `PYTHONPATH`.

```bash
pyenv activate nas                          # python 3.10 env with torch+transformers+lm_eval
# Install QEfficient or run commands with python -m from the repository root

python -c "import nas; print(nas.__version__)"   # → 0.1.0
```

Minimum versions (what CI/main currently uses):
- Python 3.10
- `torch>=2.0`
- `transformers>=4.50` (older versions still return tuples from decoder `forward`; the transform layer expects the modern bare-tensor return)
- `lm-eval>=0.4`
- `QEfficient` only needed if you use `QEffRuntime`

---

## 3. Quick start — 30 seconds

Load Qwen3-1.7B, skip layer 15, run 5 gsm8k samples — one function call:

```python
import torch
from QEfficient.model_pruning.qeff_model_optimizer.api import run
from QEfficient.model_pruning.qeff_model_optimizer.runtimes import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config import EvalSpec, ModelSpec, SkipLayersSpec, TransformationPlan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results = run(
    ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map=DEVICE),
    plan=TransformationPlan(transforms=[SkipLayersSpec(layers=[15])]),
    runtime=HuggingFaceRuntime(
        EvalSpec(tasks=["gsm8k"], batch_size=4, device=DEVICE, limit=5, num_fewshot=2),
    ),
)

print(results["results"]["gsm8k"])
# → {'exact_match,strict-match': 0.2, 'exact_match,flexible-extract': 0.2, ...}
```

`run()` handles load → apply plan → evaluate → cleanup in one call. Baseline (no plan) scores `exact_match,strict-match: 0.4`.

For multi-step workflows that keep the model resident between calls (e.g. a REPL, a server, manual generation), use `NASSession` directly — see §4.4.

---

## 4. Core concepts

### 4.1 `ModelSpec` — how to load the base model

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import ModelSpec

spec = ModelSpec(
    model_id="Qwen/Qwen3-1.7B",
    revision=None,                   # pin a HF revision if desired
    trust_remote_code=True,
    dtype="bfloat16",                # "float32" | "float16" | "bfloat16"
    device_map="cuda",               # or "auto", "cpu", a device string
)
```

Serialization: `spec.to_dict()` / `ModelSpec.from_dict(...)`.

### 4.2 Transform specs — what to change about the model

All transforms live in `nas.specs`. Each is a small, validated dataclass.

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import (
    SkipLayersSpec,                    # drop specific decoder blocks
    RemoveLayersSpec,                  # structural deletion (recognized; HF apply not yet wired)
    CompensationSpec,                  # replace skipped signal with a learned/precomputed delta
    HeadPruningSpec,                   # attention-head masking
    LinearAttentionSpec,               # swap attention implementation (spec only; apply not yet wired)
    ScaledCompensationConfig,          # one of 13 compensation strategies
)
```

### 4.3 `TransformationPlan` — an ordered list of transforms

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import (
    CompensationSpec, ScaledCompensationConfig,
    SkipLayersSpec, TransformationPlan,
)

plan = TransformationPlan(
    transforms=[
        SkipLayersSpec(layers=[10, 15]),
        CompensationSpec(
            config=ScaledCompensationConfig(
                mean_delta_path="/path/to/mean_delta.pt", alpha=0.6,
            ),
        ),
    ],
    compatibility_mode="strict",       # or "best_effort"
)
```

Order matters: `CompensationTransform` reads the most recent `skip_layers` record off the artifact, so `SkipLayersSpec` must come first.

### 4.4 `run()` — the one-shot entry point

```python
from QEfficient.model_pruning.qeff_model_optimizer.api import run

results = run(model_spec, runtime=runtime, plan=plan)   # or plan=None for baseline
```

Signature:

```python
def run(
    model_spec: ModelSpec,
    runtime: BaseRuntime,
    plan: TransformationPlan | None = None,
    *,
    loader=None,              # defaults to TransformersModelLoader()
    transform_applier=None,   # defaults to TransformApplier() with default registry
) -> Any:
```

Internally it opens a `NASSession`, loads the model, applies the plan if one was given, evaluates, and cleans up. Use this for any workflow that is "one model, one plan, one evaluation."

### 4.5 `NASSession` — for multi-step workflows

Use the session directly when you need to:
- keep a loaded model resident across multiple evaluations,
- swap plans on the same loaded model (e.g. search a plan space without reloading weights),
- run your own `model.generate()` between transform changes,
- manage several artifacts at once.

```python
from QEfficient.model_pruning.qeff_model_optimizer.api import NASSession

with NASSession() as session:          # default loader + applier
    artifact = session.load(spec)
    artifact = session.apply_plan(artifact, plan_a)
    results_a = session.evaluate(artifact, runtime)

    artifact = session.apply_plan(artifact, plan_b)   # swap without reloading
    results_b = session.evaluate(artifact, runtime)
# session.close() runs; forward hooks removed; artifacts cleared
```

If you exit without the context manager, call `session.close()` explicitly. One failing cleanup no longer strands the others — `close()` is exception-safe.

### 4.6 Runtimes — where to execute

| Runtime | Backend | Use when |
|---|---|---|
| `HuggingFaceRuntime(eval_spec)` | `lm_eval` on stock HF model (CPU/CUDA) | Benchmark on GPU or CPU |
| `QEffRuntime(compile_spec, prepare_mode="auto")` | `QEFFAutoModelForCausalLM` on QAIC | Target Cloud AI 100 cards |

Both runtimes satisfy the same `BaseRuntime.evaluate(artifact)` contract, so `run()` and `session.evaluate` are uniform across backends.

### 4.7 `ArtifactManifest` — reproducibility

```python
from QEfficient.model_pruning.qeff_model_optimizer.serialization import (
    ArtifactManifest, EnvironmentInfo, dump_manifest, load_manifest,
)

manifest = ArtifactManifest(
    model_spec=artifact.model_spec,
    plan=artifact.plan,
    applied_transforms=artifact.applied_transforms,
    environment=EnvironmentInfo.capture(),        # auto-reads transformers/lm_eval/QEfficient versions
    artifact_id=artifact.artifact_id,
    capabilities={"gsm8k_score": 0.4},
)
dump_manifest(manifest, "results/qwen3-1p7b/run.json")

reloaded = load_manifest("results/qwen3-1p7b/run.json")
assert reloaded.plan == manifest.plan
```

Schema version `nas.manifest/v1` is embedded in every manifest.

### 4.8 `analyze_weak_layers` and `generate_candidate_plans` — finding skips automatically

Instead of handpicking which layers to skip, you can let the analysis + search modules propose candidates for you:

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import analyze_weak_layers
from QEfficient.model_pruning.qeff_model_optimizer.search import generate_candidate_plans
from QEfficient.model_pruning.qeff_model_optimizer.config import ModelSpec

report = analyze_weak_layers(
    ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda"),
    datasets=["gsm8k", "hellaswag"],
    num_samples=100,          # per dataset
    batch_size=8,
    metric="cosine",          # or "l2"
)

candidates = generate_candidate_plans(report, max_skip_layers=3, top_k=5)

for cand in candidates[:3]:
    print(cand.priority, cand.metadata, cand.rationale)
```

- **`WeakLayerReport`** ranks every layer weakest-first by mean hidden-state delta. `report.weakest(n)` returns the top `n` candidates; `report.ranked_layers[i].per_dataset_scores` breaks the aggregate score down per dataset.
- **`CandidatePlan`** bundles a `TransformationPlan`, a heuristic `priority` (lower = more conservative), a human-readable `rationale`, and source metadata. Any candidate can be fed straight into `run()` or `session.apply_plan()`.
- Passing `output_dir=...` to `analyze_weak_layers` additionally writes the legacy-format `layer_contributions_<dataset>_<metric>.csv` / `.png` that the legacy pipeline consumers expect.

---

## 5. Recipes

Recipes below use `run()` for one-shot workflows and `NASSession` for multi-step ones. You can always swap one for the other — the pieces are identical, only the wrapper differs.

### 5.1 Load a model and generate (no evaluator needed)

```python
with NASSession(loader=TransformersModelLoader()) as session:
    artifact = session.load(
        ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda"),
    )
    tok = artifact.tokenizer
    inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = artifact.model(**inputs)
    next_tok = int(out.logits[0, -1].argmax())
    print(tok.decode([next_tok]))         # ' Paris'
```

### 5.2 Skip specific decoder layers

```python
with NASSession() as session:
    artifact = session.load(ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda"))
    session.apply_plan(
        artifact,
        TransformationPlan(transforms=[SkipLayersSpec(layers=[0, 1, 2])]),
    )

    # Inspect what was applied:
    for record in artifact.applied_transforms:
        print(record.kind, record.status, record.details)
    # skip_layers applied {'layers': [0, 1, 2], 'model_family': 'qwen3'}
```

Re-applying a new plan overwrites the previous one. The old layers are restored first, then the new ones applied. If the new plan fails mid-apply, the **previous** plan is restored atomically.

### 5.3 Skip + scaled compensation

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import CompensationSpec, ScaledCompensationConfig

plan = TransformationPlan(transforms=[
    SkipLayersSpec(layers=[15, 16]),
    CompensationSpec(
        config=ScaledCompensationConfig(
            mean_delta_path="artifacts/qwen3/mean_delta_layer14.pt",
            alpha=0.5,
        ),
    ),
])

# Either run one evaluation end-to-end:
results = run(model_spec, runtime=HuggingFaceRuntime(eval_spec), plan=plan)

# ...or keep the model resident if you need to introspect / generate:
with NASSession() as session:
    artifact = session.load(model_spec)
    session.apply_plan(artifact, plan)
    # artifact.applied_transforms now contains 'skip_layers' + 'compensation' records
```

The compensation hook attaches to layer `min(skip_layers) - 1` by default (the layer whose output feeds the skip). It pulls the delta tensor from `mean_delta_path` (a `torch.save`d file) and adds `alpha * delta` to the hidden state after that layer fires.

### 5.4 Phase-aware compensation (different prefill vs decode vectors)

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import CompensationSpec, PhaseAwareCompensationConfig

plan = TransformationPlan(transforms=[
    SkipLayersSpec(layers=[15]),
    CompensationSpec(
        config=PhaseAwareCompensationConfig(
            prefill_delta_path="artifacts/qwen3/prefill_delta.pt",
            decode_delta_path="artifacts/qwen3/decode_delta.pt",
            prefill_alpha=1.0,
            decode_alpha=0.8,
        ),
    ),
])
```

The hook inspects `hidden_states.shape[1]` — seq-len of 1 → decode path, otherwise prefill.

### 5.5 Run an lm-eval benchmark (one-shot)

```python
from QEfficient.model_pruning.qeff_model_optimizer.api import run
from QEfficient.model_pruning.qeff_model_optimizer.runtimes import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config import EvalSpec, ModelSpec, SkipLayersSpec, TransformationPlan

results = run(
    ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda"),
    plan=TransformationPlan(transforms=[SkipLayersSpec(layers=[15])]),
    runtime=HuggingFaceRuntime(
        EvalSpec(
            tasks=["gsm8k", "hellaswag"],
            batch_size=8,
            device="cuda",
            limit=100,
            num_fewshot=5,
        ),
    ),
)

for task, metrics in results["results"].items():
    print(task, metrics)
```

Supported dataset aliases (via `BENCHMARK_MAPPING` in `benchmarking/run_benchmark.py`): `gsm8k`, `hellaswag`, `winogrande`, `mmlu`, `arc_easy`, `arc_challenge`, `truthfulqa`, `piqa`, `boolq`, `openbookqa`.

### 5.6 Compile & run on QAIC via QEff

> **Requires** the `QEfficient` library installed and the invoking user in the `qaic` group (or sudo).

```python
from QEfficient.model_pruning.qeff_model_optimizer.api import run
from QEfficient.model_pruning.qeff_model_optimizer.runtimes import QEffRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config import ModelSpec, QEffCompileSpec, SkipLayersSpec, TransformationPlan

compile_spec = QEffCompileSpec(
    ctx_len=4096,
    prefill_seq_len=64,
    batch_size=1,
    num_cores=16,
    continuous_batching=False,
    device_group=[0],
    qaic_config={
        "include_sampler": True,
        "compile_dir": "results/model_pruning/qwen3-qpc",
        "mxfp6_matmul": True,           # compile option, auto-promoted
        "mxint8_kv_cache": True,
    },
)

result = run(
    ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cpu"),
    plan=TransformationPlan(transforms=[SkipLayersSpec(layers=[15])]),
    runtime=QEffRuntime(compile_spec, prepare_mode="auto"),   # "auto" → "object"
)

# prepare_mode="auto"/"object" returns a prepared QEFFAutoModelForCausalLM but
# does NOT call .compile() itself. Invoke compile explicitly to produce a QPC:
qpc_path = result["prepared_model"].compile(
    prefill_seq_len=compile_spec.prefill_seq_len,
    ctx_len=compile_spec.ctx_len,
    batch_size=compile_spec.batch_size,
    num_devices=len(compile_spec.device_group or [0]),
    num_cores=compile_spec.num_cores,
    mxfp6_matmul=True,
)
print(qpc_path)
```

`prepare_mode` options:
- `"auto"` — currently resolves to `"object"`.
- `"object"` — wraps the in-memory artifact with `QEFFAutoModelForCausalLM`. The result dict contains `prepared_model`, `artifact_id`, `runtime`, `prepare_mode`, `model_id`. Compile is up to the caller.
- `"export"` — reserved; currently raises `NotImplementedError`.

Compile-option keys inside `qaic_config` are auto-split: known keys (`mxfp6_matmul`, `mxint8_kv_cache`, `num_speculative_tokens`, `kv_cache_batch_size`, `comp_ctx_lengths_prefill`, etc.) flow into `compile(...)`; everything else stays in `model_qaic_config`.

### 5.7 Save and reload a full run

```python
from pathlib import Path
from QEfficient.model_pruning.qeff_model_optimizer.serialization import (
    ArtifactManifest, EnvironmentInfo, dump_manifest, load_manifest,
)

manifest = ArtifactManifest(
    model_spec=artifact.model_spec,
    plan=artifact.plan,
    applied_transforms=artifact.applied_transforms,
    environment=EnvironmentInfo.capture(),
    artifact_id=artifact.artifact_id,
    capabilities={"gsm8k_score": float(results["results"]["gsm8k"]["exact_match,strict-match"])},
)
path = dump_manifest(manifest, Path("results/qwen3-1p7b-skip15/manifest.json"))

# Later, in a different process / commit / branch:
reloaded = load_manifest(path)
print(reloaded.environment.transformers_version)   # auto-captured at save time
print(reloaded.plan)                                # fully reconstructed typed plan
```

### 5.8 Automated skip-layer search (analysis → candidates → eval)

Measure per-layer contribution, generate ranked candidate plans, evaluate each against a budget — all through the API:

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import analyze_weak_layers
from QEfficient.model_pruning.qeff_model_optimizer.api import run
from QEfficient.model_pruning.qeff_model_optimizer.runtimes import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.search import generate_candidate_plans
from QEfficient.model_pruning.qeff_model_optimizer.config import EvalSpec, ModelSpec

spec = ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")

# 1. Measure per-layer contributions across two datasets.
report = analyze_weak_layers(spec, datasets=["gsm8k", "hellaswag"], num_samples=64)

# 2. Propose candidate skip plans, weakest-first.
candidates = generate_candidate_plans(report, max_skip_layers=3, top_k=5)

# 3. Evaluate a handful of candidates and keep whichever clears an accuracy budget.
eval_spec = EvalSpec(tasks=["gsm8k"], batch_size=4, device="cuda",
                     limit=50, num_fewshot=2)
baseline = run(spec, runtime=HuggingFaceRuntime(eval_spec))
baseline_score = baseline["results"]["gsm8k"]["exact_match,strict-match"]

accepted = []
for cand in candidates[:5]:
    if not cand.plan.transforms:               # skip the baseline candidate
        continue
    result = run(spec, runtime=HuggingFaceRuntime(eval_spec), plan=cand.plan)
    score = result["results"]["gsm8k"]["exact_match,strict-match"]
    drop = baseline_score - score
    if drop <= 0.02:                           # 2-percentage-point budget
        accepted.append((cand, score))
    print(f"{cand.metadata}: score={score:.3f} drop={drop:+.3f}")

print(f"accepted: {len(accepted)} candidates within budget")
```

Each `cand.plan` is a full `TransformationPlan`, so you can also feed it directly to `session.apply_plan()` for manual inspection or to `dump_manifest(...)` for persistence.

---

## 6. Worked examples (verified)

### 6.1 End-to-end parity check on Qwen3-1.7B

```python
# verified on A100, bf16, gsm8k, limit=5, 2-shot
from QEfficient.model_pruning.qeff_model_optimizer.api import run
from QEfficient.model_pruning.qeff_model_optimizer.runtimes import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config import EvalSpec, ModelSpec, SkipLayersSpec, TransformationPlan

spec = ModelSpec(model_id="Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")
eval_spec = EvalSpec(tasks=["gsm8k"], batch_size=4, device="cuda", limit=5, num_fewshot=2)

baseline = run(spec, runtime=HuggingFaceRuntime(eval_spec))
skipped  = run(spec, runtime=HuggingFaceRuntime(eval_spec),
               plan=TransformationPlan(transforms=[SkipLayersSpec(layers=[15])]))

# baseline exact_match,strict-match = 0.4 (21.8s)
# skip[15] exact_match,strict-match = 0.2 (16.3s)
```

### 6.2 Regression-test pattern

`tests/test_nas_fixes_regression.py` shows how to exercise every layer without loading a real model (uses small `nn.Module` fixtures). It is a good starting template for project-specific tests.

---

## 7. Using via the legacy CLI

The CLI wrappers now delegate to this API under the hood whenever the flag combination is supported. You don't have to migrate scripts — existing commands still work.

### 7.1 Benchmark one model on one dataset

```bash
# Install QEfficient or run commands with python -m from the repository root
python -m benchmarking.run_benchmark \
    --model Qwen/Qwen3-1.7B \
    --dataset gsm8k \
    --batch-size 4 \
    --limit 5 \
    --num-fewshot 2 \
    --device cuda \
    --output-dir results/qwen3-baseline
```

Internally this builds `ModelSpec` + `EvalSpec`, loads through `TransformersModelLoader`, and evaluates via `HuggingFaceRuntime`.

### 7.2 Benchmark with skip layers

```bash
python -m benchmarking.run_benchmark \
    --model Qwen/Qwen3-1.7B \
    --dataset gsm8k hellaswag \
    --skip-layers 14 15 \
    --limit 50 \
    --output-dir results/qwen3-skip14-15
```

`--skip-layers` is translated to `TransformationPlan([SkipLayersSpec([14, 15])])` before running.

### 7.3 Benchmark with compensation

```bash
# scaled compensation
python -m benchmarking.run_benchmark \
    --model Qwen/Qwen3-1.7B \
    --dataset gsm8k \
    --skip-layers 15 \
    --use-compensation \
    --compensation-strategy scaled \
    --compensation-vector-file artifacts/qwen3/mean_delta_layer14.pt \
    --compensation-alpha 0.6 \
    --limit 50

# phase-aware compensation (prefill + decode vectors)
python -m benchmarking.run_benchmark \
    --model Qwen/Qwen3-1.7B \
    --dataset gsm8k \
    --skip-layers 15 \
    --use-compensation \
    --compensation-strategy phase_aware \
    --compensation-vector-file artifacts/qwen3/prefill_delta.pt \
    --compensation-decode-vector-file artifacts/qwen3/decode_delta.pt \
    --compensation-alpha 1.0 \
    --limit 50
```

All compensation strategies are also reachable programmatically via `benchmarking.run_benchmark.build_compensation_spec(...)` with the same parameter names (underscored).

### 7.4 Full pipeline (layer analysis → config generation → baseline → skip benchmarks → report)

```bash
python run_pipeline.py \
    --model Qwen/Qwen3-1.7B \
    --datasets gsm8k hellaswag \
    --num-samples 1000 \
    --output-dir runs/qwen3-nas-v1
```

`run_pipeline.py` calls `analysis/`, `optimization/layer_skipping/`, `benchmarking/` modules. Phase 5 of the API migration (moving the analysis + search stages into `nas/`) is not yet complete; these still use the legacy modules.

Resume flags (`--resume-from`, `--force-rerun`, `--skip-failed`, `--retry-failed-only`, `--clean-checkpoint`) remain unchanged — checkpoint semantics are preserved.

---

## 8. Reference

### 8.1 Supported model families (for transforms)

Resolved via `nas/transforms/adapters.py:SUPPORTED_MODEL_TYPES`:

| `config.model_type` | Verified |
|---|---|
| `llama` | Yes (existing fixtures + legacy path) |
| `mistral` | Spec-supported, not yet exercised on a real model |
| `qwen2` | Spec-supported, not yet exercised on a real model |
| `qwen3` | **Yes** — verified end-to-end on `Qwen/Qwen3-1.7B` |
| `gemma3` | Spec-supported, not yet exercised on a real model |

### 8.2 Compensation strategies

All 13 are spec-complete and wired through `CompensationTransform`. CLI strategy name → Python config class:

| `--compensation-strategy` | Python config class | Required files |
|---|---|---|
| `scaled` | `ScaledCompensationConfig` | mean delta |
| `last_token` | `LastTokenCompensationConfig` | mean delta |
| `magnitude_preserving` | `MagnitudePreservingCompensationConfig` | mean delta |
| `cascaded` | `CascadedCompensationConfig` | mean delta (pre-skip fraction) |
| `magnitude_rescaling` | `MagnitudeRescalingCompensationConfig` | mean delta |
| `phase_aware` | `PhaseAwareCompensationConfig` | prefill + decode delta |
| `phase_last_token` | `PhaseAwareLastTokenCompensationConfig` | prefill + decode delta |
| `phase_aware_magnitude_rescaling` | `PhaseAwareMagnitudeRescalingCompensationConfig` | (ratios only) |
| `position_aware` | `PositionAwareCompensationConfig` | bucket deltas |
| `pca` | `PcaCompensationConfig` | PCA file (+ optional mean delta) |
| `multiplicative` | `MultiplicativeCompensationConfig` | scale + bias vectors |
| `learnable` | `LearnableCompensationConfig` | learned module (`.pt`) |
| `multi_phase_aware_magnitude_rescaling` | `MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig` | per-cluster ratios |

### 8.3 `EvalSpec` fields

| Field | Default | Notes |
|---|---|---|
| `tasks` | required | list of lm-eval task names |
| `batch_size` | `16` | |
| `device` | `"cuda"` | `"cuda"`, `"cpu"`, `"auto"` |
| `limit` | `None` | cap samples for fast iteration |
| `num_fewshot` | `None` | overrides task default |
| `use_cache` | `False` | requires `cache_dir` |
| `cache_dir` | `None` | |
| `dtype` | `"auto"` | lm-eval wrapper only; redundant with `ModelSpec.dtype` |
| `log_samples` | `False` | emits per-sample logs |
| `random_seed` | `42` | |
| `verbosity` | `"INFO"` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 8.4 `QEffCompileSpec` fields

| Field | Default | Notes |
|---|---|---|
| `ctx_len` | `4096` | total context window |
| `prefill_seq_len` | `32` | prefill chunk size |
| `batch_size` | `1` | |
| `num_cores` | `16` | AIC cores per device |
| `continuous_batching` | `False` | enables `full_batch_size` compile arg |
| `device_group` | `None` | list of QAIC device ids |
| `qaic_config` | `{}` | dict; compile-option keys auto-split to `compile(...)` |

### 8.5 Manifest schema (`nas.manifest/v1`)

Required keys: `schema_version`, `model_spec`, `plan`, `applied_transforms`, `capabilities`.
Optional keys (omitted entirely when their value is `None`): `artifact_id`, `environment`, `source_control`.

```
{
  "schema_version": "nas.manifest/v1",
  "artifact_id": "...",               # optional
  "model_spec": { ... },              # ModelSpec.to_dict()
  "plan": { "transforms": [...], "compatibility_mode": "strict" },
  "applied_transforms": [
    { "kind": "...", "status": "applied", "details": { ... }, "warnings": [] }
  ],
  "capabilities": { ... },            # caller-defined metrics, notes
  "environment": {                    # optional; emitted only when not None
    "nas_version": "0.1.0",
    "transformers_version": "5.0.0.dev0",
    "qefficient_version": null,
    "lm_eval_version": "0.4.10"
  },
  "source_control": {                 # optional; emitted only when not None
    "repo_git_sha": null,
    "repo_dirty": null
  }
}
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Unsupported model type for v1 skip transform: None` | Model was unwrapped past its `config` or `config.model_type` is unset | Check the model's `config.model_type` and that it is in `SUPPORTED_MODEL_TYPES` |
| `ValueError: Could not locate decoder layers for supported model type '...'` | `resolve_layer_adapter` fell through all candidate paths | Add a new candidate tuple to `nas/transforms/adapters.py` for the model family |
| `TypeError: missing 1 required positional argument: 'config'` on `CompensationSpec()` | Forgot to pass a compensation config | Pass e.g. `CompensationSpec(config=ScaledCompensationConfig(...))` |
| `ValueError: Tokenizer ... has neither pad_token nor eos_token` | Some base models lack both | Set `tokenizer.pad_token` manually before loading via API |
| `NotImplementedError: QEffRuntime prepare_mode='export' is not implemented yet` | Using the reserved mode | Use `prepare_mode="auto"` or `"object"` |
| `ValueError: compensation transform requires skip_layers to be applied earlier in the plan` | `CompensationSpec` ordered before `SkipLayersSpec` | Swap the order in `TransformationPlan.transforms` |
| Long startup + `Fetching ... .safetensors` | First-time model download | Normal; subsequent runs use HF cache (`~/.cache/huggingface/hub`) |

---

## 10. Training-Free Pruning & Optimization (v0.2)

Four new training-free transforms, three new analysis modules, a multi-transform search engine, and nine modern datasets. All transforms are GPU mask-mode (reversible). Validate end-to-end on target models and hardware before publishing numeric claims.

### 10.1 New transform types

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import (
    HeadPruningSpec, LayerHeadSelection,   # mask attention heads
    MlpPruningSpec,                        # mask MLP intermediate channels
    KvCacheCompressionSpec,                # simulate KV head merging
    StructuredSparsitySpec,                # 2:4 weight sparsity pattern
)
```

### 10.2 Head pruning — mask least-important attention heads

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import HeadPruningSpec, LayerHeadSelection, TransformationPlan

plan = TransformationPlan(transforms=[
    HeadPruningSpec(selections=[
        LayerHeadSelection(layer=16, heads=[0, 1]),   # prune heads 0,1 in layer 16
        LayerHeadSelection(layer=17, heads=[3]),       # prune head 3 in layer 17
    ])
])

with NASSession() as session:
    artifact = session.load(spec)
    session.apply_plan(artifact, plan)
    # artifact.applied_transforms[0].kind == "head_pruning"
    # artifact.applied_transforms[0].status == "applied"
```

Pruning zeros the head-specific slices in the pre-`o_proj` space via `register_forward_pre_hook`. Reversible on cleanup.

### 10.3 MLP width pruning — mask least-active channels

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import MlpPruningSpec, TransformationPlan

plan = TransformationPlan(transforms=[
    MlpPruningSpec(
        target_layers=[16, 17],     # empty = all layers
        pruning_ratio=0.2,          # prune 20% of intermediate channels (max 0.5)
        metric="activation_norm",   # or "wanda"
    )
])
```

Pruning zeros the weakest channels in the intermediate activation (input to `down_proj`) via `register_forward_pre_hook`. Channel importance is computed from gate_proj weight norms (fallback) or from a `ChannelImportanceReport` stored in artifact metadata.

### 10.4 KV cache compression — simulate KV head merging

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import KvCacheCompressionSpec, TransformationPlan

plan = TransformationPlan(transforms=[
    KvCacheCompressionSpec(
        target_layers=[16],       # empty = all layers
        merge_ratio=0.5,          # merge 50% of KV heads
        similarity_metric="cosine",
    )
])
```

Only applies to GQA models (`num_kv_heads < num_attention_heads`). Hooks on `k_proj` and `v_proj` average the outputs of similar heads at runtime. Set `allow_mha_to_gqa=True` to override the GQA constraint.

### 10.5 2:4 Structured sparsity

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import StructuredSparsitySpec, TransformationPlan

plan = TransformationPlan(transforms=[
    StructuredSparsitySpec(
        target_layers=[16],             # empty = all layers
        target_modules=["q_proj", "gate_proj"],  # which nn.Linear modules to sparsify
    )
])
```

For every group of 4 contiguous weights, zeros the 2 with smallest absolute magnitude. Original weights are saved on CPU and restored on cleanup.

### 10.6 Combined multi-transform plans

Transforms compose in order: skip → head → MLP → KV → sparsity → compensation.

```python
plan = TransformationPlan(transforms=[
    SkipLayersSpec(layers=[32]),
    HeadPruningSpec(selections=[LayerHeadSelection(layer=16, heads=[0])]),
    MlpPruningSpec(target_layers=[16], pruning_ratio=0.15),
    StructuredSparsitySpec(target_layers=[15], target_modules=["gate_proj"]),
])

with NASSession() as session:
    artifact = session.load(spec)
    session.apply_plan(artifact, plan)
    # All 4 transforms applied, all reversible
```

### 10.7 Analysis — head importance

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import compute_head_importance

report = compute_head_importance(
    artifact,
    datasets=["mmlu_pro", "ifeval", "gsm_hard"],
    num_samples=50,
)

# Per-layer-per-head scores, sorted weakest-first
for head_idx, score in report.per_layer_scores[0][:3]:
    print(f"  Layer 0, Head {head_idx}: importance={score:.4f}")
```

### 10.8 Analysis — channel importance

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import compute_channel_importance

report = compute_channel_importance(
    artifact,
    datasets=["mmlu_pro", "ifeval"],
    num_samples=50,
    metric="activation_norm",    # or "wanda"
)

# Per-layer channel scores (list of floats, ascending)
print(f"Weakest channel score in layer 0: {report.per_layer_scores[0][0]:.6f}")
```

### 10.9 Analysis — KV head similarity

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import compute_kv_head_similarity

report = compute_kv_head_similarity(artifact)

# Weight-only analysis, no calibration data needed
print(f"Most similar KV heads in layer 0: {report.merge_pairs[0][0]}")
# → (2, 6) — heads 2 and 6 have the highest cosine similarity
```

### 10.10 Search — multi-transform optimization plans

```python
from QEfficient.model_pruning.qeff_model_optimizer.search import generate_optimization_plans

candidates = generate_optimization_plans(
    weak_layer_report=weak_report,
    head_importance_report=head_report,
    channel_importance_report=channel_report,
    kv_similarity_report=kv_report,
    accuracy_budget=0.05,
    enable_sparsity=True,
)

# 9 plan variants: baseline, skip-only, head-only, mlp-only,
# kv-only, sparsity-only, conservative, recommended, aggressive
for c in candidates:
    kinds = [t.kind for t in c.plan.transforms]
    print(f"  {c.metadata['kind']:20s} transforms={kinds}")
```

### 10.11 Step-by-step: Find the best optimization for your model

The complete workflow from model loading to finding the best plan. Copy-paste ready.

#### Step 1: Setup and model loading

```python
import torch
from QEfficient.model_pruning.qeff_model_optimizer.api import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.config import ModelSpec, TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry

MODEL_ID = "Qwen/Qwen3-4B"   # or "Qwen/Qwen3-32B", any supported model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

spec = ModelSpec(model_id=MODEL_ID, dtype="bfloat16", device_map="auto")
loader = TransformersModelLoader()
model, tokenizer = loader.load(spec)

print(f"Loaded: {model.config.num_hidden_layers} layers, "
      f"{model.config.num_attention_heads} heads, "
      f"{getattr(model.config, 'num_key_value_heads', '?')} KV heads")
```

#### Step 2: Create a session and artifact

```python
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from uuid import uuid4

applier = TransformApplier(default_transform_registry())
session = NASSession(loader=loader, transform_applier=applier)

artifact = ModelArtifact(
    artifact_id=uuid4().hex, model=model, tokenizer=tokenizer,
    model_spec=spec, plan=TransformationPlan(),
)
session.artifacts[artifact.artifact_id] = artifact
```

#### Step 3: Run all analyses

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis import (
    compute_weak_layer_report, compute_head_importance,
    compute_channel_importance, compute_kv_head_similarity,
)

# Modern datasets for accurate weakness detection
DATASETS = ["mmlu_pro", "bbh_causal", "bbh_logical_deduction",
            "ifeval", "gsm_hard", "humanevalpack", "orca_math"]

# Weak layer analysis (identifies which layers contribute least)
weak_report = compute_weak_layer_report(
    artifact, datasets=DATASETS, num_samples=30, batch_size=4, max_length=512,
)
print(f"Weakest layers: {[(r.layer, round(r.aggregate_score, 4)) for r in weak_report.ranked_layers[:5]]}")

# Head importance (identifies which attention heads matter least)
head_report = compute_head_importance(
    artifact, datasets=DATASETS[:4], num_samples=30, batch_size=4, max_length=512,
)

# Channel importance (identifies which MLP channels are least active)
channel_report = compute_channel_importance(
    artifact, datasets=DATASETS[:4], num_samples=30, batch_size=4, max_length=512,
)

# KV head similarity (identifies redundant KV heads — weight-only, instant)
kv_report = compute_kv_head_similarity(artifact)
```

#### Step 4: Generate optimization plans

```python
from QEfficient.model_pruning.qeff_model_optimizer.search import generate_optimization_plans

candidates = generate_optimization_plans(
    weak_layer_report=weak_report,
    head_importance_report=head_report,
    channel_importance_report=channel_report,
    kv_similarity_report=kv_report,
    accuracy_budget=0.05,       # max 5% accuracy drop tolerance
    enable_sparsity=False,      # 2:4 sparsity is aggressive, start without it
    head_prune_ratio=0.25,      # prune 25% of heads
    mlp_prune_ratio=0.2,        # prune 20% of MLP channels
    kv_merge_ratio=0.5,         # merge 50% of KV heads
)

for c in candidates:
    kinds = [t.kind for t in c.plan.transforms]
    print(f"  {c.metadata['kind']:20s} priority={c.priority:.3f} transforms={kinds}")
```

#### Step 5: Evaluate each candidate by measuring perplexity

```python
def measure_loss(model, tokenizer, prompts, max_length=512):
    """Compute average cross-entropy loss across prompts."""
    total_loss, count = 0.0, 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length).to(model.device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        total_loss += out.loss.item()
        count += 1
    return total_loss / max(count, 1)

from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import load_dataset_samples

eval_prompts = {}
for ds in ["mmlu_pro", "bbh_causal", "ifeval", "gsm_hard", "humanevalpack"]:
    eval_prompts[ds] = load_dataset_samples(ds, 50)

# Measure baseline
baseline_losses = {ds: measure_loss(model, tokenizer, prompts) for ds, prompts in eval_prompts.items()}
baseline_avg = sum(baseline_losses.values()) / len(baseline_losses)
print(f"Baseline avg loss: {baseline_avg:.4f}")

# Evaluate each candidate
results = []
for candidate in candidates:
    plan_kind = candidate.metadata.get("kind", "unknown")
    session.apply_plan(artifact, candidate.plan)

    plan_losses = {ds: measure_loss(model, tokenizer, prompts) for ds, prompts in eval_prompts.items()}
    plan_avg = sum(plan_losses.values()) / len(plan_losses)
    delta_pct = (plan_avg - baseline_avg) / baseline_avg * 100

    results.append({"kind": plan_kind, "avg_loss": plan_avg, "delta_pct": delta_pct, "plan": candidate.plan})
    print(f"  {plan_kind:20s}: loss={plan_avg:.4f} ({delta_pct:+.1f}%)")

# Sort by loss (best first)
results.sort(key=lambda r: r["avg_loss"])
```

#### Step 6: Pick the best plan and measure speed

```python
import time

# Best plan within budget
best = next(r for r in results if r["delta_pct"] <= 5.0 and r["kind"] != "baseline")
print(f"\nBest plan: {best['kind']} ({best['delta_pct']:+.1f}% accuracy loss)")

# Apply it
session.apply_plan(artifact, best["plan"])

# Measure generation speed
prompt = "Write a detailed explanation of how neural networks learn."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
torch.cuda.synchronize()
elapsed = time.time() - t0

gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print(f"Speed: {gen_tokens/elapsed:.1f} tokens/sec ({gen_tokens} tokens in {elapsed:.2f}s)")
print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}")
```

#### Step 7: Clean up when done

```python
session.close()   # Restores all model patches, removes hooks
```

### 10.12 Recommended plans (verified results)

Based on extensive benchmarking across 7 modern datasets with 50 samples each:

#### Qwen3-4B — Best: Skip 1 layer + light head prune

```python
from QEfficient.model_pruning.qeff_model_optimizer.config import (
    HeadPruningSpec, LayerHeadSelection, SkipLayersSpec, TransformationPlan,
)

# Skip layer 32 (weakest) + prune weakest 4 heads in 5 layers
best_plan_4b = TransformationPlan(transforms=[
    SkipLayersSpec(layers=[32]),
    HeadPruningSpec(selections=[
        # Prune 4 weakest heads per layer (out of 32)
        # Actual head indices come from compute_head_importance()
        LayerHeadSelection(layer=0, heads=[13, 24, 28, 31]),
        LayerHeadSelection(layer=9, heads=[5, 11, 19, 27]),
        LayerHeadSelection(layer=18, heads=[3, 7, 22, 29]),
        LayerHeadSelection(layer=27, heads=[1, 8, 14, 30]),
        LayerHeadSelection(layer=35, heads=[0, 6, 17, 25]),
    ]),
])
```

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Avg cross-entropy loss | 2.1879 | 2.2436 | **+2.5%** |
| Avg perplexity | 11.5 | 12.3 | +6.5% |
| Generation speed | 41.9 tok/s | 42.8 tok/s | **+2.1%** |
| Prefill latency | 24.7ms | 24.3ms | **-1.6%** |
| GPU memory | 8.05 GB | 8.05 GB | 0% |

Per-dataset perplexity (lower = better):

| Dataset | Baseline | Optimized | Δ |
|---------|----------|-----------|---|
| MMLU-Pro | 4.5 | 4.6 | +2.2% |
| BBH Causal | 10.7 | 11.2 | +4.7% |
| BBH Logical | 5.5 | 5.9 | +7.3% |
| GSM-Hard | 14.8 | 16.3 | +10.1% |
| HumanEvalPack | 3.7 | 4.0 | +8.1% |
| IFEval | 31.8 | 34.0 | +6.9% |
| Orca-Math | 9.7 | 10.0 | +3.1% |

#### Qwen3-32B — Best: Skip 2 layers

```python
# Skip layers 17 and 19 (weakest middle layers)
best_plan_32b = TransformationPlan(transforms=[
    SkipLayersSpec(layers=[17, 19]),
])
```

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Avg cross-entropy loss | 1.8577 | 1.8826 | **+1.3%** |
| Avg perplexity | 9.0 | 9.2 | +2.0% |
| Generation speed | 18.5 tok/s | 19.1 tok/s | **+3.2%** |
| Prefill latency | 59.8ms | 57.9ms | **-3.2%** |
| GPU memory | 65.53 GB | 65.53 GB | 0% |

Per-dataset perplexity:

| Dataset | Baseline | Optimized | Δ |
|---------|----------|-----------|---|
| MMLU-Pro | 4.0 | 4.0 | 0.0% |
| BBH Causal | 7.8 | 7.8 | 0.0% |
| BBH Logical | 3.8 | 4.1 | +7.9% |
| GSM-Hard | 10.9 | 10.6 | **-2.8%** |
| HumanEvalPack | 1.8 | 1.9 | +5.6% |
| IFEval | 28.0 | 28.4 | +1.4% |
| Orca-Math | 7.1 | 7.5 | +5.6% |

#### What to expect from each technique

| Technique | Typical accuracy cost | Speed gain | When to use |
|-----------|----------------------|------------|-------------|
| **Skip 1 layer** | +0.1% to +2% | +1.5% to +3.5% | Always — nearly free |
| **Skip 2 layers** | +1% to +5% | +3% to +6% | Large models (32B+) with redundant middle layers |
| **Light head prune** (12.5%) | +1% to +3% | +1% to +2% | Combine with skip for extra gains |
| **Head prune 25%** | +5% to +10% | -4% to -10%* | Not recommended without finetuning |
| **MLP prune 20%** | +100%+ | N/A | Too aggressive without finetuning — use 5% or lower |
| **KV compression 50%** | +30%+ | N/A | Too aggressive — use 10-20% merge ratio |
| **2:4 Sparsity** | +100%+ | N/A | Requires post-training calibration — future v2 |

*Head pruning hooks add overhead that can exceed the compute savings on GPU. On QAIC (structural export), this overhead disappears.

### 10.13 Running the benchmark scripts

Three scripts are available for model optimization:

```bash
# 1. Quick search — find the best plan for a model
python scripts/find_best_plan.py "Qwen/Qwen3-4B"

# 2. Full benchmark — detailed performance comparison with speed + quality
python scripts/benchmark_plans.py "Qwen/Qwen3-4B"

# 3. Verification — test all 14 pipeline components end-to-end
python scripts/verify_optimization_pipeline.py "Qwen/Qwen3-4B"
```

All scripts accept any HuggingFace model ID as argument. Results are saved to `results/`.

### 10.15 Modern datasets

Nine modern benchmarks added alongside the 13 legacy datasets:

| Dataset | Category | HF Source |
|---------|----------|-----------|
| `mmlu_pro` | Hard reasoning | `TIGER-Lab/MMLU-Pro` |
| `bbh_causal` | Causal reasoning | `lukaemon/bbh` (causal_judgement) |
| `bbh_logical_deduction` | Logical reasoning | `lukaemon/bbh` (logical_deduction) |
| `ifeval` | Instruction following | `HuggingFaceH4/ifeval` |
| `helpsteer2` | Instruction following | `nvidia/HelpSteer2` |
| `gsm_hard` | Math reasoning | `reasoning-machines/gsm-hard` |
| `orca_math` | Math reasoning | `microsoft/orca-math-word-problems-200k` |
| `humanevalpack` | Code generation | `bigcode/humanevalpack` (python) |
| `metamathqa` | Math + rephrasing | `meta-math/MetaMathQA` |

Default analysis datasets: `mmlu_pro`, `bbh_causal`, `bbh_logical_deduction`, `ifeval`, `gsm_hard`, `humanevalpack`, `orca_math`.

```python
from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import MODERN_DATASETS, DEFAULT_ANALYSIS_DATASETS, SUPPORTED_DATASETS

print(len(SUPPORTED_DATASETS))  # 22 total (13 legacy + 9 modern)
```

### 10.16 LayerAnatomy — sub-module resolution

```python
from QEfficient.model_pruning.qeff_model_optimizer.transforms import resolve_layer_anatomy

anatomy = resolve_layer_anatomy(model, layer_idx=0)
print(anatomy.q_proj)           # Linear(in_features=2560, out_features=4096)
print(anatomy.gate_proj)        # Linear(in_features=2560, out_features=9728)
print(anatomy.num_heads)        # 32
print(anatomy.num_kv_heads)     # 8
print(anatomy.head_dim)         # 128
print(anatomy.intermediate_size) # 9728
```

Resolution uses convention probing (`self_attn.q_proj` → `attention.q_proj` → `attn.q_proj`) with shape-based fallback. No model-type allowlist — works for any HF model following standard naming.

---

## 11. Verification Results

### 11.1 Qwen3-4B (A100 80GB, bfloat16)

Reference validation checklist from the source NAS project; rerun in this QEfficient branch before treating results as current:

| Check | Status | Time |
|-------|--------|------|
| Model Loading | PASS | 10s |
| LayerAnatomy Adapter (layers 0, 18, 35) | PASS | 0s |
| Weak Layer Analysis (5 modern datasets, 8 samples each) | PASS | 22s |
| Head Importance Analysis (3 datasets) | PASS | 9s |
| Channel Importance Analysis (activation_norm) | PASS | 8s |
| KV Head Similarity (weight-only) | PASS | 0s |
| Optimization Plan Generation (9 variants) | PASS | 0s |
| Head Pruning (layer 18, heads [0,1]) | PASS | 1s |
| MLP Pruning (layer 18, ratio=0.2) | PASS | 1s |
| KV Cache Compression (layer 18, merge=0.5) | PASS | 1s |
| 2:4 Structured Sparsity (layer 18, q_proj+gate_proj) | PASS | 1s |
| Combined Plan (skip+head+MLP+sparsity) | PASS | 1s |
| Post-Transform Inference (4 prompts) | PASS | 5s |
| Cleanup & Restoration | PASS | 0s |

**Key findings:**
- Weakest layers: 32, 33, 34, 31, 30 (late layers near output)
- Weakest head in layer 0: head 13 (importance=0.0602)
- Strongest head in layer 0: head 2 (importance=0.2174)
- q_proj sparsity after 2:4: exactly 50.00%

### 11.2 Qwen3-32B (A100 80GB, bfloat16)

All 14 verification checks passed in 187s:

| Check | Status | Time |
|-------|--------|------|
| Model Loading | PASS | 126s |
| LayerAnatomy Adapter (layers 0, 32, 63) | PASS | 0s |
| Weak Layer Analysis (5 modern datasets) | PASS | 21s |
| Head Importance Analysis (3 datasets) | PASS | 11s |
| Channel Importance Analysis (activation_norm) | PASS | 10s |
| KV Head Similarity (weight-only) | PASS | 0s |
| Optimization Plan Generation (9 variants) | PASS | 0s |
| Head Pruning (layer 32, heads [0,1]) | PASS | 2s |
| MLP Pruning (layer 32, ratio=0.2) | PASS | 2s |
| KV Cache Compression (layer 32, merge=0.5) | PASS | 2s |
| 2:4 Structured Sparsity (layer 32, q_proj+gate_proj) | PASS | 2s |
| Combined Plan (skip+head+MLP+sparsity) | PASS | 2s |
| Post-Transform Inference (4 prompts) | PASS | 11s |
| Cleanup & Restoration | PASS | 0s |

**Key findings:**
- Model: 64 layers, 64 heads, 8 KV heads (GQA ratio 8:1)
- Weakest layers: 17, 16, 18, 19, 15 (middle layers)
- Weakest head in layer 0: head 41 (importance=0.0608)
- Strongest head in layer 0: head 30 (importance=0.2904)
- Channel importance range in layer 0: 0.00013 to 0.465
- Top KV merge pair in layer 0: heads (2, 6)

---

## 12. Not yet implemented

Tracked in the [implementation plan](./api_first_nas_implementation_plan.md) and [pruning design](./superpowers/specs/2026-04-24-training-free-pruning-design.md):

- **Structural export mode** for QAIC compilation (`mode="structural"` on transforms — slices weight tensors, updates `model.config`).
- `export_optimized_model()` for one-shot structural export.
- `LinearAttentionTransform` apply logic (spec exists).
- `QEffRuntime prepare_mode="export"` real path.
- Compensation transform extension to cover non-skip transforms.
- Manifest schema version migrator.
- Setuptools-based packaging (`pyproject.toml`).

## 13. Breaking changes since 0.1.0 initial draft

- **`CompensationSpec(config=...)` is now a required argument.** The `config` field was previously declared as optional (defaulting to `None`) but `__post_init__` always rejected `None`, so zero-arg construction never worked. The current declaration moves `config` to the first dataclass field with no default, making the requirement explicit. Call sites that passed `config=` as a keyword (all in-repo call sites do) are unaffected. Any external caller relying on `CompensationSpec("compensation", ScaledCompensationConfig(...))` with positional `kind` must switch to the keyword form.
