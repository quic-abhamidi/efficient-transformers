# Training-Free Model Pruning & Optimization — Design Spec

**Date**: 2026-04-24
**Status**: Approved
**Goal**: Reduce inference latency on GPU and QAIC without finetuning, tolerating up to 5% accuracy degradation.
**Model scope**: All sizes (1B–70B+), all supported families (llama, mistral, qwen2, qwen3, gemma3).

---

## 1. Overview

Four new pruning/optimization techniques are added to the NAS framework alongside the existing layer skipping + compensation system. All techniques are training-free and operate via mask-mode hooks for GPU experimentation. A future v2 adds structural export for QAIC compilation.

**New transforms:**
- Head Pruning — mask least-important attention heads
- MLP Width Pruning — mask least-active MLP intermediate channels
- KV Cache Compression — simulate KV head merging via hooks
- 2:4 Structured Sparsity — enforce hardware-friendly weight sparsity pattern

**New analysis modules:**
- Head importance scoring
- Channel importance scoring (activation_norm, wanda)
- KV head similarity analysis

**New search:**
- `generate_optimization_plans()` — budget-aware multi-transform candidate generation

**New datasets (modern tier):**
- 9 modern benchmarks spanning hard reasoning, instruction following, math, and code

---

## 2. Architecture

### Composition order in a TransformationPlan

```
1. SkipLayersSpec           (skip entire weak layers)
2. HeadPruningSpec          (prune heads in remaining layers)
3. MlpPruningSpec           (prune MLP channels)
4. KvCacheCompressionSpec   (simulate KV head merging)
5. StructuredSparsitySpec   (apply 2:4 weight sparsity)
6. CompensationSpec         (compensate for skip-layer changes only — see note)
```

Order matters: skip first (no point pruning a skipped layer), compensation last (compensates for all preceding changes).

**Note on CompensationSpec scope:** In v1, the existing `CompensationTransform` hard-requires a prior `skip_layers` record in `artifact.applied_transforms`. Compensation applies only to layer-skipping effects. Plans that include head/MLP/KV/sparsity transforms but no layer skipping should omit `CompensationSpec`. Extending compensation to cover all transform types is a v2 concern.

### New specs added to TransformSpec union

```python
TransformSpec = Union[
    SkipLayersSpec,
    RemoveLayersSpec,
    CompensationSpec,
    HeadPruningSpec,       # existing spec, new apply logic
    LinearAttentionSpec,
    MlpPruningSpec,        # new
    KvCacheCompressionSpec,# new
    StructuredSparsitySpec,# new
]
```

### Transform registry extension

```python
def default_transform_registry():
    return {
        "skip_layers": SkipLayersTransform(),
        "compensation": CompensationTransform(),
        "head_pruning": HeadPruningTransform(),         # new
        "mlp_pruning": MlpPruningTransform(),           # new
        "kv_cache_compression": KvCacheCompressionTransform(),  # new
        "structured_sparsity": StructuredSparsityTransform(),   # new
    }
```

---

## 3. Adapter Extensions — LayerAnatomy

### Problem

The existing `LayerContainerAdapter` only exposes the decoder layer list. New transforms need access to sub-modules within each layer (attention projections, MLP projections). A per-model-type registry doesn't scale.

### Solution: Convention probing + config-derived dimensions

```python
@dataclass(frozen=True)
class LayerAnatomy:
    layer_module: nn.Module

    # Attention sub-modules
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    # MLP sub-modules
    gate_proj: nn.Linear        # or fc1 for non-SwiGLU
    up_proj: nn.Linear | None   # None for non-SwiGLU (e.g., GELU models)
    down_proj: nn.Linear        # or fc2

    # Structure (from model.config)
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
```

**Resolution function:** `resolve_layer_anatomy(model, layer_idx) -> LayerAnatomy`

1. Gets `config` from model (via existing `unwrap_model`)
2. Gets layer module from existing `LayerContainerAdapter`
3. Probes known attribute paths in priority order:
   - Attention: `self_attn.q_proj` → `attention.q_proj` → `attn.q_proj`
   - MLP: `mlp.gate_proj` → `mlp.fc1` → `feed_forward.w1`
4. Falls back to shape-based discovery: walk `layer.named_modules()`, find `nn.Linear` modules, match output dimension against config values
5. Reads `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `intermediate_size` from config

**Why this scales:**
- No model-type allowlist for sub-modules
- No rebasing when transformers updates model classes
- No per-model code — works for any HF model following standard naming
- Graceful failure with clear error messages

### Relation to existing adapter

`LayerContainerAdapter` stays unchanged. `LayerAnatomy` is a per-layer drill-down resolved on-demand by transforms that need it. Layer skipping doesn't need it.

---

## 4. Transform Apply Logic

All transforms implement mask mode only (v1). All are reversible via `run_model_cleanup()`.

### 4A. Head Pruning (HeadPruningTransform)

**Spec:** `HeadPruningSpec` (existing, mode="mask")

**Importance measurement** (`analysis/head_importance.py`):

```python
def compute_head_importance(
    artifact: ModelArtifact,
    datasets: list[str],
    num_samples: int = 50,
    batch_size: int = 4,
    max_length: int = 256,
) -> HeadImportanceReport
```

Metric: `importance(h) = mean over samples of ‖head_output_h‖₂`

Register forward hooks on attention layers, collect per-head output norms. Single forward pass, no gradients. Returns `HeadImportanceReport` with per-layer-per-head scores ranked weakest-first.

**Apply logic (mask mode):** Register a forward hook on `self_attn` that intercepts the intermediate representation **before** `o_proj`. The attention output before `o_proj` has shape `[batch, seq_len, num_heads, head_dim]`. The hook zeros the `head_dim`-wide slices for pruned heads in this pre-projection space, ensuring clean head isolation. This is critical — hooking *after* `o_proj` would zero hidden dimensions, not individual heads, because `o_proj` is a learned projection that mixes head outputs. Implementation: use a `register_forward_pre_hook` on `o_proj` that modifies its input tensor, or monkey-patch the attention forward to zero heads between attention computation and `o_proj`. Reversible via cleanup.

**Note:** `HeadPruningSpec` serialization (`transform_spec_to_dict` / `transform_spec_from_dict`) already exists in the codebase — no serialization changes needed for this transform.

### 4B. MLP Width Pruning (MlpPruningTransform)

**Spec:**

```python
@dataclass
class MlpPruningSpec:
    kind: Literal["mlp_pruning"] = "mlp_pruning"
    target_layers: list[int]     # empty = apply to all layers
    pruning_ratio: float = 0.2      # 0-0.5
    metric: Literal["activation_norm", "wanda"] = "activation_norm"
```

**Importance measurement** (`analysis/channel_importance.py`):

```python
def compute_channel_importance(
    artifact: ModelArtifact,
    datasets: list[str],
    num_samples: int = 50,
    metric: Literal["activation_norm", "wanda"] = "activation_norm",
) -> ChannelImportanceReport
```

Two metrics:
- **activation_norm**: mean absolute activation per channel across samples and tokens
- **wanda** (Sun et al. 2023): `|weight| × ‖input activation‖₂` — accounts for both weight magnitude and actual usage

**Apply logic (mask mode):** Register a forward hook on the MLP that zeros pruned channels in the intermediate activation (between gate_proj/up_proj and down_proj). Reversible via cleanup.

**Validation:** `__post_init__` rejects `pruning_ratio > 0.5`. The `target_layers` field uses "empty = all layers" semantics — this intentionally diverges from `SkipLayersSpec` which requires non-empty layers via `_canonicalize_layers`. New specs must NOT call `_canonicalize_layers` on `target_layers`; use a separate `_canonicalize_target_layers` that permits empty lists.

**Serialization format:**
```json
{"kind": "mlp_pruning", "target_layers": [0, 1, 2], "pruning_ratio": 0.2, "metric": "activation_norm"}
```

### 4C. KV Cache Compression (KvCacheCompressionTransform)

**Spec:**

```python
@dataclass
class KvCacheCompressionSpec:
    kind: Literal["kv_cache_compression"] = "kv_cache_compression"
    target_layers: list[int]      # empty = apply to all layers
    merge_ratio: float = 0.5
    similarity_metric: Literal["cosine", "l2"] = "cosine"
    allow_mha_to_gqa: bool = False  # must be True to merge in MHA models
```

**Similarity analysis** (`analysis/kv_similarity.py`):

```python
def compute_kv_head_similarity(
    artifact: ModelArtifact,
) -> KvSimilarityReport
```

Weight-only analysis — no calibration data needed. Computes pairwise cosine similarity between KV head projection weight slices. Greedy hierarchical clustering identifies merge pairs.

**Apply logic (mask mode — accuracy simulation):** This requires hooking *inside* the attention forward — a simple `register_forward_hook` on `self_attn` fires too late (after the full attention + o_proj). Instead, register `register_forward_hook` on the individual `k_proj` and `v_proj` sub-modules. These hooks fire after each projection computes its output. The hook:
1. Receives the KV projection output shaped `[batch, seq_len, num_kv_heads, head_dim]`
2. Averages outputs of heads that would be merged
3. Broadcasts averaged values back to merged head positions
4. Returns the modified output

This correctly simulates merging without changing tensor shapes or requiring monkey-patching.

**Serialization format:**
```json
{"kind": "kv_cache_compression", "target_layers": [], "merge_ratio": 0.5, "similarity_metric": "cosine", "allow_mha_to_gqa": false}
```

**Constraint:** Only applies to GQA models (`num_kv_heads < num_attention_heads`). Validation deferred to apply-time (not `__post_init__`) because the spec does not know the model's head configuration at construction. Rejects MHA models unless `allow_mha_to_gqa=True`.

### 4D. 2:4 Structured Sparsity (StructuredSparsityTransform)

**Spec:**

```python
@dataclass
class StructuredSparsitySpec:
    kind: Literal["structured_sparsity"] = "structured_sparsity"
    target_layers: list[int]     # empty = apply to all layers
    pattern: Literal["2:4"] = "2:4"
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
```

**No importance analysis needed.** Deterministic: for each group of 4 contiguous weights along the output dimension, zero the 2 with smallest absolute value.

**Apply logic (mask mode):** Store original weights, apply 2:4 mask to `weight.data` in-place. Register cleanup to restore original weights.

**Memory strategy for large models:** Original weights are stored on CPU (not GPU) to avoid doubling GPU memory. For a 70B model with 7 target modules per layer, this is ~140GB of CPU RAM — acceptable for a research workstation. Cleanup copies weights back from CPU to GPU. For smaller models where GPU memory is not a constraint, weights stay on GPU for faster cleanup.

**Hardware acceleration (opt-in, not automatic):** `to_sparse_semi_structured()` is available as an explicit opt-in flag (`use_semi_structured: bool = False`) rather than auto-detected by hardware capability. The API is unstable across PyTorch versions and should not be the default. The primary v1 path is dense masked weights. When `use_semi_structured=True`, the transform checks `torch.cuda.get_device_capability() >= (8, 0)` and falls back to dense masking if unsupported.

**Serialization format:**
```json
{"kind": "structured_sparsity", "target_layers": [0, 1], "pattern": "2:4", "target_modules": ["q_proj", "gate_proj"]}
```

---

## 5. Search Integration

### New entry point

```python
def generate_optimization_plans(
    weak_layer_report: WeakLayerReport,
    head_importance_report: HeadImportanceReport | None = None,
    channel_importance_report: ChannelImportanceReport | None = None,
    kv_similarity_report: KvSimilarityReport | None = None,
    target_speedup: float = 1.3,
    accuracy_budget: float = 0.05,
    enable_sparsity: bool = False,
    mode: Literal["mask", "structural"] = "mask",
) -> list[CandidatePlan]:
```

Takes whatever reports are available. If only weak-layer report provided, generates skip-only plans. Adding more reports enables more transform types.

### Search strategy: budget-aware greedy composition

**Step 1 — Rank all operations by efficiency** (latency gain / accuracy cost):

| Operation | Latency estimate | Accuracy cost estimate |
|-----------|-----------------|----------------------|
| Skip layer L | `1 / num_layers` | `1 - aggregate_score` from weak layer report |
| Prune p% heads in layer L | `p × attn_fraction / num_layers` | Sum of pruned head importance scores |
| Prune p% MLP in layer L | `p × mlp_fraction / num_layers` | Sum of pruned channel importance scores |
| Apply 2:4 sparsity to layer L | `~0.5 / num_layers` | Fixed empirical ~0.01 per layer |
| Merge KV heads in layer L | Decode latency ∝ merge ratio | `1 - min_similarity` of merged pairs |

`attn_fraction` ≈ 0.33, `mlp_fraction` ≈ 0.67 for SwiGLU models (from config).

**Step 2 — Greedy plan building:** Sort operations by efficiency, greedily add until `target_speedup` reached or `accuracy_budget` exhausted.

**Step 3 — Generate plan variants:**

1. **Conservative** — top 50% of greedy operations
2. **Recommended** — full greedy result
3. **Aggressive** — push to full accuracy budget
4. **Skip-only baseline** — layer skipping alone
5. **Per-technique baselines** — each technique in isolation

Each variant is a `CandidatePlan` with ordered `TransformationPlan` and metadata.

### Importance-to-spec mapping

The search layer must convert ranked importance scores into concrete spec objects:

- **Head importance → `HeadPruningSpec`**: For each target layer, sort heads by importance score, select the bottom N heads (where N is derived from the pruning budget), construct `LayerHeadSelection(layer=L, heads=[h1, h2, ...])`. Group all layer selections into a single `HeadPruningSpec`.
- **Channel importance → `MlpPruningSpec`**: The pruning ratio is computed per-layer based on the budget allocation. Channels below the threshold are implicitly pruned by the transform at apply time (the transform reads the `ChannelImportanceReport` to determine which channels to zero).
- **KV similarity → `KvCacheCompressionSpec`**: The merge ratio is set globally. The transform reads the `KvSimilarityReport` at apply time to determine which heads to merge per layer.

### Report serialization

All three new report types (`HeadImportanceReport`, `ChannelImportanceReport`, `KvSimilarityReport`) implement `to_dict()` / `from_dict()` for persistence. Analysis results are computed once and reused across the candidate evaluation loop — storing them avoids redundant forward passes.

### Composition ordering enforcement

The plan builder enforces: skip → head → MLP → KV → sparsity → compensation.

### Backward compatibility

`generate_candidate_plans()` stays unchanged (skip-only). `generate_optimization_plans()` is the new multi-transform entry point that internally calls `generate_candidate_plans()` for the skip-only baseline.

---

## 6. GPU-First Experimentation, QAIC Verification

### Primary workflow

```
Analyze → Search → Apply (mask) → Evaluate → Iterate → Best plan found
                                                              │
                                                              ▼
                                                    Export (structural) → QAIC verify
```

All experiments run on GPU with mask mode. QAIC is verification only.

### v1 scope (implement now)

- All 4 transforms: mask mode only
- Everything reversible via `run_model_cleanup()`
- `NASSession` loop: load once, try many plans, evaluate each
- No config mutation, no structural weight slicing

### v2 scope (implement later)

- `export_optimized_model()` — structural mode for QAIC export
- Config mutation (num_attention_heads, intermediate_size, num_key_value_heads)
- Uniform pruning constraint enforcement for structural mode
- `save_pretrained()` with updated config → QEffRuntime compilation

### GPU experimentation loop

```python
with NASSession() as session:
    artifact = session.load(model_spec)

    # Run analyses (once)
    weak_report = compute_weak_layer_report(artifact, datasets)
    head_report = compute_head_importance(artifact, datasets)
    channel_report = compute_channel_importance(artifact, datasets)
    kv_report = compute_kv_head_similarity(artifact)

    # Generate candidates
    candidates = generate_optimization_plans(
        weak_report, head_report, channel_report, kv_report,
        accuracy_budget=0.05,
    )

    # Try each candidate
    results = []
    for plan in candidates:
        session.apply_plan(artifact, plan.plan)
        score = session.evaluate(artifact, hf_runtime)
        results.append((plan, score))

    best_plan = pick_best(results)
```

---

## 7. Modern Datasets

### New dataset tier

Added alongside existing datasets. Both tiers available via `load_dataset_samples()`.

**Hard Reasoning:**

| Name | HF Source | Field | Notes |
|------|-----------|-------|-------|
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` test | `question` + `options` | 10 choices, chain-of-thought |
| `bbh_causal` | `lukaemon/bbh` `causal_judgement` test | `input` | Multi-step causal reasoning |
| `bbh_logical_deduction` | `lukaemon/bbh` `logical_deduction_five_objects` test | `input` | Logical constraint satisfaction |

**Instruction Following:**

| Name | HF Source | Field | Notes |
|------|-----------|-------|-------|
| `ifeval` | `HuggingFaceH4/ifeval` train | `prompt` | Verifiable instruction constraints |
| `helpsteer2` | `nvidia/HelpSteer2` train | `prompt` | Diverse real-world instructions |

**Math Reasoning:**

| Name | HF Source | Field | Notes |
|------|-----------|-------|-------|
| `gsm_hard` | `reasoning-machines/gsm-hard` train | `input` | GSM8K with harder numbers |
| `orca_math` | `microsoft/orca-math-word-problems-200k` train | `question` | 200k diverse math problems |

**Code Generation:**

| Name | HF Source | Field | Notes |
|------|-----------|-------|-------|
| `humanevalpack` | `bigcode/humanevalpack` python test | `prompt` | Extended HumanEval |
| `metamathqa` | `meta-math/MetaMathQA` train | `query` | Math + diverse rephrasing |

### Default analysis datasets

```python
DEFAULT_ANALYSIS_DATASETS = [
    "mmlu_pro", "bbh_causal", "bbh_logical_deduction",
    "ifeval", "gsm_hard", "humanevalpack", "orca_math",
]
```

Seven datasets, all four categories. Used as default for `compute_weak_layer_report` and importance analyses when no explicit dataset list is provided.

### Dataset tier constants

```python
LEGACY_DATASETS = {
    "gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande",
    "arc_challenge", "arc_easy", "openbookqa", "piqa",
    "mmlu", "boolq", "truthfulqa", "lambada",
}

MODERN_DATASETS = {
    "mmlu_pro", "bbh_causal", "bbh_logical_deduction",
    "ifeval", "helpsteer2", "gsm_hard", "orca_math",
    "humanevalpack", "metamathqa",
}

SUPPORTED_DATASETS = {**_legacy_loaders, **_modern_loaders}
```

---

## 8. Testing Strategy

### Layer 1: Unit tests (synthetic models, no GPU)

Per-transform tests with toy models and known weights:

- **Head Pruning:** 4-head model, prune head 1, verify its output is zeroed, others unchanged. Cleanup restores.
- **MLP Pruning:** Known intermediate activations, prune 50%, verify output changes. Cleanup restores.
- **KV Compression:** 4 KV heads, heads 0+1 identical, merge, verify averaged output. Cleanup restores.
- **2:4 Sparsity:** Known weight tensor, verify exactly 2-of-4 zeroed (smallest magnitude). Cleanup restores bitwise.

Analysis module tests:
- Head importance ranks known-weak heads lowest
- Channel importance ranks inactive channels lowest
- KV similarity correctly identifies identical vs. orthogonal heads

Search tests:
- `generate_optimization_plans` returns sorted candidates
- Conservative ⊂ Recommended ⊂ Aggressive
- Composition ordering enforced
- Accuracy budget respected

LayerAnatomy tests:
- Convention probing works for llama/qwen3/gemma3-style toys
- Shape-based fallback works when names differ
- Clear error when nothing matches

### Layer 2: Integration tests (small real model, GPU)

Use `Qwen/Qwen3-0.6B` or `meta-llama/Llama-3.2-1B`:
- Apply each transform, run `model.generate()`, verify coherent output
- Apply combined plan (skip + head + MLP + sparsity), verify coherent text
- Cleanup fully reverses all changes
- Analysis → Search → Apply round-trip

### Layer 3: End-to-end verification (Qwen3-4B)

Extend `scripts/verify_qwen3_4b.py`:
1. Run all 4 analyses
2. Generate optimization plans with `accuracy_budget=0.05`
3. Apply recommended plan, evaluate on all datasets
4. Compare to baseline — verify < 5% degradation
5. Summary table with per-technique contribution breakdown

### Test file organization

```
tests/
├── test_nas_head_pruning.py
├── test_nas_mlp_pruning.py
├── test_nas_kv_compression.py
├── test_nas_structured_sparsity.py
├── test_nas_layer_anatomy.py
├── test_nas_analysis_importance.py
├── test_nas_optimization_search.py
├── test_nas_combined_transforms.py
scripts/
├── verify_qwen3_4b.py               (existing, updated)
└── verify_optimization_pipeline.py   (new)
```

---

## 9. File Map — New and Modified Files

### New files

```
nas/transforms/head_pruning.py          HeadPruningTransform
nas/transforms/mlp_pruning.py           MlpPruningTransform
nas/transforms/kv_compression.py        KvCacheCompressionTransform
nas/transforms/structured_sparsity.py   StructuredSparsityTransform
nas/transforms/anatomy.py               LayerAnatomy, resolve_layer_anatomy()
nas/analysis/head_importance.py         compute_head_importance(), HeadImportanceReport
nas/analysis/channel_importance.py      compute_channel_importance(), ChannelImportanceReport
nas/analysis/kv_similarity.py           compute_kv_head_similarity(), KvSimilarityReport
nas/search/optimization.py              generate_optimization_plans()
```

### Modified files

```
nas/config/transforms.py               Add MlpPruningSpec, KvCacheCompressionSpec, StructuredSparsitySpec
                                        Update TransformSpec union, serialization helpers
nas/config/__init__.py                  Export new spec types
nas/transforms/applier.py              Register new transforms in default_transform_registry()
nas/transforms/__init__.py             Export new transforms
nas/analysis/datasets.py               Add 9 modern dataset loaders, MODERN_DATASETS, DEFAULT_ANALYSIS_DATASETS
nas/analysis/__init__.py               Export new analysis functions and report types
nas/search/__init__.py                 Export generate_optimization_plans
```
