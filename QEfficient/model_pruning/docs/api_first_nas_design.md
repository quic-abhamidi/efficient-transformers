# API-First NAS Design Review

## Objective

Move this repository from a script-first, CLI-heavy workflow with model-behavior changes partly encoded in vendored `transformers` / QEff paths to an API-first design that:

- keeps model loading on standard Hugging Face APIs,
- applies NAS modifications as Python-level model transforms,
- supports both local Hugging Face/GPU and QEff/QAIC runtimes,
- scales cleanly beyond skip layers to other inference-only techniques,
- preserves the current pipeline's useful analysis/search/reporting logic.

A core dependency goal is:

- NAS should be usable as an independent package on top of installed `transformers` and installed `QEfficient`,
- NAS should not require patched local clones of `transformers/` or `efficient-transformers/` for normal use,
- model behavior changes should live in NAS transform code or in upstream/library extension points, not in copied modeling files.

This document reviews the current codebase first, then refines the recommended design based on what already exists.

For the execution roadmap derived from this design, see [api_first_nas_implementation_plan.md](./api_first_nas_implementation_plan.md).

## Current Code Review

### What the code already does well

The current repository already has a useful split between stages:

- `analysis/measure_layer_contributions.py` provides reusable functions for prompt loading, model loading, forward passes, and contribution measurement.
- `optimization/layer_skipping/generate_config.py` is already closer to a library than a pure script. It exposes importable functions such as `load_contribution_data` and `identify_low_impact_layers`.
- `benchmarking/run_benchmark.py` already supports a strong API direction: `run_lm_eval(model, tokenizer, ...)` accepts model objects directly rather than only model IDs.
- `run_pipeline.py` mostly orchestrates existing functions rather than shelling out everywhere.
- `core/pipeline_checkpoint.py` is a good example of stateful orchestration separated from the algorithmic code.

This means the repo does not need a rewrite. It needs a cleaner set of API boundaries and a better way to represent model modifications.

### Where the current design is weak

#### 1. Model modification is not a first-class concept

There is no single abstraction representing "a model transformation plan".

Today, behavior is encoded through a mix of:

- CLI arguments such as `--skip-layers`,
- config mutation before `from_pretrained`,
- wrapper classes with forward hooks,
- compensation strategy dispatch and argument wiring inside `benchmarking/run_benchmark.py`,
- strategy implementations in `core/advanced_compensation.py` and `core/learnable_compensation.py`,
- QAIC logic in archived scripts.

As a result, the same conceptual change is represented differently in different parts of the code.

#### 2. Skip-layer support is coupled to model-loading behavior

`core/model_wrapper.py` uses `SkipLayerModelLoader.load_model_with_skip_layers(...)`, which:

- loads config,
- mutates `config.skip_layers`,
- calls `AutoModelForCausalLM.from_pretrained(..., config=config)`.

This approach only works if the loaded model implementation already knows how to interpret `skip_layers` in config. That creates a hidden dependency on modified model code in vendored `transformers` or other custom runtime assumptions.

In the current repo this coupling is real: the nested `transformers/` checkout is patched in place so model loops in Llama, Mistral, Qwen2, Qwen3, and Gemma3 read `config.skip_layers` directly.

This is exactly the coupling you want to remove.

#### 3. Compensation and skipping logic are mixed into runtime loading

`benchmarking/run_benchmark.py` currently acts as both:

- public benchmark API,
- model loader,
- feature-selection router,
- compensation-strategy dispatcher,
- legacy compatibility layer.

The actual advanced strategy factory already exists in `core/advanced_compensation.py` via `create_compensation_strategy(...)`, with wrapper support in `AdvancedCompensatedSkipLayerModel`. The problem is not missing abstraction there; the problem is that `run_benchmark.py` contains too much selection, validation, and wiring logic. This will become a scaling problem once more techniques are added.

#### 4. Hook-based wrappers are useful but too low-level to be the main external API

`core/compensated_skip_model.py` implements layer skipping via forward hooks. That is a reasonable implementation technique, but not a sufficient public design.

Problems with relying on wrappers/hooks as the main abstraction:

- behavior is runtime-only and implicit,
- serialization/reproducibility becomes weaker,
- composition across multiple transforms becomes unclear,
- QEff compatibility becomes harder to reason about,
- family-specific differences are hidden until runtime failure.

The right place for hooks is inside a transform implementation, not as the top-level user-facing interface.

#### 5. QAIC/QEff path is not integrated into the same core abstraction

The active repo no longer has a top-level `performance_benchmarking/` package; the QAIC logic currently lives mostly in `archive/` and the sibling `efficient-transformers/` checkout.

That means the repo currently has:

- one path for Hugging Face/GPU experimentation,
- another path for QAIC experimentation,
- no unified representation of a transformed model candidate that both runtimes can consume.

This is a design gap, not just an implementation gap.

#### 6. Current orchestration is stage-oriented, not object-oriented

The pipeline is organized around files and directories:

- contribution CSVs,
- generated config JSON,
- benchmark result JSON,
- reports.

That is fine for experiments, but it should not be the primary programmatic API. The programmatic API should work in-memory and optionally persist artifacts.

### Summary of current-state conclusions

The repo already contains reusable building blocks. The main missing piece is a stable object model for:

- what model to load,
- what transforms to apply,
- what runtime to evaluate on,
- what artifacts to save.

The migration should preserve the good parts of the current repo and replace the weak representation of model changes.

## Review of the Proposed Approach

The proposed approach is:

1. Load a model normally with `AutoModelForCausalLM`.
2. Apply model-level changes to the returned model object.
3. Pass the transformed object to the benchmark/runtime path.
4. For QEff, pass the object into the QEff instantiation/compile path rather than modifying QEff or vendored `transformers`.

This is directionally correct and should be the foundation of the design.

### Why this is the right direction

- It removes the need for special config semantics in vendored model code.
- It makes transformations explicit Python functionality owned by this repo.
- It lets analysis and search produce transform plans independent of runtime.
- It makes future features easier to add without touching model-loading code.
- It better matches the transform-based pattern you referenced from `efficient-transformers`.

### Where this approach needs refinement

The phrase "apply changes to the model object" is not by itself enough of a design. It needs three additions.

#### 1. Transform specs must be explicit

If transforms are represented only as direct object mutation, the system will not scale. The repo needs a typed, serializable transform-spec layer.

#### 2. Runtime compatibility must be explicit

Not every object-level transform will be acceptable to QEff/QAIC. The design must treat runtime compatibility as a first-class concern.

#### 3. Family-specific adapters are unavoidable

Skip layers, head pruning, and attention replacement all depend on model internals. The design must account for model-family handlers rather than assuming a single generic traversal works everywhere.

So the refined recommendation is:

- yes to post-load transforms,
- no to raw ad hoc mutation as the only representation,
- yes to a transform registry plus runtime adapters.

## Recommended Architecture

## Design principles

Use these principles consistently:

1. Keep model loading standard and boring.
2. Represent modifications as explicit transform specs.
3. Apply transforms through a registry of implementations.
4. Keep runtime concerns separate from transform concerns.
5. Keep analysis/search separate from execution.
6. Persist manifests for reproducibility.
7. Keep CLI as a thin wrapper over the same Python APIs.
8. Treat `transformers` and `QEfficient` as external dependencies, not source trees that NAS patches in-place.
9. Keep the core architecture inference-only; training and fine-tuning flows are out of scope for this design.

## Core object model

### 1. `ModelSpec`

Represents how to load the base model.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    model_id: str
    revision: Optional[str] = None
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
```

This replaces passing many unrelated loader arguments around the codebase.

### 2. `EvalSpec`

Represents evaluation/runtime arguments that should not leak back into ad hoc `**kwargs`.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalSpec:
    tasks: list[str]
    batch_size: int = 16
    device: str = "cuda"
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    use_cache: bool = False
    cache_dir: Optional[str] = None
    dtype: str = "auto"
    log_samples: bool = False
    random_seed: int = 42
    verbosity: str = "INFO"
```

`ModelSpec` should stay focused on loading. `EvalSpec` should own the current `run_lm_eval(...)` option set.

### 3. `QEffCompileSpec`

Represents QAIC/QEff compile and execution settings.

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QEffCompileSpec:
    ctx_len: int = 4096
    prefill_seq_len: int = 32
    batch_size: int = 1
    num_cores: int = 16
    continuous_batching: bool = False
    device_group: Optional[list[int]] = None
    qaic_config: dict = field(default_factory=dict)
```

This keeps QAIC runtime settings explicit and prevents compile/export kwargs from leaking into generic session calls.

### 4. `TransformSpec`

Use typed per-kind transform specs rather than a generic `kind + params: Dict[str, Any]`.

For v1, use dataclass-based discriminated unions with explicit `from_dict()` / `to_dict()` loaders. Avoid raw unvalidated `params` in manifests.

```python
from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class SkipLayersSpec:
    kind: Literal["skip_layers"] = "skip_layers"
    layers: list[int] = field(default_factory=list)


@dataclass
class RemoveLayersSpec:
    kind: Literal["remove_layers"] = "remove_layers"
    layers: list[int] = field(default_factory=list)


@dataclass
class ScaledCompensationConfig:
    strategy: Literal["scaled"] = "scaled"
    mean_delta_path: str = ""
    alpha: float = 1.0


@dataclass
class PhaseAwareCompensationConfig:
    strategy: Literal["phase_aware"] = "phase_aware"
    prefill_delta_path: str = ""
    decode_delta_path: str = ""
    prefill_alpha: float = 1.0
    decode_alpha: float = 1.0


CompensationConfig = Union[
    ScaledCompensationConfig,
    PhaseAwareCompensationConfig,
    # ... one typed config per supported strategy
]


@dataclass
class CompensationSpec:
    kind: Literal["compensation"] = "compensation"
    config: CompensationConfig = field(default_factory=ScaledCompensationConfig)


@dataclass
class LayerHeadSelection:
    layer: int
    heads: list[int] = field(default_factory=list)


@dataclass
class HeadPruningSpec:
    kind: Literal["head_pruning"] = "head_pruning"
    selections: list[LayerHeadSelection] = field(default_factory=list)
    mode: Literal["mask"] = "mask"


@dataclass
class LinearAttentionSpec:
    kind: Literal["linear_attention"] = "linear_attention"
    implementation: str = ""
    target_layers: list[int] = field(default_factory=list)
    apply_to_all: bool = False


TransformSpec = Union[
    SkipLayersSpec,
    RemoveLayersSpec,
    CompensationSpec,
    HeadPruningSpec,
    LinearAttentionSpec,
]
```

Examples:

- `SkipLayersSpec(layers=[6, 17])`
- `HeadPruningSpec(selections=[LayerHeadSelection(layer=12, heads=[1, 3, 7])])`
- `LinearAttentionSpec(implementation="kernel_x", apply_to_all=True)`

### 5. `TransformationPlan`

Represents an ordered set of transforms.

```python
from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class TransformationPlan:
    transforms: List[TransformSpec] = field(default_factory=list)
    compatibility_mode: Literal["strict", "best_effort"] = "strict"
```

This becomes the canonical representation of a candidate.

Compatibility mode behavior:

- `strict`: fail plan application if any transform is unsupported for the loaded model or requested runtime.
- `best_effort`: apply only supported transforms, emit warnings, and record skipped/degraded transforms in the manifest.

### 6. `ModelArtifact`

Represents a loaded and transformed model plus metadata.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class AppliedTransformRecord:
    kind: str
    status: Literal["applied", "skipped", "degraded", "failed"]
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelArtifact:
    artifact_id: str
    model: Any
    tokenizer: Any
    model_spec: ModelSpec
    plan: TransformationPlan
    applied_transforms: List[AppliedTransformRecord] = field(default_factory=list)
    capability_report: Dict[str, Any] = field(default_factory=dict)
```

This is what evaluation APIs should consume.

Mutation semantics:

- `ModelArtifact` owns one mutable model instance.
- `session.apply_plan(artifact, plan)` mutates that model instance in place and returns the same logical artifact handle for reassignment.
- Deep-copying large models is out of scope for v1.
- If the caller wants to branch multiple candidates from the same base weights, the supported mechanism is to reload/materialize a fresh artifact from `ModelSpec`, not to clone the in-memory module.

## Package layout

Add a new API-oriented package without disrupting existing directories immediately.

```text
nas/
  api/
    __init__.py
    session.py
    loaders.py
    benchmark.py
  specs/
    __init__.py
    models.py
    transforms.py
  transforms/
    __init__.py
    base.py
    registry.py
    skip_layers.py
    head_pruning.py
    linear_attention.py
    adapters/
      __init__.py
      llama.py
      qwen.py
      gemma.py
      mistral.py
  runtimes/
    __init__.py
    hf.py
    qeff.py
    capabilities.py
  analysis/
    __init__.py
    weak_layers.py
  search/
    __init__.py
    candidate_generator.py
  serialization/
    __init__.py
    manifest.py
```

This `nas/` package lives at repo root as a new namespace package. The existing top-level `analysis/`, `benchmarking/`, `optimization/`, and `core/` directories remain in place during Phases 1-3 and are reused behind the new API. The intent is coexistence during migration, not immediate replacement.

This package can reuse functions from existing `analysis/`, `optimization/`, and `benchmarking/` modules during migration.

## Transform system

### Base interface

```python
from typing import Protocol, Any, Dict


class ModelTransform(Protocol):
    kind: str

    def supports(self, model: Any) -> bool:
        ...

    def inspect(self, model: Any, spec: TransformSpec) -> Dict[str, Any]:
        ...

    def apply(self, model: Any, spec: TransformSpec) -> Any:
        ...
```

Each feature owns its own implementation.

### Registry

Use a registry to map `kind -> transform implementation`.

```python
TRANSFORM_REGISTRY = {
    "skip_layers": SkipLayersTransform(),
    "head_pruning": HeadPruningTransform(),
    "linear_attention": LinearAttentionTransform(),
}
```

This avoids hard-coding every feature inside `run_benchmark.py`.

For v1, keep the registry static plus an explicit `register_transform(kind, impl)` helper. Do not add setuptools entry points or a plugin discovery layer until the core API stabilizes.

### Model-family adapters

Transforms should delegate model-structure operations to family adapters.

For example, `SkipLayersTransform` should not directly assume `model.model.layers` for every model. Instead:

```python
adapter = resolve_family_adapter(model)
blocks = adapter.get_decoder_blocks(model)
adapter.replace_block(model, layer_idx, replacement_block)
```

This is necessary because:

- Llama/Qwen/Gemma/Mistral are similar but not identical,
- output signatures differ,
- attention internals differ more than block layout suggests.

### Adapter detection and support matrix

Use this detection order:

1. unwrap known wrappers (`PeftModel`, accelerator wrappers, thin runtime wrappers),
2. inspect `model.config.model_type` as the primary key,
3. fall back to architecture/class name only when `model_type` is missing.

Initial support matrix for skip-layer migration:

- supported in v1: `llama`, `mistral`, `qwen2`, `qwen3`, `gemma3`
- unsupported until explicitly tested: newer families such as `llama4`, older unrelated decoder stacks, and active training-time PEFT wrappers

PEFT/LoRA models are only in scope when they can be cleanly unwrapped to a supported base model for inference. Training-mode wrapper stacks are out of scope.

### Device placement contract

Adapters must preserve placement and dtype:

- transforms run after the base model is loaded,
- replacement modules must be created on the same device and dtype as the block they replace,
- `device_map="auto"` is supported only if adapter replacement preserves per-block placement before the first forward pass or export.

## Runtime system

## Why runtimes need to be separate

The same transformed model candidate may be:

- valid for Hugging Face inference,
- not valid for QEff compile,
- valid for QEff only in a reduced mode,
- valid only after export to a different prepared representation.

So runtime support cannot be implicit.

### Core runtime API

Runtimes should be first-class objects. The core session API should be:

```python
result = session.evaluate(artifact, runtime)
```

with convenience wrappers such as `session.evaluate_hf(...)` and `session.evaluate_qeff(...)` treated only as optional sugar over runtime instances.

### `HuggingFaceRuntime`

Responsibilities:

- accept `ModelArtifact`,
- run `lm_eval` or direct inference,
- save benchmark/report artifacts if requested.

This runtime can initially delegate to the existing `benchmarking/run_benchmark.py` functions.

The target end state is that `HuggingFaceRuntime` works with stock installed `transformers` and no local patched model source tree.

### `QEffRuntime`

Responsibilities:

- validate whether the transform plan is QEff-compatible,
- prepare the transformed model for QEff,
- compile and run,
- return structured metrics.

`QEffRuntime` should target the installed `QEfficient` Python package API. The NAS repo should not depend on editing a local `efficient-transformers/` checkout during normal use.

This runtime should support two modes.

#### Mode A: direct object mode

Use this when QEff can consume an instantiated transformed model object safely.

#### Mode B: prepared/export mode

Use this when QEff needs a controlled representation rather than an arbitrary in-memory object.

Selection should be explicit and logged:

- `prepare_mode="auto"`: runtime chooses and records the chosen path,
- `prepare_mode="object"`: require direct-object mode or fail,
- `prepare_mode="export"`: force prepared/export mode.

The chosen mode should be included in the evaluation result and manifest.

### Capability reporting

Every transform and runtime combination should support inspection.

Example capability report:

```python
{
    "hf_inference": True,
    "qeff_compile": False,
    "reason": "linear_attention not yet implemented for qeff runtime"
}
```

This prevents hidden failures later in the pipeline.

## How the current code should map into the new design

### Keep and reuse

These pieces should remain and gradually move behind the new API.

- `analysis/measure_layer_contributions.py`
  Use its analysis logic as the basis of `nas.analysis` APIs.

- `optimization/layer_skipping/generate_config.py`
  Reuse its candidate-generation logic as search input for skip-layer transform specs.

- `benchmarking/run_benchmark.py`
  Keep `run_lm_eval(...)` and metric extraction logic, but move model-loading/feature-routing responsibilities out of it.

- `core/advanced_compensation.py`
  Keep the strategy implementations, the `create_compensation_strategy(...)` factory, and `AdvancedCompensatedSkipLayerModel` as migration inputs for a future compensation transform.

- `core/learnable_compensation.py`
  Keep the learnable compensation module and training/loading utilities. This is part of the compensation migration story, not throwaway code.

- `core/pipeline_checkpoint.py`
  Keep for long-running artifact-producing workflows.

### Gradually replace

- `core/model_wrapper.py`
  Replace with `skip_layers` transform implementation and a standard loader.

- skip-layer routing inside `benchmarking/run_benchmark.py`
  Replace with `load -> apply_plan -> evaluate`.

- QAIC logic scattered across archive scripts
  Replace with a proper runtime adapter that primarily calls `efficient-transformers/QEfficient` as a library. Archived scripts should be treated as validation/reference material, not the primary integration surface.

### Deprecate as the primary mechanism

- config-based `skip_layers` injection before model load,
- feature selection by benchmark CLI branching,
- direct hook wrappers as the only representation of transformed candidates.
- local patched `transformers/` model loops as a runtime dependency.

## Feature-by-feature design guidance

## 1. Skip layers

This should be the first transform migrated because it is the current primary feature and the least risky place to establish the new architecture.

### Recommended implementation

Use an explicit transform implementation that:

- resolves model-family adapter,
- identifies decoder blocks,
- replaces selected blocks with lightweight no-op blocks that preserve the original forward signature without executing the original block compute,
- records metadata in `applied_transforms`.

Avoid making `config.skip_layers` the core mechanism.

### Why not rely on hooks as the main implementation forever

Hooks are acceptable for experimentation, but a replacement-block or wrapper-block approach is easier to reason about for:

- inspection,
- serialization metadata,
- compatibility checks,
- possible runtime export.

Hooks can still be used internally where necessary.

### Recommended v1 behavior

- support a small set of model families well,
- fail clearly for unsupported families,
- define `skip_layers` semantics as no-op replacement / iteration bypass semantics that preserve hidden-state shape, cache contract, and module interface while avoiding the skipped block compute,
- avoid physical layer deletion in v1.

This distinction matters because the current codebase has two different behaviors:

- hook-based bypass in `core/compensated_skip_model.py` / `core/advanced_compensation.py`,
- loop-level skipping in the patched `transformers/` checkout,
- physical layer removal in QEff (`remove_layers`) for QAIC export.

The new design should model these as different concepts rather than overload one flag.

### Structural removal should be a separate transform/runtime capability

Use a separate concept for structural removal:

- `skip_layers`: bypass/pass-through semantics, mainly for HF parity and functional experiments.
- `remove_layers`: structural deletion / export-time graph change, primarily for QAIC memory/performance optimization.

QEff already exposes `remove_layers` in `QEFFAutoModelForCausalLM.from_pretrained(...)`. The new runtime should map to that capability explicitly instead of hiding it behind `skip_layers`.

## 1A. Compensation

Compensation is the largest existing feature family after skip layers and should be modeled explicitly in the architecture.

### Recommended representation

Treat compensation as its own transform kind that depends on a prior skip transform, rather than burying 11 strategies inside `skip_layers` params.

Examples:

- `SkipLayersSpec(layers=[19, 20, 21, 22])`
- `CompensationSpec(config=PhaseAwareCompensationConfig(...))`

This keeps the plan readable and lets the registry route compensation independently.

### Migration source of truth

The migration should reuse:

- `core.compensated_skip_model` for the simple mean-vector baseline,
- `core.advanced_compensation.create_compensation_strategy(...)` for named strategies,
- `core.advanced_compensation.AdvancedCompensatedSkipLayerModel` for current wrapper semantics,
- `core.learnable_compensation.LearnableCompensation` for learnable residual compensation.

### Migration rule

Move the current `run_benchmark.py` if/elif dispatch into:

- a compensation transform registry,
- validation helpers for required artifacts/files,
- explicit transform ordering rules such as `skip_layers -> compensation`.

## 2. Weak-layer analysis

Weak-layer detection should remain a pure analysis concern.

It should produce structured reports, not mutate models directly.

Recommended return type:

```python
@dataclass
class RankedLayer:
    layer: int
    aggregate_score: float
    rank: int
    per_dataset_scores: dict[str, float]


@dataclass
class WeakLayerReport:
    model_spec: ModelSpec
    datasets: list[str]
    ranked_layers: list[RankedLayer]
    metadata: dict[str, Any]
```

This keeps downstream consumers honest about whether they are using:

- aggregate rankings,
- per-dataset rankings,
- score thresholds,
- or frequency-based heuristics.

Example flow:

```python
weak_layers = analyze_weak_layers(model_spec, datasets=["gsm8k", "hellaswag"])
plan = TransformationPlan([
    SkipLayersSpec(layers=[item.layer for item in weak_layers.ranked_layers[:2]])
])
```

This keeps analysis reusable for future techniques.

## 2A. Search

`nas.search` should consume analysis outputs and constraints, then emit executable candidate plans.

Responsibilities:

- consume `WeakLayerReport` and search constraints,
- generate `TransformationPlan` candidates,
- attach search metadata such as heuristic score, estimated risk, and generation provenance,
- stay separate from runtime execution.

Recommended output type:

```python
@dataclass
class CandidatePlan:
    plan: TransformationPlan
    priority: float
    rationale: str
    metadata: dict[str, Any]
```

Search should plug into the session API indirectly:

- `search` creates `CandidatePlan` objects,
- `session` materializes/evaluates them one at a time,
- orchestration code decides the keep/discard loop.

## 3. Head pruning

Implement this as a transform with two modes.

### Mode A: mask-only

- safer,
- keeps parameter shapes stable,
- easier for early adoption,
- likely better for runtime compatibility.

### Mode B: structural prune

- more efficient,
- more invasive,
- should be added only after the transform/runtime interfaces are stable.

Start with mask-only.

## 4. Linear attention replacement

This feature is more invasive than skip layers or mask-based pruning.

### Recommendation

- do not make it part of the first migration phase,
- support only selected architectures first,
- require explicit capability checks,
- keep the transform implementation family-specific.

This is the clearest example of why the design must include adapters and runtime capability reporting.

## 5. Quantization

Quantization is adjacent to NAS but is not part of the initial migration scope.

Decision:

- do not treat quantization as a Phase 1 blocker,
- keep it separate from skip/compensation/head-pruning transforms,
- if added later, represent it as its own typed transform or runtime capability rather than mixing it into unrelated specs.

This keeps the first migration focused on the current structural NAS features while leaving a clean extension point for future quantization support.

## Public API design

Keep the user-facing API small.

```python
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.runtimes.hf import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.runtimes.qeff import QEffRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.eval import EvalSpec, QEffCompileSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec, TransformationPlan


with NASSession() as session:
    artifact = session.load(
        ModelSpec(model_id="meta-llama/Llama-3.2-1B-Instruct")
    )

    artifact = session.apply_plan(
        artifact,
        TransformationPlan([
            SkipLayersSpec(layers=[6, 17])
        ])
    )

    gpu_results = session.evaluate(
        artifact,
        HuggingFaceRuntime(
            EvalSpec(
                tasks=["gsm8k", "hellaswag"],
                limit=50,
            )
        ),
    )

    qaic_results = session.evaluate(
        artifact,
        QEffRuntime(
            QEffCompileSpec(ctx_len=4096, batch_size=32),
            prepare_mode="auto",
        ),
    )
```

This is enough for:

- internal library usage,
- notebooks,
- later service endpoints,
- thin CLI wrappers.

## Session lifecycle

`NASSession` should be a lightweight resource manager, not a hidden model cache.

Expected v1 behavior:

- one session may manage multiple artifacts logically, but should avoid retaining multiple live GPU-resident model copies by default,
- tokenizer caching is acceptable,
- model caching should be explicit and opt-in,
- session instances are not thread-safe in v1,
- `NASSession` and `ModelArtifact` should support explicit cleanup (`close()` / context manager) to release GPU memory deterministically.

## CLI role after migration

CLI should stay, but only as a facade.

Desired pattern:

- CLI parses user input,
- converts it into `ModelSpec` and `TransformationPlan`,
- calls the same session APIs used by Python callers,
- writes artifacts.

This preserves scripting convenience without making CLI the architecture.

Backward-compatibility policy:

- keep existing flags such as `--skip-layers` working through at least Phases 1-2,
- translate legacy flags into typed specs internally,
- add deprecation warnings only after the API and manifest formats stabilize,
- avoid breaking scripted usage during the migration window.

## Testing and parity strategy

The migration is not complete unless the new API reproduces current behavior within defined tolerances.

Required checks:

- spec serialization/deserialization tests for all supported transform kinds,
- manifest version/load tests,
- family-adapter unit tests on supported architectures,
- skip-layer parity tests comparing new transform behavior against the current execution path on fixed prompts,
- greedy decode parity tests,
- `lm_eval` smoke comparisons within documented tolerance,
- QEff compile smoke tests for supported plans.

Phase 2 should not be considered complete merely because the code \"runs through the same API\". It must also demonstrate parity or intentionally documented deltas.

## Manifest and reproducibility

Every evaluated candidate should persist a manifest.

Suggested JSON shape:

```json
{
  "schema_version": "nas.manifest/v1",
  "model_spec": {
    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    "dtype": "bfloat16",
    "device_map": "auto"
  },
  "plan": {
    "transforms": [
      {
        "kind": "skip_layers",
        "params": {
          "layers": [6, 17]
        }
      }
    ]
  },
  "applied_transforms": [
    {
      "kind": "skip_layers",
      "status": "applied",
      "details": {
        "layers": [6, 17],
        "model_family": "llama"
      }
    }
  ],
  "capabilities": {
    "hf_inference": true,
    "qeff_compile": true
  },
  "environment": {
    "nas_version": "0.1.0",
    "transformers_version": "x.y.z",
    "qefficient_version": "x.y.z",
    "lm_eval_version": "x.y.z"
  },
  "source_control": {
    "repo_git_sha": "<optional>",
    "repo_dirty": true
  }
}
```

This matters because transformed objects are ephemeral; manifests are the durable truth.

Manifest requirements:

- every manifest must include `schema_version`,
- manifest loading must go through a version-aware loader that can migrate older payloads,
- dependency versions and optional repo SHA should be captured for reproducibility.

### Where manifests should live

For artifact-producing runs, manifests should live under the same run directory as results, for example:

```text
results/<run_id>/
  plan.json
  manifest.json
  metrics/
  reports/
```

`artifacts/` should remain for curated outputs, executive summaries, or presentation-ready assets. It should not become the primary storage location for run-state manifests.

## Recommended migration plan

## Phase 1: Establish the API shell

Build the following without removing current scripts:

- `ModelSpec`, typed `TransformSpec` unions, `TransformationPlan`, `ModelArtifact`,
- `AppliedTransformRecord`,
- `EvalSpec`,
- `QEffCompileSpec`,
- `NASSession.load(...)`,
- `NASSession.evaluate(artifact, runtime)`,
- transform registry,
- `HuggingFaceRuntime`,
- `QEffRuntime`,
- versioned manifest serializer/loader.

At this stage, keep existing CLIs working.

Packaging goal for this phase:

- the new `nas/` package should be importable independently of the local `transformers/` and `efficient-transformers/` checkouts,
- local checkouts may remain temporarily for migration/testing, but they should not be required by the new API surface.

## Phase 2: Migrate skip layers to object-level transforms

Implement `skip_layers` as a proper transform.

Work items:

- create model-family adapter utilities,
- implement no-op skip-layer block replacement,
- update benchmark path to use `apply_plan` rather than `SkipLayerModelLoader`,
- keep temporary compatibility path if needed,
- add parity tests against the current execution path.

Success criterion:

- baseline and skip-layer evaluation run through the same API with no patched `transformers` `config.skip_layers` dependency required.
- the repo no longer depends on the nested `transformers/` checkout modifications for skip-layer execution, so those patches can be reverted or at minimum removed from the active execution path.
- stock installed `transformers` is sufficient for HF execution of NAS transforms.
- new skip-layer execution matches the current reference path within documented decode/logit and `lm_eval` tolerances.

## Phase 3: Move analysis/search onto specs

Update analysis/config generation so they produce transform specs or transformation plans instead of only filesystem configs.

Examples:

- current `skip_configurations.json` can still be emitted,
- but its internal schema should align with `TransformationPlan`.

Success criterion:

- search output is directly consumable by the new runtime APIs.
- weak-layer analysis produces typed reports, not ad hoc lists.

## Phase 4: Introduce QEff runtime adapter

Create a first-class `QEffRuntime` with capability inspection.

Work items:

- integrate `efficient-transformers/QEfficient` as a library rather than reviving archive scripts as the primary path,
- use direct object mode first via `QEFFAutoModelForCausalLM(model=..., ...)` where compatible,
- use export/prepare fallback only where direct object mode is not sufficient,
- hide backend differences behind one runtime interface.

Dependency goal:

- QEff support should work against an installed `QEfficient` package version with documented minimum compatibility,
- if NAS needs a missing extension point from QEfficient, that should be solved by upstreaming or version pinning, not by making a local clone mandatory.

Success criterion:

- the same plan evaluated on HF can be passed to QEff runtime without redefining the candidate format.

## Phase 5: Add second and third transform families

Recommended order:

1. `head_pruning` in mask mode,
2. weak-layer APIs feeding candidate generation,
3. `linear_attention` on limited model families.

This validates that the architecture scales beyond skip layers.

## What not to do

Avoid these design mistakes:

- Do not create one giant class that owns loading, transforms, search, benchmarking, and reporting.
- Do not keep adding feature branches into `benchmarking/run_benchmark.py`.
- Do not make config mutation the long-term mechanism for model changes.
- Do not assume QEff can always consume arbitrary transformed objects.
- Do not promise all transforms for all model families from the start.
- Do not collapse analysis and execution into one step.

## Final recommendation

The right design for this repo is an API-first transform system with:

- standard model loading,
- explicit transform specs,
- ordered transformation plans,
- model-family transform adapters,
- separate Hugging Face and QEff runtime adapters,
- manifest-based reproducibility,
- CLI kept only as a thin facade.

This is the simplest design that is still scalable.

It avoids overengineering, removes the current dependence on hidden model-config semantics, and gives the repo a clean path from today's skip-layer workflow to future features such as head pruning, linear attention replacement, and more advanced weak-layer interventions.

Most importantly, the intended end state is that NAS is a standalone package layered on top of released `transformers` and `QEfficient` APIs. Local source checkouts may still be useful while developing those upstream projects, but they should not be part of the normal NAS usage model.
