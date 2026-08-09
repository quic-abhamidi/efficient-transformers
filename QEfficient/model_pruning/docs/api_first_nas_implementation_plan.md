# API-First NAS Implementation Plan

## Purpose

This document converts the API-first NAS design into a phased implementation plan.

Primary goals:

- move NAS to an API-first architecture,
- remove the runtime dependency on patched local `transformers/` and `efficient-transformers/` checkouts,
- keep the current CLI interface intact until the new APIs and manifest formats stabilize,
- migrate incrementally with parity checks at each stage.

This plan is implementation-oriented. It should be read together with [api_first_nas_design.md](./api_first_nas_design.md).

## Non-negotiable migration rules

1. Keep existing CLI entry points working through all migration phases until the stabilization gate is passed.
2. Do not require users to keep patched local clones of `transformers` or `efficient-transformers` for normal NAS usage.
3. Do not break current benchmark/result directory expectations during migration unless a compatibility layer is in place.
4. Do not ship a phase as complete without parity validation against the current reference path.
5. Keep the scope inference-only. Training and fine-tuning are out of scope for this roadmap.

## CLI compatibility policy

The current CLI stays as the user-facing compatibility layer until the API stabilizes.

Rules:

- existing flags such as `--skip-layers`, `--use-compensation`, `--compensation-strategy`, `--limit`, and pipeline stage flags must continue to work,
- CLI modules should gradually become thin translators from CLI args to typed API specs,
- no flag removals before the stabilization gate,
- if an internal implementation changes, CLI output shape and artifact locations should remain compatible unless explicitly versioned,
- deprecation warnings are allowed only after the typed API and manifest loader are stable.

Stabilization gate:

- manifest schema version `v1` is frozen,
- typed transform specs are frozen for the supported v1 feature set,
- HF runtime and QEff runtime both work through the same session/runtime contract,
- parity tests pass for the supported features,
- at least one release cycle runs with the new APIs underneath the old CLI.

Only after that gate may the CLI be simplified or deprecated.

## Phase overview

1. Phase 0: Baseline capture and migration guardrails
2. Phase 1: Typed spec and runtime skeleton
3. Phase 2: Session and runtime integration without behavior change
4. Phase 3: Skip-layer transform migration on stock `transformers`
5. Phase 4: Compensation transform migration
6. Phase 5: Typed analysis and search outputs
7. Phase 6: QEff runtime integration on installed `QEfficient`
8. Phase 7: Additional transforms and API stabilization
9. Phase 8: CLI simplification after stabilization

## Phase 0: Baseline capture and migration guardrails

### Objective

Freeze the reference behavior before changing architecture.

### Deliverables

- reference prompts, datasets, and expected outputs for:
  - baseline HF execution,
  - skip-layer execution,
  - compensation execution for representative strategies,
  - QAIC/QEff compile smoke runs.
- compatibility matrix for currently used model families:
  - `llama`,
  - `mistral`,
  - `qwen2`,
  - `qwen3`,
  - `gemma3`.
- inventory of CLI entry points and flags currently used by this repo.
- documented artifact layout expectations under `results/`.

### Code touchpoints

- [run_pipeline.py](../run_pipeline.py)
- [run_benchmark.py](../benchmarking/run_benchmark.py)
- [model_wrapper.py](../core/model_wrapper.py)
- [compensated_skip_model.py](../core/compensated_skip_model.py)
- [advanced_compensation.py](../core/advanced_compensation.py)

### Validation

- capture current logits / greedy decode outputs for fixed prompts,
- capture current `lm_eval` smoke results on small limits,
- capture one QAIC reference compile run for a known-supported model.

### Exit criteria

- reference metrics and prompts are committed as migration fixtures,
- parity thresholds are documented,
- CLI compatibility inventory is complete.

## Phase 1: Typed spec and runtime skeleton

### Objective

Introduce the new package structure and typed contracts without changing behavior.

### Deliverables

- `nas/` namespace package scaffold:
  - `nas/api/`
  - `nas/specs/`
  - `nas/transforms/`
  - `nas/runtimes/`
  - `nas/analysis/`
  - `nas/search/`
  - `nas/serialization/`
- typed specs:
  - `ModelSpec`
  - `EvalSpec`
  - `QEffCompileSpec`
  - typed transform specs for:
    - `skip_layers`
    - `remove_layers`
    - `compensation`
    - `head_pruning`
    - `linear_attention`
- `TransformationPlan`
- `AppliedTransformRecord`
- `ModelArtifact`
- manifest serializer/loader with `schema_version`

### Code touchpoints

- new files under `nas/`
- no behavior changes to current CLI modules yet

### Implementation notes

- use dataclass-based discriminated unions for v1,
- provide `to_dict()` / `from_dict()` loaders for manifests,
- keep the transform registry static for v1,
- do not add plugin discovery in this phase.

### Validation

- unit tests for spec serialization/deserialization,
- manifest round-trip tests,
- version-aware manifest loading tests.

### Exit criteria

- typed API package is importable independently,
- manifest `v1` loader works,
- current CLI remains unchanged and green.

## Phase 2: Session and runtime integration without behavior change

### Objective

Add the session/runtime abstraction while still delegating to existing implementations.

### Deliverables

- `NASSession`
- first-class runtime objects:
  - `HuggingFaceRuntime`
  - `QEffRuntime`
- core evaluation API:
  - `session.load(model_spec)`
  - `session.apply_plan(artifact, plan)`
  - `session.evaluate(artifact, runtime)`
- optional wrapper methods:
  - `session.evaluate_hf(...)`
  - `session.evaluate_qeff(...)`
- explicit session lifecycle:
  - context manager support,
  - `close()` cleanup,
  - non-thread-safe v1 behavior documented.

### Code touchpoints

- new runtime and session files under `nas/`
- bridge logic into:
  - [run_benchmark.py](../benchmarking/run_benchmark.py)
  - existing tokenizer/model loaders

### Implementation notes

- do not replace `SkipLayerModelLoader` yet,
- do not replace compensation internals yet,
- use existing code behind the session API where possible.

### CLI policy in this phase

- CLI still calls existing code paths,
- optionally add hidden/internal API bridge calls,
- no visible behavior changes.

### Validation

- session API smoke tests for baseline HF runs,
- runtime wrapper tests that confirm delegation to current benchmark path,
- resource cleanup tests to catch GPU memory leaks.

### Exit criteria

- a baseline evaluation can run end-to-end through the new session/runtime API,
- current CLI remains intact,
- no regression in existing outputs.

## Phase 3: Skip-layer transform migration on stock `transformers`

### Objective

Replace config-based skip behavior with explicit model-object transforms that work on installed `transformers`.

### Deliverables

- model-family adapter layer
- supported adapter detection for:
  - `llama`
  - `mistral`
  - `qwen2`
  - `qwen3`
  - `gemma3`
- `SkipLayersTransform`
- `RemoveLayersSpec` recognized as a separate concept, but HF focus remains `skip_layers`
- no-op replacement blocks that:
  - preserve interface,
  - preserve hidden-state shape,
  - preserve cache contract,
  - avoid executing original skipped compute.

### Code touchpoints

- new files under `nas/transforms/` and `nas/transforms/adapters/`
- migration away from:
  - [model_wrapper.py](../core/model_wrapper.py)
  - patched `transformers` `config.skip_layers` dependency

### Implementation notes

- adapter detection order:
  1. unwrap known wrappers,
  2. inspect `model.config.model_type`,
  3. fall back to class name only if needed.
- preserve device placement and dtype for `device_map="auto"`,
- branching multiple candidate plans should require re-materializing from `ModelSpec`, not copying a mutated model object.

### CLI policy in this phase

- `--skip-layers` remains supported,
- CLI translates `--skip-layers` into `SkipLayersSpec`,
- no flag changes visible to users.

### Validation

- decode/logit parity tests against the current reference path,
- small `lm_eval` parity tests within documented tolerance,
- adapter coverage tests for supported model families,
- `device_map="auto"` smoke tests on representative models.

### Exit criteria

- stock installed `transformers` is sufficient for HF skip-layer execution,
- current local `transformers/` patches are no longer required by the active execution path,
- parity thresholds are met,
- CLI behavior remains stable.

## Phase 4: Compensation transform migration

### Objective

Move the current compensation stack into the typed transform system.

### Deliverables

- `CompensationSpec`
- typed config types for supported compensation strategies
- `CompensationTransform`
- explicit transform ordering rules:
  - `skip_layers -> compensation`
- migration of current dispatch logic out of [run_benchmark.py](../benchmarking/run_benchmark.py)

### Source of truth to reuse

- [compensated_skip_model.py](../core/compensated_skip_model.py)
- [advanced_compensation.py](../core/advanced_compensation.py)
- [learnable_compensation.py](../core/learnable_compensation.py)

### Strategy rollout

Recommended order:

1. simple mean-vector compensation
2. scaled
3. phase-aware
4. magnitude_rescaling
5. learnable
6. remaining advanced strategies

This avoids migrating all 11 strategies at once without proving the transform contract first.

### CLI policy in this phase

- keep existing flags:
  - `--use-compensation`
  - `--compensation-vector-file`
  - `--compensation-strategy`
  - related strategy-specific flags
- CLI translates them into `CompensationSpec` and typed strategy config.

### Validation

- per-strategy manifest round-trip tests,
- per-strategy required-file validation tests,
- output parity tests for representative strategies,
- ordering validation tests to ensure compensation is not applied without skip context.

### Exit criteria

- compensation no longer depends on large if/elif routing in `run_benchmark.py`,
- at least the priority strategy subset runs through the transform system,
- CLI compensation flags still behave compatibly.

## Phase 5: Typed analysis and search outputs

### Objective

Move analysis and candidate generation to typed outputs while preserving current filesystem artifacts.

### Deliverables

- `WeakLayerReport`
- `RankedLayer`
- `CandidatePlan`
- search API that consumes typed analysis reports and emits `TransformationPlan` candidates
- compatibility writers for existing outputs such as `skip_configurations.json`

### Code touchpoints

- [measure_layer_contributions.py](../analysis/measure_layer_contributions.py)
- [generate_config.py](../optimization/layer_skipping/generate_config.py)

### CLI policy in this phase

- existing analysis/config-generation CLIs remain intact,
- internally they may emit typed objects first and then write legacy JSON/CSV artifacts.

### Validation

- typed report generation tests,
- candidate generation tests,
- backward-compatible artifact writer tests.

### Exit criteria

- search output is directly consumable by `session.apply_plan()` / `session.evaluate(...)`,
- legacy output files are still produced where expected.

## Phase 6: QEff runtime integration on installed `QEfficient`

### Objective

Use installed `QEfficient` as the runtime backend for QAIC rather than local checked-out integration logic.

### Deliverables

- `QEffRuntime` backed by installed `QEfficient`
- prepare mode support:
  - `auto`
  - `object`
  - `export`
- capability inspection for transform/runtime compatibility
- explicit support for `remove_layers` as a QEff-facing structural capability

### Integration direction

- primary path: call installed `QEfficient` as a library,
- first choice: direct object mode where `QEFFAutoModelForCausalLM(model=...)` is sufficient,
- fallback: export/prepare mode when direct object mode is not enough,
- archived QAIC scripts are reference material only.

### CLI policy in this phase

- preserve current QAIC-facing CLI behavior where possible,
- translate existing QAIC/grid-search args into `QEffCompileSpec` and runtime options.

### Validation

- compile smoke tests on supported models,
- object-mode vs export-mode selection logging tests,
- compatibility-report tests,
- artifact and manifest recording tests.

### Exit criteria

- normal NAS QAIC use no longer depends on editing a local `efficient-transformers/` checkout,
- installed `QEfficient` with a documented minimum version is sufficient,
- chosen prepare mode is recorded in outputs/manifests.

## Phase 7: Additional transforms and API stabilization

### Objective

Add the next transform families and freeze the v1 API surface.

### Deliverables

- `HeadPruningSpec` and `HeadPruningTransform` in mask mode
- limited-family `LinearAttentionSpec` / transform
- frozen v1 typed spec set
- frozen manifest schema `nas.manifest/v1`
- finalized runtime API contract

### Scope notes

- quantization is explicitly out of initial migration scope,
- if added later, it should be its own typed transform or runtime capability.

### Validation

- spec freeze review,
- runtime compatibility review,
- additional transform smoke tests,
- manifest backward-compatibility tests.

### Exit criteria

- v1 API surface is stable,
- v1 manifest loader is stable,
- CLI compatibility bridge has run successfully for at least one release cycle.

## Phase 8: CLI simplification after stabilization

### Objective

Reduce architectural dependence on CLI once the API is proven stable.

### Allowed changes in this phase

- re-implement CLI commands purely as thin API wrappers,
- add deprecation warnings for obsolete flags,
- simplify duplicated legacy code paths,
- document preferred API-first usage patterns.

### Not allowed before this phase

- removing major flags users rely on,
- forcing users onto the API only,
- changing result layouts without a migration note.

### Validation

- backward-compatible CLI smoke suite,
- documentation update for API-first and CLI usage,
- deprecation messaging review.

### Exit criteria

- CLI is a thin facade over stable APIs,
- legacy internals are removed or quarantined,
- users can choose CLI or API without behavior mismatch.

## Recommended file creation order

1. `nas/specs/models.py`
2. `nas/specs/eval.py`
3. `nas/specs/transforms.py`
4. `nas/serialization/manifest.py`
5. `nas/api/session.py`
6. `nas/runtimes/base.py`
7. `nas/runtimes/hf.py`
8. `nas/runtimes/qeff.py`
9. `nas/transforms/base.py`
10. `nas/transforms/registry.py`
11. `nas/transforms/adapters/*`
12. `nas/transforms/skip_layers.py`
13. `nas/transforms/compensation.py`
14. `nas/analysis/weak_layers.py`
15. `nas/search/candidate_generator.py`

## Recommended implementation order

1. Finish Phase 0 fixtures and parity thresholds.
2. Build Phase 1 typed specs and manifest loader.
3. Add Phase 2 session/runtime shell with delegation only.
4. Migrate skip layers in Phase 3.
5. Migrate compensation in Phase 4.
6. Move typed analysis/search in Phase 5.
7. Integrate installed `QEfficient` in Phase 6.
8. Add head pruning and limited linear attention in Phase 7.
9. Simplify CLI only after Phase 7 stabilization.

## Minimum acceptance criteria before calling the migration successful

- baseline HF runs work on installed `transformers` only,
- skip-layer HF runs work on installed `transformers` only,
- QAIC runs work on installed `QEfficient` only for the supported feature set,
- CLI remains usable during migration,
- manifests are versioned and reproducible,
- parity checks are documented and passing,
- current local patched source trees are no longer required for normal NAS usage.
