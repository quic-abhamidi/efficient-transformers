# Analysis

`analysis/` now holds the maintained analysis and reporting utilities for layer-skipping studies. Experiment-specific orchestration scripts were moved to `archive/analysis_experiments/` so this package stays focused, importable, and easier to maintain.

## What stays here

- `measure_layer_contributions.py`: measures per-layer contribution across datasets.
- `measure_layer_range_delta.py`: measures cumulative hidden-state change across a chosen layer range.
- `analyze_embedding_delta_patterns.py`: analyzes prefill/decode delta structure, magnitude ratios, and related diagnostics.
- `analyze_config_performance.py`: compares benchmark/config trade-offs.
- `generate_compensation_report.py`: builds compensation comparison reports.
- `generate_skip_comp_report.py`: builds compact baseline vs skip vs compensation reports.
- `collect_and_fit_advanced.py`: collects hidden states and fits advanced compensation prototypes.

## What moved to archive

These files were experiment runners rather than reusable analysis primitives, so they now live under `archive/analysis_experiments/`:

- `run_compensation_experiment.py`
- `run_comprehensive_compensation_comparison.py`
- `run_mlp_small_32b_full_comparison.py`
- `run_multicluster_phase_mag_autosearch.py`
- `run_nontraining_advanced_benchmark.py`
- `run_per_dataset_compensation_experiment.py`
- `run_phase_mag_rescaling_experiment.py`
- `run_residual_compensation_experiment.py`
- `run_skip_search_experiment.py`
- `train_learnable_compensation.py`
- `package_multicluster_best.py`

Archived scripts are preserved for reproducibility, but they are not treated as maintained package APIs.

## Core workflows

### Per-layer contribution analysis

```bash
python -m analysis.measure_layer_contributions \
    --model <hf-model> \
    --dataset gsm8k hellaswag \
    --num-samples 100 \
    --metric both
```

Outputs per-dataset CSV/PNG files that show which layers have the smallest or largest effect on hidden states.

### Layer-range delta analysis

```bash
python -m analysis.measure_layer_range_delta \
    --model <hf-model> \
    --dataset wikitext \
    --start-layer 13 \
    --end-layer 17 \
    --num-samples 100
```

Use this when you want the combined effect of a skipped span rather than sequential layer-to-layer changes.

### Delta pattern diagnostic

```bash
python -m analysis.analyze_embedding_delta_patterns \
    --model <hf-model> \
    --start-layer 13 \
    --end-layer 17 \
    --datasets wikitext gsm8k hellaswag
```

This is the diagnostic used to study whether skipped layers mostly change direction, magnitude, or both.

### Config performance analysis

```bash
python -m analysis.analyze_config_performance \
    --analysis-dir <run_dir> \
    --accuracy-threshold 5.0
```

Use this to summarize trade-offs across generated configurations and benchmark results.

## Maintained import surface

`analysis/__init__.py` exports:

- `generate_layer_analysis`
- `generate_layer_range_analysis`
- `generate_delta_pattern_analysis`
- `ConfigPerformanceAnalyzer`

## Notes

- Use small sample sizes first for smoke testing.
- If you need the older end-to-end compensation experiments, run them directly from `archive/analysis_experiments/`.
- The archive README explains why those scripts were separated.
