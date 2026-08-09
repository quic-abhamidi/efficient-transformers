# QEfficient Model Pruning

License: BSD-3-Clause | Backend: PyTorch, Transformers, QEfficient, QAIC | Methods: decoder layer skip pruning | Evaluation: lm_eval, perplexity, Video-MME

QEfficient Model Pruning is a reproducible workflow for finding low-impact decoder layers, validating quality impact, and comparing baseline versus pruned execution on Qualcomm AI Cloud hardware. It is designed for teams that need to prune multiple QEfficient-supported models with repeatable commands and JSON artifacts.

This workflow currently focuses on training-free layer skipping. It does not permanently rewrite model checkpoints. Pruning plans are applied in memory during HF evaluation or QEff/QAIC benchmarking.

## Why Use This Workflow

- Task-aware plan selection: discover candidate skip-layer plans and keep only plans that pass a quality threshold.
- Multiple scoring metrics: cosine, L2, or combined cosine+L2 weak-layer analysis.
- HF quality gate: evaluate candidate plans with perplexity, lm_eval, or Video-MME before hardware benchmarking.
- QAIC comparison: compile and run baseline QEff/QAIC and optimized QEff/QAIC paths from fresh model loads.
- Manual QAIC path: if skip layers are already known, run QAIC baseline-vs-pruned comparison without HF evaluation.
- Structured artifacts: every stage writes JSON so results can be inspected, shared, and reproduced.

## Supported Models

Layer-skip pruning is supported for these model families:

- Llama-style decoder models
- Mistral-style decoder models
- Qwen2 / Qwen3 text models
- Qwen3 MoE text models
- Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE VLM text decoders
- Gemma3-style models

Common targets:

- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `google/gemma-4-26B-A4B-it`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`

## Updates

- Added `--metric both` for combined cosine and L2 weak-layer analysis.
- Added direct `qaic --skip-layers ...` for QAIC-only manual comparisons.
- QAIC benchmarking now loads fresh QEfficient models directly for baseline and optimized runs.
- QAIC comparison reports now include compile time, TTFT, decode tokens/sec, total tokens/sec, and E2E timing.
- Added Qwen-VL skip-layer support and Qwen-VL Video-MME preprocessing via `qwen_vl_utils` when available.

## Table Of Contents

- [Quick Start](#quick-start)
- [End-To-End Flow](#end-to-end-flow)
- [Step-By-Step Commands](#step-by-step-commands)
- [Run The Requested Models](#run-the-requested-models)
- [QAIC-Only Manual Skip Layers](#qaic-only-manual-skip-layers)
- [Video-MME For VLMs](#video-mme-for-vlms)
- [Outputs](#outputs)
- [Precision And Runtime Notes](#precision-and-runtime-notes)
- [Troubleshooting](#troubleshooting)
- [Validation](#validation)

## Quick Start

### Installation

Run from the `efficient-transformers` repository root:

```bash
cd /home/abhamidi/NAS_new/efficient-transformers
source image_bug/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -e ".[test,model_pruning]"
python -m QEfficient.cloud.model_pruning --help
```

### Minimal QAIC Example With Known Skip Layers

Use this when you already know which decoder layers to skip and want only the QAIC performance comparison.

```bash
MODEL="Qwen/Qwen3.5-27B"
OUT="results/model_pruning/manual_layers_qwen35_27b"

python -m QEfficient.cloud.model_pruning qaic \
  --model "$MODEL" \
  --skip-layers 16 5 15 13 \
  --device-group 0 \
  --batch-size 1 \
  --ctx-len 4096 \
  --prefill-seq-len 128 \
  --generation-len 60 \
  --num-cores 16 \
  --compile-dir-base "$OUT/qaic_compile" \
  --qaic-output-dir "$OUT/qaic" \
  --verbose

cat "$OUT/qaic/manual_best_plan.json"
cat "$OUT/qaic/benchmark_comparison.json"
```

## End-To-End Flow

The full pipeline has three stages:

1. Analyze: HF only
   - Loads the model with Transformers.
   - Runs calibration prompts.
   - Computes weak-layer scores using cosine, L2, or both.
   - Writes `weak_layer_report.json` and `candidate_plans.json`.
   - No QEff compile happens here.

2. Evaluate: HF only
   - Loads the HF model again.
   - Applies candidate pruning plans in memory.
   - Evaluates quality using `lm_eval`, perplexity, or Video-MME.
   - Writes `plan_results.json`, `comparison_report.json`, and `best_plan.json`.

3. QAIC: QEfficient from scratch
   - Reads `best_plan.json`, or builds a manual plan from `--skip-layers`.
   - Runs baseline with an empty plan.
   - Runs optimized with the selected skip-layer plan.
   - Each run loads a fresh QEfficient model, compiles it, runs generation, and parses performance metrics.
   - Writes `baseline.json`, optimized-plan JSON, `all_results.json`, and `benchmark_comparison.json`.

## Step-By-Step Commands

The commands below use a text model and `lm_eval`. Adjust `MODEL` and `OUT` for each run.

```bash
cd /home/abhamidi/NAS_new/efficient-transformers
source image_bug/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL="Qwen/Qwen3.5-27B"
OUT="results/model_pruning/qwen35_27b_lm_eval"
```

### 1. Analyze Weak Layers

```bash
python -m QEfficient.cloud.model_pruning analyze \
  --model "$MODEL" \
  --datasets gsm8k hellaswag \
  --num-samples 64 \
  --batch-size 1 \
  --metric both \
  --dtype float16 \
  --device auto \
  --verbose \
  --output-dir "$OUT"
```

Useful metric choices:

```bash
--metric cosine
--metric l2
--metric both
```

`--metric both` writes the combined report plus metric-specific reports:

```bash
ls "$OUT"
cat "$OUT/weak_layer_report.json"
cat "$OUT/weak_layer_report_cosine.json"
cat "$OUT/weak_layer_report_l2.json"
cat "$OUT/candidate_plans.json"
```

### 2. Evaluate Candidate Plans

```bash
python -m QEfficient.cloud.model_pruning evaluate \
  --model "$MODEL" \
  --candidate-plans "$OUT/candidate_plans.json" \
  --datasets gsm8k hellaswag \
  --num-samples 64 \
  --max-candidates 5 \
  --eval-method lm_eval \
  --generation-len 40 \
  --accuracy-metric auto \
  --accuracy-threshold 5 \
  --lm-eval-batch-size 1 \
  --dtype float16 \
  --device auto \
  --verbose \
  --output-dir "$OUT"
```

Inspect quality results:

```bash
cat "$OUT/comparison_report.json"
cat "$OUT/evaluation_summary.json"
cat "$OUT/best_plan.json"
cat "$OUT/plan_results.json"
```

### 3. Compile And Benchmark On QAIC

```bash
python -m QEfficient.cloud.model_pruning qaic \
  --model "$MODEL" \
  --plan "$OUT/best_plan.json" \
  --device-group 0 \
  --batch-size 1 \
  --ctx-len 4096 \
  --prefill-seq-len 128 \
  --generation-len 60 \
  --num-cores 16 \
  --compile-dir-base "$OUT/qaic_compile" \
  --qaic-output-dir "$OUT/qaic" \
  --verbose
```

Inspect QAIC comparison:

```bash
cat "$OUT/qaic/all_results.json"
cat "$OUT/qaic/benchmark_comparison.json"
```

## Run The Requested Models

This loop runs analyze, evaluate, and QAIC for the three large model targets. It uses a small sample count for bring-up. Increase `--num-data-samples` after the flow is stable.

```bash
cd /home/abhamidi/NAS_new/efficient-transformers
source image_bug/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -euo pipefail

MODELS=(
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-35B-A3B"
  "google/gemma-4-26B-A4B-it"
)

for MODEL in "${MODELS[@]}"; do
  RUN_NAME=$(printf '%s' "$MODEL" | tr '/:' '__')
  OUT="results/model_pruning/${RUN_NAME}_qaic_both"

  python -m QEfficient.cloud.model_pruning run-all \
    --model "$MODEL" \
    --analysis-datasets gsm8k hellaswag \
    --eval-datasets gsm8k hellaswag \
    --num-data-samples 32 \
    --batch-size 1 \
    --metric both \
    --max-candidates 3 \
    --eval-method lm_eval \
    --eval-generation-len 40 \
    --accuracy-metric auto \
    --accuracy-threshold 100 \
    --lm-eval-batch-size 1 \
    --dtype float16 \
    --device auto \
    --device-group 0 \
    --qaic-batch-size 1 \
    --ctx-len 4096 \
    --prefill-seq-len 128 \
    --generation-len 60 \
    --num-cores 16 \
    --compile-dir-base "$OUT/qaic_compile" \
    --verbose \
    --output-dir "$OUT"
done
```

Inspect all results:

```bash
for MODEL in "Qwen/Qwen3.5-27B" "Qwen/Qwen3.5-35B-A3B" "google/gemma-4-26B-A4B-it"; do
  RUN_NAME=$(printf '%s' "$MODEL" | tr '/:' '__')
  OUT="results/model_pruning/${RUN_NAME}_qaic_both"
  echo "===== $MODEL ====="
  cat "$OUT/evaluation_summary.json"
  cat "$OUT/best_plan.json"
  cat "$OUT/qaic/benchmark_comparison.json"
done
```

For final quality selection, reduce `--accuracy-threshold 100` to a real quality budget such as `5` and increase `--num-data-samples`.

## QAIC-Only Manual Skip Layers

Use this path when layers are known and you do not want HF quality evaluation. This is the fastest way to compare baseline QEff/QAIC against optimized QEff/QAIC.

```bash
MODEL="Qwen/Qwen3.5-35B-A3B"
OUT="results/model_pruning/manual_layers_qwen35_35b_a3b"

python -m QEfficient.cloud.model_pruning qaic \
  --model "$MODEL" \
  --skip-layers 16 5 15 13 \
  --device-group 0 \
  --batch-size 1 \
  --ctx-len 4096 \
  --prefill-seq-len 128 \
  --generation-len 60 \
  --num-cores 16 \
  --compile-dir-base "$OUT/qaic_compile" \
  --qaic-output-dir "$OUT/qaic" \
  --verbose
```

Artifacts:

```bash
cat "$OUT/qaic/manual_best_plan.json"
cat "$OUT/qaic/baseline.json"
cat "$OUT/qaic/benchmark_comparison.json"
```

## Video-MME For VLMs

Video-MME is for VLMs such as Qwen3-VL and Qwen2.5-VL. Text-only models cannot consume video frames.

### 1. Download Video-MME Metadata And Videos

```bash
python QEfficient/model_pruning/datasets/download_datasets.py \
  --datasets videomme \
  --export-jsonl \
  --download-videomme-videos
```

Default local paths:

```bash
QEfficient/model_pruning/datasets/downloaded/videomme/videomme.jsonl
QEfficient/model_pruning/datasets/downloaded/videomme/videos
```

### 2. Analyze Candidate Layers

```bash
MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUT="results/model_pruning/qwen3_vl_videomme_nas"

python -m QEfficient.cloud.model_pruning analyze \
  --model "$MODEL" \
  --datasets gsm8k hellaswag \
  --num-samples 64 \
  --batch-size 1 \
  --metric both \
  --dtype float16 \
  --device cuda \
  --verbose \
  --output-dir "$OUT"
```

### 3. Evaluate On Video-MME GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m QEfficient.cloud.model_pruning evaluate \
  --model "$MODEL" \
  --candidate-plans "$OUT/candidate_plans.json" \
  --datasets videomme \
  --num-samples 50 \
  --eval-method videomme \
  --videomme-dataset-path QEfficient/model_pruning/datasets/downloaded/videomme/videomme.jsonl \
  --videomme-video-root QEfficient/model_pruning/datasets/downloaded/videomme/videos \
  --videomme-num-frames 8 \
  --generation-len 16 \
  --dtype float16 \
  --device cuda \
  --verbose \
  --accuracy-threshold 5 \
  --output-dir "$OUT"
```

Manual Video-MME quality check:

```bash
python -m QEfficient.cloud.model_pruning evaluate \
  --model "$MODEL" \
  --skip-layers 16 5 15 13 \
  --datasets videomme \
  --num-samples 50 \
  --eval-method videomme \
  --videomme-dataset-path QEfficient/model_pruning/datasets/downloaded/videomme/videomme.jsonl \
  --videomme-video-root QEfficient/model_pruning/datasets/downloaded/videomme/videos \
  --videomme-num-frames 8 \
  --generation-len 16 \
  --dtype float16 \
  --device cuda \
  --verbose \
  --output-dir results/model_pruning/qwen3_vl_videomme_manual
```

### 4. QAIC Video-MME Comparison

```bash
python -m QEfficient.cloud.model_pruning qaic \
  --model "$MODEL" \
  --plan "$OUT/best_plan.json" \
  --device-group 0 \
  --batch-size 1 \
  --ctx-len 4096 \
  --prefill-seq-len 128 \
  --generation-len 16 \
  --num-cores 16 \
  --videomme-dataset-path QEfficient/model_pruning/datasets/downloaded/videomme/videomme.jsonl \
  --videomme-video-root QEfficient/model_pruning/datasets/downloaded/videomme/videos \
  --videomme-num-samples 50 \
  --videomme-num-frames 8 \
  --compile-dir-base "$OUT/qaic_compile" \
  --qaic-output-dir "$OUT/qaic" \
  --verbose
```

## Outputs

| File | Stage | Purpose |
| --- | --- | --- |
| `weak_layer_report.json` | analyze | Combined or selected metric layer ranking |
| `weak_layer_report_cosine.json` | analyze | Cosine ranking when `--metric both` is used |
| `weak_layer_report_l2.json` | analyze | L2 ranking when `--metric both` is used |
| `candidate_plans.json` | analyze | Baseline plus generated skip-layer plans |
| `manual_candidate_plans.json` | evaluate | Baseline plus manual skip-layer plan |
| `plan_results.json` | evaluate | Per-plan HF quality result |
| `comparison_report.json` | evaluate | Baseline vs selected plan quality comparison |
| `evaluation_summary.json` | evaluate | Run metadata and selected plan summary |
| `best_plan.json` | evaluate | Plan consumed by QAIC |
| `manual_best_plan.json` | qaic | QAIC-only manual plan generated from `--skip-layers` |
| `baseline.json` | qaic | Baseline QAIC run result |
| `<plan_name>.json` | qaic | Optimized QAIC run result |
| `all_results.json` | qaic | Full QAIC output bundle |
| `benchmark_comparison.json` | qaic | Baseline vs optimized QAIC comparison |

`benchmark_comparison.json` includes:

- `metrics.baseline_compile_time_s`
- `metrics.optimized_compile_time_s`
- `metrics.baseline_ttft_s`
- `metrics.optimized_ttft_s`
- `metrics.baseline_decode_tokens_per_sec`
- `metrics.optimized_decode_tokens_per_sec`
- `metrics.baseline_total_tokens_per_sec`
- `metrics.optimized_total_tokens_per_sec`
- `metrics.baseline_e2e_s`
- `metrics.optimized_e2e_s`
- deltas and improvement percentages

## Precision And Runtime Notes

- Analyze and evaluate use HF/Transformers and respect `--dtype`.
- The README examples use `--dtype float16` for large GPU runs.
- QAIC benchmarking loads fresh QEfficient models in `float32` before QEff compile.
- QAIC compile uses `mxfp6_matmul` and `mxint8_kv_cache` by default.
- Disable QAIC compile quantization options with:

```bash
--no-mxfp6-matmul --no-mxint8-kv-cache
```

## Troubleshooting

### `Unsupported model type for v1 skip transform: 'qwen3_vl'`

Use the latest model-pruning branch. Qwen-VL model types are supported by the skip-layer adapter:

- `qwen2_5_vl`
- `qwen3_vl`
- `qwen3_vl_moe`

### `TypeError: a bytes-like object is required, not 'str'` during Video-MME

Install `qwen_vl_utils` and use the latest Video-MME evaluator. The evaluator uses `qwen_vl_utils.process_vision_info` for Qwen-VL processors when available.

```bash
python -m pip install qwen-vl-utils
```

### `torchcodec is not installed`

This warning is not fatal. Install `torchcodec` if you want the preferred video decoding backend:

```bash
python -m pip install torchcodec
```

### `No non-baseline plan fit the accuracy threshold`

Either no candidate met the quality budget or every non-baseline candidate failed. Inspect:

```bash
cat "$OUT/plan_results.json"
cat "$OUT/comparison_report.json"
```

For bring-up, increase the threshold:

```bash
--accuracy-threshold 100
```

For manual layers, bypass candidate selection:

```bash
python -m QEfficient.cloud.model_pruning qaic --model "$MODEL" --skip-layers 16 5 15 13 ...
```

### `lm_eval` or `datasets` import errors

Refresh the evaluation stack:

```bash
python -m pip install -U "lm-eval[api]" "datasets>=2.19,<3.0" "pyarrow>=15" "evaluate"
```

## Validation

Run syntax checks:

```bash
python -m py_compile \
  QEfficient/cloud/model_pruning.py \
  QEfficient/model_pruning/nas_pipeline.py \
  QEfficient/model_pruning/qeff_model_optimizer/evaluation/qaic_benchmark.py \
  QEfficient/model_pruning/qeff_model_optimizer/evaluation/videomme.py \
  QEfficient/model_pruning/qeff_model_optimizer/transforms/adapters.py \
  QEfficient/model_pruning/qeff_model_optimizer/transforms/skip_layers.py \
  tests/model_pruning/test_model_pruning_workflows.py
```

Run workflow tests:

```bash
python -m pytest tests/model_pruning/test_model_pruning_workflows.py -q
```

Expected result on this branch:

```text
32 passed
```

## Limitations

- Current pruning is layer skipping, not structural channel/head/MLP checkpoint rewriting.
- The pruned plan is applied in memory during evaluation or QAIC compile.
- Quality depends on the calibration/evaluation samples and threshold.
- QAIC benchmarks require QAIC hardware and a valid QEfficient installation.
- Some Video-MME source videos may be unavailable from upstream URLs; record downloaded counts when reporting results.
