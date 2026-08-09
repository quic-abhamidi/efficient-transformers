# MoE Pipelines

This folder contains the two GPT-OSS Mixture-of-Experts (MoE) pipelines added to this NAS repo.

- `run_moe_expert_importance_pipeline.py` — Pipeline 1: measure/rank expert importance.
- `run_pipeline_moe_pruning_gpu.py` — Pipeline 2: run baseline vs pruned-expert evaluation.

Both pipelines are adapted from the logic in `Reference_moe/` and kept as standalone runners under `moe_pipelines/` so they do not require changes to the core NAS transform system.

---

## What Was Added

### Pipeline 1: Expert Importance

Pipeline 1 computes how important each MoE expert is in each GPT-OSS layer.

Main runner:

```bash
moe_pipelines/run_moe_expert_importance_pipeline.py
```

Supporting modules:

```bash
analysis/measure_moe_expert_importance.py
analysis/analyze_moe_expert_importance.py
```

Pipeline 1 produces routing CSVs, expert-importance CSVs, and a JSON file containing ranked experts per layer.

### Pipeline 2: Pruned Expert Run

Pipeline 2 consumes the JSON from Pipeline 1, selects the first `N` least-important experts per layer, masks those experts at the router, and compares baseline vs pruned model accuracy.

Main runner:

```bash
moe_pipelines/run_pipeline_moe_pruning_gpu.py
```

Pipeline 2 produces baseline/pruned benchmark results, pruning summary, comparison report, and pipeline summary.

---

## GPT-OSS Router Pruning Behavior

The current repo already has GPT-OSS router masking logic in the local Transformers GPT-OSS model files:

```bash
transformers/transformers/src/transformers/models/gpt_oss/modular_gpt_oss.py
transformers/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py
```

The pruning behavior is:

1. Compute router logits for every token and every expert.
2. Before top-k expert selection, set pruned experts' logits to the minimum dtype value.
3. Run `topk` over the masked logits.
4. Route tokens only to selected non-pruned experts.

Important: pruned expert weights are not deleted, and router logits are not set to zero. They are set to a very large negative value:

```python
router_logits = router_logits.masked_fill(
    self._pruned_expert_mask,
    torch.finfo(router_logits.dtype).min,
)
```

This means a token should not route to a pruned expert as long as:

- router masking is applied,
- at least `top_k` unpruned experts remain,
- the loaded model uses the compatible GPT-OSS router path,
- no external fused router bypasses this Python router logic.

Pipeline 2 also includes a runtime fallback. If the loaded GPT-OSS model does not already expose native `set_pruned_experts`, the pipeline patches compatible router modules in memory and applies the same mask-before-top-k behavior.

---

# Pipeline 1: Expert Importance Pipeline

## Purpose

Pipeline 1 answers:

> For each MoE layer, which experts are least important and most important based on routing behavior?

It does this by running dataset text through the model and observing router decisions.

---

## Pipeline 1 Command

Minimal smoke test:

```bash
python moe_pipelines/run_moe_expert_importance_pipeline.py \
  --model openai/gpt-oss-20b \
  --datasets hellaswag \
  --max-samples 1 \
  --batch-size 1 \
  --max-length 64 \
  --device cuda \
  --clean-checkpoint
```

Run on 2000 HellaSwag samples:

```bash
python moe_pipelines/run_moe_expert_importance_pipeline.py \
  --model openai/gpt-oss-20b \
  --datasets hellaswag \
  --max-samples 2000 \
  --batch-size 1 \
  --max-length 512 \
  --device cuda \
  --clean-checkpoint
```

---

## Pipeline 1 Arguments

### `--model`

Hugging Face model ID or local model path.

Default:

```bash
openai/gpt-oss-20b
```

Example:

```bash
--model openai/gpt-oss-20b
```

### `--datasets`

Dataset aliases to use for routing profiling.

Default:

```bash
hellaswag
```

Available aliases:

```text
gsm8k
hellaswag
winogrande
wikitext
mmlu
arc_easy
arc_challenge
truthfulqa
piqa
boolq
openbookqa
all
```

Examples:

```bash
--datasets hellaswag
```

```bash
--datasets hellaswag winogrande
```

```bash
--datasets all
```

### `--max-samples`

Maximum number of non-empty samples per dataset.

Example:

```bash
--max-samples 2000
```

### `--num-samples`

Alias for `--max-samples`. If both are given, `--num-samples` takes precedence.

Example:

```bash
--num-samples 2000
```

### `--batch-size`

Number of texts per forward pass.

Default:

```bash
1
```

For large GPT-OSS models, keep this low unless you know your GPU memory can handle more.

### `--max-length`

Tokenizer truncation length.

Default:

```bash
512
```

For smoke tests, use a smaller value such as `64`.

### `--output-dir`

Pipeline output directory.

If omitted, the default is:

```text
<clean_model_name>_MoE_Expert_Importance
```

For `openai/gpt-oss-20b`, this becomes:

```text
gpt-oss-20b_MoE_Expert_Importance
```

### `--device`

Device used when no explicit device map is set.

Examples:

```bash
--device cuda
--device cuda:0
--device cpu
--device auto
```

### `--device-map`

Optional Transformers `device_map`.

Examples:

```bash
--device-map auto
--device-map none
```

If CUDA is used and `--device-map` is omitted, Pipeline 1 resolves it to `auto` for large-model loading.

### `--torch-dtype`

Model dtype.

Choices:

```text
auto
float32
float16
bfloat16
```

Default:

```bash
bfloat16
```

### `--trust-remote-code`

Trust remote Hugging Face model code.

This is enabled by default in the pipeline logic.

### `--write-importance-debug`

Also write debug matrices:

```text
importance_sum
importance_mean
```

Normally not needed unless debugging the scoring.

### `--importance-metric`

Metric used for expert ranking.

Choices:

```text
combined_score
freq_fraction
freq_counts
```

Default:

```bash
combined_score
```

### `--top-k-experts`

Number of experts per layer kept in the original top-k ranking/consensus outputs.

Default:

```bash
3
```

Important: this does not limit `expert_importance_full.csv` or `pruned_experts.json`. Those contain all experts per layer, ordered least-to-most important.

### `--force-rerun`

Ignore completed checkpoint stages and rerun all stages.

### `--resume-from`

Force rerun from a specific stage onward.

Choices:

```text
moe_routing_profile
expert_importance_analysis
pruned_experts_json
summary
```

### `--clean-checkpoint`

Delete the existing checkpoint before starting.

Use this when changing arguments and wanting a fresh run.

---

## Pipeline 1 Step-by-Step Flow

### Step 1: Load model and tokenizer

Pipeline 1 loads the model with:

- `AutoConfig.from_pretrained`
- `AutoTokenizer.from_pretrained`
- `AutoModelForCausalLM.from_pretrained`

It uses:

- `trust_remote_code=True`
- `low_cpu_mem_usage=True`
- `torch_dtype=bfloat16` by default
- `device_map=auto` for CUDA large-model loading if not explicitly set

If the tokenizer has no pad token, it uses the EOS token as the pad token.

### Step 2: Discover MoE layers

The profiler scans `model.named_modules()` and identifies MoE MLP modules that have:

```text
.router
.experts
.router.top_k
.experts.num_experts
```

For each discovered MoE layer, it records:

```text
layer_idx
module_name
router_module_name
num_experts
top_k
```

### Step 3: Register router hooks

The pipeline registers forward hooks on router modules.

During model forward passes, each router hook collects:

- selected expert IDs,
- router scores/weights,
- fallback top-k routing from logits if selected IDs are not exposed.

### Step 4: Load dataset samples

The pipeline uses the dataset registry in `analysis/measure_moe_expert_importance.py`.

Examples:

- `hellaswag` uses text column `ctx`
- `gsm8k` uses text column `question`
- `wikitext` uses text column `text`

It skips empty text rows and stops after `--max-samples` non-empty rows.

### Step 5: Forward pass and routing collection

For each batch:

1. Tokenize text.
2. Move input tensors to the model input device.
3. Store `attention_mask` in the profiler.
4. Run the model with `use_cache=False`.
5. Router hooks collect expert usage.

Padding tokens are ignored when the attention mask shape matches the routing shape.

### Step 6: Accumulate routing metrics

For each layer and expert, the profiler collects:

```text
freq_counts
importance_sum
```

Then it computes:

```text
freq_fraction
importance_mean
combined_score
```

### Metric meanings

#### `freq_counts`

Raw count of how often an expert was selected.

If expert 10 in layer 5 was selected 2,000 times, its `freq_counts` is 2000.

#### `freq_fraction`

Normalized selection frequency inside a layer.

Formula:

```text
expert_freq_count / total_expert_selections_in_layer
```

#### `importance_sum`

Sum of router weights for that expert across selections.

This is mostly a debug/intermediate metric.

#### `importance_mean`

Average router score when that expert was selected.

Formula:

```text
importance_sum / freq_counts
```

#### `combined_score`

Main importance score used by default.

Formula:

```text
combined_score = freq_fraction * importance_mean
```

This combines how often the expert is selected with how strongly the router weights it.

Lower `combined_score` means less important by this pipeline's ranking.

---

## Pipeline 1 Outputs

Default output directory:

```text
gpt-oss-20b_MoE_Expert_Importance/
```

### Checkpoint

```text
pipeline_checkpoint.json
```

Tracks completed stages.

### Routing matrix files

Inside:

```text
moe_routing/
```

Example for HellaSwag:

```text
hellaswag_freq_counts.csv
hellaswag_freq_fraction.csv
hellaswag_combined_score.csv
```

Optional debug files if `--write-importance-debug` is used:

```text
hellaswag_importance_sum.csv
hellaswag_importance_mean.csv
```

### `expert_importance_rankings.csv`

Original reference-style top-k ranking file.

Contains both:

```text
least_important
most_important
```

This file is controlled by `--top-k-experts`.

### `expert_importance_consensus.csv`

Original reference-style consensus file.

Aggregates top-k rankings across datasets.

This file is also controlled by `--top-k-experts`.

### `expert_importance_full.csv`

This is the complete all-expert importance file added for this repo workflow.

It contains exactly:

```text
num_layers * num_experts_per_layer
```

rows, aggregated across datasets.

Columns:

```text
layer_index
expert_index
importance_rank
metric
dataset_frequency
datasets
mean_score
min_score
max_score
```

Example row:

```csv
0,28,11,combined_score,1,['hellaswag'],0.001444498698,0.001444498698,0.001444498698
```

Meaning:

- layer `0`
- expert `28`
- rank `11` inside layer 0
- ranked by `combined_score`
- data came from 1 dataset: `hellaswag`
- mean/min/max score are equal because only one dataset was used

`importance_rank = 1` means least important expert in that layer.

### `pruned_experts.json`

This is generated from `expert_importance_full.csv`.

Format:

```json
{
  "pruned_experts": {
    "0": [2, 0, 1],
    "1": [1, 2, 0]
  }
}
```

Important: despite the name, this file from Pipeline 1 is a **full ranked expert list** per layer. It contains all experts per layer ordered from least important to most important.

Pipeline 2 uses `--experts-per-layer` to choose how many from the front of each list to actually prune.

### `expert_importance_summary.json`

JSON summary of Pipeline 1 analysis outputs.

### `pipeline_summary.json`

Final pipeline run summary.

---

# Pipeline 2: Pruned Expert Evaluation Pipeline

## Purpose

Pipeline 2 answers:

> If we prune the least-important MoE experts, how much does benchmark accuracy change?

It compares:

```text
baseline GPT-OSS model
vs
same GPT-OSS model with selected experts masked at router
```

---

## Pipeline 2 Command

Minimal smoke test:

```bash
python moe_pipelines/run_pipeline_moe_pruning_gpu.py \
  --model openai/gpt-oss-20b \
  --prune-config gpt-oss-20b_MoE_Expert_Importance/pruned_experts.json \
  --experts-per-layer 1 \
  --datasets hellaswag \
  --num-samples 1 \
  --batch-size 1 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --clean-checkpoint
```

Run on 200 HellaSwag samples:

```bash
python moe_pipelines/run_pipeline_moe_pruning_gpu.py \
  --model openai/gpt-oss-20b \
  --prune-config gpt-oss-20b_MoE_Expert_Importance/pruned_experts.json \
  --experts-per-layer 1 \
  --datasets hellaswag \
  --num-samples 200 \
  --batch-size 1 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --clean-checkpoint
```

---

## Pipeline 2 Arguments

### `--model`

Required model ID or local path.

Example:

```bash
--model openai/gpt-oss-20b
```

### `--prune-config`

Required path to the Pipeline 1 JSON.

Example:

```bash
--prune-config gpt-oss-20b_MoE_Expert_Importance/pruned_experts.json
```

Expected format:

```json
{
  "pruned_experts": {
    "0": [2, 0, 1],
    "1": [1, 2, 0]
  }
}
```

For Pipeline 2, this is treated as a ranked list. It does not prune every listed expert.

### `--experts-per-layer`

Required.

Number of least-important experts to prune per layer.

Example:

```bash
--experts-per-layer 1
```

If Pipeline 1 JSON says:

```json
{
  "pruned_experts": {
    "0": [7, 21, 25, 8]
  }
}
```

and you pass:

```bash
--experts-per-layer 2
```

Pipeline 2 prunes only:

```json
{
  "0": [7, 21]
}
```

The actual selected config is saved as:

```text
selected_prune_config.json
```

### `--datasets`

Benchmark datasets to evaluate.

Default:

```bash
hellaswag
```

Available:

```text
gsm8k
hellaswag
winogrande
mmlu
arc_easy
arc_challenge
truthfulqa
piqa
boolq
openbookqa
all
```

### `--num-samples`

lm-eval sample limit per dataset.

Examples:

```bash
--num-samples 1
--num-samples 200
```

If omitted, lm-eval evaluates the full task.

### `--accuracy-threshold`

Maximum acceptable relative accuracy drop percentage.

Default:

```bash
3.0
```

Used only for the recommendation in `comparison_report.json`.

### `--device`

Device passed to lm-eval and model loading.

Default:

```bash
cuda
```

### `--device-map`

Transformers device map.

Default:

```bash
auto
```

Use `none` to disable device map and call `model.to(device)`.

### `--torch-dtype`

Model dtype.

Choices:

```text
auto
float32
float16
bfloat16
```

Default:

```bash
bfloat16
```

### `--batch-size`

Batch size passed to lm-eval `HFLM`.

Default:

```bash
1
```

### `--output-dir`

Output directory.

If omitted, default is:

```text
<clean_model_name>_moe_pruning_gpu
```

For `openai/gpt-oss-20b`, this becomes:

```text
gpt-oss-20b_moe_pruning_gpu
```

### `--lm-eval-verbosity`

Verbosity for lm-eval.

Default:

```bash
WARNING
```

### `--force-rerun`

Ignore completed checkpoint stages and rerun.

### `--clean-checkpoint`

Delete existing checkpoint before starting.

---

## Pipeline 2 Step-by-Step Flow

### Step 1: Load Pipeline 1 JSON

Reads:

```text
--prune-config
```

This file contains all experts per layer ordered from least important to most important.

### Step 2: Select first N experts per layer

Uses:

```text
--experts-per-layer
```

For every layer:

```text
selected_experts = ranked_experts[:experts_per_layer]
```

Writes:

```text
selected_prune_config.json
```

This file is the actual pruning config used in the pruned benchmark.

### Step 3: Run baseline benchmark

Loads the original model without pruning.

Runs lm-eval on requested datasets.

Writes:

```text
baseline_results.json
baseline_scores.json
```

Then deletes the model/tokenizer and clears CUDA memory.

### Step 4: Run pruned benchmark

Loads a fresh copy of the model.

Applies selected expert pruning.

Runs lm-eval again.

Writes:

```text
pruned_results.json
pruned_scores.json
pruning_summary.json
```

Then deletes the model/tokenizer and clears CUDA memory.

### Step 5: Generate comparison report

Compares baseline vs pruned scores.

Writes:

```text
comparison_report.json
comparison_report.csv
```

The report includes:

```text
baseline_score
pruned_score
abs_delta
pct_delta
within_threshold
metric
```

### Step 6: Write pipeline summary

Writes:

```text
pipeline_summary.json
```

---

## Pipeline 2 Outputs

Default output directory:

```text
gpt-oss-20b_moe_pruning_gpu/
```

Expected files:

```text
pipeline_checkpoint.json
selected_prune_config.json
baseline_results.json
baseline_scores.json
pruned_results.json
pruned_scores.json
pruning_summary.json
comparison_report.json
comparison_report.csv
pipeline_summary.json
```

---

## How Experts Are Actually Pruned

Pipeline 2 does not remove expert weights from the model.

It does not set the expert weights to zero.

It does not set router logits to zero.

Instead, it masks router logits for selected experts before top-k selection:

```python
router_logits = router_logits.masked_fill(
    self._pruned_expert_mask,
    torch.finfo(router_logits.dtype).min,
)
```

Then router top-k is computed:

```python
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
```

Because the pruned expert logits are extremely negative, top-k should not select them.

That means tokens should not go to pruned experts as long as at least `top_k` unpruned experts remain.

---

## Important Safety Rule: Do Not Over-Prune

If a layer has:

```text
num_experts = 32
top_k = 4
```

then at least 4 experts must remain available.

So the maximum safe prune count is:

```text
32 - 4 = 28
```

If you prune too many, the router cannot select enough experts.

Pipeline 2 preserves the reference check:

```text
Cannot prune X experts when num_experts=... and top_k=...
```

Start small:

```bash
--experts-per-layer 1
```

Then gradually increase.

---

## Recommended End-to-End Workflow

### 1. Run Pipeline 1 on a small sample

```bash
python moe_pipelines/run_moe_expert_importance_pipeline.py \
  --model openai/gpt-oss-20b \
  --datasets hellaswag \
  --max-samples 100 \
  --batch-size 1 \
  --max-length 512 \
  --device cuda \
  --clean-checkpoint
```

### 2. Inspect full expert ranking

```bash
head gpt-oss-20b_MoE_Expert_Importance/expert_importance_full.csv
```

### 3. Inspect ranked JSON

```bash
cat gpt-oss-20b_MoE_Expert_Importance/pruned_experts.json
```

### 4. Run Pipeline 2 with one expert per layer

```bash
python moe_pipelines/run_pipeline_moe_pruning_gpu.py \
  --model openai/gpt-oss-20b \
  --prune-config gpt-oss-20b_MoE_Expert_Importance/pruned_experts.json \
  --experts-per-layer 1 \
  --datasets hellaswag \
  --num-samples 200 \
  --batch-size 1 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --clean-checkpoint
```

### 5. Inspect selected pruning config

```bash
cat gpt-oss-20b_moe_pruning_gpu/selected_prune_config.json
```

### 6. Inspect comparison report

```bash
cat gpt-oss-20b_moe_pruning_gpu/comparison_report.json
```

---

## Validation Commands

Check Pipeline 1 syntax:

```bash
python -m py_compile \
  analysis/measure_moe_expert_importance.py \
  analysis/analyze_moe_expert_importance.py \
  moe_pipelines/run_moe_expert_importance_pipeline.py
```

Check Pipeline 1 CLI:

```bash
python moe_pipelines/run_moe_expert_importance_pipeline.py --help
```

Check Pipeline 2 syntax:

```bash
python -m py_compile moe_pipelines/run_pipeline_moe_pruning_gpu.py
```

Check Pipeline 2 CLI:

```bash
python moe_pipelines/run_pipeline_moe_pruning_gpu.py --help
```

---

## Notes and Assumptions

- These pipelines target GPT-OSS-style MoE models.
- Pipeline 1 expects router modules exposing `top_k` and expert modules exposing `num_experts`.
- Pipeline 2 expects the model structure to expose `model.model.layers[*].mlp` like GPT-OSS.
- Pipeline 2 has a fallback for unpatched router classes, but it still expects a compatible router shape.
- Full model runs require enough GPU memory for `openai/gpt-oss-20b`.
- `--num-samples 1` is only a smoke test, not a meaningful benchmark.
- Pipeline 1 `pruned_experts.json` is a ranked expert list, not necessarily the exact number of experts to prune.
- Pipeline 2 decides how many experts to actually prune using `--experts-per-layer`.
