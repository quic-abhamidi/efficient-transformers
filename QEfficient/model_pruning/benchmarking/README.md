# Benchmarking Module

Model evaluation and performance comparison tools.

## Overview

The benchmarking module provides tools for running standardized benchmarks on models and generating detailed comparison reports between baseline and optimized models.

## Components

### `run_benchmark.py`

Run lm-evaluation-harness benchmarks on models.

### `generate_report.py`

Generate comparison reports between baseline and optimized models.

## Usage

### 1. Run Benchmarks

#### Standalone CLI

```bash
# Run single benchmark
python -m benchmarking.run_benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset gsm8k \
    --batch-size 16 \
    --limit 100

# Run multiple benchmarks
python -m benchmarking.run_benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset gsm8k hellaswag winogrande \
    --batch-size 16 \
    --output-dir benchmark_results

# Run with layer skipping
python -m benchmarking.run_benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset gsm8k \
    --skip-layers 5 6 \
    --limit 100
```

#### As a Module (Import)

```python
from QEfficient.model_pruning.benchmarking.run_benchmark import run_lm_eval, load_model, BENCHMARK_MAPPING

# Load model
model, tokenizer = load_model(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    device="cuda"
)

# Run evaluation
results = run_lm_eval(
    model=model,
    tokenizer=tokenizer,
    tasks=["gsm8k", "hellaswag"],
    batch_size=16,
    device="cuda",
    limit=100
)

# Access results
for task, metrics in results['results'].items():
    print(f"{task}: {metrics}")
```

### 2. Generate Comparison Reports

#### Standalone CLI

```bash
# Assumes benchmark_results/ contains baseline_model/ and target_model/ subdirectories
python -m benchmarking.generate_report
```

#### As a Module (Import)

```python
from QEfficient.model_pruning.benchmarking.generate_report import BenchmarkReportGenerator

# Create report generator
generator = BenchmarkReportGenerator(
    results_dir="benchmark_results",
    output_dir="comparison_output"
)

# Generate report
generator.generate_report()
```

## API Reference

### `run_lm_eval()`

Run lm-evaluation-harness on a pre-loaded model.

**Parameters:**
- `model`: Pre-loaded model object
- `tokenizer`: Pre-loaded tokenizer object
- `tasks` (List[str]): List of task names to evaluate
- `batch_size` (int): Batch size for evaluation
- `device` (str): Device to use
- `limit` (Optional[int]): Limit number of samples (for testing)
- `num_fewshot` (Optional[int]): Number of few-shot examples
- `use_cache` (bool): Enable request caching
- `cache_dir` (Optional[str]): Cache directory path
- `dtype` (str): Model dtype
- `log_samples` (bool): Log individual sample predictions
- `random_seed` (int): Random seed for reproducibility
- `verbosity` (str): Logging verbosity level

**Returns:**
- Dictionary containing evaluation results

### `load_model()`

Load a HuggingFace model and tokenizer.

**Parameters:**
- `model_name` (str): HuggingFace model identifier
- `device` (str): Device to use ("cuda", "cpu", "auto")
- `trust_remote_code` (bool): Whether to trust remote code

**Returns:**
- Tuple of (model, tokenizer)

### `BenchmarkReportGenerator`

Generate comparison reports for benchmark results.

**Key Methods:**
- `identify_models()`: Identify baseline and target model directories
- `load_results()`: Load benchmark results from model directory
- `generate_comparison_dataframe()`: Generate comparison DataFrame
- `save_csv_report()`: Save comparison to CSV
- `create_bar_chart()`: Create grouped bar chart visualization
- `create_delta_chart()`: Create delta/change visualization
- `generate_report()`: Generate complete report

## Supported Benchmarks

Available through `BENCHMARK_MAPPING`:
- `gsm8k`: Grade School Math
- `hellaswag`: Commonsense reasoning
- `winogrande`: Commonsense reasoning
- `mmlu`: Massive Multitask Language Understanding
- `arc_easy`: ARC Easy
- `arc_challenge`: ARC Challenge
- `truthfulqa`: TruthfulQA
- `piqa`: Physical Interaction QA
- `boolq`: Boolean Questions
- `openbookqa`: OpenBookQA

## Output Files

### Benchmark Results
- `results_{timestamp}.json`: Raw benchmark results
- `results.csv`: Extracted metrics in CSV format

### Comparison Reports
- `comparison_metrics.csv`: Side-by-side comparison table
- `performance_chart.png`: Grouped bar chart (baseline vs target)
- `delta_chart.png`: Performance delta visualization

## Dependencies

- lm-eval>=0.4.0
- transformers
- torch
- pandas
- matplotlib
- numpy
