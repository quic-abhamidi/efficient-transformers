# Optimization Module

Model optimization techniques for reducing computational cost and improving inference performance.

## Overview

The optimization module provides various techniques for optimizing transformer models. Currently supports layer skipping, with plans to add quantization, pruning, and knowledge distillation.

## Current Techniques

### Layer Skipping

Skip selected layers during inference to reduce computational cost with minimal accuracy impact.

**Location:** `optimization/layer_skipping/`

## Usage

### Layer Skipping Configuration Generation

#### Standalone CLI

```bash
python -m optimization.layer_skipping.generate_config \
    --contribution-dir Llama-3.2-1B-Instruct_Analysis/layer_contributions \
    --metric both \
    --threshold-percentile 10.0 \
    --max-skip-layers 3 \
    --output layer_skip_configs.json \
    --visualize
```

#### As a Module (Import)

```python
from QEfficient.model_pruning.optimization.layer_skipping.generate_config import generate_configurations

# Generate skip configurations
config_data = generate_configurations(
    contribution_dir="Llama-3.2-1B-Instruct_Analysis/layer_contributions",
    metric="both",
    threshold_percentile=10.0,
    max_skip_layers=3
)

# Access configurations
configurations = config_data['configurations']
for config in configurations:
    print(f"{config['name']}: skip layers {config['skip_layers']}")
```

## API Reference

### `generate_configurations()`

Generate layer-skipping configurations based on layer contribution analysis.

**Parameters:**
- `contribution_dir` (str): Directory containing layer contribution CSV files
- `metric` (str): Metric to use ("cosine", "l2", or "both")
- `threshold_percentile` (float): Percentile threshold for low-impact layers (e.g., 10.0 = bottom 10%)
- `max_skip_layers` (int): Maximum number of layers to skip in any configuration

**Returns:**
- Dictionary with structure:
  ```python
  {
      'metadata': {...},
      'layer_analysis': {...},
      'configurations': [
          {
              'id': 0,
              'name': 'baseline',
              'skip_layers': [],
              'num_skipped': 0,
              'description': '...',
              'rationale': '...',
              'confidence': 'baseline'
          },
          {
              'id': 1,
              'name': 'skip_layer_13',
              'skip_layers': [13],
              'num_skipped': 1,
              'description': '...',
              'rationale': '...',
              'confidence': 'high',
              'supporting_datasets': [...]
          },
          ...
      ]
  }
  ```

## Configuration Types Generated

1. **Baseline**: No layers skipped (for comparison)
2. **Single Layer**: Skip individual low-impact layers
3. **Pairs**: Skip combinations of 2 low-impact layers
4. **Triplets**: Skip combinations of 3 low-impact layers
5. **Consecutive**: Skip adjacent layers
6. **High Consensus**: Skip layers identified across multiple datasets

## Output Files

- `layer_skip_configs.json`: Generated configurations with metadata
- `layer_skip_configs_heatmap.png`: Visualization of skip patterns (if --visualize)

## Future Techniques

### Quantization (Planned)
- INT8/INT4 quantization
- Mixed-precision quantization
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)

### Pruning (Planned)
- Structured pruning
- Unstructured pruning
- Magnitude-based pruning
- Gradient-based pruning

### Knowledge Distillation (Planned)
- Teacher-student distillation
- Self-distillation
- Progressive distillation

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
