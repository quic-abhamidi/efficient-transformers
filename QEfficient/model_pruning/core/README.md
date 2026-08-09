# Core Module

Model loading utilities for the NAS optimization toolkit.

## Overview

The core module provides utilities for loading and wrapping models with optimization configurations, particularly for layer-skipping functionality.

## Components

### `model_wrapper.py`

Provides the `SkipLayerModelLoader` class for loading models with pre-configured skip layers.

## Usage

### As a Module (Import)

```python
from QEfficient.model_pruning.core.model_wrapper import SkipLayerModelLoader

# Load model with layer skipping
model, tokenizer, config = SkipLayerModelLoader.load_model_with_skip_layers(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    skip_layers=[5, 10, 15],
    device_map="cuda",
    trust_remote_code=True
)

# Load baseline model (no skipping)
model, tokenizer, config = SkipLayerModelLoader.load_model_with_skip_layers(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    skip_layers=None,
    device_map="cuda"
)
```

## API Reference

### `SkipLayerModelLoader.load_model_with_skip_layers()`

Load a model with optional layer-skipping configuration.

**Parameters:**
- `model_name` (str): HuggingFace model identifier
- `skip_layers` (Optional[List[int]]): List of layer indices to skip (None for baseline)
- `device_map` (str): Device mapping strategy ("auto", "cuda", "cpu")
- `trust_remote_code` (bool): Whether to trust remote code

**Returns:**
- Tuple of (model, tokenizer, config)

## Dependencies

- transformers
- torch
