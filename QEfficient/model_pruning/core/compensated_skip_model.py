#!/usr/bin/env python3
"""
Compensated Skip-Layer Model Wrapper

Wraps a transformer model to:
1. Skip specified layers during forward pass
2. Add mean embedding compensation to the layer before skip

This allows testing whether adding the mean embedding delta from skipped layers
can help maintain model performance when layers are removed.

Author: LLM Interpretability Engineer
"""

from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompensatedSkipLayerModel:
    """
    Wrapper that skips specified layers and adds compensation vector.
    
    The compensation works by:
    1. Loading a pre-computed mean embedding delta vector
    2. Adding this vector to the output of the layer before the skip
    3. Then skipping the specified layers as normal
    
    Example:
        If skipping layers 19-22 with compensation at layer 18:
        - Layer 18 output: h18
        - Add compensation: h18_compensated = h18 + mean_delta_vector
        - Skip layers 19-22
        - Continue with layer 23 using h18_compensated as input
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        skip_layers: List[int],
        compensation_vector: torch.Tensor,
        compensation_layer: Optional[int] = None
    ):
        """
        Initialize compensated skip-layer model.
        
        Args:
            model: The base transformer model
            tokenizer: Associated tokenizer
            skip_layers: List of layer indices to skip (e.g., [19, 20, 21, 22])
            compensation_vector: Mean embedding delta vector to add [hidden_dim]
            compensation_layer: Layer to add compensation to (default: layer before first skip)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.skip_layers = sorted(skip_layers)
        self.compensation_vector = compensation_vector.to(model.device)
        
        # Default: add compensation to the layer before the first skipped layer
        if compensation_layer is None:
            self.compensation_layer = min(skip_layers) - 1
        else:
            self.compensation_layer = compensation_layer
        
        # Validate
        if self.compensation_layer < 0:
            raise ValueError(f"Compensation layer {self.compensation_layer} is invalid")
        if self.compensation_layer in skip_layers:
            raise ValueError(f"Compensation layer {self.compensation_layer} cannot be in skip_layers")
        
        print(f"CompensatedSkipLayerModel initialized:")
        print(f"  Skipping layers: {skip_layers}")
        print(f"  Adding compensation at layer: {self.compensation_layer}")
        print(f"  Compensation vector shape: {compensation_vector.shape}")
        print(f"  Compensation vector norm: {torch.norm(compensation_vector).item():.6f}")
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to implement skip and compensation."""
        # Get the model's decoder layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Most models (Llama, Qwen, etc.)
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            layers = self.model.transformer.h
        else:
            raise ValueError("Could not find decoder layers in model")
        
        # Hook for compensation layer
        def compensation_hook(module, input, output):
            """Add compensation vector to layer output."""
            if isinstance(output, tuple):
                # Output is (hidden_states, ...) tuple
                hidden_states = output[0]
                # Add compensation vector (broadcast across batch and sequence)
                compensated = hidden_states + self.compensation_vector.view(1, 1, -1)
                return (compensated,) + output[1:]
            else:
                # Output is just hidden_states tensor
                return output + self.compensation_vector.view(1, 1, -1)
        
        # Hook for skipped layers
        def skip_hook(module, input, output):
            """Return input unchanged (effectively skipping the layer)."""
            if isinstance(input, tuple):
                return input[0]  # Return just the hidden states
            return input
        
        # Register hooks
        self.hooks = []
        
        # Add compensation hook
        hook = layers[self.compensation_layer].register_forward_hook(compensation_hook)
        self.hooks.append(hook)
        
        # Add skip hooks
        for layer_idx in self.skip_layers:
            hook = layers[layer_idx].register_forward_hook(skip_hook)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model with compensation and skipping."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the compensated model."""
        return self.model.generate(*args, **kwargs)
    
    @property
    def device(self):
        """Return the device of the underlying model."""
        return self.model.device
    
    @property
    def config(self):
        """Return the config of the underlying model."""
        return self.model.config
    
    @property
    def dtype(self):
        """Return the dtype of the underlying model."""
        return self.model.dtype
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        # Avoid infinite recursion for special attributes
        if name in ('model', 'tokenizer', 'skip_layers', 'compensation_vector', 
                    'compensation_layer', 'hooks'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.model, name)
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def load_compensated_skip_model(
    model_id: str,
    skip_layers: List[int],
    compensation_vector_file: Union[str, Path],
    device: str = "cuda",
    dtype: str = "bfloat16"
) -> tuple[CompensatedSkipLayerModel, AutoTokenizer]:
    """
    Load a model with compensation and layer skipping.
    
    Args:
        model_id: HuggingFace model ID
        skip_layers: List of layer indices to skip
        compensation_vector_file: Path to .pt file containing mean delta vector
        device: Device to load model on
        dtype: Model dtype
    
    Returns:
        Tuple of (compensated_model, tokenizer)
    """
    print(f"Loading compensated skip-layer model: {model_id}")
    print(f"Skip layers: {skip_layers}")
    print(f"Compensation vector: {compensation_vector_file}")
    
    # Setup device and dtype
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Handle dtype
    if dtype == "auto":
        # Auto-detect best dtype
        if device_obj.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype_obj = torch.bfloat16
        elif device_obj.type == "cuda":
            dtype_obj = torch.float16
        else:
            dtype_obj = torch.float32
    else:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        dtype_obj = dtype_map[dtype]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_obj,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    
    model.eval()
    
    # Load compensation vector
    compensation_vector = torch.load(compensation_vector_file, map_location=device_obj)
    print(f"Loaded compensation vector: shape {compensation_vector.shape}")
    
    # Create compensated model
    compensated_model = CompensatedSkipLayerModel(
        model=model,
        tokenizer=tokenizer,
        skip_layers=skip_layers,
        compensation_vector=compensation_vector
    )
    
    return compensated_model, tokenizer
