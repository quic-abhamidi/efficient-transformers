#!/usr/bin/env python3
"""
Custom Model Wrapper for Layer Skipping
Allows lm_eval to load models with pre-configured skip_layers
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple
import os
import json


def _get_tokenizer_kwargs(model_name: str, trust_remote_code: bool) -> dict:
    kwargs = {"trust_remote_code": trust_remote_code}
    if "mistral" in model_name.lower():
        kwargs["fix_mistral_regex"] = True
    return kwargs


def _apply_skip_layers_to_config(config, skip_layers: Optional[List[int]]) -> None:
    if not skip_layers:
        return

    normalized_layers = set(skip_layers)
    if hasattr(config, "skip_layers"):
        config.skip_layers = normalized_layers

    # Gemma3 keeps the decoder stack config under text_config.
    if hasattr(config, "text_config") and hasattr(config.text_config, "skip_layers"):
        config.text_config.skip_layers = normalized_layers


class SkipLayerModelLoader:
    """
    Custom model loader that configures skip_layers before model initialization.
    This ensures the layer skipping configuration is properly set.
    """
    
    @staticmethod
    def load_model_with_skip_layers(
        model_name: str,
        skip_layers: Optional[List[int]] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True
    ) -> Tuple:
        """
        Load a model with skip_layers configuration.
        
        Args:
            model_name: HuggingFace model identifier
            skip_layers: List of layer indices to skip
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Tuple of (model, tokenizer, config)
        """
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        if skip_layers:
            print(f"Skip layers configured: {skip_layers}")
        else:
            print(f"No layer skipping (baseline)")
        print(f"{'='*60}\n")
        
        # Load config
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Set skip_layers in config
        if skip_layers:
            _apply_skip_layers_to_config(config, skip_layers)
            configured = getattr(config, "skip_layers", None)
            if configured is None and hasattr(config, "text_config"):
                configured = getattr(config.text_config, "skip_layers", None)
            print(f"✓ Config updated with skip_layers: {configured}")
        
        # Load model with modified config
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=trust_remote_code,
            device_map=device_map
        )
        
        # Verify skip_layers is set
        if hasattr(model.config, 'skip_layers') and model.config.skip_layers:
            print(f"✓ Model loaded with skip_layers: {model.config.skip_layers}")
        else:
            print(f"✓ Model loaded without layer skipping (baseline)")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **_get_tokenizer_kwargs(model_name, trust_remote_code)
        )
        
        return model, tokenizer, config


def create_temp_model_config(model_name: str, skip_layers: Optional[List[int]], temp_dir: Optional[str] = None):
    """
    Create a temporary config file with skip_layers for lm_eval to use.
    This is an alternative approach if direct model loading doesn't work.
    
    Args:
        model_name: HuggingFace model identifier
        skip_layers: List of layer indices to skip
        temp_dir: Directory to save temporary config
        
    Returns:
        Path to temporary config directory
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _apply_skip_layers_to_config(config, skip_layers)
    
    # Create unique temp directory
    import tempfile
    temp_config_dir = tempfile.mkdtemp(prefix="model_config_", dir=temp_dir)
    
    # Save config
    config.save_pretrained(temp_config_dir)
    
    return temp_config_dir
