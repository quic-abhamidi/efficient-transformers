# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import nn

from QEfficient.pruning.config import LayerSkipCompensationConfig, LayerSkipConfig
from QEfficient.pruning.residual_patch import PatchedDecoderLayer, load_linear_residual_patch


@dataclass(frozen=True)
class LayerContainer:
    model_type: str | None
    container: nn.ModuleList

    @property
    def num_layers(self) -> int:
        return len(self.container)


class SkippedDecoderLayer(nn.Module):
    """Export-safe no-op replacement for a decoder layer.

    The wrapper intentionally keeps the original layer as a child module so
    debug/state-dict paths remain inspectable, but its forward path is a pure
    pass-through.  This preserves decoder layer count and cache slot layout for
    ONNX/QAIC while removing the selected layer's compute from the exported graph.
    """

    def __init__(self, original_layer: nn.Module, layer_idx: int, returns_tuple: bool):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.returns_tuple = returns_tuple
        self.qeff_layer_skip_enabled = True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if not self.returns_tuple:
            return hidden_states

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (None,)
        if kwargs.get("use_cache", False):
            outputs += (kwargs.get("past_key_value", kwargs.get("past_key_values")),)
        return outputs


def resolve_layer_container(model: nn.Module) -> LayerContainer:
    model_config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    model_type = getattr(model_config, "model_type", None)

    candidates = (
        ("layers",),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("text_model", "layers"),
        ("model", "text_model", "layers"),
        ("h",),
        ("transformer", "h"),
        ("model", "transformer", "h"),
    )
    for path in candidates:
        container = _get_nested_attr(model, path)
        if isinstance(container, nn.ModuleList):
            return LayerContainer(model_type=model_type, container=container)
        if container is not None and hasattr(container, "__len__") and hasattr(container, "__setitem__"):
            return LayerContainer(model_type=model_type, container=container)

    raise ValueError(f"Could not locate decoder layers for pruning on model type {model_type!r}.")


def apply_layer_skip(
    model: nn.Module,
    config: LayerSkipConfig,
    compensation: LayerSkipCompensationConfig | None = None,
) -> tuple[nn.Module, bool]:
    adapter = resolve_layer_container(model)
    out_of_range = [layer for layer in config.layers if layer >= adapter.num_layers]
    if out_of_range:
        raise ValueError(
            f"Requested skip layers out of range for model with {adapter.num_layers} layers: {out_of_range}"
        )

    transformed = False
    requested_layers = set(config.layers)
    injection_layer = _resolve_injection_layer(config, compensation, adapter.num_layers)

    for layer_idx, current_layer in enumerate(adapter.container):
        unwrapped_layer = _unwrap_patch(current_layer)
        if unwrapped_layer is not current_layer and layer_idx != injection_layer:
            adapter.container[layer_idx] = unwrapped_layer
            current_layer = unwrapped_layer
            transformed = True

        if isinstance(current_layer, SkippedDecoderLayer) and layer_idx not in requested_layers:
            adapter.container[layer_idx] = current_layer.original_layer
            transformed = True

    for layer_idx in config.layers:
        current_layer = _unwrap_patch(adapter.container[layer_idx])
        if isinstance(current_layer, SkippedDecoderLayer):
            current_layer.layer_idx = layer_idx
            current_layer.returns_tuple = _returns_tuple(current_layer.original_layer)
            adapter.container[layer_idx] = current_layer
            transformed = True
            continue
        adapter.container[layer_idx] = SkippedDecoderLayer(
            original_layer=current_layer,
            layer_idx=layer_idx,
            returns_tuple=_returns_tuple(current_layer),
        )
        transformed = True

    if compensation is not None:
        if injection_layer is None:
            raise ValueError("Layer skip compensation needs a surviving layer after the skipped layers.")
        if injection_layer in requested_layers:
            raise ValueError(f"Layer skip compensation injection layer {injection_layer} cannot be skipped.")
        target_layer = _unwrap_patch(adapter.container[injection_layer])
        hidden_size = _resolve_hidden_size(model, target_layer)
        device, dtype = _resolve_module_device_dtype(target_layer)
        patch = load_linear_residual_patch(
            compensation.patch_weights,
            hidden_size,
            alpha=compensation.alpha,
            device=device,
            dtype=dtype,
        )
        adapter.container[injection_layer] = PatchedDecoderLayer(
            patch=patch,
            original_layer=target_layer,
            injection_layer=injection_layer,
        )
        transformed = True

    return model, transformed


def _resolve_injection_layer(
    config: LayerSkipConfig, compensation: LayerSkipCompensationConfig | None, num_layers: int
) -> int | None:
    if compensation is None:
        return None
    if compensation.injection_layer is not None:
        if compensation.injection_layer >= num_layers:
            raise ValueError(
                f"Layer skip compensation injection layer {compensation.injection_layer} out of range "
                f"for model with {num_layers} layers."
            )
        return compensation.injection_layer

    injection_layer = max(config.layers) + 1
    if injection_layer >= num_layers:
        return None
    return injection_layer


def _unwrap_patch(layer: nn.Module) -> nn.Module:
    if isinstance(layer, PatchedDecoderLayer):
        return layer.original_layer
    return layer


def _resolve_hidden_size(model: nn.Module, target_layer: nn.Module) -> int:
    config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    text_config = getattr(config, "text_config", None)
    for candidate in (text_config, config, getattr(target_layer, "config", None)):
        hidden_size = getattr(candidate, "hidden_size", None) if candidate is not None else None
        if hidden_size is not None:
            return int(hidden_size)
    for parameter in target_layer.parameters():
        if parameter.ndim >= 1:
            return int(parameter.shape[-1])
    raise ValueError("Could not infer hidden_size for layer skip compensation patch.")


def _resolve_module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return None, None


def _returns_tuple(layer: nn.Module) -> bool:
    try:
        parameters = inspect.signature(layer.forward).parameters
    except (TypeError, ValueError):
        return False
    return "output_attentions" in parameters or "output_router_logits" in parameters


def _get_nested_attr(root: object, path: tuple[str, ...]):
    current = root
    for name in path:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current
