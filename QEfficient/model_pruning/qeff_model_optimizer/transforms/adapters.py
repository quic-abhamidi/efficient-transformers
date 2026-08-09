"""Model-family adapter helpers for transform implementations."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "gemma3"}


@dataclass(frozen=True)
class LayerContainerAdapter:
    """Resolved layer container for a supported model."""

    model_type: str
    container: object

    @property
    def num_layers(self) -> int:
        """Number of decoder blocks in this container."""
        return len(self.container)


def unwrap_model(model):
    """Unwrap simple runtime/training wrappers to the base model."""
    current = model
    visited = set()
    while True:
        marker = id(current)
        if marker in visited:
            return current
        visited.add(marker)

        if hasattr(current, "get_base_model"):
            candidate = current.get_base_model()
            if candidate is not None and candidate is not current:
                current = candidate
                continue

        if hasattr(current, "base_model") and current.base_model is not current:
            current = current.base_model
            continue

        return current


def resolve_layer_adapter(model) -> LayerContainerAdapter:
    """Resolve the decoder/block container for supported model families."""
    base_model = unwrap_model(model)
    config = getattr(base_model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type for v1 skip transform: {model_type!r}"
        )

    candidates = (
        ("layers",),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "language_model", "layers"),
        ("language_model", "layers"),
        ("h",),
        ("transformer", "h"),
        ("model", "transformer", "h"),
    )
    for path in candidates:
        container = _get_nested_attr(base_model, path)
        if container is not None:
            return LayerContainerAdapter(model_type=model_type, container=container)

    raise ValueError(
        f"Could not locate decoder layers for supported model type {model_type!r}"
    )


def _get_nested_attr(root, path):
    current = root
    for name in path:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current
