"""Sub-module anatomy resolver for individual decoder layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import _get_nested_attr, resolve_layer_adapter, unwrap_model


@dataclass(frozen=True)
class LayerAnatomy:
    """Resolved sub-modules and structure for one decoder layer."""

    layer_module: nn.Module

    # Attention sub-modules
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    # MLP sub-modules (None for MoE layers)
    gate_proj: Optional[nn.Linear]
    up_proj: Optional[nn.Linear]
    down_proj: Optional[nn.Linear]

    # Structure from config
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int


# ---------------------------------------------------------------------------
# Convention-based probing tables
# ---------------------------------------------------------------------------

_ATTN_CONVENTIONS = (
    # (prefix, q, k, v, o)
    ("self_attn", "q_proj", "k_proj", "v_proj", "o_proj"),
    ("attention", "q_proj", "k_proj", "v_proj", "o_proj"),
    ("attn", "q_proj", "k_proj", "v_proj", "o_proj"),
)

# Each entry: (gate_path, up_path_or_None, down_path)
# Paths are tuples relative to the layer root.
_MLP_CONVENTIONS = (
    (("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")),
    (("mlp", "fc1"), None, ("mlp", "fc2")),
    (("feed_forward", "w1"), ("feed_forward", "w3"), ("feed_forward", "w2")),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _probe_attention(layer: nn.Module):
    """Try convention-based attention discovery.  Returns (q, k, v, o) or None."""
    for prefix, q_name, k_name, v_name, o_name in _ATTN_CONVENTIONS:
        q = _get_nested_attr(layer, (prefix, q_name))
        k = _get_nested_attr(layer, (prefix, k_name))
        v = _get_nested_attr(layer, (prefix, v_name))
        o = _get_nested_attr(layer, (prefix, o_name))
        if all(isinstance(m, nn.Linear) for m in (q, k, v, o)):
            return q, k, v, o
    return None


def _probe_mlp(layer: nn.Module):
    """Try convention-based MLP discovery.  Returns (gate, up_or_None, down) or None."""
    for gate_path, up_path, down_path in _MLP_CONVENTIONS:
        gate = _get_nested_attr(layer, gate_path)
        down = _get_nested_attr(layer, down_path)
        if not (isinstance(gate, nn.Linear) and isinstance(down, nn.Linear)):
            continue
        if up_path is not None:
            up = _get_nested_attr(layer, up_path)
            if not isinstance(up, nn.Linear):
                continue
        else:
            up = None
        return gate, up, down
    return None


def _shape_based_attention(
    layer: nn.Module, hidden_size: int, num_kv_heads: int, head_dim: int
):
    """Fallback: walk named_modules and match Linear layers by shape.

    Raises ValueError when q_proj and o_proj are ambiguous (same shape)
    and cannot be distinguished by module name heuristics.
    """
    kv_size = num_kv_heads * head_dim
    q_size = hidden_size  # num_heads * head_dim for non-GQA; may differ from hidden_size
    candidates_qo: list[tuple[str, nn.Module]] = []
    k = v = None
    for name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        out_feat = mod.out_features
        if out_feat == kv_size and mod.in_features == hidden_size:
            if k is None:
                k = mod
            elif v is None:
                v = mod
        elif out_feat == hidden_size and mod.in_features == hidden_size:
            candidates_qo.append((name, mod))

    if k is None or v is None or len(candidates_qo) < 2:
        return None

    q = o = None
    for name, mod in candidates_qo:
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf in ("q_proj", "query", "q"):
            q = mod
        elif leaf in ("o_proj", "out_proj", "o", "dense"):
            o = mod

    if q is None or o is None:
        raise ValueError(
            "Shape-based attention discovery found "
            f"{len(candidates_qo)} modules with shape [{hidden_size}, {hidden_size}] "
            "but could not determine which is q_proj vs o_proj. "
            "Use a model with standard HF naming (self_attn.q_proj, self_attn.o_proj) "
            "or add the model family to the convention probing table."
        )

    return q, k, v, o


def _shape_based_mlp(layer: nn.Module, intermediate_size: int):
    """Fallback: walk named_modules and match Linear layers by shape for MLP."""
    gate = up = down = None
    for _name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.out_features == intermediate_size and gate is None:
            gate = mod
        elif mod.out_features == intermediate_size and up is None:
            up = mod
        elif mod.in_features == intermediate_size and down is None:
            down = mod
    if gate is not None and down is not None:
        return gate, up, down
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_layer_anatomy(model, layer_idx: int) -> LayerAnatomy:
    """Resolve the sub-module anatomy for a single decoder layer.

    Parameters
    ----------
    model:
        A HuggingFace-style causal language model (possibly wrapped).
    layer_idx:
        Zero-based index of the decoder layer to inspect.

    Returns
    -------
    LayerAnatomy
        Frozen dataclass with references to the layer's sub-modules and
        structural parameters read from the model config.

    Raises
    ------
    ValueError
        If the layer index is out of range or if the sub-modules cannot be
        resolved via convention probing or shape-based fallback.
    """
    adapter = resolve_layer_adapter(model)

    if layer_idx < 0 or layer_idx >= adapter.num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range for model with "
            f"{adapter.num_layers} layers (valid: 0..{adapter.num_layers - 1})"
        )

    layer = adapter.container[layer_idx]

    # ---- Read structural parameters from config ----
    base = unwrap_model(model)
    config = getattr(base, "config", None)
    if config is None:
        raise ValueError("Model has no config attribute; cannot read structure.")

    num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("Config missing 'num_attention_heads'.")

    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Config missing 'hidden_size'.")

    intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        raise ValueError("Config missing 'intermediate_size'.")

    head_dim = getattr(config, "head_dim", hidden_size // num_heads)

    # ---- Attention sub-modules ----
    attn_result = _probe_attention(layer)
    if attn_result is None:
        attn_result = _shape_based_attention(layer, hidden_size, num_kv_heads, head_dim)
    if attn_result is None:
        raise ValueError(
            f"Could not resolve attention sub-modules for layer {layer_idx}. "
            f"Tried convention prefixes: {[c[0] for c in _ATTN_CONVENTIONS]} "
            f"and shape-based fallback (hidden_size={hidden_size}, "
            f"kv_size={num_kv_heads * head_dim})."
        )
    q_proj, k_proj, v_proj, o_proj = attn_result

    # ---- MLP sub-modules ----
    mlp_result = _probe_mlp(layer)
    if mlp_result is None:
        mlp_result = _shape_based_mlp(layer, intermediate_size)
    if mlp_result is None:
        mlp_attr = getattr(layer, "mlp", None)
        if mlp_attr is not None and (
            hasattr(mlp_attr, "experts") or hasattr(mlp_attr, "sparse_moe")
        ):
            gate_proj, up_proj, down_proj = None, None, None
        else:
            raise ValueError(
                f"Could not resolve MLP sub-modules for layer {layer_idx}. "
                f"Tried convention paths: {[c[0] for c in _MLP_CONVENTIONS]} "
                f"and shape-based fallback (intermediate_size={intermediate_size})."
            )
    else:
        gate_proj, up_proj, down_proj = mlp_result

    return LayerAnatomy(
        layer_module=layer,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )
