"""Linear attention transform -- replace softmax attention with kernel-based linear attention."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import LinearAttentionSpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import _ATTN_CONVENTIONS, resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup

SUPPORTED_KERNELS = {"elu", "relu", "cosine"}

_KERNEL_MAP = {
    "elu": lambda x: F.elu(x) + 1.0,
    "relu": lambda x: F.relu(x),
    "cosine": lambda x: torch.cat([F.relu(x), F.relu(-x)], dim=-1),
}


def _find_attn_module(layer_module):
    """Find the parent attention sub-module within a decoder layer."""
    for prefix, *_ in _ATTN_CONVENTIONS:
        attn = getattr(layer_module, prefix, None)
        if attn is not None:
            return attn
    return None


def _apply_rope_if_available(attn_module, q, k, position_ids, seq_len):
    """Apply rotary position embeddings if available, otherwise return unchanged."""
    rotary_emb = getattr(attn_module, "rotary_emb", None)
    if rotary_emb is None:
        return q, k

    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        return q, k

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)

    cos, sin = rotary_emb(k, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k


def _make_linear_attn_forward(attn_module, anatomy, kernel_fn, mode, doubles_dim):
    """Build the replacement forward function for a self-attention module."""
    q_proj = anatomy.q_proj
    k_proj = anatomy.k_proj
    v_proj = anatomy.v_proj
    o_proj = anatomy.o_proj
    num_heads = anatomy.num_heads
    num_kv_heads = anatomy.num_kv_heads
    head_dim = anatomy.head_dim
    groups = num_heads // num_kv_heads

    def forward(self, hidden_states, *args, **kwargs):
        B, N, _ = hidden_states.shape

        if mode == "decode_only" and N > 1:
            return self._nas_original_forward(hidden_states, *args, **kwargs)

        q = q_proj(hidden_states).view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k_proj(hidden_states).view(B, N, num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj(hidden_states).view(B, N, num_kv_heads, head_dim).transpose(1, 2)

        position_ids = kwargs.get("position_ids", None)
        q, k = _apply_rope_if_available(self, q, k, position_ids, N)

        if num_kv_heads < num_heads:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        q_prime = kernel_fn(q)
        k_prime = kernel_fn(k)

        # Linear attention: φ(Q)(φ(K)^T V) with normalization
        kv = torch.einsum("bhnd,bhnv->bhdv", k_prime, v)
        numerator = torch.einsum("bhnd,bhdv->bhnv", q_prime, kv)
        denom = torch.einsum("bhnd,bhd->bhn", q_prime, k_prime.sum(dim=2))
        denom = denom.clamp(min=1e-6).unsqueeze(-1)
        attn_out = numerator / denom

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, num_heads * head_dim)
        output = o_proj(attn_out)
        return (output, None)

    return forward


class LinearAttentionTransform(BaseTransform):
    """Replace softmax attention with kernel-based linear attention."""

    kind = "linear_attention"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: LinearAttentionSpec,
    ) -> AppliedTransformRecord:
        if spec.implementation not in SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel {spec.implementation!r}; "
                f"supported: {sorted(SUPPORTED_KERNELS)}"
            )

        kernel_fn = _KERNEL_MAP[spec.implementation]
        doubles_dim = spec.implementation == "cosine"

        adapter = resolve_layer_adapter(artifact.model)
        if spec.apply_to_all:
            target_layers = list(range(adapter.num_layers))
        else:
            target_layers = list(spec.target_layers)
            out_of_range = [i for i in target_layers if i >= adapter.num_layers]
            if out_of_range:
                raise ValueError(
                    f"Layer indices {out_of_range} out of range for model with "
                    f"{adapter.num_layers} layers"
                )

        patched = []
        try:
            for layer_idx in target_layers:
                anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
                attn_module = _find_attn_module(anatomy.layer_module)
                if attn_module is None:
                    raise ValueError(
                        f"Could not find attention module in layer {layer_idx}"
                    )

                if not hasattr(attn_module, "_nas_original_forward"):
                    attn_module._nas_original_forward = attn_module.forward

                replacement = _make_linear_attn_forward(
                    attn_module, anatomy, kernel_fn, spec.mode, doubles_dim,
                )
                attn_module.forward = MethodType(replacement, attn_module)
                patched.append(attn_module)

        except Exception:
            _cleanup_all(patched)
            raise

        register_model_cleanup(
            artifact.model,
            lambda model, modules=patched: _cleanup_all(modules),
        )

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details={
                "target_layers": target_layers,
                "implementation": spec.implementation,
                "mode": spec.mode,
                "model_family": adapter.model_type,
            },
        )


def _restore_attn_forward(attn_module) -> None:
    original = getattr(attn_module, "_nas_original_forward", None)
    if original is not None:
        attn_module.forward = original
        delattr(attn_module, "_nas_original_forward")


def _cleanup_all(modules) -> None:
    for mod in modules:
        _restore_attn_forward(mod)
