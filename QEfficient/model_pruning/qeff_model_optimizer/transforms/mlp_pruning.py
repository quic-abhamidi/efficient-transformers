"""MLP width pruning transform -- mask least-active intermediate channels."""

from __future__ import annotations

import torch

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import MlpPruningSpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class MlpPruningTransform(BaseTransform):
    """Zero out the least-important intermediate MLP channels via forward pre-hooks."""

    kind = "mlp_pruning"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: MlpPruningSpec,
    ) -> AppliedTransformRecord:
        """Attach pre-hooks on ``down_proj`` that zero pruned intermediate channels.

        Target layers default to *all* layers when ``spec.target_layers`` is
        empty.  Channel importance is read from the artifact's
        ``capability_report["channel_importance"]`` when available; otherwise a
        lightweight weight-norm heuristic is used as fallback (no calibration
        data required).

        Cleanup is registered so ``run_model_cleanup`` removes all hooks.
        """
        adapter = resolve_layer_adapter(artifact.model)
        num_layers = adapter.num_layers

        # Determine target layers
        if spec.target_layers:
            target_layers = list(spec.target_layers)
            out_of_range = [idx for idx in target_layers if idx >= num_layers]
            if out_of_range:
                raise ValueError(
                    f"Requested MLP pruning layers out of range for model with "
                    f"{num_layers} layers: {out_of_range}"
                )
        else:
            target_layers = list(range(num_layers))

        # Resolve optional pre-computed importance report
        importance_report = artifact.capability_report.get("channel_importance")

        handles = []
        pruned_info: dict[int, list[int]] = {}

        try:
            for layer_idx in target_layers:
                anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
                if anatomy.gate_proj is None or anatomy.down_proj is None:
                    continue
                intermediate_size = anatomy.intermediate_size

                # Determine per-channel importance scores
                scores = _get_channel_scores(
                    anatomy, layer_idx, intermediate_size, importance_report,
                )

                # Number of channels to prune
                num_prune = int(spec.pruning_ratio * intermediate_size)
                if num_prune == 0:
                    continue

                # Pick the weakest channels (lowest scores)
                ranked = sorted(range(intermediate_size), key=lambda c: scores[c])
                pruned_channels = sorted(ranked[:num_prune])
                pruned_info[layer_idx] = pruned_channels

                # Register a forward pre-hook on down_proj
                handle = anatomy.down_proj.register_forward_pre_hook(
                    _make_mlp_prune_hook(pruned_channels),
                )
                handles.append(handle)
        except Exception:
            _remove_hook_handles(handles)
            raise

        register_model_cleanup(
            artifact.model,
            lambda model, hook_handles=handles: _remove_hook_handles(hook_handles),
        )

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details={
                "target_layers": target_layers,
                "pruning_ratio": spec.pruning_ratio,
                "metric": spec.metric,
                "pruned_channels_per_layer": {
                    str(k): v for k, v in pruned_info.items()
                },
            },
        )


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def _make_mlp_prune_hook(pruned_channels: list[int]):
    """Return a forward pre-hook that zeros ``pruned_channels`` in the input."""

    def hook(module, args):
        x = args[0]
        # x shape: [B, S, intermediate_size]
        x = x.clone()
        x[..., pruned_channels] = 0.0
        return (x,) + args[1:]

    return hook


# ---------------------------------------------------------------------------
# Importance scoring helpers
# ---------------------------------------------------------------------------


def _get_channel_scores(
    anatomy,
    layer_idx: int,
    intermediate_size: int,
    importance_report,
) -> list[float]:
    """Return per-channel importance scores for a single layer.

    If an importance report is available and contains scores for this layer,
    those scores are used directly.  Otherwise falls back to a weight-norm
    heuristic that requires no calibration data.
    """
    if importance_report is not None:
        per_layer = getattr(importance_report, "per_layer_scores", None)
        if per_layer is None and isinstance(importance_report, dict):
            per_layer = importance_report.get("per_layer_scores")
        if per_layer is not None and layer_idx in per_layer:
            scores = per_layer[layer_idx]
            if len(scores) == intermediate_size:
                return scores

    return _compute_weight_importance(anatomy, intermediate_size)


def _compute_weight_importance(anatomy, intermediate_size: int) -> list[float]:
    """Rank channels by gate_proj weight column L2 norms.

    For SwiGLU models the gate and up projections are multiplied element-wise,
    so the importance of a channel depends on both.  The heuristic multiplies
    the column norms of gate_proj and up_proj (when present).
    """
    gate_w = anatomy.gate_proj.weight.data  # [intermediate, hidden]
    norms = gate_w.norm(dim=1)  # [intermediate]

    # If up_proj exists (SwiGLU), also consider it
    if anatomy.up_proj is not None:
        up_w = anatomy.up_proj.weight.data
        norms = norms * up_w.norm(dim=1)

    scores = norms.cpu().tolist()
    if len(scores) != intermediate_size:
        raise ValueError(
            f"Weight importance produced {len(scores)} scores but "
            f"intermediate_size is {intermediate_size}"
        )
    return scores


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------


def _remove_hook_handles(handles) -> None:
    for handle in handles:
        handle.remove()
