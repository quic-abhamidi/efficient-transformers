"""Compensation transform implementation."""

from __future__ import annotations

from typing import Any

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    CascadedCompensationConfig,
    CompensationSpec,
    LastTokenCompensationConfig,
    LearnableCompensationConfig,
    MagnitudePreservingCompensationConfig,
    MagnitudeRescalingCompensationConfig,
    MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
    MultiplicativeCompensationConfig,
    PcaCompensationConfig,
    PhaseAwareCompensationConfig,
    PhaseAwareLastTokenCompensationConfig,
    PhaseAwareMagnitudeRescalingCompensationConfig,
    PositionAwareCompensationConfig,
    ScaledCompensationConfig,
)
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class CompensationTransform(BaseTransform):
    """Apply a compensation strategy as reversible forward hooks."""

    kind = "compensation"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: CompensationSpec,
    ) -> AppliedTransformRecord:
        """Attach compensation forward hooks to the layer preceding the skipped layers.

        Reads the most recently applied ``skip_layers`` record from the artifact to
        determine which layer to compensate at (``min(skip_layers) - 1`` by default,
        or per-layer for multi-cluster configs).  Registers a cleanup callback so
        the hooks are removed when ``NASSession.close()`` is called.
        Raises ``ValueError`` if no prior skip_layers record exists.
        """
        adapter = resolve_layer_adapter(artifact.model)
        skip_layers = _resolve_skip_layers(artifact)
        if not skip_layers:
            raise ValueError(
                "compensation transform requires skip_layers to be applied earlier in the plan"
            )

        compensation_layers = _resolve_compensation_layers(spec, skip_layers)
        _validate_compensation_layers(
            compensation_layers=compensation_layers,
            skip_layers=skip_layers,
            num_layers=adapter.num_layers,
        )

        post_skip_layer: int | None = None
        if isinstance(spec.config, CascadedCompensationConfig):
            post_skip_layer = max(skip_layers) + 1
            if post_skip_layer >= adapter.num_layers:
                raise ValueError(
                    "cascaded compensation requires a post-skip layer within model bounds"
                )

        strategy = _build_strategy(spec)

        handles = []
        try:
            for layer_idx in compensation_layers:
                layer = adapter.container[layer_idx]
                handles.append(
                    layer.register_forward_hook(
                        _make_compensation_hook(strategy, layer_idx),
                    )
                )
            if post_skip_layer is not None:
                layer = adapter.container[post_skip_layer]
                handles.append(layer.register_forward_hook(_make_post_skip_hook(strategy)))
        except Exception:
            _remove_hook_handles(handles)
            raise

        register_model_cleanup(
            artifact.model,
            lambda model, hook_handles=handles: _remove_hook_handles(hook_handles),
        )

        details = {
            "strategy": spec.config.strategy,
            "skip_layers": list(skip_layers),
            "compensation_layers": list(compensation_layers),
        }
        if post_skip_layer is not None:
            details["post_skip_layer"] = post_skip_layer

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details=details,
        )


def _resolve_skip_layers(artifact: ModelArtifact) -> list[int]:
    for record in reversed(artifact.applied_transforms):
        if record.kind == "skip_layers" and record.status == "applied":
            raw_layers = record.details.get("layers", [])
            return [int(layer) for layer in raw_layers]
    return []


def _resolve_compensation_layers(
    spec: CompensationSpec,
    skip_layers: list[int],
) -> list[int]:
    if isinstance(
        spec.config,
        MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
    ):
        return [item.layer for item in spec.config.layer_ratios]
    return [min(skip_layers) - 1]


def _validate_compensation_layers(
    compensation_layers: list[int],
    skip_layers: list[int],
    num_layers: int,
) -> None:
    if not compensation_layers:
        raise ValueError("compensation_layers must contain at least one layer index")
    skip_set = set(skip_layers)
    for layer_idx in compensation_layers:
        if layer_idx < 0:
            raise ValueError(
                f"Compensation layer {layer_idx} is invalid "
                f"(derived from skip_layers={skip_layers}); "
                "skip_layers=[0,...] leaves no preceding layer to compensate into"
            )
        if layer_idx >= num_layers:
            raise ValueError(
                f"Compensation layer {layer_idx} is out of range for model with {num_layers} layers"
            )
        if layer_idx in skip_set:
            raise ValueError(
                f"Compensation layer {layer_idx} cannot also be a skipped layer"
            )


def _build_strategy(spec: CompensationSpec):
    from QEfficient.model_pruning.core.advanced_compensation import (
        CascadedCompensation,
        LastTokenCompensation,
        MagnitudePreservingCompensation,
        MagnitudeRescalingCompensation,
        MultiClusterPhaseAwareMagnitudeRescalingCompensation,
        MultiplicativeCompensation,
        PCACompensation,
        PhaseAwareCompensation,
        PhaseAwareLastTokenCompensation,
        PhaseAwareMagnitudeRescalingCompensation,
        PositionAwareCompensation,
        ScaledCompensation,
    )

    config = spec.config
    if isinstance(config, ScaledCompensationConfig):
        return ScaledCompensation(
            mean_delta=config.mean_delta_path,
            alpha=config.alpha,
        )
    if isinstance(config, LastTokenCompensationConfig):
        return LastTokenCompensation(
            mean_delta=config.mean_delta_path,
            alpha=config.alpha,
        )
    if isinstance(config, MagnitudePreservingCompensationConfig):
        return MagnitudePreservingCompensation(
            mean_delta=config.mean_delta_path,
            alpha=config.alpha,
        )
    if isinstance(config, CascadedCompensationConfig):
        return CascadedCompensation(
            mean_delta=config.mean_delta_path,
            pre_skip_fraction=config.pre_skip_fraction,
            alpha=config.alpha,
        )
    if isinstance(config, MagnitudeRescalingCompensationConfig):
        return MagnitudeRescalingCompensation(
            mean_delta=config.mean_delta_path,
            norm_ratio=config.norm_ratio,
            alpha=config.alpha,
        )
    if isinstance(config, PhaseAwareCompensationConfig):
        return PhaseAwareCompensation(
            prefill_delta=config.prefill_delta_path,
            decode_delta=config.decode_delta_path,
            prefill_alpha=config.prefill_alpha,
            decode_alpha=config.decode_alpha,
        )
    if isinstance(config, PhaseAwareLastTokenCompensationConfig):
        return PhaseAwareLastTokenCompensation(
            prefill_delta=config.prefill_delta_path,
            decode_delta=config.decode_delta_path,
            prefill_alpha=config.prefill_alpha,
            decode_alpha=config.decode_alpha,
        )
    if isinstance(config, PhaseAwareMagnitudeRescalingCompensationConfig):
        return PhaseAwareMagnitudeRescalingCompensation(
            prefill_norm_ratio=config.prefill_norm_ratio,
            decode_norm_ratio=config.decode_norm_ratio,
        )
    if isinstance(config, PositionAwareCompensationConfig):
        return PositionAwareCompensation(
            bucket_deltas=config.bucket_deltas_path,
            num_buckets=config.num_buckets,
            alpha=config.alpha,
            fallback_delta=config.fallback_delta_path,
        )
    if isinstance(config, PcaCompensationConfig):
        if config.mean_delta_path is not None:
            return PCACompensation.from_mean_delta_and_pca(
                mean_delta=config.mean_delta_path,
                pca_file=config.pca_path,
                n_components=config.n_components,
                alpha=config.alpha,
            )
        return PCACompensation(
            pca_file=config.pca_path,
            n_components=config.n_components,
            alpha=config.alpha,
        )
    if isinstance(config, MultiplicativeCompensationConfig):
        return MultiplicativeCompensation(
            scale_vector=config.scale_vector_path,
            bias_vector=config.bias_vector_path,
        )
    if isinstance(config, LearnableCompensationConfig):
        from QEfficient.model_pruning.core.learnable_compensation import LearnableCompensation

        return _DeviceAwareLearnableCompensation(
            LearnableCompensation.load(config.model_path, device="cpu")
        )
    if isinstance(
        config,
        MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
    ):
        return MultiClusterPhaseAwareMagnitudeRescalingCompensation(
            layer_ratios={
                item.layer: {
                    "prefill_norm_ratio": item.prefill_norm_ratio,
                    "decode_norm_ratio": item.decode_norm_ratio,
                }
                for item in config.layer_ratios
            },
            default_prefill_norm_ratio=config.default_prefill_norm_ratio,
            default_decode_norm_ratio=config.default_decode_norm_ratio,
        )
    raise ValueError(f"Unsupported compensation config type: {type(config)!r}")


def _make_compensation_hook(strategy, layer_idx: int):
    def compensation_hook(module, inputs, output):
        """Forward hook that applies the compensation strategy to this layer's output."""
        hidden_states, remainder = _extract_hidden_states(output)
        is_decode = hidden_states.shape[1] == 1
        if hasattr(strategy, "compensate_at_layer"):
            compensated = strategy.compensate_at_layer(
                hidden_states,
                layer_idx=layer_idx,
                is_decode=is_decode,
            )
        else:
            compensated = strategy.compensate(
                hidden_states,
                is_decode=is_decode,
            )
        return _rebuild_output(compensated, remainder)

    return compensation_hook


def _make_post_skip_hook(strategy):
    def post_skip_hook(module, inputs, output):
        """Post-skip forward hook for the cascaded strategy (fires after the skipped block)."""
        hidden_states, remainder = _extract_hidden_states(output)
        is_decode = hidden_states.shape[1] == 1
        compensated = strategy.compensate_post_skip(
            hidden_states,
            is_decode=is_decode,
        )
        return _rebuild_output(compensated, remainder)

    return post_skip_hook


def _extract_hidden_states(output: Any):
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Layer output tuple cannot be empty")
        return output[0], output[1:]
    return output, None


def _rebuild_output(hidden_states, remainder):
    if remainder is None:
        return hidden_states
    return (hidden_states,) + remainder


def _remove_hook_handles(handles) -> None:
    for handle in handles:
        handle.remove()


class _DeviceAwareLearnableCompensation:
    """Thin adapter that moves a learnable compensation module to the active device."""

    def __init__(self, module) -> None:
        self.module = module

    def compensate(self, h, is_decode: bool = False, token_positions=None):
        """Move the learnable module to the active device and run compensation."""
        del is_decode, token_positions
        parameter = next(self.module.parameters(), None)
        if parameter is not None and str(parameter.device) != str(h.device):
            self.module.to(h.device)
        return self.module.compensate(h)
