"""Skip-layer transform implementation."""

from __future__ import annotations

from types import MethodType
from typing import Any

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class SkipLayersTransform(BaseTransform):
    """Apply a no-op forward override to selected decoder layers."""

    kind = "skip_layers"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: SkipLayersSpec,
    ) -> AppliedTransformRecord:
        """Patch the forward method of each layer in ``spec.layers`` to a no-op.

        The original forward is stored on the layer as ``_nas_original_forward``
        and a cleanup callback is registered so ``NASSession.close()`` (or
        ``run_model_cleanup``) can restore it.  Raises ``ValueError`` if any
        layer index is out of range for the resolved model family.
        """
        adapter = resolve_layer_adapter(artifact.model)
        out_of_range = [layer for layer in spec.layers if layer >= adapter.num_layers]
        if out_of_range:
            raise ValueError(
                f"Requested skip layers out of range for model with {adapter.num_layers} layers: "
                f"{out_of_range}"
            )

        for layer_idx in spec.layers:
            layer = adapter.container[layer_idx]
            self._patch_layer(artifact.model, layer, adapter.model_type)

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details={
                "layers": list(spec.layers),
                "model_family": adapter.model_type,
            },
        )

    def _patch_layer(self, root_model, layer, model_type: str) -> None:
        if not hasattr(layer, "_nas_original_forward"):
            layer._nas_original_forward = layer.forward
            register_model_cleanup(root_model, lambda model, target=layer: _restore_layer_forward(target))

        def _skip_forward(self, hidden_states, *args, **kwargs):
            return _skip_layer_output(hidden_states, kwargs, model_type)

        layer.forward = MethodType(_skip_forward, layer)


def _skip_layer_output(hidden_states: Any, kwargs: dict[str, Any], model_type: str) -> Any:
    """Return a no-op decoder-layer output for the resolved model family.

    QEff's Llama/Mistral/Qwen decoder loops assign ``hidden_states =
    decoder_layer(...)`` directly, so those families must return the tensor.
    Gemma3 consumes ``layer_outputs[0]``, so it keeps the tuple-style decoder
    contract.
    """
    if model_type in {"gemma3", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}:
        output = [hidden_states]
        if bool(kwargs.get("output_attentions", False)):
            output.append(None)
        return tuple(output)
    return hidden_states


def _restore_layer_forward(layer) -> None:
    original = getattr(layer, "_nas_original_forward", None)
    if original is not None:
        layer.forward = original
        delattr(layer, "_nas_original_forward")
