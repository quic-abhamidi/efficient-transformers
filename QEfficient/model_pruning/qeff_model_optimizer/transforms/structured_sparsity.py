"""2:4 structured sparsity -- enforce hardware-friendly weight sparsity pattern."""

from __future__ import annotations

import torch

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import StructuredSparsitySpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class StructuredSparsityTransform(BaseTransform):
    """Apply 2:4 structured sparsity to selected weight matrices."""

    kind = "structured_sparsity"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: StructuredSparsitySpec,
    ) -> AppliedTransformRecord:
        """Zero the 2 smallest-magnitude elements in every group of 4 along
        the input dimension of each target module's weight tensor.

        Original weights are saved on CPU so they can be restored by the
        cleanup callback registered with :func:`register_model_cleanup`.
        """
        adapter = resolve_layer_adapter(artifact.model)

        # 1. Determine target layers
        if spec.target_layers:
            target_layers = list(spec.target_layers)
            out_of_range = [l for l in target_layers if l >= adapter.num_layers]
            if out_of_range:
                raise ValueError(
                    f"Requested target layers out of range for model with "
                    f"{adapter.num_layers} layers: {out_of_range}"
                )
        else:
            target_layers = list(range(adapter.num_layers))

        saved: list[tuple[torch.nn.Module, torch.Tensor]] = []
        sparsified_modules: list[str] = []

        for layer_idx in target_layers:
            anatomy = resolve_layer_anatomy(artifact.model, layer_idx)

            for module_name in spec.target_modules:
                module = getattr(anatomy, module_name, None)
                if module is None:
                    continue
                if not hasattr(module, "weight"):
                    continue

                # Save original weight on CPU
                original = module.weight.data.clone().cpu()
                saved.append((module, original))

                # Apply 2:4 mask in-place
                _apply_2_4_mask(module.weight)
                sparsified_modules.append(f"layer_{layer_idx}.{module_name}")

        register_model_cleanup(
            artifact.model,
            lambda model, saved_weights=saved: _restore_weights(saved_weights),
        )

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details={
                "target_layers": target_layers,
                "pattern": spec.pattern,
                "target_modules": list(spec.target_modules),
                "sparsified_modules": sparsified_modules,
            },
        )


@torch.no_grad()
def _apply_2_4_mask(weight: torch.Tensor) -> None:
    """Apply 2:4 structured sparsity in-place.

    For every contiguous group of 4 elements along the last dimension of the
    weight (shape ``[out_features, in_features]``), the 2 smallest-magnitude
    elements are zeroed.  When ``in_features`` is not a multiple of 4, the
    trailing partial group is left unchanged.
    """
    data = weight.data
    out_f, in_f = data.shape
    usable = (in_f // 4) * 4

    if usable == 0:
        return

    # Reshape the usable portion into groups of 4
    flat = data[:, :usable].reshape(-1, 4)

    # For each group of 4, find the indices of the 2 smallest by abs magnitude
    abs_vals = flat.abs()
    _, indices = abs_vals.topk(2, dim=-1, largest=False)  # 2 smallest

    # Create mask: 1 for kept, 0 for pruned
    mask = torch.ones_like(flat)
    mask.scatter_(1, indices, 0.0)
    flat.mul_(mask)

    # Write back
    data[:, :usable] = flat.reshape(out_f, usable)


def _restore_weights(saved: list[tuple[torch.nn.Module, torch.Tensor]]) -> None:
    """Restore original weights from CPU copies."""
    for module, original in saved:
        module.weight.data.copy_(original.to(module.weight.device))
