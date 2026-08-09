"""Head pruning transform -- mask least-important attention heads."""

from __future__ import annotations

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import HeadPruningSpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class HeadPruningTransform(BaseTransform):
    """Apply attention-head masking via forward pre-hooks on o_proj."""

    kind = "head_pruning"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: HeadPruningSpec,
    ) -> AppliedTransformRecord:
        """Register pre-hooks on each selected layer's o_proj to zero pruned heads.

        For every ``LayerHeadSelection`` in *spec.selections*, this method:

        1. Resolves the :class:`LayerAnatomy` for the given layer index.
        2. Validates that all requested head indices are within
           ``[0, num_heads)``.
        3. Registers a ``register_forward_pre_hook`` on ``anatomy.o_proj``
           that zeros the ``[h*head_dim : (h+1)*head_dim]`` slice of the
           input tensor for each pruned head *h*.

        All hook handles are collected and a cleanup callback is registered
        so that ``NASSession.close()`` (or ``run_model_cleanup``) removes
        them, restoring the original model behaviour.

        Parameters
        ----------
        artifact : ModelArtifact
            The loaded model artifact to modify in-place.
        spec : HeadPruningSpec
            Which layers and heads to prune.

        Returns
        -------
        AppliedTransformRecord
            A record with ``kind="head_pruning"`` and ``status="applied"``.

        Raises
        ------
        ValueError
            If any head index is out of range for the resolved layer's
            ``num_heads``.
        """
        handles = []
        pruned_summary: dict[str, object] = {}

        try:
            for selection in spec.selections:
                anatomy = resolve_layer_anatomy(artifact.model, selection.layer)
                num_heads = anatomy.num_heads
                head_dim = anatomy.head_dim

                # Validate head indices are in range.
                out_of_range = [h for h in selection.heads if h >= num_heads]
                if out_of_range:
                    raise ValueError(
                        f"Head indices {out_of_range} out of range for layer "
                        f"{selection.layer} with {num_heads} heads "
                        f"(valid: 0..{num_heads - 1})"
                    )

                hook = _make_head_zeroing_hook(selection.heads, head_dim)
                handle = anatomy.o_proj.register_forward_pre_hook(hook)
                handles.append(handle)

                pruned_summary[str(selection.layer)] = list(selection.heads)

        except Exception:
            _remove_handles(handles)
            raise

        register_model_cleanup(
            artifact.model,
            lambda model, hook_handles=handles: _remove_handles(hook_handles),
        )

        return AppliedTransformRecord(
            kind=self.kind,
            status="applied",
            details={
                "mode": spec.mode,
                "pruned_heads": pruned_summary,
            },
        )


def _make_head_zeroing_hook(heads: list[int], head_dim: int):
    """Return a forward-pre-hook that zeros the specified head slices."""

    def hook(module, args):
        inp = args[0]
        modified = inp.clone()
        for h in heads:
            start = h * head_dim
            end = start + head_dim
            modified[..., start:end] = 0.0
        if len(args) > 1:
            return (modified,) + args[1:]
        return (modified,)

    return hook


def _remove_handles(handles) -> None:
    for handle in handles:
        handle.remove()
