"""KV cache compression -- simulate KV head merging via hooks."""

from __future__ import annotations

import torch

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import KvCacheCompressionSpec
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.base import BaseTransform
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import register_model_cleanup


class KvCacheCompressionTransform(BaseTransform):
    """Simulate KV head merging by averaging similar KV heads at runtime."""

    kind = "kv_cache_compression"

    def apply(
        self,
        artifact: ModelArtifact,
        spec: KvCacheCompressionSpec,
    ) -> AppliedTransformRecord:
        """Attach forward hooks to K/V projections that merge similar heads.

        Steps:
        1. Determine target layers (empty ``target_layers`` means all layers).
        2. Optionally read a pre-computed ``KvSimilarityReport`` from the
           artifact metadata; otherwise compute pairwise cosine similarity on
           projection weights inline.
        3. Validate the GQA constraint (``num_kv_heads < num_heads`` unless
           ``spec.allow_mha_to_gqa`` is set).
        4. For each target layer, register hooks on k_proj and v_proj that
           average the most similar head pairs.
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

        # 2. Get or compute similarity report
        kv_report = artifact.capability_report.get("kv_similarity")
        if kv_report is not None:
            from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import KvSimilarityReport

            if isinstance(kv_report, dict):
                kv_report = KvSimilarityReport.from_dict(kv_report)
        else:
            from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import compute_kv_head_similarity

            kv_report = compute_kv_head_similarity(artifact)

        handles: list[torch.utils.hooks.RemovableHook] = []
        merged_layers: list[int] = []
        per_layer_pairs: dict[int, list[tuple[int, int]]] = {}

        try:
            for layer_idx in target_layers:
                anatomy = resolve_layer_anatomy(artifact.model, layer_idx)

                # 3. GQA constraint
                if anatomy.num_kv_heads >= anatomy.num_heads:
                    if not spec.allow_mha_to_gqa:
                        raise ValueError(
                            f"Layer {layer_idx}: num_kv_heads ({anatomy.num_kv_heads}) "
                            f">= num_heads ({anatomy.num_heads}). "
                            f"Set allow_mha_to_gqa=True to allow MHA-to-GQA merging."
                        )

                # 4. Merge pairs from report
                layer_pairs = kv_report.merge_pairs.get(layer_idx, [])
                num_to_merge = int(anatomy.num_kv_heads * spec.merge_ratio)
                num_to_merge = max(1, min(num_to_merge, len(layer_pairs)))
                selected_pairs = layer_pairs[:num_to_merge]

                if not selected_pairs:
                    continue

                hook = _make_kv_merge_hook(
                    selected_pairs,
                    anatomy.num_kv_heads,
                    anatomy.head_dim,
                )
                handles.append(anatomy.k_proj.register_forward_hook(hook))
                handles.append(anatomy.v_proj.register_forward_hook(hook))
                merged_layers.append(layer_idx)
                per_layer_pairs[layer_idx] = selected_pairs

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
                "target_layers": merged_layers,
                "merge_ratio": spec.merge_ratio,
                "per_layer_pairs": {
                    str(k): [list(p) for p in v]
                    for k, v in per_layer_pairs.items()
                },
            },
        )


def _make_kv_merge_hook(
    merge_pairs: list[tuple[int, int]],
    num_kv_heads: int,
    head_dim: int,
):
    """Create a forward hook that averages selected KV head pairs in-place."""

    def hook(module, input, output):
        # output: [B, S, num_kv_heads * head_dim]
        B, S, _ = output.shape
        out = output.view(B, S, num_kv_heads, head_dim).clone()
        for a, b in merge_pairs:
            avg = (out[:, :, a, :] + out[:, :, b, :]) / 2.0
            out[:, :, a, :] = avg
            out[:, :, b, :] = avg
        return out.view(B, S, -1)

    return hook


def _remove_hook_handles(handles) -> None:
    for handle in handles:
        handle.remove()
