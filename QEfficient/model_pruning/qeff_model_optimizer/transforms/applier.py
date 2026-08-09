"""Transformation plan applier and default transform registry."""

from __future__ import annotations

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import run_model_cleanup


def default_transform_registry():
    """Return the default static transform registry."""
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.compensation import CompensationTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.head_pruning import HeadPruningTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.kv_compression import KvCacheCompressionTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.linear_attention import LinearAttentionTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.mlp_pruning import MlpPruningTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.skip_layers import SkipLayersTransform
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.structured_sparsity import StructuredSparsityTransform
    return {
        "compensation": CompensationTransform(),
        "head_pruning": HeadPruningTransform(),
        "kv_cache_compression": KvCacheCompressionTransform(),
        "linear_attention": LinearAttentionTransform(),
        "mlp_pruning": MlpPruningTransform(),
        "skip_layers": SkipLayersTransform(),
        "structured_sparsity": StructuredSparsityTransform(),
    }


class TransformApplier:
    """Apply a typed transformation plan to an artifact in place."""

    def __init__(self, registry=None):
        self.registry = registry or default_transform_registry()

    def apply(
        self,
        artifact: ModelArtifact,
        plan: TransformationPlan,
    ) -> ModelArtifact:
        """Apply every spec in *plan* to *artifact* in order, returning the same artifact.

        If any spec raises, the model is cleaned up and the *previous* plan is
        re-applied so the artifact stays in a consistent state.  If the rollback
        also fails, a ``RuntimeError`` is raised that includes both the original
        and rollback failure messages.
        """
        previous_plan = artifact.plan

        self._restore_model(artifact)
        try:
            applied_records = self._apply_specs(artifact, plan)
        except Exception as original_exc:
            self._restore_model(artifact)
            if previous_plan.transforms:
                try:
                    restored_records = self._apply_specs(artifact, previous_plan)
                except Exception as rollback_exc:
                    self._restore_model(artifact)
                    artifact.plan = TransformationPlan()
                    raise RuntimeError(
                        "Transform apply failed "
                        f"({type(original_exc).__name__}: {original_exc}); "
                        "rollback to previous plan also failed "
                        f"({type(rollback_exc).__name__}: {rollback_exc})"
                    ) from original_exc
                artifact.plan = previous_plan
                artifact.applied_transforms[:] = restored_records
            raise

        artifact.plan = plan
        artifact.applied_transforms[:] = applied_records
        return artifact

    def _apply_specs(
        self,
        artifact: ModelArtifact,
        plan: TransformationPlan,
    ):
        applied_records = []
        for spec in plan.transforms:
            transform = self.registry.get(spec.kind)
            if transform is None:
                raise ValueError(f"No transform registered for kind {spec.kind!r}")
            record = transform.apply(artifact, spec)
            applied_records.append(record)
            artifact.applied_transforms[:] = applied_records
        return applied_records

    def _restore_model(self, artifact: ModelArtifact) -> None:
        run_model_cleanup(artifact.model)
        artifact.applied_transforms.clear()
