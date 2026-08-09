"""Base transform contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord, ModelArtifact


class BaseTransform(ABC):
    """Base interface for mutating a model artifact in place."""

    kind: str

    @abstractmethod
    def apply(self, artifact: ModelArtifact, spec) -> AppliedTransformRecord:
        """Apply the transform to the artifact and return an applied record."""
