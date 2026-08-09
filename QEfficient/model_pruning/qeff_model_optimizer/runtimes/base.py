"""Base runtime abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact


class BaseRuntime(ABC):
    """Abstract runtime interface."""

    name: str

    @abstractmethod
    def evaluate(self, artifact: ModelArtifact):
        """Run a prepared artifact on a concrete runtime."""
