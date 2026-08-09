"""Public API layer for model-pruning NAS."""

from __future__ import annotations

from typing import Any

from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.runtimes.base import BaseRuntime


def run(
    model_spec: ModelSpec,
    runtime: BaseRuntime,
    plan: TransformationPlan | None = None,
    *,
    loader: Any = None,
    transform_applier: Any = None,
) -> Any:
    """Load, optionally transform, evaluate, and clean up in one call."""
    with NASSession(loader=loader, transform_applier=transform_applier) as session:
        artifact = session.load(model_spec)
        if plan is not None and plan.transforms:
            artifact = session.apply_plan(artifact, plan)
        return session.evaluate(artifact, runtime)


__all__ = ["NASSession", "TransformersModelLoader", "run"]
