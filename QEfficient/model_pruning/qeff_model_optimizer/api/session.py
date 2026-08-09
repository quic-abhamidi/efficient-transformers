"""Session lifecycle and orchestration for the NAS API."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier
from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import run_model_cleanup


class NASSession:
    """Lightweight resource manager for loading and evaluating NAS artifacts.

    Use as a context manager (``with NASSession() as session:``) so that
    ``close()`` is called automatically on exit, which restores any patched
    model layer forwards and removes forward hooks registered by transforms.

    Custom ``loader`` and ``transform_applier`` can be injected for testing or
    alternative model-loading strategies; both default to the standard
    implementations when omitted.
    """

    def __init__(self, loader=None, transform_applier=None):
        self.loader = loader or TransformersModelLoader()
        self.transform_applier = transform_applier or TransformApplier()
        self.artifacts: dict[str, ModelArtifact] = {}
        self._closed = False

    def __enter__(self) -> "NASSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def load(self, model_spec: ModelSpec) -> ModelArtifact:
        """Load a model from *model_spec* and register it for lifecycle management.

        Creates a :class:`~nas.config.artifacts.ModelArtifact` with a fresh
        ``uuid4`` id, stores it in ``self.artifacts`` so that :meth:`close` can
        clean it up, and returns it.  The artifact starts with an empty
        :class:`~nas.config.transforms.TransformationPlan` — call
        :meth:`apply_plan` to modify the model before evaluation.
        """
        self._ensure_open()
        model, tokenizer = self.loader.load(model_spec)
        artifact = ModelArtifact(
            artifact_id=uuid4().hex,
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            plan=TransformationPlan(),
        )
        self.artifacts[artifact.artifact_id] = artifact
        return artifact

    def apply_plan(
        self,
        artifact: ModelArtifact,
        plan: TransformationPlan,
    ) -> ModelArtifact:
        """Apply *plan* to *artifact* in place and return the same artifact.

        Transforms are applied sequentially in ``plan.transforms`` order.  If any
        transform raises, the applier rolls back to the previously-applied plan so
        the artifact model is left in a consistent state.  The artifact object
        identity is preserved — callers need not replace their reference.
        """
        self._ensure_open()
        updated = self.transform_applier.apply(artifact, plan)
        if updated is not artifact:
            raise ValueError("Transform applier must mutate and return the same artifact")
        self.artifacts[artifact.artifact_id] = artifact
        return artifact

    def evaluate(self, artifact: ModelArtifact, runtime: Any):
        """Dispatch evaluation of *artifact* through *runtime*.

        ``runtime`` is duck-typed: any object with an ``evaluate(artifact)``
        method is accepted.  Use :class:`~nas.runtimes.hf.HuggingFaceRuntime` for
        lm-eval benchmarks or :class:`~nas.runtimes.qeff.QEffRuntime` for QAIC.
        """
        self._ensure_open()
        if not hasattr(runtime, "evaluate"):
            raise TypeError("runtime must provide an evaluate(artifact) method")
        return runtime.evaluate(artifact)

    def close(self) -> None:
        """Release all resources held by this session.

        Runs registered cleanup callbacks on every loaded artifact (reversing
        forward-method patches and forward hooks), then clears ``self.artifacts``.
        Exception-safe: errors from individual artifact cleanup are collected and
        raised together at the end rather than aborting mid-cleanup.
        Idempotent — calling ``close()`` a second time is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for artifact in self.artifacts.values():
            model = artifact.model
            for step in (
                lambda: run_model_cleanup(model),
                lambda: model.remove_hooks() if hasattr(model, "remove_hooks") else None,
                lambda: model.close() if hasattr(model, "close") else None,
            ):
                try:
                    step()
                except Exception as exc:
                    errors.append(exc)
        self.artifacts.clear()
        if errors:
            raise RuntimeError(
                f"NASSession.close encountered {len(errors)} cleanup error(s): {errors}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("NASSession is closed")
