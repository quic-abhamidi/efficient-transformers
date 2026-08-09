"""Runtime artifact records produced and consumed by the NAS API layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan

ALLOWED_TRANSFORM_STATUSES = {"applied", "skipped", "degraded", "failed"}


@dataclass(eq=True)
class AppliedTransformRecord:
    """Immutable result record for a single applied (or skipped) transform step.

    ``kind`` matches the ``TransformSpec.kind`` that produced this record.
    ``status`` indicates the outcome:

    - ``"applied"`` — transform was successfully applied in full.
    - ``"skipped"`` — transform was intentionally skipped (e.g. best_effort mode
      when a dependency was absent).
    - ``"degraded"`` — transform was partially applied with a recoverable issue.
    - ``"failed"`` — transform could not be applied; the model state is unchanged.

    ``details`` carries strategy-specific provenance (e.g. which layer indices
    were modified).  ``warnings`` accumulates non-fatal messages from the
    transform implementation.
    """

    kind: str
    status: Literal["applied", "skipped", "degraded", "failed"]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind must be a non-empty string")
        if self.status not in ALLOWED_TRANSFORM_STATUSES:
            raise ValueError(f"Unsupported transform status: {self.status}")

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "kind": self.kind,
            "status": self.status,
            "details": self.details,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppliedTransformRecord":
        """Deserialise from a plain dict (e.g. loaded from a manifest)."""
        return cls(
            kind=str(payload["kind"]),
            status=str(payload["status"]),
            details=dict(payload.get("details", {})),
            warnings=list(payload.get("warnings", [])),
        )


@dataclass(eq=True)
class ModelArtifact:
    """Mutable handle bundling a loaded model with its accumulated transform state.

    Created by ``NASSession.load()`` and mutated in-place by
    ``NASSession.apply_plan()``.  Callers should not construct this directly;
    use the session API.

    ``artifact_id`` is a unique hex string assigned at load time.
    ``applied_transforms`` grows as specs are applied; it reflects the *current*
    state of the model and is cleared/rebuilt on each ``apply_plan`` call.
    ``capability_report`` is populated by runtimes or analysis steps that
    augment the artifact with post-transform metadata.
    """

    artifact_id: str
    model: Any
    tokenizer: Any
    model_spec: ModelSpec
    plan: TransformationPlan
    applied_transforms: list[AppliedTransformRecord] = field(default_factory=list)
    capability_report: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be a non-empty string")
