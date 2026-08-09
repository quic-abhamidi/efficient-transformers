"""Versioned manifest serializer for API-first NAS artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path

from QEfficient.model_pruning.qeff_model_optimizer import __version__ as nas_version
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import AppliedTransformRecord
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan, plan_from_dict, plan_to_dict


SCHEMA_VERSION = "nas.manifest/v1"


def _read_installed_version(distribution_name: str) -> str | None:
    """Return the installed version of a distribution, or None if absent."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(eq=True)
class EnvironmentInfo:
    """Snapshot of library versions at the time a manifest was saved.

    Construct via :meth:`capture` to auto-populate from installed distributions;
    or build manually for testing.  All version fields default to ``None`` when
    the corresponding distribution is not installed.
    """
    nas_version: str = nas_version
    transformers_version: str | None = None
    qefficient_version: str | None = None
    lm_eval_version: str | None = None

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Snapshot installed library versions at the time of call."""
        return cls(
            nas_version=nas_version,
            transformers_version=_read_installed_version("transformers"),
            qefficient_version=_read_installed_version("QEfficient"),
            lm_eval_version=_read_installed_version("lm-eval"),
        )

    def to_dict(self) -> dict[str, str | None]:
        """Serialise to a plain dict."""
        return {
            "nas_version": self.nas_version,
            "transformers_version": self.transformers_version,
            "qefficient_version": self.qefficient_version,
            "lm_eval_version": self.lm_eval_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EnvironmentInfo":
        """Deserialise from a plain dict."""
        return cls(
            nas_version=str(payload.get("nas_version", nas_version)),
            transformers_version=_optional_str(payload.get("transformers_version")),
            qefficient_version=_optional_str(payload.get("qefficient_version")),
            lm_eval_version=_optional_str(payload.get("lm_eval_version")),
        )


@dataclass(eq=True)
class SourceControlInfo:
    """Optional git provenance attached to a manifest (never auto-populated).

    Set ``repo_git_sha`` and ``repo_dirty`` manually when you want to pin the
    exact commit that produced a run.
    """
    repo_git_sha: str | None = None
    repo_dirty: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "repo_git_sha": self.repo_git_sha,
            "repo_dirty": self.repo_dirty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SourceControlInfo":
        """Deserialise from a plain dict."""
        return cls(
            repo_git_sha=_optional_str(payload.get("repo_git_sha")),
            repo_dirty=_optional_bool(payload.get("repo_dirty")),
        )


@dataclass(eq=True)
class ArtifactManifest:
    """Versioned record of everything needed to reproduce a NAS run.

    ``schema_version`` is validated on load; it must equal ``SCHEMA_VERSION``
    (``"nas.manifest/v1"``).  Use :func:`dump_manifest` / :func:`load_manifest`
    rather than constructing this directly from JSON.

    ``artifact_id`` is optional but recommended — it links the manifest back to
    the :class:`~nas.config.artifacts.ModelArtifact` that produced it.
    ``environment`` captures library versions for reproducibility; call
    ``EnvironmentInfo.capture()`` to populate it automatically.
    """
    model_spec: ModelSpec
    plan: TransformationPlan
    applied_transforms: list[AppliedTransformRecord] = field(default_factory=list)
    capabilities: dict[str, object] = field(default_factory=dict)
    environment: EnvironmentInfo | None = None
    source_control: SourceControlInfo | None = None
    artifact_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def manifest_to_dict(manifest: ArtifactManifest) -> dict[str, object]:
    """Serialise an :class:`ArtifactManifest` to a JSON-compatible dict."""
    payload: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "model_spec": manifest.model_spec.to_dict(),
        "plan": plan_to_dict(manifest.plan),
        "applied_transforms": [
            record.to_dict() for record in manifest.applied_transforms
        ],
        "capabilities": dict(manifest.capabilities),
    }
    if manifest.artifact_id is not None:
        payload["artifact_id"] = manifest.artifact_id
    if manifest.environment is not None:
        payload["environment"] = manifest.environment.to_dict()
    if manifest.source_control is not None:
        payload["source_control"] = manifest.source_control.to_dict()
    return payload


def manifest_from_dict(payload: dict[str, object]) -> ArtifactManifest:
    """Deserialise a plain dict to an :class:`ArtifactManifest`, validating schema_version."""
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema version: {schema_version!r}"
        )

    environment_payload = payload.get("environment")
    source_control_payload = payload.get("source_control")

    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        model_spec=ModelSpec.from_dict(_as_dict(payload.get("model_spec"), "model_spec")),
        plan=plan_from_dict(_as_dict(payload.get("plan"), "plan")),
        applied_transforms=[
            AppliedTransformRecord.from_dict(item)
            for item in _as_list(payload.get("applied_transforms"), "applied_transforms")
        ],
        capabilities=dict(_as_dict(payload.get("capabilities", {}), "capabilities")),
        artifact_id=_optional_str(payload.get("artifact_id")),
        environment=(
            EnvironmentInfo.from_dict(_as_dict(environment_payload, "environment"))
            if environment_payload is not None
            else None
        ),
        source_control=(
            SourceControlInfo.from_dict(
                _as_dict(source_control_payload, "source_control")
            )
            if source_control_payload is not None
            else None
        ),
    )


def dump_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    """Serialise *manifest* to *path* as pretty-printed JSON, creating parent dirs as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_manifest(path: str | Path) -> ArtifactManifest:
    """Read and deserialise an :class:`ArtifactManifest` from a JSON file at *path*."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest file must contain a JSON object")
    return manifest_from_dict(payload)


def _as_dict(value: object, field_name: str) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field_name} must be a JSON object")


def _as_list(value: object, field_name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    normalized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be JSON objects")
        normalized.append(item)
    return normalized
