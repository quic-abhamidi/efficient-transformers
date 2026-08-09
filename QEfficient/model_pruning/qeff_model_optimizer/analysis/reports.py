"""Typed records produced by the layer-contribution analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec


@dataclass(eq=True)
class RankedLayer:
    """One layer's aggregated contribution score with its ranking.

    ``aggregate_score`` is the mean per-layer delta (lower = weaker layer = better
    skip candidate). ``rank`` is 1-indexed: rank 1 is the weakest layer across
    the datasets that were evaluated.
    """

    layer: int
    aggregate_score: float
    rank: int
    per_dataset_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.rank < 1:
            raise ValueError("rank must be 1-indexed and positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "layer": self.layer,
            "aggregate_score": self.aggregate_score,
            "rank": self.rank,
            "per_dataset_scores": dict(self.per_dataset_scores),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RankedLayer":
        """Deserialise from a plain dict (e.g. from a manifest)."""
        return cls(
            layer=int(payload["layer"]),
            aggregate_score=float(payload["aggregate_score"]),
            rank=int(payload["rank"]),
            per_dataset_scores={
                str(k): float(v)
                for k, v in dict(payload.get("per_dataset_scores", {})).items()
            },
        )


@dataclass(eq=True)
class WeakLayerReport:
    """Structured output of ``analyze_weak_layers``.

    ``ranked_layers`` is sorted by ``aggregate_score`` ascending (weakest first).
    ``metadata`` is free-form and typically carries metric name, sample counts,
    token-range settings, and any other provenance information.
    """

    model_spec: ModelSpec
    datasets: list[str]
    ranked_layers: list[RankedLayer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("datasets must contain at least one entry")
        ranks = [r.rank for r in self.ranked_layers]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("ranked_layers must cover ranks 1..N exactly once")

    def weakest(self, n: int) -> list[RankedLayer]:
        """Return the ``n`` weakest layers (lowest aggregate_score)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return sorted(self.ranked_layers, key=lambda r: r.rank)[:n]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "model_spec": self.model_spec.to_dict(),
            "datasets": list(self.datasets),
            "ranked_layers": [r.to_dict() for r in self.ranked_layers],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeakLayerReport":
        """Deserialise from a plain dict (e.g. loaded from a manifest)."""
        return cls(
            model_spec=ModelSpec.from_dict(dict(payload["model_spec"])),
            datasets=[str(d) for d in payload.get("datasets", [])],
            ranked_layers=[
                RankedLayer.from_dict(item)
                for item in payload.get("ranked_layers", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )
