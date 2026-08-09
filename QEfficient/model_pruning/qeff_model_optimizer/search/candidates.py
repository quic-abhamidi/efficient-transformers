"""Candidate plan type and generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import RankedLayer, WeakLayerReport
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    SkipLayersSpec,
    TransformationPlan,
    plan_from_dict,
    plan_to_dict,
)


@dataclass(eq=True)
class CandidatePlan:
    """A scored :class:`TransformationPlan` candidate from the search layer."""

    plan: TransformationPlan
    priority: float
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON storage or manifest embedding."""
        return {
            "plan": plan_to_dict(self.plan),
            "priority": self.priority,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidatePlan":
        """Deserialise from a plain dict."""
        return cls(
            plan=plan_from_dict(dict(payload["plan"])),
            priority=float(payload["priority"]),
            rationale=str(payload.get("rationale", "")),
            metadata=dict(payload.get("metadata", {})),
        )


def _baseline_candidate(metric: str | None = None) -> CandidatePlan:
    metadata: dict[str, Any] = {"kind": "baseline"}
    if metric is not None:
        metadata["metric"] = metric
    return CandidatePlan(
        plan=TransformationPlan(transforms=[]),
        priority=0.0,
        rationale="baseline: no transform applied",
        metadata=metadata,
    )


def _single_layer_candidates(
    ranked: list[RankedLayer], max_candidates: int, metric: str | None = None
) -> list[CandidatePlan]:
    out: list[CandidatePlan] = []
    for layer in ranked[:max_candidates]:
        out.append(
            CandidatePlan(
                plan=TransformationPlan(
                    transforms=[SkipLayersSpec(layers=[layer.layer])],
                ),
                priority=float(layer.aggregate_score),
                rationale=(
                    f"skip layer {layer.layer}: weakest rank {layer.rank}, "
                    f"aggregate_score={layer.aggregate_score:.4g}"
                ),
                metadata={
                    "kind": "single",
                    "layers": [layer.layer],
                    "metric": metric,
                    "per_dataset_scores": dict(layer.per_dataset_scores),
                },
            )
        )
    return out


def _pair_candidates(
    ranked: list[RankedLayer], pool_size: int, max_candidates: int, metric: str | None = None
) -> list[CandidatePlan]:
    pool = ranked[:pool_size]
    out: list[CandidatePlan] = []
    for a, b in combinations(pool, 2):
        layers = sorted({a.layer, b.layer})
        if len(layers) < 2:
            continue
        priority = (a.aggregate_score + b.aggregate_score) / 2.0
        out.append(
            CandidatePlan(
                plan=TransformationPlan(
                    transforms=[SkipLayersSpec(layers=layers)],
                ),
                priority=priority,
                rationale=(
                    f"skip layers {layers}: pair of weak layers "
                    f"(ranks {a.rank} + {b.rank})"
                ),
                metadata={"kind": "pair", "layers": layers, "metric": metric},
            )
        )
    out.sort(key=lambda c: c.priority)
    return out[:max_candidates]


def _consecutive_run_candidate(
    ranked: list[RankedLayer], pool_size: int, run_length: int, metric: str | None = None
) -> CandidatePlan | None:
    """Find a run of adjacent layer indices that are all in the weak pool."""
    if run_length < 2:
        return None
    pool = {layer.layer for layer in ranked[:pool_size]}
    if not pool:
        return None
    sorted_pool = sorted(pool)
    best_run = None
    best_avg_score = float("inf")
    for start in sorted_pool:
        run = [start + offset for offset in range(run_length)]
        if all(layer in pool for layer in run):
            avg_score = (
                sum(
                    next(r for r in ranked if r.layer == layer).aggregate_score
                    for layer in run
                )
                / run_length
            )
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                best_run = run
    if best_run is not None:
        return CandidatePlan(
            plan=TransformationPlan(
                transforms=[SkipLayersSpec(layers=best_run)],
            ),
            priority=best_avg_score + 0.05 * run_length,
            rationale=f"skip consecutive layers {best_run}: all in weak pool",
            metadata={"kind": "consecutive", "layers": best_run, "metric": metric},
        )
    return None


def _consensus_candidates(
    report: WeakLayerReport, min_datasets: int, metric: str | None = None
) -> list[CandidatePlan]:
    """Layers that are among the weak half in a majority of datasets."""
    if len(report.datasets) < 2:
        return []
    weak_cutoff = max(1, len(report.ranked_layers) // 2)
    weak_half = {r.layer for r in report.ranked_layers[:weak_cutoff]}

    per_dataset_medians: dict[str, float] = {}
    for dataset in report.datasets:
        dataset_scores = sorted(
            r.per_dataset_scores[dataset]
            for r in report.ranked_layers
            if dataset in r.per_dataset_scores
        )
        if not dataset_scores:
            continue
        mid = len(dataset_scores) // 2
        if len(dataset_scores) % 2 == 0:
            per_dataset_medians[dataset] = (
                dataset_scores[mid - 1] + dataset_scores[mid]
            ) / 2
        else:
            per_dataset_medians[dataset] = dataset_scores[mid]

    per_layer_consensus: dict[int, int] = {}
    for ranked in report.ranked_layers:
        if ranked.layer not in weak_half:
            continue
        count = 0
        for dataset in report.datasets:
            if dataset not in ranked.per_dataset_scores:
                continue
            if dataset not in per_dataset_medians:
                continue
            if ranked.per_dataset_scores[dataset] <= per_dataset_medians[dataset]:
                count += 1
        if count >= min_datasets:
            per_layer_consensus[ranked.layer] = count

    out: list[CandidatePlan] = []
    for layer, count in sorted(
        per_layer_consensus.items(), key=lambda item: (-item[1], item[0])
    ):
        ranked_entry = next(r for r in report.ranked_layers if r.layer == layer)
        out.append(
            CandidatePlan(
                plan=TransformationPlan(
                    transforms=[SkipLayersSpec(layers=[layer])],
                ),
                priority=ranked_entry.aggregate_score - 0.01 * count,
                rationale=(
                    f"consensus skip layer {layer}: weak in {count}/{len(report.datasets)} datasets"
                ),
                metadata={
                    "kind": "consensus",
                    "layers": [layer],
                    "supporting_datasets": count,
                    "metric": metric,
                },
            )
        )
    return out


def generate_candidate_plans(
    report: WeakLayerReport,
    max_skip_layers: int = 3,
    top_k: int = 5,
    include_baseline: bool = True,
    metric: str | None = None,
) -> list[CandidatePlan]:
    """Emit a ranked list of candidate transformation plans from a weak-layer report.

    Generates, in order of priority (lowest first):

    1. Baseline (no transform).
    2. Single-layer skips for the top ``top_k`` weakest layers.
    3. Pair skips within the weak pool.
    4. One consecutive-run skip when adjacent weak layers exist.
    5. Cross-dataset consensus skips (only when multiple datasets were analyzed).

    ``max_skip_layers`` caps the largest individual skip plan (skip-triplets are
    emitted only when ``max_skip_layers >= 3``).
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_skip_layers < 1:
        raise ValueError("max_skip_layers must be at least 1")

    metric = metric or str(report.metadata.get("metric", "")) or None
    ranked = sorted(report.ranked_layers, key=lambda r: r.rank)
    candidates: list[CandidatePlan] = []
    non_baseline: list[CandidatePlan] = []

    non_baseline.extend(_single_layer_candidates(ranked, top_k, metric=metric))

    if max_skip_layers >= 2:
        pool = max(top_k, max_skip_layers * 2)
        non_baseline.extend(_pair_candidates(ranked, pool_size=pool, max_candidates=top_k, metric=metric))

    if max_skip_layers >= 3:
        for run_length in (2, 3):
            run_candidate = _consecutive_run_candidate(
                ranked, pool_size=max(top_k, max_skip_layers * 2), run_length=run_length, metric=metric
            )
            if run_candidate is not None:
                non_baseline.append(run_candidate)

    non_baseline.extend(
        _consensus_candidates(report, min_datasets=max(2, len(report.datasets) // 2), metric=metric)
    )

    non_baseline.sort(key=lambda c: c.priority)

    if include_baseline:
        candidates.append(_baseline_candidate(metric=metric))
    candidates.extend(non_baseline)
    return candidates


def flatten_candidate_plans(candidates: Iterable[CandidatePlan]) -> list[dict]:
    """Serialize a list of candidates for legacy JSON consumers."""
    return [candidate.to_dict() for candidate in candidates]
