"""Multi-transform optimization plan generator.

Generates a fixed set of useful plan *variants* covering different combinations
of pruning, compression, and sparsity transforms.  Each variant is wrapped in a
:class:`~nas.search.candidates.CandidatePlan` with a priority score that
reflects estimated impact (lower = more conservative).

True budget-aware search (Pareto front enumeration, latency-driven knapsack,
etc.) can be layered on top later -- this module gives the user a curated set of
actionable candidates to evaluate first.
"""

from __future__ import annotations

from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import WeakLayerReport
from QEfficient.model_pruning.qeff_model_optimizer.analysis.head_importance import HeadImportanceReport
from QEfficient.model_pruning.qeff_model_optimizer.analysis.channel_importance import ChannelImportanceReport
from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import KvSimilarityReport
from QEfficient.model_pruning.qeff_model_optimizer.search.candidates import CandidatePlan, generate_candidate_plans
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    HeadPruningSpec,
    LayerHeadSelection,
    MlpPruningSpec,
    KvCacheCompressionSpec,
    StructuredSparsitySpec,
    SkipLayersSpec,
    TransformationPlan,
    TransformSpec,
)

# -----------------------------------------------------------------------
# Composition ordering -- transforms in a combined plan are sorted by kind
# -----------------------------------------------------------------------

KIND_ORDER: dict[str, int] = {
    "skip_layers": 0,
    "head_pruning": 1,
    "mlp_pruning": 2,
    "kv_cache_compression": 3,
    "structured_sparsity": 4,
    "compensation": 5,
}


def _sort_transforms(transforms: list[TransformSpec]) -> list[TransformSpec]:
    """Return *transforms* ordered by the canonical kind priority."""
    return sorted(transforms, key=lambda t: KIND_ORDER.get(t.kind, 99))


# -----------------------------------------------------------------------
# Importance-to-spec mapping helpers
# -----------------------------------------------------------------------


def _build_head_pruning_spec(
    report: HeadImportanceReport,
    prune_ratio: float,
) -> HeadPruningSpec:
    """Build a :class:`HeadPruningSpec` by selecting the weakest heads per layer."""
    selections: list[LayerHeadSelection] = []
    for layer_idx, head_scores in report.per_layer_scores.items():
        num_to_prune = max(1, int(len(head_scores) * prune_ratio))
        weakest_heads = [h for h, _s in head_scores[:num_to_prune]]
        selections.append(LayerHeadSelection(layer=layer_idx, heads=weakest_heads))
    return HeadPruningSpec(selections=selections)


def _build_mlp_pruning_spec(
    prune_ratio: float,
    target_layers: list[int] | None = None,
) -> MlpPruningSpec:
    return MlpPruningSpec(target_layers=target_layers or [], pruning_ratio=prune_ratio)


def _build_kv_compression_spec(
    merge_ratio: float,
    target_layers: list[int] | None = None,
) -> KvCacheCompressionSpec:
    return KvCacheCompressionSpec(target_layers=target_layers or [], merge_ratio=merge_ratio)


def _build_sparsity_spec(
    target_layers: list[int] | None = None,
) -> StructuredSparsitySpec:
    return StructuredSparsitySpec(target_layers=target_layers or [])


# -----------------------------------------------------------------------
# Priority scoring
# -----------------------------------------------------------------------


def _compute_priority(
    skip_layers: list[int] | None,
    num_layers: int,
    head_prune_ratio: float | None,
    mlp_prune_ratio: float | None,
    kv_merge_ratio: float | None,
    has_sparsity: bool,
) -> float:
    """Compute a scalar priority score for a plan variant.

    Lower priority = more conservative.
    """
    priority = 0.0
    if skip_layers:
        priority += len(skip_layers) * (1.0 / max(num_layers, 1))
    if head_prune_ratio is not None:
        priority += head_prune_ratio * 0.33  # attn fraction
    if mlp_prune_ratio is not None:
        priority += mlp_prune_ratio * 0.67  # MLP fraction
    if kv_merge_ratio is not None:
        priority += kv_merge_ratio * 0.1
    if has_sparsity:
        priority += 0.5
    return priority


# -----------------------------------------------------------------------
# Plan construction helpers
# -----------------------------------------------------------------------


def _make_candidate(
    transforms: list[TransformSpec],
    kind: str,
    rationale: str,
    num_layers: int,
    *,
    skip_layers: list[int] | None = None,
    head_prune_ratio: float | None = None,
    mlp_prune_ratio: float | None = None,
    kv_merge_ratio: float | None = None,
    has_sparsity: bool = False,
) -> CandidatePlan:
    """Assemble a :class:`CandidatePlan` with sorted transforms and metadata."""
    sorted_transforms = _sort_transforms(transforms)
    priority = _compute_priority(
        skip_layers=skip_layers,
        num_layers=num_layers,
        head_prune_ratio=head_prune_ratio,
        mlp_prune_ratio=mlp_prune_ratio,
        kv_merge_ratio=kv_merge_ratio,
        has_sparsity=has_sparsity,
    )
    transform_kinds = [t.kind for t in sorted_transforms]

    # Estimate speedup from the priority (rough linear mapping for metadata).
    estimated_speedup = 1.0 + priority

    return CandidatePlan(
        plan=TransformationPlan(transforms=sorted_transforms),
        priority=priority,
        rationale=rationale,
        metadata={
            "kind": kind,
            "transforms_applied": transform_kinds,
            "estimated_speedup": round(estimated_speedup, 3),
        },
    )


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


def generate_optimization_plans(
    weak_layer_report: WeakLayerReport,
    head_importance_report: HeadImportanceReport | None = None,
    channel_importance_report: ChannelImportanceReport | None = None,
    kv_similarity_report: KvSimilarityReport | None = None,
    target_speedup: float = 1.3,
    accuracy_budget: float = 0.05,
    enable_sparsity: bool = False,
    top_k_layers: int = 5,
    head_prune_ratio: float = 0.25,
    mlp_prune_ratio: float = 0.2,
    kv_merge_ratio: float = 0.5,
) -> list[CandidatePlan]:
    """Generate a curated set of multi-transform optimisation plan variants.

    Each variant targets a different trade-off between model quality and
    inference cost.  Plans are returned sorted by *priority* ascending (most
    conservative first).

    Parameters
    ----------
    weak_layer_report:
        Layer-contribution analysis used to identify skip candidates.
    head_importance_report:
        Optional per-head importance scores for attention head pruning.
    channel_importance_report:
        Optional per-channel importance scores for MLP width pruning.
    kv_similarity_report:
        Optional KV head similarity analysis for KV cache compression.
    target_speedup:
        Desired inference speedup factor (informational -- stored in metadata).
    accuracy_budget:
        Acceptable accuracy degradation (informational -- stored in metadata).
    enable_sparsity:
        Whether to include 2:4 structured sparsity in applicable plans.
    top_k_layers:
        Number of weakest layers to consider for skip candidates.
    head_prune_ratio:
        Fraction of attention heads to prune per layer (0-1).
    mlp_prune_ratio:
        Fraction of MLP channels to prune per layer (0-1).
    kv_merge_ratio:
        Fraction of KV heads to merge per layer (0-1).

    Returns
    -------
    list[CandidatePlan]
        Plans sorted by priority ascending (baseline first, aggressive last).
    """
    num_layers = len(weak_layer_report.ranked_layers)
    candidates: list[CandidatePlan] = []

    # ------------------------------------------------------------------
    # 1. Baseline -- no transforms
    # ------------------------------------------------------------------
    candidates.append(
        CandidatePlan(
            plan=TransformationPlan(transforms=[]),
            priority=0.0,
            rationale="baseline: no transforms applied",
            metadata={
                "kind": "baseline",
                "transforms_applied": [],
                "estimated_speedup": 1.0,
            },
        )
    )

    # ------------------------------------------------------------------
    # Reusable building blocks
    # ------------------------------------------------------------------

    # Best single-layer skip from generate_candidate_plans
    skip_spec: SkipLayersSpec | None = None
    skip_candidates = generate_candidate_plans(
        weak_layer_report, max_skip_layers=1, top_k=top_k_layers, include_baseline=False,
    )
    if skip_candidates:
        best_skip = skip_candidates[0]  # lowest priority = weakest layer
        skip_spec = best_skip.plan.transforms[0] if best_skip.plan.transforms else None  # type: ignore[assignment]

    head_spec: HeadPruningSpec | None = None
    if head_importance_report is not None:
        head_spec = _build_head_pruning_spec(head_importance_report, head_prune_ratio)

    mlp_spec: MlpPruningSpec | None = None
    if channel_importance_report is not None:
        mlp_spec = _build_mlp_pruning_spec(mlp_prune_ratio)

    kv_spec: KvCacheCompressionSpec | None = None
    if kv_similarity_report is not None:
        kv_spec = _build_kv_compression_spec(kv_merge_ratio)

    sparsity_spec: StructuredSparsitySpec | None = None
    if enable_sparsity:
        sparsity_spec = _build_sparsity_spec()

    # ------------------------------------------------------------------
    # 2. Skip-only
    # ------------------------------------------------------------------
    if skip_spec is not None:
        candidates.append(
            _make_candidate(
                transforms=[skip_spec],
                kind="skip_only",
                rationale=f"skip-only: skip layers {skip_spec.layers}",
                num_layers=num_layers,
                skip_layers=skip_spec.layers,
            )
        )

    # ------------------------------------------------------------------
    # 3. Head-prune-only
    # ------------------------------------------------------------------
    if head_spec is not None:
        candidates.append(
            _make_candidate(
                transforms=[head_spec],
                kind="head_prune_only",
                rationale=f"head-prune-only: prune {head_prune_ratio:.0%} heads across all layers",
                num_layers=num_layers,
                head_prune_ratio=head_prune_ratio,
            )
        )

    # ------------------------------------------------------------------
    # 4. MLP-prune-only
    # ------------------------------------------------------------------
    if mlp_spec is not None:
        candidates.append(
            _make_candidate(
                transforms=[mlp_spec],
                kind="mlp_prune_only",
                rationale=f"mlp-prune-only: prune {mlp_prune_ratio:.0%} channels across all layers",
                num_layers=num_layers,
                mlp_prune_ratio=mlp_prune_ratio,
            )
        )

    # ------------------------------------------------------------------
    # 5. KV-compress-only
    # ------------------------------------------------------------------
    if kv_spec is not None:
        candidates.append(
            _make_candidate(
                transforms=[kv_spec],
                kind="kv_compress_only",
                rationale=f"kv-compress-only: merge {kv_merge_ratio:.0%} of KV heads",
                num_layers=num_layers,
                kv_merge_ratio=kv_merge_ratio,
            )
        )

    # ------------------------------------------------------------------
    # 6. Sparsity-only
    # ------------------------------------------------------------------
    if sparsity_spec is not None:
        candidates.append(
            _make_candidate(
                transforms=[sparsity_spec],
                kind="sparsity_only",
                rationale="sparsity-only: 2:4 structured sparsity on all layers",
                num_layers=num_layers,
                has_sparsity=True,
            )
        )

    # ------------------------------------------------------------------
    # 7. Conservative -- skip + head prune at half ratio
    # ------------------------------------------------------------------
    conservative_transforms: list[TransformSpec] = []
    conservative_skip: list[int] | None = None
    conservative_head_ratio: float | None = None
    if skip_spec is not None:
        conservative_transforms.append(skip_spec)
        conservative_skip = skip_spec.layers
    if head_importance_report is not None:
        half_head_spec = _build_head_pruning_spec(
            head_importance_report, head_prune_ratio / 2.0,
        )
        conservative_transforms.append(half_head_spec)
        conservative_head_ratio = head_prune_ratio / 2.0
    if conservative_transforms:
        candidates.append(
            _make_candidate(
                transforms=conservative_transforms,
                kind="conservative",
                rationale="conservative: skip + head prune at half ratio",
                num_layers=num_layers,
                skip_layers=conservative_skip,
                head_prune_ratio=conservative_head_ratio,
            )
        )

    # ------------------------------------------------------------------
    # 8. Recommended -- skip + head prune + MLP prune (full ratios)
    # ------------------------------------------------------------------
    recommended_transforms: list[TransformSpec] = []
    recommended_skip: list[int] | None = None
    recommended_head: float | None = None
    recommended_mlp: float | None = None
    if skip_spec is not None:
        recommended_transforms.append(skip_spec)
        recommended_skip = skip_spec.layers
    if head_spec is not None:
        recommended_transforms.append(head_spec)
        recommended_head = head_prune_ratio
    if mlp_spec is not None:
        recommended_transforms.append(mlp_spec)
        recommended_mlp = mlp_prune_ratio
    if recommended_transforms:
        candidates.append(
            _make_candidate(
                transforms=recommended_transforms,
                kind="recommended",
                rationale="recommended: skip + head prune + MLP prune",
                num_layers=num_layers,
                skip_layers=recommended_skip,
                head_prune_ratio=recommended_head,
                mlp_prune_ratio=recommended_mlp,
            )
        )

    # ------------------------------------------------------------------
    # 9. Aggressive -- everything enabled
    # ------------------------------------------------------------------
    aggressive_transforms: list[TransformSpec] = []
    agg_skip: list[int] | None = None
    agg_head: float | None = None
    agg_mlp: float | None = None
    agg_kv: float | None = None
    agg_sparsity = False
    if skip_spec is not None:
        aggressive_transforms.append(skip_spec)
        agg_skip = skip_spec.layers
    if head_spec is not None:
        aggressive_transforms.append(head_spec)
        agg_head = head_prune_ratio
    if mlp_spec is not None:
        aggressive_transforms.append(mlp_spec)
        agg_mlp = mlp_prune_ratio
    if kv_spec is not None:
        aggressive_transforms.append(kv_spec)
        agg_kv = kv_merge_ratio
    if sparsity_spec is not None:
        aggressive_transforms.append(sparsity_spec)
        agg_sparsity = True
    if aggressive_transforms:
        candidates.append(
            _make_candidate(
                transforms=aggressive_transforms,
                kind="aggressive",
                rationale="aggressive: all available transforms combined",
                num_layers=num_layers,
                skip_layers=agg_skip,
                head_prune_ratio=agg_head,
                mlp_prune_ratio=agg_mlp,
                kv_merge_ratio=agg_kv,
                has_sparsity=agg_sparsity,
            )
        )

    # ------------------------------------------------------------------
    # Sort by priority ascending (baseline stays first naturally)
    # ------------------------------------------------------------------
    candidates.sort(key=lambda c: c.priority)
    return candidates


__all__ = [
    "generate_optimization_plans",
]
