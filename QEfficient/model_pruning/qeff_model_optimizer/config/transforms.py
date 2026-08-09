"""Typed transform specifications, compensation configs, and plan serialisation.

A ``TransformationPlan`` is an ordered list of ``TransformSpec`` entries.
Order matters: a ``CompensationSpec`` must follow the ``SkipLayersSpec`` it
compensates for, because the compensation transform reads the most-recently
applied skip record to determine which layers were skipped.

The ``plan_to_dict`` / ``plan_from_dict`` helpers provide stable JSON round-
trip serialisation used by the manifest layer and the search API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Union


ALLOWED_COMPATIBILITY_MODES = {"strict", "best_effort"}


def _canonicalize_layers(layers: list[int], field_name: str = "layers") -> list[int]:
    if not layers:
        raise ValueError(f"{field_name} must contain at least one layer index")
    normalized = sorted({int(layer) for layer in layers})
    if normalized[0] < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _canonicalize_heads(heads: list[int]) -> list[int]:
    if not heads:
        raise ValueError("heads must contain at least one head index")
    normalized = sorted({int(head) for head in heads})
    if normalized[0] < 0:
        raise ValueError("heads must be non-negative")
    return normalized


def _require_path(path: str, field_name: str) -> str:
    value = str(path)
    if not value:
        raise ValueError(f"{field_name} must be provided")
    return value


def _canonicalize_target_layers(layers: list[int]) -> list[int]:
    if not layers:
        return []
    normalized = sorted({int(layer) for layer in layers})
    if normalized[0] < 0:
        raise ValueError("target_layers must be non-negative")
    return normalized


@dataclass(eq=True)
class SkipLayersSpec:
    """Instruction to replace specific decoder layers with a no-op forward pass.

    layers is canonicalised (deduped, sorted, non-negative) by __post_init__.
    """
    kind: Literal["skip_layers"] = "skip_layers"
    layers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layers = _canonicalize_layers(self.layers)


@dataclass(eq=True)
class RemoveLayersSpec:
    """Instruction to structurally delete decoder layers (for QAIC export).

    Supported at the spec level; apply logic is not yet wired for v1 HF transforms.
    layers is canonicalised by __post_init__.
    """
    kind: Literal["remove_layers"] = "remove_layers"
    layers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layers = _canonicalize_layers(self.layers)


@dataclass(eq=True)
class ScaledCompensationConfig:
    """Adds alpha * mean_delta to every token position uniformly.

    Requires: mean_delta_path — path to a .pt file created by
    collect_and_fit.py containing a mean hidden-state delta tensor.
    """
    strategy: Literal["scaled"] = "scaled"
    mean_delta_path: str = ""
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.mean_delta_path = _require_path(
            self.mean_delta_path,
            "mean_delta_path",
        )


@dataclass(eq=True)
class LastTokenCompensationConfig:
    """Applies the mean-delta compensation only to the last-token position.

    Requires: mean_delta_path.
    """
    strategy: Literal["last_token"] = "last_token"
    mean_delta_path: str = ""
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.mean_delta_path = _require_path(
            self.mean_delta_path,
            "mean_delta_path",
        )


@dataclass(eq=True)
class MagnitudePreservingCompensationConfig:
    """Adds compensation scaled to preserve the original output magnitude.

    Requires: mean_delta_path.
    """
    strategy: Literal["magnitude_preserving"] = "magnitude_preserving"
    mean_delta_path: str = ""
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.mean_delta_path = _require_path(
            self.mean_delta_path,
            "mean_delta_path",
        )


@dataclass(eq=True)
class CascadedCompensationConfig:
    """Split compensation: a fraction before the skip and the rest after.

    Requires: mean_delta_path.
    pre_skip_fraction (0–1) controls how much of the delta is applied before
    the skipped layer; the remainder is applied after.
    """
    strategy: Literal["cascaded"] = "cascaded"
    mean_delta_path: str = ""
    pre_skip_fraction: float = 0.5
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.mean_delta_path = _require_path(
            self.mean_delta_path,
            "mean_delta_path",
        )
        if not 0.0 <= float(self.pre_skip_fraction) <= 1.0:
            raise ValueError("pre_skip_fraction must be between 0.0 and 1.0")


@dataclass(eq=True)
class MagnitudeRescalingCompensationConfig:
    """Rescales the output magnitude by norm_ratio after adding the mean delta.

    Requires: mean_delta_path.
    """
    strategy: Literal["magnitude_rescaling"] = "magnitude_rescaling"
    mean_delta_path: str = ""
    norm_ratio: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.mean_delta_path = _require_path(
            self.mean_delta_path,
            "mean_delta_path",
        )


@dataclass(eq=True)
class PhaseAwareCompensationConfig:
    """Uses separate delta vectors for prefill and decode phases.

    Phase is detected by checking whether hidden_states.shape[1] == 1.
    Requires: prefill_delta_path and decode_delta_path.
    """
    strategy: Literal["phase_aware"] = "phase_aware"
    prefill_delta_path: str = ""
    decode_delta_path: str = ""
    prefill_alpha: float = 1.0
    decode_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.prefill_delta_path = _require_path(
            self.prefill_delta_path,
            "prefill_delta_path",
        )
        self.decode_delta_path = _require_path(
            self.decode_delta_path,
            "decode_delta_path",
        )


@dataclass(eq=True)
class PhaseAwareLastTokenCompensationConfig:
    """Phase-aware compensation applied only to the last token position.

    Requires: prefill_delta_path and decode_delta_path.
    """
    strategy: Literal["phase_last_token"] = "phase_last_token"
    prefill_delta_path: str = ""
    decode_delta_path: str = ""
    prefill_alpha: float = 1.0
    decode_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.prefill_delta_path = _require_path(
            self.prefill_delta_path,
            "prefill_delta_path",
        )
        self.decode_delta_path = _require_path(
            self.decode_delta_path,
            "decode_delta_path",
        )


@dataclass(eq=True)
class PhaseAwareMagnitudeRescalingCompensationConfig:
    """Rescales prefill and decode output magnitudes by separate ratios (no delta file needed)."""
    strategy: Literal["phase_aware_magnitude_rescaling"] = (
        "phase_aware_magnitude_rescaling"
    )
    prefill_norm_ratio: float = 1.0
    decode_norm_ratio: float = 1.0


@dataclass(eq=True)
class PositionAwareCompensationConfig:
    """Applies position-bucket-specific deltas with an optional fallback mean delta.

    Requires: bucket_deltas_path — a .pt file mapping bucket indices to delta
    tensors.  fallback_delta_path is used when no bucket matches.
    """
    strategy: Literal["position_aware"] = "position_aware"
    bucket_deltas_path: str = ""
    num_buckets: int = 10
    alpha: float = 1.0
    fallback_delta_path: str | None = None

    def __post_init__(self) -> None:
        self.bucket_deltas_path = _require_path(
            self.bucket_deltas_path,
            "bucket_deltas_path",
        )
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        if self.fallback_delta_path is not None:
            self.fallback_delta_path = _require_path(
                self.fallback_delta_path,
                "fallback_delta_path",
            )


@dataclass(eq=True)
class PcaCompensationConfig:
    """Projects the compensation through a PCA basis before applying.

    Requires: pca_path — a .pt file with pre-computed PCA components.
    mean_delta_path is optional; when supplied the PCA is combined with the
    mean-delta reconstruction.
    """
    strategy: Literal["pca"] = "pca"
    pca_path: str = ""
    n_components: int = 32
    alpha: float = 1.0
    mean_delta_path: str | None = None

    def __post_init__(self) -> None:
        self.pca_path = _require_path(self.pca_path, "pca_path")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.mean_delta_path is not None:
            self.mean_delta_path = _require_path(
                self.mean_delta_path,
                "mean_delta_path",
            )


@dataclass(eq=True)
class MultiplicativeCompensationConfig:
    """Applies an element-wise scale + bias to the hidden state.

    Requires: scale_vector_path and bias_vector_path.
    """
    strategy: Literal["multiplicative"] = "multiplicative"
    scale_vector_path: str = ""
    bias_vector_path: str = ""

    def __post_init__(self) -> None:
        self.scale_vector_path = _require_path(
            self.scale_vector_path,
            "scale_vector_path",
        )
        self.bias_vector_path = _require_path(
            self.bias_vector_path,
            "bias_vector_path",
        )


@dataclass(eq=True)
class LearnableCompensationConfig:
    """Applies a small learned neural network as the compensation module.

    Requires: model_path — a .pt checkpoint loadable by
    LearnableCompensation.load() in core.learnable_compensation.
    """
    strategy: Literal["learnable"] = "learnable"
    model_path: str = ""

    def __post_init__(self) -> None:
        self.model_path = _require_path(self.model_path, "model_path")


@dataclass(eq=True)
class CompensationLayerRatio:
    """Per-layer prefill/decode rescaling ratios used by multi-cluster compensation."""
    layer: int
    prefill_norm_ratio: float = 1.0
    decode_norm_ratio: float = 1.0

    def __post_init__(self) -> None:
        self.layer = int(self.layer)
        if self.layer < 0:
            raise ValueError("layer must be non-negative")


@dataclass(eq=True)
class MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig:
    """Per-layer-cluster rescaling: different prefill/decode ratios for each layer group.

    Requires: layer_ratios — a list of :class: entries.
    Layers not listed use default_prefill_norm_ratio / default_decode_norm_ratio.
    """
    strategy: Literal["multi_phase_aware_magnitude_rescaling"] = (
        "multi_phase_aware_magnitude_rescaling"
    )
    layer_ratios: list[CompensationLayerRatio] = field(default_factory=list)
    default_prefill_norm_ratio: float = 1.0
    default_decode_norm_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not self.layer_ratios:
            raise ValueError("layer_ratios must contain at least one layer ratio")
        layers = [item.layer for item in self.layer_ratios]
        if len(layers) != len(set(layers)):
            raise ValueError("layer_ratios must not contain duplicate layers")
        self.layer_ratios = sorted(self.layer_ratios, key=lambda item: item.layer)


CompensationConfig = Union[
    ScaledCompensationConfig,
    LastTokenCompensationConfig,
    MagnitudePreservingCompensationConfig,
    CascadedCompensationConfig,
    MagnitudeRescalingCompensationConfig,
    PhaseAwareCompensationConfig,
    PhaseAwareLastTokenCompensationConfig,
    PhaseAwareMagnitudeRescalingCompensationConfig,
    PositionAwareCompensationConfig,
    PcaCompensationConfig,
    MultiplicativeCompensationConfig,
    LearnableCompensationConfig,
    MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
]


ALLOWED_COMPENSATION_CONFIG_TYPES = (
    ScaledCompensationConfig,
    LastTokenCompensationConfig,
    MagnitudePreservingCompensationConfig,
    CascadedCompensationConfig,
    MagnitudeRescalingCompensationConfig,
    PhaseAwareCompensationConfig,
    PhaseAwareLastTokenCompensationConfig,
    PhaseAwareMagnitudeRescalingCompensationConfig,
    PositionAwareCompensationConfig,
    PcaCompensationConfig,
    MultiplicativeCompensationConfig,
    LearnableCompensationConfig,
    MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
)


@dataclass(eq=True)
class CompensationSpec:
    """Wraps a compensation strategy config for use in a :class:.

    config must be one of the 13 *CompensationConfig variants.
    It must be supplied explicitly — there is no default, as each strategy
    requires different mandatory file paths.
    """
    config: CompensationConfig
    kind: Literal["compensation"] = "compensation"

    def __post_init__(self) -> None:
        if not isinstance(self.config, ALLOWED_COMPENSATION_CONFIG_TYPES):
            raise ValueError(
                f"Unsupported compensation config type: {type(self.config).__name__!r}"
            )


@dataclass(eq=True)
class LayerHeadSelection:
    """Identifies which attention heads to prune in a given decoder layer."""
    layer: int
    heads: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layer = int(self.layer)
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        self.heads = _canonicalize_heads(self.heads)


@dataclass(eq=True)
class HeadPruningSpec:
    """Instruction to mask specific attention heads (mask mode only in v1).

    selections must contain at least one :class:; each
    layer may appear only once.  Head pruning is spec-complete but the apply
    logic is not yet wired in the v1 transform registry.
    """
    kind: Literal["head_pruning"] = "head_pruning"
    selections: list[LayerHeadSelection] = field(default_factory=list)
    mode: Literal["mask"] = "mask"

    def __post_init__(self) -> None:
        if self.mode != "mask":
            raise ValueError("Only mask mode is supported in v1")
        if not self.selections:
            raise ValueError("selections must contain at least one layer/head selection")
        layers = [selection.layer for selection in self.selections]
        if len(layers) != len(set(layers)):
            raise ValueError("selections must not contain duplicate layer entries")
        self.selections = sorted(self.selections, key=lambda selection: selection.layer)


@dataclass(eq=True)
class LinearAttentionSpec:
    """Instruction to swap the attention implementation with a linear-complexity variant.

    Either target_layers or apply_to_all=True must be set, but not both.
    """
    kind: Literal["linear_attention"] = "linear_attention"
    implementation: str = ""
    target_layers: list[int] = field(default_factory=list)
    apply_to_all: bool = False
    mode: Literal["both", "decode_only"] = "both"

    def __post_init__(self) -> None:
        if not self.implementation:
            raise ValueError("implementation must be a non-empty string")
        if self.mode not in ("both", "decode_only"):
            raise ValueError("mode must be 'both' or 'decode_only'")
        if self.apply_to_all and self.target_layers:
            raise ValueError("target_layers must be empty when apply_to_all=True")
        if not self.apply_to_all and not self.target_layers:
            raise ValueError("target_layers must be provided when apply_to_all=False")
        if self.target_layers:
            self.target_layers = _canonicalize_layers(
                self.target_layers,
                field_name="target_layers",
            )


@dataclass(eq=True)
class MlpPruningSpec:
    kind: Literal["mlp_pruning"] = "mlp_pruning"
    target_layers: list[int] = field(default_factory=list)
    pruning_ratio: float = 0.2
    metric: Literal["activation_norm", "wanda"] = "activation_norm"

    def __post_init__(self):
        self.target_layers = _canonicalize_target_layers(self.target_layers)
        if not 0.0 < self.pruning_ratio <= 0.5:
            raise ValueError("pruning_ratio must be between 0 (exclusive) and 0.5 (inclusive)")
        if self.metric not in ("activation_norm", "wanda"):
            raise ValueError(f"metric must be 'activation_norm' or 'wanda', got {self.metric!r}")


@dataclass(eq=True)
class KvCacheCompressionSpec:
    kind: Literal["kv_cache_compression"] = "kv_cache_compression"
    target_layers: list[int] = field(default_factory=list)
    merge_ratio: float = 0.5
    similarity_metric: Literal["cosine", "l2"] = "cosine"
    allow_mha_to_gqa: bool = False

    def __post_init__(self):
        self.target_layers = _canonicalize_target_layers(self.target_layers)
        if not 0.0 < self.merge_ratio < 1.0:
            raise ValueError("merge_ratio must be between 0 and 1 (exclusive)")


@dataclass(eq=True)
class StructuredSparsitySpec:
    kind: Literal["structured_sparsity"] = "structured_sparsity"
    target_layers: list[int] = field(default_factory=list)
    pattern: Literal["2:4"] = "2:4"
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    def __post_init__(self):
        self.target_layers = _canonicalize_target_layers(self.target_layers)
        if self.pattern != "2:4":
            raise ValueError("Only '2:4' pattern is supported")
        if not self.target_modules:
            raise ValueError("target_modules must contain at least one module name")


TransformSpec = Union[
    SkipLayersSpec,
    RemoveLayersSpec,
    CompensationSpec,
    HeadPruningSpec,
    LinearAttentionSpec,
    MlpPruningSpec,
    KvCacheCompressionSpec,
    StructuredSparsitySpec,
]


@dataclass(eq=True)
class TransformationPlan:
    """Ordered list of transforms to apply sequentially to a loaded model.

    Transforms are applied left-to-right by :class:.
    Order matters: a :class: must follow the
    :class: it compensates for, because the compensation
    transform reads the most-recently-applied skip record from the artifact.

    compatibility_mode controls rollback behaviour:
    - "strict" (default): any transform failure raises immediately.
    - "best_effort": recognised by the spec; enforcement is up to the applier.
    """
    transforms: list[TransformSpec] = field(default_factory=list)
    compatibility_mode: Literal["strict", "best_effort"] = "strict"

    def __post_init__(self) -> None:
        if self.compatibility_mode not in ALLOWED_COMPATIBILITY_MODES:
            raise ValueError(
                f"compatibility_mode must be one of {sorted(ALLOWED_COMPATIBILITY_MODES)}"
            )


def _compensation_config_to_dict(config: CompensationConfig) -> dict[str, Any]:
    return asdict(config)


def _compensation_config_from_dict(payload: dict[str, Any]) -> CompensationConfig:
    strategy = payload.get("strategy")
    if strategy == "scaled":
        return ScaledCompensationConfig(
            mean_delta_path=str(payload["mean_delta_path"]),
            alpha=float(payload.get("alpha", 1.0)),
        )
    if strategy == "last_token":
        return LastTokenCompensationConfig(
            mean_delta_path=str(payload["mean_delta_path"]),
            alpha=float(payload.get("alpha", 1.0)),
        )
    if strategy == "magnitude_preserving":
        return MagnitudePreservingCompensationConfig(
            mean_delta_path=str(payload["mean_delta_path"]),
            alpha=float(payload.get("alpha", 1.0)),
        )
    if strategy == "cascaded":
        return CascadedCompensationConfig(
            mean_delta_path=str(payload["mean_delta_path"]),
            pre_skip_fraction=float(payload.get("pre_skip_fraction", 0.5)),
            alpha=float(payload.get("alpha", 1.0)),
        )
    if strategy == "magnitude_rescaling":
        return MagnitudeRescalingCompensationConfig(
            mean_delta_path=str(payload["mean_delta_path"]),
            norm_ratio=float(payload.get("norm_ratio", 1.0)),
            alpha=float(payload.get("alpha", 1.0)),
        )
    if strategy == "phase_aware":
        return PhaseAwareCompensationConfig(
            prefill_delta_path=str(payload["prefill_delta_path"]),
            decode_delta_path=str(payload["decode_delta_path"]),
            prefill_alpha=float(payload.get("prefill_alpha", 1.0)),
            decode_alpha=float(payload.get("decode_alpha", 1.0)),
        )
    if strategy == "phase_last_token":
        return PhaseAwareLastTokenCompensationConfig(
            prefill_delta_path=str(payload["prefill_delta_path"]),
            decode_delta_path=str(payload["decode_delta_path"]),
            prefill_alpha=float(payload.get("prefill_alpha", 1.0)),
            decode_alpha=float(payload.get("decode_alpha", 1.0)),
        )
    if strategy == "phase_aware_magnitude_rescaling":
        return PhaseAwareMagnitudeRescalingCompensationConfig(
            prefill_norm_ratio=float(payload.get("prefill_norm_ratio", 1.0)),
            decode_norm_ratio=float(payload.get("decode_norm_ratio", 1.0)),
        )
    if strategy == "position_aware":
        return PositionAwareCompensationConfig(
            bucket_deltas_path=str(payload["bucket_deltas_path"]),
            num_buckets=int(payload.get("num_buckets", 10)),
            alpha=float(payload.get("alpha", 1.0)),
            fallback_delta_path=payload.get("fallback_delta_path"),
        )
    if strategy == "pca":
        return PcaCompensationConfig(
            pca_path=str(payload["pca_path"]),
            n_components=int(payload.get("n_components", 32)),
            alpha=float(payload.get("alpha", 1.0)),
            mean_delta_path=payload.get("mean_delta_path"),
        )
    if strategy == "multiplicative":
        return MultiplicativeCompensationConfig(
            scale_vector_path=str(payload["scale_vector_path"]),
            bias_vector_path=str(payload["bias_vector_path"]),
        )
    if strategy == "learnable":
        return LearnableCompensationConfig(
            model_path=str(payload["model_path"]),
        )
    if strategy == "multi_phase_aware_magnitude_rescaling":
        return MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig(
            layer_ratios=[
                CompensationLayerRatio(
                    layer=int(item["layer"]),
                    prefill_norm_ratio=float(item.get("prefill_norm_ratio", 1.0)),
                    decode_norm_ratio=float(item.get("decode_norm_ratio", 1.0)),
                )
                for item in payload.get("layer_ratios", [])
            ],
            default_prefill_norm_ratio=float(
                payload.get("default_prefill_norm_ratio", 1.0)
            ),
            default_decode_norm_ratio=float(
                payload.get("default_decode_norm_ratio", 1.0)
            ),
        )
    raise ValueError(f"Unsupported compensation strategy: {strategy}")


def transform_spec_to_dict(spec: TransformSpec) -> dict[str, Any]:
    """Serialise any :class: subclass to a JSON-compatible dict."""
    if isinstance(spec, SkipLayersSpec):
        return {"kind": spec.kind, "layers": list(spec.layers)}
    if isinstance(spec, RemoveLayersSpec):
        return {"kind": spec.kind, "layers": list(spec.layers)}
    if isinstance(spec, CompensationSpec):
        return {"kind": spec.kind, "config": _compensation_config_to_dict(spec.config)}
    if isinstance(spec, HeadPruningSpec):
        return {
            "kind": spec.kind,
            "selections": [
                {"layer": selection.layer, "heads": list(selection.heads)}
                for selection in spec.selections
            ],
            "mode": spec.mode,
        }
    if isinstance(spec, LinearAttentionSpec):
        return {
            "kind": spec.kind,
            "implementation": spec.implementation,
            "target_layers": list(spec.target_layers),
            "apply_to_all": spec.apply_to_all,
            "mode": spec.mode,
        }
    if isinstance(spec, MlpPruningSpec):
        return {
            "kind": spec.kind,
            "target_layers": list(spec.target_layers),
            "pruning_ratio": spec.pruning_ratio,
            "metric": spec.metric,
        }
    if isinstance(spec, KvCacheCompressionSpec):
        return {
            "kind": spec.kind,
            "target_layers": list(spec.target_layers),
            "merge_ratio": spec.merge_ratio,
            "similarity_metric": spec.similarity_metric,
            "allow_mha_to_gqa": spec.allow_mha_to_gqa,
        }
    if isinstance(spec, StructuredSparsitySpec):
        return {
            "kind": spec.kind,
            "target_layers": list(spec.target_layers),
            "pattern": spec.pattern,
            "target_modules": list(spec.target_modules),
        }
    raise TypeError(f"Unsupported transform spec type: {type(spec)!r}")


def transform_spec_from_dict(payload: dict[str, Any]) -> TransformSpec:
    """Deserialise a plain dict back into the matching :class: subclass."""
    kind = payload.get("kind")
    if kind == "skip_layers":
        return SkipLayersSpec(layers=list(payload.get("layers", [])))
    if kind == "remove_layers":
        return RemoveLayersSpec(layers=list(payload.get("layers", [])))
    if kind == "compensation":
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("compensation payload requires a config object")
        return CompensationSpec(config=_compensation_config_from_dict(config))
    if kind == "head_pruning":
        selections = [
            LayerHeadSelection(layer=item["layer"], heads=list(item.get("heads", [])))
            for item in payload.get("selections", [])
        ]
        return HeadPruningSpec(selections=selections, mode=str(payload.get("mode", "mask")))
    if kind == "linear_attention":
        return LinearAttentionSpec(
            implementation=str(payload["implementation"]),
            target_layers=list(payload.get("target_layers", [])),
            apply_to_all=bool(payload.get("apply_to_all", False)),
            mode=str(payload.get("mode", "both")),
        )
    if kind == "mlp_pruning":
        return MlpPruningSpec(
            target_layers=list(payload.get("target_layers", [])),
            pruning_ratio=float(payload.get("pruning_ratio", 0.2)),
            metric=str(payload.get("metric", "activation_norm")),
        )
    if kind == "kv_cache_compression":
        return KvCacheCompressionSpec(
            target_layers=list(payload.get("target_layers", [])),
            merge_ratio=float(payload.get("merge_ratio", 0.5)),
            similarity_metric=str(payload.get("similarity_metric", "cosine")),
            allow_mha_to_gqa=bool(payload.get("allow_mha_to_gqa", False)),
        )
    if kind == "structured_sparsity":
        return StructuredSparsitySpec(
            target_layers=list(payload.get("target_layers", [])),
            pattern=str(payload.get("pattern", "2:4")),
            target_modules=list(payload.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])),
        )
    raise ValueError(f"Unsupported transform kind: {kind}")


def plan_to_dict(plan: TransformationPlan) -> dict[str, Any]:
    """Serialise a :class: to a JSON-compatible dict."""
    return {
        "transforms": [transform_spec_to_dict(spec) for spec in plan.transforms],
        "compatibility_mode": plan.compatibility_mode,
    }


def plan_from_dict(payload: dict[str, Any]) -> TransformationPlan:
    """Deserialise a plain dict back into a :class:."""
    transforms = [
        transform_spec_from_dict(item)
        for item in payload.get("transforms", [])
    ]
    return TransformationPlan(
        transforms=transforms,
        compatibility_mode=str(payload.get("compatibility_mode", "strict")),
    )
