"""Per-channel importance scoring for MLP width pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import load_dataset_samples
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(eq=True)
class ChannelImportanceReport:
    """Per-layer-per-channel importance scores for MLP pruning.

    ``per_layer_scores[layer_idx]`` is a list of floats indexed by channel
    (``scores[channel_idx] = importance``).  Lower values indicate weaker
    channels that are better pruning candidates.
    """

    model_id: str
    num_layers: int
    intermediate_size: int
    # scores[layer_idx] = list of floats, one per channel, ascending (weakest first)
    per_layer_scores: dict[int, list[float]]
    metric: str = "activation_norm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "per_layer_scores": {
                str(k): list(v) for k, v in self.per_layer_scores.items()
            },
            "metric": self.metric,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelImportanceReport":
        """Deserialise from a plain dict (e.g. loaded from a manifest)."""
        return cls(
            model_id=str(payload["model_id"]),
            num_layers=int(payload["num_layers"]),
            intermediate_size=int(payload["intermediate_size"]),
            per_layer_scores={
                int(k): [float(x) for x in v]
                for k, v in payload["per_layer_scores"].items()
            },
            metric=str(payload.get("metric", "activation_norm")),
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(artifact: ModelArtifact) -> torch.device:
    """Return the device the model parameters reside on."""
    params = list(artifact.model.parameters())
    if params:
        return params[0].device
    return torch.device("cpu")


def _run_tokenize(tokenizer, prompts: list[str], max_length: int, device: torch.device):
    """Tokenise a batch of prompts and move to device."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    return inputs


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------


def _compute_activation_norm_scores(
    artifact: ModelArtifact,
    all_prompts: list[str],
    batch_size: int,
    max_length: int,
) -> dict[int, list[float]]:
    """Compute per-channel importance using mean absolute activation norm.

    For each layer, hooks the ``down_proj`` input (the SwiGLU intermediate
    activation) and accumulates ``|activation|`` averaged over tokens and
    samples.  Returns per-layer scores sorted ascending (weakest first).
    """
    device = _resolve_device(artifact)
    adapter = resolve_layer_adapter(artifact.model)
    num_layers = adapter.num_layers

    # Resolve anatomy once to get intermediate_size
    anatomy_0 = resolve_layer_anatomy(artifact.model, 0)
    intermediate_size = anatomy_0.intermediate_size

    # Accumulators: per-layer tensor of shape [intermediate_size]
    accumulators: dict[int, torch.Tensor] = {
        i: torch.zeros(intermediate_size, device=device, dtype=torch.float32)
        for i in range(num_layers)
    }
    sample_count = 0

    # Register hooks on down_proj for all layers
    hooks = []
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    for layer_idx in range(num_layers):
        anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
        down = anatomy.down_proj
        if down is None:
            continue

        def _make_hook(idx):
            def hook_fn(module, input, output):
                # input[0] is the intermediate activation: [B, S, intermediate_size]
                captured[idx].append(input[0].detach().float())
            return hook_fn

        hooks.append(down.register_forward_hook(_make_hook(layer_idx)))

    try:
        for start in range(0, len(all_prompts), batch_size):
            batch = all_prompts[start : start + batch_size]
            inputs = _run_tokenize(
                artifact.tokenizer, batch, max_length, device
            )
            with torch.no_grad():
                artifact.model(**inputs)

            # Accumulate from captured
            for layer_idx in range(num_layers):
                for tensor in captured[layer_idx]:
                    # Mean over batch and sequence dims -> [intermediate_size]
                    accumulators[layer_idx] += tensor.abs().mean(dim=(0, 1))
                captured[layer_idx].clear()

            sample_count += len(batch)
    finally:
        for h in hooks:
            h.remove()

    num_forward_passes = sum(1 for start in range(0, len(all_prompts), batch_size))
    per_layer_scores: dict[int, list[float]] = {}
    for layer_idx in range(num_layers):
        avg = accumulators[layer_idx] / max(num_forward_passes, 1)
        per_layer_scores[layer_idx] = avg.tolist()

    return per_layer_scores


def _compute_wanda_scores(
    artifact: ModelArtifact,
    all_prompts: list[str],
    batch_size: int,
    max_length: int,
) -> dict[int, list[float]]:
    """Compute per-channel importance using the Wanda metric.

    For each channel c:
        score_c = |down_proj.weight[:, c]|.mean() * |activation[:, :, c]|.mean()

    The activation norm component is accumulated over all batches, then
    combined with the weight magnitude.
    """
    device = _resolve_device(artifact)
    adapter = resolve_layer_adapter(artifact.model)
    num_layers = adapter.num_layers

    anatomy_0 = resolve_layer_anatomy(artifact.model, 0)
    intermediate_size = anatomy_0.intermediate_size

    # Accumulators for activation norms
    act_accumulators: dict[int, torch.Tensor] = {
        i: torch.zeros(intermediate_size, device=device, dtype=torch.float32)
        for i in range(num_layers)
    }
    sample_count = 0

    hooks = []
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    for layer_idx in range(num_layers):
        anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
        down = anatomy.down_proj
        if down is None:
            continue

        def _make_hook(idx):
            def hook_fn(module, input, output):
                captured[idx].append(input[0].detach().float())
            return hook_fn

        hooks.append(down.register_forward_hook(_make_hook(layer_idx)))

    try:
        for start in range(0, len(all_prompts), batch_size):
            batch = all_prompts[start : start + batch_size]
            inputs = _run_tokenize(
                artifact.tokenizer, batch, max_length, device
            )
            with torch.no_grad():
                artifact.model(**inputs)

            for layer_idx in range(num_layers):
                for tensor in captured[layer_idx]:
                    act_accumulators[layer_idx] += tensor.abs().mean(dim=(0, 1))
                captured[layer_idx].clear()

            sample_count += len(batch)
    finally:
        for h in hooks:
            h.remove()

    num_forward_passes = sum(1 for start in range(0, len(all_prompts), batch_size))
    per_layer_scores: dict[int, list[float]] = {}
    for layer_idx in range(num_layers):
        anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
        if anatomy.down_proj is None:
            continue
        down_weight = anatomy.down_proj.weight.detach().float()
        # down_weight shape: [hidden_size, intermediate_size]
        weight_norm = down_weight.abs().mean(dim=0)  # [intermediate_size]
        act_norm = act_accumulators[layer_idx] / num_forward_passes
        wanda_scores = weight_norm * act_norm  # [intermediate_size]
        per_layer_scores[layer_idx] = wanda_scores.tolist()

    return per_layer_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_channel_importance(
    artifact: ModelArtifact,
    datasets: list[str],
    num_samples: int = 50,
    batch_size: int = 4,
    max_length: int = 256,
    metric: Literal["activation_norm", "wanda"] = "activation_norm",
) -> ChannelImportanceReport:
    """Compute per-channel importance scores for MLP width pruning.

    Parameters
    ----------
    artifact:
        A loaded ``ModelArtifact`` (model + tokenizer + metadata).
    datasets:
        List of dataset names (keys in ``SUPPORTED_DATASETS``).
    num_samples:
        Number of prompt samples per dataset.
    batch_size:
        Forward-pass batch size.
    max_length:
        Maximum token length for prompt truncation.
    metric:
        Scoring strategy. ``"activation_norm"`` uses mean absolute
        activation of the SwiGLU intermediate.  ``"wanda"`` multiplies
        the weight magnitude of the down-projection column by the
        activation norm (Sun et al., 2024).

    Returns
    -------
    ChannelImportanceReport
        Per-layer channel scores sorted ascending (weakest first).
    """
    if not datasets:
        raise ValueError("datasets must contain at least one entry")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Gather prompts from all datasets
    all_prompts: list[str] = []
    for ds_name in datasets:
        all_prompts.extend(load_dataset_samples(ds_name, num_samples))

    adapter = resolve_layer_adapter(artifact.model)
    num_layers = adapter.num_layers
    anatomy_0 = resolve_layer_anatomy(artifact.model, 0)
    intermediate_size = anatomy_0.intermediate_size

    if metric == "activation_norm":
        per_layer_scores = _compute_activation_norm_scores(
            artifact, all_prompts, batch_size, max_length,
        )
    elif metric == "wanda":
        per_layer_scores = _compute_wanda_scores(
            artifact, all_prompts, batch_size, max_length,
        )
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    return ChannelImportanceReport(
        model_id=artifact.model_spec.model_id,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        per_layer_scores=per_layer_scores,
        metric=metric,
        metadata={
            "datasets": list(datasets),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "max_length": max_length,
        },
    )


__all__ = [
    "ChannelImportanceReport",
    "compute_channel_importance",
]
