"""Per-head importance scoring for attention head pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import load_dataset_samples
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy


@dataclass(eq=True)
class HeadImportanceReport:
    """Per-layer-per-head importance scores, ranked weakest-first.

    ``per_layer_scores[layer_idx]`` is a list of ``(head_idx, score)`` tuples
    sorted by score ascending (weakest first).  This ordering mirrors
    :class:`~nas.analysis.reports.WeakLayerReport` so that callers can slice
    off the *N* weakest heads directly.
    """

    model_id: str
    num_layers: int
    num_heads: int
    # scores[layer_idx] = list of (head_idx, score) sorted ascending (weakest first)
    per_layer_scores: dict[int, list[tuple[int, float]]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "per_layer_scores": {
                str(k): [[h, s] for h, s in v]
                for k, v in self.per_layer_scores.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HeadImportanceReport":
        """Deserialise from a plain dict (e.g. loaded from a manifest)."""
        return cls(
            model_id=str(payload["model_id"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            per_layer_scores={
                int(k): [(int(h), float(s)) for h, s in v]
                for k, v in payload["per_layer_scores"].items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(artifact: ModelArtifact) -> torch.device:
    """Return the device the model parameters live on."""
    params = list(artifact.model.parameters())
    if params:
        return params[0].device
    return torch.device("cpu")


def _run_forward_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: int,
) -> None:
    """Tokenise *prompts* and run a forward pass (hooks capture the data)."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_head_importance(
    artifact: ModelArtifact,
    datasets: list[str],
    num_samples: int = 50,
    batch_size: int = 4,
    max_length: int = 256,
) -> HeadImportanceReport:
    """Compute per-head importance scores by measuring activation norms.

    For every decoder layer the function registers a forward hook on the
    ``o_proj`` linear layer.  The hook captures the *input* to ``o_proj``
    (i.e. the concatenated per-head outputs before the final projection),
    reshapes it to ``[B, S, num_heads, head_dim]``, and computes the
    mean L2 norm across the batch and sequence dimensions for each head.

    The scores are averaged over all calibration samples drawn from the
    requested *datasets*.  Heads with lower scores contribute less to the
    residual stream and are therefore better pruning candidates.

    Parameters
    ----------
    artifact:
        A loaded :class:`~nas.config.artifacts.ModelArtifact`.
    datasets:
        Dataset names recognised by :func:`~nas.analysis.datasets.load_dataset_samples`.
    num_samples:
        Number of prompt samples to draw from *each* dataset.
    batch_size:
        Forward-pass batch size.
    max_length:
        Maximum token length for tokenisation.

    Returns
    -------
    HeadImportanceReport
        Report with per-layer head scores sorted weakest-first.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one entry")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = _resolve_device(artifact)
    adapter = resolve_layer_adapter(artifact.model)
    num_layers = adapter.num_layers

    # Resolve anatomy from layer 0 to get structural info
    anatomy_0 = resolve_layer_anatomy(artifact.model, 0)
    num_heads = anatomy_0.num_heads
    head_dim = anatomy_0.head_dim

    # Accumulators: layer_idx -> list of per-head norm sums (one float per head)
    head_norm_accum: dict[int, list[float]] = {
        i: [0.0] * num_heads for i in range(num_layers)
    }
    total_samples = 0

    # ------------------------------------------------------------------
    # Register hooks on every o_proj
    # ------------------------------------------------------------------
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(layer_idx: int):
        """Return a forward hook that accumulates per-head L2 norms."""

        def hook_fn(module, input, output):
            # input is a tuple; input[0] is the pre-projection tensor
            # shaped [B, S, num_heads * head_dim]
            inp = input[0]
            B, S, _ = inp.shape
            reshaped = inp.view(B, S, num_heads, head_dim)
            for h in range(num_heads):
                head_slice = reshaped[:, :, h, :]  # [B, S, head_dim]
                norm_val = torch.norm(head_slice, dim=-1).mean().item()
                head_norm_accum[layer_idx][h] += norm_val * B

        return hook_fn

    try:
        for layer_idx in range(num_layers):
            anatomy = resolve_layer_anatomy(artifact.model, layer_idx)
            hook = anatomy.o_proj.register_forward_hook(_make_hook(layer_idx))
            hooks.append(hook)

        # ------------------------------------------------------------------
        # Run calibration samples through the model
        # ------------------------------------------------------------------
        for dataset in datasets:
            prompts = load_dataset_samples(dataset, num_samples)
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                _run_forward_batch(
                    artifact.model,
                    artifact.tokenizer,
                    batch,
                    device,
                    max_length,
                )
                total_samples += len(batch)
    finally:
        # Always remove hooks, even if an error occurred
        for hook in hooks:
            hook.remove()

    # ------------------------------------------------------------------
    # Average and sort
    # ------------------------------------------------------------------
    if total_samples == 0:
        raise RuntimeError("No samples were successfully processed")

    per_layer_scores: dict[int, list[tuple[int, float]]] = {}
    for layer_idx in range(num_layers):
        head_scores: list[tuple[int, float]] = []
        for h in range(num_heads):
            avg_score = head_norm_accum[layer_idx][h] / total_samples
            head_scores.append((h, avg_score))
        # Sort by score ascending (weakest first)
        head_scores.sort(key=lambda x: x[1])
        per_layer_scores[layer_idx] = head_scores

    model_id = getattr(artifact.model_spec, "model_id", "unknown")

    return HeadImportanceReport(
        model_id=model_id,
        num_layers=num_layers,
        num_heads=num_heads,
        per_layer_scores=per_layer_scores,
        metadata={
            "datasets": list(datasets),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "max_length": max_length,
            "total_calibration_samples": total_samples,
        },
    )


__all__ = [
    "HeadImportanceReport",
    "compute_head_importance",
]
