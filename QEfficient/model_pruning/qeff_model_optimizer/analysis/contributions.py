"""Pure-tensor primitives for per-layer contribution deltas."""

from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn.functional as F


SupportedMetric = Literal["cosine", "l2"]


def compute_per_layer_deltas(
    hidden_states: Iterable[torch.Tensor],
    metric: SupportedMetric,
    mask: torch.Tensor | None = None,
) -> list[float]:
    """Return mean delta per decoder block, averaged over batch and sequence dims.

    When *mask* is provided (shape ``[B, T]``, 1 for real tokens, 0 for padding),
    only non-padding positions contribute to the per-layer average.
    """
    tensors = list(hidden_states)
    if len(tensors) < 2:
        raise ValueError(
            "hidden_states must have at least 2 entries (embedding + one block)"
        )

    shapes = {t.shape for t in tensors}
    if len(shapes) > 1:
        raise ValueError("hidden_state shapes must match across all layers")

    stacked = torch.stack(tensors)         # [L+1, B, T, H]
    h_prev, h_curr = stacked[:-1], stacked[1:]

    if metric == "cosine":
        per_token = 1.0 - F.cosine_similarity(h_curr, h_prev, dim=-1)  # [L, B, T]
    elif metric == "l2":
        per_token = torch.norm(h_curr - h_prev, p=2, dim=-1)            # [L, B, T]
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    if mask is not None:
        mask_expanded = mask.unsqueeze(0).float()
        per_token = per_token * mask_expanded
        valid_count = mask_expanded.sum(dim=(1, 2)).clamp(min=1)
        return (per_token.sum(dim=(1, 2)) / valid_count).tolist()
    return per_token.mean(dim=(1, 2)).tolist()   # one float per layer, one GPU sync


def aggregate_layer_scores(
    per_sample_deltas: list[list[float]],
) -> list[dict[str, float]]:
    """Reduce per-sample layer-delta rows into per-layer avg/std/min/max stats."""
    if not per_sample_deltas:
        raise ValueError("per_sample_deltas must not be empty")
    num_layers = len(per_sample_deltas[0])
    if any(len(row) != num_layers for row in per_sample_deltas):
        raise ValueError("per-sample delta rows must have matching length")

    m = torch.tensor(per_sample_deltas, dtype=torch.float64)  # [N, L]
    avgs = m.mean(0)
    stds = m.std(0, unbiased=False)
    mins = m.min(0).values
    maxs = m.max(0).values

    return [
        {
            "avg": float(avgs[i]),
            "std": float(stds[i]),
            "min": float(mins[i]),
            "max": float(maxs[i]),
        }
        for i in range(num_layers)
    ]
