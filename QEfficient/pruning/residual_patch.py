# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LinearResidualPatch(nn.Module):
    """Training-free full linear residual correction for skipped decoder layers."""

    def __init__(self, hidden_size: int, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.alpha.to(dtype=hidden_states.dtype) * self.linear(hidden_states)


class PatchedDecoderLayer(nn.Module):
    """Apply a residual patch before dispatching to the next surviving decoder layer."""

    def __init__(self, patch: nn.Module, original_layer: nn.Module, injection_layer: int):
        super().__init__()
        self.patch = patch
        self.original_layer = original_layer
        self.injection_layer = injection_layer
        self.qeff_layer_skip_compensation_enabled = True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = self.patch(hidden_states)
        return self.original_layer(hidden_states, *args, **kwargs)


def load_linear_residual_patch(
    patch_weights: str | Path,
    hidden_size: int,
    *,
    alpha: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LinearResidualPatch:
    payload = torch.load(patch_weights, map_location="cpu")
    state_dict = _extract_state_dict(payload)

    patch = LinearResidualPatch(hidden_size=hidden_size, alpha=alpha)
    patch.load_state_dict(state_dict, strict=False)
    if device is not None or dtype is not None:
        patch = patch.to(device=device, dtype=dtype)
    patch.eval()
    return patch


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("linear residual patch weights must be a state_dict or a dict containing state_dict.")

    if "linear.weight" in payload:
        return payload
    if "weight" in payload:
        return {"linear.weight": payload["weight"]}
    raise KeyError("linear residual patch weights must contain 'linear.weight' or 'weight'.")
