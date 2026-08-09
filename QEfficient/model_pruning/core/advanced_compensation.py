#!/usr/bin/env python3
"""
Advanced Compensation Strategies for Layer Skipping

Implements multiple compensation strategies to mitigate accuracy loss
from skipping transformer layers. Each strategy is a drop-in replacement
for the simple mean-vector compensation in CompensatedSkipLayerModel.

Strategies implemented:
  1. ScaledCompensation        — alpha * mean_delta (tunable scale)
  2. LastTokenCompensation     — apply compensation only to last token in prefill
  3. MagnitudePreservingComp   — normalize delta, scale by ||h_start||
  4. MultiplicativeCompensation— scale + bias (captures layer-norm effects)
  5. PCACompensation           — project delta into top-K PCA components
  6. PhaseAwareCompensation    — separate vectors for prefill vs. decode
  7. PositionAwareCompensation — position-bucket-specific vectors
  8. CascadedCompensation      — split compensation across start and end of skip
  9. MagnitudeRescaling        — rescale output norm to match expected h_end norm
 10. ClusterCompensation       — K-means cluster-based input-adaptive compensation

All strategies implement the same interface:
    compensate(h: Tensor, is_decode: bool, token_positions: Optional[Tensor]) -> Tensor

Usage:
    from QEfficient.model_pruning.core.advanced_compensation import (
        ScaledCompensation, PhaseAwareCompensation, PCACompensation
    )

    # Load a strategy
    strategy = PhaseAwareCompensation(
        prefill_vector_file="wikitext_prefill_mean.pt",
        decode_vector_file="wikitext_decode_mean.pt",
    )

    # Apply in a forward hook
    h_compensated = strategy.compensate(h, is_decode=False)

Author: LLM Interpretability Engineer
"""

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MIN_NORM_RATIO = 0.5
MAX_NORM_RATIO = 3.0


def _sanitize_norm_ratio(
    value: float,
    fallback: float = 1.0,
    min_ratio: float = MIN_NORM_RATIO,
    max_ratio: float = MAX_NORM_RATIO,
) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return float(fallback)

    if not math.isfinite(ratio) or ratio <= 0:
        return float(fallback)

    return max(min_ratio, min(max_ratio, ratio))


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseCompensation(ABC):
    """
    Abstract base class for all compensation strategies.

    Subclasses must implement `compensate(h, is_decode, token_positions)`.
    """

    @abstractmethod
    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply compensation to hidden state tensor.

        Args:
            h: Hidden state tensor [batch, seq_len, hidden_dim]
               During decode: [batch, 1, hidden_dim]
            is_decode: True if this is a decode step (single token)
            token_positions: Optional token position indices [batch, seq_len]
                             Used by position-aware strategies

        Returns:
            Compensated hidden state [batch, seq_len, hidden_dim]
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Scaled Compensation
# ─────────────────────────────────────────────────────────────────────────────

class ScaledCompensation(BaseCompensation):
    """
    Applies alpha * mean_delta to all tokens.

    This is the simplest improvement over the baseline (alpha=1.0).
    The optimal alpha can be found via grid search using
    analyze_embedding_delta_patterns.py.

    h_comp = h + alpha * mean_delta
    """

    def __init__(
        self,
        mean_delta: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            mean_delta: Mean delta vector [hidden_dim] or path to .pt file
            alpha: Scale factor (default 1.0 = same as baseline)
            device: Target device
        """
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        self.mean_delta = mean_delta.float()
        self.alpha = alpha
        self._device = device

    def to(self, device: torch.device) -> "ScaledCompensation":
        self.mean_delta = self.mean_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.mean_delta.to(h.device)
        return h + self.alpha * delta.view(1, 1, -1)

    def __repr__(self) -> str:
        return f"ScaledCompensation(alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Last-Token Compensation
# ─────────────────────────────────────────────────────────────────────────────

class LastTokenCompensation(BaseCompensation):
    """
    Applies compensation ONLY to the last token during prefill.
    During decode, applies to the single token (which is always the "last").

    Rationale: The last token's hidden state drives the next-token prediction.
    Compensating only the last token avoids corrupting earlier token representations
    that may be used by later layers for attention.

    h_comp[last] = h[last] + mean_delta
    h_comp[others] = h[others]  (unchanged)
    """

    def __init__(
        self,
        mean_delta: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        self.mean_delta = mean_delta.float()
        self.alpha = alpha
        self._device = device

    def to(self, device: torch.device) -> "LastTokenCompensation":
        self.mean_delta = self.mean_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.mean_delta.to(h.device)
        h_comp = h.clone()
        # Apply only to last token position
        h_comp[:, -1, :] = h[:, -1, :] + self.alpha * delta
        return h_comp

    def __repr__(self) -> str:
        return f"LastTokenCompensation(alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Magnitude-Preserving Compensation
# ─────────────────────────────────────────────────────────────────────────────

class MagnitudePreservingCompensation(BaseCompensation):
    """
    Scales the compensation by the norm of the input hidden state.

    Instead of adding a fixed absolute vector, adds a vector whose magnitude
    is proportional to the current embedding magnitude. This is more robust
    to inputs with different embedding scales.

    h_comp = h + ||h|| * normalize(mean_delta)

    where normalize(v) = v / ||v||
    """

    def __init__(
        self,
        mean_delta: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        self.mean_delta = mean_delta.float()
        # Pre-normalize the direction
        self.delta_direction = F.normalize(self.mean_delta.unsqueeze(0), dim=-1).squeeze(0)
        self.alpha = alpha
        self._device = device

    def to(self, device: torch.device) -> "MagnitudePreservingCompensation":
        self.mean_delta = self.mean_delta.to(device)
        self.delta_direction = self.delta_direction.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        direction = self.delta_direction.to(h.device)
        # Compute per-token norms: [batch, seq_len, 1]
        h_norms = h.float().norm(dim=-1, keepdim=True)
        # Scale direction by norm
        scaled_delta = self.alpha * h_norms * direction.view(1, 1, -1)
        return h + scaled_delta.to(h.dtype)

    def __repr__(self) -> str:
        return f"MagnitudePreservingCompensation(alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Multiplicative + Additive Compensation
# ─────────────────────────────────────────────────────────────────────────────

class MultiplicativeCompensation(BaseCompensation):
    """
    Applies both a scale vector and a bias vector.

    h_comp = h * scale_vector + bias_vector

    The scale_vector captures layer-norm-like multiplicative effects.
    The bias_vector captures the mean shift (same as mean_delta).

    scale_vector = mean(h_end) / mean(h_start)  (element-wise ratio)
    bias_vector  = mean(h_end - h_start * scale_vector)
    """

    def __init__(
        self,
        scale_vector: Union[torch.Tensor, str, Path],
        bias_vector: Union[torch.Tensor, str, Path],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            scale_vector: Element-wise scale [hidden_dim] or path to .pt file
            bias_vector: Additive bias [hidden_dim] or path to .pt file
        """
        if isinstance(scale_vector, (str, Path)):
            scale_vector = torch.load(scale_vector, map_location="cpu")
        if isinstance(bias_vector, (str, Path)):
            bias_vector = torch.load(bias_vector, map_location="cpu")
        self.scale_vector = scale_vector.float()
        self.bias_vector = bias_vector.float()
        self._device = device

    @classmethod
    def from_hidden_states(
        cls,
        h_start_list: List[torch.Tensor],
        h_end_list: List[torch.Tensor],
        eps: float = 1e-8,
    ) -> "MultiplicativeCompensation":
        """
        Compute scale and bias from collected hidden state pairs.

        Args:
            h_start_list: List of h_start tensors [hidden_dim]
            h_end_list: List of h_end tensors [hidden_dim]
        """
        h_start = torch.stack(h_start_list).float()  # [N, D]
        h_end = torch.stack(h_end_list).float()      # [N, D]

        # Element-wise scale: ratio of means
        mean_start = h_start.mean(dim=0)  # [D]
        mean_end = h_end.mean(dim=0)      # [D]
        scale = mean_end / (mean_start.abs() + eps)
        scale = scale.clamp(0.5, 2.0)  # Clip to reasonable range

        # Bias: residual after scaling
        bias = mean_end - scale * mean_start

        return cls(scale_vector=scale, bias_vector=bias)

    @classmethod
    def from_delta_files(
        cls,
        h_start_mean_file: Union[str, Path],
        h_end_mean_file: Union[str, Path],
        eps: float = 1e-8,
    ) -> "MultiplicativeCompensation":
        """Load from pre-computed mean files."""
        mean_start = torch.load(h_start_mean_file, map_location="cpu").float()
        mean_end = torch.load(h_end_mean_file, map_location="cpu").float()
        scale = mean_end / (mean_start.abs() + eps)
        scale = scale.clamp(0.5, 2.0)
        bias = mean_end - scale * mean_start
        return cls(scale_vector=scale, bias_vector=bias)

    def to(self, device: torch.device) -> "MultiplicativeCompensation":
        self.scale_vector = self.scale_vector.to(device)
        self.bias_vector = self.bias_vector.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.scale_vector.to(h.device).view(1, 1, -1)
        bias = self.bias_vector.to(h.device).view(1, 1, -1)
        return (h.float() * scale + bias).to(h.dtype)

    def __repr__(self) -> str:
        return "MultiplicativeCompensation(scale+bias)"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5: PCA-Based Compensation
# ─────────────────────────────────────────────────────────────────────────────

class PCACompensation(BaseCompensation):
    """
    Projects the mean delta into the top-K PCA components of the delta space.

    This filters out noise and focuses compensation on the most consistent
    directions of change. The PCA components are computed by
    analyze_embedding_delta_patterns.py.

    h_comp = h + project_to_pca_subspace(mean_delta, top_k_components)
    """

    def __init__(
        self,
        pca_file: Union[str, Path],
        n_components: int = 32,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            pca_file: Path to .pt file containing PCA components dict:
                      {"components": [n_comp, hidden_dim], "explained_variance_ratio": [n_comp]}
            n_components: Number of PCA components to use
            alpha: Scale factor
        """
        pca_data = torch.load(pca_file, map_location="cpu")
        components = pca_data["components"].float()  # [n_comp, D]
        evr = pca_data["explained_variance_ratio"].float()

        # Use top-n_components
        n_use = min(n_components, components.shape[0])
        self.components = components[:n_use]  # [n_use, D]
        self.evr = evr[:n_use]
        self.n_components = n_use
        self.alpha = alpha
        self._device = device

        total_var = float(evr[:n_use].sum() * 100)
        print(f"PCACompensation: using {n_use} components "
              f"({total_var:.1f}% of delta variance)")

    @classmethod
    def from_mean_delta_and_pca(
        cls,
        mean_delta: Union[torch.Tensor, str, Path],
        pca_file: Union[str, Path],
        n_components: int = 32,
        alpha: float = 1.0,
    ) -> "PCACompensation":
        """Create from separate mean delta and PCA files."""
        obj = cls(pca_file=pca_file, n_components=n_components, alpha=alpha)
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        obj._mean_delta = mean_delta.float()
        return obj

    def to(self, device: torch.device) -> "PCACompensation":
        self.components = self.components.to(device)
        if hasattr(self, "_mean_delta"):
            self._mean_delta = self._mean_delta.to(device)
        self._device = device
        return self

    def _project_to_subspace(self, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto the PCA subspace."""
        # v: [D]
        # components: [n_use, D]
        # projection = sum_i (v · c_i) * c_i
        coeffs = (self.components @ v)  # [n_use]
        projected = (coeffs.unsqueeze(-1) * self.components).sum(dim=0)  # [D]
        return projected

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        components = self.components.to(h.device)

        # If we have a mean delta, project it
        if hasattr(self, "_mean_delta"):
            mean_delta = self._mean_delta.to(h.device)
            # Project mean delta onto PCA subspace
            coeffs = components @ mean_delta  # [n_use]
            projected_delta = (coeffs.unsqueeze(-1) * components).sum(dim=0)  # [D]
        else:
            # Use the first PCA component as the compensation direction
            # weighted by explained variance
            projected_delta = (self.evr.to(h.device).unsqueeze(-1) * components).sum(dim=0)

        return h + self.alpha * projected_delta.view(1, 1, -1).to(h.dtype)

    def __repr__(self) -> str:
        return f"PCACompensation(n_components={self.n_components}, alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6: Phase-Aware Compensation
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareCompensation(BaseCompensation):
    """
    Uses separate compensation vectors for prefill and decode phases.

    The prefill and decode phases have different embedding distributions.
    Using phase-specific vectors addresses the prefill/decode mismatch
    identified in the diagnostic analysis.

    Prefill: h_comp = h + prefill_delta
    Decode:  h_comp = h + decode_delta
    """

    def __init__(
        self,
        prefill_delta: Union[torch.Tensor, str, Path],
        decode_delta: Union[torch.Tensor, str, Path],
        prefill_alpha: float = 1.0,
        decode_alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            prefill_delta: Prefill mean delta [hidden_dim] or path to .pt file
            decode_delta: Decode mean delta [hidden_dim] or path to .pt file
            prefill_alpha: Scale for prefill compensation
            decode_alpha: Scale for decode compensation
        """
        if isinstance(prefill_delta, (str, Path)):
            prefill_delta = torch.load(prefill_delta, map_location="cpu")
        if isinstance(decode_delta, (str, Path)):
            decode_delta = torch.load(decode_delta, map_location="cpu")
        self.prefill_delta = prefill_delta.float()
        self.decode_delta = decode_delta.float()
        self.prefill_alpha = prefill_alpha
        self.decode_alpha = decode_alpha
        self._device = device

    def to(self, device: torch.device) -> "PhaseAwareCompensation":
        self.prefill_delta = self.prefill_delta.to(device)
        self.decode_delta = self.decode_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            delta = self.decode_delta.to(h.device)
            alpha = self.decode_alpha
        else:
            delta = self.prefill_delta.to(h.device)
            alpha = self.prefill_alpha
        return h + alpha * delta.view(1, 1, -1)

    def __repr__(self) -> str:
        return (f"PhaseAwareCompensation("
                f"prefill_alpha={self.prefill_alpha:.3f}, "
                f"decode_alpha={self.decode_alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 7: Position-Aware Compensation
# ─────────────────────────────────────────────────────────────────────────────

class PositionAwareCompensation(BaseCompensation):
    """
    Uses position-bucket-specific compensation vectors.

    Different token positions undergo different transformations in the
    skipped layers. This strategy applies a different compensation vector
    based on the relative position of each token.

    For token at relative position p (0.0 to 1.0):
        bucket = int(p * num_buckets)
        h_comp[pos] = h[pos] + bucket_delta[bucket]
    """

    def __init__(
        self,
        bucket_deltas: Union[Dict[int, torch.Tensor], str, Path],
        num_buckets: int = 10,
        alpha: float = 1.0,
        fallback_delta: Optional[Union[torch.Tensor, str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            bucket_deltas: Dict mapping bucket_idx -> delta_vector [hidden_dim]
                           or path to .pt file containing this dict
            num_buckets: Number of position buckets (default 10)
            alpha: Scale factor
            fallback_delta: Fallback delta for missing buckets (e.g., global mean)
        """
        if isinstance(bucket_deltas, (str, Path)):
            bucket_deltas = torch.load(bucket_deltas, map_location="cpu")
        self.bucket_deltas = {k: v.float() for k, v in bucket_deltas.items()}
        self.num_buckets = num_buckets
        self.alpha = alpha
        self._device = device

        if fallback_delta is not None:
            if isinstance(fallback_delta, (str, Path)):
                fallback_delta = torch.load(fallback_delta, map_location="cpu")
            self.fallback_delta = fallback_delta.float()
        else:
            # Use mean of all bucket deltas as fallback
            if self.bucket_deltas:
                self.fallback_delta = torch.stack(list(self.bucket_deltas.values())).mean(dim=0)
            else:
                self.fallback_delta = None

    def to(self, device: torch.device) -> "PositionAwareCompensation":
        self.bucket_deltas = {k: v.to(device) for k, v in self.bucket_deltas.items()}
        if self.fallback_delta is not None:
            self.fallback_delta = self.fallback_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = h.shape

        if is_decode or seq_len == 1:
            # During decode, treat as last position (bucket = num_buckets - 1)
            bucket = self.num_buckets - 1
            delta = self.bucket_deltas.get(bucket, self.fallback_delta)
            if delta is None:
                return h
            return h + self.alpha * delta.to(h.device).view(1, 1, -1)

        # Prefill: apply position-specific compensation
        h_comp = h.clone()
        for pos in range(seq_len):
            # Relative position in [0, 1)
            rel_pos = pos / seq_len
            bucket = min(int(rel_pos * self.num_buckets), self.num_buckets - 1)
            delta = self.bucket_deltas.get(bucket, self.fallback_delta)
            if delta is not None:
                h_comp[:, pos, :] = h[:, pos, :] + self.alpha * delta.to(h.device)

        return h_comp

    def __repr__(self) -> str:
        return (f"PositionAwareCompensation("
                f"num_buckets={self.num_buckets}, alpha={self.alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 8: Cascaded Compensation
# ─────────────────────────────────────────────────────────────────────────────

class CascadedCompensation(BaseCompensation):
    """
    Splits compensation across two application points:
    - Before the skip (at the compensation layer): apply fraction * delta
    - After the skip (at the first layer after skip): apply (1-fraction) * delta

    This is implemented by registering hooks at two layers instead of one.
    The compensate() method here handles the "before" part; the "after" part
    is handled by a separate hook registered at the post-skip layer.

    Rationale: Applying all compensation before the skip may cause the
    intermediate layers to receive an out-of-distribution input. Splitting
    the compensation may be more stable.
    """

    def __init__(
        self,
        mean_delta: Union[torch.Tensor, str, Path],
        pre_skip_fraction: float = 0.5,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            mean_delta: Mean delta vector [hidden_dim] or path to .pt file
            pre_skip_fraction: Fraction of delta to apply before skip (0.0 to 1.0)
            alpha: Overall scale factor
        """
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        self.mean_delta = mean_delta.float()
        self.pre_skip_fraction = pre_skip_fraction
        self.post_skip_fraction = 1.0 - pre_skip_fraction
        self.alpha = alpha
        self._device = device

    def to(self, device: torch.device) -> "CascadedCompensation":
        self.mean_delta = self.mean_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the pre-skip fraction of compensation."""
        delta = self.mean_delta.to(h.device)
        return h + self.alpha * self.pre_skip_fraction * delta.view(1, 1, -1)

    def compensate_post_skip(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
    ) -> torch.Tensor:
        """Apply the post-skip fraction of compensation."""
        delta = self.mean_delta.to(h.device)
        return h + self.alpha * self.post_skip_fraction * delta.view(1, 1, -1)

    def __repr__(self) -> str:
        return (f"CascadedCompensation("
                f"pre={self.pre_skip_fraction:.2f}, "
                f"post={self.post_skip_fraction:.2f}, "
                f"alpha={self.alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 9: Magnitude Rescaling
# ─────────────────────────────────────────────────────────────────────────────

class MagnitudeRescalingCompensation(BaseCompensation):
    """
    Combines additive compensation with output magnitude rescaling.

    After adding the mean delta, rescales the output so its norm matches
    the expected norm of h_end (the output of the skipped layers).

    h_comp = h + mean_delta
    h_comp = h_comp * (expected_norm / ||h_comp||)

    where expected_norm = mean(||h_end||) from the training data.

    This addresses the case where skipped layers change the embedding magnitude
    (e.g., amplify or compress the signal).
    """

    def __init__(
        self,
        mean_delta: Union[torch.Tensor, str, Path],
        norm_ratio: float = 1.0,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            mean_delta: Mean delta vector [hidden_dim] or path to .pt file
            norm_ratio: Expected ||h_end|| / ||h_start|| ratio
                        (from magnitude_stats in the diagnostic report)
            alpha: Scale factor for the additive delta
        """
        if isinstance(mean_delta, (str, Path)):
            mean_delta = torch.load(mean_delta, map_location="cpu")
        self.mean_delta = mean_delta.float()
        self.norm_ratio = norm_ratio
        self.alpha = alpha
        self._device = device

    def to(self, device: torch.device) -> "MagnitudeRescalingCompensation":
        self.mean_delta = self.mean_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.mean_delta.to(h.device)
        h_float = h.float()

        # Step 1: Add delta
        h_comp = h_float + self.alpha * delta.view(1, 1, -1)

        # Step 2: Rescale to match expected norm
        if abs(self.norm_ratio - 1.0) > 0.01:
            # Current norm of h (before compensation)
            h_norm = h_float.norm(dim=-1, keepdim=True)  # [B, S, 1]
            # Expected norm after skipped layers
            expected_norm = h_norm * self.norm_ratio
            # Current norm of compensated h
            h_comp_norm = h_comp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            # Rescale
            h_comp = h_comp * (expected_norm / h_comp_norm)

        return h_comp.to(h.dtype)

    def __repr__(self) -> str:
        return (f"MagnitudeRescalingCompensation("
                f"norm_ratio={self.norm_ratio:.3f}, alpha={self.alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 9b: Phase-Aware Magnitude Rescaling (no additive delta)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareMagnitudeRescalingCompensation(BaseCompensation):
    """
    Rescales embedding norms using SEPARATE ratios for prefill and decode.

    No additive delta is applied — only the norm is rescaled.
    This is the recommended strategy when:
      - The global mean delta explains <10% of variance (best alpha ≈ 0)
      - Prefill and decode have significantly different norm ratios

    Prefill: h_comp = h * (prefill_norm_ratio * ||h|| / ||h||)
                     = h scaled so ||h_comp|| = prefill_norm_ratio * ||h||
    Decode:  h_comp = h scaled so ||h_comp|| = decode_norm_ratio * ||h||

    Norm ratios come from the diagnostic report's magnitude_stats:
        prefill_norm_ratio = mean(||h_end|| / ||h_start||) for prefill
        decode_norm_ratio  = mean(||h_end|| / ||h_start||) for decode
    """

    def __init__(
        self,
        prefill_norm_ratio: float = 1.0,
        decode_norm_ratio: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            prefill_norm_ratio: ||h_end|| / ||h_start|| for prefill phase
            decode_norm_ratio:  ||h_end|| / ||h_start|| for decode phase
        """
        self.prefill_norm_ratio = prefill_norm_ratio
        self.decode_norm_ratio = decode_norm_ratio
        self._device = device

    def to(self, device: torch.device) -> "PhaseAwareMagnitudeRescalingCompensation":
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = self.decode_norm_ratio if is_decode else self.prefill_norm_ratio
        if abs(ratio - 1.0) < 0.005:
            return h  # No-op if ratio is essentially 1.0
        h_float = h.float()
        h_norm = h_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
        return (h_float * ratio).to(h.dtype)

    def __repr__(self) -> str:
        return (f"PhaseAwareMagnitudeRescalingCompensation("
                f"prefill={self.prefill_norm_ratio:.3f}, "
                f"decode={self.decode_norm_ratio:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 9c: Multi-Cluster Phase-Aware Magnitude Rescaling
# ─────────────────────────────────────────────────────────────────────────────

class MultiClusterPhaseAwareMagnitudeRescalingCompensation(BaseCompensation):
    """
    Layer-aware norm rescaling for multiple skip clusters.

    Each compensation hook layer can have its own prefill/decode norm ratios.
    This enables applying separate phase-aware magnitude rescaling at each
    cluster boundary when skip layers are non-contiguous.
    """

    def __init__(
        self,
        layer_ratios: Dict[Union[int, str], Union[Dict[str, float], List[float], tuple]],
        default_prefill_norm_ratio: float = 1.0,
        default_decode_norm_ratio: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.default_prefill_norm_ratio = _sanitize_norm_ratio(default_prefill_norm_ratio)
        self.default_decode_norm_ratio = _sanitize_norm_ratio(default_decode_norm_ratio)
        self._device = device

        parsed: Dict[int, tuple] = {}
        for raw_layer, raw_value in layer_ratios.items():
            layer_idx = int(raw_layer)
            if isinstance(raw_value, dict):
                prefill = float(
                    raw_value.get(
                        "prefill_norm_ratio",
                        raw_value.get("prefill", self.default_prefill_norm_ratio),
                    )
                )
                decode = float(
                    raw_value.get(
                        "decode_norm_ratio",
                        raw_value.get("decode", self.default_decode_norm_ratio),
                    )
                )
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                prefill = float(raw_value[0])
                decode = float(raw_value[1])
            else:
                raise ValueError(
                    "layer_ratios values must be dicts with prefill/decode keys "
                    "or [prefill, decode] pairs"
                )
            parsed[layer_idx] = (
                _sanitize_norm_ratio(prefill, fallback=self.default_prefill_norm_ratio),
                _sanitize_norm_ratio(decode, fallback=self.default_decode_norm_ratio),
            )

        if not parsed:
            raise ValueError("layer_ratios cannot be empty for multi-cluster compensation")

        self.layer_ratios = parsed

    def to(self, device: torch.device) -> "MultiClusterPhaseAwareMagnitudeRescalingCompensation":
        self._device = device
        return self

    def _ratio_for(self, layer_idx: Optional[int], is_decode: bool) -> float:
        if layer_idx is not None and layer_idx in self.layer_ratios:
            prefill_ratio, decode_ratio = self.layer_ratios[layer_idx]
            ratio = decode_ratio if is_decode else prefill_ratio
        else:
            ratio = self.default_decode_norm_ratio if is_decode else self.default_prefill_norm_ratio
        fallback = self.default_decode_norm_ratio if is_decode else self.default_prefill_norm_ratio
        return _sanitize_norm_ratio(ratio, fallback=fallback)

    def compensate_at_layer(
        self,
        h: torch.Tensor,
        layer_idx: int,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = self._ratio_for(layer_idx=layer_idx, is_decode=is_decode)
        if abs(ratio - 1.0) < 0.005:
            return h
        return (h.float() * ratio).to(h.dtype)

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fallback path when no layer index is provided.
        ratio = self._ratio_for(layer_idx=None, is_decode=is_decode)
        if abs(ratio - 1.0) < 0.005:
            return h
        return (h.float() * ratio).to(h.dtype)

    def __repr__(self) -> str:
        return (
            "MultiClusterPhaseAwareMagnitudeRescalingCompensation("
            f"layers={len(self.layer_ratios)}, "
            f"default_prefill={self.default_prefill_norm_ratio:.3f}, "
            f"default_decode={self.default_decode_norm_ratio:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 10: Cluster-Based Compensation
# ─────────────────────────────────────────────────────────────────────────────

class ClusterCompensation(BaseCompensation):
    """
    Input-adaptive compensation using K-means clustering.

    Pre-computes K cluster centroids from h_start vectors and K corresponding
    mean delta vectors. At inference, finds the nearest cluster centroid and
    applies that cluster's compensation vector.

    This is a non-parametric approximation to input-adaptive compensation
    that requires no training, only offline clustering.

    h_comp = h + cluster_delta[nearest_cluster(h)]
    """

    def __init__(
        self,
        cluster_centroids: Union[torch.Tensor, str, Path],
        cluster_deltas: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            cluster_centroids: K cluster centroids [K, hidden_dim] or path to .pt file
            cluster_deltas: K mean delta vectors [K, hidden_dim] or path to .pt file
            alpha: Scale factor
        """
        if isinstance(cluster_centroids, (str, Path)):
            cluster_centroids = torch.load(cluster_centroids, map_location="cpu")
        if isinstance(cluster_deltas, (str, Path)):
            cluster_deltas = torch.load(cluster_deltas, map_location="cpu")
        self.centroids = cluster_centroids.float()  # [K, D]
        self.cluster_deltas = cluster_deltas.float()  # [K, D]
        self.n_clusters = self.centroids.shape[0]
        self.alpha = alpha
        self._device = device

    @classmethod
    def from_data(
        cls,
        h_start_list: List[torch.Tensor],
        delta_list: List[torch.Tensor],
        n_clusters: int = 8,
        alpha: float = 1.0,
    ) -> "ClusterCompensation":
        """
        Build cluster compensation from collected hidden states and deltas.

        Args:
            h_start_list: List of h_start vectors [hidden_dim]
            delta_list: List of delta vectors [hidden_dim]
            n_clusters: Number of clusters
        """
        from sklearn.cluster import KMeans

        h_start = torch.stack(h_start_list).float().numpy()  # [N, D]
        deltas = torch.stack(delta_list).float().numpy()      # [N, D]

        # Cluster h_start vectors
        n_clusters = min(n_clusters, len(h_start_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(h_start)

        # Compute mean delta per cluster
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        cluster_deltas = torch.zeros(n_clusters, deltas.shape[1])
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_deltas[k] = torch.tensor(deltas[mask].mean(axis=0))

        return cls(
            cluster_centroids=centroids,
            cluster_deltas=cluster_deltas,
            alpha=alpha,
        )

    def to(self, device: torch.device) -> "ClusterCompensation":
        self.centroids = self.centroids.to(device)
        self.cluster_deltas = self.cluster_deltas.to(device)
        self._device = device
        return self

    def _find_nearest_cluster(self, h: torch.Tensor) -> torch.Tensor:
        """
        Find nearest cluster for each token.

        Args:
            h: [batch, seq_len, D]

        Returns:
            cluster_ids: [batch, seq_len]
        """
        centroids = self.centroids.to(h.device)  # [K, D]
        # Compute cosine similarity between each token and each centroid
        h_norm = F.normalize(h.float(), dim=-1)  # [B, S, D]
        c_norm = F.normalize(centroids, dim=-1)  # [K, D]
        # [B, S, K]
        sims = torch.einsum("bsd,kd->bsk", h_norm, c_norm)
        return sims.argmax(dim=-1)  # [B, S]

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cluster_deltas = self.cluster_deltas.to(h.device)

        # Find nearest cluster for each token
        cluster_ids = self._find_nearest_cluster(h)  # [B, S]

        # Gather cluster-specific deltas
        # cluster_ids: [B, S] -> [B, S, D]
        batch_size, seq_len = cluster_ids.shape
        flat_ids = cluster_ids.view(-1)  # [B*S]
        flat_deltas = cluster_deltas[flat_ids]  # [B*S, D]
        token_deltas = flat_deltas.view(batch_size, seq_len, -1)  # [B, S, D]

        return h + self.alpha * token_deltas.to(h.dtype)

    def __repr__(self) -> str:
        return (f"ClusterCompensation("
                f"n_clusters={self.n_clusters}, alpha={self.alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Combined / Composite Strategies
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareLastTokenCompensation(BaseCompensation):
    """
    Combines PhaseAwareCompensation with LastTokenCompensation.

    During prefill: apply compensation only to the last token using prefill vector
    During decode: apply compensation to the single token using decode vector

    This is the recommended starting point based on the diagnostic analysis.
    """

    def __init__(
        self,
        prefill_delta: Union[torch.Tensor, str, Path],
        decode_delta: Union[torch.Tensor, str, Path],
        prefill_alpha: float = 1.0,
        decode_alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if isinstance(prefill_delta, (str, Path)):
            prefill_delta = torch.load(prefill_delta, map_location="cpu")
        if isinstance(decode_delta, (str, Path)):
            decode_delta = torch.load(decode_delta, map_location="cpu")
        self.prefill_delta = prefill_delta.float()
        self.decode_delta = decode_delta.float()
        self.prefill_alpha = prefill_alpha
        self.decode_alpha = decode_alpha
        self._device = device

    def to(self, device: torch.device) -> "PhaseAwareLastTokenCompensation":
        self.prefill_delta = self.prefill_delta.to(device)
        self.decode_delta = self.decode_delta.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            delta = self.decode_delta.to(h.device)
            alpha = self.decode_alpha
            return h + alpha * delta.view(1, 1, -1)
        else:
            delta = self.prefill_delta.to(h.device)
            alpha = self.prefill_alpha
            h_comp = h.clone()
            h_comp[:, -1, :] = h[:, -1, :] + alpha * delta
            return h_comp

    def __repr__(self) -> str:
        return (f"PhaseAwareLastTokenCompensation("
                f"prefill_alpha={self.prefill_alpha:.3f}, "
                f"decode_alpha={self.decode_alpha:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Skip-Layer Model with pluggable compensation
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedCompensatedSkipLayerModel:
    """
    Extended version of CompensatedSkipLayerModel that supports any
    BaseCompensation strategy.

    Tracks whether the current forward pass is prefill or decode by
    monitoring the sequence length (prefill: seq_len > 1, decode: seq_len == 1).

    Usage:
        strategy = PhaseAwareCompensation(
            prefill_delta_file="wikitext_prefill_mean.pt",
            decode_delta_file="wikitext_decode_mean.pt",
        )
        model = AdvancedCompensatedSkipLayerModel(
            model=base_model,
            tokenizer=tokenizer,
            skip_layers=[19, 20, 21, 22],
            compensation=strategy,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        skip_layers: List[int],
        compensation: BaseCompensation,
        compensation_layer: Optional[int] = None,
        post_skip_layer: Optional[int] = None,
        compensation_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            model: Base transformer model
            tokenizer: Associated tokenizer
            skip_layers: Layer indices to skip
            compensation: Compensation strategy instance
            compensation_layer: Layer to apply compensation at (default: layer before first skip)
            post_skip_layer: Layer to apply post-skip compensation (for CascadedCompensation)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.skip_layers = sorted(skip_layers)
        self.compensation = compensation

        # Move compensation to model device
        if hasattr(model, "device"):
            self.compensation.to(model.device)

        # Default compensation layer
        if compensation_layer is None:
            self.compensation_layer = min(skip_layers) - 1
        else:
            self.compensation_layer = compensation_layer

        if compensation_layers is None:
            self.compensation_layers = [self.compensation_layer]
        else:
            merged = set(compensation_layers)
            merged.add(self.compensation_layer)
            self.compensation_layers = sorted(merged)

        for layer_idx in self.compensation_layers:
            if layer_idx < 0:
                raise ValueError(f"Compensation layer {layer_idx} is invalid")
            if layer_idx in skip_layers:
                raise ValueError(f"Compensation layer {layer_idx} cannot be in skip_layers")

        self.post_skip_layer = post_skip_layer
        self._is_decode = False  # Track phase

        print(f"AdvancedCompensatedSkipLayerModel initialized:")
        print(f"  Strategy: {compensation}")
        print(f"  Skipping layers: {skip_layers}")
        if len(self.compensation_layers) == 1:
            print(f"  Compensation at layer: {self.compensation_layer}")
        else:
            print(f"  Compensation layers: {self.compensation_layers}")
        if post_skip_layer:
            print(f"  Post-skip compensation at layer: {post_skip_layer}")

        self._register_hooks()

    def _get_layers(self):
        """Get the decoder layers from the model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
            and hasattr(self.model.model.language_model, "layers")
        ):
            return self.model.model.language_model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise ValueError("Could not find decoder layers in model")

    def _register_hooks(self):
        """Register forward hooks."""
        layers = self._get_layers()
        self.hooks = []

        def _apply_compensation(
            hidden_states: torch.Tensor,
            is_decode: bool,
            layer_idx: int,
        ) -> torch.Tensor:
            if hasattr(self.compensation, "compensate_at_layer"):
                return self.compensation.compensate_at_layer(
                    hidden_states,
                    layer_idx=layer_idx,
                    is_decode=is_decode,
                )
            return self.compensation.compensate(hidden_states, is_decode=is_decode)

        def make_compensation_hook(layer_idx: int):
            def compensation_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    seq_len = hidden_states.shape[1]
                    is_decode = (seq_len == 1)
                    compensated = _apply_compensation(
                        hidden_states=hidden_states,
                        is_decode=is_decode,
                        layer_idx=layer_idx,
                    )
                    return (compensated,) + output[1:]
                seq_len = output.shape[1]
                is_decode = (seq_len == 1)
                return _apply_compensation(
                    hidden_states=output,
                    is_decode=is_decode,
                    layer_idx=layer_idx,
                )

            return compensation_hook

        for comp_layer in self.compensation_layers:
            hook = layers[comp_layer].register_forward_hook(make_compensation_hook(comp_layer))
            self.hooks.append(hook)

        # Post-skip compensation hook (for CascadedCompensation)
        if self.post_skip_layer is not None and isinstance(self.compensation, CascadedCompensation):
            def post_skip_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    seq_len = hidden_states.shape[1]
                    is_decode = (seq_len == 1)
                    compensated = self.compensation.compensate_post_skip(
                        hidden_states, is_decode=is_decode
                    )
                    return (compensated,) + output[1:]
                else:
                    seq_len = output.shape[1]
                    is_decode = (seq_len == 1)
                    return self.compensation.compensate_post_skip(output, is_decode=is_decode)

            hook = layers[self.post_skip_layer].register_forward_hook(post_skip_hook)
            self.hooks.append(hook)

        # Skip hooks
        def skip_hook(module, input, output):
            if isinstance(input, tuple):
                return input[0]
            return input

        for layer_idx in self.skip_layers:
            hook = layers[layer_idx].register_forward_hook(skip_hook)
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    @property
    def dtype(self):
        return self.model.dtype

    def __getattr__(self, name):
        if name in ("model", "tokenizer", "skip_layers", "compensation",
                    "compensation_layer", "compensation_layers", "post_skip_layer",
                    "hooks", "_is_decode"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.model, name)

    def __del__(self):
        self.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_compensation_strategy(
    strategy_name: str,
    **kwargs,
) -> BaseCompensation:
    """
    Factory function to create a compensation strategy by name.

    Args:
        strategy_name: One of:
            "scaled"              - ScaledCompensation
            "last_token"          - LastTokenCompensation
            "magnitude_preserving"- MagnitudePreservingCompensation
            "multiplicative"      - MultiplicativeCompensation
            "pca"                 - PCACompensation
            "phase_aware"         - PhaseAwareCompensation
            "position_aware"      - PositionAwareCompensation
            "cascaded"            - CascadedCompensation
            "magnitude_rescaling" - MagnitudeRescalingCompensation
            "multi_phase_aware_magnitude_rescaling"
                                 - MultiClusterPhaseAwareMagnitudeRescalingCompensation
            "cluster"             - ClusterCompensation
            "phase_last_token"    - PhaseAwareLastTokenCompensation
        **kwargs: Strategy-specific arguments

    Returns:
        BaseCompensation instance

    Example:
        strategy = create_compensation_strategy(
            "phase_aware",
            prefill_delta="wikitext_prefill_mean.pt",
            decode_delta="wikitext_decode_mean.pt",
        )
    """
    strategies = {
        "scaled": ScaledCompensation,
        "last_token": LastTokenCompensation,
        "magnitude_preserving": MagnitudePreservingCompensation,
        "multiplicative": MultiplicativeCompensation,
        "pca": PCACompensation,
        "phase_aware": PhaseAwareCompensation,
        "position_aware": PositionAwareCompensation,
        "cascaded": CascadedCompensation,
        "magnitude_rescaling": MagnitudeRescalingCompensation,
        "phase_aware_magnitude_rescaling": PhaseAwareMagnitudeRescalingCompensation,
        "multi_phase_aware_magnitude_rescaling": MultiClusterPhaseAwareMagnitudeRescalingCompensation,
        "cluster": ClusterCompensation,
        "phase_last_token": PhaseAwareLastTokenCompensation,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 11: Phase-Aware Multiplicative Compensation
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareMultiplicativeCompensation(BaseCompensation):
    """
    Element-wise scale+bias compensation, separate for prefill and decode.

    h_comp = h * scale_vec + bias_vec

    where scale_vec = mean(h_end) / mean(h_start) element-wise
    and   bias_vec  = mean(h_end) - scale_vec * mean(h_start)

    This is more expressive than scalar norm rescaling because it captures
    the per-dimension mean transformation of the skipped layers.

    Requires pre-computed mean(h_start) and mean(h_end) vectors for each phase,
    produced by collect_and_fit.py in experiments/improved_compensation_v2/.
    """

    def __init__(
        self,
        prefill_scale: Union[torch.Tensor, str, Path],
        prefill_bias: Union[torch.Tensor, str, Path],
        decode_scale: Union[torch.Tensor, str, Path],
        decode_bias: Union[torch.Tensor, str, Path],
        device: Optional[torch.device] = None,
    ):
        for name, val in [("prefill_scale", prefill_scale), ("prefill_bias", prefill_bias),
                          ("decode_scale", decode_scale), ("decode_bias", decode_bias)]:
            if isinstance(val, (str, Path)):
                val = torch.load(val, map_location="cpu")
            setattr(self, name, val.float())
        self._device = device

    @classmethod
    def from_mean_vectors(
        cls,
        prefill_h_start_mean: Union[torch.Tensor, str, Path],
        prefill_h_end_mean: Union[torch.Tensor, str, Path],
        decode_h_start_mean: Union[torch.Tensor, str, Path],
        decode_h_end_mean: Union[torch.Tensor, str, Path],
        eps: float = 1e-8,
        scale_clip: float = 5.0,
    ) -> "PhaseAwareMultiplicativeCompensation":
        """Create from pre-computed mean(h_start) and mean(h_end) vectors."""
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        def _compute(mean_start, mean_end):
            scale = mean_end / (mean_start.abs() + eps)
            scale = scale.clamp(-scale_clip, scale_clip)
            bias = mean_end - scale * mean_start
            return scale, bias

        pf_start = _load(prefill_h_start_mean)
        pf_end = _load(prefill_h_end_mean)
        dc_start = _load(decode_h_start_mean)
        dc_end = _load(decode_h_end_mean)

        pf_scale, pf_bias = _compute(pf_start, pf_end)
        dc_scale, dc_bias = _compute(dc_start, dc_end)

        return cls(
            prefill_scale=pf_scale, prefill_bias=pf_bias,
            decode_scale=dc_scale, decode_bias=dc_bias,
        )

    def to(self, device: torch.device) -> "PhaseAwareMultiplicativeCompensation":
        self.prefill_scale = self.prefill_scale.to(device)
        self.prefill_bias = self.prefill_bias.to(device)
        self.decode_scale = self.decode_scale.to(device)
        self.decode_bias = self.decode_bias.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            scale = self.decode_scale.to(h.device)
            bias = self.decode_bias.to(h.device)
        else:
            scale = self.prefill_scale.to(h.device)
            bias = self.prefill_bias.to(h.device)
        return (h.float() * scale.view(1, 1, -1) + bias.view(1, 1, -1)).to(h.dtype)

    def __repr__(self) -> str:
        return "PhaseAwareMultiplicativeCompensation(element-wise scale+bias)"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 12: Low-Rank Linear Adapter Compensation
# ─────────────────────────────────────────────────────────────────────────────

class LowRankLinearAdapterCompensation(BaseCompensation):
    """
    Low-rank linear adapter: delta ≈ U @ (V^T @ h)

    h_comp = h + alpha * U @ (V^T @ h)

    where U ∈ R^{D×r} and V ∈ R^{D×r} are fitted from (h_start, h_end) pairs
    using least-squares + SVD truncation.

    This is the most expressive fixed-parameter compensation strategy.
    It captures the mean linear transformation of the skipped layers.

    Separate U, V for prefill and decode phases.
    """

    def __init__(
        self,
        prefill_U: Union[torch.Tensor, str, Path],
        prefill_V: Union[torch.Tensor, str, Path],
        decode_U: Union[torch.Tensor, str, Path],
        decode_V: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        self.prefill_U = _load(prefill_U)  # [D, r]
        self.prefill_V = _load(prefill_V)  # [D, r]
        self.decode_U = _load(decode_U)    # [D, r]
        self.decode_V = _load(decode_V)    # [D, r]
        self.alpha = alpha
        self._device = device

        r = self.prefill_U.shape[1]
        print(f"LowRankLinearAdapterCompensation: rank={r}, alpha={alpha:.3f}")

    def to(self, device: torch.device) -> "LowRankLinearAdapterCompensation":
        self.prefill_U = self.prefill_U.to(device)
        self.prefill_V = self.prefill_V.to(device)
        self.decode_U = self.decode_U.to(device)
        self.decode_V = self.decode_V.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            U = self.decode_U.to(h.device)
            V = self.decode_V.to(h.device)
        else:
            U = self.prefill_U.to(h.device)
            V = self.prefill_V.to(h.device)

        h_float = h.float()
        # delta ≈ U @ (V^T @ h) = h @ V @ U^T
        # h: [B, S, D], V: [D, r], U: [D, r]
        delta = h_float @ V @ U.T  # [B, S, D]
        return (h_float + self.alpha * delta).to(h.dtype)

    def __repr__(self) -> str:
        r = self.prefill_U.shape[1]
        return f"LowRankLinearAdapterCompensation(rank={r}, alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 13: Norm-Adaptive Rescaling
# ─────────────────────────────────────────────────────────────────────────────

class NormAdaptiveRescaling(BaseCompensation):
    """
    Norm-adaptive rescaling: ratio = a + b / ||h||

    h_comp = h * (a + b / ||h||)

    Parameters a and b are fitted from (||h_start||, ||h_end||/||h_start||) pairs
    using linear regression: ratio = a + b / ||h_start||.

    This extends PhaseAwareMagnitudeRescalingCompensation by making the ratio
    depend on the input norm, which accounts for the observed variance in the
    norm ratio (std ≈ 0.27 for decode phase).

    Separate parameters for prefill and decode.
    """

    def __init__(
        self,
        prefill_a: float = 1.0,
        prefill_b: float = 0.0,
        decode_a: float = 1.0,
        decode_b: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.prefill_a = prefill_a
        self.prefill_b = prefill_b
        self.decode_a = decode_a
        self.decode_b = decode_b
        self._device = device

    def to(self, device: torch.device) -> "NormAdaptiveRescaling":
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.decode_a if is_decode else self.prefill_a
        b = self.decode_b if is_decode else self.prefill_b

        h_float = h.float()
        h_norm = h_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
        ratio = a + b / h_norm
        return (h_float * ratio).to(h.dtype)

    def __repr__(self) -> str:
        return (f"NormAdaptiveRescaling("
                f"prefill=({self.prefill_a:.3f}+{self.prefill_b:.1f}/||h||), "
                f"decode=({self.decode_a:.3f}+{self.decode_b:.1f}/||h||))")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 14: Cluster-Based Phase-Aware Compensation
# ─────────────────────────────────────────────────────────────────────────────

class ClusterBasedPhaseAwareCompensation(BaseCompensation):
    """
    Input-adaptive compensation using K-means clustering, separate for prefill/decode.

    For each phase (prefill/decode):
      - Pre-computed K cluster centroids from h_start vectors
      - K corresponding mean delta vectors (one per cluster)
      - At inference: find nearest centroid, apply its mean delta

    This is the most promising training-free input-adaptive approach.
    If inputs cluster into groups with similar transformations, cluster-specific
    means explain more variance than the global mean.

    Files produced by collect_and_fit.py in experiments/improved_compensation_v2/.
    """

    def __init__(
        self,
        prefill_centroids: Union[torch.Tensor, str, Path],
        prefill_deltas: Union[torch.Tensor, str, Path],
        decode_centroids: Union[torch.Tensor, str, Path],
        decode_deltas: Union[torch.Tensor, str, Path],
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        self.prefill_centroids = _load(prefill_centroids)  # [K, D]
        self.prefill_deltas = _load(prefill_deltas)        # [K, D]
        self.decode_centroids = _load(decode_centroids)    # [K, D]
        self.decode_deltas = _load(decode_deltas)          # [K, D]
        self.alpha = alpha
        self._device = device

        K = self.prefill_centroids.shape[0]
        print(f"ClusterBasedPhaseAwareCompensation: K={K}, alpha={alpha:.3f}")

    def to(self, device: torch.device) -> "ClusterBasedPhaseAwareCompensation":
        self.prefill_centroids = self.prefill_centroids.to(device)
        self.prefill_deltas = self.prefill_deltas.to(device)
        self.decode_centroids = self.decode_centroids.to(device)
        self.decode_deltas = self.decode_deltas.to(device)
        self._device = device
        return self

    def _find_nearest_cluster(
        self, h: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """Find nearest cluster centroid for each token using cosine similarity."""
        h_norm = F.normalize(h.float(), dim=-1)          # [B, S, D]
        c_norm = F.normalize(centroids.float(), dim=-1)  # [K, D]
        sims = torch.einsum("bsd,kd->bsk", h_norm, c_norm)  # [B, S, K]
        return sims.argmax(dim=-1)  # [B, S]

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            centroids = self.decode_centroids.to(h.device)
            cluster_deltas = self.decode_deltas.to(h.device)
        else:
            centroids = self.prefill_centroids.to(h.device)
            cluster_deltas = self.prefill_deltas.to(h.device)

        cluster_ids = self._find_nearest_cluster(h, centroids)  # [B, S]
        B, S = cluster_ids.shape
        flat_deltas = cluster_deltas[cluster_ids.view(-1)]  # [B*S, D]
        token_deltas = flat_deltas.view(B, S, -1)           # [B, S, D]

        return (h.float() + self.alpha * token_deltas).to(h.dtype)

    def __repr__(self) -> str:
        K = self.prefill_centroids.shape[0]
        return f"ClusterBasedPhaseAwareCompensation(K={K}, alpha={self.alpha:.3f})"


def load_advanced_compensated_model(
    model_id: str,
    skip_layers: List[int],
    strategy_name: str,
    strategy_kwargs: Dict,
    device: str = "cuda",
    dtype: str = "bfloat16",
    compensation_layer: Optional[int] = None,
    post_skip_layer: Optional[int] = None,
) -> tuple:
    """
    Load a model with advanced compensation.

    Args:
        model_id: HuggingFace model ID
        skip_layers: Layer indices to skip
        strategy_name: Compensation strategy name
        strategy_kwargs: Strategy-specific arguments
        device: Device
        dtype: Model dtype
        compensation_layer: Layer to apply compensation at
        post_skip_layer: Layer for post-skip compensation (CascadedCompensation)

    Returns:
        Tuple of (AdvancedCompensatedSkipLayerModel, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype_obj = dtype_map.get(dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_obj,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Create compensation strategy
    compensation = create_compensation_strategy(strategy_name, **strategy_kwargs)

    # Create advanced model
    advanced_model = AdvancedCompensatedSkipLayerModel(
        model=model,
        tokenizer=tokenizer,
        skip_layers=skip_layers,
        compensation=compensation,
        compensation_layer=compensation_layer,
        post_skip_layer=post_skip_layer,
    )

    return advanced_model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# v2 Improved Strategies
# ─────────────────────────────────────────────────────────────────────────────

class ClusterNormOnlyCompensation(BaseCompensation):
    """
    Cluster-based compensation that only adjusts the norm (not direction).

    For each cluster, computes the mean norm ratio: mean(||h_end|| / ||h_start||).
    At inference: find nearest cluster, scale h by the cluster-specific norm ratio.

    This is safer than adding the full cluster delta because it avoids adding
    direction noise. It's a cluster-specific version of PhaseAwareMagnitudeRescaling.
    """

    def __init__(
        self,
        prefill_centroids: Union[torch.Tensor, str, Path],
        prefill_norm_ratios: Union[torch.Tensor, str, Path],
        decode_centroids: Union[torch.Tensor, str, Path],
        decode_norm_ratios: Union[torch.Tensor, str, Path],
        device: Optional[torch.device] = None,
    ):
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        self.prefill_centroids = _load(prefill_centroids)    # [K, D]
        self.prefill_norm_ratios = _load(prefill_norm_ratios)  # [K]
        self.decode_centroids = _load(decode_centroids)      # [K, D]
        self.decode_norm_ratios = _load(decode_norm_ratios)  # [K]
        self._device = device

        K = self.prefill_centroids.shape[0]
        print(f"ClusterNormOnlyCompensation: K={K}, "
              f"prefill_ratio_range=[{self.prefill_norm_ratios.min():.3f}, {self.prefill_norm_ratios.max():.3f}], "
              f"decode_ratio_range=[{self.decode_norm_ratios.min():.3f}, {self.decode_norm_ratios.max():.3f}]")

    def to(self, device: torch.device) -> "ClusterNormOnlyCompensation":
        self.prefill_centroids = self.prefill_centroids.to(device)
        self.prefill_norm_ratios = self.prefill_norm_ratios.to(device)
        self.decode_centroids = self.decode_centroids.to(device)
        self.decode_norm_ratios = self.decode_norm_ratios.to(device)
        self._device = device
        return self

    def _find_nearest_cluster(self, h: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h.float(), dim=-1)
        c_norm = F.normalize(centroids.float(), dim=-1)
        sims = torch.einsum("bsd,kd->bsk", h_norm, c_norm)
        return sims.argmax(dim=-1)  # [B, S]

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            centroids = self.decode_centroids.to(h.device)
            norm_ratios = self.decode_norm_ratios.to(h.device)
        else:
            centroids = self.prefill_centroids.to(h.device)
            norm_ratios = self.prefill_norm_ratios.to(h.device)

        cluster_ids = self._find_nearest_cluster(h, centroids)  # [B, S]
        # Get per-token norm ratios
        token_ratios = norm_ratios[cluster_ids.view(-1)].view(cluster_ids.shape)  # [B, S]
        # Scale h by the cluster-specific norm ratio
        h_float = h.float()
        h_scaled = h_float * token_ratios.unsqueeze(-1)  # [B, S, D]
        return h_scaled.to(h.dtype)

    def __repr__(self) -> str:
        K = self.prefill_centroids.shape[0]
        return f"ClusterNormOnlyCompensation(K={K})"


class SoftClusterCompensation(BaseCompensation):
    """
    Soft cluster-based compensation using weighted average of cluster deltas.

    Instead of hard assignment to the nearest cluster, uses soft assignment:
        weight_k = softmax(-||h - centroid_k||^2 / temperature)
        delta = sum(weight_k * cluster_delta_k)

    This is smoother and less sensitive to cluster boundaries.
    """

    def __init__(
        self,
        prefill_centroids: Union[torch.Tensor, str, Path],
        prefill_deltas: Union[torch.Tensor, str, Path],
        decode_centroids: Union[torch.Tensor, str, Path],
        decode_deltas: Union[torch.Tensor, str, Path],
        temperature: float = 1.0,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        self.prefill_centroids = _load(prefill_centroids)
        self.prefill_deltas = _load(prefill_deltas)
        self.decode_centroids = _load(decode_centroids)
        self.decode_deltas = _load(decode_deltas)
        self.temperature = temperature
        self.alpha = alpha
        self._device = device

        K = self.prefill_centroids.shape[0]
        print(f"SoftClusterCompensation: K={K}, temperature={temperature:.3f}, alpha={alpha:.3f}")

    def to(self, device: torch.device) -> "SoftClusterCompensation":
        self.prefill_centroids = self.prefill_centroids.to(device)
        self.prefill_deltas = self.prefill_deltas.to(device)
        self.decode_centroids = self.decode_centroids.to(device)
        self.decode_deltas = self.decode_deltas.to(device)
        self._device = device
        return self

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_decode:
            centroids = self.decode_centroids.to(h.device)
            cluster_deltas = self.decode_deltas.to(h.device)
        else:
            centroids = self.prefill_centroids.to(h.device)
            cluster_deltas = self.prefill_deltas.to(h.device)

        h_float = h.float()
        B, S, D = h_float.shape
        K = centroids.shape[0]

        # Compute cosine similarity to each centroid
        h_norm = F.normalize(h_float, dim=-1)          # [B, S, D]
        c_norm = F.normalize(centroids, dim=-1)         # [K, D]
        sims = torch.einsum("bsd,kd->bsk", h_norm, c_norm)  # [B, S, K]

        # Soft weights via softmax
        weights = torch.softmax(sims / self.temperature, dim=-1)  # [B, S, K]

        # Weighted average of cluster deltas
        # cluster_deltas: [K, D], weights: [B, S, K]
        weighted_delta = torch.einsum("bsk,kd->bsd", weights, cluster_deltas)  # [B, S, D]

        return (h_float + self.alpha * weighted_delta).to(h.dtype)

    def __repr__(self) -> str:
        K = self.prefill_centroids.shape[0]
        return f"SoftClusterCompensation(K={K}, T={self.temperature:.3f}, alpha={self.alpha:.3f})"


class HybridClusterNormAdaptiveCompensation(BaseCompensation):
    """
    Hybrid compensation: cluster-based for prefill, norm-adaptive for decode.

    Rationale:
    - Prefill: cluster variance explained ~55%, so cluster delta is informative
    - Decode: cluster variance explained ~22%, so full delta adds too much noise;
      norm-adaptive rescaling is safer

    Prefill: h_comp = h + cluster_delta (from nearest cluster)
    Decode:  h_comp = h * (a + b / ||h||)
    """

    def __init__(
        self,
        prefill_centroids: Union[torch.Tensor, str, Path],
        prefill_deltas: Union[torch.Tensor, str, Path],
        decode_a: float,
        decode_b: float,
        prefill_alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        def _load(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float()

        self.prefill_centroids = _load(prefill_centroids)
        self.prefill_deltas = _load(prefill_deltas)
        self.decode_a = decode_a
        self.decode_b = decode_b
        self.prefill_alpha = prefill_alpha
        self._device = device

        K = self.prefill_centroids.shape[0]
        print(f"HybridClusterNormAdaptiveCompensation: K={K}, "
              f"decode: ratio={decode_a:.4f}+{decode_b:.2f}/||h||")

    def to(self, device: torch.device) -> "HybridClusterNormAdaptiveCompensation":
        self.prefill_centroids = self.prefill_centroids.to(device)
        self.prefill_deltas = self.prefill_deltas.to(device)
        self._device = device
        return self

    def _find_nearest_cluster(self, h: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h.float(), dim=-1)
        c_norm = F.normalize(centroids.float(), dim=-1)
        sims = torch.einsum("bsd,kd->bsk", h_norm, c_norm)
        return sims.argmax(dim=-1)

    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_float = h.float()

        if is_decode:
            # Norm-adaptive rescaling for decode
            norms = h_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
            ratio = self.decode_a + self.decode_b / norms
            ratio = ratio.clamp(0.5, 3.0)
            return (h_float * ratio).to(h.dtype)
        else:
            # Cluster-based for prefill
            centroids = self.prefill_centroids.to(h.device)
            cluster_deltas = self.prefill_deltas.to(h.device)
            cluster_ids = self._find_nearest_cluster(h_float, centroids)
            B, S = cluster_ids.shape
            flat_deltas = cluster_deltas[cluster_ids.view(-1)]
            token_deltas = flat_deltas.view(B, S, -1)
            return (h_float + self.prefill_alpha * token_deltas).to(h.dtype)

    def __repr__(self) -> str:
        K = self.prefill_centroids.shape[0]
        return (f"HybridClusterNormAdaptiveCompensation(K={K}, "
                f"decode_a={self.decode_a:.4f}, decode_b={self.decode_b:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 15: Gaussian Optimal Transport (Whitening) Compensation
# ─────────────────────────────────────────────────────────────────────────────

class GaussianOTCompensation(BaseCompensation):
    """
    Gaussian optimal transport map: T(h) = mu_end + A @ (h - mu_start)

    The optimal transport map between two Gaussians N(mu_s, Sigma_s) and
    N(mu_e, Sigma_e) is a linear map. This is the most principled non-training
    approach for correcting the distribution shift caused by skipping layers.

    Stored as factored form: A = Vt^T @ A_pca @ Vt  (PCA-compressed)
    where Vt [K, D] are the top-K PCA components of h_start.

    Separate maps for prefill and decode phases.
    """

    def __init__(
        self,
        prefill_mu_start, prefill_mu_end, prefill_VtT, prefill_A_pca, prefill_Vt,
        decode_mu_start,  decode_mu_end,  decode_VtT,  decode_A_pca,  decode_Vt,
        alpha: float = 1.0,
        device=None,
    ):
        def _l(v):
            if isinstance(v, (str, Path)):
                return torch.load(v, map_location="cpu").float()
            return v.float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)

        self.pf_mu_s = _l(prefill_mu_start)
        self.pf_mu_e = _l(prefill_mu_end)
        self.pf_VtT  = _l(prefill_VtT)
        self.pf_A    = _l(prefill_A_pca)
        self.pf_Vt   = _l(prefill_Vt)
        self.dc_mu_s = _l(decode_mu_start)
        self.dc_mu_e = _l(decode_mu_end)
        self.dc_VtT  = _l(decode_VtT)
        self.dc_A    = _l(decode_A_pca)
        self.dc_Vt   = _l(decode_Vt)
        self.alpha   = alpha
        self._device = device
        print(f"GaussianOTCompensation: PCA_dim={self.pf_Vt.shape[0]}, alpha={alpha:.3f}")

    def to(self, device) -> "GaussianOTCompensation":
        for attr in ["pf_mu_s","pf_mu_e","pf_VtT","pf_A","pf_Vt",
                     "dc_mu_s","dc_mu_e","dc_VtT","dc_A","dc_Vt"]:
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    def _apply_ot(self, h: torch.Tensor, mu_s, mu_e, VtT, A, Vt) -> torch.Tensor:
        """Apply OT map: T(h) = mu_e + (h - mu_s) @ VtT @ A @ Vt"""
        dev = h.device
        hc = h.float() - mu_s.to(dev)          # [B, S, D]
        z  = hc @ VtT.to(dev)                   # [B, S, K]
        z2 = z  @ A.to(dev)                     # [B, S, K]
        out = z2 @ Vt.to(dev) + mu_e.to(dev)   # [B, S, D]
        # Blend: alpha=1 → full OT, alpha=0 → identity
        return (self.alpha * out + (1 - self.alpha) * h.float()).to(h.dtype)

    def compensate(self, h, is_decode=False, token_positions=None):
        if is_decode:
            return self._apply_ot(h, self.dc_mu_s, self.dc_mu_e,
                                   self.dc_VtT, self.dc_A, self.dc_Vt)
        return self._apply_ot(h, self.pf_mu_s, self.pf_mu_e,
                               self.pf_VtT, self.pf_A, self.pf_Vt)

    def __repr__(self):
        return f"GaussianOTCompensation(K={self.pf_Vt.shape[0]}, alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 16: High-Rank LoRA (configurable rank)
# ─────────────────────────────────────────────────────────────────────────────

class HighRankLoRACompensation(BaseCompensation):
    """
    Low-rank linear adapter with configurable rank.
    Loads U, V from files named prefill_lora{rank}_U.pt etc.
    """

    def __init__(self, models_dir, rank: int = 128, alpha: float = 1.0, device=None):
        d = Path(models_dir)
        def _l(p):
            return torch.load(p, map_location="cpu").float()
        self.pf_U = _l(d / f"prefill_lora{rank}_U.pt")
        self.pf_V = _l(d / f"prefill_lora{rank}_V.pt")
        self.dc_U = _l(d / f"decode_lora{rank}_U.pt")
        self.dc_V = _l(d / f"decode_lora{rank}_V.pt")
        self.alpha = alpha
        self.rank  = rank
        self._device = device
        print(f"HighRankLoRACompensation: rank={rank}, alpha={alpha:.3f}")

    def to(self, device) -> "HighRankLoRACompensation":
        for attr in ["pf_U","pf_V","dc_U","dc_V"]:
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    def compensate(self, h, is_decode=False, token_positions=None):
        U = self.dc_U.to(h.device) if is_decode else self.pf_U.to(h.device)
        V = self.dc_V.to(h.device) if is_decode else self.pf_V.to(h.device)
        delta = h.float() @ V @ U.T
        return (h.float() + self.alpha * delta).to(h.dtype)

    def __repr__(self):
        return f"HighRankLoRACompensation(rank={self.rank}, alpha={self.alpha:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 17: High-K Cluster Compensation (configurable K)
# ─────────────────────────────────────────────────────────────────────────────

class HighKClusterCompensation(BaseCompensation):
    """Cluster-based compensation with configurable K."""

    def __init__(self, models_dir, k: int = 64, alpha: float = 1.0, device=None):
        d = Path(models_dir)
        def _l(p):
            return torch.load(p, map_location="cpu").float()
        self.pf_c = _l(d / f"prefill_cluster{k}_centroids.pt")
        self.pf_d = _l(d / f"prefill_cluster{k}_deltas.pt")
        self.dc_c = _l(d / f"decode_cluster{k}_centroids.pt")
        self.dc_d = _l(d / f"decode_cluster{k}_deltas.pt")
        self.alpha = alpha
        self.k     = k
        self._device = device
        print(f"HighKClusterCompensation: K={k}, alpha={alpha:.3f}")

    def to(self, device) -> "HighKClusterCompensation":
        for attr in ["pf_c","pf_d","dc_c","dc_d"]:
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    def _nearest(self, h, centroids):
        hn = F.normalize(h.float(), dim=-1)
        cn = F.normalize(centroids.float(), dim=-1)
        sims = torch.einsum("bsd,kd->bsk", hn, cn)
        return sims.argmax(dim=-1)

    def compensate(self, h, is_decode=False, token_positions=None):
        c = self.dc_c.to(h.device) if is_decode else self.pf_c.to(h.device)
        d = self.dc_d.to(h.device) if is_decode else self.pf_d.to(h.device)
        ids = self._nearest(h, c)
        B, S = ids.shape
        deltas = d[ids.view(-1)].view(B, S, -1)
        return (h.float() + self.alpha * deltas).to(h.dtype)

    def __repr__(self):
        return f"HighKClusterCompensation(K={self.k}, alpha={self.alpha:.3f})"
