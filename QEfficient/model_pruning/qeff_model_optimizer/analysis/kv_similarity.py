"""KV head similarity analysis for KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.transforms.adapters import resolve_layer_adapter
from QEfficient.model_pruning.qeff_model_optimizer.transforms.anatomy import resolve_layer_anatomy


@dataclass(eq=True)
class KvSimilarityReport:
    """Per-layer pairwise KV head similarity and suggested merge pairs.

    ``similarity_matrices[layer_idx]`` is a 2-D list of floats with shape
    ``[num_kv_heads, num_kv_heads]`` representing pairwise cosine similarity
    (averaged over K and V projections) between every pair of KV heads in that
    layer.

    ``merge_pairs[layer_idx]`` is a list of ``(head_a, head_b)`` tuples sorted
    by similarity descending -- the most similar pair comes first.  Only
    upper-triangle pairs (head_a < head_b) are included.
    """

    model_id: str
    num_layers: int
    num_kv_heads: int
    # similarity_matrices[layer_idx] = 2D list of floats (num_kv_heads x num_kv_heads)
    similarity_matrices: dict[int, list[list[float]]]
    # merge_pairs[layer_idx] = list of (head_a, head_b) pairs sorted by similarity descending
    merge_pairs: dict[int, list[tuple[int, int]]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for manifest or JSON storage."""
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "similarity_matrices": {
                str(k): v for k, v in self.similarity_matrices.items()
            },
            "merge_pairs": {
                str(k): [list(pair) for pair in v]
                for k, v in self.merge_pairs.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KvSimilarityReport":
        """Deserialise from a plain dict (e.g. loaded from a manifest)."""
        return cls(
            model_id=str(payload["model_id"]),
            num_layers=int(payload["num_layers"]),
            num_kv_heads=int(payload["num_kv_heads"]),
            similarity_matrices={
                int(k): [[float(x) for x in row] for row in v]
                for k, v in payload["similarity_matrices"].items()
            },
            merge_pairs={
                int(k): [tuple(pair) for pair in v]
                for k, v in payload["merge_pairs"].items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


def _compute_pairwise_similarity(
    weight: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute pairwise cosine similarity between KV heads from a projection weight.

    Parameters
    ----------
    weight:
        The projection weight matrix shaped ``[num_kv_heads * head_dim, hidden_size]``.
    num_kv_heads:
        Number of KV heads.
    head_dim:
        Dimensionality of each head.

    Returns
    -------
    torch.Tensor
        Cosine similarity matrix of shape ``[num_kv_heads, num_kv_heads]``.
    """
    # Reshape to [num_kv_heads, head_dim, hidden_size]
    head_weights = weight.reshape(num_kv_heads, head_dim, -1)
    # Flatten each head to [head_dim * hidden_size]
    head_vectors = head_weights.reshape(num_kv_heads, -1).float()
    # Normalize and compute pairwise cosine similarity
    normed = F.normalize(head_vectors, dim=-1)
    return normed @ normed.T


def _extract_merge_pairs(
    sim_matrix: torch.Tensor,
) -> list[tuple[int, int]]:
    """Extract upper-triangle pairs sorted by similarity descending.

    Parameters
    ----------
    sim_matrix:
        Square cosine similarity matrix of shape ``[N, N]``.

    Returns
    -------
    list[tuple[int, int]]
        Pairs ``(head_a, head_b)`` with ``head_a < head_b``, sorted by
        similarity score descending.
    """
    n = sim_matrix.shape[0]
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j].item(), i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [(a, b) for _, a, b in pairs]


@torch.no_grad()
def compute_kv_head_similarity(
    artifact: ModelArtifact,
    similarity_metric: str = "cosine",
) -> KvSimilarityReport:
    """Compute per-layer pairwise KV head similarity from projection weights.

    This is a **weight-only** analysis -- no calibration data or forward passes
    are needed.  For each layer the K and V projection weight matrices are
    reshaped to isolate individual KV heads, then pairwise cosine similarity is
    computed.  The final similarity score for each head pair is the average of
    the K-space and V-space similarities, giving a more robust signal for merge
    decisions.

    Parameters
    ----------
    artifact:
        A loaded ``ModelArtifact`` whose ``.model`` attribute is an HF-style
        causal LM.
    similarity_metric:
        Similarity metric to use.  Currently only ``"cosine"`` is supported.

    Returns
    -------
    KvSimilarityReport
        Report containing per-layer similarity matrices and suggested merge
        pairs sorted by similarity descending.

    Raises
    ------
    ValueError
        If an unsupported ``similarity_metric`` is requested.
    """
    if similarity_metric != "cosine":
        raise ValueError(
            f"Unsupported similarity_metric: {similarity_metric!r}. "
            f"Currently only 'cosine' is supported."
        )

    model = artifact.model
    adapter = resolve_layer_adapter(model)
    num_layers = adapter.num_layers

    similarity_matrices: dict[int, list[list[float]]] = {}
    merge_pairs: dict[int, list[tuple[int, int]]] = {}
    num_kv_heads_resolved: int | None = None

    for layer_idx in range(num_layers):
        anatomy = resolve_layer_anatomy(model, layer_idx)
        num_kv_heads = anatomy.num_kv_heads
        head_dim = anatomy.head_dim

        if num_kv_heads_resolved is None:
            num_kv_heads_resolved = num_kv_heads

        # Compute K-space similarity
        k_sim = _compute_pairwise_similarity(
            anatomy.k_proj.weight.data, num_kv_heads, head_dim
        )

        # Compute V-space similarity
        v_sim = _compute_pairwise_similarity(
            anatomy.v_proj.weight.data, num_kv_heads, head_dim
        )

        # Average K and V similarities for more robust merging decisions
        avg_sim = (k_sim + v_sim) / 2.0

        # Store similarity matrix as nested list
        similarity_matrices[layer_idx] = avg_sim.tolist()

        # Extract merge pairs sorted by similarity descending
        merge_pairs[layer_idx] = _extract_merge_pairs(avg_sim)

    return KvSimilarityReport(
        model_id=artifact.model_spec.model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads_resolved or 0,
        similarity_matrices=similarity_matrices,
        merge_pairs=merge_pairs,
        metadata={
            "similarity_metric": similarity_metric,
            "method": "weight_only",
            "averaging": "k_v_mean",
        },
    )
