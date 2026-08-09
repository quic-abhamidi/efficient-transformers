"""Weak-layer analysis: measure per-layer contribution and rank layers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import torch

from QEfficient.model_pruning.logging_utils import get_logger
from QEfficient.model_pruning.qeff_model_optimizer.analysis.contributions import (
    SupportedMetric,
    aggregate_layer_scores,
    compute_per_layer_deltas,
)
from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import DatasetLoadError, load_dataset_samples
from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import RankedLayer, WeakLayerReport
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.utils.writers import write_combined_png, write_legacy_csv, write_legacy_png

logger = get_logger(__name__)


def _resolve_text_tokenizer(tokenizer_or_processor):
    text_tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if text_tokenizer is tokenizer_or_processor and hasattr(tokenizer_or_processor, "image_processor"):
        raise ValueError(
            "Weak-layer analysis requires a text tokenizer. The loaded processor does not expose "
            "a tokenizer attribute for text calibration prompts."
        )
    if getattr(text_tokenizer, "pad_token", None) is None and getattr(text_tokenizer, "eos_token", None) is not None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    return text_tokenizer


def _run_forward_pass_batch(
    model, tokenizer, prompts: list[str], device, max_length: int
):
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    inputs = text_tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["attention_mask"]


def _resolve_device(artifact: ModelArtifact) -> torch.device:
    params = list(artifact.model.parameters())
    if params:
        return params[0].device
    return torch.device("cpu")


def _analyze_dataset(
    artifact: ModelArtifact,
    dataset: str,
    num_samples: int,
    batch_size: int,
    metric: SupportedMetric,
    max_length: int,
    *,
    verbose: bool = False,
) -> list[dict[str, float]]:
    dataset_t0 = time.time()
    if verbose:
        logger.info(
            "[analyze] loading dataset=%s requested_samples=%d metric=%s max_length=%d",
            dataset,
            num_samples,
            metric,
            max_length,
        )

    prompts = load_dataset_samples(dataset, num_samples)
    device = _resolve_device(artifact)
    per_sample_deltas: list[list[float]] = []

    if verbose:
        logger.info(
            "[analyze] dataset=%s loaded_prompts=%d device=%s batch_size=%d",
            dataset,
            len(prompts),
            device,
            batch_size,
        )

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch = prompts[start : start + batch_size]
        batch_t0 = time.time()
        if verbose:
            logger.info(
                "[analyze] dataset=%s batch=%d/%d samples=%d running forward pass",
                dataset,
                batch_idx,
                total_batches,
                len(batch),
            )

        hidden_states, attention_mask = _run_forward_pass_batch(
            artifact.model, artifact.tokenizer, batch, device, max_length
        )
        for sample_idx in range(len(batch)):
            sample_hidden = [h[sample_idx : sample_idx + 1] for h in hidden_states]
            sample_mask = attention_mask[sample_idx : sample_idx + 1]
            per_sample_deltas.append(
                compute_per_layer_deltas(sample_hidden, metric=metric, mask=sample_mask)
            )

        if verbose:
            logger.info(
                "[analyze] dataset=%s batch=%d/%d done elapsed_s=%.2f processed_samples=%d/%d",
                dataset,
                batch_idx,
                total_batches,
                time.time() - batch_t0,
                min(start + len(batch), len(prompts)),
                len(prompts),
            )

    if not per_sample_deltas:
        raise RuntimeError(
            f"No samples were successfully processed for dataset {dataset!r}"
        )

    stats = aggregate_layer_scores(per_sample_deltas)
    if verbose:
        logger.info(
            "[analyze] dataset=%s complete layers=%d elapsed_s=%.2f",
            dataset,
            len(stats),
            time.time() - dataset_t0,
        )
    return stats


def _rank_layers(
    per_dataset_stats: dict[str, list[dict[str, float]]],
) -> list[RankedLayer]:
    datasets = list(per_dataset_stats.keys())
    if not datasets:
        raise ValueError("per_dataset_stats must be non-empty")

    num_layers_by_ds = {d: len(stats) for d, stats in per_dataset_stats.items()}
    unique_lengths = set(num_layers_by_ds.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"dataset analyses disagree on layer count: {num_layers_by_ds}"
        )
    num_layers = unique_lengths.pop()

    per_layer_scores: list[tuple[int, float, dict[str, float], float]] = []

    per_dataset_ranks: dict[str, dict[int, int]] = {}
    for dataset in datasets:
        scores = [
            (i, per_dataset_stats[dataset][i]["avg"]) for i in range(num_layers)
        ]
        scores.sort(key=lambda x: x[1])
        per_dataset_ranks[dataset] = {
            idx: rank for rank, (idx, _) in enumerate(scores, start=1)
        }

    for layer_idx in range(num_layers):
        per_dataset = {
            dataset: per_dataset_stats[dataset][layer_idx]["avg"]
            for dataset in datasets
        }
        aggregate = sum(per_dataset.values()) / len(per_dataset)
        avg_rank = sum(
            per_dataset_ranks[dataset][layer_idx] for dataset in datasets
        ) / len(datasets)
        per_layer_scores.append((layer_idx, aggregate, per_dataset, avg_rank))

    per_layer_scores.sort(key=lambda row: row[3])
    ranked: list[RankedLayer] = []
    for rank, (layer, aggregate, per_dataset, _avg_rank) in enumerate(per_layer_scores, start=1):
        ranked.append(
            RankedLayer(
                layer=layer,
                aggregate_score=aggregate,
                rank=rank,
                per_dataset_scores=per_dataset,
            )
        )
    return ranked


def compute_weak_layer_report(
    artifact: ModelArtifact,
    datasets: Iterable[str],
    num_samples: int = 100,
    batch_size: int = 8,
    metric: SupportedMetric = "cosine",
    max_length: int = 512,
    output_dir: Path | str | None = None,
    *,
    verbose: bool = False,
) -> WeakLayerReport:
    """Run weak-layer analysis against an already-loaded ModelArtifact."""
    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("datasets must contain at least one entry")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if verbose:
        logger.info(
            "[analyze] starting weak-layer report model=%s datasets=%s samples=%d batch=%d metric=%s",
            artifact.model_spec.model_id,
            dataset_list,
            num_samples,
            batch_size,
            metric,
        )

    per_dataset_stats: dict[str, list[dict[str, float]]] = {}
    for dataset in dataset_list:
        per_dataset_stats[dataset] = _analyze_dataset(
            artifact=artifact,
            dataset=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            metric=metric,
            max_length=max_length,
            verbose=verbose,
        )

    ranked = _rank_layers(per_dataset_stats)

    if output_dir is not None:
        out = Path(output_dir)
        if verbose:
            logger.info("[analyze] writing analysis artifacts to %s", out)
        for dataset, stats in per_dataset_stats.items():
            write_legacy_csv(out, dataset, metric, stats)
            write_legacy_png(out, dataset, metric, stats)
        write_combined_png(
            out, metric, per_dataset_stats,
            model_id=artifact.model_spec.model_id,
        )

    if verbose:
        logger.info(
            "[analyze] weak-layer report complete top_layers=%s",
            [r.layer for r in ranked[: min(10, len(ranked))]],
        )

    return WeakLayerReport(
        model_spec=artifact.model_spec,
        datasets=dataset_list,
        ranked_layers=ranked,
        metadata={
            "metric": metric,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "max_length": max_length,
        },
    )


def analyze_weak_layers(
    model_spec: ModelSpec,
    datasets: Iterable[str],
    num_samples: int = 100,
    batch_size: int = 8,
    metric: SupportedMetric = "cosine",
    max_length: int = 512,
    output_dir: Path | str | None = None,
    *,
    loader=None,
    verbose: bool = False,
) -> WeakLayerReport:
    """Load a model, measure per-layer contributions, rank layers weakest-first."""
    if verbose:
        logger.info(
            "[analyze] loading model=%s revision=%s dtype=%s device_map=%s",
            model_spec.model_id,
            model_spec.revision,
            model_spec.dtype,
            model_spec.device_map,
        )
    with NASSession(loader=loader or TransformersModelLoader()) as session:
        artifact = session.load(model_spec)
        if verbose:
            logger.info("[analyze] model loaded; starting layer contribution analysis")
        return compute_weak_layer_report(
            artifact=artifact,
            datasets=datasets,
            num_samples=num_samples,
            batch_size=batch_size,
            metric=metric,
            max_length=max_length,
            output_dir=output_dir,
            verbose=verbose,
        )


__all__ = [
    "DatasetLoadError",
    "analyze_weak_layers",
    "compute_weak_layer_report",
]
