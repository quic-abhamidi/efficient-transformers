"""Typed layer-contribution analysis API."""

from QEfficient.model_pruning.qeff_model_optimizer.analysis.channel_importance import (
    ChannelImportanceReport,
    compute_channel_importance,
)
from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import DatasetLoadError, load_dataset_samples
from QEfficient.model_pruning.qeff_model_optimizer.analysis.head_importance import HeadImportanceReport, compute_head_importance
from QEfficient.model_pruning.qeff_model_optimizer.analysis.kv_similarity import KvSimilarityReport, compute_kv_head_similarity
from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import RankedLayer, WeakLayerReport
from QEfficient.model_pruning.qeff_model_optimizer.analysis.weak_layers import analyze_weak_layers, compute_weak_layer_report
from QEfficient.model_pruning.qeff_model_optimizer.utils.writers import write_combined_png, write_legacy_csv, write_legacy_png

__all__ = [
    "ChannelImportanceReport",
    "DatasetLoadError",
    "HeadImportanceReport",
    "KvSimilarityReport",
    "RankedLayer",
    "WeakLayerReport",
    "analyze_weak_layers",
    "compute_channel_importance",
    "compute_head_importance",
    "compute_kv_head_similarity",
    "compute_weak_layer_report",
    "load_dataset_samples",
    "write_combined_png",
    "write_legacy_csv",
    "write_legacy_png",
]
