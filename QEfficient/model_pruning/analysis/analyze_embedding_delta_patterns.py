#!/usr/bin/env python3
"""
Embedding Delta Pattern Analyzer

Deep diagnostic analysis of how embeddings change across skipped layers.
Answers the key questions needed to design better compensation strategies:

  1. How much does the delta vary by token position?
  2. How different are deltas across datasets (domain mismatch)?
  3. How different are prefill vs. decode deltas?
  4. How much variance does the mean vector explain?
  5. What are the principal directions of change (PCA)?

Outputs:
  - Per-position delta vectors (.pt files)
  - Per-dataset delta vectors (.pt files)
  - Prefill vs. decode delta vectors (.pt files)
  - Variance decomposition report (.json)
  - Visualization plots (.png)
  - Recommendation report (.txt)

Usage:
    python analysis/analyze_embedding_delta_patterns.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --start-layer 18 \\
        --end-layer 22 \\
        --datasets wikitext gsm8k hellaswag winogrande \\
        --num-samples 200 \\
        --output-dir ./delta_analysis

Author: LLM Interpretability Engineer
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_model_name(model_id: str) -> str:
    safe = re.sub(r'[/\\:*?"<>|]', "_", model_id)
    return safe.strip("._")


def setup_device_and_dtype(device_arg: str, dtype_arg: str):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_arg == "auto"
        else torch.device(device_arg)
    )
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[dtype_arg]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    print(f"Device: {device}  |  Dtype: {dtype}")
    return device, dtype


def load_model_and_tokenizer(model_id: str, device, dtype):
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    print(f"Model: {model.config.num_hidden_layers} layers, hidden_dim={model.config.hidden_size}")
    return model, tokenizer


def load_dataset_samples(dataset_name: str, num_samples: int) -> List[str]:
    print(f"  Loading {dataset_name} ({num_samples} samples)...")
    prompts: List[str] = []
    try:
        if dataset_name == "wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            for item in ds:
                text = item["text"].strip()
                if len(text) > 50 and not text.startswith("="):
                    prompts.append(text)
                    if len(prompts) >= num_samples:
                        break
        elif dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(item["question"])
        elif dataset_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(f"{item['ctx']} {item['activity_label']}")
        elif dataset_name == "winogrande":
            ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(item["sentence"])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        print(f"    Loaded {len(prompts)} samples.")
        return prompts
    except Exception as exc:
        print(f"    Error: {exc}. Using fallback.")
        return ["The quick brown fox jumps over the lazy dog."] * min(num_samples, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_prefill_deltas(
    model,
    tokenizer,
    prompts: List[str],
    start_layer: int,
    end_layer: int,
    device,
    max_input_len: int = 256,
    batch_size: int = 8,
) -> Dict:
    """
    Collect embedding delta vectors for prefill phase.

    Returns dict with:
      - all_deltas:        List[Tensor[hidden_dim]]  — per-sample mean-over-tokens delta
      - last_token_deltas: List[Tensor[hidden_dim]]  — per-sample last-token delta
      - first_token_deltas:List[Tensor[hidden_dim]]  — per-sample first-token delta
      - position_bucket_deltas: List[Dict[bucket_idx, Tensor[hidden_dim]]]
      - h_start_norms:     List[float]               — ||h_start|| per sample
      - h_end_norms:       List[float]               — ||h_end|| per sample
      - seq_lens:          List[int]
    """
    all_deltas = []
    last_token_deltas = []
    first_token_deltas = []
    position_bucket_deltas = []  # 10 buckets: 0-9%, 10-19%, ..., 90-100%
    h_start_norms = []
    h_end_norms = []
    seq_lens = []

    num_buckets = 10
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="  Collecting prefill deltas"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        try:
            # Right-pad for batching
            orig_side = tokenizer.padding_side
            tokenizer.padding_side = "right"
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
                padding=True,
            ).to(device)
            tokenizer.padding_side = orig_side

            attention_mask = inputs["attention_mask"]  # [B, seq_len]

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of [B, seq_len, D]

            for i in range(len(batch_prompts)):
                actual_len = int(attention_mask[i].sum().item())
                seq_lens.append(actual_len)

                # Extract real tokens only
                h_start = hidden_states[start_layer][i, :actual_len, :].float()  # [actual_len, D]
                h_end = hidden_states[end_layer][i, :actual_len, :].float()      # [actual_len, D]

                delta = h_end - h_start  # [actual_len, D]

                # Mean over all tokens
                all_deltas.append(delta.mean(dim=0).cpu())

                # Last token only
                last_token_deltas.append(delta[-1].cpu())

                # First token only
                first_token_deltas.append(delta[0].cpu())

                # Position bucket deltas
                bucket_dict = {}
                for pos in range(actual_len):
                    bucket = min(int(pos / actual_len * num_buckets), num_buckets - 1)
                    if bucket not in bucket_dict:
                        bucket_dict[bucket] = []
                    bucket_dict[bucket].append(delta[pos].cpu())

                # Average within each bucket
                bucket_means = {}
                for bucket, vecs in bucket_dict.items():
                    bucket_means[bucket] = torch.stack(vecs).mean(dim=0)
                position_bucket_deltas.append(bucket_means)

                # Norms
                h_start_norms.append(h_start.norm(dim=-1).mean().item())
                h_end_norms.append(h_end.norm(dim=-1).mean().item())

        except Exception as exc:
            tqdm.write(f"    Error on batch {batch_idx}: {exc}")
            continue

    return {
        "all_deltas": all_deltas,
        "last_token_deltas": last_token_deltas,
        "first_token_deltas": first_token_deltas,
        "position_bucket_deltas": position_bucket_deltas,
        "h_start_norms": h_start_norms,
        "h_end_norms": h_end_norms,
        "seq_lens": seq_lens,
    }


def collect_decode_deltas(
    model,
    tokenizer,
    prompts: List[str],
    start_layer: int,
    end_layer: int,
    device,
    max_input_len: int = 256,
    num_decode_steps: int = 32,
) -> Dict:
    """
    Collect embedding delta vectors for decode phase.

    Returns dict with:
      - step_deltas: List[Tensor[hidden_dim]]  — one per decode step (all samples)
      - h_start_norms: List[float]
      - h_end_norms: List[float]
    """
    step_deltas = []
    h_start_norms = []
    h_end_norms = []

    for prompt in tqdm(prompts[:min(len(prompts), 50)], desc="  Collecting decode deltas"):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            ).to(device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=True)

            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for _ in range(num_decode_steps):
                with torch.no_grad():
                    dec_out = model(
                        next_tok,
                        past_key_values=past_kv,
                        output_hidden_states=True,
                        use_cache=True,
                    )

                h_start = dec_out.hidden_states[start_layer][0, 0, :].float()  # [D]
                h_end = dec_out.hidden_states[end_layer][0, 0, :].float()      # [D]

                step_deltas.append((h_end - h_start).cpu())
                h_start_norms.append(h_start.norm().item())
                h_end_norms.append(h_end.norm().item())

                past_kv = dec_out.past_key_values
                next_tok = dec_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if next_tok.item() == tokenizer.eos_token_id:
                    break

        except Exception as exc:
            tqdm.write(f"    Error on decode sample: {exc}")
            continue

    return {
        "step_deltas": step_deltas,
        "h_start_norms": h_start_norms,
        "h_end_norms": h_end_norms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_vector(deltas: List[torch.Tensor]) -> torch.Tensor:
    """Compute mean delta vector from a list of delta tensors."""
    if not deltas:
        raise ValueError("Empty delta list")
    return torch.stack(deltas).mean(dim=0)


def compute_variance_explained(
    deltas: List[torch.Tensor],
    mean_vec: torch.Tensor,
) -> float:
    """
    Compute what fraction of total variance is explained by the mean vector.

    Variance explained = 1 - (residual variance / total variance)
    where residual = delta - mean_vec
    """
    if not deltas:
        return 0.0
    stacked = torch.stack(deltas).float()  # [N, D]
    total_var = stacked.var(dim=0).sum().item()
    if total_var < 1e-10:
        return 1.0
    residuals = stacked - mean_vec.float().unsqueeze(0)
    residual_var = residuals.var(dim=0).sum().item()
    return max(0.0, 1.0 - residual_var / total_var)


def compute_cosine_similarity_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(v1.float().unsqueeze(0), v2.float().unsqueeze(0)).item()


def compute_pca_components(
    deltas: List[torch.Tensor],
    n_components: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on delta vectors.

    Returns:
        components: [n_components, hidden_dim]
        explained_variance_ratio: [n_components]
    """
    stacked = torch.stack(deltas).float().numpy()  # [N, D]
    n_components = min(n_components, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(stacked)
    return pca.components_, pca.explained_variance_ratio_


def compute_pca_variance_explained(
    deltas: List[torch.Tensor],
    n_components: int = 32,
) -> float:
    """How much variance is explained by top-K PCA components."""
    _, evr = compute_pca_components(deltas, n_components)
    return float(evr.sum())


def compute_position_bucket_mean_vectors(
    position_bucket_deltas: List[Dict],
    num_buckets: int = 10,
) -> Dict[int, torch.Tensor]:
    """
    Compute mean delta vector for each position bucket.

    Returns:
        Dict mapping bucket_idx -> mean_delta_vector [hidden_dim]
    """
    bucket_accumulator = {b: [] for b in range(num_buckets)}

    for sample_buckets in position_bucket_deltas:
        for bucket, vec in sample_buckets.items():
            bucket_accumulator[bucket].append(vec)

    bucket_means = {}
    for bucket, vecs in bucket_accumulator.items():
        if vecs:
            bucket_means[bucket] = torch.stack(vecs).mean(dim=0)

    return bucket_means


def compute_alpha_grid_search(
    deltas: List[torch.Tensor],
    mean_vec: torch.Tensor,
    alphas: List[float] = None,
) -> Dict[float, float]:
    """
    For each alpha, compute the variance explained by alpha * mean_vec.

    Returns:
        Dict mapping alpha -> variance_explained
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    stacked = torch.stack(deltas).float()  # [N, D]
    total_var = stacked.var(dim=0).sum().item()

    results = {}
    for alpha in alphas:
        scaled_vec = alpha * mean_vec.float()
        residuals = stacked - scaled_vec.unsqueeze(0)
        residual_var = residuals.var(dim=0).sum().item()
        var_explained = max(0.0, 1.0 - residual_var / total_var) if total_var > 1e-10 else 1.0
        results[alpha] = var_explained

    return results


def compute_magnitude_stats(
    deltas: List[torch.Tensor],
    h_start_norms: List[float],
    h_end_norms: List[float],
) -> Dict:
    """Compute statistics about delta magnitudes relative to embedding norms."""
    delta_norms = [d.float().norm().item() for d in deltas]
    relative_norms = [dn / max(hn, 1e-8) for dn, hn in zip(delta_norms, h_start_norms)]
    norm_ratios = [en / max(sn, 1e-8) for en, sn in zip(h_end_norms, h_start_norms)]

    return {
        "delta_norm_mean": float(np.mean(delta_norms)),
        "delta_norm_std": float(np.std(delta_norms)),
        "relative_norm_mean": float(np.mean(relative_norms)),
        "relative_norm_std": float(np.std(relative_norms)),
        "h_start_norm_mean": float(np.mean(h_start_norms)),
        "h_end_norm_mean": float(np.mean(h_end_norms)),
        "norm_ratio_mean": float(np.mean(norm_ratios)),
        "norm_ratio_std": float(np.std(norm_ratios)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_comparison(
    dataset_mean_vecs: Dict[str, torch.Tensor],
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Plot cosine similarity matrix between dataset-specific mean vectors."""
    datasets = list(dataset_mean_vecs.keys())
    n = len(datasets)
    sim_matrix = np.zeros((n, n))

    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            sim_matrix[i, j] = compute_cosine_similarity_between_vectors(
                dataset_mean_vecs[d1], dataset_mean_vecs[d2]
            )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(datasets, fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if abs(sim_matrix[i, j]) < 0.7 else "white")

    ax.set_title(
        f"Cosine Similarity Between Dataset-Specific Mean Delta Vectors\n"
        f"{model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "dataset_similarity_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_position_bucket_analysis(
    bucket_means_by_dataset: Dict[str, Dict[int, torch.Tensor]],
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Plot how delta magnitude varies by token position bucket."""
    datasets = list(bucket_means_by_dataset.keys())
    num_buckets = 10
    bucket_labels = [f"{i*10}-{(i+1)*10}%" for i in range(num_buckets)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Delta norm by position bucket
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    for di, dataset in enumerate(datasets):
        bucket_means = bucket_means_by_dataset[dataset]
        norms = []
        for b in range(num_buckets):
            if b in bucket_means:
                norms.append(bucket_means[b].float().norm().item())
            else:
                norms.append(0.0)
        ax.plot(range(num_buckets), norms, marker="o", linewidth=2,
                color=colors[di], label=dataset)

    ax.set_xticks(range(num_buckets))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Token Position Bucket", fontsize=11)
    ax.set_ylabel("Mean Delta Vector Norm", fontsize=11)
    ax.set_title("Delta Magnitude by Token Position\n(higher = more transformation at that position)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cosine similarity between position buckets (for first dataset)
    if datasets:
        first_dataset = datasets[0]
        bucket_means = bucket_means_by_dataset[first_dataset]
        available_buckets = sorted(bucket_means.keys())
        n = len(available_buckets)
        sim_matrix = np.zeros((n, n))
        for i, b1 in enumerate(available_buckets):
            for j, b2 in enumerate(available_buckets):
                sim_matrix[i, j] = compute_cosine_similarity_between_vectors(
                    bucket_means[b1], bucket_means[b2]
                )

        ax2 = axes[1]
        im = ax2.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
        plt.colorbar(im, ax=ax2, label="Cosine Similarity")
        labels = [bucket_labels[b] for b in available_buckets]
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax2.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax2.set_title(
            f"Cosine Similarity Between Position Buckets\n({first_dataset})",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"Token Position Analysis  |  {model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "position_bucket_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_prefill_vs_decode_comparison(
    prefill_data: Dict[str, Dict],
    decode_data: Dict[str, Dict],
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Compare prefill and decode delta distributions."""
    datasets = list(prefill_data.keys())
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 10))
    if len(datasets) == 1:
        axes = axes.reshape(2, 1)

    for di, dataset in enumerate(datasets):
        pf_deltas = prefill_data[dataset]["last_token_deltas"]
        dc_deltas = decode_data[dataset]["step_deltas"]

        pf_norms = [d.float().norm().item() for d in pf_deltas]
        dc_norms = [d.float().norm().item() for d in dc_deltas]

        # Row 1: Distribution of delta norms
        ax = axes[0, di]
        ax.hist(pf_norms, bins=30, alpha=0.6, color="#2196F3", label="Prefill (last token)", density=True)
        ax.hist(dc_norms, bins=30, alpha=0.6, color="#FF5722", label="Decode", density=True)
        ax.set_xlabel("Delta Norm ||h_end - h_start||", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{dataset}\nDelta Norm Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Row 2: Cosine similarity between prefill and decode mean vectors
        ax2 = axes[1, di]
        if pf_deltas and dc_deltas:
            pf_mean = compute_mean_vector(pf_deltas)
            dc_mean = compute_mean_vector(dc_deltas)
            sim = compute_cosine_similarity_between_vectors(pf_mean, dc_mean)

            # Also compute per-sample cosine similarity between individual deltas and the mean
            pf_sims = [compute_cosine_similarity_between_vectors(d, pf_mean) for d in pf_deltas[:100]]
            dc_sims = [compute_cosine_similarity_between_vectors(d, dc_mean) for d in dc_deltas[:100]]

            ax2.hist(pf_sims, bins=20, alpha=0.6, color="#2196F3", label="Prefill vs. prefill mean", density=True)
            ax2.hist(dc_sims, bins=20, alpha=0.6, color="#FF5722", label="Decode vs. decode mean", density=True)
            ax2.axvline(x=sim, color="green", linestyle="--", linewidth=2,
                        label=f"Prefill mean vs. Decode mean: {sim:.3f}")
            ax2.set_xlabel("Cosine Similarity to Phase Mean Vector", fontsize=10)
            ax2.set_ylabel("Density", fontsize=10)
            ax2.set_title(f"{dataset}\nConsistency of Delta Direction", fontsize=11, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Prefill vs. Decode Delta Analysis  |  {model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "prefill_vs_decode_delta_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_variance_decomposition(
    variance_report: Dict,
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Visualize variance decomposition across datasets and strategies."""
    datasets = list(variance_report.keys())
    strategies = ["mean_vector", "last_token_mean", "pca_32", "position_specific"]
    strategy_labels = ["Global Mean\nVector", "Last-Token\nMean", "PCA\n(32 components)", "Position-\nSpecific"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Variance explained by strategy per dataset
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for si, (strategy, label) in enumerate(zip(strategies, strategy_labels)):
        values = []
        for dataset in datasets:
            val = variance_report[dataset].get(f"var_explained_{strategy}", 0.0)
            values.append(val * 100)  # Convert to percentage
        ax.bar(x + si * width, values, width, label=label, color=colors[si], alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", fontsize=11)
    ax.set_title("Variance Explained by Compensation Strategy\n(higher = better approximation of actual delta)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    # Plot 2: Alpha grid search results
    ax2 = axes[1]
    for di, dataset in enumerate(datasets):
        alpha_results = variance_report[dataset].get("alpha_grid_search", {})
        if alpha_results:
            alphas = sorted(alpha_results.keys())
            values = [alpha_results[a] * 100 for a in alphas]
            ax2.plot(alphas, values, marker="o", linewidth=2, label=dataset)

    ax2.set_xlabel("Compensation Scale (α)", fontsize=11)
    ax2.set_ylabel("Variance Explained (%)", fontsize=11)
    ax2.set_title("Effect of Scaling the Compensation Vector\n(optimal α may differ from 1.0)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="α=1 (current)")

    fig.suptitle(
        f"Variance Decomposition Analysis  |  {model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "variance_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pca_analysis(
    pca_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Plot PCA explained variance and component analysis."""
    datasets = list(pca_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Cumulative explained variance
    ax = axes[0]
    for dataset in datasets:
        _, evr = pca_data[dataset]
        cumulative = np.cumsum(evr) * 100
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3,
                linewidth=2, label=dataset)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axhline(y=80, color="gray", linestyle="-.", alpha=0.5, label="80%")
    ax.axhline(y=95, color="gray", linestyle=":", alpha=0.5, label="95%")
    ax.set_xlabel("Number of PCA Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
    ax.set_title("PCA Cumulative Variance Explained\n(how many components needed to capture the delta?)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Per-component variance (first 20)
    ax2 = axes[1]
    for dataset in datasets:
        _, evr = pca_data[dataset]
        n_show = min(20, len(evr))
        ax2.plot(range(1, n_show + 1), evr[:n_show] * 100, marker="o", markersize=4,
                 linewidth=2, label=dataset)

    ax2.set_xlabel("PCA Component Index", fontsize=11)
    ax2.set_ylabel("Variance Explained (%)", fontsize=11)
    ax2.set_title("Per-Component Variance Explained\n(steep drop = few components capture most variance)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"PCA Analysis of Delta Vectors  |  {model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "pca_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_magnitude_analysis(
    magnitude_stats: Dict[str, Dict],
    output_dir: Path,
    model_id: str,
    start_layer: int,
    end_layer: int,
):
    """Plot magnitude statistics for prefill and decode."""
    datasets = list(magnitude_stats.keys())
    phases = ["prefill", "decode"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Delta norm mean by dataset and phase
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    for pi, phase in enumerate(phases):
        values = [magnitude_stats[d][phase]["delta_norm_mean"] for d in datasets]
        errors = [magnitude_stats[d][phase]["delta_norm_std"] for d in datasets]
        ax.bar(x + pi * width, values, width, yerr=errors, capsize=5,
               label=phase.capitalize(), alpha=0.8,
               color="#2196F3" if phase == "prefill" else "#FF5722")

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Mean Delta Norm ||h_end - h_start||", fontsize=11)
    ax.set_title("Delta Magnitude by Dataset & Phase", fontsize=11, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Relative delta norm (delta / h_start)
    ax2 = axes[1]
    for pi, phase in enumerate(phases):
        values = [magnitude_stats[d][phase]["relative_norm_mean"] for d in datasets]
        errors = [magnitude_stats[d][phase]["relative_norm_std"] for d in datasets]
        ax2.bar(x + pi * width, values, width, yerr=errors, capsize=5,
                label=phase.capitalize(), alpha=0.8,
                color="#2196F3" if phase == "prefill" else "#FF5722")

    ax2.set_xlabel("Dataset", fontsize=11)
    ax2.set_ylabel("Relative Delta Norm (||delta|| / ||h_start||)", fontsize=11)
    ax2.set_title("Relative Delta Magnitude\n(how large is the change relative to embedding?)",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(datasets, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Norm ratio h_end / h_start
    ax3 = axes[2]
    for pi, phase in enumerate(phases):
        values = [magnitude_stats[d][phase]["norm_ratio_mean"] for d in datasets]
        errors = [magnitude_stats[d][phase]["norm_ratio_std"] for d in datasets]
        ax3.bar(x + pi * width, values, width, yerr=errors, capsize=5,
                label=phase.capitalize(), alpha=0.8,
                color="#2196F3" if phase == "prefill" else "#FF5722")

    ax3.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="No change")
    ax3.set_xlabel("Dataset", fontsize=11)
    ax3.set_ylabel("Norm Ratio (||h_end|| / ||h_start||)", fontsize=11)
    ax3.set_title("Embedding Magnitude Change\n(>1 = layers amplify, <1 = layers compress)",
                  fontsize=11, fontweight="bold")
    ax3.set_xticks(x + width / 2)
    ax3.set_xticklabels(datasets, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Magnitude Analysis  |  {model_id}  |  Layers {start_layer}→{end_layer}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "magnitude_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    variance_report: Dict,
    dataset_similarity: Dict,
    prefill_decode_similarity: Dict,
    magnitude_stats: Dict,
    pca_data: Dict,
) -> List[str]:
    """
    Generate actionable recommendations based on the analysis.

    Returns a list of recommendation strings, ordered by priority.
    """
    recommendations = []
    datasets = list(variance_report.keys())

    # 1. Check if mean vector explains enough variance
    mean_var_values = [variance_report[d].get("var_explained_mean_vector", 0) for d in datasets]
    avg_mean_var = np.mean(mean_var_values)

    if avg_mean_var < 0.1:
        recommendations.append(
            "CRITICAL: The global mean vector explains <10% of delta variance. "
            "The current compensation approach is fundamentally limited. "
            "→ Priority: Implement input-adaptive compensation (linear adapter or cluster-based)."
        )
    elif avg_mean_var < 0.3:
        recommendations.append(
            f"LOW: Global mean vector explains only {avg_mean_var*100:.1f}% of variance. "
            "→ Consider PCA-based compensation or input-adaptive approaches."
        )
    else:
        recommendations.append(
            f"MODERATE: Global mean vector explains {avg_mean_var*100:.1f}% of variance. "
            "→ Scaled compensation (alpha tuning) may provide meaningful improvement."
        )

    # 2. Check dataset similarity
    if dataset_similarity:
        sim_values = list(dataset_similarity.values())
        min_sim = min(sim_values)
        if min_sim < 0.5:
            recommendations.append(
                f"DOMAIN MISMATCH: Minimum cross-dataset similarity is {min_sim:.3f}. "
                "The wikitext-derived vector is poorly aligned with some benchmark distributions. "
                "→ Compute dataset-specific compensation vectors for each benchmark."
            )
        elif min_sim < 0.8:
            recommendations.append(
                f"MODERATE DOMAIN MISMATCH: Cross-dataset similarity ranges down to {min_sim:.3f}. "
                "→ Consider using a mixed-dataset compensation vector."
            )

    # 3. Check prefill vs. decode similarity
    if prefill_decode_similarity:
        sim_values = list(prefill_decode_similarity.values())
        avg_sim = np.mean(sim_values)
        if avg_sim < 0.7:
            recommendations.append(
                f"PHASE MISMATCH: Prefill vs. decode mean vector similarity is {avg_sim:.3f}. "
                "Using the same compensation for both phases is suboptimal. "
                "→ Compute separate compensation vectors for prefill and decode phases."
            )

    # 4. Check if last-token compensation is better
    last_token_var_values = [variance_report[d].get("var_explained_last_token_mean", 0) for d in datasets]
    avg_last_token_var = np.mean(last_token_var_values)
    if avg_last_token_var > avg_mean_var * 1.1:
        recommendations.append(
            f"POSITION MATTERS: Last-token mean explains {avg_last_token_var*100:.1f}% vs. "
            f"global mean {avg_mean_var*100:.1f}%. "
            "→ Apply compensation only to the last token position during prefill."
        )

    # 5. Check optimal alpha
    for dataset in datasets:
        alpha_results = variance_report[dataset].get("alpha_grid_search", {})
        if alpha_results:
            best_alpha = max(alpha_results, key=alpha_results.get)
            if abs(best_alpha - 1.0) > 0.2:
                recommendations.append(
                    f"SCALE MISMATCH ({dataset}): Optimal alpha is {best_alpha:.2f} (not 1.0). "
                    f"→ Use scaled compensation with α={best_alpha:.2f} for {dataset}-like inputs."
                )

    # 6. Check PCA dimensionality
    for dataset in datasets:
        _, evr = pca_data.get(dataset, (None, np.array([1.0])))
        if evr is not None:
            n_for_80 = int(np.searchsorted(np.cumsum(evr), 0.8)) + 1
            n_for_95 = int(np.searchsorted(np.cumsum(evr), 0.95)) + 1
            recommendations.append(
                f"PCA DIMENSIONALITY ({dataset}): {n_for_80} components explain 80% of variance, "
                f"{n_for_95} explain 95%. "
                f"→ PCA compensation with {n_for_80}-{n_for_95} components is a good trade-off."
            )

    # 7. Check magnitude scaling
    for dataset in datasets:
        pf_ratio = magnitude_stats[dataset]["prefill"]["norm_ratio_mean"]
        dc_ratio = magnitude_stats[dataset]["decode"]["norm_ratio_mean"]
        if abs(pf_ratio - 1.0) > 0.1:
            recommendations.append(
                f"MAGNITUDE CHANGE ({dataset} prefill): Skipped layers change embedding norm by "
                f"{(pf_ratio-1)*100:+.1f}%. "
                "→ Add magnitude rescaling: multiply output by norm_ratio after compensation."
            )
        if abs(dc_ratio - 1.0) > 0.1:
            recommendations.append(
                f"MAGNITUDE CHANGE ({dataset} decode): Skipped layers change embedding norm by "
                f"{(dc_ratio-1)*100:+.1f}%. "
                "→ Add magnitude rescaling for decode phase."
            )

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def generate_delta_pattern_analysis(
    model: str,
    start_layer: int,
    end_layer: int,
    datasets: List[str],
    num_samples: int = 200,
    num_decode_steps: int = 32,
    device: str = "cuda",
    dtype: str = "bfloat16",
    output_dir: Optional[Path] = None,
    max_input_len: int = 256,
    batch_size: int = 8,
    n_pca_components: int = 64,
) -> Path:
    """
    Run comprehensive embedding delta pattern analysis.

    Args:
        model: HuggingFace model ID
        start_layer: Start of skipped layer range (e.g., 18)
        end_layer: End of skipped layer range (e.g., 22)
        datasets: List of dataset names to analyze
        num_samples: Number of samples per dataset
        num_decode_steps: Number of decode steps to collect
        device: Device selection
        dtype: Model dtype
        output_dir: Output directory
        max_input_len: Maximum input token length
        batch_size: Batch size for prefill collection
        n_pca_components: Number of PCA components to compute

    Returns:
        Path to output directory
    """
    print("=" * 72)
    print("Embedding Delta Pattern Analyzer")
    print("=" * 72)
    print(f"Model       : {model}")
    print(f"Layer range : {start_layer} → {end_layer}")
    print(f"Datasets    : {', '.join(datasets)}")
    print(f"Samples     : {num_samples} per dataset")

    # Setup
    device_obj, dtype_obj = setup_device_and_dtype(device, dtype)
    model_obj, tokenizer = load_model_and_tokenizer(model, device_obj, dtype_obj)

    # Validate layer range
    num_model_layers = model_obj.config.num_hidden_layers
    if not (0 <= start_layer < end_layer <= num_model_layers):
        raise ValueError(
            f"Invalid layer range {start_layer}→{end_layer} "
            f"(model has {num_model_layers} layers)"
        )

    # Create output directory
    if output_dir is None:
        output_dir = Path(f"delta_analysis_{sanitize_model_name(model)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir.absolute()}")

    # ── Collect data for each dataset ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 1: Data Collection")
    print("=" * 72)

    prefill_data: Dict[str, Dict] = {}
    decode_data: Dict[str, Dict] = {}

    for dataset_name in datasets:
        print(f"\n[{dataset_name}]")
        prompts = load_dataset_samples(dataset_name, num_samples)

        print(f"  Collecting prefill deltas...")
        prefill_data[dataset_name] = collect_prefill_deltas(
            model_obj, tokenizer, prompts, start_layer, end_layer,
            device_obj, max_input_len, batch_size
        )

        print(f"  Collecting decode deltas...")
        decode_data[dataset_name] = collect_decode_deltas(
            model_obj, tokenizer, prompts, start_layer, end_layer,
            device_obj, max_input_len, num_decode_steps
        )

    # ── Compute mean vectors and save ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2: Computing Mean Vectors")
    print("=" * 72)

    dataset_mean_vecs: Dict[str, torch.Tensor] = {}
    dataset_last_token_vecs: Dict[str, torch.Tensor] = {}
    dataset_decode_vecs: Dict[str, torch.Tensor] = {}
    bucket_means_by_dataset: Dict[str, Dict[int, torch.Tensor]] = {}

    for dataset_name in datasets:
        pf = prefill_data[dataset_name]
        dc = decode_data[dataset_name]

        # Global mean (all tokens)
        if pf["all_deltas"]:
            mean_vec = compute_mean_vector(pf["all_deltas"])
            dataset_mean_vecs[dataset_name] = mean_vec
            vec_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}_mean_delta.pt"
            torch.save(mean_vec, vec_path)
            print(f"  [{dataset_name}] Global mean vector saved: {vec_path}")

        # Last-token mean
        if pf["last_token_deltas"]:
            last_vec = compute_mean_vector(pf["last_token_deltas"])
            dataset_last_token_vecs[dataset_name] = last_vec
            vec_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}_last_token_mean_delta.pt"
            torch.save(last_vec, vec_path)
            print(f"  [{dataset_name}] Last-token mean vector saved: {vec_path}")

        # Decode mean
        if dc["step_deltas"]:
            decode_vec = compute_mean_vector(dc["step_deltas"])
            dataset_decode_vecs[dataset_name] = decode_vec
            vec_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}_decode_mean_delta.pt"
            torch.save(decode_vec, vec_path)
            print(f"  [{dataset_name}] Decode mean vector saved: {vec_path}")

        # Position bucket means
        bucket_means = compute_position_bucket_mean_vectors(pf["position_bucket_deltas"])
        bucket_means_by_dataset[dataset_name] = bucket_means
        bucket_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}_position_bucket_deltas.pt"
        torch.save(bucket_means, bucket_path)
        print(f"  [{dataset_name}] Position bucket vectors saved: {bucket_path}")

    # ── Variance analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 3: Variance Analysis")
    print("=" * 72)

    variance_report: Dict[str, Dict] = {}
    pca_data: Dict[str, Tuple] = {}

    for dataset_name in datasets:
        pf = prefill_data[dataset_name]
        dc = decode_data[dataset_name]
        report = {}

        if pf["all_deltas"] and dataset_name in dataset_mean_vecs:
            mean_vec = dataset_mean_vecs[dataset_name]

            # Variance explained by global mean
            report["var_explained_mean_vector"] = compute_variance_explained(
                pf["all_deltas"], mean_vec
            )

            # Variance explained by last-token mean
            if dataset_name in dataset_last_token_vecs:
                report["var_explained_last_token_mean"] = compute_variance_explained(
                    pf["last_token_deltas"], dataset_last_token_vecs[dataset_name]
                )

            # Variance explained by PCA
            n_comp = min(n_pca_components, len(pf["all_deltas"]) - 1)
            if n_comp > 0:
                report["var_explained_pca_32"] = compute_pca_variance_explained(
                    pf["all_deltas"], min(32, n_comp)
                )
                components, evr = compute_pca_components(pf["all_deltas"], n_comp)
                pca_data[dataset_name] = (components, evr)

                # Save PCA components
                pca_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}_pca_components.pt"
                torch.save({
                    "components": torch.tensor(components),
                    "explained_variance_ratio": torch.tensor(evr),
                }, pca_path)
                print(f"  [{dataset_name}] PCA components saved: {pca_path}")

            # Position-specific variance explained
            bucket_means = bucket_means_by_dataset.get(dataset_name, {})
            if bucket_means and pf["position_bucket_deltas"]:
                # Compute variance explained by position-specific vectors
                all_pos_deltas = []
                all_pos_means = []
                for sample_buckets in pf["position_bucket_deltas"]:
                    for bucket, vec in sample_buckets.items():
                        if bucket in bucket_means:
                            all_pos_deltas.append(vec)
                            all_pos_means.append(bucket_means[bucket])
                if all_pos_deltas:
                    stacked = torch.stack(all_pos_deltas).float()
                    stacked_means = torch.stack(all_pos_means).float()
                    total_var = stacked.var(dim=0).sum().item()
                    residuals = stacked - stacked_means
                    residual_var = residuals.var(dim=0).sum().item()
                    report["var_explained_position_specific"] = max(
                        0.0, 1.0 - residual_var / total_var
                    ) if total_var > 1e-10 else 1.0

            # Alpha grid search
            report["alpha_grid_search"] = compute_alpha_grid_search(
                pf["all_deltas"], mean_vec
            )

        variance_report[dataset_name] = report

        print(f"  [{dataset_name}]")
        print(f"    Var explained by global mean:    {report.get('var_explained_mean_vector', 0)*100:.2f}%")
        print(f"    Var explained by last-token mean:{report.get('var_explained_last_token_mean', 0)*100:.2f}%")
        print(f"    Var explained by PCA (32 comp):  {report.get('var_explained_pca_32', 0)*100:.2f}%")
        print(f"    Var explained by position-spec:  {report.get('var_explained_position_specific', 0)*100:.2f}%")
        best_alpha = max(report.get("alpha_grid_search", {1.0: 0}),
                         key=report.get("alpha_grid_search", {1.0: 0}).get)
        print(f"    Best alpha:                      {best_alpha:.2f}")

    # ── Cross-dataset similarity ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 4: Cross-Dataset & Phase Similarity")
    print("=" * 72)

    dataset_similarity: Dict[str, float] = {}
    if len(datasets) > 1:
        ref_dataset = datasets[0]
        ref_vec = dataset_mean_vecs.get(ref_dataset)
        if ref_vec is not None:
            for other_dataset in datasets[1:]:
                other_vec = dataset_mean_vecs.get(other_dataset)
                if other_vec is not None:
                    sim = compute_cosine_similarity_between_vectors(ref_vec, other_vec)
                    key = f"{ref_dataset}_vs_{other_dataset}"
                    dataset_similarity[key] = sim
                    print(f"  {ref_dataset} vs {other_dataset}: cosine similarity = {sim:.4f}")

    prefill_decode_similarity: Dict[str, float] = {}
    for dataset_name in datasets:
        pf_vec = dataset_mean_vecs.get(dataset_name)
        dc_vec = dataset_decode_vecs.get(dataset_name)
        if pf_vec is not None and dc_vec is not None:
            sim = compute_cosine_similarity_between_vectors(pf_vec, dc_vec)
            prefill_decode_similarity[dataset_name] = sim
            print(f"  [{dataset_name}] Prefill vs. decode mean similarity: {sim:.4f}")

    # ── Magnitude statistics ───────────────────────────────────────────────────
    magnitude_stats: Dict[str, Dict] = {}
    for dataset_name in datasets:
        pf = prefill_data[dataset_name]
        dc = decode_data[dataset_name]
        magnitude_stats[dataset_name] = {
            "prefill": compute_magnitude_stats(
                pf["all_deltas"], pf["h_start_norms"], pf["h_end_norms"]
            ),
            "decode": compute_magnitude_stats(
                dc["step_deltas"],
                dc["h_start_norms"],
                dc["h_end_norms"]
            ) if dc["step_deltas"] else {
                "delta_norm_mean": 0, "delta_norm_std": 0,
                "relative_norm_mean": 0, "relative_norm_std": 0,
                "h_start_norm_mean": 0, "h_end_norm_mean": 0,
                "norm_ratio_mean": 1.0, "norm_ratio_std": 0,
            },
        }

    # ── Generate visualizations ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 5: Generating Visualizations")
    print("=" * 72)

    if len(datasets) > 1 and len(dataset_mean_vecs) > 1:
        plot_dataset_comparison(dataset_mean_vecs, output_dir, model, start_layer, end_layer)

    plot_position_bucket_analysis(bucket_means_by_dataset, output_dir, model, start_layer, end_layer)

    plot_prefill_vs_decode_comparison(prefill_data, decode_data, output_dir, model, start_layer, end_layer)

    if variance_report:
        plot_variance_decomposition(variance_report, output_dir, model, start_layer, end_layer)

    if pca_data:
        plot_pca_analysis(pca_data, output_dir, model, start_layer, end_layer)

    plot_magnitude_analysis(magnitude_stats, output_dir, model, start_layer, end_layer)

    # ── Generate recommendations ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 6: Generating Recommendations")
    print("=" * 72)

    recommendations = generate_recommendations(
        variance_report, dataset_similarity, prefill_decode_similarity,
        magnitude_stats, pca_data
    )

    # ── Save full report ───────────────────────────────────────────────────────
    # Serialize variance report (convert float keys to strings for JSON)
    serializable_variance = {}
    for ds, rep in variance_report.items():
        serializable_variance[ds] = {}
        for k, v in rep.items():
            if k == "alpha_grid_search":
                serializable_variance[ds][k] = {str(a): float(ve) for a, ve in v.items()}
            else:
                serializable_variance[ds][k] = float(v) if isinstance(v, (float, np.floating)) else v

    report = {
        "model": model,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "datasets": datasets,
        "num_samples": num_samples,
        "variance_report": serializable_variance,
        "dataset_similarity": {k: float(v) for k, v in dataset_similarity.items()},
        "prefill_decode_similarity": {k: float(v) for k, v in prefill_decode_similarity.items()},
        "magnitude_stats": {
            ds: {
                phase: {k: float(v) for k, v in stats.items()}
                for phase, stats in phase_stats.items()
            }
            for ds, phase_stats in magnitude_stats.items()
        },
        "recommendations": recommendations,
    }

    report_path = output_dir / "delta_pattern_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # ── Print recommendations ──────────────────────────────────────────────────
    rec_path = output_dir / "recommendations.txt"
    with open(rec_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("COMPENSATION IMPROVEMENT RECOMMENDATIONS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Layer range: {start_layer} → {end_layer}\n")
        f.write(f"Datasets analyzed: {', '.join(datasets)}\n\n")
        f.write("=" * 72 + "\n")
        f.write("FINDINGS & RECOMMENDATIONS\n")
        f.write("=" * 72 + "\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n\n")

    print(f"  Recommendations saved: {rec_path}")

    print("\n" + "=" * 72)
    print("RECOMMENDATIONS SUMMARY")
    print("=" * 72)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # ── GPU cleanup ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("=" * 72)

    print("\nCleaning up GPU memory...")
    del model_obj, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory — Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    print("✓ GPU memory cleanup complete")

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep diagnostic analysis of embedding delta patterns for layer skipping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--start-layer", type=int, required=True,
                        help="Start of skipped layer range (e.g., 18)")
    parser.add_argument("--end-layer", type=int, required=True,
                        help="End of skipped layer range (e.g., 22)")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["wikitext", "gsm8k", "hellaswag", "winogrande"],
        choices=["wikitext", "gsm8k", "hellaswag", "winogrande"],
        help="Datasets to analyze",
    )
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples per dataset")
    parser.add_argument("--num-decode-steps", type=int, default=32,
                        help="Number of decode steps to collect per sample")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", type=str,
                        choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max-input-len", type=int, default=256,
                        help="Maximum input token length")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for prefill collection")
    parser.add_argument("--n-pca-components", type=int, default=64,
                        help="Number of PCA components to compute")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        generate_delta_pattern_analysis(
            model=args.model,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            datasets=args.datasets,
            num_samples=args.num_samples,
            num_decode_steps=args.num_decode_steps,
            device=args.device,
            dtype=args.dtype,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            n_pca_components=args.n_pca_components,
        )
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
