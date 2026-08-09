#!/usr/bin/env python3
"""
Decoder-Layer Contribution Measurement Tool

Measures how much each decoder layer alters token embeddings in a transformer model.
Computes per-layer deltas using cosine distance or L2 norm and visualizes results.

Author: LLM Interpretability Engineer
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Literal, Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for layer contribution visualizations") from exc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def sanitize_model_name(model_id: str) -> str:
    """Convert model card name to a safe folder name."""
    # Replace slashes and special characters with underscores
    safe_name = re.sub(r'[/\\:*?"<>|]', '_', model_id)
    # Remove any leading/trailing underscores or dots
    safe_name = safe_name.strip('._')
    return safe_name


def create_output_directory(model_id: str, parent_dir: Optional[Path] = None) -> Path:
    """Create output directory for model results."""
    if parent_dir:
        # Create layer_contributions subdirectory within parent
        output_dir = parent_dir / "layer_contributions"
    else:
        # Backward compatibility: create in CWD with model name
        folder_name = sanitize_model_name(f"layer_analysis_{model_id}")
        output_dir = Path(folder_name)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    return output_dir


def load_dataset_samples(dataset_name: str, num_samples: int) -> List[str]:
    """Load samples from specified dataset.

    Delegates to :func:`nas.analysis.datasets.load_dataset_samples` so the new
    API and the legacy CLI stay in sync on dataset sources. Falls back to a
    small built-in prompt list when the dataset cannot be loaded, preserving
    the legacy behavior that tolerated offline runs.
    """
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import (
        DatasetLoadError,
        load_dataset_samples as _shared_loader,
    )

    print(f"\nLoading dataset: {dataset_name}")
    print(f"Requesting {num_samples} samples...")
    try:
        prompts = _shared_loader(dataset_name, num_samples)
    except (DatasetLoadError, ValueError) as exc:
        print(f"Error loading dataset {dataset_name}: {exc}")
        print("Falling back to default prompts")
        return [
            "Explain why transformers use self-attention in two sentences.",
            "What is the capital of France?",
            "Write a Python function to calculate factorial.",
        ][:num_samples]

    print(f"Loaded {len(prompts)} samples from {dataset_name}")
    if prompts:
        print(f"\nExample prompt:\n{prompts[0][:200]}...")
    return prompts


def setup_device_and_dtype(device_arg: str, dtype_arg: str) -> Tuple[torch.device, torch.dtype]:
    """Configure device and dtype based on CLI args and availability."""
    # Device selection
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    # Dtype selection
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[dtype_arg]
    
    # Validate bfloat16 support
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported on this GPU, falling back to float16")
            dtype = torch.float16
    
    print(f"Device: {device}, Dtype: {dtype}")
    return device, dtype


def load_model_and_tokenizer(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load model and tokenizer with specified configuration."""
    print(f"Loading model: {model_id}")
    
    try:
        tokenizer_kwargs = {"trust_remote_code": True}
        if "mistral" in model_id.lower():
            # Required by newer Mistral tokenizer fast path in some environments.
            tokenizer_kwargs["fix_mistral_regex"] = True

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        
        # Handle missing pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with hidden states output enabled
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Move to device if CPU
        if device.type == "cpu":
            model = model.to(device)
        
        # Set to evaluation mode (deterministic, no dropout)
        model.eval()
        
        print(f"Model loaded: {model.config.num_hidden_layers} layers, "
              f"hidden_dim={model.config.hidden_size}")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        print("Attempting fallback to TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return load_model_and_tokenizer(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device,
            dtype
        )


def run_forward_pass(model, tokenizer, prompt: str, device: torch.device, verbose: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Run forward pass and extract hidden states for a single prompt."""
    if verbose:
        print(f"\nPrompt: {prompt[:100]}..." if len(prompt) > 100 else f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    
    if verbose:
        print(f"Tokens: {input_ids.shape[1]} tokens")
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract hidden states: tuple of (embedding, layer_1, ..., layer_N)
    hidden_states = outputs.hidden_states
    
    if verbose:
        print(f"Hidden states captured: {len(hidden_states)} (embedding + {len(hidden_states)-1} layers)")
        print(f"Hidden state shape: {hidden_states[0].shape}")
    
    return input_ids, hidden_states, outputs.logits


def run_forward_pass_batch(model, tokenizer, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Run forward pass and extract hidden states for a batch of prompts."""
    # Tokenize batch with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract hidden states: tuple of (embedding, layer_1, ..., layer_N)
    hidden_states = outputs.hidden_states
    
    return inputs["input_ids"], hidden_states, outputs.logits


def process_dataset_samples(
    model, 
    tokenizer, 
    prompts: List[str], 
    metrics: List[str],
    device: torch.device,
    save_per_sample: bool,
    verbose: bool = False,
    batch_size: int = 1
) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Process multiple samples and aggregate layer contributions for multiple metrics."""
    print(f"\n{'='*60}")
    print(f"Processing {len(prompts)} samples from dataset")
    print(f"Batch size: {batch_size}")
    print(f"Computing metrics: {', '.join(metrics)}")
    print(f"{'='*60}")
    
    # Store per-layer deltas for each sample and metric
    all_sample_deltas = {metric: [] for metric in metrics}
    
    # Process in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        try:
            # Run forward pass for batch
            input_ids, hidden_states, logits = run_forward_pass_batch(model, tokenizer, batch_prompts, device)
            
            # Process each sample in the batch
            for batch_sample_idx, prompt in enumerate(batch_prompts):
                sample_idx = start_idx + batch_sample_idx
                
                # Optionally decode and print the prompt and predicted output
                if verbose:
                    print(f"\nPrompt: {prompt}")
                    # Get predicted token IDs by taking argmax over vocabulary dimension
                    predicted_ids = torch.argmax(logits[batch_sample_idx], dim=-1)
                    print(f"Output: {tokenizer.decode(predicted_ids, skip_special_tokens=True)}")
                
                # Compute deltas for all metrics
                for metric in metrics:
                    sample_deltas = {"sample_id": sample_idx}
                    
                    for layer_idx in range(1, len(hidden_states)):
                        # Extract hidden states for this specific sample in the batch
                        h_prev = hidden_states[layer_idx - 1][batch_sample_idx:batch_sample_idx+1]
                        h_curr = hidden_states[layer_idx][batch_sample_idx:batch_sample_idx+1]
                        
                        if metric == "cosine":
                            cos_sim = F.cosine_similarity(h_curr, h_prev, dim=-1)
                            per_token_delta = 1.0 - cos_sim
                        elif metric == "l2":
                            per_token_delta = torch.norm(h_curr - h_prev, p=2, dim=-1)
                        
                        avg_delta = per_token_delta.mean().item()
                        sample_deltas[f"layer_{layer_idx}"] = avg_delta
                    
                    all_sample_deltas[metric].append(sample_deltas)
            
        except Exception as e:
            tqdm.write(f"  ✗ Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Successfully processed samples")
    print(f"{'='*60}")
    
    # Aggregate across samples for each metric
    results = {}
    for metric in metrics:
        samples_df = pd.DataFrame(all_sample_deltas[metric])
        
        if len(samples_df) == 0:
            continue
            
        layer_columns = [col for col in samples_df.columns if col.startswith("layer_")]
        
        aggregated_results = []
        for layer_col in layer_columns:
            layer_idx = int(layer_col.split("_")[1])
            layer_deltas = samples_df[layer_col].values
            
            aggregated_results.append({
                "layer_index": layer_idx,
                "avg_delta": layer_deltas.mean(),
                "std_delta": layer_deltas.std(),
                "min_delta": layer_deltas.min(),
                "max_delta": layer_deltas.max()
            })
        
        aggregated_df = pd.DataFrame(aggregated_results)
        results[metric] = (aggregated_df, samples_df if save_per_sample else None)
        
        print(f"\n{metric.upper()}: Aggregated statistics computed for {len(aggregated_df)} layers")
    
    return results


def compute_layer_deltas(
    hidden_states: List[torch.Tensor],
    metric: Literal["cosine", "l2"],
    save_per_token: bool
) -> pd.DataFrame:
    """Compute per-layer deltas vs previous layer.

    Delegates the mean-delta math to
    :func:`nas.analysis.contributions.compute_per_layer_deltas` so the typed
    API and the legacy CLI share a single numerical implementation. Per-token
    payloads are still produced locally when ``save_per_token`` is set.
    """
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.contributions import compute_per_layer_deltas

    mean_deltas = compute_per_layer_deltas(hidden_states, metric=metric)

    results = []
    for layer_idx, avg_delta in enumerate(mean_deltas, start=1):
        result = {"layer_index": layer_idx, "avg_delta": avg_delta}

        if save_per_token:
            h_prev = hidden_states[layer_idx - 1]
            h_curr = hidden_states[layer_idx]
            if metric == "cosine":
                per_token_delta = 1.0 - F.cosine_similarity(h_curr, h_prev, dim=-1)
            else:
                per_token_delta = torch.norm(h_curr - h_prev, p=2, dim=-1)
            result["per_token_deltas"] = per_token_delta[0].float().cpu().numpy().tolist()

        results.append(result)

    return pd.DataFrame(results)


def generate_sanity_check(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    """Optional: generate text to verify model loaded correctly."""
    if max_new_tokens <= 0:
        return
    
    print(f"\n=== Sanity Check: Generating {max_new_tokens} tokens ===")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=1.0,  # Ignored when do_sample=False
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated:\n{generated_text}\n")


def save_results(df: pd.DataFrame, metric: str, model_id: str, context: str, output_dir: Path, dataset_name: Optional[str] = None):
    """Save results to CSV with metadata."""
    if dataset_name:
        # No model name prefix when in organized directory structure
        csv_path = output_dir / f"{dataset_name}_{metric}.csv"
    else:
        csv_path = output_dir / f"layer_contributions_{metric}.csv"
    
    # Add metadata as comment header
    with open(csv_path, "w") as f:
        f.write(f"# Model: {model_id}\n")
        if dataset_name:
            f.write(f"# Dataset: {dataset_name}\n")
        else:
            f.write(f"# Prompt: {context}\n")
        f.write(f"# Metric: {metric}\n")
        f.write("#\n")
    
    # Append dataframe
    df.to_csv(str(csv_path), mode="a", index=False)
    
    print(f"Saved results to {csv_path}")
    return str(csv_path)


def visualize_results(df: pd.DataFrame, metric: str, model_id: str, output_dir: Path, dataset_name: Optional[str] = None):
    """Create line plot with top-3 layer annotations and optional error bars."""
    plt = _get_plt()
    if dataset_name:
        # No model name prefix when in organized directory structure
        png_path = output_dir / f"{dataset_name}_{metric}.png"
    else:
        png_path = output_dir / f"layer_contributions_{metric}.png"
    
    # Identify top-3 layers by delta
    top3 = df.nlargest(3, "avg_delta")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if we have std_delta column (dataset mode)
    has_std = "std_delta" in df.columns
    
    if has_std:
        # Line plot with error bars
        ax.errorbar(
            df["layer_index"], 
            df["avg_delta"], 
            yerr=df["std_delta"],
            marker="o", 
            linewidth=2, 
            markersize=6,
            capsize=5,
            capthick=2
        )
    else:
        # Simple line plot
        ax.plot(df["layer_index"], df["avg_delta"], marker="o", linewidth=2, markersize=6)
    
    # Annotate top-3 layers
    for _, row in top3.iterrows():
        layer_idx = row["layer_index"]
        delta = row["avg_delta"]
        ax.annotate(
            f"Layer {layer_idx}\n{delta:.4f}",
            xy=(layer_idx, delta),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
    
    # Labels and title
    metric_label = "Cosine Distance" if metric == "cosine" else "L2 Distance"
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel(f"Average {metric_label}", fontsize=12)
    
    if dataset_name:
        title = f"Decoder-Layer Contribution Analysis\n{model_id} on {dataset_name}"
    else:
        title = f"Decoder-Layer Contribution Analysis\n{model_id}"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()
    print(f"Saved visualization to {png_path}")
    
    return str(png_path), top3


def visualize_combined_results(results_dict: Dict[str, pd.DataFrame], model_id: str, output_dir: Path, dataset_name: Optional[str] = None):
    """Create combined plot with both cosine and L2 metrics."""
    plt = _get_plt()
    if dataset_name:
        # No model name prefix when in organized directory structure
        png_path = output_dir / f"{dataset_name}_combined.png"
    else:
        png_path = output_dir / f"layer_contributions_combined.png"
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Check if we have std_delta column (dataset mode)
    has_std = "std_delta" in list(results_dict.values())[0].columns
    
    # Plot cosine on left y-axis
    if "cosine" in results_dict:
        df_cosine = results_dict["cosine"]
        color1 = 'tab:blue'
        ax1.set_xlabel("Layer Index", fontsize=12)
        ax1.set_ylabel("Cosine Distance", fontsize=12, color=color1)
        
        if has_std:
            ax1.errorbar(
                df_cosine["layer_index"], 
                df_cosine["avg_delta"], 
                yerr=df_cosine["std_delta"],
                marker="o", 
                linewidth=2, 
                markersize=6,
                capsize=5,
                capthick=2,
                color=color1,
                label="Cosine Distance"
            )
        else:
            ax1.plot(
                df_cosine["layer_index"], 
                df_cosine["avg_delta"], 
                marker="o", 
                linewidth=2, 
                markersize=6,
                color=color1,
                label="Cosine Distance"
            )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
    
    # Plot L2 on right y-axis
    if "l2" in results_dict:
        df_l2 = results_dict["l2"]
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel("L2 Distance", fontsize=12, color=color2)
        
        if has_std:
            ax2.errorbar(
                df_l2["layer_index"], 
                df_l2["avg_delta"], 
                yerr=df_l2["std_delta"],
                marker="s", 
                linewidth=2, 
                markersize=6,
                capsize=5,
                capthick=2,
                color=color2,
                label="L2 Distance"
            )
        else:
            ax2.plot(
                df_l2["layer_index"], 
                df_l2["avg_delta"], 
                marker="s", 
                linewidth=2, 
                markersize=6,
                color=color2,
                label="L2 Distance"
            )
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    if dataset_name:
        title = f"Combined Layer Contribution Analysis\n{model_id} on {dataset_name}"
    else:
        title = f"Combined Layer Contribution Analysis\n{model_id}"
    ax1.set_title(title, fontsize=14)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if "l2" in results_dict:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    else:
        ax1.legend(loc='upper left', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()
    print(f"Saved combined visualization to {png_path}")
    
    return str(png_path)


def validate_outputs(df: pd.DataFrame, hidden_states: List[torch.Tensor], csv_path: str, png_path: str):
    """Run validation checks on outputs."""
    print("\n=== Validation Checklist ===")
    
    # 1. Hidden states length
    expected_length = len(hidden_states)
    actual_length = len(df) + 1  # +1 for embedding
    check1 = expected_length == actual_length
    print(f"Hidden states length: {expected_length} == {actual_length}: {check1}")
    
    # 2. Hidden state shapes
    batch_size, seq_len, hidden_dim = hidden_states[0].shape
    check2 = all(h.shape == (batch_size, seq_len, hidden_dim) for h in hidden_states)
    print(f"All hidden states shape [{batch_size}, {seq_len}, {hidden_dim}]: {check2}")
    
    # 3. Delta values non-negative
    check3 = (df["avg_delta"] >= 0).all()
    print(f"All deltas non-negative: {check3}")
    
    # 4. CSV has correct number of rows
    check4 = len(df) == len(hidden_states) - 1
    print(f"CSV has {len(df)} rows (num_layers): {check4}")
    
    # 5. PNG file exists
    check5 = Path(png_path).exists()
    print(f"PNG file exists: {check5}")
    
    # 6. CSV file exists
    check6 = Path(csv_path).exists()
    print(f"CSV file exists: {check6}")
    
    all_passed = all([check1, check2, check3, check4, check5, check6])
    print(f"\nAll checks passed: {all_passed}")
    
    return all_passed


def print_summary(df: pd.DataFrame, top3: Optional[pd.DataFrame], metric: str):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===")
    print(f"Metric: {metric}")
    print(f"Total layers: {len(df)}")
    print(f"Min delta: {df['avg_delta'].min():.6f}")
    print(f"Max delta: {df['avg_delta'].max():.6f}")
    print(f"Mean delta: {df['avg_delta'].mean():.6f}")
    print(f"Std delta: {df['avg_delta'].std():.6f}")
    
    if top3 is not None:
        print("\n=== Top-3 Layers by Delta ===")
        for idx, row in top3.iterrows():
            print(f"  Layer {row['layer_index']}: {row['avg_delta']:.6f}")


def create_multi_dataset_comparison(all_dataset_results: Dict, datasets: List[str], metrics: List[str], model_id: str, output_dir: Path):
    """Create comparison visualizations for multiple datasets."""
    plt = _get_plt()
    print(f"\n{'='*60}")
    print("Creating multi-dataset comparison visualizations...")
    print(f"{'='*60}")
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        for idx, dataset_name in enumerate(datasets):
            df = all_dataset_results[dataset_name][metric][0]
            
            if "std_delta" in df.columns:
                ax.errorbar(
                    df["layer_index"],
                    df["avg_delta"],
                    yerr=df["std_delta"],
                    marker='o',
                    linewidth=2,
                    markersize=5,
                    capsize=4,
                    capthick=1.5,
                    label=dataset_name.upper(),
                    color=colors[idx],
                    alpha=0.8
                )
            else:
                ax.plot(
                    df["layer_index"],
                    df["avg_delta"],
                    marker='o',
                    linewidth=2,
                    markersize=5,
                    label=dataset_name.upper(),
                    color=colors[idx],
                    alpha=0.8
                )
        
        metric_label = "Cosine Distance" if metric == "cosine" else "L2 Distance"
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel(f"Average {metric_label}", fontsize=12)
        ax.set_title(f"Multi-Dataset Layer Contribution Comparison\n{model_id}\nMetric: {metric_label}", 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        overlay_path = output_dir / f"multi_dataset_{metric}.png"
        plt.savefig(str(overlay_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved multi-dataset comparison: {overlay_path}")


def process_single_dataset(model, tokenizer, dataset_name: str, metrics: List[str], 
                          num_samples: int, device: torch.device, 
                          save_per_sample: bool, model_id: str, output_dir: Path, 
                          verbose: bool = False, batch_size: int = 1):
    """Process a single dataset and save results."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset samples
    prompts = load_dataset_samples(dataset_name, num_samples)
    
    # Process all samples for all metrics
    results = process_dataset_samples(
        model, tokenizer, prompts, metrics, device, save_per_sample, verbose, batch_size
    )
    
    # Save results for each metric
    for metric in metrics:
        df, samples_df = results[metric]
        
        # Optional: Save per-sample results
        if save_per_sample and samples_df is not None:
            samples_csv = output_dir / f"{dataset_name}_{metric}_samples.csv"
            samples_df.to_csv(str(samples_csv), index=False)
            print(f"Saved per-sample results to {samples_csv}")
        
        # Save aggregated results
        save_results(df, metric, model_id, dataset_name, output_dir, dataset_name=dataset_name)
        
        # Print summary
        # print_summary(df, None, metric)
    
    return results


def run_multi_dataset_mode(model, tokenizer, datasets: List[str], metrics: List[str],
                          num_samples: int, device: torch.device, 
                          save_per_sample: bool, model_id: str, output_dir: Path, 
                          verbose: bool = False, batch_size: int = 1):
    """Run analysis on multiple datasets."""
    print(f"\n{'='*60}")
    print(f"Running in MULTI-DATASET MODE")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"{'='*60}")
    
    # Store results for all datasets
    all_dataset_results = {}
    
    for dataset_name in datasets:
        results = process_single_dataset(
            model, tokenizer, dataset_name, metrics, 
            num_samples, device, save_per_sample, model_id, output_dir, verbose, batch_size
        )
        all_dataset_results[dataset_name] = results
    
    # Create comparison visualizations
    create_multi_dataset_comparison(all_dataset_results, datasets, metrics, model_id, output_dir)


def run_single_dataset_mode(model, tokenizer, dataset_name: str, metrics: List[str],
                           num_samples: int, device: torch.device,
                           save_per_sample: bool, model_id: str, output_dir: Path, 
                           verbose: bool = False, batch_size: int = 1):
    """Run analysis on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Running in DATASET MODE: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset samples
    prompts = load_dataset_samples(dataset_name, num_samples)
    
    # Process all samples for all metrics
    results = process_dataset_samples(
        model, tokenizer, prompts, metrics, device, save_per_sample, verbose, batch_size
    )
    
    # Save and visualize results for each metric
    for metric in metrics:
        df, samples_df = results[metric]
        
        # Optional: Save per-sample results
        if save_per_sample and samples_df is not None:
            samples_csv = output_dir / f"layer_contributions_{dataset_name}_{metric}_samples.csv"
            samples_df.to_csv(str(samples_csv), index=False)
            print(f"Saved per-sample results to {samples_csv}")
        
        # Save aggregated results
        save_results(df, metric, model_id, dataset_name, output_dir, dataset_name=dataset_name)
        
        # Visualize with error bars
        png_path, top3 = visualize_results(df, metric, model_id, output_dir, dataset_name=dataset_name)
        
        # Print summary
        # print_summary(df, top3, metric)
        
        # Print dataset-specific stats
        if "std_delta" in df.columns:
            print("\n=== Dataset Variability ===")
            print(f"Average std across layers: {df['std_delta'].mean():.6f}")
            print(f"Max std: {df['std_delta'].max():.6f} (Layer {df.loc[df['std_delta'].idxmax(), 'layer_index']})")
            print(f"Min std: {df['std_delta'].min():.6f} (Layer {df.loc[df['std_delta'].idxmin(), 'layer_index']})")
    
    # Create combined visualization if both metrics were computed
    if len(metrics) == 2:
        print(f"\n{'='*60}")
        print("Creating combined visualization...")
        print(f"{'='*60}")
        results_dict = {metric: results[metric][0] for metric in metrics}
        visualize_combined_results(results_dict, model_id, output_dir, dataset_name=dataset_name)


def run_single_prompt_mode(model, tokenizer, prompt: str, metric: str,
                          save_per_token: bool, max_new_tokens: int,
                          device: torch.device, model_id: str, output_dir: Path):
    """Run analysis on a single prompt."""
    print(f"\n{'='*60}")
    print(f"Running in SINGLE-PROMPT MODE")
    print(f"{'='*60}")
    
    # Print the input prompt
    print(f"\n{'='*60}")
    print("INPUT PROMPT:")
    print(f"{'='*60}")
    print(f"{prompt}")
    print(f"{'='*60}")
    
    # Forward pass
    input_ids, hidden_states, logits = run_forward_pass(model, tokenizer, prompt, device)
    
    # Generate and display model output
    print(f"\n{'='*60}")
    print("MODEL OUTPUT (Greedy Decoding):")
    print(f"{'='*60}")
    with torch.no_grad():
        # Generate with greedy decoding
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,  # Generate up to 50 new tokens
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_only = full_output[len(prompt):].strip()
    
    print(f"\nPrompt: {prompt}")
    print(f"\nOutput: {generated_only}")
    print(f"{'='*60}\n")
    
    # Determine which metrics to compute
    metrics = ["cosine", "l2"] if metric == "both" else [metric]
    
    # Optional generation (additional sanity check if requested)
    if max_new_tokens > 0:
        generate_sanity_check(model, tokenizer, prompt, max_new_tokens, device)
    
    # Compute deltas for each metric
    results_dict = {}
    for m in metrics:
        df = compute_layer_deltas(hidden_states, m, save_per_token)
        
        # Save CSV
        csv_path = save_results(df, m, model_id, prompt, output_dir)
        
        # Visualize
        png_path, top3 = visualize_results(df, m, model_id, output_dir)
        
        # Validate
        # validate_outputs(df, hidden_states, csv_path, png_path)
        
        # Print summary
        # print_summary(df, top3, m)
        
        # Store for combined visualization
        results_dict[m] = df
    
    # Create combined visualization if both metrics were computed
    if len(metrics) == 2:
        print(f"\n{'='*60}")
        print("Creating combined visualization...")
        print(f"{'='*60}")
        visualize_combined_results(results_dict, model_id, output_dir)


def generate_layer_analysis(
    model: str,
    dataset: Optional[str] = None,
    prompt: Optional[str] = None,
    num_samples: int = 10000,
    metric: str = "both",
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 16,
    output_dir: Optional[Path] = None,
    save_per_sample: bool = False,
    save_per_token: bool = False,
    verbose: bool = False,
    max_new_tokens: int = 0
) -> Path:
    """
    Generate layer contribution analysis for a transformer model.
    
    This function can be called programmatically from other scripts or used
    via the CLI through the main() function.
    
    Args:
        model: HuggingFace model ID
        dataset: Dataset name(s) or "all" for all datasets. Can be a string or list.
        prompt: Input prompt for single-prompt mode
        num_samples: Number of samples to process from dataset
        metric: Distance metric ("cosine", "l2", or "both")
        device: Device selection ("auto", "cuda", or "cpu")
        dtype: Model dtype ("bfloat16", "float16", or "float32")
        batch_size: Batch size for processing samples
        output_dir: Parent output directory (will create layer_contributions/ subdirectory)
        save_per_sample: Save per-sample results in dataset mode
        save_per_token: Save per-token deltas to CSV
        verbose: Enable verbose output
        max_new_tokens: Generate N tokens for sanity check (0=disabled)
    
    Returns:
        Path to the layer_contributions directory containing results
    
    Raises:
        ValueError: If neither dataset nor prompt is specified
    """
    print("=" * 60)
    print("Decoder-Layer Contribution Measurement Tool")
    print("=" * 60)
    
    # Validate arguments
    if prompt is None and dataset is None:
        raise ValueError("Must specify either 'dataset' or 'prompt' parameter")
    
    if prompt and dataset:
        print("\nWarning: Both prompt and dataset specified. Using dataset mode.")
        prompt = None
    
    # Setup device and dtype
    device_obj, dtype_obj = setup_device_and_dtype(device, dtype)
    
    # Load model and tokenizer
    model_obj, tokenizer = load_model_and_tokenizer(model, device_obj, dtype_obj)
    
    # Create output directory
    parent_dir = Path(output_dir) if output_dir else None
    output_dir_path = create_output_directory(model, parent_dir)
    
    # Determine metrics to compute
    metrics = ["cosine", "l2"] if metric == "both" else [metric]
    
    # Choose mode and run
    if dataset:
        # Handle 'all' keyword and convert string to list if needed
        if isinstance(dataset, str):
            if dataset == "all":
                datasets_to_process = ["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande"]
            else:
                datasets_to_process = [dataset]
        else:
            # Already a list
            datasets_to_process = ["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande"] if "all" in dataset else dataset
        
        # Run appropriate mode
        if len(datasets_to_process) > 1:
            run_multi_dataset_mode(
                model_obj, tokenizer, datasets_to_process, metrics,
                num_samples, device_obj, save_per_sample, model, output_dir_path, 
                verbose, batch_size
            )
        else:
            run_single_dataset_mode(
                model_obj, tokenizer, datasets_to_process[0], metrics,
                num_samples, device_obj, save_per_sample, model, output_dir_path, 
                verbose, batch_size
            )
    else:
        run_single_prompt_mode(
            model_obj, tokenizer, prompt, metric,
            save_per_token, max_new_tokens, device_obj, model, output_dir_path
        )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # Explicit GPU memory cleanup before returning
    print("\nCleaning up GPU memory...")
    del model_obj, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        
        # Log memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print("✓ GPU memory cleanup complete")
    
    return output_dir_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure decoder-layer contribution in transformer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID (fallback: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for analysis (single-prompt mode)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        choices=["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande", "all"],
        default=None,
        help="Dataset(s) to use for analysis. Can specify multiple datasets or 'all' for all datasets"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to process from dataset"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2", "both"],
        default="both",
        help="Distance metric: cosine distance, L2 norm, or both"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Device selection"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype for inference"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Generate N tokens for sanity check (0=disabled)"
    )
    parser.add_argument(
        "--save-per-token",
        action="store_true",
        help="Save per-token deltas to CSV (not just averages)"
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Save per-sample results in dataset mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (print model outputs during processing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing samples (higher values = faster but more memory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent output directory (will create layer_contributions/ subdirectory within it)"
    )
    return parser.parse_args()


def main():
    """Main execution flow - thin CLI wrapper around generate_layer_analysis()."""
    args = parse_args()
    
    # Validate arguments
    if args.prompt is None and args.dataset is None:
        print("\nError: Must specify either --prompt or --dataset")
        print("Using default prompt for demonstration...")
        args.prompt = "Explain why transformers use self-attention in two sentences."
    
    # Convert dataset list to appropriate format
    dataset_arg = args.dataset
    if dataset_arg and len(dataset_arg) == 1:
        dataset_arg = dataset_arg[0]
    
    # Call the main function
    try:
        generate_layer_analysis(
            model=args.model,
            dataset=dataset_arg,
            prompt=args.prompt,
            num_samples=args.num_samples,
            metric=args.metric,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            save_per_sample=args.save_per_sample,
            save_per_token=args.save_per_token,
            verbose=args.verbose,
            max_new_tokens=args.max_new_tokens
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
