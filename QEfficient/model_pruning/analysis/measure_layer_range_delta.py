#!/usr/bin/env python3
"""
Layer Range Delta Measurement Tool

Measures the cumulative contribution of a range of layers by computing the delta
between a start layer (reference) and end layer (target) across multiple samples.

This helps understand how much a specific range of layers (e.g., layers 18-22)
contributes to the overall embedding transformation.

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
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def sanitize_model_name(model_id: str) -> str:
    """Convert model card name to a safe folder name."""
    safe_name = re.sub(r'[/\\:*?"<>|]', '_', model_id)
    safe_name = safe_name.strip('._')
    return safe_name


def create_output_directory(model_id: str, parent_dir: Optional[Path] = None) -> Path:
    """Create output directory for model results."""
    if parent_dir:
        output_dir = parent_dir / "layer_range_delta"
    else:
        folder_name = sanitize_model_name(f"layer_range_delta_{model_id}")
        output_dir = Path(folder_name)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    return output_dir


def load_dataset_samples(dataset_name: str, num_samples: int) -> List[str]:
    """Load samples from specified dataset."""
    print(f"\nLoading dataset: {dataset_name}")
    print(f"Requesting {num_samples} samples...")
    
    prompts = []
    
    try:
        if dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(item["question"])
        
        elif dataset_name == "mbpp":
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(item["text"])
        
        elif dataset_name == "wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            for item in ds:
                text = item["text"].strip()
                if len(text) > 50 and not text.startswith("="):
                    prompts.append(text)
                    if len(prompts) >= num_samples:
                        break
        
        elif dataset_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompt = f"{item['ctx']} {item['activity_label']}"
                prompts.append(prompt)
        
        elif dataset_name == "winogrande":
            ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
            for i, item in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(item["sentence"])
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Loaded {len(prompts)} samples from {dataset_name}")
        
        if prompts:
            print(f"\nExample prompt:\n{prompts[0][:200]}...")
        
        return prompts
    
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to default prompts")
        return [
            "Explain why transformers use self-attention in two sentences.",
            "What is the capital of France?",
            "Write a Python function to calculate factorial."
        ][:num_samples]


def setup_device_and_dtype(device_arg: str, dtype_arg: str) -> Tuple[torch.device, torch.dtype]:
    """Configure device and dtype based on CLI args and availability."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[dtype_arg]
    
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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device.type == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device.type == "cpu":
            model = model.to(device)
        
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


def run_forward_pass_batch(model, tokenizer, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Run forward pass and extract hidden states for a batch of prompts."""
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    return inputs["input_ids"], hidden_states, outputs.logits


def compute_layer_range_delta(
    hidden_states: List[torch.Tensor],
    start_layer: int,
    end_layer: int,
    metric: Literal["cosine", "l2"]
) -> float:
    """
    Compute delta between start_layer and end_layer for a single sample.
    
    Args:
        hidden_states: List of hidden states (embedding, layer_1, ..., layer_N)
        start_layer: Reference layer index (e.g., 18)
        end_layer: Target layer index (e.g., 22)
        metric: Distance metric ("cosine" or "l2")
    
    Returns:
        Average delta across all tokens
    """
    # Note: hidden_states[0] is embedding, hidden_states[1] is layer 1, etc.
    # So to get layer N, we use hidden_states[N]
    h_start = hidden_states[start_layer]
    h_end = hidden_states[end_layer]
    
    if metric == "cosine":
        # Cosine distance: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(h_end, h_start, dim=-1)
        per_token_delta = 1.0 - cos_sim
    
    elif metric == "l2":
        # L2 distance: ||h_end - h_start||_2
        per_token_delta = torch.norm(h_end - h_start, p=2, dim=-1)
    
    # Average across tokens
    avg_delta = per_token_delta.mean().item()
    
    return avg_delta


def process_dataset_samples(
    model, 
    tokenizer, 
    prompts: List[str], 
    start_layer: int,
    end_layer: int,
    metrics: List[str],
    device: torch.device,
    batch_size: int = 1,
    verbose: bool = False,
    save_mean_vector: bool = False
) -> Tuple[Dict[str, List[float]], Optional[torch.Tensor]]:
    """
    Process multiple samples and compute layer range deltas.
    
    Args:
        save_mean_vector: If True, also compute and return mean embedding difference vector
    
    Returns:
        Tuple of (deltas_dict, mean_delta_vector)
        - deltas_dict: Dictionary mapping metric name to list of per-sample deltas
        - mean_delta_vector: Mean embedding difference vector (if save_mean_vector=True)
    """
    print(f"\n{'='*60}")
    print(f"Processing {len(prompts)} samples from dataset")
    print(f"Layer range: {start_layer} → {end_layer}")
    print(f"Batch size: {batch_size}")
    print(f"Computing metrics: {', '.join(metrics)}")
    print(f"{'='*60}")
    
    # Store per-sample deltas for each metric
    all_deltas = {metric: [] for metric in metrics}
    
    # Store embedding differences for computing mean vector
    embedding_diffs = [] if save_mean_vector else None
    
    # Process in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        try:
            # Run forward pass for batch
            input_ids, hidden_states, logits = run_forward_pass_batch(
                model, tokenizer, batch_prompts, device
            )
            
            # Process each sample in the batch
            for batch_sample_idx in range(len(batch_prompts)):
                # Extract hidden states for this specific sample
                sample_hidden_states = [
                    h[batch_sample_idx:batch_sample_idx+1] 
                    for h in hidden_states
                ]
                
                # Compute deltas for all metrics
                for metric in metrics:
                    delta = compute_layer_range_delta(
                        sample_hidden_states,
                        start_layer,
                        end_layer,
                        metric
                    )
                    all_deltas[metric].append(delta)
                
                # Store embedding difference vector if requested
                if save_mean_vector:
                    h_start = sample_hidden_states[start_layer]
                    h_end = sample_hidden_states[end_layer]
                    # Compute difference: h_end - h_start
                    # Average across tokens to get single vector per sample
                    diff = (h_end - h_start).mean(dim=1)  # [1, hidden_dim]
                    embedding_diffs.append(diff.cpu())
                
                if verbose and batch_sample_idx == 0:
                    print(f"\nSample {start_idx + batch_sample_idx}:")
                    for metric in metrics:
                        print(f"  {metric}: {all_deltas[metric][-1]:.6f}")
            
        except Exception as e:
            tqdm.write(f"  ✗ Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(all_deltas[metrics[0]])} samples")
    print(f"{'='*60}")
    
    # Compute mean embedding difference vector if requested
    mean_delta_vector = None
    if save_mean_vector and embedding_diffs:
        # Stack all differences and compute mean
        all_diffs = torch.cat(embedding_diffs, dim=0)  # [num_samples, hidden_dim]
        mean_delta_vector = all_diffs.mean(dim=0)  # [hidden_dim]
        print(f"\n✓ Computed mean embedding delta vector: shape {mean_delta_vector.shape}")
    
    return all_deltas, mean_delta_vector


def save_results(
    all_deltas: Dict[str, List[float]],
    model_id: str,
    dataset_name: str,
    start_layer: int,
    end_layer: int,
    output_dir: Path
) -> str:
    """Save results to CSV with metadata."""
    csv_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}.csv"
    
    # Create dataframe
    df_data = {"sample_id": list(range(len(all_deltas[list(all_deltas.keys())[0]])))}
    for metric, deltas in all_deltas.items():
        df_data[f"delta_{metric}"] = deltas
    
    df = pd.DataFrame(df_data)
    
    # Add metadata as comment header
    with open(csv_path, "w") as f:
        f.write(f"# Model: {model_id}\n")
        f.write(f"# Dataset: {dataset_name}\n")
        f.write(f"# Layer Range: {start_layer} → {end_layer}\n")
        f.write(f"# Metrics: {', '.join(all_deltas.keys())}\n")
        f.write("#\n")
    
    # Append dataframe
    df.to_csv(str(csv_path), mode="a", index=False)
    
    print(f"Saved results to {csv_path}")
    return str(csv_path)


def visualize_results(
    all_deltas: Dict[str, List[float]],
    model_id: str,
    dataset_name: str,
    start_layer: int,
    end_layer: int,
    output_dir: Path,
    max_bars: int = 100
) -> str:
    """
    Create bar chart visualization with mean line.
    
    Args:
        all_deltas: Dictionary mapping metric to list of deltas
        max_bars: Maximum number of bars to show (for readability)
    """
    num_metrics = len(all_deltas)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 6 * num_metrics))
    
    # Handle single metric case
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric, deltas) in enumerate(all_deltas.items()):
        ax = axes[idx]
        
        # Limit number of bars for readability
        num_samples = len(deltas)
        if num_samples > max_bars:
            print(f"Warning: {num_samples} samples exceed max_bars={max_bars}. "
                  f"Showing first {max_bars} samples in visualization.")
            display_deltas = deltas[:max_bars]
            sample_ids = list(range(max_bars))
        else:
            display_deltas = deltas
            sample_ids = list(range(num_samples))
        
        # Create bar chart
        bars = ax.bar(sample_ids, display_deltas, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Color bars by magnitude (gradient)
        norm = plt.Normalize(vmin=min(display_deltas), vmax=max(display_deltas))
        colors = plt.cm.viridis(norm(display_deltas))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        # Add mean line
        mean_delta = np.mean(deltas)
        ax.axhline(y=mean_delta, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_delta:.6f}')
        
        # Labels and title
        metric_label = "Cosine Distance" if metric == "cosine" else "L2 Distance"
        ax.set_xlabel("Sample ID", fontsize=12)
        ax.set_ylabel(f"{metric_label}", fontsize=12)
        ax.set_title(
            f"Layer Range Delta: Layer {start_layer} → {end_layer}\n"
            f"{model_id} on {dataset_name} ({metric.upper()})",
            fontsize=13,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11)
        
        # Add statistics text box
        stats_text = (
            f"Statistics:\n"
            f"Mean: {mean_delta:.6f}\n"
            f"Std: {np.std(deltas):.6f}\n"
            f"Min: {np.min(deltas):.6f}\n"
            f"Max: {np.max(deltas):.6f}\n"
            f"Samples: {num_samples}"
        )
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    png_path = output_dir / f"{dataset_name}_layer{start_layer}-{end_layer}.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {png_path}")
    return str(png_path)


def print_summary(all_deltas: Dict[str, List[float]], start_layer: int, end_layer: int):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Layer Range: {start_layer} → {end_layer}")
    print(f"Number of Samples: {len(all_deltas[list(all_deltas.keys())[0]])}")
    print()
    
    for metric, deltas in all_deltas.items():
        print(f"{metric.upper()} Metric:")
        print(f"  Mean:   {np.mean(deltas):.6f}")
        print(f"  Std:    {np.std(deltas):.6f}")
        print(f"  Min:    {np.min(deltas):.6f}")
        print(f"  Max:    {np.max(deltas):.6f}")
        print(f"  Median: {np.median(deltas):.6f}")
        print()


def generate_layer_range_analysis(
    model: str,
    dataset: str,
    start_layer: int,
    end_layer: int,
    num_samples: int = 100,
    metric: str = "both",
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 16,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    max_bars: int = 100,
    save_mean_vector: bool = False
) -> Path:
    """
    Generate layer range delta analysis for a transformer model.
    
    Args:
        model: HuggingFace model ID
        dataset: Dataset name
        start_layer: Reference layer (e.g., 18)
        end_layer: Target layer (e.g., 22)
        num_samples: Number of samples to process
        metric: Distance metric ("cosine", "l2", or "both")
        device: Device selection ("auto", "cuda", or "cpu")
        dtype: Model dtype ("bfloat16", "float16", or "float32")
        batch_size: Batch size for processing
        output_dir: Parent output directory
        verbose: Enable verbose output
        max_bars: Maximum number of bars to show in visualization
        save_mean_vector: If True, compute and save mean embedding difference vector
    
    Returns:
        Path to the output directory containing results
    """
    print("=" * 60)
    print("Layer Range Delta Measurement Tool")
    print("=" * 60)
    
    # Setup device and dtype
    device_obj, dtype_obj = setup_device_and_dtype(device, dtype)
    
    # Load model and tokenizer
    model_obj, tokenizer = load_model_and_tokenizer(model, device_obj, dtype_obj)
    
    # Validate layer indices
    num_layers = model_obj.config.num_hidden_layers
    if start_layer < 0 or start_layer > num_layers:
        raise ValueError(f"start_layer must be between 0 and {num_layers}")
    if end_layer < 0 or end_layer > num_layers:
        raise ValueError(f"end_layer must be between 0 and {num_layers}")
    if start_layer >= end_layer:
        raise ValueError(f"start_layer must be less than end_layer")
    
    print(f"\nValidated layer range: {start_layer} → {end_layer}")
    print(f"This measures the contribution of layers {start_layer+1} through {end_layer}")
    
    # Create output directory
    parent_dir = Path(output_dir) if output_dir else None
    output_dir_path = create_output_directory(model, parent_dir)
    
    # Determine metrics to compute
    metrics = ["cosine", "l2"] if metric == "both" else [metric]
    
    # Load dataset samples
    prompts = load_dataset_samples(dataset, num_samples)
    
    # Process samples
    all_deltas, mean_delta_vector = process_dataset_samples(
        model_obj, tokenizer, prompts, start_layer, end_layer,
        metrics, device_obj, batch_size, verbose, save_mean_vector
    )
    
    # Save mean embedding delta vector if computed
    if mean_delta_vector is not None:
        vector_path = output_dir_path / f"{dataset}_layer{start_layer}-{end_layer}_mean_delta.pt"
        torch.save(mean_delta_vector, vector_path)
        print(f"\n✓ Saved mean embedding delta vector to {vector_path}")
        print(f"  Vector shape: {mean_delta_vector.shape}")
        print(f"  Vector norm: {torch.norm(mean_delta_vector).item():.6f}")
    
    # Save results
    save_results(all_deltas, model, dataset, start_layer, end_layer, output_dir_path)
    
    # Visualize results
    visualize_results(all_deltas, model, dataset, start_layer, end_layer, 
                     output_dir_path, max_bars)
    
    # Print summary
    print_summary(all_deltas, start_layer, end_layer)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # Cleanup
    print("\nCleaning up GPU memory...")
    del model_obj, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print("✓ GPU memory cleanup complete")
    
    return output_dir_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure layer range delta in transformer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande"],
        help="Dataset to use for analysis"
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        required=True,
        help="Reference layer (start of range, e.g., 18)"
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        required=True,
        help="Target layer (end of range, e.g., 22)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing samples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=100,
        help="Maximum number of bars to show in visualization"
    )
    parser.add_argument(
        "--save-mean-vector",
        action="store_true",
        help="Compute and save mean embedding difference vector for compensation"
    )
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()
    
    try:
        generate_layer_range_analysis(
            model=args.model,
            dataset=args.dataset,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            num_samples=args.num_samples,
            metric=args.metric,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            verbose=args.verbose,
            max_bars=args.max_bars,
            save_mean_vector=args.save_mean_vector
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
