#!/usr/bin/env python3
"""
Generate Layer Skip Configurations

Pipeline-ready script that analyzes layer contributions and generates
configurations for grid search optimization.

Can be used as:
1. Standalone CLI tool (writes JSON output)
2. Importable module (returns structured data)

Author: Neural Network Optimization Team
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from itertools import combinations
import pandas as pd
import numpy as np


# ============================================================================
# CORE API FUNCTIONS (Pipeline-consumable)
# ============================================================================

def load_contribution_data(contribution_dir: str, metric: str) -> Dict[str, pd.DataFrame]:
    """
    Load layer contribution CSV files from directory.
    
    Args:
        contribution_dir: Directory containing CSV files
        metric: Metric to use ('cosine', 'l2', or 'both')
    
    Returns:
        Dict mapping dataset names to DataFrames
        If metric='both', combines both metrics by averaging normalized values
    """
    contribution_dir = Path(contribution_dir)
    
    print(f"\n{'='*80}")
    print(f"Loading Layer Contribution Data")
    print(f"{'='*80}")
    print(f"Directory: {contribution_dir}")
    print(f"Metric: {metric}")
    
    if metric == 'both':
        # Load both metrics and combine them
        datasets_cosine = {}
        datasets_l2 = {}
        
        # Load cosine files
        cosine_files = list(contribution_dir.glob("*_cosine.csv"))
        print(f"Found {len(cosine_files)} cosine CSV files")
        
        for csv_file in cosine_files:
            filename = csv_file.stem
            if 'multi_dataset' in filename:
                continue
            
            parts = filename.split('_')
            try:
                contrib_idx = parts.index('contributions')
                metric_idx = len(parts) - 1
                dataset_name = '_'.join(parts[contrib_idx+1:metric_idx])
                if not dataset_name:
                    dataset_name = 'single_prompt'
            except (ValueError, IndexError):
                dataset_name = csv_file.stem.replace('_cosine', '')
            
            df = pd.read_csv(csv_file, comment='#')
            datasets_cosine[dataset_name] = df
        
        # Load L2 files
        l2_files = list(contribution_dir.glob("*_l2.csv"))
        print(f"Found {len(l2_files)} L2 CSV files")
        
        for csv_file in l2_files:
            filename = csv_file.stem
            if 'multi_dataset' in filename:
                continue
            
            parts = filename.split('_')
            try:
                contrib_idx = parts.index('contributions')
                metric_idx = len(parts) - 1
                dataset_name = '_'.join(parts[contrib_idx+1:metric_idx])
                if not dataset_name:
                    dataset_name = 'single_prompt'
            except (ValueError, IndexError):
                dataset_name = csv_file.stem.replace('_l2', '')
            
            df = pd.read_csv(csv_file, comment='#')
            datasets_l2[dataset_name] = df
        
        # Combine metrics by averaging normalized values
        datasets = {}
        common_datasets = set(datasets_cosine.keys()) & set(datasets_l2.keys())
        
        print(f"\nCombining metrics for {len(common_datasets)} datasets:\n")
        
        for dataset_name in common_datasets:
            df_cosine = datasets_cosine[dataset_name].copy()
            df_l2 = datasets_l2[dataset_name].copy()
            
            # Normalize both metrics to [0, 1] range
            cosine_norm = (df_cosine['avg_delta'] - df_cosine['avg_delta'].min()) / \
                         (df_cosine['avg_delta'].max() - df_cosine['avg_delta'].min())
            l2_norm = (df_l2['avg_delta'] - df_l2['avg_delta'].min()) / \
                     (df_l2['avg_delta'].max() - df_l2['avg_delta'].min())
            
            # Average the normalized values
            combined_df = df_cosine.copy()
            combined_df['avg_delta'] = (cosine_norm + l2_norm) / 2
            
            # Also combine std_delta if present
            if 'std_delta' in df_cosine.columns and 'std_delta' in df_l2.columns:
                cosine_std_norm = (df_cosine['std_delta'] - df_cosine['std_delta'].min()) / \
                                 (df_cosine['std_delta'].max() - df_cosine['std_delta'].min() + 1e-10)
                l2_std_norm = (df_l2['std_delta'] - df_l2['std_delta'].min()) / \
                             (df_l2['std_delta'].max() - df_l2['std_delta'].min() + 1e-10)
                combined_df['std_delta'] = (cosine_std_norm + l2_std_norm) / 2
            
            datasets[dataset_name] = combined_df
            print(f"  ✓ {dataset_name}: {len(combined_df)} layers (combined cosine + L2)")
        
        if not datasets:
            raise ValueError(f"No matching cosine and L2 files found in {contribution_dir}")
    
    else:
        # Single metric mode
        datasets = {}
        pattern = f"*_{metric}.csv"
        csv_files = list(contribution_dir.glob(pattern))
        
        if not csv_files:
            raise ValueError(f"No CSV files found matching pattern '{pattern}' in {contribution_dir}")
        
        print(f"Found {len(csv_files)} CSV files\n")
        
        for csv_file in csv_files:
            filename = csv_file.stem
            parts = filename.split('_')
            
            if 'multi_dataset' in filename:
                continue
            
            try:
                contrib_idx = parts.index('contributions')
                metric_idx = len(parts) - 1
                dataset_name = '_'.join(parts[contrib_idx+1:metric_idx])
                
                if not dataset_name:
                    dataset_name = 'single_prompt'
            except (ValueError, IndexError):
                dataset_name = csv_file.stem
            
            df = pd.read_csv(csv_file, comment='#')
            datasets[dataset_name] = df
            
            print(f"  ✓ {dataset_name}: {len(df)} layers")
    
    print(f"\n{'='*80}\n")
    
    return datasets


def identify_low_impact_layers(
    datasets: Dict[str, pd.DataFrame],
    threshold_percentile: float
) -> Dict[str, Set[int]]:
    """
    Identify low-contribution layers for each dataset.
    
    Args:
        datasets: Dict of dataset name -> DataFrame
        threshold_percentile: Percentile threshold (e.g., 10 = bottom 10%)
    
    Returns:
        Dict mapping dataset names to sets of low-impact layer indices
    """
    print(f"{'='*80}")
    print(f"Identifying Low-Impact Layers")
    print(f"Threshold: Bottom {threshold_percentile}% of layers")
    print(f"{'='*80}\n")
    
    low_impact_layers = {}
    
    for dataset_name, df in datasets.items():
        # Calculate threshold value
        threshold = np.percentile(df['avg_delta'], threshold_percentile)
        
        # Find layers below threshold
        low_layers = set(df[df['avg_delta'] <= threshold]['layer_index'].values)
        low_impact_layers[dataset_name] = low_layers
        
        print(f"{dataset_name}:")
        print(f"  Threshold value: {threshold:.6f}")
        print(f"  Low-impact layers: {sorted(low_layers)}")
        
        # Show stats for these layers
        low_df = df[df['layer_index'].isin(low_layers)]
        print(f"  Avg delta range: [{low_df['avg_delta'].min():.6f}, {low_df['avg_delta'].max():.6f}]")
        print()
    
    return low_impact_layers


def compute_layer_frequency(
    low_impact_layers: Dict[str, Set[int]]
) -> List[Tuple[int, int, List[str]]]:
    """
    Compute how often each layer appears as low-impact across datasets.
    
    Args:
        low_impact_layers: Dict of dataset -> set of layer indices
    
    Returns:
        List of (layer_id, frequency, supporting_datasets)
        Sorted by frequency (descending), then by layer_id (ascending)
    """
    print(f"{'='*80}")
    print(f"Computing Layer Frequency Across Datasets")
    print(f"{'='*80}\n")
    
    # Count how many datasets each layer appears in
    layer_counts = {}
    layer_datasets = {}
    
    for dataset_name, layers in low_impact_layers.items():
        for layer in layers:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            if layer not in layer_datasets:
                layer_datasets[layer] = []
            layer_datasets[layer].append(dataset_name)
    
    # Sort by frequency (descending), then by layer_id (ascending)
    layer_frequency = [
        (layer, count, layer_datasets[layer])
        for layer, count in sorted(
            layer_counts.items(),
            key=lambda x: (-x[1], x[0])
        )
    ]
    
    # Print summary
    print(f"Layer frequency across {len(low_impact_layers)} datasets:\n")
    print(f"{'Layer':<10} {'Frequency':<15} {'Datasets'}")
    print("-" * 80)
    
    for layer, count, datasets in layer_frequency[:15]:  # Show top 15
        datasets_str = ', '.join(datasets)
        print(f"{layer:<10} {count}/{len(low_impact_layers):<15} {datasets_str}")
    
    if len(layer_frequency) > 15:
        print(f"... and {len(layer_frequency) - 15} more layers")
    
    print(f"\n{'='*80}\n")
    
    return layer_frequency


def create_configurations(
    layer_frequency: List[Tuple[int, int, List[str]]],
    max_skip_layers: int,
    num_datasets: int
) -> List[Dict]:
    """
    Generate grid search configurations.
    
    Args:
        layer_frequency: List of (layer_id, frequency, supporting_datasets)
        max_skip_layers: Maximum number of layers to skip in any config
        num_datasets: Total number of datasets analyzed
    
    Returns:
        List of configuration dicts
    """
    print(f"{'='*80}")
    print(f"Generating Skip Configurations")
    print(f"Max layers to skip: {max_skip_layers}")
    print(f"{'='*80}\n")
    
    configurations = []
    config_id = 0
    
    # Helper function to determine confidence level
    def get_confidence(frequency: int, total_datasets: int) -> str:
        ratio = frequency / total_datasets
        if ratio >= 0.8:
            return "high"
        elif ratio >= 0.5:
            return "medium"
        else:
            return "low"
    
    # 1. Baseline configuration
    configurations.append({
        'id': config_id,
        'name': 'baseline',
        'skip_layers': [],
        'num_skipped': 0,
        'description': 'Full model - no layers skipped',
        'rationale': 'Baseline for comparison',
        'confidence': 'baseline',
        'supporting_datasets': []
    })
    config_id += 1
    
    # Get top candidate layers
    top_layers = [layer for layer, freq, _ in layer_frequency[:max_skip_layers * 3]]
    
    # 2. Single layer configurations (top candidates)
    print("Single layer configurations:")
    for i, (layer, freq, datasets) in enumerate(layer_frequency[:max_skip_layers + 2]):
        confidence = get_confidence(freq, num_datasets)
        
        config = {
            'id': config_id,
            'name': f'skip_layer_{layer}',
            'skip_layers': [layer],
            'num_skipped': 1,
            'description': f'Skip layer {layer} only',
            'rationale': f'Layer {layer} has low impact across {freq}/{num_datasets} datasets',
            'confidence': confidence,
            'supporting_datasets': datasets
        }
        configurations.append(config)
        print(f"  {config_id}. {config['name']}: {config['skip_layers']} (confidence: {confidence})")
        config_id += 1
    
    # 3. Pair configurations (combinations of top layers)
    if max_skip_layers >= 2 and len(top_layers) >= 2:
        print("\nPair configurations:")
        # Generate pairs from top candidates
        num_pairs = min(6, len(list(combinations(top_layers[:5], 2))))
        for pair in list(combinations(top_layers[:5], 2))[:num_pairs]:
            layer_set = sorted(pair)
            
            # Get combined confidence (minimum of the two)
            freq1 = next(f for l, f, _ in layer_frequency if l == layer_set[0])
            freq2 = next(f for l, f, _ in layer_frequency if l == layer_set[1])
            min_freq = min(freq1, freq2)
            confidence = get_confidence(min_freq, num_datasets)
            
            # Get supporting datasets (intersection)
            datasets1 = next(d for l, _, d in layer_frequency if l == layer_set[0])
            datasets2 = next(d for l, _, d in layer_frequency if l == layer_set[1])
            supporting = list(set(datasets1) & set(datasets2))
            
            config = {
                'id': config_id,
                'name': f'skip_layers_{"_".join(map(str, layer_set))}',
                'skip_layers': layer_set,
                'num_skipped': 2,
                'description': f'Skip layers {layer_set[0]} and {layer_set[1]}',
                'rationale': f'Both layers in bottom percentile across {len(supporting)}+ datasets',
                'confidence': confidence,
                'supporting_datasets': supporting if supporting else datasets1
            }
            configurations.append(config)
            print(f"  {config_id}. {config['name']}: {config['skip_layers']} (confidence: {confidence})")
            config_id += 1
    
    # 4. Triplet configurations (combinations of top 3-4 layers)
    if max_skip_layers >= 3 and len(top_layers) >= 3:
        print("\nTriplet configurations:")
        # Generate triplets from top candidates
        num_triplets = min(4, len(list(combinations(top_layers[:5], 3))))
        for triplet in list(combinations(top_layers[:5], 3))[:num_triplets]:
            layer_set = sorted(triplet)
            
            # Get combined confidence (minimum of the three)
            freqs = [next(f for l, f, _ in layer_frequency if l == layer) for layer in layer_set]
            min_freq = min(freqs)
            confidence = get_confidence(min_freq, num_datasets)
            
            # Get supporting datasets (intersection)
            all_datasets = [next(d for l, _, d in layer_frequency if l == layer) for layer in layer_set]
            supporting = list(set(all_datasets[0]) & set(all_datasets[1]) & set(all_datasets[2]))
            
            config = {
                'id': config_id,
                'name': f'skip_layers_{"_".join(map(str, layer_set))}',
                'skip_layers': layer_set,
                'num_skipped': 3,
                'description': f'Skip layers {", ".join(map(str, layer_set))}',
                'rationale': f'All three layers in bottom percentile across {len(supporting)}+ datasets',
                'confidence': confidence,
                'supporting_datasets': supporting if supporting else all_datasets[0]
            }
            configurations.append(config)
            print(f"  {config_id}. {config['name']}: {config['skip_layers']} (confidence: {confidence})")
            config_id += 1
    
    # 5. Consecutive layer configurations
    print("\nConsecutive layer configurations:")
    consecutive_added = 0
    for i, (layer, freq, datasets) in enumerate(layer_frequency[:max_skip_layers * 2]):
        if consecutive_added >= 3:  # Limit consecutive configs
            break
        
        # Check if next layer is also in the list and consecutive
        next_layer = layer + 1
        if any(l == next_layer for l, _, _ in layer_frequency[:max_skip_layers * 2]):
            layer_set = sorted([layer, next_layer])
            
            # Check if this config already exists
            if any(c['skip_layers'] == layer_set for c in configurations):
                continue
            
            freq_next = next(f for l, f, _ in layer_frequency if l == next_layer)
            min_freq = min(freq, freq_next)
            confidence = get_confidence(min_freq, num_datasets)
            
            datasets_next = next(d for l, _, d in layer_frequency if l == next_layer)
            supporting = list(set(datasets) & set(datasets_next))
            
            config = {
                'id': config_id,
                'name': f'skip_consecutive_{layer}_{next_layer}',
                'skip_layers': layer_set,
                'num_skipped': 2,
                'description': f'Skip consecutive layers {layer} and {next_layer}',
                'rationale': f'Adjacent layers both have low impact',
                'confidence': confidence,
                'supporting_datasets': supporting if supporting else datasets
            }
            configurations.append(config)
            print(f"  {config_id}. {config['name']}: {config['skip_layers']} (confidence: {confidence})")
            config_id += 1
            consecutive_added += 1
    
    # 6. High consensus configurations (layers appearing in most datasets)
    print("\nHigh consensus configurations:")
    high_consensus = [
        (layer, freq, datasets)
        for layer, freq, datasets in layer_frequency
        if freq >= max(2, int(num_datasets * 0.6))  # At least 60% of datasets
    ][:3]  # Top 3 high-consensus layers
    
    for layer, freq, datasets in high_consensus:
        # Check if single-layer config already exists
        if any(c['skip_layers'] == [layer] for c in configurations):
            continue
        
        confidence = get_confidence(freq, num_datasets)
        
        config = {
            'id': config_id,
            'name': f'skip_consensus_{layer}',
            'skip_layers': [layer],
            'num_skipped': 1,
            'description': f'Skip layer {layer} (high consensus)',
            'rationale': f'Layer {layer} identified as low-impact by {freq}/{num_datasets} datasets',
            'confidence': confidence,
            'supporting_datasets': datasets
        }
        configurations.append(config)
        print(f"  {config_id}. {config['name']}: {config['skip_layers']} (confidence: {confidence})")
        config_id += 1
    
    print(f"\n{'='*80}")
    print(f"Generated {len(configurations)} configurations")
    print(f"{'='*80}\n")
    
    return configurations


def generate_configurations(
    contribution_dir: str,
    metric: str = "both",
    threshold_percentile: float = 10.0,
    max_skip_layers: int = 3
) -> Dict:
    """
    Main API function - generates all configurations.
    
    This is the primary entry point for pipeline integration.
    
    Args:
        contribution_dir: Directory containing layer contribution CSV files
        metric: Metric to use ('cosine', 'l2', or 'both')
        threshold_percentile: Percentile threshold for low-impact layers
        max_skip_layers: Maximum number of layers to skip in any configuration
    
    Returns:
        Dict with structure:
        {
            'metadata': {...},
            'layer_analysis': {...},
            'configurations': [...]
        }
    """
    # Load data
    datasets = load_contribution_data(contribution_dir, metric)
    
    # Analyze layers
    low_impact_layers = identify_low_impact_layers(datasets, threshold_percentile)
    layer_frequency = compute_layer_frequency(low_impact_layers)
    
    # Generate configurations
    configurations = create_configurations(
        layer_frequency,
        max_skip_layers,
        len(datasets)
    )
    
    # Build metadata
    metadata = {
        'model': 'Qwen/Qwen3-32B',  # Inferred from directory name
        'source_directory': str(contribution_dir),
        'metric': metric,
        'threshold_percentile': threshold_percentile,
        'max_skip_layers': max_skip_layers,
        'num_layers': max(df['layer_index'].max() for df in datasets.values()),
        'datasets_analyzed': sorted(datasets.keys()),
        'num_configurations': len(configurations),
        'timestamp': datetime.now().isoformat()
    }
    
    # Build layer analysis summary
    layer_analysis = {
        'low_impact_by_dataset': {
            name: sorted(list(layers))
            for name, layers in low_impact_layers.items()
        },
        'layer_frequency': [
            {
                'layer': layer,
                'frequency': freq,
                'datasets': datasets,
                'frequency_ratio': freq / len(low_impact_layers)
            }
            for layer, freq, datasets in layer_frequency
        ]
    }
    
    return {
        'metadata': metadata,
        'layer_analysis': layer_analysis,
        'configurations': configurations
    }


# ============================================================================
# OUTPUT FUNCTIONS (For human readability)
# ============================================================================

def save_to_json(config_data: Dict, output_path: str):
    """Save configuration data to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Saved configuration to: {output_path}")


def print_summary(config_data: Dict):
    """Print human-readable summary to console."""
    metadata = config_data['metadata']
    configs = config_data['configurations']
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nModel: {metadata['model']}")
    print(f"Source: {metadata['source_directory']}")
    print(f"Metric: {metadata['metric']}")
    print(f"Threshold: Bottom {metadata['threshold_percentile']}%")
    print(f"Datasets: {', '.join(metadata['datasets_analyzed'])}")
    print(f"Total configurations: {metadata['num_configurations']}")
    
    # Count by confidence
    confidence_counts = {}
    for config in configs:
        conf = config.get('confidence', 'unknown')
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    print(f"\nConfigurations by confidence:")
    for conf, count in sorted(confidence_counts.items()):
        print(f"  {conf}: {count}")
    
    # Count by number of skipped layers
    skip_counts = {}
    for config in configs:
        num = config['num_skipped']
        skip_counts[num] = skip_counts.get(num, 0) + 1
    
    print(f"\nConfigurations by layers skipped:")
    for num, count in sorted(skip_counts.items()):
        print(f"  {num} layers: {count} configs")
    
    print(f"\n{'='*80}\n")


def create_visualization(config_data: Dict, output_path: str):
    """Create heatmap visualization of layer skip configurations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("matplotlib and seaborn are required for --visualize") from exc
    configs = config_data['configurations']
    
    # Get all unique layers that are skipped
    all_layers = set()
    for config in configs:
        if config['skip_layers']:
            all_layers.update(config['skip_layers'])
    
    if not all_layers:
        print("No layers to visualize (only baseline config)")
        return
    
    all_layers = sorted(all_layers)
    
    # Create matrix
    matrix = np.zeros((len(configs), len(all_layers)))
    config_names = []
    
    for i, config in enumerate(configs):
        config_names.append(config['name'])
        if config['skip_layers']:
            for j, layer in enumerate(all_layers):
                if layer in config['skip_layers']:
                    matrix[i, j] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_layers) * 0.5), max(10, len(configs) * 0.4)))
    
    # Create heatmap
    sns.heatmap(
        matrix,
        xticklabels=[f"L{l}" for l in all_layers],
        yticklabels=config_names,
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Layer Skipped'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Layer Skip Configuration Heatmap\n{config_data["metadata"]["model"]} - {config_data["metadata"]["metric"]} metric',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate layer skip configurations for grid search optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--contribution-dir',
        type=str,
        required=True,
        help='Directory containing layer contribution CSV files'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['cosine', 'l2', 'both'],
        default='both',
        help='Metric to use for identifying low-contribution layers'
    )
    parser.add_argument(
        '--threshold-percentile',
        type=float,
        default=10.0,
        help='Percentile threshold for identifying low-contribution layers (e.g., 10 = bottom 10%%)'
    )
    parser.add_argument(
        '--max-skip-layers',
        type=int,
        default=2,
        help='Maximum number of layers to skip in any configuration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='layer_skip_configs.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization heatmap'
    )
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()
    
    print("="*80)
    print("Layer Skip Configuration Generator")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Contribution directory: {args.contribution_dir}")
    print(f"  Metric: {args.metric}")
    print(f"  Threshold: Bottom {args.threshold_percentile}%")
    print(f"  Max skip layers: {args.max_skip_layers}")
    print(f"  Output: {args.output}")
    
    try:
        # Generate configurations (API call)
        config_data = generate_configurations(
            contribution_dir=args.contribution_dir,
            metric=args.metric,
            threshold_percentile=args.threshold_percentile,
            max_skip_layers=args.max_skip_layers
        )
        
        # Output for human consumption
        save_to_json(config_data, args.output)
        print_summary(config_data)
        
        if args.visualize:
            viz_path = args.output.replace('.json', '_heatmap.png')
            create_visualization(config_data, viz_path)
        
        print(f"{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")
        print(f"\nGenerated {config_data['metadata']['num_configurations']} configurations")
        print(f"Output saved to: {args.output}")
        
        if args.visualize:
            print(f"Visualization saved to: {viz_path}")
        
        print(f"\nTo use in pipeline:")
        print(f"  from generate_layer_skip_config import generate_configurations")
        print(f"  configs = generate_configurations('{args.contribution_dir}')")
        print(f"  for cfg in configs['configurations']:")
        print(f"      skip_layers = cfg['skip_layers']")
        print(f"      # Use skip_layers in next pipeline stage")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
