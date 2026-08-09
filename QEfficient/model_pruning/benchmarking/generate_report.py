#!/usr/bin/env python3
"""
Benchmark Report Generation and Visualization Script

This script compares benchmark performance between baseline and target models,
generating a CSV comparison table and a bar chart visualization.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for benchmark report charts") from exc


class BenchmarkReportGenerator:
    """Generate comparison reports for benchmark results."""
    
    # Metric priority for selection (higher priority first)
    METRIC_PRIORITY = [
        'acc_norm,none',
        'acc,none', 
        'exact_match,flexible-extract',
        'exact_match,strict-match'
    ]
    
    def __init__(self, results_dir: str = "benchmark_results", output_dir: Optional[Path] = None,
                 baseline_results_path: Optional[Path] = None, target_results_path: Optional[Path] = None,
                 baseline_name: Optional[str] = None, target_name: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            results_dir: Directory containing benchmark results (used when explicit paths not provided)
            output_dir: Optional output directory for comparison reports
            baseline_results_path: Optional direct path to baseline results JSON file
            target_results_path: Optional direct path to target results JSON file
            baseline_name: Optional name for baseline model (for display)
            target_name: Optional name for target model (for display)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir
        self.baseline_results_path = Path(baseline_results_path) if baseline_results_path else None
        self.target_results_path = Path(target_results_path) if target_results_path else None
        self.baseline_data = {}
        self.target_data = {}
        self.baseline_name = baseline_name
        self.target_name = target_name
        
    def identify_models(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Identify baseline and target model directories.
        
        Returns:
            Tuple of (baseline_dir, target_dir)
        """
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        subdirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        # Identify baseline (no modification indicators) and target (has skip/modified)
        baseline_dir = None
        target_dir = None
        
        for subdir in subdirs:
            name = subdir.name.lower()
            # Target models typically have indicators like "skip", "pruned", "quantized", etc.
            if any(indicator in name for indicator in ['skip', 'pruned', 'quantized', 'compressed']):
                target_dir = subdir
            else:
                baseline_dir = subdir
        
        if baseline_dir:
            self.baseline_name = baseline_dir.name
        if target_dir:
            self.target_name = target_dir.name
            
        return baseline_dir, target_dir
    
    def load_results_from_file(self, results_file: Path) -> Dict:
        """
        Load benchmark results directly from a JSON file.
        
        Args:
            results_file: Path to benchmark results JSON file
            
        Returns:
            Dictionary of dataset results
        """
        results = {}
        
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            return results
        
        print(f"Loading results from: {results_file}")
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract results for each dataset
        if 'results' in data:
            for dataset, metrics in data['results'].items():
                # Skip group aggregations (they start with spaces or are in groups)
                if dataset.startswith(' ') or dataset in data.get('groups', {}):
                    continue
                
                # Find the best metric for this dataset
                selected_metric = None
                selected_value = None
                
                for priority_metric in self.METRIC_PRIORITY:
                    if priority_metric in metrics:
                        selected_metric = priority_metric
                        selected_value = metrics[priority_metric]
                        break
                
                # If no priority metric found, use the first available metric
                if selected_metric is None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and not key.endswith('_stderr'):
                            selected_metric = key
                            selected_value = value
                            break
                
                if selected_metric and selected_value is not None:
                    results[dataset] = {
                        'metric': selected_metric,
                        'value': selected_value
                    }
        
        return results
    
    def load_results(self, model_dir: Path) -> Dict:
        """
        Load benchmark results from a model directory.
        
        Args:
            model_dir: Path to model results directory
            
        Returns:
            Dictionary of dataset results
        """
        results = {}
        
        # Find all JSON files in the directory
        json_files = list(model_dir.glob("*.json"))
        
        if not json_files:
            print(f"Warning: No JSON files found in {model_dir}")
            return results
        
        # Load the most recent results file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        return self.load_results_from_file(latest_file)
    
    def generate_comparison_dataframe(self) -> pd.DataFrame:
        """
        Generate a comparison dataframe between baseline and target.
        
        Returns:
            DataFrame with comparison data
        """
        # Find common datasets
        common_datasets = set(self.baseline_data.keys()) & set(self.target_data.keys())
        
        if not common_datasets:
            print("Warning: No common datasets found between baseline and target")
            return pd.DataFrame()
        
        comparison_data = []
        
        for dataset in sorted(common_datasets):
            baseline_info = self.baseline_data[dataset]
            target_info = self.target_data[dataset]
            
            baseline_value = baseline_info['value']
            target_value = target_info['value']
            
            # Calculate differences
            abs_diff = target_value - baseline_value
            pct_change = (abs_diff / baseline_value * 100) if baseline_value != 0 else 0
            
            comparison_data.append({
                'Dataset': dataset,
                'Baseline Score': baseline_value,
                'Target Score': target_value,
                'Absolute Difference': abs_diff,
                'Percentage Change (%)': pct_change,
                'Metric Type': baseline_info['metric']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Add summary row
        summary = {
            'Dataset': 'AVERAGE',
            'Baseline Score': df['Baseline Score'].mean(),
            'Target Score': df['Target Score'].mean(),
            'Absolute Difference': df['Absolute Difference'].mean(),
            'Percentage Change (%)': df['Percentage Change (%)'].mean(),
            'Metric Type': 'N/A'
        }
        
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        return df
    
    def create_output_directory(self) -> Path:
        """
        Create output directory for benchmark comparison results.
        
        Returns:
            Path to output directory
        """
        if self.output_dir:
            # Use provided output directory with performance_comparison subdirectory
            output_dir = self.output_dir / "performance_comparison"
        else:
            # Backward compatibility: create in CWD with target model name
            dir_name = f"benchmark_comparison_{self.target_name}"
            output_dir = Path(dir_name)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_csv_report(self, df: pd.DataFrame, output_dir: Path):
        """
        Save comparison dataframe to CSV.
        
        Args:
            df: Comparison dataframe
            output_dir: Output directory path
        """
        output_file = output_dir / "comparison_metrics.csv"
        df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"CSV report saved to: {output_file}")
    
    def create_bar_chart(self, df: pd.DataFrame, output_dir: Path):
        """
        Create a grouped bar chart comparing baseline and target performance.
        
        Args:
            df: Comparison dataframe
            output_dir: Output directory path
        """
        plt = _get_plt()
        output_file = output_dir / "performance_chart.png"
        
        # Remove the summary row for visualization
        df_plot = df[df['Dataset'] != 'AVERAGE'].copy()
        
        if df_plot.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        datasets = df_plot['Dataset'].tolist()
        baseline_scores = df_plot['Baseline Score'].tolist()
        target_scores = df_plot['Target Score'].tolist()
        
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', 
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, target_scores, width, label='Target',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Benchmark Performance Comparison\n{self.baseline_name} vs {self.target_name}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: {output_file}")
        plt.close()
    
    def create_delta_chart(self, df: pd.DataFrame, output_dir: Path):
        """
        Create a bar chart showing the performance delta (Target - Baseline) as percentage change for each dataset.
        
        Args:
            df: Comparison dataframe
            output_dir: Output directory path
        """
        plt = _get_plt()
        output_file = output_dir / "delta_chart.png"
        
        # Remove the summary row for visualization
        df_plot = df[df['Dataset'] != 'AVERAGE'].copy()
        
        if df_plot.empty:
            print("No data to plot")
            return
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        datasets = df_plot['Dataset'].tolist()
        deltas = df_plot['Percentage Change (%)'].tolist()
        
        x = np.arange(len(datasets))
        
        # Color bars based on positive (green) or negative (red) delta
        colors = ['#27ae60' if delta >= 0 else '#e74c3c' for delta in deltas]
        
        # Create bars
        bars = ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Delta (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Benchmark Performance Delta (Percentage Change)\n{self.baseline_name} vs {self.target_name}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        patches = __import__("matplotlib.patches", fromlist=["Patch"]); Patch = patches.Patch
        legend_elements = [
            Patch(facecolor='#27ae60', alpha=0.8, edgecolor='black', label='Improvement'),
            Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Degradation')
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
        
        # Add value labels on bars
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            # Position label above or below bar depending on sign
            va = 'bottom' if height >= 0 else 'top'
            y_pos = height if height >= 0 else height
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{delta:.2f}%',
                   ha='center', va=va, fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Delta chart saved to: {output_file}")
        plt.close()
    
    def generate_report(self):
        """Generate the complete benchmark report."""
        print("=" * 70)
        print("Benchmark Report Generation")
        print("=" * 70)
        
        # Check if explicit paths are provided
        if self.baseline_results_path and self.target_results_path:
            # Use explicit paths - direct loading
            print(f"\nBaseline Model: {self.baseline_name}")
            print(f"Target Model: {self.target_name}")
            print()
            
            # Load results directly from files
            print("Loading benchmark results...")
            self.baseline_data = self.load_results_from_file(self.baseline_results_path)
            self.target_data = self.load_results_from_file(self.target_results_path)
        else:
            # Use directory-based identification (backward compatibility)
            baseline_dir, target_dir = self.identify_models()
            
            if not baseline_dir or not target_dir:
                raise ValueError("Could not identify both baseline and target model directories")
            
            print(f"\nBaseline Model: {self.baseline_name}")
            print(f"Target Model: {self.target_name}")
            print()
            
            # Load results
            print("Loading benchmark results...")
            self.baseline_data = self.load_results(baseline_dir)
            self.target_data = self.load_results(target_dir)
        
        print(f"Baseline datasets: {len(self.baseline_data)}")
        print(f"Target datasets: {len(self.target_data)}")
        
        # Generate comparison
        print("\nGenerating comparison...")
        df = self.generate_comparison_dataframe()
        
        if df.empty:
            print("Error: No common datasets to compare")
            return
        
        # Display summary
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        print()
        
        # Create output directory and save outputs
        output_dir = self.create_output_directory()
        print(f"\nOutput directory: {output_dir}")
        
        self.save_csv_report(df, output_dir)
        self.create_bar_chart(df, output_dir)
        self.create_delta_chart(df, output_dir)
        
        print("\n" + "=" * 70)
        print("Report generation complete!")
        print("=" * 70)


def main():
    """Main entry point."""
    try:
        generator = BenchmarkReportGenerator()
        generator.generate_report()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
