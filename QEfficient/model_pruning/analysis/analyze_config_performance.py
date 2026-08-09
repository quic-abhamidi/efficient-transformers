#!/usr/bin/env python3
"""
Configuration Performance Analyzer

Analyzes layer-skipping configurations to find optimal trade-offs between
performance gains and accuracy impact.

Usage:
    python analyze_config_performance.py \\
        --analysis-dir Llama-3.2-1B-Instruct_Analysis \\
        --accuracy-threshold 5.0 \\
        --output-file optimization_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


class ConfigPerformanceAnalyzer:
    """Analyzes layer-skipping configurations for performance/accuracy trade-offs"""
    
    def __init__(self, analysis_dir: Path, accuracy_threshold: float = 5.0):
        self.analysis_dir = Path(analysis_dir)
        self.accuracy_threshold = accuracy_threshold
        self.baseline_results = None
        self.configurations = []
        self.config_results = {}
        self.total_layers = None
        
    def load_data(self):
        """Load all necessary data from analysis directory"""
        logger.info("Loading analysis data...")
        
        # Load skip configurations
        config_file = self.analysis_dir / "skip_configurations.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            self.configurations = config_data['configurations']
            self.total_layers = config_data['metadata'].get('num_layers', 16)
        
        logger.info(f"Loaded {len(self.configurations)} configurations")
        logger.info(f"Total layers in model: {self.total_layers}")
        
        # Load baseline results
        baseline_file = self.analysis_dir / "baseline" / "benchmark_results.json"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline results not found: {baseline_file}")
        
        with open(baseline_file, 'r') as f:
            self.baseline_results = json.load(f)
        
        logger.info("Loaded baseline results")
        
        # Load skip experiment results
        skip_exp_dir = self.analysis_dir / "skip_experiments"
        if skip_exp_dir.exists():
            for config in self.configurations:
                if config['name'] == 'baseline':
                    continue
                
                result_file = skip_exp_dir / config['name'] / "benchmark_results.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        self.config_results[config['name']] = json.load(f)
        
        logger.info(f"Loaded {len(self.config_results)} skip experiment results")
    
    def extract_metrics(self, results: Dict) -> Dict[str, Dict[str, float]]:
        """Extract key metrics from benchmark results"""
        metrics = {}
        
        for dataset, data in results.get('results', {}).items():
            metrics[dataset] = {}
            
            # Extract all numeric metrics
            for key, value in data.items():
                if isinstance(value, (int, float)) and not key.endswith('_stderr'):
                    metrics[dataset][key] = value
        
        return metrics
    
    def calculate_accuracy_drop(self, baseline_metric: float, config_metric: float) -> Tuple[float, float]:
        """Calculate absolute and relative accuracy drop"""
        absolute_drop = baseline_metric - config_metric
        
        if baseline_metric > 0:
            relative_drop = (absolute_drop / baseline_metric) * 100
        else:
            relative_drop = 0.0
        
        return absolute_drop, relative_drop
    
    def calculate_performance_gain(self, num_skipped: int) -> Dict[str, float]:
        """Calculate theoretical performance gains"""
        if self.total_layers == 0:
            return {
                'layers_skipped': num_skipped,
                'skip_ratio': 0.0,
                'theoretical_speedup': 1.0,
                'flops_reduction': 0.0,
                'memory_reduction': 0.0
            }
        
        skip_ratio = num_skipped / self.total_layers
        
        # Theoretical speedup (simplified linear model)
        # Assumes each layer contributes equally to compute time
        theoretical_speedup = 1.0 / (1.0 - skip_ratio) if skip_ratio < 1.0 else float('inf')
        
        # FLOPs reduction (proportional to layers skipped)
        flops_reduction = skip_ratio * 100
        
        # Memory reduction (parameters + activations)
        # Simplified: assume each layer has equal parameters
        memory_reduction = skip_ratio * 100
        
        return {
            'layers_skipped': num_skipped,
            'skip_ratio': skip_ratio,
            'theoretical_speedup': theoretical_speedup,
            'flops_reduction_percent': flops_reduction,
            'memory_reduction_percent': memory_reduction
        }
    
    def analyze_configuration(self, config: Dict) -> Dict[str, Any]:
        """Analyze a single configuration"""
        config_name = config['name']
        
        if config_name == 'baseline':
            return None
        
        if config_name not in self.config_results:
            logger.warning(f"No results found for {config_name}")
            return None
        
        # Extract metrics
        baseline_metrics = self.extract_metrics(self.baseline_results)
        config_metrics = self.extract_metrics(self.config_results[config_name])
        
        # Calculate accuracy drops per dataset
        dataset_analysis = {}
        all_relative_drops = []
        
        for dataset in baseline_metrics.keys():
            if dataset not in config_metrics:
                continue
            
            dataset_analysis[dataset] = {}
            
            for metric_name in baseline_metrics[dataset].keys():
                if metric_name not in config_metrics[dataset]:
                    continue
                
                baseline_val = baseline_metrics[dataset][metric_name]
                config_val = config_metrics[dataset][metric_name]
                
                abs_drop, rel_drop = self.calculate_accuracy_drop(baseline_val, config_val)
                
                dataset_analysis[dataset][metric_name] = {
                    'baseline': baseline_val,
                    'config': config_val,
                    'absolute_drop': abs_drop,
                    'relative_drop_percent': rel_drop
                }
                
                all_relative_drops.append(abs(rel_drop))
        
        # Calculate average accuracy drop
        avg_relative_drop = sum(all_relative_drops) / len(all_relative_drops) if all_relative_drops else 0.0
        max_relative_drop = max(all_relative_drops) if all_relative_drops else 0.0
        
        # Calculate performance gains
        num_skipped = config.get('num_skipped', len(config.get('skip_layers', [])))
        performance_gains = self.calculate_performance_gain(num_skipped)
        
        # Determine if configuration meets threshold
        meets_threshold = max_relative_drop <= self.accuracy_threshold
        
        return {
            'config_name': config_name,
            'config_id': config.get('id'),
            'skip_layers': config.get('skip_layers', []),
            'num_skipped': num_skipped,
            'confidence': config.get('confidence', 'unknown'),
            'dataset_analysis': dataset_analysis,
            'accuracy_summary': {
                'avg_relative_drop_percent': avg_relative_drop,
                'max_relative_drop_percent': max_relative_drop,
                'meets_threshold': meets_threshold,
                'threshold_percent': self.accuracy_threshold
            },
            'performance_gains': performance_gains,
            'efficiency_score': performance_gains['theoretical_speedup'] / (1 + max_relative_drop / 100)
        }
    
    def analyze_all_configurations(self) -> List[Dict[str, Any]]:
        """Analyze all configurations"""
        logger.info("Analyzing all configurations...")
        
        analyses = []
        for config in self.configurations:
            if config['name'] == 'baseline':
                continue
            
            analysis = self.analyze_configuration(config)
            if analysis:
                analyses.append(analysis)
        
        logger.info(f"Analyzed {len(analyses)} configurations")
        return analyses
    
    def find_optimal_configurations(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal configurations based on different criteria"""
        
        # Filter by threshold
        within_threshold = [a for a in analyses if a['accuracy_summary']['meets_threshold']]
        
        # Sort by different criteria
        best_speedup = None
        best_efficiency = None
        pareto_optimal = []
        
        if within_threshold:
            # Best speedup within threshold
            best_speedup = max(within_threshold, key=lambda x: x['performance_gains']['theoretical_speedup'])
            
            # Best efficiency score
            best_efficiency = max(within_threshold, key=lambda x: x['efficiency_score'])
            
            # Find Pareto optimal configurations
            # A config is Pareto optimal if no other config has both better speedup AND better accuracy
            for config in within_threshold:
                is_pareto = True
                for other in within_threshold:
                    if (other['performance_gains']['theoretical_speedup'] > config['performance_gains']['theoretical_speedup'] and
                        other['accuracy_summary']['max_relative_drop_percent'] < config['accuracy_summary']['max_relative_drop_percent']):
                        is_pareto = False
                        break
                if is_pareto:
                    pareto_optimal.append(config)
        
        # Also find best overall (ignoring threshold)
        best_overall_speedup = max(analyses, key=lambda x: x['performance_gains']['theoretical_speedup']) if analyses else None
        best_overall_efficiency = max(analyses, key=lambda x: x['efficiency_score']) if analyses else None
        
        return {
            'within_threshold': within_threshold,
            'best_speedup_within_threshold': best_speedup,
            'best_efficiency_within_threshold': best_efficiency,
            'pareto_optimal_within_threshold': pareto_optimal,
            'best_overall_speedup': best_overall_speedup,
            'best_overall_efficiency': best_overall_efficiency
        }
    
    def generate_report(self, analyses: List[Dict[str, Any]], optimal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report"""
        
        report = {
            'analysis_directory': str(self.analysis_dir),
            'accuracy_threshold_percent': self.accuracy_threshold,
            'total_layers': self.total_layers,
            'total_configurations_analyzed': len(analyses),
            'configurations_within_threshold': len(optimal['within_threshold']),
            'all_configurations': analyses,
            'optimal_configurations': {
                'best_speedup_within_threshold': optimal['best_speedup_within_threshold'],
                'best_efficiency_within_threshold': optimal['best_efficiency_within_threshold'],
                'pareto_optimal_within_threshold': optimal['pareto_optimal_within_threshold'],
                'best_overall_speedup': optimal['best_overall_speedup'],
                'best_overall_efficiency': optimal['best_overall_efficiency']
            },
            'recommendations': self.generate_recommendations(optimal)
        }
        
        return report
    
    def generate_recommendations(self, optimal: Dict[str, Any]) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []
        
        if not optimal['within_threshold']:
            recommendations.append(
                f"⚠️  WARNING: No configurations meet the {self.accuracy_threshold}% accuracy threshold."
            )
            recommendations.append(
                "Consider: (1) Increasing the threshold, (2) Using different layers, or (3) Skipping fewer layers."
            )
            
            if optimal['best_overall_efficiency']:
                config = optimal['best_overall_efficiency']
                recommendations.append(
                    f"\n📊 Best overall efficiency (ignoring threshold): {config['config_name']}"
                )
                recommendations.append(
                    f"   - Speedup: {config['performance_gains']['theoretical_speedup']:.2f}x"
                )
                recommendations.append(
                    f"   - Max accuracy drop: {config['accuracy_summary']['max_relative_drop_percent']:.2f}%"
                )
        else:
            recommendations.append(
                f"✅ Found {len(optimal['within_threshold'])} configuration(s) within {self.accuracy_threshold}% threshold."
            )
            
            if optimal['best_speedup_within_threshold']:
                config = optimal['best_speedup_within_threshold']
                recommendations.append(
                    f"\n🚀 RECOMMENDED: Best speedup within threshold: {config['config_name']}"
                )
                recommendations.append(
                    f"   - Skip layers: {config['skip_layers']}"
                )
                recommendations.append(
                    f"   - Theoretical speedup: {config['performance_gains']['theoretical_speedup']:.2f}x"
                )
                recommendations.append(
                    f"   - FLOPs reduction: {config['performance_gains']['flops_reduction_percent']:.1f}%"
                )
                recommendations.append(
                    f"   - Max accuracy drop: {config['accuracy_summary']['max_relative_drop_percent']:.2f}%"
                )
                recommendations.append(
                    f"   - Avg accuracy drop: {config['accuracy_summary']['avg_relative_drop_percent']:.2f}%"
                )
            
            if len(optimal['pareto_optimal_within_threshold']) > 1:
                recommendations.append(
                    f"\n📈 {len(optimal['pareto_optimal_within_threshold'])} Pareto-optimal configurations found:"
                )
                for config in optimal['pareto_optimal_within_threshold']:
                    recommendations.append(
                        f"   - {config['config_name']}: "
                        f"{config['performance_gains']['theoretical_speedup']:.2f}x speedup, "
                        f"{config['accuracy_summary']['max_relative_drop_percent']:.2f}% max drop"
                    )
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_file: Path):
        """Save report to JSON file"""
        logger.info(f"Saving report to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Report saved successfully")
    
    def save_results_csv(self, analyses: List[Dict[str, Any]], output_file: Path):
        """Save configuration results to CSV file with % deltas"""
        import csv
        
        logger.info(f"Saving results CSV to {output_file}")
        
        if not analyses:
            logger.warning("No analyses to save to CSV")
            return
        
        # Sort analyses by:
        # 1. Theoretical speedup (descending - highest first)
        # 2. Max accuracy drop (ascending - lowest first)
        # 3. Avg accuracy drop (ascending - lowest first)
        sorted_analyses = sorted(
            analyses,
            key=lambda x: (
                -x['performance_gains']['theoretical_speedup'],
                x['accuracy_summary']['max_relative_drop_percent'],
                x['accuracy_summary']['avg_relative_drop_percent']
            )
        )
        
        logger.info(f"Sorted {len(sorted_analyses)} configurations by speedup (desc) and accuracy drop (asc)")
        
        # Collect all unique datasets and metrics across all configurations
        all_datasets = set()
        all_metrics = {}  # dataset -> set of metrics
        
        for analysis in sorted_analyses:
            for dataset, metrics in analysis['dataset_analysis'].items():
                all_datasets.add(dataset)
                if dataset not in all_metrics:
                    all_metrics[dataset] = set()
                all_metrics[dataset].update(metrics.keys())
        
        # Sort for consistent ordering
        all_datasets = sorted(all_datasets)
        for dataset in all_datasets:
            all_metrics[dataset] = sorted(all_metrics[dataset])
        
        # Build CSV header
        header = [
            'Config Name',
            'Config ID',
            'Skip Layers',
            'Num Layers Skipped',
            'Theoretical Speedup',
            'FLOPs Reduction %',
            'Memory Reduction %',
            'Avg Accuracy Drop %',
            'Max Accuracy Drop %',
            'Meets Threshold',
            'Efficiency Score'
        ]
        
        # Add per-dataset metric columns
        for dataset in all_datasets:
            for metric in all_metrics[dataset]:
                header.extend([
                    f'{dataset}_{metric}_baseline',
                    f'{dataset}_{metric}_config',
                    f'{dataset}_{metric}_delta_%'
                ])
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for analysis in sorted_analyses:
                row = [
                    analysis['config_name'],
                    analysis['config_id'],
                    str(analysis['skip_layers']),
                    analysis['num_skipped'],
                    f"{analysis['performance_gains']['theoretical_speedup']:.4f}",
                    f"{analysis['performance_gains']['flops_reduction_percent']:.2f}",
                    f"{analysis['performance_gains']['memory_reduction_percent']:.2f}",
                    f"{analysis['accuracy_summary']['avg_relative_drop_percent']:.2f}",
                    f"{analysis['accuracy_summary']['max_relative_drop_percent']:.2f}",
                    'Yes' if analysis['accuracy_summary']['meets_threshold'] else 'No',
                    f"{analysis['efficiency_score']:.4f}"
                ]
                
                # Add per-dataset metrics
                for dataset in all_datasets:
                    for metric in all_metrics[dataset]:
                        if dataset in analysis['dataset_analysis'] and metric in analysis['dataset_analysis'][dataset]:
                            metric_data = analysis['dataset_analysis'][dataset][metric]
                            row.extend([
                                f"{metric_data['baseline']:.4f}",
                                f"{metric_data['config']:.4f}",
                                f"{metric_data['relative_drop_percent']:.2f}"
                            ])
                        else:
                            # Missing data
                            row.extend(['N/A', 'N/A', 'N/A'])
                
                writer.writerow(row)
        
        logger.info(f"CSV saved successfully with {len(sorted_analyses)} configurations")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console"""
        print("\n" + "="*80)
        print("CONFIGURATION PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"Analysis Directory: {report['analysis_directory']}")
        print(f"Accuracy Threshold: {report['accuracy_threshold_percent']}%")
        print(f"Total Layers: {report['total_layers']}")
        print(f"Configurations Analyzed: {report['total_configurations_analyzed']}")
        print(f"Configurations Within Threshold: {report['configurations_within_threshold']}")
        print("="*80)
        
        print("\n📋 RECOMMENDATIONS:")
        print("-" * 80)
        for rec in report['recommendations']:
            print(rec)
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for config in report['all_configurations']:
            print(f"\n{config['config_name']}:")
            print(f"  Skip Layers: {config['skip_layers']}")
            print(f"  Theoretical Speedup: {config['performance_gains']['theoretical_speedup']:.2f}x")
            print(f"  FLOPs Reduction: {config['performance_gains']['flops_reduction_percent']:.1f}%")
            print(f"  Max Accuracy Drop: {config['accuracy_summary']['max_relative_drop_percent']:.2f}%")
            print(f"  Avg Accuracy Drop: {config['accuracy_summary']['avg_relative_drop_percent']:.2f}%")
            print(f"  Meets Threshold: {'✅ Yes' if config['accuracy_summary']['meets_threshold'] else '❌ No'}")
            
            print(f"  Per-Dataset Results:")
            for dataset, metrics in config['dataset_analysis'].items():
                print(f"    {dataset}:")
                for metric_name, metric_data in metrics.items():
                    print(f"      {metric_name}: {metric_data['config']:.4f} "
                          f"(baseline: {metric_data['baseline']:.4f}, "
                          f"drop: {metric_data['relative_drop_percent']:.2f}%)")
        
        print("\n" + "="*80)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze layer-skipping configurations for optimal performance/accuracy trade-offs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help="Path to analysis directory (e.g., Llama-3.2-1B-Instruct_Analysis)"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=5.0,
        help="Maximum acceptable accuracy drop percentage"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file path (default: <analysis-dir>/optimization_report.json)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (only save to file)"
    )
    
    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()
    
    # Setup output file
    analysis_dir = Path(args.analysis_dir)
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = analysis_dir / "optimization_report.json"
    
    # Create analyzer
    analyzer = ConfigPerformanceAnalyzer(
        analysis_dir=analysis_dir,
        accuracy_threshold=args.accuracy_threshold
    )
    
    # Load data
    analyzer.load_data()
    
    # Analyze configurations
    analyses = analyzer.analyze_all_configurations()
    
    # Find optimal configurations
    optimal = analyzer.find_optimal_configurations(analyses)
    
    # Generate report
    report = analyzer.generate_report(analyses, optimal)
    
    # Save report
    analyzer.save_report(report, output_file)
    
    # Save CSV results
    csv_output_file = output_file.parent / output_file.name.replace('.json', '.csv')
    analyzer.save_results_csv(analyses, csv_output_file)
    
    # Print summary
    if not args.quiet:
        analyzer.print_summary(report)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
