#!/usr/bin/env python3
"""
Simple Pipeline Orchestrator for Layer Skipping Analysis

A thin orchestration script that calls existing modules to:
0. Measure layer contributions across datasets
1. Generate layer-skip configurations from contribution analysis
2. Run baseline benchmark and skip configuration benchmarks
3. Generate comparison reports for each configuration
4. Analyze optimal configurations based on accuracy threshold
5. Create pipeline summary

This script contains minimal logic - it just wires together existing modules.

Usage:
    python run_pipeline.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --datasets gsm8k hellaswag \\
        --num-samples 100 \\
        --accuracy-threshold 5.0 \\
        --output-dir pipeline_results
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Import existing modules
from QEfficient.model_pruning.optimization.layer_skipping.generate_config import generate_configurations
from QEfficient.model_pruning.core.model_wrapper import SkipLayerModelLoader
from QEfficient.model_pruning.core.pipeline_checkpoint import PipelineCheckpoint
from QEfficient.model_pruning.benchmarking.run_benchmark import run_lm_eval, BENCHMARK_MAPPING, make_json_serializable
from QEfficient.model_pruning.benchmarking.generate_report import BenchmarkReportGenerator
from QEfficient.model_pruning.analysis.measure_layer_contributions import generate_layer_analysis
from QEfficient.model_pruning.analysis.analyze_config_performance import ConfigPerformanceAnalyzer

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


def extract_clean_model_name(model_id: str) -> str:
    """Extract clean model name from HuggingFace model ID."""
    # Extract the model name after the last slash
    if '/' in model_id:
        model_name = model_id.split('/')[-1]
    else:
        model_name = model_id
    
    # Remove any special characters that might cause issues
    import re
    model_name = re.sub(r'[^\w\-.]', '_', model_name)
    
    return model_name


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Simple Pipeline for Layer Skipping Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model name")
    parser.add_argument("--metric", type=str, choices=['cosine', 'l2', 'both'], default='both', help="Metric for config generation")
    parser.add_argument("--threshold-percentile", type=float, default=10.0, help="Percentile threshold for low-impact layers")
    parser.add_argument("--max-skip-layers", type=int, default=5, help="Max layers to skip in any config")
    
    # Dataset arguments - separate for layer analysis and benchmarking
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets for both layer analysis and benchmarking (backward compatibility)")
    parser.add_argument("--layer-datasets", nargs="+", 
                       choices=["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande", "all"],
                       default=None, 
                       help="Datasets for layer contribution analysis (Stage 0). Available: gsm8k, mbpp, wikitext, hellaswag, winogrande, all")
    parser.add_argument("--benchmark-datasets", nargs="+",
                       choices=["gsm8k", "hellaswag", "winogrande", "mmlu", "arc_easy", "arc_challenge", "truthfulqa", "piqa", "boolq", "openbookqa", "all"],
                       default=None,
                       help="Datasets for benchmarking (Stage 2). Available: gsm8k, hellaswag, winogrande, mmlu, arc_easy, arc_challenge, truthfulqa, piqa, boolq, openbookqa, all")
    
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples for both layer analysis and benchmarking per dataset. If not specified, uses all available samples.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: {model_name}_Analysis)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--accuracy-threshold", type=float, default=5.0, help="Maximum acceptable accuracy drop percentage for optimization analysis")
    
    # Checkpoint and resume arguments
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run all stages, ignoring checkpoint")
    parser.add_argument("--resume-from", type=str, 
                       choices=["layer_contributions", "config_generation", "baseline_benchmark", 
                               "skip_benchmarks", "report_generation", "optimization_analysis", "summary"],
                       help="Force resume from specific stage (re-runs that stage and all subsequent stages)")
    parser.add_argument("--skip-failed", action="store_true", help="Skip previously failed configurations in skip_benchmarks stage")
    parser.add_argument("--retry-failed-only", action="store_true", help="Only retry previously failed configurations (skip completed ones)")
    parser.add_argument("--clean-checkpoint", action="store_true", help="Delete existing checkpoint and start fresh")
    
    return parser.parse_args()


def main():
    """Main pipeline execution"""
    args = parse_args()
    start_time = time.time()
    
    # Handle dataset arguments with backward compatibility
    layer_datasets = None
    benchmark_datasets = None
    
    # Define available datasets for each stage
    LAYER_ANALYSIS_DATASETS = ["gsm8k", "mbpp", "wikitext", "hellaswag", "winogrande"]
    BENCHMARK_DATASETS = ["gsm8k", "hellaswag", "winogrande", "mmlu", "arc_easy", "arc_challenge", "truthfulqa", "piqa", "boolq", "openbookqa"]
    
    if args.layer_datasets:
        # Use specified layer datasets
        if "all" in args.layer_datasets:
            layer_datasets = LAYER_ANALYSIS_DATASETS
        else:
            layer_datasets = args.layer_datasets
    elif args.datasets:
        # Backward compatibility: use --datasets for layer analysis
        # Filter to only valid layer analysis datasets
        layer_datasets = [ds for ds in args.datasets if ds in LAYER_ANALYSIS_DATASETS]
        if not layer_datasets:
            logger.warning(f"No valid layer analysis datasets in --datasets. Using default: gsm8k, hellaswag")
            layer_datasets = ["gsm8k", "hellaswag"]
    else:
        # Default layer datasets
        layer_datasets = ["gsm8k", "hellaswag"]
    
    if args.benchmark_datasets:
        # Use specified benchmark datasets
        if "all" in args.benchmark_datasets:
            benchmark_datasets = BENCHMARK_DATASETS
        else:
            benchmark_datasets = args.benchmark_datasets
    elif args.datasets:
        # Backward compatibility: use --datasets for benchmarking
        # Filter to only valid benchmark datasets
        benchmark_datasets = [ds for ds in args.datasets if ds in BENCHMARK_DATASETS]
        if not benchmark_datasets:
            logger.warning(f"No valid benchmark datasets in --datasets. Using default: gsm8k, hellaswag")
            benchmark_datasets = ["gsm8k", "hellaswag"]
    else:
        # Default benchmark datasets
        benchmark_datasets = ["gsm8k", "hellaswag"]
    
    # Extract clean model name and setup output directory
    clean_model_name = extract_clean_model_name(args.model)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"{clean_model_name}_Analysis")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle clean checkpoint flag
    if args.clean_checkpoint:
        checkpoint_path = output_dir / "pipeline_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Deleted existing checkpoint")
    
    # Initialize checkpoint manager
    checkpoint = PipelineCheckpoint(output_dir, args.model)
    
    # Determine if we're resuming
    is_resuming = checkpoint.checkpoint_path.exists() and not args.force_rerun
    force_rerun_active = args.force_rerun
    
    if is_resuming:
        logger.info("\n" + "="*80)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("="*80)
        checkpoint.print_status()
        
        resume_stage = checkpoint.get_resume_stage()
        if resume_stage:
            logger.info(f"Will resume from: {checkpoint.STAGES[resume_stage]['name']}")
        else:
            logger.info("All stages complete - will regenerate summary")
        logger.info("="*80 + "\n")
    
    # Handle --resume-from flag
    if args.resume_from:
        force_rerun_active = False  # Don't force rerun all
        logger.info(f"Forcing resume from stage: {args.resume_from}")
    
    logger.info("="*80)
    logger.info("LAYER SKIPPING PIPELINE")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Layer Analysis Datasets: {layer_datasets}")
    logger.info(f"Benchmark Datasets: {benchmark_datasets}")
    logger.info(f"Output: {output_dir}")
    if args.force_rerun:
        logger.info("Mode: FORCE RE-RUN (ignoring checkpoint)")
    elif args.resume_from:
        logger.info(f"Mode: FORCE RE-RUN from stage '{args.resume_from}'")
    logger.info("="*80)
    
    # Step 0: Measure layer contributions
    stage_name = "layer_contributions"
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 0/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
        contribution_dir = output_dir / "layer_contributions"
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
        
        logger.info(f"\n[STEP 0/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name)
        
        try:
            # Call generate_layer_analysis directly instead of using subprocess
            contribution_dir = generate_layer_analysis(
                model=args.model,
                dataset=layer_datasets,
                num_samples=args.num_samples,
                metric=args.metric,
                device=args.device,
                batch_size=args.batch_size,
                output_dir=output_dir
            )
            
            logger.info(f"✓ Layer contribution analysis complete")
            logger.info(f"  Results saved to: {contribution_dir}")
            
            checkpoint.mark_stage_complete(stage_name)
            
            # Defensive GPU memory cleanup after layer analysis
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("✓ GPU memory cleared after layer analysis")
            
        except Exception as e:
            logger.error(f"Layer contribution analysis failed: {e}")
            checkpoint.mark_stage_failed(stage_name, str(e))
            return
    
    # Step 1: Generate configurations using existing module
    stage_name = "config_generation"
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 1/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
        # Load existing configurations
        with open(output_dir / "skip_configurations.json", 'r') as f:
            config_data = json.load(f)
        configurations = config_data['configurations']
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
        
        logger.info(f"\n[STEP 1/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name)
        
        try:
            config_data = generate_configurations(
                contribution_dir=str(contribution_dir),
                metric=args.metric,
                threshold_percentile=args.threshold_percentile,
                max_skip_layers=args.max_skip_layers
            )
            
            configurations = config_data['configurations']
            logger.info(f"✓ Generated {len(configurations)} configurations")
            
            # Save configurations with meaningful name
            with open(output_dir / "skip_configurations.json", 'w') as f:
                json.dump(make_json_serializable(config_data), f, indent=2)
            
            checkpoint.mark_stage_complete(stage_name, {"num_configurations": len(configurations)})
            
        except Exception as e:
            logger.error(f"Configuration generation failed: {e}")
            checkpoint.mark_stage_failed(stage_name, str(e))
            return
    
    # Step 2a: Run baseline benchmark
    stage_name = "baseline_benchmark"
    baseline_config = next((c for c in configurations if c['name'] == 'baseline'), None)
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 2/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
        baseline_dir = output_dir / "baseline"
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
        
        logger.info(f"\n[STEP 2/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name)
        
        if baseline_config:
            try:
                # Load baseline model
                model, tokenizer, _ = SkipLayerModelLoader.load_model_with_skip_layers(
                    model_name=args.model,
                    skip_layers=None,
                    device_map=args.device,
                    trust_remote_code=True
                )
                
                # Run benchmark
                tasks = [BENCHMARK_MAPPING.get(ds, ds) for ds in benchmark_datasets]
                baseline_results = run_lm_eval(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=tasks,
                    batch_size=args.batch_size,
                    device=args.device,
                    limit=args.num_samples,
                    verbosity="WARNING"
                )
                
                # Save baseline results with meaningful name
                baseline_dir = output_dir / "baseline"
                baseline_dir.mkdir(exist_ok=True)
                with open(baseline_dir / "benchmark_results.json", 'w') as f:
                    json.dump(make_json_serializable(baseline_results), f, indent=2)
                
                logger.info("✓ Baseline benchmark complete")
                checkpoint.mark_stage_complete(stage_name)
                
                # Cleanup
                del model, tokenizer
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Baseline benchmark failed: {e}")
                checkpoint.mark_stage_failed(stage_name, str(e))
                return
    
    # Step 2b: Run skip configuration benchmarks
    stage_name = "skip_benchmarks"
    test_configs = [c for c in configurations if c['name'] != 'baseline']
    
    # Initialize config tracking
    checkpoint.init_config_tracking([c['name'] for c in test_configs])
    
    # Determine which configs to run based on checkpoint and flags
    if args.retry_failed_only:
        configs_to_run = [
            c for c in test_configs 
            if checkpoint.get_config_status(c['name']) == 'failed'
        ]
        if configs_to_run:
            logger.info(f"\n[STEP 2/5] Retrying {len(configs_to_run)} previously failed configurations")
    else:
        configs_to_run = [
            c for c in test_configs
            if checkpoint.should_run_config(c['name'], skip_failed=args.skip_failed)
        ]
    
    # Check if stage should be skipped entirely
    completed_count = sum(
        1 for c in test_configs 
        if checkpoint.get_config_status(c['name']) == 'complete'
    )
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 2/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
        logger.info(f"  All {len(test_configs)} configurations complete")
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
            configs_to_run = test_configs  # Force rerun all
        
        logger.info(f"\n[STEP 2/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name, {"total_configs": len(test_configs)})
        
        if completed_count > 0 and not force_rerun_active and not args.retry_failed_only:
            logger.info(f"  Resuming: {completed_count}/{len(test_configs)} configs already complete")
            logger.info(f"  Remaining: {len(configs_to_run)} configs to test")
        
        for i, config in enumerate(configs_to_run, 1):
            logger.info(f"  [{i}/{len(configs_to_run)}] Testing {config['name']}...")
            
            checkpoint.mark_config_started(config['name'])
            
            try:
                # Load model with skip layers
                model, tokenizer, _ = SkipLayerModelLoader.load_model_with_skip_layers(
                    model_name=args.model,
                    skip_layers=config['skip_layers'],
                    device_map=args.device,
                    trust_remote_code=True
                )
                
                # Run benchmark
                tasks = [BENCHMARK_MAPPING.get(ds, ds) for ds in benchmark_datasets]
                results = run_lm_eval(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=tasks,
                    batch_size=args.batch_size,
                    device=args.device,
                    limit=args.num_samples,
                    verbosity="WARNING"
                )
                
                # Save results with meaningful names
                config_dir = output_dir / "skip_experiments" / config['name']
                config_dir.mkdir(parents=True, exist_ok=True)
                
                with open(config_dir / "skip_config.json", 'w') as f:
                    json.dump(make_json_serializable(config), f, indent=2)
                
                with open(config_dir / "benchmark_results.json", 'w') as f:
                    json.dump(make_json_serializable(results), f, indent=2)
                
                logger.info(f"    ✓ {config['name']} complete")
                checkpoint.mark_config_complete(config['name'])
                
                # Cleanup
                del model, tokenizer
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"    ✗ {config['name']} failed: {e}")
                
                # Determine if error is retryable
                retryable = "out of memory" in str(e).lower() or "timeout" in str(e).lower()
                checkpoint.mark_config_failed(config['name'], str(e), retryable=retryable)
                continue
        
        # Mark stage complete if at least some configs succeeded
        final_completed = sum(
            1 for c in test_configs 
            if checkpoint.get_config_status(c['name']) == 'complete'
        )
        
        if final_completed > 0:
            checkpoint.mark_stage_complete(stage_name, {
                "total_configs": len(test_configs),
                "completed_configs": final_completed
            })
        else:
            checkpoint.mark_stage_failed(stage_name, "No configurations completed successfully")
            return
    
    # Step 3: Generate comparison reports
    stage_name = "report_generation"
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 3/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
        
        logger.info(f"\n[STEP 3/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name)
        
        try:
            reports_generated = 0
            # Use BenchmarkReportGenerator with direct paths for each config vs baseline
            for config in test_configs:
                config_dir = output_dir / "skip_experiments" / config['name']
                if not (config_dir / "benchmark_results.json").exists():
                    continue
                
                try:
                    # Generate report using direct file paths
                    report_gen = BenchmarkReportGenerator(
                        baseline_results_path=baseline_dir / "benchmark_results.json",
                        target_results_path=config_dir / "benchmark_results.json",
                        baseline_name="baseline",
                        target_name=config['name'],
                        output_dir=config_dir
                    )
                    report_gen.generate_report()
                    reports_generated += 1
                    
                except Exception as e:
                    logger.warning(f"  Could not generate report for {config['name']}: {e}")
            
            checkpoint.mark_stage_complete(stage_name, {"reports_generated": reports_generated})
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            checkpoint.mark_stage_failed(stage_name, str(e))
            # Don't return - continue to next stage
    
    # Step 4: Analyze optimal configurations
    stage_name = "optimization_analysis"
    
    if checkpoint.should_skip_stage(stage_name, force_rerun_active or (args.resume_from == stage_name)):
        logger.info(f"\n[STEP 4/5] Skipping '{checkpoint.STAGES[stage_name]['name']}' (already complete)")
    else:
        if args.resume_from == stage_name:
            force_rerun_active = True
        
        logger.info(f"\n[STEP 4/5] {checkpoint.STAGES[stage_name]['name']}...")
        checkpoint.mark_stage_started(stage_name)
        
        try:
            analyzer = ConfigPerformanceAnalyzer(
                analysis_dir=output_dir,
                accuracy_threshold=args.accuracy_threshold
            )
            
            analyzer.load_data()
            analyses = analyzer.analyze_all_configurations()
            optimal = analyzer.find_optimal_configurations(analyses)
            report = analyzer.generate_report(analyses, optimal)
            analyzer.save_report(report, output_dir / "optimization_report.json")
            
            # Save CSV results
            analyzer.save_results_csv(analyses, output_dir / "optimization_report.csv")
            
            # Print summary
            analyzer.print_summary(report)
            
            logger.info("✓ Optimization analysis complete")
            checkpoint.mark_stage_complete(stage_name)
            
        except Exception as e:
            logger.warning(f"Optimization analysis failed: {e}")
            checkpoint.mark_stage_failed(stage_name, str(e))
            # Don't return - continue to summary
    
    # Step 5: Create summary
    stage_name = "summary"
    logger.info(f"\n[STEP 5/5] {checkpoint.STAGES[stage_name]['name']}...")
    checkpoint.mark_stage_started(stage_name)
    
    try:
        summary = {
            "model": args.model,
            "clean_model_name": clean_model_name,
            "layer_analysis_datasets": layer_datasets,
            "benchmark_datasets": benchmark_datasets,
            "num_configurations": len(configurations),
            "num_tested": len([c for c in test_configs if (output_dir / "skip_experiments" / c['name'] / "benchmark_results.json").exists()]),
            "execution_time_seconds": time.time() - start_time,
            "output_directory": str(output_dir),
            "checkpoint_info": checkpoint.get_summary(),
            "resumed_from_checkpoint": is_resuming
        }
        
        with open(output_dir / "pipeline_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        checkpoint.mark_stage_complete(stage_name)
        
    except Exception as e:
        logger.error(f"Summary creation failed: {e}")
        checkpoint.mark_stage_failed(stage_name, str(e))
    
    # Done
    elapsed = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"Execution time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Configurations tested: {summary['num_tested']}/{len(test_configs)}")
    if summary.get('resumed_from_checkpoint'):
        logger.info("Pipeline resumed from checkpoint")
    logger.info("="*80)


if __name__ == "__main__":
    main()
