#!/usr/bin/env python3
"""
Weights & Biases Experiment Runner for QAIC Training Optimization Matrix

Runs 6 experiment configurations with different:
- num_workers (1, 8)
- prefetch_factor (1, 4)
- gradient_accumulation (1, 4)
- batch_size (1, 2)
- bucket_size (25MB, 50MB)

Each configuration is tracked in W&B for comparison.
"""

import os
import sys
import yaml
import wandb
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Experiment matrix definition
EXPERIMENT_MATRIX = [
    {
        "id": 19,
        "name": "google/gemma-2b: num_w=8, prefetch=4, grad_acc=4, batch=4, bucket=50MB",
        "config_overrides": {
            "model_name" : "google/gemma-2b",
            "dataloader_num_workers": 8,
            "dataloader_prefetch_factor": 4,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size": 2,
            "ddp_bucket_cap_mb": 50,
        },
    },
    {
        "id": 20,
        "name": "Qwen/Qwen2-1.5B-Instruct: num_w=8, prefetch=4, grad_acc=4, batch=4, bucket=50MB",
        "config_overrides": {
            "model_name": "Qwen/Qwen2-1.5B-Instruct",
            "dataloader_num_workers": 8,
            "dataloader_prefetch_factor": 4,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size": 2,
            "ddp_bucket_cap_mb": 50,
        },
    }
]


class WANDBExperimentRunner:
    """Orchestrates training runs with W&B tracking."""

    def __init__(
        self,
        base_config_path: str,
        base_output_dir: str = "./wandb_experiments",
        project_name: str = "qaic-training-optimization",
        entity_name: Optional[str] = None,
        num_epochs: int = 1,
        run_subset: Optional[list] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            base_config_path: Path to base YAML config file
            base_output_dir: Directory to save outputs from all runs
            project_name: W&B project name
            entity_name: W&B entity/team name (optional)
            num_epochs: Number of epochs to train
            run_subset: List of experiment IDs to run (None = run all)
        """
        self.base_config_path = Path(base_config_path)
        self.base_output_dir = Path(base_output_dir)
        self.project_name = project_name
        self.entity_name = entity_name
        self.num_epochs = num_epochs
        self.run_subset = run_subset or [e["id"] for e in EXPERIMENT_MATRIX]
 
        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
 
        # Load base config
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        self.results = []

    def _create_experiment_config(
        self, experiment: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Create a modified config for a specific experiment.

        Args:
            experiment: Experiment definition dict

        Returns:
            Tuple of (modified config dict, experiment output dir)
        """
        config = deepcopy(self.base_config)
        
        # Update training parameters
        for key, value in experiment["config_overrides"].items():
            if key in ["per_device_train_batch_size", "gradient_accumulation_steps"]:
                config["training"][key] = value
            elif key in ["ddp_bucket_cap_mb"]:
                config["training"]["ddp_config"][key] = value
            elif key in ["dataloader_num_workers", "dataloader_prefetch_factor"]:
                config["dataset"][key] = value
            elif key in ["model_name"]:
                config["model"][key] = value
        
        # Set number of epochs
        config["training"]["num_train_epochs"] = self.num_epochs
        
        # Create unique output dir for this experiment
        exp_output_dir = (
            self.base_output_dir
            / f"exp_{experiment['id']:02d}_{experiment['name'].replace(' ', '_')[:30]}"
        )
        config["training"]["output_dir"] = str(exp_output_dir)
        
        return config, exp_output_dir

    def _save_config(self, config: Dict[str, Any], path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with W&B tracking via subprocess.

        Args:
            experiment: Experiment definition dict

        Returns:
            Dict of results
        """
        exp_id = experiment["id"]
        exp_name = experiment["name"]
        
        print(f"\n{'='*80}")
        print(f"Running Experiment {exp_id}: {exp_name}")
        print(f"{'='*80}\n")
        
        # Initialize W&B run
        run_name = f"exp_{exp_id:02d}_{exp_name.split(':')[0]}"
        with wandb.init(
            project=self.project_name,
            entity=self.entity_name,
            name=run_name,
            config={
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "model": "meta-llama/Llama-3.2-1B",
                "dataset": "openai/gsm8k",
                "num_devices": 7,  # QAIC devices
                **experiment["config_overrides"],
            },
            tags=["training-optimization", "qaic", f"exp_{exp_id}"],
        ) as run:
            start_time = time.time()
            result = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "run_id": run.id,
                "config_overrides": experiment["config_overrides"],
                "status": "failed",
                "error": None,
                "duration_seconds": 0,
                "metrics": {},
            }
            
            try:
                # Create modified config
                config, exp_output_dir = self._create_experiment_config(experiment)
                config_path = exp_output_dir / "config_with_overrides.yaml"
                self._save_config(config, config_path)
                
                # Log config to W&B
                wandb.config.update(experiment["config_overrides"], allow_val_change=True)
                
                print(f"Loading config from: {config_path}")
                
                # Call training via subprocess with torchrun for distributed training
                finetune_script = "QEfficient.cloud.finetune_experimental"
                cmd = [
                    "torchrun",
                    "--nproc-per-node", "7", "-m",
                    str(finetune_script),
                    str(config_path),
                ]
                
                # Setup environment with QAIC visible devices
                training_env = {**os.environ}
                # Use QAIC_VISIBLE_DEVICES from environment, or default to 16,17,18,19
                qaic_devices = os.environ.get("QAIC_VISIBLE_DEVICES", "38,39,40,41,42")
                training_env["QAIC_VISIBLE_DEVICES"] = qaic_devices
                training_env["WANDB_RUN_ID"] = run.id
                
                print(f"Running training command: QAIC_VISIBLE_DEVICES={qaic_devices} {' '.join(cmd)}")
                result_process = subprocess.run(
                    cmd,
                    capture_output=False,
                    text=True,
                    env=training_env,
                )
                
                if result_process.returncode == 0:
                    result["status"] = "completed"
                    print(f"✓ Experiment {exp_id} completed successfully")
                    # Log basic completion metrics
                    wandb.log({"experiment_status": "completed"})
                else:
                    result["status"] = "failed"
                    result["error"] = f"Process exited with code {result_process.returncode}"
                    print(f"✗ Experiment {exp_id} failed with exit code {result_process.returncode}")
                    wandb.log({"error": result["error"]})
                
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                print(f"✗ Experiment {exp_id} failed: {e}")
                
                # Log error to W&B
                wandb.log({"error": str(e)})
                import traceback
                traceback.print_exc()
            
            finally:
                result["duration_seconds"] = time.time() - start_time
                wandb.log({"experiment_duration": result["duration_seconds"]})
                run.finish()
        
        return result

    def run_all(self) -> None:
        """Run all experiments in the matrix."""
        print(f"\nStarting W&B Experiment Matrix")
        print(f"Project: {self.project_name}")
        print(f"Total experiments: {len(EXPERIMENT_MATRIX)}")
        print(f"Running subset: {self.run_subset}\n")
        
        for experiment in EXPERIMENT_MATRIX:
            if experiment["id"] not in self.run_subset:
                print(f"Skipping experiment {experiment['id']} (not in subset)")
                continue
            
            result = self._run_experiment(experiment)
            self.results.append(result)
        
        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary of all runs."""
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}\n")
        
        completed = [r for r in self.results if r["status"] == "completed"]
        failed = [r for r in self.results if r["status"] == "failed"]
        
        print(f"Total runs: {len(self.results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}\n")
        
        if completed:
            print("COMPLETED RUNS:")
            for result in completed:
                print(f"\n  Exp {result['experiment_id']}: {result['experiment_name']}")
                print(f"  W&B Run ID: {result['run_id']}")
                print(f"  Duration: {result['duration_seconds']:.1f}s")
                print(f"  Metrics:")
                for metric_key, metric_val in result["metrics"].items():
                    print(f"    - {metric_key}: {metric_val}")
        
        if failed:
            print("\nFAILED RUNS:")
            for result in failed:
                print(f"\n  Exp {result['experiment_id']}: {result['experiment_name']}")
                print(f"  Error: {result['error']}")
        
        # Save results to JSON
        results_file = self.base_output_dir / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B experiment matrix for QAIC training optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./wandb_experiments",
        help="Base output directory for experiment results",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="qaic-training-optimization",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs for each run",
    )
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific experiment IDs to run (e.g., --exp-ids 1 2 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment configs without running",
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = WANDBExperimentRunner(
        base_config_path=args.config,
        base_output_dir=args.output_dir,
        project_name=args.project,
        entity_name=args.entity,
        num_epochs=args.epochs,
        run_subset=args.exp_ids,
    )
    
    if args.dry_run:
        print("\nDRY RUN: Showing experiment configurations\n")
        for exp in EXPERIMENT_MATRIX:
            if args.exp_ids is None or exp["id"] in args.exp_ids:
                print(f"\nExperiment {exp['id']}: {exp['name']}")
                print(f"  Overrides: {json.dumps(exp['config_overrides'], indent=4)}")
        return
    
    # Run experiments
    runner.run_all()


if __name__ == "__main__":
    main()
