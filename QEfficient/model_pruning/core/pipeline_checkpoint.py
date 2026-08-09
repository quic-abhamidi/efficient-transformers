#!/usr/bin/env python3
"""
Pipeline Checkpoint Manager

Provides fault-tolerant checkpoint management for the layer skipping pipeline.
Supports stage-level and config-level tracking with validation and atomic operations.

Key Features:
- Granular resume capability (stage and config level)
- Output validation to ensure data integrity
- Atomic file operations to prevent corruption
- Flexible retry logic for failed configurations
- Comprehensive error tracking and reporting
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


class PipelineCheckpoint:
    """
    Manages pipeline execution state with granular checkpoint tracking.
    
    Provides:
    - Stage-level completion tracking
    - Config-level tracking for benchmark stage
    - Output validation
    - Atomic file operations
    - Resume logic with multiple strategies
    """
    
    VERSION = "2.0"
    CHECKPOINT_FILE = "pipeline_checkpoint.json"
    
    # Stage definitions with expected outputs
    STAGES = {
        "layer_contributions": {
            "name": "Layer Contribution Analysis",
            "order": 0,
            "outputs": ["layer_contributions/"]
        },
        "config_generation": {
            "name": "Configuration Generation",
            "order": 1,
            "outputs": ["skip_configurations.json"]
        },
        "baseline_benchmark": {
            "name": "Baseline Benchmark",
            "order": 2,
            "outputs": ["baseline/benchmark_results.json"]
        },
        "skip_benchmarks": {
            "name": "Skip Configuration Benchmarks",
            "order": 3,
            "outputs": ["skip_experiments/"],
            "granular": True  # Has per-config tracking
        },
        "report_generation": {
            "name": "Comparison Reports",
            "order": 4,
            "outputs": ["skip_experiments/"]
        },
        "optimization_analysis": {
            "name": "Optimization Analysis",
            "order": 5,
            "outputs": ["optimization_report.json"]
        },
        "summary": {
            "name": "Pipeline Summary",
            "order": 6,
            "outputs": ["pipeline_summary.json"]
        }
    }
    
    def __init__(self, output_dir: Path, model: str):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Pipeline output directory
            model: Model identifier
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self.model = model
        self.data = self._load_or_create()
    
    def _load_or_create(self) -> Dict:
        """Load existing checkpoint or create new one."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                
                # Validate version compatibility
                if data.get("version") != self.VERSION:
                    logger.warning(
                        f"Checkpoint version mismatch (found {data.get('version')}, "
                        f"expected {self.VERSION}). Creating new checkpoint."
                    )
                    return self._create_new_checkpoint()
                
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
                return data
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Creating new one.")
                return self._create_new_checkpoint()
        
        return self._create_new_checkpoint()
    
    def _create_new_checkpoint(self) -> Dict:
        """Create new checkpoint structure."""
        return {
            "version": self.VERSION,
            "model": self.model,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stages": {}
        }
    
    def save(self):
        """Save checkpoint atomically to prevent corruption."""
        self.data["last_updated"] = datetime.now().isoformat()
        
        # Atomic write: write to temp file, then rename
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename (overwrites existing file)
            temp_path.replace(self.checkpoint_path)
            logger.debug(f"Checkpoint saved to {self.checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    # ========================================================================
    # STAGE MANAGEMENT
    # ========================================================================
    
    def mark_stage_started(self, stage_name: str, metadata: Optional[Dict] = None):
        """Mark a stage as started."""
        if stage_name not in self.STAGES:
            logger.warning(f"Unknown stage: {stage_name}")
            return
        
        stage_data = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "display_name": self.STAGES[stage_name]["name"]
        }
        
        if metadata:
            stage_data["metadata"] = metadata
        
        self.data["stages"][stage_name] = stage_data
        self.save()
    
    def mark_stage_complete(self, stage_name: str, metadata: Optional[Dict] = None):
        """Mark a stage as complete."""
        if stage_name not in self.STAGES:
            logger.warning(f"Unknown stage: {stage_name}")
            return
        
        stage_data = self.data["stages"].get(stage_name, {})
        stage_data.update({
            "status": "complete",
            "completed_at": datetime.now().isoformat(),
            "display_name": self.STAGES[stage_name]["name"]
        })
        
        if metadata:
            stage_data["metadata"] = metadata
        
        self.data["stages"][stage_name] = stage_data
        self.save()
        
        logger.info(f"✓ Stage '{self.STAGES[stage_name]['name']}' complete")
    
    def mark_stage_failed(self, stage_name: str, error: str, metadata: Optional[Dict] = None):
        """Mark a stage as failed."""
        if stage_name not in self.STAGES:
            logger.warning(f"Unknown stage: {stage_name}")
            return
        
        stage_data = self.data["stages"].get(stage_name, {})
        stage_data.update({
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": str(error),
            "display_name": self.STAGES[stage_name]["name"]
        })
        
        if metadata:
            stage_data["metadata"] = metadata
        
        self.data["stages"][stage_name] = stage_data
        self.save()
        
        logger.error(f"✗ Stage '{self.STAGES[stage_name]['name']}' failed: {error}")
    
    def is_stage_complete(self, stage_name: str, validate: bool = True) -> bool:
        """
        Check if a stage is complete.
        
        Args:
            stage_name: Name of the stage
            validate: If True, also validate outputs exist
        
        Returns:
            True if stage is complete and valid
        """
        stage_data = self.data["stages"].get(stage_name, {})
        
        if stage_data.get("status") != "complete":
            return False
        
        if validate:
            return self.validate_stage_outputs(stage_name)
        
        return True
    
    def validate_stage_outputs(self, stage_name: str) -> bool:
        """
        Validate that expected outputs exist for a stage.
        
        Args:
            stage_name: Name of the stage
        
        Returns:
            True if all expected outputs are valid
        """
        if stage_name not in self.STAGES:
            return False
        
        expected_outputs = self.STAGES[stage_name]["outputs"]
        
        for output in expected_outputs:
            output_path = self.output_dir / output
            
            if not output_path.exists():
                logger.debug(f"Missing output for {stage_name}: {output_path}")
                return False
            
            # Additional validation for specific file types
            if output.endswith('.json'):
                if not self._validate_json_file(output_path):
                    return False
            elif output.endswith('/'):  # Directory
                if not output_path.is_dir():
                    return False
        
        return True
    
    def _validate_json_file(self, filepath: Path) -> bool:
        """Validate JSON file integrity."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check it's not empty
            if not data:
                logger.debug(f"Empty JSON file: {filepath}")
                return False
            
            # Specific validation for benchmark results
            if filepath.name == "benchmark_results.json":
                if "results" not in data:
                    logger.debug(f"Invalid benchmark results structure: {filepath}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Invalid JSON file {filepath}: {e}")
            return False
    
    # ========================================================================
    # CONFIG-LEVEL MANAGEMENT (for skip_benchmarks stage)
    # ========================================================================
    
    def init_config_tracking(self, config_names: List[str]):
        """Initialize config-level tracking for skip_benchmarks stage."""
        stage_name = "skip_benchmarks"
        
        if stage_name not in self.data["stages"]:
            self.data["stages"][stage_name] = {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "display_name": self.STAGES[stage_name]["name"],
                "total_configs": len(config_names),
                "configs": {}
            }
        else:
            # Ensure configs key exists even if stage was loaded from checkpoint
            if "configs" not in self.data["stages"][stage_name]:
                self.data["stages"][stage_name]["configs"] = {}
            
            # Update total_configs if needed
            if "total_configs" not in self.data["stages"][stage_name]:
                self.data["stages"][stage_name]["total_configs"] = len(config_names)
        
        # Initialize each config if not already present
        for config_name in config_names:
            if config_name not in self.data["stages"][stage_name]["configs"]:
                self.data["stages"][stage_name]["configs"][config_name] = {
                    "status": "pending"
                }
        
        self.save()
    
    def mark_config_started(self, config_name: str):
        """Mark a config as started."""
        stage_name = "skip_benchmarks"
        
        if stage_name not in self.data["stages"]:
            self.init_config_tracking([config_name])
        
        # Ensure configs key exists (defensive programming)
        if "configs" not in self.data["stages"][stage_name]:
            self.data["stages"][stage_name]["configs"] = {}
        
        self.data["stages"][stage_name]["configs"][config_name] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat()
        }
        self.save()
    
    def mark_config_complete(self, config_name: str):
        """Mark a config as complete."""
        stage_name = "skip_benchmarks"
        
        # Ensure configs key exists (defensive programming)
        if "configs" not in self.data["stages"][stage_name]:
            self.data["stages"][stage_name]["configs"] = {}
        
        self.data["stages"][stage_name]["configs"][config_name] = {
            "status": "complete",
            "completed_at": datetime.now().isoformat()
        }
        
        # Update stage-level counts
        self._update_config_counts()
        self.save()
        
        logger.info(f"  ✓ Config '{config_name}' complete")
    
    def mark_config_failed(self, config_name: str, error: str, retryable: bool = True):
        """Mark a config as failed."""
        stage_name = "skip_benchmarks"
        
        # Ensure configs key exists (defensive programming)
        if "configs" not in self.data["stages"][stage_name]:
            self.data["stages"][stage_name]["configs"] = {}
        
        self.data["stages"][stage_name]["configs"][config_name] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": str(error),
            "retryable": retryable
        }
        
        # Update stage-level counts
        self._update_config_counts()
        self.save()
        
        logger.warning(f"  ✗ Config '{config_name}' failed: {error}")
    
    def _update_config_counts(self):
        """Update config completion counts in skip_benchmarks stage."""
        stage_name = "skip_benchmarks"
        
        if stage_name not in self.data["stages"]:
            return
        
        configs = self.data["stages"][stage_name].get("configs", {})
        
        completed = sum(1 for c in configs.values() if c.get("status") == "complete")
        failed = sum(1 for c in configs.values() if c.get("status") == "failed")
        
        self.data["stages"][stage_name]["completed_configs"] = completed
        self.data["stages"][stage_name]["failed_configs"] = failed
    
    def get_config_status(self, config_name: str) -> str:
        """Get status of a specific config."""
        stage_name = "skip_benchmarks"
        
        if stage_name not in self.data["stages"]:
            return "pending"
        
        configs = self.data["stages"][stage_name].get("configs", {})
        return configs.get(config_name, {}).get("status", "pending")
    
    def should_run_config(
        self, 
        config_name: str, 
        skip_failed: bool = False,
        retry_failed: bool = False
    ) -> bool:
        """
        Determine if a config should be run.
        
        Args:
            config_name: Name of the configuration
            skip_failed: If True, skip previously failed configs
            retry_failed: If True, retry failed configs
        
        Returns:
            True if config should be run
        """
        status = self.get_config_status(config_name)
        
        if status == "complete":
            # Validate outputs exist
            config_dir = self.output_dir / "skip_experiments" / config_name
            results_file = config_dir / "benchmark_results.json"
            
            if results_file.exists() and self._validate_json_file(results_file):
                return False  # Skip - already complete and valid
            else:
                logger.warning(
                    f"Config '{config_name}' marked complete but outputs invalid - will retry"
                )
                return True
        
        elif status == "failed":
            if skip_failed:
                logger.info(f"Skipping previously failed config: {config_name}")
                return False
            elif retry_failed:
                logger.info(f"Retrying previously failed config: {config_name}")
                return True
            else:
                # Default: retry if retryable
                stage_name = "skip_benchmarks"
                config_data = self.data["stages"][stage_name]["configs"].get(config_name, {})
                if config_data.get("retryable", True):
                    logger.info(f"Retrying retryable failed config: {config_name}")
                    return True
                else:
                    logger.info(f"Skipping non-retryable failed config: {config_name}")
                    return False
        
        else:  # pending or in_progress
            return True
    
    def get_pending_configs(self, all_configs: List[str]) -> List[str]:
        """Get list of configs that still need to be run."""
        return [
            config for config in all_configs 
            if self.should_run_config(config)
        ]
    
    # ========================================================================
    # RESUME LOGIC
    # ========================================================================
    
    def get_resume_stage(self) -> Optional[str]:
        """
        Determine which stage to resume from.
        
        Returns:
            Stage name to resume from, or None if starting fresh
        """
        # Find first incomplete stage
        sorted_stages = sorted(
            self.STAGES.items(),
            key=lambda x: x[1]["order"]
        )
        
        for stage_name, stage_info in sorted_stages:
            if not self.is_stage_complete(stage_name, validate=True):
                return stage_name
        
        return None  # All stages complete
    
    def should_skip_stage(self, stage_name: str, force_rerun: bool = False) -> bool:
        """
        Determine if a stage should be skipped.
        
        Args:
            stage_name: Name of the stage
            force_rerun: If True, never skip
        
        Returns:
            True if stage should be skipped
        """
        if force_rerun:
            return False
        
        return self.is_stage_complete(stage_name, validate=True)
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_status(self):
        """Print current checkpoint status."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT STATUS")
        logger.info("="*80)
        logger.info(f"Model: {self.model}")
        logger.info(f"Created: {self.data.get('created_at', 'unknown')}")
        logger.info(f"Last Updated: {self.data.get('last_updated', 'unknown')}")
        logger.info("-"*80)
        
        sorted_stages = sorted(
            self.STAGES.items(),
            key=lambda x: x[1]["order"]
        )
        
        for stage_name, stage_info in sorted_stages:
            stage_data = self.data["stages"].get(stage_name, {})
            status = stage_data.get("status", "pending")
            
            status_symbol = {
                "complete": "✓",
                "in_progress": "⟳",
                "failed": "✗",
                "pending": "○"
            }.get(status, "?")
            
            logger.info(f"{status_symbol} {stage_info['name']}: {status.upper()}")
            
            # Show config-level details for skip_benchmarks
            if stage_name == "skip_benchmarks" and "configs" in stage_data:
                completed = stage_data.get("completed_configs", 0)
                failed = stage_data.get("failed_configs", 0)
                total = stage_data.get("total_configs", 0)
                pending = total - completed - failed
                
                logger.info(f"    Completed: {completed}/{total}")
                logger.info(f"    Failed: {failed}/{total}")
                logger.info(f"    Pending: {pending}/{total}")
            
            elif status == "complete":
                completed_at = stage_data.get("completed_at", "unknown")
                logger.info(f"    Completed: {completed_at}")
            
            elif status == "failed":
                error = stage_data.get("error", "unknown")
                logger.info(f"    Error: {error}")
        
        logger.info("="*80 + "\n")
    
    def get_summary(self) -> Dict:
        """Get checkpoint summary for pipeline_summary.json."""
        return {
            "checkpoint_version": self.VERSION,
            "checkpoint_exists": True,
            "stages_complete": sum(
                1 for stage_data in self.data["stages"].values()
                if stage_data.get("status") == "complete"
            ),
            "total_stages": len(self.STAGES),
            "created_at": self.data.get("created_at"),
            "last_updated": self.data.get("last_updated")
        }
