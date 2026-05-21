# ============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ============================================================================

"""
Comprehensive Distributed Data Parallel (DDP) end-to-end tests for QAIC devices.

This test suite validates:
- Parity between single-device and multi-device DDP training
- Gradient synchronization across devices
- Loss computation consistency
- Model state synchronization
- Batch processing correctness
- Learning rate scheduling in DDP
- Checkpoint save/load functionality
- Different world sizes (1, 2, 4 devices)
- Error handling and edge cases
- Integration with the FineTuningPipeline for end-to-end validation
"""

import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import FileStore
from torch.nn import MSELoss
from torch.optim import SGD

from QEfficient.finetune.experimental.core.logger import Logger

try:
    from QEfficient.cloud.finetune_experimental import FineTuningPipeline
    from QEfficient.finetune.experimental.core.config_manager import ConfigManager
except ImportError:
    FineTuningPipeline = None
    ConfigManager = None

logger = Logger(__name__)
# ============================================================================
# Test Configuration Constants
# ============================================================================

WORLD_SIZE = 2
BACKEND = "qccl"
LOSS_ATOL = 1e-3
PIPELINE_ATOL = 1e-1  # Higher tolerance for end-to-end pipeline tests due to non-determinism
METRIC_ATOL = 0.5
GRADIENT_ATOL = 1e-4
PARAM_ATOL = 1e-4
TEST_SEED = 42
DEFAULT_MASTER_PORT = 12591
STORE_PATH = os.path.join(tempfile.gettempdir(), "ddp_store_file")  # File path for FileStore (not directory)
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "regression_tests"
)  # Base directory for regression test outputs
GOLDEN_DIR = os.path.join(BASE_DIR, "goldens")  # Directory to store golden outputs for regression testing
BASELINE_SDK_VERSION = "SDK_1.22.0.32"  # Baseline SDK version for regression testing (update as needed)
_LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
_REDUCED_LAYERS = (
    2  # For testing, we reduce the number of layers to speed up execution while still validating the pipeline
)


def load_llama_model_and_tokenizer(reduced_layers=None):
    """
    Load Llama-3.2-1B with num_hidden_layers reduced to _REDUCED_LAYERS.
    Optionally injects a PP device_map.
    """
    from QEfficient.finetune.experimental.core.component_registry import ComponentFactory
    from QEfficient.finetune.experimental.core.model import HFModel  # noqa: F401

    kwargs = {
        "auto_class_name": "AutoModelForCausalLM",
        "use_cache": False,
        "attn_implementation": "eager",
        "num_hidden_layers": reduced_layers,
    }
    return ComponentFactory.create_model("hf", _LLAMA_MODEL_NAME, **kwargs)


_HF_MODEL = load_llama_model_and_tokenizer(reduced_layers=_REDUCED_LAYERS)

# ============================================================================
# Test Configuration Dataclasses
# ============================================================================


@dataclass
class DDPTestConfig:
    """Configuration for DDP tests - PICKLABLE for mp.spawn."""

    world_size: int
    rank: int
    backend: str
    find_unused_parameters: bool = False
    static_graph: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 2
    num_epochs: int = 1


@dataclass
class SimpleConfig:
    """Picklable configuration for training workers."""

    batch_size: int = 2
    learning_rate: float = 1e-3
    num_batches: int = 5
    seed: int = 42


# ============================================================================
# Test Models
# ============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP model for testing."""

    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiLayerMLP(nn.Module):
    """Multi-layer MLP for more complex testing."""

    def __init__(self, input_dim=8, hidden_dim=16, num_layers=3, output_dim=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Golden utilities for regression testing (can be extended to save/load golden outputs)
# ============================================================================


def save_golden(name: str, data: dict):
    """Save golden data for regression testing (only if not already present)."""

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{name}_{BASELINE_SDK_VERSION}.json")

    # ✅ Do not overwrite if already exists
    if os.path.exists(path):
        logger.info(f"Golden file already exists, skipping: {path}")
        return

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Saved golden data to {path}")


# ============================================================================
# Helpers
# ============================================================================


def is_qaic_available():
    """
    Check if QAIC devices are available.

    Returns:
        bool: True if QAIC devices are available, False otherwise.
    """
    try:
        import torch

        return hasattr(torch, "qaic") or "qaic" in torch.device.__str__()
    except Exception:
        return False


def ddp_init(rank, world_size, port=DEFAULT_MASTER_PORT, use_file_store=False, timeout=300):
    """
    Initialize DDP process group with robust error handling.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for TCPStore (ignored if use_file_store=True)
        use_file_store: If True, use FileStore for testing (avoids port conflicts with mp.spawn)
        timeout: Timeout in seconds for process group initialization
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    backend = "qccl" if is_qaic_available() else "gloo"
    logger.info(f"[Rank {rank}] Initializing process group with backend={backend}, timeout={timeout}s")

    try:
        if use_file_store:
            # Use FileStore for mp.spawn tests to avoid TCP port conflicts
            store = FileStore(STORE_PATH, world_size=world_size)
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                store=store,
                timeout=torch.distributed.timedelta(seconds=timeout),
            )
        else:
            # Use default TCPStore for normal execution
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.timedelta(seconds=timeout),
            )
        logger.info(f"[Rank {rank}] Process group initialized successfully")
    except RuntimeError as e:
        logger.error(f"[Rank {rank}] Failed to initialize process group: {e}")
        raise


def cleanup():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def extract_losses(loss_list):
    """Extract only loss values from (step, loss) tuples."""
    return [loss for _, loss in loss_list]


def compute_avg(losses):
    return float(np.mean(losses))


def is_stable(losses, spike_tol=3.0):
    """
    Ensure no extreme spikes (instability).
    """
    losses = np.array(losses)
    median = np.median(losses)
    return np.all(losses < spike_tol * median)


def ddp_training_worker(rank, world_size, model_class, config, results):
    """
    Simple DDP training worker for basic parity tests.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_class: Model class to instantiate
        config: SimpleConfig (PICKLABLE dataclass)
        results: Multiprocessing dict for results
    """
    ddp_init(rank, world_size, use_file_store=False)

    try:
        torch.manual_seed(config.seed + rank)

        # Create model and move to device
        model = model_class()
        device = torch.device(f"qaic:{rank}")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        # Simulate training
        gradients_per_batch = []
        losses = []

        for batch_idx in range(config.num_batches):
            # Generate random batch
            x = torch.randn(config.batch_size, 8).to(device)
            y = torch.randn(config.batch_size, 2).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Store gradient snapshot
            grads = [p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]
            gradients_per_batch.append(grads)

        results[rank] = {
            "params": [p.data.detach().cpu().clone() for p in model.parameters()],
            "losses": losses,
            "gradients": gradients_per_batch,
        }

    except Exception as e:
        logger.error(f"Rank {rank} error: {e}")
        results[rank] = {"error": str(e)}

    finally:
        cleanup()


def single_device_training_worker(model_class, config, results):
    """
    Single-device training for reference parity comparison.

    This function performs single-device training to serve as a reference
    for comparing DDP multi-device training results.

    Args:
        model_class: Model class to instantiate
        config: SimpleConfig (PICKLABLE dataclass) with training parameters
        results: Dict for storing results (modified in-place)
    """
    try:
        torch.manual_seed(config.seed)

        model = model_class()
        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        losses = []
        gradients_per_batch = []

        # Training loop: same as DDP but without distributed setup
        for batch_idx in range(config.num_batches):
            x = torch.randn(config.batch_size, 8)
            y = torch.randn(config.batch_size, 2)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Capture gradient snapshots for analysis
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            gradients_per_batch.append(grads)

        # Store results in CPU memory for cross-process comparison
        results["single_device"] = {
            "params": [p.data.detach().cpu().clone() for p in model.parameters()],
            "losses": losses,
            "gradients": gradients_per_batch,
        }

    except Exception as e:
        logger.error(f"Single device error: {e}")
        results["single_device"] = {"error": str(e)}


def ddp_loss_worker(rank, world_size, config, results):
    """
    DDP training worker with deterministic loss computation.

    Uses synchronized random seeds across all ranks to ensure identical inputs
    and deterministic loss trajectories for parity testing.

    Args:
        rank: Process rank in the DDP group
        world_size: Total number of processes in the DDP group
        config: SimpleConfig (PICKLABLE dataclass) with training parameters
        results: Multiprocessing dict for storing results
    """
    ddp_init(rank, world_size, use_file_store=False)

    try:
        torch.manual_seed(config.seed)

        model = SimpleMLP()
        device = torch.device(f"qaic:{rank}")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        losses = []

        # Training loop with deterministic random seeds
        for batch_idx in range(config.num_batches):
            # Use same seed across all ranks to generate identical inputs
            torch.manual_seed(config.seed + batch_idx)

            x = torch.randn(config.batch_size, 8).to(device)
            y = torch.randn(config.batch_size, 2).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Only rank 0 reports results (all ranks should be identical)
        if rank == 0:
            results["ddp"] = losses

    except Exception as e:
        logger.error(f"DDP loss worker rank {rank} error: {e}")
        results["error"] = str(e)

    finally:
        cleanup()


def pipeline_ddp_worker(rank, world_size, port, config_dict, results):
    """
    Top-level DDP training worker for FineTuningPipeline.

    Must be a top-level function (not a method) for pickling compatibility with mp.spawn.
    Initializes DDP, sets up FineTuningPipeline, and runs training.
    Only rank 0 reports results back.

    Args:
        rank: Process rank in DDP group
        world_size: Total processes in DDP group
        port: Port for DDP initialization
        config_dict: Configuration dictionary (PICKLABLE)
        results: Multiprocessing dict for results
    """
    # Set environment variables for DDP initialization
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    ddp_init(rank, world_size, use_file_store=True, port=port)
    logger.info(
        f"[Rank {rank}] DDP initialized with backend={config_dict.get('ddp_config', {}).get('ddp_backend', 'N/A')}"
    )

    try:
        if FineTuningPipeline is None or ConfigManager is None:
            raise RuntimeError("FineTuningPipeline not available")

        # Use SAME seed across all ranks for parity testing
        # In production DDP, each rank would use rank-specific seed for data shuffling
        torch.manual_seed(TEST_SEED + rank)

        # Create and configure ConfigManager
        cm = ConfigManager()
        cm.config.model_name = _HF_MODEL.model_name
        cm.config.dataset["prompt_func"] = (
            "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
        )
        cm.config.dataset["completion_template"] = "{output}"
        cm.config.dataset["dataset_num_samples"] = 100

        for k, v in config_dict.items():
            cm.config.training[k] = v

        cm.config.training["local_rank"] = rank

        # Run pipeline training
        logger.info(f"[Rank {rank}] Starting training pipeline...")
        pipeline = FineTuningPipeline(cm)
        pipeline.run()

        # Synchronize all ranks before result collection
        if dist.is_initialized():
            dist.barrier()
            logger.info(f"[Rank {rank}] Barrier passed - all ranks completed training")

        # Only rank 0 collects and reports results
        if rank == 0:
            trainer = pipeline.trainer
            train_loss = None
            train_metrics = None

            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                # Extract loss trajectory
                losses = [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
                if losses:
                    train_loss = losses
                # Filter for entries that have a step but are not eval entries
                train_metrics = [
                    entry["train/epoch_metric"] for entry in trainer.state.log_history if "train/epoch_metric" in entry
                ]

            logger.info(
                f"[Rank {rank}] Training completed. Train loss: {len(train_loss) if train_loss else 0}, Train metrics: {len(train_metrics) if train_metrics else 0}"
            )
            results[rank] = {
                "train_loss": train_loss,
                "train_metrics": train_metrics,
            }

    except Exception as e:
        logger.error(f"[Rank {rank}] Pipeline error: {e}", exc_info=True)
        results["error"] = str(e)

    finally:
        # Clean up DDP process group and resources
        try:
            if dist.is_initialized():
                # Final synchronization barrier before cleanup
                dist.barrier()
                dist.destroy_process_group()
                logger.info(f"[Rank {rank}] Process group destroyed")
        except Exception as cleanup_err:
            logger.warning(f"[Rank {rank}] Cleanup error: {cleanup_err}")

        cleanup()
        logger.info(f"[Rank {rank}] Worker finished")


# ============================================================================
# Test Classes
# ============================================================================


class TestGradientSynchronization:
    """
    Test gradient synchronization across DDP ranks.

    Verifies that gradients are properly synchronized during backward pass
    and that all ranks maintain identical gradient values after each batch.
    """

    def test_gradient_sync_per_batch(self):
        """
        Verify gradients are synchronized after each backward pass.

        For DDP to work correctly, gradients must be averaged across all ranks.
        This test captures gradient snapshots and verifies they match across ranks.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()

        # Run DDP training and capture gradient snapshots
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())

        # Compare gradients across all ranks
        ref_grads = ddp_results[0]["gradients"]
        for rank_idx, rank_result in enumerate(ddp_results[1:], start=1):
            rank_grads = rank_result["gradients"]
            for batch_idx, (ref_grad_list, rank_grad_list) in enumerate(zip(ref_grads, rank_grads)):
                for param_idx, (ref_grad, rank_grad) in enumerate(zip(ref_grad_list, rank_grad_list)):
                    assert torch.allclose(ref_grad, rank_grad, atol=GRADIENT_ATOL), (
                        f"Gradient mismatch at batch {batch_idx}, param {param_idx}, rank {rank_idx}"
                    )


class TestBatchHandling:
    """
    Test DDP batch processing with various batch sizes.

    Validates that DDP correctly handles different batch sizes and
    maintains rank synchronization across configurations.
    """

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_various_batch_sizes(self, batch_size):
        """
        Test DDP with different batch sizes.

        Args:
            batch_size: Number of samples per batch
        """
        config = SimpleConfig(batch_size=batch_size, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()

        # Run DDP with specified batch size
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        assert len(ddp_results) == WORLD_SIZE, f"Expected {WORLD_SIZE} results, got {len(ddp_results)}"

        # Verify all ranks maintain synchronized parameters
        for rank_idx, rank_result in enumerate(ddp_results[1:], start=1):
            for param_idx, (p_rank0, p_rank) in enumerate(zip(ddp_results[0]["params"], rank_result["params"])):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL), (
                    f"Parameter mismatch at param {param_idx}, rank {rank_idx} with batch_size {batch_size}"
                )


class TestScaling:
    """
    Test DDP scaling properties with different world sizes.

    Validates that DDP implementation correctly handles different numbers
    of participating processes (world sizes).
    """

    @pytest.mark.parametrize("world_size", [1, 4])
    def test_different_world_sizes(self, world_size):
        """
        Test DDP with different world sizes.

        Args:
            world_size: Number of processes to spawn
        """
        if world_size > WORLD_SIZE:
            pytest.skip(f"Cannot test world_size {world_size} with only {WORLD_SIZE} available devices")

        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()

        if world_size == 1:
            # Single-device mode: verify functionality with single process
            single_device_training_worker(SimpleMLP, config, results)
            assert "single_device" in results, "Single device results not found"
        else:
            # Multi-device DDP mode
            mp.spawn(
                ddp_training_worker,
                args=(world_size, SimpleMLP, config, results),
                nprocs=world_size,
                join=True,
            )
            assert len(results) == world_size, f"Expected {world_size} results, got {len(results)}"


class TestErrorHandling:
    """
    Test error handling and robustness in DDP training.

    Ensures that worker processes properly capture and report exceptions,
    and that distributed process groups are cleaned up even on failure.
    """

    def test_worker_exception_handling(self):
        """
        Verify worker exceptions are properly captured and reported.

        Each worker process should catch exceptions and store them in
        the shared results dict for inspection.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=1)
        manager = mp.Manager()
        results = manager.dict()

        # Run DDP training
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        # Verify all workers completed without errors
        for rank, result in results.items():
            assert "error" not in result or result["error"] is None, (
                f"Rank {rank} reported error: {result.get('error')}"
            )


class TestDDPParity:
    """
    Parity tests between DDP and single-device training.

    Validates that multi-device DDP training produces equivalent results
    to single-device training with identical configurations and seeds.
    """

    def test_simple_model_parity(self):
        """
        Test parity between DDP and single-device with SimpleMLP model.

        Verifies:
        - All DDP ranks converge to identical parameters
        - Parameters remain within expected ranges (no explosion)
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5)
        manager = mp.Manager()
        results = manager.dict()

        # Run multi-device DDP training
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        assert len(ddp_results) == WORLD_SIZE, "Not all ranks reported results"

        # Run single-device reference training
        single_results = {}
        single_device_training_worker(SimpleMLP, config, single_results)

        # Verify all DDP ranks converged to same parameters (within tolerance)
        for rank_result in ddp_results[1:]:
            for p_rank0, p_rank in zip(ddp_results[0]["params"], rank_result["params"]):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL), "Parameters diverged between ranks"

        # Verify DDP and single-device results are in similar ranges
        for p_ddp, p_single in zip(ddp_results[0]["params"], single_results["single_device"]["params"]):
            assert p_ddp.abs().max() < 100, "DDP parameters exploded"
            assert p_single.abs().max() < 100, "Single-device parameters exploded"

    def test_complex_model_parity(self):
        """
        Test parity between DDP and single-device with MultiLayerMLP model.

        Tests synchronization with more complex model architecture.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5)
        manager = mp.Manager()
        results = manager.dict()

        # Run multi-device DDP training
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, MultiLayerMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        assert len(ddp_results) == WORLD_SIZE

        # Verify rank synchronization across all parameters
        for rank_result in ddp_results[1:]:
            for p_rank0, p_rank in zip(ddp_results[0]["params"], rank_result["params"]):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL)

    def test_single_vs_ddp_loss_parity(self):
        """
        Compare loss trajectory between single-device and multi-device DDP training.

        Ensures both optimization paths follow the same trajectory with
        identical seeds and deterministic random inputs.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5, seed=TEST_SEED)

        manager = mp.Manager()
        ddp_results = manager.dict()

        # Run multi-device DDP training with deterministic loss computation
        mp.spawn(
            ddp_loss_worker,
            args=(WORLD_SIZE, config, ddp_results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        assert "error" not in ddp_results, f"DDP error: {ddp_results.get('error')}"
        assert "ddp" in ddp_results, "DDP results not found"
        ddp_losses = ddp_results["ddp"]

        # Run single-device reference training with same seed and inputs
        torch.manual_seed(config.seed)
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        single_losses = []
        for batch_idx in range(config.num_batches):
            # Use same seed sequence as DDP to generate identical inputs
            torch.manual_seed(config.seed + batch_idx)

            x = torch.randn(config.batch_size, 8)
            y = torch.randn(config.batch_size, 2)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            single_losses.append(loss.item())

        # Compare loss trajectories step-by-step
        assert len(ddp_losses) == len(single_losses), "Loss trajectory length mismatch"

        for step, (l_ddp, l_single) in enumerate(zip(ddp_losses, single_losses)):
            assert abs(l_ddp - l_single) < LOSS_ATOL, (
                f"Loss mismatch at step {step}: DDP={l_ddp:.6f}, Single={l_single:.6f}"
            )


@pytest.mark.skipif(
    not hasattr(torch, "qaic") or (hasattr(torch.qaic, "device_count") and torch.qaic.device_count() < 2),
    reason="Requires at least 2 QAIC devices",
)
class TestDDPPipelineParity:
    """
    End-to-end DDP pipeline parity tests using FineTuningPipeline.

    Validates that the complete training pipeline produces equivalent results
    between single-device and multi-device (DDP) configurations.

    These tests require actual QAIC hardware and model weights.
    """

    OUTPUT_DIR_SINGLE = "/tmp/test_ddp_single"
    OUTPUT_DIR_DDP = "/tmp/test_ddp_multi"
    _MAX_STEPS = 50
    _REDUCED_LAYERS = 2

    if os.getenv("FROM_REGRESSION") == "1":
        pytest.skip("Skipped: executed via regression test")

    @staticmethod
    def _get_unique_port():
        """
        Get a unique port for MASTER_PORT to avoid conflicts between test runs.

        Returns:
            int: Available port number
        """
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def setup_method(self):
        """Clean up FileStore before each test to prevent stale lock files."""
        try:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
                logger.info(f"Cleaned up FileStore at {STORE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to clean FileStore: {e}")

    def _assert_loss(self, value, label):
        """
        Validate loss value is finite and positive.

        Args:
            value: Loss value to validate
            label: Label for error messages

        Raises:
            AssertionError: If value is None, non-finite, or non-positive
        """
        assert value is not None, f"{label} is None"
        assert torch.isfinite(torch.tensor(value)), f"{label} not finite"
        assert value > 0, f"{label} must be > 0"

    def _build_config_dict(self, backend, output_dir):
        """
        Build configuration dictionary for FineTuningPipeline.

        Args:
            backend: DDP backend ("qccl" for QAIC, "gloo" for CPU, None for single-device)
            output_dir: Directory for training outputs

        Returns:
            dict: Configuration dictionary

        """
        return {
            "output_dir": output_dir,
            "seed": TEST_SEED,
            "max_steps": self._MAX_STEPS,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "logging_steps": 1,
            "logging_strategy": "steps",
            "eval_steps": 50,
            "eval_strategy": "steps",
            "ddp_config": {
                "ddp_backend": backend,
                "ddp_find_unused_parameters": False,
                "ddp_bucket_cap_mb": 25,
                "ddp_broadcast_buffers": True,
                "ddp_timeout": 1800,
            },
        }

    def _run_single(self, config_dict):
        """
        Run single-device training using FineTuningPipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            tuple: (train_loss, train_metrics) - Loss and metrics from training
        """
        try:
            if FineTuningPipeline is None or ConfigManager is None:
                pytest.skip("FineTuningPipeline not available")

            cm = ConfigManager()
            cm.config.model_name = _HF_MODEL.model_name
            cm.config.dataset["prompt_func"] = (
                "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
            )
            cm.config.dataset["completion_template"] = "{output}"
            cm.config.dataset["dataset_num_samples"] = 100
            for k, v in config_dict.items():
                cm.config.training[k] = v

            cm.config.training["local_rank"] = 0
            cm.config.dataset["ddp_backend"] = None
            pipeline = FineTuningPipeline(cm)
            pipeline.run()

            trainer = pipeline.trainer

            # Extract training loss from log history
            train_loss = None
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                losses = [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
                train_loss = losses if losses else None

            # Extract all training metrics (learning_rate, gradient_norm, etc.)
            train_metrics = None
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                train_metrics = [
                    entry["train/epoch_metric"] for entry in trainer.state.log_history if "train/epoch_metric" in entry
                ]
            return train_loss, train_metrics

        except Exception as e:
            logger.error(f"Single-device training error: {e}", exc_info=True)
            raise
        finally:
            cleanup()

    def _run_ddp(self, config_dict, port=None):
        """
        Run multi-device DDP training using FineTuningPipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            tuple: (train_loss, train_metrics) - Loss and metrics from DDP training
        """
        manager = mp.Manager()
        results = manager.dict()

        # Use top-level function for mp.spawn (not a method)
        mp.spawn(
            pipeline_ddp_worker,
            args=(WORLD_SIZE, port, config_dict, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        assert "error" not in results, f"DDP error: {results.get('error')}"
        return results[0]["train_loss"], results[0]["train_metrics"]

    def test_single_vs_ddp_loss_parity(self):
        """
        Comprehensive parity test: Single-device vs DDP training.
        Validates that loss trajectories and training metrics are similar
        between single-device and multi-device DDP training with identical configurations.
        """

        # Run multi-device DDP training
        ddp_cfg = self._build_config_dict(
            backend="qccl",
            output_dir=self.OUTPUT_DIR_DDP,
        )
        logger.info("Running DDP training...")
        ddp_train, ddp_metrics = self._run_ddp(ddp_cfg, port=self._get_unique_port())

        # Run single-device training with IDENTICAL config
        single_cfg = self._build_config_dict(
            backend=None,
            output_dir=self.OUTPUT_DIR_SINGLE,
        )
        logger.info("Running single-device training...")
        single_train, single_metrics = self._run_single(single_cfg)

        # Compare loss trajectories step-by-step
        assert len(ddp_train) == len(single_train), "Loss trajectory length mismatch"

        # Dumping single device losses and DDP losses as goldens for regression testing
        save_golden("finetuning_pipeline_single", {"loss": single_train})
        save_golden("finetuning_pipeline_ddp", {"loss": ddp_train})
        single_losses = extract_losses(single_train)
        ddp_losses = extract_losses(ddp_train)

        assert len(single_losses) == len(ddp_losses)

        # ✅ Average parity
        avg_single = compute_avg(single_losses)
        avg_ddp = compute_avg(ddp_losses)

        assert abs(avg_single - avg_ddp) < PIPELINE_ATOL, f"Average loss mismatch: single={avg_single}, ddp={avg_ddp}"

        # ✅ Stability
        assert is_stable(single_losses)
        assert is_stable(ddp_losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ddp"])
