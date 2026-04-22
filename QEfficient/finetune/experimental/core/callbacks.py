# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import logging
import math
import os
import torch
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import torch.distributed as dist
from transformers import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainingArguments,
)

from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler, ProfilerAction
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry
from QEfficient.finetune.experimental.core.config_manager import ConfigManager
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.core.utils.profiler_utils import (
    get_op_verifier_ctx,
    init_qaic_profiling,
    stop_qaic_profiling,
)

registry.callback("early_stopping")(EarlyStoppingCallback)
registry.callback("printer")(PrinterCallback)
registry.callback("default_flow")(DefaultFlowCallback)
registry.callback("tensorboard")(TensorBoardCallback)

logger = Logger(__name__)
# Setting the path for dumping the log file
output_dir = Path(ConfigManager().config.training["output_dir"])
log_file_name = os.path.join(output_dir,f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
#log_file_path = os.path.join(output_dir, log_file_name)
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
import time
import json
rank = 0  # Assuming single process for simplicity; replace with actual rank in distributed setting
def log(text):
    if rank != 0:
        return
    logger.log_rank_zero(text)
    with open(log_file_name, "a") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())

'''
@registry.callback("torch_profiler")
class TorchProfilerCallback(TrainerCallback):
    def __init__(
        self,
        output_dir=os.path.join(Path(ConfigManager().config.training["output_dir"]),"profiler"),
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    ):
        self.profile = None
        self.output_dir = output_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat

    def on_train_begin(self, args, state, control, **kwargs):
        # ✅ Profile ONLY on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(self.output_dir, exist_ok=True)

            self.profile = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=schedule(
                    wait=self.wait,
                    warmup=self.warmup,
                    active=self.active,
                    repeat=self.repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(self.output_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self.profile.__enter__()


    def on_step_end(self, args, state, control, **kwargs):
        if self.profile:
            self.profile.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profile:
            self.profile.__exit__(None, None, None)

'''
@registry.callback("distributed_profiler")
class DistributedTrainingProfiler(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.comm_times = []
        self.sync_times = []
        self.rank = 0
        self.world_size = 1
        self.is_distributed = True
        self.is_qaic = True
        
        # Check if distributed training is active
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self.is_distributed = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        except:
            pass
            
        # Check for QAIC
        try:
            import torch_qaic  # noqa: F401
            self.is_qaic = True
        except ImportError:
            self.is_qaic = False
            
        self.step_start = None
        self.comm_start = None
        
        # Log initialization
        device_type = "QAIC" if self.is_qaic else "standard"
        log(f"DistributedTrainingProfiler initialized: distributed={self.is_distributed}, rank={self.rank}, device={device_type}")

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.is_distributed or self.rank != 0:
            return
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_distributed or self.rank != 0:
            return
            
        step_time = time.time() - self.step_start
        self.step_times.append(step_time)
        
        # Log step time trends
        if len(self.step_times) >= 10:
            recent_avg = sum(self.step_times[-10:]) / 10
            overall_avg = sum(self.step_times) / len(self.step_times)
            slowdown_ratio = recent_avg / overall_avg
            
            device_type = "QAIC" if self.is_qaic else "standard"
            # Always log recent step time for monitoring
            log(f"Step {state.global_step} - Time: {step_time:.3f}s, Recent avg: {recent_avg:.3f}s")
            
            if slowdown_ratio > 1.1:  # 10% slowdown
                log(f"⚠️  {device_type} DISTRIBUTED SLOWDOWN: Recent steps {recent_avg:.3f}s vs overall {overall_avg:.3f}s (ratio: {slowdown_ratio:.2f})")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.is_distributed or self.rank != 0:
            return
            
        epoch = int(state.epoch)
        
        # Analyze step time trends
        if len(self.step_times) > 50:  # Need enough data
            # Split into first half vs second half of epoch
            midpoint = len(self.step_times) // 2
            first_half_avg = sum(self.step_times[:midpoint]) / midpoint
            second_half_avg = sum(self.step_times[midpoint:]) / (len(self.step_times) - midpoint)
            slowdown = second_half_avg / first_half_avg
            
            device_type = "QAIC" if self.is_qaic else "standard"
            log(f"Epoch {epoch} - {device_type} Distributed Analysis:")
            log(f"  First half avg step time: {first_half_avg:.3f}s")
            log(f"  Second half avg step time: {second_half_avg:.3f}s")
            log(f"  Slowdown ratio: {slowdown:.3f}")
            
            if slowdown > 1.15:  # 15% slowdown within epoch
                log(f"🚨 CRITICAL: Severe within-epoch slowdown detected! Likely {device_type} distributed training issue.")
            
            # Check for device-specific communication issues
            try:
                import torch.distributed as dist
                backend = dist.get_backend()
                log(f"  Backend: {backend}")
                log(f"  World Size: {self.world_size}, Rank: {self.rank}")
                
                if self.is_qaic:
                    log("  Device: QAIC - monitoring for QAIC-specific communication patterns")
                else:
                    # Check NCCL for non-QAIC
                    nccl_version = torch.cuda.nccl.version() if torch.cuda.is_available() else "N/A"
                    log(f"  NCCL Version: {nccl_version}")
            except:
                pass


@registry.callback("communication_profiler") 
class CommunicationProfiler(TrainerCallback):
    def __init__(self):
        self.comm_overhead = []
        self.barrier_times = []
        self.is_distributed = True
        self.rank = 0
        self.is_qaic = True
        
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self.is_distributed = True
                self.rank = dist.get_rank()
        except:
            pass
            
        # Check for QAIC
        try:
            import torch_qaic  # noqa: F401
            self.is_qaic = True
        except ImportError:
            self.is_qaic = False
            
        # Log initialization
        device_type = "QAIC" if self.is_qaic else "standard"
        log(f"CommunicationProfiler initialized: distributed={self.is_distributed}, rank={self.rank}, device={device_type}")

    def measure_barrier_time(self):
        """Measure time for a distributed barrier (synchronization point)"""
        if not self.is_distributed:
            return 0
            
        try:
            import torch.distributed as dist
            start = time.time()
            dist.barrier()
            # QAIC-specific synchronization if needed
            if self.is_qaic and hasattr(torch, 'qaic'):
                try:
                    torch.qaic.synchronize()
                except:
                    pass  # QAIC sync may not be available
            return time.time() - start
        except:
            return 0

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_distributed or self.rank != 0:
            return
            
        # Measure barrier time (indicates communication overhead)
        barrier_time = self.measure_barrier_time()
        self.barrier_times.append(barrier_time)
        
        device_type = "QAIC" if self.is_qaic else "standard"
        # Always log barrier time for debugging
        log(f"{device_type} barrier time: {barrier_time:.4f}s (step {state.global_step})")
        
        # Check for increasing communication overhead
        if len(self.barrier_times) >= 20:
            recent_avg = sum(self.barrier_times[-10:]) / 10
            overall_avg = sum(self.barrier_times) / len(self.barrier_times)
            
            if recent_avg > overall_avg * 1.2:  # 20% increase
                log(f"⚠️  Increasing {device_type} communication overhead: {recent_avg:.4f}s vs {overall_avg:.4f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.is_distributed or self.rank != 0:
            return
            
        if self.barrier_times:
            avg_barrier = sum(self.barrier_times) / len(self.barrier_times)
            max_barrier = max(self.barrier_times)
            
            device_type = "QAIC" if self.is_qaic else "standard"
            log(f"Epoch {int(state.epoch)} - {device_type} Communication Stats:")
            log(f"  Avg barrier time: {avg_barrier:.4f}s")
            log(f"  Max barrier time: {max_barrier:.4f}s")
            
            # Flag if communication is a bottleneck (lower threshold for QAIC)
            threshold = 0.005 if self.is_qaic else 0.01  # QAIC is more sensitive
            if avg_barrier > threshold:
                log(f"🚨 WARNING: High {device_type} communication overhead detected!")



@registry.callback("dataloading_profiler")
class DataLoadingProfiler(TrainerCallback):
    def __init__(self):
        self.data_load_times = []
        self.iter_start = None
        self.epoch_data_loads = []  # Track per-epoch totals

    def on_step_begin(self, args, state, control, **kwargs):
        # Measure time to get next batch
        self.iter_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        data_load_time = time.time() - self.iter_start
        self.data_load_times.append(data_load_time)

        # # Flag if data loading becomes a bottleneck
        # if data_load_time > 0.5:  # Adjust threshold
        #     log(
        #         f"SLOW DATA LOAD: {data_load_time:.3f}s (step {state.global_step})"
        #     )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.data_load_times:
            avg_time = sum(self.data_load_times) / len(self.data_load_times)
            max_time = max(self.data_load_times)
            total_time = sum(self.data_load_times)
            
            epoch_num = int(state.epoch)
            
            # Store epoch-level metrics
            epoch_stats = {
                "epoch": epoch_num,
                "avg_dataload_time_sec": avg_time,
                "max_dataload_time_sec": max_time,
                "total_dataload_time_sec": total_time,
                "num_steps": len(self.data_load_times),
            }
            self.epoch_data_loads.append(epoch_stats)
            
            # Log to wandb if available
            if args.local_rank in [-1, 0]:  # Only rank 0
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        f"dataload/epoch_{epoch_num}_avg_time_sec": avg_time,
                        f"dataload/epoch_{epoch_num}_max_time_sec": max_time,
                        f"dataload/epoch_{epoch_num}_total_time_sec": total_time,
                        "epoch": epoch_num,
                    })
            
            log(
                f"Epoch {epoch_num} - Data Load Stats: "
                f"avg={avg_time:.4f}s, max={max_time:.4f}s, total={total_time:.2f}s"
            )
            
            # Reset for next epoch
            self.data_load_times = []

@registry.callback("experiment_tracker")
class ExperimentTracker(TrainerCallback):
    def __init__(self, experiment_name="baseline_experiment"):
        self.experiment_name = experiment_name
        self.epoch_times = []
        self.batch_times = []
        self.epoch_start = None
        self.batch_start = None
 
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
 
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append({
            "epoch": int(state.epoch),
            "time_seconds": epoch_time
        })
        log(
            f"[{self.experiment_name}] Epoch {int(state.epoch)}: {epoch_time:.2f}s"
        )

    def on_step_begin(self, args, state, control, **kwargs):
        self.batch_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)

    def save_results(self, output_path):
        results = {
            "experiment": self.experiment_name,
            "epoch_times": self.epoch_times,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times),
            "batch_times_std": (sum((x - sum(self.batch_times)/len(self.batch_times))**2 
                               for x in self.batch_times) / len(self.batch_times))**0.5,
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results

@registry.callback("train_logger")
class TrainingLogger(TrainerCallback):
    def __init__(self,rank=0, log_file: str | None = log_file_name):
        self.rank = rank  # rank-safe logging (only rank 0)
        # Log file setup
        self.log_file = log_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.epoch_start_time = None
        self.best_eval_loss = float("inf")
    # ----------------------------------------------------
    # Safe write to log (only rank 0)
    # ----------------------------------------------------
    def write(self, text):
        if self.rank != 0:
            return
        logger.log_rank_zero(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ----------------------------------------------------
    # EPOCH BEGIN
    # ----------------------------------------------------
    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.rank != 0:
            return

        epoch = int(state.epoch) + 1
        self.epoch_start_time = time.time()
        if state.is_world_process_zero:
            self.write(f"TRAINING INFO: Starting epoch {epoch}/{int(args.num_train_epochs)}")

    # ----------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.rank != 0:
            return

        epoch = int(state.epoch)
        eval_loss = None
        eval_metric = None

        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break
        if eval_loss is not None:
            eval_metric = math.exp(eval_loss)
        # Track best eval loss
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            if state.is_world_process_zero:
                self.write(f"EVALUATION INFO: Best eval loss on epoch {epoch} is {eval_loss:.4f}")
        if state.is_world_process_zero:
            self.write(
                f"EVALUATION INFO: Epoch {epoch}: Eval Loss: {eval_loss:.4f} || Eval metric: {eval_metric:.4f}"
            )

    # ----------------------------------------------------
    # EPOCH END — TRAIN LOSS + METRIC + TIME
    # ----------------------------------------------------
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.rank != 0:
            return

        epoch = int(state.epoch)
        epoch_time = time.time() - self.epoch_start_time

        # Extract the last recorded train loss
        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break

        # Compute perplexity safely
        train_metric = None
        if train_loss is not None:
            train_metric = math.exp(train_loss)
        if state.is_world_process_zero:
            self.write(
                f"TRAINING INFO: Epoch {epoch}: "
                f" Train epoch loss: {train_loss:.4f} || "
                f" Train metric: {train_metric} || "
                f" Epoch time {epoch_time:.2f} sec"
            )
        state.log_history.append({"train/epoch_time_sec": epoch_time, "epoch": state.epoch})
        control.should_log = True




'''
logger = Logger(__name__)
# Extracting the user input name for the log file
log_file_name = ConfigManager().config.training["log_file_name"]


@registry.callback("train_logger")
class TrainingLogger(TrainerCallback):
    """
    A [`TrainerCallback`] that logs per epoch time, training metric (perplexity),training loss, evaluation metrics and loss etc.
    These are only logged for rank = 0.
    """

    def __init__(self, log_file: str | None = log_file_name):
        # Log file setup
        output_dir = Path(ConfigMlog_file: str | None = log_file_nameanager().config.training["output_dir"])
        self.log_file = os.path.join(output_dir, log_file)
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.epoch_start_time = None
        self.best_eval_loss = float("inf")

    # ----------------------------------------------------
    # Safe write to log (only rank 0)
    # ----------------------------------------------------
    def write(self, text):
        #if not is_main_process():
        #    return
        logger.log_rank_zero(text)
        try:
            with open(self.log_file, "a") as f:
                f.write(text + "\n")
                f.flush()
                os.fsync(f.fileno())

        except OSError:
            logging.exception("Failed to write to log file: %s", self.log_file)

    # ----------------------------------------------------
    # EPOCH BEGIN
    # ----------------------------------------------------
    def on_epoch_begin(self, args, state, control, **kwargs):
        if not  state.is_world_process_zero:
            return

        epoch = int(state.epoch) + 1
        self.epoch_start_time = time.time()
        if state.is_world_process_zero:
            self.write(f"TRAINING INFO: Starting epoch {epoch}/{int(args.num_train_epochs)}")

    # ----------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not  state.is_world_process_zero:
            return

        epoch = int(state.epoch)
        eval_loss = None
        eval_metric = None

        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break
        if eval_loss is not None:
            eval_metric = math.exp(eval_loss)
        # Track best eval loss
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            if state.is_world_process_zero:
                self.write(f"EVALUATION INFO: Best eval loss on epoch {epoch} is {eval_loss:.4f}")
        if state.is_world_process_zero:
            self.write(f"EVALUATION INFO: Epoch {epoch}: Eval Loss: {eval_loss:.4f} || Eval metric: {eval_metric:.4f}")

    # ----------------------------------------------------
    # EPOCH END — TRAIN LOSS + METRIC + TIME
    # ----------------------------------------------------
    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        epoch = int(state.epoch)
        epoch_time = time.time() - self.epoch_start_time

        # Extract the last recorded train loss
        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break
        # Compute perplexity safely
        train_metric = None
        if train_loss is not None:
            train_metric = math.exp(train_loss)

        if state.is_world_process_zero:
            self.write(
                f"TRAINING INFO: Epoch {epoch}: "
                f" Train epoch loss: {train_loss:.4f} || "
                f" Train metric: {train_metric} || "
                f" Epoch time {epoch_time:.2f} sec"
            )
        state.log_history.append({"train/epoch_time_sec": epoch_time, "epoch": state.epoch})
        control.should_log = True
'''

@registry.callback("enhanced_progressbar")
class EnhancedProgressCallback(ProgressCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        super().__init__(*args, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        """Set progress bar description at the start of training."""
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Training Progress")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Override the default `on_log` behavior during training to display
        the current epoch number, loss, and learning rate in the logs.
        """
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            updated_dict = {}
            if "epoch" in shallow_logs:
                updated_dict["epoch"] = shallow_logs["epoch"]
            if "loss" in shallow_logs:
                updated_dict["loss"] = shallow_logs["loss"]
            if "learning_rate" in shallow_logs:
                updated_dict["lr"] = shallow_logs["learning_rate"]
            self.training_bar.set_postfix(updated_dict)


@registry.callback("json_logger")
class JSONLoggerCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs training and evaluation metrics to a JSON file.
    """

    def __init__(self, log_path=None, *args, **kwargs):
        """
        Initialize the callback with the path to the JSON log file.

        Args:
            log_path (`str`):
                Path to the jsonl file where logs will be saved.
        """
        super().__init__(*args, **kwargs)
        if log_path is None:
            log_path = os.path.join(os.environ.get("OUTPUT_DIR", "./"), "training_logs.jsonl")
        self.log_path = log_path
        # Ensure the log file is created and empty
        with open(self.log_path, "w") as _:
            pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ):
        """Append sanitized log metrics (including global_step) to a JSONL file."""
        if logs is None:
            return
        logs.pop("entropy", None)
        logs.pop("mean_token_accuracy", None)
        if state.global_step:
            logs["global_step"] = state.global_step
        if logs is not None:
            with open(self.log_path, "a") as f:
                json_line = json.dumps(logs, separators=(",", ":"))
                f.write(json_line + "\n")


@registry.callback("qaic_profiler_callback")
class QAICProfilerCallback(TrainerCallback):
    """Callback to profile QAIC devices over a specified training step range."""

    def __init__(self, *args, **kwargs):
        """
        Initialize QAIC profiler settings (start/end steps and target device IDs).
        """

        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.device_ids = kwargs.get("device_ids", [0])

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.global_step == self.start_step:
            for device_id in self.device_ids:
                init_qaic_profiling(True, f"qaic:{device_id}")
        elif state.global_step == self.end_step:
            for device_id in self.device_ids:
                stop_qaic_profiling(True, f"qaic:{device_id}")


@registry.callback("qaic_op_by_op_verifier_callback")
class QAICOpByOpVerifierCallback(TrainerCallback):
    """Callback to verify QAIC operations step-by-step during a specified training range."""

    def __init__(self, *args, **kwargs):
        """ "
        Initialize QAIC Op-by-Op verifier callback with profiling and tolerance settings.
        """
        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.trace_dir = kwargs.get("trace_dir", "qaic_op_by_op_traces")
        self.atol = kwargs.get("atol", 1e-1)
        self.rtol = kwargs.get("rtol", 1e-5)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            self.op_verifier_ctx_step = get_op_verifier_ctx(
                use_op_by_op_verifier=True,
                device_type="qaic",
                dump_dir=self.trace_dir,
                step=state.global_step,
                atol=self.atol,
                rtol=self.rtol,
            )
            self.op_verifier_ctx_step.__enter__()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            if self.op_verifier_ctx_step is not None:
                self.op_verifier_ctx_step.__exit__(None, None, None)


def replace_progress_callback(trainer: Any, callbacks: list[Any], logger: Any = None) -> None:
    """
    Replace default ProgressCallback with EnhancedProgressCallback if not already present.

    Args:
        trainer: Trainer instance
        callbacks: List of callbacks already added
        logger: Optional logger instance for warning messages
    """
    # Check if EnhancedProgressCallback is already in callbacks
    has_enhanced = any(callback.__class__.__name__ == "EnhancedProgressCallback" for callback in callbacks)
    if not has_enhanced:
        try:
            # Remove default ProgressCallback if present
            trainer.remove_callback(ProgressCallback)
        except (AttributeError, ValueError) as e:
            # Callback not present or method doesn't exist, continue
            if logger:
                logger.log_rank_zero(
                    f"Debug: Could not remove default ProgressCallback: {e}. This is expected if callback is not present.",
                    level="debug",
                )
            pass

        try:
            # Add EnhancedProgressCallback
            enhanced_callback = ComponentFactory.create_callback("enhanced_progressbar")
            trainer.add_callback(enhanced_callback)
        except Exception as e:
            if logger:
                logger.log_rank_zero(f"Warning: Could not add enhanced progress callback: {e}", level="warning")
            else:
                import warnings

                warnings.warn(f"Could not add enhanced progress callback: {e}")
        try:
            # Add Train Logger
            train_logger = ComponentFactory.create_callback("train_logger")
            trainer.add_callback(train_logger)
        except Exception as e:
            if logger:
                logger.log_rank_zero(f"Warning: Could not add train logger callback: {e}", level="warning")
            else:
                import warnings

                warnings.warn(f"Could not add train warning callback: {e}")
       