#!/usr/bin/env python3
"""
Simple Benchmark Runner - Refactored

A clean, modular benchmark runner that:
- Accepts model objects directly for evaluation
- Can be imported and used in other classes
- Provides separate model loading functionality
- Maintains simple, clean code structure

Usage as script:
  python run_benchmark_refactored.py --model meta-llama/Meta-Llama-3-8B-Instruct --dataset gsm8k

Usage as module:
  from run_benchmark_refactored import run_lm_eval, load_model
  model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
  results = run_lm_eval(model, tokenizer, tasks=["gsm8k"])
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
from uuid import uuid4

from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.runtimes.hf import HuggingFaceRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.eval import EvalSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import (
    CascadedCompensationConfig,
    CompensationLayerRatio,
    CompensationSpec,
    LastTokenCompensationConfig,
    LearnableCompensationConfig,
    MagnitudePreservingCompensationConfig,
    MagnitudeRescalingCompensationConfig,
    MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig,
    MultiplicativeCompensationConfig,
    PcaCompensationConfig,
    PhaseAwareCompensationConfig,
    PhaseAwareLastTokenCompensationConfig,
    PhaseAwareMagnitudeRescalingCompensationConfig,
    PositionAwareCompensationConfig,
    ScaledCompensationConfig,
    SkipLayersSpec,
    TransformationPlan,
)
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier
from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)

# Available benchmarks
BENCHMARK_MAPPING = {
    "gsm8k": "gsm8k",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "truthfulqa": "truthfulqa_mc2",
    "piqa": "piqa",
    "boolq": "boolq",
    "openbookqa": "openbookqa"
}


def get_tokenizer_kwargs(model_name: str, trust_remote_code: bool) -> Dict[str, Any]:
    del model_name
    return {"trust_remote_code": trust_remote_code}


def make_json_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to serializable format."""
    import numpy as np
    import torch
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.dtype, np.dtype)):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


SUPPORTED_API_COMPENSATION_STRATEGIES = {
    None,
    "scaled",
    "last_token",
    "magnitude_preserving",
    "phase_aware",
    "phase_last_token",
    "pca",
    "position_aware",
    "multiplicative",
    "cascaded",
    "magnitude_rescaling",
    "phase_aware_magnitude_rescaling",
    "multi_phase_aware_magnitude_rescaling",
    "learnable",
}


def should_use_api_load_path(
    use_compensation: bool,
    compensation_strategy: Optional[str] = None,
) -> bool:
    """Return True when model loading should use the API-first loader path."""

    if not use_compensation:
        return True
    return compensation_strategy in SUPPORTED_API_COMPENSATION_STRATEGIES


def _sanitize_ratio(raw_value: Any, fallback: float) -> float:
    try:
        ratio = float(raw_value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(ratio) or ratio <= 0:
        return float(fallback)
    return max(0.5, min(3.0, ratio))


def build_compensation_spec(
    use_compensation: bool = False,
    compensation_vector_file: Optional[str] = None,
    compensation_strategy: Optional[str] = None,
    compensation_alpha: float = 1.0,
    compensation_decode_vector_file: Optional[str] = None,
    compensation_pca_file: Optional[str] = None,
    compensation_n_pca_components: int = 32,
    compensation_norm_ratio: float = 1.0,
    compensation_pre_skip_fraction: float = 0.5,
    compensation_bucket_file: Optional[str] = None,
    compensation_cluster_ratios_file: Optional[str] = None,
    learnable_compensation_model: Optional[str] = None,
) -> Optional[CompensationSpec]:
    """Build a typed compensation spec from the legacy CLI/programmatic args."""

    if not use_compensation:
        return None

    strategy = compensation_strategy or "scaled"

    if strategy == "scaled":
        if compensation_vector_file is None:
            raise ValueError("scaled compensation requires compensation_vector_file")
        alpha = compensation_alpha if compensation_strategy else 1.0
        return CompensationSpec(
            config=ScaledCompensationConfig(
                mean_delta_path=compensation_vector_file,
                alpha=alpha,
            )
        )
    if strategy == "last_token":
        if compensation_vector_file is None:
            raise ValueError("last_token compensation requires compensation_vector_file")
        return CompensationSpec(
            config=LastTokenCompensationConfig(
                mean_delta_path=compensation_vector_file,
                alpha=compensation_alpha,
            )
        )
    if strategy == "magnitude_preserving":
        if compensation_vector_file is None:
            raise ValueError(
                "magnitude_preserving compensation requires compensation_vector_file"
            )
        return CompensationSpec(
            config=MagnitudePreservingCompensationConfig(
                mean_delta_path=compensation_vector_file,
                alpha=compensation_alpha,
            )
        )
    if strategy == "phase_aware":
        if compensation_vector_file is None:
            raise ValueError("phase_aware compensation requires compensation_vector_file")
        if compensation_decode_vector_file is None:
            raise ValueError(
                "phase_aware compensation requires compensation_decode_vector_file"
            )
        return CompensationSpec(
            config=PhaseAwareCompensationConfig(
                prefill_delta_path=compensation_vector_file,
                decode_delta_path=compensation_decode_vector_file,
                prefill_alpha=compensation_alpha,
                decode_alpha=compensation_alpha,
            )
        )
    if strategy == "phase_last_token":
        if compensation_vector_file is None:
            raise ValueError(
                "phase_last_token compensation requires compensation_vector_file"
            )
        if compensation_decode_vector_file is None:
            raise ValueError(
                "phase_last_token compensation requires compensation_decode_vector_file"
            )
        return CompensationSpec(
            config=PhaseAwareLastTokenCompensationConfig(
                prefill_delta_path=compensation_vector_file,
                decode_delta_path=compensation_decode_vector_file,
                prefill_alpha=compensation_alpha,
                decode_alpha=compensation_alpha,
            )
        )
    if strategy == "pca":
        if compensation_pca_file is None:
            raise ValueError("pca compensation requires compensation_pca_file")
        return CompensationSpec(
            config=PcaCompensationConfig(
                pca_path=compensation_pca_file,
                n_components=compensation_n_pca_components,
                alpha=compensation_alpha,
                mean_delta_path=compensation_vector_file,
            )
        )
    if strategy == "position_aware":
        if compensation_bucket_file is None:
            raise ValueError(
                "position_aware compensation requires compensation_bucket_file"
            )
        return CompensationSpec(
            config=PositionAwareCompensationConfig(
                bucket_deltas_path=compensation_bucket_file,
                alpha=compensation_alpha,
                fallback_delta_path=compensation_vector_file,
            )
        )
    if strategy == "multiplicative":
        if compensation_vector_file is None:
            raise ValueError(
                "multiplicative compensation requires compensation_vector_file"
            )
        if compensation_decode_vector_file is None:
            raise ValueError(
                "multiplicative compensation requires compensation_decode_vector_file"
            )
        return CompensationSpec(
            config=MultiplicativeCompensationConfig(
                scale_vector_path=compensation_vector_file,
                bias_vector_path=compensation_decode_vector_file,
            )
        )
    if strategy == "cascaded":
        if compensation_vector_file is None:
            raise ValueError("cascaded compensation requires compensation_vector_file")
        return CompensationSpec(
            config=CascadedCompensationConfig(
                mean_delta_path=compensation_vector_file,
                pre_skip_fraction=compensation_pre_skip_fraction,
                alpha=compensation_alpha,
            )
        )
    if strategy == "magnitude_rescaling":
        if compensation_vector_file is None:
            raise ValueError(
                "magnitude_rescaling compensation requires compensation_vector_file"
            )
        return CompensationSpec(
            config=MagnitudeRescalingCompensationConfig(
                mean_delta_path=compensation_vector_file,
                norm_ratio=compensation_norm_ratio,
                alpha=compensation_alpha,
            )
        )
    if strategy == "phase_aware_magnitude_rescaling":
        decode_ratio = (
            compensation_alpha
            if compensation_alpha != 1.0
            else compensation_norm_ratio
        )
        return CompensationSpec(
            config=PhaseAwareMagnitudeRescalingCompensationConfig(
                prefill_norm_ratio=compensation_norm_ratio,
                decode_norm_ratio=decode_ratio,
            )
        )
    if strategy == "multi_phase_aware_magnitude_rescaling":
        if compensation_cluster_ratios_file is None:
            raise ValueError(
                "multi_phase_aware_magnitude_rescaling requires "
                "compensation_cluster_ratios_file"
            )
        with open(compensation_cluster_ratios_file, "r", encoding="utf-8") as handle:
            raw_ratios = json.load(handle)

        if isinstance(raw_ratios, dict) and "clusters" in raw_ratios:
            cluster_entries = raw_ratios["clusters"]
        elif isinstance(raw_ratios, list):
            cluster_entries = raw_ratios
        else:
            raise ValueError(
                "cluster ratios file must be a list or an object with a 'clusters' list"
            )

        default_decode_ratio = (
            compensation_alpha if compensation_alpha != 1.0 else compensation_norm_ratio
        )
        layer_ratios = []
        for entry in cluster_entries:
            if not isinstance(entry, dict):
                raise ValueError("Each cluster ratio entry must be an object")

            if "compensation_layer" in entry:
                comp_layer = int(entry["compensation_layer"])
            elif "start_layer" in entry:
                comp_layer = int(entry["start_layer"]) - 1
            else:
                raise ValueError(
                    "Each cluster entry must include 'compensation_layer' or 'start_layer'"
                )

            if comp_layer < 0:
                raise ValueError(f"Invalid compensation layer {comp_layer}")

            prefill_ratio = _sanitize_ratio(
                entry.get(
                    "prefill_norm_ratio",
                    entry.get("prefill", compensation_norm_ratio),
                ),
                fallback=compensation_norm_ratio,
            )
            decode_ratio = _sanitize_ratio(
                entry.get(
                    "decode_norm_ratio",
                    entry.get("decode", default_decode_ratio),
                ),
                fallback=default_decode_ratio,
            )
            layer_ratios.append(
                CompensationLayerRatio(
                    layer=comp_layer,
                    prefill_norm_ratio=prefill_ratio,
                    decode_norm_ratio=decode_ratio,
                )
            )

        if not layer_ratios:
            raise ValueError("No valid layer ratios found in cluster ratios file")

        return CompensationSpec(
            config=MultiClusterPhaseAwareMagnitudeRescalingCompensationConfig(
                layer_ratios=layer_ratios,
                default_prefill_norm_ratio=compensation_norm_ratio,
                default_decode_norm_ratio=default_decode_ratio,
            )
        )
    if strategy == "learnable":
        if learnable_compensation_model is None:
            raise ValueError(
                "learnable compensation requires learnable_compensation_model"
            )
        return CompensationSpec(
            config=LearnableCompensationConfig(
                model_path=learnable_compensation_model,
            )
        )

    raise ValueError(f"Unsupported API compensation strategy: {strategy}")


def build_api_transformation_plan(
    skip_layers: Optional[List[int]] = None,
    use_compensation: bool = False,
    compensation_vector_file: Optional[str] = None,
    compensation_strategy: Optional[str] = None,
    compensation_alpha: float = 1.0,
    compensation_decode_vector_file: Optional[str] = None,
    compensation_pca_file: Optional[str] = None,
    compensation_n_pca_components: int = 32,
    compensation_norm_ratio: float = 1.0,
    compensation_pre_skip_fraction: float = 0.5,
    compensation_bucket_file: Optional[str] = None,
    compensation_cluster_ratios_file: Optional[str] = None,
    learnable_compensation_model: Optional[str] = None,
) -> Optional[TransformationPlan]:
    """Build the combined API-first transform plan for load/eval paths."""

    transforms = []
    if skip_layers:
        transforms.append(SkipLayersSpec(layers=list(skip_layers)))

    compensation_spec = build_compensation_spec(
        use_compensation=use_compensation,
        compensation_vector_file=compensation_vector_file,
        compensation_strategy=compensation_strategy,
        compensation_alpha=compensation_alpha,
        compensation_decode_vector_file=compensation_decode_vector_file,
        compensation_pca_file=compensation_pca_file,
        compensation_n_pca_components=compensation_n_pca_components,
        compensation_norm_ratio=compensation_norm_ratio,
        compensation_pre_skip_fraction=compensation_pre_skip_fraction,
        compensation_bucket_file=compensation_bucket_file,
        compensation_cluster_ratios_file=compensation_cluster_ratios_file,
        learnable_compensation_model=learnable_compensation_model,
    )
    if compensation_spec is not None:
        transforms.append(compensation_spec)

    if not transforms:
        return None
    return TransformationPlan(
        transforms=transforms,
        compatibility_mode="strict",
    )


def load_model_via_api(
    model_name: str,
    trust_remote_code: bool = True,
    skip_layers: Optional[List[int]] = None,
    dtype: str = "bfloat16",
    plan: Optional[TransformationPlan] = None,
    loader=None,
    transform_applier=None,
) -> Tuple[Any, Any]:
    """Load a model through the typed API stack and optionally apply skip-layers."""

    model_loader = loader or TransformersModelLoader()
    applier = transform_applier or TransformApplier()

    model_spec = ModelSpec(
        model_id=model_name,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map="auto",
    )
    model, tokenizer = model_loader.load(model_spec)
    artifact = ModelArtifact(
        artifact_id=uuid4().hex,
        model=model,
        tokenizer=tokenizer,
        model_spec=model_spec,
        plan=TransformationPlan(),
    )

    applied_plan = plan or (
        TransformationPlan(transforms=[SkipLayersSpec(layers=list(skip_layers))])
        if skip_layers else None
    )
    if applied_plan is not None:
        updated = applier.apply(artifact, applied_plan)
        if updated is not artifact:
            raise ValueError("Transform applier must mutate and return the same artifact")

    return artifact.model, artifact.tokenizer


def load_model(
    model_name: str,
    device: str = "cuda",
    trust_remote_code: bool = True,
    skip_layers: Optional[List[int]] = None,
    use_compensation: bool = False,
    compensation_vector_file: Optional[str] = None,
    dtype: str = "bfloat16",
    # Advanced compensation strategy arguments
    compensation_strategy: Optional[str] = None,
    compensation_alpha: float = 1.0,
    compensation_decode_vector_file: Optional[str] = None,
    compensation_pca_file: Optional[str] = None,
    compensation_n_pca_components: int = 32,
    compensation_norm_ratio: float = 1.0,
    compensation_pre_skip_fraction: float = 0.5,
    compensation_bucket_file: Optional[str] = None,
    compensation_cluster_ratios_file: Optional[str] = None,
    learnable_compensation_model: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace model and tokenizer with optional layer skipping and compensation.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use (cuda, cpu, auto)
        trust_remote_code: Whether to trust remote code
        skip_layers: Optional list of layer indices to skip
        use_compensation: Whether to use compensation when skipping layers
        compensation_vector_file: Path to compensation vector file (.pt)
        dtype: Model dtype (bfloat16, float16, float32)
        compensation_strategy: Advanced strategy name (scaled, last_token, phase_aware, etc.)
        compensation_alpha: Scale factor for scaled/last_token strategies
        compensation_decode_vector_file: Decode-phase vector for phase_aware strategies
        compensation_pca_file: PCA components file for pca strategy
        compensation_n_pca_components: Number of PCA components to use
        compensation_norm_ratio: Norm ratio for magnitude_rescaling strategy
        compensation_pre_skip_fraction: Pre-skip fraction for cascaded strategy
        compensation_bucket_file: Position bucket deltas file for position_aware strategy
        compensation_cluster_ratios_file: JSON file with per-cluster norm ratios

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If transformers is not installed
        RuntimeError: If model loading fails
    """
    if use_compensation and not skip_layers:
        raise ValueError("use_compensation requires skip_layers to be specified")

    if should_use_api_load_path(use_compensation, compensation_strategy):
        logger.info("Loading model through API-first loader path")
        plan = build_api_transformation_plan(
            skip_layers=skip_layers,
            use_compensation=use_compensation,
            compensation_vector_file=compensation_vector_file,
            compensation_strategy=compensation_strategy,
            compensation_alpha=compensation_alpha,
            compensation_decode_vector_file=compensation_decode_vector_file,
            compensation_pca_file=compensation_pca_file,
            compensation_n_pca_components=compensation_n_pca_components,
            compensation_norm_ratio=compensation_norm_ratio,
            compensation_pre_skip_fraction=compensation_pre_skip_fraction,
            compensation_bucket_file=compensation_bucket_file,
            compensation_cluster_ratios_file=compensation_cluster_ratios_file,
            learnable_compensation_model=learnable_compensation_model,
        )
        return load_model_via_api(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            skip_layers=skip_layers,
            dtype=dtype,
            plan=plan,
        )

    raise ValueError(
        f"Compensation strategy {compensation_strategy!r} is not supported. "
        f"Supported strategies: {sorted(s for s in SUPPORTED_API_COMPENSATION_STRATEGIES if s)}"
    )

def run_lm_eval(
    model: Any,
    tokenizer: Any,
    tasks: List[str],
    batch_size: int = 16,
    device: str = "cuda",
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    use_cache: bool = False,
    cache_dir: Optional[str] = None,
    dtype: str = "auto",
    log_samples: bool = False,
    random_seed: int = 42,
    verbosity: str = "INFO",
    max_gen_toks: Optional[int] = None,
) -> Dict:
    """
    Run lm-evaluation-harness on a model.
    
    This function accepts a pre-loaded model object and runs evaluation.
    It can be imported and used in other classes.
    
    Args:
        model: Pre-loaded model object (transformers.PreTrainedModel)
        tokenizer: Pre-loaded tokenizer object
        tasks: List of task names to evaluate
        batch_size: Batch size for evaluation
        device: Device to use (cuda, cpu, etc.)
        limit: Limit number of samples (for testing)
        num_fewshot: Number of few-shot examples
        use_cache: Enable request caching
        cache_dir: Cache directory path (if use_cache is True)
        dtype: Model dtype (auto, float32, float16, bfloat16)
        log_samples: Log individual sample predictions
        random_seed: Random seed for reproducibility
        verbosity: Logging verbosity level
        max_gen_toks: Optional maximum generated tokens for generate_until tasks
    
    Returns:
        Dictionary containing evaluation results
    
    Raises:
        ImportError: If lm_eval is not installed
        RuntimeError: If evaluation fails
        ValueError: If results are invalid
    """
    # Import lm_eval
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            f"Failed to import lm_eval: {e}\n"
            "Install with: pip install lm-eval>=0.4.0"
        ) from e
    
    logger.info("="*60)
    logger.info("Running LM Evaluation")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Batch size: {batch_size}")
    if max_gen_toks is not None:
        logger.info(f"Max generation tokens: {max_gen_toks}")
    logger.info("="*60)
    
    # Wrap model for lm_eval
    try:
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
            logits_cache=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to wrap model for lm_eval: {e}") from e
    
    # Prepare cache if enabled
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Caching enabled: {cache_dir}")
    
    def _evaluate_task_list(task_list: List[str]) -> Dict:
        return simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=100000,
            log_samples=log_samples,
            write_out=False,  # Don't write automatically
            use_cache=cache_dir if use_cache else None,
            cache_requests=use_cache,
            random_seed=random_seed,
            numpy_random_seed=random_seed,
            torch_random_seed=random_seed,
            fewshot_random_seed=random_seed,
            verbosity=verbosity,
            gen_kwargs={"max_gen_toks": max_gen_toks, "do_sample": False} if max_gen_toks is not None else None,
        )

    def _merge_lm_eval_results(partials: List[Dict], task_errors: Dict[str, str]) -> Dict:
        merged: Dict[str, Any] = {"results": {}, "task_errors": task_errors}
        for partial in partials:
            if not partial or "results" not in partial:
                continue
            for key, value in partial.items():
                if isinstance(value, dict):
                    merged.setdefault(key, {}).update(value)
                elif key not in merged:
                    merged[key] = value
        return merged

    # Run evaluation. If a multi-task run fails because one task's dataset loader
    # is broken in the local environment, retry task-by-task so healthy tasks can
    # still produce an accuracy regression report.
    task_errors: Dict[str, str] = {}
    try:
        results = _evaluate_task_list(tasks)
    except Exception as multi_task_error:
        logger.warning(
            "lm_eval multi-task evaluation failed (%s: %s); retrying tasks individually",
            type(multi_task_error).__name__,
            multi_task_error,
        )
        partials: List[Dict] = []
        for task in tasks:
            try:
                partials.append(_evaluate_task_list([task]))
            except Exception as task_error:
                task_errors[task] = f"{type(task_error).__name__}: {task_error}"
                logger.warning("lm_eval task %s failed: %s", task, task_errors[task])
        results = _merge_lm_eval_results(partials, task_errors)
        if not results.get("results"):
            raise RuntimeError(
                "Evaluation failed for all requested lm_eval tasks. "
                f"Initial error: {multi_task_error}. Task errors: {task_errors}"
            ) from multi_task_error

    # Validate results
    if not results or "results" not in results:
        raise ValueError("Evaluation returned invalid results")
    
    logger.info("✓ Evaluation completed successfully")
    return results


def resolve_tasks(datasets: List[str]) -> List[str]:
    """Resolve CLI dataset aliases to lm_eval task names."""

    if "all" in datasets:
        selected = list(BENCHMARK_MAPPING.keys())
    else:
        selected = datasets
    return [BENCHMARK_MAPPING[dataset] for dataset in selected]


def build_model_spec(args: argparse.Namespace) -> ModelSpec:
    """Build a typed model-loading spec from CLI arguments."""

    return ModelSpec(
        model_id=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        device_map="auto",
    )


def build_eval_spec(
    args: argparse.Namespace,
    tasks: List[str],
    cache_dir: Optional[str],
) -> EvalSpec:
    """Build a typed evaluation spec from CLI arguments."""

    return EvalSpec(
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        use_cache=args.use_cache,
        cache_dir=cache_dir,
        dtype=args.dtype,
        log_samples=args.log_samples,
        random_seed=args.random_seed,
        verbosity=args.verbosity,
    )


def build_transformation_plan(args: argparse.Namespace) -> Optional[TransformationPlan]:
    """Translate CLI transform flags into a typed transformation plan."""

    return build_api_transformation_plan(
        skip_layers=args.skip_layers,
        use_compensation=args.use_compensation,
        compensation_vector_file=getattr(args, "compensation_vector_file", None),
        compensation_strategy=getattr(args, "compensation_strategy", None),
        compensation_alpha=getattr(args, "compensation_alpha", 1.0),
        compensation_decode_vector_file=getattr(
            args,
            "compensation_decode_vector_file",
            None,
        ),
        compensation_pca_file=getattr(args, "compensation_pca_file", None),
        compensation_n_pca_components=getattr(
            args,
            "compensation_n_pca_components",
            32,
        ),
        compensation_norm_ratio=getattr(args, "compensation_norm_ratio", 1.0),
        compensation_pre_skip_fraction=getattr(
            args,
            "compensation_pre_skip_fraction",
            0.5,
        ),
        compensation_bucket_file=getattr(args, "compensation_bucket_file", None),
        compensation_cluster_ratios_file=getattr(
            args,
            "compensation_cluster_ratios_file",
            None,
        ),
        learnable_compensation_model=getattr(
            args,
            "learnable_compensation_model",
            None,
        ),
    )


def should_use_api_path(args: argparse.Namespace) -> bool:
    """Return True when the benchmark should run through the API-first stack."""

    return should_use_api_load_path(
        getattr(args, "use_compensation", False),
        getattr(args, "compensation_strategy", None),
    )


def run_benchmark_via_api(
    args: argparse.Namespace,
    tasks: List[str],
    session_factory=NASSession,
    runtime_factory=HuggingFaceRuntime,
) -> Dict:
    """Execute a benchmark through the typed NAS session/runtime API."""

    cache_dir = os.path.join(args.output_dir, "cache") if args.use_cache else None
    model_spec = build_model_spec(args)
    eval_spec = build_eval_spec(args, tasks=tasks, cache_dir=cache_dir)
    plan = build_transformation_plan(args)

    with session_factory() as session:
        artifact = session.load(model_spec)
        if plan is not None:
            artifact = session.apply_plan(artifact, plan)
        runtime = runtime_factory(eval_spec)
        return session.evaluate(artifact, runtime)


def save_results(
    results: Dict,
    output_dir: str,
    model_name: str = "model"
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary from lm_eval
        output_dir: Directory to save results
        model_name: Model name for subdirectory
    
    Returns:
        Path to saved results file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model-specific subdirectory
    model_name_sanitized = model_name.replace("/", "__")
    model_output_dir = os.path.join(output_dir, model_name_sanitized)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().isoformat()
    results_file = os.path.join(model_output_dir, f"results_{timestamp}.json")
    
    # Convert to JSON-serializable format
    serializable_results = make_json_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {results_file}")
    return results_file


def extract_metrics(results: Dict, tasks: List[str]) -> pd.DataFrame:
    """
    Extract metrics from results.
    
    Args:
        results: Results dictionary from lm_eval
        tasks: List of task names
    
    Returns:
        DataFrame with extracted metrics
    """
    if not results or "results" not in results:
        raise ValueError("Invalid results structure")
    
    data = []
    metric_priority = ["acc_norm", "acc", "exact_match", "em", "mc2"]
    
    for task in tasks:
        if task not in results["results"]:
            logger.warning(f"Task '{task}' not found in results")
            continue
        
        task_results = results["results"][task]
        
        # Find primary metric
        metric_value = None
        metric_name = None
        
        for metric in metric_priority:
            # Check for exact match
            if metric in task_results:
                metric_value = task_results[metric]
                metric_name = metric
                break
            # Check flexible-extract variant
            metric_flex = f"{metric},flexible-extract"
            if metric_flex in task_results:
                metric_value = task_results[metric_flex]
                metric_name = metric
                break
            # Check ,none variant (common in lm_eval)
            metric_none = f"{metric},none"
            if metric_none in task_results:
                metric_value = task_results[metric_none]
                metric_name = metric
                break
        
        if metric_value is not None:
            data.append({
                "benchmark": task,
                "metric": metric_name,
                "score": metric_value
            })
    
    if not data:
        raise ValueError("No valid metrics extracted")
    
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        choices=list(BENCHMARK_MAPPING.keys()) + ["all"],
        default=["gsm8k"],
        help="Benchmark(s) to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable request caching"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype"
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log individual samples"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity"
    )
    parser.add_argument(
        "--skip-layers",
        type=int,
        nargs='+',
        default=None,
        help="Layer indices to skip (e.g., --skip-layers 5 6)"
    )
    parser.add_argument(
        "--use-compensation",
        action="store_true",
        help="Use compensation when skipping layers"
    )
    parser.add_argument(
        "--compensation-vector-file",
        type=str,
        default=None,
        help="Path to compensation vector file (.pt)"
    )
    # Advanced compensation strategy arguments
    parser.add_argument(
        "--compensation-strategy",
        type=str,
        default=None,
        help="Advanced compensation strategy name (scaled, last_token, phase_aware, "
             "phase_last_token, pca, position_aware, magnitude_preserving, "
             "magnitude_rescaling, phase_aware_magnitude_rescaling, "
             "multi_phase_aware_magnitude_rescaling, cascaded, multiplicative)"
    )
    parser.add_argument(
        "--compensation-alpha",
        type=float,
        default=1.0,
        help="Scale factor for compensation vector"
    )
    parser.add_argument(
        "--compensation-decode-vector-file",
        type=str,
        default=None,
        help="Path to decode-phase compensation vector (.pt) for phase_aware strategies"
    )
    parser.add_argument(
        "--compensation-pca-file",
        type=str,
        default=None,
        help="Path to PCA components file (.pt) for pca strategy"
    )
    parser.add_argument(
        "--compensation-n-pca-components",
        type=int,
        default=32,
        help="Number of PCA components to use"
    )
    parser.add_argument(
        "--compensation-norm-ratio",
        type=float,
        default=1.0,
        help="Norm ratio (||h_end||/||h_start||) for magnitude_rescaling strategy"
    )
    parser.add_argument(
        "--compensation-pre-skip-fraction",
        type=float,
        default=0.5,
        help="Fraction of delta to apply before skip for cascaded strategy"
    )
    parser.add_argument(
        "--compensation-bucket-file",
        type=str,
        default=None,
        help="Path to position bucket deltas file (.pt) for position_aware strategy"
    )
    parser.add_argument(
        "--compensation-cluster-ratios-file",
        type=str,
        default=None,
        help="Path to JSON with per-cluster norm ratios for multi_phase_aware_magnitude_rescaling"
    )
    parser.add_argument(
        "--learnable-compensation-model",
        type=str,
        default=None,
        help="Path to trained learnable compensation model (.pt)"
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    # Set logging level
    logger.setLevel(args.verbosity)
    
    logger.info("="*60)
    logger.info("Simple Benchmark Runner")
    logger.info("="*60)
    
    # Determine tasks
    if "all" in args.dataset:
        datasets = list(BENCHMARK_MAPPING.keys())
    else:
        datasets = args.dataset

    tasks = resolve_tasks(args.dataset)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Batch Size: {args.batch_size}")
    if args.skip_layers:
        logger.info(f"Skip Layers: {args.skip_layers}")
        if args.use_compensation:
            logger.info(f"Using Compensation: {args.compensation_vector_file}")
    
    # Validate compensation arguments
    if args.use_compensation and not args.skip_layers:
        logger.error("--use-compensation requires --skip-layers to be specified")
        return 1
    # For advanced strategies, vector file may not be required (e.g. position_aware uses bucket file)
    strategy = getattr(args, "compensation_strategy", None)
    if args.use_compensation and not args.compensation_vector_file and strategy is None:
        logger.error("--use-compensation requires --compensation-vector-file (or --compensation-strategy)")
        return 1
    
    try:
        if should_use_api_path(args):
            logger.info("Using API-first NAS evaluation path")
            results = run_benchmark_via_api(args, tasks=tasks)
        else:
            model, tokenizer = load_model(
                model_name=args.model,
                device=args.device,
                skip_layers=args.skip_layers,
                use_compensation=args.use_compensation,
                compensation_vector_file=args.compensation_vector_file,
                dtype=args.dtype,
                compensation_strategy=getattr(args, "compensation_strategy", None),
                compensation_alpha=getattr(args, "compensation_alpha", 1.0),
                compensation_decode_vector_file=getattr(args, "compensation_decode_vector_file", None),
                compensation_pca_file=getattr(args, "compensation_pca_file", None),
                compensation_n_pca_components=getattr(args, "compensation_n_pca_components", 32),
                compensation_norm_ratio=getattr(args, "compensation_norm_ratio", 1.0),
                compensation_pre_skip_fraction=getattr(args, "compensation_pre_skip_fraction", 0.5),
                compensation_bucket_file=getattr(args, "compensation_bucket_file", None),
                compensation_cluster_ratios_file=getattr(args, "compensation_cluster_ratios_file", None),
                learnable_compensation_model=getattr(args, "learnable_compensation_model", None),
            )

            cache_dir = os.path.join(args.output_dir, "cache") if args.use_cache else None
            results = run_lm_eval(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
                use_cache=args.use_cache,
                cache_dir=cache_dir,
                dtype=args.dtype,
                log_samples=args.log_samples,
                random_seed=args.random_seed,
                verbosity=args.verbosity,
            )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    # Save results
    try:
        results_file = save_results(results, args.output_dir, args.model)
        
        # Extract and display metrics
        df = extract_metrics(results, tasks)
        csv_file = os.path.join(args.output_dir, "results.csv")
        df.to_csv(csv_file, index=False)
        
        logger.info(f"\n✅ Results saved to: {csv_file}")
        logger.info("\nResults:")
        logger.info("\n" + df.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("✅ Evaluation complete!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
