"""Hugging Face runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from QEfficient.model_pruning.qeff_model_optimizer.runtimes.base import BaseRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.eval import EvalSpec


def _default_hf_evaluator(artifact: ModelArtifact, eval_spec: EvalSpec) -> Any:
    from QEfficient.model_pruning.benchmarking.run_benchmark import run_lm_eval

    return run_lm_eval(
        model=artifact.model,
        tokenizer=artifact.tokenizer,
        tasks=eval_spec.tasks,
        batch_size=eval_spec.batch_size,
        device=eval_spec.device,
        limit=eval_spec.limit,
        num_fewshot=eval_spec.num_fewshot,
        use_cache=eval_spec.use_cache,
        cache_dir=eval_spec.cache_dir,
        dtype=eval_spec.dtype,
        log_samples=eval_spec.log_samples,
        random_seed=eval_spec.random_seed,
        verbosity=eval_spec.verbosity,
    )


@dataclass
class HuggingFaceRuntime(BaseRuntime):
    """Runtime wrapper for HF model-object evaluation."""

    eval_spec: EvalSpec
    evaluator: Callable[[ModelArtifact, EvalSpec], Any] | None = None
    name: str = "hf"

    def evaluate(self, artifact: ModelArtifact):
        """Run lm-eval tasks against *artifact* and return the raw results dict.

        Delegates to ``benchmarking.run_benchmark.run_lm_eval``, which wraps the
        artifact's model in an ``HFLM`` object understood by the lm-evaluation-harness.
        Results are keyed by task name with per-metric nested dicts.
        """
        evaluator = self.evaluator or _default_hf_evaluator
        return evaluator(artifact, self.eval_spec)
