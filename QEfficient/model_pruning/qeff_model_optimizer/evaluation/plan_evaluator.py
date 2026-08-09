"""Plan evaluator: apply TransformationPlans and rank by quality."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch

from QEfficient.model_pruning.benchmarking.run_benchmark import resolve_tasks, run_lm_eval
from QEfficient.model_pruning.logging_utils import get_logger
from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import TransformersModelLoader
from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.evaluation.perplexity import PerplexityReport, compute_perplexity
from QEfficient.model_pruning.qeff_model_optimizer.evaluation.videomme import VideoMMEReport, evaluate_videomme
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry

logger = get_logger(__name__)

_ACCURACY_METRIC_PRIORITY = ["acc_norm", "acc", "exact_match", "em", "mc2"]


@dataclass(eq=True)
class PlanResult:
    """Evaluation result for a single plan."""

    plan_name: str
    plan: TransformationPlan
    transform_kinds: list[str]
    skip_layers: list[int]
    perplexity_report: PerplexityReport | None = None
    lm_eval_results: dict[str, Any] | None = None
    videomme_report: VideoMMEReport | None = None
    accuracy_score: float | None = None
    accuracy_metric: str | None = None
    completions: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    eval_time_s: float = 0.0

    @property
    def overall_perplexity(self) -> float:
        if self.perplexity_report is None:
            return float("inf")
        return self.perplexity_report.overall_perplexity

    @property
    def quality_score(self) -> float:
        if self.accuracy_score is not None:
            return self.accuracy_score
        return -self.overall_perplexity

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "transform_kinds": list(self.transform_kinds),
            "skip_layers": list(self.skip_layers),
            "num_layers_skipped": len(self.skip_layers),
            "perplexity_report": self.perplexity_report.to_dict() if self.perplexity_report else None,
            "lm_eval_results": self.lm_eval_results,
            "videomme_report": self.videomme_report.to_dict() if self.videomme_report else None,
            "accuracy_score": self.accuracy_score,
            "accuracy_metric": self.accuracy_metric,
            "completions": dict(self.completions),
            "error": self.error,
            "eval_time_s": round(self.eval_time_s, 2),
            "overall_perplexity": self.overall_perplexity,
        }


class PlanEvaluator:
    """Apply multiple TransformationPlans and rank by perplexity or lm_eval accuracy."""

    DEFAULT_EVAL_PROMPTS = [
        "The capital of France is",
        "Write a Python function to compute fibonacci numbers:",
        "Explain quantum computing in simple terms:",
    ]

    def __init__(
        self,
        model,
        tokenizer,
        model_spec: ModelSpec,
        *,
        datasets: list[str],
        num_samples: int = 50,
        max_length: int = 512,
        eval_prompts: list[str] | None = None,
        generation_len: int = 40,
        eval_method: str = "perplexity",
        accuracy_metric: str = "auto",
        lm_eval_batch_size: int = 1,
        lm_eval_limit: int | None = None,
        videomme_dataset_path: str | None = None,
        videomme_video_root: str | None = None,
        videomme_split: str = "test",
        videomme_num_frames: int = 8,
        videomme_fps: float | None = None,
        videomme_use_subtitles: bool = False,
    ):
        if not datasets:
            raise ValueError("datasets must contain at least one entry")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if eval_method not in {"perplexity", "lm_eval", "videomme"}:
            raise ValueError("eval_method must be 'perplexity', 'lm_eval', or 'videomme'")
        if lm_eval_batch_size <= 0:
            raise ValueError("lm_eval_batch_size must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.model_spec = model_spec
        self.datasets = list(datasets)
        self.num_samples = num_samples
        self.max_length = max_length
        self.eval_prompts = list(eval_prompts or self.DEFAULT_EVAL_PROMPTS)
        self.generation_len = generation_len
        self.eval_method = eval_method
        self.accuracy_metric = accuracy_metric
        self.lm_eval_batch_size = lm_eval_batch_size
        self.lm_eval_limit = lm_eval_limit if lm_eval_limit is not None else num_samples
        self.videomme_dataset_path = videomme_dataset_path
        self.videomme_video_root = videomme_video_root
        self.videomme_split = videomme_split
        self.videomme_num_frames = videomme_num_frames
        self.videomme_fps = videomme_fps
        self.videomme_use_subtitles = videomme_use_subtitles

        self._applier = TransformApplier(default_transform_registry())
        self._loader = TransformersModelLoader()

    def _extract_skip_layers(self, plan: TransformationPlan) -> list[int]:
        for spec in plan.transforms:
            if hasattr(spec, "layers") and spec.layers:
                return list(spec.layers)
        return []

    def _generate_completion(self, prompt: str) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_len,
                do_sample=False,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

    def _device_for_lm_eval(self) -> str:
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            return "cuda" if device.index is None else f"cuda:{device.index}"
        return device.type

    def _extract_accuracy(self, results: dict[str, Any]) -> tuple[float, str]:
        task_results = results.get("results", {})
        values: list[float] = []
        used_metrics: list[str] = []

        for task, metrics in task_results.items():
            metric_name, metric_value = self._select_metric(metrics)
            if metric_name is None:
                logger.warning("[evaluate] no accuracy metric found for lm_eval task=%s", task)
                continue
            values.append(float(metric_value))
            used_metrics.append(metric_name)

        if not values:
            raise ValueError(
                f"No lm_eval accuracy metrics found for requested metric {self.accuracy_metric!r}"
            )
        metric_label = self.accuracy_metric if self.accuracy_metric != "auto" else "+".join(sorted(set(used_metrics)))
        return sum(values) / len(values), metric_label

    def _select_metric(self, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
        candidates = _ACCURACY_METRIC_PRIORITY if self.accuracy_metric == "auto" else [self.accuracy_metric]
        for metric in candidates:
            possible_keys = [metric, f"{metric},none", f"{metric},flexible-extract", f"{metric},strict-match"]
            for key in possible_keys:
                if key in metrics:
                    return key, metrics[key]
        return None, None

    def _evaluate_lm_eval(self) -> tuple[dict[str, Any], float, str]:
        tasks = resolve_tasks(self.datasets)
        logger.info(
            "[evaluate] running lm_eval tasks=%s limit=%s batch=%d metric=%s max_gen_toks=%d",
            tasks,
            self.lm_eval_limit,
            self.lm_eval_batch_size,
            self.accuracy_metric,
            self.generation_len,
        )
        results = run_lm_eval(
            self.model,
            self.tokenizer,
            tasks=tasks,
            batch_size=self.lm_eval_batch_size,
            device=self._device_for_lm_eval(),
            limit=self.lm_eval_limit,
            dtype="auto",
            verbosity="INFO",
            max_gen_toks=self.generation_len,
        )
        accuracy_score, metric_name = self._extract_accuracy(results)
        return results, accuracy_score, metric_name

    def _evaluate_videomme(self) -> tuple[VideoMMEReport, float, str]:
        logger.info(
            "[evaluate] running Video-MME samples=%d frames=%d subtitles=%s",
            self.num_samples,
            self.videomme_num_frames,
            self.videomme_use_subtitles,
        )
        report = evaluate_videomme(
            self.model,
            self.tokenizer,
            dataset_path=self.videomme_dataset_path,
            video_root=self.videomme_video_root,
            split=self.videomme_split,
            num_samples=self.num_samples,
            generation_len=self.generation_len,
            num_frames=self.videomme_num_frames,
            fps=self.videomme_fps,
            use_subtitles=self.videomme_use_subtitles,
        )
        return report, report.overall_accuracy, "videomme_accuracy"

    def evaluate_one(self, plan_name: str, plan: TransformationPlan) -> PlanResult:
        t0 = time.time()
        skip_layers = self._extract_skip_layers(plan)
        transform_kinds = [t.kind for t in plan.transforms]

        session = NASSession(loader=self._loader, transform_applier=self._applier)
        artifact = ModelArtifact(
            artifact_id=uuid4().hex,
            model=self.model,
            tokenizer=self.tokenizer,
            model_spec=self.model_spec,
            plan=TransformationPlan(),
        )
        session.artifacts[artifact.artifact_id] = artifact

        try:
            if plan.transforms:
                session.apply_plan(artifact, plan)

            perplexity_report = None
            lm_eval_results = None
            videomme_report = None
            accuracy_score = None
            accuracy_metric = None

            if self.eval_method == "lm_eval":
                lm_eval_results, accuracy_score, accuracy_metric = self._evaluate_lm_eval()
            elif self.eval_method == "videomme":
                videomme_report, accuracy_score, accuracy_metric = self._evaluate_videomme()
            else:
                perplexity_report = compute_perplexity(
                    self.model,
                    self.tokenizer,
                    datasets=self.datasets,
                    num_samples=self.num_samples,
                    max_length=self.max_length,
                )

            completions = (
                {}
                if self.eval_method == "videomme"
                else {prompt: self._generate_completion(prompt) for prompt in self.eval_prompts}
            )

            return PlanResult(
                plan_name=plan_name,
                plan=plan,
                transform_kinds=transform_kinds,
                skip_layers=skip_layers,
                perplexity_report=perplexity_report,
                lm_eval_results=lm_eval_results,
                videomme_report=videomme_report,
                accuracy_score=accuracy_score,
                accuracy_metric=accuracy_metric,
                completions=completions,
                eval_time_s=time.time() - t0,
            )
        except Exception as e:
            return PlanResult(
                plan_name=plan_name,
                plan=plan,
                transform_kinds=transform_kinds,
                skip_layers=skip_layers,
                error=f"{type(e).__name__}: {e}",
                eval_time_s=time.time() - t0,
            )
        finally:
            session.close()

    def evaluate_all(self, plans: dict[str, TransformationPlan]) -> list[PlanResult]:
        results = [self.evaluate_one(plan_name, plan) for plan_name, plan in plans.items()]
        if self.eval_method in {"lm_eval", "videomme"}:
            results.sort(key=lambda r: (r.accuracy_score is not None, r.accuracy_score or float("-inf")), reverse=True)
        else:
            results.sort(key=lambda r: r.overall_perplexity)
        return results

    def select_best(
        self,
        results: list[PlanResult],
        *,
        accuracy_threshold: float = 10.0,
        baseline_name: str = "baseline",
    ) -> PlanResult | None:
        """Select the most aggressive non-baseline plan within the quality budget.

        This mirrors the NAS selection policy: among plans that do not exceed the
        allowed regression from baseline, prefer the lowest-quality acceptable
        result as a proxy for maximum optimization. For ``lm_eval`` this means
        the lowest accuracy still above threshold; for perplexity it means the
        highest perplexity still below threshold.
        """
        baseline = next((r for r in results if r.plan_name == baseline_name and r.error is None), None)
        if baseline is None:
            return None

        if self.eval_method in {"lm_eval", "videomme"}:
            if baseline.accuracy_score is None:
                return None
            threshold = baseline.accuracy_score * (1.0 - accuracy_threshold / 100.0)
            for result in reversed(results):
                if result.plan_name == baseline_name or result.error or result.accuracy_score is None:
                    continue
                if result.accuracy_score >= threshold:
                    return result
            return None

        threshold = baseline.overall_perplexity * (1.0 + accuracy_threshold / 100.0)
        for result in reversed(results):
            if result.plan_name == baseline_name or result.error:
                continue
            if result.overall_perplexity <= threshold:
                return result
        return None


__all__ = ["PlanEvaluator", "PlanResult"]
