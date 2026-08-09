"""QAIC benchmark runner — compile a plan to QPC, run, and measure perf.

Wraps the full compile-and-run loop for Qualcomm AI Cloud (QAIC) hardware:
load model on CPU → apply plan → QEff prepare → compile to QPC → run prompts →
parse performance stats (TTFT, decode tok/s, E2E latency).

Typical use:

.. code-block:: python

    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import QAICBenchmarkRunner

    runner = QAICBenchmarkRunner(
        model_id="Qwen/Qwen3-14B",
        prompts=["The capital of France is"],
        generation_len=60,
    )

    # Baseline (no transforms)
    baseline = runner.run(
        name="baseline",
        plan=TransformationPlan(),
        device_group=[0], batch_size=1,
    )

    # Optimized (skip 3 weak layers)
    optimized = runner.run(
        name="skip_3_spread",
        plan=TransformationPlan(transforms=[SkipLayersSpec(layers=[3,23,34])]),
        device_group=[0,1,2,3], batch_size=8,
    )

    print(f"Speedup: {baseline.avg_stats['decode_tps']} -> {optimized.avg_stats['decode_tps']}")

Design notes:
- Each call to ``run()`` loads a fresh QEfficient CPU float32 model. Required
  because QEff tracing mutates the model and can't be rolled back cleanly.
- QEff cache is scoped under each compile directory to force fresh ONNX export per plan.
- Multi-device compilation requires the compile_dir to exist before invocation;
  the runner ensures this.
- Performance stats are extracted from QEff's stdout via regex; the underlying
  API does not return these programmatically.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import io
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from QEfficient.model_pruning.qeff_model_optimizer.api.session import NASSession
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
from QEfficient.model_pruning.qeff_model_optimizer.evaluation.videomme import (
    build_videomme_inputs,
    extract_choice,
    load_videomme_examples,
    normalize_answer,
)
from QEfficient.model_pruning.qeff_model_optimizer.transforms.applier import TransformApplier, default_transform_registry


# --- Result dataclasses ---

@dataclass(eq=True)
class QAICRunResult:
    """Result from a single QAIC compile+run cycle.

    ``avg_stats`` holds the averaged performance metrics across all prompts.
    ``per_prompt_stats`` has the raw stats for each prompt individually.
    ``error`` is populated if compilation failed; check it before using
    ``avg_stats`` (which will be empty or zero).
    """
    plan_name: str
    batch_size: int
    num_devices: int
    skip_layers: list[int]
    compile_time_s: float = 0.0
    avg_stats: dict[str, float] = field(default_factory=dict)
    per_prompt_stats: list[dict[str, float]] = field(default_factory=list)
    completions: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    qpc_path: str | None = None
    accuracy_score: float | None = None
    accuracy_metric: str | None = None
    videomme_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "batch_size": self.batch_size,
            "num_devices": self.num_devices,
            "skip_layers": list(self.skip_layers),
            "compile_time_s": self.compile_time_s,
            "avg_stats": dict(self.avg_stats),
            "per_prompt_stats": list(self.per_prompt_stats),
            "completions": list(self.completions),
            "error": self.error,
            "qpc_path": self.qpc_path,
            "accuracy_score": self.accuracy_score,
            "accuracy_metric": self.accuracy_metric,
            "videomme_report": self.videomme_report,
        }


# --- Performance parsing ---

# QEff writes performance lines like "TTFT is= 0.190 sec" to stdout.
# No programmatic API exposes these, so we capture stdout and regex them.
_PERF_PATTERNS = {
    "ttft": r"TTFT is=\s*([\d.]+)\s*sec",
    "decode_tps": r"Decode is=\s*([\d.]+)\s*tokens/sec",
    "total_tps": r"Total is=\s*([\d.]+)\s*tokens/sec",
    "e2e": r"Total \(E2E\).*?is=\s*([\d.]+)\s*sec",
}


def parse_qeff_perf(output: str) -> dict[str, float]:
    """Extract perf stats from QEff ``generate()`` stdout.

    Returns a dict with any of ttft, decode_tps, total_tps, e2e that were found.
    Returns empty dict if none matched (e.g. output captured early or format changed).
    """
    stats: dict[str, float] = {}
    for key, pattern in _PERF_PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            stats[key] = float(m.group(1))
    return stats


def _extract_completion(output: str) -> str:
    """Pull the generated completion text out of QEff stdout."""
    m = re.search(r"Completion\s*:\s*(.+?)\n=", output, re.DOTALL)
    return m.group(1).strip()[:200] if m else ""


def _decode_generated(processor, generated: Any) -> str:
    if generated is None:
        return ""
    if isinstance(generated, dict) and "generated_text" in generated:
        return str(generated["generated_text"])[:200]
    if hasattr(processor, "batch_decode"):
        try:
            return str(processor.batch_decode(generated, skip_special_tokens=True)[0]).strip()[:200]
        except Exception:
            return ""
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "batch_decode"):
        try:
            return str(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]).strip()[:200]
        except Exception:
            return ""
    return ""


def _per_duration_accuracy(results: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for result in results:
        duration = str(result.get("duration") or "unknown")
        totals[duration] = totals.get(duration, 0) + 1
        correct[duration] = correct.get(duration, 0) + int(bool(result.get("correct")))
    return {duration: correct.get(duration, 0) / total for duration, total in totals.items()}


# --- QAIC benchmark runner ---

class QAICBenchmarkRunner:
    """Compile and benchmark transformation plans on QAIC hardware.

    Each ``run()`` call is a full cycle: load a fresh model on CPU (float32),
    apply the plan via NASSession, hand to QEff, compile to QPC, execute prompts,
    parse performance stats.

    Parameters
    ----------
    model_id:
        HuggingFace repo id (e.g. "Qwen/Qwen3-14B") or local path.
    prompts:
        Prompts to run for performance measurement. Results are averaged across
        prompts. Keep short (3-5) since each prompt adds decode time.
    generation_len:
        Tokens to generate per prompt.
    ctx_len:
        KV cache context length (total prefill + decode).
    prefill_seq_len:
        Max prefill sequence length in a single step.
    num_cores:
        QAIC cores per device.
    compile_dir_base:
        Root directory where each run's QPC is compiled. Each run gets a
        subdirectory named after the plan.
    mxfp6_matmul:
        Enable 6-bit mixed precision for matmul weights.
    mxint8_kv_cache:
        Enable 8-bit integer KV cache.
    """

    DEFAULT_PROMPTS = [
        "The capital of France is",
        "Write a Python function to compute fibonacci numbers:",
        "Explain quantum computing in simple terms:",
    ]

    def __init__(
        self,
        model_id: str,
        *,
        prompts: list[str] | None = None,
        generation_len: int = 60,
        ctx_len: int = 4096,
        prefill_seq_len: int = 128,
        num_cores: int = 16,
        compile_dir_base: str | None = None,
        mxfp6_matmul: bool = True,
        mxint8_kv_cache: bool = True,
        videomme_dataset_path: str | None = None,
        videomme_video_root: str | None = None,
        videomme_split: str = "test",
        videomme_num_samples: int | None = None,
        videomme_num_frames: int = 8,
        videomme_fps: float | None = None,
        videomme_use_subtitles: bool = False,
    ):
        self.model_id = model_id
        self.prompts = list(prompts or self.DEFAULT_PROMPTS)
        self.generation_len = generation_len
        self.ctx_len = ctx_len
        self.prefill_seq_len = prefill_seq_len
        self.num_cores = num_cores
        self.compile_dir_base = compile_dir_base or str(Path.cwd() / "results" / "model_pruning" / "qaic_compile")
        self.mxfp6_matmul = mxfp6_matmul
        self.mxint8_kv_cache = mxint8_kv_cache
        self.videomme_dataset_path = videomme_dataset_path
        self.videomme_video_root = videomme_video_root
        self.videomme_split = videomme_split
        self.videomme_num_samples = videomme_num_samples
        self.videomme_num_frames = videomme_num_frames
        self.videomme_fps = videomme_fps
        self.videomme_use_subtitles = videomme_use_subtitles

    @property
    def _uses_videomme(self) -> bool:
        return bool(self.videomme_dataset_path)

    def _prepare_compile_dir(self, plan_name: str) -> str:
        """Create a fresh compile_dir. Multi-device compile fails without it."""
        compile_dir = str(Path(self.compile_dir_base) / plan_name)
        if Path(compile_dir).exists():
            shutil.rmtree(compile_dir)
        os.makedirs(compile_dir, exist_ok=True)
        return compile_dir

    @staticmethod
    def _clear_qeff_cache(cache_dir: Path) -> None:
        """Force fresh ONNX export by clearing this run's scoped QEff cache."""
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    @staticmethod
    @contextlib.contextmanager
    def _qeff_cache_scope(cache_dir: Path):
        """Point QEff cache lookups at a per-run directory.

        QEff resolves cache paths from environment variables and module-level
        constants. Patch both forms so this benchmark never deletes or reuses
        the user's global ``~/.cache/qeff_models`` contents.
        """
        old_env = os.environ.get("QEFF_HOME")
        cache_dir = cache_dir.resolve()
        models_dir = cache_dir / "qeff_models"
        patched: list[tuple[Any, str, Any]] = []

        def patch_attr(module: Any, attr: str, value: Any) -> None:
            if hasattr(module, attr):
                patched.append((module, attr, getattr(module, attr)))
                setattr(module, attr, value)

        os.environ["QEFF_HOME"] = str(cache_dir)
        for module_name in ("QEfficient.utils.cache", "QEfficient.utils.constants"):
            try:
                importlib.import_module(module_name)
            except Exception:
                pass

        for module in list(sys.modules.values()):
            if module is None:
                continue
            patch_attr(module, "QEFF_HOME", cache_dir)
            patch_attr(module, "QEFF_MODELS_DIR", str(models_dir))

        try:
            yield
        finally:
            for module, attr, value in reversed(patched):
                setattr(module, attr, value)
            if old_env is None:
                os.environ.pop("QEFF_HOME", None)
            else:
                os.environ["QEFF_HOME"] = old_env

    def _load_fresh_qeff_model(self):
        """Load a fresh QEfficient CPU float32 model and its processor/tokenizer."""
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
        from QEfficient.model_pruning.qeff_model_optimizer.api.loaders import _looks_like_vlm

        spec = ModelSpec(model_id=self.model_id, dtype="float32", device_map="cpu")
        common_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "trust_remote_code": True,
        }
        if _looks_like_vlm(self.model_id):
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                self.model_id,
                kv_offload=False,
                **common_kwargs,
            )
            tokenizer = processor
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                self.model_id,
                max_seq_len_cached=self.ctx_len,
                **common_kwargs,
            )
        return qeff_model, tokenizer, spec

    def _aggregate_stats(
        self, per_prompt: list[dict[str, float]]
    ) -> dict[str, float]:
        """Average per-prompt stats across runs. Missing keys become 0."""
        result: dict[str, float] = {}
        for key in _PERF_PATTERNS:
            values = [s.get(key, 0.0) for s in per_prompt if key in s]
            result[key] = sum(values) / len(values) if values else 0.0
        return result

    def run(
        self,
        name: str,
        plan: TransformationPlan,
        *,
        device_group: list[int],
        batch_size: int,
    ) -> QAICRunResult:
        """Compile and run a single plan configuration.

        Parameters
        ----------
        name:
            Unique label for this run (used in compile_dir and logs).
        plan:
            TransformationPlan to apply before compilation.
        device_group:
            List of QAIC device ids to use (length determines num_devices).
        batch_size:
            Inference batch size. Must match compile batch_size.

        Returns
        -------
        QAICRunResult
            Always returned; check ``.error`` to detect compilation failure.
        """
        # Extract skip layers for reporting.
        skip_layers: list[int] = []
        for spec in plan.transforms:
            if hasattr(spec, "layers") and spec.layers:
                skip_layers = list(spec.layers)
                break

        num_devices = len(device_group)
        compile_dir = self._prepare_compile_dir(name)
        cache_dir = Path(compile_dir) / ".qeff_cache"
        self._clear_qeff_cache(cache_dir)

        with self._qeff_cache_scope(cache_dir):
            # Load QEfficient model fresh and apply plan to its underlying model.
            try:
                qeff_model, tokenizer, model_spec = self._load_fresh_qeff_model()
                model = qeff_model.model
            except Exception as e:
                gc.collect()
                return QAICRunResult(
                    plan_name=name, batch_size=batch_size, num_devices=num_devices,
                    skip_layers=skip_layers,
                    error=f"QEfficient model load failed: {type(e).__name__}: {str(e)[:300]}",
                )

            applier = TransformApplier(default_transform_registry())
            session = NASSession(loader=None, transform_applier=applier)
            artifact = ModelArtifact(
                artifact_id=uuid4().hex,
                model=model, tokenizer=tokenizer,
                model_spec=model_spec, plan=TransformationPlan(),
            )
            session.artifacts[artifact.artifact_id] = artifact

            if plan.transforms:
                try:
                    session.apply_plan(artifact, plan)
                except Exception as e:
                    session.close()
                    del qeff_model, model; gc.collect()
                    return QAICRunResult(
                        plan_name=name, batch_size=batch_size, num_devices=num_devices,
                        skip_layers=skip_layers,
                        error=f"Plan apply failed: {type(e).__name__}: {e}",
                    )

            # Compile the already-loaded QEfficient model to QPC.
            prepared = qeff_model
            t0 = time.time()
            try:
                qpc_path = prepared.compile(
                    prefill_seq_len=self.prefill_seq_len,
                    ctx_len=self.ctx_len,
                    batch_size=batch_size,
                    num_devices=num_devices,
                    num_cores=self.num_cores,
                    mxfp6_matmul=self.mxfp6_matmul,
                    mxint8_kv_cache=self.mxint8_kv_cache,
                    compile_dir=compile_dir,
                )
                compile_time = time.time() - t0
            except Exception as e:
                session.close()
                del qeff_model, model; gc.collect()
                return QAICRunResult(
                    plan_name=name, batch_size=batch_size, num_devices=num_devices,
                    skip_layers=skip_layers,
                    compile_time_s=time.time() - t0,
                    error=f"Compile failed: {str(e)[:300]}",
                )

            # Run through the compiled model, capturing stdout so we can parse
            # performance metrics and, for Video-MME, multiple-choice accuracy.
            try:
                if self._uses_videomme:
                    per_prompt, completions, videomme_report = self._run_videomme(prepared, tokenizer)
                    accuracy_score = videomme_report["overall_accuracy"]
                    accuracy_metric = "videomme_accuracy"
                else:
                    per_prompt, completions = self._run_text_prompts(prepared, tokenizer)
                    videomme_report = None
                    accuracy_score = None
                    accuracy_metric = None
            except Exception as e:
                session.close()
                del qeff_model, model; gc.collect()
                return QAICRunResult(
                    plan_name=name, batch_size=batch_size, num_devices=num_devices,
                    skip_layers=skip_layers,
                    compile_time_s=round(compile_time, 2),
                    qpc_path=str(qpc_path),
                    error=f"Generation failed: {type(e).__name__}: {str(e)[:300]}",
                )

            session.close()
            del qeff_model, model; gc.collect()

        return QAICRunResult(
            plan_name=name,
            batch_size=batch_size,
            num_devices=num_devices,
            skip_layers=skip_layers,
            compile_time_s=round(compile_time, 2),
            avg_stats=self._aggregate_stats(per_prompt),
            per_prompt_stats=per_prompt,
            completions=completions,
            qpc_path=str(qpc_path),
            accuracy_score=accuracy_score,
            accuracy_metric=accuracy_metric,
            videomme_report=videomme_report,
        )

    def _run_text_prompts(self, prepared, tokenizer) -> tuple[list[dict[str, float]], list[dict[str, str]]]:
        per_prompt: list[dict[str, float]] = []
        completions: list[dict[str, str]] = []
        for prompt in self.prompts:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                prepared.generate(
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    generation_len=self.generation_len,
                )
            output = captured.getvalue()
            per_prompt.append(parse_qeff_perf(output))
            completions.append({
                "prompt": prompt,
                "completion": _extract_completion(output),
            })
        return per_prompt, completions

    def _run_videomme(self, prepared, processor) -> tuple[list[dict[str, float]], list[dict[str, str]], dict[str, Any]]:
        examples = load_videomme_examples(
            dataset_path=self.videomme_dataset_path,
            video_root=self.videomme_video_root,
            split=self.videomme_split,
            num_samples=self.videomme_num_samples,
            use_subtitles=self.videomme_use_subtitles,
        )
        per_prompt: list[dict[str, float]] = []
        completions: list[dict[str, str]] = []
        sample_results: list[dict[str, Any]] = []
        for example in examples:
            prompt = example.prompt(use_subtitles=self.videomme_use_subtitles)
            inputs = build_videomme_inputs(
                processor,
                example,
                prompt=prompt,
                num_frames=self.videomme_num_frames,
                fps=self.videomme_fps,
            )
            captured = io.StringIO()
            generated = None
            with contextlib.redirect_stdout(captured):
                try:
                    generated = prepared.generate(inputs=inputs, generation_len=self.generation_len)
                except TypeError:
                    generated = prepared.generate(processor=processor, inputs=inputs, generation_len=self.generation_len)
            output = captured.getvalue()
            completion = _extract_completion(output) or _decode_generated(processor, generated) or output.strip()[:200]
            prediction = extract_choice(completion or output)
            answer = normalize_answer(example.answer)
            correct = prediction == answer
            per_prompt.append(parse_qeff_perf(output))
            completions.append({
                "prompt": prompt,
                "completion": completion,
            })
            sample_results.append({
                "sample_id": example.sample_id,
                "video_id": example.video_id,
                "duration": example.duration,
                "answer": answer,
                "prediction": prediction,
                "correct": correct,
                "output_text": completion,
            })
        correct_count = sum(1 for item in sample_results if item["correct"])
        return per_prompt, completions, {
            "dataset_path": self.videomme_dataset_path,
            "video_root": self.videomme_video_root,
            "use_subtitles": self.videomme_use_subtitles,
            "num_samples": len(sample_results),
            "correct": correct_count,
            "overall_accuracy": correct_count / len(sample_results) if sample_results else 0.0,
            "per_duration_accuracy": _per_duration_accuracy(sample_results),
            "with_subtitle_accuracy": (correct_count / len(sample_results) if sample_results and self.videomme_use_subtitles else None),
            "without_subtitle_accuracy": (correct_count / len(sample_results) if sample_results and not self.videomme_use_subtitles else None),
            "results": sample_results,
        }

    def compute_speedups(
        self,
        results: list[QAICRunResult],
        *,
        baseline_name: str = "baseline",
    ) -> dict[str, dict[str, float]]:
        """Compute percentage speedups of each result vs a baseline result.

        Returns ``{plan_name: {ttft_pct, decode_pct, e2e_pct}}``. Positive
        numbers mean 'better' (lower TTFT, higher decode, lower E2E).
        """
        baseline = next(
            (r for r in results if r.plan_name == baseline_name and not r.error),
            None,
        )
        if baseline is None:
            return {}

        bs = baseline.avg_stats
        speedups: dict[str, dict[str, float]] = {}
        for r in results:
            if r.plan_name == baseline_name or r.error:
                continue
            s = r.avg_stats
            # Guard against zero division; baseline stats should always be populated
            # for a successful run, but defensive anyway.
            speedups[r.plan_name] = {
                "ttft_pct": (
                    (bs.get("ttft", 0) - s.get("ttft", 0))
                    / max(bs.get("ttft", 1e-6), 1e-6) * 100
                ),
                "decode_pct": (
                    (s.get("decode_tps", 0) - bs.get("decode_tps", 0))
                    / max(bs.get("decode_tps", 1e-6), 1e-6) * 100
                ),
                "e2e_pct": (
                    (bs.get("e2e", 0) - s.get("e2e", 0))
                    / max(bs.get("e2e", 1e-6), 1e-6) * 100
                ),
            }
        return speedups


__all__ = [
    "QAICBenchmarkRunner",
    "QAICRunResult",
    "parse_qeff_perf",
]
