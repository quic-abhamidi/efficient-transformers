# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run lm-evaluation-harness tasks against one or more QEfficient language QPCs.

This bridge is intentionally generation-first: it implements the lm_eval
``generate_until`` path by calling QEfficient's QPC text generation runtime.
Tasks that require token loglikelihood will raise a clear error until a
logits-backed QPC adapter is added.

Example:
    python scripts/eval/lm_eval_qpc_accuracy.py \
        --tokenizer-name Qwen/Qwen3-VL-30B-A3B \
        --variant baseline=/home/tmp/qpc_baseline/qpc-abc/qpc \
        --variant skip_32_36=/home/tmp/qpc_skip/qpc-def/qpc \
        --variant skip_32_36_linear_patch=/home/tmp/qpc_patch/qpc-ghi/qpc \
        --baseline-variant baseline \
        --skip-variant skip_32_36 \
        --tasks gsm8k \
        --limit 50 \
        --generation-len 256 \
        --output-json /home/tmp/qwen3vl_lm_eval.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
from transformers import AutoTokenizer


@dataclass
class VariantEvalResult:
    name: str
    qpc_path: str
    raw_results: dict[str, Any]
    flattened_metrics: dict[str, float]
    deltas_vs_baseline: dict[str, float]
    recovery_vs_skip: dict[str, float]


def _parse_variant(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--variant must be in the form name=/path/to/qpc")
    name, qpc_path = value.split("=", 1)
    name = name.strip()
    qpc_path = qpc_path.strip()
    if not name:
        raise argparse.ArgumentTypeError("variant name cannot be empty")
    if not qpc_path:
        raise argparse.ArgumentTypeError("variant QPC path cannot be empty")
    return name, qpc_path


def _parse_device_ids(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.strip("[]").split(",") if part.strip()]


def _flatten_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            texts.extend(_flatten_texts(item))
        return texts
    return [str(value)]


def _request_args(request: Any) -> tuple[str, dict[str, Any]]:
    args = request.args if hasattr(request, "args") else request
    if not isinstance(args, (list, tuple)) or len(args) != 2:
        raise TypeError(f"expected lm_eval generate request args as (context, gen_kwargs), got {type(args)}")
    context, gen_kwargs = args
    if gen_kwargs is None:
        gen_kwargs = {}
    if not isinstance(context, str):
        context = str(context)
    if not isinstance(gen_kwargs, dict):
        raise TypeError(f"expected generation kwargs to be a dict, got {type(gen_kwargs)}")
    return context, gen_kwargs


def _until_sequences(gen_kwargs: dict[str, Any]) -> list[str]:
    until = gen_kwargs.get("until", [])
    if until is None:
        return []
    if isinstance(until, str):
        return [until]
    return [str(item) for item in until]


def _truncate_at_until(text: str, until: list[str]) -> str:
    cut = len(text)
    for stop in until:
        if not stop:
            continue
        position = text.find(stop)
        if position >= 0:
            cut = min(cut, position)
    return text[:cut]


def _strip_prompt_echo(generated_text: str, prompt: str) -> str:
    return generated_text.removeprefix(prompt)


def _unsupported_loglikelihood(method_name: str) -> NotImplementedError:
    return NotImplementedError(
        f"{method_name} is not implemented for QEfficient QPC lm_eval bridge. "
        "Use generate_until tasks, or add a logits-backed QPC adapter before running "
        "multiple-choice/loglikelihood tasks."
    )


def _load_qeff_runtime() -> tuple[Any, Any]:
    from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv, get_compilation_dims

    return cloud_ai_100_exec_kv, get_compilation_dims


def _build_lm_class(base_lm_class: type[Any]) -> type[Any]:
    class QEffQPCGenerateUntilLM(base_lm_class):
        def __init__(
            self,
            *,
            tokenizer: Any,
            qpc_path: str,
            device_ids: list[int] | None,
            generation_len: int,
        ) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.qpc_path = qpc_path
            self.device_ids = device_ids
            self.generation_len = generation_len
            self._cloud_ai_100_exec_kv, get_compilation_dims = _load_qeff_runtime()
            self._batch_size, self._ctx_len, _ = get_compilation_dims(qpc_path)

        @property
        def eot_token_id(self) -> int | None:
            return self.tokenizer.eos_token_id

        @property
        def max_length(self) -> int:
            return self._ctx_len

        @property
        def max_gen_toks(self) -> int:
            return self.generation_len

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self) -> str:
            return "qaic"

        def tok_encode(
            self,
            string: str,
            left_truncate_len: int | None = None,
            add_special_tokens: bool | None = None,
        ) -> list[int]:
            token_ids = self.tokenizer.encode(
                string,
                add_special_tokens=False if add_special_tokens is None else add_special_tokens,
            )
            if left_truncate_len is not None:
                token_ids = token_ids[-left_truncate_len:]
            return token_ids

        def tok_decode(self, tokens: list[int]) -> str:
            return self.tokenizer.decode(tokens)

        def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
            del disable_tqdm
            responses = []
            for request in requests:
                context, gen_kwargs = _request_args(request)
                generation_len = int(
                    gen_kwargs.get("max_gen_toks")
                    or gen_kwargs.get("max_new_tokens")
                    or gen_kwargs.get("max_length")
                    or self.generation_len
                )
                exec_info = self._cloud_ai_100_exec_kv(
                    tokenizer=self.tokenizer,
                    qpc_path=self.qpc_path,
                    prompt=[context],
                    device_id=self.device_ids,
                    generation_len=generation_len,
                    stream=False,
                    automation=True,
                )
                generated_texts = _flatten_texts(exec_info.generated_texts)
                generated_text = generated_texts[0] if generated_texts else ""
                generated_text = _strip_prompt_echo(generated_text, context)
                responses.append(_truncate_at_until(generated_text, _until_sequences(gen_kwargs)))
            return responses

        def greedy_until(self, requests: list[Any]) -> list[str]:
            return self.generate_until(requests)

        def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
            raise _unsupported_loglikelihood("loglikelihood")

        def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
            raise _unsupported_loglikelihood("loglikelihood_rolling")

    return QEffQPCGenerateUntilLM


def _load_lm_eval() -> tuple[Any, type[Any]]:
    try:
        from lm_eval import evaluator
        from lm_eval.api.model import LM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lm_eval is not installed in this environment. Install EleutherAI lm-evaluation-harness "
            "in the active virtualenv, for example: pip install lm-eval"
        ) from exc
    return evaluator, LM


def _simple_evaluate(evaluator: Any, lm: Any, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "model": lm,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "bootstrap_iters": args.bootstrap_iters,
        "log_samples": args.log_samples,
    }
    if args.gen_kwargs is not None:
        kwargs["gen_kwargs"] = args.gen_kwargs

    signature = inspect.signature(evaluator.simple_evaluate)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return evaluator.simple_evaluate(**supported_kwargs)


def _flatten_numeric_metrics(results: dict[str, Any]) -> dict[str, float]:
    flattened = {}
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                flattened[f"{task_name}.{metric_name}"] = float(value)
    return flattened


def _compute_deltas(metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    return {key: value - baseline_metrics[key] for key, value in metrics.items() if key in baseline_metrics}


def _compute_recovery(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    skip_metrics: dict[str, float],
) -> dict[str, float]:
    recovery = {}
    for key, value in metrics.items():
        if key not in baseline_metrics or key not in skip_metrics:
            continue
        denominator = baseline_metrics[key] - skip_metrics[key]
        if abs(denominator) < 1e-12:
            continue
        recovery[key] = (value - skip_metrics[key]) / denominator
    return recovery


def _write_json(path: str | None, results: list[VariantEvalResult]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
    print(f"Wrote lm_eval comparison results to {output_path}")


def _print_summary(results: list[VariantEvalResult]) -> None:
    print("\n========================= lm_eval QPC Accuracy Summary =========================")
    for result in results:
        print(f"\nVariant: {result.name}")
        print(f"  QPC: {result.qpc_path}")
        if not result.flattened_metrics:
            print("  No numeric lm_eval metrics found.")
            continue
        for key, value in sorted(result.flattened_metrics.items()):
            delta = result.deltas_vs_baseline.get(key)
            suffix = f" (delta {delta:+.4f})" if delta is not None else ""
            print(f"  {key}: {value:.4f}{suffix}")
        if result.recovery_vs_skip:
            compact = ", ".join(f"{key}={value:.4f}" for key, value in sorted(result.recovery_vs_skip.items()))
            print(f"  Recovery vs skip: {compact}")
    print("===============================================================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm_eval generate_until tasks against baseline/skipped/compensated QEfficient QPCs."
    )
    parser.add_argument("--tokenizer-name", required=True, help="Tokenizer name or local tokenizer path.")
    parser.add_argument(
        "--variant",
        action="append",
        type=_parse_variant,
        required=True,
        help="Variant in name=/path/to/qpc form. Pass multiple times.",
    )
    parser.add_argument(
        "--baseline-variant",
        default=None,
        help="Variant name to treat as baseline. Defaults to the first --variant.",
    )
    parser.add_argument(
        "--skip-variant",
        default=None,
        help="Uncompensated skipped-layer variant used to compute recovery ratios.",
    )
    parser.add_argument("--tasks", required=True, help="Comma-separated lm_eval task names, for example gsm8k.")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples passed to lm_eval.")
    parser.add_argument("--limit", default=None, help="lm_eval limit. Use a small value for smoke tests.")
    parser.add_argument("--bootstrap-iters", type=int, default=100000, help="lm_eval bootstrap iterations.")
    parser.add_argument("--generation-len", type=int, default=256, help="Default max generation tokens.")
    parser.add_argument("--gen-kwargs", default=None, help="Optional lm_eval generation kwargs string.")
    parser.add_argument(
        "--device-id",
        default=None,
        help="QAIC device IDs, for example '0' or '[0,1]'. Defaults to runtime auto-device selection.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--log-samples", action="store_true", help="Ask lm_eval to include per-sample logs.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None:
        try:
            args.limit = int(args.limit)
        except ValueError:
            args.limit = float(args.limit)

    variants = dict(args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("variant names must be unique")

    baseline_name = args.baseline_variant or args.variant[0][0]
    if baseline_name not in variants:
        raise ValueError(f"baseline variant {baseline_name!r} was not provided")
    if args.skip_variant is not None and args.skip_variant not in variants:
        raise ValueError(f"skip variant {args.skip_variant!r} was not provided")

    try:
        evaluator, base_lm_class = _load_lm_eval()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    qeff_lm_class = _build_lm_class(base_lm_class)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_ids = _parse_device_ids(args.device_id)
    raw_metrics = {}
    raw_results = {}
    for name, qpc_path in args.variant:
        print(f"\nRunning lm_eval for variant {name}: {qpc_path}")
        lm = qeff_lm_class(
            tokenizer=tokenizer,
            qpc_path=qpc_path,
            device_ids=device_ids,
            generation_len=args.generation_len,
        )
        result = _simple_evaluate(evaluator, lm, args)
        raw_results[name] = result
        raw_metrics[name] = _flatten_numeric_metrics(result)

    baseline_metrics = raw_metrics[baseline_name]
    skip_metrics = raw_metrics.get(args.skip_variant, {}) if args.skip_variant is not None else {}

    results = []
    for name, qpc_path in args.variant:
        metrics = raw_metrics[name]
        results.append(
            VariantEvalResult(
                name=name,
                qpc_path=qpc_path,
                raw_results=raw_results[name],
                flattened_metrics=metrics,
                deltas_vs_baseline=_compute_deltas(metrics, baseline_metrics),
                recovery_vs_skip=_compute_recovery(metrics, baseline_metrics, skip_metrics) if skip_metrics else {},
            )
        )

    _print_summary(results)
    _write_json(args.output_json, results)


if __name__ == "__main__":
    main()
