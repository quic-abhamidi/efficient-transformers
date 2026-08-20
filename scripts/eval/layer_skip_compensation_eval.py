# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compare layer-skipped QPC variants against a baseline and optional references.

This script is intended for quick quantitative checks after layer skipping or
accuracy compensation. It runs already compiled language QPCs on the same prompt
set and reports:

* task-style scores against optional references
* generated-text similarity to the baseline QPC
* generated-token similarity to the baseline QPC
* throughput and latency deltas

Example:
    python scripts/eval/layer_skip_compensation_eval.py \
        --tokenizer-name Qwen/Qwen3-VL-30B-A3B \
        --variant baseline=/home/tmp/qpc_baseline/qpc-abc/qpc \
        --variant skip_32_36=/home/tmp/qpc_skip/qpc-def/qpc \
        --variant skip_32_36_linear_patch=/home/tmp/qpc_patch/qpc-ghi/qpc \
        --baseline-variant baseline \
        --skip-variant skip_32_36 \
        --prompt "Hello!" \
        --generation-len 128 \
        --output-json /home/tmp/layer_skip_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@dataclass
class PromptRecord:
    prompt: str
    reference: str | None = None


@dataclass
class VariantResult:
    name: str
    qpc_path: str
    generated_texts: list[str]
    generated_ids: list[list[int]]
    perf: dict[str, float]
    reference_metrics: dict[str, float]
    baseline_text_metrics: dict[str, float]
    baseline_token_metrics: dict[str, float]
    deltas_vs_baseline: dict[str, float]
    recovery_vs_skip: dict[str, float]


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", _normalize_text(text))


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _word_tokens(prediction)
    ref_tokens = _word_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _sequence_similarity(left: list[Any], right: list[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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


def _load_json_records(path: Path) -> list[PromptRecord]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return [_record_from_mapping(item, path) for item in data]


def _load_jsonl_records(path: Path) -> list[PromptRecord]:
    records = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_mapping(json.loads(line), path, line_no))
    return records


def _record_from_mapping(item: Any, path: Path, line_no: int | None = None) -> PromptRecord:
    if not isinstance(item, dict):
        location = f"{path}:{line_no}" if line_no is not None else str(path)
        raise TypeError(f"{location} must contain JSON objects")

    prompt = item.get("prompt", item.get("input", item.get("question")))
    reference = item.get("reference", item.get("answer", item.get("target")))
    if not isinstance(prompt, str) or not prompt:
        location = f"{path}:{line_no}" if line_no is not None else str(path)
        raise ValueError(f"{location} is missing a non-empty prompt/input/question field")
    if reference is not None and not isinstance(reference, str):
        reference = str(reference)
    return PromptRecord(prompt=prompt, reference=reference)


def _load_text_records(path: Path) -> list[PromptRecord]:
    return [PromptRecord(prompt=line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _load_records(prompts_file: str | None, prompt: list[str], references_file: str | None) -> list[PromptRecord]:
    records = [PromptRecord(prompt=value) for value in prompt]

    if prompts_file is not None:
        path = Path(prompts_file)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records = _load_jsonl_records(path)
        elif suffix == ".json":
            records = _load_json_records(path)
        else:
            records = _load_text_records(path)

    if references_file is not None:
        references = [line.strip() for line in Path(references_file).read_text().splitlines()]
        references = [line for line in references if line]
        if len(references) != len(records):
            raise ValueError(
                f"references file has {len(references)} entries, but prompt set has {len(records)} entries"
            )
        records = [
            PromptRecord(prompt=record.prompt, reference=reference) for record, reference in zip(records, references)
        ]

    if not records:
        raise ValueError("provide at least one prompt through --prompt or --prompts-file")
    return records


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


def _flatten_ids(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 0:
            return [[int(array.item())]]
        if array.ndim == 1:
            return [[int(token) for token in array.tolist()]]
        return [[int(token) for token in row] for row in array.reshape((-1, array.shape[-1])).tolist()]
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, np.integer)) for item in value):
            return [[int(item) for item in value]]
        rows = []
        for item in value:
            rows.extend(_flatten_ids(item))
        return rows
    return []


def _perf_dict(exec_info: Any) -> dict[str, float]:
    metrics = exec_info.perf_metrics
    batch_size = int(getattr(exec_info, "batch_size", 1))
    return {
        "batch_size": float(batch_size),
        "ttft_sec": float(metrics.prefill_time),
        "decode_tokens_per_sec": float(metrics.decode_perf * batch_size),
        "total_tokens_per_sec": float(metrics.total_perf * batch_size),
        "e2e_time_sec": float(metrics.total_time),
    }


def _score_references(texts: list[str], records: list[PromptRecord]) -> dict[str, float]:
    pairs = [(text, record.reference) for text, record in zip(texts, records) if record.reference is not None]
    if not pairs:
        return {}

    exact = []
    contains = []
    token_f1 = []
    char_similarity = []
    for text, reference in pairs:
        assert reference is not None
        normalized_text = _normalize_text(text)
        normalized_reference = _normalize_text(reference)
        exact.append(float(normalized_text == normalized_reference))
        contains.append(float(normalized_reference in normalized_text))
        token_f1.append(_token_f1(text, reference))
        char_similarity.append(SequenceMatcher(a=normalized_text, b=normalized_reference, autojunk=False).ratio())

    return {
        "reference_count": float(len(pairs)),
        "exact_match": _mean(exact),
        "contains_reference": _mean(contains),
        "token_f1": _mean(token_f1),
        "char_similarity": _mean(char_similarity),
    }


def _score_against_baseline_texts(texts: list[str], baseline_texts: list[str]) -> dict[str, float]:
    if not baseline_texts:
        return {}
    count = min(len(texts), len(baseline_texts))
    if count == 0:
        return {}

    exact = []
    token_f1 = []
    char_similarity = []
    for text, baseline in zip(texts[:count], baseline_texts[:count]):
        normalized_text = _normalize_text(text)
        normalized_baseline = _normalize_text(baseline)
        exact.append(float(normalized_text == normalized_baseline))
        token_f1.append(_token_f1(text, baseline))
        char_similarity.append(SequenceMatcher(a=normalized_text, b=normalized_baseline, autojunk=False).ratio())

    return {
        "baseline_text_exact": _mean(exact),
        "baseline_text_token_f1": _mean(token_f1),
        "baseline_text_char_similarity": _mean(char_similarity),
    }


def _score_against_baseline_ids(ids: list[list[int]], baseline_ids: list[list[int]]) -> dict[str, float]:
    if not baseline_ids:
        return {}
    count = min(len(ids), len(baseline_ids))
    if count == 0:
        return {}

    exact = []
    similarity = []
    common_prefix = []
    for current, baseline in zip(ids[:count], baseline_ids[:count]):
        exact.append(float(current == baseline))
        similarity.append(_sequence_similarity(current, baseline))
        prefix_len = 0
        for left, right in zip(current, baseline):
            if left != right:
                break
            prefix_len += 1
        common_prefix.append(prefix_len / max(len(baseline), 1))

    return {
        "baseline_token_exact": _mean(exact),
        "baseline_token_edit_similarity": _mean(similarity),
        "baseline_token_common_prefix": _mean(common_prefix),
    }


def _compute_deltas(metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value - baseline_metrics[key]
        for key, value in metrics.items()
        if key in baseline_metrics and key != "reference_count"
    }


def _compute_recovery(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    skip_metrics: dict[str, float],
) -> dict[str, float]:
    recovery = {}
    for key, value in metrics.items():
        if key not in baseline_metrics or key not in skip_metrics or key == "reference_count":
            continue
        denominator = baseline_metrics[key] - skip_metrics[key]
        if abs(denominator) < 1e-12:
            continue
        recovery[key] = (value - skip_metrics[key]) / denominator
    return recovery


def _run_variant(
    name: str,
    qpc_path: str,
    tokenizer: Any,
    prompts: list[str],
    generation_len: int,
    device_ids: list[int] | None,
) -> tuple[list[str], list[list[int]], dict[str, float]]:
    exec_info = cloud_ai_100_exec_kv(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        prompt=prompts,
        device_id=device_ids,
        generation_len=generation_len,
        stream=False,
        automation=True,
    )
    texts = _flatten_texts(exec_info.generated_texts)
    ids = _flatten_ids(exec_info.generated_ids)
    print(f"[{name}] collected {len(texts)} generated texts and {len(ids)} generated id rows")
    return texts, ids, _perf_dict(exec_info)


def _write_outputs(results: list[VariantResult], output_json: str | None, output_csv: str | None) -> None:
    payload = [asdict(result) for result in results]
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON results to {path}")

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(
            {
                "variant",
                "qpc_path",
                *[f"perf.{key}" for result in results for key in result.perf],
                *[f"reference.{key}" for result in results for key in result.reference_metrics],
                *[f"baseline_text.{key}" for result in results for key in result.baseline_text_metrics],
                *[f"baseline_token.{key}" for result in results for key in result.baseline_token_metrics],
                *[f"delta.{key}" for result in results for key in result.deltas_vs_baseline],
                *[f"recovery.{key}" for result in results for key in result.recovery_vs_skip],
            }
        )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row: dict[str, Any] = {"variant": result.name, "qpc_path": result.qpc_path}
                row.update({f"perf.{key}": value for key, value in result.perf.items()})
                row.update({f"reference.{key}": value for key, value in result.reference_metrics.items()})
                row.update({f"baseline_text.{key}": value for key, value in result.baseline_text_metrics.items()})
                row.update({f"baseline_token.{key}": value for key, value in result.baseline_token_metrics.items()})
                row.update({f"delta.{key}": value for key, value in result.deltas_vs_baseline.items()})
                row.update({f"recovery.{key}": value for key, value in result.recovery_vs_skip.items()})
                writer.writerow(row)
        print(f"Wrote CSV results to {path}")


def _print_summary(results: list[VariantResult]) -> None:
    print("\n========================= Layer-Skip Evaluation Summary =========================")
    for result in results:
        print(f"\nVariant: {result.name}")
        print(f"  QPC: {result.qpc_path}")
        print(f"  TTFT: {result.perf['ttft_sec']:.4f} sec")
        print(f"  Decode: {result.perf['decode_tokens_per_sec']:.2f} token/sec")
        print(f"  Total: {result.perf['total_tokens_per_sec']:.2f} token/sec")
        if result.reference_metrics:
            print(
                "  Reference: "
                f"EM={result.reference_metrics.get('exact_match', 0.0):.4f}, "
                f"F1={result.reference_metrics.get('token_f1', 0.0):.4f}, "
                f"char_sim={result.reference_metrics.get('char_similarity', 0.0):.4f}"
            )
        if result.baseline_text_metrics:
            print(
                "  Baseline text: "
                f"exact={result.baseline_text_metrics.get('baseline_text_exact', 0.0):.4f}, "
                f"F1={result.baseline_text_metrics.get('baseline_text_token_f1', 0.0):.4f}, "
                f"char_sim={result.baseline_text_metrics.get('baseline_text_char_similarity', 0.0):.4f}"
            )
        if result.baseline_token_metrics:
            print(
                "  Baseline tokens: "
                f"exact={result.baseline_token_metrics.get('baseline_token_exact', 0.0):.4f}, "
                f"edit_sim={result.baseline_token_metrics.get('baseline_token_edit_similarity', 0.0):.4f}, "
                f"prefix={result.baseline_token_metrics.get('baseline_token_common_prefix', 0.0):.4f}"
            )
        if result.recovery_vs_skip:
            compact = ", ".join(f"{key}={value:.4f}" for key, value in sorted(result.recovery_vs_skip.items()))
            print(f"  Recovery vs skip: {compact}")
    print("===============================================================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitatively compare baseline, layer-skipped, and compensated language QPCs."
    )
    parser.add_argument(
        "--tokenizer-name",
        required=True,
        help="Tokenizer name or local tokenizer path matching the compiled QPCs.",
    )
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
        help="Variant name to treat as the full-model baseline. Defaults to the first --variant.",
    )
    parser.add_argument(
        "--skip-variant",
        default=None,
        help="Uncompensated skipped-layer variant. Enables recovery ratios for compensated variants.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help=(
            "Prompt file. .txt uses one prompt per line; .jsonl/.json accepts objects with "
            "prompt/input/question and optional reference/answer/target."
        ),
    )
    parser.add_argument(
        "--references-file",
        default=None,
        help="Optional line-aligned references for .txt prompts or --prompt inputs.",
    )
    parser.add_argument("--generation-len", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument(
        "--device-id",
        default=None,
        help="QAIC device IDs, for example '0' or '[0,1]'. Defaults to runtime auto-device selection.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON result path.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV summary path.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = dict(args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("variant names must be unique")

    baseline_name = args.baseline_variant or args.variant[0][0]
    if baseline_name not in variants:
        raise ValueError(f"baseline variant {baseline_name!r} was not provided")
    if args.skip_variant is not None and args.skip_variant not in variants:
        raise ValueError(f"skip variant {args.skip_variant!r} was not provided")

    records = _load_records(args.prompts_file, args.prompt, args.references_file)
    prompts = [record.prompt for record in records]
    device_ids = _parse_device_ids(args.device_id)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_outputs: dict[str, tuple[list[str], list[list[int]], dict[str, float]]] = {}
    for name, qpc_path in args.variant:
        raw_outputs[name] = _run_variant(
            name=name,
            qpc_path=qpc_path,
            tokenizer=tokenizer,
            prompts=prompts,
            generation_len=args.generation_len,
            device_ids=device_ids,
        )

    baseline_texts, baseline_ids, baseline_perf = raw_outputs[baseline_name]
    baseline_reference_metrics = _score_references(baseline_texts, records)
    skip_reference_metrics = {}
    if args.skip_variant is not None:
        skip_texts, _, _ = raw_outputs[args.skip_variant]
        skip_reference_metrics = _score_references(skip_texts, records)

    baseline_self_text_metrics = _score_against_baseline_texts(baseline_texts, baseline_texts)
    baseline_self_token_metrics = _score_against_baseline_ids(baseline_ids, baseline_ids)

    results = []
    for name, qpc_path in args.variant:
        texts, ids, perf = raw_outputs[name]
        reference_metrics = _score_references(texts, records)
        baseline_text_metrics = _score_against_baseline_texts(texts, baseline_texts)
        baseline_token_metrics = _score_against_baseline_ids(ids, baseline_ids)

        deltas = _compute_deltas(perf, baseline_perf)
        deltas.update(
            {
                f"reference_{key}": value
                for key, value in _compute_deltas(reference_metrics, baseline_reference_metrics).items()
            }
        )

        recovery = {}
        if skip_reference_metrics:
            recovery = _compute_recovery(reference_metrics, baseline_reference_metrics, skip_reference_metrics)
        if args.skip_variant is not None:
            skip_texts, skip_ids, _ = raw_outputs[args.skip_variant]
            skip_text_metrics = _score_against_baseline_texts(skip_texts, baseline_texts)
            skip_token_metrics = _score_against_baseline_ids(skip_ids, baseline_ids)
            recovery.update(
                {
                    f"text_{key}": value
                    for key, value in _compute_recovery(
                        baseline_text_metrics,
                        baseline_self_text_metrics,
                        skip_text_metrics,
                    ).items()
                }
            )
            recovery.update(
                {
                    f"token_{key}": value
                    for key, value in _compute_recovery(
                        baseline_token_metrics,
                        baseline_self_token_metrics,
                        skip_token_metrics,
                    ).items()
                }
            )

        results.append(
            VariantResult(
                name=name,
                qpc_path=qpc_path,
                generated_texts=texts,
                generated_ids=ids,
                perf=perf,
                reference_metrics=reference_metrics,
                baseline_text_metrics=baseline_text_metrics,
                baseline_token_metrics=baseline_token_metrics,
                deltas_vs_baseline=deltas,
                recovery_vs_skip=recovery,
            )
        )

    _print_summary(results)
    _write_outputs(results, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
