#!/usr/bin/env python3
"""Profile GPT-OSS MoE expert routing and write layer-by-expert matrices.

# START: MoE router profiling module
# This module is adapted from the reference MoE routing profiler.  It loads a
# GPT-OSS-style causal LM, attaches forward hooks to router modules, runs text
# datasets through the model, and saves layer-by-expert CSV matrices that are
# consumed by ``analyze_moe_expert_importance.py`` and the pipeline runner.
# END: MoE router profiling module
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DATASET_REGISTRY = {
    "gsm8k": ("gsm8k", "main", "train", "question"),
    "hellaswag": ("Rowan/hellaswag", None, "validation", "ctx"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", "sentence"),
    "wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text"),
    "mmlu": ("cais/mmlu", "all", "validation", "question"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation", "question"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", "question"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "question"),
    "piqa": ("piqa", None, "validation", "goal"),
    "boolq": ("google/boolq", None, "validation", "question"),
    "openbookqa": ("allenai/openbookqa", "main", "validation", "question_stem"),
}


@dataclass(frozen=True)
class MoeLayerSpec:
    """Metadata for one discovered MoE layer."""

    layer_idx: int
    module_name: str
    router_module_name: str
    num_experts: int
    top_k: int


class RoutingStatsAccumulator:
    """Accumulate routing statistics into layer-by-expert matrices."""

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.freq_counts = torch.zeros((num_layers, num_experts), dtype=torch.long)
        self.importance_sum = torch.zeros((num_layers, num_experts), dtype=torch.float64)

    def update_from_selected(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        """Update stats from actual selected expert ids and router weights."""

        selected_experts = selected_experts.detach()
        selected_weights = selected_weights.detach()

        if selected_experts.ndim == 3:
            selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
        if selected_weights.ndim == 3:
            selected_weights = selected_weights.reshape(-1, selected_weights.shape[-1])

        if attention_mask is not None:
            valid_token_mask = attention_mask.reshape(-1).to(device=selected_experts.device, dtype=torch.bool)
            if valid_token_mask.numel() == selected_experts.shape[0]:
                selected_experts = selected_experts[valid_token_mask]
                selected_weights = selected_weights[valid_token_mask]

        if selected_experts.numel() == 0:
            return

        selected_experts_cpu = selected_experts.reshape(-1).to(device="cpu", dtype=torch.long)
        selected_weights_cpu = selected_weights.reshape(-1).to(device="cpu", dtype=torch.float64)

        num_experts = self.freq_counts.shape[1]
        batch_counts = torch.bincount(selected_experts_cpu, minlength=num_experts)
        batch_importance = torch.bincount(selected_experts_cpu, weights=selected_weights_cpu, minlength=num_experts)

        self.freq_counts[layer_idx] += batch_counts
        self.importance_sum[layer_idx] += batch_importance

    def update_from_logits(
        self,
        layer_idx: int,
        router_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        top_k: int,
    ) -> None:
        """Fallback path for routers that expose logits but not selected ids."""

        router_logits = router_logits.detach()
        if router_logits.ndim == 3:
            router_logits = router_logits.reshape(-1, router_logits.shape[-1])

        if attention_mask is not None:
            valid_token_mask = attention_mask.reshape(-1).to(device=router_logits.device, dtype=torch.bool)
            if valid_token_mask.numel() == router_logits.shape[0]:
                router_logits = router_logits[valid_token_mask]

        if router_logits.numel() == 0:
            return

        top_values, top_experts = torch.topk(router_logits, k=top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1, dtype=torch.float32)
        self.update_from_selected(layer_idx, top_experts, top_weights, attention_mask=None)

    def finalize(self) -> dict[str, torch.Tensor]:
        """Return final matrices."""

        freq_counts_float = self.freq_counts.to(torch.float64)
        layer_totals = freq_counts_float.sum(dim=1, keepdim=True)

        freq_fraction = torch.zeros_like(freq_counts_float)
        non_empty_layers = layer_totals.squeeze(1) > 0
        freq_fraction[non_empty_layers] = freq_counts_float[non_empty_layers] / layer_totals[non_empty_layers]

        importance_mean = torch.zeros_like(self.importance_sum)
        selected_mask = self.freq_counts > 0
        importance_mean[selected_mask] = self.importance_sum[selected_mask] / freq_counts_float[selected_mask]

        combined_score = freq_fraction * importance_mean

        return {
            "freq_counts": self.freq_counts.clone(),
            "freq_fraction": freq_fraction,
            "importance_sum": self.importance_sum.clone(),
            "importance_mean": importance_mean,
            "combined_score": combined_score,
        }


class GptOssRoutingProfiler:
    """Attach hooks to GPT-OSS router modules and collect expert routing stats."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layer_specs = self._discover_moe_layers()
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_mask: Optional[torch.Tensor] = None
        self.accumulator = self._new_accumulator()

    def _discover_moe_layers(self) -> list[MoeLayerSpec]:
        layer_specs: list[MoeLayerSpec] = []
        for module_name, module in self.model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            if router is None or experts is None:
                continue

            top_k = getattr(router, "top_k", None)
            num_experts = getattr(experts, "num_experts", None)
            if top_k is None or num_experts is None:
                continue

            layer_specs.append(
                MoeLayerSpec(
                    layer_idx=len(layer_specs),
                    module_name=module_name,
                    router_module_name=f"{module_name}.router",
                    num_experts=int(num_experts),
                    top_k=int(top_k),
                )
            )

        if not layer_specs:
            raise RuntimeError(
                "No GPT-OSS MoE layers found. Expected modules with `.router.top_k` "
                "and `.experts.num_experts`."
            )

        expert_counts = {spec.num_experts for spec in layer_specs}
        if len(expert_counts) != 1:
            raise RuntimeError(f"Expected same num_experts for all layers, found: {sorted(expert_counts)}")

        return layer_specs

    def _new_accumulator(self) -> RoutingStatsAccumulator:
        return RoutingStatsAccumulator(num_layers=len(self.layer_specs), num_experts=self.layer_specs[0].num_experts)

    def reset(self) -> None:
        self.accumulator = self._new_accumulator()

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._attention_mask = attention_mask

    def register_hooks(self) -> None:
        self.remove_hooks()
        modules = dict(self.model.named_modules())
        for spec in self.layer_specs:
            router_module = modules.get(spec.router_module_name)
            if router_module is None:
                raise RuntimeError(f"Could not find router module `{spec.router_module_name}`.")
            self._hook_handles.append(router_module.register_forward_hook(self._make_router_hook(spec)))

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _make_router_hook(self, spec: MoeLayerSpec):
        def hook(_module, _inputs, output) -> None:
            if isinstance(output, tuple) and len(output) >= 3:
                router_logits, router_scores, router_indices = output[:3]
                if torch.is_tensor(router_scores) and torch.is_tensor(router_indices):
                    self.accumulator.update_from_selected(
                        layer_idx=spec.layer_idx,
                        selected_experts=router_indices,
                        selected_weights=router_scores,
                        attention_mask=self._attention_mask,
                    )
                    return
                if torch.is_tensor(router_logits):
                    self.accumulator.update_from_logits(
                        layer_idx=spec.layer_idx,
                        router_logits=router_logits,
                        attention_mask=self._attention_mask,
                        top_k=spec.top_k,
                    )
                    return

            if torch.is_tensor(output):
                self.accumulator.update_from_logits(
                    layer_idx=spec.layer_idx,
                    router_logits=output,
                    attention_mask=self._attention_mask,
                    top_k=spec.top_k,
                )
                return

            raise RuntimeError(f"Router output from {spec.router_module_name} has unsupported format.")

        return hook


def parse_dataset_aliases(raw_datasets: str | list[str]) -> list[str]:
    """Turn ``all``, comma-separated text, or a list into validated dataset aliases."""

    if isinstance(raw_datasets, str):
        if raw_datasets.strip().lower() == "all":
            return list(DATASET_REGISTRY)
        aliases = [part.strip() for part in raw_datasets.replace(",", " ").split() if part.strip()]
    else:
        aliases = []
        for item in raw_datasets:
            if item.strip().lower() == "all":
                return list(DATASET_REGISTRY)
            aliases.extend(part.strip() for part in item.replace(",", " ").split() if part.strip())

    unknown_aliases = sorted(set(aliases) - set(DATASET_REGISTRY))
    if unknown_aliases:
        raise ValueError(f"Unknown dataset aliases: {unknown_aliases}. Valid aliases: {sorted(DATASET_REGISTRY)}")
    return aliases


def parse_torch_dtype(raw_dtype: str):
    if raw_dtype == "auto":
        return "auto"
    if raw_dtype == "float32":
        return torch.float32
    if raw_dtype == "float16":
        return torch.float16
    if raw_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {raw_dtype}")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def get_model_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def load_text_dataset(alias: str, trust_remote_code: bool):
    dataset_name, config_name, split_name, _text_column = DATASET_REGISTRY[alias]
    dataset_kwargs = {"split": split_name}
    if trust_remote_code:
        dataset_kwargs["trust_remote_code"] = True
    if config_name is None:
        return load_dataset(dataset_name, **dataset_kwargs)
    return load_dataset(dataset_name, config_name, **dataset_kwargs)


def iter_non_empty_texts(dataset, text_column: str, max_samples: Optional[int]) -> Iterable[str]:
    yielded = 0
    for row in dataset:
        value = row.get(text_column, "")
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if not text:
            continue

        yield text
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_matrix_csv(path: Path, matrix: torch.Tensor, value_kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_cpu = matrix.detach().cpu()

    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer", *[f"expert_{idx}" for idx in range(matrix_cpu.shape[1])]])

        for layer_idx, row in enumerate(matrix_cpu):
            if value_kind == "int":
                values = [str(int(value)) for value in row.tolist()]
            else:
                values = [f"{float(value):.10g}" for value in row.tolist()]
            writer.writerow([f"layer_{layer_idx}", *values])


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    device_map: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """Load model/tokenizer using the same large-model defaults as the reference pipeline."""

    dtype = parse_torch_dtype(torch_dtype)

    effective_device_map = device_map
    resolved_device = resolve_device(device)
    if effective_device_map is None and resolved_device.type == "cuda":
        effective_device_map = "auto"
        print("  device_map resolved to 'auto' for CUDA large-model loading")

    print(f"  Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if effective_device_map is not None:
        model_kwargs["device_map"] = effective_device_map

    print("  Loading weights " f"(torch_dtype={torch_dtype}, low_cpu_mem_usage=True, device_map={effective_device_map})")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if effective_device_map is None:
        model.to(resolved_device)

    return model, tokenizer


def profile_dataset(
    alias: str,
    model: torch.nn.Module,
    tokenizer,
    profiler: GptOssRoutingProfiler,
    max_samples: Optional[int],
    batch_size: int,
    max_length: int,
    device: str,
    trust_remote_code: bool,
) -> dict[str, torch.Tensor]:
    _dataset_name, _config_name, _split_name, text_column = DATASET_REGISTRY[alias]
    dataset = load_text_dataset(alias, trust_remote_code=trust_remote_code)
    input_device = get_model_input_device(model, resolve_device(device))

    profiler.reset()
    processed_samples = 0
    texts = iter_non_empty_texts(dataset, text_column=text_column, max_samples=max_samples)

    # START: Dataset forward-pass profiling loop
    # Each non-empty text batch is tokenized, moved to the model input device,
    # and forwarded once.  Router hooks collect expert ids/weights during the
    # forward pass; the language-model outputs themselves are not used here.
    for text_batch in batched(texts, batch_size):
        processed_samples += len(text_batch)
        encoded = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        profiler.set_attention_mask(encoded.get("attention_mask"))

        with torch.no_grad():
            model(**encoded, use_cache=False)
    # END: Dataset forward-pass profiling loop

    if processed_samples == 0:
        raise RuntimeError(f"Dataset {alias} produced zero non-empty samples from column `{text_column}`.")

    print(f"Profiled {processed_samples} samples for dataset `{alias}`.")
    return profiler.accumulator.finalize()


def write_dataset_outputs(
    alias: str,
    matrices: dict[str, torch.Tensor],
    output_dir: Path,
    write_importance_debug: bool = False,
) -> None:
    write_matrix_csv(output_dir / f"{alias}_freq_counts.csv", matrices["freq_counts"], value_kind="int")
    write_matrix_csv(output_dir / f"{alias}_freq_fraction.csv", matrices["freq_fraction"], value_kind="float")
    write_matrix_csv(output_dir / f"{alias}_combined_score.csv", matrices["combined_score"], value_kind="float")

    if write_importance_debug:
        write_matrix_csv(output_dir / f"{alias}_importance_sum.csv", matrices["importance_sum"], value_kind="float")
        write_matrix_csv(output_dir / f"{alias}_importance_mean.csv", matrices["importance_mean"], value_kind="float")


def profile_moe_expert_importance(
    model_name: str = "openai/gpt-oss-20b",
    datasets: str | list[str] = "all",
    max_samples: Optional[int] = 100,
    batch_size: int = 1,
    max_length: int = 512,
    output_dir: str | Path = "outputs/moe_routing",
    device: str = "cuda",
    device_map: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    write_importance_debug: bool = False,
) -> dict:
    """Profile GPT-OSS expert routing and write matrix CSVs."""

    # START: Full MoE expert-routing profiling flow
    # This is the top-level profiling function called by the pipeline runner.
    # It loads one model, discovers all MoE routers, profiles every requested
    # dataset, writes matrix CSV files, and returns metadata for checkpointing.
    dataset_aliases = parse_dataset_aliases(datasets)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    print("Discovering GPT-OSS MoE layers...")
    profiler = GptOssRoutingProfiler(model)
    print(
        f"Found {len(profiler.layer_specs)} MoE layers, "
        f"{profiler.layer_specs[0].num_experts} experts/layer, top_k={profiler.layer_specs[0].top_k}."
    )

    profiler.register_hooks()
    processed_datasets = []
    try:
        for alias in dataset_aliases:
            print(f"\nProfiling dataset: {alias}")
            matrices = profile_dataset(
                alias=alias,
                model=model,
                tokenizer=tokenizer,
                profiler=profiler,
                max_samples=max_samples,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            write_dataset_outputs(alias, matrices, output_dir, write_importance_debug=write_importance_debug)
            processed_datasets.append(alias)
            print(f"Wrote matrix CSV files for `{alias}` to {output_dir}")
    finally:
        profiler.remove_hooks()

    metadata = {
        "output_dir": str(output_dir),
        "model": model_name,
        "datasets": processed_datasets,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_moe_layers": len(profiler.layer_specs),
        "num_experts": profiler.layer_specs[0].num_experts,
        "top_k": profiler.layer_specs[0].top_k,
        "layer_specs": [spec.__dict__ for spec in profiler.layer_specs],
    }

    profiler.model = None
    del model, tokenizer

    return metadata
    # END: Full MoE expert-routing profiling flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GPT-OSS MoE expert routing matrices.")
    parser.add_argument("--model-name", "--model", dest="model_name", default="openai/gpt-oss-20b")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset aliases, comma-separated aliases, or all.")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum non-empty samples per dataset.")
    parser.add_argument("--num-samples", type=int, default=None, help="Alias for --max-samples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/moe_routing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--write-importance-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_moe_expert_importance(
        model_name=args.model_name,
        datasets=args.datasets,
        max_samples=args.num_samples if args.num_samples is not None else args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        write_importance_debug=args.write_importance_debug,
    )


if __name__ == "__main__":
    main()
