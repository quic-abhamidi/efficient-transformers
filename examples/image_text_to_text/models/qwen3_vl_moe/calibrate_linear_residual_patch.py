# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient.pruning.config import PruningConfig
from QEfficient.pruning.layer_skip import resolve_layer_container
from QEfficient.transformers.models.pytorch_transforms import PruningTransform

DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "Explain why the sky appears blue in two sentences.",
    "Give three examples of tasks a vision-language assistant can help with.",
    "Solve step by step: if a box has 12 red balls and 8 blue balls, how many balls are there?",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a training-free full linear residual patch for Qwen3-VL layer skipping."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--skip-layers", nargs="+", type=int, default=[32, 33, 34, 35, 36])
    parser.add_argument("--injection-layer", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output .pt file for the calibrated patch weights.")
    parser.add_argument("--prompts-file", default=None, help="Optional text file with one calibration prompt per line.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def read_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def build_inputs(processor, prompt: str, max_length: int):
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    return processor(
        text=text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def first_tensor_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def move_inputs(inputs, device: torch.device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def resolve_injection_layer(skip_layers: list[int], injection_layer: int | None) -> int:
    return max(skip_layers) + 1 if injection_layer is None else injection_layer


def load_model(model_id: str, dtype: torch.dtype, device_map: str, trust_remote_code: bool):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()
    return model


def apply_skip(model, skip_layers: list[int]):
    pruning_config = PruningConfig.from_qaic_config(
        {
            "enable_pruning": True,
            "pruning_config": {"skip_layers": skip_layers},
        }
    )
    model, _ = PruningTransform.apply(model, pruning_config)
    return model


def collect_layer_inputs(model, processor, prompts: list[str], injection_layer: int, max_length: int, max_tokens: int):
    layers = resolve_layer_container(model).container
    if injection_layer >= len(layers):
        raise ValueError(f"Injection layer {injection_layer} out of range for {len(layers)} layers.")

    captured = []

    def hook(_module, args, kwargs):
        hidden_states = kwargs.get("hidden_states") if kwargs else None
        if hidden_states is None:
            hidden_states = args[0]
        captured.append(hidden_states.detach().float().cpu())

    handle = layers[injection_layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for prompt in prompts:
                before = len(captured)
                inputs = build_inputs(processor, prompt, max_length)
                attention_mask = inputs.get("attention_mask")
                inputs = move_inputs(inputs, first_tensor_device(model))
                model(**inputs, use_cache=False)
                if len(captured) == before:
                    raise RuntimeError(f"No activation captured for prompt: {prompt!r}")

                hidden = captured[-1]
                if attention_mask is not None and attention_mask.shape[:2] == hidden.shape[:2]:
                    mask = attention_mask.bool().cpu().view(-1)
                    hidden = hidden.view(-1, hidden.shape[-1])[mask]
                else:
                    hidden = hidden.reshape(-1, hidden.shape[-1])
                captured[-1] = hidden

                if sum(chunk.shape[0] for chunk in captured) >= max_tokens:
                    break
    finally:
        handle.remove()

    activations = torch.cat(captured, dim=0)[:max_tokens].contiguous()
    if activations.numel() == 0:
        raise RuntimeError("No calibration activations were collected.")
    return activations


def solve_ridge_patch(full_hidden: torch.Tensor, pruned_hidden: torch.Tensor, ridge_lambda: float):
    if full_hidden.shape != pruned_hidden.shape:
        raise ValueError(f"Activation shape mismatch: full={full_hidden.shape}, pruned={pruned_hidden.shape}")

    x = pruned_hidden.float()
    target = (full_hidden - pruned_hidden).float()
    xtx = x.T @ x
    xtx.diagonal().add_(ridge_lambda)
    xtd = x.T @ target
    w = torch.linalg.solve(xtx, xtd)
    return w.T.contiguous()


def main():
    args = parse_args()
    prompts = read_prompts(args.prompts_file)
    injection_layer = resolve_injection_layer(args.skip_layers, args.injection_layer)
    dtype = dtype_from_name(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    print(f"Calibrating linear residual patch for skipped layers {args.skip_layers}")
    print(f"Injection layer: {injection_layer}")
    print(f"Calibration prompts: {len(prompts)}")

    full_model = load_model(args.model_id, dtype, args.device_map, args.trust_remote_code)
    full_hidden = collect_layer_inputs(
        full_model, processor, prompts, injection_layer, args.max_length, args.max_tokens
    )
    del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pruned_model = load_model(args.model_id, dtype, args.device_map, args.trust_remote_code)
    pruned_model = apply_skip(pruned_model, args.skip_layers)
    pruned_hidden = collect_layer_inputs(
        pruned_model, processor, prompts, injection_layer, args.max_length, args.max_tokens
    )
    del pruned_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_tokens = min(full_hidden.shape[0], pruned_hidden.shape[0])
    full_hidden = full_hidden[:num_tokens]
    pruned_hidden = pruned_hidden[:num_tokens]
    weight = solve_ridge_patch(full_hidden, pruned_hidden, args.ridge_lambda)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {
            "linear.weight": weight,
            "alpha": torch.tensor(float(args.alpha), dtype=torch.float32),
        },
        "metadata": {
            "type": "linear_residual_patch",
            "model_id": args.model_id,
            "skip_layers": args.skip_layers,
            "injection_layer": injection_layer,
            "num_tokens": int(num_tokens),
            "hidden_size": int(weight.shape[0]),
            "ridge_lambda": float(args.ridge_lambda),
            "alpha": float(args.alpha),
        },
    }
    torch.save(payload, output)

    residual_before = torch.mean((full_hidden - pruned_hidden).pow(2)).item()
    corrected = pruned_hidden + pruned_hidden @ weight.T
    residual_after = torch.mean((full_hidden - corrected).pow(2)).item()
    summary = {
        "output": str(output),
        "num_tokens": int(num_tokens),
        "hidden_size": int(weight.shape[0]),
        "mse_before": residual_before,
        "mse_after": residual_after,
        "mse_reduction_pct": round((residual_before - residual_after) / residual_before * 100, 4)
        if residual_before
        else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
