# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo-based export and inference for Causal LM models on Cloud AI 100.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/causal_lm/requirements.txt
"""

import argparse

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo-based export and inference for Causal LM models on Cloud AI 100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="tiny-random/gpt-oss-mxfp4", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Override number of hidden layers")
    parser.add_argument("--prompt", type=str, default="My name is", help="Input prompt for generation")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context (KV-cache) length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--blocking-mode", choices=["kv", "kv_headpar", "q", "h", "qkv", "hq", "hkv", "hqkv"], default=None)
    parser.add_argument("--num-kv-blocks", type=int, default=None)
    parser.add_argument("--num-q-blocks", type=int, default=None)
    parser.add_argument("--head-block-size", type=int, default=None)
    parser.add_argument("--headpar-split", type=int, default=None)
    parser.add_argument("--disable-kv-loop-op", action="store_true")
    parser.add_argument("--skip-kv", action="store_true", help="Enable skipping KV blocks beyond the live context")
    parser.add_argument("--kv-loop-dynamic-trip-count", action="store_true")
    parser.add_argument("--enable-onnx-subfunctions", action="store_true")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of AI 100 cores")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="AIC hardware version"
    )
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated), e.g. [0,1]",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        weight_free=args.weight_free,
    )

    qaic_config = None
    if args.blocking_mode is not None:
        qaic_config = {"blocking_mode": args.blocking_mode}
        for key in ("num_kv_blocks", "num_q_blocks", "head_block_size", "headpar_split"):
            value = getattr(args, key)
            if value is not None:
                qaic_config[key] = value
        if args.disable_kv_loop_op:
            qaic_config["use_kv_loop_op"] = False
        if args.skip_kv:
            qaic_config["skip_kv"] = True
        if args.kv_loop_dynamic_trip_count:
            qaic_config["kv_loop_dynamic_trip_count"] = True

    # Export (via torch.export / dynamo) + compile to QPC
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=args.enable_onnx_subfunctions,
    )
    print(f"Model compiled to: {qpc_path}")

    # Run inference on device
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt   : {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
