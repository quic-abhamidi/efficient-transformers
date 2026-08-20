# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL with layer skip + optional linear residual patch.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--skip-layers", nargs="+", type=int, default=[32, 33, 34, 35, 36])
    parser.add_argument("--layer-skip-config", required=True)
    parser.add_argument("--linear-patch-weights", default=None)
    parser.add_argument("--injection-layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--compile-dir", required=True)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    return parser.parse_args()


def build_qaic_config(args):
    config_path = Path(args.layer_skip_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"skip_layers": args.skip_layers}, indent=2) + "\n")

    qaic_config = {
        "enable_layer_skipping": True,
        "layer_skip_config": str(config_path),
    }
    if args.linear_patch_weights is not None:
        compensation = {
            "type": "linear_residual_patch",
            "patch_weights": args.linear_patch_weights,
            "alpha": args.alpha,
        }
        if args.injection_layer is not None:
            compensation["injection_layer"] = args.injection_layer
        qaic_config["layer_skip_compensation"] = compensation
    return qaic_config


def build_text_inputs(processor):
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Tell me about yourself."}],
            }
        ]
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def main():
    args = parse_args()
    qaic_config = build_qaic_config(args)

    print(f"RUN_NAME={args.run_name}")
    print(f"MODEL={args.model_id}")
    print(f"SKIP_LAYERS={args.skip_layers}")
    print(f"QAIC_CONFIG={qaic_config}")

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        qaic_config=qaic_config,
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    qeff_model.compile(
        batch_size=1,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        height=args.height,
        width=args.width,
        mxfp6_matmul=True,
        mxint8_kv_cache=args.mxint8_kv_cache,
        aic_enable_depth_first=True,
        skip_vision=args.skip_vision,
        mos=1,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        qaic_config=qaic_config,
        compile_dir=args.compile_dir,
    )

    inputs = build_text_inputs(processor)
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=args.prefill_seq_len,
        batch_size=1,
    )
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)

    print("QPC_PATHS=", getattr(qeff_model, "qpc_paths", None))
    print("GENERATED_IDS=", output.generated_ids)
    print("DECODED=", tokenizer.batch_decode(output.generated_ids))
    print(output)


if __name__ == "__main__":
    main()
