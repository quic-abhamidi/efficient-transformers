# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""End-to-end Qwen3.5 image+text inference with Dynamo export.

This example compiles the Qwen3.5 VLM with ``dynamo=True`` and then runs a
single image+text generation request against the compiled QPC.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants


def _load_image(image_path: Optional[str], image_url: Optional[str], width: int, height: int) -> Image.Image:
    if image_path:
        image = Image.open(Path(image_path)).convert("RGB")
    else:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
    return image.resize((width, height))



def _tensorize_generated_ids(generated_ids):
    if isinstance(generated_ids, torch.Tensor):
        return generated_ids.detach().cpu().to(torch.long)
    return torch.as_tensor(generated_ids, dtype=torch.long)


def _run_hf_generation(args, processor, messages, torch_dtype):
    hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    hf_model.eval()
    hf_model.to(args.hf_device)

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    hf_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    hf_inputs = {name: value.to(args.hf_device) for name, value in hf_inputs.items()}

    with torch.no_grad():
        hf_ids = hf_model.generate(
            **hf_inputs,
            max_new_tokens=args.generation_len,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = hf_inputs["input_ids"].shape[1]
    return hf_ids[:, prompt_len:].detach().cpu()


def _print_parity_report(tokenizer, qaic_generated_ids, hf_generated_ids, require_exact):
    qaic_ids = _tensorize_generated_ids(qaic_generated_ids)
    hf_ids = _tensorize_generated_ids(hf_generated_ids)
    min_len = min(qaic_ids.shape[-1], hf_ids.shape[-1])
    prefix_match = bool(torch.equal(qaic_ids[:, :min_len], hf_ids[:, :min_len]))
    exact_match = qaic_ids.shape == hf_ids.shape and prefix_match

    print("HF generated IDs:")
    print(hf_ids.numpy())
    print("HF decoded output:")
    print(tokenizer.batch_decode(hf_ids, skip_special_tokens=False))
    print("Parity summary:")
    print(f"  exact_token_match={exact_match}")
    print(f"  common_prefix_match={prefix_match}")
    print(f"  qaic_shape={tuple(qaic_ids.shape)}")
    print(f"  hf_shape={tuple(hf_ids.shape)}")

    if not exact_match:
        mismatch = None
        for batch_idx in range(min(qaic_ids.shape[0], hf_ids.shape[0])):
            for token_idx in range(min_len):
                if int(qaic_ids[batch_idx, token_idx]) != int(hf_ids[batch_idx, token_idx]):
                    mismatch = (
                        batch_idx,
                        token_idx,
                        int(qaic_ids[batch_idx, token_idx]),
                        int(hf_ids[batch_idx, token_idx]),
                    )
                    break
            if mismatch is not None:
                break
        if mismatch is None and qaic_ids.shape != hf_ids.shape:
            print("  first_mismatch=length differs after matching common prefix")
        elif mismatch is not None:
            batch_idx, token_idx, qaic_token, hf_token = mismatch
            print(
                "  first_mismatch="
                f"batch={batch_idx} token={token_idx} qaic={qaic_token} hf={hf_token}"
            )

    if require_exact and not exact_match:
        raise AssertionError("QAIC generated IDs do not exactly match HuggingFace generated IDs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile and run Qwen3.5 image+text generation with Dynamo enabled.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-0.8B", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Describe all the colors seen in the image.")
    parser.add_argument("--image-path", type=str, default=None, help="Local image path. Overrides --image-url.")
    parser.add_argument("--image-url", type=str, default="https://picsum.photos/id/237/536/354")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION)
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--torch-dtype", type=str, default="float32")
    parser.add_argument("--disable-mxfp6-matmul", action="store_true")
    parser.add_argument("--disable-mxint8-kv-cache", action="store_true")
    parser.add_argument(
        "--disable-onnx-subfunctions",
        action="store_true",
        help="Disable experimental ONNX subfunctions. Subfunctions are enabled by default.",
    )
    parser.add_argument(
        "--hf-parity",
        action="store_true",
        help="Run a HuggingFace eager baseline and compare generated tokens.",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for the HuggingFace parity baseline.",
    )
    parser.add_argument(
        "--parity-require-exact",
        action="store_true",
        help="Exit with an error if QAIC generated IDs do not exactly match the HuggingFace baseline.",
    )
    parser.add_argument("--no-streamer", action="store_true", help="Disable token streaming during the QAIC run.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    torch_dtype = getattr(torch, args.torch_dtype)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    config.torch_dtype = torch_dtype

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        trust_remote_code=args.trust_remote_code,
    )
    qeff_model.model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    print("Compiling Qwen3.5 VLM with dynamo=True...")
    qpc_path = qeff_model.compile(
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        aic_hw_version=args.aic_hw_version,
        height=args.height,
        width=args.width,
        mxfp6_matmul=not args.disable_mxfp6_matmul,
        mxint8_kv_cache=not args.disable_mxint8_kv_cache,
        aic_enable_depth_first=False,
        mos=args.mos,
        split_model_io=True,
        use_onnx_subfunctions=not args.disable_onnx_subfunctions,
        dynamo=True,
    )
    print(f"Model compiled to: {qpc_path}")

    image = _load_image(args.image_path, args.image_url, args.width, args.height)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    messages = [messages] * args.batch_size

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    qeff_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    qeff_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=qeff_inputs,
        prefill_seq_len=args.prefill_seq_len,
        batch_size=args.batch_size,
    )

    hf_generated_ids = None
    if args.hf_parity:
        print("Running HuggingFace eager baseline for parity...")
        hf_generated_ids = _run_hf_generation(args, processor, messages, torch_dtype)

    streamer = None if args.no_streamer or args.hf_parity else TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=qeff_inputs, generation_len=args.generation_len, streamer=streamer)

    print("QAIC generated IDs:")
    print(output.generated_ids)
    print("QAIC decoded output:")
    print(tokenizer.batch_decode(output.generated_ids, skip_special_tokens=False))
    print(output)

    if hf_generated_ids is not None:
        _print_parity_report(tokenizer, output.generated_ids, hf_generated_ids, args.parity_require_exact)


if __name__ == "__main__":
    main()
