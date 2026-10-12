# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free Qwen3-VL-MoE continuous-batching KV batch-fold example.

This mirrors the continuous-batching KV-share batch-fold example, but exports the
language graphs with weight_free=True. This file is intentionally scoped to the
decode mode that requires continuous batching:

    qaic_config = {
        "blocking_mode": "kv_batch_fold",
        "num_kv_blocks": 4,
        "ctx_len": 10240,
    }

    prefill_qaic_config = {
        "blocking_mode": "prefill_qkv",
        "ctx_len": 10240,
        "num_kv_blocks": 4,
        "num_q_blocks": 16,
        "n_rep_chunk": 4,
    }
"""

import argparse
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "What is the capital of France?",
    "Who is the President of USA?",
    "Explain quantum computing in simple terms?",
] * 64
DEFAULT_IMAGE_PROMPTS = [
    "Describe all the colors seen in the image",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "describe the image?",
]
DEFAULT_IMAGE_URLS = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/230/536/354",
    "https://picsum.photos/id/234/536/354",
    "https://picsum.photos/id/235/536/354",
]
DEFAULT_PREFILL_SEQ_LEN = 1024
DEFAULT_CTX_LEN = 10240
DEFAULT_GENERATION_LEN = 50
DEFAULT_FULL_BATCH_SIZE = 256
DEFAULT_HEAD_BLOCK_SIZE = 8
DEFAULT_NUM_KV_BLOCKS = 4
DEFAULT_PREFILL_NUM_Q_BLOCKS = 16
DEFAULT_N_REP_CHUNK = 4
DEFAULT_MOE_PREFILL_PACKED_CHUNK_SIZE = 256
VISION_INPUT_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")


def _build_config(args):
    config = AutoConfig.from_pretrained(args.model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16

    if args.full_model:
        return config

    if args.vision_depth is not None:
        config.vision_config.depth = args.vision_depth
    if args.num_text_layers is not None:
        config.text_config.num_hidden_layers = args.num_text_layers
    if args.deepstack_visual_index is not None:
        config.vision_config.deepstack_visual_indexes = [args.deepstack_visual_index]
    return config


def _decode_qaic_config(args) -> dict:
    cfg = {
        "blocking_mode": args.decode_blocking_mode,
        "ctx_len": args.ctx_len,
        "moe_config": {"flavour": "expert_parallel", "tree_reduce": True, "cores_per_expert": 2},
    }
    cfg["num_kv_blocks"] = args.num_kv_blocks
    return cfg


def _prefill_qaic_config(args) -> dict:
    cfg = {
        "blocking_mode": args.prefill_blocking_mode,
        "ctx_len": args.ctx_len,
        "moe_config": {"expert_parallel_chunk_size": args.moe_prefill_packed_chunk_size},
    }
    if args.prefill_blocking_mode in {"prefill_kv", "prefill_qkv", "prefill_online"}:
        cfg["num_kv_blocks"] = args.num_kv_blocks
    if args.prefill_blocking_mode in {"prefill_q", "prefill_qkv", "prefill_online"}:
        cfg["num_q_blocks"] = args.prefill_num_q_blocks
    if args.prefill_blocking_mode in {"prefill_kv", "prefill_qkv"}:
        cfg["headpar_split"] = args.headpar_split
    cfg["n_rep_chunk"] = args.n_rep_chunk
    return cfg


def _slice_position_ids_for_session(position_ids, session, batch_slice, token_slice):
    sliced = position_ids[..., batch_slice, token_slice] if position_ids.ndim == 3 else position_ids[batch_slice, token_slice]
    return _fit_position_ids_to_session(sliced, session)


def _next_position_ids_for_session(position_ids, session):
    next_pos = np.max(position_ids, axis=-1, keepdims=True) + 1
    return _fit_position_ids_to_session(next_pos, session)


def _fit_position_ids_to_session(position_ids, session):
    expected_shape = tuple(int(dim) for dim in session.bindings[session.binding_index_map["position_ids"]].dims)
    return _fit_array_to_session_binding("position_ids", position_ids, session, expected_shape=expected_shape)


def _fit_array_to_session_binding(name, array, session, expected_shape=None):
    if name not in session.binding_index_map:
        return array
    binding = session.bindings[session.binding_index_map[name]]
    if expected_shape is None:
        expected_shape = tuple(int(dim) for dim in binding.dims)
    expected_rank = len(expected_shape)
    fitted = np.asarray(array)
    while fitted.ndim > expected_rank:
        fitted = fitted[0]
    while fitted.ndim < expected_rank:
        fitted = np.expand_dims(fitted, axis=0)

    slices = []
    for axis, expected_dim in enumerate(expected_shape):
        actual_dim = fitted.shape[axis]
        if actual_dim == expected_dim:
            slices.append(slice(None))
            continue
        if actual_dim > expected_dim:
            slices.append(slice(0, expected_dim))
            continue
        if actual_dim == 1:
            slices.append(slice(None))
            continue
        raise ValueError(
            f"Cannot fit {name} shape {tuple(array.shape)} to compiled binding shape {expected_shape}"
        )

    fitted = fitted[tuple(slices)]
    if fitted.shape != expected_shape and all(actual == 1 or actual == expected for actual, expected in zip(fitted.shape, expected_shape)):
        fitted = np.broadcast_to(fitted, expected_shape)
    if fitted.shape != expected_shape:
        raise ValueError(f"{name} shape {tuple(array.shape)} became {fitted.shape}, expected {expected_shape}")
    expected_dtype = session.aic_to_np_dtype_mapping[binding.type]
    if fitted.dtype != expected_dtype:
        fitted = fitted.astype(expected_dtype)
    return np.ascontiguousarray(fitted)


def _fit_inputs_to_session_bindings(inputs, session):
    for name, value in list(inputs.items()):
        if name in session.binding_index_map and not name.startswith(("past_", "compressed_", "k_pe")):
            inputs[name] = _fit_array_to_session_binding(name, value, session)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="image URL (repeatable); paired with prompts by index. Defaults to a set of 4",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH_SIZE)
    parser.add_argument("--prefill-kv-cache-batch-size", type=int, default=7)
    parser.add_argument(
        "--decode-blocking-mode",
        default="kv_batch_fold",
        choices=("kv_batch_fold",),
    )
    parser.add_argument(
        "--prefill-blocking-mode",
        default="prefill_qkv",
        choices=("prefill_q", "prefill_kv", "prefill_qkv", "prefill_online"),
    )
    parser.add_argument("--head-block-size", type=int, default=DEFAULT_HEAD_BLOCK_SIZE)
    parser.add_argument("--num-kv-blocks", type=int, default=DEFAULT_NUM_KV_BLOCKS)
    parser.add_argument("--prefill-num-q-blocks", type=int, default=DEFAULT_PREFILL_NUM_Q_BLOCKS)
    parser.add_argument("--headpar-split", type=int, default=4)
    parser.add_argument("--n-rep-chunk", type=int, default=DEFAULT_N_REP_CHUNK)
    parser.add_argument("--moe-prefill-packed-chunk-size", type=int, default=DEFAULT_MOE_PREFILL_PACKED_CHUNK_SIZE)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--prefill-num-devices", type=int, default=2)
    parser.add_argument("--decode-num-devices", type=int, default=1)
    parser.add_argument("--vision-num-devices", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--num-text-layers", type=int, default=2)
    parser.add_argument("--vision-depth", type=int, default=9)
    parser.add_argument("--deepstack-visual-index", type=int, default=8)
    parser.add_argument("--full-model", action="store_true", help="use the full HF model config without truncating layers")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--compile-only", action="store_true", help="compile QPCs and exit without running inference")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _build_config(args)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype=torch.float16,
        weight_free=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    common = {
        "ctx_len": args.ctx_len,
        "height": args.height,
        "width": args.width,
        "num_cores": args.num_cores,
        "mos": 1,
        "use_onnx_subfunctions": True,
        "layerwise": False,
    }

    if not args.skip_vision:
        vision_qpc = qeff_model.compile(
            batch_size=1,
            full_batch_size=args.full_batch_size,
            prefill_seq_len=args.prefill_seq_len,
            num_devices=args.vision_num_devices,
            skip_vision=False,
            skip_lang=True,
            split_model_io=True,
            aic_enable_depth_first=True,
            mxfp6_matmul=True,
            **common,
        )
        print(f"Vision QPC: {vision_qpc['vision_qpc_path']}")

    decode_qaic_config = _decode_qaic_config(args)
    decode_qpc = qeff_model.compile(
        batch_size=args.full_batch_size,
        full_batch_size=args.full_batch_size,
        kv_cache_batch_size=args.full_batch_size,
        prefill_seq_len=1,
        num_devices=args.decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        user_tiled=True,
        split_retained_state_io=True,
        split_model_io=True,
        prefill_only=False,
        skip_vision=True,
        qaic_config=decode_qaic_config,
        **common,
    )
    print(f"Decode QPC: {decode_qpc['lang_decode_qpc_path']}")
    print(f"Decode qaic_config: {decode_qaic_config}")

    prefill_qaic_config = _prefill_qaic_config(args)
    prefill_qpc = qeff_model.compile(
        batch_size=1,
        full_batch_size=1,
        kv_cache_batch_size=args.prefill_kv_cache_batch_size,
        prefill_seq_len=args.prefill_seq_len,
        num_devices=args.prefill_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,
        split_model_io=True,
        aic_enable_depth_first=True,
        mdp_num_partitions=args.stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        qaic_config=prefill_qaic_config,
        **common,
    )
    print(f"Prefill QPC: {prefill_qpc['lang_prefill_qpc_path']}")
    print(f"Prefill qaic_config: {prefill_qaic_config}")

    if args.compile_only:
        return

    prompts = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
    image_urls = list(args.image_urls) if args.image_urls else list(DEFAULT_IMAGE_URLS)
    if not args.skip_vision and args.prompts is None:
        prompts = list(DEFAULT_IMAGE_PROMPTS)
    if len(prompts) == 1:
        prompts = prompts * args.full_batch_size
    if len(image_urls) == 1:
        image_urls = image_urls * args.full_batch_size

    vision_session = None
    if not args.skip_vision:
        vision_session = QAICInferenceSession(vision_qpc["vision_qpc_path"])
    prefill_session = QAICInferenceSession(
        prefill_qpc["lang_prefill_qpc_path"],
        kv_dma_share=True,
        full_batch_size=args.full_batch_size,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc["lang_decode_qpc_path"],
        kv_dma_share=True,
        full_batch_size=args.full_batch_size,
        cluster_id="decode",
    )

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == args.full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size "
        f"{args.full_batch_size}"
    )
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def _prepare_prompt_batch(prompt: str, image_url: str):
        if args.skip_vision:
            content = [{"type": "text", "text": prompt}]
        else:
            image = Image.open(requests.get(image_url, stream=True, timeout=30).raw).convert("RGB")
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [[{"role": "user", "content": content}]]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs,
            prefill_seq_len=args.prefill_seq_len,
            batch_size=1,
        )

        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -args.prefill_seq_len)
        padded_len = num_chunks * args.prefill_seq_len
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.nn.functional.pad(
                inputs["mm_token_type_ids"], (0, padded_len - input_ids_length), "constant", 0
            )

        inputs = {name: np.asarray(value) for name, value in inputs.items()}
        vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUT_KEYS}
        vision_inputs.update(
            {name: vision_inputs[name].astype(np.float16) for name in VISION_FP16_KEYS if name in vision_inputs}
        )

        vision_start = perf_counter()
        vision_outputs = {}
        if not args.skip_vision and vision_inputs:
            raw = vision_session.run(vision_inputs)
            vision_outputs = {name: raw[name] for name in VISION_OUTPUT_KEYS if name in raw}
        vision_time = perf_counter() - vision_start

        lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
        if "mm_token_type_ids" in lang_inputs and "mm_token_type_ids" not in prefill_session.input_names:
            lang_inputs.pop("mm_token_type_ids")
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

        if not args.skip_vision:
            for name in VISION_OUTPUT_KEYS:
                if name in vision_outputs:
                    lang_inputs[name] = vision_outputs[name]

        return lang_inputs, vision_outputs, num_chunks, vision_time

    prompt = prompts[0]
    prompt_image_url = image_urls[0]
    lang_inputs, vision_outputs, num_chunks, vision_time = _prepare_prompt_batch(prompt, prompt_image_url)

    all_outputs = []
    chunk_inputs = lang_inputs.copy()
    chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
    slot_kv_view = [kv_cache[0:1] for kv_cache in kv_caches]
    prefill_out = None

    lang_start = perf_counter()
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.prefill_seq_len
        chunk_end = (chunk_idx + 1) * args.prefill_seq_len
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][0:1, chunk_start:chunk_end]
        chunk_inputs["position_ids"] = _slice_position_ids_for_session(
            lang_inputs["position_ids"], prefill_session, slice(0, 1), slice(chunk_start, chunk_end)
        )
        if "mm_token_type_ids" in lang_inputs:
            chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][0:1, chunk_start:chunk_end]
        last_chunk = chunk_idx == num_chunks - 1
        _fit_inputs_to_session_bindings(chunk_inputs, prefill_session)
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        prefill_out = prefill_session.get_outputs(index=exec_idx)
        chunk_inputs["image_idx"] = prefill_out["image_idx_output"]
    print(f"Prefill time : {perf_counter() - lang_start + vision_time:.2f} secs")

    next_token_id = int(np.argmax(prefill_out["logits"]))
    all_outputs.append(next_token_id)
    next_pos = _next_position_ids_for_session(lang_inputs["position_ids"], decode_session)

    batch_index = np.random.default_rng(1234).permutation(args.full_batch_size).reshape(args.full_batch_size, 1)
    batch_index = batch_index.astype(np.int64)
    physical_slots = batch_index[:, 0]
    for kv_cache in kv_caches:
        source_row = kv_cache[0:1].copy()
        reps = (args.full_batch_size,) + (1,) * (kv_cache.ndim - 1)
        kv_cache[physical_slots] = np.tile(source_row, reps)

    if not args.skip_vision and vision_outputs:
        persistent = {}
        for name in VISION_OUTPUT_KEYS:
            if name not in decode_session.binding_index_map:
                continue
            binding = decode_session.bindings[decode_session.binding_index_map[name]]
            dtype = decode_session.aic_to_np_dtype_mapping[binding.type]
            persistent[name] = np.zeros(tuple(binding.dims), dtype=dtype)
        decode_session.set_persistent_inputs(persistent)

    decode_inputs = {
        "input_ids": np.full((args.full_batch_size, 1), next_token_id, dtype=lang_inputs["input_ids"].dtype),
        "position_ids": next_pos,
        "batch_index": batch_index,
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

    def _run_decode_step(inputs):
        _fit_inputs_to_session_bindings(inputs, decode_session)
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        return decode_session.get_outputs(index=exec_idx)

    try:
        decode_start = perf_counter()
        decode_out = _run_decode_step(decode_inputs)
        print(f"time for first run of decode with KV as input = {perf_counter() - decode_start} sec\n")

        if args.generation_len > 1:
            all_outputs.append(int(np.argmax(decode_out["logits"][0])))
        pos_id = decode_inputs["position_ids"] + 1
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"], axis=-1),
            "position_ids": pos_id,
            "batch_index": batch_index,
        }
        if decode_has_image_idx:
            loop_decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

        decode_loop_start = perf_counter()
        for _ in range(max(args.generation_len - 2, 0)):
            decode_out = _run_decode_step(loop_decode_inputs)
            all_outputs.append(int(np.argmax(decode_out["logits"][0])))
            pos_id += 1
            loop_decode_inputs.update(
                {
                    "input_ids": np.argmax(decode_out["logits"], axis=-1),
                    "position_ids": pos_id,
                    "batch_index": batch_index,
                }
            )
            if decode_has_image_idx:
                loop_decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        decode_time = perf_counter() - decode_loop_start
        if args.generation_len > 2:
            print(f"decode tok/sec={(args.generation_len - 2) / decode_time}")
        print(f"\ninput\n{prompt}\noutput\n{tokenizer.decode(all_outputs, skip_special_tokens=True)}")
    finally:
        if vision_session is not None:
            vision_session.deactivate()
        prefill_session.deactivate()
        decode_session.deactivate()


if __name__ == "__main__":
    main()
