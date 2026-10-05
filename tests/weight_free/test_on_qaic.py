# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally) and use_onnx_subfunctions=True.

Covers:
  - Generate smoke test (weight-free export -> compile -> generate on QAIC)
  - HF PT vs QAIC HW parity (HF PT tokens == weight-free QAIC top-1 token)
  - Weight-free vs legacy/dynamo QAIC parity (two independently compiled QPCs
    of the same model, generated output compared directly)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.device_utils import get_available_device_id, get_qaic_mdp_device_groups

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_BLOCKING_MODE_CASES,
    WEIGHT_FREE_BLOCKING_QAIC_CASES,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    WEIGHT_FREE_VLM_MODEL_PARAMS,
    assert_blocked_kv_ops_for_mode,
    assert_has_subfunctions,
    exported_onnx_path,
    load_causal_lm_config,
    load_hf_model,
    load_tokenizer,
    load_weight_free_vlm_model,
    skip_on_model_fetch_error,
)

VLM_PREFILL_SEQ_LEN = 64
VLM_CTX_LEN = 512
VLM_IMAGE_HEIGHT = 354
VLM_IMAGE_WIDTH = 536
VLM_NUM_CORES = 4
VLM_GENERATION_LEN = 2
VLM_PROMPT = "Describe this image."


def _load_vlm_hf_reference(model_id: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.text_config.num_hidden_layers = 1
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    return model.eval()


def _prepare_vlm_inputs(processor: AutoProcessor, image: Image.Image) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _run_vlm_hf_reference(model, inputs: dict) -> np.ndarray:
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=VLM_GENERATION_LEN,
            min_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_VLM_MODEL_PARAMS)
def test_weight_free_vlm_combined_compile(model_type, model_id, tmp_export_dir):
    """Run a weight-free dual-QPC VLM and compare generated tokens with HF."""
    try:
        hf_model = _load_vlm_hf_reference(model_id)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        qeff_model = load_weight_free_vlm_model(model_id)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    image = Image.new("RGB", (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT), color=(127, 127, 127))
    hf_tokens = _run_vlm_hf_reference(hf_model, _prepare_vlm_inputs(processor, image))
    qeff_inputs = _prepare_vlm_inputs(processor, image)
    qeff_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=qeff_inputs,
        prefill_seq_len=VLM_PREFILL_SEQ_LEN,
        batch_size=1,
    )

    qpc_paths = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_combined"),
        batch_size=1,
        prefill_seq_len=VLM_PREFILL_SEQ_LEN,
        ctx_len=VLM_CTX_LEN,
        height=VLM_IMAGE_HEIGHT,
        width=VLM_IMAGE_WIDTH,
        num_cores=VLM_NUM_CORES,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    assert qpc_paths.get("vision_qpc_path")
    assert qpc_paths.get("lang_qpc_path")

    qaic_output = qeff_model.generate(inputs=qeff_inputs, generation_len=VLM_GENERATION_LEN)
    assert qaic_output is not None, "Weight-free VLM QAIC generate returned None"
    assert qaic_output.generated_ids is not None, "Weight-free VLM QAIC generate returned no token IDs"

    qaic_tokens = qaic_output.generated_ids[:, :VLM_GENERATION_LEN]
    assert qaic_tokens.shape == hf_tokens.shape == (1, VLM_GENERATION_LEN)
    assert np.array_equal(qaic_tokens, hf_tokens), (
        f"Weight-free VLM QAIC/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


BLOCKING_PARITY_PROMPT = "Hello world"
HEAD_BLOCKING_NUM_DEVICES = 4


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _blocking_runtime_options(blocking_key: str) -> tuple[dict, list[int] | None]:
    if blocking_key in {"h", "hq", "hkv"}:
        device_groups = get_qaic_mdp_device_groups(devices_per_group=HEAD_BLOCKING_NUM_DEVICES)
        if not device_groups:
            pytest.skip(f"No available QAIC MDP device group with {HEAD_BLOCKING_NUM_DEVICES} devices")
        return {"num_devices": HEAD_BLOCKING_NUM_DEVICES, "user_tiled": True}, device_groups[0]

    device_ids = get_available_device_id()
    if device_ids is None:
        pytest.skip("No available QAIC device")
    return {}, device_ids


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_generate_fp16(model_type, model_id, tmp_export_dir):
    """End-to-end weight-free export -> compile -> generate on real QAIC hardware."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_gen_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    tokenizer = load_tokenizer(model_id)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """HF PT tokens == weight-free QAIC FP16 tokens (exact equality)."""
    from QEfficient.utils.run_utils import ApiRunner

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_hw_parity_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"Weight-free HW/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_vs_legacy_qaic_parity(model_type, model_id, tmp_export_dir):
    """Weight-free-compiled and legacy dynamo-compiled QPCs produce identical tokens on QAIC."""
    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
        if model_type == "gpt_oss":
            model_hf = model_hf.to(torch.float32)
            model_hf.config.torch_dtype = torch.float32
            model_hf.config.dtype = torch.float32
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    # Legacy/dynamo leg — real weights, no weight-free export.
    qeff_legacy = QEFFAutoModelForCausalLM(model_hf)

    legacy_onnx_path = exported_onnx_path(
        qeff_legacy.export(
            tmp_export_dir / "legacy_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_legacy.compile(
        onnx_path=str(legacy_onnx_path),
        compile_dir=str(tmp_export_dir / "legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    legacy_output = qeff_legacy.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert legacy_output is not None, "Legacy QAIC generate returned None"

    # Weight-free leg — meta-device model, matching layer count.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_weight_free = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    weight_free_onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            tmp_export_dir / "wf_vs_legacy_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_weight_free.compile(
        onnx_path=str(weight_free_onnx_path),
        compile_dir=str(tmp_export_dir / "wf_vs_legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    weight_free_output = qeff_weight_free.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert weight_free_output is not None, "Weight-free QAIC generate returned None"

    if (
        hasattr(legacy_output, "generated_ids")
        and legacy_output.generated_ids is not None
        and hasattr(weight_free_output, "generated_ids")
        and weight_free_output.generated_ids is not None
    ):
        legacy_tokens = legacy_output.generated_ids[0].flatten()
        weight_free_tokens = weight_free_output.generated_ids[0].flatten()
        assert np.array_equal(legacy_tokens, weight_free_tokens), (
            f"Weight-free vs legacy QAIC parity failed for {model_id}: "
            f"legacy={legacy_tokens.tolist()}, weight_free={weight_free_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.weight_free_blocking
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id,blocking_key", WEIGHT_FREE_BLOCKING_QAIC_CASES)
def test_weight_free_blocked_hw_hf_parity_and_artifact_metrics(
    model_type,
    model_id,
    blocking_key,
    tmp_export_dir,
    record_property,
):
    """HF PT tokens == blocked weight-free QAIC tokens, with export/compile artifact metrics."""
    from QEfficient.utils.run_utils import ApiRunner

    use_onnx_subfunctions = True
    qaic_config = dict(WEIGHT_FREE_BLOCKING_MODE_CASES[blocking_key])
    prompts = [BLOCKING_PARITY_PROMPT]

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:  # noqa: BLE001
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompts,
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    config = load_causal_lm_config(model_id)
    config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        weight_free=True,
        qaic_config=qaic_config,
    )

    export_start = time.perf_counter()
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"wf_blocking_{model_type}_{blocking_key}_{use_onnx_subfunctions}_export",
            qaic_config=qaic_config,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )
    export_seconds = time.perf_counter() - export_start
    onnx_size_bytes = onnx_path.stat().st_size

    blocking_config = qeff_model.hash_params.get("blocking_kwargs")
    assert blocking_config is not None
    assert blocking_config.mode == BlockingMode(qaic_config["blocking_mode"])
    if use_onnx_subfunctions:
        assert_has_subfunctions(onnx_path, qeff_model)
    assert_blocked_kv_ops_for_mode(onnx_path, qeff_model, blocking_key)

    compile_start = time.perf_counter()
    compile_options, device_ids = _blocking_runtime_options(blocking_key)
    qpc_path = Path(
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(
                tmp_export_dir / f"wf_blocking_{model_type}_{blocking_key}_{use_onnx_subfunctions}_compile"
            ),
            prefill_seq_len=PROMPT_LEN,
            ctx_len=CTX_LEN,
            num_cores=16,
            batch_size=BATCH_SIZE,
            qaic_config=qaic_config,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compile_options,
        )
    )
    compile_seconds = time.perf_counter() - compile_start
    qpc_size_bytes = _directory_size_bytes(qpc_path)

    for name, value in {
        "onnx_export_seconds": export_seconds,
        "onnx_size_bytes": onnx_size_bytes,
        "qpc_compile_seconds": compile_seconds,
        "qpc_size_bytes": qpc_size_bytes,
    }.items():
        record_property(name, value)
    print(
        f"weight_free_blocking_metrics model={model_type} mode={blocking_key} "
        f"use_onnx_subfunctions={use_onnx_subfunctions} "
        f"onnx_export_seconds={export_seconds:.3f} onnx_size_bytes={onnx_size_bytes} "
        f"qpc_compile_seconds={compile_seconds:.3f} qpc_size_bytes={qpc_size_bytes}"
    )

    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_ids=device_ids,
    )
    assert qaic_output is not None, "Blocked weight-free QAIC generate returned None"
    assert hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None

    gen_len = CTX_LEN - PROMPT_LEN
    qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
    qaic_text = getattr(qaic_output, "generated_texts", None)
    print("Blocked weight-free QAIC output:")
    print("Prompt:", repr(prompts))
    print("Completion:", repr(qaic_text))
    hf_tokens_for_compare = np.asarray(hf_tokens).flatten()[:gen_len]
    print("HF token IDs:", hf_tokens_for_compare.tolist())
    print("QAIC token IDs:", qaic_tokens.tolist())
    assert np.array_equal(hf_tokens_for_compare, qaic_tokens), (
        f"Blocked weight-free HW/HF parity failed for {model_id} mode={blocking_key}: "
        f"HF={hf_tokens_for_compare.tolist()}, QAIC={qaic_tokens.tolist()}"
    )
