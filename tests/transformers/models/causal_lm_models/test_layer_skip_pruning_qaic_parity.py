# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json
import os

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import load_hf_causal_lm_model

SKIPPED_LAYER = 1
NUM_TEST_LAYERS = 2
PROMPT_LEN = 32
CTX_LEN = 128
GENERATION_LEN = 4


LAYER_SKIP_FEATURE_CASES = [
    ("regular", {}),
    ("regular_subfunctions", {"use_onnx_subfunctions": True}),
    ("regular_mxint8", {"mxint8_kv_cache": True}),
    ("regular_subfunctions_mxint8", {"use_onnx_subfunctions": True, "mxint8_kv_cache": True}),
    ("disagg_prefill", {"prefill_only": True}),
    ("disagg_prefill_subfunctions", {"prefill_only": True, "use_onnx_subfunctions": True}),
    ("disagg_prefill_mxint8", {"prefill_only": True, "mxint8_kv_cache": True}),
    (
        "disagg_prefill_subfunctions_mxint8",
        {"prefill_only": True, "use_onnx_subfunctions": True, "mxint8_kv_cache": True},
    ),
    ("disagg_decode", {"prefill_only": False, "prefill_seq_len": 1}),
    ("disagg_decode_subfunctions", {"prefill_only": False, "prefill_seq_len": 1, "use_onnx_subfunctions": True}),
    ("disagg_decode_mxint8", {"prefill_only": False, "prefill_seq_len": 1, "mxint8_kv_cache": True}),
    (
        "disagg_decode_subfunctions_mxint8",
        {"prefill_only": False, "prefill_seq_len": 1, "use_onnx_subfunctions": True, "mxint8_kv_cache": True},
    ),
]

LAYERWISE_PREFILL_LEN = 256
LAYERWISE_CTX_LEN = 256

LAYER_SKIP_LAYERWISE_CASES = [
    ("layerwise_regular", {}),
    ("layerwise_regular_subfunctions", {"use_onnx_subfunctions": True}),
    ("layerwise_regular_mxint8", {"mxint8_kv_cache": True}),
    (
        "layerwise_regular_subfunctions_mxint8",
        {"use_onnx_subfunctions": True, "mxint8_kv_cache": True},
    ),
    (
        "layerwise_disagg_prefill",
        {"prefill_only": True, "prefill_seq_len": LAYERWISE_PREFILL_LEN, "ctx_len": LAYERWISE_CTX_LEN},
    ),
    (
        "layerwise_disagg_prefill_subfunctions",
        {
            "prefill_only": True,
            "prefill_seq_len": LAYERWISE_PREFILL_LEN,
            "ctx_len": LAYERWISE_CTX_LEN,
            "use_onnx_subfunctions": True,
        },
    ),
    (
        "layerwise_disagg_prefill_mxint8",
        {
            "prefill_only": True,
            "prefill_seq_len": LAYERWISE_PREFILL_LEN,
            "ctx_len": LAYERWISE_CTX_LEN,
            "mxint8_kv_cache": True,
        },
    ),
    (
        "layerwise_disagg_prefill_subfunctions_mxint8",
        {
            "prefill_only": True,
            "prefill_seq_len": LAYERWISE_PREFILL_LEN,
            "ctx_len": LAYERWISE_CTX_LEN,
            "use_onnx_subfunctions": True,
            "mxint8_kv_cache": True,
        },
    ),
    ("layerwise_disagg_decode", {"prefill_only": False, "prefill_seq_len": 1}),
    (
        "layerwise_disagg_decode_subfunctions",
        {"prefill_only": False, "prefill_seq_len": 1, "use_onnx_subfunctions": True},
    ),
    (
        "layerwise_disagg_decode_mxint8",
        {"prefill_only": False, "prefill_seq_len": 1, "mxint8_kv_cache": True},
    ),
    (
        "layerwise_disagg_decode_subfunctions_mxint8",
        {"prefill_only": False, "prefill_seq_len": 1, "use_onnx_subfunctions": True, "mxint8_kv_cache": True},
    ),
]


def _zero_decoder_layer(model: torch.nn.Module, layer_idx: int) -> None:
    container = _resolve_decoder_layers(model)
    if layer_idx >= len(container):
        raise ValueError(f"layer_idx {layer_idx} out of range for model with {len(container)} layers")
    for parameter in container[layer_idx].parameters():
        parameter.data.zero_()


def _resolve_decoder_layers(model: torch.nn.Module):
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "layers"),
    )
    for path in candidates:
        current = model
        for name in path:
            if not hasattr(current, name):
                current = None
                break
            current = getattr(current, name)
        if current is not None:
            return current
    raise ValueError(f"Could not locate decoder layers for {type(model).__name__}.")


def _write_layer_skip_qaic_config(tmp_path, name: str, skip_layers=None):
    if skip_layers is None:
        skip_layers = [SKIPPED_LAYER]
    layer_skip_config_path = tmp_path / f"{name}_layer_skip.json"
    layer_skip_config_path.write_text(json.dumps({"skip_layers": list(skip_layers)}))
    return {
        "enable_layer_skipping": True,
        "layer_skip_config": str(layer_skip_config_path),
    }


def _assert_compile_output(qpc_path):
    qpc_paths = qpc_path.values() if isinstance(qpc_path, dict) else [qpc_path]
    for path in qpc_paths:
        assert path is not None
        assert os.path.isfile(os.path.join(os.path.dirname(path), "qconfig.json"))


def _compile_layer_skip_on_qaic(model, qaic_config, compile_dir, manual_cleanup, **compile_kwargs):
    qeff_model = QEFFAutoModelForCausalLM(
        model,
        pretrained_model_name_or_path=getattr(model.config, "_name_or_path", None),
        qaic_config=qaic_config,
    )
    qpc_path = qeff_model.compile(
        prefill_seq_len=compile_kwargs.pop("prefill_seq_len", PROMPT_LEN),
        ctx_len=compile_kwargs.pop("ctx_len", CTX_LEN),
        compile_dir=str(compile_dir),
        num_cores=16,
        mxfp6_matmul=False,
        aic_enable_depth_first=False,
        qaic_config=qaic_config,
        **compile_kwargs,
    )
    _assert_compile_output(qpc_path)
    manual_cleanup(qeff_model.onnx_path)
    return qpc_path


def _tiny_qwen3_moe_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "tiny_qwen3_moe"
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=NUM_TEST_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        dtype="float32",
    )
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config).eval()
    _zero_decoder_layer(model, SKIPPED_LAYER)
    model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def _compile_and_generate_on_qaic(
    model, tokenizer, qaic_config, compile_dir, manual_cleanup, mxint8_kv_cache=False, **compile_kwargs
):
    qeff_model = QEFFAutoModelForCausalLM(
        model,
        pretrained_model_name_or_path=getattr(model.config, "_name_or_path", None),
        qaic_config=qaic_config,
    )
    mxint8_kv_cache = compile_kwargs.pop("mxint8_kv_cache", mxint8_kv_cache)
    qpc_path = qeff_model.compile(
        prefill_seq_len=compile_kwargs.pop("prefill_seq_len", PROMPT_LEN),
        ctx_len=compile_kwargs.pop("ctx_len", CTX_LEN),
        compile_dir=str(compile_dir),
        num_cores=16,
        mxfp6=False,
        mxint8_kv_cache=mxint8_kv_cache,
        aic_enable_depth_first=False,
        qaic_config=qaic_config,
        **compile_kwargs,
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR, generation_len=GENERATION_LEN)
    generated_ids = np.asarray(exec_info.generated_ids[0])
    manual_cleanup(qeff_model.onnx_path)
    return generated_ids


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_layer_skip_pruning_qaic_parity_with_zeroed_layer(tmp_path, manual_cleanup):
    model_name = os.environ.get("QEFF_LAYER_SKIP_QAIC_MODEL", "gpt2")
    torch.manual_seed(42)

    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=NUM_TEST_LAYERS)
    _zero_decoder_layer(model_hf, SKIPPED_LAYER)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)

    layer_skip_config_path = tmp_path / "layer_skip.json"
    layer_skip_config_path.write_text(json.dumps({"skip_layers": [SKIPPED_LAYER]}))
    pruned_qaic_config = {
        "enable_layer_skipping": True,
        "layer_skip_config": str(layer_skip_config_path),
    }

    baseline_ids = _compile_and_generate_on_qaic(
        copy.deepcopy(model_hf),
        tokenizer,
        qaic_config=None,
        compile_dir=tmp_path / "baseline",
        manual_cleanup=manual_cleanup,
    )
    pruned_ids = _compile_and_generate_on_qaic(
        copy.deepcopy(model_hf),
        tokenizer,
        qaic_config=pruned_qaic_config,
        compile_dir=tmp_path / "pruned",
        manual_cleanup=manual_cleanup,
    )

    assert baseline_ids.shape == pruned_ids.shape
    assert np.array_equal(baseline_ids, pruned_ids)


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_layer_skip_pruning_qaic_parity_with_mxint8_kv_cache(tmp_path, manual_cleanup):
    model_name = os.environ.get("QEFF_LAYER_SKIP_QAIC_MODEL", "gpt2")
    torch.manual_seed(42)

    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=NUM_TEST_LAYERS)
    _zero_decoder_layer(model_hf, SKIPPED_LAYER)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)

    layer_skip_config_path = tmp_path / "layer_skip_mxint8.json"
    layer_skip_config_path.write_text(json.dumps({"skip_layers": [SKIPPED_LAYER]}))
    pruned_qaic_config = {
        "enable_layer_skipping": True,
        "layer_skip_config": str(layer_skip_config_path),
    }

    baseline_ids = _compile_and_generate_on_qaic(
        copy.deepcopy(model_hf),
        tokenizer,
        qaic_config=None,
        compile_dir=tmp_path / "baseline_mxint8",
        manual_cleanup=manual_cleanup,
        mxint8_kv_cache=True,
    )
    pruned_ids = _compile_and_generate_on_qaic(
        copy.deepcopy(model_hf),
        tokenizer,
        qaic_config=pruned_qaic_config,
        compile_dir=tmp_path / "pruned_mxint8",
        manual_cleanup=manual_cleanup,
        mxint8_kv_cache=True,
    )

    assert baseline_ids.shape == pruned_ids.shape
    assert np.array_equal(baseline_ids, pruned_ids)


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "case_name, compile_kwargs",
    LAYER_SKIP_FEATURE_CASES,
    ids=[case_name for case_name, _ in LAYER_SKIP_FEATURE_CASES],
)
def test_layer_skip_pruning_qaic_compile_feature_matrix(case_name, compile_kwargs, tmp_path, manual_cleanup):
    model_name = os.environ.get("QEFF_LAYER_SKIP_QAIC_MODEL", "gpt2")
    torch.manual_seed(42)

    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=NUM_TEST_LAYERS)
    _zero_decoder_layer(model_hf, SKIPPED_LAYER)
    pruned_qaic_config = _write_layer_skip_qaic_config(tmp_path, case_name)

    _compile_layer_skip_on_qaic(
        model_hf,
        qaic_config=pruned_qaic_config,
        compile_dir=tmp_path / case_name,
        manual_cleanup=manual_cleanup,
        **compile_kwargs,
    )


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "case_name, compile_kwargs",
    LAYER_SKIP_LAYERWISE_CASES,
    ids=[case_name for case_name, _ in LAYER_SKIP_LAYERWISE_CASES],
)
def test_layer_skip_pruning_qaic_compile_layerwise_matrix(case_name, compile_kwargs, tmp_path, manual_cleanup):
    checkpoint_dir = _tiny_qwen3_moe_checkpoint(tmp_path)
    pruned_qaic_config = _write_layer_skip_qaic_config(tmp_path, case_name)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        qaic_config=pruned_qaic_config,
        layerwise=True,
        torch_dtype=torch.float32,
    )
    qpc_path = qeff_model.compile(
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        compile_dir=str(tmp_path / case_name),
        num_cores=16,
        mxfp6_matmul=False,
        aic_enable_depth_first=False,
        qaic_config=pruned_qaic_config,
        layerwise=True,
        layerwise_window_size=1,
        **compile_kwargs,
    )
    _assert_compile_output(qpc_path)
    manual_cleanup(qeff_model.onnx_path)
