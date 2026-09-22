# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run tiny CausalLM KV-headpar Dynamo Loop models on QAIC.

This mirrors examples/dynamo/causal_lm/basic_dynamo_inference.py, but keeps the
coverage in pytest form and pins the KV-loop dimensions requested for regression:
PL=64, CL=128, num_kv_blocks=2.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    assert_hf_hw_parity,
    exported_onnx_path,
    get_hf_tokens,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

PREFILL_SEQ_LEN = 64
CTX_LEN = 128
NUM_KV_BLOCKS = 2
HEADPAR_SPLIT = 2
BATCH_SIZE = 1
GENERATION_LEN = 2
NUM_CORES = 2
PROMPT = "My name is"

KV_HEADPAR_LOOP_TINY_MODEL_TYPES = {
    "gpt_oss",
    "granite",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "qwen2",
    "starcoder2",
}
KV_HEADPAR_LOOP_TINY_MODEL_IDS = {
    model_type: model_id
    for model_type, model_id in DYNAMO_CAUSAL_LM_MODEL_IDS.items()
    if model_type in KV_HEADPAR_LOOP_TINY_MODEL_TYPES
}


def _collect_onnx_graph_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    nodes = []

    def visit(node_list):
        for node in node_list:
            nodes.append(node)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g.node)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attr.graphs:
                        visit(nested_graph.node)

    visit(graph.node)
    return nodes


def _decoder_layer_count(qeff_model: QEFFAutoModelForCausalLM) -> int:
    layer_count = getattr(qeff_model, "num_layers", None)
    if layer_count is not None:
        return int(layer_count)

    config = qeff_model.model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        layer_count = getattr(config, attr, None)
        if layer_count is not None:
            return int(layer_count)

    raise AssertionError(f"Could not resolve decoder layer count for {qeff_model.model.__class__.__name__}")


def _iter_decoder_layers(qeff_model: QEFFAutoModelForCausalLM):
    expected_count = _decoder_layer_count(qeff_model)
    for module in qeff_model.model.modules():
        for attr in ("layers", "blocks", "h"):
            layers = getattr(module, attr, None)
            if layers is None:
                continue
            try:
                layer_list = list(layers)
            except TypeError:
                continue
            if len(layer_list) == expected_count and all(
                hasattr(layer, "self_attn") or hasattr(layer, "attn") for layer in layer_list
            ):
                return layer_list
    return None


def _expected_kv_loop_count(qeff_model: QEFFAutoModelForCausalLM) -> int:
    decoder_layers = _decoder_layer_count(qeff_model)
    layers = _iter_decoder_layers(qeff_model)
    if layers is None:
        return decoder_layers

    eligible_layers = 0
    for layer in layers:
        attention = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if getattr(attention, "sliding_window", None) is None:
            eligible_layers += 1
    return eligible_layers


def _assert_decoder_layers_match_loop_ops(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    decoder_layers = _decoder_layer_count(qeff_model)
    expected_loop_ops = _expected_kv_loop_count(qeff_model)
    loop_ops = [node for node in _collect_onnx_graph_nodes(onnx_model.graph) if node.op_type == "Loop"]

    assert len(loop_ops) == expected_loop_ops, (
        f"Expected one ONNX Loop op per kv-loop eligible decoder layer in {onnx_path.name}; "
        f"decoder layers={decoder_layers}, expected loop ops={expected_loop_ops}, actual loop ops={len(loop_ops)}"
    )


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(KV_HEADPAR_LOOP_TINY_MODEL_IDS.items()),
    ids=sorted(KV_HEADPAR_LOOP_TINY_MODEL_IDS),
)
def test_kv_headpar_dynamo_loop_subfunctions_tiny_generate_on_qaic(model_type, model_id, tmp_export_dir):
    del model_type
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": NUM_KV_BLOCKS,
        "headpar_split": HEADPAR_SPLIT,
    }

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    hf_tokens = get_hf_tokens(
        tokenizer, model_hf, [PROMPT], prompt_len=PREFILL_SEQ_LEN, ctx_len=CTX_LEN
    )
    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=False, qaic_config=qaic_config)
    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=PREFILL_SEQ_LEN,
        bs=BATCH_SIZE,
        num_devices=1,
        qaic_config=qaic_config,
        dynamo=True,
        num_cores=NUM_CORES,
        prefill_seq_len=PREFILL_SEQ_LEN,
    )
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "kv_headpar_loop_tiny_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
            prefill_seq_len=PREFILL_SEQ_LEN,
        )
    )
    _assert_decoder_layers_match_loop_ops(onnx_path, qeff_model)

    qpc_path = qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "kv_headpar_loop_tiny_compile"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[PROMPT],
        generation_len=GENERATION_LEN,
    )

    assert Path(qpc_path).is_dir()
    assert exec_info is not None
    assert exec_info.generated_texts is not None
    assert len(exec_info.generated_texts) == BATCH_SIZE
    assert_hf_hw_parity(
        model_id, hf_tokens, exec_info, gen_len=GENERATION_LEN,
        context="kv_headpar torch.while_loop",
    )
