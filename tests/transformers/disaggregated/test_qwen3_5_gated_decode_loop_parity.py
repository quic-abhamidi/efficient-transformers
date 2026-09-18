"""Qwen3.5 decode-only gated ``torch.while_loop`` parity tests.

The full multimodal disaggregated test is intentionally opt-in because it
requires model weights, qwen-vl-utils, ORT, and QAIC. The unit tests below run
without those external services and validate the numerical decode recurrence.
"""

import os

import numpy as np
import pytest
import torch


def _inputs(batch=2, heads=4, key_dim=8, value_dim=8):
    torch.manual_seed(0)
    query = torch.randn(batch, 1, heads, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(batch, 1, heads, value_dim)
    gate = torch.randn(batch, 1, heads)
    beta = torch.sigmoid(torch.randn(batch, 1, heads))
    position_ids = torch.zeros(batch, 1, dtype=torch.long)
    state = torch.randn(batch, heads, key_dim, value_dim)
    return query, key, value, gate, beta, position_ids, state


def test_qwen3_5_decode_loop_matches_recurrent_step():
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import (
        QEffQwen3_5GatedDeltaNet,
    )

    query, key, value, gate, beta, position_ids, state = _inputs()
    module = object.__new__(QEffQwen3_5GatedDeltaNet)

    loop_out, loop_state = module._torch_chunk_gated_delta_rule_loop(
        query,
        key,
        value,
        gate,
        beta,
        position_ids,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_out, ref_state = module._recurrent_step_batched(
        query, key, value, gate, beta, state
    )

    torch.testing.assert_close(loop_out, ref_out, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(loop_state, ref_state, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(
    os.environ.get("QEFF_RUN_QWEN35_ORT") != "1",
    reason="set QEFF_RUN_QWEN35_ORT=1 for the opt-in Dynamo/ORT test",
)
def test_qwen3_5_decode_loop_export_contains_loop(tmp_path):
    """Export a tiny decode wrapper and verify ORT executes the loop graph."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import (
        QEffQwen3_5GatedDeltaNet,
    )

    query, key, value, gate, beta, position_ids, state = _inputs(batch=1)
    module = object.__new__(QEffQwen3_5GatedDeltaNet)

    class Wrapper(torch.nn.Module):
        def forward(self, q, k, v, g, b, p, s):
            return module._torch_chunk_gated_delta_rule_loop(
                q, k, v, g, b, p, initial_state=s, output_final_state=True
            )

    path = tmp_path / "qwen35_decode_loop.onnx"
    torch.onnx.export(
        Wrapper(),
        (query, key, value, gate, beta, position_ids, state),
        path,
        input_names=["q", "k", "v", "g", "b", "p", "s"],
        output_names=["out", "state_out"],
        opset_version=18,
        dynamo=True,
    )
    model = onnx.load(path)
    assert any(node.op_type == "Loop" for node in model.graph.node)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {name: tensor.numpy() for name, tensor in zip(
        ["q", "k", "v", "g", "b", "p", "s"],
        [query, key, value, gate, beta, position_ids, state],
    )})
    torch_out, torch_state = module._torch_chunk_gated_delta_rule_loop(
        query, key, value, gate, beta, position_ids,
        initial_state=state, output_final_state=True,
    )
    ref_out, ref_state = module._recurrent_step_batched(
        query, key, value, gate, beta, state,
    )
    np.testing.assert_allclose(outputs[0], torch_out.numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(outputs[1], torch_state.numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(outputs[0], ref_out.numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(outputs[1], ref_state.numpy(), rtol=2e-4, atol=2e-4)
