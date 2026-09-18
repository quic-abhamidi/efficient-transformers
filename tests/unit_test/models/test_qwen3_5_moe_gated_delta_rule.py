# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import (
    QEffQwen3_5GatedDeltaNet,
    _use_gated_delta_loop,
    qeff_apply_interleaved_mrope,
    qeff_prepare_mrope_cos_sin,
    qeff_torch_causal_conv1d_update,
)
from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeGatedDeltaNet

MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL = 1e-4


def test_qwen3_5_gated_delta_loop_is_default_for_decode_only(monkeypatch):
    monkeypatch.delenv("QEFF_QWEN3_5_ENABLE_GATED_DELTA_LOOP", raising=False)
    monkeypatch.delenv("QEFF_QWEN3_5_DISABLE_GATED_DELTA_LOOP", raising=False)

    assert _use_gated_delta_loop(torch.empty(2, 1, 3, 4))
    assert not _use_gated_delta_loop(torch.empty(2, 64, 3, 4))

    monkeypatch.setenv("QEFF_QWEN3_5_DISABLE_GATED_DELTA_LOOP", "1")
    assert not _use_gated_delta_loop(torch.empty(2, 1, 3, 4))

    monkeypatch.delenv("QEFF_QWEN3_5_DISABLE_GATED_DELTA_LOOP", raising=False)
    monkeypatch.setenv("QEFF_QWEN3_5_ENABLE_GATED_DELTA_LOOP", "0")
    assert not _use_gated_delta_loop(torch.empty(2, 1, 3, 4))


def _old_apply_interleaved_mrope(freqs, mrope_section):
    half_shape = freqs[0].shape[-1] // 2
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += half_shape
        length += half_shape
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def test_qeff_apply_interleaved_mrope_matches_original_assignment_logic():
    torch.manual_seed(0)
    freqs = torch.randn(3, 2, 5, 64, dtype=torch.float32)
    mrope_section = [11, 11, 10]

    actual = qeff_apply_interleaved_mrope(freqs, mrope_section)
    expected = _old_apply_interleaved_mrope(freqs, mrope_section)

    assert torch.equal(actual, expected)


def test_qeff_prepare_mrope_cos_sin_preserves_decode_seq_len_shape():
    head_dim = 64
    cos = torch.randn(128, head_dim, dtype=torch.float32)
    sin = torch.randn(128, head_dim, dtype=torch.float32)
    mrope_section = [11, 11, 10]

    for seq_len in (1, 64):
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).repeat(3, 2, 1)
        out_cos, out_sin = qeff_prepare_mrope_cos_sin(cos, sin, position_ids, mrope_section, dtype=torch.float16)

        assert out_cos.shape == (2, 1, seq_len, head_dim)
        assert out_sin.shape == (2, 1, seq_len, head_dim)
        assert out_cos.dtype == torch.float16
        assert out_sin.dtype == torch.float16


def test_qeff_torch_causal_conv1d_update_matches_negative_tail_slice():
    torch.manual_seed(0)
    batch_size, hidden_size, state_len = 2, 4, 4
    weight = torch.randn(hidden_size, state_len, dtype=torch.float32)
    bias = torch.randn(hidden_size, dtype=torch.float32)
    conv_state = torch.randn(batch_size, hidden_size, state_len, dtype=torch.float32)

    for seq_len in (1, 64):
        hidden_states = torch.randn(batch_size, hidden_size, seq_len, dtype=torch.float32)
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).repeat(1, batch_size, 1)

        actual, _ = qeff_torch_causal_conv1d_update(hidden_states, conv_state, weight, position_ids, bias)
        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        expected = torch.nn.functional.silu(
            torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=hidden_size)[:, :, -seq_len:]
        ).to(hidden_states.dtype)

        assert actual.shape == (batch_size, hidden_size, seq_len)
        assert torch.allclose(actual, expected)


def test_qwen3_5_decode_recurrent_step_keeps_unpadded_token_shape():
    torch.manual_seed(0)
    batch_size, seq_len, num_heads, k_head_dim, v_head_dim = 16, 1, 8, 64, 128
    layer = object.__new__(QEffQwen3_5GatedDeltaNet)

    query = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.float32)
    g = torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32))
    recurrent_state = torch.randn(batch_size, num_heads, k_head_dim, v_head_dim, dtype=torch.float32)

    output, final_state = layer._recurrent_step_batched(query, key, value, g, beta, recurrent_state)

    assert output.shape == (batch_size, seq_len, num_heads, v_head_dim)
    assert output.reshape(-1, v_head_dim).shape == (batch_size * num_heads, v_head_dim)
    assert final_state.shape == recurrent_state.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def _build_chunk_masks(chunk_size: int, device: torch.device):
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=0)
    mask_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=1)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=device)
    return mask_causal, mask_strict, eye


def _run_chunk_rule(solver: str, chunk_size: int = 8):
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size, seq_len, num_heads, k_head_dim, v_head_dim = 2, 11, 3, 4, 4

    # Method only depends on these attrs/helpers; full HF model init is not required for this unit test.
    layer = object.__new__(QEffQwen3_5MoeGatedDeltaNet)
    layer.chunk_gated_delta_solver = solver

    query = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32, device=device)
    key = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32, device=device)
    value = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.float32, device=device)
    g = torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device=device))

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).repeat(1, batch_size, 1)
    position_ids[:, 1, -2:] = -1

    mask_causal, mask_strict, eye = _build_chunk_masks(chunk_size, device)

    output, final_state = layer.torch_chunk_gated_delta_rule_qeff(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        position_ids=position_ids,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        mask_causal=mask_causal,
        mask_strict=mask_strict,
        ones_lower=None,
        eye=eye,
    )
    return (
        output,
        final_state,
        (batch_size, seq_len, num_heads, v_head_dim),
        (batch_size, num_heads, k_head_dim, v_head_dim),
    )


def test_torch_chunk_gated_delta_rule_qeff_output_and_state_shapes():
    output, final_state, out_shape, state_shape = _run_chunk_rule("recursive_sns")

    assert output.shape == out_shape
    assert final_state is not None
    assert final_state.shape == state_shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def test_torch_chunk_gated_delta_rule_qeff_supports_all_solver_modes():
    for solver in ("recursive_sns", "scaled_newton_schulz", "original", "factorized", "horner"):
        output, final_state, _, _ = _run_chunk_rule(solver)
        assert torch.isfinite(output).all(), f"non-finite output for solver={solver}"
        assert torch.isfinite(final_state).all(), f"non-finite final_state for solver={solver}"


def test_torch_chunk_gated_delta_rule_qeff_recursive_sns_matches_original_max_abs_dev():
    output_recursive, _, _, _ = _run_chunk_rule("recursive_sns")
    # TODO: It would be better to test directly between original HF vs QEFF instead of making copy of original code snippet within qeff.
    output_original, _, _, _ = _run_chunk_rule("original")

    max_abs_dev = (output_recursive - output_original).abs().max().item()
    assert max_abs_dev <= MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL, (
        f"max abs deviation {max_abs_dev:.6e} exceeded threshold "
        f"{MAX_ABS_DEV_RECURSIVE_VS_ORIGINAL:.6e} for recursive_sns vs original"
    )
