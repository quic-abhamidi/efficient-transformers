# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json

import pytest
import torch

VOCAB_SIZE = 500
CTX_LEN = 32


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_qwen3_vl():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    text_cfg = Qwen3VLTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def make_tiny_qwen3_vl_moe():
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    text_cfg = Qwen3VLMoeTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    vision_cfg = Qwen3VLMoeVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLMoeConfig(text_config=text_cfg, vision_config=vision_cfg)
    return Qwen3VLMoeForConditionalGeneration(cfg).eval()


def make_tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=64,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=CTX_LEN,
        rope_parameters={"rope_type": "default"},
    )
    return GptOssForCausalLM(cfg).eval()


@pytest.mark.transforms
class TestLayerSkipPruningTransform:
    def test_layer_skip_config_from_json_canonicalizes_skip_layers(self, tmp_path):
        from QEfficient.pruning.config import PruningConfig

        config_path = tmp_path / "layer_skip.json"
        config_path.write_text(json.dumps({"skip_layers": [2, 0, 2]}))

        config = PruningConfig.from_qaic_config(
            {
                "enable_layer_skipping": True,
                "layer_skip_config": str(config_path),
            }
        )

        assert config is not None
        assert config.to_dict() == {"skip_layers": [0, 2]}

    def test_layer_skip_config_accepts_nas_transform_json(self, tmp_path):
        from QEfficient.pruning.config import PruningConfig

        config_path = tmp_path / "nas_plan.json"
        config_path.write_text(json.dumps({"transforms": [{"kind": "skip_layers", "layers": [1]}]}))

        config = PruningConfig.from_qaic_config(
            {
                "enable_layer_skipping": True,
                "layer_skip_config": str(config_path),
            }
        )

        assert config is not None
        assert config.to_dict() == {"skip_layers": [1]}

    def test_layer_skip_flag_requires_json_file(self):
        from QEfficient.pruning.config import PruningConfig

        with pytest.raises(TypeError, match="layer_skip_config must be a JSON file path"):
            PruningConfig.from_qaic_config(
                {
                    "enable_layer_skipping": True,
                    "layer_skip_config": {"skip_layers": [0]},
                }
            )

    def test_pruning_config_legacy_alias_canonicalizes_skip_layers(self):
        from QEfficient.pruning.config import PruningConfig

        config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [2, 0, 2]},
            }
        )

        assert config is not None
        assert config.to_dict() == {"skip_layers": [0, 2]}

    def test_pruning_config_rejects_unsupported_pruning_keys(self):
        from QEfficient.pruning.config import PruningConfig

        with pytest.raises(ValueError, match="Only layer skipping is supported"):
            PruningConfig.from_qaic_config(
                {
                    "enable_pruning": True,
                    "pruning_config": {
                        "skip_layers": [0],
                        "moe_pruned_experts": {"0": [1]},
                    },
                }
            )

    def test_layer_skip_transform_replaces_selected_decoder_layers(self):
        from QEfficient.pruning import SkippedDecoderLayer
        from QEfficient.pruning.config import PruningConfig
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, PruningTransform

        model = make_tiny_qwen3()
        model, _ = KVCacheTransform.apply(model)
        config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [1]},
            }
        )

        model, transformed = PruningTransform.apply(model, config)

        assert transformed
        assert isinstance(model.model.layers[1], SkippedDecoderLayer)
        assert not isinstance(model.model.layers[0], SkippedDecoderLayer)

        hidden_states = torch.randn(1, 2, model.config.hidden_size)
        skipped_output = model.model.layers[1](hidden_states)
        assert skipped_output is hidden_states

    @pytest.mark.parametrize(
        "model_factory, layer_path",
        [
            (make_tiny_qwen3_vl, ("model", "language_model", "layers")),
            (make_tiny_qwen3_vl_moe, ("model", "language_model", "layers")),
        ],
    )
    def test_layer_skip_transform_replaces_qwen3_vl_language_layers(self, model_factory, layer_path, tmp_path):
        from QEfficient.pruning import SkippedDecoderLayer
        from QEfficient.pruning.config import PruningConfig
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, PruningTransform

        config_path = tmp_path / "layer_skip.json"
        config_path.write_text(json.dumps({"layers": [1]}))

        model = model_factory()
        model, _ = KVCacheTransform.apply(model)
        config = PruningConfig.from_qaic_config(
            {
                "enable_layer_skipping": True,
                "layer_skip_config": str(config_path),
            }
        )

        model, transformed = PruningTransform.apply(model, config)

        layers = model
        for attr in layer_path:
            layers = getattr(layers, attr)
        assert transformed
        assert isinstance(layers[1], SkippedDecoderLayer)
        assert not isinstance(layers[0], SkippedDecoderLayer)

    def test_reapplying_layer_skip_restores_previous_wrappers(self):
        from QEfficient.pruning import SkippedDecoderLayer
        from QEfficient.pruning.config import PruningConfig
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, PruningTransform

        model = make_tiny_qwen3()
        model, _ = KVCacheTransform.apply(model)
        first_config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [1]},
            }
        )
        second_config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [0]},
            }
        )

        model, _ = PruningTransform.apply(model, first_config)
        model, transformed = PruningTransform.apply(model, second_config)

        assert transformed
        assert isinstance(model.model.layers[0], SkippedDecoderLayer)
        assert not isinstance(model.model.layers[1], SkippedDecoderLayer)

    def test_layer_skip_transform_preserves_tuple_contract_for_gpt_oss_layers(self):
        from QEfficient.pruning import SkippedDecoderLayer
        from QEfficient.pruning.config import PruningConfig
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, PruningTransform

        model = make_tiny_gpt_oss()
        model, _ = KVCacheTransform.apply(model)
        config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [0]},
            }
        )

        model, transformed = PruningTransform.apply(model, config)

        assert transformed
        assert isinstance(model.model.layers[0], SkippedDecoderLayer)

        hidden_states = torch.randn(1, 2, model.config.hidden_size)
        past_key_value = object()
        skipped_output = model.model.layers[0](
            hidden_states,
            output_attentions=True,
            use_cache=True,
            past_key_value=past_key_value,
        )

        assert skipped_output[0] is hidden_states
        assert skipped_output[1] is None
        assert skipped_output[2] is past_key_value

    def test_layer_skip_transform_rejects_out_of_range_layers(self):
        from QEfficient.pruning.config import PruningConfig
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, PruningTransform

        model = make_tiny_qwen3()
        model, _ = KVCacheTransform.apply(model)
        config = PruningConfig.from_qaic_config(
            {
                "enable_pruning": True,
                "pruning_config": {"skip_layers": [model.config.num_hidden_layers]},
            }
        )

        with pytest.raises(ValueError, match="out of range"):
            PruningTransform.apply(model, config)

    def test_layer_skip_onnx_transform_preserves_skipped_retained_state_inputs(self):
        import onnx
        from onnx import TensorProto, helper

        from QEfficient.base.onnx_transforms import PreserveSkippedLayerRetainedStateTransform

        shape = [1, 4, CTX_LEN, 32]
        graph_inputs = [
            helper.make_tensor_value_info("past_key.0", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_value.0", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_key.1", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_value.1", TensorProto.FLOAT, shape),
        ]
        graph_outputs = [
            helper.make_tensor_value_info("past_key.0_RetainedState", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_value.0_RetainedState", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_key.1_RetainedState", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("past_value.1_RetainedState", TensorProto.FLOAT, shape),
        ]
        graph = helper.make_graph(
            [
                helper.make_node("Add", ["past_key.0", "past_key.0"], ["past_key.0_RetainedState"]),
                helper.make_node("Add", ["past_value.0", "past_value.0"], ["past_value.0_RetainedState"]),
                helper.make_node("Add", ["past_key.1", "past_key.1"], ["past_key.1_RetainedState"]),
                helper.make_node("Add", ["past_value.1", "past_value.1"], ["past_value.1_RetainedState"]),
            ],
            "layer_skip_retained_state_test",
            graph_inputs,
            graph_outputs,
        )
        model = helper.make_model(graph)

        transformed = PreserveSkippedLayerRetainedStateTransform.apply(model, skipped_layers=[1])

        assert transformed
        producers = {output: node for node in model.graph.node for output in node.output}
        assert producers["past_key.1_RetainedState"].op_type == "Identity"
        assert list(producers["past_key.1_RetainedState"].input) == ["past_key.1"]
        assert producers["past_value.1_RetainedState"].op_type == "Identity"
        assert list(producers["past_value.1_RetainedState"].input) == ["past_value.1"]
        assert producers["past_key.0_RetainedState"].op_type == "Add"
        assert producers["past_value.0_RetainedState"].op_type == "Add"
        onnx.checker.check_model(model)
