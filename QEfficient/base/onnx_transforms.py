# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxChunkScatterBatch,
    CtxChunkScatterBatchFunc,
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherBlockedKVBatch,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.customop.ctx_scatter_gather_cb import (
    CtxGatherBlockedKVCB,
    CtxGatherCB,
    CtxGatherCB3D,
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterCB,
    CtxScatterCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)
from QEfficient.customop.onnxscript_utils import get_onnxscript_func
from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils import constants
from QEfficient.utils.constants import FILE_CHUNK_SIZE_DEFAULT, SIZE_THRESHOLD_DEFAULT

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(BaseOnnxTransform):
    """Clip FP32 tensors to FP16 range to avoid overflow during conversion."""

    @classmethod
    def apply(cls, tensor: TensorProto, onnx_base_dir: str, fp16_max: float, fp16_min: float) -> bool:
        nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

            if neg_inf_mask.any():
                clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

            tensor.CopyFrom(numpy_helper.from_array(clipped_tensor, tensor.name))
            return True
        return False


class SplitTensorsTransform(BaseOnnxTransform):
    """Split large tensors into external data files for efficient storage."""

    @classmethod
    def apply(
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: Dict[str, Tuple[TensorProto, str]]
    ) -> None:
        file_name = f"{model_name}_{file_num}.onnx.data"
        mapping[tensor.name] = (tensor, file_name)


class CustomOpTransform(BaseOnnxTransform):
    """Register custom ONNX ops and append their function prototypes to the model."""

    _custom_ops: Dict[str, Tuple[Any, Any]] = {
        "CustomRMSNormFunc": (CustomRMSNormFunc, CustomRMSNorm),
        "CtxScatterFunc": (CtxScatterFunc, CtxScatter),
        "CtxScatterFunc3D": (CtxScatterFunc3D, CtxScatter3D),
        "CtxScatterFunc3DInt": (CtxScatterFunc3DInt, CtxScatter3DInt),
        "CtxScatterFunc3DGeneralized": (CtxScatterFunc3DGeneralized, CtxScatter3D),
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
        "CtxGatherFunc3D": (CtxGatherFunc3D, CtxGather3D),
        "CtxGatherFunc3DGeneralized": (CtxGatherFunc3DGeneralized, CtxGather3D),
        "CtxScatterFuncCB3D": (CtxScatterFuncCB3D, CtxScatterCB3D),
        "CtxGatherFuncCB3D": (CtxGatherFuncCB3D, CtxGatherCB3D),
        "CtxGatherFuncBlockedKV": (CtxGatherFuncBlockedKV, CtxGatherBlockedKV),
        "CtxGatherFuncBlockedKVCB": (CtxGatherFuncBlockedKVCB, CtxGatherBlockedKVCB),
        "CtxScatterFuncCB": (CtxScatterFuncCB, CtxScatterCB),
        "CtxGatherFuncCB": (CtxGatherFuncCB, CtxGatherCB),
        "CastToUInt4": (CastToUInt4Func, CastToUInt4),
        "CtxChunkScatterBatchFunc": (CtxChunkScatterBatchFunc, CtxChunkScatterBatch),
        "CtxGatherFuncBlockedKVBatch": (CtxGatherFuncBlockedKVBatch, CtxGatherBlockedKVBatch),
    }

    @classmethod
    def apply(cls, model: ModelProto, onnx_export_opset: int = constants.ONNX_LEGACY_EXPORT_OPSET) -> bool:
        op_applied = False

        # Register with PyTorch ONNX exporter (for export time)
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, onnx_export_opset)

        used_op_types = {node.op_type for node in model.graph.node}
        for function_proto in model.functions:
            used_op_types.update(node.op_type for node in function_proto.node)

        # Add function prototypes to model
        existing = {f.name for f in model.functions}

        for func_name, onnxscript_func in cls._custom_ops.values():
            proto = get_onnxscript_func(onnxscript_func, onnx_export_opset).to_function_proto()
            if proto.name not in used_op_types:
                continue
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
                cls._ensure_opset_imports(model, proto.domain, 1)
        cls._propagate_function_opset_imports(model)
        return op_applied

    @staticmethod
    def _ensure_opset_imports(container: Union[ModelProto, onnx.FunctionProto], domain: str, version: int) -> None:
        if any(opset.domain == domain for opset in container.opset_import):
            return
        container.opset_import.append(onnx.helper.make_opsetid(domain, version))

    @classmethod
    def _propagate_function_opset_imports(cls, model: ModelProto) -> None:
        for fn in model.functions:
            for node in fn.node:
                if node.domain:
                    cls._ensure_opset_imports(fn, node.domain, 1)
                    cls._ensure_opset_imports(model, node.domain, 1)


class RemovePrefix(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        renamed = False

        def strip_prefix(name: str) -> str:
            parts = name.rsplit("/", 1)
            return parts[1] if len(parts) == 2 else parts[0]

        input_names = []
        for i, inputs in enumerate(graph.input):
            original = inputs.name
            new = strip_prefix(original)
            if new != original:
                renamed = True
            inputs.name = new
            graph.input[i].name = new
            input_names.append(new)

        input_name_set = set(input_names)
        output_rename_map = {}

        # Rename model graph outputs and keep mapping so producer/consumer edges can be fixed.
        for out in graph.output:
            original = out.name
            new = strip_prefix(original)
            if new != original:
                out.name = new
                output_rename_map[original] = new
                renamed = True

        for node in graph.node:
            for i, out in enumerate(node.output):
                if out in output_rename_map and output_rename_map[out] != out:
                    node.output[i] = output_rename_map[out]
                    renamed = True

            new_inputs = []
            for s in node.input:
                # Keep node inputs in sync for renamed model outputs.
                if s in output_rename_map:
                    new_inputs.append(output_rename_map[s])
                    continue

                if s in input_name_set:
                    new_inputs.append(s)
                    continue

                replaced = s
                if "/" in s:
                    tail = s.rsplit("/", 1)[1]
                    if tail in input_name_set:
                        replaced = tail
                new_inputs.append(replaced)

            for idx in range(len(node.input)):
                if node.input[idx] != new_inputs[idx]:
                    node.input[idx] = new_inputs[idx]
                    renamed = True

        return renamed


class RenameFunctionOutputsTransform(BaseOnnxTransform):
    """Rename decoder function retained-state outputs to public graph names.

    When subfunction export emits ``*_InternalRetainedState``, this transform rewrites
    them to ``*_RetainedState`` while preserving any optional KV prefix infix
    (for example ``past_key.0_vllmKvCache``).
    """

    @classmethod
    def apply(cls, model: ModelProto, layer_idx=0) -> bool:
        graph = model.graph
        op_type_to_func = {f.name: f for f in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False
        model_out_map = {v.name: i for i, v in enumerate(graph.output)}

        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if not func:
                    continue
                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        orig = node.output[i]
                        if orig.endswith("_InternalRetainedState"):
                            new = orig[: -len("_InternalRetainedState")] + "_RetainedState"
                        else:
                            base = out_name[: -len("_InternalRetainedState")]
                            new = orig
                            for token in (
                                "past_key.",
                                "past_value.",
                                "compressed_kv.",
                                "k_pe.",
                                "recurrent_state.",
                                "conv_state.",
                            ):
                                if not base.startswith(token):
                                    continue
                                tail = base[len(token) :]
                                _, _, infix = tail.partition("_")
                                infix = f"_{infix}" if infix else ""
                                new = f"{token}{layer_idx}{infix}_RetainedState"
                                break
                        node.output[i] = new
                        if orig in model_out_map:
                            graph.output[model_out_map[orig]].name = new
                layer_idx += 1
        return renamed


class PreserveNestedCacheRetainedStateTransform(BaseOnnxTransform):
    """Expose nested decoder cache side effects as explicit ONNX values.

    Must run BEFORE RenameRepeatedSubgraphTransform: this transform looks up
    functions by the dynamo-assigned name (repeated_subgraphN) via fn_by_name.
    Renaming them first would break that lookup.
    """

    # Match past_key.N / past_value.N regardless of any suffix that follows
    # (plain, _RetainedState, or _<prefix>_RetainedState for kv_cache_prefix).
    _KV_INPUT_RE = re.compile(r"^past_(key|value)\.(\d+)")

    # All scatter op_type names that write back a KV cache tensor.
    # Keep in sync with CustomOpTransform._custom_ops and the dynamo
    # custom_translation_table in base/modeling_qeff.py.
    _SCATTER_OP_TYPES = frozenset(
        {
            "CtxScatter",
            "CtxScatterCB",
            "CtxScatter3D",
            "CtxScatter3DInt",
            "CtxScatterCB3D",
        }
    )

    @staticmethod
    def _scatter_sort_key(n) -> int:
        output_name = n.output[0] if n.output else ""
        if "key" in output_name:
            return 0
        if "value" in output_name:
            return 1
        return 2

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        produced_names = {name for node in graph.node for name in node.output}
        dangling_retained_outputs = {
            out.name for out in graph.output if out.name.endswith("_RetainedState") and out.name not in produced_names
        }
        if not dangling_retained_outputs:
            return False

        fn_by_name = {fn.name: fn for fn in model.functions}
        changed = False
        kv_rename_map: Dict[str, str] = {}

        for node in graph.node:
            fn = fn_by_name.get(node.op_type)
            if fn is None:
                continue

            # Collect scatter nodes that write back the KV cache.
            # Sort by first-input name so the key scatter reliably precedes the
            # value scatter: dynamo names function-body args generically (arg7_1
            # etc.), so we sort by the scatter output name instead — dynamo
            # preserves "key"/"value" in output tensor names even when input
            # argument names are opaque.
            scatter_nodes = [
                fn_node for fn_node in fn.node if fn_node.op_type in cls._SCATTER_OP_TYPES and fn_node.output
            ]
            if len(scatter_nodes) != 2:
                logger.debug(
                    "PreserveNestedCacheRetainedStateTransform: function '%s' has %d scatter node(s), expected 2 — skipping.",
                    node.op_type,
                    len(scatter_nodes),
                )
                continue

            scatter_nodes.sort(key=cls._scatter_sort_key)
            # Only the first two scatter outputs map to key / value respectively.
            scatter_outputs = [n.output[0] for n in scatter_nodes[:2]]

            # Identify layer index from KV inputs on this call node.
            layer_idx = None
            kv_inputs = {}
            for inp_name in node.input:
                match = cls._KV_INPUT_RE.match(inp_name)
                if match is None:
                    continue
                kind, idx = match.groups()
                layer_idx = idx if layer_idx is None else layer_idx
                if layer_idx != idx:
                    kv_inputs = {}
                    break
                kv_inputs[kind] = inp_name

            if layer_idx is None or set(kv_inputs) != {"key", "value"}:
                continue

            desired_outputs = [
                f"past_key.{layer_idx}_RetainedState",
                f"past_value.{layer_idx}_RetainedState",
            ]
            # Skip layers whose retained-state outputs are not dangling —
            # either the graph is already correctly wired or a previous call
            # to this transform already fixed them.
            if not any(name in dangling_retained_outputs for name in desired_outputs):
                continue

            # Expose scatter outputs in the function's output list, rename KV
            # inputs and append retained-state output names to the call node —
            # all in one pass over the two key/value pairs.
            for kind, scatter_output, desired_output in zip(("key", "value"), scatter_outputs, desired_outputs):
                if scatter_output not in fn.output:
                    fn.output.append(scatter_output)
                    changed = True

                retained_input = kv_inputs[kind]
                plain_input = f"past_{kind}.{layer_idx}"
                if retained_input.endswith("_RetainedState"):
                    kv_rename_map[retained_input] = plain_input

                if desired_output not in node.output:
                    node.output.append(desired_output)
                    changed = True

        if kv_rename_map:
            changed |= cls._rename_graph_inputs_bulk(graph, kv_rename_map)
        return changed

    @staticmethod
    def _rename_graph_inputs_bulk(graph: onnx.GraphProto, rename_map: Dict[str, str]) -> bool:
        if not rename_map:
            return False
        changed = False
        for value in graph.input:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for value in graph.value_info:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for node in graph.node:
            new_inputs = [rename_map.get(n, n) for n in node.input]
            if new_inputs != list(node.input):
                node.input[:] = new_inputs
                changed = True
        return changed


class RenameRepeatedSubgraphTransform(BaseOnnxTransform):
    """Rename dynamo repeated_subgraph function names to model-specific layer class names.

    Must run AFTER PreserveNestedCacheRetainedStateTransform: that transform
    looks up functions by the dynamo-assigned name.  Renaming them first would
    break that lookup.
    """

    # Primary pattern emitted by torch.export repeated-subgraph canonicalization.
    # Extended to also match alternative patterns seen across PyTorch 2.x releases
    # so a dynamo-internal rename does not silently produce a no-op transform.
    _REPEATED_SUBGRAPH_PATTERNS = [
        re.compile(r"^repeated_subgraph(\d+)$"),  # torch >= 2.5 canonical name
        re.compile(r"^subgraph_(\d+)$"),  # alternative seen in some 2.x nightlies
        re.compile(r"^invoke_subgraph_(\d+)$"),  # earlier 2.x name
    ]

    @classmethod
    def _iter_all_nodes(cls, nodes):
        """Yield every NodeProto reachable from `nodes`, including nodes nested
        inside If/Loop subgraph attributes."""
        for node in nodes:
            yield node
            for attr in node.attribute:
                if attr.HasField("g"):
                    yield from cls._iter_all_nodes(attr.g.node)

    @staticmethod
    def _rename_op_types(nodes, old_to_new: Dict[str, str]) -> None:
        for node in RenameRepeatedSubgraphTransform._iter_all_nodes(nodes):
            if node.op_type in old_to_new:
                node.op_type = old_to_new[node.op_type]

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: Optional[List[str]] = None, **kwargs) -> bool:
        target_classnames = [name for name in (target_classnames or []) if name]
        if not target_classnames:
            logger.warning(
                "RenameRepeatedSubgraphTransform: target_classnames is empty — transform is a no-op. "
                "Check that get_submodules_for_export() returns the decoder layer classes."
            )
            return False

        repeated_functions = []
        for fn in model.functions:
            for pattern in cls._REPEATED_SUBGRAPH_PATTERNS:
                match = pattern.match(fn.name)
                if match:
                    repeated_functions.append((int(match.group(1)), fn))
                    break

        if not repeated_functions:
            logger.warning(
                "RenameRepeatedSubgraphTransform: no repeated_subgraph functions found in the ONNX model. "
                "This may indicate that dynamo changed its internal subgraph naming convention. "
                "The transform is a no-op; function names remain as emitted by torch.export."
            )
            return False

        repeated_functions.sort(key=lambda item: item[0])
        old_to_new = {}
        used_names = {fn.name for fn in model.functions}

        for idx, (_, fn) in enumerate(repeated_functions):
            if idx >= len(target_classnames):
                logger.warning(
                    f"RenameRepeatedSubgraphTransform: more repeated subgraph functions ({len(repeated_functions)}) "
                    f"than target class names ({len(target_classnames)}). "
                    f"Function '{fn.name}' (index {idx}) will be assigned the last available name "
                    f"'{target_classnames[-1]}' with a numeric suffix — verify get_submodules_for_export() "
                    "returns all repeated block classes for this model."
                )
            base_name = target_classnames[min(idx, len(target_classnames) - 1)]
            candidate = base_name
            suffix = 1
            while candidate in used_names and candidate != fn.name:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.discard(fn.name)
            used_names.add(candidate)
            old_to_new[fn.name] = candidate

        if not old_to_new:
            return False

        for fn in model.functions:
            if fn.name in old_to_new:
                fn.name = old_to_new[fn.name]

        cls._rename_op_types(model.graph.node, old_to_new)
        for fn in model.functions:
            cls._rename_op_types(fn.node, old_to_new)

        return True


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> Tuple[onnx.ModelProto, bool]:
        transformed = False
        removed_initializers = []

        # Find nodes with lora weights as inputs
        weight_suffix = f".{adapter_name}.weight"
        lora_weight_nodes = {
            inp: node for node in model.graph.node for inp in node.input if inp.endswith(weight_suffix)
        }

        for i, weight in enumerate(model.graph.initializer):
            if weight.name.endswith(weight_suffix):
                transformed = True

                # Create input/output for lora weights
                new_weight_name = weight.name[: -len(weight_suffix)] + ".weight"
                type_proto = onnx.helper.make_tensor_type_proto(weight.data_type, shape=list(weight.dims))
                inp = onnx.ValueInfoProto(name=new_weight_name, type=type_proto)
                out = onnx.ValueInfoProto(name=new_weight_name + "_RetainedState", type=type_proto)
                model.graph.input.append(inp)
                model.graph.output.append(out)

                # Create a node that connects input -> output
                node = onnx.helper.make_node("Identity", [inp.name], [out.name], new_weight_name + "_identity")
                model.graph.node.append(node)

                # Rename weight input
                lora_weight_node = lora_weight_nodes[weight.name]
                for j, inp in enumerate(lora_weight_node.input):
                    if inp == weight.name:
                        lora_weight_node.input[j] = new_weight_name

                # Remove weight initializers
                removed_initializers.append(i)

        if transformed:
            for i in sorted(removed_initializers, reverse=True):
                model.graph.initializer.pop(i)

        return model, transformed


class PruneFakeInitializersTransform(BaseOnnxTransform):
    """Remove initializers backed by FakeTensors from a dynamo onnx_program before serialisation.

    Operates on the torch.onnx.ONNXProgram object returned by dynamo export, not on a
    ModelProto, because FakeTensors cannot survive onnx.load round-trips.
    """

    @classmethod
    def apply(cls, onnx_program) -> bool:
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = onnx_program.model.graph.initializers
        used_names = {name for node in onnx_program.model.graph for name in node.inputs}
        used_names.update(output.name for output in onnx_program.model.graph.outputs)

        pruned = False
        for name in list(initializers):
            const_value = getattr(initializers[name], "const_value", None)
            raw_value = getattr(const_value, "raw", None)
            if isinstance(raw_value, FakeTensor) and name not in used_names:
                del initializers[name]
                pruned = True
        return pruned


class RenameWsubNodesTransform(BaseOnnxTransform):
    """Name local-function operators from their semantic layer parameters."""

    _LAYER_PARAMETER_RE = re.compile(r"(?:^|\.)layers\.\d+\.(?P<role>.+)\.weight$")
    _WEIGHTED_OPS = {"MatMul", "Gemm", "CustomRMSNorm"}

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        transformed = False
        for function in model.functions:
            function_inputs = set(function.input)
            for node in function.node:
                if node.op_type not in cls._WEIGHTED_OPS:
                    continue
                for input_name in node.input[1:]:
                    if input_name not in function_inputs:
                        continue
                    match = cls._LAYER_PARAMETER_RE.search(input_name)
                    if match is None:
                        continue
                    role = match.group("role").replace(".", "/")
                    semantic_name = f"{role}/{node.op_type}"
                    if node.name != semantic_name:
                        node.name = semantic_name
                        transformed = True
                    break
        return transformed


class DynamicAxesMetadataTransform(BaseOnnxTransform):
    """Restore symbolic graph input/output dimensions from QEff dynamic_axes metadata.

    Dynamo export can materialize static dimensions in model inputs even when
    torch.export dynamic_shapes were provided. QAIC network specializations are
    keyed by ONNX symbolic dimensions, so restore those dim_param names from the
    same dynamic_axes map used by the legacy exporter.
    """

    @classmethod
    def apply(cls, model: ModelProto, dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None, **kwargs) -> bool:
        if not dynamic_axes:
            return False

        def iter_value_infos():
            yield from model.graph.input
            yield from model.graph.output
            yield from model.graph.value_info

        value_infos = {value_info.name: value_info for value_info in iter_value_infos()}
        transformed = False

        for value_name, axes in dynamic_axes.items():
            value_info = value_infos.get(value_name)
            if value_info is None or not value_info.type.HasField("tensor_type"):
                continue
            shape = value_info.type.tensor_type.shape
            for axis, dim_name in axes.items():
                if axis >= len(shape.dim):
                    continue
                dim = shape.dim[axis]
                if dim.dim_param == dim_name:
                    continue
                if dim.HasField("dim_value"):
                    dim.ClearField("dim_value")
                dim.dim_param = dim_name
                transformed = True

        return transformed


class Qwen3_5DecoderLayerDynamicSeqLenTransform(BaseOnnxTransform):
    """Make Qwen3.5 decoder-layer subfunction reshape shapes use runtime seq_len.

    Dynamo emits ONNX local functions for repeated decoder layers. In a combined
    Prefill+Decode language QPC, those function bodies are traced with the
    prefill example length, so some reshape shape tensors contain a literal
    ``prefill_seq_len``. QAIC then reuses the same function body for the Decode
    specialization and fails on reshapes like ``[batch, 64, -1]`` when the input
    sequence length is 1.

    Only source-marked sequence-length reshape constants are rewritten. Gated
    delta chunk-size constants also happen to be 64 and must remain static.
    """

    _TARGET_PREFIX = "QEffQwen3_5DecoderLayer"
    _SEQ_LEN_STACK_PATTERNS = (
        "reshape(batch_size, seq_len",
        "reshape(*input_shape",
    )
    _CHUNK_STACK_PATTERNS = (
        "chunk_size",
        "zeros = torch.zeros(g.shape",
        "qkv_zeros = torch.zeros(key.shape",
    )

    @staticmethod
    def _metadata_text(node) -> str:
        return "\n".join(prop.value for prop in node.metadata_props)

    @classmethod
    def _is_prefill_seq_len_constant(cls, node, export_prefill_seq_len: int) -> bool:
        if node.op_type != "Constant" or len(node.output) != 1:
            return False
        has_target_value = False
        for attr in node.attribute:
            if attr.name == "value_ints" and list(attr.ints) == [export_prefill_seq_len]:
                has_target_value = True
                break
            if attr.name == "value" and list(attr.t.dims) == [1]:
                try:
                    arr = numpy_helper.to_array(attr.t)
                except Exception:
                    continue
                if arr.size == 1 and int(arr.reshape(-1)[0]) == export_prefill_seq_len:
                    has_target_value = True
                    break
        if not has_target_value:
            return False

        metadata = cls._metadata_text(node)
        if any(pattern in metadata for pattern in cls._CHUNK_STACK_PATTERNS):
            return False
        return any(pattern in metadata for pattern in cls._SEQ_LEN_STACK_PATTERNS)

    @classmethod
    def apply(cls, model: ModelProto, *, export_prefill_seq_len: Optional[int] = None, **kwargs) -> bool:
        if export_prefill_seq_len is None:
            return False
        export_prefill_seq_len = int(export_prefill_seq_len)
        transformed = False

        for function in model.functions:
            if not function.name.startswith(cls._TARGET_PREFIX) or len(function.input) < 2:
                continue

            replacements = [
                node for node in function.node if cls._is_prefill_seq_len_constant(node, export_prefill_seq_len)
            ]
            if not replacements:
                continue

            hidden_states_name = function.input[1]
            axis_name = f"qeff_{function.name}_seq_axis"
            shape_name = f"qeff_{function.name}_hidden_shape"
            seq_len_name = f"qeff_{function.name}_seq_len"
            prefix_nodes = [
                onnx.helper.make_node("Constant", [], [axis_name], value_ints=[1]),
                onnx.helper.make_node("Shape", [hidden_states_name], [shape_name]),
                onnx.helper.make_node("Gather", [shape_name, axis_name], [seq_len_name], axis=0),
            ]

            for node in replacements:
                original_output = node.output[0]
                del node.input[:]
                node.input.extend([seq_len_name])
                del node.attribute[:]
                node.op_type = "Identity"
                node.domain = ""
                node.name = f"{node.name}_dynamic_seq_len"
                del node.output[:]
                node.output.extend([original_output])

            existing_nodes = list(function.node)
            del function.node[:]
            function.node.extend(prefix_nodes)
            function.node.extend(existing_nodes)
            transformed = True

        return transformed




class InlineLoopSubfunctionsTransform(BaseOnnxTransform):
    """Inline local ONNX functions that contain Loop nodes.

    QAIC accepts ``-sub-functions`` for the surrounding model, but Qwen3.5 gated
    delta currently produces incorrect runtime results when an ONNX Loop is
    nested inside a local function body. This pass leaves normal subfunctions in
    place and expands only the function-call nodes whose target function contains
    a Loop.
    """

    @staticmethod
    def _contains_loop_in_nodes(nodes) -> bool:
        for node in nodes:
            if node.op_type == "Loop":
                return True
            for attr in node.attribute:
                if attr.HasField("g") and InlineLoopSubfunctionsTransform._contains_loop_in_nodes(attr.g.node):
                    return True
                for nested_graph in attr.graphs:
                    if InlineLoopSubfunctionsTransform._contains_loop_in_nodes(nested_graph.node):
                        return True
        return False

    @staticmethod
    def _remap_graph_names(graph, name_map, prefix: str) -> None:
        def remap(name: str) -> str:
            if not name:
                return name
            if name in name_map:
                return name_map[name]
            new_name = f"{prefix}_{name}"
            name_map[name] = new_name
            return new_name

        for graph_input in graph.input:
            graph_input.name = remap(graph_input.name)
        for graph_output in graph.output:
            graph_output.name = remap(graph_output.name)
        for value_info in graph.value_info:
            value_info.name = remap(value_info.name)
        for initializer in graph.initializer:
            initializer.name = remap(initializer.name)
        for node in graph.node:
            node.input[:] = [remap(name) for name in node.input]
            node.output[:] = [remap(name) for name in node.output]
            if node.name:
                node.name = f"{prefix}_{node.name}"
            for attr in node.attribute:
                if attr.HasField("g"):
                    InlineLoopSubfunctionsTransform._remap_graph_names(attr.g, name_map, prefix)
                for nested_graph in attr.graphs:
                    InlineLoopSubfunctionsTransform._remap_graph_names(nested_graph, name_map, prefix)

    @classmethod
    def _inline_graph(cls, graph, loop_functions: Dict[Tuple[str, str], Any], prefix_seed: str) -> bool:
        changed = False
        new_nodes = []
        call_index = 0

        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    changed |= cls._inline_graph(attr.g, loop_functions, f"{prefix_seed}_{node.name or node.op_type}")
                for nested_graph in attr.graphs:
                    changed |= cls._inline_graph(nested_graph, loop_functions, f"{prefix_seed}_{node.name or node.op_type}")

            function = loop_functions.get((node.domain, node.op_type))
            if function is None:
                new_nodes.append(node)
                continue

            prefix = f"qeff_inline_{prefix_seed}_{call_index}_{node.name or node.op_type}"
            name_map = {formal: actual for formal, actual in zip(function.input, node.input)}
            name_map.update({formal: actual for formal, actual in zip(function.output, node.output)})

            for function_node in function.node:
                inlined_node = copy.deepcopy(function_node)

                def remap(name: str) -> str:
                    if not name:
                        return name
                    if name in name_map:
                        return name_map[name]
                    new_name = f"{prefix}_{name}"
                    name_map[name] = new_name
                    return new_name

                inlined_node.input[:] = [remap(name) for name in inlined_node.input]
                inlined_node.output[:] = [remap(name) for name in inlined_node.output]
                if inlined_node.name:
                    inlined_node.name = f"{prefix}_{inlined_node.name}"
                for attr in inlined_node.attribute:
                    if attr.HasField("g"):
                        cls._remap_graph_names(attr.g, name_map, prefix)
                    for nested_graph in attr.graphs:
                        cls._remap_graph_names(nested_graph, name_map, prefix)
                new_nodes.append(inlined_node)

            changed = True
            call_index += 1

        if changed:
            del graph.node[:]
            graph.node.extend(new_nodes)
        return changed

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
        loop_functions = {
            (function.domain, function.name): function
            for function in model.functions
            if cls._contains_loop_in_nodes(function.node)
        }
        if not loop_functions:
            return False

        transformed = cls._inline_graph(model.graph, loop_functions, model.graph.name or "graph")
        if transformed:
            kept_functions = [
                function for function in model.functions if (function.domain, function.name) not in loop_functions
            ]
            del model.functions[:]
            model.functions.extend(kept_functions)
        return transformed

class RetainedStateInputOutputNameTransform(BaseOnnxTransform):
    """Split pass-through retained-state graph input/output names.

    Some dynamo exports leave a retained-state graph output backed directly by a
    graph input with the same ``*_RetainedState`` name. The compiler creates one
    placeholder per graph input/output, so the shared name is rejected as a
    pre-existing placeholder. Rename the graph input back to the base state name
    and insert an Identity producing the retained-state output name.
    """

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
        graph = model.graph
        produced_names = {output for node in graph.node for output in node.output if output}
        graph_output_names = {output.name for output in graph.output}
        rename_map: Dict[str, str] = {}
        identity_nodes = []

        for graph_input in graph.input:
            old_name = graph_input.name
            if not old_name.endswith("_RetainedState") or old_name not in graph_output_names:
                continue
            if old_name in produced_names:
                continue
            new_name = old_name[: -len("_RetainedState")]
            if not new_name:
                continue
            rename_map[old_name] = new_name
            graph_input.name = new_name
            identity_nodes.append(onnx.helper.make_node("Identity", [new_name], [old_name], f"{new_name}_identity"))

        if not rename_map:
            return False

        def rename_inputs_in_nodes(nodes) -> None:
            for node in nodes:
                node.input[:] = [rename_map.get(name, name) for name in node.input]
                for attr in node.attribute:
                    if attr.HasField("g"):
                        rename_inputs_in_nodes(attr.g.node)
                    for nested_graph in attr.graphs:
                        rename_inputs_in_nodes(nested_graph.node)

        for value_info in graph.value_info:
            if value_info.name in rename_map:
                value_info.name = rename_map[value_info.name]
        rename_inputs_in_nodes(graph.node)

        existing_nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(identity_nodes)
        graph.node.extend(existing_nodes)
        return True


class ConstantLoopConditionTransform(BaseOnnxTransform):
    """Rewrite ONNX Loop control inputs into compiler-friendly constants.

    ``torch.while_loop`` exports the ONNX Loop initial condition as a graph
    value produced by the loop condition graph. The AIC compiler requires this
    Loop input to be a compile-time constant. When a static loop trip count is
    provided, this transform also fills Loop.input[0] so the compiler sees a
    bounded loop. The Loop body still returns the real per-iteration continuation
    condition, so runtime termination semantics are preserved.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, loop_trip_count: Optional[int] = None, **kwargs) -> bool:
        transformed = False
        loop_index = 0

        def next_names() -> Tuple[Optional[str], str]:
            nonlocal loop_index
            trip_name = f"qeff_loop_trip_count_{loop_index}" if loop_trip_count is not None else None
            cond_name = f"qeff_loop_cond_true_{loop_index}"
            loop_index += 1
            return trip_name, cond_name

        def make_trip_tensor(name: str) -> TensorProto:
            return onnx.helper.make_tensor(name, TensorProto.INT64, [], [int(loop_trip_count)])

        def make_cond_tensor(name: str) -> TensorProto:
            return onnx.helper.make_tensor(name, TensorProto.BOOL, [], [True])

        def rewrite_loop_node(node, trip_name: Optional[str], cond_name: str) -> bool:
            if len(node.input) < 2:
                logger.warning(
                    "ConstantLoopConditionTransform: Loop node '%s' has fewer than 2 inputs; skipping.", node.name
                )
                return False
            if trip_name is not None:
                node.input[0] = trip_name
            node.input[1] = cond_name
            return True

        def rewrite_graph(graph) -> bool:
            graph_changed = False
            initializer_names = {initializer.name for initializer in graph.initializer}

            def add_initializer_once(name: str, tensor: Optional[TensorProto] = None) -> None:
                if name not in initializer_names:
                    graph.initializer.append(tensor if tensor is not None else make_cond_tensor(name))
                    initializer_names.add(name)

            for node in graph.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        graph_changed |= rewrite_graph(attr.g)
                    for nested_graph in attr.graphs:
                        graph_changed |= rewrite_graph(nested_graph)

                if node.op_type != "Loop":
                    continue
                trip_name, cond_name = next_names()
                if trip_name is not None:
                    add_initializer_once(trip_name, make_trip_tensor(trip_name))
                add_initializer_once(cond_name)
                graph_changed |= rewrite_loop_node(node, trip_name, cond_name)
            return graph_changed

        def rewrite_function(function) -> bool:
            function_changed = False
            constant_nodes = []
            for node in function.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        function_changed |= rewrite_graph(attr.g)
                    for nested_graph in attr.graphs:
                        function_changed |= rewrite_graph(nested_graph)

                if node.op_type != "Loop":
                    continue
                trip_name, cond_name = next_names()
                if trip_name is not None:
                    constant_nodes.append(
                        onnx.helper.make_node(
                            "Constant", [], [trip_name], value=make_trip_tensor(f"{trip_name}_value")
                        )
                    )
                constant_nodes.append(
                    onnx.helper.make_node(
                        "Constant", [], [cond_name], value=make_cond_tensor(f"{cond_name}_value")
                    )
                )
                function_changed |= rewrite_loop_node(node, trip_name, cond_name)

            if constant_nodes:
                existing_nodes = list(function.node)
                del function.node[:]
                function.node.extend(constant_nodes)
                function.node.extend(existing_nodes)
            return function_changed

        transformed |= rewrite_graph(model.graph)
        for function in model.functions:
            transformed |= rewrite_function(function)

        return transformed


class OnnxTransformPipeline(BaseOnnxTransform):
    """Pipeline to apply multiple ONNX transformations in sequence."""

    def __init__(self, transforms: List[Type[BaseOnnxTransform]]):
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: Dict[str, Tuple[TensorProto, str]] = {}
        requested = set(self.transforms)
        applied = {t: False for t in requested}
        f16_applied = False
        do_fp16 = FP16ClipTransform in requested
        do_split = SplitTensorsTransform in requested
        fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
        file_num_tracker = {"num": 0, "size": 0}
        if onnx_base_dir is not None:
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)

        if do_fp16 or do_split:
            for tensor in external_data_helper._get_all_tensors(model):
                if do_fp16 and FP16ClipTransform.apply(tensor, onnx_base_dir, fp16_max, fp16_min):
                    f16_applied = True
                applied[FP16ClipTransform] = f16_applied

                if do_split and tensor.HasField("raw_data"):
                    tsize = len(tensor.raw_data)
                    if tsize > size_threshold:
                        if file_num_tracker["size"] + tsize > file_chunk_size:
                            file_num_tracker["num"] += 1
                            file_num_tracker["size"] = tsize
                        else:
                            file_num_tracker["size"] += tsize
                        applied[SplitTensorsTransform] = True
                        SplitTensorsTransform.apply(tensor, model_name, file_num_tracker["num"], mapping)

        def _set_external_data(tensor, file_name):
            external_data_helper.set_external_data(tensor, file_name)

        max_workers = min(32, (os.cpu_count() or 1) * 4)
        logger.info(f"Applying external data mapping with {max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_set_external_data, tensor, file_name) for tensor, file_name in mapping.values()]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to set external data: {e}")

        # Non-looping transforms
        if CustomOpTransform in requested:
            applied[CustomOpTransform] = CustomOpTransform.apply(
                model, onnx_export_opset=kwargs.get("onnx_export_opset", constants.ONNX_LEGACY_EXPORT_OPSET)
            )

        if RenameFunctionOutputsTransform in requested:
            applied[RenameFunctionOutputsTransform] = RenameFunctionOutputsTransform.apply(
                model, layer_idx=kwargs.get("layer_idx", 0)
            )

        if RenameWsubNodesTransform in requested:
            applied[RenameWsubNodesTransform] = RenameWsubNodesTransform.apply(model)

        if DynamicAxesMetadataTransform in requested:
            applied[DynamicAxesMetadataTransform] = DynamicAxesMetadataTransform.apply(model, **kwargs)

        if PreserveNestedCacheRetainedStateTransform in requested:
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(model)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if Qwen3_5DecoderLayerDynamicSeqLenTransform in requested:
            applied[Qwen3_5DecoderLayerDynamicSeqLenTransform] = Qwen3_5DecoderLayerDynamicSeqLenTransform.apply(
                model, **kwargs
            )

        if InlineLoopSubfunctionsTransform in requested:
            applied[InlineLoopSubfunctionsTransform] = InlineLoopSubfunctionsTransform.apply(model, **kwargs)

        if RetainedStateInputOutputNameTransform in requested:
            applied[RetainedStateInputOutputNameTransform] = RetainedStateInputOutputNameTransform.apply(model, **kwargs)

        if ConstantLoopConditionTransform in requested:
            applied[ConstantLoopConditionTransform] = ConstantLoopConditionTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
