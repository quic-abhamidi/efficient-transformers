"""QEff runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from QEfficient.model_pruning.qeff_model_optimizer.runtimes.base import BaseRuntime
from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
from QEfficient.model_pruning.qeff_model_optimizer.config.eval import QEffCompileSpec

ALLOWED_PREPARE_MODES = {"auto", "object", "export"}

QEFF_COMPILE_OPTION_KEYS = {
    "onnx_path",
    "compile_dir",
    "comp_ctx_lengths_prefill",
    "comp_ctx_lengths_decode",
    "kv_cache_batch_size",
    "mxfp6_matmul",
    "mxint8_kv_cache",
    "num_speculative_tokens",
    "prefill_only",
    "use_onnx_subfunctions",
    "offload_pt_weights",
    "enable_chunking",
    "retain_full_kv",
}


def _load_qeff_auto_model_cls(model_type: str | None = None):
    try:
        from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
    except Exception as exc:
        raise ImportError(
            "Failed to import QEfficient. Ensure the installed QEfficient package "
            "is compatible with the installed transformers package and does not rely "
            "on a local patched checkout."
        ) from exc
    model_key = (model_type or "").lower()
    if any(marker in model_key for marker in ("vl", "vision", "gemma3", "gemma4", "llava", "internvl", "mllama", "molmo")):
        return QEFFAutoModelForImageTextToText
    return QEFFAutoModelForCausalLM


def _split_qeff_config(raw_config: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object]]:
    payload = dict(raw_config)
    compile_options = dict(payload.pop("compile_options", {}))
    model_qaic_config = payload.pop("model_qaic_config", None)

    for key in tuple(payload.keys()):
        if key in QEFF_COMPILE_OPTION_KEYS:
            compile_options[key] = payload.pop(key)

    if model_qaic_config is None:
        model_qaic_config = payload or None
    elif not isinstance(model_qaic_config, dict):
        raise ValueError("model_qaic_config must be a dict when provided")

    return model_qaic_config, compile_options


def _build_compile_kwargs(compile_spec: QEffCompileSpec, compile_options: dict[str, object]) -> dict[str, object]:
    num_devices = len(compile_spec.device_group or [0])
    compile_kwargs: dict[str, object] = {
        "prefill_seq_len": compile_spec.prefill_seq_len,
        "ctx_len": compile_spec.ctx_len,
        "batch_size": compile_spec.batch_size,
        "num_devices": num_devices,
        "num_cores": compile_spec.num_cores,
    }
    if compile_spec.continuous_batching:
        compile_kwargs["full_batch_size"] = compile_spec.batch_size
    compile_kwargs.update(compile_options)
    return compile_kwargs


def _default_qeff_evaluator(
    artifact: ModelArtifact,
    compile_spec: QEffCompileSpec,
    prepare_mode: Literal["auto", "object", "export"],
) -> Any:
    resolved_mode = "object" if prepare_mode == "auto" else prepare_mode
    if resolved_mode == "export":
        raise NotImplementedError(
            "QEffRuntime prepare_mode='export' is not implemented yet. "
            "Use prepare_mode='object' or 'auto' until an explicit export path lands."
        )
    model_type = getattr(getattr(artifact.model, "config", None), "model_type", artifact.model_spec.model_id)
    qeff_model_cls = _load_qeff_auto_model_cls(str(model_type))
    model_qaic_config, compile_options = _split_qeff_config(compile_spec.qaic_config)

    prepared_model = qeff_model_cls(
        artifact.model,
        continuous_batching=compile_spec.continuous_batching,
        qaic_config=model_qaic_config,
        max_seq_len_cached=compile_spec.ctx_len,
        pretrained_model_name_or_path=artifact.model_spec.model_id,
    )

    result = {
        "artifact_id": artifact.artifact_id,
        "runtime": "qeff",
        "prepare_mode": resolved_mode,
        "prepared_model": prepared_model,
        "model_id": artifact.model_spec.model_id,
    }

    if resolved_mode == "object":
        return result

    compile_kwargs = _build_compile_kwargs(compile_spec, compile_options)
    result["compile_kwargs"] = compile_kwargs
    result["qpc_path"] = prepared_model.compile(**compile_kwargs)
    return result


@dataclass
class QEffRuntime(BaseRuntime):
    """Runtime wrapper for QEff-backed execution."""

    compile_spec: QEffCompileSpec
    prepare_mode: Literal["auto", "object", "export"] = "auto"
    evaluator: Callable[[ModelArtifact, QEffCompileSpec, str], Any] | None = None
    name: str = "qeff"

    def __post_init__(self) -> None:
        if self.prepare_mode not in ALLOWED_PREPARE_MODES:
            raise ValueError(
                f"prepare_mode must be one of {sorted(ALLOWED_PREPARE_MODES)}"
            )

    def evaluate(self, artifact: ModelArtifact):
        """Prepare *artifact* for QEff execution and return a result dict.

        In ``"auto"`` / ``"object"`` mode the model is wrapped with
        ``QEFFAutoModelForCausalLM`` and returned without compiling so the caller
        can inspect or call ``.compile()`` manually.  ``"export"`` mode raises
        ``NotImplementedError`` until a real export path is implemented.
        The result dict always contains ``artifact_id``, ``runtime``,
        ``prepare_mode``, ``prepared_model``, and ``model_id``.
        """
        evaluator = self.evaluator or _default_qeff_evaluator
        return evaluator(artifact, self.compile_spec, self.prepare_mode)
