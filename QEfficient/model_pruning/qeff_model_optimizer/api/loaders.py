"""Model loader implementations for the NAS API."""

from __future__ import annotations

from dataclasses import dataclass

from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec


def _get_tokenizer_kwargs(model_id: str, trust_remote_code: bool) -> dict[str, object]:
    del model_id
    return {"trust_remote_code": trust_remote_code}


def _looks_like_vlm(model_id: str) -> bool:
    model_key = model_id.lower()
    return any(
        marker in model_key
        for marker in (
            "vl",
            "vision",
            "gemma-3",
            "gemma3",
            "gemma-4",
            "gemma4",
            "llava",
            "internvl",
            "mllama",
            "molmo",
        )
    )


@dataclass
class TransformersModelLoader:
    """Default loader backed by installed Hugging Face transformers."""

    def load(self, model_spec: ModelSpec):
        """Load model and tokenizer according to *model_spec* and return ``(model, tokenizer)``.

        ``dtype`` is mapped to the matching ``torch.dtype``; unrecognised strings
        fall back to the model's saved dtype.  If the tokenizer has no ``pad_token``
        the ``eos_token`` is used as a fallback; if both are absent a ``ValueError``
        is raised.  The model is set to eval mode before returning.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        model_kwargs = {
            "revision": model_spec.revision,
            "trust_remote_code": model_spec.trust_remote_code,
            "device_map": model_spec.device_map,
        }
        if model_spec.dtype in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[model_spec.dtype]

        if _looks_like_vlm(model_spec.model_id):
            tokenizer = AutoProcessor.from_pretrained(
                model_spec.model_id,
                revision=model_spec.revision,
                **_get_tokenizer_kwargs(model_spec.model_id, model_spec.trust_remote_code),
            )
            model = AutoModelForImageTextToText.from_pretrained(model_spec.model_id, **model_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_spec.model_id,
                revision=model_spec.revision,
                **_get_tokenizer_kwargs(
                    model_spec.model_id,
                    model_spec.trust_remote_code,
                ),
            )
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is None:
                    raise ValueError(
                        f"Tokenizer for {model_spec.model_id!r} has neither pad_token nor "
                        "eos_token; assign one on a custom loader subclass or pre-configure "
                        "the tokenizer before calling NASSession.load()."
                    )
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_spec.model_id,
                **model_kwargs,
            )
        model.eval()
        return model, tokenizer
