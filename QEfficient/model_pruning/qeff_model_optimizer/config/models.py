"""Spec describing a base model to load, independent of any transform or runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=True)
class ModelSpec:
    """Describes how to locate and load a base causal-LM checkpoint.

    ``model_id`` is any string accepted by
    ``AutoModelForCausalLM.from_pretrained`` — a Hub repo ID, a local path, or
    a URI understood by the active loader.  ``revision`` pins a specific git
    commit or branch; ``None`` means the default branch.

    ``dtype`` is resolved to the matching ``torch.dtype`` by the loader;
    unrecognised strings fall back to the model's saved dtype
    (equivalent to ``torch_dtype="auto"``).
    """

    model_id: str
    revision: str | None = None
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict suitable for JSON / manifest storage."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "device_map": self.device_map,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelSpec":
        """Deserialise from a plain dict (e.g. a manifest or config file)."""
        return cls(
            model_id=str(payload["model_id"]),
            revision=payload.get("revision"),
            trust_remote_code=bool(payload.get("trust_remote_code", True)),
            dtype=str(payload.get("dtype", "bfloat16")),
            device_map=str(payload.get("device_map", "auto")),
        )
