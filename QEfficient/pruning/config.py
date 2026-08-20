# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerSkipConfig:
    layers: tuple[int, ...]

    @classmethod
    def from_layers(cls, layers: Any) -> LayerSkipConfig:
        if not isinstance(layers, (list, tuple)):
            raise TypeError("skip_layers must be a list of non-negative layer indices.")
        if not layers:
            raise ValueError("skip_layers must contain at least one layer index.")

        normalized = tuple(sorted({int(layer) for layer in layers}))
        if normalized[0] < 0:
            raise ValueError("skip_layers must contain only non-negative layer indices.")
        return cls(layers=normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"layers": list(self.layers)}


@dataclass(frozen=True)
class PruningConfig:
    layer_skip: LayerSkipConfig | None = None

    @classmethod
    def from_qaic_config(cls, qaic_config: dict | None) -> PruningConfig | None:
        if not qaic_config:
            return None

        enable_layer_skipping = qaic_config.get("enable_layer_skipping", False)
        enable_pruning = qaic_config.get("enable_pruning", False)
        if not enable_layer_skipping and not enable_pruning:
            return None

        if enable_layer_skipping:
            raw_config = qaic_config.get("layer_skip_config")
            if raw_config is None:
                raise ValueError(
                    "enable_layer_skipping=True requires qaic_config.layer_skip_config to be a JSON file path."
                )
            raw_config = cls._load_json_config(raw_config, field_name="layer_skip_config")
        else:
            raw_config = qaic_config.get("pruning_config")
            if raw_config is None:
                raw_config = {}
            if isinstance(raw_config, (str, Path)):
                raw_config = cls._load_json_config(raw_config, field_name="pruning_config")

        if not isinstance(raw_config, dict):
            raise TypeError("layer skip configuration must be a dictionary or a JSON file path.")

        raw_skip_layers = _extract_skip_layers(raw_config)
        if raw_skip_layers is None:
            raw_skip_layers = qaic_config.get("skip_layers")

        if enable_pruning:
            unsupported_keys = set(raw_config) - {"skip_layers", "layers", "kind", "plan", "transforms"}
        else:
            unsupported_keys = set()
        if unsupported_keys:
            raise ValueError(
                "Only layer skipping is supported in this version. "
                f"Unsupported layer skip configuration keys: {sorted(unsupported_keys)}"
            )
        if raw_skip_layers is None:
            raise ValueError(
                "Layer skipping requires skip_layers/layers in the JSON file, "
                "or a NAS transform entry with kind='skip_layers'."
            )

        return cls(layer_skip=LayerSkipConfig.from_layers(raw_skip_layers))

    @staticmethod
    def _load_json_config(config_path: str | Path, field_name: str) -> dict[str, Any]:
        if not isinstance(config_path, (str, Path)):
            raise TypeError(f"{field_name} must be a JSON file path.")
        path = Path(config_path)
        if path.suffix.lower() != ".json":
            raise ValueError(f"{field_name} must point to a JSON file.")
        with path.open() as config_file:
            return json.load(config_file)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.layer_skip is not None:
            result["skip_layers"] = list(self.layer_skip.layers)
        return result


def _extract_skip_layers(raw_config: dict[str, Any]) -> Any | None:
    if "skip_layers" in raw_config:
        return raw_config["skip_layers"]
    if "layers" in raw_config:
        return raw_config["layers"]

    transforms = raw_config.get("transforms")
    if transforms is None and isinstance(raw_config.get("plan"), dict):
        transforms = raw_config["plan"].get("transforms")
    if transforms is None:
        return None
    if not isinstance(transforms, list):
        raise TypeError("transforms must be a list when provided in a layer skip configuration.")

    for transform in transforms:
        if isinstance(transform, dict) and transform.get("kind") == "skip_layers":
            return transform.get("skip_layers", transform.get("layers"))
    return None
