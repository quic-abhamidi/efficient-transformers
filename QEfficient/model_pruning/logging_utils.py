"""QEfficient logger integration for model_pruning."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

try:
    from ..utils.logging_utils import logger as _qeff_logger
except Exception:  # pragma: no cover - fallback for isolated tooling
    _qeff_logger = logging.getLogger("QEfficient")

_PACKAGE_PREFIX = "QEfficient.model_pruning"
_LOG_HANDLER_ATTR = "_qeff_model_pruning_file_handler"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a QEfficient child logger for model_pruning modules."""
    if not name:
        suffix = "model_pruning"
    else:
        normalized = name.removeprefix(_PACKAGE_PREFIX).lstrip(".")
        suffix = "model_pruning" if not normalized else f"model_pruning.{normalized}"
    return _qeff_logger.getChild(suffix)


def set_verbose_logging(enabled: bool) -> None:
    """Enable INFO-level model_pruning logs for long-running CLI stages."""
    if enabled:
        get_logger().setLevel(logging.INFO)


def configure_file_logging(output_dir: str | Path, *, enabled: bool = True) -> Path | None:
    """Attach a timestamped file handler for verbose model_pruning CLI logs."""
    if not enabled:
        return None

    root = get_logger()
    root.setLevel(logging.INFO)

    existing = getattr(root, _LOG_HANDLER_ATTR, None)
    if existing is not None:
        return Path(existing.baseFilename)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out / f"model_pruning_{timestamp}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    root.addHandler(handler)
    setattr(root, _LOG_HANDLER_ATTR, handler)
    root.info("[logging] writing verbose model_pruning logs to %s", log_path)
    return log_path


logger = get_logger()
