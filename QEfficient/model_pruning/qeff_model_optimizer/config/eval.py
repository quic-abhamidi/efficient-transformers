"""Evaluation and compile specs for the HuggingFace and QEff runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=True)
class EvalSpec:
    """Evaluation settings forwarded to the HuggingFace lm-eval harness.

    ``tasks`` must contain at least one lm-evaluation-harness task name (e.g.
    ``["hellaswag", "arc_challenge"]``).  All other fields map 1-to-1 to the
    matching ``lm_eval`` arguments; see the lm-eval docs for full semantics.

    ``limit`` caps the number of examples evaluated per task (useful for
    smoke-testing); ``None`` evaluates the full task split.
    ``num_fewshot`` overrides the task-default shot count; ``None`` keeps the
    task default.
    ``use_cache`` / ``cache_dir`` enable result caching between runs.
    """

    tasks: list[str]
    batch_size: int = 16
    device: str = "cuda"
    limit: int | None = None
    num_fewshot: int | None = None
    use_cache: bool = False
    cache_dir: str | None = None
    dtype: str = "auto"
    log_samples: bool = False
    random_seed: int = 42
    verbosity: str = "INFO"

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("tasks must contain at least one task")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative when provided")


@dataclass(eq=True)
class QEffCompileSpec:
    """Compile and execution settings for the QEff (QAIC) runtime.

    ``ctx_len`` is the total KV-cache context length (prefill + decode).
    ``prefill_seq_len`` is the maximum number of tokens in a single prefill
    step; it must be <= ``ctx_len``.

    ``device_group`` specifies which QAIC device IDs to use; ``None`` defaults
    to a single device (device 0).  The number of devices determines the
    ``num_devices`` compile argument.

    ``continuous_batching`` enables the continuous-batching path; when ``True``
    the ``full_batch_size`` compile argument is set equal to ``batch_size``.

    ``qaic_config`` is a free-form dict forwarded to the QEff model constructor
    and compiler.  Keys matching ``QEFF_COMPILE_OPTION_KEYS`` in the runtime
    are extracted and passed to ``compile()``.
    """

    ctx_len: int = 4096
    prefill_seq_len: int = 32
    batch_size: int = 1
    num_cores: int = 16
    continuous_batching: bool = False
    device_group: list[int] | None = None
    qaic_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ctx_len <= 0:
            raise ValueError("ctx_len must be positive")
        if self.prefill_seq_len <= 0:
            raise ValueError("prefill_seq_len must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_cores <= 0:
            raise ValueError("num_cores must be positive")
