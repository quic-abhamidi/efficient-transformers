"""Benchmarking and report generation helpers for model pruning."""

__all__ = [
    "BENCHMARK_MAPPING",
    "BenchmarkReportGenerator",
    "load_model",
    "make_json_serializable",
    "run_lm_eval",
]


def __getattr__(name):
    if name == "BenchmarkReportGenerator":
        from QEfficient.model_pruning.benchmarking.generate_report import BenchmarkReportGenerator

        return BenchmarkReportGenerator
    if name in {"BENCHMARK_MAPPING", "load_model", "make_json_serializable", "run_lm_eval"}:
        from QEfficient.model_pruning.benchmarking import run_benchmark

        return getattr(run_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
