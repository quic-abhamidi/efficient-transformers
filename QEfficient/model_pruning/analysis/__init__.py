"""Maintained analysis utilities for layer contribution studies."""

__all__ = [
    "generate_layer_analysis",
    "generate_layer_range_analysis",
    "generate_delta_pattern_analysis",
    "ConfigPerformanceAnalyzer",
]


def __getattr__(name):
    if name == "generate_layer_analysis":
        from QEfficient.model_pruning.analysis.measure_layer_contributions import generate_layer_analysis

        return generate_layer_analysis
    if name == "generate_layer_range_analysis":
        from QEfficient.model_pruning.analysis.measure_layer_range_delta import generate_layer_range_analysis

        return generate_layer_range_analysis
    if name == "generate_delta_pattern_analysis":
        from QEfficient.model_pruning.analysis.analyze_embedding_delta_patterns import generate_delta_pattern_analysis

        return generate_delta_pattern_analysis
    if name == "ConfigPerformanceAnalyzer":
        from QEfficient.model_pruning.analysis.analyze_config_performance import ConfigPerformanceAnalyzer

        return ConfigPerformanceAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
