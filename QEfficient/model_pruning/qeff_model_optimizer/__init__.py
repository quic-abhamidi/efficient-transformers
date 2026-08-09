"""Neural Architecture Search (NAS) package.

Provides an API-first toolkit for analysing decoder-only transformer models,
identifying weak layers, applying reversible structural transforms (layer
skipping, compensation, head pruning), and evaluating the resulting artifacts
on HuggingFace or QEff runtimes.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
