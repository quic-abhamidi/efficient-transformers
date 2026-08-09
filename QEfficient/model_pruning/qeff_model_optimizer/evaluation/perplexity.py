"""Perplexity and language-modeling quality metrics for NAS plans.

Provides dataset-aware perplexity computation that integrates with the existing
NAS analysis datasets (``nas.analysis.datasets``). The primary entry point,
:func:`compute_perplexity`, accepts either a list of raw texts or a list of
dataset names and returns per-dataset plus overall perplexity.

Typical use:

.. code-block:: python

    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import compute_perplexity

    # Using datasets loaded on-the-fly
    report = compute_perplexity(
        model, tokenizer,
        datasets=["wikitext", "mmlu_pro"],
        num_samples=50, max_length=512,
    )
    print(report.overall_perplexity)
    print(report.per_dataset["wikitext"].perplexity)

Design notes:
- Computes exp(mean cross-entropy loss) over all tokens (standard HF perplexity).
- Skips empty or single-token inputs that can't produce a meaningful loss.
- Per-dataset sub-reports are weighted by token count when aggregated.
- Does NOT apply any transforms — assumes the model is already in the desired
  state. Use with a transformed artifact after ``session.apply_plan(...)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import load_dataset_samples


# --- Report dataclasses ---

@dataclass(eq=True)
class DatasetPerplexity:
    """Perplexity result for a single dataset.

    ``perplexity`` is exp(mean cross-entropy over ``num_tokens`` tokens).
    ``num_samples`` is the count of non-empty texts that contributed.
    """
    dataset: str
    perplexity: float
    num_samples: int
    num_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "perplexity": self.perplexity,
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
        }


@dataclass(eq=True)
class PerplexityReport:
    """Combined perplexity report across one or more datasets.

    ``overall_perplexity`` is computed by pooling all tokens across every
    dataset and exponentiating the token-weighted mean loss. It is NOT the mean
    of per-dataset perplexities (those are geometric means already).
    """
    overall_perplexity: float
    total_tokens: int
    per_dataset: dict[str, DatasetPerplexity] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_perplexity": self.overall_perplexity,
            "total_tokens": self.total_tokens,
            "per_dataset": {
                name: ppl.to_dict() for name, ppl in self.per_dataset.items()
            },
            "metadata": dict(self.metadata),
        }


# --- Core computation ---

def _compute_ppl_on_texts(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
) -> tuple[float, int, int]:
    """Compute perplexity on a list of texts.

    Returns ``(perplexity, num_tokens, num_samples_used)``.
    Skips empty/whitespace texts and inputs tokenized to < 2 tokens
    (HF can't compute a loss without at least 2 tokens).
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    used_samples = 0

    for text in texts:
        if not text or not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Need at least 2 tokens: labels are shifted, so 1-token inputs
        # produce a zero-length loss.
        n_tokens = inputs["input_ids"].shape[1]
        if n_tokens < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        # HF returns mean loss across tokens; multiply back out so we can
        # aggregate properly with per-text token counts.
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
        used_samples += 1

    if total_tokens == 0:
        return float("inf"), 0, 0

    return math.exp(total_loss / total_tokens), total_tokens, used_samples


# --- Public API ---

def compute_perplexity(
    model,
    tokenizer,
    *,
    datasets: list[str] | None = None,
    texts: list[str] | None = None,
    num_samples: int = 50,
    max_length: int = 512,
) -> PerplexityReport:
    """Compute perplexity on real datasets or raw texts.

    Exactly one of ``datasets`` or ``texts`` must be provided.

    Parameters
    ----------
    model, tokenizer:
        A loaded causal LM and its tokenizer. The model should already have
        any transforms applied (e.g. after ``session.apply_plan``).
    datasets:
        List of dataset names recognised by
        :func:`~nas.analysis.datasets.load_dataset_samples`. ``num_samples``
        prompts are drawn from each.
    texts:
        Alternative to ``datasets`` — a raw list of strings. Used as a single
        unnamed dataset ("custom").
    num_samples:
        Number of samples to draw per dataset (ignored when ``texts`` is used).
    max_length:
        Maximum token length per sample; longer texts are truncated.

    Returns
    -------
    PerplexityReport
        Overall pooled-token perplexity plus per-dataset sub-reports.

    Raises
    ------
    ValueError
        If both ``datasets`` and ``texts`` are provided, or neither is.
    """
    if (datasets is None) == (texts is None):
        raise ValueError(
            "Exactly one of `datasets` or `texts` must be provided"
        )

    per_dataset: dict[str, DatasetPerplexity] = {}
    all_texts: list[str] = []

    if datasets is not None:
        # Load each dataset fresh and score independently.
        for ds_name in datasets:
            ds_texts = load_dataset_samples(ds_name, num_samples)
            ppl, n_tok, n_samp = _compute_ppl_on_texts(
                model, tokenizer, ds_texts, max_length,
            )
            per_dataset[ds_name] = DatasetPerplexity(
                dataset=ds_name,
                perplexity=round(ppl, 4),
                num_samples=n_samp,
                num_tokens=n_tok,
            )
            all_texts.extend(ds_texts)
    else:
        # Raw-text mode: treat as one unnamed dataset.
        ppl, n_tok, n_samp = _compute_ppl_on_texts(
            model, tokenizer, texts, max_length,
        )
        per_dataset["custom"] = DatasetPerplexity(
            dataset="custom",
            perplexity=round(ppl, 4),
            num_samples=n_samp,
            num_tokens=n_tok,
        )
        all_texts = list(texts)

    # Overall PPL pools ALL tokens across all datasets and recomputes; this is
    # the correct way to aggregate because per-dataset PPLs are geometric means.
    overall_ppl, total_tokens, _ = _compute_ppl_on_texts(
        model, tokenizer, all_texts, max_length,
    )

    return PerplexityReport(
        overall_perplexity=round(overall_ppl, 4),
        total_tokens=total_tokens,
        per_dataset=per_dataset,
        metadata={
            "num_samples_per_dataset": num_samples,
            "max_length": max_length,
            "mode": "datasets" if datasets is not None else "texts",
        },
    )


__all__ = [
    "DatasetPerplexity",
    "PerplexityReport",
    "compute_perplexity",
]
