# Multi-Signal Layer Weakness Analysis

**Date:** 2026-04-21
**Status:** Approved
**Replaces:** Threshold-based weak layer detection (`nas/analysis/weak_layers.py`)

## Problem

The current layer weakness detection uses a single metric (hidden-state cosine/L2 delta) with a percentile threshold. This produces inconsistent rankings across datasets and prompts — a layer ranked weakest on GSM8k may rank differently on HellaSwag. The root cause: hidden-state delta measures layer *activity* (how much it changes the representation), not layer *importance* (how much the model's output depends on that change).

## Solution

Replace the single-metric threshold approach with a multi-signal ensemble that combines 10 complementary importance signals across three compute tiers. The system outputs a continuous weakness index (0.0–1.0) with confidence bounds, eliminating the brittle percentile threshold.

## Requirements

- **Architectures:** Dense decoder-only (Llama, Qwen, Mistral, Gemma, Phi) and Mixture of Experts (Mixtral, DeepSeek-MoE, Qwen-MoE)
- **Compute budget:** Configurable via tiers (Tier 1: ~5-10 min, Tier 2: ~30-60 min, Tier 3: ~30-45 min loss-based / ~2-4 hours with full benchmark ablation, on A100 for a 7B model)
- **Output use:** General optimization signal — layer skipping, structured pruning, quantization priority, early exit
- **Recall/Precision:** Balanced — don't miss candidates, but keep the list actionable
- **Key metric:** Cross-dataset ranking stability must be significantly better than current approach

## API

### Entry Point

```python
def analyze_layer_weakness(
    model,
    tokenizer,
    tier: int = 2,                    # 1=fast, 2=medium, 3=full
    datasets: list[str] = DEFAULT_DATASETS,  # ["gsm8k", "hellaswag", "wikitext", "winogrande", "arc_easy"]
    num_samples: int = 200,
    batch_size: int = 16,
) -> WeaknessReport:
```

### Output Data Structures

```python
@dataclass
class LayerWeakness:
    layer: int
    weakness_index: float            # 0.0 = critical, 1.0 = most skippable
    confidence: float                # 0.0 = uncertain, 1.0 = all signals agree
    rank: int                        # 1 = weakest layer
    signal_scores: dict[str, float]  # Per-signal normalized scores (0-1)
    cross_dataset_stability: float   # How stable this ranking is across datasets
    tier_used: int                   # Highest tier computed for this layer

@dataclass
class WeaknessReport:
    model_spec: ModelSpec
    layers: list[LayerWeakness]      # Sorted by rank (weakest first)
    tier_used: int                   # Highest tier run
    datasets_used: list[str]
    timing: dict[str, float]         # Per-tier wall-clock time
    metadata: dict[str, Any]

    def weak_layers(
        self,
        threshold: float = 0.7,
        min_confidence: float = 0.5
    ) -> list[LayerWeakness]:
        """Layers above weakness threshold with sufficient confidence."""

    def candidates_for_skipping(self, max_layers: int = 5) -> list[LayerWeakness]:
        """Top-N weakest layers filtered by: weakness_index > 0.5 AND confidence > 0.3.
        Returns at most max_layers results, sorted weakest first."""
```

### CLI

```bash
python -m nas.analysis.weakness \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tier 2 \
    --datasets gsm8k hellaswag wikitext \
    --num-samples 200 \
    --batch-size 16 \
    --output-dir results/weakness/ \
    --output-format json,csv,plot
```

## Architecture

### Pipeline

```
CalibrationDataLoader → SignalAnalyzers (Tier 1/2/3) → RankAggregation → ConfidenceEstimator → WeaknessReport
```

### Module Structure

```
nas/analysis/
├── weakness/
│   ├── __init__.py                 # Public API: analyze_layer_weakness()
│   ├── analyzer.py                 # WeaknessAnalyzer orchestrator (tier management)
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseSignalAnalyzer protocol
│   │   ├── hidden_state_delta.py   # Signal 1 (refactored from existing)
│   │   ├── cka.py                  # Signal 2
│   │   ├── attention_entropy.py    # Signal 3
│   │   ├── weight_spectral.py      # Signal 4
│   │   ├── activation_stats.py     # Signal 5
│   │   ├── taylor_importance.py    # Signal 6
│   │   ├── fisher_information.py   # Signal 7
│   │   ├── gradient_flow.py        # Signal 8
│   │   ├── ablation.py             # Signal 9
│   │   └── pairwise_ablation.py    # Signal 10
│   ├── aggregation.py              # Rank aggregation, weight calibration
│   ├── confidence.py               # Bootstrap confidence, cross-dataset stability
│   └── report.py                   # WeaknessReport, LayerWeakness dataclasses
```

### BaseSignalAnalyzer Protocol

```python
class BaseSignalAnalyzer(Protocol):
    name: str
    tier: int  # 1, 2, or 3

    def analyze(
        self,
        model,
        tokenizer,
        calibration_data: list[str],
        batch_size: int = 16,
    ) -> dict[int, float]:
        """Returns {layer_index: raw_score} for all layers."""
        ...
```

Each analyzer is independent and can be run in any order. The orchestrator manages tier progression and passes cached forward-pass data to avoid recomputation.

## Tier 1 Signals (Forward-Only, ~5-10 min)

### Signal 1: Hidden-State Delta (existing, refactored)

Cosine distance between adjacent layer hidden states, averaged across batch and sequence dimensions.

```
score(l) = 1.0 - mean(cosine_sim(h_out_l, h_in_l))
```

Low delta = layer barely transforms hidden states = likely weak. This is the existing metric, kept as one signal among many.

### Signal 2: CKA Similarity

Centered Kernel Alignment between each layer's output and the final layer's output. Measures representational similarity — if removing a layer barely changes CKA with the final output, the layer is redundant.

Linear CKA implementation using Frobenius inner products of centered Gram matrices. Cost: O(N² × D) per layer pair, where N = batch size and D = hidden dimension.

More robust than cosine delta because CKA captures representational structure (which features are represented), not just direction/magnitude.

### Signal 3: Attention Entropy

Shannon entropy of each attention head's pattern: `H = -Σ p_ij log(p_ij)`, aggregated as mean entropy across heads within each layer.

- High entropy = diffuse, unfocused attention = potentially less important
- Low entropy with high mass on specific positions = structured operation

For GQA architectures (Llama, etc.): normalize per unique head group.

### Signal 4: Weight Spectral Analysis

Effective rank of each layer's weight matrices via singular values:

```
effective_rank = exp(entropy(σ / Σσ))
```

Computed for Q, K, V, O projection matrices and MLP weights. Low effective rank relative to full rank = weight matrix is approximately low-rank = likely compressible/skippable.

For MoE: compute per-expert and take weighted average by router frequency.

### Signal 5: Activation Statistics

- Layer output L2 norms, variance across tokens, kurtosis
- Dead neuron fraction: % of neurons firing on <1% of tokens
- For MoE: expert load balance (router entropy)

These sub-metrics are combined into a single score: compute z-scores for each sub-metric across layers, then take the mean z-score per layer. Higher mean z-score (more anomalous activation patterns) indicates a weaker layer.

Cheap to compute during the same forward pass as signals 1-3.

### Shared Forward Pass Optimization

Tier 1 signals 1-3 and 5 all need hidden states and attention weights from the same forward pass. One forward pass with hooks captures everything. Signal 4 reads model parameters directly (no forward pass needed).

## Tier 2 Signals (+ Gradients, ~30-60 min)

### Signal 6: Taylor Importance Score

First-order Taylor approximation of removing a layer:

```
importance(l) = Σ_d |h_l[d] × ∂L/∂h_l[d]|
```

Product of "how large is this layer's output" × "how sensitive is the loss to it." Most direct proxy for "what happens if I zero out this layer's contribution."

### Signal 7: Fisher Information Estimate

Diagonal Fisher information per layer:

```
Fisher(l) = E[(∂L/∂θ_l)²]
```

Aggregated: sum of squared gradients for all parameters in layer l, divided by parameter count. Accumulated over 10-20 mini-batches for stability.

### Signal 8: Gradient Flow Magnitude

```
grad_norm(l) = ||∂L/∂h_l||₂
grad_ratio(l) = ||∂L/∂h_l|| / ||∂L/∂h_{l-1}||
```

Layers with vanishing gradients or unit gradient ratio are passthrough layers — less impactful.

### Shared Backward Pass

All three Tier 2 signals share the same forward+backward pass. Hooks capture gradients w.r.t. hidden states (Taylor) and gradients w.r.t. parameters (Fisher, gradient flow) simultaneously.

### Loss Function

Next-token prediction loss (cross-entropy) for causal LMs. Model-agnostic — works for any architecture producing logits.

### MoE Gradient Handling

Gradients flow through the router's top-k selection via straight-through estimator. Fisher information for MoE layers includes router + all expert parameters.

## Tier 3 Signals (+ Ablation, ~2-4 hours)

### Signal 9: Single-Layer Ablation Score

For each layer: replace forward with identity, measure loss on calibration set.

```
ablation_score(l) = loss_with_layer_skipped - baseline_loss
```

Most direct measurement — exactly what downstream optimization will do.

#### Smart Sampling

- **Progressive evaluation:** Start with 50 samples. If ablation_score > 5× median after 50, mark layer as "critical" and stop
- **Tier-informed ordering:** Evaluate weakest layers first (from Tier 1+2). If budget runs low, skip layers already identified as critical
- **Early stopping:** If top-K and bottom-K layers are stable after N% of calibration set, stop

### Signal 10: Pairwise Interaction Ablation

For top-M candidate weak layers from Tier 1+2, test pairwise skip combinations:

```
interaction(l1, l2) = ablation(l1 + l2) - ablation(l1) - ablation(l2)
```

- Positive interaction = layers are complementary (skipping both is worse than sum of individual skips)
- Negative interaction = layers are redundant with each other

Only the top-10 weakest candidates are tested pairwise = 45 pair evaluations.

### Compute Budget

- Single-layer ablation (28 layers, 200 samples, batch 16): ~12 min on A100 for 7B model
- With progressive evaluation: typically 40-60% cheaper
- Pairwise ablation (45 pairs): ~20 additional minutes
- **Total Tier 3 (loss-based): ~30-45 min**
- Optional Tier 3b (full benchmark ablation): ~2-4 hours

## Rank Aggregation

### Step 1: Normalization

Convert raw scores to rank percentiles per signal:

```python
normalized[signal][l] = rank_position(l, raw_scores[signal]) / num_layers
# 0.0 = strongest layer, 1.0 = weakest layer
```

Rank-based normalization is robust to outliers and non-linear scales.

### Step 2: Weighted Average

```python
weakness_index(l) = Σ (weight[s] × normalized_rank[s][l]) / Σ weight[s]
```

Default signal weights:

| Signal | Weight | Tier | Rationale |
|--------|--------|------|-----------|
| Hidden-state delta | 1.0 | 1 | Baseline signal |
| CKA similarity | 1.5 | 1 | Structurally informative |
| Attention entropy | 0.8 | 1 | Supplementary, can be noisy |
| Weight spectral | 0.7 | 1 | Static property |
| Activation stats | 0.8 | 1 | Supplementary |
| Taylor importance | 2.0 | 2 | Best single gradient signal |
| Fisher information | 1.5 | 2 | Complementary to Taylor |
| Gradient flow | 1.0 | 2 | Catches passthrough layers |
| Ablation score | 3.0 | 3 | Ground truth |
| Pairwise interaction | 1.5 | 3 | Catches complementary layers |

### Auto-Calibration

When Tier 3 ablation data is available, auto-calibrate weights to maximize rank correlation (Kendall's tau) between the ensemble and ablation ground truth.

## Confidence Estimation

### Bootstrap Confidence

Bootstrap resample per-sample signal scores (already collected during analysis), re-aggregate, and measure rank stability:

```python
confidence(l) = 1.0 - (std_of_rank_position(l) across bootstrap resamples / num_layers)
```

High confidence = layer's rank is stable across resamples. Low confidence = sensitive to calibration data.

### Cross-Dataset Stability

```python
per_dataset_rankings[dataset] = aggregate(signals on this dataset only)
stability(l) = 1.0 - normalized_rank_variance_across_datasets(l)
```

Directly measures whether a layer's ranking is consistent across data distributions.

### Signal Disagreement Detection

```python
disagreement(l) = std(normalized_ranks[all_signals][l])
```

Layers with high disagreement are flagged as "investigation needed" — borderline layers surfaced for human review rather than silently classified.

## Integration

### Pipeline Change

```
Before: measure_layer_contributions → analyze_weak_layers (threshold) → generate_configurations
After:  analyze_layer_weakness (tiered) → WeaknessReport → generate_configurations
```

`analyze_layer_weakness()` replaces Stage 0 + part of Stage 1 in the current pipeline.

### Backward Compatibility

- `WeakLayerReport` stays available, marked deprecated
- `analyze_weak_layers()` becomes a thin wrapper calling `analyze_layer_weakness(tier=1)` and converting to old format
- Existing CSV/PNG output preserved as option alongside the new report
- Downstream config generation switches to `WeaknessReport`

### MoE Support

- Layer adapter extension: `resolve_layer_adapter()` extended to expose attention heads and expert modules per layer
- Expert utilization: router entropy added as bonus Tier 1 signal for MoE architectures
- Ablation granularity: Tier 3 ablation skips entire transformer blocks. Expert-level ablation is a future extension.

## Success Criteria

1. **Ranking stability:** Cross-dataset rank correlation (Kendall's tau) for top-10 weakest layers should be >0.7 (vs. current ~0.3-0.5 with cosine delta alone)
2. **Prediction accuracy:** Tier 2 weakness ranking should correlate with Tier 3 ablation ground truth at tau >0.8
3. **No missed candidates:** All layers that cause <1% accuracy drop when skipped (per ablation) should appear in the top-50% of the weakness ranking
4. **Compute targets:** Tier 1 <10 min, Tier 2 <60 min, Tier 3 <4 hours (7B model, A100)
