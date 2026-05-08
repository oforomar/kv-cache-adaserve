"""LayerSignals dataclass and the cheap signal extractors used across stages.

The three runtime signals are computed once per (sequence, layer) during the
prefill pass in `calibration.collect_signals` and consumed by the Phase A
heuristic (`selector.select_phase_a`) and the Phase B classifier.

Reductions happen inside `attention_entropy`/`head_variance`; both are designed
to be called inside a forward hook with eager attention weights and reduced to
a scalar before the `[B, H, q, k]` tensor escapes to caller scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerSignals:
    entropy: float              # mean attention entropy in nats
    entropy_normalized: float   # entropy / log(seq_len) ∈ [0, 1]
    head_variance: float        # cross-head heterogeneity, see head_variance()
    seq_len: int
    layer_idx: int


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Per-(batch, head, query) attention entropy in nats.

    weights: [B, H, q, k] from an HF attention module with output_attentions=True.
    Returns a [B, H, q] tensor; callers typically `.mean()` to a scalar.
    """
    eps = 1e-12
    p = weights.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def head_variance(weights: torch.Tensor) -> torch.Tensor:
    """Cross-head heterogeneity: variance across heads of per-head peak attention.

    weights: [B, H, q, k]. Returns a 0-d tensor.

    Many "head variance" definitions are reasonable. We use the variance across
    heads of each head's mean peak (max over keys) attention probability:
    peaky heads → high mean-max, diffuse heads → low. Variance captures how
    heterogeneously peaked the heads are. The exact statistic affects
    `HeuristicConfig.tau_head_var` calibration — keep them consistent.

    Note on GQA: HF returns weights post K/V repetition, so the head axis is
    Q-heads, not KV-heads. The design doc calls for variance over KV heads;
    with GQA, group adjacent Q-heads using `num_key_value_groups` before
    taking the variance. Not done here — flag for the GQA-specific run.
    """
    per_head_peak = weights.amax(dim=-1).mean(dim=(0, 2))  # [H]
    return per_head_peak.var(unbiased=False)
