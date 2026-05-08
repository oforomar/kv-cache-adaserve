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

    Chunked by head: the fp32 cast (needed because fp16's smallest subnormal
    is ~6e-8 and a 1e-12 clamp would underflow to 0, propagating NaN through
    `0 * log(0)`) doubles memory; on a long prompt (q=k=8K, H=24, fp16) the
    full-tensor cast is ~6 GB. Per-head pass is ~256 MB peak.
    """
    eps = 1e-12
    out = []
    for h in range(weights.shape[1]):
        p = weights[:, h:h + 1].float().clamp_min(eps)
        out.append(-(p * p.log()).sum(dim=-1))
    return torch.cat(out, dim=1)


def head_variance(weights: torch.Tensor, num_kv_groups: int = 1) -> torch.Tensor:
    """Cross-head heterogeneity: variance across KV heads of per-head peak attention.

    weights: [B, num_q_heads, q, k]. Returns a 0-d tensor.

    HF returns attention weights post K/V repetition, so the head axis is
    Q-heads. The design doc calls for variance over KV heads. With GQA we
    fold every `num_kv_groups` adjacent Q-heads into one KV-head value
    (average their peak-attention statistic) before taking the variance.
    For MHA (num_kv_groups=1) the fold is a no-op and behavior matches
    "variance across all heads".

    The statistic per head is the mean-over-(batch, query) of the row's
    peak-key probability — peaky heads → high, diffuse heads → low.
    Variance across (folded) heads captures how heterogeneously peaked
    the KV-head groups are. The exact statistic affects
    `HeuristicConfig.tau_head_var` calibration — keep them consistent.
    """
    per_q_head_peak = weights.amax(dim=-1).mean(dim=(0, 2))  # [num_q_heads]
    if num_kv_groups > 1:
        n_q = per_q_head_peak.shape[0]
        if n_q % num_kv_groups != 0:
            raise ValueError(
                f"num_q_heads ({n_q}) not divisible by num_kv_groups "
                f"({num_kv_groups})"
            )
        n_kv = n_q // num_kv_groups
        per_kv_head = per_q_head_peak.view(n_kv, num_kv_groups).mean(dim=1)
    else:
        per_kv_head = per_q_head_peak
    return per_kv_head.var(unbiased=False)
