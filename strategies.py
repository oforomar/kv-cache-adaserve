"""KV-cache strategy enum and the per-prompt scoring function.

Strategies are integrated via per-backend runner scripts under
`backends/runners/`, each running in its own environment and emitting a
JSONL of `{prompt_id, ppl, cratio}`. The orchestration layer
(`calibration.join_labels`) joins those per-strategy JSONLs with the
baseline JSONL and applies `score()` to pick the per-prompt label.

QAQ is offline-only and excluded from the runtime classifier; it remains
in the enum so Phase A can return it for QAQ-capable models, but the
runtime label space lives in `RUNTIME_STRATEGIES`.
"""
from __future__ import annotations

from enum import Enum


class Strategy(str, Enum):
    KVQUANT_8B = "kvquant_8b"
    KVQUANT_3B = "kvquant_3b"
    DYNAMICKV = "dynamickv"
    ADAKV = "adakv"
    QAQ = "qaq"


RUNTIME_STRATEGIES: tuple[Strategy, ...] = (
    Strategy.KVQUANT_8B,
    Strategy.KVQUANT_3B,
    Strategy.DYNAMICKV,
    Strategy.ADAKV,
)


def score(ppl_baseline: float, ppl_strategy: float, cratio: float,
          lambda_compress: float) -> float:
    """Higher is better. cratio = compressed_bytes / fp16_bytes ∈ (0, 1].

    score = -(Δppl) - λ·(1 - cratio)
    """
    delta_ppl = ppl_strategy - ppl_baseline
    return -delta_ppl - lambda_compress * (1.0 - cratio)
