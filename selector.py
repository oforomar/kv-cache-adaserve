"""Phase A heuristic selector — explicit precedence, no scoring function.

Precedence (highest priority first), per the design doc:
  1. QAQ-capable model (offline flag)        → QAQ
  2. high head variance                       → Ada-KV
  3. low entropy                              → DynamicKV
  4. high entropy + long context (≥4096)      → KVQuant 3-bit
  5. high entropy + short context             → KVQuant 8-bit
  6. mid-entropy default                      → KVQuant 8-bit

Phase A operates on `entropy_normalized` (∈ [0, 1]), not raw entropy. The
mock validation showed `tau_head_var` is the most sensitive single number;
the head-variance check fires before everything else, so a mis-tuned
threshold produces 100% Ada-KV labels.
"""
from __future__ import annotations

from dataclasses import dataclass

from signals import LayerSignals
from strategies import Strategy


@dataclass
class HeuristicConfig:
    qaq_capable: bool = False
    tau_head_var: float = 1e-4
    tau_entropy_low: float = 0.5
    tau_entropy_high: float = 0.8
    tau_seq_long: int = 4096


def select_phase_a(s: LayerSignals, cfg: HeuristicConfig) -> Strategy:
    if cfg.qaq_capable:
        return Strategy.QAQ
    if s.head_variance > cfg.tau_head_var:
        return Strategy.ADAKV
    if s.entropy_normalized < cfg.tau_entropy_low:
        return Strategy.DYNAMICKV
    if s.entropy_normalized > cfg.tau_entropy_high:
        if s.seq_len >= cfg.tau_seq_long:
            return Strategy.KVQUANT_3B
        return Strategy.KVQUANT_8B
    return Strategy.KVQUANT_8B
