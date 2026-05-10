"""Phase B runtime module: MLP definition + inference helper.

Kept separate from train_classifier.py so the runtime path only imports
this file and not the training code.

Runtime usage:
    from phase_b import load_phase_b, select_phase_b
    model, meta = load_phase_b("models/phase_b_lam1.0.pt")
    strategy = select_phase_b(layer_signals, model, num_layers=28)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from signals import LayerSignals
from strategies import RUNTIME_STRATEGIES, Strategy

LABEL2IDX: dict[str, int] = {s.value: i for i, s in enumerate(RUNTIME_STRATEGIES)}
IDX2LABEL: dict[int, str] = {v: k for k, v in LABEL2IDX.items()}
N_CLASSES = len(LABEL2IDX)


class PhaseBMLP(nn.Module):
    """3-layer MLP: 4 signals → one of 4 runtime KV cache strategies."""

    def __init__(self, n_features: int = 4, n_classes: int = N_CLASSES, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),     nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def featurize_signals(s: LayerSignals, num_layers: int) -> list[float]:
    """Map a LayerSignals to the 4-element feature vector the MLP expects.

    Matches the featurization documented in TRAINING.md:
      [entropy_normalized, log1p(seq_len), head_variance, layer_idx/(num_layers-1)]
    """
    return [
        s.entropy_normalized,
        math.log1p(s.seq_len),
        s.head_variance,
        s.layer_idx / max(1, num_layers - 1),
    ]


def load_phase_b(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[PhaseBMLP, dict]:
    """Load a trained Phase B checkpoint. Returns (model, metadata dict).

    The metadata dict contains: label2idx, idx2label, feature_names,
    lambda_compress, hidden, val_accuracy, phase_a_agreement.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = PhaseBMLP(n_features=4, n_classes=N_CLASSES, hidden=ckpt.get("hidden", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def select_phase_b(
    s: LayerSignals,
    model: PhaseBMLP,
    num_layers: int,
    device: torch.device | None = None,
) -> Strategy:
    """Predict the best KV cache strategy for a (prompt, layer) signal vector.

    Drop-in replacement for select_phase_a() — same inputs, same output type.
    Inference takes < 0.1 ms on CPU.
    """
    feats = featurize_signals(s, num_layers)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    idx = int(model(x).argmax(dim=-1).item())
    return Strategy(IDX2LABEL[idx])
