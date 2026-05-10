"""Phase B MLP classifier trainer.

Trains one 3-layer MLP per call on the training JSONL produced by make_labels.py.
Saves a .pt checkpoint loadable by phase_b.load_phase_b().

Input features (per TRAINING.md):
  [entropy_normalized, log1p(seq_len), head_variance, layer_idx / (num_layers-1)]

Output classes (RUNTIME_STRATEGIES order from strategies.py):
  kvquant_8b=0  kvquant_3b=1  dynamickv=2  adakv=3

Train/val split is by prompt_id (not row-randomly) — each prompt contributes
one row per layer, so a random row split would leak same-prompt data across
the boundary and inflate val accuracy.

Usage:
  uv run python train_classifier.py \\
      --data calib100/training_lam1.0.jsonl \\
      --out models/phase_b_lam1.0.pt

  # All three lambdas:
  for LAM in 0.1 1.0 10.0; do
    uv run python train_classifier.py \\
        --data calib100/training_lam${LAM}.jsonl \\
        --out models/phase_b_lam${LAM}.pt
  done
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_b import (  # noqa: E402
    LABEL2IDX, IDX2LABEL, N_CLASSES,
    PhaseBMLP, featurize_signals,
)
from selector import HeuristicConfig, select_phase_a  # noqa: E402
from signals import LayerSignals                      # noqa: E402

FEATURE_NAMES = [
    "entropy_normalized",
    "log1p_seq_len",
    "head_variance",
    "layer_idx_normalized",
]


# ── Featurize a training-JSONL row ────────────────────────────────────────────

def _row_to_signals(row: dict) -> LayerSignals:
    s = row["signals"]
    return LayerSignals(
        entropy=s["entropy"],
        entropy_normalized=s["entropy_normalized"],
        head_variance=s["head_variance"],
        seq_len=s["seq_len"],
        layer_idx=s["layer_idx"],
    )


def featurize_row(row: dict) -> tuple[list[float], int]:
    feats = featurize_signals(_row_to_signals(row), row["num_layers"])
    return feats, LABEL2IDX[row["label"]]


# ── Dataset ───────────────────────────────────────────────────────────────────

class CalibDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        feats, label = featurize_row(self.rows[i])
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ── Prompt-level train/val split ──────────────────────────────────────────────

def split_by_prompt(
    rows: list[dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["prompt_id"]].append(r)

    pids = sorted(by_pid)
    random.Random(seed).shuffle(pids)
    n_val = max(1, int(len(pids) * val_frac))

    val_pids = set(pids[:n_val])
    train_rows = [r for pid in pids[n_val:] for r in by_pid[pid]]
    val_rows   = [r for pid in pids[:n_val]  for r in by_pid[pid]]
    return train_rows, val_rows


def split_by_prompt_stratified(
    rows: list[dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Prompt-level split that guarantees every label class appears in val.

    Each prompt contributes one label (all its layer-rows share the same label).
    Prompts are grouped by label first; val_frac of each group goes to val,
    with a minimum of 1 prompt per class regardless of how rare it is.
    This prevents minority classes (e.g. adakv with 2 prompts) from falling
    entirely into the training split by chance.
    """
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["prompt_id"]].append(r)

    # Each prompt has one label — take the first row's label as representative.
    by_label: dict[str, list[str]] = defaultdict(list)
    for pid, prows in by_pid.items():
        by_label[prows[0]["label"]].append(pid)

    rng = random.Random(seed)
    val_pids: set[str] = set()
    train_pids: set[str] = set()

    for label, pids in by_label.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        n_val = max(1, int(len(pids) * val_frac))
        val_pids.update(pids[:n_val])
        train_pids.update(pids[n_val:])

    train_rows = [r for pid in train_pids for r in by_pid[pid]]
    val_rows   = [r for pid in val_pids   for r in by_pid[pid]]
    return train_rows, val_rows


# ── Class weights (inverse-frequency) ────────────────────────────────────────

def compute_class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = [0] * N_CLASSES
    for r in rows:
        counts[LABEL2IDX[r["label"]]] += 1
    total = sum(counts)
    weights = [total / (N_CLASSES * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: PhaseBMLP,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    correct = total = 0
    per_correct = [0] * N_CLASSES
    per_total   = [0] * N_CLASSES

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=-1)
        correct += int((preds == y).sum())
        total   += y.size(0)
        for c in range(N_CLASSES):
            mask = y == c
            per_correct[c] += int((preds[mask] == c).sum())
            per_total[c]   += int(mask.sum())

    return {
        "accuracy": correct / max(1, total),
        "per_class": {
            IDX2LABEL[i]: per_correct[i] / max(1, per_total[i])
            for i in range(N_CLASSES)
        },
    }


# ── Phase A agreement diagnostic ─────────────────────────────────────────────

@torch.no_grad()
def phase_a_agreement(
    model: PhaseBMLP,
    val_rows: list[dict],
    device: torch.device,
) -> float:
    """Fraction of val rows where Phase B prediction matches Phase A heuristic.

    Per the design doc:
      > 80%  → per-prompt labels are sufficient
      < 50%  → need real per-layer labels (much larger GPU budget)
    """
    cfg = HeuristicConfig()
    model.eval()
    agree = 0
    for r in val_rows:
        ls = _row_to_signals(r)
        phase_a_pred = select_phase_a(ls, cfg).value

        feats, _ = featurize_row(r)
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        phase_b_pred = IDX2LABEL[int(model(x).argmax(dim=-1).item())]

        if phase_a_pred == phase_b_pred:
            agree += 1
    return agree / max(1, len(val_rows))


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path: str,
    out_path: str,
    epochs: int = 50,
    hidden: int = 64,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 42,
    device_str: str = "auto",
    stratify_split: bool = False,
) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    rows = [json.loads(line) for line in open(data_path) if line.strip()]
    print(f"loaded {len(rows)} rows from {data_path}")

    split_fn = split_by_prompt_stratified if stratify_split else split_by_prompt
    split_label = "stratified" if stratify_split else "random"
    train_rows, val_rows = split_fn(rows, val_frac=val_frac, seed=seed)
    n_train_pids = len({r["prompt_id"] for r in train_rows})
    n_val_pids   = len({r["prompt_id"] for r in val_rows})
    print(f"split ({split_label}): {len(train_rows)} train rows ({n_train_pids} prompts) / "
          f"{len(val_rows)} val rows ({n_val_pids} prompts)")

    dist: dict[str, int] = defaultdict(int)
    for r in train_rows:
        dist[r["label"]] += 1
    print("train label distribution:")
    for lbl in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {lbl:<14}  {dist[lbl]:>5}  ({dist[lbl] / len(train_rows):.1%})")

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"device: {device}")

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        CalibDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(CalibDataset(val_rows), batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    weights = compute_class_weights(train_rows, device)
    print(f"class weights: { {IDX2LABEL[i]: f'{weights[i].item():.3f}' for i in range(N_CLASSES)} }")

    torch.manual_seed(seed)
    model    = PhaseBMLP(n_features=4, n_classes=N_CLASSES, hidden=hidden).to(device)
    optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn  = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_acc = 0.0
    best_state: dict | None = None

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\ntraining {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * y.size(0)
        scheduler.step()

        avg_loss = total_loss / len(train_rows)
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["accuracy"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{epochs}  loss {avg_loss:.4f}  "
                  f"val_acc {val_acc:.3f}  best {best_val_acc:.3f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)

    final     = evaluate(model, val_loader, device)
    agreement = phase_a_agreement(model, val_rows, device)

    print(f"\n{'─' * 52}")
    print(f"val accuracy       : {final['accuracy']:.3f}")
    print(f"phase A agreement  : {agreement:.3f}  "
          f"({'OK — per-prompt labels sufficient' if agreement >= 0.8 else 'LOW — consider per-layer labels'})")
    print("per-class accuracy :")
    for lbl, acc in final["per_class"].items():
        print(f"  {lbl:<14}  {acc:.3f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    stem    = Path(data_path).stem
    lam_str = stem.split("lam")[-1] if "lam" in stem else "unknown"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":        best_state,
        "label2idx":         LABEL2IDX,
        "idx2label":         IDX2LABEL,
        "feature_names":     FEATURE_NAMES,
        "lambda_compress":   lam_str,
        "hidden":            hidden,
        "val_accuracy":      final["accuracy"],
        "phase_a_agreement": agreement,
        "per_class_accuracy": final["per_class"],
    }, out_path)
    print(f"saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Phase B MLP classifier.")
    ap.add_argument(
        "--data", required=True,
        help="training_lam*.jsonl produced by make_labels.py",
    )
    ap.add_argument(
        "--out", required=True,
        help="output .pt checkpoint path (e.g. models/phase_b_lam1.0.pt)",
    )
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--hidden",       type=int,   default=64,
                    help="hidden layer width (default 64)")
    ap.add_argument("--batch-size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac",     type=float, default=0.2,
                    help="fraction of prompts held out for validation (default 0.2)")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",          default="auto",
                    help="cuda / cpu / auto (default: auto)")
    ap.add_argument("--stratify-split",  action="store_true",
                    help="guarantee every label class appears in val "
                         "(use when minority classes fall entirely into train)")
    args = ap.parse_args()

    train(
        data_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        hidden=args.hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        device_str=args.device,
        stratify_split=args.stratify_split,
    )
