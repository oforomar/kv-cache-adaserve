"""K-fold cross-validation for the Phase B MLP classifier.

Splits the 100 prompts into k folds at the prompt level (not row level),
trains one model per fold, and aggregates precision / recall / F1 across
all folds. The aggregate confusion matrix is the sum over all held-out folds
so every row is evaluated exactly once.

Usage:
  python kfold_eval.py --data calib100/training_lam1.0.jsonl --k 5
  python kfold_eval.py --data calib100/training_lam1.0.jsonl --k 5 --epochs 100
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_b import LABEL2IDX, IDX2LABEL, N_CLASSES, PhaseBMLP, featurize_signals
from train_classifier import (
    CalibDataset, compute_class_weights, featurize_row,
)

LABELS = [IDX2LABEL[i] for i in range(N_CLASSES)]


# ── Prompt-level k-fold splitter ──────────────────────────────────────────────

def make_folds(rows: list[dict], k: int, seed: int) -> list[tuple[list, list]]:
    """Return k (train_rows, val_rows) splits at the prompt level."""
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["prompt_id"]].append(r)

    pids = sorted(by_pid)
    random.Random(seed).shuffle(pids)

    # Divide prompts into k chunks
    fold_size = len(pids) // k
    chunks = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else len(pids)
        chunks.append(pids[start:end])

    folds = []
    for i in range(k):
        val_pids   = set(chunks[i])
        train_pids = set(p for j, chunk in enumerate(chunks)
                         if j != i for p in chunk)
        train_rows = [r for pid in train_pids for r in by_pid[pid]]
        val_rows   = [r for pid in val_pids   for r in by_pid[pid]]
        folds.append((train_rows, val_rows))
    return folds


# ── Train one fold ────────────────────────────────────────────────────────────

def train_fold(
    train_rows: list[dict],
    val_rows: list[dict],
    epochs: int,
    hidden: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> PhaseBMLP:
    torch.manual_seed(seed)

    train_loader = DataLoader(
        CalibDataset(train_rows), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    weights  = compute_class_weights(train_rows, device)
    model    = PhaseBMLP(n_features=4, n_classes=N_CLASSES, hidden=hidden).to(device)
    optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn  = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_acc   = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optim.zero_grad(); loss.backward(); optim.step()
        scheduler.step()

        # Track best by val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for r in val_rows:
                feats, true_idx = featurize_row(r)
                x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
                if int(model(x).argmax(dim=-1)) == true_idx:
                    correct += 1
                total += 1
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    return model


# ── Evaluate one fold ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_fold(model: PhaseBMLP, val_rows: list[dict], device: torch.device) -> list[list[int]]:
    cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    model.eval()
    for r in val_rows:
        feats, true_idx = featurize_row(r)
        x    = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        pred = int(model(x).argmax(dim=-1))
        cm[true_idx][pred] += 1
    return cm


def cm_to_metrics(cm: list[list[int]]) -> dict[str, dict]:
    out = {}
    for i, lbl in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(N_CLASSES)) - tp
        fn = sum(cm[i][c] for c in range(N_CLASSES)) - tp
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[lbl] = {"precision": p, "recall": r, "f1": f1,
                    "support": sum(cm[i]), "tp": tp}
    return out


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def add_cm(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[a[i][j] + b[i][j] for j in range(N_CLASSES)] for i in range(N_CLASSES)]


def mean_std(values: list[float]) -> tuple[float, float]:
    n    = len(values)
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def print_cm(cm: list[list[int]]) -> None:
    w      = 11
    corner = "true \\ pred"
    header = f"  {corner:<14}" + "".join(f"{l[:w]:>{w}}" for l in LABELS) + f"{'|  support':>11}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, lbl in enumerate(LABELS):
        row_sum = sum(cm[i])
        print(f"  {lbl:<14}" + "".join(f"{cm[i][j]:>{w}}" for j in range(N_CLASSES))
              + f"  |{row_sum:>7}")
    col_sums = [sum(cm[r][c] for r in range(N_CLASSES)) for c in range(N_CLASSES)]
    print("  " + "─" * (len(header) - 2))
    print(f"  {'predicted':<14}" + "".join(f"{s:>{w}}" for s in col_sums)
          + f"  |{sum(col_sums):>7}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         required=True,
                    help="training_lam*.jsonl from make_labels.py")
    ap.add_argument("--k",            type=int,   default=5)
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--hidden",       type=int,   default=64)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size",   type=int,   default=128)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",       default="auto")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    n_prompts = len({r["prompt_id"] for r in rows})
    print(f"loaded {len(rows)} rows from {args.data}  ({n_prompts} prompts)")
    print(f"k={args.k} folds → ~{n_prompts//args.k} val prompts / ~{n_prompts - n_prompts//args.k} train prompts per fold")
    print(f"device: {device}\n")

    folds = make_folds(rows, k=args.k, seed=args.seed)

    # Per-fold metrics storage
    fold_accs:    list[float]             = []
    fold_metrics: list[dict[str, dict]]   = []
    agg_cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]

    for fold_idx, (train_rows, val_rows) in enumerate(folds):
        n_train_pids = len({r["prompt_id"] for r in train_rows})
        n_val_pids   = len({r["prompt_id"] for r in val_rows})

        # Label distribution in this fold's val
        val_dist: dict[str, int] = defaultdict(int)
        for r in val_rows:
            val_dist[r["label"]] += 1

        print(f"  fold {fold_idx+1}/{args.k}  "
              f"train={len(train_rows)} rows ({n_train_pids} prompts)  "
              f"val={len(val_rows)} rows ({n_val_pids} prompts)")
        print(f"    val labels: " +
              "  ".join(f"{l}:{val_dist.get(l,0)}" for l in LABELS))

        model = train_fold(
            train_rows, val_rows,
            epochs=args.epochs, hidden=args.hidden,
            lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, device=device,
            seed=args.seed + fold_idx,
        )

        cm      = eval_fold(model, val_rows, device)
        metrics = cm_to_metrics(cm)
        acc     = sum(cm[i][i] for i in range(N_CLASSES)) / max(1, len(val_rows))

        fold_accs.append(acc)
        fold_metrics.append(metrics)
        agg_cm = add_cm(agg_cm, cm)

        per_class_str = "  ".join(
            f"{l[:6]}:F1={metrics[l]['f1']:.2f}" for l in LABELS
        )
        print(f"    acc={acc:.3f}  {per_class_str}\n")

    # ── Aggregate results ─────────────────────────────────────────────────────
    print("=" * 64)
    print("  AGGREGATE CONFUSION MATRIX  (all folds, every row evaluated once)")
    print("=" * 64)
    print_cm(agg_cm)
    agg_metrics = cm_to_metrics(agg_cm)
    agg_acc = sum(agg_cm[i][i] for i in range(N_CLASSES)) / max(1, sum(sum(r) for r in agg_cm))

    print(f"\n{'=' * 64}")
    print(f"  PER-CLASS METRICS  (mean ± std across {args.k} folds)")
    print(f"{'=' * 64}")
    print(f"  {'class':<14}  {'precision':>12}  {'recall':>12}  {'f1':>12}  {'support':>8}")
    print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")

    active_f1s = []
    for lbl in LABELS:
        ps  = [fold_metrics[f][lbl]["precision"] for f in range(args.k)]
        rs  = [fold_metrics[f][lbl]["recall"]    for f in range(args.k)]
        f1s = [fold_metrics[f][lbl]["f1"]        for f in range(args.k)]
        sup = agg_metrics[lbl]["support"]

        mp, sp = mean_std(ps)
        mr, sr = mean_std(rs)
        mf, sf = mean_std(f1s)

        flag = ""
        if sup == 0:
            flag = "  ← never in val (any fold)"
        elif all(v == 0 for v in f1s):
            flag = "  ← F1=0 every fold"
        elif any(v == 0 for v in f1s):
            flag = f"  ← F1=0 in {sum(1 for v in f1s if v==0)}/{args.k} folds"

        print(f"  {lbl:<14}  {mp:>6.3f}±{sp:.3f}  {mr:>6.3f}±{sr:.3f}  "
              f"{mf:>6.3f}±{sf:.3f}  {sup:>8}{flag}")

        if sup > 0:
            active_f1s.append(mf)

    mac_f1_mean = sum(active_f1s) / len(active_f1s) if active_f1s else 0.0
    acc_mean, acc_std = mean_std(fold_accs)

    print(f"\n  {'macro F1 (active)':<30}  {mac_f1_mean:.3f}")
    print(f"  {'accuracy mean ± std':<30}  {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"  {'aggregate accuracy':<30}  {agg_acc:.3f}")

    print(f"\n  Per-fold accuracies: " +
          "  ".join(f"fold{i+1}={v:.3f}" for i, v in enumerate(fold_accs)))


if __name__ == "__main__":
    main()
