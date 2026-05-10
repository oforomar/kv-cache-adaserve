"""Evaluate trained Phase B classifiers.

Loads each models/phase_b_lam*.pt checkpoint, reconstructs the same
prompt-level val split used during training, and reports:
  - Overall accuracy vs. majority-class baseline
  - Confusion matrix
  - Per-bucket accuracy (short / medium / long)
  - Feature statistics per predicted class
  - Phase A agreement with explanation

Usage:
  python evaluate_classifier.py --data-dir calib100 --model-dir models
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_b import (
    LABEL2IDX, IDX2LABEL, N_CLASSES,
    PhaseBMLP, featurize_signals, load_phase_b,
)
from selector import HeuristicConfig, select_phase_a
from signals import LayerSignals
from train_classifier import (
    featurize_row, split_by_prompt, split_by_prompt_stratified, _row_to_signals,
)

LABELS = [IDX2LABEL[i] for i in range(N_CLASSES)]

LENGTH_BUCKETS = {
    "short":  (0,    1024),
    "medium": (1024, 4096),
    "long":   (4096, float("inf")),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def seq_bucket(seq_len: int) -> str:
    for name, (lo, hi) in LENGTH_BUCKETS.items():
        if lo <= seq_len < hi:
            return name
    return "long"


def majority_baseline(rows: list[dict]) -> float:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    return max(counts.values()) / len(rows)


def confusion_matrix(
    model: PhaseBMLP, rows: list[dict], device: torch.device
) -> list[list[int]]:
    cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    model.eval()
    with torch.no_grad():
        for r in rows:
            feats, true_idx = featurize_row(r)
            x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            pred_idx = int(model(x).argmax(dim=-1).item())
            cm[true_idx][pred_idx] += 1
    return cm


def accuracy_by_bucket(
    model: PhaseBMLP, rows: list[dict], device: torch.device
) -> dict[str, dict]:
    bucket_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket_rows[seq_bucket(r["signals"]["seq_len"])].append(r)

    results = {}
    model.eval()
    with torch.no_grad():
        for bucket, brows in sorted(bucket_rows.items()):
            correct = 0
            dist: dict[str, int] = defaultdict(int)
            pred_dist: dict[str, int] = defaultdict(int)
            for r in brows:
                feats, true_idx = featurize_row(r)
                x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
                pred_idx = int(model(x).argmax(dim=-1).item())
                dist[IDX2LABEL[true_idx]] += 1
                pred_dist[IDX2LABEL[pred_idx]] += 1
                if pred_idx == true_idx:
                    correct += 1
            results[bucket] = {
                "n": len(brows),
                "accuracy": correct / max(1, len(brows)),
                "true_dist": dict(dist),
                "pred_dist": dict(pred_dist),
            }
    return results


def feature_stats_by_pred(
    model: PhaseBMLP, rows: list[dict], device: torch.device
) -> dict[str, dict]:
    """Mean feature values grouped by predicted class — shows what drives predictions."""
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for r in rows:
            feats, _ = featurize_row(r)
            x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            pred = IDX2LABEL[int(model(x).argmax(dim=-1).item())]
            buckets[pred].append(feats)

    stats = {}
    feat_names = ["entropy_norm", "log1p_seq_len", "head_var", "layer_idx_norm"]
    for lbl, feat_list in buckets.items():
        t = torch.tensor(feat_list)
        stats[lbl] = {
            name: {"mean": float(t[:, i].mean()), "std": float(t[:, i].std())}
            for i, name in enumerate(feat_names)
        }
    return stats


def phase_a_dist(val_rows: list[dict]) -> dict[str, int]:
    cfg = HeuristicConfig()
    dist: dict[str, int] = defaultdict(int)
    for r in val_rows:
        ls = _row_to_signals(r)
        dist[select_phase_a(ls, cfg).value] += 1
    return dict(dist)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_confusion(cm: list[list[int]], labels: list[str]) -> None:
    col_w = 11
    header_lbl = "true\\pred"
    header = f"{header_lbl:<14}" + "".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    print(header)
    print("─" * len(header))
    for i, row_lbl in enumerate(labels):
        row = f"{row_lbl:<14}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(labels)))
        print(row)


def evaluate_one(
    model_path: str,
    data_dir: str,
    val_frac: float,
    seed: int,
    device: torch.device,
    stratify: bool = False,
) -> None:
    model, meta = load_phase_b(model_path, device=str(device))
    lam = meta["lambda_compress"]

    data_path = Path(data_dir) / f"training_lam{lam}.jsonl"
    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    split_fn = split_by_prompt_stratified if stratify else split_by_prompt
    _, val_rows = split_fn(rows, val_frac=val_frac, seed=seed)

    print(f"\n{'═' * 60}")
    print(f"  λ = {lam}   |   checkpoint: {Path(model_path).name}")
    print(f"{'═' * 60}")
    print(f"  val set: {len(val_rows)} rows, "
          f"{len({r['prompt_id'] for r in val_rows})} prompts")

    # ── Baseline ──────────────────────────────────────────────────────────────
    maj = majority_baseline(val_rows)
    print(f"\n  majority-class baseline : {maj:.3f}")
    print(f"  Phase B val accuracy    : {meta['val_accuracy']:.3f}  "
          f"({'▲ +' if meta['val_accuracy'] > maj else '▼ '}"
          f"{abs(meta['val_accuracy'] - maj):.3f} vs majority)")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(model, val_rows, device)
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print("  ", end="")
    print_confusion(cm, LABELS)

    # Per-class precision / recall
    print(f"\n  Per-class precision / recall:")
    print(f"  {'class':<14}  {'precision':>10}  {'recall':>8}  {'support':>8}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*8}")
    for i, lbl in enumerate(LABELS):
        tp = cm[i][i]
        pred_total = sum(cm[r][i] for r in range(N_CLASSES))
        true_total = sum(cm[i][c] for c in range(N_CLASSES))
        prec = tp / max(1, pred_total)
        rec  = tp / max(1, true_total)
        print(f"  {lbl:<14}  {prec:>10.3f}  {rec:>8.3f}  {true_total:>8}")

    # ── Accuracy by length bucket ─────────────────────────────────────────────
    print(f"\n  Accuracy by sequence-length bucket:")
    print(f"  {'bucket':<8}  {'n':>5}  {'accuracy':>9}  "
          f"{'true labels (top 2)':>24}  {'predicted (top 2)':>20}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*9}  {'─'*24}  {'─'*20}")
    buckets = accuracy_by_bucket(model, val_rows, device)
    for bname in ("short", "medium", "long"):
        if bname not in buckets:
            continue
        b = buckets[bname]
        top_true = sorted(b["true_dist"].items(), key=lambda x: -x[1])[:2]
        top_pred = sorted(b["pred_dist"].items(), key=lambda x: -x[1])[:2]
        true_str = ", ".join(f"{l}:{n}" for l, n in top_true)
        pred_str = ", ".join(f"{l}:{n}" for l, n in top_pred)
        print(f"  {bname:<8}  {b['n']:>5}  {b['accuracy']:>9.3f}  "
              f"{true_str:>24}  {pred_str:>20}")

    # ── Feature stats by predicted class ─────────────────────────────────────
    print(f"\n  Mean feature values by predicted class:")
    print(f"  {'class':<14}  {'entropy_n':>10}  {'log1p_len':>10}  "
          f"{'head_var':>10}  {'layer_n':>8}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    fstats = feature_stats_by_pred(model, val_rows, device)
    for lbl in LABELS:
        if lbl not in fstats:
            print(f"  {lbl:<14}  (never predicted)")
            continue
        s = fstats[lbl]
        print(f"  {lbl:<14}  "
              f"{s['entropy_norm']['mean']:>10.3f}  "
              f"{s['log1p_seq_len']['mean']:>10.3f}  "
              f"{s['head_var']['mean']:>10.4f}  "
              f"{s['layer_idx_norm']['mean']:>8.3f}")

    # ── Phase A comparison ────────────────────────────────────────────────────
    pa_dist = phase_a_dist(val_rows)
    print(f"\n  Phase A prediction distribution on val set:")
    for lbl, cnt in sorted(pa_dist.items(), key=lambda x: -x[1]):
        print(f"    {lbl:<14}  {cnt:>5}  ({cnt / len(val_rows):.1%})")
    print(f"  → Phase A agreement: {meta['phase_a_agreement']:.3f}  "
          f"(expected ~0 — tau_head_var=1e-4 is miscalibrated; "
          f"see CLAUDE.md §'Phase A precedence')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",      default="calib100")
    ap.add_argument("--model-dir",     default="models")
    ap.add_argument("--val-frac",      type=float, default=0.2)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--device",        default="cpu")
    ap.add_argument("--stratify-split", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)

    checkpoints = sorted(model_dir.glob("phase_b_lam*.pt"))
    if not checkpoints:
        sys.exit(f"no checkpoints found in {model_dir}")

    print(f"Found {len(checkpoints)} checkpoint(s): "
          f"{[p.name for p in checkpoints]}")

    for ckpt in checkpoints:
        evaluate_one(
            model_path=str(ckpt),
            data_dir=args.data_dir,
            val_frac=args.val_frac,
            seed=args.seed,
            device=device,
            stratify=args.stratify_split,
        )

    print(f"\n{'═' * 60}")
    print("  Pareto summary across λ values:")
    print(f"  {'λ':<6}  {'val_acc':>8}  {'note'}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*40}")
    rows_summary = []
    for ckpt in checkpoints:
        meta = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        lam  = meta["lambda_compress"]
        acc  = meta["val_accuracy"]
        dom  = max(meta["per_class_accuracy"], key=lambda k: meta["per_class_accuracy"][k])
        rows_summary.append((lam, acc, dom))
    for lam, acc, dom in rows_summary:
        print(f"  {lam:<6}  {acc:>8.3f}  dominant class: {dom}")
    print()


if __name__ == "__main__":
    main()
