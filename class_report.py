"""Class representation analysis + per-class confusion / precision / recall / F1."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_b import LABEL2IDX, IDX2LABEL, N_CLASSES, load_phase_b
from train_classifier import featurize_row, split_by_prompt

LABELS = [IDX2LABEL[i] for i in range(N_CLASSES)]
LAMS   = ["0.1", "1.0", "10.0"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_cm(model, rows, device):
    cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    model.eval()
    with torch.no_grad():
        for r in rows:
            feats, true_idx = featurize_row(r)
            x    = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            pred = int(model(x).argmax(dim=-1))
            cm[true_idx][pred] += 1
    return cm


def metrics_from_cm(cm):
    results = {}
    for i, lbl in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(N_CLASSES)) - tp
        fn = sum(cm[i][c] for c in range(N_CLASSES)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(cm[i])
        results[lbl] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": support, "tp": tp, "fp": fp, "fn": fn}
    return results


def print_cm(cm, title="Confusion matrix (rows=true, cols=predicted)"):
    w = 11
    print(f"\n  {title}")
    corner = "true \\ pred"
    header = f"  {corner:<14}" + "".join(f"{l[:w]:>{w}}" for l in LABELS) + f"{'|  support':>11}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, lbl in enumerate(LABELS):
        row_sum = sum(cm[i])
        row = f"  {lbl:<14}" + "".join(f"{cm[i][j]:>{w}}" for j in range(N_CLASSES))
        row += f"  |{row_sum:>7}"
        print(row)
    # Column totals
    col_sums = [sum(cm[r][c] for r in range(N_CLASSES)) for c in range(N_CLASSES)]
    total    = sum(col_sums)
    print("  " + "─" * (len(header) - 2))
    foot = f"  {'predicted':<14}" + "".join(f"{s:>{w}}" for s in col_sums)
    foot += f"  |{total:>7}"
    print(foot)


def print_metrics(metrics):
    print(f"\n  {'class':<14}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>8}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for lbl in LABELS:
        m = metrics[lbl]
        flag = "  ← never predicted" if m["support"] > 0 and m["tp"] == 0 else ""
        flag = "  ← never in val"   if m["support"] == 0                   else flag
        print(f"  {lbl:<14}  {m['precision']:>10.3f}  {m['recall']:>8.3f}  "
              f"{m['f1']:>8.3f}  {m['support']:>8}{flag}")


# ── 1. Class representation across the full dataset (all λ) ──────────────────

def show_representation(data_dir: str) -> None:
    print("=" * 64)
    print("  CLASS REPRESENTATION (full dataset, before split)")
    print("=" * 64)

    overall: dict[str, int] = defaultdict(int)
    for lam in LAMS:
        path = Path(data_dir) / f"training_lam{lam}.jsonl"
        rows = [json.loads(l) for l in open(path) if l.strip()]
        dist: dict[str, int] = defaultdict(int)
        for r in rows:
            dist[r["label"]] += 1
            overall[r["label"]] += 1

        print(f"\n  λ={lam}  (total {len(rows)} rows, "
              f"{len({r['prompt_id'] for r in rows})} prompts)")
        print(f"  {'class':<14}  {'rows':>6}  {'%':>6}  {'bar'}")
        print(f"  {'─'*14}  {'─'*6}  {'─'*6}  {'─'*30}")
        for lbl in sorted(dist, key=lambda k: -dist[k]):
            pct  = dist[lbl] / len(rows)
            bar  = "█" * int(pct * 40)
            flag = "  ✓ well-represented" if pct >= 0.15 else \
                   "  ⚠ sparse"           if pct >= 0.05 else \
                   "  ✗ severely under-represented"
            print(f"  {lbl:<14}  {dist[lbl]:>6}  {pct:>5.1%}  {bar:<30}{flag}")

    print(f"\n  Aggregate across all λ (8400 rows total):")
    print(f"  {'class':<14}  {'rows':>6}  {'%':>6}  verdict")
    print(f"  {'─'*14}  {'─'*6}  {'─'*6}  {'─'*35}")
    total_rows = sum(overall.values())
    for lbl in sorted(overall, key=lambda k: -overall[k]):
        pct = overall[lbl] / total_rows
        verdict = "well-represented" if pct >= 0.15 else \
                  "sparse (needs more)" if pct >= 0.05 else \
                  "SEVERELY under-represented"
        print(f"  {lbl:<14}  {overall[lbl]:>6}  {pct:>5.1%}  {verdict}")


# ── 2. Per-λ confusion matrix + precision / recall / F1 ──────────────────────

def show_per_lambda(data_dir: str, model_dir: str) -> None:
    device = torch.device("cpu")
    for lam in LAMS:
        ckpt = Path(model_dir) / f"phase_b_lam{lam}.pt"
        if not ckpt.exists():
            print(f"\n  [skip] {ckpt} not found")
            continue

        model, meta = load_phase_b(str(ckpt))
        path  = Path(data_dir) / f"training_lam{lam}.jsonl"
        rows  = [json.loads(l) for l in open(path) if l.strip()]
        _, val_rows = split_by_prompt(rows, val_frac=0.2, seed=42)

        print(f"\n{'=' * 64}")
        print(f"  λ = {lam}  |  val={len(val_rows)} rows  "
              f"|  val_acc={meta['val_accuracy']:.3f}")
        print(f"{'=' * 64}")

        cm      = build_cm(model, val_rows, device)
        metrics = metrics_from_cm(cm)

        print_cm(cm)
        print_metrics(metrics)

        # macro averages (exclude classes with 0 support)
        active = [lbl for lbl in LABELS if metrics[lbl]["support"] > 0]
        mac_p  = sum(metrics[l]["precision"] for l in active) / len(active)
        mac_r  = sum(metrics[l]["recall"]    for l in active) / len(active)
        mac_f1 = sum(metrics[l]["f1"]        for l in active) / len(active)
        print(f"\n  macro avg (active classes only: {len(active)}/4)")
        print(f"    precision {mac_p:.3f}  recall {mac_r:.3f}  f1 {mac_f1:.3f}")


# ── 3. Data limitation solutions ──────────────────────────────────────────────

def show_solutions() -> None:
    print(f"\n{'=' * 64}")
    print("  DATA LIMITATION SOLUTIONS")
    print("=" * 64)
    solutions = [
        ("Scale to full TARGET_MIX",
         "Run the full pipeline on ~1850 prompts (6 sources × 4 buckets).\n"
         "     52K rows → all 4 classes get adequate coverage at every λ.\n"
         "     This is the primary fix — the dataset is just too small."),

        ("Oversample minority classes",
         "In make_labels.py --target-size, use stratified oversampling\n"
         "     (sample with replacement for rare classes) rather than subsampling.\n"
         "     Keeps rare classes at ≥15% without adding GPU runs."),

        ("Focal loss instead of weighted CrossEntropy",
         "FocalLoss(gamma=2) down-weights easy majority examples and forces\n"
         "     the model to focus on hard minority ones (adakv, kvquant_3b at λ=0.1)."),

        ("Include 'long' bucket prompts",
         "The 100-prompt run excluded xlong and had few long prompts.\n"
         "     adakv is selected mostly at long context — adding 20-30 long\n"
         "     prompts would double its label count without a full re-run."),

        ("Stratify the train/val split by label",
         "Current split is by prompt_id only. Force each val fold to contain\n"
         "     at least 1 prompt per class. With 100 prompts and 4 classes\n"
         "     this is feasible; prevents adakv from landing 100% in train."),

        ("Per-layer labels (large budget)",
         "Per-prompt labels assume the same strategy is optimal for all 28\n"
         "     layers of a prompt. Per-layer labels (run each backend per layer)\n"
         "     would capture intra-prompt variation and multiply the effective\n"
         "     dataset size by 28× for free. Cost: ~100× more GPU runs."),
    ]
    for i, (name, detail) in enumerate(solutions, 1):
        print(f"\n  {i}. {name}")
        print(f"     {detail}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="calib100")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    show_representation(args.data_dir)
    show_per_lambda(args.data_dir, args.model_dir)
    show_solutions()
