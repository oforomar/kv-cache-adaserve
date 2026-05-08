"""Join signals + measurements → training JSONL consumed by train_classifier.py.

Each output row is in the format the trainer expects:
  {"signals": {entropy, entropy_normalized, head_variance, seq_len, layer_idx},
   "num_layers": int,
   "label": "kvquant_8b" | ...}

Usage:
  python make_labels.py \
      --signals signals.jsonl \
      --measurements measurements.jsonl \
      --out training.jsonl \
      --target-size 10000

If signals contains more (prompt, layer) pairs than --target-size, we
stratified-sample by label to preserve class balance.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(rows: list[dict], target_size: int,
                      seed: int = 0) -> list[dict]:
    """Sample to target_size while preserving label proportions."""
    if len(rows) <= target_size:
        return rows
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    quota = {lbl: max(1, round(target_size * len(rs) / len(rows)))
             for lbl, rs in by_label.items()}
    out: list[dict] = []
    for lbl, rs in by_label.items():
        rng.shuffle(rs)
        out.extend(rs[:quota[lbl]])
    rng.shuffle(out)
    return out[:target_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", default="training.jsonl")
    ap.add_argument("--target-size", type=int, default=10_000)
    args = ap.parse_args()

    sig_rows = load_jsonl(args.signals)
    meas_rows = load_jsonl(args.measurements)
    label_by_pid = {r["prompt_id"]: r for r in meas_rows}

    joined: list[dict] = []
    skipped = 0
    for s in sig_rows:
        pid = s["prompt_id"]
        if pid not in label_by_pid:
            skipped += 1
            continue
        m = label_by_pid[pid]
        joined.append({
            "signals": {
                "entropy": s["entropy"],
                "entropy_normalized": s["entropy_normalized"],
                "head_variance": s["head_variance"],
                "seq_len": s["seq_len"],
                "layer_idx": s["layer_idx"],
            },
            "num_layers": s["num_layers"],
            "label": m["label"],
        })

    sampled = stratified_sample(joined, args.target_size)
    with open(args.out, "w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")

    # Print label distribution for sanity.
    dist: dict[str, int] = defaultdict(int)
    for r in sampled:
        dist[r["label"]] += 1
    print(f"wrote {len(sampled)} rows → {args.out}  (skipped {skipped} unjoinable)")
    print("label distribution:")
    for lbl in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {lbl:<14}  {dist[lbl]:>5}  ({dist[lbl] / len(sampled):.1%})")


if __name__ == "__main__":
    main()
