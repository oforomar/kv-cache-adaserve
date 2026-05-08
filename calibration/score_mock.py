"""Stage 3 mock: assign per-prompt labels via Phase A on aggregated signals.

This is *not* a substitute for the real per-backend scoring runs. It exists
to validate that prompts → signals → labels → training all wire up before
standing up the GPU pipeline.

Output: measurements.jsonl in the same shape `make_labels.py` consumes:
    {"prompt_id": ..., "label": ..., "num_layers": ...}

For real runs, use `score_baseline.py` plus the per-backend runners under
`backends/runners/`, joined via `join_labels.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector import HeuristicConfig, select_phase_a   # noqa: E402
from signals import LayerSignals                        # noqa: E402
from calibration.prompts import read_prompts            # noqa: E402


def measure_mock(prompts_path: str, signals_path: str, out_path: str,
                 cfg: HeuristicConfig | None = None) -> int:
    cfg = cfg or HeuristicConfig()
    prompts = {p.id: p for p in read_prompts(prompts_path)}

    agg: dict[str, dict] = defaultdict(lambda: {
        "entropy": 0.0, "entropy_normalized": 0.0,
        "head_variance": 0.0, "seq_len": 0, "n": 0, "num_layers": 0,
    })
    with open(signals_path) as f:
        for line in f:
            r = json.loads(line)
            a = agg[r["prompt_id"]]
            a["entropy"] += r["entropy"]
            a["entropy_normalized"] += r["entropy_normalized"]
            a["head_variance"] += r["head_variance"]
            a["seq_len"] = r["seq_len"]
            a["num_layers"] = r["num_layers"]
            a["n"] += 1

    n = 0
    with open(out_path, "w") as f:
        for pid, a in agg.items():
            if pid not in prompts or a["n"] == 0:
                continue
            mean = LayerSignals(
                entropy=a["entropy"] / a["n"],
                entropy_normalized=a["entropy_normalized"] / a["n"],
                head_variance=a["head_variance"] / a["n"],
                seq_len=a["seq_len"],
                layer_idx=0,
            )
            label = select_phase_a(mean, cfg).value
            f.write(json.dumps({
                "prompt_id": pid,
                "label": label,
                "num_layers": a["num_layers"],
            }) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--signals", required=True,
                    help="signals.jsonl from collect_signals.py")
    ap.add_argument("--out", default="measurements.jsonl")
    args = ap.parse_args()
    n = measure_mock(args.prompts, args.signals, args.out)
    print(f"wrote {n} mock measurement rows → {args.out}")
