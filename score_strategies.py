"""Score each strategy on each prompt → pick winner → emit per-prompt label.

The expensive pass. For every prompt:
  1. Compute baseline FP16 perplexity on the continuation.
  2. For each strategy in {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV}:
       - Apply strategy (uniform across all layers).
       - Compute perplexity on the same continuation.
       - Compute compression ratio (compressed bytes / fp16 bytes).
       - score(s) = -(ppl_s - ppl_baseline) - lambda * (1 - cratio_s)
  3. label = argmax score
  4. Emit one row per prompt to measurements.jsonl.

To wire in real backends, replace `_apply_strategy` with calls to the actual
KVQuant / DynamicKV / Ada-KV / QAQ adapters. The current implementation
patches the model's KV cache via a callback registered before generation.

Mock mode (--mock) skips the model entirely and assigns labels using the
Phase A heuristic on the captured signals, so you can validate the rest of
the pipeline before standing up the GPU pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector import HeuristicConfig, select_phase_a   # noqa: E402
from signals import LayerSignals                        # noqa: E402
from strategies import Strategy                         # noqa: E402
from calibration.prompts import read_prompts            # noqa: E402

# QAQ excluded from the runtime classifier — see design notes.
RUNTIME_STRATEGIES = [
    Strategy.KVQUANT_8B,
    Strategy.KVQUANT_3B,
    Strategy.DYNAMICKV,
    Strategy.ADAKV,
]


def score(ppl_baseline: float, ppl_strategy: float, cratio: float,
          lambda_compress: float) -> float:
    """Higher is better. cratio = compressed_bytes / fp16_bytes ∈ (0, 1]."""
    delta_ppl = ppl_strategy - ppl_baseline
    return -delta_ppl - lambda_compress * (1.0 - cratio)


# ---------------------------------------------------------------------------
# Real measurement path — fill in the backend adapters.
# ---------------------------------------------------------------------------

def measure_real(model_name: str, prompts_path: str, out_path: str,
                 lambda_compress: float = 1.0, device: str = "cuda") -> int:
    raise NotImplementedError(
        "Wire your KV-cache backends into this function: for each strategy, "
        "patch the model's attention so K/V are encoded/decoded by the "
        "strategy, then call the standard perplexity loop on the prompt's "
        "target_text. Use strategies.REGISTRY to plug adapters in."
    )


# ---------------------------------------------------------------------------
# Mock measurement — labels via Phase A on aggregated signals.
# ---------------------------------------------------------------------------

def measure_mock(prompts_path: str, signals_path: str, out_path: str,
                 cfg: HeuristicConfig | None = None) -> int:
    """Assign a per-prompt label using Phase A on each prompt's mean signals.

    This is *not* a substitute for real measurement. It exists only to
    validate that prompts → signals → labels → training all wire up.
    """
    cfg = cfg or HeuristicConfig()
    prompts = {p.id: p for p in read_prompts(prompts_path)}

    # Aggregate signals per prompt (mean across layers).
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
    ap.add_argument("--mock", action="store_true",
                    help="label via Phase A heuristic instead of real measurement")
    ap.add_argument("--model", help="(real mode) HF model id")
    ap.add_argument("--lambda-compress", type=float, default=1.0)
    args = ap.parse_args()

    if args.mock:
        n = measure_mock(args.prompts, args.signals, args.out)
    else:
        if not args.model:
            raise SystemExit("--model is required without --mock")
        n = measure_real(args.model, args.prompts, args.out,
                         args.lambda_compress)
    print(f"wrote {n} measurement rows → {args.out}")
