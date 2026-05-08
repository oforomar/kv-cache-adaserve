"""Per-backend runner skeleton: KVQuant 8-bit and 3-bit.

This script runs in its OWN environment (separate from the main calibration
venv) — KVQuant has custom CUDA kernels and pinned torch/transformers
versions; isolating it here avoids dep conflicts with the rest of the
pipeline. See `backends/runners/README.md` for env setup.

What this needs to do (TODO):
  1. Load the model in the upstream KVQuant fashion (its own model class
     or quantize-then-load helpers from `backends/kvquant/quant/`).
  2. For each prompt: compute teacher-forced perplexity on `target_text`
     under KVQuant's K/V quantization (one bitwidth per run).
  3. Compute compression ratio: bitwidth/16 with whatever overhead the
     codebook adds for outliers (KVQuant has both dense + sparse parts).
  4. Emit JSONL: {"prompt_id": ..., "ppl": ..., "cratio": ...}

Reference points in the upstream repo:
  - quantization driver / model patcher (look for top-level entrypoint
    or `quant/` package)
  - their LM-eval / perplexity loop (often a `lm_eval` adapter)

This file is intentionally a stub — fill it in after reading the upstream
README and choosing the entrypoint that fits a per-prompt, per-bitwidth run.
"""
from __future__ import annotations

import argparse
import json
import sys


def _read_prompts(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(model_name: str, prompts_path: str, out_path: str, bitwidth: int) -> int:
    raise NotImplementedError(
        "KVQuant runner is a skeleton. Wire upstream KVQuant's quantize "
        "+ perplexity loop here. Inputs: HF model id, prompts JSONL "
        "(id/text/target_text). Output: per-prompt {prompt_id, ppl, cratio} "
        "JSONL. Bitwidth selects 8b vs 3b."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True,
                    help="e.g. kvquant_8b.jsonl or kvquant_3b.jsonl")
    ap.add_argument("--bitwidth", type=int, choices=(3, 8), required=True)
    args = ap.parse_args()
    n = run(args.model, args.prompts, args.out, args.bitwidth)
    print(f"wrote {n} rows → {args.out}")
