"""Per-backend runner skeleton: DynamicKV.

This script runs in its OWN environment. DynamicKV monkey-patches HF
transformers internals; that's why isolation matters. See
`backends/runners/README.md` for env setup.

What this needs to do (TODO):
  1. Load model with DynamicKV's monkey-patched attention path applied
     (upstream `backends/dynamickv` typically does this via a patcher
     module imported before model construction).
  2. For each prompt: teacher-forced perplexity on `target_text`.
  3. Compression ratio: derive from the layer-adaptive token-retention
     budget DynamicKV ends up using (it's ~1.7%–6.9% retention in the
     paper's settings; the upstream config decides the target).
  4. Emit JSONL: {"prompt_id": ..., "ppl": ..., "cratio": ...}

This file is intentionally a stub.
"""
from __future__ import annotations

import argparse
import json


def _read_prompts(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(model_name: str, prompts_path: str, out_path: str) -> int:
    raise NotImplementedError(
        "DynamicKV runner is a skeleton. Wire upstream DynamicKV's "
        "patcher + perplexity loop. Inputs: HF model id, prompts JSONL. "
        "Output: per-prompt {prompt_id, ppl, cratio} JSONL."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="dynamickv.jsonl")
    args = ap.parse_args()
    n = run(args.model, args.prompts, args.out)
    print(f"wrote {n} rows → {args.out}")
