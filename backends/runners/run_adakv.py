"""Per-backend runner skeleton: Ada-KV.

This script runs in its OWN environment. Ada-KV is closer to a library
than DynamicKV/KVQuant but still benefits from isolation (model-version
pins, optional GQA branch). See `backends/runners/README.md`.

If your target model uses GQA (Llama-3, Mistral, Qwen), consider the
upstream `gqa_support` branch:
    cd backends/adakv && git checkout gqa_support && cd -
Then update the submodule pin in this repo before running calibration.

What this needs to do (TODO):
  1. Load model + apply Ada-KV's head-adaptive eviction
     (upstream exposes a wrapper / hook over HF attention).
  2. For each prompt: teacher-forced perplexity on `target_text`.
  3. Compression ratio: derive from the eviction budget config used.
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
        "Ada-KV runner is a skeleton. Wire upstream Ada-KV's eviction "
        "wrapper + perplexity loop. Inputs: HF model id, prompts JSONL. "
        "Output: per-prompt {prompt_id, ppl, cratio} JSONL."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="adakv.jsonl")
    args = ap.parse_args()
    n = run(args.model, args.prompts, args.out)
    print(f"wrote {n} rows → {args.out}")
