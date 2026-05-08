"""Stage 3a: FP16 baseline perplexity per prompt → baseline.jsonl.

Per-prompt teacher-forced perplexity on the held-out continuation
(`target_text`), with the prompt portion masked from the loss. This is
the reference point every strategy is compared against in `score()`.

Output: one row per prompt:
    {"prompt_id": ..., "ppl_baseline": ..., "num_layers": ...}

The per-strategy runners (`backends/runners/run_<name>.py`) write similar
JSONLs of `{prompt_id, ppl, cratio}` for their compressor; `join_labels.py`
joins all of them and picks the per-prompt argmax.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.prompts import read_prompts  # noqa: E402


def perplexity(model, tokenizer, prompt_text: str, target_text: str,
               device: str, max_length: int) -> float:
    """Teacher-forced perplexity of target_text conditioned on prompt_text.

    Prompt portion is masked (labels = -100), so we measure only the
    model's likelihood on the held-out continuation.
    """
    import torch

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(device)
    target_ids = tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    if input_ids.shape[1] > max_length:
        # Trim leading prompt tokens; never trim the target.
        overflow = input_ids.shape[1] - max_length
        input_ids = input_ids[:, overflow:]
        prompt_len = max(0, prompt_ids.shape[1] - overflow)
    else:
        prompt_len = prompt_ids.shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
    return math.exp(out.loss.item())


def run(model_name: str, prompts_path: str, out_path: str,
        device: str = "cuda", max_length: int = 8192) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    num_layers = model.config.num_hidden_layers

    prompts = read_prompts(prompts_path)
    n_written = 0
    with open(out_path, "w") as f:
        for p in prompts:
            ppl = perplexity(model, tok, p.text, p.target_text, device, max_length)
            f.write(json.dumps({
                "prompt_id": p.id,
                "ppl_baseline": ppl,
                "num_layers": num_layers,
            }) + "\n")
            n_written += 1
    return n_written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="baseline.jsonl")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n = run(args.model, args.prompts, args.out, args.device, args.max_length)
    print(f"wrote {n} baseline rows → {args.out}")
