"""Per-backend runner: Ada-KV.

Runs in its own environment (`backends/runners/adakv_env/`) since AdaKV
pins transformers==4.44.2 and needs flash-attn built against the local
torch ABI.

What this does:
  1. Monkey-patch transformers.Llama with AdaKV's adaptive eviction path
     (`replace_llama_adaptive`) BEFORE loading the model.
  2. Load the HF model with flash_attention_2 + bfloat16 (AdaKV's required
     execution config).
  3. `config_compress(model, base_capacity, gqa_support=True, gqa_func="mean")`
     wires hyperparameters onto model.config; the patched attention reads
     them per-layer.
  4. For each prompt: prefill the prompt (eviction fires here when seq_len
     exceeds base_capacity), then teacher-force the target_text against
     the post-eviction cache to get a clean perplexity number on what
     AdaKV's eviction kept.
  5. Compression ratio: base_capacity / max(seq_len, base_capacity). This
     is the design ratio AdaKV targets; actual stored bytes also include
     the small per-head allocation metadata, but that's strategy-agnostic
     overhead and falls out of the score() comparison.

Output: per-prompt JSONL of {prompt_id, ppl, cratio}.

Usage:
    uv run --project backends/runners/adakv_env python backends/runners/run_adakv.py \
        --model meta-llama/Llama-3.1-8B \
        --prompts prompts.jsonl --out adakv.jsonl --base-capacity 1024
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Iterable


def _read_prompts(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_adakv_llama(model_name: str, base_capacity: int,
                      window_size: int, kernel_size: int,
                      floor_alpha: float, device: str):
    """Apply AdaKV's monkey-patch, then load + configure the model."""
    import torch
    from adaptive_snapkv.monkeypatch.monkeypatch import (
        config_compress, replace_llama_adaptive,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    replace_llama_adaptive()

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    model = config_compress(
        model,
        window_size=window_size,
        base_capacity=base_capacity,
        kernel_size=kernel_size,
        pooling="maxpool",
        floor_alpha=floor_alpha,
        pyram_mode=False,
        skip=0,
        gqa_support=True,    # Llama-3.1-8B is GQA (8 KV / 32 Q)
        gqa_func="mean",
    )
    return model, tok


def _perplexity_evicted(model, tok, prompt_text: str, target_text: str,
                        max_length: int, device: str) -> tuple[float, int]:
    """Two-step perplexity: prefill the prompt (eviction fires), then score
    the target tokens against the post-eviction KV cache.

    Returns (ppl, prompt_len_tokens). prompt_len is what AdaKV saw before
    eviction — used to compute cratio per prompt.
    """
    import torch
    import torch.nn.functional as F

    prompt_ids = tok(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).input_ids.to(device)
    target_ids = tok(
        target_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)

    prompt_len = int(prompt_ids.shape[1])

    with torch.inference_mode():
        # Prefill: eviction happens inside the patched attention forward
        # when seq_len exceeds base_capacity.
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        # Last logit of prefill predicts the first target token.
        prev_logits = out.logits[:, -1:, :]

        nll_terms: list[torch.Tensor] = []
        for t in range(target_ids.shape[1]):
            tok_id = target_ids[:, t:t + 1]
            log_probs = F.log_softmax(prev_logits, dim=-1)
            nll_terms.append(-log_probs.gather(-1, tok_id.unsqueeze(-1)).squeeze())

            if t < target_ids.shape[1] - 1:
                step = model(input_ids=tok_id, past_key_values=past, use_cache=True)
                past = step.past_key_values
                prev_logits = step.logits[:, -1:, :]

    nll = torch.stack(nll_terms).mean().item()
    return math.exp(nll), prompt_len


def run(model_name: str, prompts_path: str, out_path: str,
        base_capacity: int, window_size: int, kernel_size: int,
        floor_alpha: float, max_length: int, device: str) -> int:
    model, tok = _load_adakv_llama(
        model_name, base_capacity, window_size, kernel_size,
        floor_alpha, device,
    )

    n_written = 0
    with open(out_path, "w") as f:
        for p in _read_prompts(prompts_path):
            ppl, prompt_len = _perplexity_evicted(
                model, tok, p["text"], p["target_text"], max_length, device,
            )
            # Approximate cratio: AdaKV's design ratio is base_capacity / seq_len
            # for sequences longer than the budget; shorter prompts aren't
            # actually compressed.
            cratio = min(1.0, base_capacity / max(prompt_len, 1))
            f.write(json.dumps({
                "prompt_id": p["id"],
                "ppl": ppl,
                "cratio": cratio,
            }) + "\n")
            n_written += 1
    return n_written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="adakv.jsonl")
    ap.add_argument("--base-capacity", type=int, default=1024,
                    help="Tokens retained per layer (summed across heads).")
    ap.add_argument("--window-size", type=int, default=32,
                    help="AdaKV recent-window size kept verbatim per head.")
    ap.add_argument("--kernel-size", type=int, default=7,
                    help="AdaKV importance-pooling kernel size.")
    ap.add_argument("--floor-alpha", type=float, default=0.2,
                    help="Per-head minimum-budget fraction.")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    n = run(
        args.model, args.prompts, args.out,
        args.base_capacity, args.window_size, args.kernel_size,
        args.floor_alpha, args.max_length, args.device,
    )
    print(f"wrote {n} rows → {args.out}", file=sys.stderr)
