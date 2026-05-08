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
        --model meta-llama/Llama-3.2-3B \
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
        gqa_support=True,    # Llama-3.2-3B is GQA (8 KV / 24 Q)
        gqa_func="mean",
    )
    return model, tok


def _perplexity_evicted(model, tok, prompt_text: str, target_text: str,
                        max_length: int, device: str) -> tuple[float, int]:
    """Teacher-forced perplexity through `model.generate` with the target
    sequence as a constraint.

    Why this shape: AdaKV's flattened post-eviction K/V layout makes
    standard teacher-forced decoding via `past_key_values=...` break
    (rank-2 vs rank-4). Single-pass with use_cache=True breaks because
    eviction mid-forward shrinks logits' seq dim out of step with labels.
    Token-by-token decode via past_kv runs but re-triggers AdaKV's
    eviction each step, inflating ppl wildly.

    `model.generate` is the eval shape AdaKV's upstream LongBench pipeline
    uses — eviction fires once during prefill, and decode steps treat the
    flattened cache correctly. We constrain generation to the target
    sequence via `prefix_allowed_tokens_fn` (teacher-forcing through
    generate) and capture the model's raw logits at each step via a
    forward hook (raw logits, not the post-processor masked ones, so the
    softmax denominator is the full vocabulary).
    """
    import torch
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    prompt_ids = tok(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).input_ids.to(device)
    target_ids = tok(
        target_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    target_list = target_ids[0].tolist()
    target_len = len(target_list)
    prompt_len = int(prompt_ids.shape[1])

    if target_len == 0:
        return float("inf"), prompt_len

    # Capture raw logits at each forward call. The last position's logit
    # at call k predicts step-k's token (the prefill's last logit predicts
    # target[0]; subsequent decode-step logits predict target[1], target[2], ...).
    captured: list[torch.Tensor] = []

    def _hook(module, args, kwargs, output):
        # CausalLMOutputWithPast has a `.logits` attribute.
        logits = getattr(output, "logits", None)
        if logits is None:
            return
        captured.append(logits[:, -1:, :].detach())

    handle = model.register_forward_hook(_hook, with_kwargs=True)

    def _prefix_fn(batch_idx: int, input_ids):
        step = int(input_ids.shape[-1]) - prompt_len
        if 0 <= step < target_len:
            return [target_list[step]]
        # After we've forced the full target, anything is fine; generate
        # will stop at max_new_tokens anyway.
        return [tok.eos_token_id]

    try:
        with torch.inference_mode():
            model.generate(
                input_ids=prompt_ids,
                max_new_tokens=target_len,
                do_sample=False,
                num_beams=1,
                prefix_allowed_tokens_fn=_prefix_fn,
                pad_token_id=tok.eos_token_id,
            )
    finally:
        handle.remove()

    # captured[0] is the prefill's logits (last position predicts target[0]);
    # captured[k] for k=1..target_len is the decode-step k's logits
    # (predicts target[k]). We want target_len log-probs total.
    if len(captured) < target_len:
        # Fall back: if we got fewer logits than expected (some HF version
        # quirk), score what we have.
        target_len = len(captured)
        target_list = target_list[:target_len]

    log_probs = torch.cat([
        torch.log_softmax(captured[i].float(), dim=-1)
        for i in range(target_len)
    ], dim=1)  # [B, target_len, V]
    target_t = torch.tensor(target_list, device=log_probs.device).unsqueeze(0).unsqueeze(-1)
    nll = -log_probs.gather(-1, target_t).squeeze(-1).mean()
    return math.exp(nll.item()), prompt_len


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
                    help="HF model id, e.g. meta-llama/Llama-3.2-3B")
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
