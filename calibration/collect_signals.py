"""Capture LayerSignals for every (prompt, layer) by running prefill once.

Cheap pass — one forward per prompt. Materializes attention weights, so for
long context we use a hook that reduces to scalars immediately instead of
holding the full [B, H, q, k] tensor.

Output: signals.jsonl with one row per (prompt, layer):
  {"prompt_id": ..., "layer_idx": ...,
   "entropy": ..., "entropy_normalized": ...,
   "head_variance": ..., "seq_len": ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Allow running as a script from the calibration/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals import LayerSignals, attention_entropy, head_variance  # noqa: E402
from calibration.prompts import read_prompts                         # noqa: E402


def install_hooks(model) -> tuple[list, dict[int, dict]]:
    """Hook every attention module to capture per-layer signals on the fly.

    HF transformer attention modules expose attention weights via the second
    output element when `output_attentions=True`. We avoid that path because
    it stores all layers in CPU memory; instead we hook each layer and
    reduce to scalars immediately.

    Returns (handles, results). Results is filled in during forward.
    """
    handles = []
    results: dict[int, dict] = {}

    # Heuristic: HF models expose attention layers as model.model.layers[i].self_attn
    # (Llama-family). Adjust for other architectures.
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        # GQA: HF Llama/Mistral/Qwen attention modules expose this directly.
        # Defaults to 1 (MHA) for older or non-GQA architectures.
        n_kv_groups = getattr(attn, "num_key_value_groups", 1)

        def make_hook(layer_idx: int, kv_groups: int):
            def hook(module, args, kwargs, output):
                # HF returns (attn_output, attn_weights, past_kv) when
                # output_attentions=True. attn_weights: [B, Hq, q, k].
                weights = output[1] if isinstance(output, tuple) else None
                if weights is None:
                    return
                with torch.no_grad():
                    H = attention_entropy(weights).mean().item()
                    V = head_variance(weights, num_kv_groups=kv_groups).item()
                    seq_len = int(weights.shape[-1])
                results[layer_idx] = {
                    "entropy": H,
                    "head_variance": V,
                    "seq_len": seq_len,
                }
            return hook

        handles.append(
            attn.register_forward_hook(
                make_hook(idx, n_kv_groups), with_kwargs=True
            )
        )
    return handles, results


def collect(
    model_name: str,
    prompts_path: str,
    out_path: str,
    max_length: int = 8192,
    device: str = "cuda",
) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    num_layers = model.config.num_hidden_layers

    prompts = read_prompts(prompts_path)
    handles, results = install_hooks(model)

    n_written = 0
    try:
        with open(out_path, "w") as f, torch.inference_mode():
            for p in prompts:
                ids = tok(p.text, return_tensors="pt",
                          truncation=True, max_length=max_length).input_ids.to(device)
                results.clear()
                model(input_ids=ids, output_attentions=True, use_cache=False)
                for layer_idx in sorted(results):
                    r = results[layer_idx]
                    import math
                    s = LayerSignals(
                        entropy=r["entropy"],
                        entropy_normalized=r["entropy"] / math.log(max(2, r["seq_len"])),
                        head_variance=r["head_variance"],
                        seq_len=r["seq_len"],
                        layer_idx=layer_idx,
                    )
                    row = {"prompt_id": p.id, "num_layers": num_layers, **asdict(s)}
                    f.write(json.dumps(row) + "\n")
                    n_written += 1
    finally:
        for h in handles:
            h.remove()
    return n_written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="signals.jsonl")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n = collect(args.model, args.prompts, args.out, args.max_length, args.device)
    print(f"wrote {n} signal rows → {args.out}")
