"""Per-backend runner: KVQuant.

Runs in `backends/runners/kvquant_env/`. Uses KVQuant's `make_quant_sim`
to patch the model's K/V projections with simulated NUQ quantization,
then runs standard teacher-forced perplexity per prompt.

KVQuant requires **offline pre-calibration**: the user must first compute
Fisher information (KVQuant's `gradients/` pipeline) and run NUQ codebook
calibration (`llama_simquant.py --quantize`) to produce a `quantizers.pickle`
per bitwidth. This runner consumes those pickles via `--quantizer-path`
and does not perform calibration itself. See
`backends/runners/kvquant_env/README.md` for the prep steps.

Bitwidth mapping (per project's strategy enum):
  kvquant_8b → KVQuant 4-bit  (KVQuant has no native 8-bit; 4-bit is the
                                closest "light-compression" option)
  kvquant_3b → KVQuant 3-bit

Compression ratio reported as `abits / 16` (KVQuant's design ratio for K/V
storage, ignoring the small dense-and-sparse codebook overhead).

Usage:
    uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \\
        --model meta-llama/Llama-3.1-8B \\
        --prompts prompts.jsonl --out kvquant_8b.jsonl --bitwidth 8 \\
        --quantizer-path quantizers_4bit.pickle
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from typing import Iterable

# KVQuant's strategy enum 8b ↔ KVQuant 4-bit; everything else is literal.
ABITS_MAP = {8: 4, 3: 3}


def _read_prompts(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_kvquant_model(model_name: str, abits: int, quantizer_path: str,
                        include_sparse: bool, sparsity_threshold: float,
                        nuq: bool, nf_nuq: bool, first_few_fp16: int,
                        device: str):
    """Load the model in fp16 and patch K/V projections with simquant wrappers.

    Returns (model, tokenizer). After this, standard `model(input_ids, ...)`
    calls run with quantized K/V transparently.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # KVQuant's own simquant package, installed via the runner env.
    from kvquant.simquant_module_quantizer import make_quant_sim

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Try flash-attn first for speed; fall back to eager if it isn't built.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    except (ImportError, ValueError) as e:
        print(f"  flash-attn unavailable ({e}); using eager attention",
              file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    model = model.to(device).eval()

    with open(quantizer_path, "rb") as f:
        quantizers = pickle.load(f)

    # KVQuant defaults: per-channel for k_proj, per-token for v_proj.
    perchannel_match = ["k_proj"]
    pertoken_match = ["v_proj"]
    perchannel_q: dict = {}
    pertoken_q: dict = {}
    for k in quantizers:
        if any(p in k for p in perchannel_match):
            perchannel_q[k] = quantizers[k]
        if any(p in k for p in pertoken_match):
            pertoken_q[k] = quantizers[k]

    # Patch in-place. Two passes: per-channel first, then per-token (dynamic).
    make_quant_sim(
        model, perchannel_q, abits,
        perchannel=True,
        include_sparse=include_sparse,
        sparsity_threshold=sparsity_threshold,
        dynamicquantization=False,
        nuq=nuq, nf_nuq=nf_nuq,
        first_few_fp16=first_few_fp16,
    )
    make_quant_sim(
        model, pertoken_q, abits,
        perchannel=False,
        include_sparse=include_sparse,
        sparsity_threshold=sparsity_threshold,
        dynamicquantization=True,
        nuq=nuq, nf_nuq=nf_nuq,
        first_few_fp16=first_few_fp16,
    )
    return model, tok


def _perplexity(model, tok, prompt_text: str, target_text: str,
                max_length: int, device: str) -> float:
    """Teacher-forced perplexity on target_text with prompt portion masked."""
    import torch

    prompt_ids = tok(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).input_ids.to(device)
    target_ids = tok(
        target_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    if input_ids.shape[1] > max_length:
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


def run(model_name: str, prompts_path: str, out_path: str, bitwidth: int,
        quantizer_path: str, include_sparse: bool, sparsity_threshold: float,
        nuq: bool, nf_nuq: bool, first_few_fp16: int,
        max_length: int, device: str) -> int:
    abits = ABITS_MAP.get(bitwidth, bitwidth)
    if abits not in (2, 3, 4, 5):
        raise SystemExit(
            f"--bitwidth {bitwidth} maps to abits={abits}; KVQuant supports "
            f"{{2, 3, 4, 5}}. Pick a different bitwidth or update ABITS_MAP."
        )

    print(f"loading model with KVQuant {abits}-bit (mapped from "
          f"--bitwidth {bitwidth})", file=sys.stderr)
    model, tok = _load_kvquant_model(
        model_name, abits, quantizer_path,
        include_sparse, sparsity_threshold,
        nuq, nf_nuq, first_few_fp16, device,
    )

    cratio = abits / 16.0
    n_written = 0
    with open(out_path, "w") as f:
        for p in _read_prompts(prompts_path):
            ppl = _perplexity(model, tok, p["text"], p["target_text"],
                              max_length, device)
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
    ap.add_argument("--out", required=True,
                    help="Per-strategy JSONL, e.g. kvquant_8b.jsonl or kvquant_3b.jsonl")
    ap.add_argument("--bitwidth", type=int, choices=(3, 8), required=True,
                    help="Strategy bitwidth label (8 → KVQuant 4-bit, 3 → 3-bit)")
    ap.add_argument("--quantizer-path", required=True,
                    help="Pre-computed quantizers.pickle (bitwidth-specific)")
    # NUQ and sparse settings — match the upstream defaults from llama_simquant.
    ap.add_argument("--no-nuq", action="store_true",
                    help="Disable non-uniform quantization (default: NUQ on).")
    ap.add_argument("--nf", action="store_true",
                    help="Use NormalFloat NUQ instead of k-means NUQ.")
    ap.add_argument("--include-sparse", action="store_true", default=True,
                    help="Use dense-and-sparse quantization (recommended).")
    ap.add_argument("--sparsity-threshold", type=float, default=0.99,
                    help="Outlier percentile retained in dense form.")
    ap.add_argument("--first-few-fp16", type=int, default=-1,
                    help="Keep first N tokens in fp16 (attention sink).")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    n = run(
        args.model, args.prompts, args.out, args.bitwidth,
        args.quantizer_path, args.include_sparse, args.sparsity_threshold,
        nuq=not args.no_nuq, nf_nuq=args.nf,
        first_few_fp16=args.first_few_fp16,
        max_length=args.max_length, device=args.device,
    )
    print(f"wrote {n} rows → {args.out}", file=sys.stderr)
