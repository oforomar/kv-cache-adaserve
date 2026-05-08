"""Per-backend runner: DynamicKV.

Runs in `backends/runners/dynamickv_env/`. DynamicKV monkey-patches the
`flash_attention_2` forward of HF Llama/Mistral/Qwen2/InternLM. We use
the `dynamickv_v11` variant (the upstream's recommended head per their
LongBench scripts).

Two upstream quirks worth knowing:

  1. `kv_compression.token_drop.monkeypatch` does an unconditional
     `import transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2`
     at module import time, even when not using InternLM. We stub that
     module in `sys.modules` BEFORE importing.

  2. The DynamicKV repo isn't pip-installable (no setup.py / pyproject).
     We add `backends/dynamickv` to sys.path so its `kv_compression`
     package is importable.

Per-layer config (window_size, max_capacity_prompt, kernel_size, pooling,
radio_max) is set on each `model.model.layers[i].self_attn.config`
after the patch is applied — that's what the patched forward reads.

Compression ratio: `max_capacity_prompt / seq_len` for prompts longer
than the budget; saturates at 1.0 for shorter prompts.

Usage:
    uv run --project backends/runners/dynamickv_env python backends/runners/run_dynamickv.py \\
        --model meta-llama/Llama-3.1-8B \\
        --prompts prompts.jsonl --out dynamickv.jsonl \\
        --max-capacity-prompt 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Iterable


def _read_prompts(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _stub_internlm_module() -> None:
    """Pre-empt DynamicKV's hardcoded `transformers_modules.<...>` import.

    Upstream `monkeypatch.py` imports an InternLM-specific transformers
    module at the top level, even when only the Llama path is exercised.
    Provide an empty stub so the import succeeds; the InternLM branch of
    `replace_attention()` is never taken in this runner.
    """
    parent = "transformers_modules"
    sub = f"{parent}.internlm2_5_7b_chat_1m"
    leaf = f"{sub}.modeling_internlm2"
    for name in (parent, sub, leaf):
        sys.modules.setdefault(name, types.ModuleType(name))


def _ensure_dynamickv_on_path() -> None:
    """Locate `backends/dynamickv` relative to this script and add it to sys.path."""
    runner_path = Path(__file__).resolve()
    repo_root = runner_path.parents[2]
    dynamickv_path = repo_root / "backends" / "dynamickv"
    if not dynamickv_path.is_dir():
        raise SystemExit(
            f"DynamicKV submodule not found at {dynamickv_path}. "
            f"Run `git submodule update --init --recursive` from the repo root."
        )
    if str(dynamickv_path) not in sys.path:
        sys.path.insert(0, str(dynamickv_path))


def _load_dynamickv_llama(model_name: str, max_capacity_prompt: int,
                          window_size: int, kernel_size: int, pooling: str,
                          radio_max: int, device: str):
    """Apply DynamicKV's monkey-patch, then load + configure the model."""
    _ensure_dynamickv_on_path()
    _stub_internlm_module()

    import torch
    from kv_compression.token_drop.monkeypatch import replace_attention
    from transformers import AutoModelForCausalLM, AutoTokenizer

    replace_attention(model_type="llama", method="dynamickv_v11")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    # The patched flash_attention_2 forward reads these from each layer's
    # self_attn.config — one assignment per layer.
    for layer in model.model.layers:
        cfg = layer.self_attn.config
        cfg.window_size = window_size
        cfg.max_capacity_prompt = max_capacity_prompt
        cfg.kernel_size = kernel_size
        cfg.pooling = pooling
        cfg.radio_max = radio_max
    return model, tok


def _perplexity(model, tok, prompt_text: str, target_text: str,
                max_length: int, device: str) -> tuple[float, int]:
    """Teacher-forced perplexity on target_text. DynamicKV's eviction
    fires inside the patched flash_attention_2 forward when
    `seq_len > max_capacity_prompt`."""
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
    return math.exp(out.loss.item()), prompt_len


def run(model_name: str, prompts_path: str, out_path: str,
        max_capacity_prompt: int, window_size: int, kernel_size: int,
        pooling: str, radio_max: int, max_length: int, device: str) -> int:
    model, tok = _load_dynamickv_llama(
        model_name, max_capacity_prompt, window_size, kernel_size,
        pooling, radio_max, device,
    )

    n_written = 0
    with open(out_path, "w") as f:
        for p in _read_prompts(prompts_path):
            ppl, prompt_len = _perplexity(
                model, tok, p["text"], p["target_text"], max_length, device,
            )
            cratio = min(1.0, max_capacity_prompt / max(prompt_len, 1))
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
    ap.add_argument("--out", default="dynamickv.jsonl")
    ap.add_argument("--max-capacity-prompt", type=int, default=512,
                    help="Per-layer KV-cache budget in tokens.")
    ap.add_argument("--window-size", type=int, default=8,
                    help="Recent-window size kept verbatim per layer.")
    ap.add_argument("--kernel-size", type=int, default=7,
                    help="Importance-pooling kernel size.")
    ap.add_argument("--pooling", choices=("avgpool", "maxpool"),
                    default="avgpool")
    ap.add_argument("--radio-max", type=int, default=10,
                    help="DynamicKV's max-ratio between layer budgets.")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    n = run(
        args.model, args.prompts, args.out,
        args.max_capacity_prompt, args.window_size, args.kernel_size,
        args.pooling, args.radio_max,
        args.max_length, args.device,
    )
    print(f"wrote {n} rows → {args.out}", file=sys.stderr)
