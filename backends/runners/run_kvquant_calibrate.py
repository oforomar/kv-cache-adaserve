"""KVQuant offline calibration: build a `quantizers.pickle` for one bitwidth.

Wraps upstream `llama_simquant.py --quantize` with two patches:

  1. Modern transformers (>=4.45) requires `attn_implementation` instead of
     the deprecated `use_flash_attention_2=True` kwarg. The upstream
     get_model still uses the old kwarg, which raises on transformers 4.45.
  2. Llama-3.2's RoPE scaling format (`rope_type: 'llama3'`) needs
     transformers 4.45+ — older releases reject the config dict.

This script reimplements `get_model` and reuses upstream's `llama_calibration`
function and dataloader directly. Output is identical in shape to upstream's
`--quantize` mode.

Calibration runs k-means clustering for NUQ codebooks per (layer, projection).
For Llama-3.2-3B (28 layers × 2 projs), expect tens of minutes of CPU work.

Skipping `--fisher`: KVQuant accepts `fisher=None`, producing non-fisher-
weighted NUQ codebooks. Slightly lower quality than with fisher info, but
avoids the heavyweight gradients pipeline (a separate transformers fork).

Usage:
    uv run --project backends/runners/kvquant_env python \\
        backends/runners/run_kvquant_calibrate.py \\
        --model meta-llama/Llama-3.2-3B \\
        --abits 4 \\
        --quantizer-path smoke/quantizers/quantizers_4bit.pickle
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path


def _ensure_kvquant_quant_on_path() -> None:
    runner_path = Path(__file__).resolve()
    repo_root = runner_path.parents[2]
    quant_path = repo_root / "backends" / "kvquant" / "quant"
    if not quant_path.is_dir():
        raise SystemExit(f"missing {quant_path}")
    if str(quant_path) not in sys.path:
        sys.path.insert(0, str(quant_path))


def _patch_llama_layer_for_per_layer_rotary() -> None:
    """Modern transformers (4.45+) moved Llama rotary to a model-level
    `rotary_emb` and made layers expect a precomputed `position_embeddings`
    tuple from the model's forward. KVQuant's calibration calls layers
    directly (CPU-offload pattern), so `position_embeddings` is never
    provided — and the inv_freq buffer the layer falls back to is on CPU
    while the layer was just moved to GPU, hence the device mismatch.

    Patch LlamaDecoderLayer.forward to lazily compute its own
    `position_embeddings` if not supplied, using a per-layer rotary that
    follows the layer's device.
    """
    import functools
    import torch
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer, LlamaRotaryEmbedding,
    )

    if getattr(LlamaDecoderLayer.forward, "__kvquant_patched__", False):
        return

    original_forward = LlamaDecoderLayer.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states, attention_mask=None,
                        position_ids=None, past_key_value=None,
                        output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None,
                        **kwargs):
        if position_embeddings is None and position_ids is not None:
            rotary = getattr(self, "_kvquant_rotary", None)
            if rotary is None or rotary.inv_freq.device != hidden_states.device:
                rotary = LlamaRotaryEmbedding(
                    config=self.self_attn.config
                ).to(hidden_states.device)
                self._kvquant_rotary = rotary
            with torch.no_grad():
                position_embeddings = rotary(hidden_states, position_ids)
        return original_forward(
            self, hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs,
        )

    patched_forward.__kvquant_patched__ = True
    LlamaDecoderLayer.forward = patched_forward


def _load_model_modern_kwargs(model_name: str, seqlen: int, maxseqlen: int):
    """Replacement for `llama_simquant.get_model` using `attn_implementation`
    instead of the removed `use_flash_attention_2=True` kwarg."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    def _skip(*a, **kw):
        pass
    torch.nn.init.kaiming_uniform_ = _skip
    torch.nn.init.uniform_ = _skip
    torch.nn.init.normal_ = _skip

    config = AutoConfig.from_pretrained(model_name)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    model.seqlen = seqlen
    return model


def calibrate(model_name: str, abits: int, quantizer_path: str,
              nsamples: int, seqlen: int, maxseqlen: int,
              include_sparse: bool, sparsity_threshold: float,
              nuq: bool, dataset: str = "wikitext2", seed: int = 0) -> None:
    _ensure_kvquant_quant_on_path()
    _patch_llama_layer_for_per_layer_rotary()

    import torch
    import types
    from kvquant.datautils import get_loaders  # type: ignore
    # llama_simquant lives as a script next to the kvquant package; we put
    # backends/kvquant/quant on sys.path above so this import resolves.
    import llama_simquant  # type: ignore

    # Upstream `llama_calibration` references a script-local `args.nsamples`
    # via a closed-over global. Inject a stub so the function call succeeds
    # without rewriting the upstream source.
    llama_simquant.args = types.SimpleNamespace(nsamples=nsamples)
    llama_calibration = llama_simquant.llama_calibration

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"loading model {model_name}...", file=sys.stderr)
    model = _load_model_modern_kwargs(model_name, seqlen, maxseqlen)
    model.eval()
    model = model.half()

    # Modern Llama-3 has a model-level `rotary_emb`. Upstream KVQuant's
    # calibration runs an initial full forward (the Catcher pass) which
    # needs rotary on the same device as the input batch. The CPU-offload
    # pattern leaves rotary_emb on CPU; move it to GPU here so the Catcher
    # pass works. Layer-by-layer calibration uses our patched
    # LlamaDecoderLayer rotary (above).
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    print(f"loading {dataset} calibration data ({nsamples} samples × "
          f"{seqlen} tokens)...", file=sys.stderr)
    dataloader, _ = get_loaders(
        dataset, nsamples=nsamples, seed=seed, model=model_name, seqlen=seqlen,
    )

    print(f"running KVQuant calibration at {abits}-bit "
          f"(this is the slow part — k-means per layer)...", file=sys.stderr)
    quantizers = llama_calibration(
        model, dataloader, dev,
        ["k_proj"],   # perchannel_match
        ["v_proj"],   # pertensor_match
        abits,
        include_sparse=include_sparse,
        sparsity_threshold=sparsity_threshold,
        nuq=nuq,
        fisher=None,
        norm=False,
        cap_outliers=-1,
        first_few_fp16=-1,
    )

    Path(quantizer_path).parent.mkdir(parents=True, exist_ok=True)
    with open(quantizer_path, "wb") as f:
        pickle.dump(quantizers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote quantizer pickle ({len(quantizers)} tensors) → "
          f"{quantizer_path}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. meta-llama/Llama-3.2-3B")
    ap.add_argument("--abits", type=int, choices=(2, 3, 4, 5), required=True)
    ap.add_argument("--quantizer-path", required=True)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--maxseqlen", type=int, default=2048)
    ap.add_argument("--include-sparse", action="store_true", default=True)
    ap.add_argument("--sparsity-threshold", type=float, default=0.99)
    ap.add_argument("--no-nuq", action="store_true")
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    calibrate(
        args.model, args.abits, args.quantizer_path,
        nsamples=args.nsamples, seqlen=args.seqlen, maxseqlen=args.maxseqlen,
        include_sparse=args.include_sparse,
        sparsity_threshold=args.sparsity_threshold,
        nuq=not args.no_nuq, dataset=args.dataset, seed=args.seed,
    )
