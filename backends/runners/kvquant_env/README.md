# KVQuant runner env

Hosts the eval-time portion of KVQuant. KVQuant requires **offline pre-calibration** that must be done once per (model, bitwidth) combination before this runner can produce useful labels.

## One-time prep: build the quantizer pickles

KVQuant's calibration pipeline lives in two subfolders of the upstream submodule, each with its own deps. Both run in **separate envs** from this runner; they're heavyweight scripts that produce artifacts, not part of our calibration loop.

### Step 1 — Fisher information

```bash
# In a fresh conda or uv env (NOT this runner's env), per upstream's gradients/README.md
cd backends/kvquant/gradients
pip install -e .
# Compute fisher info for the target model — produces a directory of safetensors
python compute_fisher.py --model meta-llama/Llama-3.2-3B --output /tmp/fisher_llama32_3b ...
```

Refer to `backends/kvquant/gradients/README.md` for the exact CLI; it varies by upstream version.

### Step 2 — Quantizer codebook (one per bitwidth)

```bash
# Again in a fresh env per upstream/quant/README.md
cd backends/kvquant/quant
pip install -e .

# 4-bit (used for our `kvquant_8b` strategy)
python llama_simquant.py meta-llama/Llama-3.2-3B \
    --abits 4 --nsamples 16 --seqlen 2048 \
    --nuq --fisher /tmp/fisher_llama32_3b \
    --quantize --include_sparse --sparsity-threshold 0.99 \
    --quantizer-path /tmp/quantizers_4bit.pickle

# 3-bit (used for our `kvquant_3b` strategy)
python llama_simquant.py meta-llama/Llama-3.2-3B \
    --abits 3 --nsamples 16 --seqlen 2048 \
    --nuq --fisher /tmp/fisher_llama32_3b \
    --quantize --include_sparse --sparsity-threshold 0.99 \
    --quantizer-path /tmp/quantizers_3bit.pickle
```

These steps are slow (k-means on per-channel codebooks) and CPU-heavy. They produce two pickle files this runner consumes via `--quantizer-path`.

## This env's setup

Once the pickles exist:

```bash
# from repo root
uv sync --project backends/runners/kvquant_env

# Optional but recommended — flash-attn (custom CUDA build):
uv pip install --project backends/runners/kvquant_env flash-attn --no-build-isolation
```

The runner falls back to eager attention if flash-attn isn't built.

## Running

Two invocations, one per bitwidth:

```bash
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model meta-llama/Llama-3.2-3B --prompts prompts.jsonl \
    --out kvquant_8b.jsonl --bitwidth 8 \
    --quantizer-path /tmp/quantizers_4bit.pickle

uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model meta-llama/Llama-3.2-3B --prompts prompts.jsonl \
    --out kvquant_3b.jsonl --bitwidth 3 \
    --quantizer-path /tmp/quantizers_3bit.pickle
```

`--bitwidth 8` maps to KVQuant's 4-bit (closest "light compression" option since KVQuant has no native 8-bit). `--bitwidth 3` is literal.

`cratio` reported as `abits / 16` — matches KVQuant's design ratio for K/V storage; ignores the small dense-and-sparse codebook overhead.
