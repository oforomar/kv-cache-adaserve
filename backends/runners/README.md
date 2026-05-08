# Per-backend runners

Each runner is a thin script that:

1. Loads the model with one specific compressor active (KVQuant 8b/3b, DynamicKV, or Ada-KV).
2. Iterates the calibration prompts and computes teacher-forced perplexity on each prompt's `target_text`.
3. Writes a per-prompt JSONL: `{"prompt_id", "ppl", "cratio"}`.

The four resulting JSONLs are joined into `measurements.jsonl` by `calibration/join_labels.py`, which applies `score()` and picks the per-prompt argmax.

## Why each runner has its own environment

The three upstream repos pin specific torch / transformers / CUDA-toolkit versions and (in DynamicKV's case) monkey-patch transformers internals. Trying to satisfy all of them in one venv is fragile. Each runner gets its own.

Suggested layout (each `pyproject.toml` is independent):

```
backends/runners/
├── kvquant_env/pyproject.toml       # deps to run KVQuant
├── dynamickv_env/pyproject.toml     # deps to run DynamicKV
├── adakv_env/pyproject.toml         # deps to run Ada-KV
├── run_kvquant.py
├── run_dynamickv.py
└── run_adakv.py
```

Then:

```bash
uv sync --project backends/runners/kvquant_env
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model <hf-id> --prompts prompts.jsonl --out kvquant_8b.jsonl --bitwidth 8
```

(The env folders are not committed yet; they get created when the runners are filled in and we know each upstream's actual pin requirements.)

## Status

- **`run_adakv.py`** — implemented. Uses upstream `replace_llama_adaptive` + `config_compress`; two-step perplexity (prefill prompt → score target against post-eviction cache). Env: `backends/runners/adakv_env/` with `transformers==4.44.2` and the AdaKV submodule as an editable path source.
- **`run_kvquant.py`** — skeleton; raises `NotImplementedError`.
- **`run_dynamickv.py`** — skeleton; raises `NotImplementedError`.

### Setting up the AdaKV env

```bash
# from repo root
uv sync --project backends/runners/adakv_env

# flash-attn must be built against the installed torch ABI:
uv pip install --project backends/runners/adakv_env flash-attn --no-build-isolation
```

Then run a smoke check on a few prompts:

```bash
uv run --project backends/runners/adakv_env python backends/runners/run_adakv.py \
    --model meta-llama/Llama-3.1-8B \
    --prompts prompts.jsonl --out adakv.jsonl --base-capacity 1024
```

The `gqa_support=True` + `gqa_func="mean"` flags are baked in — required for Llama-3.1-8B (32 Q / 8 KV heads). To target a different model, adjust those defaults.

## Production sequence (once the runners are real)

```bash
# 1. baseline (in our main env)
uv run python calibration/score_baseline.py --model <hf-id> \
    --prompts prompts.jsonl --out baseline.jsonl

# 2. four per-backend runs (each in its own env)
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model <hf-id> --prompts prompts.jsonl --out kvquant_8b.jsonl --bitwidth 8
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model <hf-id> --prompts prompts.jsonl --out kvquant_3b.jsonl --bitwidth 3
uv run --project backends/runners/dynamickv_env python backends/runners/run_dynamickv.py \
    --model <hf-id> --prompts prompts.jsonl --out dynamickv.jsonl
uv run --project backends/runners/adakv_env python backends/runners/run_adakv.py \
    --model <hf-id> --prompts prompts.jsonl --out adakv.jsonl

# 3. join + argmax (back in our main env)
uv run python calibration/join_labels.py \
    --baseline baseline.jsonl \
    --strategy kvquant_8b=kvquant_8b.jsonl \
    --strategy kvquant_3b=kvquant_3b.jsonl \
    --strategy dynamickv=dynamickv.jsonl \
    --strategy adakv=adakv.jsonl \
    --out measurements.jsonl --lambda-compress 1.0
```

λ sweep: re-run only step 3 with different `--lambda-compress` values; the per-strategy {ppl, cratio} are reused.
