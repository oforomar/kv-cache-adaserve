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

All three runners are implemented. Each has its own per-env README under `<name>_env/README.md` covering setup quirks specific to that backend.

| Runner | Env | Notes |
|---|---|---|
| `run_adakv.py` | `adakv_env/` | `transformers==4.44.2`, AdaKV submodule as editable path source. Two-step perplexity (prefill prompt → score target against post-eviction cache). |
| `run_kvquant.py` | `kvquant_env/` | KVQuant submodule as editable path source. **Requires offline pre-calibration** to produce a `quantizers.pickle` per bitwidth — see `kvquant_env/README.md`. |
| `run_dynamickv.py` | `dynamickv_env/` | DynamicKV is not pip-installable; runner adds it to `sys.path` and stubs an upstream hardcoded `transformers_modules` import. |

flash-attn is required for DynamicKV (patches `flash_attention_2`'s forward) and recommended for the other two for speed. Each env's README documents the install step (`uv pip install flash-attn --no-build-isolation`).

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
