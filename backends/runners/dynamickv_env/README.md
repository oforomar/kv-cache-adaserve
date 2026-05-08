# DynamicKV runner env

Hosts the eval-time portion of DynamicKV. No offline calibration needed — DynamicKV is purely a runtime cache-management policy ("plug-and-play" per upstream).

## Setup

```bash
# from repo root
uv sync --project backends/runners/dynamickv_env

# flash-attn is REQUIRED — DynamicKV's patches replace flash_attention_2's forward:
uv pip install --project backends/runners/dynamickv_env flash-attn --no-build-isolation
```

## Running

```bash
uv run --project backends/runners/dynamickv_env python backends/runners/run_dynamickv.py \
    --model meta-llama/Llama-3.1-8B --prompts prompts.jsonl \
    --out dynamickv.jsonl --max-capacity-prompt 512
```

`--max-capacity-prompt` is the per-layer KV budget in tokens. The upstream paper's tested range is 64–4096; 512 is a reasonable default for 7B-class models. Smaller values → heavier compression. The compression ratio reported in the output JSONL is `max_capacity_prompt / seq_len` (saturates at 1.0 for short prompts).

## Two upstream quirks worked around in the runner

1. **Hardcoded InternLM import.** DynamicKV's `monkeypatch.py` does an unconditional `import transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2` even when only the Llama path is used. The runner stubs that module in `sys.modules` before the import.
2. **Not pip-installable.** The DynamicKV submodule has no `pyproject.toml`/`setup.py`, so this env's deps don't include `dynamickv`. The runner adds `backends/dynamickv` to `sys.path` at startup.
