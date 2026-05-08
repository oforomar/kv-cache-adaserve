# TASKS

Open work, in rough priority order. Update as items land or new ones surface.

---

## 1. Resolve the `score()` sign inconsistency — **DONE**

Resolved by flipping the formula sign in `strategies.score()` to `-(Δppl) + λ·(1 − cratio)`, matching the design doc's prose. Updated `CALIBRATION_PIPELINE.md`, `README.md`, `CLAUDE.md` to reflect. Smoke-tested with the synthetic λ sweep:

- λ=0.1 → `kvquant_8b` leads (mildest compression, smallest ppl hit)
- λ=10 → `dynamickv`/`adakv` dominate; `kvquant_8b` drops to 1/1850

Behavior now matches "λ=10 favors aggressive compression".

---

## 2. Real prompt loaders in `calibration/prompts.py` — **DONE**

Implemented all six HF dataset sources (WikiText, LongBench, MMLU, HumanEval, CNN/DailyMail, Alpaca) behind a shared "pool tokens, slice into buckets" core. Decisions: xlong via concatenation across adjacent passages; fixed 128-token target_text. CLI: `--tokenizer <hf-id> --out prompts.jsonl` (the tokenizer must match the calibration target model so bucket lengths are tokenwise consistent with downstream signal collection).

Smoke-tested the pooler with a fake tokenizer + iterator: bucket counts honored, prompt lengths randomized within bounds. Real-network validation deferred to first end-to-end run (task 5) — dataset IDs may need adjusting if any have moved on HF Hub.

---

## 3. Pick the calibration target model — **DONE**

**Initial choice: `meta-llama/Llama-3.1-8B`** (32 Q / 8 KV, group 4). **Revised to `meta-llama/Llama-3.2-3B`** (24 Q / 8 KV, group 3, 128K ctx) after hardware probe revealed the available GPU is a 16GB laptop card; 8B bf16 weights alone fill it before any KV cache. Both gated on HF Hub.

The GQA `head_variance` fold is divisibility-checked, so it works for either group size; switching targets needs no code changes beyond the `--model` / `--tokenizer` CLI args.

Follow-on changes landed with this task:

- **AdaKV submodule switched to `gqa_support` branch** (commit `1c1d99a3`) — required for any GQA-aware compressor work.
- **`signals.head_variance` is now GQA-aware**: takes `num_kv_groups` (defaults to 1 = MHA, no behavior change), folds adjacent Q-heads into KV-head groups by averaging the per-head peak statistic, then takes variance across KV-heads. `collect_signals.install_hooks` reads `attn.num_key_value_groups` from each Llama-family attention module and threads it through. Smoke-tested: MHA case unchanged; GQA folded variance < unfolded, matching expectations.
- `tau_head_var = 1e-4` was tuned against the unfolded statistic; expect it to need adjustment on real Llama-3.1 signals (already tracked under "Lower-priority cleanups").

---

## 4. Fill in the per-backend runners
**Status:** skeletons committed · **Blocks:** real label runs · **Effort:** ~1 week per backend, work in parallel

Three independent tracks. Each runner needs: a uv project under `backends/runners/<name>_env/`, real model loading via the upstream submodule, per-prompt teacher-forced perplexity, compression-ratio reporting, JSONL output.

### 4a. Ada-KV (`backends/runners/run_adakv.py`) — **DONE pending GPU validation**

Implemented:
- Env `backends/runners/adakv_env/` with `transformers==4.44.2`, `torch>=2.2`, the AdaKV submodule as an editable path source, and a note about installing flash-attn separately (`--no-build-isolation`).
- Runner uses `replace_llama_adaptive()` + `config_compress(...)` with `gqa_support=True`, `gqa_func="mean"` (Llama-3.1 GQA).
- Two-step perplexity: prefill the prompt (eviction fires inside the patched attention forward when `seq_len > base_capacity`), then teacher-force `target_text` token-by-token against the post-eviction KV cache.
- `cratio = min(1.0, base_capacity / max(prompt_len, 1))` — AdaKV's design ratio.
- CLI exposes `--base-capacity`, `--window-size`, `--kernel-size`, `--floor-alpha`, `--max-length`, `--device`.

Open follow-ups (need GPU + HF auth to validate):
- Confirm the AdaKV submodule installs cleanly via `uv sync --project backends/runners/adakv_env`. The submodule's own `pyproject.toml` strictly pins `numpy==1.24.0` and `tqdm==4.66.1` — if those clash with anything else, relax via uv's override.
- Run a sanity check on ~5 prompts and verify the output JSONL has plausible ppl (close to baseline at `base_capacity ≥ seq_len`, higher when `base_capacity « seq_len`).
- The two-step perplexity loop runs decode autoregressively (one forward per target token). For 128-token targets × 1850 prompts that's ~237K extra forwards — measure the actual time on H100; if it's a meaningful fraction of total budget, batch the target-side forwards.

### 4b. KVQuant (`backends/runners/run_kvquant.py`) — **DONE pending GPU validation + offline prep**

Implemented:
- Env `backends/runners/kvquant_env/` with KVQuant submodule (`backends/kvquant/quant`) as editable path source.
- Runner uses `kvquant.simquant_module_quantizer.make_quant_sim` to patch K/V projections (per-channel for `k_proj`, per-token + dynamic for `v_proj`); falls back from flash-attn to eager if FA isn't built.
- Per-prompt teacher-forced perplexity, standard pattern (no two-step needed; quantization is in the patched projections).
- Bitwidth mapping: `--bitwidth 8 → KVQuant 4-bit`, `--bitwidth 3 → 3-bit`. KVQuant has no native 8-bit; 4-bit is the closest "light-compression" point. `cratio = abits / 16`.
- CLI exposes NUQ on/off, NormalFloat NUQ, sparsity threshold, attention-sink fp16-prefix, max-length.

**Critical prerequisite (cannot be automated):** the user must run KVQuant's upstream Fisher-info pipeline + `llama_simquant.py --quantize` once per bitwidth, producing `quantizers_4bit.pickle` and `quantizers_3bit.pickle`. See `backends/runners/kvquant_env/README.md` for the prep CLI. This calibration is hours of mostly-CPU work.

Open follow-ups (need GPU + the prep pickles):
- Verify the runner finds and applies the pickle without errors against Llama-3.1-8B.
- The `--first-few-fp16` attention-sink option may improve quality at low bitwidths; sweep on a small set after validation.

### 4c. DynamicKV (`backends/runners/run_dynamickv.py`) — **DONE pending GPU validation**

Implemented:
- Env `backends/runners/dynamickv_env/` with `transformers>=4.44,<4.46` (matches upstream) and a note that flash-attn is mandatory (the runner uses DynamicKV's `flash_attention_2` patches).
- Runner uses `kv_compression.token_drop.monkeypatch.replace_attention(model_type="llama", method="dynamickv_v11")` (the upstream-recommended head per their LongBench scripts), then sets per-layer `self_attn.config.{window_size, max_capacity_prompt, kernel_size, pooling, radio_max}`.
- Per-prompt teacher-forced perplexity. `cratio = max_capacity_prompt / seq_len` (saturates at 1.0 for short prompts).

Two upstream quirks worked around in the runner:
- DynamicKV's `monkeypatch.py` does an unconditional `import transformers_modules.internlm2_5_7b_chat_1m.modeling_internlm2` at module-import time; the runner stubs that module in `sys.modules` before importing.
- The DynamicKV submodule has no `setup.py` / `pyproject.toml`, so it isn't pip-installable. The runner adds `backends/dynamickv` to `sys.path` at startup.

Open follow-ups (need GPU):
- Verify the patched flash_attention_2 forward runs cleanly on Llama-3.1-8B.
- Sweep `--max-capacity-prompt` (paper tested 64–4096) on a small set after validation.

---

## 5. End-to-end smoke run on a small prompt set — **PARTIAL DONE**

5-prompt smoke run on Llama-3.2-3B with AdaKV + DynamicKV (KVQuant skipped pending offline pre-calibration). Surfaced and fixed several real issues; the pipeline now runs end-to-end on this hardware.

**Fixes that landed during the smoke run:**
- `signals.attention_entropy` produced NaN in fp16 (`clamp_min(1e-12)` underflows to 0 there). Cast to fp32 before the clamp.
- Per-runner-env tooling: the system Python 3.11 lacked `Python.h`; switched both runner envs to a uv-managed Python 3.11 install. Also pinned `torch==2.5.1+cu121` so prebuilt flash-attn wheels exist (the auto-resolved torch 2.11+cu130 had no wheel for our cp311 + nvcc 12 combo). flash-attn URL-pinned in both envs.
- AdaKV's custom CUDA extension (`tiny_api_cuda` from `backends/adakv/csrc`) required a separate `python build.py install` run. Built cleanly against `CUDA_HOME=/usr` with the system nvcc 12.0 + torch's cu121 bindings. Documented; one-time build per env.
- DynamicKV's runner needed two compatibility shims:
  - The InternLM stub now exposes `InternLM2FlashAttention2` and `InternLM2ForCausalLM` classes, since `replace_attention()` assigns to those unconditionally regardless of `model_type`.
  - `_flash_attention_forward` was a method on `LlamaFlashAttention2` in older transformers but was moved to a module-level helper in 4.44+. The runner adds a method-form shim that delegates to the new helper.
- `join_labels.py` now has `--allow-partial` for smoke runs that intentionally cover a subset of strategies.

**Smoke results:** 5 wikitext-short prompts, base_capacity=128, max_capacity_prompt=128:

| Stage | Output |
|---|---|
| baseline (FP16) | ppl 8.1, 14.0, 18.5 (≈), ?, 11.8 |
| DynamicKV | ppl 8.1, 14.0, 16.3, 23.9, 11.8 — healthy (~baseline) |
| AdaKV | ppl 137, 206, 557, 429, 187 — **inflated** (see below) |
| join_labels | 5 measurements, argmax = dynamickv for all 5 |
| make_labels | 100 training rows with the trainer-expected schema |

**Known issue: AdaKV perplexity is way above baseline at any base_capacity.** AdaKV's API is built around `model.generate()` with internal cache state; teacher-forced perplexity via prompt-prefill + token-by-token decode causes per-step re-evictions on its flattened post-eviction layout. Bulk-decoding the target via `past_key_values=...` doesn't work because the post-eviction K/V tensor is rank-2 (flattened) rather than rank-4. **The runner produces JSONL — pipeline is unblocked — but AdaKV's ppl numbers are not a valid measurement of its quality.** Fix options: use `model.generate()` with log-prob extraction, or freeze AdaKV's eviction after the first prefill via a hook. Tracked separately as task 5b below.

## 5b. Fix AdaKV teacher-forced perplexity
**Status:** open · **Effort:** half-day

AdaKV's design assumes one prefill + autoregressive generation, not teacher-forcing. Two paths:
- `model.generate()` with `output_scores=True` to extract per-token log-probs of the target sequence — requires injecting the target tokens as `decoder_input_ids` or using `forced_decoder_ids`.
- Hook into AdaKV's eviction trigger to fire only on the first prefill, then run a normal teacher-forced second forward.

Until this is fixed, AdaKV will lose every per-prompt argmax (its score is wildly negative).

---

## 6. Phase A ↔ Phase B agreement-rate diagnostic
**Status:** waiting on #4–#5 · **Effort:** small once real measurements exist

Per the design doc: compute the agreement rate between Phase A labels and the per-prompt argmax labels on a held-out subset.

- **>80% agreement** → per-prompt labels are sufficient; Phase B's main value is inference speed.
- **<50%** → real per-layer labels are needed; calibration set must shrink or the budget must grow.

This is the diagnostic that justifies (or kills) the per-prompt labeling shortcut.

---

## 7. Phase B classifier trainer
**Status:** out of scope for this repo (per CLAUDE.md) · **Effort:** separate repo

Consumes `training.jsonl`. Documented separately. Listed here only so it doesn't get forgotten — once #1–#6 are done, this is what `training.jsonl` is *for*.

---

## Lower-priority cleanups

- **`signals.head_variance` statistic choice.** Currently cross-head variance of mean peak attention probability — placeholder pick. The exact statistic affects `tau_head_var` calibration. Re-tune on real signals before any production labeling run.
- ~~**GQA-aware `head_variance`.**~~ Done as part of task 3.
- **Cleanup `tau_*` thresholds in `selector.HeuristicConfig`.** The defaults (`tau_head_var=1e-4`, etc.) are educated guesses. Recalibrate against the first real signals batch.
