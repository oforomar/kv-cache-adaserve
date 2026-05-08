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

**Choice: `meta-llama/Llama-3.1-8B`.** GQA (32 Q-heads / 8 KV-heads, group size 4), 128K native context — the xlong bucket actually exercises the model. Gated on HF Hub; the user needs to accept the license and `huggingface-cli login` before any GPU stage.

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

### 4b. KVQuant (`backends/runners/run_kvquant.py`)
Has custom CUDA kernels under `backends/kvquant/quant/`. Building the kernels against a specific torch ABI is the env's main constraint. Upstream has a documented eval pipeline; lift it into the per-prompt loop. Single runner emits both `kvquant_8b.jsonl` and `kvquant_3b.jsonl` via `--bitwidth`.

### 4c. DynamicKV (`backends/runners/run_dynamickv.py`)
Youngest of the three; expect more time reading source to find the right entrypoint. Monkey-patches transformers internals — env isolation is non-negotiable.

---

## 5. End-to-end smoke run on a small prompt set
**Status:** open · **Blocks:** committing 50 GPU-hours · **Effort:** ~hours, GPU

Once #4 has at least one real backend, run the *whole* chain — collect_signals → score_baseline → run_<backend> → join_labels → make_labels — on ~50 prompts before the full 2000-prompt run. Goals: surface env-conflict bugs, verify JSONL formats line up across backends, confirm `join_labels` produces a sane label distribution.

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
