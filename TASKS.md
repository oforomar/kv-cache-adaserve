# TASKS

Open work, in rough priority order. Update as items land or new ones surface.

---

## 1. Resolve the `score()` sign inconsistency — **DONE**

Resolved by flipping the formula sign in `strategies.score()` to `-(Δppl) + λ·(1 − cratio)`, matching the design doc's prose. Updated `CALIBRATION_PIPELINE.md`, `README.md`, `CLAUDE.md` to reflect. Smoke-tested with the synthetic λ sweep:

- λ=0.1 → `kvquant_8b` leads (mildest compression, smallest ppl hit)
- λ=10 → `dynamickv`/`adakv` dominate; `kvquant_8b` drops to 1/1850

Behavior now matches "λ=10 favors aggressive compression".

---

## 2. Real prompt loaders in `calibration/prompts.py`
**Status:** open · **Blocks:** any non-mock pipeline run · **Effort:** ~1 day, no GPU

Currently `--mock` only — the real path is a `SystemExit`. Wire HuggingFace `datasets` for the six sources in `TARGET_MIX`:

- WikiText (200/200/100/50 across short/medium/long/xlong)
- LongBench (0/100/200/100)
- MMLU (150/50/0/0)
- HumanEval (100/50/0/0)
- CNN/DailyMail (100/150/0/0)
- Alpaca (200/100/0/0)

Each source needs: tokenize, length-bucket, split into `(text, target_text)` for perplexity scoring, write to JSONL. Verify per-bucket counts match `TARGET_MIX` before declaring done. ~1500–2000 prompts total.

---

## 3. Pick the calibration target model
**Status:** open · **Blocks:** task #4 · **Effort:** decision

Choose the 7B-class model (Llama-2-7B? Llama-3-8B? Mistral-7B?). This decides:

- Whether AdaKV's `gqa_support` branch is required (yes for Llama-3, Mistral, Qwen — most modern 7B-class models use GQA).
- Per-backend env pins (each backend supports a different transformers/torch range).
- Whether the GQA `head_variance` issue (CLAUDE.md "What's NOT done yet" #4) needs to be fixed before signal collection or can be deferred.

---

## 4. Fill in the per-backend runners
**Status:** skeletons committed · **Blocks:** real label runs · **Effort:** ~1 week per backend, work in parallel

Three independent tracks. Each runner needs: a uv project under `backends/runners/<name>_env/`, real model loading via the upstream submodule, per-prompt teacher-forced perplexity, compression-ratio reporting, JSONL output.

### 4a. Ada-KV (`backends/runners/run_adakv.py`)
Closest to a library — head-adaptive eviction is a relatively clean wrapper. Probably the best starting point for proving the per-backend env pattern works. Decide `main` vs `gqa_support` branch first (see task #3).

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
- **GQA-aware `head_variance`.** Operates on Q-heads (post-repetition) today; the doc calls for KV-heads. With GQA models, group Q-heads by `num_key_value_groups` first. Probably tied to task #3.
- **Cleanup `tau_*` thresholds in `selector.HeuristicConfig`.** The defaults (`tau_head_var=1e-4`, etc.) are educated guesses. Recalibrate against the first real signals batch.
