# KV Cache Adaptive Selector — Calibration Pipeline

Phase B calibration data pipeline for a runtime-adaptive KV cache strategy selector. Generates training data (`training.jsonl`) for a small MLP that picks one of {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV} per layer group at prefill time, based on three cheap signals: attention entropy, sequence length, and head-level variance.

The full design rationale (orthogonality argument, label-granularity trade-off, λ sweep, mock validation findings) lives in [`CALIBRATION_PIPELINE.md`](./CALIBRATION_PIPELINE.md). [`CLAUDE.md`](./CLAUDE.md) is the operational summary for AI coders.

## Status

| Component | State |
|---|---|
| Length-stratified prompt curation (`calibration/prompts.py`) | mock generator + real HF `datasets` loaders for all six sources (xlong via concatenation) |
| Per-(prompt, layer) signal capture (`calibration/collect_signals.py`) | implemented; needs a real model to run |
| Baseline FP16 perplexity (`calibration/score_baseline.py`) | implemented |
| Per-prompt label join + λ scoring (`calibration/join_labels.py`) | implemented |
| Mock label assignment via Phase A (`calibration/score_mock.py`) | implemented |
| Stratified training-set assembly (`calibration/make_labels.py`) | implemented |
| Selector core: `signals.py`, `selector.py`, `strategies.py` | vendored from the design doc |
| KV-cache backends (KVQuant 8b/3b, DynamicKV, Ada-KV) | upstream code vendored under `backends/` as submodules; per-backend runner scripts under `backends/runners/` are skeletons |
| Phase B classifier trainer | not yet present |

End-to-end mock pipeline runs and reproduces the doc's reported label distribution (Ada-KV ~50%, KVQuant 8b ~42%, DynamicKV ~7%, KVQuant 3b absent under synthetic signals).

## Pipeline shape

The doc's original "score every strategy in one process" idea was replaced with a **per-backend job** model: each compressor runs in its own environment, writes a per-strategy JSONL of `{prompt_id, ppl, cratio}`, and a final join step applies `score()` and picks the per-prompt argmax. Reasons: each upstream pins conflicting torch/transformers versions, KVQuant has custom CUDA kernels, DynamicKV monkey-patches transformers internals — one shared venv would be a constant source of dep fights.

```
prompts.jsonl ──▶ collect_signals ──▶ signals.jsonl ─────────────────┐
                                                                     │
prompts.jsonl ──▶ score_baseline ──▶ baseline.jsonl ───┐             │
                                                       ▼             ▼
prompts.jsonl ──▶ run_kvquant      ──▶ kvquant_8b.jsonl              │
                                       kvquant_3b.jsonl              │
prompts.jsonl ──▶ run_dynamickv    ──▶ dynamickv.jsonl  ──▶ join_labels ──▶ measurements.jsonl
prompts.jsonl ──▶ run_adakv        ──▶ adakv.jsonl     ──┘                          │
                                                                                    │
                                                            signals.jsonl  ─────────┤
                                                                                    ▼
                                                                              make_labels ──▶ training.jsonl
```

`join_labels` applies `score()` post-hoc, so the λ sweep is a re-run of *only* the join step against the same per-strategy JSONLs — no model re-runs. Mock mode (`score_mock.py`) bypasses everything between prompts and measurements by labeling via Phase A on aggregated signals.

## Quick start

```bash
git clone --recurse-submodules <this-repo>      # backends/ live in submodules
# or, if you cloned without --recurse-submodules:
git submodule update --init --recursive

uv sync                                                            # creates .venv

# Mock pipeline (no GPU, validates wiring)
uv run python calibration/prompts.py --mock --out prompts.jsonl
# (signals.jsonl needs a real model; see Real run below)
uv run python calibration/score_mock.py \
    --prompts prompts.jsonl --signals signals.jsonl --out measurements.jsonl
uv run python calibration/make_labels.py \
    --signals signals.jsonl --measurements measurements.jsonl \
    --out training.jsonl --target-size 10000
```

Real run (~50 GPU-hours per λ on H100, 7B model — once per-backend runners are filled in; see `backends/runners/README.md`):

```bash
# 0. real prompts (main env; downloads from HF Hub)
uv run python calibration/prompts.py --tokenizer <hf-id> --out prompts.jsonl

# 1. signals + baseline (main env)
uv run python calibration/collect_signals.py --model <hf-id> \
    --prompts prompts.jsonl --out signals.jsonl
uv run python calibration/score_baseline.py --model <hf-id> \
    --prompts prompts.jsonl --out baseline.jsonl

# 2. four per-backend runs (each in its own env)
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model <hf-id> --prompts prompts.jsonl --out kvquant_8b.jsonl --bitwidth 8
# ... kvquant_3b, dynamickv, adakv ...

# 3. join + argmax (main env)
uv run python calibration/join_labels.py --baseline baseline.jsonl \
    --strategy kvquant_8b=kvquant_8b.jsonl --strategy kvquant_3b=kvquant_3b.jsonl \
    --strategy dynamickv=dynamickv.jsonl --strategy adakv=adakv.jsonl \
    --out measurements.jsonl --lambda-compress 1.0

# 4. final stratified training set
uv run python calibration/make_labels.py --signals signals.jsonl \
    --measurements measurements.jsonl --out training.jsonl --target-size 10000
```

`join_labels` preserves per-strategy `{ppl, cratio}`, so the λ sweep ({0.1, 1, 10}) recommended in the design doc is a re-run of *only* step 3 with different `--lambda-compress` — no GPU re-runs.

## Layout

```
code/
├── CALIBRATION_PIPELINE.md   # design doc + full implementation reference
├── CLAUDE.md                 # operational summary for Claude Code
├── signals.py                # LayerSignals + attention_entropy / head_variance
├── selector.py               # HeuristicConfig + select_phase_a (Phase A)
├── strategies.py             # Strategy enum + score()
├── calibration/
│   ├── prompts.py            # stage 1 — curate length-stratified prompts
│   ├── collect_signals.py    # stage 2 — per-(prompt, layer) signal capture
│   ├── score_baseline.py     # stage 3a — FP16 baseline perplexity per prompt
│   ├── score_mock.py         # stage 3 mock — Phase A on aggregated signals
│   ├── join_labels.py        # stage 3b — join baseline + per-strategy → measurements
│   └── make_labels.py        # stage 4 — join + stratified-sample → training.jsonl
├── backends/                 # upstream compressor implementations
│   ├── kvquant/              # submodule: SqueezeAILab/KVQuant   (NeurIPS 2024)
│   ├── dynamickv/            # submodule: DreamMr/DynamicKV      (EMNLP 2025)
│   ├── adakv/                # submodule: FFY0/AdaKV             (NeurIPS 2025)
│   └── runners/              # per-backend runner scripts (one env each)
├── pyproject.toml
└── .python-version           # 3.11
```

Each calibration stage reads/writes JSONL and is independent — re-run any one without redoing the others.

## Wiring up the backend runners

The four runners under `backends/runners/` are skeletons. Each one's job is the same shape:

1. Load the model with one specific compressor active.
2. For each prompt, compute teacher-forced perplexity on `target_text` (mask the prompt portion from the loss).
3. Emit `{prompt_id, ppl, cratio}` JSONL.

Each runner ships in **its own uv environment** under `backends/runners/<name>_env/` (not yet committed — created when each runner is filled in and we know its real pin requirements). This isolation is necessary: KVQuant has custom CUDA kernels with specific torch ABI requirements; DynamicKV monkey-patches transformers internals; AdaKV pins specific transformers versions per branch. One shared env would constantly break.

Notes on the upstream repos:

- **AdaKV** has a `gqa_support` branch as an alternative to `main`. We track `main`; if your target model uses GQA (Llama-3, Mistral, Qwen), `cd backends/adakv && git checkout gqa_support` may be the right move. Pin the resulting submodule commit before running production calibration.
- The runner scripts deliberately import nothing from the rest of this repo — they communicate via JSONL on disk. That keeps each runner's env free of our calibration deps and vice versa.

## Open issues to know about before a production run

- **`τ_head_var` is the most sensitive number in Phase A.** The mock validation hit 100% Ada-KV labels with the default; recalibrate against real signals before any production labeling run.
- **`signals.head_variance` statistic is a placeholder.** Cross-head variance of mean peak attention probability — the design doc doesn't pin down the exact statistic. Threshold calibration depends on this choice.
- **GQA**: `head_variance` runs over Q-heads (post-repetition); the doc calls for KV-heads. With GQA models, group Q-heads by `num_key_value_groups` first.
- **λ sweep**: don't commit to one λ. Sweep {0.1, 1, 10} via post-hoc relabeling from saved measurements and report the Pareto frontier. The score function is `-(Δppl) + λ·(1 - cratio)` — λ=0.1 favors quality, λ=10 favors aggressive compression.
