# CLAUDE.md

Guide for Claude Code working in this repo. Read `CALIBRATION_PIPELINE.md` for the full design rationale; this file is the operational summary.

## What this project is

The Phase B calibration **data pipeline** for a runtime-adaptive KV cache strategy selector. It generates training data (`training.jsonl`) for a small MLP classifier that picks one of {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV} per layer group at prefill time. QAQ is offline-only and excluded from the runtime classifier.

The selector core (`signals.py`, `selector.py`, `strategies.py`) is vendored here, derived from the design doc. The trainer (`train_classifier.py`) is documented separately and not yet present.

## Layout

```
code/
├── CALIBRATION_PIPELINE.md   # design doc + full implementation reference
├── signals.py                # LayerSignals + attention_entropy / head_variance
├── selector.py               # HeuristicConfig + select_phase_a (Phase A)
├── strategies.py             # Strategy enum + score(); no adapter registry
├── calibration/              # data-pipeline scripts (run in main env)
│   ├── prompts.py            # stage 1 — curate length-stratified prompts
│   ├── collect_signals.py    # stage 2 — per-(prompt, layer) signal capture (GPU)
│   ├── score_baseline.py     # stage 3a — FP16 baseline ppl per prompt (GPU)
│   ├── score_mock.py         # stage 3 mock — Phase A labels on aggregated signals
│   ├── join_labels.py        # stage 3b — baseline + per-strategy → measurements
│   └── make_labels.py        # stage 4 — join + stratified-sample → training.jsonl
├── backends/                 # upstream compressor code + per-backend runners
│   ├── kvquant/              # submodule: SqueezeAILab/KVQuant   (NeurIPS 2024)
│   ├── dynamickv/            # submodule: DreamMr/DynamicKV      (EMNLP 2025)
│   ├── adakv/                # submodule: FFY0/AdaKV             (NeurIPS 2025)
│   └── runners/              # per-backend runner scripts (each its own env)
├── pyproject.toml            # uv-managed project metadata
└── .python-version           # 3.11
```

After cloning, run `git submodule update --init --recursive` (or clone with `--recurse-submodules`). AdaKV is pinned to its `gqa_support` branch since the calibration target is Llama-3.1-8B (GQA). KVQuant and DynamicKV track `main`.

**Calibration target: `meta-llama/Llama-3.2-3B`.** Gated on HF Hub — accept the license and `huggingface-cli login` before any GPU stage. 24 Q-heads / 8 KV-heads (GQA group size 3); 128K native context. Switched from Llama-3.1-8B because the 16GB VRAM on the available laptop GPU (RTX 3080 Laptop) can't hold 8B weights plus a meaningful KV cache.

`calibration/` scripts each own one stage and write JSONL; stages are independent. Imports rely on the repo root being on `sys.path`; each script does this via `sys.path.insert(0, parents[1])`, so `uv run python calibration/<stage>.py` works.

`backends/runners/` scripts run in **separate uv environments** under `backends/runners/<name>_env/` (not yet committed). They communicate with the rest of the pipeline via JSONL on disk only — no Python imports across the boundary. This is deliberate: KVQuant has custom CUDA kernels, DynamicKV monkey-patches transformers, each pins its own torch/transformers versions; one shared env would constantly break.

## Environment setup

```bash
uv sync                       # creates .venv and installs deps from pyproject.toml
uv run python calibration/prompts.py --mock --out prompts.jsonl
```

Dependencies pinned in `pyproject.toml`: `torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Mock paths only need stdlib; GPU stages need torch + transformers.

## Running the pipeline

End-to-end with mocks (no GPU, validates wiring — `score_mock` labels via Phase A on prompt-aggregated signals, so it still requires a `signals.jsonl`):

```bash
uv run python calibration/prompts.py --mock --out prompts.jsonl
uv run python calibration/score_mock.py \
    --prompts prompts.jsonl --signals signals.jsonl --out measurements.jsonl
uv run python calibration/make_labels.py \
    --signals signals.jsonl --measurements measurements.jsonl \
    --out training.jsonl --target-size 10000
```

Real run (~50 GPU-hours per λ on H100, 7B model — once per-backend runners are filled in):

```bash
# 0. real prompts (main env; pulls from HF datasets)
uv run python calibration/prompts.py --tokenizer <hf-id> --out prompts.jsonl

# 1. signals + baseline (main env)
uv run python calibration/collect_signals.py --model <hf-id> \
    --prompts prompts.jsonl --out signals.jsonl
uv run python calibration/score_baseline.py --model <hf-id> \
    --prompts prompts.jsonl --out baseline.jsonl

# 2. per-backend runs (each in its own env — see backends/runners/README.md)
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

`join_labels` preserves per-strategy `{ppl, cratio}`, so the λ sweep is a re-run of *only* step 3 with different `--lambda-compress`. No GPU re-runs.

## Design decisions worth knowing before editing

- **Per-prompt labels with layer-varying signals.** Each prompt gets one label, but emits one row per (prompt × layer). True per-layer labels would cost ~1700 GPU-hours and are not budgeted. The agreement rate between Phase A and Phase B on a held-out set is the diagnostic for whether per-prompt is sufficient (>80% agreement = yes; <50% = need real per-layer labels).
- **λ in the scoring function** (`-(Δppl) + λ·(1 - cratio)`) is the most consequential single number. λ=0.1 favors quality, λ=10 favors aggressive compression. Recommendation is to sweep λ ∈ {0.1, 1, 10} and report the Pareto frontier rather than commit to one value.
- **Phase A precedence is explicit, not a scoring function:** QAQ-capable → high head-variance → low entropy → high entropy + long ctx → high entropy + short ctx → default KVQuant 8b. Order matters; head-variance check fires first and dominates if `τ_head_var` is mis-tuned (the mock run hit 100% Ada-KV with the default threshold — see "Mock Validation Findings" in the design doc).
- **Strategy is prefill-locked**, frozen for the rest of generation. Backends have incompatible memory layouts.
- **Eager attention is required** for `collect_signals.py` — FlashAttention does not materialize attention weights.
- **GQA**: `signals.head_variance(weights, num_kv_groups)` folds adjacent Q-heads into KV-head groups before taking the cross-head variance. `collect_signals` reads `attn.num_key_value_groups` from each Llama-family attention module and threads it through. Entropy stays over Q-heads.
- **Feature normalization** for the MLP: `entropy / log(L)`, `log1p(L)`, `layer_idx / (num_layers - 1)`. Head variance stays raw.

## What's NOT done yet

1. Per-backend runners under `backends/runners/` are implemented (AdaKV, KVQuant, DynamicKV) but pending GPU validation against `meta-llama/Llama-3.2-3B`. KVQuant additionally requires the user to run upstream's offline calibration (Fisher info + NUQ codebook) once per bitwidth before its runner can produce useful labels — see `backends/runners/kvquant_env/README.md`.
3. `signals.head_variance` uses cross-head variance of mean peak attention probability as the heterogeneity statistic — the design doc doesn't pin this down, and `tau_head_var` calibration depends on it. Re-tune the threshold against real signals before any production labeling run.
4. The classifier trainer.

## Conventions

- Scripts read/write JSONL, one record per line, joined on `prompt_id`.
- All stages support being re-run independently.
- Mock modes (`--mock`, `score_mock.py`) exist for shape validation; never use them for production data.
- Per-backend runners do not import from this repo's modules — JSONL on disk is the only contract. Keep it that way to preserve env isolation.

## doc lookup
Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.
