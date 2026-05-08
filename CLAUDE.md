# CLAUDE.md

Guide for Claude Code working in this repo. Read `CALIBRATION_PIPELINE.md` for the full design rationale; this file is the operational summary.

## What this project is

The Phase B calibration **data pipeline** for a runtime-adaptive KV cache strategy selector. It generates training data (`training.jsonl`) for a small MLP classifier that picks one of {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV} per layer group at prefill time. QAQ is offline-only and excluded from the runtime classifier.

This repo currently contains **only the data-generation scripts**. The selector core (`signals.py`, `selector.py`, `strategies.py`) and the trainer (`train_classifier.py`) are documented separately and not yet vendored here — `collect_signals.py` and `score_strategies.py` import from them and will not run as-is until those modules are added to the project root.

## Layout

```
code/
├── CALIBRATION_PIPELINE.md   # design doc + full implementation reference
├── prompts.py                # stage 1 — curate length-stratified prompts
├── collect_signals.py        # stage 2 — per-(prompt, layer) signal capture (GPU)
├── score_strategies.py       # stage 3 — run each strategy, pick winner (GPU)
├── make_labels.py            # stage 4 — join + stratified-sample → training.jsonl
├── pyproject.toml            # uv-managed project metadata
└── .python-version           # 3.11
```

The four scripts each own one stage and write JSONL. Stages are independent — re-run any one without redoing the others. The doc describes them as living under `calibration/`; here they sit at the repo root, but the scripts still use `from calibration.prompts import …` and `from signals import …` style imports. **Either move them into a `calibration/` package and add the parent-package modules, or fix the imports — current state will fail on import.**

## Environment setup

```bash
uv sync                       # creates .venv and installs deps from pyproject.toml
uv run python prompts.py --mock --out prompts.jsonl
```

Dependencies pinned in `pyproject.toml`: `torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Mock paths only need stdlib + numpy; GPU stages need torch + transformers.

## Running the pipeline

End-to-end with mocks (no GPU, validates wiring only):

```bash
uv run python prompts.py --mock --out prompts.jsonl
# requires signals.jsonl from a real run; mock signals path TBD
uv run python score_strategies.py --mock --prompts prompts.jsonl \
    --signals signals.jsonl --out measurements.jsonl
uv run python make_labels.py --signals signals.jsonl \
    --measurements measurements.jsonl --out training.jsonl --target-size 10000
```

Real run (~50 GPU-hours per λ on H100, 7B model):

```bash
uv run python collect_signals.py --model <hf-id> --prompts prompts.jsonl --out signals.jsonl
uv run python score_strategies.py --model <hf-id> --prompts prompts.jsonl \
    --signals signals.jsonl --out measurements.jsonl --lambda-compress 1.0
uv run python make_labels.py ...
```

## Design decisions worth knowing before editing

- **Per-prompt labels with layer-varying signals.** Each prompt gets one label, but emits one row per (prompt × layer). True per-layer labels would cost ~1700 GPU-hours and are not budgeted. The agreement rate between Phase A and Phase B on a held-out set is the diagnostic for whether per-prompt is sufficient (>80% agreement = yes; <50% = need real per-layer labels).
- **λ in the scoring function** (`-(Δppl) - λ·(1 - cratio)`) is the most consequential single number. Recommendation is to sweep λ ∈ {0.1, 1, 10} and report the Pareto frontier rather than commit to one value.
- **Phase A precedence is explicit, not a scoring function:** QAQ-capable → high head-variance → low entropy → high entropy + long ctx → high entropy + short ctx → default KVQuant 8b. Order matters; head-variance check fires first and dominates if `τ_head_var` is mis-tuned (the mock run hit 100% Ada-KV with the default threshold — see "Mock Validation Findings" in the design doc).
- **Strategy is prefill-locked**, frozen for the rest of generation. Backends have incompatible memory layouts.
- **Eager attention is required** for `collect_signals.py` — FlashAttention does not materialize attention weights.
- **GQA**: head variance is over KV heads, entropy is over Q heads.
- **Feature normalization** for the MLP: `entropy / log(L)`, `log1p(L)`, `layer_idx / (num_layers - 1)`. Head variance stays raw.

## What's NOT done yet

1. Real prompt loaders in `prompts.py` (currently raises `SystemExit` without `--mock`).
2. KV-cache backend adapters in `score_strategies.measure_real` (raises `NotImplementedError`).
3. The selector core modules (`signals.py`, `selector.py`, `strategies.py`) — referenced via imports, not present in this repo.
4. The classifier trainer.

## Conventions

- Scripts read/write JSONL, one record per line, joined on `prompt_id`.
- All four stages support being re-run independently.
- Mock modes (`--mock`) exist for shape validation; never use them for production data.
- Don't add backwards-compat shims for the missing selector modules — fix imports or vendor the modules properly.
