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
├── strategies.py             # Strategy enum + adapter REGISTRY (stubbed)
├── calibration/              # data-pipeline scripts, one per stage
│   ├── prompts.py            # stage 1 — curate length-stratified prompts
│   ├── collect_signals.py    # stage 2 — per-(prompt, layer) signal capture (GPU)
│   ├── score_strategies.py   # stage 3 — run each strategy, pick winner (GPU)
│   └── make_labels.py        # stage 4 — join + stratified-sample → training.jsonl
├── pyproject.toml            # uv-managed project metadata
└── .python-version           # 3.11
```

The four scripts each own one stage and write JSONL. Stages are independent — re-run any one without redoing the others. Imports rely on `calibration/`'s parent (the repo root) being on `sys.path`; each script does this via `sys.path.insert(0, parents[1])`, so running them as `uv run python calibration/<stage>.py` works.

## Environment setup

```bash
uv sync                       # creates .venv and installs deps from pyproject.toml
uv run python calibration/prompts.py --mock --out prompts.jsonl
```

Dependencies pinned in `pyproject.toml`: `torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Mock paths only need stdlib; GPU stages need torch + transformers.

## Running the pipeline

End-to-end with mocks (no GPU, validates wiring only — `score_strategies --mock` labels via Phase A on prompt-aggregated signals, so it still requires a `signals.jsonl`):

```bash
uv run python calibration/prompts.py --mock --out prompts.jsonl
uv run python calibration/score_strategies.py --mock \
    --prompts prompts.jsonl --signals signals.jsonl --out measurements.jsonl
uv run python calibration/make_labels.py \
    --signals signals.jsonl --measurements measurements.jsonl \
    --out training.jsonl --target-size 10000
```

Real run (~50 GPU-hours per λ on H100, 7B model):

```bash
uv run python calibration/collect_signals.py --model <hf-id> \
    --prompts prompts.jsonl --out signals.jsonl
uv run python calibration/score_strategies.py --model <hf-id> \
    --prompts prompts.jsonl --signals signals.jsonl \
    --out measurements.jsonl --lambda-compress 1.0
uv run python calibration/make_labels.py ...
```

`measure_real` writes per-strategy `{ppl, cratio, score}` for every prompt, so you can sweep λ post-hoc by recomputing `score()` from the saved measurements without re-running the model.

## Design decisions worth knowing before editing

- **Per-prompt labels with layer-varying signals.** Each prompt gets one label, but emits one row per (prompt × layer). True per-layer labels would cost ~1700 GPU-hours and are not budgeted. The agreement rate between Phase A and Phase B on a held-out set is the diagnostic for whether per-prompt is sufficient (>80% agreement = yes; <50% = need real per-layer labels).
- **λ in the scoring function** (`-(Δppl) - λ·(1 - cratio)`) is the most consequential single number. Recommendation is to sweep λ ∈ {0.1, 1, 10} and report the Pareto frontier rather than commit to one value.
- **Phase A precedence is explicit, not a scoring function:** QAQ-capable → high head-variance → low entropy → high entropy + long ctx → high entropy + short ctx → default KVQuant 8b. Order matters; head-variance check fires first and dominates if `τ_head_var` is mis-tuned (the mock run hit 100% Ada-KV with the default threshold — see "Mock Validation Findings" in the design doc).
- **Strategy is prefill-locked**, frozen for the rest of generation. Backends have incompatible memory layouts.
- **Eager attention is required** for `collect_signals.py` — FlashAttention does not materialize attention weights.
- **GQA**: head variance is over KV heads, entropy is over Q heads.
- **Feature normalization** for the MLP: `entropy / log(L)`, `log1p(L)`, `layer_idx / (num_layers - 1)`. Head variance stays raw.

## What's NOT done yet

1. Real prompt loaders in `calibration/prompts.py` (currently raises `SystemExit` without `--mock`).
2. KV-cache backend adapters in `strategies.REGISTRY` are stubs — they raise `NotImplementedError` when entered. `measure_real` is otherwise complete (loads model + tokenizer, runs baseline + per-strategy perplexity via the registry, writes per-prompt measurements with per-strategy scores) but needs real adapters before it produces useful labels.
3. `signals.head_variance` uses cross-head variance of mean peak attention probability as the heterogeneity statistic — the design doc doesn't pin this down, and `tau_head_var` calibration depends on it. Re-tune the threshold against real signals before any production labeling run.
4. GQA: `head_variance` operates on Q-heads (post-repetition); the doc calls for KV-heads. With GQA models, group Q-heads by `num_key_value_groups` first.
5. The classifier trainer.

## Conventions

- Scripts read/write JSONL, one record per line, joined on `prompt_id`.
- All four stages support being re-run independently.
- Mock modes (`--mock`) exist for shape validation; never use them for production data.
- Don't add backwards-compat shims for the missing selector modules — fix imports or vendor the modules properly.

## doc lookup
Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.
