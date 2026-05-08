# KV Cache Adaptive Selector — Calibration Pipeline

Phase B calibration data pipeline for a runtime-adaptive KV cache strategy selector. Generates training data (`training.jsonl`) for a small MLP that picks one of {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV} per layer group at prefill time, based on three cheap signals: attention entropy, sequence length, and head-level variance.

The full design rationale (orthogonality argument, label-granularity trade-off, λ sweep, mock validation findings) lives in [`CALIBRATION_PIPELINE.md`](./CALIBRATION_PIPELINE.md). [`CLAUDE.md`](./CLAUDE.md) is the operational summary for AI coders.

## Status

| Component | State |
|---|---|
| Length-stratified prompt curation (`calibration/prompts.py`) | mock generator works; real HF `datasets` loaders not wired |
| Per-(prompt, layer) signal capture (`calibration/collect_signals.py`) | implemented; needs a real model to run |
| Strategy scoring + per-prompt labeling (`calibration/score_strategies.py`) | `measure_real` end-to-end (model load, perplexity, scoring loop, JSONL output); backend adapters are stubs |
| Stratified label assembly (`calibration/make_labels.py`) | implemented |
| Selector core: `signals.py`, `selector.py`, `strategies.py` | vendored from the design doc |
| KV-cache backend adapters (KVQuant 8b/3b, DynamicKV, Ada-KV) | **stubbed** — raise `NotImplementedError` until real compressors are registered |
| Phase B classifier trainer | not yet present |

End-to-end mock pipeline runs and reproduces the doc's reported label distribution (Ada-KV ~50%, KVQuant 8b ~42%, DynamicKV ~7%, KVQuant 3b absent under synthetic signals).

## Quick start

```bash
uv sync                                                            # creates .venv

# Mock pipeline (no GPU, validates wiring)
uv run python calibration/prompts.py --mock --out prompts.jsonl
# (signals.jsonl needs a real model; see Real run below)
uv run python calibration/score_strategies.py --mock \
    --prompts prompts.jsonl --signals signals.jsonl --out measurements.jsonl
uv run python calibration/make_labels.py \
    --signals signals.jsonl --measurements measurements.jsonl \
    --out training.jsonl --target-size 10000
```

Real run (~50 GPU-hours per λ on H100, 7B model — once backend adapters are wired):

```bash
uv run python calibration/collect_signals.py --model <hf-id> \
    --prompts prompts.jsonl --out signals.jsonl
uv run python calibration/score_strategies.py --model <hf-id> \
    --prompts prompts.jsonl --signals signals.jsonl \
    --out measurements.jsonl --lambda-compress 1.0
uv run python calibration/make_labels.py \
    --signals signals.jsonl --measurements measurements.jsonl \
    --out training.jsonl
```

`measure_real` records per-strategy `{ppl, cratio, score}` for every prompt, so the λ sweep ({0.1, 1, 10}) recommended in the design doc can be done post-hoc by recomputing `score()` from saved measurements without re-running the model.

## Layout

```
code/
├── CALIBRATION_PIPELINE.md   # design doc + full implementation reference
├── CLAUDE.md                 # operational summary for Claude Code
├── signals.py                # LayerSignals + attention_entropy / head_variance
├── selector.py               # HeuristicConfig + select_phase_a (Phase A)
├── strategies.py             # Strategy enum + adapter REGISTRY (stubbed)
├── calibration/
│   ├── prompts.py            # stage 1 — curate length-stratified prompts
│   ├── collect_signals.py    # stage 2 — per-(prompt, layer) signal capture
│   ├── score_strategies.py   # stage 3 — run each strategy, pick winner
│   └── make_labels.py        # stage 4 — join + stratified-sample → training.jsonl
├── pyproject.toml
└── .python-version           # 3.11
```

Each calibration stage reads/writes JSONL and is independent — re-run any one without redoing the others.

## Plugging in real backend adapters

Each strategy in `strategies.REGISTRY` is a context manager `(model) -> ContextManager[float]` that patches the model's K/V handling on enter, yields the compression ratio, and unpatches on exit. To wire in a real backend:

```python
from contextlib import contextmanager
from strategies import Strategy, register

@contextmanager
def kvquant_8b_adapter(model):
    # ... patch model attention to use KVQuant 8-bit K/V encoding ...
    cratio = 8 / 16  # bitwidth ratio (replace with measured value)
    try:
        yield cratio
    finally:
        # ... restore original attention ...
        pass

register(Strategy.KVQUANT_8B, kvquant_8b_adapter)
```

Repeat for `KVQUANT_3B`, `DYNAMICKV`, `ADAKV`.

## Open issues to know about before a production run

- **`τ_head_var` is the most sensitive number in Phase A.** The mock validation hit 100% Ada-KV labels with the default; recalibrate against real signals before any production labeling run.
- **`signals.head_variance` statistic is a placeholder.** Cross-head variance of mean peak attention probability — the design doc doesn't pin down the exact statistic. Threshold calibration depends on this choice.
- **GQA**: `head_variance` runs over Q-heads (post-repetition); the doc calls for KV-heads. With GQA models, group Q-heads by `num_key_value_groups` first.
- **λ sweep**: don't commit to one λ. Sweep {0.1, 1, 10} via post-hoc relabeling from saved measurements and report the Pareto frontier.
