# KV Cache Adaptive Selector — Calibration Pipeline & Phase B Classifier

Runtime-adaptive KV cache strategy selector for LLM inference. Dynamically picks one of **{KVQuant 4-bit, KVQuant 3-bit, DynamicKV, Ada-KV}** per layer at prefill time based on three cheap signals: attention entropy, sequence length, and head-level variance.

Part of the **AdaptiveServe** framework — see the research proposal for the full five-layer architecture (KV cache + activation sparsity + query routing + cloud-edge splitting).

---

## What's in this repo

| File / Folder | Purpose |
|---|---|
| `signals.py` | `LayerSignals` dataclass + `attention_entropy()` / `head_variance()` |
| `selector.py` | Phase A rule-based heuristic (`select_phase_a`) |
| `strategies.py` | `Strategy` enum + `score()` function |
| `phase_b.py` | Phase B MLP definition + `load_phase_b()` / `select_phase_b()` runtime |
| `train_classifier.py` | Phase B trainer — trains one MLP per λ |
| `evaluate_classifier.py` | Full evaluation: confusion matrix, P/R/F1, bucket analysis |
| `class_report.py` | Class representation + per-class P/R/F1 report |
| `kfold_eval.py` | K-fold cross-validation at the prompt level |
| `calibration/` | 5-stage data pipeline (prompts → signals → baseline → join → labels) |
| `backends/runners/` | Per-backend runner scripts (each in its own isolated uv env) |
| `calib100/` | Validation run: 100 prompts × 28 layers on Llama-3.2-3B |
| `models/` | Trained Phase B checkpoints for λ ∈ {0.1, 1.0, 10.0} |

---

## Two-phase selector

### Phase A — Rule-based heuristic (`selector.py`)

Fires at prefill using explicit precedence rules on the three signals:

```
head_variance > τ_head_var          →  Ada-KV
entropy_normalized < τ_low          →  DynamicKV
entropy_normalized > τ_high, L≥4096 →  KVQuant 3-bit
entropy_normalized > τ_high, L<4096 →  KVQuant 8-bit
default                             →  KVQuant 8-bit
```

### Phase B — Learned MLP classifier (`phase_b.py` + `train_classifier.py`)

A tiny 3-layer MLP (4 → 64 → 64 → 4) trained on calibration data to replace the hand-tuned thresholds.

**Input features:**

| Feature | Formula |
|---|---|
| entropy_normalized | `entropy / log(seq_len)` — already in [0, 1] |
| log1p_seq_len | `log1p(seq_len)` — compresses the long-tail range |
| head_variance | raw — MLP can learn the scale |
| layer_idx_normalized | `layer_idx / (num_layers - 1)` |

**Training data:** `calib100/training_lam*.jsonl` — 2800 rows (100 prompts × 28 layers), three files for λ ∈ {0.1, 1.0, 10.0}.

**Results (5-fold cross-validation, λ=1.0):**

| Class | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| kvquant_8b | 0.672 ± 0.125 | 0.725 ± 0.196 | 0.672 ± 0.113 | stable |
| kvquant_3b | 0.348 ± 0.185 | 0.330 ± 0.211 | 0.322 ± 0.175 | unstable — small dataset |
| dynamickv | 0.833 ± 0.211 | 0.867 ± 0.194 | 0.808 ± 0.132 | best class |
| adakv | 0.000 | 0.000 | 0.000 | needs more long-context prompts |
| **overall accuracy** | | | **0.610 ± 0.074** | vs 0.49 majority baseline |

λ=1.0 is the recommended classifier. λ=10.0 produces 92% accuracy but the problem is near-binary (kvquant_3b dominates); λ=0.1 gives only +5% over majority. See [TRAINING.md](./TRAINING.md) for the full λ analysis.

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/oforomar/kv-cache-adaserve.git
uv sync
```

**Train Phase B (uses pre-built calibration data):**

```bash
# Train all three λ values
for LAM in 0.1 1.0 10.0; do
    python train_classifier.py \
        --data calib100/training_lam${LAM}.jsonl \
        --out  models/phase_b_lam${LAM}.pt \
        --epochs 50 --device cpu
done
```

**Evaluate:**

```bash
python evaluate_classifier.py --data-dir calib100 --model-dir models
python class_report.py        --data-dir calib100 --model-dir models
python kfold_eval.py          --data calib100/training_lam1.0.jsonl --k 5
```

**Runtime inference (Phase B):**

```python
from phase_b import load_phase_b, select_phase_b
from signals import LayerSignals

model, meta = load_phase_b("models/phase_b_lam1.0.pt")

# Called inside a forward hook during prefill
layer_signals = LayerSignals(
    entropy=1.5, entropy_normalized=0.4,
    head_variance=0.03, seq_len=512, layer_idx=5,
)
strategy = select_phase_b(layer_signals, model, num_layers=28)
# → Strategy.KVQUANT_8B  (drop-in replacement for select_phase_a)
```

---

## Data pipeline

The calibration pipeline runs in 5 stages, each writing JSONL and independent of the others:

```
[HF Datasets]
    │
    ▼
prompts.py          → prompts.jsonl          (id, source, bucket, text, target_text)
    │
    ├──▶ collect_signals.py  → signals.jsonl      (per prompt × layer: entropy, head_var, seq_len)
    │
    ├──▶ score_baseline.py   → baseline.jsonl     (per prompt: FP16 perplexity)
    │
    ├──▶ run_kvquant.py  ┐
    ├──▶ run_dynamickv.py├──▶ per-strategy JSONLs  (per prompt: ppl, cratio)
    └──▶ run_adakv.py   ┘
                │
                ▼
         join_labels.py      → measurements.jsonl  (per prompt: argmax label, scores)
                │
                ▼
         make_labels.py      → training_lam*.jsonl  (per prompt × layer: signals + label)
```

Re-running `join_labels.py` with a different `--lambda-compress` sweeps λ without any GPU re-runs.

---

## Calibration target

**Model:** `meta-llama/Llama-3.2-3B` (GQA: 24 Q-heads / 8 KV-heads, 128K context).  
**Hardware:** RTX 3080 Laptop (16 GB VRAM) — max 4096 tokens per prompt.  
**Validation run:** 100 prompts (70 wikitext + 30 alpaca), ~12 min end-to-end.

The validated dataset lives in `calib100/`. The full TARGET_MIX (~1850 prompts) is defined in `calibration/prompts.py` and would take ~10–13 h.

---

## Known limitations

- **adakv F1 = 0** across all folds — the dataset needs more long-context prompts where AdaKV's per-head budget allocation outperforms DynamicKV.
- **τ_head_var is miscalibrated** — Phase A's default `1e-4` causes it to predict Ada-KV for every input. Re-tune against real signals before using Phase A in production.
- **Per-prompt labels** — all 28 layers of a prompt share one label. True per-layer labels would cost ~100× more GPU time but potentially improve the classifier.
- **No Fisher information** in KVQuant calibration pickles — quality is slightly below the paper's `<0.1 ppl` claim at 3-bit.
