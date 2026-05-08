# Phase B classifier training — data + how to use it

Dataset: per-(prompt, layer) signals → one of four KV-cache strategies. Generated end-to-end on this machine by the calibration pipeline (`prompts.py` → `collect_signals.py` → `score_baseline.py` → 4 backend runners → `join_labels.py` → `make_labels.py`). Phase B's classifier consumes `training.jsonl` and learns to predict the per-prompt strategy that maximizes `score = -(Δppl) + λ·(1 - cratio)`.

## Validation run results (2026-05-08)

100-prompt mid-scale run on `meta-llama/Llama-3.2-3B`, max_length=4096, KVQuant pickles built without Fisher info. Source mix: 70 wikitext + 30 alpaca; bucket mix: 50 short + 30 medium + 20 long (no xlong — eager-attention's softmax tensor at q=k=16K won't fit on a 16 GB card). Wall time end-to-end: ~12 minutes. Compares each backend at its paper-default budget: KVQuant abits 4/3, AdaKV `base_capacity=512`, DynamicKV `max_capacity_prompt=512`.

### Per-prompt perplexity (mean Δppl vs FP16 baseline)

| Backend | mean Δppl | typical cratio |
|---|---|---|
| KVQuant 4-bit (`kvquant_8b` label) | +0.10 | 0.250 |
| KVQuant 3-bit (`kvquant_3b` label) | +0.50 | 0.188 |
| DynamicKV @ 512 | +0.02 (baseline-equivalent at this budget) | varies (≈0.13 medium, ≈0.5 short) |
| AdaKV @ 512 | +1.5 | varies (≈0.13 medium, ≈0.5 short) |

### λ sweep label distributions (2800 training rows = 100 prompts × 28 layers)

| | KVQuant 4b | KVQuant 3b | DynamicKV | AdaKV |
|---|---|---|---|---|
| **λ=0.1** (favor quality) | 40% | 14% | 29% | 17% |
| **λ=1.0** (balanced) | 49% | 32% | 17% | 2% |
| **λ=10** (favor compression) | 2% | 78% | 18% | 2% |

All four classes appear at every λ (no collapse to a single class). The **per-bucket structure** is what Phase B has to learn:

```
λ=1.0:
  short  (n=50):  kvquant_8b:33  kvquant_3b:17
  medium (n=30):  kvquant_8b:16  kvquant_3b:12  dynamickv:2
  long   (n=20):  dynamickv:15   kvquant_3b:3   adakv:2

λ=10:
  short  (n=50):  kvquant_3b:48  kvquant_8b:2
  medium (n=30):  kvquant_3b:30
  long   (n=20):  dynamickv:18   adakv:2
```

At λ=10 the long bucket flips to DynamicKV almost universally — uniform quantization wins on shorter contexts; layer-adaptive eviction wins at long context where per-layer attention pattern variance pays off. This is exactly the kind of structure a 4-feature MLP should be able to capture.

## Where the data lives

Per the project `.gitignore`, `*.jsonl` files are not checked in — they're regenerable from the pipeline. The validation run's output sits at:

```
calib100/
├── prompts.jsonl                        100 prompts (id, source, bucket, text, target_text)
├── signals.jsonl                        2800 rows: per-(prompt, layer) signals
├── baseline.jsonl                       100 rows: FP16 perplexity per prompt
├── kvquant_8b.jsonl                     100 rows: KVQuant 4-bit per-prompt {ppl, cratio}
├── kvquant_3b.jsonl                     100 rows: KVQuant 3-bit per-prompt {ppl, cratio}
├── dynamickv.jsonl                      100 rows: DynamicKV per-prompt {ppl, cratio}
├── adakv.jsonl                          100 rows: AdaKV per-prompt {ppl, cratio}
├── measurements_lam0.1.jsonl            100 rows: per-prompt argmax + per-strategy scores
├── measurements_lam1.0.jsonl
├── measurements_lam10.0.jsonl
├── training_lam0.1.jsonl                2800 rows: classifier-ready (signals + label)
├── training_lam1.0.jsonl
└── training_lam10.0.jsonl
```

The trainer consumes one of the `training_lam*.jsonl` files. Three λs → three classifiers; report the Pareto frontier (per the design doc, not one hand-picked value).

## JSONL schema

Each row of `training_lam*.jsonl`:

```json
{
  "signals": {
    "entropy": 1.786,
    "entropy_normalized": 0.319,
    "head_variance": 0.0426,
    "seq_len": 270,
    "layer_idx": 0
  },
  "num_layers": 28,
  "label": "kvquant_8b"
}
```

`label` is one of `kvquant_8b`, `kvquant_3b`, `dynamickv`, `adakv` — the four members of `RUNTIME_STRATEGIES` in `strategies.py`. (`qaq` is the fifth `Strategy` enum value but Phase B doesn't predict it; it's handled offline.)

## Featurization (per design doc)

Four input features for the MLP. Each row's `signals` block maps to a feature vector as:

| feature | source | reason |
|---|---|---|
| `entropy_normalized` | row.signals.entropy_normalized | already in [0, 1]; tokenizer-agnostic |
| `log1p(seq_len)` | `math.log1p(row.signals.seq_len)` | stretches dynamic range; bounded gradient |
| `head_variance` | row.signals.head_variance | left raw — interacts with seq_len, MLP can learn |
| `layer_idx_normalized` | `row.signals.layer_idx / (row.num_layers - 1)` | makes the position comparable across model sizes |

Phase A thresholds (`HeuristicConfig` in `selector.py`) operate on `entropy_normalized` and raw `head_variance` — keep this consistent for the agreement-rate diagnostic.

## Loading into Python (PyTorch)

```python
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader

# Same string→int order as RUNTIME_STRATEGIES in strategies.py.
LABEL2IDX = {
    "kvquant_8b": 0,
    "kvquant_3b": 1,
    "dynamickv":  2,
    "adakv":      3,
}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def featurize(row: dict) -> tuple[list[float], int]:
    s = row["signals"]
    feats = [
        s["entropy_normalized"],
        math.log1p(s["seq_len"]),
        s["head_variance"],
        s["layer_idx"] / max(1, row["num_layers"] - 1),
    ]
    label = LABEL2IDX[row["label"]]
    return feats, label


class CalibDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        feats, label = featurize(self.rows[i])
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Usage:
ds = CalibDataset("calib100/training_lam1.0.jsonl")
loader = DataLoader(ds, batch_size=128, shuffle=True)
```

## Train/val split

Important: **split by `prompt_id`**, not row-randomly. Each prompt contributes 28 rows (one per layer); a random row split would let layers from the same prompt land in both train and val, leaking information. Group by prompt:

```python
import random
prompt_ids = sorted({r["signals"].get("prompt_id") or _infer_pid(r) for r in ds.rows})
# (Or attach prompt_id to each row at make_labels time; current schema strips it.)
```

The current `make_labels.py` does **not** carry `prompt_id` into `training.jsonl`. If you want a clean prompt-level split, the simplest path is to re-emit `training.jsonl` with `prompt_id` included — small change in `calibration/make_labels.py` (joined['prompt_id'] = s['prompt_id']). For the validation run, randomly splitting layers across prompts is acceptable but won't give a true generalization estimate.

## A minimal MLP that fits the four features

The design doc calls for a "small MLP". Reasonable starting point:

```python
import torch.nn as nn

class PhaseBMLP(nn.Module):
    def __init__(self, n_features: int = 4, n_classes: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),     nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = PhaseBMLP()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad(); loss.backward(); optim.step()
```

Class imbalance is real (label distribution is skewed at λ=10 — KVQuant 3-bit dominates). Either weight the loss (`CrossEntropyLoss(weight=...)`) or stratify the sampler.

## Reproducing the dataset

The validation run was generated by:

```bash
# 1. Build calibration pickles once per bitwidth (~12-15 min each, CPU-heavy)
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant_calibrate.py \
    --model meta-llama/Llama-3.2-3B --abits 4 \
    --quantizer-path smoke/quantizers/quantizers_4bit.pickle
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant_calibrate.py \
    --model meta-llama/Llama-3.2-3B --abits 3 \
    --quantizer-path smoke/quantizers/quantizers_3bit.pickle

# 2. Generate prompts (custom subset; full TARGET_MIX is in calibration/prompts.py)
uv run --project backends/runners/dynamickv_env python -c "
import sys, random; sys.path.insert(0, 'calibration')
from prompts import _emit_for_source, write_prompts
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
mix = {'wikitext': {'short': 30, 'medium': 20, 'long': 20},
       'alpaca':   {'short': 20, 'medium': 10}}
all_p = []
for src, t in mix.items():
    for p in _emit_for_source(src, tok, t, target_tokens=128, seed=hash(src) & 0xffffff):
        all_p.append(p)
random.Random(0).shuffle(all_p)
write_prompts(all_p, 'calib100/prompts.jsonl')
"

# 3. Signals + baseline (main env)
uv run python calibration/collect_signals.py --model meta-llama/Llama-3.2-3B \
    --prompts calib100/prompts.jsonl --out calib100/signals.jsonl --max-length 4096
uv run python calibration/score_baseline.py --model meta-llama/Llama-3.2-3B \
    --prompts calib100/prompts.jsonl --out calib100/baseline.jsonl --max-length 4096

# 4. Four backends
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model meta-llama/Llama-3.2-3B --prompts calib100/prompts.jsonl \
    --out calib100/kvquant_8b.jsonl --bitwidth 8 \
    --quantizer-path smoke/quantizers/quantizers_4bit.pickle --max-length 4096
uv run --project backends/runners/kvquant_env python backends/runners/run_kvquant.py \
    --model meta-llama/Llama-3.2-3B --prompts calib100/prompts.jsonl \
    --out calib100/kvquant_3b.jsonl --bitwidth 3 \
    --quantizer-path smoke/quantizers/quantizers_3bit.pickle --max-length 4096
uv run --project backends/runners/dynamickv_env python backends/runners/run_dynamickv.py \
    --model meta-llama/Llama-3.2-3B --prompts calib100/prompts.jsonl \
    --out calib100/dynamickv.jsonl --max-capacity-prompt 512 --max-length 4096
uv run --project backends/runners/adakv_env python backends/runners/run_adakv.py \
    --model meta-llama/Llama-3.2-3B --prompts calib100/prompts.jsonl \
    --out calib100/adakv.jsonl --base-capacity 512 --max-length 4096

# 5. Lambda sweep: join + make_labels three times
for LAM in 0.1 1.0 10.0; do
  uv run python calibration/join_labels.py \
      --baseline calib100/baseline.jsonl \
      --strategy kvquant_8b=calib100/kvquant_8b.jsonl \
      --strategy kvquant_3b=calib100/kvquant_3b.jsonl \
      --strategy dynamickv=calib100/dynamickv.jsonl \
      --strategy adakv=calib100/adakv.jsonl \
      --out "calib100/measurements_lam${LAM}.jsonl" --lambda-compress $LAM
  uv run python calibration/make_labels.py \
      --signals calib100/signals.jsonl \
      --measurements "calib100/measurements_lam${LAM}.jsonl" \
      --out "calib100/training_lam${LAM}.jsonl" --target-size 2800
done
```

For a full TARGET_MIX run on this hardware: skip step 2's custom mix, use `prompts.py --tokenizer ...` against the full `TARGET_MIX` dict (drop xlong by editing `TARGET_MIX` in place — won't fit in 16 GB), expect ~10-13 h end-to-end.

## Open issues that affect the trainer

- **Class imbalance** — at λ=10, KVQuant 3-bit is 78% of labels. Class-weighted loss or stratified sampling matters for model quality.
- **Train/val split needs prompt_id** — `make_labels.py` strips `prompt_id` from training rows. Add it back before splitting, otherwise val leaks layers from train prompts.
- **Calibration pickles were built without Fisher info** — KVQuant accepts `fisher=None`; quality is slightly below upstream's "<0.1 ppl degradation at 3-bit" claim (we saw ~0.5 here). For the production training set, consider running the upstream `gradients/` pipeline once and re-building pickles with `--fisher`.
- **Phase A vs Phase B agreement rate** — diagnostic hasn't been computed yet (no `agreement_rate.py` script). Per the design doc, >80% agreement = per-prompt labels are sufficient; <50% = need real per-layer labels (much bigger budget).
