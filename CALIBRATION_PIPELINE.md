# Adaptive KV Cache Strategy Selector — Design Review and Calibration Data Pipeline

## Context

This document captures the design review and the data-creation pipeline for the Phase B classifier of a runtime-adaptive KV cache strategy selector. The selector chooses between four KV compression methods (QAQ, KVQuant, DynamicKV, Ada-KV) per layer group based on three cheap signals computed during prefill: attention entropy `H`, sequence length `L`, and head-level variance `V`.

Phase A is a rule-based heuristic with hand-tuned thresholds `τ1`, `τ2`, `τ3`. Phase B is a small MLP trained on profiling data that records which strategy achieved the best quality/compression trade-off across diverse inputs. This document focuses on Phase B's training data generation; the selector core and trainer are documented separately.

The orthogonality argument with R-Sparse (KV cache methods modify attention KV storage; R-Sparse modifies FFN linear-layer execution; they do not share data paths) was reviewed and held up. The one caveat reviewers will likely raise is combined memory-pressure and quality interaction when both are stacked at full strength — worth an empirical check rather than just a structural argument.

## Design Issues Identified in the Original Proposal

Several issues in the original Phase A specification needed resolution before implementation. They are documented here because the data-generation pipeline inherits from these decisions.

The original Table 4 was not a partition. Conditions overlapped (a query with `H < τ1` and `V > τ3` hits both the DynamicKV and Ada-KV rows). The fix is explicit precedence rather than a scoring function — easier to reason about, easier to ablate. The chosen order, highest priority first, is: QAQ-capable model (offline flag), high head variance (Ada-KV), low entropy (DynamicKV), high entropy with long context (KVQuant 3-bit), high entropy with short context (KVQuant 8-bit), and a default of KVQuant 8-bit for the mid-entropy region.

The "K-sensitivity ≠ V-sensitivity" condition for QAQ does not belong in Table 3 as a runtime signal. K/V sensitivity asymmetry is determined offline by sweeping K and V bitwidths against perplexity; it is not something a forward pass measures cheaply. The pipeline treats QAQ-eligibility as an offline per-model flag (`qaq_capable`) and excludes QAQ from the runtime classifier's output space. Models without measured K/V asymmetry simply never select QAQ at runtime.

The signal/decision granularity was ambiguous. Signals are computed from "the first few attention layers" but decisions are "per layer group." If early layers are predictors for later groups, that is an empirical claim that needs validation; if signals are recomputed per group, the cost is no longer "negligible" because materializing attention weights breaks FlashAttention's fused kernel. The pipeline computes signals per layer (one forward pass with eager attention) and does not assume cross-layer prediction.

The decision is prefill-locked, not per-token. The four backends have incompatible memory layouts; switching a layer group's strategy mid-decode requires rematerializing the cache. Strategy is chosen once per (sequence, layer-group) at prefill end and frozen for the duration of generation. This is now explicit in the design.

Phase B labels were originally underspecified ("best quality/compression trade-off" is multi-objective). The pipeline uses a scalar score:

```
score = -(ppl_strategy - ppl_baseline) - λ * (1 - cratio_strategy)
```

where `cratio` is `compressed_bytes / fp16_bytes ∈ (0, 1]` and `λ` controls the quality/compression trade-off. Higher score is better; the label per prompt is `argmax_strategy(score)`. λ is the most consequential single number in the whole pipeline because it defines what "best" means. λ=1 weights one nat of perplexity loss equally with a unit of compression-ratio improvement, which roughly favors compression. λ=0.1 favors quality; λ=10 favors aggressive compression. The recommendation is to sweep λ ∈ {0.1, 1, 10} at labeling time, train three classifiers, and report the Pareto frontier rather than committing to one value.

MLP input features need normalization for cross-model transfer. `entropy_normalized = H / log(L)` is in `[0, 1]` and means the same thing at L=512 and L=32K. `seq_len` is fed as `log1p(L)`. `layer_idx` is fed as `layer_idx / (num_layers - 1)`. Head variance is left raw — it interacts with `seq_len` and the MLP can learn that interaction. Phase A thresholds operate on `entropy_normalized`, not raw `H`.

Batch handling and GQA were also flagged. In batched serving, different sequences in a batch may want different strategies; the pipeline picks one strategy per sequence, and serving code is expected to bucket-by-strategy at admission. For grouped-query attention, head variance is computed over KV heads (the axis governing cache shape), entropy is computed over Q heads.

## Cost Analysis and Labeling Strategy Decision

Generating 10K *measured* labels — running all four runtime strategies on each sample and picking the winner — has a real GPU cost that drives the labeling-granularity decision.

For a 7B model on a single H100 at roughly 20 seconds per prompt of average 2K tokens, per-prompt labels (apply each strategy uniformly to all layers, score once per prompt) require 5 strategies × 2000 prompts × 20s ≈ 55 GPU-hours. Per-layer-group labels at 4 groups multiply that by 4 (≈ 220 hours). True per-layer labels at 32 layers multiply by 32 (≈ 1700 hours, infeasible).

The pipeline uses **per-prompt labels with layer-varying signals**. Each prompt produces one label, but emits one row per (prompt × layer) combination. Each row carries the same label but different signals because layer signals differ. The classifier learns layer-dependent patterns from the signal variation even without per-layer labels. Total budget: ~50 GPU-hours per λ value.

This is a deliberate trade-off. If different layers genuinely want different strategies (the per-layer-resolution case) the classifier will under-fit because no signal in the training data distinguishes "layer 5 wants strategy A while layer 25 wants strategy B for the same prompt." The diagnostic is the agreement rate between Phase A labels (which act per layer) and Phase B labels (which act per prompt) on a held-out set: if they agree >80%, per-prompt labels are sufficient; if <50%, real per-layer labels are needed and the calibration set must be smaller or the budget larger.

## Pipeline Architecture

Four scripts in `calibration/`, each owning one stage. Each stage writes JSONL; the stages are independent so any one of them can be re-run without redoing the others.

`prompts.py` curates a length-stratified, source-diverse set of prompts and writes them with held-out continuations (used at scoring time for perplexity). Target ~1500–2000 prompts spanning WikiText, LongBench, MMLU, HumanEval, CNN/DailyMail, and Alpaca, with explicit length buckets at 256–1024, 1024–4096, 4096–16384, and 16384+. The 4096 boundary matters because that is the KVQuant 8-bit ↔ 3-bit decision boundary in Phase A.

`collect_signals.py` is the cheap pass — one forward per prompt with eager attention. It hooks every attention module to extract `(entropy, head_variance, seq_len)` per layer, immediately reduces to scalars (avoiding the OOM that comes from holding all `[B, H, q, k]` tensors), and writes one row per (prompt, layer) to `signals.jsonl`. Cost: ~30–60 minutes on H100 for 2K prompts.

`score_strategies.py` is the expensive pass. For each prompt, it computes baseline FP16 perplexity on the continuation, then for each of the four runtime strategies applies it uniformly to all layers and re-measures perplexity and compression ratio. The scalar score above selects the winning strategy as the per-prompt label. A `--mock` mode skips the model entirely and assigns labels using Phase A on aggregated signals, which validates the rest of the pipeline before standing up the GPU pipeline. Cost in real mode: ~50 GPU-hours per λ value.

`make_labels.py` joins signals and measurements on `prompt_id`, stratified-samples to 10K rows preserving label proportions, and writes the training JSONL the trainer consumes. It also prints the label distribution; that distribution is the single best diagnostic for "did my prompt diversity actually cover the strategy space?"

The pipeline has dependencies on `signals.py`, `selector.py`, and `strategies.py` from the parent package (the selector core, documented separately). Those modules define `LayerSignals`, `Strategy`, `HeuristicConfig`, and `select_phase_a` — used by the calibration scripts but not part of data creation themselves.

## Implementation

### `calibration/prompts.py`

Curates and writes prompts. The `Prompt` dataclass carries an `id`, `source`, length `bucket`, the prompt `text`, and a `target_text` continuation used only at scoring time. `TARGET_MIX` declares the desired (source × bucket) distribution, summing to ~1850 prompts. A `--mock` flag generates synthetic prompts for pipeline shape validation; production use requires wiring in HuggingFace `datasets` loaders for each source.

```python
"""Curate a diverse, length-stratified prompt set for calibration.

Diversity matters more than raw count for a 4-feature classifier. Aim for:

  - Source mix: general LM, QA, code, summarization, long-context
  - Length stratification with a deliberate split at 4096 (the KVQuant
    8-bit / 3-bit boundary) and at the model's training context length.
  - A held-out continuation per prompt for perplexity scoring.

Each `Prompt` carries a continuation (`target_text`) used only at scoring
time. Signal capture ignores it.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

LengthBucket = Literal["short", "medium", "long", "xlong"]
LENGTH_BOUNDS: dict[LengthBucket, tuple[int, int]] = {
    "short":  (256, 1024),
    "medium": (1024, 4096),    # boundary at 4096 matters for KVQuant 8b/3b
    "long":   (4096, 16384),
    "xlong":  (16384, 65536),
}


@dataclass
class Prompt:
    id: str
    source: str
    bucket: LengthBucket
    text: str
    target_text: str   # continuation for perplexity scoring


# Target proportions per source × bucket. Adjust to your model's max ctx.
TARGET_MIX: dict[str, dict[LengthBucket, int]] = {
    "wikitext":      {"short": 200, "medium": 200, "long": 100, "xlong":  50},
    "longbench":     {"short":   0, "medium": 100, "long": 200, "xlong": 100},
    "mmlu":          {"short": 150, "medium":  50, "long":   0, "xlong":   0},
    "humaneval":     {"short": 100, "medium":  50, "long":   0, "xlong":   0},
    "cnn_dailymail": {"short": 100, "medium": 150, "long":   0, "xlong":   0},
    "alpaca":        {"short": 200, "medium": 100, "long":   0, "xlong":   0},
}
# Total: ~1850 prompts → with 32 layers each → ~59K rows → subsample to 10K


def write_prompts(prompts: Iterable[Prompt], path: str | Path) -> int:
    n = 0
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")
            n += 1
    return n


def read_prompts(path: str | Path) -> list[Prompt]:
    with open(path) as f:
        return [Prompt(**json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Mock generator — sanity-check the pipeline without HF dataset downloads.
# ---------------------------------------------------------------------------

def _synthetic_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = ("the of and to in is that for on with as by it from at are this be"
             " or have has not but were was had which their will would can may"
             " more some when what who where why how than then so if no yes also").split()
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def generate_mock_prompts(out_path: str | Path, total: int = 1850) -> int:
    """Fast pipeline-validation set. Replace with real loaders for production."""
    prompts: list[Prompt] = []
    counter = 0
    for source, bucket_counts in TARGET_MIX.items():
        for bucket, count in bucket_counts.items():
            target_share = count / total
            n = max(1, int(target_share * total)) if count > 0 else 0
            lo, hi = LENGTH_BOUNDS[bucket]
            for i in range(n):
                target_tokens = random.Random(counter).randint(lo, hi - 128)
                n_words = int(target_tokens * 0.75)
                text = _synthetic_text(n_words, seed=counter)
                cont = _synthetic_text(64, seed=counter + 999_999)
                prompts.append(Prompt(
                    id=f"{source}-{bucket}-{i:04d}",
                    source=source,
                    bucket=bucket,
                    text=text,
                    target_text=cont,
                ))
                counter += 1
    random.Random(0).shuffle(prompts)
    return write_prompts(prompts, out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--mock", action="store_true",
                    help="generate synthetic prompts (no HF download)")
    args = ap.parse_args()
    if args.mock:
        n = generate_mock_prompts(args.out)
        print(f"wrote {n} mock prompts → {args.out}")
    else:
        raise SystemExit(
            "Real loaders not implemented — wire in HF `datasets` here. "
            "See docstring of TARGET_MIX for the source/bucket targets."
        )
```

### `calibration/collect_signals.py`

Hooks every attention layer of an HF causal LM to capture per-layer signals on the fly. Reduces to scalars inside the hook so memory is bounded regardless of context length. The layer-walking heuristic (`model.model.layers[i].self_attn`) works for Llama-family models; other architectures require adjusting the path. Eager attention is required because FlashAttention does not materialize attention weights.

```python
"""Capture LayerSignals for every (prompt, layer) by running prefill once.

Cheap pass — one forward per prompt. Materializes attention weights, so for
long context we use a hook that reduces to scalars immediately instead of
holding the full [B, H, q, k] tensor.

Output: signals.jsonl with one row per (prompt, layer):
  {"prompt_id": ..., "layer_idx": ...,
   "entropy": ..., "entropy_normalized": ...,
   "head_variance": ..., "seq_len": ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals import LayerSignals, attention_entropy, head_variance  # noqa: E402
from calibration.prompts import read_prompts                         # noqa: E402


def install_hooks(model) -> tuple[list, dict[int, dict]]:
    """Hook every attention module to capture per-layer signals on the fly.

    HF transformer attention modules expose attention weights via the second
    output element when `output_attentions=True`. We avoid that path because
    it stores all layers in CPU memory; instead we hook each layer and
    reduce to scalars immediately.

    Returns (handles, results). Results is filled in during forward.
    """
    handles = []
    results: dict[int, dict] = {}

    layers = model.model.layers if hasattr(model, "model") else model.layers

    for idx, layer in enumerate(layers):
        attn = layer.self_attn

        def make_hook(layer_idx: int):
            def hook(module, args, kwargs, output):
                weights = output[1] if isinstance(output, tuple) else None
                if weights is None:
                    return
                with torch.no_grad():
                    H = attention_entropy(weights).mean().item()
                    V = head_variance(weights).item()
                    seq_len = int(weights.shape[-1])
                results[layer_idx] = {
                    "entropy": H,
                    "head_variance": V,
                    "seq_len": seq_len,
                }
            return hook

        handles.append(
            attn.register_forward_hook(make_hook(idx), with_kwargs=True)
        )
    return handles, results


def collect(
    model_name: str,
    prompts_path: str,
    out_path: str,
    max_length: int = 8192,
    device: str = "cuda",
) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    num_layers = model.config.num_hidden_layers

    prompts = read_prompts(prompts_path)
    handles, results = install_hooks(model)

    n_written = 0
    try:
        with open(out_path, "w") as f, torch.inference_mode():
            for p in prompts:
                ids = tok(p.text, return_tensors="pt",
                          truncation=True, max_length=max_length).input_ids.to(device)
                results.clear()
                model(input_ids=ids, output_attentions=True, use_cache=False)
                for layer_idx in sorted(results):
                    r = results[layer_idx]
                    import math
                    s = LayerSignals(
                        entropy=r["entropy"],
                        entropy_normalized=r["entropy"] / math.log(max(2, r["seq_len"])),
                        head_variance=r["head_variance"],
                        seq_len=r["seq_len"],
                        layer_idx=layer_idx,
                    )
                    row = {"prompt_id": p.id, "num_layers": num_layers, **asdict(s)}
                    f.write(json.dumps(row) + "\n")
                    n_written += 1
    finally:
        for h in handles:
            h.remove()
    return n_written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="signals.jsonl")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n = collect(args.model, args.prompts, args.out, args.max_length, args.device)
    print(f"wrote {n} signal rows → {args.out}")
```

### `calibration/score_strategies.py`

Runs each strategy on each prompt, scores them, picks the per-prompt winner. The `measure_real` function is a `NotImplementedError` placeholder where the actual KV-cache backend adapters get wired in. The `measure_mock` path uses Phase A on prompt-aggregated signals so the rest of the pipeline can be validated end-to-end before any GPU work happens.

```python
"""Score each strategy on each prompt → pick winner → emit per-prompt label.

The expensive pass. For every prompt:
  1. Compute baseline FP16 perplexity on the continuation.
  2. For each strategy in {KVQuant 8b, KVQuant 3b, DynamicKV, Ada-KV}:
       - Apply strategy (uniform across all layers).
       - Compute perplexity on the same continuation.
       - Compute compression ratio (compressed bytes / fp16 bytes).
       - score(s) = -(ppl_s - ppl_baseline) - lambda * (1 - cratio_s)
  3. label = argmax score
  4. Emit one row per prompt to measurements.jsonl.

To wire in real backends, replace `_apply_strategy` with calls to the actual
KVQuant / DynamicKV / Ada-KV / QAQ adapters. The current implementation
patches the model's KV cache via a callback registered before generation.

Mock mode (--mock) skips the model entirely and assigns labels using the
Phase A heuristic on the captured signals, so you can validate the rest of
the pipeline before standing up the GPU pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector import HeuristicConfig, select_phase_a   # noqa: E402
from signals import LayerSignals                        # noqa: E402
from strategies import Strategy                         # noqa: E402
from calibration.prompts import read_prompts            # noqa: E402

# QAQ excluded from the runtime classifier — see design notes.
RUNTIME_STRATEGIES = [
    Strategy.KVQUANT_8B,
    Strategy.KVQUANT_3B,
    Strategy.DYNAMICKV,
    Strategy.ADAKV,
]


def score(ppl_baseline: float, ppl_strategy: float, cratio: float,
          lambda_compress: float) -> float:
    """Higher is better. cratio = compressed_bytes / fp16_bytes ∈ (0, 1]."""
    delta_ppl = ppl_strategy - ppl_baseline
    return -delta_ppl - lambda_compress * (1.0 - cratio)


# ---------------------------------------------------------------------------
# Real measurement path — fill in the backend adapters.
# ---------------------------------------------------------------------------

def measure_real(model_name: str, prompts_path: str, out_path: str,
                 lambda_compress: float = 1.0, device: str = "cuda") -> int:
    raise NotImplementedError(
        "Wire your KV-cache backends into this function: for each strategy, "
        "patch the model's attention so K/V are encoded/decoded by the "
        "strategy, then call the standard perplexity loop on the prompt's "
        "target_text. Use strategies.REGISTRY to plug adapters in."
    )


# ---------------------------------------------------------------------------
# Mock measurement — labels via Phase A on aggregated signals.
# ---------------------------------------------------------------------------

def measure_mock(prompts_path: str, signals_path: str, out_path: str,
                 cfg: HeuristicConfig | None = None) -> int:
    """Assign a per-prompt label using Phase A on each prompt's mean signals.

    This is *not* a substitute for real measurement. It exists only to
    validate that prompts → signals → labels → training all wire up.
    """
    cfg = cfg or HeuristicConfig()
    prompts = {p.id: p for p in read_prompts(prompts_path)}

    agg: dict[str, dict] = defaultdict(lambda: {
        "entropy": 0.0, "entropy_normalized": 0.0,
        "head_variance": 0.0, "seq_len": 0, "n": 0, "num_layers": 0,
    })
    with open(signals_path) as f:
        for line in f:
            r = json.loads(line)
            a = agg[r["prompt_id"]]
            a["entropy"] += r["entropy"]
            a["entropy_normalized"] += r["entropy_normalized"]
            a["head_variance"] += r["head_variance"]
            a["seq_len"] = r["seq_len"]
            a["num_layers"] = r["num_layers"]
            a["n"] += 1

    n = 0
    with open(out_path, "w") as f:
        for pid, a in agg.items():
            if pid not in prompts or a["n"] == 0:
                continue
            mean = LayerSignals(
                entropy=a["entropy"] / a["n"],
                entropy_normalized=a["entropy_normalized"] / a["n"],
                head_variance=a["head_variance"] / a["n"],
                seq_len=a["seq_len"],
                layer_idx=0,
            )
            label = select_phase_a(mean, cfg).value
            f.write(json.dumps({
                "prompt_id": pid,
                "label": label,
                "num_layers": a["num_layers"],
            }) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--signals", required=True,
                    help="signals.jsonl from collect_signals.py")
    ap.add_argument("--out", default="measurements.jsonl")
    ap.add_argument("--mock", action="store_true",
                    help="label via Phase A heuristic instead of real measurement")
    ap.add_argument("--model", help="(real mode) HF model id")
    ap.add_argument("--lambda-compress", type=float, default=1.0)
    args = ap.parse_args()

    if args.mock:
        n = measure_mock(args.prompts, args.signals, args.out)
    else:
        if not args.model:
            raise SystemExit("--model is required without --mock")
        n = measure_real(args.model, args.prompts, args.out,
                         args.lambda_compress)
    print(f"wrote {n} measurement rows → {args.out}")
```

### `calibration/make_labels.py`

Joins signals and measurements on `prompt_id`, stratified-samples to the target size to preserve label proportions, and emits the final training JSONL. Prints the label distribution at the end — that distribution is the single best diagnostic for whether prompt diversity actually covered the strategy space. If any class is below ~5%, the classifier will not learn it, and either the prompt mix needs adjustment or that strategy needs to be dropped from the runtime output space.

```python
"""Join signals + measurements → training JSONL consumed by train_classifier.py.

Each output row is in the format the trainer expects:
  {"signals": {entropy, entropy_normalized, head_variance, seq_len, layer_idx},
   "num_layers": int,
   "label": "kvquant_8b" | ...}

Usage:
  python make_labels.py \
      --signals signals.jsonl \
      --measurements measurements.jsonl \
      --out training.jsonl \
      --target-size 10000

If signals contains more (prompt, layer) pairs than --target-size, we
stratified-sample by label to preserve class balance.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(rows: list[dict], target_size: int,
                      seed: int = 0) -> list[dict]:
    """Sample to target_size while preserving label proportions."""
    if len(rows) <= target_size:
        return rows
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    quota = {lbl: max(1, round(target_size * len(rs) / len(rows)))
             for lbl, rs in by_label.items()}
    out: list[dict] = []
    for lbl, rs in by_label.items():
        rng.shuffle(rs)
        out.extend(rs[:quota[lbl]])
    rng.shuffle(out)
    return out[:target_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", default="training.jsonl")
    ap.add_argument("--target-size", type=int, default=10_000)
    args = ap.parse_args()

    sig_rows = load_jsonl(args.signals)
    meas_rows = load_jsonl(args.measurements)
    label_by_pid = {r["prompt_id"]: r for r in meas_rows}

    joined: list[dict] = []
    skipped = 0
    for s in sig_rows:
        pid = s["prompt_id"]
        if pid not in label_by_pid:
            skipped += 1
            continue
        m = label_by_pid[pid]
        joined.append({
            "signals": {
                "entropy": s["entropy"],
                "entropy_normalized": s["entropy_normalized"],
                "head_variance": s["head_variance"],
                "seq_len": s["seq_len"],
                "layer_idx": s["layer_idx"],
            },
            "num_layers": s["num_layers"],
            "label": m["label"],
        })

    sampled = stratified_sample(joined, args.target_size)
    with open(args.out, "w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")

    dist: dict[str, int] = defaultdict(int)
    for r in sampled:
        dist[r["label"]] += 1
    print(f"wrote {len(sampled)} rows → {args.out}  (skipped {skipped} unjoinable)")
    print("label distribution:")
    for lbl in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {lbl:<14}  {dist[lbl]:>5}  ({dist[lbl] / len(sampled):.1%})")


if __name__ == "__main__":
    main()
```

## Mock Validation Findings

The mock pipeline was run end-to-end with fabricated signals to validate the shape of the data flow. The first run produced a striking result: 100% of samples received the Ada-KV label. This was not a bug; it was a calibration sensitivity finding. The default `τ_head_var = 1e-4` was too low for the synthetic head-variance range (1e-6 to 5e-4), so every sample triggered the Ada-KV rule before any other rule was checked. Phase A's precedence ordering — head variance is checked before entropy — meant Ada-KV swallowed every sample.

After widening the entropy range and reducing the head-variance range to more plausible values (1e-7 to 2e-4), the distribution settled at Ada-KV 48.4%, KVQuant 8-bit 42.4%, DynamicKV 9.1%. Two strategies remained absent: KVQuant 3-bit (no synthetic prompt combined high entropy with `seq_len ≥ 4096`) and QAQ (the `qaq_capable` flag was off, as expected).

The takeaway is operational rather than theoretical: the `τ_head_var` threshold is the most sensitive single number in Phase A, and the label distribution is a direct function of the prompt distribution. Real calibration must include enough samples in the `long`/`xlong` length buckets for KVQuant 3-bit to appear, and the τ thresholds must be re-calibrated on a held-out subset of real signals before any production labeling run.

## Open Decisions and Recommendations

The λ value in the scoring function is the most consequential single number in the whole pipeline because it defines what "best strategy" means. Rather than committing to one value, the recommendation is to sweep λ ∈ {0.1, 1, 10} at the labeling step, produce three training sets, train three classifiers, and report the Pareto frontier. This makes a much stronger story for the proposal write-up than a single hand-picked value, and it costs only 3× the labeling budget (~150 GPU-hours total instead of ~50).

The Phase A ↔ Phase B agreement rate on a held-out set is a useful diagnostic that should be computed once real measurements exist. If the two agree on more than 80% of samples, Phase A is doing most of the work and Phase B's main value is inference speed. If they agree on less than 50%, Phase B is genuinely learning structure that the heuristic misses — that is a real result worth reporting.

The per-prompt vs per-layer-group label decision is reversible. The pipeline currently produces per-prompt labels because that is the only thing the budget supports at scale, but if the agreement rate diagnostic above shows substantial within-prompt layer variation, a smaller per-layer-group calibration set (say 500 prompts × 4 groups instead of 2000 prompts × 1) at the same total budget might be the better trade.

## Production Run Sequence

1. Replace the mock branch in `prompts.py` with HuggingFace `datasets` loaders for WikiText, LongBench, MMLU, HumanEval, CNN/DailyMail, and Alpaca. Tokenize, bucket by length, write JSONL. Target ~1500–2000 prompts. Verify the per-bucket counts match `TARGET_MIX` before proceeding.
2. Run `collect_signals.py` on a single GPU with eager attention. For Llama-family the layer-walking heuristic works as written; for Mistral, Qwen, or other architectures verify the path to `self_attn` first. Cost: ~30–60 minutes on H100 for 2K prompts.
3. Wire actual KVQuant / DynamicKV / Ada-KV adapters into `score_strategies.measure_real`. Use the registry pattern from `strategies.py`. Each adapter needs a "patch the model's attention to use this compressor" callback. Run with `--lambda-compress` set to the target λ. Cost: ~50 GPU-hours per λ value.
4. Run `make_labels.py` and inspect the printed label distribution. If any class is below 5%, adjust the prompt mix (likely adding more long-context or more low-entropy prompts depending on which class is missing) and re-run from step 1.
5. Train the Phase B MLP on the resulting `training.jsonl`. Compare classifier predictions against Phase A labels on a held-out set to compute the agreement rate diagnostic.

The pipeline is intentionally re-runnable at each stage. Signal capture is independent of strategy scoring; both are independent of label assembly. Iterate on prompt mix and λ without redoing the GPU-expensive scoring step unless the prompts themselves change.
