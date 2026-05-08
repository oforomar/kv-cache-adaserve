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
from strategies import REGISTRY, Strategy               # noqa: E402
from calibration.prompts import Prompt, read_prompts    # noqa: E402

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
# Real measurement path
# ---------------------------------------------------------------------------

def perplexity(model, tokenizer, prompt_text: str, target_text: str,
               device: str, max_length: int) -> float:
    """Teacher-forced perplexity of `target_text` conditioned on `prompt_text`.

    The prompt portion is masked from the loss (labels = -100), so we measure
    only the model's likelihood on the held-out continuation.
    """
    import torch
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(device)
    target_ids = tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    if input_ids.shape[1] > max_length:
        # Trim the prompt's leading tokens; never trim the target.
        overflow = input_ids.shape[1] - max_length
        input_ids = input_ids[:, overflow:]
        prompt_len = max(0, prompt_ids.shape[1] - overflow)
    else:
        prompt_len = prompt_ids.shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
    return math.exp(out.loss.item())


def measure_real(model_name: str, prompts_path: str, out_path: str,
                 lambda_compress: float = 1.0, device: str = "cuda",
                 max_length: int = 8192) -> int:
    """Run baseline + each strategy on every prompt, pick winner per prompt.

    Each output row carries the per-strategy (ppl, cratio, score) so that the
    same measurements can be relabeled later under a different λ without
    re-running the model — the relabeling is just a `score()` recompute.

    Strategies are applied via `strategies.REGISTRY` adapters: each adapter is
    a context manager that patches the model's K/V handling on enter and
    yields the strategy's compression ratio. Adapters are stubbed by default;
    register real ones before calling this in production.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    num_layers = model.config.num_hidden_layers

    prompts = read_prompts(prompts_path)
    n_written = 0
    with open(out_path, "w") as f:
        for p in prompts:
            ppl_baseline = perplexity(
                model, tok, p.text, p.target_text, device, max_length
            )

            scores: dict[str, dict[str, float]] = {}
            for strat in RUNTIME_STRATEGIES:
                adapter = REGISTRY[strat]
                with adapter(model) as cratio:
                    ppl_s = perplexity(
                        model, tok, p.text, p.target_text, device, max_length
                    )
                scores[strat.value] = {
                    "ppl": ppl_s,
                    "cratio": cratio,
                    "score": score(ppl_baseline, ppl_s, cratio, lambda_compress),
                }

            label = max(scores, key=lambda k: scores[k]["score"])
            f.write(json.dumps({
                "prompt_id": p.id,
                "label": label,
                "ppl_baseline": ppl_baseline,
                "scores": scores,
                "lambda_compress": lambda_compress,
                "num_layers": num_layers,
            }) + "\n")
            n_written += 1
    return n_written


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

    # Aggregate signals per prompt (mean across layers).
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
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.mock:
        n = measure_mock(args.prompts, args.signals, args.out)
    else:
        if not args.model:
            raise SystemExit("--model is required without --mock")
        n = measure_real(args.model, args.prompts, args.out,
                         lambda_compress=args.lambda_compress,
                         device=args.device,
                         max_length=args.max_length)
    print(f"wrote {n} measurement rows → {args.out}")
