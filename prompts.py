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
                # Approximate words-to-tokens at ~0.75
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
