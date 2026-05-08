"""Curate a diverse, length-stratified prompt set for calibration.

Diversity matters more than raw count for a 4-feature classifier. Aim for:

  - Source mix: general LM, QA, code, summarization, long-context
  - Length stratification with a deliberate split at 4096 (the KVQuant
    8-bit / 3-bit boundary) and at the model's training context length.
  - A held-out continuation per prompt for perplexity scoring.

Each `Prompt` carries a continuation (`target_text`) used only at scoring
time. Signal capture ignores it.

Real loaders pool tokenized text per source and slice (prompt, target)
pairs honoring TARGET_MIX. xlong prompts are produced by concatenating
adjacent passages — neither WikiText nor LongBench has native 16K-65K
single examples in their default splits.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

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


# ---------------------------------------------------------------------------
# Real loaders — pool tokenized text from HF datasets, slice into buckets.
# ---------------------------------------------------------------------------

# Continuation length for perplexity scoring (target_text).
TARGET_TOKENS = 128


def _source_iter(source: str) -> Iterator[str]:
    """Yield raw text snippets from one of the six known sources via HF datasets.

    Each yielded string can be tokenized and concatenated with its neighbours;
    the pooler turns those tokens into prompts of the right bucket length.
    """
    from datasets import load_dataset

    if source == "wikitext":
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            split="train", streaming=True,
        )
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if t:
                yield t

    elif source == "longbench":
        # LongBench is a suite of long-context tasks. Cycle through a
        # length-diverse subset; merge `context` + `input` into a single
        # passage. Subtasks unavailable on the user's HF account are
        # skipped with a stderr warning.
        subtasks = ("qasper", "narrativeqa", "multifieldqa_en",
                    "hotpotqa", "2wikimqa", "qmsum", "multi_news")
        iters = []
        for sub in subtasks:
            try:
                iters.append((sub, iter(load_dataset(
                    "THUDM/LongBench", sub, split="test", trust_remote_code=True
                ))))
            except Exception as e:
                print(f"  longbench/{sub} unavailable: {e}", file=sys.stderr)
        active = list(range(len(iters)))
        while active:
            for i in list(active):
                try:
                    ex = next(iters[i][1])
                except StopIteration:
                    active.remove(i)
                    continue
                parts = [ex.get("context") or "", ex.get("input") or ""]
                merged = "\n\n".join(p for p in parts if p)
                if merged:
                    yield merged

    elif source == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        for ex in ds:
            choices = ex.get("choices") or []
            yield ex.get("question", "") + "\n" + "\n".join(
                f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
            )

    elif source == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        for ex in ds:
            yield ex.get("prompt", "")

    elif source == "cnn_dailymail":
        ds = load_dataset(
            "abisee/cnn_dailymail", "3.0.0",
            split="train", streaming=True,
        )
        for ex in ds:
            article = ex.get("article") or ""
            if article.strip():
                yield article

    elif source == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for ex in ds:
            parts = [ex.get("instruction") or "",
                     ex.get("input") or "",
                     ex.get("output") or ""]
            merged = "\n".join(p for p in parts if p)
            if merged:
                yield merged

    else:
        raise ValueError(f"unknown source: {source!r}")


def _emit_for_source(
    source: str,
    tokenizer,
    bucket_targets: dict[LengthBucket, int],
    target_tokens: int,
    seed: int,
) -> Iterator[Prompt]:
    """Pool tokenized text from `source`; emit Prompts honoring bucket targets.

    Strategy: keep a token buffer; whenever it can satisfy the smallest
    still-unfilled bucket, slice off (prompt, target) and emit. Smaller
    buckets are filled first (they need less buffer); xlong fills last
    because it requires the longest concatenation.
    """
    counts: dict[LengthBucket, int] = {b: 0 for b in bucket_targets}
    rng = random.Random(seed)

    def remaining() -> dict[LengthBucket, int]:
        return {b: bucket_targets[b] - counts[b]
                for b in bucket_targets if counts[b] < bucket_targets[b]}

    # Bucket order: short → xlong, fill smallest-first.
    bucket_order: tuple[LengthBucket, ...] = ("short", "medium", "long", "xlong")

    buf: list[int] = []
    for text in _source_iter(source):
        if not remaining():
            return
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        buf.extend(ids)

        while True:
            rem = remaining()
            if not rem:
                return
            chosen: LengthBucket | None = None
            for b in bucket_order:
                if b not in rem:
                    continue
                lo, hi = LENGTH_BOUNDS[b]
                if len(buf) >= lo + target_tokens:
                    chosen = b
                    break
            if chosen is None:
                break  # need more text for any unfilled bucket

            lo, hi = LENGTH_BOUNDS[chosen]
            max_prompt = min(hi - target_tokens, len(buf) - target_tokens)
            prompt_len = rng.randint(lo, max_prompt)

            prompt_ids = buf[:prompt_len]
            target_ids = buf[prompt_len:prompt_len + target_tokens]

            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

            idx = counts[chosen]
            yield Prompt(
                id=f"{source}-{chosen}-{idx:04d}",
                source=source,
                bucket=chosen,
                text=prompt_text,
                target_text=target_text,
            )
            counts[chosen] += 1
            buf = buf[prompt_len + target_tokens:]

    # Source exhausted before all buckets filled — warn loudly.
    leftover = remaining()
    if leftover:
        print(f"  warning: source {source!r} exhausted; missing "
              f"{leftover} (got {counts})", file=sys.stderr)


def generate_real_prompts(
    out_path: str | Path,
    tokenizer_id: str,
    target_tokens: int = TARGET_TOKENS,
    seed: int = 0,
) -> int:
    """Build the full TARGET_MIX from real HF datasets and write JSONL.

    `tokenizer_id` should match the model used for signal collection — the
    bucket lengths (256, 1024, 4096, 16384) are token counts, so different
    tokenizers produce different prompts for the same TARGET_MIX.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    all_prompts: list[Prompt] = []
    for src_seed, (source, bucket_targets) in enumerate(TARGET_MIX.items()):
        targets = {b: c for b, c in bucket_targets.items() if c > 0}
        if not targets:
            continue
        print(f"loading {source}: {targets}", file=sys.stderr)
        for p in _emit_for_source(source, tok, targets, target_tokens,
                                   seed=seed + src_seed):
            all_prompts.append(p)

    random.Random(seed).shuffle(all_prompts)
    return write_prompts(all_prompts, out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--mock", action="store_true",
                    help="generate synthetic prompts (no HF download)")
    ap.add_argument("--tokenizer",
                    help="HF tokenizer id; required without --mock. "
                         "Should match the model used downstream.")
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                    help="Length of target_text for perplexity scoring.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mock:
        n = generate_mock_prompts(args.out)
        print(f"wrote {n} mock prompts → {args.out}")
    else:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required without --mock")
        n = generate_real_prompts(args.out, args.tokenizer,
                                  args.target_tokens, args.seed)
        print(f"wrote {n} real prompts → {args.out}")
