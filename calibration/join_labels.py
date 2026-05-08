"""Stage 3b: join baseline + per-strategy JSONLs → measurements.jsonl.

Inputs:
  --baseline  baseline.jsonl                # from score_baseline.py
  --strategy  kvquant_8b=path/to/kvquant_8b.jsonl
  --strategy  kvquant_3b=path/to/kvquant_3b.jsonl
  --strategy  dynamickv=path/to/dynamickv.jsonl
  --strategy  adakv=path/to/adakv.jsonl
  --lambda-compress 1.0

Each per-strategy JSONL is one row per prompt:
    {"prompt_id": ..., "ppl": ..., "cratio": ...}

Output: one row per prompt with the per-strategy table and the argmax label:
    {"prompt_id": ..., "label": ...,
     "ppl_baseline": ...,
     "scores": {"kvquant_8b": {"ppl", "cratio", "score"}, ...},
     "lambda_compress": ..., "num_layers": ...}

Per-strategy {ppl, cratio} are preserved so λ can be re-swept post-hoc by
re-running this join with a different `--lambda-compress` and the same
inputs — no model re-runs.

Prompts missing a measurement from any strategy are skipped, with a count
printed at the end. That's the diagnostic: if it's nonzero, one of the
backend runs didn't cover the full prompt set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies import RUNTIME_STRATEGIES, Strategy, score  # noqa: E402


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _index_by_pid(rows: list[dict]) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in rows}


def join(baseline_path: str, strategy_paths: dict[str, str],
         out_path: str, lambda_compress: float,
         allow_partial: bool = False) -> tuple[int, int]:
    baseline = _index_by_pid(_load_jsonl(baseline_path))
    strat_rows = {name: _index_by_pid(_load_jsonl(p))
                  for name, p in strategy_paths.items()}

    needed = {s.value for s in RUNTIME_STRATEGIES}
    missing_strats = needed - set(strat_rows)
    if missing_strats and not allow_partial:
        raise SystemExit(
            f"missing per-strategy JSONLs for: {sorted(missing_strats)}. "
            f"All four runtime strategies must be measured before joining. "
            f"Pass --allow-partial for smoke runs that intentionally cover "
            f"a subset (label argmax will be over present strategies only)."
        )

    # When allow_partial is set, argmax over whichever strategies were given.
    strategies_in_use = [s for s in RUNTIME_STRATEGIES if s.value in strat_rows]

    n_written = 0
    n_skipped = 0
    with open(out_path, "w") as f:
        for pid, b in baseline.items():
            ppl_baseline = b["ppl_baseline"]
            num_layers = b.get("num_layers")
            scores: dict[str, dict[str, float]] = {}
            ok = True
            for s in strategies_in_use:
                row = strat_rows[s.value].get(pid)
                if row is None:
                    ok = False
                    break
                ppl_s = row["ppl"]
                cratio = row["cratio"]
                scores[s.value] = {
                    "ppl": ppl_s,
                    "cratio": cratio,
                    "score": score(ppl_baseline, ppl_s, cratio, lambda_compress),
                }
            if not ok:
                n_skipped += 1
                continue
            label = max(scores, key=lambda k: scores[k]["score"])
            f.write(json.dumps({
                "prompt_id": pid,
                "label": label,
                "ppl_baseline": ppl_baseline,
                "scores": scores,
                "lambda_compress": lambda_compress,
                "num_layers": num_layers,
            }) + "\n")
            n_written += 1
    return n_written, n_skipped


def _parse_strategy_arg(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--strategy expects NAME=PATH, got {s!r}"
        )
    name, path = s.split("=", 1)
    valid = {x.value for x in Strategy}
    if name not in valid:
        raise argparse.ArgumentTypeError(
            f"unknown strategy {name!r}; valid: {sorted(valid)}"
        )
    return name, path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--strategy", action="append", default=[],
                    type=_parse_strategy_arg,
                    help="NAME=PATH; repeat once per runtime strategy")
    ap.add_argument("--out", default="measurements.jsonl")
    ap.add_argument("--lambda-compress", type=float, default=1.0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="Permit a subset of runtime strategies (smoke runs).")
    args = ap.parse_args()

    strategy_paths = dict(args.strategy)
    n, skipped = join(args.baseline, strategy_paths, args.out,
                      args.lambda_compress, allow_partial=args.allow_partial)
    print(f"wrote {n} measurement rows → {args.out}  (skipped {skipped} prompts "
          f"missing from one or more strategies)")
