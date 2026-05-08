"""KV-cache strategy enum + adapter registry.

An adapter is a context manager: entering it patches the model's KV cache
to use the strategy's compressor and yields the compression ratio
(`compressed_bytes / fp16_bytes ∈ (0, 1]`); exiting unpatches.

```python
with REGISTRY[Strategy.KVQUANT_8B](model) as cratio:
    ppl_s = perplexity(model, ...)
    score_s = score(ppl_baseline, ppl_s, cratio, lambda_compress)
```

The four runtime strategies are stubbed — they raise NotImplementedError on
enter. Replace with real backend adapters before the production scoring run
(see `score_strategies.measure_real`). QAQ has an adapter slot too, but
runtime selection excludes it (see `score_strategies.RUNTIME_STRATEGIES`).
"""
from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, ContextManager, Iterator


class Strategy(str, Enum):
    KVQUANT_8B = "kvquant_8b"
    KVQUANT_3B = "kvquant_3b"
    DYNAMICKV = "dynamickv"
    ADAKV = "adakv"
    QAQ = "qaq"


# Adapter signature: callable(model) -> ContextManager[float], where the
# yielded float is the strategy's measured compression ratio for this model.
Adapter = Callable[[Any], ContextManager[float]]

REGISTRY: dict[Strategy, Adapter] = {}


def register(strategy: Strategy, adapter: Adapter) -> None:
    REGISTRY[strategy] = adapter


def _stub_adapter(name: str) -> Adapter:
    @contextmanager
    def adapter(model: Any) -> Iterator[float]:
        raise NotImplementedError(
            f"Backend adapter for {name!r} is not wired up. "
            f"Implement it as a context manager that patches the model's "
            f"K/V handling, yields the compression ratio, and unpatches on "
            f"exit. Register via strategies.register(Strategy.{name.upper()}, "
            f"your_adapter)."
        )
        yield 0.0  # unreachable; satisfies the contextmanager protocol
    return adapter


for _s in Strategy:
    register(_s, _stub_adapter(_s.value))
