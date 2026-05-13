"""
PhaseTimer: tiny accumulator for sub-stage timings inside a single
orchestrator stage call.

Used by stages that have non-trivial internal phases (e.g. RetrieveStage,
which inside one call may run multiple Kuzu queries + AST parsing + semantic
ranking) so we can see *which sub-phase* owns the wall time, not just the
stage total.

Usage::

    timer = PhaseTimer()
    with timer.phase("kuzu_query"):
        ...
    with timer.phase("ast_parse"):
        ...
    timer.snapshot()  # {"kuzu_query": 12.3, "ast_parse": 45.6}

Multiple `with` blocks under the same name accumulate. Reset by creating
a new instance — meant to be scoped to one orchestrator stage call.
"""
import time
from contextlib import contextmanager


class PhaseTimer:
    """Per-call accumulator of named-phase timings, in milliseconds."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._totals[name] = self._totals.get(name, 0.0) + elapsed_ms

    def add(self, name: str, ms: float) -> None:
        """Manual add when you measured the time yourself."""
        self._totals[name] = self._totals.get(name, 0.0) + float(ms)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of the accumulated phase totals (rounded)."""
        return {k: round(v, 2) for k, v in self._totals.items()}

    def reset(self) -> None:
        self._totals.clear()
