"""
StageTrace: the complete record of one stage's execution.

Stored as (ctx_before, delta) so the full history can be replayed from any
point without storing redundant context copies.  ctx_after is computed on
demand via the ctx_before.apply(delta) round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from klareco.orchestrator.context import QueryContext, ContextDelta, StageMetrics


@dataclass
class StageTrace:
    """One entry in the orchestrator's execution trace."""
    stage_name: str
    ctx_before: QueryContext        # immutable snapshot before the stage ran
    delta: Optional[ContextDelta]   # None when the stage was skipped
    metrics: Optional[StageMetrics]
    skipped: bool = False

    @property
    def ctx_after(self) -> QueryContext:
        """Reconstruct the post-stage context by replaying the delta."""
        if self.delta is None:
            return self.ctx_before
        return self.ctx_before.apply(self.delta)

    def summary(self) -> str:
        if self.skipped:
            return f"[{self.stage_name}] SKIPPED"
        m = self.metrics
        if m is None:
            return f"[{self.stage_name}] no metrics"
        return (
            f"[{self.stage_name}] "
            f"{m.timing_ms:.1f}ms  "
            f"conf {m.confidence_before:.2f}→{m.confidence_after:.2f}  "
            f"sym_cov={m.symbolic_coverage:.2f}"
        )
