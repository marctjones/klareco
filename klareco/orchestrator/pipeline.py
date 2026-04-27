"""
Orchestrator: the thin pipeline runner.

Routing intelligence lives in each stage's should_skip(), not here.
The orchestrator's only responsibilities are:
  - sequence stages
  - measure timing
  - apply deltas to produce the next immutable context
  - accumulate the trace
  - surface the final answer
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from klareco.orchestrator.context import (
    QueryContext, ContextDelta, StageMetrics, CitationRecord, Segment
)
from klareco.orchestrator.metrics import StageTrace
from klareco.orchestrator.stage import PipelineStage, ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """The pipeline's output, including full execution trace for inspection."""
    question: str
    text: str                        # final formatted answer (Esperanto)
    citations: tuple                 # tuple[CitationRecord]
    confidence: float
    trace: list                      # list[StageTrace]

    def print_trace(self) -> None:
        for entry in self.trace:
            print(entry.summary())

    @property
    def has_answer(self) -> bool:
        return bool(self.text.strip())


class Orchestrator:
    """
    Runs a fixed sequence of PipelineStages, threading an immutable QueryContext
    through each one via the delta pattern.

    The orchestrator is intentionally dumb about what stages do.  It does not
    branch on question type, answer quality, or model availability — those
    decisions belong inside the stages' should_skip() methods.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        models: Optional[ModelRegistry] = None,
        debug: bool = False,
    ):
        self.stages = stages
        self.models = models or ModelRegistry()
        self.debug = debug

    def answer(self, question: str) -> AnswerResult:
        ctx = QueryContext(question=question)
        trace: list[StageTrace] = []

        for stage in self.stages:
            ctx_before = ctx

            if stage.should_skip(ctx):
                trace.append(StageTrace(
                    stage_name=stage.name,
                    ctx_before=ctx_before,
                    delta=None,
                    metrics=None,
                    skipped=True,
                ))
                logger.debug(f"[{stage.name}] skipped")
                continue

            t0 = time.perf_counter()
            try:
                delta = stage.run(ctx)
            except Exception as exc:
                try:
                    delta = stage.on_failure(ctx, exc)
                except Exception:
                    logger.exception(f"[{stage.name}] unrecoverable failure")
                    raise
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self.debug:
                _validate_delta(delta, stage.name)

            if delta.metrics is None:
                delta.metrics = StageMetrics(
                    stage_name=stage.name,
                    timing_ms=elapsed_ms,
                    confidence_before=ctx.confidence,
                    confidence_after=ctx.confidence,
                    symbolic_coverage=1.0,
                )
            else:
                delta.metrics.timing_ms = elapsed_ms

            trace.append(StageTrace(
                stage_name=stage.name,
                ctx_before=ctx_before,
                delta=delta,
                metrics=delta.metrics,
            ))

            ctx = ctx.apply(delta)
            logger.debug(f"[{stage.name}] {elapsed_ms:.1f}ms  conf={ctx.confidence:.3f}")

            if ctx.flag('abort_pipeline'):
                logger.info(f"Pipeline aborted after [{stage.name}]")
                break

        return AnswerResult(
            question=question,
            text=ctx.symbolic.final_text,
            citations=ctx.symbolic.citations,
            confidence=ctx.confidence,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

_SYMBOLIC_FIELDS = {
    'question_ast', 'question_type', 'passage_asts',
    'fact_fragments', 'answer_segments', 'citations', 'final_text',
}
_LATENT_FIELDS = {
    'question_embedding', 'passage_embeddings',
    'relevance_matrix', 'stage_attention',
}


def _validate_delta(delta: ContextDelta, stage_name: str) -> None:
    bad_sym = set(delta.symbolic) - _SYMBOLIC_FIELDS
    bad_lat = set(delta.latent) - _LATENT_FIELDS
    if bad_sym:
        raise ValueError(f"[{stage_name}] unknown symbolic fields: {bad_sym}")
    if bad_lat:
        raise ValueError(f"[{stage_name}] unknown latent fields: {bad_lat}")
