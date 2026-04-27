"""
RerankStage: re-order retrieved passages using a neural reranker.

Currently a stub: skipped whenever no reranker is loaded in ModelRegistry.
When a reranker is provided, it receives (question_ast, passages, embeddings)
and returns a reordered tuple of ParsedPassage.

Failure is handled gracefully — a crash here means the pipeline continues
with the original BM25-ranked order from RetrieveStage.
"""
from __future__ import annotations

import logging

from klareco.orchestrator.context import QueryContext, ContextDelta, StageMetrics
from klareco.orchestrator.stage import PipelineStage, ModelRegistry

logger = logging.getLogger(__name__)


class RerankStage(PipelineStage):
    name = 'rerank'

    def __init__(self, models: ModelRegistry):
        self.models = models

    def should_skip(self, ctx: QueryContext) -> bool:
        return (
            not self.models.has('reranker')
            or not ctx.symbolic.passage_asts
            or ctx.flag('retrieval_empty')
        )

    def run(self, ctx: QueryContext) -> ContextDelta:
        passages = ctx.symbolic.passage_asts
        embeddings = ctx.latent.passage_embeddings

        reranked = self.models.reranker.rerank(
            question_ast=ctx.symbolic.question_ast,
            passages=passages,
            embeddings=embeddings if embeddings else None,
        )

        return ContextDelta(
            symbolic={'passage_asts': tuple(reranked)},
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=ctx.confidence + 0.1,
                symbolic_coverage=1.0,
                stage_specific={'reranked': len(reranked)},
            ),
        )

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.warning(f"[rerank] failed ({exc}), continuing with BM25 order")
        return ContextDelta()
