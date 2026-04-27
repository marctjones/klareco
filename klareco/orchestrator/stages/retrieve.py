"""
RetrieveStage: question AST → ranked passages with pre-built ASTs.

Wraps WhooshRetriever.retrieve_with_ast_roles(), which already handles
question-type routing internally.  The stage converts raw dicts from the
retriever into immutable ParsedPassage objects and populates both the
symbolic layer (passage_asts) and latent layer (passage_embeddings,
if an embedder is available in the ModelRegistry).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from klareco.orchestrator.context import (
    QueryContext, ContextDelta, StageMetrics, ParsedPassage
)
from klareco.orchestrator.stage import PipelineStage, ModelRegistry

if TYPE_CHECKING:
    from klareco.rag.whoosh_retriever import WhooshRetriever

logger = logging.getLogger(__name__)


class RetrieveStage(PipelineStage):
    name = 'retrieve'

    def __init__(
        self,
        retriever: WhooshRetriever,
        models: ModelRegistry,
        top_k: int = 20,
    ):
        self.retriever = retriever
        self.models = models
        self.top_k = top_k

    def should_skip(self, ctx: QueryContext) -> bool:
        return ctx.symbolic.question_ast is None

    def run(self, ctx: QueryContext) -> ContextDelta:
        raw = self.retriever.retrieve_with_ast_roles(
            ctx.symbolic.question_ast, self.top_k
        )

        passages = tuple(
            ParsedPassage(
                sentence_id=r.get('id', ''),
                text=r.get('text', ''),
                ast=r.get('ast'),
                score=float(r.get('score', 0.0)),
                source_doc=r.get('doc_title', ''),
                source_type=r.get('source', 'unknown'),
            )
            for r in raw
        )

        ast_count = sum(1 for p in passages if p.ast is not None)
        symbolic_coverage = ast_count / len(passages) if passages else 0.0
        top_score = passages[0].score if passages else 0.0

        # Rough confidence contribution: capped at +0.4 even for perfect retrieval.
        confidence_gain = min(0.4, top_score / 15.0)
        confidence_after = ctx.confidence + confidence_gain

        symbolic_updates = {'passage_asts': passages}
        latent_updates: dict = {}

        if self.models.has('embedder') and passages:
            embeddings = _embed_passages(self.models.embedder, passages)
            if embeddings is not None:
                latent_updates['passage_embeddings'] = embeddings

        flags: dict = {}
        if not passages:
            flags['retrieval_empty'] = True
            logger.warning("Retrieval returned no passages")

        return ContextDelta(
            symbolic=symbolic_updates,
            latent=latent_updates,
            flags=flags,
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=confidence_after,
                symbolic_coverage=symbolic_coverage,
                stage_specific={
                    'passages_retrieved': len(passages),
                    'ast_hit_rate': round(symbolic_coverage, 3),
                    'top_score': round(top_score, 4),
                },
            ),
        )

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.error(f"[retrieve] failed: {exc}")
        return ContextDelta(flags={'retrieval_empty': True})


def _embed_passages(embedder, passages: tuple) -> tuple | None:
    try:
        import numpy as np
        vecs = tuple(
            embedder.embed_text(p.text) for p in passages
        )
        return vecs
    except Exception as e:
        logger.warning(f"Passage embedding failed: {e}")
        return None
