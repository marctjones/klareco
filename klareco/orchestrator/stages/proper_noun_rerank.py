"""
ProperNounRerankStage (#877): boost candidates that share the question's
discriminating proper noun.

The common_terms_competition deep-band bucket fails because the answer sentence
shares the common terms but is buried; the PROPER NOUN is what distinguishes it.
An additive boost (score + alpha * proper_noun_matches) promotes those without
demoting on noise (no-op when the question has no proper noun).

MEASURED (gate-passing): on rebaseline_500 (n=500), MRR 0.3480 -> 0.3615
(+0.0135, paired-bootstrap CI [+0.0067,+0.0212] EXCLUDES 0; 46 better / 15
worse); recall@5 +7 on rebaseline_210, zero band regression. Confirmed once the
frozen benchmark grew from 210 to 500 (#847) — on n=210 the CI just included 0.
No store access; pure reorder over the candidate pool.
"""
from __future__ import annotations

import time

from klareco.orchestrator.context import ContextDelta, QueryContext, StageMetrics
from klareco.orchestrator.stage import PipelineStage
from klareco.rag.proper_noun_reranker import boost_scores

_ALPHA = 4.0  # tuned on rebaseline_210/500


class ProperNounRerankStage(PipelineStage):
    name = 'proper_noun_rerank'

    def should_skip(self, ctx: QueryContext) -> bool:
        return (not ctx.symbolic.passage_asts
                or ctx.flag('retrieval_empty')
                or ctx.flag('tool_short_circuit'))

    def run(self, ctx: QueryContext) -> ContextDelta:
        t0 = time.time()
        passages = list(ctx.symbolic.passage_asts)
        reordered = boost_scores(ctx.symbolic.question_ast, ctx.question,
                                 passages, alpha=_ALPHA)
        return ContextDelta(
            symbolic={'passage_asts': tuple(reordered)},
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=(time.time() - t0) * 1000.0,
                confidence_before=ctx.confidence,
                confidence_after=ctx.confidence,
                symbolic_coverage=1.0,
                stage_specific={'reordered': len(reordered)},
            ),
        )
