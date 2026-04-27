"""
FormatOutputStage: answer segments + citations → final_text.

Assembles the answer text and appends a numbered citation list.  This is the
only stage that produces human-readable Esperanto text — everything upstream
operates on AST-native structures.
"""
from __future__ import annotations

import logging

from klareco.orchestrator.context import QueryContext, ContextDelta, StageMetrics
from klareco.orchestrator.stage import PipelineStage

logger = logging.getLogger(__name__)

_NO_ANSWER = "Pardonu, mi ne trovis sufiĉan informon por respondi vian demandon."


class FormatOutputStage(PipelineStage):
    name = 'format_output'

    def should_skip(self, ctx: QueryContext) -> bool:
        return bool(ctx.flag('abort_pipeline'))

    def run(self, ctx: QueryContext) -> ContextDelta:
        segments = ctx.symbolic.answer_segments
        citations = ctx.symbolic.citations

        if not segments or not any(s.text.strip() for s in segments):
            final_text = _NO_ANSWER
        else:
            body = ' '.join(s.text.strip() for s in segments if s.text.strip())
            if citations:
                ref_lines = [
                    f"[{c.id}] {c.doc_title} — {c.snippet[:80].rstrip()}…"
                    for c in citations
                ]
                final_text = body + '\n\n' + '\n'.join(ref_lines)
            else:
                final_text = body

        confidence_after = ctx.confidence if segments else 0.0

        return ContextDelta(
            symbolic={'final_text': final_text},
            metrics=StageMetrics(
                stage_name=self.name,
                timing_ms=0.0,
                confidence_before=ctx.confidence,
                confidence_after=confidence_after,
                symbolic_coverage=1.0,
                stage_specific={
                    'segments': len(segments),
                    'citations': len(citations),
                    'has_answer': bool(segments),
                },
            ),
        )
