"""
MathToolStage: short-circuit when the question is a math expression (#772).

Runs early in the pipeline. If detect_and_evaluate returns a number,
the answer is written to ctx.symbolic.final_text and a flag
'tool_short_circuit' is raised. Downstream stages skip themselves
when they see this flag.
"""
from __future__ import annotations

import logging

from klareco.orchestrator.context import QueryContext, ContextDelta
from klareco.orchestrator.stage import PipelineStage
from klareco.tools.math import detect_and_evaluate

logger = logging.getLogger(__name__)


class MathToolStage(PipelineStage):
    name = 'math_tool'

    def should_skip(self, ctx: QueryContext) -> bool:
        # Skip if a prior stage already short-circuited
        return bool(ctx.flag('tool_short_circuit'))

    def run(self, ctx: QueryContext) -> ContextDelta:
        result = detect_and_evaluate(ctx.question, ctx.symbolic.question_ast)
        if result is None:
            return ContextDelta()
        logger.info(f'[math_tool] short-circuit: {ctx.question!r} → {result}')
        return ContextDelta(
            symbolic={'final_text': result},
            flags={'tool_short_circuit': True, 'math_tool_result': result},
        )

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.warning(f'[math_tool] failed ({exc}); falling through')
        return ContextDelta()
