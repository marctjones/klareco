"""
PlannerStage: decompose nested questions into a plan, execute, return
the result directly (#771 orchestrator wiring).

Runs after parse + dialog, before retrieve. If decompose() returns a
plan, we execute it and short-circuit. Otherwise it's a single-hop
question and the regular retrieval pipeline handles it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


from klareco.orchestrator.context import QueryContext, ContextDelta
from klareco.orchestrator.dependencies import TableDependency
from klareco.orchestrator.stage import PipelineStage
from klareco.orchestrator.store_view import StoreView
from klareco.planning import decompose, execute

logger = logging.getLogger(__name__)


class PlannerStage(PipelineStage):
    name = 'planner'

    # Loud-failure contract (#884). As of #881 the planner reads entity_facts
    # through the SLOTS adapter over the live TRIPLE schema, so the real
    # requirement is the triple columns. (The stage stays default-off — the
    # facts are too thin/noisy to answer, tracked by #745 — but if enabled it
    # now passes preflight and runs gracefully instead of crashing.)
    REQUIRES = (
        TableDependency('entity_facts',
                        columns=('sid', 'entito', 'rilato', 'valoro'),
                        issue='#745'),
    )

    def __init__(self, store):
        # #885: receive the shared StoreView; never open a private connection.
        self.store = StoreView.coerce(store)

    def _get_conn(self):
        return self.store.connection

    def should_skip(self, ctx: QueryContext) -> bool:
        return (bool(ctx.flag('tool_short_circuit'))
                or ctx.symbolic.question_ast is None)

    def run(self, ctx: QueryContext) -> ContextDelta:
        plan = decompose(ctx.symbolic.question_ast, ctx.question)
        if plan is None:
            return ContextDelta()
        conn = self._get_conn()
        result = execute(plan, conn)
        answer = result.get('result')
        if not answer:
            logger.info(f'[planner] plan returned no result; '
                        f'falling through to retrieval')
            return ContextDelta(flags={'planner_attempted': True})
        # Render the first result (the values are already simple strings)
        if isinstance(answer, list):
            answer_text = answer[0] if answer else ''
        else:
            answer_text = str(answer)
        logger.info(f'[planner] short-circuit: {ctx.question!r} → {answer_text}')
        return ContextDelta(
            symbolic={'final_text': answer_text},
            flags={'tool_short_circuit': True,
                   'planner_result': answer_text,
                   'planner_trace': result.get('trace', [])},
        )

    # on_failure deliberately NOT overridden (#884): this stage is default-off
    # until #881 lands; when explicitly enabled, a failure must be LOUD.
