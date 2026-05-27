"""
DialogStage: resolve Esperanto pronouns to specific entities using
conversational state (#767 orchestrator wiring).

Holds a DialogState across calls. Mutates the state on each turn
(non-pure side-effect, intentional — dialog state is per-conversation).
Caller should construct one pipeline per conversation, OR pass an
external DialogState via the optional `state` argument to run().

This stage modifies ctx.symbolic.question_ast in-place via the delta:
pronoun nodes are replaced with their resolved entity. Subsequent
stages see the resolved AST.
"""
from __future__ import annotations

import logging

from klareco.orchestrator.context import QueryContext, ContextDelta
from klareco.orchestrator.stage import PipelineStage
from klareco.dialog.state import DialogState, resolve_anaphora, update_state

logger = logging.getLogger(__name__)


class DialogStage(PipelineStage):
    name = 'dialog'

    def __init__(self):
        # One DialogState per stage instance. Caller creates one
        # pipeline per conversation.
        self.state = DialogState()

    def should_skip(self, ctx: QueryContext) -> bool:
        return ctx.symbolic.question_ast is None

    def run(self, ctx: QueryContext) -> ContextDelta:
        original = ctx.symbolic.question_ast
        resolved = resolve_anaphora(self.state, original)
        # Always update state after a turn (push named entities from the
        # question itself onto the stack so later turns can reference them)
        update_state(self.state, resolved)
        # If nothing changed (no pronouns resolved), return empty delta
        if resolved is original:
            return ContextDelta()
        return ContextDelta(symbolic={'question_ast': resolved})

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        logger.warning(f'[dialog] failed ({exc}); proceeding with original AST')
        return ContextDelta()
