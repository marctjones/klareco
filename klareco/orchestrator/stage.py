"""
PipelineStage ABC and ModelRegistry.

Every component in the orchestration pipeline implements PipelineStage.
Stages are stateful objects (they hold expensive resources like loaded models
or open index connections) but their run() method behaves as a pure function:
same input context → same output delta, no hidden mutation.

ModelRegistry holds references to optional learned models.  Stages query it
at construction time to decide whether they're active.  Models are never
loaded inside a request — they are injected once at startup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from klareco.orchestrator.context import QueryContext, ContextDelta


@dataclass
class ModelRegistry:
    """
    Container for optional learned models, injected into stages at construction.

    Fields are None until the corresponding model is loaded.  Stages call
    has() to decide whether to activate neural processing paths.
    """
    reranker: Any = None     # future neural passage reranker
    reasoner: Any = None     # future 20M–100M AST reasoning core
    embedder: Any = None     # compositional morpheme embedder

    def has(self, name: str) -> bool:
        return getattr(self, name, None) is not None


class PipelineStage(ABC):
    """
    Base class for all orchestration stages.

    Contract
    --------
    run(ctx)        Pure function: reads ctx, returns a ContextDelta describing
                    what changed.  Must not mutate ctx or any shared state.

    should_skip(ctx) Return True to skip this stage entirely.  The orchestrator
                    records a skipped StageTrace and moves on.  Default: False.

    on_failure(ctx, exc)
                    Called when run() raises.  Default re-raises so the
                    orchestrator propagates the error.  Override to return a
                    safe fallback delta (e.g. empty ContextDelta()) for
                    graceful degradation when a neural model crashes.
    """
    name: str = 'unnamed'

    @abstractmethod
    def run(self, ctx: QueryContext) -> ContextDelta: ...

    def should_skip(self, ctx: QueryContext) -> bool:
        return False

    def on_failure(self, ctx: QueryContext, exc: Exception) -> ContextDelta:
        raise exc
