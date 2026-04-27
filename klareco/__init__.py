"""
Klareco — Pure Esperanto AI.

The public API centres on the orchestration pipeline.  Import
build_default_pipeline to get a ready-to-run Orchestrator, or compose
custom pipelines from the individual stage classes.

Quick start::

    from klareco import build_default_pipeline

    pipeline = build_default_pipeline(
        whoosh_index_dir='data/indexes/whoosh',
        kuzu_db_path='data/indexes/v2.1_kuzu_index_full',
    )
    result = pipeline.answer("Kiu fondis Esperanton?")
    print(result.text)
"""
from klareco.orchestrator import (
    Orchestrator,
    AnswerResult,
    QueryContext,
    ContextDelta,
    SymbolicLayer,
    LatentLayer,
    ParsedPassage,
    FactFragment,
    Segment,
    CitationRecord,
    StageMetrics,
    StageTrace,
    PipelineStage,
    ModelRegistry,
    build_default_pipeline,
)
from klareco.parser import parse
from klareco.deparser import deparse

__all__ = [
    # Pipeline
    'Orchestrator',
    'AnswerResult',
    'build_default_pipeline',
    # Context types
    'QueryContext',
    'ContextDelta',
    'SymbolicLayer',
    'LatentLayer',
    # Leaf types
    'ParsedPassage',
    'FactFragment',
    'Segment',
    'CitationRecord',
    'StageMetrics',
    'StageTrace',
    # Stage authoring
    'PipelineStage',
    'ModelRegistry',
    # Core language tools
    'parse',
    'deparse',
]
