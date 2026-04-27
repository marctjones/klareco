"""
Klareco orchestration pipeline.

The pipeline runs a fixed sequence of immutable-context stages:
  ParseQuestion → Retrieve → Rerank → ExtractGenerate → FormatOutput

Quick start
-----------
    from klareco.orchestrator import build_default_pipeline

    pipeline = build_default_pipeline(
        whoosh_index_dir='data/indexes/whoosh',
        kuzu_db_path='data/indexes/kuzu_index',
    )
    result = pipeline.answer("Kiu fondis Esperanton?")
    print(result.text)
    result.print_trace()
"""
from klareco.orchestrator.pipeline import Orchestrator, AnswerResult
from klareco.orchestrator.context import (
    QueryContext, ContextDelta,
    SymbolicLayer, LatentLayer,
    ParsedPassage, FactFragment, Segment, CitationRecord,
    StageMetrics,
)
from klareco.orchestrator.metrics import StageTrace
from klareco.orchestrator.stage import PipelineStage, ModelRegistry
from klareco.orchestrator.factory import build_default_pipeline

__all__ = [
    'Orchestrator',
    'AnswerResult',
    'QueryContext',
    'ContextDelta',
    'SymbolicLayer',
    'LatentLayer',
    'ParsedPassage',
    'FactFragment',
    'Segment',
    'CitationRecord',
    'StageMetrics',
    'StageTrace',
    'PipelineStage',
    'ModelRegistry',
    'build_default_pipeline',
]
