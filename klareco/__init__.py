# This file makes the 'klareco' directory a Python package.

from klareco.enriched_ast import EnrichedAST
from klareco.semantic_pipeline import SemanticPipeline, SemanticModel
from klareco.thought_decoder import ThoughtDecoder, DecodedThought

# Active retrievers for hybrid embeddings (128d = 64d linguistic + 64d topical)
# All memory-efficient and handle 4.4M corpus
from klareco.rag import (
    ASTAwareRetriever,
    HNSWSlotRetriever,
    FAISSSlotRetriever,
    HybridFAISSMmapRetriever,
)

__all__ = [
    'EnrichedAST',
    'SemanticPipeline',
    'SemanticModel',
    'ThoughtDecoder',
    'DecodedThought',
    # Active retrievers (hybrid embeddings)
    'ASTAwareRetriever',
    'HNSWSlotRetriever',
    'FAISSSlotRetriever',
    'HybridFAISSMmapRetriever',
]
