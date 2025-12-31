"""
Q&A Module for Klareco.

Provides deterministic answer extraction and reranking for Q&A tasks.
"""

from .extractor import AnswerExtractor, ExtractionResult
from .reranker import DeterministicReranker, RerankResult, rerank_documents

__all__ = [
    'AnswerExtractor', 'ExtractionResult',
    'DeterministicReranker', 'RerankResult', 'rerank_documents',
]
