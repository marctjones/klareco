"""
Klareco RAG (Retrieval-Augmented Generation) module.

Active components
-----------------
WhooshRetriever      — BM25 + AST-role scoring over the Whoosh full-text index.
ExtractiveAnswerGenerator — Fact extraction → discourse planning → answer text.
UnifiedASTExtractor  — Single entry point for all AST-native fact extraction.

Kuzu was retired 2026-05 (measured: KuzuASTReconstructor ~17 s/AST;
graph traversal ~338x slower than a flat indexed store). Retrieval is
being migrated to a DuckDB store (shredded query columns + ast_json
blob). Until that lands, the Kuzu-dependent retrieval path is disabled.
"""
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.unified_extractor import UnifiedASTExtractor
from klareco.rag.importance_scorer import ImportanceScorer, QuestionType
from klareco.rag.question_classifier import QuestionClassifier
from klareco.rag.entity_recognizer import EntityRecognizer

__all__ = [
    'WhooshRetriever',
    'ExtractiveAnswerGenerator',
    'UnifiedASTExtractor',
    'ImportanceScorer',
    'QuestionType',
    'QuestionClassifier',
    'EntityRecognizer',
]
