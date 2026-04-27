"""
Klareco RAG (Retrieval-Augmented Generation) module.

Active components
-----------------
WhooshRetriever      — BM25 + AST-role scoring over the Whoosh full-text index.
ExtractiveAnswerGenerator — Fact extraction → discourse planning → answer text.
UnifiedASTExtractor  — Single entry point for all AST-native fact extraction.

The retrieval backend is Whoosh (full-text) combined with Kuzu (graph DB) for
pre-built AST storage.  KuzuASTReconstructor reads sentence ASTs from the
graph at <5ms, avoiding runtime re-parsing.
"""
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.unified_extractor import UnifiedASTExtractor
from klareco.rag.importance_scorer import ImportanceScorer, QuestionType
from klareco.rag.question_classifier import QuestionClassifier
from klareco.rag.entity_recognizer import EntityRecognizer
from klareco.rag.kuzu_ast_reconstructor import KuzuASTReconstructor

__all__ = [
    'WhooshRetriever',
    'ExtractiveAnswerGenerator',
    'UnifiedASTExtractor',
    'ImportanceScorer',
    'QuestionType',
    'QuestionClassifier',
    'EntityRecognizer',
    'KuzuASTReconstructor',
]
