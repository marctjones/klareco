# RAG (Retrieval-Augmented Generation) module
#
# AST-First Retrieval Architecture using Kuzu Graph Database:
# - KuzuInvertedIndex: Graph-backed retrieval with integrated semantic relations
# - ASTAwareRetriever: High-level API with question classification
#
# Features:
# - Root-based inverted index (O(1) lookup)
# - Transitive synonym expansion via graph traversal
# - Hypernym chain traversal
# - Grammar-aware scoring
# - Sentence context retrieval
#
# The retriever is PURE DETERMINISTIC by default. A/B testing showed
# that deterministic lookup has equal recall with lower latency than
# embedding-based fallbacks (see issue #246).

from klareco.rag.kuzu_inverted_index import (
    KuzuInvertedIndex,
    FallbackMode,
    RetrievalStats,
    SearchResult,
)
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.semantic_db import SemanticRelationDB
from klareco.rag.entity_recognizer import EntityRecognizer
from klareco.rag.question_classifier import QuestionClassifier
from klareco.rag.ast_pattern_matcher import ASTPatternMatcher

__all__ = [
    # Core retrieval - Kuzu backend
    'KuzuInvertedIndex',
    'FallbackMode',
    'RetrievalStats',
    'SearchResult',
    # High-level API
    'ASTAwareRetriever',
    # Supporting components
    'SemanticRelationDB',
    'EntityRecognizer',
    'QuestionClassifier',
    'ASTPatternMatcher',
]
