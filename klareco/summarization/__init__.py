"""
Klareco Summarization Module

Implements deterministic schema-based summarization with citations.

Components:
- schema_classifier: Detect summary type (biographical, definitional, event)
- importance_scorer: Score facts using semantic properties
- fact_selector: Select top facts per schema slot
- citation_tracker: Track source provenance
- synthesizer: Generate coherent text with citations

Architecture: 70% deterministic, 30% learned (future Phase 2)
"""

from .schema_classifier import SchemaClassifier, ClassificationResult
from .importance_scorer import ImportanceScorer, ScoredFact
from .fact_selector import FactSelector, SelectedFact, SchemaSlot
from .citation_tracker import CitationTracker, SourceSentence, Citation, FactWithCitations
from .synthesizer import Synthesizer, Summary
from .retriever import Retriever, RetrievedSentence
from .fact_extractor import FactExtractor
from .discourse_planner import DiscoursePlanner, DiscourseFact

__all__ = [
    'SchemaClassifier',
    'ClassificationResult',
    'ImportanceScorer',
    'ScoredFact',
    'FactSelector',
    'SelectedFact',
    'SchemaSlot',
    'CitationTracker',
    'SourceSentence',
    'Citation',
    'FactWithCitations',
    'Synthesizer',
    'Summary',
    'Retriever',
    'RetrievedSentence',
    'FactExtractor',
    'DiscoursePlanner',
    'DiscourseFact',
]
