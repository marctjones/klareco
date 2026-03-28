"""
Klareco Ontology Module

Provides clean API for querying the 4-layer semantic ontology in Kuzu database.

Replaces all hardcoded gazetteers, synonym lists, and pattern matching.

VERSION: v2.2
"""

from .semantic_query import SemanticQuery

__all__ = ['SemanticQuery']
