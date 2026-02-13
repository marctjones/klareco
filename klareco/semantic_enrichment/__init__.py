"""
AST Semantic Enrichment for Pure Esperanto AI.

This module enriches AST nodes with semantic annotations following the principle:
**Maximize deterministic, minimize learned.**

Architecture:
1. Deterministic feature extraction (0 params) - from correlatives, affixes, grammar
2. Learned semantic gap filling (~5M params) - only when deterministic fails
3. AST enrichment - adds semantic_annotation layer without modifying original AST

Modules:
- taxonomy: Entity type hierarchy (Tier 1/2/3)
- deterministic: Deterministic feature extractor (0 params)
- enricher: AST semantic enricher (coordinates deterministic + learned)

Usage:
    from klareco.semantic_enrichment import ASTSemanticEnricher

    enricher = ASTSemanticEnricher()  # No model = deterministic only
    enriched_ast = enricher.enrich(word_ast, context_ast)

    # Access semantic annotations
    annotation = enriched_ast['semantic_annotation']
    tier3_type = annotation['final_classification']['tier3_type']
"""

from .taxonomy import (
    TopLevelCategory,
    EntityType,
    PersonType,
    LocationType,
    TimeType,
    ThingType,
)
from .deterministic import DeterministicFeatureExtractor
from .enricher import ASTSemanticEnricher

__all__ = [
    'TopLevelCategory',
    'EntityType',
    'PersonType',
    'LocationType',
    'TimeType',
    'ThingType',
    'DeterministicFeatureExtractor',
    'ASTSemanticEnricher',
]
