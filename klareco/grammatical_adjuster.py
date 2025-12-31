"""
GrammaticalAdjuster: Deterministic grammatical similarity adjustment.

This module adjusts semantic similarity scores based on grammatical features
extracted from the AST. Unlike a learned model, this uses explicit rules
aligned with Klareco's philosophy of maximizing deterministic processing.

The AST already contains all grammatical annotations:
- negita: True/False (negation)
- tempo: 'pasinteco', 'prezenco', 'futuro' (tense)
- fraztipo: 'deklaro', 'demando', 'ordono' (sentence type)
- modo: 'indikativo', 'kondiĉa', 'vola', 'infinitivo' (mood)

This replaces the learned Stage 2 model with zero parameters.

Usage:
    from klareco import SemanticPipeline
    from klareco.grammatical_adjuster import GrammaticalAdjuster

    pipeline = SemanticPipeline.load()
    adjuster = GrammaticalAdjuster()

    enriched1 = pipeline.for_retrieval("La kato dormas.")
    enriched2 = pipeline.for_retrieval("La kato ne dormas.")

    # Stage 1 similarity (semantic only)
    semantic_sim = cosine_similarity(enriched1.embedding, enriched2.embedding)
    # → ~1.0 (same roots)

    # Adjusted similarity (with grammatical features)
    adjusted_sim = adjuster.adjust(enriched1, enriched2, semantic_sim)
    # → ~-0.8 (negation detected)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class GrammaticalAdjustment:
    """Record of adjustments made to similarity."""
    original_similarity: float
    adjusted_similarity: float
    adjustments: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Original: {self.original_similarity:.3f}"]
        for feature, factor in self.adjustments.items():
            lines.append(f"  {feature}: ×{factor:.2f}")
        lines.append(f"Adjusted: {self.adjusted_similarity:.3f}")
        return "\n".join(lines)


class GrammaticalAdjuster:
    """
    Deterministic grammatical similarity adjustment.

    Uses AST annotations to adjust semantic similarity based on
    grammatical features. No learned parameters - pure rules.
    """

    # Default adjustment factors (can be tuned empirically)
    DEFAULT_ADJUSTMENTS = {
        # Negation: flip polarity
        'negation_mismatch': -0.8,  # Multiply by this (flips sign, scales down)

        # Tense: reduce similarity for temporal mismatch
        'tense_adjacent': 0.8,      # present↔past or present↔future
        'tense_distant': 0.6,       # past↔future (larger gap)

        # Mood: reduce for factuality mismatch
        'mood_mismatch': 0.5,       # indicative↔conditional

        # Sentence type: reduce for illocution mismatch
        'type_mismatch': 0.7,       # statement↔question or statement↔command
    }

    # Tense ordering for distance calculation
    TENSE_ORDER = {
        'pasinteco': 0,   # past
        'prezenco': 1,    # present
        'futuro': 2,      # future
    }

    def __init__(self, adjustments: Optional[Dict[str, float]] = None):
        """
        Initialize with optional custom adjustment factors.

        Args:
            adjustments: Override default adjustment factors
        """
        self.adjustments = self.DEFAULT_ADJUSTMENTS.copy()
        if adjustments:
            self.adjustments.update(adjustments)

    def _get_tense_distance(self, tempo1: Optional[str], tempo2: Optional[str]) -> int:
        """Get temporal distance between two tenses (0, 1, or 2)."""
        if tempo1 is None or tempo2 is None:
            return 0

        order1 = self.TENSE_ORDER.get(tempo1)
        order2 = self.TENSE_ORDER.get(tempo2)

        if order1 is None or order2 is None:
            return 0

        return abs(order1 - order2)

    def adjust(self, ast1, ast2, semantic_similarity: float) -> float:
        """
        Adjust semantic similarity based on grammatical features.

        Args:
            ast1: First EnrichedAST
            ast2: Second EnrichedAST
            semantic_similarity: Stage 1 semantic similarity (-1 to 1)

        Returns:
            Adjusted similarity incorporating grammatical features
        """
        result = self.adjust_with_explanation(ast1, ast2, semantic_similarity)
        return result.adjusted_similarity

    def adjust_with_explanation(
        self, ast1, ast2, semantic_similarity: float
    ) -> GrammaticalAdjustment:
        """
        Adjust similarity and return explanation of adjustments.

        Args:
            ast1: First EnrichedAST
            ast2: Second EnrichedAST
            semantic_similarity: Stage 1 semantic similarity

        Returns:
            GrammaticalAdjustment with original, adjusted, and factors
        """
        sim = semantic_similarity
        applied = {}

        # 1. Negation check (most important - can flip meaning)
        negita1 = getattr(ast1, 'negita', False)
        negita2 = getattr(ast2, 'negita', False)

        if negita1 != negita2:
            factor = self.adjustments['negation_mismatch']
            sim = sim * factor
            applied['negation'] = factor

        # 2. Tense check
        tempo1 = getattr(ast1, 'tempo', None)
        tempo2 = getattr(ast2, 'tempo', None)

        if tempo1 and tempo2 and tempo1 != tempo2:
            distance = self._get_tense_distance(tempo1, tempo2)
            if distance == 1:
                factor = self.adjustments['tense_adjacent']
            else:  # distance == 2
                factor = self.adjustments['tense_distant']
            sim = sim * factor
            applied['tense'] = factor

        # 3. Mood check
        modo1 = getattr(ast1, 'modo', None)
        modo2 = getattr(ast2, 'modo', None)

        if modo1 and modo2 and modo1 != modo2:
            # Only apply if one is indicative and other is conditional
            if {modo1, modo2} & {'kondiĉa', 'indikativo'}:
                factor = self.adjustments['mood_mismatch']
                sim = sim * factor
                applied['mood'] = factor

        # 4. Sentence type check
        tipo1 = getattr(ast1, 'fraztipo', None)
        tipo2 = getattr(ast2, 'fraztipo', None)

        if tipo1 and tipo2 and tipo1 != tipo2:
            factor = self.adjustments['type_mismatch']
            sim = sim * factor
            applied['sentence_type'] = factor

        # Clamp to [-1, 1]
        sim = max(-1.0, min(1.0, sim))

        return GrammaticalAdjustment(
            original_similarity=semantic_similarity,
            adjusted_similarity=sim,
            adjustments=applied,
        )

    def compare(self, ast1, ast2) -> Dict[str, Any]:
        """
        Compare grammatical features of two ASTs.

        Returns dict with feature comparisons (no similarity adjustment).
        """
        return {
            'negation': {
                'ast1': getattr(ast1, 'negita', False),
                'ast2': getattr(ast2, 'negita', False),
                'match': getattr(ast1, 'negita', False) == getattr(ast2, 'negita', False),
            },
            'tense': {
                'ast1': getattr(ast1, 'tempo', None),
                'ast2': getattr(ast2, 'tempo', None),
                'match': getattr(ast1, 'tempo', None) == getattr(ast2, 'tempo', None),
                'distance': self._get_tense_distance(
                    getattr(ast1, 'tempo', None),
                    getattr(ast2, 'tempo', None)
                ),
            },
            'mood': {
                'ast1': getattr(ast1, 'modo', None),
                'ast2': getattr(ast2, 'modo', None),
                'match': getattr(ast1, 'modo', None) == getattr(ast2, 'modo', None),
            },
            'sentence_type': {
                'ast1': getattr(ast1, 'fraztipo', None),
                'ast2': getattr(ast2, 'fraztipo', None),
                'match': getattr(ast1, 'fraztipo', None) == getattr(ast2, 'fraztipo', None),
            },
        }


# Convenience function for quick adjustment
def adjust_similarity(ast1, ast2, semantic_similarity: float) -> float:
    """
    Quick function to adjust similarity using default settings.

    Args:
        ast1: First EnrichedAST
        ast2: Second EnrichedAST
        semantic_similarity: Stage 1 semantic similarity

    Returns:
        Adjusted similarity
    """
    return GrammaticalAdjuster().adjust(ast1, ast2, semantic_similarity)
