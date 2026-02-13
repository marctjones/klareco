"""
AST Semantic Enricher - Orchestrates deterministic + learned features.

This module implements the "Pure Esperanto AI" philosophy:
1. Try deterministic extraction first (0 params)
2. Only invoke learned model if deterministic fails (confidence < 0.90)
3. Merge features into semantic_annotation layer
4. Preserve original AST structure

Architecture:
    word_ast + context_ast
           ↓
    Deterministic Extractor (0 params)
           ↓
    Fully determined? → Yes → Return enriched AST
           ↓ No
    Learned Model (~5M params)
           ↓
    Merge features → Return enriched AST

Output: Original AST + semantic_annotation layer
"""

from typing import Dict, Optional
import logging
from copy import deepcopy

from .deterministic import DeterministicFeatureExtractor
from .taxonomy import TopLevelCategory, EntityType

logger = logging.getLogger(__name__)


class ASTSemanticEnricher:
    """
    Enrich AST nodes with semantic annotations.

    Philosophy: Maximize deterministic, minimize learned.

    The enricher adds a semantic_annotation layer to AST nodes without
    modifying the original structure. This allows downstream components
    to access both grammatical (deterministic) and semantic (learned)
    information.
    """

    def __init__(self, model=None):
        """
        Initialize AST semantic enricher.

        Args:
            model: Optional learned semantic model (~5M params).
                   If None, only deterministic features are extracted.
        """
        self.deterministic_extractor = DeterministicFeatureExtractor()
        self.model = model  # Will be a trained GNN model (Task 1.5)

        logger.info(
            f"ASTSemanticEnricher initialized "
            f"(deterministic={0} params, learned={self._get_model_params()} params)"
        )

    def _get_model_params(self) -> int:
        """Get number of learned parameters in model."""
        if self.model is None:
            return 0
        # Will implement when model exists
        return 5_000_000  # Target ~5M params

    def enrich(
        self,
        word_ast: Dict,
        context_ast: Optional[Dict] = None
    ) -> Dict:
        """
        Add semantic annotations to AST node.

        Args:
            word_ast: AST node for word to classify
            context_ast: Optional surrounding context
                {
                    'before': [ast1, ast2, ...],  # ±3 words before
                    'after': [ast3, ast4, ...],   # ±3 words after
                    'sentence': {...},             # Full sentence AST
                    'position': 'mid_sentence' | 'sentence_initial' | 'sentence_final'
                }

        Returns:
            Enriched AST with semantic_annotation layer added:
            {
                # UNCHANGED: Original AST structure
                'tipo': 'vorto',
                'radiko': '...',
                'sufiksoj': [...],
                'vortspeco': '...',
                ...

                # NEW: Semantic enrichment layer
                'semantic_annotation': {
                    'deterministic_features': {...},
                    'learned_features': {...},  # None if not needed
                    'final_classification': {...},
                    'explanation': {...}
                }
            }
        """
        # Validate input
        if not word_ast:
            logger.warning("Empty word_ast provided to enricher")
            return self._create_error_annotation(word_ast, "empty_input")

        # STEP 1: Deterministic feature extraction (0 params)
        det_features = self.deterministic_extractor.extract(word_ast, context_ast)

        # STEP 2: Check if deterministic is sufficient
        if det_features['is_fully_determined']:
            logger.debug(
                f"Word '{word_ast.get('teksto', '?')}' fully determined by "
                f"deterministic features (confidence={det_features['confidence']:.2f})"
            )
            return self._create_deterministic_annotation(word_ast, det_features)

        # STEP 3: Invoke learned model for semantic gap
        if self.model is None:
            logger.debug(
                f"Word '{word_ast.get('teksto', '?')}' needs model but none loaded "
                f"(deterministic confidence={det_features['confidence']:.2f})"
            )
            return self._create_partial_annotation(word_ast, det_features)

        # STEP 4: Get learned features from model
        learned_features = self._extract_learned_features(
            word_ast,
            context_ast,
            det_features
        )

        # STEP 5: Merge deterministic + learned
        return self._create_merged_annotation(word_ast, det_features, learned_features)

    def _extract_learned_features(
        self,
        word_ast: Dict,
        context_ast: Optional[Dict],
        det_features: Dict
    ) -> Dict:
        """
        Extract learned semantic features using model.

        Args:
            word_ast: AST node
            context_ast: Surrounding context
            det_features: Deterministic features (used as priors)

        Returns:
            Learned features:
            {
                'tier3_type': PersonType.PERSON_NAME,  # Fine-grained type
                'specificity': 'SPECIFIC',              # vs GENERIC
                'referentiality': 'REFERENTIAL',        # vs DESCRIPTIVE
                'confidence': 0.92,
                'type_distribution': {...},             # Probability distribution
                'evidence': {...}                       # What model used
            }
        """
        # TODO: Implement when model is trained (Task 1.5)
        # For now, return stub
        logger.debug(
            f"Model inference stub called for '{word_ast.get('teksto', '?')}' "
            f"(will implement in Task 1.5)"
        )

        return {
            'tier3_type': None,
            'specificity': 'UNKNOWN',
            'referentiality': 'UNKNOWN',
            'confidence': 0.0,
            'type_distribution': {},
            'evidence': {'model': 'not_yet_implemented'}
        }

    def _create_deterministic_annotation(
        self,
        word_ast: Dict,
        det_features: Dict
    ) -> Dict:
        """
        Create enriched AST with only deterministic features.

        Used when deterministic extraction is sufficient (confidence >= 0.90).
        """
        enriched = deepcopy(word_ast)

        enriched['semantic_annotation'] = {
            'deterministic_features': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                'confidence': det_features['confidence'],
                'evidence': det_features['evidence']
            },
            'learned_features': None,  # Not needed!
            'final_classification': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                'confidence': det_features['confidence'],
                'source': 'deterministic_only'
            },
            'explanation': {
                'deterministic_reasoning': det_features['reasoning'],
                'learned_reasoning': None,
                'decision_source': 'deterministic_only',
                'model_invoked': False,
                'confidence_breakdown': {
                    'deterministic': det_features['confidence'],
                    'learned': 0.0
                }
            }
        }

        return enriched

    def _create_partial_annotation(
        self,
        word_ast: Dict,
        det_features: Dict
    ) -> Dict:
        """
        Create enriched AST with only deterministic features (no model available).

        Used when word needs model but model is not loaded.
        Flags the annotation as incomplete.
        """
        enriched = deepcopy(word_ast)

        enriched['semantic_annotation'] = {
            'deterministic_features': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                'confidence': det_features['confidence'],
                'evidence': det_features['evidence']
            },
            'learned_features': None,  # Model not available
            'final_classification': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                'confidence': det_features['confidence'],
                'source': 'deterministic_partial',
                'needs_model': True  # Flag for incomplete annotation
            },
            'explanation': {
                'deterministic_reasoning': det_features['reasoning'],
                'learned_reasoning': None,
                'decision_source': 'deterministic_partial',
                'model_invoked': False,
                'warning': 'Model needed but not available',
                'confidence_breakdown': {
                    'deterministic': det_features['confidence'],
                    'learned': 0.0
                }
            }
        }

        return enriched

    def _create_merged_annotation(
        self,
        word_ast: Dict,
        det_features: Dict,
        learned_features: Dict
    ) -> Dict:
        """
        Create enriched AST with both deterministic and learned features.

        Used when deterministic is insufficient and model provides additional info.
        """
        enriched = deepcopy(word_ast)

        # Merge tier3 type (learned overrides if more confident)
        final_tier3 = learned_features['tier3_type'] if learned_features['confidence'] > det_features['confidence'] else det_features.get('tier3_type')

        # Combined confidence (weighted average if both contribute)
        if det_features['confidence'] > 0 and learned_features['confidence'] > 0:
            # Weight: 60% learned, 40% deterministic (learned is more specific)
            final_confidence = (
                learned_features['confidence'] * 0.6 +
                det_features['confidence'] * 0.4
            )
        else:
            final_confidence = max(det_features['confidence'], learned_features['confidence'])

        enriched['semantic_annotation'] = {
            'deterministic_features': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': det_features['tier3_type'].value if det_features['tier3_type'] else None,
                'confidence': det_features['confidence'],
                'evidence': det_features['evidence']
            },
            'learned_features': {
                'tier3_type': learned_features['tier3_type'].value if hasattr(learned_features['tier3_type'], 'value') else None,
                'specificity': learned_features['specificity'],
                'referentiality': learned_features['referentiality'],
                'confidence': learned_features['confidence'],
                'type_distribution': learned_features['type_distribution'],
                'evidence': learned_features['evidence']
            },
            'final_classification': {
                'tier1_category': det_features['tier1_category'].value if det_features['tier1_category'] else None,
                'tier2_type': det_features['tier2_type'].value if det_features['tier2_type'] else None,
                'tier3_type': final_tier3.value if hasattr(final_tier3, 'value') else None,
                'specificity': learned_features['specificity'],
                'referentiality': learned_features['referentiality'],
                'confidence': final_confidence,
                'source': 'deterministic_and_learned'
            },
            'explanation': {
                'deterministic_reasoning': det_features['reasoning'],
                'learned_reasoning': f"Model classified as {learned_features['tier3_type']} (confidence={learned_features['confidence']:.2f})",
                'decision_source': 'merged',
                'model_invoked': True,
                'confidence_breakdown': {
                    'deterministic': det_features['confidence'],
                    'learned': learned_features['confidence'],
                    'final': final_confidence
                }
            }
        }

        return enriched

    def _create_error_annotation(
        self,
        word_ast: Dict,
        error_type: str
    ) -> Dict:
        """Create enriched AST with error annotation."""
        enriched = deepcopy(word_ast) if word_ast else {}

        enriched['semantic_annotation'] = {
            'deterministic_features': None,
            'learned_features': None,
            'final_classification': {
                'tier1_category': None,
                'tier2_type': None,
                'tier3_type': None,
                'confidence': 0.0,
                'source': 'error',
                'error': error_type
            },
            'explanation': {
                'deterministic_reasoning': None,
                'learned_reasoning': None,
                'decision_source': 'error',
                'model_invoked': False,
                'error': error_type
            }
        }

        return enriched

    def enrich_batch(
        self,
        word_asts: list,
        context_asts: Optional[list] = None
    ) -> list:
        """
        Enrich multiple AST nodes in batch.

        Args:
            word_asts: List of word AST nodes
            context_asts: Optional list of context ASTs (same length)

        Returns:
            List of enriched ASTs
        """
        if context_asts is None:
            context_asts = [None] * len(word_asts)

        enriched_asts = []
        for word_ast, context_ast in zip(word_asts, context_asts):
            enriched = self.enrich(word_ast, context_ast)
            enriched_asts.append(enriched)

        return enriched_asts

    def get_enrichment_stats(self, enriched_asts: list) -> Dict:
        """
        Analyze enrichment statistics on a batch.

        Args:
            enriched_asts: List of enriched ASTs

        Returns:
            Statistics:
            {
                'total': int,
                'deterministic_only': int,    # Fully determined by rules
                'needs_model': int,            # Flagged as needing model
                'deterministic_and_learned': int,  # Model invoked
                'errors': int,
                'deterministic_coverage': float,  # % fully determined
                'avg_confidence': float
            }
        """
        total = len(enriched_asts)
        deterministic_only = 0
        needs_model = 0
        deterministic_and_learned = 0
        errors = 0
        total_confidence = 0.0

        for ast in enriched_asts:
            if 'semantic_annotation' not in ast:
                errors += 1
                continue

            annot = ast['semantic_annotation']
            final = annot.get('final_classification', {})
            source = final.get('source', 'unknown')

            if source == 'deterministic_only':
                deterministic_only += 1
            elif source == 'deterministic_partial':
                needs_model += 1
            elif source == 'deterministic_and_learned':
                deterministic_and_learned += 1
            elif source == 'error':
                errors += 1

            total_confidence += final.get('confidence', 0.0)

        valid_total = total - errors
        avg_confidence = (total_confidence / valid_total) if valid_total > 0 else 0.0
        deterministic_coverage = (deterministic_only / valid_total * 100) if valid_total > 0 else 0.0

        return {
            'total': total,
            'deterministic_only': deterministic_only,
            'needs_model': needs_model,
            'deterministic_and_learned': deterministic_and_learned,
            'errors': errors,
            'deterministic_coverage': deterministic_coverage,
            'avg_confidence': avg_confidence
        }
