"""
Importance Scorer - Schema-Aware Fact Scoring

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema with semantic properties
STAGE: Summarization - Phase 0

Description:
    Scores facts based on semantic properties and schema type.
    Uses root-level importance scores from Kuzu database.

Scoring Strategy:
    - Biographical: Prioritize life events, achievements, roles
    - Definitional: Prioritize category, properties, function
    - Event: Prioritize time, location, participants, outcome

Usage:
    from klareco.summarization import ImportanceScorer

    scorer = ImportanceScorer(db_path='data/indexes/v2.1_kuzu_index_full')
    score = scorer.score_fact(
        fact={'predicate': 'fond', 'subject': 'Zamenhof'},
        schema='biographical'
    )

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: klareco/schema/semantic_properties.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import kuzu
try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)


@dataclass
class ScoredFact:
    """A fact with its importance score."""
    fact: Dict[str, Any]
    score: float
    schema: str
    explanation: List[str]  # Why this score?


class ImportanceScorer:
    """
    Schema-aware fact importance scorer.

    Queries Kuzu database for semantic properties and applies
    schema-specific importance weights.
    """

    def __init__(self, db_path: str):
        """
        Initialize scorer with database connection.

        Args:
            db_path: Path to Kuzu database directory
        """
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

        # Cache for semantic properties
        self._property_cache = {}

    def get_root_properties(self, radiko: str) -> Optional[Dict[str, Any]]:
        """
        Get semantic properties for a root from database.

        Args:
            radiko: Root to query

        Returns:
            Dict with semantic properties, or None if not found/annotated
        """
        # Check cache first
        if radiko in self._property_cache:
            return self._property_cache[radiko]

        try:
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{radiko}'}})
                RETURN r.radiko, r.verba_klaso, r.aspekta_klaso,
                       r.substantiva_klaso, r.semantika_kampo,
                       r.graveco_biografia, r.graveco_difina, r.graveco_okazaĵa,
                       r.funda_stato, r.ofteca_tavolo, r.konfido
            """)

            if result.has_next():
                row = result.get_next()
                properties = {
                    'radiko': row[0],
                    'verba_klaso': row[1],
                    'aspekta_klaso': row[2],
                    'substantiva_klaso': row[3],
                    'semantika_kampo': row[4],
                    'graveco_biografia': row[5] if row[5] is not None else 0.5,
                    'graveco_difina': row[6] if row[6] is not None else 0.5,
                    'graveco_okazaĵa': row[7] if row[7] is not None else 0.5,
                    'funda_stato': row[8],
                    'ofteca_tavolo': row[9],
                    'konfido': row[10] if row[10] is not None else 1.0
                }

                # Cache it
                self._property_cache[radiko] = properties
                return properties

            return None

        except Exception as e:
            print(f"Warning: Failed to query properties for '{radiko}': {e}")
            return None

    def score_fact(self, fact: Dict[str, Any], schema: str) -> ScoredFact:
        """
        Score a single fact based on schema.

        Args:
            fact: Fact dictionary with 'predicate' (root), 'subject', 'object', etc.
            schema: Schema type ('biographical', 'definitional', 'event')

        Returns:
            ScoredFact with score and explanation
        """
        # Extract root from fact
        predicate = fact.get('predicate', '')
        subject_root = fact.get('subject_root', '')
        object_root = fact.get('object_root', '')

        # Get semantic properties
        predicate_props = self.get_root_properties(predicate) if predicate else None
        subject_props = self.get_root_properties(subject_root) if subject_root else None
        object_props = self.get_root_properties(object_root) if object_root else None

        # Score based on schema
        if schema == 'biographical':
            score, explanation = self._score_biographical(
                fact, predicate_props, subject_props, object_props
            )
        elif schema == 'definitional':
            score, explanation = self._score_definitional(
                fact, predicate_props, subject_props, object_props
            )
        elif schema == 'event':
            score, explanation = self._score_event(
                fact, predicate_props, subject_props, object_props
            )
        else:
            score = 0.5
            explanation = ['unknown_schema']

        return ScoredFact(
            fact=fact,
            score=score,
            schema=schema,
            explanation=explanation
        )

    def _score_biographical(
        self,
        fact: Dict[str, Any],
        predicate_props: Optional[Dict],
        subject_props: Optional[Dict],
        object_props: Optional[Dict]
    ) -> tuple:
        """Score fact for biographical summary."""
        score = 0.0
        explanation = []

        # Use predicate's biographical importance
        if predicate_props and predicate_props['graveco_biografia']:
            score = predicate_props['graveco_biografia']
            explanation.append(f"predicate_bio_score:{score:.2f}")

            # Boost for life events
            verb_class = predicate_props.get('verba_klaso') or ''
            if verb_class and 'ekzisto' in verb_class:  # Life events (lived, died, etc.)
                score += 0.10
                explanation.append('life_event_boost')
            elif verb_class and 'kreado' in verb_class:  # Creation/achievement
                score += 0.08
                explanation.append('achievement_boost')

        # Boost for person-related nouns
        if subject_props and subject_props.get('substantiva_klaso'):
            noun_class = subject_props['substantiva_klaso']
            if noun_class == 'persono':
                score += 0.05
                explanation.append('person_subject')
            elif noun_class == 'rolo':
                score += 0.08
                explanation.append('role_subject')

        # Boost for Fundamento roots (higher confidence)
        if predicate_props and predicate_props.get('funda_stato') == 'fundamento_kerno':
            score += 0.05
            explanation.append('fundamento_root')

        # Confidence adjustment
        if predicate_props and predicate_props.get('konfido'):
            score *= predicate_props['konfido']
            explanation.append(f"confidence_adj:{predicate_props['konfido']:.2f}")

        # Fallback score
        if score == 0.0:
            score = 0.40  # Default for unannotated roots
            explanation.append('fallback_biographical')

        # Cap at 1.0
        score = min(score, 1.0)

        return score, explanation

    def _score_definitional(
        self,
        fact: Dict[str, Any],
        predicate_props: Optional[Dict],
        subject_props: Optional[Dict],
        object_props: Optional[Dict]
    ) -> tuple:
        """Score fact for definitional summary."""
        score = 0.0
        explanation = []

        # Use predicate's definitional importance
        if predicate_props and predicate_props['graveco_difina']:
            score = predicate_props['graveco_difina']
            explanation.append(f"predicate_def_score:{score:.2f}")

            # Boost for "estas" (is/category)
            verb_class = predicate_props.get('verba_klaso') or ''
            if verb_class and 'ekzisto' in verb_class:  # "estas" - category assignment
                score += 0.10
                explanation.append('category_verb_boost')
            elif verb_class and 'havado' in verb_class:  # "havas" - essential property
                score += 0.08
                explanation.append('property_verb_boost')

        # Boost for concept/category nouns
        if object_props and object_props.get('substantiva_klaso'):
            noun_class = object_props['substantiva_klaso']
            if noun_class == 'koncepto':
                score += 0.10
                explanation.append('concept_object')
            elif noun_class == 'kvalito':
                score += 0.08
                explanation.append('quality_object')

        # Boost for Fundamento roots
        if predicate_props and predicate_props.get('funda_stato') == 'fundamento_kerno':
            score += 0.05
            explanation.append('fundamento_root')

        # Confidence adjustment
        if predicate_props and predicate_props.get('konfido'):
            score *= predicate_props['konfido']
            explanation.append(f"confidence_adj:{predicate_props['konfido']:.2f}")

        # Fallback score
        if score == 0.0:
            score = 0.40
            explanation.append('fallback_definitional')

        score = min(score, 1.0)

        return score, explanation

    def _score_event(
        self,
        fact: Dict[str, Any],
        predicate_props: Optional[Dict],
        subject_props: Optional[Dict],
        object_props: Optional[Dict]
    ) -> tuple:
        """Score fact for event summary."""
        score = 0.0
        explanation = []

        # Use predicate's event importance
        if predicate_props and predicate_props['graveco_okazaĵa']:
            score = predicate_props['graveco_okazaĵa']
            explanation.append(f"predicate_event_score:{score:.2f}")

            # Boost for action verbs
            verb_class = predicate_props.get('verba_klaso') or ''
            if verb_class and 'kreado' in verb_class:  # Main action
                score += 0.10
                explanation.append('action_verb_boost')
            elif verb_class and 'movo' in verb_class:  # Movement/arrival
                score += 0.08
                explanation.append('movement_verb_boost')

        # Boost for event nouns
        if subject_props and subject_props.get('substantiva_klaso'):
            noun_class = subject_props['substantiva_klaso']
            if noun_class == 'evento':
                score += 0.10
                explanation.append('event_noun')
            elif noun_class == 'loko':
                score += 0.08
                explanation.append('location_noun')

        # Boost for temporal/spatial information
        if fact.get('temporal_marker'):
            score += 0.10
            explanation.append('temporal_info')
        if fact.get('spatial_marker'):
            score += 0.08
            explanation.append('spatial_info')

        # Boost for Fundamento roots
        if predicate_props and predicate_props.get('funda_stato') == 'fundamento_kerno':
            score += 0.05
            explanation.append('fundamento_root')

        # Confidence adjustment
        if predicate_props and predicate_props.get('konfido'):
            score *= predicate_props['konfido']
            explanation.append(f"confidence_adj:{predicate_props['konfido']:.2f}")

        # Fallback score
        if score == 0.0:
            score = 0.40
            explanation.append('fallback_event')

        score = min(score, 1.0)

        return score, explanation

    def score_facts(self, facts: List[Dict[str, Any]], schema: str) -> List[ScoredFact]:
        """
        Score multiple facts.

        Args:
            facts: List of fact dictionaries
            schema: Schema type

        Returns:
            List of ScoredFact objects, sorted by score (descending)
        """
        scored = [self.score_fact(fact, schema) for fact in facts]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def explain_score(self, scored_fact: ScoredFact) -> str:
        """Generate human-readable explanation of score."""
        explanation = f"Score: {scored_fact.score:.2f} ({scored_fact.schema} schema)\n"
        explanation += f"Fact: {scored_fact.fact}\n"
        explanation += f"Components:\n"
        for component in scored_fact.explanation:
            explanation += f"  - {component}\n"
        return explanation
