#!/usr/bin/env python3
"""
Fact Importance Scorer - Rank Facts by Relevance to Query

Scores fact importance using question-aware heuristics. Deterministic scoring
with explainable breakdown.

Design Philosophy:
- Mostly deterministic (weighted heuristics)
- Explainable (score breakdown for each component)
- Question-aware (different scoring for WHAT/WHO/WHERE/WHEN)
- Entity-centric (query entity gets priority)

Scoring Components (weights):
- Question Relevance: 0.4 (most important!)
- Definitional Priority: 0.3
- Entity Centrality: 0.2
- Semantic Completeness: 0.1
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

from klareco.rag.fact_extractor import Fact, RelationType


class QuestionType(Enum):
    """Question types for question-aware scoring."""
    WHAT = "what"
    WHO = "who"
    WHERE = "where"
    WHEN = "when"
    HOW = "how"
    WHY = "why"
    OTHER = "other"


@dataclass
class ScoreBreakdown:
    """Explainable score breakdown."""
    question_relevance: float
    definitional_priority: float
    entity_centrality: float
    semantic_completeness: float
    final_score: float

    def __str__(self):
        return (f"Score={self.final_score:.2f} "
                f"[Q:{self.question_relevance:.2f}, "
                f"D:{self.definitional_priority:.2f}, "
                f"E:{self.entity_centrality:.2f}, "
                f"C:{self.semantic_completeness:.2f}]")


class FactImportanceScorer:
    """Score fact importance with explainable breakdown."""

    # Scoring weights (sum to 1.0)
    WEIGHTS = {
        'question_relevance': 0.4,
        'definitional': 0.3,
        'centrality': 0.2,
        'completeness': 0.1
    }

    # Source quality weights (Issue #683 - Quick Win +10% precision)
    SOURCE_WEIGHTS = {
        'wikipedia': 1.0,      # Highest quality
        'fundamento': 0.9,     # Official Esperanto foundation
        'gutenberg': 0.7,      # Literary texts, variable quality
        'revo': 0.85,          # Dictionary definitions
        'database': 0.5,       # Generic corpus
        'unknown': 0.3         # Fallback for unspecified sources
    }

    def score(self, fact: Fact, question_type: QuestionType,
              query_entity: Optional[str] = None,
              source_metadata: Optional[Dict] = None) -> ScoreBreakdown:
        """
        Score fact importance with explainable breakdown.

        Args:
            fact: Fact to score
            question_type: Type of question (WHAT, WHO, etc.)
            query_entity: Entity being queried about
            source_metadata: Source document metadata (position, source, etc.)

        Returns:
            ScoreBreakdown with component scores and final score
        """
        # Compute component scores
        q_score = self._score_question_relevance(fact, question_type, query_entity)
        d_score = self._score_definitional(fact, source_metadata or {})
        e_score = self._score_entity_centrality(fact, query_entity)
        c_score = self._score_completeness(fact)

        # Weighted combination
        final = (
            q_score * self.WEIGHTS['question_relevance'] +
            d_score * self.WEIGHTS['definitional'] +
            e_score * self.WEIGHTS['centrality'] +
            c_score * self.WEIGHTS['completeness']
        )

        return ScoreBreakdown(
            question_relevance=q_score,
            definitional_priority=d_score,
            entity_centrality=e_score,
            semantic_completeness=c_score,
            final_score=final
        )

    def _score_question_relevance(self, fact: Fact, question_type: QuestionType,
                                   query_entity: Optional[str]) -> float:
        """
        Score how well fact answers the question type.

        WHAT questions → prioritize IS-A facts about query entity
        WHO questions → prioritize agent/person facts
        WHERE questions → prioritize LOCATED-AT facts
        WHEN questions → prioritize facts with temporal modifiers
        """
        score = 0.0

        if question_type == QuestionType.WHAT:
            # "What is X?" → IS-A facts about X are perfect
            if fact.relation == RelationType.IS_A:
                if query_entity and fact.entity.lower() == query_entity.lower():
                    score = 1.0  # Perfect match!
                elif query_entity and query_entity.lower() in fact.entity.lower():
                    score = 0.9  # Close match
                else:
                    score = 0.5  # IS-A fact, but not about query entity

            # Other facts about query entity
            elif query_entity and fact.entity.lower() == query_entity.lower():
                score = 0.7  # Relevant fact about entity

            # Related facts
            elif query_entity and query_entity.lower() in str(fact).lower():
                score = 0.4

            else:
                score = 0.2  # Generic fact

        elif question_type == QuestionType.WHO:
            # "Who created X?" → prioritize CREATED-BY, FOUNDED, etc.
            if fact.relation in [RelationType.CREATED_BY, RelationType.FOUNDED]:
                if query_entity and query_entity.lower() in fact.entity.lower():
                    score = 1.0  # Perfect match
                else:
                    score = 0.6

            # Facts with agent argument
            elif 'agent' in fact.arguments:
                score = 0.8

            else:
                score = 0.2

        elif question_type == QuestionType.WHERE:
            # "Where is X?" → prioritize LOCATED-AT, BORN
            if fact.relation in [RelationType.LOCATED_AT, RelationType.BORN]:
                if query_entity and query_entity.lower() in fact.entity.lower():
                    score = 1.0
                else:
                    score = 0.7

            # Has location modifier
            elif 'location' in fact.modifiers or 'location' in fact.arguments:
                score = 0.8

            else:
                score = 0.2

        elif question_type == QuestionType.WHEN:
            # "When was X created?" → prioritize facts with time modifiers
            if 'time' in fact.modifiers:
                if query_entity and query_entity.lower() in fact.entity.lower():
                    score = 1.0
                else:
                    score = 0.8

            # CREATED-BY, PUBLISHED, BORN often have time info
            elif fact.relation in [RelationType.CREATED_BY, RelationType.PUBLISHED,
                                   RelationType.BORN]:
                score = 0.6

            else:
                score = 0.2

        else:
            # Generic scoring for other question types
            if query_entity and fact.entity.lower() == query_entity.lower():
                score = 0.7
            elif query_entity and query_entity.lower() in str(fact).lower():
                score = 0.4
            else:
                score = 0.3

        return score

    def _score_definitional(self, fact: Fact, source_metadata: Dict) -> float:
        """
        Score how definitional/central the fact is.

        IS-A relations are inherently definitional.
        First sentences in documents are often definitional.
        Wikipedia lead paragraphs contain definitions.

        Issue #683: Apply source quality weighting for +10% precision.
        """
        score = 0.0

        # IS-A facts are definitional by nature
        if fact.relation == RelationType.IS_A:
            score += 0.5

        # First sentence in document
        sentence_pos = source_metadata.get('sentence_position', -1)
        if sentence_pos == 0:
            score += 0.3
        elif sentence_pos == 1:
            score += 0.2
        elif sentence_pos == 2:
            score += 0.1

        # Quick Win #683: Apply source quality weighting
        source = self._detect_source(source_metadata)
        source_weight = self.SOURCE_WEIGHTS.get(source, self.SOURCE_WEIGHTS['unknown'])

        # Boost score based on source quality
        # High-quality sources (wikipedia=1.0) get full boost
        # Lower-quality sources (database=0.5) get reduced boost
        score = score * source_weight

        # Additional bonus for Wikipedia (definitive source)
        if source == 'wikipedia':
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _detect_source(self, source_metadata: Dict) -> str:
        """
        Detect source from metadata (Issue #683).

        Returns: Source name (wikipedia, fundamento, gutenberg, revo, database, unknown)
        """
        # Check explicit source field
        if 'source' in source_metadata:
            source_str = source_metadata['source'].lower()
            for known_source in self.SOURCE_WEIGHTS.keys():
                if known_source in source_str:
                    return known_source

        # Check metadata dict for source indicators
        metadata_str = str(source_metadata).lower()
        if 'wikipedia' in metadata_str:
            return 'wikipedia'
        elif 'fundamento' in metadata_str:
            return 'fundamento'
        elif 'gutenberg' in metadata_str:
            return 'gutenberg'
        elif 'revo' in metadata_str:
            return 'revo'
        elif source_metadata:
            return 'database'
        else:
            return 'unknown'

    def _score_entity_centrality(self, fact: Fact, query_entity: Optional[str]) -> float:
        """
        Score how central the query entity is to this fact.

        Entity as subject (highest centrality) > object > modifier
        """
        if not query_entity:
            return 0.5  # Neutral if no query entity

        query_lower = query_entity.lower()

        # Entity is the main subject of fact
        if fact.entity.lower() == query_lower:
            return 1.0

        # Entity appears in arguments (object position)
        for arg_val in fact.arguments.values():
            if isinstance(arg_val, str) and query_lower in arg_val.lower():
                return 0.7

        # Entity appears in modifiers
        for mod_val in fact.modifiers.values():
            if isinstance(mod_val, str) and query_lower in mod_val.lower():
                return 0.4

        # Entity doesn't appear
        return 0.0

    def _score_completeness(self, fact: Fact) -> float:
        """
        Score semantic completeness of the fact.

        Facts with more information (arguments + modifiers) are more complete.
        """
        score = 0.3  # Base score for having entity + relation

        # Has required arguments
        if fact.arguments:
            score += 0.4

        # Has enriching modifiers
        if fact.modifiers:
            score += 0.3

        return min(score, 1.0)  # Cap at 1.0


def classify_question_type(query: str) -> QuestionType:
    """
    Classify question type from query text.

    Simple rule-based classification using question words.
    """
    query_lower = query.lower()

    if query_lower.startswith('kio') or 'what' in query_lower:
        return QuestionType.WHAT
    elif query_lower.startswith('kiu') or 'who' in query_lower:
        return QuestionType.WHO
    elif query_lower.startswith('kie') or 'where' in query_lower:
        return QuestionType.WHERE
    elif query_lower.startswith('kiam') or 'when' in query_lower:
        return QuestionType.WHEN
    elif query_lower.startswith('kiel') or 'how' in query_lower:
        return QuestionType.HOW
    elif query_lower.startswith('kial') or 'why' in query_lower:
        return QuestionType.WHY
    else:
        return QuestionType.OTHER
