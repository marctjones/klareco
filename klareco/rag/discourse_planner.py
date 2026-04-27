#!/usr/bin/env python3
"""
Discourse Planner - Plan Multi-Sentence Discourse Structure

Plans discourse structure using RST (Rhetorical Structure Theory) relations.
Identifies relations between facts and adds discourse markers.

Design Philosophy:
- Rule-based RST relation identification
- Surface pattern matching (60-70% accuracy expected)
- Discourse markers for coherence
- Temporal and entity-based ordering

Discourse Relations (RST):
- ELABORATION: Fact B adds detail to A (same entity)
- SEQUENCE: Temporal ordering (time-based)
- CONTRAST: Contradictory information (requires antonymy - limited)
- CAUSE: Causal relationship (verb-based heuristics)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import random

from klareco.rag.unified_extractor import Fact, RelationType


class DiscourseRelation(Enum):
    """RST discourse relations."""
    ELABORATION = "elaboration"   # B adds detail to A
    SEQUENCE = "sequence"          # Temporal ordering
    CONTRAST = "contrast"          # Contradictory info
    CAUSE = "cause"                # Causal relationship
    EXAMPLE = "example"            # B exemplifies A
    NONE = "none"                  # No explicit relation


# Discourse markers for each relation
DISCOURSE_MARKERS = {
    DiscourseRelation.ELABORATION: ['Krome', 'Ankaŭ', 'Aldone', 'Plie'],
    DiscourseRelation.CONTRAST: ['Tamen', 'Sed', 'Kontraŭe', 'Malgraŭ tio'],
    DiscourseRelation.CAUSE: ['Pro tio', 'Sekve', 'Tial', 'Konsekve'],
    DiscourseRelation.SEQUENCE: ['Poste', 'Antaŭe', 'Sekve', 'Tiam'],
    DiscourseRelation.EXAMPLE: ['Ekzemple', 'Precipe', 'Nome'],
    DiscourseRelation.NONE: []
}


@dataclass
class DiscoursePlan:
    """Planned discourse structure for facts."""
    facts: List[Fact]
    relations: List[Tuple[int, int, DiscourseRelation]]  # (idx_a, idx_b, relation)
    markers: List[Optional[str]]  # Discourse marker for each fact (None for first)


class DiscoursePlanner:
    """Plan discourse structure for multi-sentence output."""

    def plan(self, facts: List[Fact], max_facts: int = 4) -> DiscoursePlan:
        """
        Plan discourse structure for facts.

        Args:
            facts: List of facts (assumed already ranked by importance)
            max_facts: Maximum number of facts to include

        Returns:
            DiscoursePlan with ordered facts, relations, and markers
        """
        # Select top facts
        selected = facts[:max_facts]

        if len(selected) == 0:
            return DiscoursePlan(facts=[], relations=[], markers=[])

        # Identify discourse relations between consecutive facts
        relations = self._identify_relations(selected)

        # Assign discourse markers
        markers = self._assign_markers(selected, relations)

        return DiscoursePlan(
            facts=selected,
            relations=relations,
            markers=markers
        )

    def _identify_relations(self, facts: List[Fact]) -> List[Tuple[int, int, DiscourseRelation]]:
        """
        Identify discourse relations between consecutive facts.

        Uses surface patterns:
        - Same entity → ELABORATION
        - Temporal ordering → SEQUENCE
        - Creation + Publication → CAUSE
        """
        relations = []

        for i in range(len(facts) - 1):
            fact_a = facts[i]
            fact_b = facts[i + 1]

            # Determine relation
            relation = self._determine_relation(fact_a, fact_b)
            relations.append((i, i + 1, relation))

        return relations

    def _determine_relation(self, fact_a: Fact, fact_b: Fact) -> DiscourseRelation:
        """
        Determine discourse relation between two facts.

        Heuristics (in priority order):
        1. SEQUENCE: Temporal ordering (explicit time modifiers)
        2. CAUSE: Creation → Publication, Founding → Development
        3. ELABORATION: Same entity, different properties
        4. NONE: Default
        """
        # 1. SEQUENCE: Check temporal ordering
        time_a = fact_a.modifiers.get('tempo')
        time_b = fact_b.modifiers.get('tempo')

        if time_a and time_b:
            # Try to parse years
            year_a = self._extract_year(time_a)
            year_b = self._extract_year(time_b)

            if year_a and year_b and year_a < year_b:
                return DiscourseRelation.SEQUENCE

        # 2. CAUSE: Certain relation pairs imply causality
        if self._is_causal_pair(fact_a.relation, fact_b.relation):
            return DiscourseRelation.CAUSE

        # 3. ELABORATION: Same entity, different relations
        if (fact_a.entity and fact_b.entity and
            fact_a.entity.lower() == fact_b.entity.lower() and
            fact_a.relation != fact_b.relation):
            return DiscourseRelation.ELABORATION

        # Default: No explicit relation
        return DiscourseRelation.NONE

    def _is_causal_pair(self, rel_a: RelationType, rel_b: RelationType) -> bool:
        """Check if relation pair implies causal relationship."""
        causal_pairs = [
            (RelationType.CREATED_BY, RelationType.PUBLISHED),
            (RelationType.FOUNDED, RelationType.HAS),
            (RelationType.BORN, RelationType.CREATED_BY),
        ]

        return (rel_a, rel_b) in causal_pairs

    def _extract_year(self, time_str: str) -> Optional[int]:
        """Extract year from time string."""
        # Simple extraction: find 4-digit number
        import re
        match = re.search(r'\b(1\d{3}|20\d{2})\b', str(time_str))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    def _assign_markers(self, facts: List[Fact],
                       relations: List[Tuple[int, int, DiscourseRelation]]) -> List[Optional[str]]:
        """
        Assign discourse markers to facts based on relations.

        First fact gets no marker.
        Subsequent facts get marker based on relation to previous fact.
        """
        markers = [None]  # First fact has no marker

        for i, (idx_a, idx_b, relation) in enumerate(relations):
            if relation == DiscourseRelation.NONE:
                markers.append(None)  # No marker for no relation
            else:
                # Choose random marker from options
                marker_options = DISCOURSE_MARKERS[relation]
                if marker_options:
                    marker = random.choice(marker_options)
                    markers.append(marker)
                else:
                    markers.append(None)

        return markers
