"""
Discourse Planner - Structure Facts for Coherent Text

VERSION: v2.1
COMPATIBLE WITH: Phase 1 deterministic components
STAGE: Summarization - Phase 1

Description:
    Plans discourse structure using RST (Rhetorical Structure Theory) principles.
    Aggregates related facts, deduplicates information, and adds discourse markers
    for coherent text flow.

Discourse Relations (RST):
    - Elaboration: Additional detail about same entity
    - Sequence: Temporal ordering of events
    - Contrast: Opposing or contrasting information
    - Cause-Effect: Causal relationships
    - Attribution: Source attribution

Usage:
    from klareco.summarization import DiscoursePlanner

    planner = DiscoursePlanner()
    discourse_plan = planner.plan_discourse(
        selected_facts=selected_facts,
        schema='biographical'
    )

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: Phase 1 implementation
See Also: docs/PHASE_1_PROGRESS.md
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .fact_selector import SelectedFact


@dataclass
class DiscourseFact:
    """A fact with discourse relation information."""
    fact: SelectedFact
    relation: str  # RST relation type
    relation_target: Optional[int] = None  # Index of related fact (if any)
    discourse_marker: Optional[str] = None  # Marker to use (Ankaŭ, Sed, etc.)
    should_aggregate: bool = False  # Should aggregate with next fact?
    deduplicated: bool = False  # Marked as duplicate?


class DiscoursePlanner:
    """
    Plan discourse structure for coherent text generation.

    Uses RST (Rhetorical Structure Theory) to structure facts with
    discourse relations and markers.
    """

    def __init__(self):
        """Initialize discourse planner with relation rules."""

        # Discourse markers by relation type
        self.discourse_markers = {
            'elaboration': ['Krome', 'Aldone', 'Plie'],  # Additional info
            'sequence': ['Poste', 'Tiam', 'Sekve'],  # Temporal ordering
            'contrast': ['Sed', 'Tamen', 'Kontraŭe'],  # Opposing facts
            'cause_effect': ['Do', 'Tial', 'Sekve'],  # Causal relations
            'example': ['Ekzemple', 'Jen'],  # Examples
            'continuation': ['Ankaŭ', 'Same'],  # Continuing same topic
        }

        # Schema-specific discourse preferences
        self.schema_discourse_patterns = {
            'biographical': ['sequence', 'elaboration', 'cause_effect'],
            'definitional': ['elaboration', 'example', 'contrast'],
            'event': ['sequence', 'cause_effect', 'elaboration'],
        }

    def plan_discourse(
        self,
        selected_facts: List[SelectedFact],
        schema: str
    ) -> List[DiscourseFact]:
        """
        Plan discourse structure for facts.

        Args:
            selected_facts: Facts selected for inclusion
            schema: Schema type (biographical, definitional, event)

        Returns:
            List of facts with discourse relation information
        """
        if not selected_facts:
            return []

        # Step 1: Deduplicate repeated facts
        deduplicated = self._deduplicate_facts(selected_facts)

        # Step 2: Identify aggregation opportunities
        with_aggregation = self._identify_aggregations(deduplicated)

        # Step 3: Assign discourse relations
        with_relations = self._assign_discourse_relations(
            with_aggregation, schema
        )

        # Step 4: Select discourse markers
        with_markers = self._select_discourse_markers(with_relations)

        return with_markers

    def _deduplicate_facts(
        self,
        facts: List[SelectedFact]
    ) -> List[DiscourseFact]:
        """
        Deduplicate repeated facts (same subject + predicate).

        Args:
            facts: Input facts

        Returns:
            Facts with duplicates marked
        """
        discourse_facts = []
        seen = set()

        for fact in facts:
            fact_dict = fact.scored_fact.fact

            # Create signature: (subject_root, predicate)
            signature = (
                fact_dict.get('subject_root', ''),
                fact_dict.get('predicate', '')
            )

            # Check if seen before
            is_duplicate = signature in seen and signature != ('', '')

            discourse_facts.append(DiscourseFact(
                fact=fact,
                relation='none',  # Will be assigned later
                deduplicated=is_duplicate
            ))

            seen.add(signature)

        return discourse_facts

    def _identify_aggregations(
        self,
        facts: List[DiscourseFact]
    ) -> List[DiscourseFact]:
        """
        Identify facts that can be aggregated (same subject, different predicates).

        Example: "Li naskiĝis en 1859. Li mortis en 1917."
                 → "Li naskiĝis en 1859 kaj mortis en 1917."

        Args:
            facts: Input facts

        Returns:
            Facts with aggregation flags
        """
        # Group by subject
        by_subject = defaultdict(list)
        for i, dfact in enumerate(facts):
            if dfact.deduplicated:
                continue  # Skip duplicates

            fact_dict = dfact.fact.scored_fact.fact
            subject_root = fact_dict.get('subject_root', '')

            if subject_root:
                by_subject[subject_root].append(i)

        # Mark aggregation opportunities
        for indices in by_subject.values():
            if len(indices) >= 2:
                # Mark first N-1 facts for aggregation
                for i in indices[:-1]:
                    facts[i].should_aggregate = True

        return facts

    def _assign_discourse_relations(
        self,
        facts: List[DiscourseFact],
        schema: str
    ) -> List[DiscourseFact]:
        """
        Assign RST discourse relations between facts.

        Args:
            facts: Input facts
            schema: Schema type

        Returns:
            Facts with discourse relations assigned
        """
        if not facts:
            return facts

        # First fact has no relation (nucleus)
        facts[0].relation = 'nucleus'

        # Assign relations to subsequent facts
        for i in range(1, len(facts)):
            if facts[i].deduplicated:
                facts[i].relation = 'duplicate'
                continue

            prev_fact = facts[i-1].fact.scored_fact.fact
            curr_fact = facts[i].fact.scored_fact.fact

            # Determine relation based on content
            relation = self._infer_relation(prev_fact, curr_fact, schema)
            facts[i].relation = relation
            facts[i].relation_target = i - 1  # Relates to previous fact

        return facts

    def _infer_relation(
        self,
        prev_fact: Dict[str, Any],
        curr_fact: Dict[str, Any],
        schema: str
    ) -> str:
        """
        Infer discourse relation between two facts.

        Args:
            prev_fact: Previous fact
            curr_fact: Current fact
            schema: Schema type

        Returns:
            Discourse relation type
        """
        prev_pred = prev_fact.get('predicate', '')
        curr_pred = curr_fact.get('predicate', '')
        prev_subj = prev_fact.get('subject_root', '')
        curr_subj = curr_fact.get('subject_root', '')

        # Same subject → elaboration or sequence
        if prev_subj == curr_subj and prev_subj != '':
            # Check for temporal markers
            if curr_fact.get('temporal_marker'):
                return 'sequence'
            else:
                return 'elaboration'

        # Different subject → contrast or elaboration
        if prev_pred == curr_pred:
            return 'contrast'  # Same action, different actors

        # Default: elaboration
        return 'elaboration'

    def _select_discourse_markers(
        self,
        facts: List[DiscourseFact]
    ) -> List[DiscourseFact]:
        """
        Select appropriate discourse markers for each fact.

        Args:
            facts: Facts with discourse relations

        Returns:
            Facts with discourse markers assigned
        """
        # Track marker usage to avoid repetition
        used_markers = set()

        for i, dfact in enumerate(facts):
            if dfact.deduplicated:
                continue  # No marker for duplicates

            if dfact.relation == 'nucleus':
                continue  # First fact has no marker

            # Get candidate markers for this relation
            candidates = self.discourse_markers.get(dfact.relation, [])

            # Select unused marker (or use first if all used)
            marker = None
            for cand in candidates:
                if cand not in used_markers:
                    marker = cand
                    used_markers.add(cand)
                    break

            if not marker and candidates:
                marker = candidates[0]  # Fallback

            dfact.discourse_marker = marker

        return facts

    def get_statistics(
        self,
        discourse_facts: List[DiscourseFact]
    ) -> Dict[str, Any]:
        """Get discourse planning statistics."""
        if not discourse_facts:
            return {
                'total_facts': 0,
                'deduplicated': 0,
                'aggregations': 0,
                'discourse_markers': 0,
                'relations': {}
            }

        relations = defaultdict(int)
        deduplicated = 0
        aggregations = 0
        with_markers = 0

        for dfact in discourse_facts:
            relations[dfact.relation] += 1
            if dfact.deduplicated:
                deduplicated += 1
            if dfact.should_aggregate:
                aggregations += 1
            if dfact.discourse_marker:
                with_markers += 1

        return {
            'total_facts': len(discourse_facts),
            'deduplicated': deduplicated,
            'aggregations': aggregations,
            'discourse_markers': with_markers,
            'relations': dict(relations)
        }
