"""
Fact Selector - Schema-Based Fact Selection with Novelty Discount

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Summarization - Phase 0

Description:
    Selects top-scoring facts per schema slot, applying novelty discount
    to avoid repetition. Uses RST discourse structure for coherence.

Schema Slots:
    - Biographical: identigo, naskiĝo_morto, ĉefa_realigo, profesio, kunteksto
    - Definitional: kategorio, esenca_eco, funkcio, origino, ekzemploj
    - Event: kio_okazis, kiam, kie, partoprenantoj, rezulto

Usage:
    from klareco.summarization import FactSelector

    selector = FactSelector()
    selected = selector.select_facts(
        scored_facts=scored_facts,
        schema='biographical',
        max_facts=10
    )

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md
"""

from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .importance_scorer import ScoredFact


@dataclass
class SchemaSlot:
    """Definition of a schema slot."""
    name: str
    description: str
    priority: float  # 0.0-1.0, higher = more important
    max_facts: int  # Maximum facts per slot
    keywords: List[str] = field(default_factory=list)  # Verb/noun classes for this slot


@dataclass
class SelectedFact:
    """A fact selected for inclusion in summary."""
    scored_fact: ScoredFact
    slot: str
    novelty_adjusted_score: float
    selection_order: int


# Schema definitions
BIOGRAPHICAL_SCHEMA = [
    SchemaSlot(
        name="identigo",
        description="Who this person is (name, basic identity)",
        priority=1.0,
        max_facts=2,
        keywords=["est", "estas", "persono", "rolo"]
    ),
    SchemaSlot(
        name="naskiĝo_morto",
        description="Birth and death (when, where)",
        priority=0.95,
        max_facts=2,
        keywords=["naskiĝ", "mort", "viv", "ekzisto"]
    ),
    SchemaSlot(
        name="ĉefa_realigo",
        description="Main achievements/contributions",
        priority=0.90,
        max_facts=3,
        keywords=["fond", "kre", "skrib", "far", "kreado"]
    ),
    SchemaSlot(
        name="profesio",
        description="Profession/occupation",
        priority=0.80,
        max_facts=2,
        keywords=["profesio", "labor", "est"]
    ),
    SchemaSlot(
        name="kunteksto",
        description="Context/background/influences",
        priority=0.70,
        max_facts=2,
        keywords=["inspir", "stud", "lern", "pens"]
    ),
]

DEFINITIONAL_SCHEMA = [
    SchemaSlot(
        name="kategorio",
        description="What category/type it belongs to",
        priority=1.0,
        max_facts=1,
        keywords=["est", "estas", "tip", "spec", "ekzisto"]
    ),
    SchemaSlot(
        name="esenca_eco",
        description="Essential properties/characteristics",
        priority=0.90,
        max_facts=3,
        keywords=["hav", "konsist", "inkluziv", "karakteriz", "havado"]
    ),
    SchemaSlot(
        name="funkcio",
        description="Purpose/function/use",
        priority=0.85,
        max_facts=2,
        keywords=["uz", "serv", "cel", "help", "permis"]
    ),
    SchemaSlot(
        name="origino",
        description="Origin/creation/history",
        priority=0.75,
        max_facts=2,
        keywords=["kre", "fond", "origin", "veni", "kreado"]
    ),
    SchemaSlot(
        name="ekzemploj",
        description="Examples/instances",
        priority=0.60,
        max_facts=2,
        keywords=["ekzempl", "kiel", "inkluziv", "konsist"]
    ),
]

EVENT_SCHEMA = [
    SchemaSlot(
        name="kio_okazis",
        description="What happened (main action)",
        priority=1.0,
        max_facts=2,
        keywords=["okaz", "far", "kreado", "ŝanĝo"]
    ),
    SchemaSlot(
        name="kiam",
        description="When it happened (time)",
        priority=0.95,
        max_facts=1,
        keywords=["temporal"]  # Special marker
    ),
    SchemaSlot(
        name="kie",
        description="Where it happened (location)",
        priority=0.90,
        max_facts=1,
        keywords=["spatial", "loko"]  # Special marker
    ),
    SchemaSlot(
        name="partoprenantoj",
        description="Who participated/was involved",
        priority=0.85,
        max_facts=3,
        keywords=["persono", "rolo", "hom"]
    ),
    SchemaSlot(
        name="rezulto",
        description="Outcome/result/consequence",
        priority=0.75,
        max_facts=2,
        keywords=["rezult", "sekv", "kaŭz", "efik"]
    ),
]

SCHEMAS = {
    'biographical': BIOGRAPHICAL_SCHEMA,
    'definitional': DEFINITIONAL_SCHEMA,
    'event': EVENT_SCHEMA,
}


class FactSelector:
    """
    Select facts based on schema slots with novelty discount.

    Ensures diverse, non-repetitive facts that fill all important schema slots.
    """

    def __init__(self, novelty_discount: float = 0.3):
        """
        Initialize fact selector.

        Args:
            novelty_discount: How much to penalize similar facts (0.0-1.0)
                             Higher = more penalty for repetition
        """
        self.novelty_discount = novelty_discount

    def select_facts(
        self,
        scored_facts: List[ScoredFact],
        schema: str,
        max_facts: int = 10
    ) -> List[SelectedFact]:
        """
        Select facts for summary based on schema.

        Args:
            scored_facts: Facts with importance scores
            schema: Schema type ('biographical', 'definitional', 'event')
            max_facts: Maximum total facts to select

        Returns:
            List of SelectedFact objects with slot assignments
        """
        if schema not in SCHEMAS:
            raise ValueError(f"Unknown schema: {schema}")

        schema_slots = SCHEMAS[schema]
        selected_facts = []
        used_roots: Set[str] = set()  # Track roots to avoid repetition
        selection_order = 0

        # Process slots in priority order
        for slot in sorted(schema_slots, key=lambda s: s.priority, reverse=True):
            # Find facts matching this slot
            slot_facts = self._match_slot(scored_facts, slot)

            # Select top facts for this slot
            slot_selected = 0
            for scored_fact in slot_facts:
                if slot_selected >= slot.max_facts:
                    break
                if len(selected_facts) >= max_facts:
                    break

                # Apply novelty discount
                novelty_score = self._calculate_novelty_score(
                    scored_fact, used_roots
                )

                # Skip if too similar to already selected facts
                if novelty_score < 0.3:
                    continue

                # Add to selection
                selected_facts.append(SelectedFact(
                    scored_fact=scored_fact,
                    slot=slot.name,
                    novelty_adjusted_score=novelty_score,
                    selection_order=selection_order
                ))

                # Update tracking
                predicate = scored_fact.fact.get('predicate', '')
                if predicate:
                    used_roots.add(predicate)
                slot_selected += 1
                selection_order += 1

        return selected_facts

    def _match_slot(
        self,
        scored_facts: List[ScoredFact],
        slot: SchemaSlot
    ) -> List[ScoredFact]:
        """
        Find facts that match a schema slot.

        Args:
            scored_facts: All scored facts
            slot: Schema slot to match

        Returns:
            List of facts matching this slot, sorted by score
        """
        matched = []

        for scored_fact in scored_facts:
            fact = scored_fact.fact
            predicate = fact.get('predicate', '')

            # Check if predicate matches slot keywords
            matches = False

            # Check temporal/spatial markers (special case for events)
            if 'temporal' in slot.keywords and fact.get('temporal_marker'):
                matches = True
            elif 'spatial' in slot.keywords and fact.get('spatial_marker'):
                matches = True

            # Check predicate against keywords
            for keyword in slot.keywords:
                if keyword in predicate:
                    matches = True
                    break

            # Check verb/noun class from explanation
            if not matches:
                explanation = ' '.join(scored_fact.explanation)
                for keyword in slot.keywords:
                    if keyword in explanation:
                        matches = True
                        break

            if matches:
                matched.append(scored_fact)

        # Sort by score (descending)
        matched.sort(key=lambda f: f.score, reverse=True)
        return matched

    def _calculate_novelty_score(
        self,
        scored_fact: ScoredFact,
        used_roots: Set[str]
    ) -> float:
        """
        Calculate novelty-adjusted score.

        Penalize facts that use roots already seen.

        Args:
            scored_fact: Fact to score
            used_roots: Set of roots already used

        Returns:
            Novelty-adjusted score (0.0-1.0)
        """
        fact = scored_fact.fact
        base_score = scored_fact.score

        # Check for overlapping roots
        predicate = fact.get('predicate', '')
        subject_root = fact.get('subject_root', '')
        object_root = fact.get('object_root', '')

        overlap_count = 0
        total_roots = 0

        for root in [predicate, subject_root, object_root]:
            if root:
                total_roots += 1
                if root in used_roots:
                    overlap_count += 1

        # Calculate novelty penalty
        if total_roots == 0:
            novelty_penalty = 0
        else:
            overlap_ratio = overlap_count / total_roots
            novelty_penalty = overlap_ratio * self.novelty_discount

        # Apply penalty
        novelty_score = base_score * (1.0 - novelty_penalty)

        return novelty_score

    def explain_selection(self, selected_facts: List[SelectedFact]) -> str:
        """Generate human-readable explanation of selection."""
        explanation = f"Selected {len(selected_facts)} facts:\n\n"

        # Group by slot
        by_slot = defaultdict(list)
        for selected in selected_facts:
            by_slot[selected.slot].append(selected)

        # Display by slot
        for slot_name, facts in sorted(by_slot.items()):
            explanation += f"Slot: {slot_name}\n"
            for selected in facts:
                fact = selected.scored_fact.fact
                explanation += f"  [{selected.selection_order}] Score: {selected.novelty_adjusted_score:.2f} | "
                explanation += f"{fact.get('predicate', '?')} | "
                explanation += f"{fact.get('subject', '?')} {fact.get('object', '?')}\n"
            explanation += "\n"

        return explanation
