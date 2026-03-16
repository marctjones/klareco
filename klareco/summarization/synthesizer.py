"""
Synthesizer - Generate Coherent Text with Citations

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Summarization - Phase 0

Description:
    Synthesizes selected facts into coherent Esperanto text with
    inline citations. Preserves factual accuracy and adds discourse markers.

Synthesis Strategy:
    - Template-based generation (deterministic)
    - Sentence ordering by schema slot priority
    - Discourse markers (sed, ankaŭ, do, etc.)
    - Inline citations after each fact

Usage:
    from klareco.summarization import Synthesizer

    synth = Synthesizer()
    summary = synth.synthesize(
        selected_facts=selected_facts,
        schema='biographical',
        tracker=citation_tracker
    )

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #666
See Also: docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .fact_selector import SelectedFact
from .citation_tracker import CitationTracker
from .discourse_planner import DiscoursePlanner, DiscourseFact


@dataclass
class Summary:
    """A complete summary with text and citations."""
    text: str  # Esperanto summary text with inline citations
    citations: str  # Formatted reference list
    schema: str
    num_facts: int
    num_citations: int


class Synthesizer:
    """
    Synthesize facts into coherent Esperanto text.

    Uses template-based generation for deterministic, accurate output.
    """

    def __init__(self, use_discourse_planning: bool = True):
        """
        Initialize synthesizer with discourse planner.

        Args:
            use_discourse_planning: If True, use Phase 1 discourse planning
        """
        self.use_discourse_planning = use_discourse_planning

        # Phase 1: Discourse planner for coherent text
        if use_discourse_planning:
            self.discourse_planner = DiscoursePlanner()

        # Legacy: Discourse markers for Phase 0 (kept for backwards compatibility)
        self.discourse_markers = {
            'continuation': ['Ankaŭ', 'Krome', 'Aldone'],
            'contrast': ['Sed', 'Tamen', 'Malgraŭ tio'],
            'result': ['Do', 'Tial', 'Sekve'],
            'example': ['Ekzemple', 'Jen', 'Kiel ekzemplo'],
            'temporal': ['Poste', 'Antaŭe', 'Tiam'],
        }

        # Schema-specific opening phrases (Phase 0 - not used with discourse planning)
        self.schema_openings = {
            'biographical': {
                'identigo': '',  # No marker for first sentence
                'naskiĝo_morto': 'Li/Ŝi',
                'ĉefa_realigo': 'Lia/Ŝia plej grava kontribuo estas',
                'profesio': 'Profesie,',
                'kunteksto': 'Kuntekste,',
            },
            'definitional': {
                'kategorio': '',  # No marker for definition
                'esenca_eco': 'Ĝi karakteriziĝas per',
                'funkcio': 'Ĝia funkcio estas',
                'origino': 'Origine,',
                'ekzemploj': 'Ekzemple,',
            },
            'event': {
                'kio_okazis': '',
                'kiam': 'Tio okazis',
                'kie': 'La evento okazis en',
                'partoprenantoj': 'Partoprenis',
                'rezulto': 'Kiel rezulto,',
            }
        }

    def synthesize(
        self,
        selected_facts: List[SelectedFact],
        schema: str,
        tracker: CitationTracker,
        subject: Optional[str] = None
    ) -> Summary:
        """
        Synthesize facts into coherent summary.

        Args:
            selected_facts: Facts selected for inclusion
            schema: Schema type
            tracker: Citation tracker with source info
            subject: Subject of summary (person name, concept, event)

        Returns:
            Summary object with text and citations
        """
        if not selected_facts:
            return Summary(
                text="Neniu informo trovita.",
                citations="",
                schema=schema,
                num_facts=0,
                num_citations=0
            )

        # Phase 1: Use discourse planning for coherent text structure
        if self.use_discourse_planning:
            discourse_facts = self.discourse_planner.plan_discourse(
                selected_facts, schema
            )
            sentences = self._synthesize_with_discourse(
                discourse_facts, schema, tracker, subject
            )
        else:
            # Phase 0: Simple slot-based generation
            facts_by_slot = self._group_by_slot(selected_facts)
            sentences = []
            for slot_name in self._get_slot_order(schema):
                if slot_name not in facts_by_slot:
                    continue
                slot_facts = facts_by_slot[slot_name]
                slot_sentences = self._synthesize_slot(
                    slot_name, slot_facts, schema, tracker, subject
                )
                sentences.extend(slot_sentences)

        # Combine into paragraph
        text = ' '.join(sentences)

        # Format citations
        citations = tracker.format_reference_list()

        return Summary(
            text=text,
            citations=citations,
            schema=schema,
            num_facts=len(selected_facts),
            num_citations=len(tracker.get_all_citations())
        )

    def _group_by_slot(self, selected_facts: List[SelectedFact]) -> Dict[str, List[SelectedFact]]:
        """Group facts by schema slot."""
        by_slot = {}
        for fact in selected_facts:
            if fact.slot not in by_slot:
                by_slot[fact.slot] = []
            by_slot[fact.slot].append(fact)
        return by_slot

    def _get_slot_order(self, schema: str) -> List[str]:
        """Get slot ordering for schema."""
        if schema == 'biographical':
            return ['identigo', 'naskiĝo_morto', 'ĉefa_realigo', 'profesio', 'kunteksto']
        elif schema == 'definitional':
            return ['kategorio', 'esenca_eco', 'funkcio', 'origino', 'ekzemploj']
        elif schema == 'event':
            return ['kio_okazis', 'kiam', 'kie', 'partoprenantoj', 'rezulto']
        else:
            return []

    def _synthesize_with_discourse(
        self,
        discourse_facts: List[DiscourseFact],
        schema: str,
        tracker: CitationTracker,
        subject: Optional[str]
    ) -> List[str]:
        """
        Synthesize facts with discourse planning (Phase 1).

        Args:
            discourse_facts: Facts with discourse relations
            schema: Schema type
            tracker: Citation tracker
            subject: Summary subject

        Returns:
            List of coherent sentences with discourse markers
        """
        sentences = []

        for i, dfact in enumerate(discourse_facts):
            # Skip duplicates
            if dfact.deduplicated:
                continue

            fact = dfact.fact.scored_fact.fact

            # Generate base sentence
            sentence = self._generate_sentence(fact, dfact.fact.slot, schema, subject)

            # Add discourse marker
            if dfact.discourse_marker and sentence:
                sentence = f"{dfact.discourse_marker}, {sentence[0].lower()}{sentence[1:]}"

            # Add citations
            citation_text = tracker.format_inline_citations(dfact.fact.selection_order)
            if citation_text:
                sentence = sentence.rstrip('.') + f" {citation_text}."
            else:
                sentence = sentence.rstrip('.') + "."

            sentences.append(sentence)

        return sentences

    def _synthesize_slot(
        self,
        slot_name: str,
        slot_facts: List[SelectedFact],
        schema: str,
        tracker: CitationTracker,
        subject: Optional[str]
    ) -> List[str]:
        """
        Synthesize facts for a single slot into sentences.

        Args:
            slot_name: Schema slot name
            slot_facts: Facts in this slot
            schema: Schema type
            tracker: Citation tracker
            subject: Summary subject

        Returns:
            List of sentences for this slot
        """
        sentences = []

        for i, selected_fact in enumerate(slot_facts):
            fact = selected_fact.scored_fact.fact

            # Generate sentence from fact (template-based)
            sentence = self._generate_sentence(fact, slot_name, schema, subject)

            # Add inline citations
            # For now, we'll use a placeholder fact_id
            # In a real implementation, facts would have persistent IDs
            citation_text = tracker.format_inline_citations(selected_fact.selection_order)

            if citation_text:
                sentence = sentence.rstrip('.') + f" {citation_text}."
            else:
                sentence = sentence.rstrip('.') + "."

            sentences.append(sentence)

        return sentences

    def _generate_sentence(
        self,
        fact: Dict[str, Any],
        slot_name: str,
        schema: str,
        subject: Optional[str]
    ) -> str:
        """
        Generate Esperanto sentence from fact using AST deparser.

        Phase 1: Uses AST deparser for grammatically perfect output.
        Falls back to template if AST not available.

        Args:
            fact: Fact dictionary (with 'ast' field from Phase 1)
            slot_name: Schema slot
            schema: Schema type
            subject: Summary subject

        Returns:
            Esperanto sentence
        """
        # Phase 1: Try to use AST deparser
        ast = fact.get('ast')
        if ast:
            try:
                from klareco.deparser import deparse
                # Deparse AST to grammatically perfect sentence
                sentence = deparse(ast)
                # Remove trailing period (will be added with citations)
                return sentence.rstrip('.')
            except Exception as e:
                # Fallback to template if deparser fails
                print(f"Warning: Deparser failed for fact: {e}")
                pass

        # Fallback: Template-based generation (Phase 0)
        return self._generate_sentence_template(fact, subject)

    def _generate_sentence_template(
        self,
        fact: Dict[str, Any],
        subject: Optional[str]
    ) -> str:
        """
        Template-based sentence generation (fallback for Phase 0 compatibility).

        Args:
            fact: Fact dictionary
            subject: Summary subject

        Returns:
            Esperanto sentence
        """
        predicate = fact.get('predicate', '')
        subj = fact.get('subject', subject or 'Ĝi')
        obj = fact.get('object', '')

        # Convert root to verb form (simple approximation)
        verb = self._root_to_verb(predicate)

        # Build sentence
        parts = [subj, verb]
        if obj:
            parts.append(obj)

        sentence = ' '.join(parts)

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        return sentence

    def _root_to_verb(self, root: str) -> str:
        """
        Convert root to verb form (simple approximation).

        TODO: Use proper morphology rules or AST deparser.

        Args:
            root: Verb root

            Returns:
            Conjugated verb (simplified)
        """
        if not root:
            return ''

        # Simple heuristic: add -is for past tense
        # (Real implementation would check context)
        if root.endswith('i'):  # Already infinitive
            return root[:-1] + 'is'
        else:
            return root + 'is'

    def explain_synthesis(self, summary: Summary) -> str:
        """Generate explanation of synthesis process."""
        explanation = f"Summary Synthesis Explanation\n"
        explanation += f"{'='*60}\n\n"
        explanation += f"Schema: {summary.schema}\n"
        explanation += f"Facts used: {summary.num_facts}\n"
        explanation += f"Citations: {summary.num_citations}\n\n"
        explanation += f"Generated text:\n{summary.text}\n\n"
        explanation += f"{summary.citations}\n"

        return explanation
