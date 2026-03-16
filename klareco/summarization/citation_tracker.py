"""
Citation Tracker - Source Provenance Through Pipeline

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema
STAGE: Summarization - Phase 0

Description:
    Tracks source sentences through the entire summarization pipeline
    from retrieval → extraction → selection → synthesis.

Citation Format:
    - Inline: "Zamenhof fondis Esperanton en 1887 [1,2]."
    - Reference list: [1] Source sentence from Wikipedia
                      [2] Another source sentence

Usage:
    from klareco.summarization import CitationTracker

    tracker = CitationTracker()
    tracker.add_source(fact_id=1, source_id="wiki_123", sentence="...")
    citations = tracker.get_citations(fact_id=1)

Last Updated: 2026-03-09
Author: Claude Code
Related Issues: #674, #675
See Also: docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md
"""

from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SourceSentence:
    """A source sentence with metadata."""
    source_id: str  # Unique ID (e.g., "wiki_article_123_sent_5")
    sentence: str  # Original sentence text
    article_title: Optional[str] = None
    url: Optional[str] = None
    sentence_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A citation linking fact to source."""
    citation_id: int  # Sequential ID for display ([1], [2], etc.)
    source: SourceSentence
    relevance_score: float = 1.0  # How relevant is this source to the fact?


@dataclass
class FactWithCitations:
    """A fact with its supporting citations."""
    fact: Dict[str, Any]
    citations: List[Citation]
    synthesized_text: Optional[str] = None  # Final text with inline citations


class CitationTracker:
    """
    Track citations through the summarization pipeline.

    Maintains bidirectional mapping:
    - fact_id → source_ids (what sources support this fact?)
    - source_id → fact_ids (what facts come from this source?)
    """

    def __init__(self):
        """Initialize citation tracker."""
        # Source management
        self.sources: Dict[str, SourceSentence] = {}  # source_id → SourceSentence
        self.source_counter = 0  # For generating sequential IDs

        # Fact → Sources mapping
        self.fact_sources: Dict[int, Set[str]] = defaultdict(set)  # fact_id → {source_ids}

        # Source → Facts mapping
        self.source_facts: Dict[str, Set[int]] = defaultdict(set)  # source_id → {fact_ids}

        # Citation display order
        self.citation_order: List[str] = []  # Ordered list of source_ids for [1], [2], etc.

    def add_source(self, source: SourceSentence) -> str:
        """
        Add a source sentence to the tracker.

        Args:
            source: SourceSentence object

        Returns:
            source_id
        """
        if source.source_id not in self.sources:
            self.sources[source.source_id] = source
        return source.source_id

    def link_fact_to_source(self, fact_id: int, source_id: str, relevance_score: float = 1.0):
        """
        Link a fact to its source sentence.

        Args:
            fact_id: Unique fact identifier
            source_id: Source sentence ID
            relevance_score: How relevant is this source? (0.0-1.0)
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} not found. Add it first with add_source().")

        self.fact_sources[fact_id].add(source_id)
        self.source_facts[source_id].add(fact_id)

    def get_citations(self, fact_id: int) -> List[Citation]:
        """
        Get citations for a fact.

        Args:
            fact_id: Fact identifier

        Returns:
            List of Citation objects with sequential IDs
        """
        source_ids = self.fact_sources.get(fact_id, set())
        citations = []

        for source_id in source_ids:
            if source_id not in self.sources:
                continue

            # Get or assign citation ID
            if source_id not in self.citation_order:
                self.citation_order.append(source_id)

            citation_id = self.citation_order.index(source_id) + 1

            citations.append(Citation(
                citation_id=citation_id,
                source=self.sources[source_id]
            ))

        # Sort by citation ID
        citations.sort(key=lambda c: c.citation_id)
        return citations

    def get_all_citations(self) -> List[Citation]:
        """
        Get all citations in display order.

        Returns:
            List of all citations [1], [2], [3], etc.
        """
        citations = []
        for i, source_id in enumerate(self.citation_order, 1):
            if source_id in self.sources:
                citations.append(Citation(
                    citation_id=i,
                    source=self.sources[source_id]
                ))
        return citations

    def format_inline_citations(self, fact_id: int) -> str:
        """
        Format inline citations for a fact.

        Args:
            fact_id: Fact identifier

        Returns:
            Citation string like "[1,2,3]" or ""
        """
        citations = self.get_citations(fact_id)
        if not citations:
            return ""

        citation_ids = [str(c.citation_id) for c in citations]
        return f"[{','.join(citation_ids)}]"

    def format_reference_list(self) -> str:
        """
        Format complete reference list.

        Returns:
            Multi-line string with all references
        """
        citations = self.get_all_citations()
        if not citations:
            return ""

        lines = ["## Fontoj / Sources\n"]
        for citation in citations:
            source = citation.source
            line = f"[{citation.citation_id}] "

            if source.article_title:
                line += f"{source.article_title}: "

            # Truncate long sentences
            sentence = source.sentence
            if len(sentence) > 150:
                sentence = sentence[:147] + "..."

            line += f'"{sentence}"'

            if source.url:
                line += f" ({source.url})"

            lines.append(line)

        return "\n".join(lines)

    def merge_facts(self, fact_ids: List[int]) -> int:
        """
        Merge multiple facts into one (aggregate citations).

        Args:
            fact_ids: List of fact IDs to merge

        Returns:
            New merged fact ID
        """
        merged_id = max(self.fact_sources.keys()) + 1 if self.fact_sources else 1

        # Aggregate all sources
        all_sources = set()
        for fact_id in fact_ids:
            all_sources.update(self.fact_sources.get(fact_id, set()))

        # Link merged fact to all sources
        for source_id in all_sources:
            self.link_fact_to_source(merged_id, source_id)

        return merged_id

    def get_statistics(self) -> Dict[str, int]:
        """Get citation statistics."""
        return {
            'total_sources': len(self.sources),
            'total_facts': len(self.fact_sources),
            'total_citations': len(self.citation_order),
            'avg_citations_per_fact': (
                sum(len(sources) for sources in self.fact_sources.values()) / len(self.fact_sources)
                if self.fact_sources else 0
            )
        }

    def explain_provenance(self, fact_id: int) -> str:
        """Explain provenance for a fact (debugging)."""
        citations = self.get_citations(fact_id)

        explanation = f"Fact {fact_id} provenance:\n"
        explanation += f"  {len(citations)} source(s)\n\n"

        for citation in citations:
            source = citation.source
            explanation += f"[{citation.citation_id}] {source.source_id}\n"
            explanation += f"    Sentence: {source.sentence[:100]}...\n"
            if source.article_title:
                explanation += f"    Article: {source.article_title}\n"
            explanation += "\n"

        return explanation

    def save_to_dict(self) -> Dict[str, Any]:
        """Save tracker state to dictionary (for serialization)."""
        return {
            'sources': {
                sid: {
                    'source_id': s.source_id,
                    'sentence': s.sentence,
                    'article_title': s.article_title,
                    'url': s.url,
                    'sentence_index': s.sentence_index,
                    'metadata': s.metadata
                }
                for sid, s in self.sources.items()
            },
            'fact_sources': {
                str(fid): list(sources)
                for fid, sources in self.fact_sources.items()
            },
            'citation_order': self.citation_order
        }

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> 'CitationTracker':
        """Load tracker from dictionary."""
        tracker = cls()

        # Restore sources
        for sid, s_data in data['sources'].items():
            tracker.sources[sid] = SourceSentence(**s_data)

        # Restore fact-source links
        for fid_str, source_ids in data['fact_sources'].items():
            fid = int(fid_str)
            for sid in source_ids:
                tracker.fact_sources[fid].add(sid)
                tracker.source_facts[sid].add(fid)

        # Restore citation order
        tracker.citation_order = data['citation_order']

        return tracker
