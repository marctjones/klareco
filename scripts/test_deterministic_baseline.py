#!/usr/bin/env python3
"""
Test Deterministic Baseline - Full Pipeline Integration

VERSION: v2.1
STAGE: Evaluation
DEPENDENCIES: All summarization components

Description:
    Tests the complete deterministic summarization pipeline:
    Query → Schema Classification → Fact Scoring → Selection → Synthesis → Citations

Usage:
    python scripts/test_deterministic_baseline.py \
        --database data/indexes/v2.1_kuzu_index_full \
        --query "Rakontu pri Zamenhof"

Last Updated: 2026-03-09
Author: Claude Code
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.summarization import (
    SchemaClassifier,
    ImportanceScorer,
    FactSelector,
    CitationTracker,
    SourceSentence,
    Synthesizer
)


def test_pipeline(db_path: str, query: str):
    """Test complete pipeline on a single query."""
    print("=" * 100)
    print("DETERMINISTIC BASELINE PIPELINE TEST")
    print("=" * 100)
    print()

    # Step 1: Schema Classification
    print("Step 1: Schema Classification")
    print("-" * 60)
    classifier = SchemaClassifier()
    classification = classifier.classify(query)
    print(f"Query: {query}")
    print(f"Schema: {classification.schema}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Subject: {classification.subject}")
    print()

    # Step 2: Mock Fact Extraction (normally from retrieval + parsing)
    print("Step 2: Fact Extraction (Mock)")
    print("-" * 60)
    print("In a full implementation, this would:")
    print("  1. Retrieve relevant sentences from Kuzu")
    print("  2. Parse sentences to ASTs")
    print("  3. Extract facts from ASTs")
    print()
    print("For testing, using pre-defined facts...")
    print()

    # Mock facts for "Rakontu pri Zamenhof"
    mock_facts = [
        {
            'fact_id': 1,
            'predicate': 'est',
            'subject': 'Zamenhof',
            'object': 'kuracisto',
            'subject_root': 'hom',
            'object_root': 'hom',
            'source_id': 'wiki_zamenhof_sent_1',
            'source_text': 'Ludoviko Lazaro Zamenhof estis kuracisto kaj la kreinto de Esperanto.'
        },
        {
            'fact_id': 2,
            'predicate': 'fond',
            'subject': 'Zamenhof',
            'object': 'Esperanton',
            'subject_root': 'hom',
            'object_root': 'nom',
            'source_id': 'wiki_zamenhof_sent_2',
            'source_text': 'Li fondis Esperanton en 1887.'
        },
        {
            'fact_id': 3,
            'predicate': 'naskiĝ',
            'subject': 'Zamenhof',
            'object': 'Bjalistoko',
            'subject_root': 'hom',
            'object_root': 'urb',
            'source_id': 'wiki_zamenhof_sent_3',
            'source_text': 'Li naskiĝis en Bjalistoko, Pollando en 1859.'
        },
        {
            'fact_id': 4,
            'predicate': 'mort',
            'subject': 'Zamenhof',
            'object': 'Varsovio',
            'subject_root': 'hom',
            'object_root': 'urb',
            'source_id': 'wiki_zamenhof_sent_4',
            'source_text': 'Li mortis en Varsovio en 1917.'
        },
        {
            'fact_id': 5,
            'predicate': 'kre',
            'subject': 'Zamenhof',
            'object': 'internacian lingvon',
            'subject_root': 'hom',
            'object_root': 'nom',
            'source_id': 'wiki_zamenhof_sent_5',
            'source_text': 'Lia celo estis krei internacian helplingvon por paco.'
        },
    ]

    print(f"Extracted {len(mock_facts)} facts from corpus")
    print()

    # Step 3: Importance Scoring
    print("Step 3: Importance Scoring")
    print("-" * 60)
    scorer = ImportanceScorer(db_path)

    scored_facts = scorer.score_facts(
        facts=mock_facts,
        schema=classification.schema
    )

    for scored in scored_facts[:5]:
        fact = scored.fact
        print(f"  {scored.score:.2f} | {fact['predicate']:10} | {fact['subject']} {fact['object']}")

    print()

    # Step 4: Fact Selection
    print("Step 4: Fact Selection")
    print("-" * 60)
    selector = FactSelector()

    selected_facts = selector.select_facts(
        scored_facts=scored_facts,
        schema=classification.schema,
        max_facts=5
    )

    print(f"Selected {len(selected_facts)} facts:")
    for selected in selected_facts:
        fact = selected.scored_fact.fact
        print(f"  [{selected.selection_order}] {selected.slot:15} | {fact['predicate']:10} | Score: {selected.novelty_adjusted_score:.2f}")

    print()

    # Step 5: Citation Tracking
    print("Step 5: Citation Tracking")
    print("-" * 60)
    tracker = CitationTracker()

    # Add sources
    for mock_fact in mock_facts:
        source = SourceSentence(
            source_id=mock_fact['source_id'],
            sentence=mock_fact['source_text'],
            article_title="Ludoviko Zamenhof",
            url="https://eo.wikipedia.org/wiki/Ludoviko_Zamenhof"
        )
        tracker.add_source(source)
        tracker.link_fact_to_source(
            fact_id=mock_fact['fact_id'],
            source_id=mock_fact['source_id']
        )

    # Link selected facts to sources
    for selected in selected_facts:
        fact = selected.scored_fact.fact
        if 'source_id' in fact:
            # Re-link with selection order as fact_id
            tracker.link_fact_to_source(
                fact_id=selected.selection_order,
                source_id=fact['source_id']
            )

    stats = tracker.get_statistics()
    print(f"Citations tracked: {stats['total_citations']}")
    print(f"Unique sources: {stats['total_sources']}")
    print()

    # Step 6: Synthesis
    print("Step 6: Synthesis")
    print("-" * 60)
    synthesizer = Synthesizer()

    summary = synthesizer.synthesize(
        selected_facts=selected_facts,
        schema=classification.schema,
        tracker=tracker,
        subject=classification.subject
    )

    print("Generated Summary:")
    print()
    print(summary.text)
    print()
    print(summary.citations)
    print()

    # Final Summary
    print("=" * 100)
    print("PIPELINE SUMMARY")
    print("=" * 100)
    print(f"Schema: {classification.schema}")
    print(f"Facts extracted: {len(mock_facts)}")
    print(f"Facts scored: {len(scored_facts)}")
    print(f"Facts selected: {len(selected_facts)}")
    print(f"Citations: {summary.num_citations}")
    print(f"Summary length: {len(summary.text)} characters")
    print()
    print("✅ Pipeline test complete!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test deterministic baseline pipeline"
    )

    parser.add_argument(
        '--database',
        type=str,
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database'
    )

    parser.add_argument(
        '--query',
        type=str,
        default='Rakontu pri Zamenhof',
        help='Test query'
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    test_pipeline(str(db_path), args.query)


if __name__ == '__main__':
    main()
