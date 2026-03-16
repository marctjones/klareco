#!/usr/bin/env python3
"""
Test Full Pipeline with Real Corpus Retrieval

VERSION: v2.1
STAGE: Evaluation
DEPENDENCIES: All summarization components, parser, Kuzu database

Description:
    Tests the complete summarization pipeline with real corpus retrieval:
    Query → Classification → Retrieval → Parsing → Fact Extraction →
    Scoring → Selection → Citation Tracking → Synthesis

Usage:
    python scripts/test_full_pipeline.py \
        --database data/indexes/v2.1_kuzu_index_full \
        --query "Rakontu pri Zamenhof"

Last Updated: 2026-03-09
Author: Claude Code
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.summarization import (
    SchemaClassifier,
    Retriever,
    FactExtractor,
    ImportanceScorer,
    FactSelector,
    CitationTracker,
    SourceSentence,
    Synthesizer
)


def test_full_pipeline(db_path: str, query: str, top_k: int = 20, max_facts: int = 10):
    """Test complete pipeline with real corpus retrieval."""
    print("=" * 100)
    print("FULL PIPELINE TEST - WITH REAL CORPUS RETRIEVAL")
    print("=" * 100)
    print()

    # Step 1: Schema Classification
    print("Step 1: Schema Classification")
    print("-" * 80)
    classifier = SchemaClassifier()
    classification = classifier.classify(query)
    print(f"Query: {query}")
    print(f"Schema: {classification.schema}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Subject: {classification.subject or '(not detected)'}")
    print()

    # Step 2: Retrieval
    print("Step 2: Corpus Retrieval")
    print("-" * 80)
    retriever = Retriever(db_path)

    # Get database statistics
    stats = retriever.get_statistics()
    print(f"Database: {stats.get('total_sentences', 0):,} sentences, "
          f"{stats.get('total_articles', 0):,} articles, "
          f"{stats.get('total_roots', 0):,} roots")

    # Retrieve relevant sentences
    print(f"\nRetrieving top {top_k} relevant sentences...")
    sentences = retriever.retrieve(
        query=query,
        top_k=top_k,
        subject=classification.subject
    )

    print(f"Retrieved {len(sentences)} sentences")
    if sentences:
        print(f"\nTop 5 by relevance:")
        for i, sent in enumerate(sentences[:5], 1):
            text_preview = sent.text[:80] + "..." if len(sent.text) > 80 else sent.text
            print(f"  {i}. [Score: {sent.relevance_score:.1f}] {text_preview}")
            if sent.article_title:
                print(f"     Article: {sent.article_title}")
    else:
        print("⚠️  No sentences retrieved! Check database and query.")
        return
    print()

    # Step 3: Fact Extraction
    print("Step 3: Fact Extraction (AST Parsing)")
    print("-" * 80)
    extractor = FactExtractor()

    print(f"Parsing {len(sentences)} sentences and extracting facts...")
    facts = extractor.extract_facts(sentences)

    fact_stats = extractor.get_statistics(facts)
    print(f"Extracted {fact_stats['total_facts']} facts")
    print(f"  - With subject: {fact_stats['with_subject']}")
    print(f"  - With object: {fact_stats['with_object']}")
    print(f"  - Temporal info: {fact_stats['temporal']}")
    print(f"  - Spatial info: {fact_stats['spatial']}")
    print(f"  - Unique predicates: {fact_stats['unique_predicates']}")

    if facts:
        print(f"\nSample facts:")
        for i, fact in enumerate(facts[:5], 1):
            print(f"  {i}. {fact['predicate']:10} | {fact['subject']:20} {fact['object']:20}")
    else:
        print("⚠️  No facts extracted! Check parser and sentences.")
        return
    print()

    # Step 4: Importance Scoring
    print("Step 4: Importance Scoring")
    print("-" * 80)
    scorer = ImportanceScorer(db_path)

    scored_facts = scorer.score_facts(
        facts=facts,
        schema=classification.schema
    )

    print(f"Scored {len(scored_facts)} facts")
    print(f"\nTop 10 by importance:")
    for i, scored in enumerate(scored_facts[:10], 1):
        fact = scored.fact
        print(f"  {i:2}. {scored.score:.3f} | {fact['predicate']:10} | "
              f"{fact['subject'][:15]:15} {fact['object'][:15]:15}")
    print()

    # Step 5: Fact Selection
    print("Step 5: Fact Selection (Schema Slots)")
    print("-" * 80)
    selector = FactSelector()

    selected_facts = selector.select_facts(
        scored_facts=scored_facts,
        schema=classification.schema,
        max_facts=max_facts
    )

    print(f"Selected {len(selected_facts)} facts across schema slots:")
    # Group by slot
    by_slot = {}
    for selected in selected_facts:
        if selected.slot not in by_slot:
            by_slot[selected.slot] = []
        by_slot[selected.slot].append(selected)

    for slot_name, slot_facts in sorted(by_slot.items()):
        print(f"\n  {slot_name} ({len(slot_facts)} facts):")
        for selected in slot_facts:
            fact = selected.scored_fact.fact
            print(f"    [{selected.selection_order}] {fact['predicate']:10} | "
                  f"Score: {selected.novelty_adjusted_score:.2f}")
    print()

    # Step 6: Citation Tracking
    print("Step 6: Citation Tracking")
    print("-" * 80)
    tracker = CitationTracker()

    # Add all source sentences
    for fact in facts:
        source_id = fact.get('source_id', '')
        source_text = fact.get('source_text', '')

        if source_id and source_text:
            # Find article title from retrieved sentences
            article_title = None
            for sent in sentences:
                if sent.sentence_id == source_id:
                    article_title = sent.article_title
                    break

            source = SourceSentence(
                source_id=source_id,
                sentence=source_text,
                article_title=article_title
            )
            tracker.add_source(source)

    # Link selected facts to sources
    fact_id = 1
    for selected in selected_facts:
        fact = selected.scored_fact.fact
        source_id = fact.get('source_id')
        if source_id:
            tracker.link_fact_to_source(fact_id, source_id)
            fact_id += 1

    cit_stats = tracker.get_statistics()
    print(f"Citations: {cit_stats['total_citations']} unique sources")
    print(f"Facts with citations: {cit_stats['total_facts']}")
    print()

    # Step 7: Synthesis
    print("Step 7: Summary Synthesis")
    print("-" * 80)
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
    if summary.citations:
        print(summary.citations)
    print()

    # Final Summary
    print("=" * 100)
    print("PIPELINE SUMMARY")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Schema: {classification.schema} (confidence: {classification.confidence:.2f})")
    print(f"Sentences retrieved: {len(sentences)}")
    print(f"Facts extracted: {len(facts)}")
    print(f"Facts scored: {len(scored_facts)}")
    print(f"Facts selected: {len(selected_facts)}")
    print(f"Citations: {summary.num_citations}")
    print(f"Summary length: {len(summary.text)} characters")
    print()

    if len(sentences) > 0 and len(facts) > 0:
        print("✅ Full pipeline test complete!")
    else:
        print("⚠️  Pipeline completed but with limited data. Check database content.")
    print()


def test_batch_queries(db_path: str, queries_file: str):
    """Test pipeline on batch of queries from file."""
    print("=" * 100)
    print("BATCH PIPELINE TEST")
    print("=" * 100)
    print()

    # Load queries
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query_data = json.loads(line)
                queries.append(query_data)

    print(f"Loaded {len(queries)} test queries")
    print()

    # Test each query
    results = []
    for i, query_data in enumerate(queries, 1):
        query = query_data['query']
        expected_schema = query_data.get('schema_type', 'unknown')

        print(f"\n{'='*100}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"Expected schema: {expected_schema}")
        print('=' * 100)
        print()

        try:
            test_full_pipeline(db_path, query, top_k=10, max_facts=5)
            results.append({'query': query, 'success': True})
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({'query': query, 'success': False, 'error': str(e)})

    # Summary
    print("\n" + "=" * 100)
    print("BATCH TEST SUMMARY")
    print("=" * 100)
    successful = sum(1 for r in results if r['success'])
    print(f"Queries tested: {len(queries)}")
    print(f"Successful: {successful}/{len(queries)} ({successful/len(queries)*100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test full pipeline with real corpus retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help='Single test query'
    )

    parser.add_argument(
        '--queries-file',
        type=str,
        help='JSONL file with multiple test queries'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of sentences to retrieve (default: 20)'
    )

    parser.add_argument(
        '--max-facts',
        type=int,
        default=10,
        help='Maximum facts to select (default: 10)'
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    # Run appropriate test mode
    if args.queries_file:
        queries_path = Path(args.queries_file)
        if not queries_path.exists():
            print(f"❌ Queries file not found: {queries_path}")
            sys.exit(1)
        test_batch_queries(str(db_path), str(queries_path))
    elif args.query:
        test_full_pipeline(str(db_path), args.query, args.top_k, args.max_facts)
    else:
        # Default: test with a sample query
        print("No query specified, using default: 'Kio estas Esperanto?'")
        test_full_pipeline(str(db_path), "Kio estas Esperanto?", args.top_k, args.max_facts)


if __name__ == '__main__':
    main()
