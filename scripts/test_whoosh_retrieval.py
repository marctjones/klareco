#!/usr/bin/env python3
"""
Test Whoosh FTS Retrieval Quality

Tests:
1. Recall: Does Whoosh find known good sentences?
2. Ranking: Are relevant sentences ranked higher?
3. Coverage: Does it find sentences Kuzu was missing?
4. Speed: How fast is retrieval?
5. Correctness: Do queries work as expected?

Usage:
    python scripts/test_whoosh_retrieval.py
    python scripts/test_whoosh_retrieval.py --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_recall(index_path: Path):
    """
    Test 1: Recall - Does Whoosh find known good sentences?

    We know "Zamenhof kreis Esperanton" exists and should be found
    for queries about who founded Esperanto.
    """
    print("\n" + "="*80)
    print("TEST 1: RECALL - Finding Known Good Sentences")
    print("="*80)

    ix = open_dir(str(index_path))

    test_cases = [
        {
            'name': 'WHO founded Esperanto',
            'query': 'zamenhof* AND (kre* OR fond* OR establ*)',
            'expected_phrases': ['zamenhof', 'kreis', 'esperanton'],
            'description': 'Should find "Zamenhof kreis Esperanton"',
            'use_and': True  # Don't use OrGroup for AND queries
        },
        {
            'name': 'WHAT is Esperanto',
            'query': 'esperanto*',
            'expected_phrases': ['esperanto'],
            'description': 'Should find sentences about Esperanto',
            'use_and': False
        },
        {
            'name': 'Zamenhof biography',
            'query': 'zamenhof*',
            'expected_phrases': ['zamenhof'],
            'description': 'Should find sentences about Zamenhof',
            'use_and': False
        }
    ]

    results_summary = []

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        for test in test_cases:
            print(f"\n{test['name']}: {test['description']}")
            print(f"Query: {test['query']}")

            # Use default parser for AND queries, OrGroup for OR queries
            if test.get('use_and', False):
                query = QueryParser("text_lower", ix.schema).parse(test['query'])
            else:
                query = QueryParser("text_lower", ix.schema, group=OrGroup).parse(test['query'])
            results = searcher.search(query, limit=20)

            print(f"Found: {len(results)} results")

            # Check if expected phrases appear in top results
            found_matches = []
            for i, hit in enumerate(results[:10], 1):
                text_lower = hit['text'].lower()
                matches = [phrase for phrase in test['expected_phrases'] if phrase in text_lower]
                if matches:
                    found_matches.append({
                        'rank': i,
                        'score': hit.score,
                        'text': hit['text'][:100] + '...',
                        'matches': matches
                    })

            if found_matches:
                print(f"✓ Found {len(found_matches)} matches in top 10:")
                for match in found_matches[:3]:
                    print(f"  Rank {match['rank']}: Score={match['score']:.4f}")
                    print(f"    Matches: {match['matches']}")
                    print(f"    Text: {match['text']}")
                results_summary.append(('✓', test['name'], len(found_matches)))
            else:
                print(f"✗ No matches found in top 10")
                results_summary.append(('✗', test['name'], 0))

    # Summary
    print("\n" + "="*80)
    print("RECALL TEST SUMMARY")
    print("="*80)
    for status, name, count in results_summary:
        print(f"{status} {name}: {count} matches in top 10")


def test_ranking(index_path: Path):
    """
    Test 2: Ranking - Are relevant sentences ranked higher than irrelevant?

    BM25 should rank sentences with more query terms higher.
    """
    print("\n" + "="*80)
    print("TEST 2: RANKING - BM25 Score Quality")
    print("="*80)

    ix = open_dir(str(index_path))

    query_str = 'zamenhof* AND kre*'
    print(f"\nQuery: {query_str}")
    print("Expected: Sentences with both 'zamenhof' AND 'kre' should rank highest\n")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        # Don't use OrGroup for AND queries
        query = QueryParser("text_lower", ix.schema).parse(query_str)
        results = searcher.search(query, limit=10)

        print(f"Top 10 results (showing score and term presence):")
        print("-" * 80)

        for i, hit in enumerate(results, 1):
            text_lower = hit['text'].lower()
            has_zamenhof = 'zamenhof' in text_lower
            has_kre = 'kre' in text_lower

            # Count how many times each term appears
            zamenhof_count = text_lower.count('zamenhof')
            kre_count = len([w for w in text_lower.split() if 'kre' in w])

            print(f"{i}. Score: {hit.score:.4f}")
            print(f"   Zamenhof: {'✓' if has_zamenhof else '✗'} ({zamenhof_count}x)")
            print(f"   Kre*: {'✓' if has_kre else '✗'} ({kre_count}x)")
            print(f"   Text: {hit['text'][:80]}...")
            print()

        # Check if ranking makes sense
        # Top results should have more matching terms
        top3_scores = [results[i].score for i in range(min(3, len(results)))]
        if len(top3_scores) >= 2 and top3_scores[0] >= top3_scores[1]:
            print("✓ Ranking appears correct (top result has highest score)")
        else:
            print("✗ Ranking may be incorrect")


def test_coverage(index_path: Path):
    """
    Test 3: Coverage - Does Whoosh find sentences Kuzu was missing?

    Test the specific case that was failing with Kuzu.
    """
    print("\n" + "="*80)
    print("TEST 3: COVERAGE - Finding Previously Missing Sentences")
    print("="*80)

    ix = open_dir(str(index_path))

    # The query that was failing with Kuzu - add wildcards for root morphology
    query_roots = ['esperant*', 'fond*', 'kre*', 'establ*', 'startig*']
    query_str = ' OR '.join(query_roots)

    print(f"\nOriginal problem query (WHO founded Esperanto):")
    print(f"Roots: {query_roots}")
    print(f"Query: {query_str}\n")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        query = QueryParser("text_lower", ix.schema, group=OrGroup).parse(query_str)
        results = searcher.search(query, limit=100)

        print(f"Total results: {len(results)}")

        # Look for the golden answer
        golden_phrases = [
            'zamenhof kreis esperanton',
            'zamenhof fondis esperanton',
            'kreinto de esperanto'
        ]

        print(f"\nSearching for golden answers (phrases indicating Zamenhof created Esperanto):")
        found_golden = []

        for i, hit in enumerate(results, 1):
            text_lower = hit['text'].lower()
            for phrase in golden_phrases:
                if phrase in text_lower:
                    found_golden.append({
                        'rank': i,
                        'score': hit.score,
                        'phrase': phrase,
                        'text': hit['text']
                    })
                    break

        if found_golden:
            print(f"✓ Found {len(found_golden)} golden answer(s)!")
            for match in found_golden[:3]:
                print(f"\n  Rank {match['rank']}: Score={match['score']:.4f}")
                print(f"  Matched: '{match['phrase']}'")
                print(f"  Text: {match['text'][:150]}...")
        else:
            print(f"✗ No golden answers found in top 100 results")
            print(f"  Showing top 5 results instead:")
            for i, hit in enumerate(results[:5], 1):
                print(f"  {i}. {hit['text'][:100]}...")


def test_speed(index_path: Path):
    """
    Test 4: Speed - How fast is Whoosh retrieval?
    """
    print("\n" + "="*80)
    print("TEST 4: SPEED - Retrieval Performance")
    print("="*80)

    ix = open_dir(str(index_path))

    test_queries = [
        ('zamenhof*', 'Single term'),
        ('zamenhof* AND kre*', 'AND query'),
        ('esperant* OR fund* OR establ*', 'OR query'),
        ('zamenhof* AND (kre* OR fond* OR establ* OR startig*)', 'Complex query'),
    ]

    print("\nTiming 10 runs of each query:\n")

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        for query_str, description in test_queries:
            times = []

            for _ in range(10):
                start = time.time()
                # Use default parser for AND queries, OrGroup for OR queries
                if ' AND ' in query_str:
                    query = QueryParser("text_lower", ix.schema).parse(query_str)
                else:
                    query = QueryParser("text_lower", ix.schema, group=OrGroup).parse(query_str)
                results = searcher.search(query, limit=20)
                _ = len(results)  # Force evaluation
                elapsed = (time.time() - start) * 1000  # ms
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"{description}:")
            print(f"  Query: {query_str}")
            print(f"  Avg: {avg_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
            print()


def test_correctness(index_path: Path):
    """
    Test 5: Correctness - Do boolean operators work as expected?
    """
    print("\n" + "="*80)
    print("TEST 5: CORRECTNESS - Boolean Operator Behavior")
    print("="*80)

    ix = open_dir(str(index_path))

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        # Test AND
        print("\n1. AND operator (should find sentences with BOTH terms):")
        query = QueryParser("text_lower", ix.schema).parse("zamenhof* AND esperanto*")
        results = searcher.search(query, limit=5)

        correct = 0
        for hit in results:
            text_lower = hit['text'].lower()
            has_both = 'zamenhof' in text_lower and 'esperanto' in text_lower
            if has_both:
                correct += 1

        print(f"   Results: {len(results)}")
        print(f"   Correct (have both): {correct}/{len(results)}")
        if correct == len(results):
            print("   ✓ AND operator works correctly")
        else:
            print("   ✗ AND operator may not be working")

        # Test OR
        print("\n2. OR operator (should find sentences with EITHER term):")
        query = QueryParser("text_lower", ix.schema, group=OrGroup).parse("zamenhof* OR esperanto*")
        results = searcher.search(query, limit=10)

        has_zamenhof = sum(1 for hit in results if 'zamenhof' in hit['text'].lower())
        has_esperanto = sum(1 for hit in results if 'esperanto' in hit['text'].lower())

        print(f"   Results: {len(results)}")
        print(f"   Have 'zamenhof': {has_zamenhof}")
        print(f"   Have 'esperanto': {has_esperanto}")
        if has_zamenhof > 0 and has_esperanto > 0:
            print("   ✓ OR operator works correctly")
        else:
            print("   ✗ OR operator may not be working")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--index', type=Path, default=Path('data/indexes/whoosh_fts'),
                       help='Path to Whoosh index')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if index exists
    if not args.index.exists():
        print(f"ERROR: Index not found at {args.index}")
        print("Build the index first with: python scripts/build_whoosh_index.py")
        sys.exit(1)

    print("="*80)
    print("WHOOSH FTS QUALITY TEST SUITE")
    print("="*80)
    print(f"Index: {args.index}")

    # Run all tests
    test_recall(args.index)
    test_ranking(args.index)
    test_coverage(args.index)
    test_speed(args.index)
    test_correctness(args.index)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
