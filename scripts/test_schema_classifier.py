#!/usr/bin/env python3
"""
Test Schema Classifier on Phase 0 Queries

VERSION: v2.1
STAGE: Evaluation
DEPENDENCIES: klareco.summarization.SchemaClassifier

Description:
    Tests deterministic schema classifier on Phase 0 test queries.
    Validates classification accuracy against expected schema types.

Usage:
    python scripts/test_schema_classifier.py \
        --queries data/test_queries/phase_0.jsonl

Last Updated: 2026-03-09
Author: Claude Code
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.summarization import SchemaClassifier


def load_test_queries(queries_path: str) -> List[Dict]:
    """Load test queries from JSONL file."""
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def test_classifier(queries_path: str, verbose: bool = False):
    """Test schema classifier on test queries."""
    print(f"📖 Loading test queries from: {queries_path}\n")

    # Load queries
    queries = load_test_queries(queries_path)
    print(f"✅ Loaded {len(queries)} test queries\n")

    # Initialize classifier
    classifier = SchemaClassifier()

    # Test each query
    correct = 0
    total = 0
    results = []

    print("=" * 100)
    print("CLASSIFICATION RESULTS")
    print("=" * 100)

    for test_case in queries:
        query = test_case['query']
        expected_schema = test_case.get('schema_type', 'unknown')

        # Classify
        result = classifier.classify(query)

        # Check if correct
        is_correct = result.schema == expected_schema
        if is_correct:
            correct += 1
        total += 1

        # Display result
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Query {test_case['id']}: {query}")
        print(f"   Expected: {expected_schema}")
        print(f"   Got:      {result.schema} (confidence: {result.confidence:.2f})")
        if result.subject:
            print(f"   Subject:  {result.subject}")

        if verbose or not is_correct:
            print(f"   Indicators: {', '.join(result.indicators)}")

        results.append({
            'id': test_case['id'],
            'query': query,
            'expected': expected_schema,
            'predicted': result.schema,
            'confidence': result.confidence,
            'correct': is_correct,
            'subject': result.subject,
            'indicators': result.indicators
        })

    # Summary
    accuracy = correct / total if total > 0 else 0.0
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Accuracy: {correct}/{total} ({accuracy*100:.1f}%)")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")

    # Per-schema breakdown
    print("\nPer-Schema Accuracy:")
    schema_stats = {}
    for result in results:
        schema = result['expected']
        if schema not in schema_stats:
            schema_stats[schema] = {'correct': 0, 'total': 0}
        schema_stats[schema]['total'] += 1
        if result['correct']:
            schema_stats[schema]['correct'] += 1

    for schema, stats in sorted(schema_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {schema:20} {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"\nAverage Confidence: {avg_confidence:.2f}")

    print("\n" + "=" * 100)

    if accuracy >= 0.80:
        print("✅ PASS: Accuracy ≥80% (target achieved)")
    elif accuracy >= 0.70:
        print("⚠️  ACCEPTABLE: Accuracy ≥70% but <80% (room for improvement)")
    else:
        print("❌ FAIL: Accuracy <70% (needs improvement)")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description="Test schema classifier on Phase 0 queries",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--queries',
        type=str,
        default='data/test_queries/phase_0.jsonl',
        help='Path to test queries JSONL file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output including indicators'
    )

    parser.add_argument(
        '--explain',
        type=str,
        help='Explain classification for a specific query'
    )

    args = parser.parse_args()

    # Explain mode
    if args.explain:
        classifier = SchemaClassifier()
        print(classifier.explain(args.explain))
        return

    # Test mode
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"❌ Test queries not found: {queries_path}")
        sys.exit(1)

    accuracy, results = test_classifier(str(queries_path), args.verbose)

    # Exit with appropriate code
    sys.exit(0 if accuracy >= 0.80 else 1)


if __name__ == '__main__':
    main()
