#!/usr/bin/env python3
"""
Evaluate RAG System Against Test Set

Runs the RAG pipeline on each test question and compares output to expected answers.

Usage:
    # Evaluate current RAG system
    python scripts/evaluate_rag_test_set.py \\
        --test-set data/evaluation/rag_test_set.jsonl \\
        --output data/evaluation/rag_results.jsonl

    # Filter by category
    python scripts/evaluate_rag_test_set.py \\
        --category factual_simple \\
        --category grammar

    # Filter by expected performance
    python scripts/evaluate_rag_test_set.py \\
        --expected works  # Only test questions that should work
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_set(
    test_set_path: Path,
    categories: Optional[List[str]] = None,
    expected_performance: Optional[str] = None
) -> List[Dict]:
    """Load test set with optional filtering."""
    questions = []

    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)

            # Filter by category
            if categories and q['category'] not in categories:
                continue

            # Filter by expected performance
            if expected_performance and q['expected_performance'] != expected_performance:
                continue

            questions.append(q)

    return questions


def run_rag_query(question: str) -> Dict:
    """
    Run RAG pipeline on a single question.

    Returns:
        {
            'answer': str,
            'retrieved_passages': List[Dict],
            'confidence': float
        }
    """
    # TODO: Integrate with actual RAG pipeline
    # For now, return placeholder

    return {
        'answer': "[RAG pipeline not yet integrated]",
        'retrieved_passages': [],
        'confidence': 0.0,
        'error': "RAG pipeline integration pending"
    }


def evaluate_answer(result: Dict, expected: Dict) -> Dict:
    """
    Evaluate RAG answer against expected answer.

    Returns:
        {
            'correct': bool,
            'partial': bool,
            'confidence': float,
            'notes': str
        }
    """
    # TODO: Implement evaluation logic
    # For now, return placeholder

    return {
        'correct': False,
        'partial': False,
        'confidence': 0.0,
        'notes': "Evaluation logic not yet implemented"
    }


def run_evaluation(
    test_set: List[Dict],
    output_path: Optional[Path] = None
) -> Dict:
    """
    Run evaluation on test set.

    Returns summary statistics.
    """
    results = []
    stats = {
        'total': len(test_set),
        'correct': 0,
        'partial': 0,
        'incorrect': 0,
        'errors': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0}),
        'by_performance': defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0}),
    }

    print(f"Running evaluation on {len(test_set)} questions...")
    print()

    for i, test_q in enumerate(test_set, 1):
        question_id = test_q['id']
        question = test_q['question']
        category = test_q['category']
        expected_perf = test_q['expected_performance']

        print(f"[{i}/{len(test_set)}] {question_id}: {question}")

        # Run RAG
        try:
            rag_result = run_rag_query(question)
            evaluation = evaluate_answer(rag_result, test_q)

            result = {
                'question_id': question_id,
                'question': question,
                'category': category,
                'expected_performance': expected_perf,
                'rag_answer': rag_result.get('answer'),
                'expected_answer_pattern': test_q['expected_answer_pattern'],
                'evaluation': evaluation,
                'retrieved_passages': rag_result.get('retrieved_passages', []),
            }

            # Update stats
            if evaluation['correct']:
                stats['correct'] += 1
                stats['by_category'][category]['correct'] += 1
                stats['by_performance'][expected_perf]['correct'] += 1
            elif evaluation['partial']:
                stats['partial'] += 1
                stats['by_category'][category]['partial'] += 1
                stats['by_performance'][expected_perf]['partial'] += 1
            else:
                stats['incorrect'] += 1

            stats['by_category'][category]['total'] += 1
            stats['by_performance'][expected_perf]['total'] += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            result = {
                'question_id': question_id,
                'question': question,
                'category': category,
                'error': str(e)
            }
            stats['errors'] += 1

        results.append(result)
        print()

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"✓ Results saved to: {output_path}")

    return stats, results


def print_statistics(stats: Dict):
    """Print evaluation statistics."""
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()

    total = stats['total']
    correct = stats['correct']
    partial = stats['partial']
    incorrect = stats['incorrect']
    errors = stats['errors']

    accuracy = (correct / total * 100) if total > 0 else 0
    partial_accuracy = ((correct + partial) / total * 100) if total > 0 else 0

    print(f"Overall:")
    print(f"  Total questions: {total}")
    print(f"  ✅ Correct: {correct} ({accuracy:.1f}%)")
    print(f"  ⚠️  Partial: {partial}")
    print(f"  ❌ Incorrect: {incorrect}")
    print(f"  💥 Errors: {errors}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Partial accuracy: {partial_accuracy:.1f}%")
    print()

    print("By Category:")
    for category, cat_stats in sorted(stats['by_category'].items()):
        cat_total = cat_stats['total']
        cat_correct = cat_stats['correct']
        cat_partial = cat_stats['partial']
        cat_acc = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        print(f"  {category}: {cat_correct}/{cat_total} correct ({cat_acc:.0f}%), {cat_partial} partial")

    print()
    print("By Expected Performance:")
    for perf, perf_stats in sorted(stats['by_performance'].items()):
        perf_total = perf_stats['total']
        perf_correct = perf_stats['correct']
        perf_partial = perf_stats['partial']
        perf_acc = (perf_correct / perf_total * 100) if perf_total > 0 else 0
        symbol = {"works": "✅", "partial": "⚠️", "fails": "❌"}.get(perf, "?")
        print(f"  {symbol} {perf}: {perf_correct}/{perf_total} correct ({perf_acc:.0f}%), {perf_partial} partial")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system against test set"
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        default=Path('data/evaluation/rag_test_set.jsonl'),
        help='Path to test set'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/evaluation/rag_results.jsonl'),
        help='Output path for results'
    )
    parser.add_argument(
        '--category',
        action='append',
        help='Filter by category (can specify multiple times)'
    )
    parser.add_argument(
        '--expected',
        choices=['works', 'partial', 'fails'],
        help='Filter by expected performance'
    )

    args = parser.parse_args()

    # Load test set
    print("Loading test set...")
    test_set = load_test_set(
        args.test_set,
        categories=args.category,
        expected_performance=args.expected
    )

    if not test_set:
        print("No questions match filters")
        return

    print(f"Loaded {len(test_set)} questions")
    print()

    # Run evaluation
    stats, results = run_evaluation(test_set, args.output)

    # Print statistics
    print()
    print_statistics(stats)

    print()
    print("Next steps:")
    print("1. Integrate RAG pipeline (uncomment TODO in run_rag_query)")
    print("2. Implement evaluation logic (uncomment TODO in evaluate_answer)")
    print("3. Track progress as Stage 3/4 features are added")


if __name__ == '__main__':
    main()
