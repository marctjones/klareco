#!/usr/bin/env python3
"""
Generate Ground Truth Candidates for RAG Test Set

Finds top-K candidate answer sentences from corpus for each test question
using current retrieval pipeline.

Usage:
    # Generate candidates using structural retrieval
    python scripts/annotate_rag_ground_truth.py \\
        --test-set data/evaluation/rag_test_set.jsonl \\
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \\
        --output data/evaluation/rag_ground_truth_candidates.jsonl \\
        --top-k 100

    # Quick test on "works" questions only
    python scripts/annotate_rag_ground_truth.py \\
        --expected works \\
        --top-k 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_set(test_set_path: Path, expected_filter: str = None) -> List[Dict]:
    """Load test questions, optionally filtered."""
    questions = []

    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)

            if expected_filter and q['expected_performance'] != expected_filter:
                continue

            questions.append(q)

    return questions


def retrieve_candidates(question: str, corpus_path: Path, top_k: int = 100) -> List[Dict]:
    """
    Retrieve top-K candidate answer sentences from corpus.

    TODO: Integrate with actual retrieval pipeline
    - Option 1: Use existing RAG retriever
    - Option 2: Simple TF-IDF/BM25 baseline
    - Option 3: Structural AST-based retrieval

    Returns:
        List of candidates with scores and metadata
    """
    # PLACEHOLDER: Replace with actual retrieval

    # For now, return empty list
    # When implemented, should return:
    # [
    #     {
    #         'sentence_id': 'wikipedia:Esperanto:0:15',
    #         'text': 'Esperanto estas planlingvo...',
    #         'source': {'name': 'wikipedia', 'quality': 'BRONZE', ...},
    #         'parse_rate': 0.95,
    #         'retrieval_score': 0.85,
    #         'retrieval_rank': 1,
    #         'ast': {...}
    #     },
    #     ...
    # ]

    return []


def generate_candidates(
    test_set: List[Dict],
    corpus_path: Path,
    top_k: int = 100
) -> List[Dict]:
    """Generate retrieval candidates for each test question."""

    results = []

    print(f"Generating candidates for {len(test_set)} questions...")
    print(f"Top-K: {top_k}")
    print()

    for i, test_q in enumerate(test_set, 1):
        question_id = test_q['id']
        question = test_q['question']
        category = test_q['category']

        print(f"[{i}/{len(test_set)}] {question_id}: {question}")

        # Retrieve candidates
        candidates = retrieve_candidates(question, corpus_path, top_k)

        result = {
            'question_id': question_id,
            'question': question,
            'category': category,
            'expected_performance': test_q['expected_performance'],
            'expected_sources': test_q['expected_sources'],
            'candidates': candidates,
            'num_candidates': len(candidates)
        }

        results.append(result)

        print(f"  Retrieved: {len(candidates)} candidates")
        if candidates:
            print(f"  Top score: {candidates[0]['retrieval_score']:.3f}")
            print(f"  Top source: {candidates[0]['source']['name']}")
        print()

    return results


def save_candidates(results: List[Dict], output_path: Path):
    """Save candidates to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"✓ Saved candidates to: {output_path}")

    # Statistics
    total_questions = len(results)
    total_candidates = sum(r['num_candidates'] for r in results)
    avg_candidates = total_candidates / total_questions if total_questions > 0 else 0

    print()
    print(f"Statistics:")
    print(f"  Questions: {total_questions}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Avg per question: {avg_candidates:.1f}")

    # By category
    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = {'count': 0, 'candidates': 0}
        by_category[cat]['count'] += 1
        by_category[cat]['candidates'] += r['num_candidates']

    print()
    print("By category:")
    for cat, stats in sorted(by_category.items()):
        avg = stats['candidates'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {cat}: {stats['count']} questions, avg {avg:.1f} candidates")


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval candidates for RAG ground truth annotation"
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        default=Path('data/evaluation/rag_test_set.jsonl'),
        help='Path to test set'
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Path to corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/evaluation/rag_ground_truth_candidates.jsonl'),
        help='Output path for candidates'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of candidates to retrieve per question'
    )
    parser.add_argument(
        '--expected',
        choices=['works', 'partial', 'fails'],
        help='Filter by expected performance'
    )

    args = parser.parse_args()

    # Check corpus exists
    if not args.corpus.exists():
        print(f"❌ Corpus not found: {args.corpus}")
        print("   Run corpus rebuild first: ./scripts/rebuild_corpus.sh")
        return 1

    # Load test set
    print("Loading test set...")
    test_set = load_test_set(args.test_set, args.expected)

    if not test_set:
        print("No questions match filter")
        return 1

    print(f"Loaded {len(test_set)} questions")
    print()

    # Generate candidates
    results = generate_candidates(test_set, args.corpus, args.top_k)

    # Save
    save_candidates(results, args.output)

    print()
    print("Next steps:")
    print("1. TODO: Implement retrieval in retrieve_candidates()")
    print("2. Review candidates: cat data/evaluation/rag_ground_truth_candidates.jsonl | jq '.'")
    print("3. Annotate relevance: python scripts/annotate_ground_truth_ui.py")
    print("4. Generate final ground truth: python scripts/finalize_ground_truth.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
