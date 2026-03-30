#!/usr/bin/env python3
"""
Evaluate Retrieval Accuracy at Different top-k Values

Measures whether correct answers appear in retrieved documents (retrieval-only metric).
Does NOT test extraction or generation - pure retrieval quality.

Usage:
    python scripts/evaluate_retrieval_accuracy.py --top-k 5,10,20,30,50
    python scripts/evaluate_retrieval_accuracy.py --verbose  # Show which documents contain answers
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import classify_question_type

# Import helper to extract roots from AST
sys.path.insert(0, str(Path(__file__).parent))
from demo_extractive_qa import extract_roots_from_ast, extract_query_entity

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_answer_in_documents(documents: List[Dict], expected_answer: str) -> Tuple[bool, int]:
    """
    Check if expected answer appears in any retrieved document.

    Args:
        documents: List of retrieved document dicts with 'text' field
        expected_answer: Expected answer keyword/phrase

    Returns:
        (found, rank): Whether answer was found, and at what rank (1-indexed, 0 if not found)
    """
    answer_lower = expected_answer.lower()

    for i, doc in enumerate(documents):
        text = doc.get('text', '').lower()
        if answer_lower in text:
            return True, i + 1  # 1-indexed rank

    return False, 0


def evaluate_retrieval_at_k(
    retriever: WhooshRetriever,
    test_questions: List[Dict],
    top_k: int,
    verbose: bool = False
) -> Dict:
    """
    Evaluate retrieval accuracy at a specific k value.

    Args:
        retriever: WhooshRetriever instance
        test_questions: List of test questions with 'question' and 'expected_keywords' fields
        top_k: Number of documents to retrieve
        verbose: Show detailed results per question

    Returns:
        Dict with evaluation metrics
    """
    results = []

    for i, item in enumerate(test_questions, 1):
        question = item['question']
        expected_keywords = item.get('expected_keywords', [])

        if not expected_keywords:
            logger.warning(f"Question {i} has no expected keywords, skipping")
            continue

        # Use first expected keyword as primary answer
        primary_answer = expected_keywords[0]

        # Parse query AST (required for AST-first retrieval)
        query_ast = parse(question)

        # Extract roots from AST
        roots = extract_roots_from_ast(query_ast)

        # Classify question type
        question_type = classify_question_type(question)
        question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)

        # Extract query entity (optional)
        query_entity = extract_query_entity(query_ast, question_type)

        # Strip endings to get root
        entity_root = None
        if query_entity:
            entity_root = query_entity.lower()
            if entity_root.endswith('jn'):
                entity_root = entity_root[:-2]
            elif entity_root.endswith('n') or entity_root.endswith('j'):
                entity_root = entity_root[:-1]
            if entity_root.endswith('o') or entity_root.endswith('a') or entity_root.endswith('e'):
                entity_root = entity_root[:-1]

        # Retrieve documents using AST-first retrieval
        try:
            documents = retriever.retrieve(
                query_roots=list(roots),
                top_k=top_k,
                retrieval_limit=200,
                question_type=question_type_str,
                query_entity=entity_root,
                query_ast=query_ast
            )
        except Exception as e:
            logger.error(f"Retrieval failed for question {i}: {e}")
            results.append({
                'question_id': i,
                'question': question,
                'expected': primary_answer,
                'found': False,
                'rank': 0,
                'num_retrieved': 0
            })
            continue

        # Check if answer is in retrieved documents
        found, rank = check_answer_in_documents(documents, primary_answer)

        results.append({
            'question_id': i,
            'question': question,
            'expected': primary_answer,
            'found': found,
            'rank': rank,
            'num_retrieved': len(documents)
        })

        if verbose:
            status = f"✓ Rank {rank}" if found else "✗ Not found"
            print(f"[{i}/{len(test_questions)}] {status}: {question[:60]}...")
            if found and rank <= 5:
                print(f"    Found in top-5 at rank {rank}")

    # Calculate metrics
    total = len(results)
    found_count = sum(1 for r in results if r['found'])
    recall = found_count / total if total > 0 else 0.0

    # Calculate recall@5, @10, @20 (if k is large enough)
    recall_at_5 = sum(1 for r in results if r['rank'] > 0 and r['rank'] <= 5) / total if top_k >= 5 else None
    recall_at_10 = sum(1 for r in results if r['rank'] > 0 and r['rank'] <= 10) / total if top_k >= 10 else None
    recall_at_20 = sum(1 for r in results if r['rank'] > 0 and r['rank'] <= 20) / total if top_k >= 20 else None

    # Calculate MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0 / r['rank'] for r in results if r['rank'] > 0]
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    return {
        'top_k': top_k,
        'total_questions': total,
        'found_count': found_count,
        'recall': recall,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10,
        'recall_at_20': recall_at_20,
        'mrr': mrr,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--whoosh-index', type=Path, default=Path('data/indexes/whoosh_fts'))
    parser.add_argument('--top-k', type=str, default='5,10,20,30,50',
                       help='Comma-separated list of k values to test (e.g., "5,10,20,30,50")')
    parser.add_argument('--output', type=Path, help='Output CSV file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed results')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Parse k values
    k_values = [int(k.strip()) for k in args.top_k.split(',')]

    print(f"Loading test set from {args.test_set}")
    test_questions = []
    with open(args.test_set) as f:
        for line in f:
            test_questions.append(json.loads(line))

    print(f"Loaded {len(test_questions)} questions")
    print(f"Testing k values: {k_values}\n")

    # Initialize retriever
    print("Initializing retriever...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.db
    )
    print("✓ Retriever initialized\n")

    # Evaluate at each k value
    all_results = []

    print("=" * 80)
    print("RETRIEVAL ACCURACY EVALUATION")
    print("=" * 80)
    print()

    for k in k_values:
        print(f"Evaluating at top-k={k}...")
        result = evaluate_retrieval_at_k(retriever, test_questions, k, verbose=args.verbose)
        all_results.append(result)

        print(f"  Recall@{k}: {result['recall']:.1%} ({result['found_count']}/{result['total_questions']})")
        if result['recall_at_5'] is not None:
            print(f"  Recall@5:  {result['recall_at_5']:.1%}")
        if result['recall_at_10'] is not None:
            print(f"  Recall@10: {result['recall_at_10']:.1%}")
        if result['recall_at_20'] is not None:
            print(f"  Recall@20: {result['recall_at_20']:.1%}")
        print(f"  MRR: {result['mrr']:.3f}")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'top-k':<10} {'Recall':<12} {'Recall@5':<12} {'Recall@10':<12} {'Recall@20':<12} {'MRR':<10}")
    print("-" * 80)

    for result in all_results:
        r5 = f"{result['recall_at_5']:.1%}" if result['recall_at_5'] is not None else "N/A"
        r10 = f"{result['recall_at_10']:.1%}" if result['recall_at_10'] is not None else "N/A"
        r20 = f"{result['recall_at_20']:.1%}" if result['recall_at_20'] is not None else "N/A"
        print(f"{result['top_k']:<10} {result['recall']:<12.1%} {r5:<12} {r10:<12} {r20:<12} {result['mrr']:<10.3f}")

    print()

    # Find optimal k
    print("RECOMMENDATIONS:")
    print()

    # Find k with highest recall
    best_recall = max(all_results, key=lambda x: x['recall'])
    print(f"  Best recall: top-k={best_recall['top_k']} with {best_recall['recall']:.1%}")

    # Find k with highest MRR (balances recall and rank)
    best_mrr = max(all_results, key=lambda x: x['mrr'])
    print(f"  Best MRR:    top-k={best_mrr['top_k']} with MRR={best_mrr['mrr']:.3f}")

    # Find smallest k that achieves 90% of max recall
    max_recall_value = best_recall['recall']
    threshold = 0.9 * max_recall_value
    good_k_values = [r for r in all_results if r['recall'] >= threshold]
    if good_k_values:
        smallest_good_k = min(good_k_values, key=lambda x: x['top_k'])
        print(f"  Smallest k achieving 90% of max recall: top-k={smallest_good_k['top_k']} ({smallest_good_k['recall']:.1%})")

    print()

    # Analyze recall gains
    print("RECALL GAINS:")
    for i in range(1, len(all_results)):
        prev = all_results[i-1]
        curr = all_results[i]
        gain = curr['recall'] - prev['recall']
        gain_pct = (gain / prev['recall'] * 100) if prev['recall'] > 0 else 0
        print(f"  {prev['top_k']} → {curr['top_k']}: +{gain:.1%} (relative +{gain_pct:.1f}%)")

    print()

    # Save results to CSV if requested
    if args.output:
        import csv

        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['top_k', 'recall', 'recall_at_5', 'recall_at_10', 'recall_at_20', 'mrr', 'found_count', 'total'])

            for result in all_results:
                writer.writerow([
                    result['top_k'],
                    f"{result['recall']:.4f}",
                    f"{result['recall_at_5']:.4f}" if result['recall_at_5'] is not None else '',
                    f"{result['recall_at_10']:.4f}" if result['recall_at_10'] is not None else '',
                    f"{result['recall_at_20']:.4f}" if result['recall_at_20'] is not None else '',
                    f"{result['mrr']:.4f}",
                    result['found_count'],
                    result['total_questions']
                ])

        print(f"Results saved to {args.output}")
        print()


if __name__ == '__main__':
    main()
