#!/usr/bin/env python3
"""
Ablation Study: Importance Scoring Components

Tests individual contribution of each improvement:
- Baseline: Old scoring (no proper noun detection, no embeddings)
- Phase 1: Proper noun detection + exact root matching
- Phase 2: Embedding similarity
- Phase 1+2: Both combined

Usage:
    python scripts/ablation_study_importance_scoring.py --limit 10
    python scripts/ablation_study_importance_scoring.py  # Full 50 questions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator
from klareco.rag.importance_scorer import FactImportanceScorer, classify_question_type
from scripts.demo_extractive_qa import extract_roots_from_ast, extract_query_entity

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner output
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(test_questions, retriever, use_embeddings: bool,
                   use_proper_noun_detection: bool, top_k: int = 30):
    """
    Run evaluation with specific configuration.

    Args:
        test_questions: List of test questions
        retriever: WhooshRetriever instance
        use_embeddings: Whether to use embedding similarity
        use_proper_noun_detection: Whether to use proper noun detection
        top_k: Number of documents to retrieve

    Returns:
        Dict with results
    """
    # Initialize generator with specific configuration
    generator = ExtractiveAnswerGenerator(
        use_reranker=False,
        use_m1=False
    )

    # Override importance scorer with specific configuration
    generator.importance_scorer = FactImportanceScorer(
        use_embeddings=use_embeddings
    )

    # If disabling proper noun detection, we need to patch the entity matching
    # (This is a hack for ablation - in production we'd have a cleaner API)
    if not use_proper_noun_detection:
        # Monkey-patch _entity_matches to ignore proper noun info
        original_entity_matches = generator.importance_scorer._entity_matches

        def baseline_entity_matches(query_entity: str, fact, exact: bool = True):
            """Baseline matching without proper noun awareness."""
            fact_entity = fact.entity if hasattr(fact, 'entity') else str(fact)
            query_lower = query_entity.lower()
            fact_lower = fact_entity.lower()

            if exact:
                # Old behavior: substring matching at word boundary
                return (fact_lower.startswith(query_lower) and
                        (len(fact_lower) == len(query_lower) or
                         fact_lower[len(query_lower)] in 'oaej'))
            else:
                return query_lower in fact_lower

        generator.importance_scorer._entity_matches = baseline_entity_matches

    # Run evaluation
    results = []
    for item in test_questions:
        question = item['question']
        expected = item['expected_keywords'][0]

        # Parse query
        query_ast = parse(question)
        roots = extract_roots_from_ast(query_ast)
        question_type = classify_question_type(question)
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

        # Retrieve
        try:
            documents = retriever.retrieve(
                query_roots=list(roots),
                top_k=top_k,
                retrieval_limit=200,
                question_type=question_type.value,
                query_entity=entity_root,
                query_ast=query_ast
            )

            # Check if answer in retrieved documents
            answer_in_docs = any(expected.lower() in doc.get('text', '').lower()
                                 for doc in documents)

            if not answer_in_docs:
                # Retrieval failure
                results.append({'success': False, 'stage': 'retrieval'})
                continue

            # Generate answer
            answer = generator.generate(
                documents,
                question,
                question_type=question_type,
                query_entity=query_entity,
                max_facts=4
            )

            # Check if answer correct
            answer_correct = expected.lower() in answer.text.lower()
            results.append({'success': answer_correct, 'stage': 'generation' if not answer_correct else 'success'})

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            results.append({'success': False, 'stage': 'error'})

    # Compute metrics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    retrieval_failed = sum(1 for r in results if r['stage'] == 'retrieval')
    generation_failed = sum(1 for r in results if r['stage'] == 'generation')

    retrieval_success = total - retrieval_failed
    generation_success = successful

    return {
        'total': total,
        'successful': successful,
        'accuracy': successful / total if total > 0 else 0,
        'retrieval_success_rate': retrieval_success / total if total > 0 else 0,
        'generation_success_rate': generation_success / retrieval_success if retrieval_success > 0 else 0,
        'retrieval_failures': retrieval_failed,
        'generation_failures': generation_failed
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--whoosh-index', type=Path, default=Path('data/indexes/whoosh_fts'))
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--limit', type=int, help='Limit to first N questions')

    args = parser.parse_args()

    # Load test set
    print(f"Loading test set from {args.test_set}")
    test_questions = []
    with open(args.test_set) as f:
        for line in f:
            test_questions.append(json.loads(line))

    if args.limit:
        test_questions = test_questions[:args.limit]

    print(f"Loaded {len(test_questions)} questions\n")

    # Initialize retriever (shared across all runs)
    print("Initializing retriever...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.db
    )
    print("✓ Initialized\n")

    # Run ablation study
    configurations = [
        {
            'name': 'Baseline',
            'description': 'No improvements (old substring matching, no embeddings)',
            'use_embeddings': False,
            'use_proper_noun_detection': False
        },
        {
            'name': 'Phase 1 Only',
            'description': 'Proper noun detection + exact root matching',
            'use_embeddings': False,
            'use_proper_noun_detection': True
        },
        {
            'name': 'Phase 2 Only',
            'description': 'Embedding similarity (10% weight)',
            'use_embeddings': True,
            'use_proper_noun_detection': False
        },
        {
            'name': 'Phase 1+2 Combined',
            'description': 'All improvements together',
            'use_embeddings': True,
            'use_proper_noun_detection': True
        }
    ]

    print("=" * 80)
    print("ABLATION STUDY: Importance Scoring Components")
    print("=" * 80)
    print()

    all_results = {}

    for config in configurations:
        print(f"Testing: {config['name']}")
        print(f"  {config['description']}")

        results = run_evaluation(
            test_questions,
            retriever,
            use_embeddings=config['use_embeddings'],
            use_proper_noun_detection=config['use_proper_noun_detection'],
            top_k=args.top_k
        )

        all_results[config['name']] = results

        print(f"  ✓ Accuracy: {results['accuracy']*100:.1f}% ({results['successful']}/{results['total']})")
        print(f"  ✓ Retrieval: {results['retrieval_success_rate']*100:.1f}%")
        print(f"  ✓ Generation (given retrieval): {results['generation_success_rate']*100:.1f}%")
        print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Configuration':<25} {'Accuracy':<12} {'Retrieval':<12} {'Generation':<12}")
    print("-" * 80)

    for config in configurations:
        results = all_results[config['name']]
        print(f"{config['name']:<25} "
              f"{results['accuracy']*100:>5.1f}%      "
              f"{results['retrieval_success_rate']*100:>5.1f}%      "
              f"{results['generation_success_rate']*100:>5.1f}%")

    print()

    # Calculate improvements
    baseline = all_results['Baseline']
    phase1 = all_results['Phase 1 Only']
    phase2 = all_results['Phase 2 Only']
    combined = all_results['Phase 1+2 Combined']

    print("=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print()

    print(f"Phase 1 improvement: {(phase1['accuracy'] - baseline['accuracy'])*100:+.1f}% absolute")
    print(f"Phase 2 improvement: {(phase2['accuracy'] - baseline['accuracy'])*100:+.1f}% absolute")
    print(f"Combined improvement: {(combined['accuracy'] - baseline['accuracy'])*100:+.1f}% absolute")
    print()

    print(f"Generation quality improvement (Phase 1): "
          f"{(phase1['generation_success_rate'] - baseline['generation_success_rate'])*100:+.1f}% absolute")
    print(f"Generation quality improvement (Combined): "
          f"{(combined['generation_success_rate'] - baseline['generation_success_rate'])*100:+.1f}% absolute")
    print()

    # Save results
    output_file = Path('results/ablation_study_importance_scoring.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
