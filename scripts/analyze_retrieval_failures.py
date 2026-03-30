#!/usr/bin/env python3
"""
Analyze Retrieval Failures - Deep Diagnosis of Why Retrieval Fails

Investigates the 66% retrieval failure rate to identify:
1. Corpus gaps (answer not in corpus)
2. AST pattern mismatches (pattern doesn't match sentence structure)
3. Synonym gaps (query and corpus use different words)
4. Ranking failures (answer retrieved but ranked too low)

Usage:
    python scripts/analyze_retrieval_failures.py --limit 20
    python scripts/analyze_retrieval_failures.py  # Full 50 questions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from scripts.demo_extractive_qa import extract_roots_from_ast, extract_query_entity
from klareco.rag.importance_scorer import classify_question_type

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_answer_in_text(text: str, expected: str) -> bool:
    """Check if expected answer appears in text (case-insensitive)."""
    return expected.lower() in text.lower()


def analyze_retrieval_failure(question: str, expected: str, retriever: WhooshRetriever,
                               corpus_db) -> Dict:
    """
    Deep analysis of why retrieval failed for a question.

    Returns dict with:
    - failure_type: 'corpus_gap', 'pattern_mismatch', 'synonym_gap', 'ranking_failure', 'success'
    - details: Specific diagnostic information
    """
    result = {
        'question': question,
        'expected': expected,
        'failure_type': None,
        'details': {}
    }

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

    result['details']['query_roots'] = list(roots)
    result['details']['query_entity'] = entity_root
    result['details']['question_type'] = question_type.value

    # Step 1: Test retrieval at different k values
    for top_k in [10, 30, 100]:
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
            answer_found = False
            answer_rank = None
            for i, doc in enumerate(documents):
                if check_answer_in_text(doc.get('text', ''), expected):
                    answer_found = True
                    answer_rank = i + 1
                    break

            result['details'][f'retrieval@{top_k}'] = {
                'found': answer_found,
                'rank': answer_rank,
                'num_docs': len(documents)
            }

            if answer_found and top_k == 30:
                # Found at k=30 - check if it's a ranking issue
                if answer_rank <= 10:
                    result['failure_type'] = 'success'
                    return result
                else:
                    result['failure_type'] = 'ranking_failure'
                    result['details']['ranking_issue'] = f"Answer at rank {answer_rank} (too low)"

        except Exception as e:
            result['details'][f'retrieval@{top_k}'] = {'error': str(e)}

    # Step 2: Check if answer exists in corpus at all (full-text search)
    try:
        # Use Whoosh BM25 search (no AST constraints)
        bm25_results = retriever._search_whoosh_bm25(list(roots), limit=200)

        corpus_has_answer = False
        for doc in bm25_results:
            if check_answer_in_text(doc.get('text', ''), expected):
                corpus_has_answer = True
                break

        result['details']['corpus_has_answer'] = corpus_has_answer

        if not corpus_has_answer:
            result['failure_type'] = 'corpus_gap'
            result['details']['diagnosis'] = "Answer not in corpus (even with BM25 full-text search)"
            return result

    except Exception as e:
        result['details']['corpus_check_error'] = str(e)

    # Step 3: If answer is in corpus but not retrieved, it's likely a pattern/synonym issue
    if result['details'].get('corpus_has_answer', False):
        # Check if AST retrieval returned anything
        retrieval_30 = result['details'].get('retrieval@30', {})
        if retrieval_30.get('num_docs', 0) == 0:
            result['failure_type'] = 'pattern_mismatch'
            result['details']['diagnosis'] = "AST pattern doesn't match any corpus sentences"
        elif not retrieval_30.get('found', False):
            result['failure_type'] = 'synonym_gap'
            result['details']['diagnosis'] = "AST retrieval succeeds but answer not in results (likely synonym mismatch)"
        else:
            # This shouldn't happen but handle it
            result['failure_type'] = 'unknown'

    # Ensure failure_type is set
    if result['failure_type'] is None:
        result['failure_type'] = 'unknown'
        result['details']['diagnosis'] = 'Unable to determine failure cause'

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--whoosh-index', type=Path, default=Path('data/indexes/whoosh_fts'))
    parser.add_argument('--limit', type=int, help='Limit to first N questions')
    parser.add_argument('--verbose', action='store_true', help='Show details per question')

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

    # Initialize retriever
    print("Initializing retriever...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.db
    )
    print("✓ Initialized\n")

    # Analyze each question
    print("=" * 80)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 80)
    print()

    results = []
    failure_counts = defaultdict(int)

    for i, item in enumerate(test_questions, 1):
        question = item['question']
        expected = item.get('expected_keywords', [''])[0]

        if args.verbose:
            print(f"[{i}/{len(test_questions)}] {question}")

        result = analyze_retrieval_failure(question, expected, retriever, args.db)
        results.append(result)

        failure_type = result['failure_type']
        failure_counts[failure_type] += 1

        if args.verbose:
            if failure_type == 'success':
                rank = result['details'].get('retrieval@30', {}).get('rank', '?')
                print(f"  ✓ Success (rank {rank})")
            else:
                print(f"  ✗ {failure_type}: {result['details'].get('diagnosis', 'See details')}")
            print()

    # Summary statistics
    total = len(results)
    print()
    print("=" * 80)
    print("FAILURE TYPE BREAKDOWN")
    print("=" * 80)
    print()

    print(f"Total questions: {total}")
    print()

    # Sort by frequency
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

    for failure_type, count in sorted_failures:
        pct = count / total * 100
        if failure_type == 'success':
            emoji = "✓"
        else:
            emoji = "✗"

        # Handle None failure_type
        type_str = failure_type if failure_type else "unknown"
        print(f"  {emoji} {type_str:20s}: {count:3d} ({pct:5.1f}%)")

    # Detailed analysis by failure type
    print()
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Corpus gaps
    corpus_gaps = [r for r in results if r['failure_type'] == 'corpus_gap']
    if corpus_gaps:
        print()
        print(f"CORPUS GAPS ({len(corpus_gaps)} questions)")
        print("-" * 80)
        print("These questions cannot be answered because the answer is not in the corpus.")
        print()
        for i, r in enumerate(corpus_gaps[:5], 1):
            print(f"{i}. {r['question']}")
            print(f"   Expected: {r['expected']}")
            print(f"   Query roots: {r['details']['query_roots']}")
        if len(corpus_gaps) > 5:
            print(f"   ... and {len(corpus_gaps) - 5} more")
        print()

    # Pattern mismatches
    pattern_mismatches = [r for r in results if r['failure_type'] == 'pattern_mismatch']
    if pattern_mismatches:
        print()
        print(f"PATTERN MISMATCHES ({len(pattern_mismatches)} questions)")
        print("-" * 80)
        print("AST grammatical pattern doesn't match any corpus sentences.")
        print("Possible solutions: Relax pattern matching, add pattern variants")
        print()
        for i, r in enumerate(pattern_mismatches[:5], 1):
            print(f"{i}. {r['question']}")
            print(f"   Expected: {r['expected']}")
            print(f"   Query roots: {r['details']['query_roots']}")
            print(f"   Question type: {r['details']['question_type']}")
        if len(pattern_mismatches) > 5:
            print(f"   ... and {len(pattern_mismatches) - 5} more")
        print()

    # Synonym gaps
    synonym_gaps = [r for r in results if r['failure_type'] == 'synonym_gap']
    if synonym_gaps:
        print()
        print(f"SYNONYM GAPS ({len(synonym_gaps)} questions)")
        print("-" * 80)
        print("AST retrieval succeeds but answer not in results (likely different verbs/roots).")
        print("Possible solutions: Expand synonym dictionary, use semantic similarity")
        print()
        for i, r in enumerate(synonym_gaps[:5], 1):
            print(f"{i}. {r['question']}")
            print(f"   Expected: {r['expected']}")
            print(f"   Query roots: {r['details']['query_roots']}")
        if len(synonym_gaps) > 5:
            print(f"   ... and {len(synonym_gaps) - 5} more")
        print()

    # Ranking failures
    ranking_failures = [r for r in results if r['failure_type'] == 'ranking_failure']
    if ranking_failures:
        print()
        print(f"RANKING FAILURES ({len(ranking_failures)} questions)")
        print("-" * 80)
        print("Answer retrieved but ranked too low (rank > 10).")
        print("Possible solutions: Improve semantic ranking, boost relevant patterns")
        print()
        for i, r in enumerate(ranking_failures[:5], 1):
            rank = r['details'].get('retrieval@30', {}).get('rank', '?')
            print(f"{i}. {r['question']} (rank {rank})")
            print(f"   Expected: {r['expected']}")
            print(f"   Query roots: {r['details']['query_roots']}")
        if len(ranking_failures) > 5:
            print(f"   ... and {len(ranking_failures) - 5} more")
        print()

    # Retrieval@k analysis
    print()
    print("=" * 80)
    print("RETRIEVAL@K ANALYSIS")
    print("=" * 80)
    print()

    for k in [10, 30, 100]:
        found_count = sum(1 for r in results
                         if r['details'].get(f'retrieval@{k}', {}).get('found', False))
        pct = found_count / total * 100
        print(f"  Retrieval@{k:3d}: {found_count:3d}/{total} ({pct:5.1f}%)")

    # Recommendations
    print()
    print("=" * 80)
    print("RECOMMENDATIONS (Priority Order)")
    print("=" * 80)
    print()

    if corpus_gaps:
        print(f"1. ADDRESS CORPUS GAPS ({len(corpus_gaps)} questions, {len(corpus_gaps)/total*100:.1f}%)")
        print("   - Expand corpus with more diverse sources")
        print("   - OR: Revise test set to match corpus coverage")
        print()

    if pattern_mismatches:
        print(f"2. FIX PATTERN MISMATCHES ({len(pattern_mismatches)} questions, {len(pattern_mismatches)/total*100:.1f}%)")
        print("   - Relax AST pattern matching (allow more grammatical variants)")
        print("   - Add fallback to BM25 when AST retrieval returns 0 results")
        print("   - Support more question patterns (passive voice, etc.)")
        print()

    if synonym_gaps:
        print(f"3. EXPAND SYNONYM COVERAGE ({len(synonym_gaps)} questions, {len(synonym_gaps)/total*100:.1f}%)")
        print("   - Add more verb synonyms (krei ≈ fondi ≈ establi)")
        print("   - Use ReVo dictionary for automatic synonym expansion")
        print("   - Consider learned semantic similarity (if fixed)")
        print()

    if ranking_failures:
        print(f"4. IMPROVE RANKING ({len(ranking_failures)} questions, {len(ranking_failures)/total*100:.1f}%)")
        print("   - Boost facts with exact entity match")
        print("   - Prioritize direct answer patterns")
        print("   - Reduce weight of generic facts")
        print()

    # Save detailed results
    output_file = Path('results/retrieval_failure_analysis.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Detailed results saved to {output_file}")
    print()


if __name__ == '__main__':
    main()
