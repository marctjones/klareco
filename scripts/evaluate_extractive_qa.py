#!/usr/bin/env python3
"""
Evaluate Extractive QA System on Test Set

Runs demo_extractive_qa.py logic on each test question and checks if answer contains expected keywords.

Usage:
    python scripts/evaluate_extractive_qa.py
    python scripts/evaluate_extractive_qa.py --no-m1 --no-rerank  # Deterministic baseline
    python scripts/evaluate_extractive_qa.py --limit 10           # Test first 10 questions
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.path.insert(0, str(Path(__file__).parent))  # Add scripts/ to path
from demo_extractive_qa import retrieve_sentences, expand_with_embeddings, extract_query_entity

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator, classify_question_type

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs during evaluation
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_keywords_in_text(text: str, keywords: List[str]) -> Dict:
    """
    Check if any keywords appear in text.

    Returns:
        {
            'found': bool,
            'matched_keywords': List[str],
            'match_count': int
        }
    """
    text_lower = text.lower()
    matched = []

    for kw in keywords:
        if kw.lower() in text_lower:
            matched.append(kw)

    return {
        'found': len(matched) > 0,
        'matched_keywords': matched,
        'match_count': len(matched)
    }


def evaluate_question(
    question: str,
    expected_keywords: List[str],
    generator: ExtractiveAnswerGenerator,
    retriever: WhooshRetriever,
    top_k: int = 20,
) -> Dict:
    """
    Run extractive QA on a single question and check if answer contains expected keywords.

    Returns:
        {
            'answer_text': str,
            'found_keywords': bool,
            'matched_keywords': List[str],
            'facts_extracted': int,
            'facts_selected': int
        }
    """
    # Parse question
    query_ast = parse(question)
    question_type = classify_question_type(question)
    query_entity = extract_query_entity(query_ast, question_type)

    # Extract roots from question
    roots = []
    def extract_roots(node):
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                # For proper names (capitalized words), use full word instead of root
                # This handles cases like "Lincoln" which parser might misparse as compound
                plena_vorto = node.get('plena_vorto', '')
                root = node.get('radiko', '')

                # Exclude question words and correlatives from being treated as proper names
                question_words = {'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel', 'kiom', 'kia', 'kies'}

                if plena_vorto and plena_vorto[0].isupper() and plena_vorto.lower() not in question_words:
                    # Proper name - use full word, strip Esperanto endings (-n, -j, -jn)
                    word = plena_vorto.rstrip('n').rstrip('j').rstrip('n')  # Strip -n, -jn
                    roots.append(word)
                elif root:
                    # Common word - use lowercase root
                    roots.append(root.lower())
            elif node.get('tipo') == 'vortgrupo':
                extract_roots(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract_roots(p)
            elif node.get('tipo') == 'frazo':
                extract_roots(node.get('subjekto'))
                extract_roots(node.get('verbo'))
                extract_roots(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract_roots(a)

    extract_roots(query_ast)

    # Manual synonyms for common words
    synonyms = {
        'fond': ['kre', 'establ', 'startig'],
        'est': ['est'],
    }

    query_roots = set(roots)
    for root in roots:
        if root in synonyms:
            query_roots.update(synonyms[root])

    # Expand with embeddings
    embeddings_path = Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt')
    if embeddings_path.exists():
        expanded = expand_with_embeddings(
            list(query_roots),
            embeddings_path,
            k=5,
            threshold=0.65  # Use updated threshold
        )
        query_roots = expanded

    # Retrieve sentences using AST role constraints
    sentences = retrieve_sentences(retriever, list(query_roots), question_type.value, query_entity, top_k, query_ast=query_ast)

    if not sentences:
        return {
            'answer_text': '',
            'found_keywords': False,
            'matched_keywords': [],
            'facts_extracted': 0,
            'facts_selected': 0,
            'error': 'No sentences retrieved'
        }

    # Generate answer
    answer = generator.generate(sentences, question, question_type=question_type, query_entity=query_entity)

    # Check if answer contains expected keywords
    keyword_check = check_keywords_in_text(answer.text, expected_keywords)

    return {
        'answer_text': answer.text[:200] + '...' if len(answer.text) > 200 else answer.text,
        'found_keywords': keyword_check['found'],
        'matched_keywords': keyword_check['matched_keywords'],
        'facts_extracted': answer.num_facts_extracted,
        'facts_selected': answer.num_facts_selected,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--no-m1', action='store_true', help='Disable M1 filtering')
    parser.add_argument('--no-rerank', action='store_true', help='Disable neural reranking')
    parser.add_argument('--single-span-types', type=str, nargs='+',
                       choices=['KIU', 'KIO', 'KIE', 'KIAM', 'KIAL', 'KIEL'],
                       help='Question types that should return single-span answers (Esperanto: kiu/kio/kie/kiam/kial/kiel)')
    parser.add_argument('--limit', type=int, help='Limit to first N questions')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                       help='Process N questions in parallel (default: 1 = sequential)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load test set
    print(f"Loading test set from {args.test_set}")
    test_questions = []
    with open(args.test_set) as f:
        for line in f:
            test_questions.append(json.loads(line))

    if args.limit:
        test_questions = test_questions[:args.limit]

    print(f"Evaluating {len(test_questions)} questions")
    print()

    # Initialize Whoosh retriever
    print("Loading Whoosh FTS index...")
    retriever = WhooshRetriever(
        whoosh_index_dir=Path('data/indexes/whoosh_fts'),
        kuzu_db_path=args.db
    )
    print("✓ Whoosh loaded\n")

    # Build multi_sentence_question_types dict from --single-span-types flag
    multi_sentence_config = None
    if args.single_span_types:
        from klareco.rag.importance_scorer import QuestionType

        # Map Esperanto question words to QuestionType enum
        eo_to_enum = {
            'KIU': QuestionType.WHO,
            'KIO': QuestionType.WHAT,
            'KIE': QuestionType.WHERE,
            'KIAM': QuestionType.WHEN,
            'KIAL': QuestionType.WHY,
            'KIEL': QuestionType.HOW,
        }

        single_span_enums = {eo_to_enum[word] for word in args.single_span_types}

        multi_sentence_config = {
            QuestionType.WHO: QuestionType.WHO not in single_span_enums,
            QuestionType.WHAT: QuestionType.WHAT not in single_span_enums,
            QuestionType.WHERE: QuestionType.WHERE not in single_span_enums,
            QuestionType.WHEN: QuestionType.WHEN not in single_span_enums,
            QuestionType.WHY: QuestionType.WHY not in single_span_enums,
            QuestionType.HOW: QuestionType.HOW not in single_span_enums,
            QuestionType.OTHER: True,
        }
        print(f"Single-span types: {args.single_span_types} (Esperanto question words)")

    # Initialize generator
    print("Loading extractive QA system...")
    generator = ExtractiveAnswerGenerator(
        use_reranker=not args.no_rerank,
        use_m1=not args.no_m1,
        multi_sentence_question_types=multi_sentence_config
    )
    print()

    # Evaluate each question (sequential or parallel)
    results = []
    correct = 0
    total = 0

    def evaluate_one_question(test_q, i):
        """Wrapper for evaluating a single question (for parallel processing)."""
        question = test_q['question']
        expected_keywords = test_q['expected_keywords']
        question_type = test_q['question_type']

        result = evaluate_question(
            question,
            expected_keywords,
            generator,
            retriever,
            args.top_k
        )

        return {
            'index': i,
            'question_id': test_q['id'],
            'question': question,
            'question_type': question_type,
            'expected_keywords': expected_keywords,
            **result
        }

    if args.parallel > 1:
        # Parallel processing
        print(f"Processing {len(test_questions)} questions with {args.parallel} workers in parallel\n")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all questions
            futures = {
                executor.submit(evaluate_one_question, test_q, i): i
                for i, test_q in enumerate(test_questions, 1)
            }

            # Process as they complete
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()

                total += 1
                if result['found_keywords']:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"[{result['index']}/{len(test_questions)}] {result['question_type']}: {result['question']}")
                print(f"  {status} Expected: {result['expected_keywords']}")
                print(f"    Matched: {result['matched_keywords']}")

                if args.verbose or not result['found_keywords']:
                    print(f"    Answer: {result['answer_text']}")

                print()
                results.append(result)

    else:
        # Sequential processing (original behavior)
        for i, test_q in enumerate(test_questions, 1):
            question = test_q['question']
            expected_keywords = test_q['expected_keywords']
            question_type = test_q['question_type']

            print(f"[{i}/{len(test_questions)}] {question_type}: {question}")

            result = evaluate_question(
                question,
                expected_keywords,
                generator,
                retriever,
                args.top_k
            )

            total += 1
            if result['found_keywords']:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            print(f"  {status} Expected: {expected_keywords}")
            print(f"    Matched: {result['matched_keywords']}")

            if args.verbose or not result['found_keywords']:
                print(f"    Answer: {result['answer_text']}")

            print()

            results.append({
                'question_id': test_q['id'],
                'question': question,
                'question_type': question_type,
                'expected_keywords': expected_keywords,
                **result
            })

    # Summary
    accuracy = (correct / total * 100) if total > 0 else 0
    print("=" * 80)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)

    # Breakdown by question type
    by_type = {}
    for r in results:
        qtype = r['question_type']
        if qtype not in by_type:
            by_type[qtype] = {'correct': 0, 'total': 0}
        by_type[qtype]['total'] += 1
        if r['found_keywords']:
            by_type[qtype]['correct'] += 1

    print("\nBy Question Type:")
    for qtype, stats in sorted(by_type.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")


if __name__ == '__main__':
    main()
