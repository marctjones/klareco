#!/usr/bin/env python3
"""
Diagnose Extraction Failures - Where and Why Does Extraction Fail?

Analyzes the QA pipeline to identify extraction failure patterns:
1. Retrieval succeeds but extraction fails
2. Extraction succeeds but wrong facts selected
3. Selection succeeds but answer generation fails

Usage:
    python scripts/diagnose_extraction_failures.py --verbose
    python scripts/diagnose_extraction_failures.py --limit 10  # Test first 10 questions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator, classify_question_type
from demo_extractive_qa import extract_roots_from_ast, extract_query_entity

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_answer_in_text(text: str, expected: str) -> bool:
    """Check if expected answer appears in text (case-insensitive)."""
    return expected.lower() in text.lower()


def diagnose_question(
    question: str,
    expected_answer: str,
    retriever: WhooshRetriever,
    generator: ExtractiveAnswerGenerator,
    top_k: int = 30
) -> Dict:
    """
    Diagnose a single question through the full pipeline.

    Returns dict with failure analysis at each stage.
    """
    result = {
        'question': question,
        'expected': expected_answer,
        'stages': {}
    }

    # Parse query
    query_ast = parse(question)
    question_type = classify_question_type(question)
    query_entity = extract_query_entity(query_ast, question_type)

    # Extract roots
    roots = extract_roots_from_ast(query_ast)

    # Strip endings from query_entity to get root
    entity_root = None
    if query_entity:
        entity_root = query_entity.lower()
        if entity_root.endswith('jn'):
            entity_root = entity_root[:-2]
        elif entity_root.endswith('n') or entity_root.endswith('j'):
            entity_root = entity_root[:-1]
        if entity_root.endswith('o') or entity_root.endswith('a') or entity_root.endswith('e'):
            entity_root = entity_root[:-1]

    question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)

    # STAGE 1: RETRIEVAL
    try:
        documents = retriever.retrieve(
            query_roots=list(roots),
            top_k=top_k,
            retrieval_limit=200,
            question_type=question_type_str,
            query_entity=entity_root,
            query_ast=query_ast
        )

        # Check if answer is in retrieved documents
        answer_found = False
        answer_rank = None
        for i, doc in enumerate(documents):
            if check_answer_in_text(doc.get('text', ''), expected_answer):
                answer_found = True
                answer_rank = i + 1
                break

        result['stages']['retrieval'] = {
            'success': answer_found,
            'num_docs': len(documents),
            'answer_rank': answer_rank,
            'sample_texts': [doc.get('text', '')[:100] + '...' for doc in documents[:3]]
        }

        if not answer_found:
            result['failure_stage'] = 'retrieval'
            result['failure_reason'] = 'Answer not in retrieved documents'
            return result

    except Exception as e:
        result['stages']['retrieval'] = {
            'success': False,
            'error': str(e)
        }
        result['failure_stage'] = 'retrieval'
        result['failure_reason'] = f'Retrieval error: {e}'
        return result

    # STAGE 2: EXTRACTION
    try:
        answer = generator.generate(
            documents,
            question,
            question_type=question_type,
            query_entity=query_entity,
            max_facts=4
        )

        # Check extraction
        result['stages']['extraction'] = {
            'num_facts_extracted': answer.num_facts_extracted,
            'num_facts_selected': answer.num_facts_selected,
            'facts_used': [str(f) for f in answer.facts_used[:3]]  # Show first 3
        }

        if answer.num_facts_extracted == 0:
            result['failure_stage'] = 'extraction'
            result['failure_reason'] = 'No facts extracted from documents'
            return result

        if answer.num_facts_selected == 0:
            result['failure_stage'] = 'selection'
            result['failure_reason'] = 'Facts extracted but none selected (filtered out)'
            return result

    except Exception as e:
        result['stages']['extraction'] = {
            'success': False,
            'error': str(e)
        }
        result['failure_stage'] = 'extraction'
        result['failure_reason'] = f'Extraction error: {e}'
        return result

    # STAGE 3: GENERATION
    generated_answer = answer.text
    answer_correct = check_answer_in_text(generated_answer, expected_answer)

    result['stages']['generation'] = {
        'success': answer_correct,
        'answer_text': generated_answer[:200] + '...' if len(generated_answer) > 200 else generated_answer,
        'contains_expected': answer_correct
    }

    if not answer_correct:
        result['failure_stage'] = 'generation'
        result['failure_reason'] = 'Answer generated but does not contain expected keyword'
        # Check if answer is in the selected facts
        answer_in_facts = any(check_answer_in_text(str(f), expected_answer) for f in answer.facts_used)
        if answer_in_facts:
            result['failure_reason'] += ' (answer was in selected facts but not in final text)'
        else:
            result['failure_reason'] += ' (answer was NOT in selected facts - selection error)'
        return result

    # SUCCESS!
    result['failure_stage'] = None
    result['failure_reason'] = None
    result['success'] = True

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--whoosh-index', type=Path, default=Path('data/indexes/whoosh_fts'))
    parser.add_argument('--top-k', type=int, default=30, help='Number of documents to retrieve')
    parser.add_argument('--limit', type=int, help='Limit to first N questions')
    parser.add_argument('--verbose', action='store_true', help='Show detailed results per question')

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

    # Initialize retriever and generator
    print("Initializing retriever and generator...")
    retriever = WhooshRetriever(
        whoosh_index_dir=args.whoosh_index,
        kuzu_db_path=args.db
    )
    generator = ExtractiveAnswerGenerator(
        use_reranker=False,
        use_m1=False
    )
    print("✓ Initialized\n")

    # Diagnose each question
    results = []
    failure_counts = {
        'retrieval': 0,
        'extraction': 0,
        'selection': 0,
        'generation': 0,
        'success': 0
    }

    print("=" * 80)
    print("DIAGNOSING EXTRACTION FAILURES")
    print("=" * 80)
    print()

    for i, item in enumerate(test_questions, 1):
        question = item['question']
        expected = item.get('expected_keywords', [''])[0]

        if args.verbose:
            print(f"[{i}/{len(test_questions)}] {question}")

        result = diagnose_question(question, expected, retriever, generator, args.top_k)
        results.append(result)

        # Count failures
        failure_stage = result.get('failure_stage')
        if failure_stage:
            failure_counts[failure_stage] += 1
            if args.verbose:
                print(f"  ✗ Failed at {failure_stage}: {result['failure_reason']}")
        else:
            failure_counts['success'] += 1
            if args.verbose:
                print(f"  ✓ Success")

        if args.verbose:
            print()

    # Summary
    total = len(results)
    print()
    print("=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total questions: {total}")
    print(f"Successful: {failure_counts['success']} ({failure_counts['success']/total*100:.1f}%)")
    print()
    print("Failures by stage:")
    print(f"  Retrieval: {failure_counts['retrieval']} ({failure_counts['retrieval']/total*100:.1f}%)")
    print(f"  Extraction: {failure_counts['extraction']} ({failure_counts['extraction']/total*100:.1f}%)")
    print(f"  Selection: {failure_counts['selection']} ({failure_counts['selection']/total*100:.1f}%)")
    print(f"  Generation: {failure_counts['generation']} ({failure_counts['generation']/total*100:.1f}%)")
    print()

    # Calculate pipeline success rates
    retrieval_success = total - failure_counts['retrieval']
    extraction_success = retrieval_success - failure_counts['extraction']
    selection_success = extraction_success - failure_counts['selection']
    generation_success = selection_success - failure_counts['generation']

    print("Pipeline success rates (conditional on previous stage):")
    print(f"  Retrieval: {retrieval_success}/{total} = {retrieval_success/total*100:.1f}%")
    print(f"  Extraction (given retrieval): {extraction_success}/{retrieval_success} = {extraction_success/retrieval_success*100:.1f}%" if retrieval_success > 0 else "  Extraction: N/A")
    print(f"  Selection (given extraction): {selection_success}/{extraction_success} = {selection_success/extraction_success*100:.1f}%" if extraction_success > 0 else "  Selection: N/A")
    print(f"  Generation (given selection): {generation_success}/{selection_success} = {generation_success/selection_success*100:.1f}%" if selection_success > 0 else "  Generation: N/A")
    print()

    # Analyze extraction failures in detail
    extraction_failures = [r for r in results if r.get('failure_stage') == 'extraction']
    if extraction_failures:
        print("=" * 80)
        print(f"EXTRACTION FAILURE DETAILS ({len(extraction_failures)} cases)")
        print("=" * 80)
        print()

        for i, failure in enumerate(extraction_failures[:10], 1):  # Show first 10
            print(f"{i}. {failure['question']}")
            print(f"   Expected: {failure['expected']}")
            print(f"   Retrieved docs: {failure['stages']['retrieval']['num_docs']}")
            print(f"   Answer in docs at rank: {failure['stages']['retrieval'].get('answer_rank')}")
            if 'extraction' in failure['stages'] and 'num_facts_extracted' in failure['stages']['extraction']:
                print(f"   Facts extracted: {failure['stages']['extraction']['num_facts_extracted']}")
            print(f"   Reason: {failure['failure_reason']}")
            print()

    # Analyze selection failures in detail
    selection_failures = [r for r in results if r.get('failure_stage') == 'selection']
    if selection_failures:
        print("=" * 80)
        print(f"SELECTION FAILURE DETAILS ({len(selection_failures)} cases)")
        print("=" * 80)
        print()

        for i, failure in enumerate(selection_failures[:10], 1):  # Show first 10
            print(f"{i}. {failure['question']}")
            print(f"   Expected: {failure['expected']}")
            print(f"   Facts extracted: {failure['stages']['extraction']['num_facts_extracted']}")
            print(f"   Facts selected: {failure['stages']['extraction']['num_facts_selected']}")
            print(f"   Reason: {failure['failure_reason']}")
            print()

    # Save detailed results
    output_file = Path('results/extraction_diagnosis.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Detailed results saved to {output_file}")
    print()


if __name__ == '__main__':
    main()
