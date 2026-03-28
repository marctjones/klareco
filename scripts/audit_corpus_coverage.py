#!/usr/bin/env python3
"""
Corpus Coverage Audit - Verify if answers exist in corpus

VERSION: v2.1
COMPATIBLE WITH: v2.1 Whoosh index
STAGE: Evaluation

Description:
    For each question in the test set, manually search the Whoosh corpus
    to determine if the correct answer sentence exists. This helps diagnose
    whether failures are due to retrieval issues or corpus coverage gaps.

Usage:
    python scripts/audit_corpus_coverage.py --test-set /tmp/qa_test_10.jsonl

Outputs:
    - Console: Table showing which questions have answers in corpus
    - /tmp/corpus_coverage_audit.md: Detailed findings
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh import scoring

def load_test_set(path: str) -> List[Dict]:
    """Load JSONL test set."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def search_corpus(index_path: str, query_terms: List[str], top_n: int = 100) -> List[Dict]:
    """Search Whoosh index for query terms."""
    ix = open_dir(index_path)

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        # Create query string
        query_str = " ".join(query_terms)

        # Parse and search
        parser = QueryParser("roots", schema=ix.schema)
        query = parser.parse(query_str)

        results = searcher.search(query, limit=top_n)

        docs = []
        for hit in results:
            docs.append({
                'text': hit['text'],
                'score': hit.score,
                'sentence_id': hit.get('sentence_id', 'unknown'),
                'doc_title': hit.get('doc_title', 'Unknown')
            })

        return docs

def check_answer_in_text(text: str, expected_keywords: List[str]) -> bool:
    """Check if any expected keyword appears in text."""
    text_lower = text.lower()
    for keyword in expected_keywords:
        if keyword.lower() in text_lower:
            return True
    return False

def audit_question(question: Dict, index_path: str) -> Dict:
    """Audit one question to see if answer exists in corpus."""

    # Build search query from question text
    # Extract roots from question
    from klareco.parser import parse

    query_text = question['question']

    try:
        ast = parse(query_text)

        # Extract all roots from AST
        def extract_roots(node):
            roots = set()
            if isinstance(node, dict):
                if 'radiko' in node:
                    roots.add(node['radiko'])
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        roots.update(extract_roots(value))
            elif isinstance(node, list):
                for item in node:
                    roots.update(extract_roots(item))
            return roots

        query_roots = extract_roots(ast)

        # Add expected keywords to search
        search_terms = list(query_roots)
        for keyword in question['expected_keywords']:
            if keyword not in search_terms:
                search_terms.append(keyword)

    except Exception as e:
        # Fallback: use expected keywords
        search_terms = question['expected_keywords']

    # Search corpus
    results = search_corpus(index_path, search_terms, top_n=200)

    # Check if any result contains the expected answer
    matches = []
    for i, doc in enumerate(results):
        if check_answer_in_text(doc['text'], question['expected_keywords']):
            matches.append({
                'rank': i + 1,
                'score': doc['score'],
                'text': doc['text'],
                'sentence_id': doc['sentence_id'],
                'doc_title': doc['doc_title']
            })

    return {
        'question_id': question['id'],
        'question': question['question'],
        'expected_answer': question['answer'],
        'expected_keywords': question['expected_keywords'],
        'search_terms': search_terms,
        'total_results': len(results),
        'matches_found': len(matches),
        'has_answer': len(matches) > 0,
        'best_match': matches[0] if matches else None,
        'all_matches': matches[:5]  # Top 5 matches
    }

def main():
    parser = argparse.ArgumentParser(description='Audit corpus coverage for test questions')
    parser.add_argument('--test-set', required=True, help='Path to test set JSONL')
    parser.add_argument('--index', default='data/indexes/whoosh_bm25',
                       help='Path to Whoosh index')
    parser.add_argument('--output', default='/tmp/corpus_coverage_audit.md',
                       help='Output markdown file')

    args = parser.parse_args()

    # Load test set
    print(f"Loading test set from {args.test_set}")
    questions = load_test_set(args.test_set)
    print(f"Loaded {len(questions)} questions")

    # Audit each question
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Auditing: {question['question']}")
        result = audit_question(question, args.index)
        results.append(result)

        if result['has_answer']:
            print(f"  ✓ Found answer in corpus (rank {result['best_match']['rank']})")
            print(f"    Sentence: {result['best_match']['text'][:100]}...")
        else:
            print(f"  ✗ Answer NOT found in corpus")

    # Generate summary
    total_questions = len(results)
    questions_with_answers = sum(1 for r in results if r['has_answer'])
    questions_without_answers = total_questions - questions_with_answers

    print(f"\n{'='*80}")
    print(f"CORPUS COVERAGE SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {total_questions}")
    print(f"Answers found in corpus: {questions_with_answers} ({questions_with_answers/total_questions*100:.1f}%)")
    print(f"Answers NOT in corpus: {questions_without_answers} ({questions_without_answers/total_questions*100:.1f}%)")

    # Write detailed report
    with open(args.output, 'w') as f:
        f.write("# Corpus Coverage Audit\n\n")
        f.write(f"**Test Set**: {args.test_set}\n")
        f.write(f"**Index**: {args.index}\n")
        f.write(f"**Date**: {Path(__file__).stat().st_mtime}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total questions: {total_questions}\n")
        f.write(f"- Answers found: {questions_with_answers} ({questions_with_answers/total_questions*100:.1f}%)\n")
        f.write(f"- Answers missing: {questions_without_answers} ({questions_without_answers/total_questions*100:.1f}%)\n\n")

        f.write("## Detailed Results\n\n")

        for result in results:
            status = "✓" if result['has_answer'] else "✗"
            f.write(f"### {status} Q{result['question_id']}: {result['question']}\n\n")
            f.write(f"**Expected answer**: {result['expected_answer']}\n\n")
            f.write(f"**Expected keywords**: {', '.join(result['expected_keywords'])}\n\n")
            f.write(f"**Search terms used**: {', '.join(result['search_terms'])}\n\n")

            if result['has_answer']:
                f.write(f"**Status**: Answer FOUND in corpus\n\n")
                f.write(f"**Best match** (rank {result['best_match']['rank']}, score {result['best_match']['score']:.2f}):\n\n")
                f.write(f"> {result['best_match']['text']}\n\n")
                f.write(f"*Source*: {result['best_match']['doc_title']} ({result['best_match']['sentence_id']})\n\n")

                if len(result['all_matches']) > 1:
                    f.write(f"**Other matches** ({len(result['all_matches'])-1} more):\n\n")
                    for match in result['all_matches'][1:]:
                        f.write(f"- Rank {match['rank']}: {match['text'][:100]}...\n")
                    f.write("\n")
            else:
                f.write(f"**Status**: Answer NOT found in corpus\n\n")
                f.write(f"**Total results retrieved**: {result['total_results']}\n\n")
                f.write(f"**Diagnosis**: Corpus gap - the information does not appear to exist in the corpus\n\n")

            f.write("---\n\n")

    print(f"\nDetailed report written to: {args.output}")

if __name__ == '__main__':
    main()
