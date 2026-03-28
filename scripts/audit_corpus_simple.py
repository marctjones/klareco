#!/usr/bin/env python3
"""
Simple Corpus Coverage Audit using existing WhooshRetriever

VERSION: v2.1
STAGE: Evaluation
"""

import argparse
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.whoosh_retriever import WhooshRetriever

def load_test_set(path: str):
    """Load JSONL test set."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def check_answer_in_results(results, expected_keywords):
    """Check if any result contains expected answer keywords."""
    matches = []
    for i, doc in enumerate(results):
        text_lower = doc['text'].lower()
        for keyword in expected_keywords:
            if keyword.lower() in text_lower:
                matches.append({
                    'rank': i + 1,
                    'score': doc['score'],
                    'text': doc['text'],
                    'keyword_matched': keyword
                })
                break
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--index', default='data/indexes/whoosh_bm25')
    args = parser.parse_args()

    # Load retriever
    print("Loading Whoosh retriever...")
    retriever = WhooshRetriever(args.index)

    # Load test set
    print(f"Loading test set from {args.test_set}")
    questions = load_test_set(args.test_set)
    print(f"Loaded {len(questions)} questions\n")

    # Audit each question
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        print(f"  Expected: {q['answer']}")
        print(f"  Keywords: {', '.join(q['expected_keywords'])}")

        # Search using question as query
        docs = retriever.search(q['question'], top_n=100)

        # Check for matches
        matches = check_answer_in_results(docs, q['expected_keywords'])

        if matches:
            print(f"  ✓ FOUND in corpus (rank {matches[0]['rank']}, score {matches[0]['score']:.2f})")
            print(f"    Matched keyword: {matches[0]['keyword_matched']}")
            print(f"    Sentence: {matches[0]['text'][:120]}...")
        else:
            print(f"  ✗ NOT FOUND in corpus (searched {len(docs)} results)")

        results.append({
            'question': q,
            'found': len(matches) > 0,
            'matches': matches
        })
        print()

    # Summary
    total = len(results)
    found = sum(1 for r in results if r['found'])
    print("="*80)
    print(f"SUMMARY: {found}/{total} ({found/total*100:.1f}%) questions have answers in corpus")
    print("="*80)

    # Write detailed report
    output_path = '/tmp/corpus_coverage_audit.md'
    with open(output_path, 'w') as f:
        f.write("# Corpus Coverage Audit\n\n")
        f.write(f"- Test set: {args.test_set}\n")
        f.write(f"- Index: {args.index}\n")
        f.write(f"- Results: {found}/{total} ({found/total*100:.1f}%) found\n\n")
        f.write("---\n\n")

        for i, r in enumerate(results, 1):
            q = r['question']
            status = "✓" if r['found'] else "✗"

            f.write(f"## {status} Q{i}: {q['question']}\n\n")
            f.write(f"**Expected**: {q['answer']}\n\n")
            f.write(f"**Keywords**: {', '.join(q['expected_keywords'])}\n\n")

            if r['found']:
                best = r['matches'][0]
                f.write(f"**Status**: FOUND (rank {best['rank']}, score {best['score']:.2f})\n\n")
                f.write(f"**Matched keyword**: {best['keyword_matched']}\n\n")
                f.write(f"**Sentence**:\n> {best['text']}\n\n")

                if len(r['matches']) > 1:
                    f.write(f"**Additional matches**: {len(r['matches'])-1} more sentences found\n\n")
            else:
                f.write(f"**Status**: NOT FOUND in corpus\n\n")
                f.write(f"**Diagnosis**: Corpus coverage gap - answer does not appear to exist\n\n")

            f.write("---\n\n")

    print(f"\nDetailed report: {output_path}")

if __name__ == '__main__':
    main()
