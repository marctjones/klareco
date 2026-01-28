#!/usr/bin/env python3
"""
Analyze RAG evaluation results to understand performance.

Usage:
    python scripts/analyze_rag_results.py data/evaluation/rag_results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

def analyze_results(results_path: Path):
    """Analyze evaluation results."""
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    print("=" * 80)
    print(f"RAG Results Analysis ({len(results)} questions)")
    print("=" * 80)

    for result in results[:10]:  # Show first 10
        print(f"\n{'='*80}")
        print(f"Q: {result['question']}")
        print(f"Category: {result['category']}")
        print(f"Expected: {result.get('expected_answer_pattern', 'N/A')[:70]}...")
        print(f"-" * 80)

        evaluation = result.get('evaluation', {})
        print(f"Status: {'✅ CORRECT' if evaluation.get('correct') else '⚠️ PARTIAL' if evaluation.get('partial') else '❌ INCORRECT'}")
        print(f"Notes: {evaluation.get('notes', 'N/A')}")

        answer = result.get('rag_answer', '')
        if answer:
            print(f"\nTop Answer: {answer[:200]}...")

            passages = result.get('retrieved_passages', [])
            if passages:
                top = passages[0]
                print(f"Source: {top.get('source', 'unknown')}")
                print(f"Scores: retrieval={top.get('score', 0):.3f}, M1={top.get('m1_score', 0):.3f}, rerank={top.get('rerank_score', 0):.3f}")
        else:
            print("\nNo answer returned")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    correct = sum(1 for r in results if r.get('evaluation', {}).get('correct'))
    partial = sum(1 for r in results if r.get('evaluation', {}).get('partial'))
    incorrect = len(results) - correct - partial

    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"Partial: {partial}/{len(results)} ({partial/len(results)*100:.1f}%)")
    print(f"Incorrect: {incorrect}/{len(results)} ({incorrect/len(results)*100:.1f}%)")

    # Common issues
    notes = [r.get('evaluation', {}).get('notes', '') for r in results]
    print(f"\nCommon evaluation notes:")
    for note in set(notes):
        count = notes.count(note)
        if count > 1:
            print(f"  [{count}x] {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path, help='Path to results.jsonl')
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        sys.exit(1)

    analyze_results(args.results)


if __name__ == '__main__':
    main()
