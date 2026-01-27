#!/usr/bin/env python3
"""
Demo: Full RAG Pipeline with Answer Extraction

Shows the complete pipeline:
1. Query parsing
2. Document retrieval (from corpus)
3. Answer extraction (AST-based)
4. Validation against expected answer

Usage:
    python scripts/demo_rag_with_extraction.py
    python scripts/demo_rag_with_extraction.py --query "Kiu fondis Esperanton?"
    python scripts/demo_rag_with_extraction.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.answer_extractor import ASTAnswerExtractor


def load_test_set(test_set_path: Path = None) -> list:
    """Load test set from JSON file."""
    if test_set_path is None:
        test_set_path = Path('data/test_sets/qa_test_set_50.json')

    if not test_set_path.exists():
        print(f"Warning: Test set not found at {test_set_path}")
        print("Using minimal test set...")
        # Fallback to minimal test set
        return [
            {
                'query': 'Kiu fondis Esperanton?',
                'expected_keywords': ['zamenhof'],
                'question_type': 'WHO',
            },
            {
                'query': 'Kio estas Esperanto?',
                'expected_keywords': ['lingv', 'internaci'],
                'question_type': 'WHAT',
            },
            {
                'query': 'Kie naskiĝis Zamenhof?',
                'expected_keywords': ['bjalistok', 'pol'],
                'question_type': 'WHERE',
            },
            {
                'query': 'Kiam estis fondita Esperanto?',
                'expected_keywords': ['1887'],
                'question_type': 'WHEN',
            },
            {
                'query': 'Kiom da homoj parolas Esperanton?',
                'expected_keywords': ['mil'],
                'question_type': 'HOW_MANY',
            },
        ]

    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    return test_set.get('queries', [])


def check_answer(answer_text: str, expected_keywords: list) -> bool:
    """Check if extracted answer contains expected keywords."""
    if not answer_text:
        return False
    answer_lower = answer_text.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def demo_query(retriever: ASTAwareRetriever, extractor: ASTAnswerExtractor,
               query: str, expected_keywords: list = None, top_k: int = 5):
    """
    Demo full pipeline for a single query.

    Args:
        retriever: Document retriever
        extractor: Answer extractor
        query: Question text
        expected_keywords: Expected keywords in answer (for validation)
        top_k: Number of documents to retrieve
    """
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # Parse query
    query_ast = parse(query)

    # Retrieve documents
    print(f"\nRetrieving top-{top_k} documents...")
    results = retriever.search(query, top_k=top_k, use_m1_expansion=False)

    if not results:
        print("✗ No documents found")
        return None

    print(f"✓ Retrieved {len(results)} documents")

    # Try to extract answer from each document
    best_answer = None
    best_score = 0.0

    for i, (score, doc, stats) in enumerate(results, 1):
        doc_text = doc.get('text', '')
        doc_ast = parse(doc_text)

        # Extract answer
        answer = extractor.extract_answer(query_ast, doc_ast, doc_text)

        if answer:
            # Found an answer
            combined_score = score * answer['confidence']

            if combined_score > best_score:
                best_score = combined_score
                best_answer = answer
                best_answer['doc_rank'] = i
                best_answer['doc_score'] = score
                best_answer['doc_text'] = doc_text

    # Display result
    print("\n" + "-" * 70)
    if best_answer:
        print("✓ ANSWER EXTRACTED:")
        print(f"  Text: {best_answer['text']}")
        print(f"  Confidence: {best_answer['confidence']:.2f}")
        print(f"  Extraction method: {best_answer['method']}")
        print(f"  Explanation: {best_answer['explanation']}")
        print(f"  From document rank: #{best_answer['doc_rank']}")
        print(f"  Retrieval score: {best_answer['doc_score']:.4f}")
        print(f"  Combined score: {best_score:.4f}")

        # Show document snippet
        doc_snippet = best_answer['doc_text'][:200] + "..." if len(best_answer['doc_text']) > 200 else best_answer['doc_text']
        print(f"\n  Source document: {doc_snippet}")

        # Validate if expected keywords provided
        if expected_keywords:
            is_correct = check_answer(best_answer['text'], expected_keywords)
            if is_correct:
                print(f"\n  ✅ CORRECT (matches expected keywords: {expected_keywords})")
            else:
                print(f"\n  ❌ INCORRECT (expected keywords: {expected_keywords})")

            return is_correct
        else:
            return True
    else:
        print("✗ NO ANSWER EXTRACTED")
        print("  Reason: No extraction pattern matched in any retrieved document")

        # Show top document for debugging
        if results:
            top_doc = results[0][1].get('text', '')
            doc_snippet = top_doc[:200] + "..." if len(top_doc) > 200 else top_doc
            print(f"\n  Top retrieved document: {doc_snippet}")

        return False


def main():
    parser = argparse.ArgumentParser(description="Demo RAG with answer extraction")
    parser.add_argument('--index-dir', type=str, default='data/indexes/kuzu_index',
                       help='Path to Kuzu index')
    parser.add_argument('--query', type=str, help='Single query to test')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of documents to retrieve')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    args = parser.parse_args()

    # Load retriever
    print("=" * 70)
    print("RAG Pipeline with Answer Extraction")
    print("=" * 70)
    print("\nLoading retriever...")
    retriever = ASTAwareRetriever(index_path=Path(args.index_dir))
    print("✓ Retriever loaded")

    # Load extractor
    print("Loading answer extractor...")
    extractor = ASTAnswerExtractor()
    print("✓ Extractor loaded")

    if args.query:
        # Single query
        demo_query(retriever, extractor, args.query, top_k=args.top_k)
    elif args.interactive:
        # Interactive mode
        print("\n" + "=" * 70)
        print("Interactive Mode (type 'exit' to quit)")
        print("=" * 70)

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if not query:
                    continue

                demo_query(retriever, extractor, query, top_k=args.top_k)
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
    else:
        # Run all test queries
        test_queries = load_test_set()
        print(f"\nRunning {len(test_queries)} test queries...\n")

        correct = 0
        total = len(test_queries)

        for test_case in test_queries:
            result = demo_query(
                retriever,
                extractor,
                test_case['query'],
                test_case['expected_keywords'],
                top_k=args.top_k
            )
            if result:
                correct += 1

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Questions answered correctly: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"Questions failed: {total-correct}/{total}")

        if correct == total:
            print("\n🎉 ALL QUESTIONS ANSWERED CORRECTLY!")
        elif correct >= total * 0.8:
            print(f"\n✅ Good performance ({correct/total*100:.1f}%)")
        else:
            print(f"\n⚠️  Needs improvement ({correct/total*100:.1f}%)")

    retriever.close()


if __name__ == '__main__':
    main()
