#!/usr/bin/env python3
"""
Debug AST-aware retrieval to understand why accuracy is low.

Compare pre-filter results vs final AST results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever

def main():
    index_path = Path("data/indexes/slot_full")

    print("Debugging AST-Aware Retrieval...")
    print()

    # Initialize retriever
    retriever = ASTAwareRetriever(index_path, use_prefilter=True)

    # Test query that failed
    query = "Kiam aperis la Fundamento de Esperanto?"
    expected = "1905"

    print(f"Query: {query}")
    print(f"Expected answer contains: {expected}")
    print()

    # Get results
    print("=" * 80)
    print("AST-Aware Results (with pre-filtering):")
    print("=" * 80)
    results = retriever.search(query, top_k=10, strategy='auto')

    for i, (score, doc) in enumerate(results, 1):
        text = doc['text'][:120]
        print(f"{i}. Score: {score:.3f}")
        print(f"   {text}...")

        # Check if answer is in there
        if expected.lower() in doc['text'].lower():
            print(f"   ✓ CONTAINS ANSWER!")
        print()

    # Now check what the pre-filter returns
    print("=" * 80)
    print("HNSW Pre-filter Results (no AST matching):")
    print("=" * 80)

    if retriever.prefilter_retriever:
        from klareco.parser import parse
        query_ast = parse(query)
        query_text = retriever._reconstruct_query(query_ast)

        print(f"Reconstructed query: {query_text}")
        print()

        prefilter_results = retriever.prefilter_retriever.search(
            query_text,
            top_k=10,
            hnsw_top_n=10,
            slot_top_n=10
        )

        for i, (score, doc) in enumerate(prefilter_results, 1):
            text = doc['text'][:120]
            print(f"{i}. Score: {score:.3f}")
            print(f"   {text}...")

            if expected.lower() in doc['text'].lower():
                print(f"   ✓ CONTAINS ANSWER!")
            print()

if __name__ == '__main__':
    main()
