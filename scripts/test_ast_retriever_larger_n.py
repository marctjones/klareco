#!/usr/bin/env python3
"""
Test AST-aware retriever with larger prefilter_n.

Try prefilter_n=2000 instead of default 500.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever

def main():
    index_path = Path("data/indexes/slot_full")

    print("Testing AST-Aware Retriever with prefilter_n=2000...")
    print()

    # Initialize retriever
    retriever = ASTAwareRetriever(index_path, use_prefilter=True)

    # Test queries
    test_cases = [
        ("Kiam aperis la Fundamento de Esperanto?", "1905"),
        ("Kiu fondis Esperanton?", "Zamenhof"),
        ("En kiu jaro naskiĝis Zamenhof?", "1859"),
        ("Kie naskiĝis Zamenhof?", "Bjalistoko"),
    ]

    for query, expected in test_cases:
        print(f"Query: {query}")
        print(f"Expected: {expected}")

        # Search with larger prefilter_n=2000
        results = retriever.search(query, top_k=10, strategy='auto', prefilter_n=2000)

        found = False
        for i, (score, doc) in enumerate(results, 1):
            if expected.lower() in doc['text'].lower():
                print(f"  ✓ Found at rank {i} (score: {score:.3f})")
                found = True
                break

        if not found:
            print(f"  ❌ Not found in top 10")

        print()

if __name__ == '__main__':
    main()
