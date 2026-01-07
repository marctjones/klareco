#!/usr/bin/env python3
"""
Quick test script for AST-aware retriever.

Tests that the retriever can load and search successfully.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever

def main():
    print("Testing AST-Aware Retriever...")
    print()

    # Index path
    index_path = Path("data/indexes/slot_hybrid")

    if not (index_path / "slot_index.jsonl").exists():
        print(f"❌ Index not found: {index_path}")
        print("   Run: python scripts/index_slot_based.py first")
        sys.exit(1)

    # Initialize retriever
    print("1. Loading AST-aware retriever...")
    try:
        retriever = ASTAwareRetriever(index_path)
        print("   ✓ Retriever loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        sys.exit(1)

    print()

    # Test queries
    test_queries = [
        "Kiu fondis Esperanton?",
        "Kio estas Esperanto?",
        "Kie naskiĝis Zamenhof?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Testing query: {query}")

        try:
            results = retriever.search(query, top_k=3, strategy='auto')

            if results:
                print(f"   ✓ Found {len(results)} results")
                print(f"   Top result: {results[0][1]['text'][:80]}...")
                print(f"   Score: {results[0][0]:.3f}")
            else:
                print("   ⚠ No results found")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()

        print()

    print("✅ All tests completed!")

if __name__ == '__main__':
    main()
