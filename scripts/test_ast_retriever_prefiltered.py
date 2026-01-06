#!/usr/bin/env python3
"""
Test AST-aware retriever WITH pre-filtering (HNSW).

Demonstrates two-stage retrieval: HNSW pre-filter → AST pattern matching
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever

def main():
    print("Testing AST-Aware Retriever with HNSW Pre-filtering...")
    print()

    # Index path with HNSW
    index_path = Path("data/indexes/slot_full")

    if not (index_path / "slot_index.jsonl").exists():
        print(f"❌ Index not found: {index_path}")
        print("   Run: python scripts/index_slot_based.py first")
        sys.exit(1)

    if not (index_path / "hnsw").exists():
        print(f"❌ HNSW index not found: {index_path}/hnsw")
        print("   This test requires HNSW pre-filtering")
        sys.exit(1)

    # Initialize retriever with pre-filtering
    print("1. Loading AST-aware retriever with HNSW pre-filtering...")
    try:
        retriever = ASTAwareRetriever(index_path, use_prefilter=True)
        print("   ✓ Retriever loaded with pre-filtering")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
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
    print()
    print("Pre-filtering architecture:")
    print("  Stage 1: HNSW pre-filter (4.3M docs → 500 candidates in ~5ms)")
    print("  Stage 2: AST pattern matching (500 candidates in ~500ms)")
    print("  Total: ~1-2 seconds per query for full corpus coverage")

if __name__ == '__main__':
    main()
