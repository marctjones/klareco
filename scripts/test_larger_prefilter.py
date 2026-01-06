#!/usr/bin/env python3
"""
Test if increasing prefilter_n helps find answers.

Try prefilter_n of 5000 instead of 500.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.parser import parse

def main():
    index_path = Path("data/indexes/slot_full")

    print("Testing larger prefilter_n...")
    print()

    # Initialize retriever
    retriever = ASTAwareRetriever(index_path, use_prefilter=True)

    # Test query that failed
    query = "Kiam aperis la Fundamento de Esperanto?"
    expected = "1905"

    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print()

    # Check if answer is in top 5000 pre-filter results
    query_ast = parse(query)
    query_text = retriever._reconstruct_query(query_ast)

    print("Checking pre-filter results with prefilter_n=5000...")
    prefilter_results = retriever.prefilter_retriever.search(
        query_text,
        top_k=5000,
        hnsw_top_n=5000,
        slot_top_n=5000
    )

    # Find where answer appears
    for i, (score, doc) in enumerate(prefilter_results, 1):
        if expected in doc['text']:
            print(f"✓ FOUND answer at rank {i}!")
            print(f"  Score: {score:.3f}")
            print(f"  Text: {doc['text'][:200]}...")
            return

    print(f"❌ Answer NOT found in top 5000 pre-filter results")
    print()
    print("This means the embedding model doesn't place semantically")
    print("relevant documents near the query embedding.")

if __name__ == '__main__':
    main()
