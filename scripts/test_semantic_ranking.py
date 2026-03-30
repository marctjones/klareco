#!/usr/bin/env python3
"""
Test Semantic AST Ranking Implementation

Quick test to verify semantic ranking is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever

def test_query(question: str):
    """Test a single query with semantic ranking."""
    print("=" * 70)
    print(f"Query: {question}")
    print("=" * 70)

    # Parse query
    query_ast = parse(question)
    print(f"Query AST: {query_ast.get('tipo')}")

    # Initialize retriever
    retriever = WhooshRetriever(
        whoosh_index_dir=Path('data/indexes/whoosh_fts'),
        kuzu_db_path=Path('data/indexes/v2.1_kuzu_index_full')
    )

    # Retrieve with AST-aware method
    print("\nRetrieving with semantic AST ranking...")
    results = retriever.retrieve(
        query_roots=[],  # Not used anymore
        top_k=10,
        query_ast=query_ast
    )

    print(f"\nRetrieved {len(results)} results:\n")

    # Show query details
    from klareco.rag.ast_semantic_ranker import get_ast_verb_root, get_ast_object_root
    query_verb = get_ast_verb_root(query_ast)
    query_obj = get_ast_object_root(query_ast)
    print(f"Query verb: {query_verb}, object: {query_obj}\n")

    for i, result in enumerate(results[:5], 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')[:100]
        breakdown = result.get('score_breakdown', {})

        # Show candidate details
        cand_ast = result.get('ast')
        cand_verb = get_ast_verb_root(cand_ast) if cand_ast else None
        cand_obj = get_ast_object_root(cand_ast) if cand_ast else None

        print(f"{i}. Score: {score:.2f}")
        print(f"   Cand verb: {cand_verb}, object: {cand_obj}")
        print(f"   Text: {text}...")

        if breakdown:
            print(f"   Breakdown:")
            print(f"     - Verb similarity: {breakdown.get('verb_similarity', 0.0):.2f}")
            print(f"     - Structural match: {breakdown.get('structural_match', 0.0):.2f}")
            print(f"     - Total: {breakdown.get('total', 0.0):.2f}")

        print()

if __name__ == '__main__':
    # Test WHO question
    test_query("Kiu fondis Esperanton?")

    # Test WHAT question
    test_query("Kio estas Esperanto?")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
