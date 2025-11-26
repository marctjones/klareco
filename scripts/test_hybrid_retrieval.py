#!/usr/bin/env python3
"""
Compare pure semantic vs hybrid retrieval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever


def test_retrieval_methods():
    """Compare retrieval methods."""

    query = "Kiu estas Mitrandiro?"

    print("=" * 70)
    print("RETRIEVAL COMPARISON: Pure Semantic vs Hybrid")
    print("=" * 70)
    print(f"\nQuery: {query}\n")

    # Parse query
    ast = parse(query)

    # Load retriever
    retriever = KlarecoRetriever(
        index_dir="data/corpus_index",
        model_path="models/tree_lstm/checkpoint_epoch_12.pt",
        mode='tree_lstm',
        device='cpu'
    )

    # Method 1: Pure semantic (current)
    print("-" * 70)
    print("METHOD 1: Pure Semantic Search (Tree-LSTM only)")
    print("-" * 70)

    semantic_results = retriever.retrieve_from_ast(ast, k=5, return_scores=True)

    for i, result in enumerate(semantic_results, 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')
        source = result.get('source_name', 'Unknown')
        print(f"\n{i}. [{score:.4f}] {source}")
        print(f"   {text[:100]}")

    # Method 2: Hybrid (keyword filter + semantic rerank)
    print("\n")
    print("-" * 70)
    print("METHOD 2: Hybrid Search (Keyword filter → Semantic rerank)")
    print("-" * 70)

    hybrid_results = retriever.retrieve_hybrid(
        ast,
        k=5,
        keyword_candidates=100,
        return_scores=True
    )

    for i, result in enumerate(hybrid_results, 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')
        source = result.get('source_name', 'Unknown')
        print(f"\n{i}. [{score:.4f}] {source}")
        print(f"   {text[:100]}")

    # Analysis
    print("\n")
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if Mitrandiro appears in results
    semantic_has_mitrandiro = any('Mitrandiro' in r.get('text', '') for r in semantic_results)
    hybrid_has_mitrandiro = any('Mitrandiro' in r.get('text', '') for r in hybrid_results)

    print(f"\nPure Semantic - Found 'Mitrandiro': {semantic_has_mitrandiro}")
    if semantic_has_mitrandiro:
        for i, r in enumerate(semantic_results, 1):
            if 'Mitrandiro' in r.get('text', ''):
                print(f"  → Ranked #{i}")

    print(f"\nHybrid Search - Found 'Mitrandiro': {hybrid_has_mitrandiro}")
    if hybrid_has_mitrandiro:
        for i, r in enumerate(hybrid_results, 1):
            if 'Mitrandiro' in r.get('text', ''):
                print(f"  → Ranked #{i}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_retrieval_methods()
