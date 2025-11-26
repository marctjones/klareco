#!/usr/bin/env python3
"""
Test RAG retrieval step-by-step to find where it hangs.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever

def test_retrieval():
    """Test RAG retrieval step by step."""

    # Test query
    query = "Kiu estas Gandalf?"

    print("=" * 70)
    print("RAG RETRIEVAL DEBUG TEST")
    print("=" * 70)
    print()
    print(f"Query: {query}")
    print()

    # Step 1: Parse query
    print("Step 1: Parsing query to AST...")
    start = time.time()
    ast = parse(query)
    elapsed = time.time() - start
    print(f"✓ Parsed in {elapsed:.3f}s")
    print(f"  AST type: {ast.get('tipo')}")
    print()

    # Step 2: Initialize retriever
    print("Step 2: Loading retriever (Tree-LSTM + FAISS)...")
    start = time.time()

    retriever = KlarecoRetriever(
        index_dir="data/corpus_index",
        model_path="models/tree_lstm/checkpoint_epoch_12.pt",
        mode='tree_lstm',
        device='cpu'
    )

    elapsed = time.time() - start
    print(f"✓ Retriever loaded in {elapsed:.3f}s")
    print(f"  Corpus size: {len(retriever.metadata):,} sentences")
    print()

    # Step 3: Encode query AST to embedding
    print("Step 3: Encoding query AST with Tree-LSTM...")
    start = time.time()

    try:
        query_embedding = retriever._encode_ast(ast)
        elapsed = time.time() - start
        print(f"✓ Encoded in {elapsed:.3f}s")
        print(f"  Embedding shape: {query_embedding.shape}")
        print(f"  Embedding norm: {(query_embedding ** 2).sum() ** 0.5:.3f}")
        print()
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED after {elapsed:.3f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Search FAISS index
    print("Step 4: Searching FAISS index...")
    start = time.time()

    try:
        results = retriever.retrieve_from_ast(ast, k=3, return_scores=True)
        elapsed = time.time() - start
        print(f"✓ Retrieved in {elapsed:.3f}s")
        print(f"  Found {len(results)} results")
        print()
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED after {elapsed:.3f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Show results
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print()

    for i, result in enumerate(results, 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')
        source = result.get('source_name', 'Unknown')
        line = result.get('line', '?')

        print(f"{i}. Score: {score:.4f} | {source}:{line}")
        print(f"   {text[:150]}{'...' if len(text) > 150 else ''}")
        print()

    print("=" * 70)
    print("✓ RAG RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_retrieval()
