#!/usr/bin/env python3
"""
Demo: Two-Stage Hybrid Retrieval

Shows the complete two-stage retrieval process:
  Stage 1: Keyword filtering (recall)
  Stage 2: Semantic reranking (precision)

Usage:
    python scripts/demo_two_stage.py
    python scripts/demo_two_stage.py "Kiu estas Frodo?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever


def demo_two_stage(query: str):
    """Demo two-stage retrieval with detailed output."""

    print("=" * 70)
    print("TWO-STAGE HYBRID RETRIEVAL DEMO")
    print("=" * 70)
    print()
    print(f"Query: {query}")
    print()

    # Parse query
    print("Parsing query to AST...")
    ast = parse(query)
    print("✓ AST created")
    print()

    # Load retriever
    print("Loading retriever...")
    retriever = KlarecoRetriever(
        index_dir="data/corpus_index",
        model_path="models/tree_lstm/checkpoint_epoch_12.pt",
        mode='tree_lstm',
        device='cpu'
    )
    print(f"✓ Retriever loaded (corpus: {len(retriever.metadata):,} sentences)")
    print()

    # Retrieve with stage1 info
    print("=" * 70)
    print("RETRIEVING WITH TWO-STAGE PIPELINE")
    print("=" * 70)
    print()

    result = retriever.retrieve_hybrid(
        ast,
        k=5,
        keyword_candidates=100,
        return_scores=True,
        return_stage1_info=True
    )

    # Extract stage info
    stage1_info = result['stage1']
    stage2_results = result['results']

    # Display Stage 1
    print("-" * 70)
    print("STAGE 1: Keyword Filtering")
    print("-" * 70)
    print()
    print(f"Keywords extracted: {stage1_info['keywords']}")
    print(f"Total candidates found: {stage1_info['total_candidates']}")
    print(f"Candidates for reranking: {stage1_info['candidates_reranked']}")
    print()

    if stage1_info['candidates_shown']:
        print(f"First {len(stage1_info['candidates_shown'])} keyword matches:")
        print()
        for i, candidate in enumerate(stage1_info['candidates_shown'], 1):
            text = candidate.get('text', '')
            source = candidate.get('source_name', 'Unknown')
            line = candidate.get('line', '?')
            print(f"{i:2d}. {source}:{line}")
            print(f"    {text[:100]}{'...' if len(text) > 100 else ''}")
        print()

    # Display Stage 2
    print("-" * 70)
    print("STAGE 2: Semantic Reranking (Tree-LSTM)")
    print("-" * 70)
    print()
    print(f"Reranked top {len(stage2_results)} results by semantic similarity:")
    print()

    for i, result in enumerate(stage2_results, 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')
        source = result.get('source_name', 'Unknown')
        line = result.get('line', '?')

        print(f"{i}. Score: {score:.4f}")
        print(f"   {source}:{line}")
        print(f"   {text[:150]}{'...' if len(text) > 150 else ''}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Stage 1 filtered {stage1_info['total_candidates']:,} candidates from {len(retriever.metadata):,} total")
    print(f"Stage 2 reranked top {stage1_info['candidates_reranked']} by semantic similarity")
    print(f"Final results: {len(stage2_results)} most relevant sentences")
    print()

    # Check if all results contain keywords
    keywords = stage1_info['keywords']
    all_contain_keywords = all(
        any(kw in r.get('text', '').lower() for kw in keywords)
        for r in stage2_results
    )
    print(f"All final results contain keywords: {'✓ Yes' if all_contain_keywords else '✗ No'}")
    print()


if __name__ == "__main__":
    # Default query or from command line
    query = sys.argv[1] if len(sys.argv) > 1 else "Kiu estas Mitrandiro?"
    demo_two_stage(query)
