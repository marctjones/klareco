#!/usr/bin/env python3
"""
Analyze two-stage retrieval performance.

Tests:
1. Literary queries (characters, objects)
2. Non-literary queries (concepts, actions)
3. Stage 1 vs Stage 2 quality comparison
4. GNN reranking effectiveness
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever
import numpy as np


def analyze_query(retriever, query, description):
    """Analyze a single query."""
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Description: {description}")
    print("=" * 80)

    ast = parse(query)

    # Get hybrid retrieval with stage1 info
    result = retriever.retrieve_hybrid(
        ast,
        k=5,
        keyword_candidates=100,
        return_scores=True,
        return_stage1_info=True
    )

    stage1 = result['stage1']
    stage2 = result['results']

    # Stage 1 analysis
    print(f"\n📊 STAGE 1: Keyword Filtering")
    print(f"Keywords extracted: {stage1['keywords']}")
    print(f"Total candidates found: {stage1['total_candidates']}")
    print(f"\nFirst 5 Stage 1 candidates (BEFORE reranking):")
    for i, cand in enumerate(stage1['candidates_shown'][:5], 1):
        text = cand.get('text', '')
        source = cand.get('source_name', 'Unknown')
        print(f"  {i}. [{source}]")
        print(f"     {text[:100]}...")

    # Stage 2 analysis
    print(f"\n🎯 STAGE 2: Semantic Reranking (Top 5 AFTER reranking)")
    for i, result in enumerate(stage2, 1):
        score = result.get('score', 0.0)
        text = result.get('text', '')
        source = result.get('source_name', 'Unknown')
        print(f"  {i}. Score: {score:.4f} [{source}]")
        print(f"     {text[:100]}...")

    # Score analysis
    scores = [r.get('score', 0.0) for r in stage2]
    print(f"\n📈 Score Statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    print(f"  Range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  Score spread: {max(scores) - min(scores):.4f}")

    # Check if reranking made a difference
    stage1_texts = [c.get('text', '') for c in stage1['candidates_shown'][:5]]
    stage2_texts = [r.get('text', '') for r in stage2]

    reordered = sum(1 for i, text in enumerate(stage2_texts)
                   if i < len(stage1_texts) and text != stage1_texts[i])
    print(f"\n🔄 Reranking Impact:")
    print(f"  {reordered}/5 results reordered from Stage 1")

    return {
        'query': query,
        'keywords': stage1['keywords'],
        'stage1_count': stage1['total_candidates'],
        'stage2_results': stage2,
        'scores': scores,
        'reordered': reordered
    }


def main():
    # Initialize retriever
    index_dir = Path("data/corpus_index")
    model_path = Path("models/tree_lstm/checkpoint_epoch_12.pt")

    retriever = KlarecoRetriever(
        index_dir=str(index_dir),
        model_path=str(model_path),
        mode='tree_lstm',
        device='cpu'
    )

    print("\n" + "=" * 80)
    print("TWO-STAGE RETRIEVAL ANALYSIS")
    print("=" * 80)

    # Test queries
    queries = [
        # Literary - Characters
        ("Kiu estas Gandalfo?", "Character query - Gandalf"),
        ("Kiu estas Frodo?", "Character query - Frodo"),
        ("Kiu estas Aragorno?", "Character query - Aragorn"),

        # Literary - Objects/Concepts
        ("Kio estas la Unu Ringo?", "Object query - the One Ring"),
        ("Kio estas hobito?", "Concept query - hobbit"),

        # Non-literary (if in corpus)
        ("Kio estas saĝo?", "Abstract concept - wisdom"),
        ("Kio estas amikeco?", "Abstract concept - friendship"),
        ("Kiel oni batalas?", "Action query - how to fight"),
    ]

    results = []
    for query, description in queries:
        try:
            result = analyze_query(retriever, query, description)
            results.append(result)
            print("\n" + "=" * 80)
            input("Press Enter for next query...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    print("\n📊 Average metrics across all queries:")
    avg_stage1 = np.mean([r['stage1_count'] for r in results])
    avg_scores = np.mean([np.mean(r['scores']) for r in results])
    avg_reordered = np.mean([r['reordered'] for r in results])

    print(f"  Avg Stage 1 candidates: {avg_stage1:.0f}")
    print(f"  Avg Stage 2 score: {avg_scores:.4f}")
    print(f"  Avg reordering: {avg_reordered:.1f}/5 results changed")

    print("\n🎯 Observations:")
    print("  1. Are Stage 1 keywords correct?")
    print("  2. Are Stage 1 candidates relevant?")
    print("  3. Does Stage 2 improve ordering?")
    print("  4. Are scores well-distributed?")
    print("  5. Are top results actually most relevant?")


if __name__ == "__main__":
    main()
