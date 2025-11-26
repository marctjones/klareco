#!/usr/bin/env python3
"""
Compare old vs new GNN model performance on RAG retrieval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import KlarecoRetriever

# Test queries
TEST_QUERIES = [
    ("Kiu estas Frodo?", "Who is Frodo?"),
    ("Kiu estas Gandalfo?", "Who is Gandalf?"),
    ("Kio estas hobito?", "What is a hobbit?"),
    ("Kio estas la Unu Ringo?", "What is the One Ring?"),
]

def test_model(model_name: str, model_path: str, index_path: str):
    """Test a model and return results."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"  Model: {model_path}")
    print(f"  Index: {index_path}")
    print(f"{'='*70}\n")

    retriever = KlarecoRetriever(
        index_dir=index_path,
        model_path=model_path,
        mode="tree_lstm"
    )

    results = {}

    for esperanto, english in TEST_QUERIES:
        print(f"Query: {esperanto} ({english})")

        try:
            ast = parse(esperanto)
            result = retriever.retrieve_hybrid(
                ast,
                k=3,
                keyword_candidates=100,
                return_stage1_info=True
            )

            results[esperanto] = {
                'stage1_count': result['stage1']['total_candidates'],
                'top_scores': [doc['score'] for doc in result['results']],
                'top_sentences': [doc['text'][:100] + '...' for doc in result['results']],
            }

            print(f"  Stage 1: {result['stage1']['total_candidates']} candidates")
            print(f"  Top 3 scores: {', '.join(f'{s:.3f}' for s in results[esperanto]['top_scores'])}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[esperanto] = {'error': str(e)}
            print()

    return results


def compare_results(old_results, new_results):
    """Compare and print differences."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    for query in TEST_QUERIES:
        esperanto, english = query
        print(f"{esperanto} ({english}):")

        if esperanto in old_results and esperanto in new_results:
            old = old_results[esperanto]
            new = new_results[esperanto]

            if 'error' not in old and 'error' not in new:
                print(f"  Stage 1 candidates: {old['stage1_count']} → {new['stage1_count']}")
                print(f"  Top score: {old['top_scores'][0]:.3f} → {new['top_scores'][0]:.3f}")

                score_diff = new['top_scores'][0] - old['top_scores'][0]
                if score_diff > 0:
                    print(f"  ✅ Improved by {score_diff:.3f}")
                elif score_diff < 0:
                    print(f"  ⚠️  Decreased by {abs(score_diff):.3f}")
                else:
                    print(f"  → No change")

        print()


def main():
    print("="*70)
    print("GNN MODEL COMPARISON")
    print("="*70)

    # Test old model
    old_results = test_model(
        "OLD MODEL (5.5K pairs, 12 epochs)",
        "models/tree_lstm_old/checkpoint_epoch_12.pt",
        "data/corpus_index_old"
    )

    # Test new model
    new_results = test_model(
        "NEW MODEL (58K pairs, 20 epochs)",
        "models/tree_lstm/checkpoint_epoch_20.pt",
        "data/corpus_index"
    )

    # Compare
    compare_results(old_results, new_results)

    print("\n" + "="*70)
    print("✓ Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
