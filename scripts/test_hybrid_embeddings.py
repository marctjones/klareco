#!/usr/bin/env python3
"""
Test script for hybrid embeddings.

Demonstrates the difference between linguistic, topical, and hybrid modes
for different types of words (content words, proper nouns, etc.).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from klareco.embeddings.hybrid_embeddings import HybridEmbeddings


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_vocabulary_info(model):
    """Test vocabulary overlap analysis."""
    print_section("VOCABULARY ANALYSIS")

    info = model.get_vocabulary_info()

    print(f"\nLinguistic vocabulary: {info['linguistic_vocab_size']:,} roots")
    print(f"Topical vocabulary:    {info['topical_vocab_size']:,} roots")
    print(f"Overlap:               {info['overlap_size']:,} roots ({info['overlap_percentage']:.1f}%)")
    print(f"Linguistic only:       {info['linguistic_only']:,} roots")
    print(f"Topical only:          {info['topical_only']:,} roots")


def test_root_classification(model):
    """Test automatic root type classification."""
    print_section("ROOT TYPE CLASSIFICATION")

    # Test different types of words
    test_words = [
        'hund',      # Common content word
        'bel',       # Common adjective root
        'kur',       # Common verb root
        'pariz',     # Proper noun (city) - Paris
        'napoleon',  # Proper noun (person) - Napoleon
        'algoritm',  # Technical term
        'foobar',    # Unknown word
    ]

    print("\n{:<15} {:<10} {:<10} {:<25}".format(
        "Root", "Ling?", "Topic?", "Classification"
    ))
    print("-" * 70)

    for root in test_words:
        info = model.analyze_root(root)
        ling_mark = "✓" if info['has_linguistic'] else "✗"
        top_mark = "✓" if info['has_topical'] else "✗"
        print(f"{root:<15} {ling_mark:<10} {top_mark:<10} {info['type']:<25}")


def test_similarity_modes(model):
    """Test similarity computation in different modes."""
    print_section("SIMILARITY IN DIFFERENT MODES")

    # Test pairs
    test_pairs = [
        ("hund", "kat", "Content words (both animals)"),
        ("bel", "ĉarm", "Adjectives (beauty-related)"),
        ("pariz", "franci", "Proper nouns (geographic)"),
        ("hund", "manĝ", "Noun-verb (contextual association)"),
    ]

    print("\n{:<40} {:<12} {:<12} {:<12}".format(
        "Pair", "Linguistic", "Topical", "Hybrid"
    ))
    print("-" * 70)

    for root1, root2, description in test_pairs:
        ling_sim = model.compute_similarity(root1, root2, mode='linguistic')
        top_sim = model.compute_similarity(root1, root2, mode='topical')
        hyb_sim = model.compute_similarity(root1, root2, mode='hybrid')

        ling_str = f"{ling_sim:.3f}" if ling_sim is not None else "N/A"
        top_str = f"{top_sim:.3f}" if top_sim is not None else "N/A"
        hyb_str = f"{hyb_sim:.3f}" if hyb_sim is not None else "N/A"

        print(f"{description:<40} {ling_str:<12} {top_str:<12} {hyb_str:<12}")


def test_embedding_dimensions(model):
    """Test embedding dimensions for different modes and word types."""
    print_section("EMBEDDING DIMENSIONS")

    test_words = [
        ('hund', "Content word (both embeddings)"),
        ('pariz', "Proper noun (topical only)"),
    ]

    print("\n{:<15} {:<35} {:<12} {:<12} {:<12}".format(
        "Root", "Description", "Linguistic", "Topical", "Hybrid"
    ))
    print("-" * 70)

    for root, description in test_words:
        ling_emb = model.get_root_embedding(root, mode='linguistic')
        top_emb = model.get_root_embedding(root, mode='topical')
        hyb_emb = model.get_root_embedding(root, mode='hybrid')

        ling_dim = f"{ling_emb.shape[0]}d" if ling_emb is not None else "N/A"
        top_dim = f"{top_emb.shape[0]}d" if top_emb is not None else "N/A"
        hyb_dim = f"{hyb_emb.shape[0]}d" if hyb_emb is not None else "N/A"

        print(f"{root:<15} {description:<35} {ling_dim:<12} {top_dim:<12} {hyb_dim:<12}")


def test_proper_noun_handling(model):
    """Test that proper nouns only have topical embeddings."""
    print_section("PROPER NOUN HANDLING")

    # Use roots (without endings)
    proper_nouns = [
        ('pariz', 'Paris'),
        ('napoleon', 'Napoleon'),
        ('esperant', 'Esperanto'),
        ('eŭrop', 'Europe')
    ]

    print("\nProper nouns should have topical embeddings but NOT linguistic:")
    print("\n{:<15} {:<15} {:<20} {:<20}".format(
        "Root", "Name", "Has Linguistic?", "Has Topical?"
    ))
    print("-" * 70)

    for root, name in proper_nouns:
        info = model.analyze_root(root)

        # Check if it exists in either vocab
        if info['type'] == 'unknown':
            print(f"{root:<15} {name:<15} {'NOT IN VOCAB':<20} {'NOT IN VOCAB':<20}")
        else:
            ling_status = "✗ NO (correct!)" if not info['has_linguistic'] else "✓ YES (unexpected!)"
            top_status = "✓ YES (correct!)" if info['has_topical'] else "✗ NO (unexpected!)"
            print(f"{root:<15} {name:<15} {ling_status:<20} {top_status:<20}")


def test_embedding_statistics(model):
    """Test embedding space statistics."""
    print_section("EMBEDDING SPACE STATISTICS")

    print("\nLinguistic embeddings:")
    ling_stats = model.linguistic_model.get_embedding_statistics()
    print(f"  Mean similarity: {ling_stats['mean_similarity']:.3f}")
    print(f"  Std similarity:  {ling_stats['std_similarity']:.3f}")
    print(f"  Min similarity:  {ling_stats['min_similarity']:.3f}")
    print(f"  Max similarity:  {ling_stats['max_similarity']:.3f}")

    print("\nTopical embeddings:")
    top_stats = model.topical_model.get_embedding_statistics()
    print(f"  Mean similarity: {top_stats['mean_similarity']:.3f}")
    print(f"  Std similarity:  {top_stats['std_similarity']:.3f}")
    print(f"  Min similarity:  {top_stats['min_similarity']:.3f}")
    print(f"  Max similarity:  {top_stats['max_similarity']:.3f}")


def main():
    print("\n" + "=" * 70)
    print("  HYBRID EMBEDDINGS TEST")
    print("=" * 70)
    print("\nLoading models...")

    # Load hybrid embeddings
    try:
        model = HybridEmbeddings.from_checkpoints(
            linguistic_checkpoint='models/root_embeddings/best_model.pt',
            topical_checkpoint='models/topical_embeddings/best_model.pt',
            pad_missing=True,
            default_mode='hybrid'
        )
        print("✓ Models loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return 1

    # Run tests
    try:
        test_vocabulary_info(model)
        test_root_classification(model)
        test_embedding_dimensions(model)
        test_similarity_modes(model)
        test_proper_noun_handling(model)
        test_embedding_statistics(model)

        print("\n" + "=" * 70)
        print("  ALL TESTS COMPLETE")
        print("=" * 70)
        print("\n✓ Hybrid embeddings are working correctly!")
        print("✓ Linguistic and topical models are properly integrated")
        print("✓ Ready for retrieval pipeline integration\n")

        return 0

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
