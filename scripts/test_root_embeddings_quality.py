#!/usr/bin/env python3
"""
Test Root Embeddings Quality - Semantic Similarity Demo

Demonstrates that the trained root embeddings capture semantic relationships
by showing nearest neighbors and computing similarities for test pairs.

Usage:
    python scripts/test_root_embeddings_quality.py
    python scripts/test_root_embeddings_quality.py --embeddings path/to/model.pt
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path


def load_embeddings(checkpoint_path):
    """Load embeddings from checkpoint."""
    print(f"Loading embeddings from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']
    root_to_idx = checkpoint['root_to_idx']

    # Normalize embeddings for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    print(f"Loaded {len(vocab)} roots with {embeddings.shape[1]}D embeddings\n")
    return embeddings, vocab, root_to_idx


def find_similar_roots(target_root, embeddings, vocab, root_to_idx, k=10):
    """Find k most similar roots to target."""
    if target_root not in root_to_idx:
        return None

    target_idx = root_to_idx[target_root]
    target_emb = embeddings[target_idx]

    # Compute similarities to all roots
    similarities = embeddings @ target_emb

    # Get top k (excluding self at index 0)
    top_k_indices = similarities.argsort(descending=True)[1:k+1]

    results = []
    for idx in top_k_indices:
        root = vocab[idx]
        sim = similarities[idx].item()
        results.append((root, sim))

    return results


def demo_semantic_queries():
    """Demonstrate semantic query expansion using embeddings."""
    parser = argparse.ArgumentParser(description='Test root embeddings quality')
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'),
        help='Path to root embeddings checkpoint'
    )
    args = parser.parse_args()

    if not args.embeddings.exists():
        print(f"ERROR: Embeddings not found at {args.embeddings}")
        return

    # Load embeddings
    embeddings, vocab, root_to_idx = load_embeddings(args.embeddings)

    print("="*80)
    print("SEMANTIC SIMILARITY DEMONSTRATION")
    print("="*80)
    print()

    # Test queries - what would semantic expansion give us?
    test_cases = [
        ("hund", "Query about dogs → find related animal concepts"),
        ("kur", "Query about running → find related movement verbs"),
        ("manĝ", "Query about eating → find related consumption"),
        ("bel", "Query about beauty → find related positive qualities"),
        ("land", "Query about countries → find related geographic concepts"),
    ]

    for root, description in test_cases:
        print(f"📝 {description}")
        print(f"   Root: '{root}'")
        print()

        similar = find_similar_roots(root, embeddings, vocab, root_to_idx, k=10)

        if similar is None:
            print(f"   ⚠️  Root '{root}' not in vocabulary")
            print()
            continue

        print(f"   🔍 Semantically similar roots (for query expansion):")
        for i, (similar_root, sim) in enumerate(similar, 1):
            print(f"      {i:2d}. {similar_root:15s} (similarity: {sim:.3f})")

        print()
        print()

    # Show some specific similarity examples
    print("="*80)
    print("SPECIFIC SIMILARITY EXAMPLES")
    print("="*80)
    print()

    test_pairs = [
        ("hund", "kat", "dog - cat (both pets)"),
        ("kur", "mar", "run - walk (both movement)"),
        ("tag", "nokt", "day - night (time periods)"),
        ("grand", "malgranda", "big - small (opposites but both size)"),
        ("manĝ", "trink", "eat - drink (consumption)"),
    ]

    for root1, root2, desc in test_pairs:
        if root1 not in root_to_idx or root2 not in root_to_idx:
            continue

        idx1 = root_to_idx[root1]
        idx2 = root_to_idx[root2]
        sim = (embeddings[idx1] @ embeddings[idx2]).item()

        status = "✓" if sim > 0.3 else "~"
        print(f"{status} {root1:12s} - {root2:12s}: {sim:5.3f}  ({desc})")

    print()
    print("="*80)
    print("APPLICATION: Semantic Query Expansion")
    print("="*80)
    print()
    print("Query: 'Kiu fondis Esperanton?' (Who founded Esperanto?)")
    print("Original roots: fond (found), esperant (Esperanto)")
    print()

    # Expand "fond"
    print("Expanding 'fond' (found):")
    similar = find_similar_roots("fond", embeddings, vocab, root_to_idx, k=5)
    if similar:
        print("  Would add to query:", ", ".join([r for r, s in similar if s > 0.4]))
    else:
        print("  Root not in vocabulary")

    print()
    print("Expanding 'esperant':")
    similar = find_similar_roots("esperant", embeddings, vocab, root_to_idx, k=5)
    if similar:
        print("  Would add to query:", ", ".join([r for r, s in similar if s > 0.4]))
    else:
        print("  Root not in vocabulary")

    print()
    print("="*80)
    print("✅ Embeddings loaded and working!")
    print("   Ready for integration into retrieval pipeline")
    print("="*80)


if __name__ == '__main__':
    demo_semantic_queries()
