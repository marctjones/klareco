#!/usr/bin/env python3
"""
Quick Embedding Quality Test - Run on Checkpoints During Training

Tests semantic similarity on known Esperanto word pairs to validate
that embeddings are learning meaningful relationships.

Usage:
    python scripts/test_embedding_checkpoint.py \
        --checkpoint models/root_embeddings_phase1_fast/root_embeddings_checkpoint.pt

Example output:
    Similar pairs (should be >0.5):
      hund - kat: 0.73 ✓
      kur - mar: 0.68 ✓

    Dissimilar pairs (should be <0.3):
      hund - matematik: 0.12 ✓
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys


def load_embeddings(checkpoint_path):
    """Load embeddings from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']
    root_to_idx = checkpoint['root_to_idx']

    # Normalize embeddings for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings, vocab, root_to_idx


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return (emb1 @ emb2).item()


def find_nearest_neighbors(target_root, embeddings, vocab, root_to_idx, k=5):
    """Find k nearest neighbors to target root."""
    if target_root not in root_to_idx:
        return None

    target_idx = root_to_idx[target_root]
    target_emb = embeddings[target_idx]

    # Compute similarities to all other roots
    similarities = embeddings @ target_emb

    # Get top k (excluding self)
    top_k_indices = similarities.argsort(descending=True)[1:k+1]

    neighbors = []
    for idx in top_k_indices:
        root = vocab[idx]
        sim = similarities[idx].item()
        neighbors.append((root, sim))

    return neighbors


def test_checkpoint(checkpoint_path):
    """Run validation tests on checkpoint."""
    print("=" * 80)
    print(f"Testing checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Load embeddings
    embeddings, vocab, root_to_idx = load_embeddings(checkpoint_path)
    print(f"Loaded {len(vocab)} root embeddings ({embeddings.shape[1]}D)")
    print()

    # Define test pairs (similar should be high, dissimilar should be low)
    similar_pairs = [
        ("hund", "kat", "dog - cat (both animals)"),
        ("kur", "mar", "run - walk (both movement verbs)"),
        ("grand", "etn", "big - tiny (size adjectives)"),
        ("manĝ", "trink", "eat - drink (consumption verbs)"),
        ("tag", "nokt", "day - night (time periods)"),
        ("bel", "bon", "beautiful - good (positive qualities)"),
        ("parol", "dir", "speak - say (communication verbs)"),
        ("aŭt", "trajn", "car - train (vehicles)"),
    ]

    dissimilar_pairs = [
        ("hund", "matematik", "dog - mathematics"),
        ("kur", "arb", "run - tree"),
        ("manĝ", "pens", "eat - think"),
        ("tag", "teler", "day - plate"),
    ]

    # Test similar pairs
    print("SIMILAR PAIRS (should be >0.4):")
    print("-" * 80)
    similar_scores = []
    for root1, root2, desc in similar_pairs:
        if root1 not in root_to_idx or root2 not in root_to_idx:
            print(f"  {root1:10s} - {root2:10s}: MISSING (not in vocab)")
            continue

        idx1 = root_to_idx[root1]
        idx2 = root_to_idx[root2]
        sim = cosine_similarity(embeddings[idx1], embeddings[idx2])
        similar_scores.append(sim)

        status = "✓" if sim > 0.4 else "✗"
        print(f"  {root1:10s} - {root2:10s}: {sim:5.3f} {status}  ({desc})")

    print()

    # Test dissimilar pairs
    print("DISSIMILAR PAIRS (should be <0.3):")
    print("-" * 80)
    dissimilar_scores = []
    for root1, root2, desc in dissimilar_pairs:
        if root1 not in root_to_idx or root2 not in root_to_idx:
            print(f"  {root1:10s} - {root2:10s}: MISSING (not in vocab)")
            continue

        idx1 = root_to_idx[root1]
        idx2 = root_to_idx[root2]
        sim = cosine_similarity(embeddings[idx1], embeddings[idx2])
        dissimilar_scores.append(sim)

        status = "✓" if sim < 0.3 else "✗"
        print(f"  {root1:10s} - {root2:10s}: {sim:5.3f} {status}  ({desc})")

    print()

    # Summary statistics
    if similar_scores and dissimilar_scores:
        avg_similar = sum(similar_scores) / len(similar_scores)
        avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores)
        gap = avg_similar - avg_dissimilar

        print("SUMMARY:")
        print("-" * 80)
        print(f"  Average similar:    {avg_similar:.3f}")
        print(f"  Average dissimilar: {avg_dissimilar:.3f}")
        print(f"  Gap:                {gap:.3f}")

        if gap > 0.3:
            print(f"  Status: ✓ GOOD - Clear separation (gap > 0.3)")
        elif gap > 0.2:
            print(f"  Status: ⚠ OK - Moderate separation (gap > 0.2)")
        else:
            print(f"  Status: ✗ POOR - Insufficient separation (gap < 0.2)")
        print()

    # Show nearest neighbors for a few test roots
    print("NEAREST NEIGHBORS:")
    print("-" * 80)
    test_roots = ["hund", "kur", "bel", "manĝ"]

    for root in test_roots:
        if root not in root_to_idx:
            print(f"  {root}: MISSING (not in vocab)")
            continue

        neighbors = find_nearest_neighbors(root, embeddings, vocab, root_to_idx, k=5)
        print(f"  {root}:")
        for neighbor, sim in neighbors:
            print(f"    {neighbor:15s} {sim:.3f}")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test embedding quality from checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('models/root_embeddings_phase1_fast/root_embeddings_checkpoint.pt'),
        help='Path to checkpoint file (default: latest checkpoint)'
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"Training may not have completed first epoch yet.")
        sys.exit(1)

    test_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
