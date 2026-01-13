#!/usr/bin/env python3
"""
Query root embeddings to find semantically similar roots.

Usage:
    python scripts/query_root_similarity.py hund
    python scripts/query_root_similarity.py --top-k 20 pens
    python scripts/query_root_similarity.py --batch roots.txt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


def load_embeddings(model_path: Path) -> Tuple[torch.nn.Embedding, Dict[str, int], Dict[int, str]]:
    """Load root embeddings from trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')

    embedding = torch.nn.Embedding(
        checkpoint['vocab_size'],
        checkpoint['embedding_dim']
    )
    embedding.weight.data = checkpoint['model_state_dict']['embeddings.weight']

    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = checkpoint['idx_to_root']

    return embedding, root_to_idx, idx_to_root


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return (a @ b) / (torch.norm(a) * torch.norm(b))


def find_similar_roots(root: str,
                       embedding: torch.nn.Embedding,
                       root_to_idx: Dict[str, int],
                       idx_to_root: Dict[int, str],
                       top_k: int = 10,
                       min_similarity: float = 0.0) -> List[Tuple[str, float]]:
    """
    Find top-k most similar roots to the given root.

    Args:
        root: Root to query
        embedding: Trained embedding model
        root_to_idx: Root to index mapping
        idx_to_root: Index to root mapping
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of (root, similarity) tuples, sorted by similarity
    """
    if root not in root_to_idx:
        return []

    query_idx = root_to_idx[root]
    query_vec = embedding(torch.tensor(query_idx))

    # Compute similarities to all roots
    similarities = []
    for other_root, other_idx in root_to_idx.items():
        if other_root == root:
            continue  # Skip self

        other_vec = embedding(torch.tensor(other_idx))
        sim = cosine_similarity(query_vec, other_vec).item()

        if sim >= min_similarity:
            similarities.append((other_root, sim))

    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description='Query root embeddings for semantic similarity'
    )
    parser.add_argument('root', nargs='?', help='Root to query')
    parser.add_argument('--model', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to trained embedding model')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of similar roots to return')
    parser.add_argument('--min-similarity', type=float, default=0.0,
                       help='Minimum similarity threshold')
    parser.add_argument('--batch', type=Path,
                       help='File with roots to query (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.model}...")
    embedding, root_to_idx, idx_to_root = load_embeddings(args.model)
    print(f"Loaded {len(root_to_idx)} root embeddings\n")

    def query_root(root: str):
        """Query and display results for a single root."""
        if root not in root_to_idx:
            print(f"❌ Root '{root}' not in vocabulary")
            return

        results = find_similar_roots(
            root, embedding, root_to_idx, idx_to_root,
            args.top_k, args.min_similarity
        )

        if not results:
            print(f"No similar roots found for '{root}'")
            return

        print(f"Most similar to '{root}':")
        for rank, (similar_root, similarity) in enumerate(results, 1):
            print(f"  {rank:2d}. {similar_root:15s} {similarity:.3f}")
        print()

    # Batch mode
    if args.batch:
        with open(args.batch) as f:
            for line in f:
                root = line.strip()
                if root and not root.startswith('#'):
                    query_root(root)

    # Interactive mode
    elif args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        while True:
            try:
                root = input("\nQuery root: ").strip()
                if root.lower() in ['quit', 'exit', 'q']:
                    break
                if root:
                    query_root(root)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    # Single query mode
    elif args.root:
        query_root(args.root)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
