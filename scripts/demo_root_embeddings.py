#!/usr/bin/env python3
"""
Demo: Root Embeddings from Fundamento-Centered Training

This script demonstrates what we've accomplished in Stage 1 Phase 1:
- Trained 64-dimensional embeddings for Esperanto roots
- Using pure Esperanto sources (Fundamento UV, Ekzercaro, ReVo definitions)
- No cross-lingual contamination

Run:
    python scripts/demo_root_embeddings.py           # Run all demos
    python scripts/demo_root_embeddings.py -i       # Interactive mode
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_model(model_path: Path):
    """Load trained root embeddings."""
    checkpoint = torch.load(model_path, map_location='cpu')

    root_to_idx = checkpoint['root_to_idx']
    idx_to_root = checkpoint['idx_to_root']
    embedding_dim = checkpoint['embedding_dim']
    vocab_size = checkpoint['vocab_size']

    # Extract embeddings
    embeddings = checkpoint['model_state_dict']['embeddings.weight']

    return {
        'embeddings': embeddings,
        'root_to_idx': root_to_idx,
        'idx_to_root': idx_to_root,
        'embedding_dim': embedding_dim,
        'vocab_size': vocab_size,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'accuracy': checkpoint.get('correlation', 'unknown'),
    }


def get_embedding(model, root: str) -> torch.Tensor:
    """Get normalized embedding for a root."""
    if root not in model['root_to_idx']:
        return None
    idx = model['root_to_idx'][root]
    emb = model['embeddings'][idx]
    return F.normalize(emb, dim=0)


def similarity(model, root1: str, root2: str) -> float:
    """Compute cosine similarity between two roots."""
    emb1 = get_embedding(model, root1)
    emb2 = get_embedding(model, root2)
    if emb1 is None or emb2 is None:
        return None
    return (emb1 * emb2).sum().item()


def find_similar(model, root: str, top_k: int = 10):
    """Find most similar roots to a given root."""
    emb = get_embedding(model, root)
    if emb is None:
        return []

    # Compute similarities to all roots
    all_embs = F.normalize(model['embeddings'], dim=1)
    sims = (all_embs @ emb).tolist()

    # Sort and get top-k (excluding self)
    idx_to_root = model['idx_to_root']
    results = [(idx_to_root[i], sim) for i, sim in enumerate(sims)]
    results.sort(key=lambda x: x[1], reverse=True)

    # Filter out self
    results = [(r, s) for r, s in results if r != root]
    return results[:top_k]


def demo_semantic_relationships(model):
    """Demonstrate semantic relationships captured by embeddings."""
    print("\n" + "=" * 60)
    print("DEMO: Semantic Relationships in Root Embeddings")
    print("=" * 60)

    # Test pairs with expected relationships
    test_pairs = [
        # Family relationships
        ("patr", "fil", "father-son (family)"),
        ("patr", "frat", "father-brother (family)"),
        ("patr", "tabul", "father-table (unrelated)"),

        # Animals
        ("hund", "kat", "dog-cat (animals)"),
        ("hund", "best", "dog-beast (animals)"),
        ("hund", "libr", "dog-book (unrelated)"),

        # Actions
        ("kur", "mar", "run-walk (movement)"),
        ("leg", "skrib", "read-write (literacy)"),
        ("manĝ", "trink", "eat-drink (consumption)"),

        # Concepts
        ("bon", "bel", "good-beautiful (positive)"),
        ("grand", "alt", "big-tall (size)"),
        ("rapid", "lent", "fast-slow (speed)"),
    ]

    print("\nPairwise Similarities:")
    print("-" * 50)
    for root1, root2, description in test_pairs:
        sim = similarity(model, root1, root2)
        if sim is not None:
            bar = "█" * int(abs(sim) * 20)
            print(f"  {root1:8} ↔ {root2:8} = {sim:+.3f} {bar}")
            print(f"    ({description})")
        else:
            print(f"  {root1:8} ↔ {root2:8} = [not in vocab]")
    print()


def demo_nearest_neighbors(model):
    """Demonstrate nearest neighbor search."""
    print("\n" + "=" * 60)
    print("DEMO: Nearest Neighbors")
    print("=" * 60)

    query_roots = ["hund", "am", "lern", "dom", "akv"]

    for root in query_roots:
        if root not in model['root_to_idx']:
            print(f"\n'{root}' not in vocabulary")
            continue

        print(f"\nNearest neighbors to '{root}':")
        neighbors = find_similar(model, root, top_k=8)
        for neighbor, sim in neighbors:
            bar = "█" * int(sim * 20)
            print(f"  {neighbor:12} {sim:.3f} {bar}")


def demo_analogy(model):
    """Demonstrate word analogies (if relationships are captured)."""
    print("\n" + "=" * 60)
    print("DEMO: Semantic Clusters")
    print("=" * 60)

    # Group related roots and check if they cluster
    clusters = {
        "Family": ["patr", "matr", "fil", "frat", "fianĉ", "edz"],
        "Animals": ["hund", "kat", "bird", "fiŝ", "ĉeval", "bov"],
        "Body": ["kap", "man", "okul", "buŝ", "nas", "orel"],
        "Time": ["tag", "nokt", "hor", "minut", "jar", "monat"],
        "Places": ["dom", "urb", "land", "lok", "ĉambr", "strat"],
    }

    for cluster_name, roots in clusters.items():
        print(f"\n{cluster_name}:")
        # Find which roots are in vocab
        valid_roots = [r for r in roots if r in model['root_to_idx']]
        if len(valid_roots) < 2:
            print("  (insufficient roots in vocabulary)")
            continue

        # Compute average pairwise similarity within cluster
        sims = []
        for i, r1 in enumerate(valid_roots):
            for r2 in valid_roots[i+1:]:
                sim = similarity(model, r1, r2)
                if sim is not None:
                    sims.append(sim)

        if sims:
            avg_sim = sum(sims) / len(sims)
            print(f"  Roots: {', '.join(valid_roots)}")
            print(f"  Average intra-cluster similarity: {avg_sim:.3f}")




def interactive_mode(model):
    """Interactive exploration of root embeddings."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  <root>           - Find similar roots")
    print("  <root1> <root2>  - Compute similarity between two roots")
    print("  list             - Show sample roots")
    print("  quit             - Exit")
    print()

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == 'quit':
            print("Goodbye!")
            break

        if query.lower() == 'list':
            # Show sample roots
            sample = list(model['root_to_idx'].keys())[:50]
            print("Sample roots:", ", ".join(sample))
            continue

        parts = query.split()

        if len(parts) == 1:
            # Find similar roots
            root = parts[0]
            if root not in model['root_to_idx']:
                print(f"  '{root}' not in vocabulary")
                # Suggest similar
                suggestions = [r for r in model['root_to_idx'] if root in r][:5]
                if suggestions:
                    print(f"  Did you mean: {', '.join(suggestions)}?")
                continue

            neighbors = find_similar(model, root, top_k=10)
            print(f"  Most similar to '{root}':")
            for neighbor, sim in neighbors:
                bar = "█" * int(sim * 20)
                print(f"    {neighbor:12} {sim:.3f} {bar}")

        elif len(parts) == 2:
            # Compare two roots
            root1, root2 = parts
            sim = similarity(model, root1, root2)
            if sim is None:
                missing = [r for r in [root1, root2] if r not in model['root_to_idx']]
                print(f"  Not in vocabulary: {', '.join(missing)}")
            else:
                bar = "█" * int(abs(sim) * 20)
                print(f"  {root1} ↔ {root2} = {sim:+.3f} {bar}")

        else:
            print("  Usage: <root> or <root1> <root2>")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Demo root embeddings')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'),
                        help='Path to model checkpoint')
    args = parser.parse_args()

    model_path = args.model

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run the training first: ./scripts/run_fundamento_training.sh")
        sys.exit(1)

    print("Loading root embeddings...")
    model = load_model(model_path)

    print(f"\nModel Statistics:")
    print(f"  Vocabulary size: {model['vocab_size']:,} roots")
    print(f"  Embedding dimension: {model['embedding_dim']}")
    print(f"  Training epoch: {model['epoch']}")
    print(f"  Correlation: {model['accuracy']:.4f}" if isinstance(model['accuracy'], float) else f"  Correlation: {model['accuracy']}")

    if args.interactive:
        interactive_mode(model)
    else:
        # Run demos
        demo_semantic_relationships(model)
        demo_nearest_neighbors(model)
        demo_analogy(model)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        corr_str = f"{model['accuracy']:.4f}" if isinstance(model['accuracy'], float) else str(model['accuracy'])
        print(f"""
Root Embedding Model Statistics:
  - Vocabulary: {model['vocab_size']:,} roots
  - Dimensions: {model['embedding_dim']}d
  - Correlation: {corr_str}

Training Data (TRAINING_PLAN_V3):
  - Fundamento Universala Vortaro (roots)
  - Fundamento Ekzercaro (sentences)
  - Reta Vortaro (ReVo) definitions

Pure Esperanto semantics - no cross-lingual contamination!
""")
        print("Try interactive mode: python scripts/demo_root_embeddings.py -i")


if __name__ == '__main__':
    main()
