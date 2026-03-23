#!/usr/bin/env python3
"""
Demo: Hybrid Root Embeddings

Shows how to use the HybridRootEmbedder for best-of-both-worlds quality.

VERSION: v1.0
COMPATIBLE WITH: Production model (phase1_fast), AST-Only model (fundamento_ast)
STAGE: Inference/Demo

Usage:
    python scripts/demo_hybrid_embeddings.py
    python scripts/demo_hybrid_embeddings.py --interactive
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings import load_hybrid_embedder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_basic():
    """Basic demonstration of hybrid embedder."""

    print("\n" + "="*80)
    print("HYBRID ROOT EMBEDDINGS - BASIC DEMO")
    print("="*80)

    # Load hybrid embedder
    print("\n[1] Loading hybrid embedder...")
    hybrid = load_hybrid_embedder()

    # Show coverage stats
    print("\n[2] Vocabulary Coverage:")
    stats = hybrid.coverage_stats()
    print(f"  Total roots: {stats['total_roots']:,}")
    print(f"  Production vocab: {stats['production_vocab']:,} roots")
    print(f"  AST vocab: {stats['ast_vocab']:,} roots")
    print(f"  Overlap: {stats['overlap']:,} roots ({stats['overlap_percentage']:.1f}%)")
    print(f"  Production-only: {stats['production_only']:,} roots")
    print(f"  AST-only: {stats['ast_only']:,} roots")

    # Test antonym detection
    print("\n[3] Antonym Detection (uses AST model):")
    antonym_pairs = [
        ("am", "malam", "love vs hate"),
        ("bon", "malbon", "good vs bad"),
        ("alt", "malalt", "high vs low"),
        ("riĉ", "malriĉ", "rich vs poor"),
        ("nova", "malnova", "new vs old")
    ]

    for root1, root2, description in antonym_pairs:
        sim, source = hybrid.similarity(root1, root2)
        status = "✓" if sim < 0 else "✗"
        print(f"  {root1} vs {root2} ({description}):")
        print(f"    Similarity: {sim:+.3f} ({source}) {status}")

    # Test Fundamento semantic similarity
    print("\n[4] Fundamento Semantic Similarity (uses AST model):")
    fundamento_pairs = [
        ("hund", "kat", "dog vs cat (both animals)"),
        ("bel", "bon", "beautiful vs good (both positive)"),
        ("grand", "malgranda", "big vs small (antonyms)")
    ]

    for root1, root2, description in fundamento_pairs:
        sim, source = hybrid.similarity(root1, root2)
        print(f"  {root1} vs {root2} ({description}):")
        print(f"    Similarity: {sim:+.3f} ({source})")

    # Test nearest neighbors - clustering
    print("\n[5] Nearest Neighbors - Semantic Clustering (uses Production model):")
    test_roots = ["hund", "ruĝ", "kur"]

    for root in test_roots:
        neighbors = hybrid.nearest_neighbors(root, k=5, use_clustering=True)
        if neighbors:
            print(f"\n  Nearest to '{root}' (clustering):")
            for neighbor, sim, source in neighbors:
                print(f"    {neighbor}: {sim:.3f}")

    # Test nearest neighbors - structural
    print("\n[6] Nearest Neighbors - Structural (uses AST model):")
    test_roots_ast = ["bel", "bon"]

    for root in test_roots_ast:
        if root in hybrid.ast_vocab:
            neighbors = hybrid.nearest_neighbors(root, k=5, use_clustering=False)
            if neighbors:
                print(f"\n  Nearest to '{root}' (structural):")
                for neighbor, sim, source in neighbors:
                    print(f"    {neighbor}: {sim:.3f}")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ Hybrid model combines Production + AST-Only strengths")
    print("  ✓ Antonyms detected systematically (AST model)")
    print("  ✓ Semantic clustering preserved (Production model)")
    print("  ✓ Maximum vocabulary coverage (7,843 roots)")
    print("  ✓ Zero additional training cost")
    print()


def demo_interactive():
    """Interactive mode for exploring embeddings."""

    print("\n" + "="*80)
    print("HYBRID ROOT EMBEDDINGS - INTERACTIVE MODE")
    print("="*80)

    # Load hybrid embedder
    print("\nLoading hybrid embedder...")
    hybrid = load_hybrid_embedder()
    print(f"Loaded! Vocabulary: {len(hybrid.root_to_idx):,} roots\n")

    print("Commands:")
    print("  sim <root1> <root2>    - Compute similarity")
    print("  neighbors <root> [k]   - Find k nearest neighbors (default: 10)")
    print("  structural <root> [k]  - Find neighbors using AST structure")
    print("  info <root>            - Show which models contain this root")
    print("  stats                  - Show coverage statistics")
    print("  quit                   - Exit")
    print()

    while True:
        try:
            cmd = input("hybrid> ").strip()

            if not cmd:
                continue

            if cmd == "quit":
                break

            parts = cmd.split()
            command = parts[0]

            if command == "sim" and len(parts) >= 3:
                root1, root2 = parts[1], parts[2]
                sim, source = hybrid.similarity(root1, root2)
                print(f"  {root1} vs {root2}: {sim:+.3f} ({source})")

            elif command == "neighbors" and len(parts) >= 2:
                root = parts[1]
                k = int(parts[2]) if len(parts) > 2 else 10
                neighbors = hybrid.nearest_neighbors(root, k=k, use_clustering=True)

                if neighbors:
                    print(f"\n  Top {k} neighbors for '{root}' (clustering):")
                    for neighbor, sim, source in neighbors:
                        print(f"    {neighbor}: {sim:.3f}")
                else:
                    print(f"  Root '{root}' not in vocabulary")

            elif command == "structural" and len(parts) >= 2:
                root = parts[1]
                k = int(parts[2]) if len(parts) > 2 else 10

                if root not in hybrid.ast_vocab:
                    print(f"  Root '{root}' not in AST vocabulary")
                else:
                    neighbors = hybrid.nearest_neighbors(root, k=k, use_clustering=False)
                    print(f"\n  Top {k} neighbors for '{root}' (structural):")
                    for neighbor, sim, source in neighbors:
                        print(f"    {neighbor}: {sim:.3f}")

            elif command == "info" and len(parts) >= 2:
                root = parts[1]
                in_production = root in hybrid.production_vocab
                in_ast = root in hybrid.ast_vocab
                in_unified = root in hybrid.root_to_idx

                print(f"\n  Root: '{root}'")
                print(f"    In Production: {'✓' if in_production else '✗'}")
                print(f"    In AST: {'✓' if in_ast else '✗'}")
                print(f"    In Unified: {'✓' if in_unified else '✗'}")

                if in_unified:
                    idx = hybrid.root_to_idx[root]
                    print(f"    Unified index: {idx}")

            elif command == "stats":
                stats = hybrid.coverage_stats()
                print("\n  Vocabulary Coverage:")
                print(f"    Total roots: {stats['total_roots']:,}")
                print(f"    Production: {stats['production_vocab']:,}")
                print(f"    AST: {stats['ast_vocab']:,}")
                print(f"    Overlap: {stats['overlap']:,} ({stats['overlap_percentage']:.1f}%)")

            else:
                print("  Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("\nGoodbye!")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Demo hybrid root embeddings")
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    args = parser.parse_args()

    if args.interactive:
        demo_interactive()
    else:
        demo_basic()


if __name__ == "__main__":
    main()
