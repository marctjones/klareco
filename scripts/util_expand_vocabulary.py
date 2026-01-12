#!/usr/bin/env python3
"""
Expand Vocabulary with New Roots.

Adds new roots to the vocabulary and updates model embeddings.
Can optionally fine-tune the new embeddings on example sentences.

Usage:
    # Add specific roots
    python scripts/expand_vocabulary.py --roots "blogo,novvorto,podkasto"

    # Add roots from file
    python scripts/expand_vocabulary.py --roots-file candidates.txt

    # Add and fine-tune
    python scripts/expand_vocabulary.py --roots-file candidates.txt --fine-tune
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.embeddings.unknown_tracker import UnknownRootTracker


def load_roots_from_file(filepath: Path) -> List[str]:
    """Load roots from a text file (one per line)."""
    roots = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            root = line.strip()
            if root and not root.startswith('#'):
                roots.append(root)
    return roots


def find_similar_roots(
    new_root: str,
    existing_vocab: dict,
    max_similar: int = 5,
) -> List[str]:
    """
    Find morphologically similar roots in vocabulary.

    Simple heuristic: roots that share prefix characters.
    """
    similar = []

    # Try to find roots with same prefix
    for prefix_len in range(min(4, len(new_root)), 0, -1):
        prefix = new_root[:prefix_len]
        for root in existing_vocab:
            if root.startswith(prefix) and root != new_root:
                similar.append(root)
                if len(similar) >= max_similar:
                    return similar

    return similar


def main():
    parser = argparse.ArgumentParser(
        description="Expand vocabulary with new roots"
    )
    parser.add_argument(
        "--roots",
        type=str,
        help="Comma-separated list of roots to add"
    )
    parser.add_argument(
        "--roots-file",
        type=Path,
        help="File with roots to add (one per line)"
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("data/vocabularies"),
        help="Vocabulary directory"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/tree_lstm_compositional/best_model.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help="Output path for expanded model (default: overwrites input)"
    )
    parser.add_argument(
        "--initialization",
        choices=["average", "similar", "random", "zero"],
        default="similar",
        help="How to initialize new embeddings (default: similar)"
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune new embeddings on tracked contexts"
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--tracker-file",
        type=Path,
        default=Path("data/unknown_roots.json"),
        help="Unknown roots tracker file (for contexts)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Get roots to add
    new_roots = []
    if args.roots:
        new_roots = [r.strip() for r in args.roots.split(',')]
    elif args.roots_file:
        if not args.roots_file.exists():
            print(f"Error: Roots file not found: {args.roots_file}")
            return 1
        new_roots = load_roots_from_file(args.roots_file)
    else:
        print("Error: Must specify --roots or --roots-file")
        return 1

    if not new_roots:
        print("No roots to add.")
        return 0

    print(f"\nRoots to add: {len(new_roots)}")
    for root in new_roots[:10]:
        print(f"  - {root}")
    if len(new_roots) > 10:
        print(f"  ... and {len(new_roots) - 10} more")

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab_dir}...")
    if not args.vocab_dir.exists():
        print(f"Error: Vocabulary directory not found: {args.vocab_dir}")
        return 1

    embedding = CompositionalEmbedding.from_vocabulary_files(
        args.vocab_dir,
        embed_dim=128,
    )
    old_vocab_size = len(embedding.root_vocab)
    print(f"Current vocabulary size: {old_vocab_size:,} roots")

    # Filter already-known roots
    unknown_roots = [r for r in new_roots if r not in embedding.root_vocab]
    known_roots = [r for r in new_roots if r in embedding.root_vocab]

    if known_roots:
        print(f"\nSkipping {len(known_roots)} roots already in vocabulary:")
        for root in known_roots[:5]:
            print(f"  - {root}")
        if len(known_roots) > 5:
            print(f"  ... and {len(known_roots) - 5} more")

    if not unknown_roots:
        print("\nNo new roots to add (all already in vocabulary).")
        return 0

    print(f"\nWill add {len(unknown_roots)} new roots")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return 0

    # Find similar roots for initialization
    similar_roots = {}
    if args.initialization == 'similar':
        print("\nFinding similar roots for initialization...")
        for root in unknown_roots:
            similar = find_similar_roots(root, embedding.root_vocab)
            if similar:
                similar_roots[root] = similar
                print(f"  {root} -> {', '.join(similar[:3])}")

    # Load model if exists
    model_loaded = False
    if args.model_path.exists():
        print(f"\nLoading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location='cpu')

        # Get config
        config = checkpoint.get('config', {})
        embed_dim = config.get('embed_dim', 128)

        # Reload with correct embed_dim
        embedding = CompositionalEmbedding.from_vocabulary_files(
            args.vocab_dir,
            embed_dim=embed_dim,
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            # Extract just the embedding weights
            state_dict = checkpoint['model_state_dict']
            if 'compositional_embedding.root_embed.weight' in state_dict:
                embedding.root_embed.weight.data = state_dict[
                    'compositional_embedding.root_embed.weight'
                ]
                model_loaded = True
                print("  Loaded embedding weights from model")
    else:
        print(f"\nNote: Model not found at {args.model_path}")
        print("  Will save expanded vocabulary only (no model update)")

    # Expand vocabulary
    print(f"\nExpanding vocabulary...")
    num_added = embedding.expand_vocabulary(
        unknown_roots,
        initialization=args.initialization,
        similar_roots=similar_roots if args.initialization == 'similar' else None,
    )

    new_vocab_size = len(embedding.root_vocab)
    print(f"Vocabulary expanded: {old_vocab_size:,} -> {new_vocab_size:,}")

    # Fine-tune if requested
    if args.fine_tune and args.tracker_file.exists():
        print("\nFine-tuning new embeddings...")
        tracker = UnknownRootTracker(args.tracker_file)

        # Get contexts for new roots
        contexts = {}
        for root in unknown_roots:
            candidates = tracker.get_candidates(min_count=1, limit=1000)
            for c in candidates:
                if c['root'] == root and c['contexts']:
                    contexts[root] = c['contexts']
                    break

        if contexts:
            print(f"  Found contexts for {len(contexts)} roots")
            # TODO: Implement actual fine-tuning loop
            # This would require parsing the contexts and training
            print("  [Fine-tuning not yet implemented - using initialization only]")
        else:
            print("  No contexts found for fine-tuning")

    # Save updated vocabulary
    print(f"\nSaving expanded vocabulary to {args.vocab_dir}...")
    embedding.save_vocabularies(args.vocab_dir)

    # Save updated model if we loaded one
    if model_loaded:
        output_path = args.output_model or args.model_path
        print(f"Saving updated model to {output_path}...")

        # Update the checkpoint
        checkpoint['model_state_dict']['compositional_embedding.root_embed.weight'] = \
            embedding.root_embed.weight.data

        # Update config
        if 'config' in checkpoint:
            checkpoint['config']['num_roots'] = new_vocab_size

        torch.save(checkpoint, output_path)
        print("  Model saved successfully")

    # Update tracker
    if args.tracker_file.exists():
        tracker = UnknownRootTracker(args.tracker_file)
        tracker.mark_added(unknown_roots)
        tracker.save()
        print(f"  Marked {len(unknown_roots)} roots as added in tracker")

    print("\n" + "=" * 60)
    print("Vocabulary expansion complete!")
    print("=" * 60)
    print(f"  Roots added:      {num_added}")
    print(f"  New vocab size:   {new_vocab_size:,}")
    print(f"  Initialization:   {args.initialization}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
