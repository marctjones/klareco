#!/usr/bin/env python3
"""
Interactive demo for exploring trained morpheme embeddings.

Usage:
    python scripts/demo_morpheme_embeddings.py --similar "hundo"
    python scripts/demo_morpheme_embeddings.py --compose "dom" "-et" "o"
    python scripts/demo_morpheme_embeddings.py --interactive
"""

import argparse
import torch
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

from klareco.embeddings.compositional import CompositionalEmbedding


def load_model(checkpoint_path: Path):
    """Load trained morpheme model."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    root_vocab = checkpoint['root_vocab']
    prefix_vocab = checkpoint['prefix_vocab']
    suffix_vocab = checkpoint['suffix_vocab']

    # Reconstruct compositional embedding
    config = checkpoint.get('config', {})
    embed_dim = config.get('embed_dim', 128)
    composition_method = config.get('composition_method', 'sum')

    # Initialize embedding
    embedding = CompositionalEmbedding(
        root_vocab=root_vocab,
        prefix_vocab=prefix_vocab,
        suffix_vocab=suffix_vocab,
        embed_dim=embed_dim,
        composition_method=composition_method
    )

    # Load trained weights
    # Extract just the embedding weights from the full model state dict
    embedding_state = {
        k.replace('embedding.', ''): v
        for k, v in checkpoint['model_state_dict'].items()
        if k.startswith('embedding.')
    }
    embedding.load_state_dict(embedding_state)
    embedding.eval()

    # Handle both old and new checkpoint formats
    loss = checkpoint.get('train_loss', checkpoint.get('loss', 'unknown'))
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else loss

    print(f"✓ Loaded model (epoch {checkpoint['epoch']}, loss: {loss_str})")
    print(f"  Vocabulary: {len(root_vocab)} roots, {len(prefix_vocab)} prefixes, {len(suffix_vocab)} suffixes")

    return embedding, root_vocab, prefix_vocab, suffix_vocab


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return (v1 @ v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)


def find_similar_roots(
    embedding: CompositionalEmbedding,
    root_vocab: dict,
    target_root: str,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Find most similar roots to target root."""

    if target_root not in root_vocab:
        return [(f"❌ Root '{target_root}' not in vocabulary", 0.0)]

    # Get target embedding
    target_idx = root_vocab[target_root]
    target_embed = embedding.root_embeddings.weight[target_idx]

    # Compute similarities to all roots
    similarities = []
    for root, idx in root_vocab.items():
        if root == target_root or root in ['<PAD>', '<UNK>', '<MASK>']:
            continue

        root_embed = embedding.root_embeddings.weight[idx]
        sim = cosine_similarity(target_embed, root_embed).item()
        similarities.append((root, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def compose_word(
    embedding: CompositionalEmbedding,
    root: str,
    prefixes: List[str],
    suffixes: List[str],
    ending: str,
    root_vocab: dict,
    prefix_vocab: dict,
    suffix_vocab: dict
) -> torch.Tensor:
    """Compose a word from morphemes."""

    # Build word structure
    word_data = {
        'radiko': root,
        'prefikso': prefixes[0] if prefixes else None,
        'sufiksoj': suffixes,
        'vortspeco': ending  # o, a, e, etc.
    }

    # Get embedding
    with torch.no_grad():
        embed = embedding([word_data])[0]

    return embed


def find_nearest_words(
    embedding: CompositionalEmbedding,
    target_embed: torch.Tensor,
    root_vocab: dict,
    prefix_vocab: dict,
    suffix_vocab: dict,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Find words with embeddings nearest to target."""

    similarities = []

    # Sample common word patterns from vocabulary
    common_endings = ['o', 'a', 'e', 'as', 'is', 'os']
    common_suffixes = ['et', 'eg', 'il', 'ej', 'ul', 'in', 'an']
    common_prefixes = ['mal', 're', 'ek', 'dis', 'mis']

    # Generate candidate words
    for root in list(root_vocab.keys())[:500]:  # Sample top 500 roots
        if root in ['<PAD>', '<UNK>', '<MASK>']:
            continue

        # Base word
        for ending in common_endings:
            word_str = root + ending
            word_embed = compose_word(embedding, root, [], [], ending, root_vocab, prefix_vocab, suffix_vocab)
            sim = cosine_similarity(target_embed, word_embed).item()
            similarities.append((word_str, sim))

        # With suffix
        for suffix in common_suffixes:
            if suffix in suffix_vocab:
                for ending in ['o', 'a', 'e']:
                    word_str = f"{root}{suffix}{ending}"
                    word_embed = compose_word(embedding, root, [], [suffix], ending, root_vocab, prefix_vocab, suffix_vocab)
                    sim = cosine_similarity(target_embed, word_embed).item()
                    similarities.append((word_str, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def analyze_affix(
    embedding: CompositionalEmbedding,
    affix: str,
    affix_type: str,  # 'prefix' or 'suffix'
    vocab: dict
) -> torch.Tensor:
    """Get embedding for an affix."""

    if affix not in vocab:
        return None

    idx = vocab[affix]

    if affix_type == 'prefix':
        return embedding.prefix_embeddings.weight[idx]
    else:
        return embedding.suffix_embeddings.weight[idx]


def interactive_mode(
    embedding: CompositionalEmbedding,
    root_vocab: dict,
    prefix_vocab: dict,
    suffix_vocab: dict
):
    """Interactive exploration mode."""

    print("\n" + "="*60)
    print("MORPHEME EMBEDDING EXPLORER")
    print("="*60)
    print("\nCommands:")
    print("  similar <root>           - Find similar roots")
    print("  compose <root> <affix>*  - Compose word and find nearest")
    print("  affix <affix>            - Analyze affix")
    print("  vocab                    - Show vocabulary stats")
    print("  quit                     - Exit")
    print()

    while True:
        try:
            command = input("> ").strip()

            if not command:
                continue

            if command == 'quit':
                break

            parts = command.split()
            cmd = parts[0]

            if cmd == 'similar' and len(parts) >= 2:
                root = parts[1]
                print(f"\nMost similar to '{root}':")
                results = find_similar_roots(embedding, root_vocab, root, top_k=10)
                for word, sim in results:
                    print(f"  {word:<20} {sim:.3f}")

            elif cmd == 'compose' and len(parts) >= 2:
                root = parts[1]
                affixes = parts[2:] if len(parts) > 2 else []

                # Parse affixes
                prefixes = [a for a in affixes if a.startswith('-') is False and a in prefix_vocab]
                suffixes = [a for a in affixes if a in suffix_vocab]
                ending = affixes[-1] if affixes and len(affixes[-1]) == 1 else 'o'

                print(f"\nComposing: {root} + {' + '.join(affixes)}")
                embed = compose_word(embedding, root, prefixes, suffixes, ending, root_vocab, prefix_vocab, suffix_vocab)

                print("Nearest words:")
                results = find_nearest_words(embedding, embed, root_vocab, prefix_vocab, suffix_vocab, top_k=5)
                for word, sim in results:
                    print(f"  {word:<20} {sim:.3f}")

            elif cmd == 'affix' and len(parts) >= 2:
                affix = parts[1]

                # Determine type
                if affix in prefix_vocab:
                    affix_type = 'prefix'
                    vocab = prefix_vocab
                elif affix in suffix_vocab:
                    affix_type = 'suffix'
                    vocab = suffix_vocab
                else:
                    print(f"❌ Affix '{affix}' not found")
                    continue

                affix_embed = analyze_affix(embedding, affix, affix_type, vocab)
                if affix_embed is not None:
                    print(f"\nAffix: {affix} ({affix_type})")
                    print(f"Embedding norm: {torch.norm(affix_embed).item():.3f}")
                    print(f"Embedding: {affix_embed[:10].tolist()}")  # Show first 10 dims

            elif cmd == 'vocab':
                print(f"\nVocabulary Statistics:")
                print(f"  Roots:    {len(root_vocab):,}")
                print(f"  Prefixes: {len(prefix_vocab):,}")
                print(f"  Suffixes: {len(suffix_vocab):,}")
                print(f"\nTop 20 prefixes: {list(prefix_vocab.keys())[:20]}")
                print(f"Top 20 suffixes: {list(suffix_vocab.keys())[:20]}")

            else:
                print("❌ Unknown command or invalid arguments")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore morpheme embeddings")
    parser.add_argument('--model', type=Path, default=Path('models/morpheme_aware/best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--similar', type=str, help='Find similar roots to this root')
    parser.add_argument('--compose', nargs='+', help='Compose word from morphemes')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    # Load model
    embedding, root_vocab, prefix_vocab, suffix_vocab = load_model(args.model)

    if args.similar:
        print(f"\nMost similar to '{args.similar}':")
        results = find_similar_roots(embedding, root_vocab, args.similar, top_k=10)
        for word, sim in results:
            print(f"  {word:<20} {sim:.3f}")

    elif args.compose:
        root = args.compose[0]
        affixes = args.compose[1:] if len(args.compose) > 1 else []

        prefixes = [a for a in affixes if a.startswith('-') is False and a in prefix_vocab]
        suffixes = [a for a in affixes if a in suffix_vocab]
        ending = affixes[-1] if affixes and len(affixes[-1]) == 1 else 'o'

        print(f"\nComposing: {root} + {' + '.join(affixes)}")
        embed = compose_word(embedding, root, prefixes, suffixes, ending, root_vocab, prefix_vocab, suffix_vocab)

        print("Nearest words:")
        results = find_nearest_words(embedding, embed, root_vocab, prefix_vocab, suffix_vocab, top_k=10)
        for word, sim in results:
            print(f"  {word:<20} {sim:.3f}")

    elif args.interactive:
        interactive_mode(embedding, root_vocab, prefix_vocab, suffix_vocab)

    else:
        print("No action specified. Use --similar, --compose, or --interactive")
        print("Example: python scripts/demo_morpheme_embeddings.py --similar hundo")


if __name__ == '__main__':
    main()
