#!/usr/bin/env python3
"""
Demo: Affix Embeddings from Fundamento-Centered Training

This script demonstrates what we've accomplished in Phase 2:
- Trained 32-dimensional embeddings for Esperanto prefixes and suffixes
- Using pure Esperanto sources

Run: python scripts/demo_affix_embeddings.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_model(model_path: Path):
    """Load trained affix embeddings."""
    checkpoint = torch.load(model_path, map_location='cpu')

    prefix_vocab = checkpoint['prefix_vocab']
    suffix_vocab = checkpoint['suffix_vocab']
    embedding_dim = checkpoint['embedding_dim']

    # Extract embeddings
    prefix_embs = checkpoint['model_state_dict']['prefix_embeddings.weight']
    suffix_embs = checkpoint['model_state_dict']['suffix_embeddings.weight']

    return {
        'prefix_embeddings': prefix_embs,
        'suffix_embeddings': suffix_embs,
        'prefix_vocab': prefix_vocab,
        'suffix_vocab': suffix_vocab,
        'embedding_dim': embedding_dim,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'accuracy': checkpoint.get('accuracy', 'unknown'),
    }


def prefix_similarity(model, p1: str, p2: str) -> float:
    """Compute cosine similarity between two prefixes."""
    if p1 not in model['prefix_vocab'] or p2 not in model['prefix_vocab']:
        return None
    idx1, idx2 = model['prefix_vocab'][p1], model['prefix_vocab'][p2]
    emb1 = F.normalize(model['prefix_embeddings'][idx1], dim=0)
    emb2 = F.normalize(model['prefix_embeddings'][idx2], dim=0)
    return (emb1 * emb2).sum().item()


def suffix_similarity(model, s1: str, s2: str) -> float:
    """Compute cosine similarity between two suffixes."""
    if s1 not in model['suffix_vocab'] or s2 not in model['suffix_vocab']:
        return None
    idx1, idx2 = model['suffix_vocab'][s1], model['suffix_vocab'][s2]
    emb1 = F.normalize(model['suffix_embeddings'][idx1], dim=0)
    emb2 = F.normalize(model['suffix_embeddings'][idx2], dim=0)
    return (emb1 * emb2).sum().item()


def demo_prefix_relationships(model):
    """Demonstrate prefix embedding relationships."""
    print("\n" + "=" * 60)
    print("DEMO: Prefix Relationships")
    print("=" * 60)

    # Expected semantic groups
    prefix_pairs = [
        # Opposites/modifiers
        ("mal", "re", "mal (opposite) vs re (again)"),
        ("ek", "dis", "ek (begin) vs dis (apart)"),
        ("ge", "pra", "ge (both genders) vs pra (ancestral)"),

        # Spatial
        ("en", "el", "en (into) vs el (out of)"),
        ("sub", "super", "sub (under) vs super (over)"),
        ("antaŭ", "post", "antaŭ (before) vs post (after)"),

        # Negation/intensity
        ("mal", "ne", "mal (opposite) vs ne (not)" if "ne" in model['prefix_vocab'] else "N/A"),
        ("ĉef", "vic", "ĉef (chief) vs vic (vice)"),
    ]

    print("\nPrefix Pair Similarities:")
    print("-" * 50)
    for p1, p2, desc in prefix_pairs:
        sim = prefix_similarity(model, p1, p2)
        if sim is not None:
            bar = "█" * int(abs(sim) * 20)
            print(f"  {p1:8} ↔ {p2:8} = {sim:+.3f} {bar}")
        else:
            print(f"  {p1:8} ↔ {p2:8} = [not in vocab]")
    print()


def demo_suffix_relationships(model):
    """Demonstrate suffix embedding relationships."""
    print("\n" + "=" * 60)
    print("DEMO: Suffix Relationships")
    print("=" * 60)

    # Expected semantic groups
    suffix_pairs = [
        # Verbal/aspectual
        ("ig", "iĝ", "ig (causative) vs iĝ (inchoative)"),
        ("ant", "int", "ant (present part.) vs int (past part.)"),
        ("ant", "ont", "ant (present) vs ont (future)"),

        # Modal
        ("ebl", "ind", "ebl (can be) vs ind (should be)"),
        ("em", "end", "em (tendency) vs end (must be)"),

        # Size/degree
        ("et", "eg", "et (diminutive) vs eg (augmentative)"),

        # Derivational
        ("ej", "uj", "ej (place) vs uj (container)"),
        ("ist", "ism", "ist (person) vs ism (ideology)"),
        ("il", "ul", "il (tool) vs ul (person)"),
    ]

    print("\nSuffix Pair Similarities:")
    print("-" * 50)
    for s1, s2, desc in suffix_pairs:
        sim = suffix_similarity(model, s1, s2)
        if sim is not None:
            bar = "█" * int(abs(sim) * 20)
            print(f"  {s1:8} ↔ {s2:8} = {sim:+.3f} {bar}")
            print(f"    ({desc})")
        else:
            print(f"  {s1:8} ↔ {s2:8} = [not in vocab]")
    print()


def demo_affix_inventory(model):
    """Show all available affixes."""
    print("\n" + "=" * 60)
    print("DEMO: Affix Inventory")
    print("=" * 60)

    prefixes = [p for p in model['prefix_vocab'] if p != '<NONE>']
    suffixes = [s for s in model['suffix_vocab'] if s != '<NONE>']

    print(f"\nPrefixes ({len(prefixes)}):")
    print("  " + ", ".join(sorted(prefixes)))

    print(f"\nSuffixes ({len(suffixes)}):")
    print("  " + ", ".join(sorted(suffixes)))


def main():
    model_path = Path("models/affix_embeddings/best_model.pt")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run Phase 2 training first")
        sys.exit(1)

    print("Loading affix embeddings...")
    model = load_model(model_path)

    print(f"\nModel Statistics:")
    print(f"  Prefixes: {len(model['prefix_vocab'])}")
    print(f"  Suffixes: {len(model['suffix_vocab'])}")
    print(f"  Embedding dimension: {model['embedding_dim']}")
    print(f"  Training epoch: {model['epoch']}")
    print(f"  Best accuracy: {model['accuracy']:.4f}" if isinstance(model['accuracy'], float) else f"  Best accuracy: {model['accuracy']}")

    # Run demos
    demo_affix_inventory(model)
    demo_prefix_relationships(model)
    demo_suffix_relationships(model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
What we've accomplished (Phase 2):

  1. Trained 32d embeddings for 61 prefixes
  2. Trained 32d embeddings for 38 suffixes
  3. Used affix co-occurrence with roots
  4. Added known semantic affix pairs

Note: Affix training had limited pairs (33 total).
Future improvement: Use more sources for affix relationships.
""")


if __name__ == '__main__':
    main()
