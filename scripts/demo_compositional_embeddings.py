#!/usr/bin/env python3
"""
Demo: Compositional Embeddings (Root + Affix)

This script demonstrates how the trained Stage 1 models work together:
- Root embeddings: Semantic meaning of base words (64d, ~10K roots)
- Affix embeddings: Semantic transforms from prefixes/suffixes (12d)

Key insight: Word meaning = root_embedding + prefix_transforms + suffix_transforms

Examples:
    python scripts/demo_compositional_embeddings.py
    python scripts/demo_compositional_embeddings.py --interactive
    python scripts/demo_compositional_embeddings.py --word malbonulo
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


class CompositionalEmbedder:
    """Combine root and affix embeddings for full word representation."""

    def __init__(
        self,
        root_model_path: Path = Path("models/root_embeddings/best_model.pt"),
        affix_model_path: Path = Path("models/affix_embeddings/best_model.pt"),
    ):
        # Load root embeddings
        print(f"Loading root embeddings from {root_model_path}...")
        root_ckpt = torch.load(root_model_path, map_location='cpu', weights_only=False)
        self.root_emb = root_ckpt['model_state_dict']['embeddings.weight'].numpy()
        self.root_to_idx = root_ckpt['root_to_idx']
        self.idx_to_root = {v: k for k, v in self.root_to_idx.items()}
        self.root_dim = self.root_emb.shape[1]
        print(f"  Loaded {len(self.root_to_idx)} roots, {self.root_dim}d")

        # Load affix embeddings
        print(f"Loading affix embeddings from {affix_model_path}...")
        affix_ckpt = torch.load(affix_model_path, map_location='cpu', weights_only=False)
        self.prefix_emb = affix_ckpt['model_state_dict']['prefix_embeddings.weight'].numpy()
        self.suffix_emb = affix_ckpt['model_state_dict']['suffix_embeddings.weight'].numpy()
        self.prefix_vocab = affix_ckpt['prefix_vocab']
        self.suffix_vocab = affix_ckpt['suffix_vocab']
        self.affix_dim = self.prefix_emb.shape[1]
        print(f"  Loaded {len(self.prefix_vocab)} prefixes, {len(self.suffix_vocab)} suffixes, {self.affix_dim}d")

        # Precompute normalized embeddings for similarity search
        self.root_emb_norm = self.root_emb / (np.linalg.norm(self.root_emb, axis=1, keepdims=True) + 1e-8)

    def parse_word(self, word: str) -> Optional[Dict]:
        """Parse a single word to extract root and affixes."""
        try:
            ast = parse(word)
            # Find the word node in the AST
            def find_word(node):
                if isinstance(node, dict):
                    if node.get('tipo') == 'vorto':
                        return node
                    for v in node.values():
                        result = find_word(v)
                        if result:
                            return result
                elif isinstance(node, list):
                    for item in node:
                        result = find_word(item)
                        if result:
                            return result
                return None
            return find_word(ast)
        except Exception as e:
            return None

    def get_root_embedding(self, root: str) -> Optional[np.ndarray]:
        """Get embedding for a root."""
        if root in self.root_to_idx:
            return self.root_emb[self.root_to_idx[root]]
        # Try lowercase
        root_lower = root.lower()
        if root_lower in self.root_to_idx:
            return self.root_emb[self.root_to_idx[root_lower]]
        return None

    def get_prefix_embedding(self, prefix: str) -> Optional[np.ndarray]:
        """Get embedding for a prefix."""
        if prefix in self.prefix_vocab and prefix != '<NONE>':
            return self.prefix_emb[self.prefix_vocab[prefix]]
        return None

    def get_suffix_embedding(self, suffix: str) -> Optional[np.ndarray]:
        """Get embedding for a suffix."""
        if suffix in self.suffix_vocab and suffix != '<NONE>':
            return self.suffix_emb[self.suffix_vocab[suffix]]
        return None

    def embed_word(self, word: str, verbose: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Create compositional embedding for a word.

        Returns (embedding, info) where info contains parse details.
        """
        info = {
            'word': word,
            'root': None,
            'prefixes': [],
            'suffixes': [],
            'root_found': False,
            'prefix_contributions': [],
            'suffix_contributions': [],
        }

        # Parse the word
        word_node = self.parse_word(word)
        if not word_node:
            info['error'] = 'Parse failed'
            return None, info

        # Extract components
        root = word_node.get('radiko', '')
        info['root'] = root

        # Get prefixes (handle both 'prefikso' and 'prefiksoj')
        prefixes = word_node.get('prefiksoj', [])
        if not prefixes:
            p = word_node.get('prefikso')
            if p:
                prefixes = [p]
        info['prefixes'] = prefixes

        # Get suffixes
        suffixes = word_node.get('sufiksoj', [])
        info['suffixes'] = suffixes

        # Get root embedding
        root_emb = self.get_root_embedding(root)
        if root_emb is None:
            info['error'] = f'Unknown root: {root}'
            return None, info

        info['root_found'] = True
        emb = root_emb.copy()

        if verbose:
            print(f"  Root '{root}': {self.root_dim}d embedding")

        # Add prefix contributions (scaled, projected to root dimension)
        for p in prefixes:
            prefix_emb = self.get_prefix_embedding(p)
            if prefix_emb is not None:
                # Zero-pad affix embedding to match root dimension
                contribution = np.zeros(self.root_dim)
                contribution[:self.affix_dim] = prefix_emb * 0.3  # Scale down
                emb = emb + contribution
                info['prefix_contributions'].append(p)
                if verbose:
                    print(f"  Prefix '{p}': added {self.affix_dim}d transform (scaled 0.3)")

        # Add suffix contributions
        for s in suffixes:
            suffix_emb = self.get_suffix_embedding(s)
            if suffix_emb is not None:
                contribution = np.zeros(self.root_dim)
                contribution[:self.affix_dim] = suffix_emb * 0.2  # Scale down
                emb = emb + contribution
                info['suffix_contributions'].append(s)
                if verbose:
                    print(f"  Suffix '{s}': added {self.affix_dim}d transform (scaled 0.2)")

        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb, info

    def similarity(self, word1: str, word2: str) -> Tuple[float, Dict, Dict]:
        """Compute cosine similarity between two words."""
        emb1, info1 = self.embed_word(word1)
        emb2, info2 = self.embed_word(word2)

        if emb1 is None or emb2 is None:
            return 0.0, info1, info2

        sim = float(np.dot(emb1, emb2))
        return sim, info1, info2

    def find_similar_roots(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find roots most similar to the given word's embedding."""
        emb, info = self.embed_word(word)
        if emb is None:
            return []

        # Normalize query
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        # Compute similarities to all roots
        sims = np.dot(self.root_emb_norm, emb_norm)

        # Get top-k
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            root = self.idx_to_root[idx]
            results.append((root, float(sims[idx])))

        return results

    def demonstrate_affix_effect(self, root: str, affixes: List[str]):
        """Show how affixes transform a root's meaning."""
        print(f"\n{'='*60}")
        print(f"Affix Effects on Root: {root}")
        print(f"{'='*60}")

        # Get base root embedding
        base_emb = self.get_root_embedding(root)
        if base_emb is None:
            print(f"  Unknown root: {root}")
            return

        base_norm = base_emb / (np.linalg.norm(base_emb) + 1e-8)

        # Show similar roots to base
        print(f"\nBase '{root}' similar to:")
        sims = np.dot(self.root_emb_norm, base_norm)
        top_indices = np.argsort(sims)[::-1][1:6]  # Skip self
        for idx in top_indices:
            print(f"  {self.idx_to_root[idx]}: {sims[idx]:.3f}")

        # Try each affix
        for affix in affixes:
            # Construct word
            if affix in self.prefix_vocab:
                word = affix + root + 'o'  # Assume noun
                affix_type = 'prefix'
            elif affix in self.suffix_vocab:
                word = root + affix + 'o'
                affix_type = 'suffix'
            else:
                print(f"\n  Unknown affix: {affix}")
                continue

            print(f"\nWith {affix_type} '{affix}' → {word}:")
            emb, info = self.embed_word(word)
            if emb is not None:
                # Find similar roots
                sims = np.dot(self.root_emb_norm, emb)
                top_indices = np.argsort(sims)[::-1][:5]
                for idx in top_indices:
                    r = self.idx_to_root[idx]
                    marker = "←" if r == root else ""
                    print(f"  {r}: {sims[idx]:.3f} {marker}")


def demo_word_pairs(embedder: CompositionalEmbedder):
    """Demonstrate similarity between word pairs."""
    print("\n" + "="*60)
    print("Word Pair Similarities")
    print("="*60)

    pairs = [
        # Same root, different affixes
        ("bona", "malbona"),      # good vs bad (mal- = opposite)
        ("hundo", "hundeto"),     # dog vs little dog (-et = diminutive)
        ("legi", "relegi"),       # read vs re-read (re- = again)
        ("skribi", "skribisto"),  # write vs writer (-ist = profession)

        # Similar meanings
        ("granda", "grandega"),   # big vs huge (-eg = augmentative)
        ("paroli", "diri"),       # speak vs say (different roots, similar meaning)

        # Related concepts
        ("libro", "librejo"),     # book vs library (-ej = place)
        ("instrui", "instruisto"),# teach vs teacher
        ("lerni", "lernejo"),     # learn vs school

        # Antonyms via mal-
        ("rapida", "malrapida"),  # fast vs slow
        ("juna", "maljuna"),      # young vs old
        ("riĉa", "malriĉa"),      # rich vs poor
    ]

    for w1, w2 in pairs:
        sim, info1, info2 = embedder.similarity(w1, w2)

        # Build component strings
        def format_components(info):
            parts = [info['root'] or '?']
            if info['prefixes']:
                parts = info['prefixes'] + parts
            if info['suffixes']:
                parts = parts + info['suffixes']
            return '+'.join(parts)

        c1 = format_components(info1)
        c2 = format_components(info2)

        status = ""
        if info1.get('error'):
            status = f" [{info1['error']}]"
        elif info2.get('error'):
            status = f" [{info2['error']}]"

        print(f"  {w1} ({c1}) vs {w2} ({c2}): {sim:.3f}{status}")


def demo_morphological_composition(embedder: CompositionalEmbedder):
    """Show how morphemes combine to create meaning."""
    print("\n" + "="*60)
    print("Morphological Composition")
    print("="*60)

    # Complex words with multiple affixes
    words = [
        "malbonulo",      # mal+bon+ul+o = bad person
        "gepatroj",       # ge+patr+o+j = parents (both genders)
        "lernejestro",    # lern+ej+estr+o = school principal
        "rehospitaligo",  # re+hospital+ig+o = re-hospitalization
        "malriĉulejo",    # mal+riĉ+ul+ej+o = place for poor people
    ]

    for word in words:
        print(f"\n{word}:")
        emb, info = embedder.embed_word(word, verbose=True)

        if emb is not None:
            # Find similar roots
            similar = embedder.find_similar_roots(word, top_k=5)
            print(f"  Similar roots: {', '.join([f'{r}({s:.2f})' for r, s in similar])}")


def interactive_mode(embedder: CompositionalEmbedder):
    """Interactive exploration mode."""
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  <word>           - Analyze a word")
    print("  <word1> <word2>  - Compare two words")
    print("  similar <word>   - Find similar roots")
    print("  affix <root>     - Show affix effects")
    print("  quit             - Exit")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.lower() == 'quit':
            break

        parts = line.split()

        if parts[0] == 'similar' and len(parts) > 1:
            word = parts[1]
            print(f"\nRoots similar to '{word}':")
            similar = embedder.find_similar_roots(word, top_k=15)
            for root, sim in similar:
                print(f"  {root}: {sim:.3f}")

        elif parts[0] == 'affix' and len(parts) > 1:
            root = parts[1]
            affixes = ['mal', 're', 'ek', 'ul', 'ej', 'ist', 'et', 'eg']
            embedder.demonstrate_affix_effect(root, affixes)

        elif len(parts) == 2:
            w1, w2 = parts
            sim, info1, info2 = embedder.similarity(w1, w2)
            print(f"\n{w1} vs {w2}:")
            print(f"  {w1}: root={info1['root']}, prefixes={info1['prefixes']}, suffixes={info1['suffixes']}")
            print(f"  {w2}: root={info2['root']}, prefixes={info2['prefixes']}, suffixes={info2['suffixes']}")
            print(f"  Similarity: {sim:.3f}")

        else:
            word = parts[0]
            print(f"\nAnalyzing '{word}':")
            emb, info = embedder.embed_word(word, verbose=True)
            if emb is not None:
                print(f"\n  Top similar roots:")
                similar = embedder.find_similar_roots(word, top_k=10)
                for root, sim in similar:
                    print(f"    {root}: {sim:.3f}")
            else:
                print(f"  Error: {info.get('error', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description='Demo: Compositional Embeddings (Root + Affix)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_compositional_embeddings.py
  python scripts/demo_compositional_embeddings.py --interactive
  python scripts/demo_compositional_embeddings.py --word malbonulo
  python scripts/demo_compositional_embeddings.py --compare bona malbona
        """
    )
    parser.add_argument('--root-model', type=Path,
                        default=Path('models/root_embeddings/best_model.pt'))
    parser.add_argument('--affix-model', type=Path,
                        default=Path('models/affix_embeddings/best_model.pt'))
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive exploration mode')
    parser.add_argument('--word', '-w', type=str,
                        help='Analyze a specific word')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('WORD1', 'WORD2'),
                        help='Compare two words')

    args = parser.parse_args()

    # Validate model paths
    if not args.root_model.exists():
        print(f"Error: Root model not found: {args.root_model}")
        print("Run: ./scripts/run_fundamento_training.sh")
        sys.exit(1)
    if not args.affix_model.exists():
        print(f"Error: Affix model not found: {args.affix_model}")
        print("Run: python scripts/training/train_affix_embeddings.py")
        sys.exit(1)

    # Initialize embedder
    embedder = CompositionalEmbedder(args.root_model, args.affix_model)

    if args.word:
        # Analyze single word
        print(f"\nAnalyzing '{args.word}':")
        emb, info = embedder.embed_word(args.word, verbose=True)
        if emb is not None:
            print(f"\nTop similar roots:")
            similar = embedder.find_similar_roots(args.word, top_k=15)
            for root, sim in similar:
                print(f"  {root}: {sim:.3f}")
        else:
            print(f"Error: {info.get('error', 'Unknown')}")

    elif args.compare:
        # Compare two words
        w1, w2 = args.compare
        sim, info1, info2 = embedder.similarity(w1, w2)
        print(f"\nComparing '{w1}' and '{w2}':")
        print(f"\n{w1}:")
        print(f"  Root: {info1['root']}")
        print(f"  Prefixes: {info1['prefixes']}")
        print(f"  Suffixes: {info1['suffixes']}")
        print(f"\n{w2}:")
        print(f"  Root: {info2['root']}")
        print(f"  Prefixes: {info2['prefixes']}")
        print(f"  Suffixes: {info2['suffixes']}")
        print(f"\nCosine Similarity: {sim:.3f}")

    elif args.interactive:
        interactive_mode(embedder)

    else:
        # Run full demo
        print("\n" + "="*60)
        print("Compositional Embeddings Demo")
        print("="*60)
        print("\nThis demo shows how root + affix embeddings combine")
        print("to represent word meaning compositionally.")

        demo_word_pairs(embedder)
        demo_morphological_composition(embedder)

        # Show affix effects on a common root
        embedder.demonstrate_affix_effect('bon', ['mal', 'ul', 'eg', 'et'])

        print("\n" + "="*60)
        print("Try interactive mode: python scripts/demo_compositional_embeddings.py -i")
        print("="*60)


if __name__ == '__main__':
    main()
