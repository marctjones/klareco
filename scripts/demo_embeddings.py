#!/usr/bin/env python3
"""
Demo: Compositional Embeddings (Root + Affix V2)

This script demonstrates how the trained Stage 1 models work together:
- Root embeddings: Semantic meaning of base words (64d, ~11K roots)
- Affix transforms V2: Low-rank transformations for prefixes/suffixes

Key insight: Word meaning = suffix_transform(prefix_transform(root_embedding))
Affixes are TRANSFORMS (not additive vectors) that modify the embedding space.

Examples:
    python scripts/demo_compositional_embeddings.py
    python scripts/demo_compositional_embeddings.py --interactive
    python scripts/demo_compositional_embeddings.py --word malbonulo
    python scripts/demo_compositional_embeddings.py --compare bona malbona
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


class LowRankTransform(nn.Module):
    """Low-rank transformation for affixes (V2 architecture)."""
    def __init__(self, dim: int = 64, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class CompositionalEmbedder:
    """Combine root embeddings with V2 affix transforms for word representation."""

    def __init__(
        self,
        root_model_path: Path = Path("models/root_embeddings/best_model.pt"),
        affix_model_path: Path = Path("models/affix_transforms_v2/best_model.pt"),
    ):
        self.root_model_path = root_model_path
        self.affix_model_path = affix_model_path

        # Load root embeddings
        print(f"Loading root embeddings from {root_model_path}...")
        root_ckpt = torch.load(root_model_path, map_location='cpu', weights_only=False)
        self.root_emb = root_ckpt['model_state_dict']['embeddings.weight'].numpy()
        self.root_to_idx = root_ckpt['root_to_idx']
        self.idx_to_root = {v: k for k, v in self.root_to_idx.items()}
        self.root_dim = self.root_emb.shape[1]
        print(f"  Loaded {len(self.root_to_idx)} roots, {self.root_dim}d")

        # Precompute average root embedding for fallbacks
        self.avg_root_embedding = torch.from_numpy(self.root_emb.mean(axis=0)).float()

        # Load V2 affix transforms (low-rank)
        print(f"Loading affix transforms (V2) from {affix_model_path}...")
        affix_ckpt = torch.load(affix_model_path, map_location='cpu', weights_only=False)

        self.prefix_transforms = {}
        self.suffix_transforms = {}
        self.embedding_dim = affix_ckpt.get('embedding_dim', 64)
        rank = affix_ckpt.get('rank', 8)

        # Load prefix transforms
        for p in affix_ckpt['prefixes']:
            t = LowRankTransform(self.embedding_dim, rank)
            t.down.weight.data = affix_ckpt['model_state_dict'][f'prefix_transforms.{p}.down.weight']
            t.up.weight.data = affix_ckpt['model_state_dict'][f'prefix_transforms.{p}.up.weight']
            self.prefix_transforms[p] = t

        # Load suffix transforms
        for s in affix_ckpt['suffixes']:
            t = LowRankTransform(self.embedding_dim, rank)
            t.down.weight.data = affix_ckpt['model_state_dict'][f'suffix_transforms.{s}.down.weight']
            t.up.weight.data = affix_ckpt['model_state_dict'][f'suffix_transforms.{s}.up.weight']
            self.suffix_transforms[s] = t

        print(f"  Loaded {len(self.prefix_transforms)} prefix transforms, {len(self.suffix_transforms)} suffix transforms")
        print(f"  Architecture: low-rank (rank={rank})")

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

    def get_root_embedding_tensor(self, root: str) -> Optional[torch.Tensor]:
        """Get embedding for a root as a tensor (for transform application)."""
        emb = self.get_root_embedding(root)
        if emb is not None:
            return torch.from_numpy(emb).float()
        return None

    def _is_proper_noun(self, root: str) -> bool:
        """Detect if a root is likely a proper noun (capitalized)."""
        return bool(root and root[0].isupper())

    def _char_hash_embedding(self, root: str) -> torch.Tensor:
        """
        Create embedding for unknown root using character trigram hashing.
        Similar-looking unknown words get similar embeddings.
        """
        root_lower = root.lower()
        padded = f"^{root_lower}$"

        # Hash character trigrams to embedding positions
        emb = torch.zeros(self.embedding_dim)
        for i in range(len(padded) - 2):
            trigram = padded[i:i+3]
            h = hash(trigram) % self.embedding_dim
            emb[h] += 1.0

        # Normalize
        norm = torch.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Blend: 70% hash + 30% average root (keeps in similar embedding space)
        emb = 0.7 * emb + 0.3 * self.avg_root_embedding
        return emb

    def embed_word(self, word: str, verbose: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Create compositional embedding for a word using V2 low-rank transforms.

        Pipeline: root → prefix_transforms → suffix_transforms → normalize

        Fallback strategies for unknown roots:
        1. Proper noun (capitalized): Character trigram hash
        2. Has known affixes: Average root + affix transforms
        3. Otherwise: Character trigram hash

        Returns (embedding, info) where info contains parse details.
        """
        info = {
            'word': word,
            'root': None,
            'prefixes': [],
            'suffixes': [],
            'root_found': False,
            'fallback': None,
            'prefix_transforms_applied': [],
            'suffix_transforms_applied': [],
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

        # Get root embedding as tensor
        emb = self.get_root_embedding_tensor(root)
        if emb is None:
            # Unknown root - apply fallback strategies
            has_known_affixes = (
                any(p in self.prefix_transforms for p in prefixes if p) or
                any(s in self.suffix_transforms for s in suffixes if s)
            )

            if self._is_proper_noun(root):
                # Strategy 1: Proper noun - character hash embedding
                emb = self._char_hash_embedding(root)
                info['fallback'] = 'proper_noun'
                if verbose:
                    print(f"  Root '{root}': proper noun fallback (char hash)")
            elif has_known_affixes:
                # Strategy 2: Unknown root but known affixes - use average root
                emb = self.avg_root_embedding.clone()
                info['fallback'] = 'morpheme_only'
                if verbose:
                    print(f"  Root '{root}': unknown, using average root + affixes")
            else:
                # Strategy 3: Unknown root, no known affixes - character hash
                emb = self._char_hash_embedding(root)
                info['fallback'] = 'char_hash'
                if verbose:
                    print(f"  Root '{root}': unknown, using char hash embedding")
        else:
            info['root_found'] = True
            if verbose:
                print(f"  Root '{root}': {self.root_dim}d embedding")

        # Apply prefix transforms (in order)
        with torch.no_grad():
            for p in prefixes:
                if p in self.prefix_transforms:
                    emb = self.prefix_transforms[p](emb)
                    info['prefix_transforms_applied'].append(p)
                    if verbose:
                        print(f"  Prefix '{p}': applied low-rank transform")
                elif verbose:
                    print(f"  Prefix '{p}': no transform available")

            # Apply suffix transforms (in order)
            for s in suffixes:
                if s in self.suffix_transforms:
                    emb = self.suffix_transforms[s](emb)
                    info['suffix_transforms_applied'].append(s)
                    if verbose:
                        print(f"  Suffix '{s}': applied low-rank transform")
                elif verbose:
                    print(f"  Suffix '{s}': no transform available")

        # Convert back to numpy and normalize
        emb_np = emb.numpy()
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        return emb_np, info

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
        """Show how affixes transform a root's meaning using V2 low-rank transforms."""
        print(f"\n{'='*60}")
        print(f"Affix Effects on Root: {root} (V2 Transforms)")
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
            # Check if we have a transform for this affix
            if affix in self.prefix_transforms:
                word = affix + root + 'o'  # Assume noun
                affix_type = 'prefix'
            elif affix in self.suffix_transforms:
                word = root + affix + 'o'
                affix_type = 'suffix'
            else:
                print(f"\n  Unknown/untrained affix: {affix}")
                continue

            print(f"\nWith {affix_type} '{affix}' → {word}:")
            emb, info = self.embed_word(word)
            if emb is not None:
                # Compute distance from original root
                dist = 1 - np.dot(base_norm, emb)
                print(f"  Distance from '{root}': {dist:.3f}")

                # Find similar roots
                sims = np.dot(self.root_emb_norm, emb)
                top_indices = np.argsort(sims)[::-1][:5]
                print(f"  Most similar roots:")
                for idx in top_indices:
                    r = self.idx_to_root[idx]
                    marker = "←" if r == root else ""
                    print(f"    {r}: {sims[idx]:.3f} {marker}")


def demo_word_pairs(embedder: CompositionalEmbedder):
    """Demonstrate similarity between word pairs."""
    print("\n" + "="*60)
    print("Word Pair Similarities (V2 Low-Rank Transforms)")
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

        # For mal- pairs, show distance instead of similarity (clearer for antonyms)
        if 'mal' in info1.get('prefixes', []) or 'mal' in info2.get('prefixes', []):
            distance = 1 - sim
            print(f"  {w1} ({c1}) vs {w2} ({c2}): sim={sim:.3f} (distance={distance:.3f}){status}")
        else:
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


def get_model_info(model_path: Path) -> dict:
    """Get model file info including modification time and version detection."""
    info = {
        'path': str(model_path),
        'name': model_path.name,
        'exists': model_path.exists(),
        'modified': None,
        'version': None,
        'size_kb': None,
    }
    if model_path.exists():
        stat = model_path.stat()
        info['modified'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        info['size_kb'] = stat.st_size // 1024
        if '_v2' in str(model_path) or 'v2' in model_path.parent.name:
            info['version'] = 'V2'
        elif '_v1' in str(model_path) or 'v1' in model_path.parent.name:
            info['version'] = 'V1'
        else:
            info['version'] = 'unknown'
    return info


def print_session_header(embedder: CompositionalEmbedder):
    """Print session header with timestamp and model info."""
    print("=" * 60)
    print(" Compositional Embeddings Demo (V2)")
    print("=" * 60)
    print(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Root model info
    root_info = get_model_info(embedder.root_model_path)
    print("Root Embeddings Model:")
    print(f"  Path: {root_info['path']}")
    print(f"  Modified: {root_info['modified']}")
    print(f"  Roots: {len(embedder.root_to_idx):,}")
    print(f"  Dimensions: {embedder.root_dim}d")

    # Affix model info
    affix_info = get_model_info(embedder.affix_model_path)
    print()
    print("Affix Transforms Model:")
    print(f"  Path: {affix_info['path']}")
    print(f"  Version: {affix_info['version']}")
    print(f"  Modified: {affix_info['modified']}")
    print(f"  Prefixes: {len(embedder.prefix_transforms)}")
    print(f"  Suffixes: {len(embedder.suffix_transforms)}")
    print(f"  Architecture: Low-rank transforms")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Demo: Compositional Embeddings (Root + Affix V2)',
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
                        default=Path('models/affix_transforms_v2/best_model.pt'),
                        help='V2 affix transforms model (low-rank)')
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
        print(f"Error: Affix transforms model not found: {args.affix_model}")
        print("Run: python scripts/training/train_affix_transforms_v2.py")
        sys.exit(1)

    # Initialize embedder
    embedder = CompositionalEmbedder(args.root_model, args.affix_model)

    # Print session header
    print()
    print_session_header(embedder)

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
        print("\nThis demo shows how root embeddings + V2 affix transforms")
        print("compose to represent word meaning.")
        print()
        print("Key insight: Affixes are TRANSFORMS (not additive vectors)")
        print("  mal- flips polarity: bon -> malbon (distinct)")
        print("  re-  preserves meaning: fari -> refari (similar)")
        print()

        demo_word_pairs(embedder)
        demo_morphological_composition(embedder)

        # Show affix effects on a common root
        embedder.demonstrate_affix_effect('bon', ['mal', 'ul', 'eg', 'et'])

        print("\n" + "="*60)
        print("Try interactive mode: python scripts/demo_compositional_embeddings.py -i")
        print("="*60)


if __name__ == '__main__':
    main()
