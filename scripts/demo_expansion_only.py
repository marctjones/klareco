#!/usr/bin/env python3
"""
Demo Hybrid Expansion Only - Show Query Expansion Without Database

Demonstrates hybrid query expansion (ReVo + Embeddings) without requiring
a Kuzu database connection. Useful for testing expansion logic independently.

Usage:
    python scripts/demo_expansion_only.py "Kio estas Esperanto?"
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def extract_roots_from_ast(ast):
    """Extract all content word roots from AST."""
    roots = set()
    skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

    def extract(node):
        if not node or not isinstance(node, dict):
            return
        if node.get('tipo') == 'vorto':
            vortspeco = node.get('vortspeco', '')
            if vortspeco not in skip_vortspeco:
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    roots.add(root.lower())
        elif node.get('tipo') == 'vortgrupo':
            extract(node.get('kerno'))
            for p in node.get('priskriboj', []):
                extract(p)
        elif node.get('tipo') == 'frazo':
            extract(node.get('subjekto'))
            extract(node.get('verbo'))
            extract(node.get('objekto'))
            for a in node.get('aliaj', []):
                extract(a)

    extract(ast)
    return roots


def main():
    parser = argparse.ArgumentParser(description='Demo hybrid expansion without database')
    parser.add_argument('query', help='Esperanto query')
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'),
        help='Path to embeddings'
    )
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of embedding neighbors')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Min similarity threshold')

    args = parser.parse_args()

    print("="*70)
    print("HYBRID QUERY EXPANSION DEMO (Embeddings Only)")
    print("="*70)
    print()

    # Parse query
    print(f"Query: {args.query}")
    print("-"*70)
    ast = parse(args.query)
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")
    print()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    checkpoint = torch.load(args.embeddings, map_location='cpu', weights_only=False)
    embeddings = checkpoint['embeddings']
    vocab = checkpoint['vocab']

    # Create reverse vocab (root -> index)
    root_to_idx = {root: idx for idx, root in enumerate(vocab)}
    print(f"Loaded {len(vocab)} roots with {embeddings.shape[1]}D embeddings")
    print()

    # Expand each root
    print("Expansion Results:")
    print("-"*70)
    all_expanded = set(original_roots)

    for root in sorted(original_roots):
        # Check if root exists in vocab
        if root not in root_to_idx:
            print(f"\n'{root}' → (not in vocabulary)")
            continue

        # Get embedding for this root
        root_idx = root_to_idx[root]
        root_emb = embeddings[root_idx].unsqueeze(0)

        # Compute similarities with all roots
        similarities = F.cosine_similarity(root_emb, embeddings)

        # Get top-k neighbors (excluding self)
        top_k_sims, top_k_indices = torch.topk(similarities, k=args.top_k + 1)

        # Filter by threshold and exclude self
        neighbors = []
        for sim, idx in zip(top_k_sims.tolist(), top_k_indices.tolist()):
            if idx == root_idx:  # Skip self
                continue
            if sim >= args.threshold:
                neighbor_root = vocab[idx]
                neighbors.append((neighbor_root, sim))

        if neighbors:
            expanded_roots = [n[0] for n in neighbors]
            all_expanded.update(expanded_roots)

            print(f"\n'{root}' →")
            for neighbor_root, similarity in neighbors:
                print(f"  {neighbor_root:15s} (similarity: {similarity:.3f})")
        else:
            print(f"\n'{root}' → (no expansions above threshold {args.threshold})")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original query roots: {len(original_roots)}")
    print(f"Total expanded roots: {len(all_expanded)}")
    print(f"Expansion factor: {len(all_expanded) / len(original_roots):.2f}x")
    print()
    print(f"Full expanded set ({len(all_expanded)} roots):")
    print(f"  {', '.join(sorted(all_expanded))}")
    print()
    print("="*70)
    print()
    print("NOTE: This shows embedding-based expansion only.")
    print("In full hybrid system, ReVo synonyms would also be added")
    print("(deterministic dictionary-based expansion).")
    print()


if __name__ == '__main__':
    main()
