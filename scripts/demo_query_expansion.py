#!/usr/bin/env python3
"""
Demo semantic query expansion.

Shows how expanding query roots with semantic neighbors improves retrieval.

Usage:
    python scripts/demo_query_expansion.py "Kio estas Esperanto?"
    python scripts/demo_query_expansion.py -i  # Interactive mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.query_expansion import SemanticQueryExpander, extract_all_roots_from_ast
from klareco.parser import parse


def demo_expansion(query: str, expander: SemanticQueryExpander):
    """Demonstrate query expansion for a single query."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    # Parse query
    try:
        ast = parse(query)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return

    # Extract original roots
    from klareco.rag.query_expansion import extract_all_roots_from_ast
    original_roots = extract_all_roots_from_ast(ast, include_expansions=False)

    print(f"\n📝 Original roots ({len(original_roots)}):")
    for root in sorted(original_roots):
        print(f"   • {root}")

    # Expand each root
    print(f"\n🔍 Semantic expansion:")
    expansion = expander.expand_roots(list(original_roots))

    expanded_roots = set(original_roots)
    for root, similar_list in expansion.items():
        if similar_list:
            print(f"\n   {root} →")
            for similar_root, similarity in similar_list:
                print(f"      • {similar_root:15s} ({similarity:.3f})")
                expanded_roots.add(similar_root)

    print(f"\n✨ Total expanded roots: {len(expanded_roots)} (added {len(expanded_roots) - len(original_roots)})")
    print(f"\n   All roots: {', '.join(sorted(expanded_roots))}")


def main():
    parser = argparse.ArgumentParser(
        description='Demo semantic query expansion'
    )
    parser.add_argument('query', nargs='?', help='Esperanto query')
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to root embeddings')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of similar roots per query root')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Minimum similarity threshold')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    # Load expander
    print(f"Loading embeddings from {args.embeddings}...")
    expander = SemanticQueryExpander(
        args.embeddings,
        top_k=args.top_k,
        threshold=args.threshold
    )
    print(f"Loaded {len(expander.root_to_idx)} root embeddings")
    print(f"Expansion: top-k={args.top_k}, threshold={args.threshold}")

    # Example queries
    example_queries = [
        "Kio estas Esperanto?",
        "La hundo kuras.",
        "Mi pensas pri tio.",
        "La bela tago.",
    ]

    if args.interactive:
        print("\n" + "="*70)
        print("Interactive mode (type 'quit' to exit)")
        print("="*70)
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    demo_expansion(query, expander)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    elif args.query:
        demo_expansion(args.query, expander)

    else:
        # Run examples
        print("\n" + "="*70)
        print("Example Queries")
        print("="*70)
        for query in example_queries:
            demo_expansion(query, expander)
            print()


if __name__ == '__main__':
    main()
