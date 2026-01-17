#!/usr/bin/env python3
"""
Demo AST retriever with semantic query expansion.

Enhances retrieval recall by expanding query roots with semantic neighbors.

Usage:
    python scripts/demo_ast_retriever_enhanced.py                    # Example queries
    python scripts/demo_ast_retriever_enhanced.py -i                  # Interactive
    python scripts/demo_ast_retriever.py "Kio estas Esperanto?"      # Single query
    python scripts/demo_ast_retriever.py --no-expansion "Kio..."     # Baseline
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.kuzu_inverted_index import FallbackMode
from klareco.rag.query_expansion import SemanticQueryExpander, expand_ast_roots
from klareco.parser import parse


def run_query_with_expansion(retriever, query: str, expander=None, top_k: int = 5):
    """Run query with optional semantic expansion."""
    from klareco.parser import parse

    print(f"\nQuery: {query}")
    print("-" * 60)

    # Parse query
    ast = parse(query)

    # Expand if expander provided
    if expander:
        print("\n[Semantic Expansion Enabled]")
        from klareco.rag.query_expansion import extract_all_roots_from_ast, expand_ast_roots

        # Show original roots
        original_roots = extract_all_roots_from_ast(ast, include_expansions=False)
        print(f"Original roots: {', '.join(sorted(original_roots))}")

        # Expand
        expanded_ast = expand_ast_roots(ast, expander)
        expanded_roots = extract_all_roots_from_ast(expanded_ast, include_expansions=True)

        print(f"Expanded roots: {', '.join(sorted(expanded_roots - set(extract_all_roots_from_ast(ast))))}")
        print()

    # Continue with normal retrieval using expanded AST
    query_ast = expanded_ast if expander else ast

    # Retrieve
    start = time.time()
    results = retriever.retrieve(query_ast, top_k=top_k)
    elapsed = time.time() - start

    # Display results
    print(f"\n🔎 Found {len(results)} results in {elapsed:.3f}s\n")

    for i, (score, doc_id, document) in enumerate(results, 1):
        text = document.get('text', 'NO TEXT')[:120]
        source = document.get('source', 'unknown')
        print(f"{i}. [score={score:.3f}] [{source}]")
        print(f"   {text}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Demo AST retriever with semantic query expansion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_semantic_retrieval.py                    # Run examples
  python scripts/demo_semantic_retrieval.py -i                 # Interactive
  python scripts/demo_semantic_retrieval.py "Kio estas Esperanto?"
  python scripts/demo_semantic_retrieval.py --no-expansion "Kio estas Esperanto?"
        """
    )
    parser.add_argument('query', nargs='?', help='Esperanto query')
    parser.add_argument('--index-dir', type=Path,
                       default=Path('data/indexes/kuzu_index'),
                       help='Path to Kuzu index directory')
    parser.add_argument('--embeddings', type=Path,
                       default=Path('models/root_embeddings/best_model.pt'),
                       help='Path to root embeddings')
    parser.add_argument('--no-expansion', action='store_true',
                       help='Disable semantic expansion (baseline)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--expansion-k', type=int, default=5,
                       help='Number of similar roots per query root')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Minimum similarity threshold for expansion')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    # Initialize retriever
    print(f"Loading retriever from {args.index_dir}...")
    retriever = ASTAwareRetriever(
        index_dir=args.index_dir,
        fallback_mode=FallbackMode.ROOT
    )
    print(f"Loaded {retriever.index.num_documents} documents\n")

    # Initialize expander if not disabled
    expander = None
    if not args.no_expansion:
        print(f"Loading semantic expander from {args.embeddings}...")
        expander = SemanticQueryExpander(
            args.embeddings,
            top_k=args.expansion_k,
            threshold=args.threshold
        )
        print(f"Loaded {len(expander.root_to_idx)} root embeddings")
        print(f"Expansion: top-k={args.expansion_k}, threshold={args.threshold}\n")
    else:
        print("⚠️  Semantic expansion DISABLED (baseline mode)\n")

    # Example queries
    example_queries = [
        "Kio estas Esperanto?",
        "Kiu fondis Esperanton?",
        "La hundo kuras.",
    ]

    if args.interactive:
        print("="*70)
        print("Interactive mode (type 'quit' to exit)")
        print("="*70)
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    run_query_with_expansion(retriever, query, expander, args.top_k)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    elif args.query:
        run_query_with_expansion(retriever, args.query, expander, args.top_k)

    else:
        # Run examples
        print("="*70)
        print("Example Queries")
        print("="*70)
        for query in example_queries:
            run_query_with_expansion(retriever, query, expander, args.top_k)
            print()


if __name__ == '__main__':
    main()
