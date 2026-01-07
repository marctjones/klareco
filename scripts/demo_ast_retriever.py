#!/usr/bin/env python3
"""
Interactive demo for AST-aware retriever.

Usage:
    python scripts/demo_ast_retriever.py                    # With prefilter (needs HNSW)
    python scripts/demo_ast_retriever.py --no-prefilter     # Without prefilter (slower but works)
    python scripts/demo_ast_retriever.py -i                 # Interactive mode
    python scripts/demo_ast_retriever.py "Kiu fondis Esperanton?"  # Single query
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever


def format_result(rank: int, score: float, doc: dict, show_ast: bool = False) -> str:
    """Format a single result for display."""
    text = doc.get('text', '')[:120]
    source = doc.get('source', {}).get('name', 'unknown')

    lines = [f"  {rank}. [{score:.3f}] {text}..."]
    lines.append(f"     Source: {source}")

    if show_ast and 'slots' in doc:
        slots = doc['slots']
        slot_info = []
        for slot_name in ['SUBJ', 'VERB', 'OBJ']:
            if slot_name in slots and slots[slot_name]:
                slot_info.append(slot_name)
        if slot_info:
            lines.append(f"     Slots: {', '.join(slot_info)}")

    return '\n'.join(lines)


def run_query(retriever, query: str, top_k: int = 5, show_ast: bool = False):
    """Run a single query and display results."""
    print(f"\nQuery: {query}")
    print("-" * 60)

    start = time.time()
    try:
        results = retriever.search(query, top_k=top_k, strategy='auto')
        elapsed = time.time() - start

        if not results:
            print("  No results found.")
            return

        print(f"  Found {len(results)} results in {elapsed:.2f}s\n")

        for i, (score, doc) in enumerate(results[:top_k], 1):
            print(format_result(i, score, doc, show_ast))
            print()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(retriever, show_ast: bool = False):
    """Run interactive query loop."""
    print("\n" + "=" * 60)
    print("AST-Aware Retriever - Interactive Mode")
    print("=" * 60)
    print("Enter queries in Esperanto. Type 'quit' to exit.")
    print("Commands: :ast (toggle AST display), :help")
    print()

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', ':q'):
            print("Goodbye!")
            break

        if query == ':ast':
            show_ast = not show_ast
            print(f"AST display: {'ON' if show_ast else 'OFF'}")
            continue

        if query == ':help':
            print("Commands:")
            print("  :ast   - Toggle AST slot display")
            print("  :q     - Quit")
            print("  quit   - Quit")
            continue

        run_query(retriever, query, show_ast=show_ast)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive AST-aware retriever demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_ast_retriever.py -i                     # Interactive mode
  python scripts/demo_ast_retriever.py "Kiu fondis Esperanton?"
  python scripts/demo_ast_retriever.py --no-prefilter -i      # Without HNSW (slower)
  python scripts/demo_ast_retriever.py --compare              # Compare with/without prefilter
        """
    )
    parser.add_argument('query', nargs='?', help='Query to search (or use -i for interactive)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--index', type=str, default='data/indexes/slot_hybrid',
                        help='Index directory (default: slot_hybrid)')
    parser.add_argument('--no-prefilter', action='store_true',
                        help='Disable HNSW/embedding prefiltering (slower but works without HNSW)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--ast', action='store_true', help='Show AST slot info')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with and without prefilter')

    args = parser.parse_args()

    index_path = Path(args.index)

    # Check index exists
    if not (index_path / "slot_index.jsonl").exists():
        print(f"Error: Index not found at {index_path}")
        print("Run: python scripts/index_slot_based.py --help")
        sys.exit(1)

    # Compare mode
    if args.compare:
        test_queries = [
            "Kiu fondis Esperanton?",
            "Kio estas la ĉefurbo de Francio?",
            "Kiam naskiĝis Zamenhof?",
        ]

        print("=" * 60)
        print("Comparing AST-Aware Retriever: Prefilter vs No Prefilter")
        print("=" * 60)

        # With prefilter
        print("\n[1] Loading WITH prefilter (HNSW)...")
        try:
            retriever_pf = ASTAwareRetriever(index_path, use_prefilter=True)
            has_prefilter = retriever_pf.prefilter_retriever is not None
            print(f"    Prefilter loaded: {has_prefilter}")
        except Exception as e:
            print(f"    Failed: {e}")
            has_prefilter = False
            retriever_pf = None

        # Without prefilter
        print("\n[2] Loading WITHOUT prefilter...")
        retriever_no_pf = ASTAwareRetriever(index_path, use_prefilter=False)
        print("    Loaded (pure AST matching)")

        # Run comparisons
        for query in test_queries:
            print(f"\n{'=' * 60}")
            print(f"Query: {query}")
            print("=" * 60)

            if retriever_pf and has_prefilter:
                print("\n[With Prefilter]")
                start = time.time()
                results = retriever_pf.search(query, top_k=3)
                elapsed = time.time() - start
                print(f"  Time: {elapsed:.2f}s")
                if results:
                    print(f"  Top: {results[0][1]['text'][:80]}...")

            print("\n[Without Prefilter]")
            start = time.time()
            results = retriever_no_pf.search(query, top_k=3)
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.2f}s")
            if results:
                print(f"  Top: {results[0][1]['text'][:80]}...")

        return

    # Normal mode - load retriever
    print(f"Loading AST-aware retriever from {index_path}...")
    use_prefilter = not args.no_prefilter

    try:
        retriever = ASTAwareRetriever(index_path, use_prefilter=use_prefilter)
        if use_prefilter and retriever.prefilter_retriever is None:
            print("  Warning: Prefilter requested but HNSW not available")
            print("  Falling back to pure AST matching (slower)")
    except Exception as e:
        print(f"Error loading retriever: {e}")
        sys.exit(1)

    print("Ready!\n")

    # Interactive or single query
    if args.interactive:
        interactive_mode(retriever, show_ast=args.ast)
    elif args.query:
        run_query(retriever, args.query, top_k=args.top_k, show_ast=args.ast)
    else:
        # Default: run example queries
        example_queries = [
            "Kiu fondis Esperanton?",
            "Kio estas Esperanto?",
            "Kie naskiĝis Zamenhof?",
        ]
        print("Running example queries (use -i for interactive mode):\n")
        for query in example_queries:
            run_query(retriever, query, top_k=3, show_ast=args.ast)


if __name__ == '__main__':
    main()
