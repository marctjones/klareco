#!/usr/bin/env python3
"""
Interactive demo for AST-aware retriever.

Uses KuzuInvertedIndex with Kuzu graph database for memory efficiency.

Usage:
    python scripts/demo_ast_retriever.py                         # Run example queries
    python scripts/demo_ast_retriever.py -i                      # Interactive mode
    python scripts/demo_ast_retriever.py "Kiu fondis Esperanton?"  # Single query
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.kuzu_inverted_index import FallbackMode


def compact_ast(ast: dict) -> dict:
    """Convert full AST to compact representation."""
    def extract_root_info(node):
        """Extract root and key info from a word/group node."""
        if node is None:
            return None
        if isinstance(node, dict):
            if node.get('tipo') == 'vortgrupo':
                kerno = node.get('kerno', {})
                root = kerno.get('radiko', '?')
                info = root
                if kerno.get('kazo') == 'akuzativo':
                    info += ' (akuz)'
                priskriboj = node.get('priskriboj', [])
                if priskriboj:
                    adj_roots = [p.get('radiko', '?') for p in priskriboj]
                    info += f" [{', '.join(adj_roots)}]"
                return info
            elif node.get('tipo') == 'vorto':
                root = node.get('radiko', '?')
                info = root
                if node.get('tempo'):
                    info += f" ({node['tempo'][:3]})"  # pas/prez/est
                if node.get('kazo') == 'akuzativo':
                    info += ' (akuz)'
                return info
        return str(node)

    compact = {
        'fraztipo': ast.get('fraztipo', '?'),
    }

    if ast.get('negita'):
        compact['negita'] = True

    compact['subjekto'] = extract_root_info(ast.get('subjekto'))
    compact['verbo'] = extract_root_info(ast.get('verbo'))
    compact['objekto'] = extract_root_info(ast.get('objekto'))

    # Compact aliaj - just roots
    aliaj = ast.get('aliaj', [])
    if aliaj:
        aliaj_roots = [a.get('radiko', '?') for a in aliaj if isinstance(a, dict)]
        compact['aliaj'] = aliaj_roots

    # Parse stats summary
    stats = ast.get('parse_statistics', {})
    if stats:
        compact['parse_rate'] = f"{stats.get('success_rate', 0):.0%}"

    return compact


def format_result(rank: int, score: float, doc: dict, show_ast: str = None, translator=None) -> str:
    """Format a single result for display.

    Args:
        show_ast: None (no AST), 'compact', or 'full'
        translator: Optional EsperantoTranslator instance for EN translation
    """
    import json
    from klareco.parser import parse

    text = doc.get('text', '')
    source = doc.get('source', {}).get('name', 'unknown')

    lines = [f"  {rank}. [{score:.3f}] {text}"]

    # Add English translation if translator provided
    if translator:
        try:
            translation = translator.translate(text)
            lines.append(f"     EN: {translation}")
        except Exception as e:
            lines.append(f"     EN: (translation error: {e})")

    lines.append(f"     Source: {source}")

    if show_ast:
        # Re-parse the text to get full AST (not stored in index)
        try:
            ast = parse(text)
            if show_ast == 'compact':
                compact = compact_ast(ast)
                lines.append(f"     AST: {json.dumps(compact, ensure_ascii=False)}")
            else:  # full
                lines.append(f"     AST: {json.dumps(ast, indent=6, ensure_ascii=False)}")
        except Exception as e:
            lines.append(f"     AST: (parse error: {e})")

    return '\n'.join(lines)


def run_query(retriever, query: str, top_k: int = 5, show_ast: str = None, translator=None):
    """Run a single query and display results.

    Args:
        show_ast: None, 'compact', or 'full'
        translator: Optional EsperantoTranslator for EN translations
    """
    print(f"\nQuery: {query}")
    print("-" * 60)

    start = time.time()
    try:
        results = retriever.search(query, top_k=top_k)
        elapsed = time.time() - start

        if not results:
            print("  No results found.")
            return

        print(f"  Found {len(results)} results in {elapsed:.2f}s\n")

        for i, (score, doc, stats) in enumerate(results[:top_k], 1):
            print(format_result(i, score, doc, show_ast, translator))
            print()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(retriever, show_ast: str = None, translator=None):
    """Run interactive query loop.

    Args:
        show_ast: None, 'compact', or 'full'
        translator: Optional EsperantoTranslator for EN translations
    """
    print("\n" + "=" * 60)
    print("AST-Aware Retriever - Interactive Mode")
    print("=" * 60)
    print("Enter queries in Esperanto. Type 'quit' to exit.")
    print("Commands: :ast (cycle: off → compact → full), :help")
    print()

    ast_modes = [None, 'compact', 'full']
    ast_mode_idx = ast_modes.index(show_ast) if show_ast in ast_modes else 0

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
            ast_mode_idx = (ast_mode_idx + 1) % len(ast_modes)
            show_ast = ast_modes[ast_mode_idx]
            mode_name = show_ast if show_ast else 'off'
            print(f"AST display: {mode_name}")
            continue

        if query == ':help':
            print("Commands:")
            print("  :ast   - Cycle AST display (off → compact → full)")
            print("  :q     - Quit")
            print("  quit   - Quit")
            continue

        run_query(retriever, query, show_ast=show_ast, translator=translator)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive AST-aware retriever demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_ast_retriever.py -i                     # Interactive mode
  python scripts/demo_ast_retriever.py "Kiu fondis Esperanton?"
  python scripts/demo_ast_retriever.py --ast compact "Kiu fondis Esperanton?"
  python scripts/demo_ast_retriever.py --ast full "Kiu fondis Esperanton?"
  python scripts/demo_ast_retriever.py --fallback embedding   # Use embedding fallback
        """
    )
    parser.add_argument('query', nargs='?', help='Query to search (or use -i for interactive)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--index', type=str, default='data/indexes/kuzu_index',
                        help='Index directory (default: data/indexes/kuzu_index)')
    parser.add_argument('--fallback', type=str, choices=['none', 'embedding', 'rerank', 'full'],
                        default='none', help='Fallback mode (default: none = pure deterministic)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--ast', type=str, nargs='?', const='compact', choices=['compact', 'full'],
                        help='Show AST: compact (default) or full JSON')
    parser.add_argument('--translate', action='store_true',
                        help='Add English translations to results (requires transformers)')

    args = parser.parse_args()

    index_path = Path(args.index)

    # Check index exists
    if not (index_path / "kuzu.db").exists():
        print(f"Error: Kuzu index not found at {index_path}/kuzu.db")
        print("Run: ./scripts/build_kuzu_index.sh")
        sys.exit(1)

    # Initialize translator if requested
    translator = None
    if args.translate:
        print("Loading Esperanto→English translator...")
        try:
            # Import translator inline to avoid dependency if not used
            sys.path.insert(0, str(Path(__file__).parent))
            from translate_eo_inline import EsperantoTranslator
            translator = EsperantoTranslator()
            print("  Translator ready!")
        except ImportError as e:
            print(f"  Error: {e}")
            print("  Install with: pip install transformers sentencepiece")
            sys.exit(1)
        except Exception as e:
            print(f"  Error loading translator: {e}")
            sys.exit(1)

    # Parse fallback mode
    fallback_modes = {
        'none': FallbackMode.NONE,
        'embedding': FallbackMode.EMBEDDING,
        'rerank': FallbackMode.RERANK,
        'full': FallbackMode.FULL,
    }
    fallback_mode = fallback_modes[args.fallback]

    # Load retriever
    print(f"Loading AST-aware retriever from {index_path}...")
    print(f"  Fallback mode: {args.fallback}")

    try:
        retriever = ASTAwareRetriever(
            index_path=index_path,
            fallback_mode=fallback_mode,
        )
    except Exception as e:
        print(f"Error loading retriever: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Ready!\n")

    # Interactive or single query
    if args.interactive:
        interactive_mode(retriever, show_ast=args.ast, translator=translator)
    elif args.query:
        run_query(retriever, args.query, top_k=args.top_k, show_ast=args.ast, translator=translator)
    else:
        # Default: run example queries
        example_queries = [
            "Kiu fondis Esperanton?",
            "Kio estas Esperanto?",
            "Kie naskiĝis Zamenhof?",
        ]
        print("Running example queries (use -i for interactive mode):\n")
        for query in example_queries:
            run_query(retriever, query, top_k=3, show_ast=args.ast, translator=translator)


if __name__ == '__main__':
    main()
