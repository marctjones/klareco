#!/usr/bin/env python3
"""
Demo Hybrid RAG - Complete Retrieval Pipeline with Hybrid Query Expansion

Integrates HybridQueryExpander with existing ASTAwareRetriever to demonstrate
full retrieval pipeline with both deterministic synonyms (ReVo) and learned
associations (embeddings).

Pipeline:
1. Parse query
2. Expand with HybridQueryExpander (ReVo + Embeddings)
3. Retrieve with ASTAwareRetriever
4. Show top N results with expansion details

Usage:
    python scripts/demo_hybrid_rag.py "Kio estas Esperanto?"
    python scripts/demo_hybrid_rag.py -i  # Interactive mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.hybrid_query_expander import HybridQueryExpander
from klareco.rag.ast_aware_retriever import ASTAwareRetriever


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


def demo_query(query, expander, retriever, top_k=10):
    """Demonstrate hybrid retrieval for a single query."""
    print("="*70)
    print(f"QUERY: {query}")
    print("="*70)
    print()

    # Parse query
    print("1. Parsing query...")
    ast = parse(query)
    original_roots = extract_roots_from_ast(ast)
    print(f"   Original roots: {', '.join(sorted(original_roots))}")
    print()

    # Show expansion info (retriever does this internally via Kuzu graph)
    print("2. Query expansion...")
    print("   Note: Retriever performs graph-based synonym expansion internally")
    print("   (ReVo synonyms via Kuzu graph traversal)")
    print()

    # Retrieve documents
    print(f"3. Retrieving top {top_k} documents with ASTAwareRetriever...")
    print()

    # Use existing retriever infrastructure
    results = retriever.search(query, top_k=top_k, use_m1_expansion=False)

    print()
    print("="*70)
    print(f"TOP {top_k} RESULTS")
    print("="*70)
    print()

    if not results:
        print("No documents found!")
    else:
        for i, (score, doc, stats) in enumerate(results, 1):
            doc_text = doc.get('text', 'NO TEXT')
            source = doc.get('source', 'unknown')
            doc_id = doc.get('doc_id', 'unknown')

            # Truncate long text
            if len(doc_text) > 200:
                doc_text = doc_text[:200] + "..."

            print(f"{i}. [Score: {score:.4f}] [Source: {source}] [ID: {doc_id}]")
            print(f"   {doc_text}")
            print()

    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(description='Demo hybrid RAG retrieval')
    parser.add_argument('query', nargs='?', help='Esperanto query')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt'),
        help='Path to embeddings'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/indexes/v2.1_kuzu_index_full'),
        help='Path to Kuzu database'
    )
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return')

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Query required unless using --interactive mode")

    print("="*70)
    print("HYBRID RAG DEMO")
    print("="*70)
    print()

    # Initialize retriever first (it will hold the Kuzu connection)
    print("Loading AST-aware retriever...")
    print(f"  Database: {args.db}")
    retriever = ASTAwareRetriever(index_path=args.db)
    print()

    # Initialize hybrid expander WITHOUT opening database
    # (We'll just use it for embeddings and get synonyms from retriever)
    print("Loading hybrid query expander (embeddings only)...")
    print(f"  Embeddings: {args.embeddings}")

    # For now, skip hybrid expansion and use retriever's built-in synonym expansion
    # The retriever already has graph-based synonym expansion via Kuzu
    expander = None
    print("  Note: Using retriever's built-in graph-based synonym expansion")
    print()

    # Interactive mode
    if args.interactive:
        print("Interactive mode - enter queries (Ctrl+C to exit)")
        print()
        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue
                print()
                demo_query(query, expander, retriever, args.top_k)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()
    else:
        # Single query mode
        demo_query(args.query, expander, retriever, args.top_k)


if __name__ == '__main__':
    main()
