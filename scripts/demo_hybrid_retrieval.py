#!/usr/bin/env python3
"""
Demo Hybrid Retrieval - ReVo Synonyms + Embedding Associations

Demonstrates two-track query expansion for RAG:
1. Track 1: ReVo synonyms (deterministic, high precision)  
2. Track 2: Embedding associations (learned, high recall)

Usage:
    python scripts/demo_hybrid_retrieval.py                      # Example queries
    python scripts/demo_hybrid_retrieval.py -i                    # Interactive
    python scripts/demo_hybrid_retrieval.py "Kio estas Esperanto?"  
    python scripts/demo_hybrid_retrieval.py --revo-only "..."    # Only synonyms
    python scripts/demo_hybrid_retrieval.py --embeddings-only "..."  # Only associations
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.hybrid_query_expander import HybridQueryExpander


def extract_roots_from_ast(ast):
    """Extract all roots from AST (simplified)."""
    roots = set()
    
    def extract(node):
        if isinstance(node, dict):
            if 'radiko' in node:
                roots.add(node['radiko'])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    extract(value)
        elif isinstance(node, list):
            for item in node:
                extract(item)
    
    extract(ast)
    return roots


def run_query(query: str, expander: HybridQueryExpander):
    """Run query with hybrid expansion."""
    print(f"\nQuery: {query}")
    print("-" * 70)
    
    # Parse query
    ast = parse(query)
    
    # Extract roots
    original_roots = extract_roots_from_ast(ast)
    print(f"Original roots: {', '.join(sorted(original_roots))}")
    
    # Expand
    expansion = expander.expand(original_roots)
    
    print()
    print("Expansion Results:")
    if expansion['revo_synonyms']:
        print(f"  ReVo synonyms ({len(expansion['revo_synonyms'])}): {', '.join(sorted(expansion['revo_synonyms']))}")
    else:
        print(f"  ReVo synonyms: (none)")
    
    if expansion['embedding_associations']:
        emb_list = sorted(list(expansion['embedding_associations']))[:10]
        print(f"  Embedding assoc ({len(expansion['embedding_associations'])}): {', '.join(emb_list)}")
    else:
        print(f"  Embedding assoc: (none)")
    
    print()
    expansion_factor = len(expansion['all']) / len(original_roots) if original_roots else 1.0
    print(f"Total: {len(expansion['all'])} roots (×{expansion_factor:.1f} expansion)")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Demo hybrid query expansion',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('query', nargs='?', help='Esperanto query')
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
    parser.add_argument('--revo-only', action='store_true',
                       help='Use only ReVo synonyms (deterministic)')
    parser.add_argument('--embeddings-only', action='store_true',
                       help='Use only embeddings (learned)')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize expander
    print("="*70)
    print("HYBRID QUERY EXPANDER - ReVo + Embeddings")
    print("="*70)
    print()
    
    use_revo = not args.embeddings_only
    use_embeddings = not args.revo_only
    
    if args.revo_only:
        print("Mode: ReVo synonyms ONLY (deterministic)")
    elif args.embeddings_only:
        print("Mode: Embeddings ONLY (learned associations)")
    else:
        print("Mode: HYBRID (ReVo + Embeddings)")
    print()
    
    expander = HybridQueryExpander(
        embedding_path=args.embeddings,
        db_path=args.db,
        use_revo=use_revo,
        use_embeddings=use_embeddings
    )
    print()
    
    # Example queries
    example_queries = [
        "Kio estas Esperanto?",
        "Kiu fondis Esperanton?",
        "Kie loĝis Zamenhof?",
    ]
    
    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        print("="*70)
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    run_query(query, expander)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
    
    elif args.query:
        run_query(args.query, expander)
    
    else:
        # Run examples
        for query in example_queries:
            run_query(query, expander)


if __name__ == '__main__':
    main()
