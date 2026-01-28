#!/usr/bin/env python3
"""
Demo: M1 Query Expansion with Plausibility Filtering

This demonstrates M1's INTENDED PURPOSE: filtering synonym expansions
BEFORE search to avoid retrieving nonsense documents.

Example:
    Query: "Kiu manĝas insektojn?" (Who eats insects?)
    Synonyms: manĝ, konsum, absorb, nutr, devorar
    M1 filtering:
      ✓ manĝ (0.95) - plausible
      ✓ konsum (0.87) - plausible
      ✗ absorb (0.12) - implausible (liquids absorb, not eat)
      ✓ nutr (0.82) - plausible
    → Search with: [manĝ, konsum, nutr] only

Usage:
    python scripts/demo_m1_expansion.py                          # Example queries
    python scripts/demo_m1_expansion.py -i                       # Interactive mode
    python scripts/demo_m1_expansion.py "Kiu fondis Esperanton?" # Single query
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.models.m1_inference import M1Inference


def run_query(retriever: ASTAwareRetriever, query: str, top_k: int = 10):
    """Run a single query and display results."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)

    start = time.time()
    try:
        # Search with M1 expansion enabled
        results = retriever.search(
            query,
            top_k=top_k,
            use_m1_expansion=True,
            m1_min_plausibility=0.5
        )
        elapsed = time.time() - start

        if not results:
            print("  No results found.")
            return

        print(f"\nFound {len(results)} results in {elapsed:.2f}s\n")

        for i, (score, doc, stats) in enumerate(results[:5], 1):
            text = doc.get('text', '')
            source = doc.get('source', {}).get('name', 'unknown')

            print(f"{i}. Score: {score:.3f}")
            print(f"   {text}")
            print(f"   Source: {source}")
            print()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(retriever: ASTAwareRetriever):
    """Run interactive query loop."""
    print("\n" + "="*70)
    print("M1 Query Expansion Demo - Interactive Mode")
    print("="*70)
    print("Enter questions in Esperanto. Type 'quit' to exit.")
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

        run_query(retriever, query)


def main():
    parser = argparse.ArgumentParser(
        description="Demo M1 query expansion with plausibility filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('query', nargs='?', help='Query to search (or use -i for interactive)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--index', type=str, default='data/indexes/kuzu_index',
                        help='Index directory')
    parser.add_argument('--m1-model', type=str, default='models/m1_compositional/best_model.pt',
                        help='M1 model path')
    parser.add_argument('--comp-model', type=str, default='models/root_embeddings_tier0/best_model.pt',
                        help='CompositionalEmbedding path')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to retrieve')

    args = parser.parse_args()

    index_path = Path(args.index)
    m1_model_path = Path(args.m1_model)
    comp_model_path = Path(args.comp_model)

    # Check paths exist
    if not (index_path / "kuzu.db").exists():
        print(f"Error: Kuzu index not found at {index_path}/kuzu.db")
        print("Build index: python scripts/index_kuzu.py")
        sys.exit(1)

    if not m1_model_path.exists():
        print(f"Error: M1 model not found at {m1_model_path}")
        print("Train M1: ./scripts/train_m1_semantic_tier_priority.sh")
        sys.exit(1)

    if not comp_model_path.exists():
        print(f"Error: CompositionalEmbedding not found at {comp_model_path}")
        print("Train embeddings: ./scripts/train_roots.sh")
        sys.exit(1)

    # Initialize M1 model
    print("Initializing M1 model for query expansion...")
    try:
        m1 = M1Inference(
            model_path=m1_model_path,
            comp_model_path=comp_model_path,
            device='cpu'
        )
        print("  ✓ M1 model loaded")
    except Exception as e:
        print(f"Error loading M1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize retriever with M1
    print("\nInitializing retriever with M1 expansion...")
    try:
        retriever = ASTAwareRetriever(
            index_path=index_path,
            m1_model=m1  # Enable M1 query expansion
        )
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nReady!\n")

    # Interactive or single query
    if args.interactive:
        interactive_mode(retriever)
    elif args.query:
        run_query(retriever, args.query, top_k=args.top_k)
    else:
        # Default: run example queries
        example_queries = [
            "Kiu fondis Esperanton?",
            "Kio estas Esperanto?",
            "Kie naskiĝis Zamenhof?",
        ]
        print("Running example queries (use -i for interactive mode):\n")
        for query in example_queries:
            run_query(retriever, query, top_k=5)


if __name__ == '__main__':
    main()
