#!/usr/bin/env python3
"""
Demo script for Klareco RAG system with Index V3.

Shows end-to-end query → retrieval → answer generation.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.retriever import create_retriever
from klareco.experts.extractive import create_extractive_responder
from klareco.parser import parse


def demo_queries():
    """Run demo queries through the RAG system."""

    print("=" * 80)
    print("Klareco RAG System Demo - Index V3 with Complete Sentences")
    print("=" * 80)
    print()

    # Create retriever with index v3
    print("📚 Loading index...")
    retriever = create_retriever(
        'data/corpus_index_v3',
        'models/tree_lstm/best_model.pt'
    )
    print("✅ Index loaded: 26,725 sentences")
    print()

    # Create extractive responder
    responder = create_extractive_responder(retriever, top_k=3)

    # Demo queries
    queries = [
        ("Esperanto", "Kio estas Esperanto?"),
        ("Ring", "Kio estas la Unu Ringo?"),
        ("Hobbits", "Kie loĝas la hobitoj?"),
        ("Gandalf", "Kiu estas Gandalfo?"),
        ("Frodo", "Kiu estas Frodo Baginzo?"),
    ]

    for category, query in queries:
        print(f"🔍 [{category}] {query}")
        print()

        # Parse query
        query_ast = parse(query)

        # Get answer
        result = responder.execute(query_ast, query)

        # Display answer
        answer = result['answer']
        if len(answer) > 200:
            answer = answer[:200] + "..."

        print(f"💬 ANSWER:")
        print(f"   {answer}")
        print()
        print(f"⭐ Confidence: {result['confidence']:.3f}")
        print()

        # Show sources
        print(f"📖 Sources:")
        for i, src in enumerate(result['sources'], 1):
            score = f"[{src['score']:.2f}]" if src.get('score') is not None else ""
            text = src['text']
            if len(text) > 100:
                text = text[:100] + "..."
            source_name = src.get('source', 'unknown')
            print(f"   {i}. {score} {text}")
            print(f"      (from: {source_name})")

        print()
        print("-" * 80)
        print()


def interactive_mode():
    """Interactive query mode."""

    print("=" * 80)
    print("Klareco RAG - Interactive Mode")
    print("=" * 80)
    print()
    print("Enter queries in Esperanto (or 'quit' to exit)")
    print()

    # Load system
    print("Loading index...")
    retriever = create_retriever(
        'data/corpus_index_v3',
        'models/tree_lstm/best_model.pt'
    )
    responder = create_extractive_responder(retriever, top_k=3)
    print("✅ Ready!")
    print()

    while True:
        try:
            query = input("🔍 Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nĜis revido! 👋")
                break

            if not query:
                continue

            # Parse and query
            query_ast = parse(query)
            result = responder.execute(query_ast, query)

            # Display
            print()
            print(f"💬 {result['answer'][:200]}...")
            print(f"⭐ Confidence: {result['confidence']:.3f}")
            print()

        except KeyboardInterrupt:
            print("\n\nĜis revido! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo Klareco RAG system")
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive query mode'
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Query to run (if not interactive)'
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        # Single query mode
        query = ' '.join(args.query)
        print(f"Query: {query}\n")

        retriever = create_retriever(
            'data/corpus_index_v3',
            'models/tree_lstm/best_model.pt'
        )
        responder = create_extractive_responder(retriever, top_k=3)

        query_ast = parse(query)
        result = responder.execute(query_ast, query)

        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nSources:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {src['text'][:100]}...")
    else:
        # Run demo
        demo_queries()


if __name__ == '__main__':
    main()
