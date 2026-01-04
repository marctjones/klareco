#!/usr/bin/env python3
"""
Demo: Slot-Based Retrieval

Test slot-based retrieval on challenging queries that failed with averaging.

Defaults favor accuracy over speed:
- top_k: 20 results (interactive), 10 results (demo mode)
- rerank_top_n: 500 candidates for stage 2 reranking

Usage:
    # Demo mode (4 test queries, 10 results each, 500 rerank candidates)
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full

    # Demo mode with translations
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full --translate

    # Interactive mode (20 results, 500 rerank candidates)
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full -i

    # Interactive mode with translations
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full -i --translate

    # Fast mode (5 results, 100 rerank candidates)
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full -i -k 5 --rerank-top-n 100

    # Very thorough mode (50 results, 1000 rerank candidates)
    python scripts/demo_slot_retrieval.py --index data/indexes/slot_full -i -k 50 --rerank-top-n 1000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever import SlotBasedRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Translation support
def load_translator():
    """Load EO→EN translation model."""
    try:
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-eo-en"
        logger.info(f"Loading EO→EN translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        logger.info("Translation model loaded!")

        def translate(text: str) -> str:
            """Translate Esperanto to English."""
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=512)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated

        return translate
    except ImportError:
        logger.warning("transformers not installed - translations disabled")
        logger.warning("Install with: pip install transformers sentencepiece")
        return None
    except Exception as e:
        logger.warning(f"Failed to load translation model: {e}")
        return None


def demo_queries(retriever: SlotBasedRetriever, top_k: int = 10, rerank_top_n: int = 500, translator=None):
    """Run demo queries that previously failed."""

    test_queries = [
        ("Kiu kreis Esperanton?", "Who created Esperanto?"),
        ("Kio estas Esperanto?", "What is Esperanto?"),
        ("Kiam Zamenhof kreis Esperanton?", "When did Zamenhof create Esperanto?"),
        ("Kie naskiĝis Zamenhof?", "Where was Zamenhof born?"),
    ]

    print("=" * 70)
    print("SLOT-BASED RETRIEVAL DEMO")
    if translator:
        print("(with translations)")
    print("=" * 70)
    print()

    for query_eo, query_en in test_queries:
        print(f"Query: {query_eo}")
        print(f"  EN: {query_en}")
        print()

        results = retriever.search(query_eo, top_k=top_k, rerank_top_n=rerank_top_n)

        if results:
            print(f"Top {len(results)} results:")
            for i, (score, doc) in enumerate(results, 1):
                text = doc['text']
                text_display = text if len(text) <= 80 else text[:77] + "..."
                print(f"  {i}. [{score:.3f}] {text_display}")

                # Add translation if available
                if translator:
                    try:
                        translation = translator(text)
                        translation_display = translation if len(translation) <= 80 else translation[:77] + "..."
                        print(f"      EN: {translation_display}")
                    except Exception as e:
                        logger.debug(f"Translation failed: {e}")

                # Show slot matching explanation for top result
                if i == 1:
                    explanation = retriever.explain_match(query_eo, doc)
                    print(f"     Slot matches:")
                    for slot, info in explanation['slot_matches'].items():
                        if info.get('status') == 'matched':
                            sim = info['similarity']
                            print(f"       {slot}: {sim:.3f}")
                        else:
                            print(f"       {slot}: {info['status']}")
        else:
            print("  No results found!")

        print()
        print("-" * 70)
        print()


def interactive_mode(retriever: SlotBasedRetriever, top_k: int = 20, rerank_top_n: int = 500, translator=None):
    """Interactive query mode."""

    print("=" * 70)
    print("INTERACTIVE SLOT-BASED RETRIEVAL")
    if translator:
        print("(with translations)")
    print("=" * 70)
    print()
    print("Enter queries in Esperanto (or 'quit' to exit)")
    print(f"Returning top {top_k} results")
    print()

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == 'quit':
            break

        print()
        results = retriever.search(query, top_k=top_k, rerank_top_n=rerank_top_n)

        if results:
            print(f"Top {len(results)} results:")
            for i, (score, doc) in enumerate(results, 1):
                text = doc['text']
                text_display = text if len(text) <= 80 else text[:77] + "..."
                print(f"  {i}. [{score:.3f}] {text_display}")

                # Add translation if available
                if translator:
                    try:
                        translation = translator(text)
                        translation_display = translation if len(translation) <= 80 else translation[:77] + "..."
                        print(f"      EN: {translation_display}")
                    except Exception as e:
                        logger.debug(f"Translation failed: {e}")

            # Show explanation for top result
            if results:
                print()
                print("Top result slot analysis:")
                explanation = retriever.explain_match(query, results[0][1])
                for slot, info in explanation['slot_matches'].items():
                    if info.get('status') == 'matched':
                        sim = info['similarity']
                        bar = '█' * int(sim * 20)
                        print(f"  {slot}: {sim:.3f} {bar}")
                    else:
                        print(f"  {slot}: {info['status']}")
        else:
            print("  No results found!")

        print()

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='Demo slot-based retrieval')
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--root-model',
        type=Path,
        default=Path('models/root_embeddings/best_model.pt'),
        help='Path to root embeddings model'
    )
    parser.add_argument(
        '--affix-model',
        type=Path,
        default=Path('models/affix_transforms_v2/best_model.pt'),
        help='Path to affix transforms model'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=20,
        help='Number of results to return (default: 20)'
    )
    parser.add_argument(
        '--rerank-top-n',
        type=int,
        default=500,
        help='Number of candidates for stage 2 reranking (default: 500)'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Enable EO→EN translation for results (requires transformers)'
    )

    args = parser.parse_args()

    # Set mode-specific defaults if user didn't specify -k
    # Check if -k was explicitly provided by user
    import sys
    k_specified = any(arg in sys.argv for arg in ['-k', '--top-k'])

    if not k_specified:
        # User didn't specify -k, use mode-appropriate defaults
        if args.interactive:
            args.top_k = 20  # Interactive: more results
        else:
            args.top_k = 10  # Demo: fewer results for readability

    # Validate inputs
    index_file = args.index / "slot_index.jsonl"
    if not index_file.exists():
        logger.error(f"Index not found: {index_file}")
        logger.error(f"Run: python scripts/index_slot_based.py --corpus <corpus> --output {args.index}")
        sys.exit(1)

    # Load indexer (for query embedding)
    logger.info("Loading models...")
    indexer = SlotBasedIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.index,  # Not used for retrieval
    )

    # Load retriever
    retriever = SlotBasedRetriever(
        index_path=index_file,
        indexer=indexer,
    )

    # Load translator if requested
    translator = None
    if args.translate:
        print()
        translator = load_translator()
        if translator is None:
            logger.warning("Continuing without translations")

    print()

    if args.interactive:
        interactive_mode(retriever, top_k=args.top_k, rerank_top_n=args.rerank_top_n, translator=translator)
    else:
        demo_queries(retriever, top_k=args.top_k, rerank_top_n=args.rerank_top_n, translator=translator)


if __name__ == '__main__':
    main()
