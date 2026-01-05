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
import time
import psutil
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever import SlotBasedRetriever

# Import specialized retrievers for large indexes
try:
    from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
    MULTIFAISS_AVAILABLE = True
except ImportError:
    MULTIFAISS_AVAILABLE = False

try:
    from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

try:
    from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever
    SCANN_AVAILABLE = True
except ImportError:
    SCANN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Resource monitoring utilities
def get_resource_usage():
    """Get current CPU and memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)
    return memory_mb, cpu_percent

def log_resources(prefix=""):
    """Log current resource usage."""
    memory_mb, cpu_percent = get_resource_usage()
    logger.info(f"{prefix}Memory: {memory_mb:.1f} MB | CPU: {cpu_percent:.1f}%")

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

    for query_num, (query_eo, query_en) in enumerate(test_queries, 1):
        print(f"Query {query_num}/{len(test_queries)}: {query_eo}")
        print(f"  EN: {query_en}")
        print()

        log_resources(f"Q{query_num} before: ")
        start_time = time.time()

        # Call search with appropriate parameters based on retriever type
        if 'ScaNN' in retriever.__class__.__name__:
            # ScaNNSlotRetriever uses scann_top_n
            results = retriever.search(query_eo, top_k=top_k, scann_top_n=rerank_top_n)
        elif 'HNSW' in retriever.__class__.__name__:
            # HNSWSlotRetriever uses hnsw_top_n
            results = retriever.search(query_eo, top_k=top_k, hnsw_top_n=rerank_top_n)
        elif 'Hybrid' in retriever.__class__.__name__:
            # HybridFAISSMmapRetriever uses faiss_top_n
            results = retriever.search(query_eo, top_k=top_k, faiss_top_n=rerank_top_n)
        elif 'MultiFAISS' in retriever.__class__.__name__:
            # MultiFAISSSlotRetriever uses slot_top_n
            results = retriever.search(query_eo, top_k=top_k, slot_top_n=rerank_top_n)
        else:
            # SlotBasedRetriever and MemoryMappedSlotRetriever use rerank_top_n
            results = retriever.search(query_eo, top_k=top_k, rerank_top_n=rerank_top_n)

        query_time = (time.time() - start_time) * 1000  # ms
        log_resources(f"Q{query_num} after: ")
        logger.info(f"Query {query_num} time: {query_time:.1f} ms")

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
        log_resources("Before query: ")
        start_time = time.time()

        # Call search with appropriate parameters based on retriever type
        if 'ScaNN' in retriever.__class__.__name__:
            results = retriever.search(query, top_k=top_k, scann_top_n=rerank_top_n)
        elif 'HNSW' in retriever.__class__.__name__:
            results = retriever.search(query, top_k=top_k, hnsw_top_n=rerank_top_n)
        elif 'Hybrid' in retriever.__class__.__name__:
            results = retriever.search(query, top_k=top_k, faiss_top_n=rerank_top_n)
        elif 'MultiFAISS' in retriever.__class__.__name__:
            results = retriever.search(query, top_k=top_k, slot_top_n=rerank_top_n)
        else:
            results = retriever.search(query, top_k=top_k, rerank_top_n=rerank_top_n)

        query_time = (time.time() - start_time) * 1000  # ms
        log_resources("After query: ")
        logger.info(f"Query time: {query_time:.1f} ms")

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

    # Check index size to determine which retriever to use
    import subprocess
    try:
        result = subprocess.run(['wc', '-l', str(index_file)], capture_output=True, text=True)
        num_docs = int(result.stdout.split()[0])
        logger.info(f"Index contains {num_docs:,} documents")
    except:
        num_docs = 0
        logger.warning("Could not determine index size")

    # Auto-select retriever based on index size
    log_resources("Initial: ")

    use_scann = False
    use_hnsw = False
    use_hybrid = False
    use_multifaiss = False

    if num_docs > 100000:  # More than 100K docs
        # Prefer HNSW if available (fastest + simplest)
        if HNSW_AVAILABLE and (args.index / "hnsw").exists() and (args.index / "mmap").exists():
            use_hnsw = True
            logger.info(f"Large index ({num_docs:,} docs) - using HNSWSlot retriever")
            logger.info("Note: HNSW graph traversal + mmap slot reranking")
            logger.info("Expected: 85-90% recall, ~2-3ms latency (fastest + simplest)")
        # Use ScaNN if available (highest accuracy but requires TensorFlow)
        elif SCANN_AVAILABLE and (args.index / "scann").exists() and (args.index / "mmap").exists():
            use_scann = True
            logger.info(f"Large index ({num_docs:,} docs) - using ScaNNSlot retriever")
            logger.info("Note: ScaNN anisotropic quantization + mmap slot reranking")
            logger.info("Expected: 90-95% recall, ~3-5ms latency (highest accuracy)")
        # Fall back to Hybrid if HNSW/ScaNN not available
        elif HYBRID_AVAILABLE and (args.index / "faiss").exists() and (args.index / "mmap").exists():
            use_hybrid = True
            logger.info(f"Large index ({num_docs:,} docs) - using HybridFAISSMmap retriever")
            logger.info("Note: Combines FAISS pre-filtering + mmap slot reranking")
            logger.info("Expected: 90% recall, ~3.5ms latency (FAISS-based)")
        elif MULTIFAISS_AVAILABLE and (args.index / "multifaiss").exists():
            use_multifaiss = True
            logger.info(f"Large index ({num_docs:,} docs) - using MultiFAISS retriever")
            logger.info("Note: Separate FAISS index per slot (SUBJ/VERB/OBJ)")
            logger.info("Expected: 75% recall, 1.1ms latency (fallback option)")
        else:
            logger.warning(f"Large index ({num_docs:,} docs) but no optimized retriever available")
            logger.warning("This may use significant memory. Consider using a smaller test index.")
            logger.warning("Or build indexes:")
            logger.warning(f"  HNSW (recommended): ./scripts/build_hnsw_index.sh --index {args.index}")
            logger.warning(f"  ScaNN (highest accuracy): ./scripts/build_scann_index.sh --index {args.index}")
            logger.warning(f"  Hybrid: python scripts/index_slot_based.py --index {args.index} --build-faiss")
            logger.warning(f"  MultiFAISS: python scripts/index_slot_based.py --index {args.index} --build-multifaiss")

    # Load indexer (for query embedding)
    logger.info("Loading models...")
    log_resources("Before model load: ")

    indexer = SlotBasedIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.index,  # Not used for retrieval
    )

    log_resources("After model load: ")

    # Load retriever (hnsw > scann > hybrid > multifaiss > slotbased)
    logger.info("Loading retriever...")
    if use_hnsw:
        logger.info("Using HNSWSlotRetriever (HNSW + mmap, 85-90% recall)")
        log_resources("Before retriever load: ")
        retriever = HNSWSlotRetriever(
            index_path=args.index,
            indexer=indexer,
        )
        log_resources("After retriever load: ")
    elif use_scann:
        logger.info("Using ScaNNSlotRetriever (ScaNN + mmap, 90-95% recall)")
        log_resources("Before retriever load: ")
        retriever = ScaNNSlotRetriever(
            index_path=args.index,
            indexer=indexer,
        )
        log_resources("After retriever load: ")
    elif use_hybrid:
        logger.info("Using HybridFAISSMmapRetriever (FAISS + mmap, 90% recall)")
        log_resources("Before retriever load: ")
        retriever = HybridFAISSMmapRetriever(
            index_path=args.index,
            indexer=indexer,
        )
        log_resources("After retriever load: ")
    elif use_multifaiss:
        logger.info("Using MultiFAISSSlotRetriever (3 separate indexes, 75% recall)")
        log_resources("Before retriever load: ")
        retriever = MultiFAISSSlotRetriever(
            index_path=args.index,
            indexer=indexer,
        )
        log_resources("After retriever load: ")
    else:
        logger.info("Using SlotBasedRetriever (loads all docs in memory)")
        log_resources("Before retriever load: ")
        retriever = SlotBasedRetriever(
            index_path=index_file,
            indexer=indexer,
        )
        log_resources("After retriever load: ")

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
