#!/usr/bin/env python3
"""
Test a single retriever implementation with a small query set.

Usage:
    python scripts/test_single_retriever.py --retriever faiss --queries 3
    python scripts/test_single_retriever.py --retriever sqlite --queries 5 --top-k 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from klareco.rag.slot_indexer import SlotBasedIndexer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_retriever(retriever_name: str, index_path: Path, num_queries: int = 3, top_k: int = 10):
    """Test a single retriever with a few queries."""

    logger.info("=" * 60)
    logger.info(f"Testing {retriever_name} retriever")
    logger.info("=" * 60)
    logger.info(f"Index: {index_path}")
    logger.info(f"Test queries: {num_queries}")
    logger.info(f"Top-K: {top_k}")
    logger.info("")

    # Initialize indexer (for embeddings)
    logger.info("Step 1: Loading embedding models...")
    indexer = SlotBasedIndexer(
        root_model_path=Path("models/root_embeddings/best_model.pt"),
        affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
        output_dir=index_path,  # Use index path as output dir
    )
    logger.info(f"  ✓ Loaded {len(indexer.root_to_idx)} roots")
    logger.info("")

    # Initialize retriever
    logger.info(f"Step 2: Initializing {retriever_name} retriever...")
    try:
        if retriever_name == "faiss":
            from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
            retriever = FAISSSlotRetriever(index_path, indexer)
        elif retriever_name == "multifaiss":
            from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
            retriever = MultiFAISSSlotRetriever(index_path, indexer)
        elif retriever_name == "sqlite":
            from klareco.rag.slot_retriever_sqlite import SQLiteSlotRetriever
            retriever = SQLiteSlotRetriever(index_path, indexer)
        elif retriever_name == "mmap":
            from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
            retriever = MemoryMappedSlotRetriever(index_path, indexer)
        else:
            logger.error(f"Unknown retriever: {retriever_name}")
            return False

        logger.info(f"  ✓ {retriever_name} retriever initialized")
        logger.info("")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize retriever: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load a few test queries from the index
    logger.info(f"Step 3: Loading {num_queries} test queries...")
    test_queries = []
    with open(index_path / "slot_index.jsonl") as f:
        for i, line in enumerate(f):
            if i >= num_queries:
                break
            doc = json.loads(line)
            test_queries.append(doc['text'])

    logger.info(f"  ✓ Loaded {len(test_queries)} queries")
    for i, q in enumerate(test_queries, 1):
        logger.info(f"    {i}. {q[:60]}...")
    logger.info("")

    # Test each query
    logger.info(f"Step 4: Running {len(test_queries)} test queries...")
    logger.info("")

    all_passed = True
    for i, query in enumerate(test_queries, 1):
        logger.info(f"Query {i}/{len(test_queries)}: {query}")

        try:
            results = retriever.search(query, top_k=top_k)

            if not results:
                logger.warning(f"  ⚠ No results returned")
                all_passed = False
            else:
                logger.info(f"  ✓ Returned {len(results)} results")

                # Show top 3 results
                for rank, (score, doc) in enumerate(results[:3], 1):
                    text_preview = doc['text'][:60] + "..." if len(doc['text']) > 60 else doc['text']
                    logger.info(f"    [{rank}] Score: {score:.4f} | {text_preview}")

                # Check if query itself appears in top results
                query_found = any(doc['text'] == query for _, doc in results[:top_k])
                if query_found:
                    logger.info(f"  ✓ Query found in top-{top_k} results (good recall)")
                else:
                    logger.warning(f"  ⚠ Query NOT in top-{top_k} (poor recall)")
                    all_passed = False

        except Exception as e:
            logger.error(f"  ✗ Query failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

        logger.info("")

    # Summary
    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.warning("⚠ SOME TESTS FAILED")
    logger.info("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test a single retriever implementation")
    parser.add_argument("--retriever", required=True, choices=["faiss", "multifaiss", "sqlite", "mmap"],
                       help="Retriever to test")
    parser.add_argument("--index", default="data/indexes/slot_full",
                       help="Index directory (default: data/indexes/slot_full)")
    parser.add_argument("--queries", type=int, default=3,
                       help="Number of test queries (default: 3)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results to retrieve (default: 10)")

    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        sys.exit(1)

    success = test_retriever(args.retriever, index_path, args.queries, args.top_k)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
