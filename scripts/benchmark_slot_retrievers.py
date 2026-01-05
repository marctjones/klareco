#!/usr/bin/env python3
"""
Benchmark slot-based retriever implementations.

Compares 4 solutions:
1. Memory-mapped index (low memory, slower)
2. FAISS pre-filtering (fast, medium memory)
3. Multi-FAISS per slot (fastest, medium memory)
4. SQLite backend (low memory, medium speed)

Measures:
- Peak memory usage (RSS)
- Query latency (mean, p50, p95, p99)
- Accuracy (Recall@10, MRR, NDCG)

Usage:
    # Benchmark all solutions on test index
    python scripts/benchmark_slot_retrievers.py \
        --index data/indexes/slot_test \
        --queries scripts/benchmark_queries.jsonl \
        --output benchmark_results.json

    # Benchmark specific solution
    python scripts/benchmark_slot_retrievers.py \
        --index data/indexes/slot_full \
        --queries scripts/benchmark_queries.jsonl \
        --solution faiss \
        --output faiss_results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever import SlotBasedRetriever
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
from klareco.rag.slot_retriever_sqlite import SQLiteSlotRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class MemoryTracker:
    """Track peak memory usage during execution."""

    def __init__(self):
        self.baseline = get_memory_usage_mb()
        self.peak = self.baseline

    def update(self):
        """Update peak memory."""
        current = get_memory_usage_mb()
        self.peak = max(self.peak, current)

    def get_peak_delta(self) -> float:
        """Get peak memory increase from baseline."""
        return self.peak - self.baseline


def load_queries(queries_path: Path) -> List[Dict]:
    """
    Load benchmark queries from JSONL file.

    Format:
        {"query": "Kiu kreis Esperanton?", "relevant_docs": ["text1", "text2"]}
    """
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def compute_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Recall@K."""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    return len(retrieved_k.intersection(relevant_set)) / len(relevant_set)


def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    relevant_set = set(relevant)

    for i, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            return 1.0 / i

    return 0.0


def compute_ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain@K."""
    # Simple binary relevance version
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal DCG (all relevant docs at top)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def benchmark_retriever(
    retriever,
    queries: List[Dict],
    top_k: int = 10,
    checkpoint_path: Path = None,
) -> Dict:
    """
    Benchmark a retriever implementation.

    Returns:
        Dictionary with metrics: latency, accuracy, memory
    """
    logger.info(f"Benchmarking {retriever.__class__.__name__}")

    # Check for existing checkpoint
    start_idx = 0
    query_times = []
    recall_at_10 = []
    mrr_scores = []
    ndcg_at_10 = []

    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"  Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('completed_queries', 0)
            query_times = checkpoint.get('query_times', [])
            recall_at_10 = checkpoint.get('recall_at_10', [])
            mrr_scores = checkpoint.get('mrr_scores', [])
            ndcg_at_10 = checkpoint.get('ndcg_at_10', [])
        logger.info(f"  Resuming from query {start_idx}/{len(queries)}")

    # Track memory
    mem_tracker = MemoryTracker()
    mem_tracker.update()

    # Run queries
    total_queries = len(queries)
    benchmark_start = time.time()
    last_update_time = benchmark_start

    for i in range(start_idx, total_queries):
        query_data = queries[i]
        query = query_data['query']
        relevant_docs = query_data.get('relevant_docs', [])

        # Time query
        start_time = time.time()
        results = retriever.search(query, top_k=top_k)
        end_time = time.time()

        query_time = (end_time - start_time) * 1000  # ms
        query_times.append(query_time)

        # Update memory tracker
        prev_mem = mem_tracker.peak
        mem_tracker.update()
        current_mem = get_memory_usage_mb()

        # Extract retrieved document texts
        retrieved_texts = [doc['text'] for score, doc in results]

        # Compute accuracy metrics (if we have ground truth)
        if relevant_docs:
            recall_at_10.append(compute_recall_at_k(retrieved_texts, relevant_docs, 10))
            mrr_scores.append(compute_mrr(retrieved_texts, relevant_docs))
            ndcg_at_10.append(compute_ndcg_at_k(retrieved_texts, relevant_docs, 10))

        # Detect meaningful events
        avg_query_time = np.mean(query_times) if len(query_times) > 1 else query_time
        is_slow_query = query_time > avg_query_time * 2  # Query took 2x longer than average
        is_memory_spike = current_mem > prev_mem + 100  # Memory increased by >100MB

        # Progress reporting with stats
        # Update on: time (1 min), milestones (5 queries), checkpoints (10), events (slow/memory), or completion
        current_time = time.time()
        time_since_last_update = current_time - last_update_time
        is_time_based = time_since_last_update >= 60  # 1 minute
        is_query_milestone = (i + 1) % 5 == 0
        is_checkpoint = (i + 1) % 10 == 0
        is_final = i + 1 == total_queries

        if is_time_based or is_query_milestone or is_checkpoint or is_slow_query or is_memory_spike or is_final:
            avg_time = np.mean(query_times[-5:]) if len(query_times) >= 5 else np.mean(query_times)
            avg_recall = np.mean(recall_at_10[-5:]) if len(recall_at_10) >= 5 else (np.mean(recall_at_10) if recall_at_10 else 0.0)
            current_mem = get_memory_usage_mb()

            # Estimate time remaining
            elapsed = time.time() - benchmark_start
            queries_done = i + 1 - start_idx
            queries_left = total_queries - (i + 1)
            if queries_done > 0:
                time_per_query = elapsed / queries_done
                eta_seconds = time_per_query * queries_left
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds >= 60 else f"{int(eta_seconds)}s"
            else:
                eta_str = "calculating..."

            # Build progress message
            progress_parts = [f"[{i+1}/{total_queries}]"]
            progress_parts.append(f"Latency: {avg_time:.1f}ms")
            progress_parts.append(f"Recall: {avg_recall:.3f}")
            progress_parts.append(f"Memory: {current_mem:.0f}MB")
            progress_parts.append(f"ETA: {eta_str}")

            # Add indicator for why we're updating (prioritize meaningful events)
            event_indicators = []
            if is_slow_query:
                event_indicators.append(f"⚠ slow query: {query_time:.0f}ms")
            if is_memory_spike:
                event_indicators.append(f"📈 mem spike: +{current_mem - prev_mem:.0f}MB")
            if is_checkpoint:
                event_indicators.append("💾 checkpoint")
            elif is_time_based:
                event_indicators.append("⏰ 1min")

            if event_indicators:
                progress_parts.append(" ".join(event_indicators))

            logger.info("  " + " | ".join(progress_parts))
            last_update_time = current_time

            # Save checkpoint every 10 queries
            if checkpoint_path and is_checkpoint:
                checkpoint_data = {
                    'completed_queries': i + 1,
                    'query_times': query_times,
                    'recall_at_10': recall_at_10,
                    'mrr_scores': mrr_scores,
                    'ndcg_at_10': ndcg_at_10,
                }
                temp_path = checkpoint_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                temp_path.rename(checkpoint_path)

    # Compute statistics
    metrics = {
        'retriever': retriever.__class__.__name__,
        'num_queries': len(queries),
        'latency': {
            'mean_ms': float(np.mean(query_times)),
            'median_ms': float(np.median(query_times)),
            'p95_ms': float(np.percentile(query_times, 95)),
            'p99_ms': float(np.percentile(query_times, 99)),
            'min_ms': float(np.min(query_times)),
            'max_ms': float(np.max(query_times)),
        },
        'memory': {
            'peak_mb': float(mem_tracker.peak),
            'delta_mb': float(mem_tracker.get_peak_delta()),
        },
    }

    # Add accuracy metrics if available
    if recall_at_10:
        metrics['accuracy'] = {
            'recall_at_10': float(np.mean(recall_at_10)),
            'mrr': float(np.mean(mrr_scores)),
            'ndcg_at_10': float(np.mean(ndcg_at_10)),
        }

    return metrics


def create_benchmark_queries(index_path: Path, output_path: Path, num_queries: int = 50):
    """
    Create benchmark queries from index.

    Samples random documents and uses them as queries with known relevant docs.
    """
    logger.info(f"Creating {num_queries} benchmark queries from {index_path}")

    index_file = index_path / "slot_index.jsonl"

    # Load all documents with progress
    logger.info("  Loading index documents...")
    documents = []
    total_lines = 0
    with open(index_file) as f:
        for i, line in enumerate(f, 1):
            doc = json.loads(line)
            documents.append(doc['text'])

            if i % 500000 == 0:
                logger.info(f"    Loaded {i:,} documents...")
            total_lines = i

    logger.info(f"  Loaded {total_lines:,} documents from index")

    # Sample queries
    logger.info(f"  Sampling {num_queries} queries...")
    np.random.seed(42)
    query_indices = np.random.choice(len(documents), num_queries, replace=False)

    queries = []
    for idx in query_indices:
        # Use document as query, mark itself as relevant
        queries.append({
            'query': documents[idx],
            'relevant_docs': [documents[idx]],
        })

    # Save queries
    logger.info(f"  Writing queries to {output_path}...")
    with open(output_path, 'w') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    logger.info(f"  ✓ Saved {num_queries} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark slot-based retrievers')
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--queries',
        type=Path,
        help='Path to benchmark queries JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results.json'),
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--solution',
        choices=['baseline', 'mmap', 'faiss', 'multifaiss', 'sqlite', 'all'],
        default='all',
        help='Which solution to benchmark'
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
        '--create-queries',
        action='store_true',
        help='Create benchmark queries from index'
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=50,
        help='Number of benchmark queries to create'
    )

    args = parser.parse_args()

    # Validate inputs
    index_file = args.index / "slot_index.jsonl"
    if not index_file.exists():
        logger.error(f"Index not found: {index_file}")
        sys.exit(1)

    # Create queries if requested
    if args.create_queries:
        queries_path = args.index / "benchmark_queries.jsonl"
        create_benchmark_queries(args.index, queries_path, args.num_queries)
        args.queries = queries_path

    if not args.queries:
        logger.error("--queries is required (or use --create-queries)")
        sys.exit(1)

    if not args.queries.exists():
        logger.error(f"Queries file not found: {args.queries}")
        sys.exit(1)

    # Load queries
    logger.info("=" * 60)
    logger.info("Loading benchmark queries")
    logger.info("=" * 60)
    queries = load_queries(args.queries)
    logger.info(f"✓ Loaded {len(queries)} queries from {args.queries}")
    logger.info("")

    # Load indexer (shared across all retrievers)
    logger.info("=" * 60)
    logger.info("Initializing indexer (loading embedding models)")
    logger.info("=" * 60)
    logger.info(f"  Root model: {args.root_model}")
    logger.info(f"  Affix model: {args.affix_model}")

    start_time = time.time()
    indexer = SlotBasedIndexer(
        root_model_path=args.root_model,
        affix_model_path=args.affix_model,
        output_dir=args.index,
    )
    load_time = time.time() - start_time
    logger.info(f"✓ Indexer loaded in {load_time:.1f}s")
    logger.info("")

    # Benchmark solutions
    results = {
        'index_path': str(args.index),
        'queries_path': str(args.queries),
        'num_queries': len(queries),
        'solutions': [],
    }

    solutions_to_test = []

    if args.solution == 'all' or args.solution == 'baseline':
        solutions_to_test.append(('baseline', SlotBasedRetriever))

    if args.solution == 'all' or args.solution == 'mmap':
        solutions_to_test.append(('mmap', MemoryMappedSlotRetriever))

    if args.solution == 'all' or args.solution == 'faiss':
        solutions_to_test.append(('faiss', FAISSSlotRetriever))

    if args.solution == 'all' or args.solution == 'multifaiss':
        solutions_to_test.append(('multifaiss', MultiFAISSSlotRetriever))

    if args.solution == 'all' or args.solution == 'sqlite':
        solutions_to_test.append(('sqlite', SQLiteSlotRetriever))

    for name, RetrieverClass in solutions_to_test:
        logger.info("=" * 60)
        logger.info(f"Benchmarking: {name}")
        logger.info("=" * 60)

        # Check if this solution already has completed results
        output_dir = args.output.parent if args.output.parent != Path('.') else Path('.')
        solution_result_file = output_dir / f"{name}_results.json"

        if solution_result_file.exists():
            logger.info(f"  Found existing results for {name}, loading from {solution_result_file}")
            try:
                with open(solution_result_file) as f:
                    existing_metrics = json.load(f)
                    # Check if it's complete
                    if existing_metrics.get('num_queries') == len(queries):
                        logger.info(f"  Skipping {name} (already completed)")
                        results['solutions'].append(existing_metrics)
                        continue
            except Exception as e:
                logger.warning(f"  Could not load existing results: {e}, re-running benchmark")

        try:
            # Create retriever
            logger.info(f"  Initializing {name} retriever...")
            init_start = time.time()

            if RetrieverClass == SlotBasedRetriever:
                retriever = RetrieverClass(
                    index_path=index_file,
                    indexer=indexer,
                )
            else:
                retriever = RetrieverClass(
                    index_path=args.index,
                    indexer=indexer,
                )

            init_time = time.time() - init_start
            logger.info(f"  ✓ Retriever initialized in {init_time:.1f}s")
            logger.info("")

            # Create checkpoint path for this solution
            checkpoint_path = output_dir / f"{name}_checkpoint.json"

            # Benchmark
            logger.info(f"  Starting benchmark run ({len(queries)} queries)...")
            logger.info("")
            bench_start = time.time()
            metrics = benchmark_retriever(retriever, queries, checkpoint_path=checkpoint_path)
            bench_time = time.time() - bench_start

            logger.info("")
            logger.info(f"  ✓ Benchmark completed in {bench_time:.1f}s")
            results['solutions'].append(metrics)

            # Save individual solution results immediately
            with open(solution_result_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  Saved results to {solution_result_file}")

            # Clean up checkpoint after successful completion
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # Print summary
            logger.info("=" * 60)
            logger.info(f"RESULTS SUMMARY: {name}")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Latency Metrics:")
            logger.info(f"  Mean:     {metrics['latency']['mean_ms']:>8.2f} ms")
            logger.info(f"  Median:   {metrics['latency']['median_ms']:>8.2f} ms")
            logger.info(f"  P95:      {metrics['latency']['p95_ms']:>8.2f} ms")
            logger.info(f"  P99:      {metrics['latency']['p99_ms']:>8.2f} ms")
            logger.info(f"  Range:    {metrics['latency']['min_ms']:>8.2f} - {metrics['latency']['max_ms']:.2f} ms")
            logger.info("")
            logger.info("Memory Usage:")
            logger.info(f"  Peak:     {metrics['memory']['peak_mb']:>8.1f} MB")
            logger.info(f"  Delta:    {metrics['memory']['delta_mb']:>8.1f} MB (increase from baseline)")
            logger.info("")

            if 'accuracy' in metrics:
                logger.info("Accuracy Metrics:")
                logger.info(f"  Recall@10: {metrics['accuracy']['recall_at_10']:>7.3f} ({metrics['accuracy']['recall_at_10']*100:.1f}%)")
                logger.info(f"  MRR:       {metrics['accuracy']['mrr']:>7.3f}")
                logger.info(f"  NDCG@10:   {metrics['accuracy']['ndcg_at_10']:>7.3f}")
                logger.info("")

            logger.info("=" * 60)
            logger.info("")

            # Cleanup
            if hasattr(retriever, 'close'):
                retriever.close()

        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}", exc_info=True)
            results['solutions'].append({
                'retriever': name,
                'error': str(e),
            })

    # Save results
    logger.info("=" * 60)
    logger.info(f"Saving results to {args.output}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 100)
    print("FINAL BENCHMARK COMPARISON")
    print("=" * 100)
    print()
    print(f"Index: {args.index}")
    print(f"Queries: {len(queries)}")
    print()
    print(f"{'Solution':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'Memory (MB)':<12} {'Recall@10':<12}")
    print("-" * 100)

    for sol in results['solutions']:
        if 'error' in sol:
            print(f"{sol['retriever']:<15} ERROR: {sol['error']}")
            continue

        name = sol['retriever']
        mean_lat = sol['latency']['mean_ms']
        median_lat = sol['latency']['median_ms']
        p95_lat = sol['latency']['p95_ms']
        memory = sol['memory']['delta_mb']
        recall = sol.get('accuracy', {}).get('recall_at_10', 0.0)

        print(f"{name:<15} {mean_lat:<12.2f} {median_lat:<12.2f} {p95_lat:<12.2f} {memory:<12.1f} {recall:<12.3f}")

    print()
    print("=" * 100)
    print()

    # Print recommendations
    if len(results['solutions']) > 1:
        valid_solutions = [s for s in results['solutions'] if 'error' not in s]
        if valid_solutions:
            print("RECOMMENDATIONS:")
            print()

            # Fastest solution
            fastest = min(valid_solutions, key=lambda s: s['latency']['mean_ms'])
            print(f"  🚀 FASTEST:      {fastest['retriever']} ({fastest['latency']['mean_ms']:.1f}ms mean latency)")

            # Most accurate
            if all('accuracy' in s for s in valid_solutions):
                most_accurate = max(valid_solutions, key=lambda s: s['accuracy']['recall_at_10'])
                print(f"  🎯 MOST ACCURATE: {most_accurate['retriever']} ({most_accurate['accuracy']['recall_at_10']*100:.1f}% recall)")

            # Lowest memory
            lowest_mem = min(valid_solutions, key=lambda s: s['memory']['delta_mb'])
            print(f"  💾 LOWEST MEMORY: {lowest_mem['retriever']} ({lowest_mem['memory']['delta_mb']:.0f}MB)")

            print()
            print("=" * 100)
            print()


if __name__ == '__main__':
    main()
