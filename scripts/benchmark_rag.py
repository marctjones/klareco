"""
RAG Performance Benchmarking Script

Benchmarks the RAG retrieval system across multiple dimensions:
- Retrieval latency (single query, batch queries)
- Throughput (queries per second)
- Memory usage
- Index load time
- Scalability (varying k values)

Usage:
    python scripts/benchmark_rag.py
    python scripts/benchmark_rag.py --queries 1000 --k 10
    python scripts/benchmark_rag.py --profile-memory
"""

import argparse
import time
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.retriever import create_retriever
from klareco.parser import parse


class RAGBenchmark:
    """
    Comprehensive RAG system benchmarking.
    """

    def __init__(self, retriever=None):
        """
        Initialize benchmark.

        Args:
            retriever: Pre-initialized retriever (optional)
        """
        self.retriever = retriever
        self.results = {}

    def benchmark_initialization(self) -> Dict[str, Any]:
        """
        Benchmark retriever initialization time.

        Returns:
            Timing statistics
        """
        print("Benchmarking initialization...")

        times = []
        for i in range(3):
            start = time.time()
            retriever = create_retriever()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

            if i == 0 and not self.retriever:
                self.retriever = retriever

        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0
        }

    def benchmark_single_query_latency(
        self,
        num_queries: int = 100,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark single query latency.

        Args:
            num_queries: Number of queries to test
            k: Number of results per query

        Returns:
            Latency statistics
        """
        print(f"\nBenchmarking single query latency ({num_queries} queries, k={k})...")

        test_queries = [
            "Mi amas hundojn",
            "La kato dormas",
            "Bona tago",
            "Kio estas Esperanto?",
            "La suno brilas",
            "Mi trinkas kafon",
            "La birdo kantas",
            "Hodiaŭ estas bela tago"
        ]

        latencies = []

        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]

            start = time.perf_counter()
            results = self.retriever.retrieve(query, k=k)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_queries} queries...")

        return {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'p95_ms': sorted(latencies)[int(0.95 * len(latencies))],
            'p99_ms': sorted(latencies)[int(0.99 * len(latencies))],
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        }

    def benchmark_batch_retrieval(
        self,
        batch_sizes: List[int] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark batch retrieval performance.

        Args:
            batch_sizes: List of batch sizes to test
            k: Number of results per query

        Returns:
            Batch performance statistics
        """
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20, 50]

        print(f"\nBenchmarking batch retrieval (k={k})...")

        test_queries = [
            "Mi amas hundojn",
            "La kato dormas",
            "Bona tago",
            "Kio estas Esperanto?",
            "La suno brilas"
        ]

        results = {}

        for batch_size in batch_sizes:
            queries = (test_queries * (batch_size // len(test_queries) + 1))[:batch_size]

            start = time.perf_counter()
            batch_results = self.retriever.batch_retrieve(queries, k=k)
            elapsed = time.perf_counter() - start

            qps = batch_size / elapsed

            results[f'batch_{batch_size}'] = {
                'total_time_s': elapsed,
                'queries_per_second': qps,
                'ms_per_query': (elapsed / batch_size) * 1000
            }

            print(f"  Batch size {batch_size:3d}: {qps:7.1f} QPS, {elapsed*1000/batch_size:6.2f} ms/query")

        return results

    def benchmark_varying_k(
        self,
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval with varying k values.

        Args:
            k_values: List of k values to test

        Returns:
            Performance for each k
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]

        print(f"\nBenchmarking varying k values...")

        query = "Mi amas hundojn"
        results = {}

        for k in k_values:
            times = []

            for _ in range(20):
                start = time.perf_counter()
                retrieved = self.retriever.retrieve(query, k=k)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            results[f'k_{k}'] = {
                'mean_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'results_returned': len(retrieved) if retrieved else 0
            }

            print(f"  k={k:3d}: {statistics.mean(times):6.2f} ms (median: {statistics.median(times):6.2f} ms)")

        return results

    def benchmark_throughput(
        self,
        duration_seconds: int = 10,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark sustained throughput.

        Args:
            duration_seconds: How long to run
            k: Number of results per query

        Returns:
            Throughput statistics
        """
        print(f"\nBenchmarking sustained throughput ({duration_seconds}s)...")

        test_queries = [
            "Mi amas hundojn",
            "La kato dormas",
            "Bona tago",
        ]

        query_count = 0
        latencies = []

        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        while time.perf_counter() < end_time:
            query = test_queries[query_count % len(test_queries)]

            query_start = time.perf_counter()
            self.retriever.retrieve(query, k=k)
            latency = (time.perf_counter() - query_start) * 1000

            latencies.append(latency)
            query_count += 1

        elapsed = time.perf_counter() - start_time
        qps = query_count / elapsed

        print(f"  Processed {query_count} queries in {elapsed:.2f}s")
        print(f"  Throughput: {qps:.1f} QPS")

        return {
            'total_queries': query_count,
            'duration_s': elapsed,
            'qps': qps,
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies)
        }

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """
        Benchmark memory usage (requires psutil).

        Returns:
            Memory statistics
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            print("\nBenchmarking memory usage...")

            # Get baseline memory
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Create retriever
            retriever = create_retriever()

            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            # Perform some queries
            for i in range(100):
                retriever.retrieve("Test query", k=10)

            mem_after_queries = process.memory_info().rss / 1024 / 1024  # MB

            print(f"  Baseline memory: {mem_before:.1f} MB")
            print(f"  After loading: {mem_after:.1f} MB")
            print(f"  After 100 queries: {mem_after_queries:.1f} MB")

            return {
                'baseline_mb': mem_before,
                'after_load_mb': mem_after,
                'after_queries_mb': mem_after_queries,
                'load_overhead_mb': mem_after - mem_before,
                'query_overhead_mb': mem_after_queries - mem_after
            }

        except ImportError:
            print("  psutil not available, skipping memory benchmark")
            return {'error': 'psutil not installed'}

    def run_all_benchmarks(
        self,
        num_queries: int = 100,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            num_queries: Number of queries for latency test
            k: Default k value

        Returns:
            Complete benchmark results
        """
        print("="*70)
        print("RAG PERFORMANCE BENCHMARK")
        print("="*70)

        results = {}

        # Initialization
        results['initialization'] = self.benchmark_initialization()

        # Single query latency
        results['single_query_latency'] = self.benchmark_single_query_latency(
            num_queries=num_queries,
            k=k
        )

        # Batch retrieval
        results['batch_retrieval'] = self.benchmark_batch_retrieval(k=k)

        # Varying k
        results['varying_k'] = self.benchmark_varying_k()

        # Throughput
        results['throughput'] = self.benchmark_throughput(duration_seconds=5, k=k)

        # Memory
        results['memory'] = self.benchmark_memory_usage()

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        # Initialization
        if 'initialization' in results:
            init = results['initialization']
            print(f"\n📦 Initialization:")
            print(f"  Mean: {init['mean']:.3f}s")
            print(f"  Range: {init['min']:.3f}s - {init['max']:.3f}s")

        # Latency
        if 'single_query_latency' in results:
            latency = results['single_query_latency']
            print(f"\n⚡ Single Query Latency:")
            print(f"  Mean: {latency['mean_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")
            print(f"  P95: {latency['p95_ms']:.2f} ms")
            print(f"  P99: {latency['p99_ms']:.2f} ms")

        # Throughput
        if 'throughput' in results:
            tp = results['throughput']
            print(f"\n🚀 Sustained Throughput:")
            print(f"  {tp['qps']:.1f} queries/second")
            print(f"  {tp['total_queries']} queries in {tp['duration_s']:.1f}s")

        # Best batch size
        if 'batch_retrieval' in results:
            batch = results['batch_retrieval']
            best_qps = max((v['queries_per_second'] for v in batch.values()))
            print(f"\n📊 Best Batch Performance:")
            print(f"  {best_qps:.1f} queries/second")

        # Memory
        if 'memory' in results and 'error' not in results['memory']:
            mem = results['memory']
            print(f"\n💾 Memory Usage:")
            print(f"  Load overhead: {mem['load_overhead_mb']:.1f} MB")
            print(f"  Query overhead: {mem['query_overhead_mb']:.2f} MB")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='RAG Performance Benchmark')
    parser.add_argument(
        '--queries',
        type=int,
        default=100,
        help='Number of queries for latency test'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to retrieve'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--profile-memory',
        action='store_true',
        help='Include memory profiling (requires psutil)'
    )

    args = parser.parse_args()

    # Run benchmarks
    benchmark = RAGBenchmark()
    results = benchmark.run_all_benchmarks(
        num_queries=args.queries,
        k=args.k
    )

    # Print summary
    benchmark.print_summary(results)

    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
