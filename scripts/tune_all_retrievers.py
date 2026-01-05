#!/usr/bin/env python3
"""
Comprehensive retriever tuning and comparison.

This script:
1. Tunes parameters for ALL available retrievers
2. Finds the best parameter configuration for each
3. Compares all retrievers using their optimal parameters
4. Recommends the best retriever overall
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Parameter configurations to test
CONFIGS = [
    # (prefilter_n, rerank_n, name, description)
    (100, 20, "ultra_fast", "Ultra fast - minimal accuracy loss"),
    (200, 50, "fast", "Fast - good for interactive UI"),
    (500, 100, "balanced", "Balanced - default settings"),
    (1000, 200, "accurate", "Accurate - better recall"),
    (2000, 500, "very_accurate", "Very accurate - best recall, slower"),
]


def check_retriever_availability(index_path: Path) -> Dict[str, bool]:
    """Check which retrievers are available based on index files."""
    return {
        'mmap': (index_path / 'mmap').exists(),
        'multifaiss': (index_path / 'multifaiss').exists(),
        'hybrid': (index_path / 'faiss').exists() and (index_path / 'mmap').exists(),
        'hnsw': (index_path / 'hnsw').exists() and (index_path / 'mmap').exists(),
        'scann': (index_path / 'scann').exists() and (index_path / 'mmap').exists(),
    }


def run_benchmark(
    index_path: Path,
    retriever: str,
    prefilter_n: int,
    rerank_n: int,
    top_k: int = 10
) -> Dict:
    """Run Q&A benchmark with specific parameters."""

    output_file = Path(f"/tmp/tune_qa_{retriever}_{prefilter_n}_{rerank_n}.json")

    cmd = [
        "python", "scripts/benchmark_qa_retrieval.py",
        "--index", str(index_path),
        "--retrievers", retriever,
        "-k", str(top_k),
        "--prefilter-n", str(prefilter_n),
        "--rerank-n", str(rerank_n),
        "--output", str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"    ✗ Benchmark failed: {result.stderr[:200]}")
            return None

        # Load results
        with open(output_file) as f:
            data = json.load(f)

        # Clean up
        output_file.unlink()

        # Extract metrics (handle list format)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data

    except subprocess.TimeoutExpired:
        logger.error(f"    ✗ Benchmark timeout (10 minutes)")
        return None
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
        return None


def tune_retriever(index_path: Path, retriever: str) -> Dict:
    """Tune a single retriever and find best configuration."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Tuning {retriever.upper()} retriever")
    logger.info(f"{'='*80}\n")

    results = []

    for prefilter_n, rerank_n, config_name, description in CONFIGS:
        logger.info(f"  Config: {config_name:15} (prefilter={prefilter_n:4}, rerank={rerank_n:3})")

        result_data = run_benchmark(index_path, retriever, prefilter_n, rerank_n)

        if result_data:
            total = result_data.get('total_questions', 1)
            top10 = result_data.get('answer_in_top_10', 0)
            latency = result_data.get('avg_time_ms', 0)

            results.append({
                'config_name': config_name,
                'prefilter_n': prefilter_n,
                'rerank_n': rerank_n,
                'top_10_count': top10,
                'total_questions': total,
                'accuracy': top10 / total if total > 0 else 0,
                'latency_ms': latency,
                'data': result_data
            })

            logger.info(f"    ✓ Top-10: {top10}/{total} ({top10/total*100:.1f}%), Latency: {latency:.1f}ms")
        else:
            logger.warning(f"    ⚠ Skipped due to error")

    if not results:
        return None

    # Find best configuration (highest accuracy, then fastest if tied)
    best = max(results, key=lambda x: (x['accuracy'], -x['latency_ms']))

    logger.info(f"\n  🏆 Best config for {retriever}: {best['config_name']}")
    logger.info(f"     Accuracy: {best['top_10_count']}/{best['total_questions']} ({best['accuracy']*100:.1f}%)")
    logger.info(f"     Latency:  {best['latency_ms']:.1f}ms")
    logger.info(f"     Params:   --prefilter-n {best['prefilter_n']} --rerank-n {best['rerank_n']}")

    return {
        'retriever': retriever,
        'best_config': best,
        'all_configs': results
    }


def print_final_comparison(tuning_results: List[Dict]):
    """Print final comparison of all retrievers with their best configs."""

    print("\n" + "=" * 120)
    print("FINAL COMPARISON - ALL RETRIEVERS WITH OPTIMAL PARAMETERS")
    print("=" * 120)
    print()

    header = f"{'Retriever':<15} {'Best Config':<15} {'Params':<20} {'Top-10':>10} {'Latency':>10}"
    print(header)
    print("-" * 120)

    # Sort by accuracy, then speed
    sorted_results = sorted(
        tuning_results,
        key=lambda x: (x['best_config']['accuracy'], -x['best_config']['latency_ms']),
        reverse=True
    )

    for result in sorted_results:
        ret = result['retriever']
        best = result['best_config']

        params = f"{best['prefilter_n']}/{best['rerank_n']}"
        accuracy_pct = best['accuracy'] * 100

        print(f"{ret:<15} "
              f"{best['config_name']:<15} "
              f"{params:<20} "
              f"{accuracy_pct:>9.1f}% "
              f"{best['latency_ms']:>8.1f}ms")

    print()
    print("=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)
    print()

    # Best overall accuracy
    best_accuracy = sorted_results[0]
    print(f"🎯 BEST FOR ACCURACY:")
    print(f"   Retriever: {best_accuracy['retriever']}")
    print(f"   Accuracy:  {best_accuracy['best_config']['accuracy']*100:.1f}% "
          f"({best_accuracy['best_config']['top_10_count']}/{best_accuracy['best_config']['total_questions']} questions)")
    print(f"   Latency:   {best_accuracy['best_config']['latency_ms']:.1f}ms")
    print(f"   Config:    {best_accuracy['best_config']['config_name']}")
    print(f"   Command:   --retrievers {best_accuracy['retriever']} "
          f"--prefilter-n {best_accuracy['best_config']['prefilter_n']} "
          f"--rerank-n {best_accuracy['best_config']['rerank_n']}")
    print()

    # Fastest
    fastest = min(sorted_results, key=lambda x: x['best_config']['latency_ms'])
    print(f"⚡ FASTEST:")
    print(f"   Retriever: {fastest['retriever']}")
    print(f"   Latency:   {fastest['best_config']['latency_ms']:.1f}ms")
    print(f"   Accuracy:  {fastest['best_config']['accuracy']*100:.1f}%")
    print(f"   Config:    {fastest['best_config']['config_name']}")
    print(f"   Command:   --retrievers {fastest['retriever']} "
          f"--prefilter-n {fastest['best_config']['prefilter_n']} "
          f"--rerank-n {fastest['best_config']['rerank_n']}")
    print()

    # Best efficiency (accuracy per millisecond)
    best_efficiency = max(
        sorted_results,
        key=lambda x: x['best_config']['accuracy'] / max(x['best_config']['latency_ms'], 1)
    )
    print(f"⚖️  BEST TRADEOFF (accuracy per ms):")
    print(f"   Retriever: {best_efficiency['retriever']}")
    print(f"   Accuracy:  {best_efficiency['best_config']['accuracy']*100:.1f}%")
    print(f"   Latency:   {best_efficiency['best_config']['latency_ms']:.1f}ms")
    print(f"   Efficiency: {best_efficiency['best_config']['accuracy']/max(best_efficiency['best_config']['latency_ms'],1)*1000:.2f} accuracy/sec")
    print(f"   Config:    {best_efficiency['best_config']['config_name']}")
    print(f"   Command:   --retrievers {best_efficiency['retriever']} "
          f"--prefilter-n {best_efficiency['best_config']['prefilter_n']} "
          f"--rerank-n {best_efficiency['best_config']['rerank_n']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Tune all retrievers and find the best one",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--skip-mmap',
        action='store_true',
        help='Skip the slow mmap retriever (recommended)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Check which retrievers are available
    available = check_retriever_availability(args.index)
    to_test = [name for name, is_avail in available.items() if is_avail]

    if args.skip_mmap and 'mmap' in to_test:
        to_test.remove('mmap')

    if not to_test:
        logger.error("No retrievers available to test")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE RETRIEVER TUNING")
    logger.info("=" * 80)
    logger.info(f"Testing retrievers: {', '.join(to_test)}")
    logger.info(f"Configurations per retriever: {len(CONFIGS)}")
    logger.info(f"Total benchmarks to run: {len(to_test) * len(CONFIGS)}")
    logger.info("=" * 80)

    # Tune each retriever
    tuning_results = []

    for retriever in to_test:
        result = tune_retriever(args.index, retriever)
        if result:
            tuning_results.append(result)

    if not tuning_results:
        logger.error("No successful tuning runs")
        sys.exit(1)

    # Print final comparison
    print_final_comparison(tuning_results)

    # Save detailed results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(tuning_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()
