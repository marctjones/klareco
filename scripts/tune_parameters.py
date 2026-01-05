#!/usr/bin/env python3
"""
Systematic parameter tuning for retrievers.

Tests different combinations of prefilter_n and rerank_n to find optimal settings
for your use case (speed vs accuracy tradeoff).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
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


def run_benchmark(
    index_path: Path,
    retriever: str,
    prefilter_n: int,
    rerank_n: int,
    benchmark_type: str = "qa",
    top_k: int = 10
) -> Dict:
    """Run benchmark with specific parameters."""

    output_file = Path(f"/tmp/tune_results_{retriever}_{prefilter_n}_{rerank_n}.json")

    if benchmark_type == "qa":
        script = "scripts/benchmark_qa_retrieval.py"
        benchmark_file = "data/benchmarks/datasets/qa_benchmark_v1.jsonl"
    else:
        script = "scripts/compare_retrievers.py"
        benchmark_file = None

    cmd = [
        "python", script,
        "--index", str(index_path),
        "--retrievers", retriever,
        "-k", str(top_k),
        "--prefilter-n", str(prefilter_n),
        "--rerank-n", str(rerank_n),
        "--output", str(output_file)
    ]

    try:
        logger.info(f"  Testing {retriever}: prefilter={prefilter_n}, rerank={rerank_n}")
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

        # Extract metrics
        if isinstance(data, list):
            # Multiple retrievers, find the one we want
            results = [r for r in data if r['name'] == retriever or retriever in r['name']]
            if results:
                return results[0]
        return data

    except subprocess.TimeoutExpired:
        logger.error(f"    ✗ Benchmark timeout (10 minutes)")
        return None
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
        return None


def analyze_results(all_results: List[Dict], benchmark_type: str):
    """Analyze and print parameter tuning results."""

    print("\n" + "=" * 120)
    print("PARAMETER TUNING RESULTS")
    print("=" * 120)
    print()

    if benchmark_type == "qa":
        # Q&A benchmark metrics
        header = f"{'Config':<15} {'Prefilter':>10} {'Rerank':>8} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Latency':>10}"
        print(header)
        print("-" * 120)

        for result in all_results:
            config_name = result['config_name']
            prefilter = result['prefilter_n']
            rerank = result['rerank_n']

            total = result['data'].get('total_questions', 1)
            top1 = result['data'].get('answer_in_top_1', 0)
            top5 = result['data'].get('answer_in_top_5', 0)
            top10 = result['data'].get('answer_in_top_10', 0)
            latency = result['data'].get('avg_time_ms', 0)

            top1_pct = top1 / total * 100 if total > 0 else 0
            top5_pct = top5 / total * 100 if total > 0 else 0
            top10_pct = top10 / total * 100 if total > 0 else 0

            print(f"{config_name:<15} {prefilter:>10} {rerank:>8} "
                  f"{top1_pct:>7.1f}% {top5_pct:>7.1f}% {top10_pct:>7.1f}% "
                  f"{latency:>8.1f}ms")

        print()

        # Find best configurations
        best_accuracy = max(all_results, key=lambda x: x['data'].get('answer_in_top_10', 0))
        best_speed = min(all_results, key=lambda x: x['data'].get('avg_time_ms', float('inf')))

        # Find best speed/accuracy tradeoff (top-10 per millisecond)
        best_efficiency = max(
            all_results,
            key=lambda x: x['data'].get('answer_in_top_10', 0) / max(x['data'].get('avg_time_ms', 1), 1)
        )

        print("Recommendations:")
        print()
        print(f"  🎯 Best Accuracy:   {best_accuracy['config_name']}")
        print(f"     - Top-10: {best_accuracy['data'].get('answer_in_top_10', 0)}/{best_accuracy['data'].get('total_questions', 0)}")
        print(f"     - Latency: {best_accuracy['data'].get('avg_time_ms', 0):.1f}ms")
        print(f"     - Use: --prefilter-n {best_accuracy['prefilter_n']} --rerank-n {best_accuracy['rerank_n']}")
        print()

        print(f"  ⚡ Fastest:         {best_speed['config_name']}")
        print(f"     - Latency: {best_speed['data'].get('avg_time_ms', 0):.1f}ms")
        print(f"     - Top-10: {best_speed['data'].get('answer_in_top_10', 0)}/{best_speed['data'].get('total_questions', 0)}")
        print(f"     - Use: --prefilter-n {best_speed['prefilter_n']} --rerank-n {best_speed['rerank_n']}")
        print()

        print(f"  ⚖️  Best Tradeoff:   {best_efficiency['config_name']}")
        print(f"     - Top-10: {best_efficiency['data'].get('answer_in_top_10', 0)}/{best_efficiency['data'].get('total_questions', 0)}")
        print(f"     - Latency: {best_efficiency['data'].get('avg_time_ms', 0):.1f}ms")
        print(f"     - Use: --prefilter-n {best_efficiency['prefilter_n']} --rerank-n {best_efficiency['rerank_n']}")
        print()

    else:
        # Exact match benchmark (recall@k)
        header = f"{'Config':<15} {'Prefilter':>10} {'Rerank':>8} {'Recall@10':>10} {'MRR':>8} {'Latency':>10}"
        print(header)
        print("-" * 120)

        for result in all_results:
            config_name = result['config_name']
            prefilter = result['prefilter_n']
            rerank = result['rerank_n']

            recall = result['data'].get('recall@k', 0) * 100
            mrr = result['data'].get('mrr', 0)
            latency = result['data'].get('avg_time', 0)

            print(f"{config_name:<15} {prefilter:>10} {rerank:>8} "
                  f"{recall:>9.1f}% {mrr:>8.3f} {latency:>8.1f}ms")

        print()

        best_recall = max(all_results, key=lambda x: x['data'].get('recall@k', 0))
        best_speed = min(all_results, key=lambda x: x['data'].get('avg_time', float('inf')))

        print("Recommendations:")
        print()
        print(f"  🎯 Best Recall:  {best_recall['config_name']}")
        print(f"     - Recall@10: {best_recall['data'].get('recall@k', 0)*100:.1f}%")
        print(f"     - Use: --prefilter-n {best_recall['prefilter_n']} --rerank-n {best_recall['rerank_n']}")
        print()
        print(f"  ⚡ Fastest:      {best_speed['config_name']}")
        print(f"     - Latency: {best_speed['data'].get('avg_time', 0):.1f}ms")
        print(f"     - Use: --prefilter-n {best_speed['prefilter_n']} --rerank-n {best_speed['rerank_n']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Systematic parameter tuning for retrievers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        help='Retriever to tune (scann, hnsw, hybrid, multifaiss)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='qa',
        choices=['qa', 'exact'],
        help='Benchmark type: qa (Q&A task) or exact (exact sentence matching). Default: qa'
    )
    parser.add_argument(
        '--configs',
        type=str,
        help='Custom configs as comma-separated tuples: "prefilter:rerank:name,..." '
             'e.g. "100:20:fast,500:100:balanced,2000:500:accurate"'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Parse custom configs if provided
    if args.configs:
        configs = []
        for config_str in args.configs.split(','):
            parts = config_str.split(':')
            if len(parts) == 3:
                prefilter, rerank, name = parts
                configs.append((int(prefilter), int(rerank), name, name))
        if configs:
            global CONFIGS
            CONFIGS = configs

    logger.info(f"Tuning parameters for {args.retriever} retriever")
    logger.info(f"Testing {len(CONFIGS)} configurations")
    logger.info(f"Benchmark: {args.benchmark}")
    print()

    # Run benchmarks for each configuration
    all_results = []

    for prefilter_n, rerank_n, config_name, description in CONFIGS:
        logger.info(f"Configuration: {config_name} - {description}")

        result_data = run_benchmark(
            args.index,
            args.retriever,
            prefilter_n,
            rerank_n,
            benchmark_type=args.benchmark
        )

        if result_data:
            all_results.append({
                'config_name': config_name,
                'description': description,
                'prefilter_n': prefilter_n,
                'rerank_n': rerank_n,
                'data': result_data
            })
            logger.info(f"    ✓ Complete")
        else:
            logger.warning(f"    ⚠ Skipped due to error")

        print()

    # Analyze results
    if all_results:
        analyze_results(all_results, args.benchmark)

        # Save detailed results if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed results saved to {args.output}")
    else:
        logger.error("No successful benchmark runs")
        sys.exit(1)


if __name__ == '__main__':
    main()
