#!/usr/bin/env python3
"""
Analyze benchmark results from compare_retrievers.py against ground truth.

Computes recall@k, MRR (Mean Reciprocal Rank), and other metrics.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set


def load_ground_truth(benchmark_file: Path) -> Dict[str, Set[str]]:
    """Load ground truth relevant docs for each query."""
    ground_truth = {}
    with open(benchmark_file) as f:
        for line in f:
            data = json.loads(line.strip())
            query = data.get('query', '')
            relevant = set(data.get('relevant_docs', []))
            if query and relevant:
                ground_truth[query] = relevant
    return ground_truth


def compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Compute recall@k: fraction of relevant docs in top-k results."""
    if not relevant:
        return 0.0

    retrieved_set = set(retrieved[:k])
    hits = len(retrieved_set & relevant)
    return hits / len(relevant)


def compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def analyze_retriever(results: Dict, ground_truth: Dict[str, Set[str]], k: int = 10) -> Dict:
    """Analyze a single retriever's results."""
    recalls = []
    mrrs = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for query_result in results['queries']:
        query = query_result['query_eo']

        # Get ground truth for this query
        relevant = ground_truth.get(query, set())
        if not relevant:
            continue

        # Get retrieved docs
        retrieved = [r['text'] for r in query_result.get('results', [])]

        # Compute metrics
        recall_k = compute_recall_at_k(retrieved, relevant, k)
        mrr = compute_mrr(retrieved, relevant)

        recalls.append(recall_k)
        mrrs.append(mrr)

        # Check if any relevant doc in top-1, top-5, top-10
        if any(doc in relevant for doc in retrieved[:1]):
            hits_at_1 += 1
        if any(doc in relevant for doc in retrieved[:5]):
            hits_at_5 += 1
        if any(doc in relevant for doc in retrieved[:10]):
            hits_at_10 += 1

    num_queries = len(recalls)

    return {
        'name': results['name'],
        'recall@k': sum(recalls) / num_queries if recalls else 0.0,
        'mrr': sum(mrrs) / num_queries if mrrs else 0.0,
        'hits@1': hits_at_1 / num_queries if num_queries > 0 else 0.0,
        'hits@5': hits_at_5 / num_queries if num_queries > 0 else 0.0,
        'hits@10': hits_at_10 / num_queries if num_queries > 0 else 0.0,
        'num_queries': num_queries,
        'avg_time_ms': results.get('avg_time', 0),
        'memory_mb': results.get('memory_peak', 0),
    }


def print_analysis(analyses: List[Dict], k: int):
    """Print formatted analysis table."""
    print("=" * 120)
    print(f"BENCHMARK ANALYSIS (k={k})")
    print("=" * 120)
    print()

    # Header
    header = f"{'Retriever':<15} {'Recall@k':>10} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'Latency':>10} {'Memory':>10} {'Queries':>8}"
    print(header)
    print("-" * 120)

    # Sort by recall@k (descending)
    sorted_analyses = sorted(analyses, key=lambda x: x['recall@k'], reverse=True)

    for analysis in sorted_analyses:
        print(f"{analysis['name']:<15} "
              f"{analysis['recall@k']:>9.1%} "
              f"{analysis['mrr']:>8.3f} "
              f"{analysis['hits@1']:>7.1%} "
              f"{analysis['hits@5']:>7.1%} "
              f"{analysis['hits@10']:>7.1%} "
              f"{analysis['avg_time_ms']:>8.1f}ms "
              f"{analysis['memory_mb']:>8.0f}MB "
              f"{analysis['num_queries']:>8}")

    print()
    print("Rankings:")

    # Best recall
    best_recall = sorted_analyses[0]
    print(f"  🎯 Best Recall:  {best_recall['name']} ({best_recall['recall@k']:.1%})")

    # Best MRR
    best_mrr = max(sorted_analyses, key=lambda x: x['mrr'])
    print(f"  🥇 Best MRR:     {best_mrr['name']} ({best_mrr['mrr']:.3f})")

    # Fastest
    fastest = min(sorted_analyses, key=lambda x: x['avg_time_ms'])
    print(f"  ⚡ Fastest:      {fastest['name']} ({fastest['avg_time_ms']:.1f}ms avg)")

    # Lowest memory
    lowest_mem = min(sorted_analyses, key=lambda x: x['memory_mb'])
    print(f"  💾 Lowest Mem:   {lowest_mem['name']} ({lowest_mem['memory_mb']:.0f}MB peak)")

    print()
    print("Metrics explained:")
    print("  Recall@k:  % of relevant docs found in top-k results (higher is better)")
    print("  MRR:       Mean Reciprocal Rank - 1/rank of first relevant doc (higher is better)")
    print("  Hit@N:     % of queries with at least 1 relevant doc in top-N (higher is better)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results against ground truth")
    parser.add_argument('results_file', type=Path, help='JSON results from compare_retrievers.py')
    parser.add_argument('--benchmark', type=Path,
                       default=Path('data/indexes/slot_full/benchmark_queries.jsonl'),
                       help='Ground truth benchmark file (JSONL)')
    parser.add_argument('-k', type=int, default=10, help='k for recall@k (default: 10)')

    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from {args.benchmark}...")
    ground_truth = load_ground_truth(args.benchmark)
    print(f"  Loaded {len(ground_truth)} queries with ground truth")
    print()

    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file) as f:
        all_results = json.load(f)
    print(f"  Loaded results for {len(all_results)} retrievers")
    print()

    # Analyze each retriever
    analyses = []
    for results in all_results:
        analysis = analyze_retriever(results, ground_truth, args.k)
        analyses.append(analysis)

    # Print analysis
    print_analysis(analyses, args.k)


if __name__ == '__main__':
    main()
