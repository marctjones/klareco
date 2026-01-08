#!/usr/bin/env python3
"""
Tiered Retrieval Benchmark Evaluation Script

Evaluates ASTAwareRetriever against a tiered benchmark designed to test
progressively harder retrieval tasks.

Tiers:
  1. Direct Root Match - simplest, should work now
  2. Multi-Root Conjunction - all roots must appear
  3. Synonym Expansion - requires synonym graph
  4. Role-Aware Matching - roots in specific grammatical roles
  5. Cross-Document Inference - advanced, future work

Usage:
    python scripts/evaluate_retrieval.py                    # Run all tiers
    python scripts/evaluate_retrieval.py --tier 1          # Run only Tier 1
    python scripts/evaluate_retrieval.py --tier 1 2        # Run Tiers 1 and 2
    python scripts/evaluate_retrieval.py --verbose         # Show per-query results
    python scripts/evaluate_retrieval.py --output results.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class QueryResult:
    """Result of a single retrieval query."""
    query_id: str
    query: str
    tier: str
    expected_doc_ids: List[int]
    retrieved_doc_ids: List[int]
    retrieved_texts: List[str]

    # Metrics
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    first_relevant_rank: Optional[int] = None

    # Diagnostics
    latency_ms: float = 0.0
    roots_found: List[str] = field(default_factory=list)
    roots_not_found: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TierResult:
    """Aggregated results for a tier."""
    tier_name: str
    description: str
    expected_threshold: float
    query_count: int = 0

    # Aggregated metrics
    avg_recall_at_5: float = 0.0
    avg_recall_at_10: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_precision_at_10: float = 0.0
    avg_mrr: float = 0.0
    avg_latency_ms: float = 0.0

    # Pass/fail
    pass_rate: float = 0.0
    passed: bool = False

    query_results: List[QueryResult] = field(default_factory=list)


def compute_recall_at_k(retrieved: List[int], expected: Set[int], k: int) -> float:
    """Compute recall@K: fraction of expected docs found in top K."""
    if not expected:
        return 1.0  # If no expected docs, consider it a pass

    retrieved_set = set(retrieved[:k])
    found = retrieved_set & expected
    return len(found) / len(expected)


def compute_precision_at_k(retrieved: List[int], expected: Set[int], k: int) -> float:
    """Compute precision@K: fraction of top K that are relevant."""
    if k == 0:
        return 0.0

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in expected)
    return relevant_count / len(retrieved_k)


def compute_mrr(retrieved: List[int], expected: Set[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def find_first_relevant_rank(retrieved: List[int], expected: Set[int]) -> Optional[int]:
    """Find rank of first relevant document (1-indexed)."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in expected:
            return i + 1
    return None


def load_benchmark(benchmark_path: Path) -> Dict[str, Any]:
    """Load the retrieval benchmark."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_query(
    query_data: Dict[str, Any],
    tier_name: str,
    retriever,
    top_k: int = 10,
) -> QueryResult:
    """Evaluate a single retrieval query."""
    query_id = query_data['id']
    query = query_data['query']
    expected_ids = set(query_data.get('expected_doc_ids', []))

    # Run retrieval
    start_time = time.time()
    try:
        results = retriever.search(query, top_k=top_k)
        latency_ms = (time.time() - start_time) * 1000

        # Extract doc IDs and texts
        retrieved_ids = []
        retrieved_texts = []
        for score, doc, stats in results:
            doc_id = doc.get('source', {}).get('doc_id')
            if doc_id is None:
                # Try to get from metadata
                doc_id = doc.get('doc_id', -1)
            retrieved_ids.append(doc_id)
            retrieved_texts.append(doc.get('text', '')[:100])

        # Get stats from last result if available
        roots_found = []
        roots_not_found = []
        if results:
            _, _, stats = results[0]
            if stats:
                roots_found = getattr(stats, 'roots_found_in_index', [])
                roots_not_found = getattr(stats, 'roots_not_found', [])

    except Exception as e:
        return QueryResult(
            query_id=query_id,
            query=query,
            tier=tier_name,
            expected_doc_ids=list(expected_ids),
            retrieved_doc_ids=[],
            retrieved_texts=[],
            error=str(e),
        )

    # Compute metrics
    result = QueryResult(
        query_id=query_id,
        query=query,
        tier=tier_name,
        expected_doc_ids=list(expected_ids),
        retrieved_doc_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        recall_at_5=compute_recall_at_k(retrieved_ids, expected_ids, 5),
        recall_at_10=compute_recall_at_k(retrieved_ids, expected_ids, 10),
        precision_at_5=compute_precision_at_k(retrieved_ids, expected_ids, 5),
        precision_at_10=compute_precision_at_k(retrieved_ids, expected_ids, 10),
        mrr=compute_mrr(retrieved_ids, expected_ids),
        first_relevant_rank=find_first_relevant_rank(retrieved_ids, expected_ids),
        latency_ms=latency_ms,
        roots_found=roots_found,
        roots_not_found=roots_not_found,
    )

    return result


def evaluate_tier(
    tier_name: str,
    tier_data: Dict[str, Any],
    retriever,
    verbose: bool = False,
) -> TierResult:
    """Evaluate all queries in a tier."""
    description = tier_data.get('description', '')
    expected_threshold = tier_data.get('expected_recall_at_10', 0.5)
    queries = tier_data.get('queries', [])

    tier_result = TierResult(
        tier_name=tier_name,
        description=description,
        expected_threshold=expected_threshold,
        query_count=len(queries),
    )

    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_precision_5 = 0.0
    total_precision_10 = 0.0
    total_mrr = 0.0
    total_latency = 0.0
    passes = 0

    for i, query_data in enumerate(queries):
        if verbose:
            print(f"  [{i+1}/{len(queries)}] {query_data['id']}: {query_data['query'][:50]}...")

        result = evaluate_query(query_data, tier_name, retriever)
        tier_result.query_results.append(result)

        if result.error:
            if verbose:
                print(f"    ERROR: {result.error}")
            continue

        total_recall_5 += result.recall_at_5
        total_recall_10 += result.recall_at_10
        total_precision_5 += result.precision_at_5
        total_precision_10 += result.precision_at_10
        total_mrr += result.mrr
        total_latency += result.latency_ms

        if result.recall_at_10 >= expected_threshold:
            passes += 1

        if verbose:
            status = "✓" if result.recall_at_10 >= expected_threshold else "✗"
            print(f"    {status} R@10={result.recall_at_10:.2f} P@10={result.precision_at_10:.2f} "
                  f"MRR={result.mrr:.2f} ({result.latency_ms:.0f}ms)")
            if result.first_relevant_rank:
                print(f"      First relevant at rank {result.first_relevant_rank}")

    n = len(queries)
    if n > 0:
        tier_result.avg_recall_at_5 = total_recall_5 / n
        tier_result.avg_recall_at_10 = total_recall_10 / n
        tier_result.avg_precision_at_5 = total_precision_5 / n
        tier_result.avg_precision_at_10 = total_precision_10 / n
        tier_result.avg_mrr = total_mrr / n
        tier_result.avg_latency_ms = total_latency / n
        tier_result.pass_rate = passes / n
        tier_result.passed = tier_result.avg_recall_at_10 >= expected_threshold

    return tier_result


def print_results(tier_results: List[TierResult], show_details: bool = False):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("TIERED RETRIEVAL BENCHMARK RESULTS")
    print("=" * 70)

    # Summary table
    print(f"\n{'Tier':<25} {'Queries':>8} {'R@10':>8} {'P@10':>8} {'MRR':>8} {'Pass%':>8} {'Status':>8}")
    print("-" * 70)

    for tier in tier_results:
        status = "✓ PASS" if tier.passed else "✗ FAIL"
        print(f"{tier.tier_name:<25} {tier.query_count:>8} "
              f"{tier.avg_recall_at_10:>7.1%} {tier.avg_precision_at_10:>7.1%} "
              f"{tier.avg_mrr:>7.2f} {tier.pass_rate:>7.1%} {status:>8}")

    print("-" * 70)

    # Overall stats
    total_queries = sum(t.query_count for t in tier_results)
    total_passed = sum(1 for t in tier_results if t.passed)
    avg_latency = sum(t.avg_latency_ms * t.query_count for t in tier_results) / total_queries if total_queries > 0 else 0

    print(f"\nTotal Queries: {total_queries}")
    print(f"Tiers Passed: {total_passed}/{len(tier_results)}")
    print(f"Avg Latency: {avg_latency:.0f}ms")

    # Show failing queries if requested
    if show_details:
        print("\n" + "=" * 70)
        print("FAILING QUERIES (R@10 < threshold)")
        print("=" * 70)

        for tier in tier_results:
            failing = [q for q in tier.query_results
                      if q.recall_at_10 < tier.expected_threshold and not q.error]
            if failing:
                print(f"\n{tier.tier_name}:")
                for q in failing[:5]:
                    print(f"\n  {q.query_id}: {q.query}")
                    print(f"    Expected: {q.expected_doc_ids[:5]}...")
                    print(f"    Got: {q.retrieved_doc_ids[:5]}...")
                    print(f"    R@10={q.recall_at_10:.2f}, First relevant: {q.first_relevant_rank}")


def save_results(tier_results: List[TierResult], output_path: Path):
    """Save results to JSON."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_queries": sum(t.query_count for t in tier_results),
            "tiers_passed": sum(1 for t in tier_results if t.passed),
            "tiers_total": len(tier_results),
        },
        "tiers": {}
    }

    for tier in tier_results:
        output["tiers"][tier.tier_name] = {
            "description": tier.description,
            "query_count": tier.query_count,
            "expected_threshold": tier.expected_threshold,
            "avg_recall_at_10": tier.avg_recall_at_10,
            "avg_precision_at_10": tier.avg_precision_at_10,
            "avg_mrr": tier.avg_mrr,
            "avg_latency_ms": tier.avg_latency_ms,
            "pass_rate": tier.pass_rate,
            "passed": tier.passed,
            "queries": [
                {
                    "id": q.query_id,
                    "query": q.query,
                    "recall_at_10": q.recall_at_10,
                    "precision_at_10": q.precision_at_10,
                    "mrr": q.mrr,
                    "first_relevant_rank": q.first_relevant_rank,
                    "latency_ms": q.latency_ms,
                    "expected_doc_ids": q.expected_doc_ids,
                    "retrieved_doc_ids": q.retrieved_doc_ids[:10],
                    "error": q.error,
                }
                for q in tier.query_results
            ]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate tiered retrieval benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_retrieval.py                    # Run all tiers
  python scripts/evaluate_retrieval.py --tier 1          # Tier 1 only
  python scripts/evaluate_retrieval.py --tier 1 2        # Tiers 1 and 2
  python scripts/evaluate_retrieval.py -v                # Verbose output
  python scripts/evaluate_retrieval.py --details         # Show failing queries
        """
    )
    parser.add_argument('--benchmark', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'retrieval_benchmark_v1.json',
                        help='Path to benchmark JSON')
    parser.add_argument('--tier', type=int, nargs='+',
                        help='Specific tier(s) to run (1-5)')
    parser.add_argument('--index', type=Path,
                        default=PROJECT_ROOT / 'data' / 'indexes' / 'kuzu_index',
                        help='Path to Kuzu index')
    parser.add_argument('--output', type=Path,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show progress and per-query results')
    parser.add_argument('--details', action='store_true',
                        help='Show detailed failing queries')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to retrieve')

    args = parser.parse_args()

    # Load benchmark
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    benchmark = load_benchmark(args.benchmark)
    print(f"Loaded benchmark v{benchmark.get('version', '?')}")
    print(f"Corpus: {benchmark.get('corpus_stats', {}).get('total_docs', '?'):,} docs, "
          f"{benchmark.get('corpus_stats', {}).get('total_roots', '?'):,} roots")

    # Check index
    if not (args.index / 'kuzu.db').exists():
        print(f"Error: Kuzu index not found at {args.index}")
        sys.exit(1)

    # Load retriever
    from klareco.rag.ast_aware_retriever import ASTAwareRetriever
    from klareco.rag.kuzu_inverted_index import FallbackMode

    print(f"\nLoading retriever from {args.index}...")
    retriever = ASTAwareRetriever(
        index_path=args.index,
        fallback_mode=FallbackMode.NONE,
    )
    print("Ready!\n")

    # Determine which tiers to run
    tier_map = {
        1: 'tier1_direct',
        2: 'tier2_conjunction',
        3: 'tier3_synonym',
        4: 'tier4_role',
        5: 'tier5_inference',
    }

    if args.tier:
        tiers_to_run = [tier_map[t] for t in args.tier if t in tier_map]
    else:
        tiers_to_run = list(tier_map.values())

    # Run evaluation
    tier_results = []
    for tier_name in tiers_to_run:
        if tier_name not in benchmark['tiers']:
            print(f"Warning: Tier {tier_name} not in benchmark, skipping")
            continue

        tier_data = benchmark['tiers'][tier_name]
        print(f"\n{'='*60}")
        print(f"Evaluating: {tier_name}")
        print(f"  {tier_data.get('description', '')}")
        print(f"  Queries: {len(tier_data.get('queries', []))}")
        print(f"  Expected R@10: {tier_data.get('expected_recall_at_10', 0.5):.0%}")
        print('='*60)

        result = evaluate_tier(tier_name, tier_data, retriever, verbose=args.verbose)
        tier_results.append(result)

    # Print results
    print_results(tier_results, show_details=args.details)

    # Save results
    if args.output:
        save_results(tier_results, args.output)
    else:
        default_output = PROJECT_ROOT / 'data' / 'benchmarks' / 'retrieval_results.json'
        save_results(tier_results, default_output)

    # Exit with error code if any tier failed
    if not all(t.passed for t in tier_results):
        sys.exit(1)


if __name__ == '__main__':
    main()
