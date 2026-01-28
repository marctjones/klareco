#!/usr/bin/env python3
"""
Benchmark M1 Impact on RAG Pipeline

Compares RAG performance WITH and WITHOUT M1 plausibility filtering:
- Retrieval precision (% of plausible results)
- Filtering effectiveness (% of implausible filtered)
- Answer quality (manual inspection)
- Speed overhead

Usage:
    python scripts/benchmark_m1_impact.py
    python scripts/benchmark_m1_impact.py --queries queries.txt
    python scripts/benchmark_m1_impact.py --show-examples
    python scripts/benchmark_m1_impact.py --output results.json
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.models.m1_inference import M1Inference

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class M1ImpactBenchmark:
    """Benchmark M1 impact on retrieval quality."""

    def __init__(
        self,
        retriever: ASTAwareRetriever,
        m1: M1Inference,
        m1_threshold: float = 0.5,
    ):
        self.retriever = retriever
        self.m1 = m1
        self.threshold = m1_threshold
        self.results = []

    def extract_svo_triple(self, ast: Dict) -> Tuple:
        """Extract subject-verb-object triple from AST."""
        def get_root(node):
            if node is None:
                return None
            if isinstance(node, dict):
                if node.get('tipo') == 'vortgrupo':
                    kerno = node.get('kerno', {})
                    return kerno.get('radiko')
                elif node.get('tipo') == 'vorto':
                    return node.get('radiko')
            return None

        subj = get_root(ast.get('subjekto'))
        verb = get_root(ast.get('verbo'))
        obj = get_root(ast.get('objekto'))

        return (subj, verb, obj)

    def evaluate_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Evaluate a single query with and without M1.

        Returns:
            Dict with comparison metrics
        """
        # Retrieve WITHOUT M1
        start_time = time.time()
        results_no_m1 = self.retriever.search(query, top_k=top_k*2)  # Get more candidates
        retrieval_time = time.time() - start_time

        if not results_no_m1:
            logger.warning(f"No results for query: {query}")
            return None

        # Score all results with M1
        start_time = time.time()
        scored_results = []
        for score, doc, stats in results_no_m1:
            try:
                doc_text = doc.get('text', '')
                doc_ast = parse(doc_text)

                subj, verb, obj = self.extract_svo_triple(doc_ast)

                if subj and verb and obj:
                    m1_score = self.m1.score_triple(subj, verb, obj)
                    plausible = m1_score >= self.threshold
                else:
                    m1_score = 0.5
                    plausible = False

                scored_results.append({
                    'retrieval_score': score,
                    'm1_score': m1_score,
                    'plausible': plausible,
                    'text': doc_text,
                    'triple': (subj, verb, obj) if subj and verb and obj else None,
                    'source': doc.get('source', {}).get('name', 'unknown'),
                })
            except Exception as e:
                logger.debug(f"Failed to score document: {e}")
                scored_results.append({
                    'retrieval_score': score,
                    'm1_score': 0.5,
                    'plausible': False,
                    'text': doc.get('text', ''),
                    'triple': None,
                    'source': doc.get('source', {}).get('name', 'unknown'),
                })

        m1_time = time.time() - start_time

        # Results WITHOUT M1 (top K by retrieval score)
        results_without_m1 = scored_results[:top_k]

        # Results WITH M1 (filter then take top K)
        results_with_m1 = [r for r in scored_results if r['plausible']][:top_k]

        # Calculate metrics
        metrics = {
            'query': query,
            'num_retrieved': len(scored_results),
            'without_m1': self._analyze_results(results_without_m1, "WITHOUT M1"),
            'with_m1': self._analyze_results(results_with_m1, "WITH M1"),
            'filtering': self._analyze_filtering(scored_results),
            'timing': {
                'retrieval_ms': retrieval_time * 1000,
                'm1_scoring_ms': m1_time * 1000,
                'total_ms': (retrieval_time + m1_time) * 1000,
                'm1_overhead_pct': (m1_time / (retrieval_time + m1_time) * 100),
            },
            'examples': {
                'without_m1': results_without_m1[:5],
                'with_m1': results_with_m1[:5],
                'filtered_out': [r for r in scored_results[:top_k] if not r['plausible']][:5],
            }
        }

        return metrics

    def _analyze_results(self, results: List[Dict], label: str) -> Dict:
        """Analyze result quality."""
        if not results:
            return {
                'total': 0,
                'plausible': 0,
                'implausible': 0,
                'precision': 0.0,
                'avg_m1_score': 0.0,
            }

        total = len(results)
        plausible = sum(1 for r in results if r['plausible'])
        implausible = total - plausible
        precision = plausible / total if total > 0 else 0
        avg_m1_score = sum(r['m1_score'] for r in results) / total if total > 0 else 0

        return {
            'total': total,
            'plausible': plausible,
            'implausible': implausible,
            'precision': precision,
            'avg_m1_score': avg_m1_score,
        }

    def _analyze_filtering(self, scored_results: List[Dict]) -> Dict:
        """Analyze M1 filtering effectiveness."""
        total = len(scored_results)
        plausible = sum(1 for r in scored_results if r['plausible'])
        implausible = total - plausible

        # Of the implausible results, how many would have been in top-K?
        top_k_implausible = sum(1 for r in scored_results[:10] if not r['plausible'])

        return {
            'total_candidates': total,
            'plausible': plausible,
            'implausible': implausible,
            'plausible_rate': plausible / total if total > 0 else 0,
            'implausible_rate': implausible / total if total > 0 else 0,
            'top_10_implausible': top_k_implausible,
        }

    def run_benchmark(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict:
        """
        Run benchmark on multiple queries.

        Returns:
            Aggregated metrics
        """
        logger.info(f"Benchmarking {len(queries)} queries...")

        all_metrics = []
        for i, query in enumerate(queries, 1):
            logger.info(f"  [{i}/{len(queries)}] {query}")
            metrics = self.evaluate_query(query, top_k)
            if metrics:
                all_metrics.append(metrics)
                self.results.append(metrics)

        # Aggregate statistics
        summary = self._aggregate_metrics(all_metrics)
        return summary

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across queries."""
        if not all_metrics:
            return {}

        # Average metrics WITHOUT M1
        avg_precision_without = sum(m['without_m1']['precision'] for m in all_metrics) / len(all_metrics)
        avg_m1_score_without = sum(m['without_m1']['avg_m1_score'] for m in all_metrics) / len(all_metrics)
        total_without = sum(m['without_m1']['total'] for m in all_metrics)
        plausible_without = sum(m['without_m1']['plausible'] for m in all_metrics)

        # Average metrics WITH M1
        avg_precision_with = sum(m['with_m1']['precision'] for m in all_metrics) / len(all_metrics)
        avg_m1_score_with = sum(m['with_m1']['avg_m1_score'] for m in all_metrics) / len(all_metrics)
        total_with = sum(m['with_m1']['total'] for m in all_metrics)
        plausible_with = sum(m['with_m1']['plausible'] for m in all_metrics)

        # Filtering stats
        avg_plausible_rate = sum(m['filtering']['plausible_rate'] for m in all_metrics) / len(all_metrics)
        avg_top10_implausible = sum(m['filtering']['top_10_implausible'] for m in all_metrics) / len(all_metrics)

        # Timing
        avg_retrieval_time = sum(m['timing']['retrieval_ms'] for m in all_metrics) / len(all_metrics)
        avg_m1_time = sum(m['timing']['m1_scoring_ms'] for m in all_metrics) / len(all_metrics)
        avg_total_time = sum(m['timing']['total_ms'] for m in all_metrics) / len(all_metrics)
        avg_overhead = sum(m['timing']['m1_overhead_pct'] for m in all_metrics) / len(all_metrics)

        # Improvement calculation
        precision_improvement = avg_precision_with - avg_precision_without
        precision_improvement_pct = (precision_improvement / avg_precision_without * 100) if avg_precision_without > 0 else 0

        return {
            'num_queries': len(all_metrics),
            'without_m1': {
                'avg_precision': avg_precision_without,
                'avg_m1_score': avg_m1_score_without,
                'total_results': total_without,
                'plausible_results': plausible_without,
            },
            'with_m1': {
                'avg_precision': avg_precision_with,
                'avg_m1_score': avg_m1_score_with,
                'total_results': total_with,
                'plausible_results': plausible_with,
            },
            'improvement': {
                'precision_improvement': precision_improvement,
                'precision_improvement_pct': precision_improvement_pct,
                'm1_score_improvement': avg_m1_score_with - avg_m1_score_without,
            },
            'filtering': {
                'avg_plausible_rate': avg_plausible_rate,
                'avg_implausible_rate': 1 - avg_plausible_rate,
                'avg_top10_implausible_without_m1': avg_top10_implausible,
            },
            'timing': {
                'avg_retrieval_ms': avg_retrieval_time,
                'avg_m1_scoring_ms': avg_m1_time,
                'avg_total_ms': avg_total_time,
                'avg_m1_overhead_pct': avg_overhead,
            },
            'details': all_metrics,
        }


def print_summary(summary: Dict):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("M1 IMPACT BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nQueries tested: {summary['num_queries']}")

    print(f"\n{'Metric':<40} {'WITHOUT M1':<20} {'WITH M1':<20} {'Improvement':<15}")
    print(f"{'-'*40} {'-'*20} {'-'*20} {'-'*15}")

    without = summary['without_m1']
    with_m1 = summary['with_m1']
    improvement = summary['improvement']

    print(f"{'Precision (% plausible)':<40} "
          f"{without['avg_precision']:<20.2%} "
          f"{with_m1['avg_precision']:<20.2%} "
          f"{improvement['precision_improvement_pct']:>+14.1f}%")

    print(f"{'Avg M1 Score':<40} "
          f"{without['avg_m1_score']:<20.3f} "
          f"{with_m1['avg_m1_score']:<20.3f} "
          f"{improvement['m1_score_improvement']:>+14.3f}")

    print(f"{'Total Results':<40} "
          f"{without['total_results']:<20} "
          f"{with_m1['total_results']:<20} "
          f"{with_m1['total_results'] - without['total_results']:>+14}")

    print(f"\nFiltering Statistics:")
    filtering = summary['filtering']
    print(f"  Plausible rate in candidates: {filtering['avg_plausible_rate']:.1%}")
    print(f"  Implausible rate in candidates: {filtering['avg_implausible_rate']:.1%}")
    print(f"  Avg implausible in top-10 without M1: {filtering['avg_top10_implausible_without_m1']:.1f}")

    print(f"\nPerformance:")
    timing = summary['timing']
    print(f"  Avg retrieval time: {timing['avg_retrieval_ms']:.1f}ms")
    print(f"  Avg M1 scoring time: {timing['avg_m1_scoring_ms']:.1f}ms")
    print(f"  Avg total time: {timing['avg_total_ms']:.1f}ms")
    print(f"  M1 overhead: {timing['avg_m1_overhead_pct']:.1f}%")


def print_examples(summary: Dict):
    """Print example comparisons."""
    print("\n" + "=" * 80)
    print("EXAMPLE COMPARISONS")
    print("=" * 80)

    for i, detail in enumerate(summary['details'][:3], 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {detail['query']}")
        print(f"{'='*80}")

        print("\n❌ WITHOUT M1 (Top 5 by retrieval score):")
        for j, result in enumerate(detail['examples']['without_m1'], 1):
            status = "✓" if result['plausible'] else "✗"
            print(f"  {j}. {status} [M1: {result['m1_score']:.3f}] {result['text'][:70]}...")

        print("\n✅ WITH M1 (Top 5 plausible):")
        for j, result in enumerate(detail['examples']['with_m1'], 1):
            print(f"  {j}. ✓ [M1: {result['m1_score']:.3f}] {result['text'][:70]}...")

        if detail['examples']['filtered_out']:
            print("\n🗑️  FILTERED OUT by M1:")
            for j, result in enumerate(detail['examples']['filtered_out'], 1):
                triple_str = str(result['triple']) if result['triple'] else "N/A"
                print(f"  {j}. ✗ [M1: {result['m1_score']:.3f}] {result['text'][:60]}...")
                print(f"      Triple: {triple_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark M1 impact on RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/indexes/kuzu_index',
        help='Path to Kuzu index'
    )
    parser.add_argument(
        '--m1-model',
        type=str,
        default='models/m1_semantic_tier_priority/best_model.pt',
        help='Path to M1 model'
    )
    parser.add_argument(
        '--stage1-model',
        type=str,
        default='models/root_embeddings_tier0/best_model.pt',
        help='Path to Stage 1 embeddings'
    )
    parser.add_argument(
        '--queries',
        type=str,
        help='File with queries (one per line)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return'
    )
    parser.add_argument(
        '--m1-threshold',
        type=float,
        default=0.5,
        help='M1 plausibility threshold'
    )
    parser.add_argument(
        '--show-examples',
        action='store_true',
        help='Show detailed examples'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("M1 Impact Benchmark")
    logger.info("=" * 80)

    # Load retriever
    logger.info("Loading retriever...")
    index_path = Path(args.index_dir)
    retriever = ASTAwareRetriever(index_path=index_path)
    logger.info("  ✓ Retriever loaded")

    # Load M1
    logger.info("Loading M1 model...")
    m1 = M1Inference(
        model_path=Path(args.m1_model),
        stage1_path=Path(args.stage1_model),
        device='cpu'
    )
    logger.info("  ✓ M1 loaded")

    # Load queries
    if args.queries:
        logger.info(f"Loading queries from {args.queries}...")
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        logger.info("Using default test queries...")
        queries = [
            "Kiu fondis Esperanton?",
            "Kio estas Esperanto?",
            "Kie naskiĝis Zamenhof?",
            "Kio manĝas insektojn?",
            "Kiu skribas librojn?",
            "Kion manĝas hundoj?",
            "Kiu instruas studentojn?",
            "Kio kreskas en ĝardeno?",
        ]

    logger.info(f"Testing {len(queries)} queries")
    logger.info("")

    # Run benchmark
    benchmark = M1ImpactBenchmark(retriever, m1, args.m1_threshold)
    summary = benchmark.run_benchmark(queries, args.top_k)

    # Print results
    print_summary(summary)

    if args.show_examples:
        print_examples(summary)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Results saved to: {output_path}")

    retriever.close()
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
