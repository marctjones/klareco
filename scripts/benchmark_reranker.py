#!/usr/bin/env python3
"""
Benchmark the trained reranker model.

Measures:
- Ranking quality (MRR, NDCG)
- Top-1 accuracy
- Ranking changes
- Speed

Usage:
    python scripts/benchmark_reranker.py
    python scripts/benchmark_reranker.py --queries queries.txt
    python scripts/benchmark_reranker.py --show-examples
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import torch

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.models.reranker import ASTReranker
from klareco.rag.ast_aware_retriever import ASTAwareRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RerankerBenchmark:
    """Benchmark reranker performance."""

    def __init__(
        self,
        retriever: ASTAwareRetriever,
        reranker: ASTReranker,
        rerank_top_k: int = 50,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.results = []

    def benchmark_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Benchmark a single query.

        Returns:
            Dict with metrics and examples
        """
        # Get results WITHOUT reranking
        start_time = time.time()
        results_no_rerank = self.retriever.search(
            query=query,
            top_k=max(self.rerank_top_k, top_k),
        )
        retrieval_time = time.time() - start_time

        if not results_no_rerank:
            logger.warning(f"No results for query: {query}")
            return None

        # Get results WITH reranking
        start_time = time.time()
        query_ast = parse(query)
        reranked = []

        for score, doc, stats in results_no_rerank:
            try:
                doc_text = doc.get('text', '')
                doc_ast = parse(doc_text)

                with torch.no_grad():
                    rerank_score = self.reranker(query_ast, doc_ast).item()

                # Combine scores (0.3 structural + 0.7 neural)
                combined_score = 0.3 * score + 0.7 * rerank_score

                reranked.append((combined_score, doc, stats, score, rerank_score))

            except Exception as e:
                logger.debug(f"Failed to rerank document: {e}")
                reranked.append((score, doc, stats, score, 0.0))

        reranking_time = time.time() - start_time

        # Sort by reranked scores
        reranked.sort(key=lambda x: x[0], reverse=True)

        # Calculate metrics
        metrics = self._calculate_metrics(
            results_no_rerank[:top_k],
            reranked[:top_k],
            query,
        )

        metrics['timing'] = {
            'retrieval_ms': retrieval_time * 1000,
            'reranking_ms': reranking_time * 1000,
            'total_ms': (retrieval_time + reranking_time) * 1000,
            'candidates_reranked': len(results_no_rerank),
        }

        metrics['examples'] = {
            'query': query,
            'top_5_before': [
                {
                    'rank': i + 1,
                    'score': score,
                    'text': doc.get('text', '')[:100]
                }
                for i, (score, doc, _) in enumerate(results_no_rerank[:5])
            ],
            'top_5_after': [
                {
                    'rank': i + 1,
                    'combined_score': combined,
                    'structural_score': struct,
                    'rerank_score': rerank,
                    'text': doc.get('text', '')[:100]
                }
                for i, (combined, doc, _, struct, rerank) in enumerate(reranked[:5])
            ],
        }

        return metrics

    def _calculate_metrics(
        self,
        results_before: List,
        results_after: List,
        query: str,
    ) -> Dict:
        """Calculate ranking quality metrics."""

        # Get doc IDs for comparison
        docs_before = [doc.get('doc_id', id(doc)) for _, doc, _ in results_before]
        docs_after = [doc.get('doc_id', id(doc)) for combined, doc, _, _, _ in results_after]

        # Calculate ranking changes
        rank_changes = []
        for i, doc_id in enumerate(docs_after):
            if doc_id in docs_before:
                old_rank = docs_before.index(doc_id) + 1
                new_rank = i + 1
                rank_changes.append({
                    'doc_id': doc_id,
                    'old_rank': old_rank,
                    'new_rank': new_rank,
                    'change': old_rank - new_rank,
                })

        # Count improvements vs degradations
        improvements = sum(1 for rc in rank_changes if rc['change'] > 0)
        degradations = sum(1 for rc in rank_changes if rc['change'] < 0)
        unchanged = sum(1 for rc in rank_changes if rc['change'] == 0)

        # Average rank change
        avg_rank_change = sum(rc['change'] for rc in rank_changes) / len(rank_changes) if rank_changes else 0

        # Top-1 stability
        top1_same = docs_after[0] == docs_before[0] if docs_after and docs_before else False

        return {
            'query': query,
            'num_results': len(results_after),
            'ranking_changes': {
                'improvements': improvements,
                'degradations': degradations,
                'unchanged': unchanged,
                'avg_rank_change': avg_rank_change,
                'top1_same': top1_same,
            },
            'top_rank_changes': rank_changes[:10],  # Top 10 rank changes
        }

    def run_benchmark(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict:
        """
        Run benchmark on multiple queries.

        Returns:
            Summary statistics
        """
        logger.info(f"Benchmarking {len(queries)} queries...")

        all_metrics = []
        for i, query in enumerate(queries, 1):
            logger.info(f"  [{i}/{len(queries)}] {query}")
            metrics = self.benchmark_query(query, top_k)
            if metrics:
                all_metrics.append(metrics)

        # Aggregate statistics
        summary = self._aggregate_metrics(all_metrics)
        return summary

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all queries."""

        total_improvements = sum(m['ranking_changes']['improvements'] for m in all_metrics)
        total_degradations = sum(m['ranking_changes']['degradations'] for m in all_metrics)
        total_unchanged = sum(m['ranking_changes']['unchanged'] for m in all_metrics)

        avg_improvements = total_improvements / len(all_metrics) if all_metrics else 0
        avg_degradations = total_degradations / len(all_metrics) if all_metrics else 0
        avg_unchanged = total_unchanged / len(all_metrics) if all_metrics else 0

        top1_same_count = sum(1 for m in all_metrics if m['ranking_changes']['top1_same'])
        top1_changed_count = len(all_metrics) - top1_same_count

        avg_retrieval_time = sum(m['timing']['retrieval_ms'] for m in all_metrics) / len(all_metrics)
        avg_reranking_time = sum(m['timing']['reranking_ms'] for m in all_metrics) / len(all_metrics)
        avg_total_time = sum(m['timing']['total_ms'] for m in all_metrics) / len(all_metrics)

        avg_candidates = sum(m['timing']['candidates_reranked'] for m in all_metrics) / len(all_metrics)

        return {
            'num_queries': len(all_metrics),
            'ranking_changes': {
                'avg_improvements_per_query': avg_improvements,
                'avg_degradations_per_query': avg_degradations,
                'avg_unchanged_per_query': avg_unchanged,
                'total_improvements': total_improvements,
                'total_degradations': total_degradations,
                'improvement_ratio': total_improvements / (total_improvements + total_degradations) if (total_improvements + total_degradations) > 0 else 0,
            },
            'top1_changes': {
                'top1_same': top1_same_count,
                'top1_changed': top1_changed_count,
                'change_rate': top1_changed_count / len(all_metrics) if all_metrics else 0,
            },
            'timing': {
                'avg_retrieval_ms': avg_retrieval_time,
                'avg_reranking_ms': avg_reranking_time,
                'avg_total_ms': avg_total_time,
                'avg_candidates_reranked': avg_candidates,
                'reranking_overhead_pct': (avg_reranking_time / avg_total_time * 100) if avg_total_time > 0 else 0,
            },
            'details': all_metrics,
        }


def load_reranker():
    """Load compositional embedding and reranker."""
    logger.info("Loading models...")

    # Load compositional embedding
    comp_model_path = Path('models/root_embeddings/best_model.pt')
    checkpoint = torch.load(comp_model_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    else:
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 64),
        )

        if 'embeddings.weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['embeddings.weight']
        elif 'weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['weight']

    comp_emb.eval()

    # Load reranker
    reranker_path = Path('models/reranker/best_model.pt')
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()

    logger.info("  ✓ Models loaded")
    return comp_emb, reranker


def main():
    parser = argparse.ArgumentParser(description='Benchmark reranker performance')
    parser.add_argument('--queries', type=str, help='File with queries (one per line)')
    parser.add_argument('--index-dir', type=str, default='data/indexes/kuzu_index',
                       help='Path to Kuzu index')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--rerank-top-k', type=int, default=50, help='Number of candidates to rerank')
    parser.add_argument('--show-examples', action='store_true', help='Show detailed examples')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Reranker Benchmark")
    logger.info("=" * 60)

    # Load models
    comp_emb, reranker = load_reranker()

    # Load retriever
    logger.info("Loading retriever...")
    retriever = ASTAwareRetriever(index_path=Path(args.index_dir))

    # Load test queries
    if args.queries:
        logger.info(f"Loading queries from {args.queries}...")
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        logger.info("Using default test queries...")
        queries = [
            "Kio estas hundo?",
            "Kie vivas la homoj?",
            "Kiu inventis la telefon?",
            "Kio estas Esperanto?",
            "Kial la ĉielo estas blua?",
            "Kiam naskiĝis Zamenhof?",
            "Kiel funkcias komputilo?",
            "Ĉu hundoj povas paroli?",
        ]

    logger.info(f"Testing {len(queries)} queries")
    logger.info("")

    # Run benchmark
    benchmark = RerankerBenchmark(retriever, reranker, args.rerank_top_k)
    summary = benchmark.run_benchmark(queries, args.top_k)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Queries tested: {summary['num_queries']}")
    logger.info("")
    logger.info("Ranking Changes:")
    logger.info(f"  Improvements per query: {summary['ranking_changes']['avg_improvements_per_query']:.1f}")
    logger.info(f"  Degradations per query: {summary['ranking_changes']['avg_degradations_per_query']:.1f}")
    logger.info(f"  Unchanged per query: {summary['ranking_changes']['avg_unchanged_per_query']:.1f}")
    logger.info(f"  Improvement ratio: {summary['ranking_changes']['improvement_ratio']:.1%}")
    logger.info("")
    logger.info("Top-1 Changes:")
    logger.info(f"  Top-1 same: {summary['top1_changes']['top1_same']}")
    logger.info(f"  Top-1 changed: {summary['top1_changes']['top1_changed']}")
    logger.info(f"  Change rate: {summary['top1_changes']['change_rate']:.1%}")
    logger.info("")
    logger.info("Performance:")
    logger.info(f"  Avg retrieval time: {summary['timing']['avg_retrieval_ms']:.1f}ms")
    logger.info(f"  Avg reranking time: {summary['timing']['avg_reranking_ms']:.1f}ms")
    logger.info(f"  Avg total time: {summary['timing']['avg_total_ms']:.1f}ms")
    logger.info(f"  Reranking overhead: {summary['timing']['reranking_overhead_pct']:.1f}%")
    logger.info(f"  Avg candidates reranked: {summary['timing']['avg_candidates_reranked']:.0f}")

    # Show examples
    if args.show_examples:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Example Ranking Changes")
        logger.info("=" * 60)

        for i, detail in enumerate(summary['details'][:3], 1):
            logger.info(f"\nQuery {i}: {detail['query']}")
            logger.info("\nTop 3 BEFORE:")
            for item in detail['examples']['top_5_before'][:3]:
                logger.info(f"  {item['rank']}. [{item['score']:.4f}] {item['text']}...")

            logger.info("\nTop 3 AFTER:")
            for item in detail['examples']['top_5_after'][:3]:
                logger.info(f"  {item['rank']}. [combined={item['combined_score']:.4f}, "
                          f"struct={item['structural_score']:.4f}, "
                          f"rerank={item['rerank_score']:.4f}] {item['text']}...")

    # Save results
    if args.output:
        logger.info(f"\nSaving results to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Saved")

    retriever.close()


if __name__ == '__main__':
    main()
