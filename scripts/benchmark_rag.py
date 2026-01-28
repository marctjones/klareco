#!/usr/bin/env python3
"""
Benchmark RAG Pipeline

Evaluates retrieval + reranking quality on a test set.

Metrics:
- Recall@K: % of queries where correct answer is in top K
- MRR (Mean Reciprocal Rank): Average 1/rank of first correct answer
- Precision@K: % of top K results that are relevant

Usage:
    python scripts/benchmark_rag.py                    # Run all tests
    python scripts/benchmark_rag.py --baseline-only   # Just retrieval (no reranker)
    python scripts/benchmark_rag.py --full-only       # Just full pipeline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.models.reranker import ASTReranker
from klareco.embeddings.compositional import CompositionalEmbedding
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Test queries with expected answers
TEST_QUERIES = [
    {
        'query': 'Kiu fondis Esperanton?',
        'expected_keywords': ['zamenhof'],
        'expected_type': 'person',
    },
    {
        'query': 'Kio estas Esperanto?',
        'expected_keywords': ['lingv', 'internaciALL'],
        'expected_type': 'definition',
    },
    {
        'query': 'Kie naskiĝis Zamenhof?',
        'expected_keywords': ['bjalistok', 'pol'],
        'expected_type': 'place',
    },
    {
        'query': 'Kiam estis fondita Esperanto?',
        'expected_keywords': ['1887'],
        'expected_type': 'date',
    },
    {
        'query': 'Kiom da homoj parolas Esperanton?',
        'expected_keywords': ['mil'],
        'expected_type': 'number',
    },
]


def check_answer(text: str, expected_keywords: List[str]) -> bool:
    """Check if text contains expected keywords (case-insensitive)."""
    text_lower = text.lower()
    # At least one keyword must match
    return any(kw.lower() in text_lower for kw in expected_keywords)


def calculate_metrics(results_by_query: Dict[str, List[Tuple[float, Dict]]]) -> Dict:
    """
    Calculate IR metrics.

    Args:
        results_by_query: {query: [(score, doc), ...]}

    Returns:
        Dict with metrics: recall@1, recall@5, recall@10, MRR
    """
    metrics = {
        'recall@1': [],
        'recall@5': [],
        'recall@10': [],
        'mrr': [],
        'total_queries': len(TEST_QUERIES),
    }

    for test_case in TEST_QUERIES:
        query = test_case['query']
        expected = test_case['expected_keywords']

        if query not in results_by_query:
            # No results for this query
            metrics['recall@1'].append(0)
            metrics['recall@5'].append(0)
            metrics['recall@10'].append(0)
            metrics['mrr'].append(0.0)
            continue

        results = results_by_query[query]

        # Find rank of first correct answer
        first_correct_rank = None
        for rank, (score, doc) in enumerate(results, 1):
            text = doc.get('text', '')
            if check_answer(text, expected):
                first_correct_rank = rank
                break

        # Update metrics
        if first_correct_rank is not None:
            metrics['recall@1'].append(1 if first_correct_rank <= 1 else 0)
            metrics['recall@5'].append(1 if first_correct_rank <= 5 else 0)
            metrics['recall@10'].append(1 if first_correct_rank <= 10 else 0)
            metrics['mrr'].append(1.0 / first_correct_rank)
        else:
            # No correct answer found
            metrics['recall@1'].append(0)
            metrics['recall@5'].append(0)
            metrics['recall@10'].append(0)
            metrics['mrr'].append(0.0)

    # Calculate averages
    return {
        'recall@1': sum(metrics['recall@1']) / len(metrics['recall@1']),
        'recall@5': sum(metrics['recall@5']) / len(metrics['recall@5']),
        'recall@10': sum(metrics['recall@10']) / len(metrics['recall@10']),
        'mrr': sum(metrics['mrr']) / len(metrics['mrr']),
        'total_queries': metrics['total_queries'],
    }


def load_reranker() -> ASTReranker:
    """Load reranker model."""
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

    comp_emb.eval()

    reranker_path = Path('models/reranker/best_model.pt')
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()

    return reranker


def benchmark_baseline(retriever: ASTAwareRetriever, top_k: int = 10) -> Dict:
    """Benchmark baseline retrieval (no reranking)."""
    print("\n" + "="*70)
    print("BASELINE: Retrieval Only (No Reranking)")
    print("="*70)

    results_by_query = {}
    total_time = 0.0

    for test_case in TEST_QUERIES:
        query = test_case['query']

        start = time.time()
        results = retriever.search(query, top_k=top_k, use_m1_expansion=False)
        elapsed = time.time() - start
        total_time += elapsed

        # Store results (score, doc)
        results_by_query[query] = [(score, doc) for score, doc, _ in results]

        print(f"\nQuery: {query}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Top result: {results[0][1].get('text', '')[:100]}..." if results else "  No results")

    metrics = calculate_metrics(results_by_query)
    metrics['avg_latency'] = total_time / len(TEST_QUERIES)

    print("\n" + "-"*70)
    print("BASELINE METRICS:")
    print(f"  Recall@1:  {metrics['recall@1']:.1%}")
    print(f"  Recall@5:  {metrics['recall@5']:.1%}")
    print(f"  Recall@10: {metrics['recall@10']:.1%}")
    print(f"  MRR:       {metrics['mrr']:.3f}")
    print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")

    return metrics


def benchmark_with_reranker(retriever: ASTAwareRetriever, reranker: ASTReranker,
                            top_k: int = 10, rerank_top_k: int = 50) -> Dict:
    """Benchmark full pipeline (retrieval + reranking)."""
    print("\n" + "="*70)
    print("FULL PIPELINE: Retrieval + Reranking")
    print("="*70)

    from klareco.parser import parse

    results_by_query = {}
    total_time = 0.0

    for test_case in TEST_QUERIES:
        query = test_case['query']

        start = time.time()

        # Stage 1: Retrieval
        candidates = retriever.search(query, top_k=rerank_top_k, use_m1_expansion=False)

        # Stage 2: Reranking
        query_ast = parse(query)
        reranked = []

        for score, doc, _ in candidates:
            try:
                doc_text = doc.get('text', '')
                doc_ast = parse(doc_text)

                with torch.no_grad():
                    rerank_score = reranker(query_ast, doc_ast).item()

                # Combine: 30% retrieval, 70% reranker
                combined = 0.3 * score + 0.7 * rerank_score
                reranked.append((combined, doc))
            except:
                reranked.append((score, doc))

        reranked.sort(key=lambda x: x[0], reverse=True)
        results_by_query[query] = reranked[:top_k]

        elapsed = time.time() - start
        total_time += elapsed

        print(f"\nQuery: {query}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Top result: {reranked[0][1].get('text', '')[:100]}..." if reranked else "  No results")

    metrics = calculate_metrics(results_by_query)
    metrics['avg_latency'] = total_time / len(TEST_QUERIES)

    print("\n" + "-"*70)
    print("FULL PIPELINE METRICS:")
    print(f"  Recall@1:  {metrics['recall@1']:.1%}")
    print(f"  Recall@5:  {metrics['recall@5']:.1%}")
    print(f"  Recall@10: {metrics['recall@10']:.1%}")
    print(f"  MRR:       {metrics['mrr']:.3f}")
    print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline")
    parser.add_argument('--index-dir', type=str, default='data/indexes/kuzu_index')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--full-only', action='store_true')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    args = parser.parse_args()

    print("="*70)
    print("RAG Pipeline Benchmark")
    print("="*70)
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"Top-K: {args.top_k}")

    # Load retriever
    print("\nLoading retriever...")
    retriever = ASTAwareRetriever(index_path=Path(args.index_dir))

    results = {}

    # Baseline
    if not args.full_only:
        baseline_metrics = benchmark_baseline(retriever, top_k=args.top_k)
        results['baseline'] = baseline_metrics

    # Full pipeline
    if not args.baseline_only:
        print("\nLoading reranker...")
        reranker = load_reranker()

        full_metrics = benchmark_with_reranker(retriever, reranker, top_k=args.top_k)
        results['full_pipeline'] = full_metrics

    # Summary
    if not args.baseline_only and not args.full_only:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        baseline = results['baseline']
        full = results['full_pipeline']

        print(f"Recall@1:  {baseline['recall@1']:.1%} → {full['recall@1']:.1%} "
              f"({full['recall@1'] - baseline['recall@1']:+.1%})")
        print(f"Recall@5:  {baseline['recall@5']:.1%} → {full['recall@5']:.1%} "
              f"({full['recall@5'] - baseline['recall@5']:+.1%})")
        print(f"MRR:       {baseline['mrr']:.3f} → {full['mrr']:.3f} "
              f"({full['mrr'] - baseline['mrr']:+.3f})")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")

    retriever.close()


if __name__ == '__main__':
    main()
