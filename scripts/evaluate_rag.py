"""
RAG Retrieval Quality Evaluation Script

Evaluates the quality of the RAG retriever across various dimensions:
- Retrieval precision and recall (if ground truth available)
- Semantic coherence of results
- Query response time
- Diversity of retrieved results

Usage:
    python scripts/evaluate_rag.py --index data/corpus_index --model models/tree_lstm/checkpoint_epoch_12.pt
    python scripts/evaluate_rag.py --queries data/eval_queries.jsonl --output evaluation_results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import statistics

from klareco.rag.retriever import KlarecoRetriever
from klareco.parser import parse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG retrieval quality across multiple dimensions.
    """

    def __init__(
        self,
        retriever: KlarecoRetriever,
        queries: List[Dict[str, Any]]
    ):
        """
        Initialize evaluator.

        Args:
            retriever: Initialized KlarecoRetriever
            queries: List of query dicts with 'text' and optionally 'expected_results'
        """
        self.retriever = retriever
        self.queries = queries
        self.results = []

    def evaluate_all(self, k: int = 10) -> Dict[str, Any]:
        """
        Run full evaluation suite.

        Args:
            k: Number of results to retrieve per query

        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating {len(self.queries)} queries with k={k}")

        # Run retrieval for all queries
        for i, query in enumerate(self.queries):
            logger.info(f"Query {i+1}/{len(self.queries)}: {query['text'][:50]}...")
            result = self._evaluate_query(query, k=k)
            self.results.append(result)

        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics()

        logger.info("Evaluation complete")
        return metrics

    def _evaluate_query(self, query: Dict[str, Any], k: int) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            query: Query dictionary
            k: Number of results to retrieve

        Returns:
            Query result with metrics
        """
        query_text = query['text']

        # Measure retrieval time
        start_time = time.time()
        try:
            results = self.retriever.retrieve(query_text, k=k, return_scores=True)
            retrieval_time = time.time() - start_time
            success = True
            error = None
        except Exception as e:
            retrieval_time = time.time() - start_time
            results = []
            success = False
            error = str(e)
            logger.error(f"Query failed: {error}")

        # Compute metrics
        result = {
            'query': query_text,
            'success': success,
            'error': error,
            'retrieval_time_ms': retrieval_time * 1000,
            'num_results': len(results),
            'results': results
        }

        if success and results:
            # Diversity: unique first words in results
            first_words = [r['text'].split()[0] if r['text'] else '' for r in results]
            result['diversity_score'] = len(set(first_words)) / len(first_words) if first_words else 0.0

            # Score statistics
            scores = [r['score'] for r in results if 'score' in r]
            if scores:
                result['score_mean'] = statistics.mean(scores)
                result['score_std'] = statistics.stdev(scores) if len(scores) > 1 else 0.0
                result['score_max'] = max(scores)
                result['score_min'] = min(scores)

            # Precision/recall if ground truth provided
            if 'expected_indices' in query:
                retrieved_indices = {r['index'] for r in results}
                expected_indices = set(query['expected_indices'])

                true_positives = len(retrieved_indices & expected_indices)
                precision = true_positives / len(retrieved_indices) if retrieved_indices else 0.0
                recall = true_positives / len(expected_indices) if expected_indices else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                result['precision'] = precision
                result['recall'] = recall
                result['f1'] = f1

        return result

    def _compute_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all queries.

        Returns:
            Dictionary of aggregate metrics
        """
        successful_results = [r for r in self.results if r['success']]

        if not successful_results:
            logger.warning("No successful queries to aggregate")
            return {
                'total_queries': len(self.queries),
                'successful_queries': 0,
                'success_rate': 0.0
            }

        # Basic stats
        metrics = {
            'total_queries': len(self.queries),
            'successful_queries': len(successful_results),
            'success_rate': len(successful_results) / len(self.queries),
        }

        # Retrieval time stats
        times = [r['retrieval_time_ms'] for r in successful_results]
        metrics['avg_retrieval_time_ms'] = statistics.mean(times)
        metrics['median_retrieval_time_ms'] = statistics.median(times)
        metrics['max_retrieval_time_ms'] = max(times)
        metrics['min_retrieval_time_ms'] = min(times)

        # Diversity stats
        diversities = [r.get('diversity_score', 0.0) for r in successful_results if 'diversity_score' in r]
        if diversities:
            metrics['avg_diversity'] = statistics.mean(diversities)
            metrics['median_diversity'] = statistics.median(diversities)

        # Score stats
        score_means = [r.get('score_mean', 0.0) for r in successful_results if 'score_mean' in r]
        if score_means:
            metrics['avg_score_mean'] = statistics.mean(score_means)
            metrics['avg_score_std'] = statistics.mean([r.get('score_std', 0.0) for r in successful_results if 'score_std' in r])

        # Precision/recall/F1 (if available)
        precisions = [r.get('precision', 0.0) for r in successful_results if 'precision' in r]
        recalls = [r.get('recall', 0.0) for r in successful_results if 'recall' in r]
        f1s = [r.get('f1', 0.0) for r in successful_results if 'f1' in r]

        if precisions:
            metrics['avg_precision'] = statistics.mean(precisions)
            metrics['avg_recall'] = statistics.mean(recalls)
            metrics['avg_f1'] = statistics.mean(f1s)

        return metrics

    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to output file
        """
        output = {
            'metrics': self._compute_aggregate_metrics(),
            'queries': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")


def load_queries_from_file(queries_path: str) -> List[Dict[str, Any]]:
    """
    Load queries from JSONL file.

    Each line should be a JSON object with:
    - 'text': Query text (required)
    - 'expected_indices': List of expected result indices (optional)

    Args:
        queries_path: Path to queries file

    Returns:
        List of query dictionaries
    """
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            queries.append(query)

    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def create_default_queries() -> List[Dict[str, Any]]:
    """
    Create a default set of test queries.

    Returns:
        List of query dictionaries
    """
    return [
        {'text': 'Mi amas hundojn'},
        {'text': 'La kato dormas'},
        {'text': 'Bona tago'},
        {'text': 'Kio estas Esperanto?'},
        {'text': 'Mi volas lerni'},
        {'text': 'La suno brilas'},
        {'text': 'Du plus tri'},
        {'text': 'Ĉu vi parolas Esperanton?'},
        {'text': 'La libro estas interesa'},
        {'text': 'Mi iras al la butiko'},
        {'text': 'Hodiaŭ estas bela tago'},
        {'text': 'La birdo kantas'},
        {'text': 'Mi trinkas kafon'},
        {'text': 'La infanoj ludas'},
        {'text': 'Dankon pro via helpo'},
    ]


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG retrieval quality')
    parser.add_argument(
        '--index-dir',
        default='data/corpus_index',
        help='Directory containing FAISS index and metadata'
    )
    parser.add_argument(
        '--model-path',
        default='models/tree_lstm/checkpoint_epoch_12.pt',
        help='Path to Tree-LSTM checkpoint'
    )
    parser.add_argument(
        '--queries',
        help='Path to queries JSONL file (optional, uses defaults if not provided)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to retrieve per query'
    )
    parser.add_argument(
        '--output',
        default='evaluation_results.json',
        help='Output file for evaluation results'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for model inference'
    )

    args = parser.parse_args()

    # Check if index and model exist
    if not Path(args.index_dir).exists():
        logger.error(f"Index directory not found: {args.index_dir}")
        return 1

    if not Path(args.model_path).exists():
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return 1

    # Initialize retriever
    logger.info("Initializing retriever...")
    retriever = KlarecoRetriever(
        index_dir=args.index_dir,
        model_path=args.model_path,
        mode='tree_lstm',
        device=args.device
    )

    # Load or create queries
    if args.queries:
        queries = load_queries_from_file(args.queries)
    else:
        logger.info("No queries file provided, using default test queries")
        queries = create_default_queries()

    # Run evaluation
    evaluator = RAGEvaluator(retriever, queries)
    metrics = evaluator.evaluate_all(k=args.k)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total queries: {metrics['total_queries']}")
    logger.info(f"Successful queries: {metrics['successful_queries']}")
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"\nAverage retrieval time: {metrics['avg_retrieval_time_ms']:.2f}ms")
    logger.info(f"Median retrieval time: {metrics['median_retrieval_time_ms']:.2f}ms")

    if 'avg_diversity' in metrics:
        logger.info(f"\nAverage diversity: {metrics['avg_diversity']:.3f}")

    if 'avg_score_mean' in metrics:
        logger.info(f"\nAverage score (mean): {metrics['avg_score_mean']:.4f}")
        logger.info(f"Average score (std): {metrics['avg_score_std']:.4f}")

    if 'avg_precision' in metrics:
        logger.info(f"\nAverage precision: {metrics['avg_precision']:.3f}")
        logger.info(f"Average recall: {metrics['avg_recall']:.3f}")
        logger.info(f"Average F1: {metrics['avg_f1']:.3f}")

    logger.info("="*60)

    # Save results
    evaluator.save_results(args.output)

    return 0


if __name__ == '__main__':
    exit(main())
