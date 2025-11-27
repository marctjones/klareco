#!/usr/bin/env python3
"""
Benchmark structural vs neural retrieval to demonstrate efficiency gains.

Compares three modes:
1. Structural-only: Filter by slot overlap, no neural reranking
2. Hybrid (two-stage): Structural filter → neural rerank
3. Neural-only: Full FAISS search (current baseline)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from klareco.rag.retriever import create_retriever
from klareco.parser import parse
from klareco.structural_index import rank_candidates_by_slot_overlap
from klareco.canonicalizer import canonicalize_sentence


def benchmark_query(
    retriever,
    query: str,
    k: int = 10,
    structural_candidates: int = 500,
) -> Dict[str, Any]:
    """
    Benchmark a single query across all three modes.

    Returns:
        Dict with timing and result info for each mode
    """
    results = {}

    # Parse query once
    ast = parse(query)
    query_slots = canonicalize_sentence(ast)
    query_slot_roots = {role: slot.root for role, slot in query_slots.items() if slot and slot.root}

    # Mode 1: Structural-only (no neural)
    start = time.time()
    structural_indices = rank_candidates_by_slot_overlap(
        query_slot_roots,
        retriever.metadata,
        limit=k,
    )
    structural_time = time.time() - start

    results['structural_only'] = {
        'time_ms': structural_time * 1000,
        'num_results': len(structural_indices),
        'indices': structural_indices[:k],
    }

    # Mode 2: Hybrid (structural filter → neural rerank)
    start = time.time()
    hybrid_results = retriever.retrieve_from_ast(
        ast,
        k=k,
        return_scores=True,
        structural_candidates=structural_candidates,
    )
    hybrid_time = time.time() - start

    results['hybrid'] = {
        'time_ms': hybrid_time * 1000,
        'num_results': len(hybrid_results),
        'top_scores': [r['score'] for r in hybrid_results[:3]],
    }

    # Mode 3: Neural-only (force full search by setting structural_candidates=0)
    # We'll do this manually by encoding and searching FAISS directly
    start = time.time()
    query_embedding = retriever._encode_ast(ast).reshape(1, -1).astype('float32')
    scores, indices = retriever.index.search(query_embedding, k)
    neural_time = time.time() - start

    results['neural_only'] = {
        'time_ms': neural_time * 1000,
        'num_results': len(indices[0]),
        'top_scores': scores[0][:3].tolist(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark structural vs neural retrieval")
    parser.add_argument("--index-dir", default="data/corpus_index_v2", help="Index directory")
    parser.add_argument("--model", default="models/tree_lstm/best_model.pt", help="Model checkpoint")
    parser.add_argument("--queries", type=int, default=20, help="Number of test queries")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--output", default="benchmark_structural_results.json", help="Output file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Load retriever
    logging.info(f"Loading retriever from {args.index_dir}...")
    retriever = create_retriever(
        index_dir=args.index_dir,
        model_path=args.model,
    )
    logging.info(f"Loaded {len(retriever.metadata)} sentences\n")

    # Test queries
    test_queries = [
        "Kio estas hobito?",
        "Kiu estas Gandalf?",
        "Kie estas la ringo?",
        "Kial Frodo iras al Mordoro?",
        "Kiam komenciĝas la aventuro?",
        "Kiel Bilbo trovis la ringon?",
        "Kiom da hobitoj ekiras?",
        "Kies estas la ringo?",
        "La hobito vidas la oron.",
        "Gandalf helpas la hobitojn.",
        "La ringo havas povon.",
        "Frodo portas la ringon.",
        "La montaro estas dang era.",
        "Bilbo revenas hejmen.",
        "La drako gardas la trezoron.",
        "Thorin estas reĝo.",
        "La elfoj kantas.",
        "La hobito dormas.",
        "Gandalf scias la veron.",
        "La grupo atingas la monton.",
    ]

    queries = test_queries[:args.queries]

    # Benchmark each query
    all_results = []
    structural_times = []
    hybrid_times = []
    neural_times = []

    logging.info(f"Benchmarking {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        logging.info(f"[{i}/{len(queries)}] {query}")

        try:
            result = benchmark_query(retriever, query, k=args.k)
            all_results.append({
                'query': query,
                **result
            })

            structural_times.append(result['structural_only']['time_ms'])
            hybrid_times.append(result['hybrid']['time_ms'])
            neural_times.append(result['neural_only']['time_ms'])

            logging.info(f"  Structural: {result['structural_only']['time_ms']:.2f}ms")
            logging.info(f"  Hybrid:     {result['hybrid']['time_ms']:.2f}ms")
            logging.info(f"  Neural:     {result['neural_only']['time_ms']:.2f}ms")
            logging.info("")

        except Exception as e:
            logging.error(f"  Failed: {e}\n")
            continue

    # Compute statistics
    summary = {
        'structural_only': {
            'mean_ms': np.mean(structural_times),
            'median_ms': np.median(structural_times),
            'min_ms': np.min(structural_times),
            'max_ms': np.max(structural_times),
        },
        'hybrid': {
            'mean_ms': np.mean(hybrid_times),
            'median_ms': np.median(hybrid_times),
            'min_ms': np.min(hybrid_times),
            'max_ms': np.max(hybrid_times),
        },
        'neural_only': {
            'mean_ms': np.mean(neural_times),
            'median_ms': np.median(neural_times),
            'min_ms': np.min(neural_times),
            'max_ms': np.max(neural_times),
        },
        'speedup': {
            'hybrid_vs_neural': np.mean(neural_times) / np.mean(hybrid_times),
            'structural_vs_neural': np.mean(neural_times) / np.mean(structural_times),
            'structural_vs_hybrid': np.mean(hybrid_times) / np.mean(structural_times),
        }
    }

    # Print summary
    logging.info("="*60)
    logging.info("BENCHMARK SUMMARY")
    logging.info("="*60)
    logging.info(f"\nStructural-only (deterministic, no neural):")
    logging.info(f"  Mean: {summary['structural_only']['mean_ms']:.2f}ms")
    logging.info(f"  Median: {summary['structural_only']['median_ms']:.2f}ms")

    logging.info(f"\nHybrid (structural filter + neural rerank):")
    logging.info(f"  Mean: {summary['hybrid']['mean_ms']:.2f}ms")
    logging.info(f"  Median: {summary['hybrid']['median_ms']:.2f}ms")

    logging.info(f"\nNeural-only (full FAISS search):")
    logging.info(f"  Mean: {summary['neural_only']['mean_ms']:.2f}ms")
    logging.info(f"  Median: {summary['neural_only']['median_ms']:.2f}ms")

    logging.info(f"\nSpeedup factors:")
    logging.info(f"  Hybrid vs Neural: {summary['speedup']['hybrid_vs_neural']:.2f}x faster")
    logging.info(f"  Structural vs Neural: {summary['speedup']['structural_vs_neural']:.2f}x faster")
    logging.info(f"  Structural vs Hybrid: {summary['speedup']['structural_vs_hybrid']:.2f}x faster")

    # Save results
    output = {
        'summary': summary,
        'queries': all_results,
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logging.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
