#!/usr/bin/env python3
"""
Benchmark retrievers on Q&A task using qa_benchmark_v1.jsonl.

Tests how well retrievers find documents that contain answers to questions.
This is a more realistic benchmark than exact sentence matching.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_qa_benchmark(benchmark_file: Path) -> List[Dict]:
    """Load Q&A benchmark questions."""
    questions = []
    with open(benchmark_file) as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data)
    return questions


def contains_answer(text: str, acceptable_answers: List[str]) -> bool:
    """Check if text contains any of the acceptable answers."""
    text_lower = text.lower()
    for answer in acceptable_answers:
        # Simple substring match (could be improved with fuzzy matching)
        if answer.lower() in text_lower:
            return True
    return False


def evaluate_retrieval(retriever, retriever_name: str, questions: List[Dict],
                       top_k: int = 10, prefilter_n: int = 500, rerank_n: int = 100) -> Dict:
    """Evaluate a retriever on Q&A benchmark."""
    results = {
        'name': retriever_name,
        'answer_in_top_1': 0,
        'answer_in_top_5': 0,
        'answer_in_top_10': 0,
        'total_questions': len(questions),
        'avg_time_ms': 0,
        'questions': []
    }

    total_time = 0

    for qa in questions:
        question = qa['question']
        acceptable_answers = qa['acceptable_answers']

        # Search
        start = time.time()
        try:
            if 'ScaNN' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    scann_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'HNSW' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    hnsw_top_n=prefilter_n,
                    slot_top_n=rerank_n
                )
            elif 'Hybrid' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    faiss_top_n=prefilter_n
                )
            elif 'MultiFAISS' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    slot_top_n=rerank_n
                )
            else:
                search_results = retriever.search(
                    question,
                    top_k=top_k,
                    rerank_top_n=prefilter_n
                )
        except Exception as e:
            logger.error(f"  ✗ Query failed: {e}")
            continue

        query_time = (time.time() - start) * 1000
        total_time += query_time

        # Check if any result contains the answer
        found_at = None
        for i, (score, doc_data) in enumerate(search_results, 1):
            if contains_answer(doc_data['text'], acceptable_answers):
                found_at = i
                break

        # Update metrics
        if found_at:
            if found_at == 1:
                results['answer_in_top_1'] += 1
            if found_at <= 5:
                results['answer_in_top_5'] += 1
            if found_at <= 10:
                results['answer_in_top_10'] += 1

        # Store per-question results
        results['questions'].append({
            'id': qa['id'],
            'question': question,
            'category': qa['category'],
            'difficulty': qa['difficulty'],
            'found_at_rank': found_at,
            'time_ms': query_time,
            'top_result': search_results[0][1]['text'][:200] if search_results else None
        })

    results['avg_time_ms'] = total_time / len(questions) if questions else 0
    return results


def print_results(all_results: List[Dict]):
    """Print formatted Q&A benchmark results."""
    print("\n" + "=" * 120)
    print("Q&A RETRIEVAL BENCHMARK RESULTS")
    print("=" * 120)
    print()

    header = f"{'Retriever':<15} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Latency':>10} {'Questions':>10}"
    print(header)
    print("-" * 120)

    # Sort by top-10 accuracy
    sorted_results = sorted(all_results, key=lambda x: x['answer_in_top_10'], reverse=True)

    for result in sorted_results:
        total = result['total_questions']
        top1_pct = result['answer_in_top_1'] / total * 100 if total > 0 else 0
        top5_pct = result['answer_in_top_5'] / total * 100 if total > 0 else 0
        top10_pct = result['answer_in_top_10'] / total * 100 if total > 0 else 0

        print(f"{result['name']:<15} "
              f"{top1_pct:>7.1f}% "
              f"{top5_pct:>7.1f}% "
              f"{top10_pct:>7.1f}% "
              f"{result['avg_time_ms']:>8.1f}ms "
              f"{total:>10}")

    print()
    print("Rankings:")

    best_top10 = sorted_results[0]
    print(f"  🎯 Best Accuracy: {best_top10['name']} ({best_top10['answer_in_top_10']}/{best_top10['total_questions']} found in top-10)")

    fastest = min(sorted_results, key=lambda x: x['avg_time_ms'])
    print(f"  ⚡ Fastest:       {fastest['name']} ({fastest['avg_time_ms']:.1f}ms avg)")

    print()
    print("Metrics explained:")
    print("  Top-N:  % of questions where answer was found in top-N retrieved documents")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrievers on Q&A task")
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to slot index directory'
    )
    parser.add_argument(
        '--benchmark',
        type=Path,
        default=Path('data/benchmarks/datasets/qa_benchmark_v1.jsonl'),
        help='Path to Q&A benchmark file (default: qa_benchmark_v1.jsonl)'
    )
    parser.add_argument(
        '--retrievers',
        type=str,
        help='Comma-separated list of retrievers (default: all available). '
             'Options: mmap, multifaiss, hybrid, hnsw, scann'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    parser.add_argument(
        '--prefilter-n',
        type=int,
        default=500,
        help='Pre-filtering candidates (default: 500)'
    )
    parser.add_argument(
        '--rerank-n',
        type=int,
        default=100,
        help='Reranking candidates (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Load Q&A benchmark
    logger.info(f"Loading Q&A benchmark from {args.benchmark}...")
    questions = load_qa_benchmark(args.benchmark)
    logger.info(f"  Loaded {len(questions)} questions")
    print()

    # Load indexer (needed by all retrievers)
    logger.info("Loading embedding models...")
    indexer = SlotBasedIndexer(
        root_model_path=Path("models/root_embeddings/best_model.pt"),
        affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
        output_dir=args.index
    )
    logger.info("  ✓ Models loaded")
    print()

    # Determine which retrievers to test
    RETRIEVERS = {
        'mmap': ('MemoryMapped', MemoryMappedSlotRetriever),
        'multifaiss': ('MultiFAISS', MultiFAISSSlotRetriever),
        'hybrid': ('Hybrid', HybridFAISSMmapRetriever),
        'hnsw': ('HNSW', HNSWSlotRetriever),
        'scann': ('ScaNN', ScaNNSlotRetriever),
    }

    # Check availability
    available = {
        'mmap': (args.index / 'mmap').exists(),
        'multifaiss': (args.index / 'multifaiss').exists(),
        'hybrid': (args.index / 'faiss').exists() and (args.index / 'mmap').exists(),
        'hnsw': (args.index / 'hnsw').exists() and (args.index / 'mmap').exists(),
        'scann': (args.index / 'scann').exists() and (args.index / 'mmap').exists(),
    }

    if args.retrievers:
        requested = [r.strip() for r in args.retrievers.split(',')]
        to_test = [r for r in requested if available.get(r, False)]
    else:
        to_test = [r for r in RETRIEVERS.keys() if available.get(r, False)]

    if not to_test:
        logger.error("No retrievers available to test")
        sys.exit(1)

    logger.info(f"Testing retrievers: {', '.join(to_test)}")
    print()

    # Test each retriever
    all_results = []

    for retriever_key in to_test:
        name, retriever_class = RETRIEVERS[retriever_key]

        logger.info(f"Loading {name} retriever...")
        retriever = retriever_class(args.index, indexer)
        logger.info(f"  ✓ {name} loaded")
        print()

        logger.info(f"Testing {name} on {len(questions)} questions...")
        results = evaluate_retrieval(
            retriever,
            name,
            questions,
            top_k=args.top_k,
            prefilter_n=args.prefilter_n,
            rerank_n=args.rerank_n
        )
        all_results.append(results)
        logger.info(f"  ✓ {name} complete: {results['answer_in_top_10']}/{len(questions)} found")
        print()

    # Print results
    print_results(all_results)

    # Save to JSON if requested
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Results saved")


if __name__ == '__main__':
    main()
