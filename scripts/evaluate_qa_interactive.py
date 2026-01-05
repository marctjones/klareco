#!/usr/bin/env python3
"""
Interactive Q&A evaluation using Claude Code as the evaluator.

This script writes evaluation tasks to a file and waits for Claude Code
to evaluate them and write the results back.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

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


def create_evaluation_batch(
    retriever,
    retriever_name: str,
    questions: List[Dict],
    top_k: int = 10,
    prefilter_n: int = 500,
    rerank_n: int = 100,
    output_dir: Path = Path("/tmp/qa_eval")
) -> Path:
    """
    Run retrieval and create evaluation batch file for Claude Code to evaluate.

    Returns path to the batch file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_file = output_dir / "eval_batch.jsonl"

    logger.info(f"Running retrieval for {len(questions)} questions...")

    batch_data = []

    for i, qa in enumerate(questions, 1):
        question = qa['question']
        gold_answer = qa.get('gold_answer', '')
        acceptable_answers = qa.get('acceptable_answers', [])

        # Run retrieval
        logger.info(f"  [{i}/{len(questions)}] {question[:60]}...")

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

        # Get retrieved documents
        retrieved_docs = [
            {
                'rank': i+1,
                'text': doc_data['text'][:500],  # Limit to 500 chars
                'score': float(score)
            }
            for i, (score, doc_data) in enumerate(search_results[:10])
        ]

        # Add to batch
        batch_data.append({
            'id': qa['id'],
            'question': question,
            'gold_answer': gold_answer,
            'acceptable_answers': acceptable_answers,
            'retrieved_docs': retrieved_docs
        })

    # Write batch file
    with open(batch_file, 'w') as f:
        for item in batch_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"\n✓ Created evaluation batch: {batch_file}")
    logger.info(f"  {len(batch_data)} questions ready for evaluation")

    return batch_file


def wait_for_results(batch_file: Path, output_dir: Path, timeout: int = 3600) -> Path:
    """Wait for Claude Code to write evaluation results."""

    results_file = output_dir / "eval_results.jsonl"

    print("\n" + "=" * 80)
    print("WAITING FOR CLAUDE CODE TO EVALUATE")
    print("=" * 80)
    print(f"\nBatch file created: {batch_file}")
    print(f"Expected results file: {results_file}")
    print("\nPlease ask Claude Code to:")
    print(f"  1. Read the batch file: {batch_file}")
    print(f"  2. Evaluate each question (which doc answers it best)")
    print(f"  3. Write results to: {results_file}")
    print("\nWaiting for results...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        if results_file.exists():
            # Check if file is complete (has same number of lines as batch)
            try:
                with open(batch_file) as f:
                    num_questions = len(f.readlines())
                with open(results_file) as f:
                    num_results = len(f.readlines())

                if num_results >= num_questions:
                    logger.info(f"\n✓ Results file complete: {results_file}")
                    return results_file
                else:
                    logger.info(f"  Results file exists but incomplete ({num_results}/{num_questions})")
            except Exception as e:
                logger.debug(f"Error checking results: {e}")

        time.sleep(2)

    raise TimeoutError(f"No results received after {timeout} seconds")


def compute_metrics(results_file: Path) -> Dict:
    """Compute metrics from evaluation results."""

    results = {
        'perfect_in_top_1': 0,
        'perfect_in_top_5': 0,
        'perfect_in_top_10': 0,
        'good_in_top_1': 0,
        'good_in_top_5': 0,
        'good_in_top_10': 0,
        'any_answer_in_top_10': 0,
        'total_questions': 0,
        'questions': []
    }

    with open(results_file) as f:
        for line in f:
            data = json.loads(line.strip())
            results['total_questions'] += 1

            best_rank = data.get('best_rank')
            quality = data.get('quality', 'none')

            if best_rank:
                results['any_answer_in_top_10'] += 1

                if quality == 'perfect':
                    if best_rank == 1:
                        results['perfect_in_top_1'] += 1
                    if best_rank <= 5:
                        results['perfect_in_top_5'] += 1
                    if best_rank <= 10:
                        results['perfect_in_top_10'] += 1

                if quality in ['perfect', 'good']:
                    if best_rank == 1:
                        results['good_in_top_1'] += 1
                    if best_rank <= 5:
                        results['good_in_top_5'] += 1
                    if best_rank <= 10:
                        results['good_in_top_10'] += 1

            results['questions'].append(data)

    return results


def print_results(results: Dict, retriever_name: str):
    """Print formatted results."""
    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)
    print()

    total = results['total_questions']

    print(f"Retriever: {retriever_name}")
    print(f"Total Questions: {total}")
    print()

    print("Perfect Answers (directly answers the question):")
    print(f"  Top-1:  {results['perfect_in_top_1']}/{total} ({results['perfect_in_top_1']/total*100:.1f}%)")
    print(f"  Top-5:  {results['perfect_in_top_5']}/{total} ({results['perfect_in_top_5']/total*100:.1f}%)")
    print(f"  Top-10: {results['perfect_in_top_10']}/{total} ({results['perfect_in_top_10']/total*100:.1f}%)")
    print()

    print("Good Answers (perfect or good):")
    print(f"  Top-1:  {results['good_in_top_1']}/{total} ({results['good_in_top_1']/total*100:.1f}%)")
    print(f"  Top-5:  {results['good_in_top_5']}/{total} ({results['good_in_top_5']/total*100:.1f}%)")
    print(f"  Top-10: {results['good_in_top_10']}/{total} ({results['good_in_top_10']/total*100:.1f}%)")
    print()

    print(f"Any Answer in Top-10: {results['any_answer_in_top_10']}/{total} ({results['any_answer_in_top_10']/total*100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive Q&A evaluation with Claude Code")
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
        help='Path to Q&A benchmark file'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        choices=['mmap', 'multifaiss', 'hybrid', 'hnsw', 'scann'],
        help='Retriever to evaluate'
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
        '--limit',
        type=int,
        help='Limit to first N questions (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/tmp/qa_eval'),
        help='Directory for batch and results files (default: /tmp/qa_eval)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save final results to JSON file'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Create batch file and exit (don\'t wait for results)'
    )

    args = parser.parse_args()

    # Load Q&A benchmark
    logger.info(f"Loading Q&A benchmark from {args.benchmark}...")
    questions = load_qa_benchmark(args.benchmark)

    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"  Limited to first {args.limit} questions")

    logger.info(f"  Loaded {len(questions)} questions")
    print()

    # Load indexer
    logger.info("Loading embedding models...")
    indexer = SlotBasedIndexer(
        root_model_path=Path("models/root_embeddings/best_model.pt"),
        affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
        output_dir=args.index
    )
    logger.info("  ✓ Models loaded")
    print()

    # Load retriever
    RETRIEVERS = {
        'mmap': ('MemoryMapped', MemoryMappedSlotRetriever),
        'multifaiss': ('MultiFAISS', MultiFAISSSlotRetriever),
        'hybrid': ('Hybrid', HybridFAISSMmapRetriever),
        'hnsw': ('HNSW', HNSWSlotRetriever),
        'scann': ('ScaNN', ScaNNSlotRetriever),
    }

    name, retriever_class = RETRIEVERS[args.retriever]

    logger.info(f"Loading {name} retriever...")
    retriever = retriever_class(args.index, indexer)
    logger.info(f"  ✓ {name} loaded")
    print()

    # Create evaluation batch
    batch_file = create_evaluation_batch(
        retriever,
        name,
        questions,
        top_k=args.top_k,
        prefilter_n=args.prefilter_n,
        rerank_n=args.rerank_n,
        output_dir=args.output_dir
    )

    if args.no_wait:
        print(f"\nBatch file created: {batch_file}")
        print("Exiting without waiting for results (use --no-wait flag to wait)")
        sys.exit(0)

    # Wait for Claude Code to evaluate
    try:
        results_file = wait_for_results(batch_file, args.output_dir)
    except TimeoutError as e:
        logger.error(str(e))
        sys.exit(1)

    # Compute metrics
    logger.info("Computing metrics...")
    results = compute_metrics(results_file)

    # Print results
    print_results(results, name)

    # Save to JSON if requested
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Results saved")


if __name__ == '__main__':
    main()
