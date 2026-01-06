#!/usr/bin/env python3
"""
Enhanced Q&A benchmark with checkpointing, resource monitoring, and progress tracking.

Features:
- Checkpoint every 10 questions (restartable)
- Memory and CPU tracking per retriever
- Detailed progress logging with ETA
- Full results for Claude Code analysis
- Saves retrieved document texts for manual review
"""

import argparse
import json
import logging
import os
import psutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever
from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever
from klareco.rag.ast_aware_retriever import ASTAwareRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_memory_cpu():
    """Get current memory (MB) and CPU (%)."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = process.cpu_percent(interval=0.1)
    return mem_mb, cpu_pct


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
        if answer.lower() in text_lower:
            return True
    return False


def evaluate_retrieval_enhanced(
    retriever,
    retriever_name: str,
    questions: List[Dict],
    top_k: int = 10,
    prefilter_n: int = 500,
    rerank_n: int = 100,
    checkpoint_file: Optional[Path] = None,
    resume: bool = True
) -> Dict:
    """
    Evaluate a retriever with checkpointing and resource monitoring.
    """
    # Load checkpoint if resuming
    start_idx = 0
    results = {
        'name': retriever_name,
        'answer_in_top_1': 0,
        'answer_in_top_5': 0,
        'answer_in_top_10': 0,
        'total_questions': len(questions),
        'avg_time_ms': 0,
        'peak_memory_mb': 0,
        'avg_cpu_pct': 0,
        'questions': []
    }

    if resume and checkpoint_file and checkpoint_file.exists():
        logger.info(f"  Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('completed', 0)
            results = checkpoint.get('results', results)
        logger.info(f"  Resuming from question {start_idx}/{len(questions)}")

    # Resource tracking
    baseline_mem, _ = get_memory_cpu()
    peak_mem = baseline_mem
    cpu_samples = []
    total_time = 0

    # Progress tracking
    benchmark_start = time.time()
    last_update = benchmark_start

    for i in range(start_idx, len(questions)):
        qa = questions[i]
        question = qa['question']
        acceptable_answers = qa['acceptable_answers']

        # Search with appropriate parameters per retriever
        start = time.time()
        try:
            if 'ASTAware' in retriever.__class__.__name__:
                # AST-aware retriever uses strategy parameter
                search_results = retriever.search(
                    question, top_k=top_k, strategy='auto'
                )
            elif 'ScaNN' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question, top_k=top_k,
                    scann_top_n=prefilter_n, slot_top_n=rerank_n
                )
            elif 'HNSW' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question, top_k=top_k,
                    hnsw_top_n=prefilter_n, slot_top_n=rerank_n
                )
            elif 'Hybrid' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question, top_k=top_k, faiss_top_n=prefilter_n
                )
            elif 'MultiFAISS' in retriever.__class__.__name__:
                search_results = retriever.search(
                    question, top_k=top_k, slot_top_n=rerank_n
                )
            else:
                search_results = retriever.search(
                    question, top_k=top_k, rerank_top_n=prefilter_n
                )
        except Exception as e:
            logger.error(f"  ✗ Query {i+1} failed: {e}")
            search_results = []

        query_time = (time.time() - start) * 1000
        total_time += query_time

        # Track resources
        current_mem, current_cpu = get_memory_cpu()
        peak_mem = max(peak_mem, current_mem)
        cpu_samples.append(current_cpu)

        # Evaluate results
        found_at = None
        retrieved_texts = []

        for rank, (score, doc) in enumerate(search_results, 1):
            doc_text = doc['text']
            retrieved_texts.append(doc_text)

            if found_at is None and contains_answer(doc_text, acceptable_answers):
                found_at = rank

        # Update counters
        if found_at:
            if found_at == 1:
                results['answer_in_top_1'] += 1
            if found_at <= 5:
                results['answer_in_top_5'] += 1
            if found_at <= 10:
                results['answer_in_top_10'] += 1

        # Store detailed results for Claude Code analysis
        results['questions'].append({
            'id': qa['id'],
            'question': question,
            'category': qa.get('category', 'unknown'),
            'gold_answer': qa.get('gold_answer', ''),
            'acceptable_answers': acceptable_answers,
            'found_at_rank': found_at,
            'query_time_ms': query_time,
            'retrieved_docs': retrieved_texts,  # Full texts for Claude analysis
            'top_result': retrieved_texts[0] if retrieved_texts else None
        })

        # Progress reporting
        current_time = time.time()
        is_milestone = (i + 1) % 5 == 0
        is_checkpoint = (i + 1) % 10 == 0
        is_timed = (current_time - last_update) >= 60  # Every minute
        is_final = (i + 1) == len(questions)

        if is_milestone or is_checkpoint or is_timed or is_final:
            # Calculate statistics
            completed = i + 1
            remaining = len(questions) - completed
            elapsed = current_time - benchmark_start
            avg_time_per_q = elapsed / (completed - start_idx) if completed > start_idx else 0
            eta_seconds = avg_time_per_q * remaining if avg_time_per_q > 0 else 0

            # Recent accuracy (last 10 questions)
            recent_results = results['questions'][-10:]
            recent_found = sum(1 for r in recent_results if r['found_at_rank'] is not None)
            recent_accuracy = recent_found / len(recent_results) * 100 if recent_results else 0

            # Format ETA
            if eta_seconds >= 60:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"

            # Build progress message
            progress_parts = [
                f"[{completed}/{len(questions)}]",
                f"Latency: {query_time:.1f}ms",
                f"Accuracy: {recent_accuracy:.0f}%",
                f"Memory: {current_mem:.0f}MB",
                f"CPU: {current_cpu:.0f}%",
                f"ETA: {eta_str}"
            ]

            if is_checkpoint:
                progress_parts.append("💾 checkpoint")
            elif is_timed:
                progress_parts.append("⏰ 1min")

            logger.info("  " + " | ".join(progress_parts))
            last_update = current_time

        # Save checkpoint every 10 questions
        if checkpoint_file and is_checkpoint:
            checkpoint_data = {
                'completed': i + 1,
                'results': results
            }
            temp_checkpoint = checkpoint_file.with_suffix('.tmp')
            with open(temp_checkpoint, 'w') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            temp_checkpoint.rename(checkpoint_file)

    # Final statistics
    results['avg_time_ms'] = total_time / len(questions) if questions else 0
    results['peak_memory_mb'] = peak_mem
    results['memory_delta_mb'] = peak_mem - baseline_mem
    results['avg_cpu_pct'] = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

    return results


def print_results(all_results: List[Dict]):
    """Print formatted Q&A benchmark results."""
    print("\n" + "=" * 120)
    print("Q&A RETRIEVAL BENCHMARK RESULTS (ENHANCED)")
    print("=" * 120)
    print()

    header = f"{'Retriever':<15} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Latency':>10} {'Memory':>10} {'CPU':>8}"
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
              f"{result['peak_memory_mb']:>8.0f}MB "
              f"{result['avg_cpu_pct']:>7.1f}%")

    print()
    print("Rankings:")

    best_top10 = sorted_results[0]
    print(f"  🎯 Best Accuracy: {best_top10['name']} "
          f"({best_top10['answer_in_top_10']}/{best_top10['total_questions']} found in top-10)")

    fastest = min(sorted_results, key=lambda x: x['avg_time_ms'])
    print(f"  ⚡ Fastest:       {fastest['name']} ({fastest['avg_time_ms']:.1f}ms avg)")

    lowest_mem = min(sorted_results, key=lambda x: x['peak_memory_mb'])
    print(f"  💾 Lowest Memory: {lowest_mem['name']} ({lowest_mem['peak_memory_mb']:.0f}MB peak)")

    print()
    print("Metrics explained:")
    print("  Top-N:   % of questions where answer was found in top-N retrieved documents")
    print("  Latency: Average query time in milliseconds")
    print("  Memory:  Peak memory usage in megabytes")
    print("  CPU:     Average CPU usage during queries")
    print()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Q&A benchmark with checkpointing")
    parser.add_argument('--index', type=Path, required=True,
                        help='Path to slot index directory')
    parser.add_argument('--benchmark', type=Path,
                        default=Path('data/benchmarks/datasets/qa_benchmark_v1.jsonl'),
                        help='Path to Q&A benchmark file')
    parser.add_argument('--retrievers', type=str,
                        help='Comma-separated list of retrievers (default: all available)')
    parser.add_argument('-k', '--top-k', type=int, default=10,
                        help='Number of results to retrieve')
    parser.add_argument('--prefilter-n', type=int, default=500,
                        help='Pre-filtering candidates')
    parser.add_argument('--rerank-n', type=int, default=100,
                        help='Reranking candidates')
    parser.add_argument('--output', type=Path,
                        help='Save detailed results to JSON file')
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='Directory for checkpoint files (default: same as output)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore checkpoints)')

    args = parser.parse_args()

    # Load Q&A benchmark
    logger.info(f"Loading Q&A benchmark from {args.benchmark}...")
    questions = load_qa_benchmark(args.benchmark)
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

    # Determine retrievers to test
    RETRIEVERS = {
        'ast': ('AST-Aware', ASTAwareRetriever, False),  # Doesn't need indexer
        'mmap': ('MemoryMapped', MemoryMappedSlotRetriever, True),
        'multifaiss': ('MultiFAISS', MultiFAISSSlotRetriever, True),
        'hybrid': ('Hybrid', HybridFAISSMmapRetriever, True),
        'hnsw': ('HNSW', HNSWSlotRetriever, True),
        'scann': ('ScaNN', ScaNNSlotRetriever, True),
    }

    available = {
        'ast': (args.index / 'slot_index.jsonl').exists(),  # Only needs slot index
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

    # Setup checkpoint directory
    checkpoint_dir = args.checkpoint_dir or (args.output.parent if args.output else Path('benchmark_results/qa'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Test each retriever
    all_results = []

    for retriever_key in to_test:
        name, retriever_class, needs_indexer = RETRIEVERS[retriever_key]

        logger.info(f"Loading {name} retriever...")
        if needs_indexer:
            retriever = retriever_class(args.index, indexer)
        else:
            # AST-aware retriever doesn't need indexer
            retriever = retriever_class(args.index)
        logger.info(f"  ✓ {name} loaded")
        print()

        checkpoint_file = checkpoint_dir / f"{retriever_key}_checkpoint.json"
        if args.fresh and checkpoint_file.exists():
            logger.info(f"  Removing old checkpoint: {checkpoint_file}")
            checkpoint_file.unlink()

        logger.info(f"Testing {name} on {len(questions)} questions...")
        results = evaluate_retrieval_enhanced(
            retriever, name, questions,
            top_k=args.top_k,
            prefilter_n=args.prefilter_n,
            rerank_n=args.rerank_n,
            checkpoint_file=checkpoint_file,
            resume=not args.fresh
        )
        all_results.append(results)
        logger.info(f"  ✓ {name} complete: {results['answer_in_top_10']}/{len(questions)} found")
        logger.info(f"  ✓ Peak memory: {results['peak_memory_mb']:.0f}MB, Avg CPU: {results['avg_cpu_pct']:.1f}%")
        print()

        # Remove checkpoint after successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # Print results
    print_results(all_results)

    # Save to JSON
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("  ✓ Results saved")
        logger.info("")
        logger.info("Results ready for Claude Code analysis!")
        logger.info(f"  Pass this file to Claude: {args.output}")


if __name__ == '__main__':
    main()
