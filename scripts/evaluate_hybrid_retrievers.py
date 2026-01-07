#!/usr/bin/env python3
"""
Evaluate all active retrievers on hybrid embeddings (128d).

Compares:
1. ASTAwareRetriever - Full AST analysis + HNSW prefilter
2. HNSWSlotRetriever - HNSW prefilter + mmap slots (fastest)
3. FAISSSlotRetriever - FAISS prefilter + slot rerank
4. HybridFAISSMmapRetriever - FAISS + mmap (best accuracy expected)

Uses only questions that require retrieval (17 of 50).

Usage:
    python scripts/evaluate_hybrid_retrievers.py
    python scripts/evaluate_hybrid_retrievers.py --fresh
    python scripts/evaluate_hybrid_retrievers.py --retriever ASTAware  # Single retriever
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
INDEX_DIR = Path("data/indexes/slot_hybrid")
BENCHMARK_FILE = Path("data/benchmarks/datasets/qa_benchmark_v1.jsonl")
RESULTS_DIR = Path("data/benchmarks/results")
CHECKPOINT_FILE = Path("data/benchmarks/hybrid_eval_checkpoint.json")


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_retrieval_questions() -> List[Dict]:
    """Load only questions that require retrieval."""
    questions = []
    with open(BENCHMARK_FILE) as f:
        for line in f:
            q = json.loads(line)
            if q.get('requires_retrieval', False):
                questions.append(q)
    logger.info(f"Loaded {len(questions)} retrieval-requiring questions")
    return questions


def contains_answer(text: str, acceptable_answers: List[str]) -> bool:
    """Check if text contains any acceptable answer."""
    text_lower = text.lower()
    for answer in acceptable_answers:
        if answer.lower() in text_lower:
            return True
    return False


def load_checkpoint() -> Dict:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed_retrievers': [], 'results': []}


def save_checkpoint(data: Dict):
    """Save checkpoint atomically."""
    temp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp.rename(CHECKPOINT_FILE)


def initialize_retriever(name: str) -> Optional[object]:
    """Initialize a retriever by name."""
    logger.info(f"Initializing {name}...")

    try:
        if name == "ASTAware":
            from klareco.rag.ast_aware_retriever import ASTAwareRetriever
            return ASTAwareRetriever(
                index_path=INDEX_DIR,
                use_prefilter=True,
                use_keyword_prefilter=True,
            )

        elif name == "HNSW":
            from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            # Check if mmap exists
            mmap_dir = INDEX_DIR / "mmap"
            if not mmap_dir.exists():
                logger.warning(f"  mmap/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            # Initialize indexer for query embedding
            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return HNSWSlotRetriever(INDEX_DIR, indexer)

        elif name == "FAISS":
            from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            # Check if faiss exists
            faiss_dir = INDEX_DIR / "faiss"
            if not faiss_dir.exists():
                logger.warning(f"  faiss/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return FAISSSlotRetriever(INDEX_DIR, indexer)

        elif name == "HybridFAISS":
            from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever
            from klareco.rag.slot_indexer import SlotBasedIndexer

            # Check if both exist
            mmap_dir = INDEX_DIR / "mmap"
            faiss_dir = INDEX_DIR / "faiss"
            if not mmap_dir.exists() or not faiss_dir.exists():
                logger.warning(f"  mmap/ or faiss/ not found - run build_hybrid_mmap_faiss.sh first")
                return None

            indexer = SlotBasedIndexer(
                root_model_path=Path("models/root_embeddings/best_model.pt"),
                affix_model_path=Path("models/affix_transforms_v2/best_model.pt"),
                output_dir=INDEX_DIR,
                topical_model_path=Path("models/topical_embeddings/best_model.pt"),
                use_hybrid=True,
            )
            return HybridFAISSMmapRetriever(INDEX_DIR, indexer)

        else:
            logger.error(f"Unknown retriever: {name}")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize {name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def evaluate_retriever(
    retriever,
    retriever_name: str,
    questions: List[Dict],
    top_k: int = 10,
) -> Dict:
    """Evaluate a single retriever."""
    logger.info(f"Evaluating {retriever_name} on {len(questions)} questions...")

    results = {
        'name': retriever_name,
        'recall_at_1': 0,
        'recall_at_5': 0,
        'recall_at_10': 0,
        'mrr': 0.0,
        'total_questions': len(questions),
        'avg_latency_ms': 0,
        'peak_memory_mb': 0,
        'questions': [],
    }

    baseline_mem = get_memory_mb()
    peak_mem = baseline_mem
    total_time = 0
    mrr_sum = 0.0

    for i, qa in enumerate(questions):
        question = qa['question']
        acceptable_answers = qa['acceptable_answers']

        # Search
        start = time.time()
        try:
            if 'ASTAware' in retriever_name or hasattr(retriever, 'search'):
                search_results = retriever.search(question, top_k=top_k)
            else:
                search_results = retriever.search(question, top_k=top_k)
        except Exception as e:
            logger.error(f"  Query {i+1} failed: {e}")
            search_results = []

        latency_ms = (time.time() - start) * 1000
        total_time += latency_ms

        # Track memory
        current_mem = get_memory_mb()
        peak_mem = max(peak_mem, current_mem)

        # Find answer rank
        found_at = None
        for rank, (score, doc) in enumerate(search_results, 1):
            doc_text = doc.get('text', '')
            if contains_answer(doc_text, acceptable_answers):
                found_at = rank
                break

        # Update metrics
        if found_at:
            if found_at == 1:
                results['recall_at_1'] += 1
            if found_at <= 5:
                results['recall_at_5'] += 1
            if found_at <= 10:
                results['recall_at_10'] += 1
            mrr_sum += 1.0 / found_at

        # Store details
        results['questions'].append({
            'id': qa['id'],
            'question': question,
            'category': qa.get('category', 'unknown'),
            'found_at_rank': found_at,
            'latency_ms': latency_ms,
            'top_result': search_results[0][1].get('text', '')[:200] if search_results else None,
        })

        # Progress
        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            found = sum(1 for q in results['questions'] if q['found_at_rank'] is not None)
            pct = 100 * found / (i + 1)
            logger.info(f"  [{i+1}/{len(questions)}] Recall: {pct:.1f}% | Latency: {latency_ms:.1f}ms")

    # Final stats
    results['avg_latency_ms'] = total_time / len(questions) if questions else 0
    results['peak_memory_mb'] = peak_mem
    results['memory_delta_mb'] = peak_mem - baseline_mem
    results['mrr'] = mrr_sum / len(questions) if questions else 0

    return results


def print_comparison(all_results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("HYBRID RETRIEVER COMPARISON (128d embeddings)")
    print("=" * 100)
    print()

    header = f"{'Retriever':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Latency':>12} {'Memory':>10}"
    print(header)
    print("-" * 100)

    # Sort by recall@10
    sorted_results = sorted(all_results, key=lambda x: x['recall_at_10'], reverse=True)

    for r in sorted_results:
        total = r['total_questions']
        r1 = 100 * r['recall_at_1'] / total if total > 0 else 0
        r5 = 100 * r['recall_at_5'] / total if total > 0 else 0
        r10 = 100 * r['recall_at_10'] / total if total > 0 else 0

        print(f"{r['name']:<20} "
              f"{r1:>7.1f}% "
              f"{r5:>7.1f}% "
              f"{r10:>7.1f}% "
              f"{r['mrr']:>7.3f} "
              f"{r['avg_latency_ms']:>10.1f}ms "
              f"{r['peak_memory_mb']:>8.0f}MB")

    print()
    print("Legend: R@k = Recall at k (% of questions where answer found in top k)")
    print("        MRR = Mean Reciprocal Rank (higher is better)")
    print()

    if sorted_results:
        best = sorted_results[0]
        fastest = min(sorted_results, key=lambda x: x['avg_latency_ms'])
        print(f"Best Accuracy: {best['name']} ({best['recall_at_10']}/{best['total_questions']} in top-10)")
        print(f"Fastest:       {fastest['name']} ({fastest['avg_latency_ms']:.1f}ms avg)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrievers")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--retriever", type=str, help="Evaluate single retriever (ASTAware, HNSW, FAISS, HybridFAISS)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    args = parser.parse_args()

    # Load questions
    questions = load_retrieval_questions()

    if not questions:
        logger.error("No retrieval questions found!")
        sys.exit(1)

    # Determine which retrievers to test
    if args.retriever:
        retriever_names = [args.retriever]
    else:
        retriever_names = ["ASTAware", "HNSW", "FAISS", "HybridFAISS"]

    # Load checkpoint
    checkpoint = load_checkpoint() if not args.fresh else {'completed_retrievers': [], 'results': []}

    all_results = checkpoint['results']

    # Evaluate each retriever
    for name in retriever_names:
        if name in checkpoint['completed_retrievers']:
            logger.info(f"Skipping {name} (already evaluated, use --fresh to re-run)")
            continue

        retriever = initialize_retriever(name)
        if retriever is None:
            logger.warning(f"Skipping {name} (initialization failed)")
            continue

        results = evaluate_retriever(retriever, name, questions, top_k=args.top_k)
        all_results.append(results)

        # Save checkpoint
        checkpoint['completed_retrievers'].append(name)
        checkpoint['results'] = all_results
        save_checkpoint(checkpoint)

        # Clean up retriever to free memory
        del retriever

    # Print comparison
    print_comparison(all_results)

    # Save final results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"hybrid_retriever_comparison_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'index_dir': str(INDEX_DIR),
            'benchmark_file': str(BENCHMARK_FILE),
            'total_questions': len(questions),
            'retrievers': all_results,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Clean up checkpoint on success
    if len(all_results) == len(retriever_names):
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("Evaluation complete, checkpoint removed")


if __name__ == "__main__":
    main()
