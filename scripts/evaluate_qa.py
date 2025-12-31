#!/usr/bin/env python3
"""
Q&A Benchmark Evaluation Script (CP1/CP2)

Evaluates a Q&A system against the M1 benchmark.
Computes exact match, partial match, F1, and retrieval metrics.

Usage:
    python scripts/evaluate_qa.py                    # Evaluate current Klareco system
    python scripts/evaluate_qa.py --system klareco   # Same as above
    python scripts/evaluate_qa.py --system baseline  # No retrieval, just return "Mi ne scias"
    python scripts/evaluate_qa.py --dry-run          # Show questions without evaluation
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class QuestionResult:
    """Result of answering a single question."""
    question_id: str
    question: str
    category: str
    gold_answer: str
    acceptable_answers: List[str]
    predicted_answer: str
    retrieved_docs: List[str] = field(default_factory=list)
    exact_match: bool = False
    partial_match: bool = False
    f1_score: float = 0.0
    retrieval_hit: bool = False  # Was answer in retrieved docs?
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    total_questions: int = 0
    exact_match: float = 0.0
    partial_match: float = 0.0
    f1_score: float = 0.0
    retrieval_recall_1: float = 0.0
    retrieval_recall_5: float = 0.0
    retrieval_recall_10: float = 0.0
    avg_latency_ms: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    question_results: List[QuestionResult] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove punctuation except Esperanto special chars
    text = re.sub(r'[^\w\sĉĝĥĵŝŭ]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> set:
    """Tokenize text into words."""
    return set(normalize_text(text).split())


def compute_f1(prediction: str, reference: str) -> float:
    """Compute F1 score between prediction and reference."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def check_exact_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check if prediction exactly matches gold or any acceptable answer."""
    pred_norm = normalize_text(prediction)

    # Check gold answer
    if pred_norm == normalize_text(gold):
        return True

    # Check acceptable answers
    for acc in acceptable:
        if pred_norm == normalize_text(acc):
            return True

    return False


def check_partial_match(prediction: str, gold: str, acceptable: List[str]) -> bool:
    """Check if prediction contains any acceptable answer."""
    pred_norm = normalize_text(prediction)

    # Check if gold is contained
    if normalize_text(gold) in pred_norm:
        return True

    # Check acceptable answers
    for acc in acceptable:
        if normalize_text(acc) in pred_norm:
            return True

    return False


def check_retrieval_hit(retrieved_docs: List[str], gold: str, acceptable: List[str]) -> bool:
    """Check if any retrieved document contains the answer."""
    for doc in retrieved_docs:
        doc_norm = normalize_text(doc)
        if normalize_text(gold) in doc_norm:
            return True
        for acc in acceptable:
            if normalize_text(acc) in doc_norm:
                return True
    return False


def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    """Load the Q&A benchmark from JSONL file."""
    questions = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


# ============================================================================
# Q&A Systems
# ============================================================================

def baseline_system(question: str, top_k: int = 10) -> tuple:
    """Baseline system that just returns 'Mi ne scias.'"""
    return "Mi ne scias.", []


def klareco_system(question: str, top_k: int = 10) -> tuple:
    """
    Klareco RAG system with reranking and answer extraction.

    Pipeline: Retrieve -> Rerank (CP4) -> Extract (CP5)

    Returns:
        tuple: (answer, retrieved_docs)
    """
    try:
        from klareco import Retriever
        from klareco.qa import AnswerExtractor, DeterministicReranker

        # Load components (cached after first call)
        if not hasattr(klareco_system, '_retriever'):
            klareco_system._retriever = Retriever.load()
        if not hasattr(klareco_system, '_reranker'):
            klareco_system._reranker = DeterministicReranker()
        if not hasattr(klareco_system, '_extractor'):
            klareco_system._extractor = AnswerExtractor()

        retriever = klareco_system._retriever
        reranker = klareco_system._reranker
        extractor = klareco_system._extractor

        # Retrieve relevant documents (fetch more than top_k for reranking)
        results = retriever.search(question, top_k=top_k * 2)
        retrieved_docs = [r.text for r in results]
        original_scores = [getattr(r, 'score', 1.0 - i/len(results))
                          for i, r in enumerate(results)]

        # Rerank documents (CP4)
        if retrieved_docs:
            reranked = reranker.rerank(
                question,
                retrieved_docs,
                original_scores=original_scores,
                top_k=top_k
            )
            reranked_docs = [r.text for r in reranked]
        else:
            reranked_docs = []

        # Use answer extractor (CP5)
        if reranked_docs:
            extraction = extractor.extract(question, reranked_docs)
            answer = extraction.answer
        else:
            answer = "Mi ne scias."

        return answer, reranked_docs

    except Exception as e:
        return f"Eraro: {e}", []


def get_system(system_name: str) -> Callable:
    """Get Q&A system by name."""
    systems = {
        'baseline': baseline_system,
        'klareco': klareco_system,
    }
    if system_name not in systems:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(systems.keys())}")
    return systems[system_name]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_question(
    question: Dict[str, Any],
    qa_system: Callable,
    top_k: int = 10
) -> QuestionResult:
    """Evaluate a single question."""
    q_id = question['id']
    q_text = question['question']
    gold = question['gold_answer']
    acceptable = question.get('acceptable_answers', [])
    category = question['category']

    # Time the system
    start_time = time.time()
    try:
        predicted, retrieved = qa_system(q_text, top_k=top_k)
    except Exception as e:
        return QuestionResult(
            question_id=q_id,
            question=q_text,
            category=category,
            gold_answer=gold,
            acceptable_answers=acceptable,
            predicted_answer="",
            error=str(e)
        )
    latency_ms = (time.time() - start_time) * 1000

    # Compute metrics
    exact = check_exact_match(predicted, gold, acceptable)
    partial = check_partial_match(predicted, gold, acceptable)
    f1 = compute_f1(predicted, gold)

    # Check retrieval hit
    retrieval_hit = check_retrieval_hit(retrieved, gold, acceptable)

    return QuestionResult(
        question_id=q_id,
        question=q_text,
        category=category,
        gold_answer=gold,
        acceptable_answers=acceptable,
        predicted_answer=predicted,
        retrieved_docs=retrieved,
        exact_match=exact,
        partial_match=partial,
        f1_score=f1,
        retrieval_hit=retrieval_hit,
        latency_ms=latency_ms
    )


def evaluate_benchmark(
    questions: List[Dict[str, Any]],
    qa_system: Callable,
    top_k: int = 10,
    verbose: bool = False
) -> EvaluationResults:
    """Evaluate all questions in the benchmark."""
    results = EvaluationResults()
    results.total_questions = len(questions)

    # Category-level accumulators
    category_counts = defaultdict(int)
    category_exact = defaultdict(int)
    category_partial = defaultdict(int)
    category_f1 = defaultdict(float)

    total_latency = 0.0
    total_exact = 0
    total_partial = 0
    total_f1 = 0.0

    # Retrieval recall tracking
    retrieval_hits_at_k = defaultdict(int)
    retrieval_questions = 0

    for i, question in enumerate(questions):
        if verbose:
            print(f"\r[{i+1}/{len(questions)}] Evaluating: {question['id']}...", end='', flush=True)

        result = evaluate_question(question, qa_system, top_k=top_k)
        results.question_results.append(result)

        if result.error:
            continue

        # Aggregate metrics
        category = result.category
        category_counts[category] += 1

        if result.exact_match:
            total_exact += 1
            category_exact[category] += 1

        if result.partial_match:
            total_partial += 1
            category_partial[category] += 1

        total_f1 += result.f1_score
        category_f1[category] += result.f1_score

        total_latency += result.latency_ms

        # Retrieval recall (only for retrieval questions)
        if question.get('requires_retrieval', False):
            retrieval_questions += 1
            if result.retrieval_hit:
                retrieval_hits_at_k[top_k] += 1

    if verbose:
        print()  # Newline after progress

    # Compute final metrics
    n = results.total_questions
    results.exact_match = total_exact / n if n > 0 else 0
    results.partial_match = total_partial / n if n > 0 else 0
    results.f1_score = total_f1 / n if n > 0 else 0
    results.avg_latency_ms = total_latency / n if n > 0 else 0

    # Retrieval recall
    if retrieval_questions > 0:
        results.retrieval_recall_10 = retrieval_hits_at_k.get(10, 0) / retrieval_questions

    # By-category metrics
    for cat in category_counts:
        cat_n = category_counts[cat]
        results.by_category[cat] = {
            'count': cat_n,
            'exact_match': category_exact[cat] / cat_n if cat_n > 0 else 0,
            'partial_match': category_partial[cat] / cat_n if cat_n > 0 else 0,
            'f1_score': category_f1[cat] / cat_n if cat_n > 0 else 0,
        }

    return results


def print_results(results: EvaluationResults, show_errors: bool = False):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("Q&A BENCHMARK EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal Questions: {results.total_questions}")
    print(f"\nOverall Metrics:")
    print(f"  Exact Match:   {results.exact_match:.1%}")
    print(f"  Partial Match: {results.partial_match:.1%}")
    print(f"  F1 Score:      {results.f1_score:.3f}")
    print(f"  Avg Latency:   {results.avg_latency_ms:.0f}ms")

    if results.retrieval_recall_10 > 0:
        print(f"\nRetrieval Metrics:")
        print(f"  Recall@10: {results.retrieval_recall_10:.1%}")

    print(f"\nBy Category:")
    print(f"  {'Category':<15} {'Count':>6} {'Exact':>8} {'Partial':>8} {'F1':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for cat, metrics in sorted(results.by_category.items()):
        print(f"  {cat:<15} {metrics['count']:>6} "
              f"{metrics['exact_match']:>7.1%} "
              f"{metrics['partial_match']:>7.1%} "
              f"{metrics['f1_score']:>7.3f}")

    # Show errors
    errors = [r for r in results.question_results if r.error]
    if errors:
        print(f"\nErrors: {len(errors)}")
        if show_errors:
            for r in errors:
                print(f"  {r.question_id}: {r.error}")

    # Show sample wrong answers
    wrong = [r for r in results.question_results if not r.exact_match and not r.error]
    if wrong and show_errors:
        print(f"\nSample Wrong Answers (first 5):")
        for r in wrong[:5]:
            print(f"\n  Q: {r.question}")
            print(f"  Gold: {r.gold_answer}")
            print(f"  Pred: {r.predicted_answer[:100]}...")


def save_results(results: EvaluationResults, output_path: Path):
    """Save results to JSON file."""
    output = {
        'total_questions': results.total_questions,
        'exact_match': results.exact_match,
        'partial_match': results.partial_match,
        'f1_score': results.f1_score,
        'retrieval_recall_10': results.retrieval_recall_10,
        'avg_latency_ms': results.avg_latency_ms,
        'by_category': results.by_category,
        'question_results': [
            {
                'question_id': r.question_id,
                'question': r.question,
                'category': r.category,
                'gold_answer': r.gold_answer,
                'predicted_answer': r.predicted_answer,
                'exact_match': r.exact_match,
                'partial_match': r.partial_match,
                'f1_score': r.f1_score,
                'latency_ms': r.latency_ms,
                'error': r.error,
            }
            for r in results.question_results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Q&A benchmark')
    parser.add_argument('--benchmark', type=Path,
                        default=PROJECT_ROOT / 'data' / 'benchmarks' / 'qa_benchmark_v1.jsonl',
                        help='Path to benchmark JSONL file')
    parser.add_argument('--system', type=str, default='klareco',
                        choices=['baseline', 'klareco'],
                        help='Q&A system to evaluate')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of documents to retrieve')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show progress during evaluation')
    parser.add_argument('--show-errors', action='store_true',
                        help='Show error details and wrong answers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show questions without evaluation')
    parser.add_argument('--category', type=str, default=None,
                        help='Only evaluate specific category')

    args = parser.parse_args()

    # Load benchmark
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    questions = load_benchmark(args.benchmark)
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    # Filter by category if specified
    if args.category:
        questions = [q for q in questions if q['category'] == args.category]
        print(f"Filtered to {len(questions)} questions in category '{args.category}'")

    # Dry run - just show questions
    if args.dry_run:
        print("\n" + "=" * 60)
        print("BENCHMARK QUESTIONS (Dry Run)")
        print("=" * 60)
        for q in questions:
            print(f"\n[{q['id']}] ({q['category']}) {q['question']}")
            print(f"  Gold: {q['gold_answer']}")
        return

    # Get system
    qa_system = get_system(args.system)
    print(f"Evaluating system: {args.system}")

    # Run evaluation
    results = evaluate_benchmark(
        questions,
        qa_system,
        top_k=args.top_k,
        verbose=args.verbose
    )

    # Print results
    print_results(results, show_errors=args.show_errors)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path
        default_output = PROJECT_ROOT / 'data' / 'benchmarks' / f'results_{args.system}.json'
        save_results(results, default_output)


if __name__ == '__main__':
    main()
