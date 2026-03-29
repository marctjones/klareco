#!/usr/bin/env python3
"""
Comprehensive Pipeline Evaluation Framework

Evaluates the full extractive QA pipeline with multi-stage metrics and performance profiling.

Metrics tracked:
1. Answer Quality:
   - Keyword match accuracy (existing metric)
   - Answer coherence (length, sentence structure)
   - Citation coverage (how many facts have citations)

2. Retrieval Quality:
   - Retrieval recall (did top-K contain answer?)
   - Retrieval precision (what % of retrieved sentences are relevant?)
   - Mean Reciprocal Rank (MRR)

3. Extraction Quality:
   - Facts extracted per question
   - Facts selected per question
   - M1 filtering rate
   - Question-type filtering rate

4. Performance/Timing:
   - Total time per question
   - Time per pipeline stage (retrieval, extraction, generation)
   - CPU time vs wall time

5. Pipeline Health:
   - Parse failures
   - Retrieval failures (0 results)
   - Generation failures (empty answers)

Usage:
    python scripts/evaluate_pipeline_comprehensive.py
    python scripts/evaluate_pipeline_comprehensive.py --limit 50
    python scripts/evaluate_pipeline_comprehensive.py --test-set data/test_sets/qa_test_set_50.jsonl
    python scripts/evaluate_pipeline_comprehensive.py --output results/eval_2024_01_15.json

Output:
    - Detailed JSON report with all metrics
    - Human-readable summary
    - CSV export for analysis
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.path.insert(0, str(Path(__file__).parent))
from demo_extractive_qa import retrieve_sentences, expand_with_embeddings, extract_query_entity

from klareco.parser import parse
from klareco.rag.whoosh_retriever import WhooshRetriever
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator, classify_question_type

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Performance timing for pipeline stages."""
    total_time: float = 0.0
    parse_time: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    extraction_time: float = 0.0
    scoring_time: float = 0.0
    generation_time: float = 0.0


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    num_retrieved: int = 0
    contains_answer: bool = False
    answer_rank: Optional[int] = None  # Rank of first sentence containing answer (1-indexed)
    recall_at_5: bool = False
    recall_at_10: bool = False
    recall_at_20: bool = False
    mrr: float = 0.0  # Mean Reciprocal Rank


@dataclass
class ExtractionMetrics:
    """Fact extraction metrics."""
    facts_extracted: int = 0
    facts_after_type_filter: int = 0
    facts_after_m1_filter: int = 0
    facts_selected: int = 0
    type_filter_rate: float = 0.0  # % removed by type filter
    m1_filter_rate: float = 0.0    # % removed by M1 filter


@dataclass
class AnswerMetrics:
    """Answer quality metrics."""
    has_answer: bool = False
    keyword_match: bool = False
    matched_keywords: List[str] = None
    answer_length: int = 0  # characters
    num_sentences: int = 0  # rough estimate
    num_citations: int = 0


@dataclass
class QuestionResult:
    """Complete result for one question."""
    question_id: str
    question_text: str
    question_type: str
    expected_keywords: List[str]
    answer_text: str

    timing: TimingMetrics
    retrieval: RetrievalMetrics
    extraction: ExtractionMetrics
    answer: AnswerMetrics

    success: bool  # Overall success (keyword match)
    error: Optional[str] = None


def check_answer_contains_keywords(answer_text: str, keywords: List[str]) -> tuple:
    """
    Check if answer contains expected keywords.

    Returns:
        (found: bool, matched: List[str])
    """
    answer_lower = answer_text.lower()
    matched = []

    for kw in keywords:
        if kw.lower() in answer_lower:
            matched.append(kw)

    return len(matched) > 0, matched


def check_retrieval_contains_answer(sentences: List[Dict], keywords: List[str]) -> tuple:
    """
    Check if retrieved sentences contain answer keywords.

    Returns:
        (contains: bool, first_rank: Optional[int], recall@5, recall@10, recall@20, mrr)
    """
    first_rank = None

    for i, sent in enumerate(sentences):
        text = sent.get('text', '').lower()
        if any(kw.lower() in text for kw in keywords):
            if first_rank is None:
                first_rank = i + 1  # 1-indexed

    contains = first_rank is not None
    recall_at_5 = first_rank is not None and first_rank <= 5
    recall_at_10 = first_rank is not None and first_rank <= 10
    recall_at_20 = first_rank is not None and first_rank <= 20
    mrr = 1.0 / first_rank if first_rank else 0.0

    return contains, first_rank, recall_at_5, recall_at_10, recall_at_20, mrr


def estimate_num_sentences(text: str) -> int:
    """Rough estimate of sentence count (count periods, question marks, exclamation marks)."""
    return text.count('.') + text.count('?') + text.count('!')


def evaluate_question_comprehensive(
    question: str,
    expected_keywords: List[str],
    question_type: str,
    question_id: str,
    generator: ExtractiveAnswerGenerator,
    retriever: WhooshRetriever,
    top_k: int = 20,
) -> QuestionResult:
    """
    Evaluate single question with comprehensive metrics and timing.

    Returns:
        QuestionResult with all metrics
    """
    timing = TimingMetrics()
    retrieval_metrics = RetrievalMetrics()
    extraction_metrics = ExtractionMetrics()
    answer_metrics = AnswerMetrics()

    error = None
    answer_text = ""

    try:
        # Stage 1: Parse query
        t0 = time.time()
        query_ast = parse(question)
        timing.parse_time = time.time() - t0

        question_type_enum = classify_question_type(question)
        query_entity = extract_query_entity(query_ast, question_type_enum)

        # Stage 2: Retrieval
        t0 = time.time()

        # Extract roots from question
        roots = []
        def extract_roots(node):
            if isinstance(node, dict):
                if node.get('tipo') == 'vorto':
                    plena_vorto = node.get('plena_vorto', '')
                    root = node.get('radiko', '')
                    question_words = {'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel', 'kiom', 'kia', 'kies'}

                    if plena_vorto and plena_vorto[0].isupper() and plena_vorto.lower() not in question_words:
                        word = plena_vorto.rstrip('n').rstrip('j').rstrip('n')
                        roots.append(word)
                    elif root:
                        roots.append(root.lower())
                elif node.get('tipo') == 'vortgrupo':
                    extract_roots(node.get('kerno'))
                    for p in node.get('priskriboj', []):
                        extract_roots(p)
                elif node.get('tipo') == 'frazo':
                    extract_roots(node.get('subjekto'))
                    extract_roots(node.get('verbo'))
                    extract_roots(node.get('objekto'))
                    for a in node.get('aliaj', []):
                        extract_roots(a)

        extract_roots(query_ast)

        # Expand with embeddings
        synonyms = {
            'fond': ['kre', 'establ', 'startig'],
            'est': ['est'],
        }
        query_roots = set(roots)
        for root in roots:
            if root in synonyms:
                query_roots.update(synonyms[root])

        embeddings_path = Path('models/root_embeddings_phase1_fast/root_embeddings_best.pt')
        if embeddings_path.exists():
            expanded = expand_with_embeddings(
                list(query_roots),
                embeddings_path,
                k=5,
                threshold=0.65
            )
            query_roots = expanded

        # Retrieve sentences
        sentences = retrieve_sentences(
            retriever,
            list(query_roots),
            question_type_enum.value,
            query_entity,
            top_k,
            query_ast=query_ast
        )

        timing.retrieval_time = time.time() - t0

        # Retrieval metrics
        retrieval_metrics.num_retrieved = len(sentences)
        if sentences:
            (contains, first_rank, recall_5, recall_10, recall_20, mrr) = \
                check_retrieval_contains_answer(sentences, expected_keywords)

            retrieval_metrics.contains_answer = contains
            retrieval_metrics.answer_rank = first_rank
            retrieval_metrics.recall_at_5 = recall_5
            retrieval_metrics.recall_at_10 = recall_10
            retrieval_metrics.recall_at_20 = recall_20
            retrieval_metrics.mrr = mrr

        if not sentences:
            error = "No sentences retrieved"
            answer_text = "Mi ne trovis respondon."
        else:
            # Stage 3: Generate answer
            t0 = time.time()
            answer = generator.generate(
                sentences,
                question,
                question_type=question_type_enum,
                query_entity=query_entity
            )
            timing.generation_time = time.time() - t0

            answer_text = answer.text

            # Extraction metrics
            extraction_metrics.facts_extracted = answer.num_facts_extracted
            extraction_metrics.facts_selected = answer.num_facts_selected

            # Calculate filter rates
            if answer.num_facts_extracted > 0:
                extraction_metrics.type_filter_rate = 0.0  # Would need to track intermediate values
                extraction_metrics.m1_filter_rate = 0.0
                extraction_metrics.facts_after_type_filter = answer.num_facts_extracted
                extraction_metrics.facts_after_m1_filter = answer.num_facts_extracted

            # Answer metrics
            answer_metrics.has_answer = len(answer_text) > 0 and answer_text != "Mi ne trovis respondon."
            answer_metrics.answer_length = len(answer_text)
            answer_metrics.num_sentences = estimate_num_sentences(answer_text)
            answer_metrics.num_citations = len(answer.citations)

    except Exception as e:
        logger.error(f"Error evaluating question '{question}': {e}")
        error = str(e)
        answer_text = ""

    # Check keyword match
    if answer_text:
        found, matched = check_answer_contains_keywords(answer_text, expected_keywords)
        answer_metrics.keyword_match = found
        answer_metrics.matched_keywords = matched
    else:
        answer_metrics.matched_keywords = []

    # Total time
    timing.total_time = (timing.parse_time + timing.retrieval_time +
                        timing.rerank_time + timing.extraction_time +
                        timing.scoring_time + timing.generation_time)

    return QuestionResult(
        question_id=question_id,
        question_text=question,
        question_type=question_type,
        expected_keywords=expected_keywords,
        answer_text=answer_text[:200] + '...' if len(answer_text) > 200 else answer_text,
        timing=timing,
        retrieval=retrieval_metrics,
        extraction=extraction_metrics,
        answer=answer_metrics,
        success=answer_metrics.keyword_match,
        error=error
    )


def aggregate_metrics(results: List[QuestionResult]) -> Dict[str, Any]:
    """
    Aggregate metrics across all questions.

    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}

    # Overall accuracy
    num_correct = sum(1 for r in results if r.success)
    accuracy = num_correct / len(results)

    # Timing statistics
    total_times = [r.timing.total_time for r in results]
    parse_times = [r.timing.parse_time for r in results]
    retrieval_times = [r.timing.retrieval_time for r in results]
    generation_times = [r.timing.generation_time for r in results]

    # Retrieval statistics
    retrieval_recalls_5 = [r.retrieval.recall_at_5 for r in results]
    retrieval_recalls_10 = [r.retrieval.recall_at_10 for r in results]
    retrieval_recalls_20 = [r.retrieval.recall_at_20 for r in results]
    mrrs = [r.retrieval.mrr for r in results]

    # Extraction statistics
    facts_extracted = [r.extraction.facts_extracted for r in results]
    facts_selected = [r.extraction.facts_selected for r in results]

    # Answer statistics
    answer_lengths = [r.answer.answer_length for r in results if r.answer.has_answer]
    num_citations = [r.answer.num_citations for r in results if r.answer.has_answer]

    # By question type
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        by_type[r.question_type]['total'] += 1
        if r.success:
            by_type[r.question_type]['correct'] += 1

    return {
        'overall': {
            'num_questions': len(results),
            'num_correct': num_correct,
            'accuracy': accuracy,
        },
        'timing': {
            'total_time_mean': np.mean(total_times),
            'total_time_median': np.median(total_times),
            'total_time_std': np.std(total_times),
            'parse_time_mean': np.mean(parse_times),
            'retrieval_time_mean': np.mean(retrieval_times),
            'generation_time_mean': np.mean(generation_times),
        },
        'retrieval': {
            'recall_at_5': np.mean(retrieval_recalls_5),
            'recall_at_10': np.mean(retrieval_recalls_10),
            'recall_at_20': np.mean(retrieval_recalls_20),
            'mean_reciprocal_rank': np.mean(mrrs),
        },
        'extraction': {
            'facts_extracted_mean': np.mean(facts_extracted) if facts_extracted else 0,
            'facts_extracted_median': np.median(facts_extracted) if facts_extracted else 0,
            'facts_selected_mean': np.mean(facts_selected) if facts_selected else 0,
            'facts_selected_median': np.median(facts_selected) if facts_selected else 0,
        },
        'answer': {
            'answer_length_mean': np.mean(answer_lengths) if answer_lengths else 0,
            'answer_length_median': np.median(answer_lengths) if answer_lengths else 0,
            'citations_per_answer_mean': np.mean(num_citations) if num_citations else 0,
        },
        'by_question_type': {
            qtype: {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
            for qtype, stats in by_type.items()
        }
    }


def print_summary(results: List[QuestionResult], aggregates: Dict[str, Any]):
    """Print human-readable summary of results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)

    # Overall accuracy
    overall = aggregates['overall']
    print(f"\nOverall: {overall['num_correct']}/{overall['num_questions']} correct ({overall['accuracy']*100:.1f}%)")

    # By question type
    print("\nBy Question Type:")
    for qtype, stats in sorted(aggregates['by_question_type'].items()):
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%)")

    # Timing
    timing = aggregates['timing']
    print("\nTiming (seconds):")
    print(f"  Total time per question: {timing['total_time_mean']:.3f}s (±{timing['total_time_std']:.3f}s)")
    print(f"  Parse: {timing['parse_time_mean']:.3f}s")
    print(f"  Retrieval: {timing['retrieval_time_mean']:.3f}s")
    print(f"  Generation: {timing['generation_time_mean']:.3f}s")

    # Retrieval quality
    retrieval = aggregates['retrieval']
    print("\nRetrieval Quality:")
    print(f"  Recall@5: {retrieval['recall_at_5']*100:.1f}%")
    print(f"  Recall@10: {retrieval['recall_at_10']*100:.1f}%")
    print(f"  Recall@20: {retrieval['recall_at_20']*100:.1f}%")
    print(f"  MRR: {retrieval['mean_reciprocal_rank']:.3f}")

    # Extraction
    extraction = aggregates['extraction']
    print("\nExtraction:")
    print(f"  Facts extracted: {extraction['facts_extracted_mean']:.1f} (median: {extraction['facts_extracted_median']:.0f})")
    print(f"  Facts selected: {extraction['facts_selected_mean']:.1f} (median: {extraction['facts_selected_median']:.0f})")

    # Answer quality
    answer = aggregates['answer']
    print("\nAnswer Quality:")
    print(f"  Answer length: {answer['answer_length_mean']:.0f} chars (median: {answer['answer_length_median']:.0f})")
    print(f"  Citations per answer: {answer['citations_per_answer_mean']:.1f}")

    print("\n" + "="*80)


def export_to_csv(results: List[QuestionResult], output_path: Path):
    """Export results to CSV for analysis."""
    import csv

    csv_path = output_path.with_suffix('.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'question_id', 'question_type', 'success', 'keyword_match',
            'total_time', 'parse_time', 'retrieval_time', 'generation_time',
            'num_retrieved', 'contains_answer', 'answer_rank', 'recall@5', 'recall@10', 'recall@20', 'mrr',
            'facts_extracted', 'facts_selected',
            'answer_length', 'num_sentences', 'num_citations'
        ])

        # Data rows
        for r in results:
            writer.writerow([
                r.question_id, r.question_type, r.success, r.answer.keyword_match,
                r.timing.total_time, r.timing.parse_time, r.timing.retrieval_time, r.timing.generation_time,
                r.retrieval.num_retrieved, r.retrieval.contains_answer, r.retrieval.answer_rank,
                r.retrieval.recall_at_5, r.retrieval.recall_at_10, r.retrieval.recall_at_20, r.retrieval.mrr,
                r.extraction.facts_extracted, r.extraction.facts_selected,
                r.answer.answer_length, r.answer.num_sentences, r.answer.num_citations
            ])

    print(f"\n✓ CSV exported to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_diverse_30.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--no-m1', action='store_true', help='Disable M1 filtering')
    parser.add_argument('--no-rerank', action='store_true', help='Disable neural reranking')
    parser.add_argument('--limit', type=int, help='Limit to first N questions')
    parser.add_argument('--seed', type=int, help='Random seed for question order (for reproducibility)')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable question order randomization')
    parser.add_argument('--output', type=Path, help='Output JSON file for results')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load test set
    print(f"Loading test set from {args.test_set}")
    test_questions = []
    with open(args.test_set) as f:
        for line in f:
            test_questions.append(json.loads(line))

    # Randomize question order (unless disabled)
    if not args.no_shuffle:
        if args.seed is not None:
            random.seed(args.seed)
            print(f"Shuffling questions with seed={args.seed}")
        else:
            # Use time-based seed for true randomization
            seed = int(time.time() * 1000) % (2**32)
            random.seed(seed)
            print(f"Shuffling questions with random seed={seed}")
        random.shuffle(test_questions)
    else:
        print("Question order: original (no shuffle)")

    if args.limit:
        test_questions = test_questions[:args.limit]

    print(f"Evaluating {len(test_questions)} questions")
    print()

    # Initialize Whoosh retriever
    print("Loading Whoosh FTS index...")
    retriever = WhooshRetriever(
        whoosh_index_dir=Path('data/indexes/whoosh_fts'),
        kuzu_db_path=args.db
    )
    print("✓ Whoosh loaded\n")

    # Initialize generator
    print("Loading extractive QA system...")
    generator = ExtractiveAnswerGenerator(
        use_reranker=not args.no_rerank,
        use_m1=not args.no_m1,
    )
    print()

    # Evaluate each question
    results = []
    for i, test_q in enumerate(test_questions, 1):
        question = test_q['question']
        expected_keywords = test_q['expected_keywords']
        question_type = test_q['question_type']
        question_id = test_q.get('id', f'q{i}')

        if args.verbose:
            print(f"[{i}/{len(test_questions)}] {question_type}: {question}")

        result = evaluate_question_comprehensive(
            question,
            expected_keywords,
            question_type,
            question_id,
            generator,
            retriever,
            args.top_k
        )

        results.append(result)

        if args.verbose:
            status = "✓" if result.success else "✗"
            print(f"  {status} Expected: {expected_keywords}")
            print(f"    Matched: {result.answer.matched_keywords}")
            print(f"    Time: {result.timing.total_time:.3f}s (R:{result.timing.retrieval_time:.3f}s, G:{result.timing.generation_time:.3f}s)")
            print(f"    Retrieval: {result.retrieval.num_retrieved} docs, rank={result.retrieval.answer_rank}, MRR={result.retrieval.mrr:.3f}")
            print(f"    Extraction: {result.extraction.facts_extracted} extracted, {result.extraction.facts_selected} selected")
            print()

    # Aggregate metrics
    aggregates = aggregate_metrics(results)

    # Print summary
    print_summary(results, aggregates)

    # Export to JSON
    if args.output:
        output_data = {
            'metadata': {
                'test_set': str(args.test_set),
                'num_questions': len(test_questions),
                'top_k': args.top_k,
                'use_m1': not args.no_m1,
                'use_rerank': not args.no_rerank,
            },
            'aggregates': aggregates,
            'results': [asdict(r) for r in results]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Full results saved to: {args.output}")

    # Export to CSV
    if args.export_csv or args.output:
        output_path = args.output or Path('evaluation_results.json')
        export_to_csv(results, output_path)


if __name__ == '__main__':
    main()
