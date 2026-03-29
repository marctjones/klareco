#!/usr/bin/env python3
"""
Analyze Evaluation Results - Visualize Pipeline Performance

Analyzes comprehensive evaluation JSON output and generates insights.

Features:
1. Performance breakdown by pipeline stage
2. Failure analysis (where does the pipeline fail?)
3. Question type analysis
4. Retrieval vs extraction vs generation performance
5. Timing analysis and bottlenecks

Usage:
    python scripts/analyze_evaluation_results.py results/eval_baseline.json
    python scripts/analyze_evaluation_results.py results/eval_baseline.json results/eval_improved.json --compare
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def load_results(path: Path) -> Dict:
    """Load evaluation results from JSON."""
    with open(path) as f:
        return json.load(f)


def analyze_failure_points(results: List[Dict]) -> Dict[str, int]:
    """
    Analyze where in the pipeline questions fail.

    Returns:
        Dict mapping failure point to count
    """
    failure_points = {
        'retrieval_empty': 0,      # No sentences retrieved
        'retrieval_no_answer': 0,  # Retrieved sentences don't contain answer
        'extraction_no_facts': 0,  # No facts extracted
        'extraction_wrong_facts': 0,  # Facts extracted but wrong ones selected
        'generation_poor': 0,      # Answer generated but doesn't match keywords
        'success': 0,              # Correct answer
    }

    for r in results:
        if r['success']:
            failure_points['success'] += 1
        elif r['retrieval']['num_retrieved'] == 0:
            failure_points['retrieval_empty'] += 1
        elif not r['retrieval']['contains_answer']:
            failure_points['retrieval_no_answer'] += 1
        elif r['extraction']['facts_extracted'] == 0:
            failure_points['extraction_no_facts'] += 1
        elif r['extraction']['facts_selected'] == 0:
            failure_points['extraction_wrong_facts'] += 1
        else:
            failure_points['generation_poor'] += 1

    return failure_points


def analyze_timing_bottlenecks(results: List[Dict]) -> Dict[str, float]:
    """
    Identify timing bottlenecks in the pipeline.

    Returns:
        Dict mapping stage to percentage of total time
    """
    total_times = []
    parse_times = []
    retrieval_times = []
    generation_times = []

    for r in results:
        timing = r['timing']
        total_times.append(timing['total_time'])
        parse_times.append(timing['parse_time'])
        retrieval_times.append(timing['retrieval_time'])
        generation_times.append(timing['generation_time'])

    total_sum = sum(total_times)

    return {
        'parse': sum(parse_times) / total_sum * 100,
        'retrieval': sum(retrieval_times) / total_sum * 100,
        'generation': sum(generation_times) / total_sum * 100,
    }


def analyze_question_type_performance(results: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze performance breakdown by question type.

    Returns:
        Dict mapping question type to performance metrics
    """
    by_type = {}

    for r in results:
        qtype = r['question_type']

        if qtype not in by_type:
            by_type[qtype] = {
                'total': 0,
                'correct': 0,
                'retrieval_success': 0,
                'extraction_success': 0,
                'avg_retrieval_time': [],
                'avg_mrr': [],
            }

        stats = by_type[qtype]
        stats['total'] += 1

        if r['success']:
            stats['correct'] += 1

        if r['retrieval']['contains_answer']:
            stats['retrieval_success'] += 1

        if r['extraction']['facts_extracted'] > 0:
            stats['extraction_success'] += 1

        stats['avg_retrieval_time'].append(r['timing']['retrieval_time'])
        stats['avg_mrr'].append(r['retrieval']['mrr'])

    # Compute averages
    for qtype in by_type:
        stats = by_type[qtype]
        stats['accuracy'] = stats['correct'] / stats['total']
        stats['retrieval_recall'] = stats['retrieval_success'] / stats['total']
        stats['extraction_rate'] = stats['extraction_success'] / stats['total']
        stats['avg_retrieval_time'] = np.mean(stats['avg_retrieval_time'])
        stats['avg_mrr'] = np.mean(stats['avg_mrr'])

    return by_type


def print_analysis(data: Dict):
    """Print comprehensive analysis of evaluation results."""
    metadata = data['metadata']
    aggregates = data['aggregates']
    results = data['results']

    print("\n" + "="*80)
    print("EVALUATION ANALYSIS")
    print("="*80)

    print(f"\nTest Set: {metadata['test_set']}")
    print(f"Questions: {metadata['num_questions']}")
    print(f"Configuration: top_k={metadata['top_k']}, M1={metadata['use_m1']}, Rerank={metadata['use_rerank']}")

    # Overall metrics
    overall = aggregates['overall']
    print(f"\n{'OVERALL ACCURACY':-^80}")
    print(f"Correct: {overall['num_correct']}/{overall['num_questions']} ({overall['accuracy']*100:.1f}%)")

    # Failure analysis
    print(f"\n{'FAILURE ANALYSIS':-^80}")
    failures = analyze_failure_points(results)

    total = overall['num_questions']
    print(f"✓ Success: {failures['success']} ({failures['success']/total*100:.1f}%)")
    print(f"✗ Retrieval empty: {failures['retrieval_empty']} ({failures['retrieval_empty']/total*100:.1f}%)")
    print(f"✗ Retrieval no answer: {failures['retrieval_no_answer']} ({failures['retrieval_no_answer']/total*100:.1f}%)")
    print(f"✗ Extraction no facts: {failures['extraction_no_facts']} ({failures['extraction_no_facts']/total*100:.1f}%)")
    print(f"✗ Extraction wrong facts: {failures['extraction_wrong_facts']} ({failures['extraction_wrong_facts']/total*100:.1f}%)")
    print(f"✗ Generation poor: {failures['generation_poor']} ({failures['generation_poor']/total*100:.1f}%)")

    # Retrieval performance
    print(f"\n{'RETRIEVAL PERFORMANCE':-^80}")
    retrieval = aggregates['retrieval']
    print(f"Recall@5:  {retrieval['recall_at_5']*100:.1f}%  (answer in top 5 sentences)")
    print(f"Recall@10: {retrieval['recall_at_10']*100:.1f}%  (answer in top 10 sentences)")
    print(f"Recall@20: {retrieval['recall_at_20']*100:.1f}%  (answer in top 20 sentences)")
    print(f"MRR: {retrieval['mean_reciprocal_rank']:.3f}  (mean reciprocal rank)")

    # Extraction performance
    print(f"\n{'EXTRACTION PERFORMANCE':-^80}")
    extraction = aggregates['extraction']
    print(f"Facts extracted (avg): {extraction['facts_extracted_mean']:.1f}")
    print(f"Facts extracted (median): {extraction['facts_extracted_median']:.0f}")
    print(f"Facts selected (avg): {extraction['facts_selected_mean']:.1f}")
    print(f"Facts selected (median): {extraction['facts_selected_median']:.0f}")

    # Timing breakdown
    print(f"\n{'TIMING BREAKDOWN':-^80}")
    timing = aggregates['timing']
    print(f"Total time per question: {timing['total_time_mean']:.3f}s (±{timing['total_time_std']:.3f}s)")
    print(f"  Parse:      {timing['parse_time_mean']:.3f}s  ({timing['parse_time_mean']/timing['total_time_mean']*100:.1f}%)")
    print(f"  Retrieval:  {timing['retrieval_time_mean']:.3f}s  ({timing['retrieval_time_mean']/timing['total_time_mean']*100:.1f}%)")
    print(f"  Generation: {timing['generation_time_mean']:.3f}s  ({timing['generation_time_mean']/timing['total_time_mean']*100:.1f}%)")

    bottlenecks = analyze_timing_bottlenecks(results)
    print(f"\nBottlenecks:")
    for stage, pct in sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True):
        print(f"  {stage.capitalize()}: {pct:.1f}% of total time")

    # Question type breakdown
    print(f"\n{'PERFORMANCE BY QUESTION TYPE':-^80}")
    by_type = analyze_question_type_performance(results)

    print(f"{'Type':<12} {'Acc':<8} {'Retr':<8} {'Extr':<8} {'Time':<8} {'MRR':<8}")
    print("-" * 80)

    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        print(f"{qtype:<12} {stats['accuracy']*100:>6.1f}%  {stats['retrieval_recall']*100:>6.1f}%  "
              f"{stats['extraction_rate']*100:>6.1f}%  {stats['avg_retrieval_time']:>6.3f}s  {stats['avg_mrr']:>6.3f}")

    # Answer quality
    print(f"\n{'ANSWER QUALITY':-^80}")
    answer = aggregates['answer']
    print(f"Answer length (avg): {answer['answer_length_mean']:.0f} chars")
    print(f"Answer length (median): {answer['answer_length_median']:.0f} chars")
    print(f"Citations per answer (avg): {answer['citations_per_answer_mean']:.1f}")

    print("\n" + "="*80)


def compare_results(baseline_path: Path, improved_path: Path):
    """Compare two evaluation results to show improvements/regressions."""
    baseline = load_results(baseline_path)
    improved = load_results(improved_path)

    print("\n" + "="*80)
    print("COMPARISON: BASELINE vs IMPROVED")
    print("="*80)

    print(f"\nBaseline: {baseline_path.name}")
    print(f"Improved: {improved_path.name}")

    # Overall accuracy
    baseline_acc = baseline['aggregates']['overall']['accuracy']
    improved_acc = improved['aggregates']['overall']['accuracy']
    diff_acc = (improved_acc - baseline_acc) * 100

    print(f"\n{'OVERALL ACCURACY':-^80}")
    print(f"Baseline: {baseline_acc*100:.1f}%")
    print(f"Improved: {improved_acc*100:.1f}%")
    print(f"Change:   {diff_acc:+.1f}% {'✓' if diff_acc > 0 else '✗' if diff_acc < 0 else '='}")

    # Timing
    baseline_time = baseline['aggregates']['timing']['total_time_mean']
    improved_time = improved['aggregates']['timing']['total_time_mean']
    speedup = (baseline_time - improved_time) / baseline_time * 100

    print(f"\n{'TIMING':-^80}")
    print(f"Baseline: {baseline_time:.3f}s per question")
    print(f"Improved: {improved_time:.3f}s per question")
    print(f"Speedup:  {speedup:+.1f}% {'✓' if speedup > 0 else '✗' if speedup < 0 else '='}")

    # Retrieval quality
    baseline_mrr = baseline['aggregates']['retrieval']['mean_reciprocal_rank']
    improved_mrr = improved['aggregates']['retrieval']['mean_reciprocal_rank']
    diff_mrr = (improved_mrr - baseline_mrr)

    print(f"\n{'RETRIEVAL QUALITY (MRR)':-^80}")
    print(f"Baseline: {baseline_mrr:.3f}")
    print(f"Improved: {improved_mrr:.3f}")
    print(f"Change:   {diff_mrr:+.3f} {'✓' if diff_mrr > 0 else '✗' if diff_mrr < 0 else '='}")

    # By question type
    print(f"\n{'ACCURACY BY QUESTION TYPE':-^80}")
    print(f"{'Type':<12} {'Baseline':<10} {'Improved':<10} {'Change':<10}")
    print("-" * 80)

    baseline_by_type = baseline['aggregates']['by_question_type']
    improved_by_type = improved['aggregates']['by_question_type']

    all_types = set(baseline_by_type.keys()) | set(improved_by_type.keys())

    for qtype in sorted(all_types):
        baseline_acc = baseline_by_type.get(qtype, {}).get('accuracy', 0) * 100
        improved_acc = improved_by_type.get(qtype, {}).get('accuracy', 0) * 100
        diff = improved_acc - baseline_acc

        print(f"{qtype:<12} {baseline_acc:>8.1f}%  {improved_acc:>8.1f}%  {diff:>+8.1f}% "
              f"{'✓' if diff > 0 else '✗' if diff < 0 else '='}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('results', type=Path, help='Evaluation results JSON file')
    parser.add_argument('compare_with', type=Path, nargs='?', help='Second results file for comparison')

    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: File not found: {args.results}")
        return 1

    # Load and analyze results
    data = load_results(args.results)
    print_analysis(data)

    # Compare if second file provided
    if args.compare_with:
        if not args.compare_with.exists():
            print(f"Error: File not found: {args.compare_with}")
            return 1

        compare_results(args.results, args.compare_with)

    return 0


if __name__ == '__main__':
    sys.exit(main())
