#!/usr/bin/env python3
"""
Experiment: Optimal Top-K Sentence Count for Answer Generation

Tests different top-k values (5, 10, 20, 30, 50, 100) to find optimal number
of sentences to use for answer generation.

Key Questions:
1. Does more sentences = better accuracy? (or add noise?)
2. What's the sweet spot for accuracy vs speed?
3. How does top-k affect extraction stats and M1 filtering?
4. Does optimal top-k differ by question type?

Usage:
    python scripts/experiment_top_k_optimization.py --output results/top_k_experiment/

Output:
    - JSON results for each top-k value
    - Comparison CSV showing accuracy, timing, extraction stats
    - Visualization of accuracy vs top-k curve
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import pandas as pd
import sys

def run_evaluation(top_k: int, output_dir: Path, test_set: Path) -> Dict:
    """Run comprehensive evaluation with specific top-k value."""
    output_file = output_dir / f"top_k_{top_k}.json"

    if output_file.exists():
        print(f"✓ Results for top_k={top_k} already exist, loading...")
        with open(output_file) as f:
            return json.load(f)

    print(f"\n{'='*80}")
    print(f"Running evaluation with top_k={top_k}")
    print(f"{'='*80}")

    cmd = [
        'python', 'scripts/evaluate_pipeline_comprehensive.py',
        '--test-set', str(test_set),
        '--top-k', str(top_k),
        '--output', str(output_file),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Load and return results
        with open(output_file) as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for top_k={top_k}")
        print(e.stderr)
        return None


def compare_results(results_by_k: Dict[int, Dict], output_dir: Path):
    """Generate comparison analysis and visualization."""

    # Extract key metrics
    comparison_data = []

    for k, data in sorted(results_by_k.items()):
        if data is None:
            continue

        agg = data['aggregates']

        row = {
            'top_k': k,
            'accuracy': agg['overall']['accuracy'] * 100,
            'num_correct': agg['overall']['num_correct'],
            'total_time_mean': agg['timing']['total_time_mean'],
            'retrieval_time_mean': agg['timing']['retrieval_time_mean'],
            'generation_time_mean': agg['timing']['generation_time_mean'],
            'facts_extracted_mean': agg['extraction']['facts_extracted_mean'],
            'facts_selected_mean': agg['extraction']['facts_selected_mean'],
            'recall_at_5': agg['retrieval']['recall_at_5'] * 100,
            'recall_at_10': agg['retrieval']['recall_at_10'] * 100,
            'recall_at_20': agg['retrieval']['recall_at_20'] * 100 if k >= 20 else None,
            'mean_reciprocal_rank': agg['retrieval']['mean_reciprocal_rank'],
        }

        # Add question type breakdown
        for qtype, stats in agg['by_question_type'].items():
            row[f'{qtype}_accuracy'] = stats['accuracy'] * 100

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Save CSV
    csv_path = output_dir / 'top_k_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison CSV saved to: {csv_path}")

    # Print summary table
    print("\n" + "="*80)
    print("TOP-K OPTIMIZATION RESULTS")
    print("="*80)
    print()

    # Overall metrics
    print("Overall Performance:")
    print(f"{'Top-K':<10} {'Accuracy':<12} {'Correct':<10} {'Total Time':<12} {'Facts Extracted':<18} {'Facts Selected':<16}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['top_k']:<10} {row['accuracy']:>10.1f}%  {row['num_correct']:>8}/50  "
              f"{row['total_time_mean']:>10.3f}s  {row['facts_extracted_mean']:>16.1f}  "
              f"{row['facts_selected_mean']:>14.1f}")

    print()
    print("Retrieval Quality:")
    print(f"{'Top-K':<10} {'Recall@5':<12} {'Recall@10':<12} {'Recall@20':<12} {'MRR':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        r20 = f"{row['recall_at_20']:.1f}%" if row['recall_at_20'] is not None else "N/A"
        print(f"{row['top_k']:<10} {row['recall_at_5']:>10.1f}%  {row['recall_at_10']:>10.1f}%  "
              f"{r20:>10}  {row['mean_reciprocal_rank']:>8.3f}")

    # Find optimal
    best_accuracy = df.loc[df['accuracy'].idxmax()]
    fastest = df.loc[df['total_time_mean'].idxmin()]
    best_value = df.assign(value_score=lambda x: x['accuracy'] / x['total_time_mean']).loc[
        lambda x: x['value_score'].idxmax()
    ]

    print()
    print("="*80)
    print("OPTIMIZATION FINDINGS")
    print("="*80)
    print()
    print(f"Best Accuracy: top_k={int(best_accuracy['top_k'])} ({best_accuracy['accuracy']:.1f}%)")
    print(f"Fastest: top_k={int(fastest['top_k'])} ({fastest['total_time_mean']:.3f}s)")
    print(f"Best Value (accuracy/time): top_k={int(best_value['top_k'])} ({best_value['accuracy']:.1f}% in {best_value['total_time_mean']:.3f}s)")
    print()

    # Analyze diminishing returns
    print("Diminishing Returns Analysis:")
    print("-" * 80)
    prev_acc = None
    for _, row in df.iterrows():
        if prev_acc is not None:
            gain = row['accuracy'] - prev_acc
            cost = row['total_time_mean'] - prev_time
            if gain > 0:
                efficiency = gain / cost
                print(f"top_k {int(prev_k)}→{int(row['top_k'])}: +{gain:.1f}% accuracy, +{cost:.3f}s time (efficiency: {efficiency:.2f}%/s)")
            elif gain < 0:
                print(f"top_k {int(prev_k)}→{int(row['top_k'])}: {gain:.1f}% accuracy (REGRESSION), +{cost:.3f}s time")
            else:
                print(f"top_k {int(prev_k)}→{int(row['top_k'])}: No change in accuracy, +{cost:.3f}s time")
        prev_acc = row['accuracy']
        prev_time = row['total_time_mean']
        prev_k = row['top_k']

    print()

    # Noise analysis
    print("Noise Analysis (Are more sentences adding noise?):")
    print("-" * 80)
    for _, row in df.iterrows():
        filter_rate = (row['facts_extracted_mean'] - row['facts_selected_mean']) / row['facts_extracted_mean'] * 100
        print(f"top_k={int(row['top_k'])}: Extract {row['facts_extracted_mean']:.1f} facts → "
              f"Select {row['facts_selected_mean']:.1f} facts ({filter_rate:.1f}% filtered)")
    print()

    # Question type analysis
    print("Optimal Top-K by Question Type:")
    print("-" * 80)
    question_types = [col.replace('_accuracy', '') for col in df.columns if col.endswith('_accuracy')]
    for qtype in question_types:
        col = f'{qtype}_accuracy'
        if col in df.columns:
            best_k_for_type = df.loc[df[col].idxmax()]
            print(f"{qtype:<10}: top_k={int(best_k_for_type['top_k'])} ({best_k_for_type[col]:.1f}%)")

    print()
    print("="*80)
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)

    if best_accuracy['top_k'] == df['top_k'].min():
        print("⚠️  Lowest top_k tested has best accuracy!")
        print(f"   → Test even lower values (e.g., {int(df['top_k'].min())//2})")
    elif best_accuracy['top_k'] == df['top_k'].max():
        print("⚠️  Highest top_k tested has best accuracy!")
        print(f"   → Test even higher values (e.g., {int(df['top_k'].max())*2})")
    else:
        print(f"✓ Optimal top_k found: {int(best_accuracy['top_k'])}")
        print(f"  → Accuracy: {best_accuracy['accuracy']:.1f}%")
        print(f"  → Time: {best_accuracy['total_time_mean']:.3f}s")

    print()

    # Value vs accuracy tradeoff
    if best_value['top_k'] != best_accuracy['top_k']:
        print(f"💡 For best value (speed vs accuracy tradeoff): use top_k={int(best_value['top_k'])}")
        print(f"   → Accuracy: {best_value['accuracy']:.1f}% (vs {best_accuracy['accuracy']:.1f}% at best)")
        print(f"   → Time: {best_value['total_time_mean']:.3f}s (vs {best_accuracy['total_time_mean']:.3f}s at best)")
        acc_loss = best_accuracy['accuracy'] - best_value['accuracy']
        time_save = best_accuracy['total_time_mean'] - best_value['total_time_mean']
        print(f"   → Trade: -{acc_loss:.1f}% accuracy for -{time_save:.3f}s time")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', type=Path, default=Path('results/top_k_experiment'),
                       help='Output directory for results')
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_set_50.jsonl'),
                       help='Test set to use')
    parser.add_argument('--top-k-values', type=int, nargs='+', default=[5, 10, 20, 30, 50, 100],
                       help='Top-K values to test (default: 5 10 20 30 50 100)')

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TOP-K OPTIMIZATION EXPERIMENT")
    print("="*80)
    print()
    print(f"Testing top-k values: {args.top_k_values}")
    print(f"Test set: {args.test_set}")
    print(f"Output directory: {args.output}")
    print()

    # Run evaluations for each top-k value
    results_by_k = {}

    for k in args.top_k_values:
        results = run_evaluation(k, args.output, args.test_set)
        results_by_k[k] = results

    # Compare results
    compare_results(results_by_k, args.output)

    print(f"\n✓ All results saved to: {args.output}")
    print(f"✓ Comparison CSV: {args.output / 'top_k_comparison.csv'}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
