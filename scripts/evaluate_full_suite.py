#!/usr/bin/env python3
"""
Full Evaluation Suite - Comprehensive Pipeline Assessment

Runs complete evaluation suite including:
1. Baseline evaluation (current configuration)
2. Top-K optimization sweep (5, 10, 20, 30, 50)
3. M1 ablation test (with/without M1 filtering)
4. Reranker ablation test (with/without reranking)
5. Comparative analysis and recommendations

This provides full context for interpreting metrics and identifying bottlenecks.

Usage:
    # Full suite (all tests, ~45 minutes)
    python scripts/evaluate_full_suite.py --output results/full_suite_$(date +%Y%m%d)/

    # Skip ablations (just baseline + top-k sweep, ~20 minutes)
    python scripts/evaluate_full_suite.py --skip-ablations --output results/suite_quick/

    # Custom top-k values
    python scripts/evaluate_full_suite.py --top-k-values 5 15 30 --output results/custom/
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import sys
from datetime import datetime

def run_evaluation(
    top_k: int,
    output_file: Path,
    test_set: Path,
    use_m1: bool = True,
    use_rerank: bool = True,
    description: str = "",
    seed: Optional[int] = None,
    no_shuffle: bool = False
) -> Optional[Dict]:
    """Run single evaluation with specific configuration."""

    if output_file.exists():
        print(f"  ✓ Results exist, loading from {output_file.name}")
        with open(output_file) as f:
            return json.load(f)

    print(f"  Running: {description}")

    cmd = [
        'python', 'scripts/evaluate_pipeline_comprehensive.py',
        '--test-set', str(test_set),
        '--top-k', str(top_k),
        '--output', str(output_file),
    ]

    if not use_m1:
        cmd.append('--no-m1')
    if not use_rerank:
        cmd.append('--no-rerank')
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    if no_shuffle:
        cmd.append('--no-shuffle')

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Only show last 10 lines to avoid clutter
        lines = result.stdout.split('\n')
        for line in lines[-10:]:
            if line.strip():
                print(f"    {line}")

        with open(output_file) as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Evaluation failed: {description}")
        print(e.stderr)
        return None


def generate_suite_report(
    baseline: Dict,
    top_k_results: Dict[int, Dict],
    ablations: Dict[str, Dict],
    output_dir: Path
):
    """Generate comprehensive analysis report."""

    report = []
    report.append("=" * 80)
    report.append("FULL EVALUATION SUITE REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Test Set: {baseline['metadata']['test_set']}")
    report.append("")

    # Baseline results
    report.append("=" * 80)
    report.append("BASELINE RESULTS (Current Configuration)")
    report.append("=" * 80)
    report.append("")

    agg = baseline['aggregates']
    report.append(f"Accuracy: {agg['overall']['accuracy']*100:.1f}% ({agg['overall']['num_correct']}/50)")
    report.append(f"Configuration: top_k={baseline['metadata']['top_k']}, M1={baseline['metadata']['use_m1']}, Rerank={baseline['metadata']['use_rerank']}")
    report.append("")

    report.append("By Question Type:")
    for qtype, stats in agg['by_question_type'].items():
        report.append(f"  {qtype:<12} {stats['accuracy']*100:>6.1f}%")
    report.append("")

    report.append("Timing:")
    report.append(f"  Total: {agg['timing']['total_time_mean']:.3f}s")
    report.append(f"  Retrieval: {agg['timing']['retrieval_time_mean']:.3f}s ({agg['timing']['retrieval_time_mean']/agg['timing']['total_time_mean']*100:.1f}%)")
    report.append(f"  Generation: {agg['timing']['generation_time_mean']:.3f}s ({agg['timing']['generation_time_mean']/agg['timing']['total_time_mean']*100:.1f}%)")
    report.append("")

    report.append("Retrieval Quality:")
    report.append(f"  Recall@5:  {agg['retrieval']['recall_at_5']*100:.1f}%")
    report.append(f"  Recall@10: {agg['retrieval']['recall_at_10']*100:.1f}%")
    report.append(f"  Recall@20: {agg['retrieval']['recall_at_20']*100:.1f}%")
    report.append(f"  MRR: {agg['retrieval']['mean_reciprocal_rank']:.3f}")
    report.append("")

    report.append("Extraction:")
    report.append(f"  Facts extracted (avg): {agg['extraction']['facts_extracted_mean']:.1f}")
    report.append(f"  Facts selected (avg): {agg['extraction']['facts_selected_mean']:.1f}")
    filter_rate = (agg['extraction']['facts_extracted_mean'] - agg['extraction']['facts_selected_mean']) / agg['extraction']['facts_extracted_mean'] * 100
    report.append(f"  Filter rate: {filter_rate:.1f}%")
    report.append("")

    # Top-K optimization results
    if top_k_results:
        report.append("=" * 80)
        report.append("TOP-K OPTIMIZATION SWEEP")
        report.append("=" * 80)
        report.append("")

        report.append(f"{'Top-K':<8} {'Accuracy':<10} {'Time':<8} {'Facts Extr':<12} {'Facts Sel':<12} {'Filter %':<10}")
        report.append("-" * 80)

        accuracies = []
        best_k = None
        best_acc = 0

        for k in sorted(top_k_results.keys()):
            if top_k_results[k] is None:
                continue

            agg = top_k_results[k]['aggregates']
            acc = agg['overall']['accuracy'] * 100
            accuracies.append((k, acc))

            if acc > best_acc:
                best_acc = acc
                best_k = k

            time = agg['timing']['total_time_mean']
            facts_extr = agg['extraction']['facts_extracted_mean']
            facts_sel = agg['extraction']['facts_selected_mean']
            filter_pct = (facts_extr - facts_sel) / facts_extr * 100 if facts_extr > 0 else 0

            marker = " ✓" if k == best_k else ""
            marker += " ←" if k == baseline['metadata']['top_k'] else ""

            report.append(f"{k:<8} {acc:>8.1f}%  {time:>6.3f}s  {facts_extr:>10.1f}  {facts_sel:>10.1f}  {filter_pct:>8.1f}%{marker}")

        report.append("")

        # Analyze curve shape
        report.append("Analysis:")
        if len(accuracies) >= 3:
            # Check if plateau
            early_acc = accuracies[0][1]
            mid_acc = accuracies[len(accuracies)//2][1]
            late_acc = accuracies[-1][1]

            if abs(late_acc - mid_acc) < 2.0 and abs(mid_acc - early_acc) > 5.0:
                report.append("  📊 PLATEAU PATTERN: Accuracy plateaus after initial increase")
                report.append("  → Diagnosis: Answer usually in first few sentences")
                report.append("  → Bottleneck: EXTRACTION (we have the answer but don't extract it correctly)")
                report.append("  → Recommendation: Fix extraction patterns")
            elif late_acc > early_acc + 5.0:
                report.append("  📈 LINEAR GROWTH: Accuracy increases with more sentences")
                report.append("  → Diagnosis: Answer often ranked deep (position 20-50)")
                report.append("  → Bottleneck: RANKING (answer not in top sentences)")
                report.append("  → Recommendation: Improve reranking or query expansion")
            elif late_acc < mid_acc - 2.0:
                report.append("  📉 PEAK THEN DECLINE: Accuracy drops at high top-k")
                report.append("  → Diagnosis: Adding noise at high top-k")
                report.append("  → Bottleneck: M1 FILTERING (overwhelmed by noise)")
                report.append("  → Recommendation: Improve M1 filter or extraction patterns")
            else:
                report.append("  📊 FLAT: No significant change across top-k values")
                report.append("  → Diagnosis: Retrieval quality is independent of ranking depth")
                report.append("  → Bottleneck: RETRIEVAL (answer not retrieved at all)")
                report.append("  → Recommendation: Drastically improve query expansion")

        report.append("")
        report.append(f"Optimal top-k: {best_k} ({best_acc:.1f}% accuracy)")

        # Noise analysis
        report.append("")
        report.append("Noise Analysis:")
        for k in sorted(top_k_results.keys()):
            if top_k_results[k] is None:
                continue
            agg = top_k_results[k]['aggregates']
            extr = agg['extraction']['facts_extracted_mean']
            sel = agg['extraction']['facts_selected_mean']
            rate = (extr - sel) / extr * 100 if extr > 0 else 0
            report.append(f"  top_k={k}: M1 filters {rate:.1f}% of extracted facts")

        # Check if filter rate increases with top-k
        filter_rates = [(k, (top_k_results[k]['aggregates']['extraction']['facts_extracted_mean'] -
                             top_k_results[k]['aggregates']['extraction']['facts_selected_mean']) /
                            top_k_results[k]['aggregates']['extraction']['facts_extracted_mean'] * 100)
                       for k in sorted(top_k_results.keys()) if top_k_results[k]]

        if len(filter_rates) >= 3:
            if filter_rates[-1][1] > filter_rates[0][1] + 5.0:
                report.append("")
                report.append("  ✓ Filter rate increases with top-k (M1 handling noise well)")
            elif abs(filter_rates[-1][1] - filter_rates[0][1]) < 2.0:
                report.append("")
                report.append("  ⚠️  Filter rate constant (signal-to-noise ratio unchanged)")

        report.append("")

    # Ablation tests
    if ablations:
        report.append("=" * 80)
        report.append("ABLATION TESTS")
        report.append("=" * 80)
        report.append("")

        baseline_acc = baseline['aggregates']['overall']['accuracy'] * 100

        if 'no_m1' in ablations:
            no_m1_acc = ablations['no_m1']['aggregates']['overall']['accuracy'] * 100
            diff = no_m1_acc - baseline_acc

            report.append(f"M1 Filter:")
            report.append(f"  With M1:    {baseline_acc:.1f}%")
            report.append(f"  Without M1: {no_m1_acc:.1f}%")
            report.append(f"  Difference: {diff:+.1f}%")

            if diff > 3.0:
                report.append("  ✗ M1 is HURTING accuracy (over-filtering correct facts)")
                report.append("  → Recommendation: Relax M1 threshold or retrain")
            elif diff < -3.0:
                report.append("  ✓ M1 is HELPING accuracy (filtering noise effectively)")
                report.append("  → Recommendation: Keep M1 enabled")
            else:
                report.append("  = M1 has minimal impact on accuracy")
                report.append("  → Recommendation: M1 is neutral, keep for quality control")
            report.append("")

        if 'no_rerank' in ablations:
            no_rerank_acc = ablations['no_rerank']['aggregates']['overall']['accuracy'] * 100
            diff = no_rerank_acc - baseline_acc

            report.append(f"Neural Reranker:")
            report.append(f"  With reranker:    {baseline_acc:.1f}%")
            report.append(f"  Without reranker: {no_rerank_acc:.1f}%")
            report.append(f"  Difference: {diff:+.1f}%")

            if diff > 3.0:
                report.append("  ✗ Reranker is HURTING accuracy (ranking wrong sentences high)")
                report.append("  → Recommendation: Retrain reranker or use BM25 only")
            elif diff < -3.0:
                report.append("  ✓ Reranker is HELPING accuracy (improving ranking)")
                report.append("  → Recommendation: Keep reranker enabled")
            else:
                report.append("  = Reranker has minimal impact on accuracy")
                report.append("  → Recommendation: Consider disabling for speed")
            report.append("")

    # Overall recommendations
    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")

    # Based on all data, generate recommendations
    if top_k_results:
        current_k = baseline['metadata']['top_k']
        best_k = max(top_k_results.keys(),
                     key=lambda k: top_k_results[k]['aggregates']['overall']['accuracy'] if top_k_results[k] else 0)

        if best_k != current_k:
            best_acc = top_k_results[best_k]['aggregates']['overall']['accuracy'] * 100
            current_acc = baseline['aggregates']['overall']['accuracy'] * 100
            improvement = best_acc - current_acc

            report.append(f"1. CHANGE TOP-K: Use top_k={best_k} instead of {current_k}")
            report.append(f"   Expected improvement: {improvement:+.1f}% accuracy")
            report.append("")

    # Identify bottleneck from failure analysis
    results = baseline['results']
    failures_with_answer = sum(1 for r in results if not r['success'] and r['retrieval']['contains_answer'])
    failures_no_answer = sum(1 for r in results if not r['success'] and not r['retrieval']['contains_answer'])

    if failures_with_answer > failures_no_answer:
        report.append("2. PRIMARY BOTTLENECK: EXTRACTION")
        report.append(f"   {failures_with_answer} questions fail even with answer in retrieved set")
        report.append("   → Recommendation: Fix extraction patterns (object verification, definition patterns)")
        report.append("")
    else:
        report.append("2. PRIMARY BOTTLENECK: RETRIEVAL")
        report.append(f"   {failures_no_answer} questions fail because answer not retrieved")
        report.append("   → Recommendation: Improve query expansion (temporal, person, causal synonyms)")
        report.append("")

    # Timing recommendations
    retrieval_pct = baseline['aggregates']['timing']['retrieval_time_mean'] / baseline['aggregates']['timing']['total_time_mean'] * 100
    if retrieval_pct > 70:
        report.append("3. PERFORMANCE OPTIMIZATION: RETRIEVAL")
        report.append(f"   Retrieval consumes {retrieval_pct:.1f}% of total time")
        report.append("   → Recommendation: Cache Kuzu queries or optimize indexes")
        report.append("")

    report.append("=" * 80)

    # Save report
    report_path = output_dir / 'SUITE_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print()
    print(f"✓ Full report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for all results')
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_diverse_30.jsonl'),
                       help='Test set to use')
    parser.add_argument('--top-k-values', type=int, nargs='+', default=[5, 10, 20, 30, 50],
                       help='Top-K values to test (default: 5 10 20 30 50)')
    parser.add_argument('--skip-ablations', action='store_true',
                       help='Skip ablation tests (faster)')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run only baseline evaluation (fastest)')
    parser.add_argument('--seed', type=int, help='Random seed for question order (for reproducibility)')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable question order randomization')

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FULL EVALUATION SUITE")
    print("=" * 80)
    print()
    print(f"Output directory: {args.output}")
    print(f"Test set: {args.test_set}")
    print()

    # Run baseline evaluation
    print("[1/4] Running baseline evaluation...")
    baseline = run_evaluation(
        top_k=20,
        output_file=args.output / 'baseline.json',
        test_set=args.test_set,
        use_m1=True,
        use_rerank=True,
        description="Baseline (top_k=20, M1=True, Rerank=True)",
        seed=args.seed,
        no_shuffle=args.no_shuffle
    )

    if baseline is None:
        print("✗ Baseline evaluation failed, aborting")
        return 1

    if args.baseline_only:
        print(f"\n✓ Baseline complete: {baseline['aggregates']['overall']['accuracy']*100:.1f}% accuracy")
        return 0

    # Run top-k sweep
    print()
    print("[2/4] Running top-k optimization sweep...")
    top_k_results = {}

    for k in args.top_k_values:
        result = run_evaluation(
            top_k=k,
            output_file=args.output / f'top_k_{k}.json',
            test_set=args.test_set,
            use_m1=True,
            use_rerank=True,
            description=f"top_k={k}",
            seed=args.seed,
            no_shuffle=args.no_shuffle
        )
        top_k_results[k] = result

    # Run ablation tests
    ablations = {}

    if not args.skip_ablations:
        print()
        print("[3/4] Running ablation tests...")

        # M1 ablation
        ablations['no_m1'] = run_evaluation(
            top_k=20,
            output_file=args.output / 'ablation_no_m1.json',
            test_set=args.test_set,
            use_m1=False,
            use_rerank=True,
            description="Without M1 filter",
            seed=args.seed,
            no_shuffle=args.no_shuffle
        )

        # Reranker ablation
        ablations['no_rerank'] = run_evaluation(
            top_k=20,
            output_file=args.output / 'ablation_no_rerank.json',
            test_set=args.test_set,
            use_m1=True,
            use_rerank=False,
            description="Without neural reranker",
            seed=args.seed,
            no_shuffle=args.no_shuffle
        )
    else:
        print()
        print("[3/4] Skipping ablation tests (--skip-ablations)")

    # Generate comprehensive report
    print()
    print("[4/4] Generating comprehensive report...")
    generate_suite_report(baseline, top_k_results, ablations, args.output)

    print()
    print("=" * 80)
    print("SUITE COMPLETE")
    print("=" * 80)
    print()
    print(f"All results saved to: {args.output}")
    print(f"  - baseline.json")
    print(f"  - top_k_*.json")
    if ablations:
        print(f"  - ablation_*.json")
    print(f"  - SUITE_REPORT.txt (comprehensive analysis)")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
