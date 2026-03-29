#!/usr/bin/env python3
"""
Adaptive Full Evaluation Suite - Time-Constrained Testing

Automatically adjusts the number of questions to meet a time budget while
maintaining representative coverage across question types.

Key Features:
- Time-limited: Keeps suite under target time (default: 10 minutes)
- Stratified sampling: Ensures all question types represented
- Adaptive: Estimates time per question and adjusts sample size
- Progress tracking: Shows estimated time remaining

Usage:
    # Quick suite (10 minutes)
    python scripts/evaluate_full_suite_adaptive.py \
      --time-limit 10 \
      --output results/suite_quick/

    # Very quick (5 minutes)
    python scripts/evaluate_full_suite_adaptive.py \
      --time-limit 5 \
      --output results/suite_fast/

    # Use all questions (no limit)
    python scripts/evaluate_full_suite_adaptive.py \
      --no-limit \
      --output results/suite_full/
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import sys
from datetime import datetime
import random

def load_test_set(path: Path) -> List[Dict]:
    """Load test set from JSONL file."""
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def stratified_sample(questions: List[Dict], target_count: int) -> List[Dict]:
    """
    Sample questions while maintaining question type distribution.

    Ensures all question types are represented proportionally.
    Questions are randomized within each type.
    """
    # Group by question type
    by_type = {}
    for q in questions:
        qtype = q.get('question_type', 'OTHER')
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(q)

    # Calculate samples per type (proportional)
    total = len(questions)
    sampled = []

    for qtype, qs in sorted(by_type.items()):
        proportion = len(qs) / total
        target_for_type = max(1, int(target_count * proportion))  # At least 1 per type

        if target_for_type >= len(qs):
            # Use all questions of this type
            sampled.extend(qs)
        else:
            # Random sample
            sampled.extend(random.sample(qs, target_for_type))

    # If we're still under target, add more from largest groups
    if len(sampled) < target_count:
        remaining = target_count - len(sampled)
        unused = [q for q in questions if q not in sampled]
        sampled.extend(random.sample(unused, min(remaining, len(unused))))

    # Shuffle final sample so question types are mixed
    result = sampled[:target_count]
    random.shuffle(result)
    return result


def estimate_time_per_question(test_set: Path, sample_size: int = 5) -> float:
    """
    Estimate time per question by running quick sample.

    Uses first sample_size questions to estimate per-question time.
    """
    print(f"  Estimating time per question (sample size: {sample_size})...")

    temp_file = Path('/tmp/timing_test.jsonl')

    # Create sample test set
    questions = load_test_set(test_set)
    sample = questions[:sample_size]

    with open(temp_file, 'w') as f:
        for q in sample:
            f.write(json.dumps(q) + '\n')

    # Run quick evaluation
    output_file = Path('/tmp/timing_test_result.json')

    cmd = [
        'python', 'scripts/evaluate_pipeline_comprehensive.py',
        '--test-set', str(temp_file),
        '--top-k', '20',
        '--output', str(output_file),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)

        with open(output_file) as f:
            data = json.load(f)

        per_question_time = data['aggregates']['timing']['total_time_mean']

        # Cleanup
        temp_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)

        print(f"  ✓ Estimated: {per_question_time:.2f}s per question")
        return per_question_time

    except Exception as e:
        print(f"  ⚠️  Estimation failed, using default: 1.7s")
        return 1.7  # Fallback default


def calculate_optimal_sample_size(
    time_limit_minutes: int,
    per_question_time: float,
    num_evaluations: int = 8
) -> int:
    """Calculate optimal number of questions to fit time budget."""

    time_limit_seconds = time_limit_minutes * 60
    questions_per_eval = time_limit_seconds / (per_question_time * num_evaluations)

    # Round down to be safe
    optimal = int(questions_per_eval * 0.95)  # 5% buffer

    return max(10, optimal)  # Minimum 10 questions


def create_sampled_test_set(
    test_set: Path,
    output_path: Path,
    sample_size: int
) -> Path:
    """Create sampled test set file."""

    questions = load_test_set(test_set)

    if sample_size >= len(questions):
        # Use all questions
        print(f"  Using all {len(questions)} questions")
        return test_set

    # Stratified sample
    sampled = stratified_sample(questions, sample_size)

    # Write sampled test set
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for q in sampled:
            f.write(json.dumps(q) + '\n')

    # Show distribution
    by_type = {}
    for q in sampled:
        qtype = q.get('question_type', 'OTHER')
        by_type[qtype] = by_type.get(qtype, 0) + 1

    print(f"  Sampled {len(sampled)} questions:")
    for qtype in sorted(by_type.keys()):
        print(f"    {qtype}: {by_type[qtype]}")

    return output_path


def run_evaluation(
    top_k: int,
    output_file: Path,
    test_set: Path,
    use_m1: bool = True,
    use_rerank: bool = True,
    description: str = ""
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

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines[-15:]:
            if 'correct' in line.lower() or 'accuracy' in line.lower():
                print(f"    {line.strip()}")

        with open(output_file) as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Evaluation failed: {description}")
        print(e.stderr[-500:] if len(e.stderr) > 500 else e.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for all results')
    parser.add_argument('--test-set', type=Path, default=Path('data/test_sets/qa_test_diverse_30.jsonl'),
                       help='Test set to use')
    parser.add_argument('--time-limit', type=int, default=10,
                       help='Target time limit in minutes (default: 10)')
    parser.add_argument('--no-limit', action='store_true',
                       help='Use all questions regardless of time')
    parser.add_argument('--top-k-values', type=int, nargs='+', default=[5, 10, 20, 30, 50],
                       help='Top-K values to test')
    parser.add_argument('--skip-ablations', action='store_true',
                       help='Skip ablation tests')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate time, don\'t run suite')
    parser.add_argument('--seed', type=int, help='Random seed for question order (for reproducibility)')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable question order randomization')

    args = parser.parse_args()

    # Set random seed for reproducibility
    if not args.no_shuffle:
        if args.seed is not None:
            random.seed(args.seed)
            print(f"Using random seed: {args.seed}")
        else:
            # Use time-based seed for true randomization
            import time
            seed = int(time.time() * 1000) % (2**32)
            random.seed(seed)
            print(f"Using random seed: {seed}")
    else:
        print("Randomization disabled (--no-shuffle)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ADAPTIVE FULL EVALUATION SUITE")
    print("=" * 80)
    print()
    print(f"Output directory: {args.output}")
    print(f"Test set: {args.test_set}")
    print(f"Time limit: {args.time_limit} minutes" if not args.no_limit else "Time limit: None (use all questions)")
    print()

    # Step 1: Estimate time per question
    print("[1/4] Estimating time per question...")
    per_question_time = estimate_time_per_question(args.test_set, sample_size=5)

    # Step 2: Calculate optimal sample size
    num_evaluations = len(args.top_k_values) + 1  # top-k + baseline
    if not args.skip_ablations:
        num_evaluations += 2  # +M1 ablation +reranker ablation

    if args.no_limit:
        # Use all questions
        questions = load_test_set(args.test_set)
        sample_size = len(questions)
        estimated_time = (sample_size * per_question_time * num_evaluations) / 60
        print(f"\nUsing all {sample_size} questions")
        print(f"Estimated total time: {estimated_time:.1f} minutes")
        sampled_test_set = args.test_set
    else:
        sample_size = calculate_optimal_sample_size(
            args.time_limit,
            per_question_time,
            num_evaluations
        )

        estimated_time = (sample_size * per_question_time * num_evaluations) / 60

        print()
        print(f"[2/4] Calculating optimal sample size...")
        print(f"  Time budget: {args.time_limit} minutes")
        print(f"  Per-question time: {per_question_time:.2f}s")
        print(f"  Number of evaluations: {num_evaluations}")
        print(f"  Optimal sample size: {sample_size} questions")
        print(f"  Estimated total time: {estimated_time:.1f} minutes")
        print()

        if args.estimate_only:
            print("Estimate complete (--estimate-only flag set)")
            return 0

        # Create sampled test set
        print("[3/4] Creating stratified sample...")
        sampled_test_set = create_sampled_test_set(
            args.test_set,
            args.output / 'test_set_sampled.jsonl',
            sample_size
        )
        print()

    if args.estimate_only:
        return 0

    # Run full suite using sampled test set
    print(f"[{4 if not args.no_limit else 3}/4] Running evaluation suite...")
    print()

    # Import and use existing suite logic
    from evaluate_full_suite import run_evaluation, generate_suite_report

    # Run baseline
    print("  [1/N] Baseline evaluation...")
    baseline = run_evaluation(
        top_k=20,
        output_file=args.output / 'baseline.json',
        test_set=sampled_test_set,
        use_m1=True,
        use_rerank=True,
        description="Baseline (top_k=20, M1=True, Rerank=True)"
    )

    if baseline is None:
        print("✗ Baseline evaluation failed, aborting")
        return 1

    # Run top-k sweep
    print()
    top_k_results = {}
    for i, k in enumerate(args.top_k_values, start=2):
        print(f"  [{i}/N] Top-k={k} evaluation...")
        result = run_evaluation(
            top_k=k,
            output_file=args.output / f'top_k_{k}.json',
            test_set=sampled_test_set,
            use_m1=True,
            use_rerank=True,
            description=f"top_k={k}"
        )
        top_k_results[k] = result

    # Run ablations
    ablations = {}
    if not args.skip_ablations:
        print()
        print(f"  [N-1/N] M1 ablation...")
        ablations['no_m1'] = run_evaluation(
            top_k=20,
            output_file=args.output / 'ablation_no_m1.json',
            test_set=sampled_test_set,
            use_m1=False,
            use_rerank=True,
            description="Without M1 filter"
        )

        print()
        print(f"  [N/N] Reranker ablation...")
        ablations['no_rerank'] = run_evaluation(
            top_k=20,
            output_file=args.output / 'ablation_no_rerank.json',
            test_set=sampled_test_set,
            use_m1=True,
            use_rerank=False,
            description="Without neural reranker"
        )

    # Generate report
    print()
    print("[4/4] Generating comprehensive report...")
    generate_suite_report(baseline, top_k_results, ablations, args.output)

    print()
    print("=" * 80)
    print("SUITE COMPLETE")
    print("=" * 80)
    print()
    print(f"Sample size: {sample_size} questions")
    print(f"Estimated time: {estimated_time:.1f} minutes")
    print(f"All results saved to: {args.output}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
