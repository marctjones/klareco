#!/usr/bin/env python3
"""
Track RAG Evaluation Progress Over Time

Creates timestamped snapshots of evaluation results and compares progress.

Usage:
    # Create new snapshot from current results
    python scripts/track_evaluation_progress.py --save

    # Compare all snapshots
    python scripts/track_evaluation_progress.py --compare

    # Show current metrics
    python scripts/track_evaluation_progress.py

    # Save with custom name
    python scripts/track_evaluation_progress.py --save --name "after_extraction_fix"
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def load_results(results_file: Path) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute summary metrics from results."""
    total = len(results)
    if total == 0:
        return {}

    # Binary metrics
    correct = sum(1 for r in results if r.get('is_correct', False))
    partial = sum(1 for r in results if r.get('is_partial', False))
    errors = sum(1 for r in results if 'error' in r)
    incorrect = total - correct - partial - errors

    # Granular metrics
    # Handle both dict and float formats
    retrieval_scores = []
    extraction_scores = []
    alignment_scores = []
    robustness_scores = []
    combined_scores = []

    for r in results:
        granular = r.get('granular_score', {})
        if isinstance(granular, dict):
            # New format: granular_score is a dict
            retrieval_scores.append(granular.get('retrieval_score', 0.0))
            extraction_scores.append(granular.get('extraction_score', 0.0))
            alignment_scores.append(granular.get('alignment_score', 0.0))
            robustness_scores.append(granular.get('robustness_score', 0.0))
            combined_scores.append(granular.get('combined_score', 0.0))
        else:
            # Old format: granular_score is a float
            combined_scores.append(granular)
            # Try to get from granular_components
            components = r.get('granular_components', {})
            retrieval_scores.append(components.get('retrieval_score', 0.0))
            extraction_scores.append(components.get('extraction_score', 0.0))
            alignment_scores.append(components.get('alignment_score', 0.0))
            robustness_scores.append(components.get('robustness_score', 0.0))

    avg_granular = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0

    # Retrieval metrics
    answer_in_top_1 = sum(1 for r in results
                          if r.get('quality_metrics', {}).get('answer_in_top_1', False))
    answer_in_top_3 = sum(1 for r in results
                          if r.get('quality_metrics', {}).get('answer_in_top_3', False))
    answer_in_top_10 = sum(1 for r in results
                           if r.get('quality_metrics', {}).get('answer_in_retrieved', False))

    # Extraction metrics
    extraction_success = sum(1 for r in results
                            if r.get('extraction_method') not in [None, 'fulltext_fallback'])
    exact_matches = sum(1 for r in results
                       if r.get('pipeline_diagnostics', {})
                         .get('extraction', {})
                         .get('metrics', {})
                         .get('match_type') == 'exact')
    fuzzy_matches = sum(1 for r in results
                       if r.get('pipeline_diagnostics', {})
                         .get('extraction', {})
                         .get('metrics', {})
                         .get('match_type') == 'fuzzy')

    return {
        'timestamp': datetime.now().isoformat(),
        'total_questions': total,
        'binary': {
            'correct': correct,
            'partial': partial,
            'incorrect': incorrect,
            'errors': errors,
            'accuracy': correct / total,
            'partial_accuracy': (correct + partial) / total,
        },
        'granular': {
            'overall_score': avg_granular,
            'retrieval': sum(retrieval_scores) / len(retrieval_scores),
            'extraction': sum(extraction_scores) / len(extraction_scores),
            'alignment': sum(alignment_scores) / len(alignment_scores),
            'robustness': sum(robustness_scores) / len(robustness_scores),
        },
        'retrieval': {
            'answer_in_top_1': answer_in_top_1,
            'answer_in_top_3': answer_in_top_3,
            'answer_in_top_10': answer_in_top_10,
            'top_1_rate': answer_in_top_1 / total,
            'top_3_rate': answer_in_top_3 / total,
            'top_10_rate': answer_in_top_10 / total,
        },
        'extraction': {
            'successful': extraction_success,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'success_rate': extraction_success / total,
            'exact_rate': exact_matches / total,
            'fuzzy_rate': fuzzy_matches / total,
        },
    }


def save_snapshot(metrics: Dict, name: str = None):
    """Save metrics snapshot with timestamp."""
    snapshots_dir = Path('data/evaluation/snapshots')
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # Prepend timestamp to custom name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"{timestamp}_{name}"

    output_file = snapshots_dir / f'{name}.json'

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Snapshot saved: {output_file}")
    return output_file


def load_snapshots() -> List[Tuple[str, Dict]]:
    """Load all snapshots sorted by timestamp."""
    snapshots_dir = Path('data/evaluation/snapshots')
    if not snapshots_dir.exists():
        return []

    snapshots = []
    for f in sorted(snapshots_dir.glob('*.json')):
        with open(f, 'r') as fp:
            metrics = json.load(fp)
            snapshots.append((f.stem, metrics))

    return snapshots


def print_metrics(metrics: Dict, label: str = "Current"):
    """Pretty print metrics."""
    print(f"\n{'='*80}")
    print(f"{label} RAG Evaluation Metrics")
    print(f"{'='*80}")
    print(f"Timestamp: {metrics.get('timestamp', 'unknown')}")
    print(f"Questions: {metrics['total_questions']}")

    print(f"\n📊 Binary Accuracy:")
    binary = metrics['binary']
    total = metrics['total_questions']
    print(f"  ✓ Correct:   {binary['correct']:2d}/{total} ({binary['accuracy']*100:5.1f}%)")
    print(f"  ⚠ Partial:   {binary['partial']:2d}/{total} ({(binary['partial']/total)*100:5.1f}%)")
    print(f"  ✗ Incorrect: {binary['incorrect']:2d}/{total} ({(binary['incorrect']/total)*100:5.1f}%)")
    if binary['errors'] > 0:
        print(f"  ❌ Errors:    {binary['errors']:2d}/{total}")

    print(f"\n📈 Granular Scores (weighted: 40%R + 30%E + 20%A + 10%B):")
    granular = metrics['granular']
    print(f"  Overall:    {granular['overall_score']:.3f} / 1.000")
    print(f"  Retrieval:  {granular['retrieval']:.3f} (40% weight)")
    print(f"  Extraction: {granular['extraction']:.3f} (30% weight)")
    print(f"  Alignment:  {granular['alignment']:.3f} (20% weight)")
    print(f"  Robustness: {granular['robustness']:.3f} (10% weight)")

    print(f"\n🔍 Retrieval Breakdown:")
    retrieval = metrics['retrieval']
    print(f"  Top-1:  {retrieval['answer_in_top_1']:2d}/{total} ({retrieval['top_1_rate']*100:5.1f}%)")
    print(f"  Top-3:  {retrieval['answer_in_top_3']:2d}/{total} ({retrieval['top_3_rate']*100:5.1f}%)")
    print(f"  Top-10: {retrieval['answer_in_top_10']:2d}/{total} ({retrieval['top_10_rate']*100:5.1f}%)")

    print(f"\n✂️  Extraction Breakdown:")
    extraction = metrics['extraction']
    print(f"  Successful: {extraction['successful']:2d}/{total} ({extraction['success_rate']*100:5.1f}%)")
    print(f"  Exact:      {extraction['exact_matches']:2d}/{total} ({extraction['exact_rate']*100:5.1f}%)")
    print(f"  Fuzzy:      {extraction['fuzzy_matches']:2d}/{total} ({extraction['fuzzy_rate']*100:5.1f}%)")
    print(f"{'='*80}\n")


def compare_snapshots(snapshots: List[Tuple[str, Dict]]):
    """Show comparison table of snapshots."""
    if len(snapshots) < 2:
        print("Need at least 2 snapshots to compare")
        return

    print(f"\n{'='*100}")
    print("Evaluation Progress Over Time")
    print(f"{'='*100}")

    # Header with snapshot names
    print(f"{'Metric':<30} ", end='')
    recent = snapshots[-5:]  # Show last 5
    for name, _ in recent:
        # Extract timestamp or custom name
        parts = name.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            label = f"{parts[0][:8]}"  # Date only
            if len(parts) > 2:
                label += f"_{parts[2][:8]}"  # Add custom suffix
        else:
            label = name[:16]
        print(f"{label:>16}", end=' ')
    print()
    print('-' * 100)

    # Granular scores
    print(f"{'Granular Score':<30} ", end='')
    for _, m in recent:
        print(f"{m['granular']['overall_score']:>16.3f}", end=' ')
    print()

    print(f"{'  - Retrieval (40%)':<30} ", end='')
    for _, m in recent:
        print(f"{m['granular']['retrieval']:>16.3f}", end=' ')
    print()

    print(f"{'  - Extraction (30%)':<30} ", end='')
    for _, m in recent:
        print(f"{m['granular']['extraction']:>16.3f}", end=' ')
    print()

    print(f"{'  - Alignment (20%)':<30} ", end='')
    for _, m in recent:
        print(f"{m['granular']['alignment']:>16.3f}", end=' ')
    print()

    print(f"{'  - Robustness (10%)':<30} ", end='')
    for _, m in recent:
        print(f"{m['granular']['robustness']:>16.3f}", end=' ')
    print()

    print()

    # Binary accuracy
    print(f"{'Binary Accuracy':<30} ", end='')
    for _, m in recent:
        print(f"{m['binary']['accuracy']*100:>15.1f}%", end=' ')
    print()

    # Retrieval metrics
    print(f"{'Answer in Top-1':<30} ", end='')
    for _, m in recent:
        print(f"{m['retrieval']['top_1_rate']*100:>15.1f}%", end=' ')
    print()

    print(f"{'Answer in Top-10':<30} ", end='')
    for _, m in recent:
        print(f"{m['retrieval']['top_10_rate']*100:>15.1f}%", end=' ')
    print()

    # Extraction metrics
    print(f"{'Extraction Success':<30} ", end='')
    for _, m in recent:
        print(f"{m['extraction']['success_rate']*100:>15.1f}%", end=' ')
    print()

    print(f"{'Extraction Exact Match':<30} ", end='')
    for _, m in recent:
        print(f"{m['extraction']['exact_rate']*100:>15.1f}%", end=' ')
    print()

    print(f"{'='*100}\n")

    # Show deltas
    if len(snapshots) >= 2:
        first = snapshots[0][1]
        latest = snapshots[-1][1]

        print(f"📈 Overall Progress ({snapshots[0][0]} → {snapshots[-1][0]}):")

        granular_delta = latest['granular']['overall_score'] - first['granular']['overall_score']
        print(f"  Granular: {first['granular']['overall_score']:.3f} → {latest['granular']['overall_score']:.3f} "
              f"({granular_delta:+.3f})")

        binary_delta = latest['binary']['accuracy'] - first['binary']['accuracy']
        print(f"  Binary:   {first['binary']['accuracy']*100:.1f}% → {latest['binary']['accuracy']*100:.1f}% "
              f"({binary_delta*100:+.1f}%)")

        retr_delta = latest['granular']['retrieval'] - first['granular']['retrieval']
        print(f"  Retrieval: {first['granular']['retrieval']:.3f} → {latest['granular']['retrieval']:.3f} "
              f"({retr_delta:+.3f})")

        extr_delta = latest['granular']['extraction'] - first['granular']['extraction']
        print(f"  Extraction: {first['granular']['extraction']:.3f} → {latest['granular']['extraction']:.3f} "
              f"({extr_delta:+.3f})")
        print()


def main():
    parser = argparse.ArgumentParser(description='Track RAG evaluation progress')
    parser.add_argument('--save', action='store_true', help='Save current results as snapshot')
    parser.add_argument('--compare', action='store_true', help='Compare all snapshots')
    parser.add_argument('--name', type=str, help='Custom name for snapshot')
    parser.add_argument('--results', type=str, default='data/evaluation/rag_results.jsonl',
                       help='Path to results file')

    args = parser.parse_args()

    # Default: show current metrics
    if not any([args.save, args.compare]):
        results_file = Path(args.results)
        if not results_file.exists():
            print(f"Error: Results file not found: {results_file}")
            print("Run evaluation first: python scripts/evaluate_rag_test_set.py")
            return 1

        print(f"Loading results from: {results_file}")
        results = load_results(results_file)
        metrics = compute_metrics(results)
        print_metrics(metrics)
        return 0

    # Save snapshot
    if args.save:
        results_file = Path(args.results)
        if not results_file.exists():
            print(f"Error: Results file not found: {results_file}")
            return 1

        print(f"Loading results from: {results_file}")
        results = load_results(results_file)
        metrics = compute_metrics(results)

        print_metrics(metrics, "Saving")
        save_snapshot(metrics, args.name)

    # Compare snapshots
    if args.compare:
        snapshots = load_snapshots()
        if not snapshots:
            print("No snapshots found. Run with --save first.")
            return 1

        compare_snapshots(snapshots)

    return 0


if __name__ == '__main__':
    exit(main())
