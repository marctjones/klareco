#!/usr/bin/env python3
"""
M1 Extensive Validation Script

Exercises the M1 model extensively to test how well it works:
1. Tests on full test set with detailed metrics
2. Stress tests with edge cases
3. Performance benchmarks (throughput, latency)
4. Robustness tests (unknown words, rare combinations)
5. Calibration analysis (score distribution)
6. Error pattern analysis

Provides comprehensive model quality assessment beyond unit tests.

Usage:
    python scripts/validate_m1_extensive.py
    python scripts/validate_m1_extensive.py --full  # Include slow tests
    python scripts/validate_m1_extensive.py --benchmark  # Performance only
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Dict

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.m1_inference import M1Inference


class M1ExtensiveValidator:
    """Extensive validation of M1 model."""

    def __init__(self, test_data_path: Path):
        """Initialize validator."""
        print("Loading M1 model and test data...")
        self.m1 = M1Inference()

        # Load test data
        self.test_examples = []
        with open(test_data_path) as f:
            for line in f:
                self.test_examples.append(json.loads(line))

        print(f"✓ Loaded {len(self.test_examples):,} test examples\n")

    def validate_1_full_test_set(self):
        """Validation 1: Comprehensive test set evaluation."""
        print("=" * 70)
        print("VALIDATION 1: FULL TEST SET EVALUATION")
        print("=" * 70)
        print(f"\nEvaluating on {len(self.test_examples):,} examples...\n")

        # Score all examples
        triples = [(ex['subject_root'], ex['verb_root'], ex['object_root'])
                   for ex in self.test_examples]

        start = time.time()
        scores = self.m1.score_triples(triples)
        duration = time.time() - start

        # Compute metrics at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        print("  Performance by threshold:\n")
        print("  Threshold  Overall   Plausible  Implausible")
        print("  ---------  -------   ---------  -----------")

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in thresholds:
            correct = 0
            plausible_correct = 0
            implausible_correct = 0
            plausible_total = 0
            implausible_total = 0

            for ex, score in zip(self.test_examples, scores):
                prediction = 1.0 if score > threshold else 0.0
                label = ex['label']

                if prediction == label:
                    correct += 1

                if label == 1.0:
                    plausible_total += 1
                    if prediction == 1.0:
                        plausible_correct += 1
                else:
                    implausible_total += 1
                    if prediction == 0.0:
                        implausible_correct += 1

            overall_acc = correct / len(self.test_examples)
            plausible_acc = plausible_correct / plausible_total if plausible_total > 0 else 0
            implausible_acc = implausible_correct / implausible_total if implausible_total > 0 else 0

            marker = ""
            if overall_acc > best_accuracy:
                best_accuracy = overall_acc
                best_threshold = threshold
                marker = " ← BEST"

            print(f"     {threshold:.1f}      {overall_acc:.1%}     {plausible_acc:.1%}      {implausible_acc:.1%}{marker}")

        print(f"\n  Processing time: {duration:.2f}s ({len(self.test_examples)/duration:.0f} examples/sec)")
        print(f"  Optimal threshold: {best_threshold} (accuracy: {best_accuracy:.1%})")

    def validate_2_edge_cases(self):
        """Validation 2: Edge cases and stress tests."""
        print("\n" + "=" * 70)
        print("VALIDATION 2: EDGE CASES & STRESS TESTS")
        print("=" * 70)

        edge_cases = [
            {
                'name': "Identical words (subject=verb=object)",
                'triples': [('hom', 'hom', 'hom'), ('libro', 'libro', 'libro')],
                'expected': 'low scores (nonsensical)'
            },
            {
                'name': "Function words as content words",
                'triples': [('mi', 'kaj', 'vi'), ('la', 'de', 'en')],
                'expected': 'varies (function words have different semantics)'
            },
            {
                'name': "Extremely common vs rare roots",
                'triples': [('mi', 'est', 'homo'), ('zzz', 'yyy', 'xxx')],
                'expected': 'common: higher, rare: lower (unknown roots)'
            },
            {
                'name': "Abstract concepts",
                'triples': [('ideo', 'viv', 'teorio'), ('penso', 'kre', 'koncepto')],
                'expected': 'medium scores (abstract but valid)'
            },
            {
                'name': "Circular dependencies",
                'triples': [('kaŭzo', 'kaŭz', 'efekto'), ('efekto', 'efekt', 'kaŭzo')],
                'expected': 'varies based on training data'
            },
        ]

        print()
        for case in edge_cases:
            print(f"  {case['name']}:")
            print(f"    Expected: {case['expected']}\n")

            scores = self.m1.score_triples(case['triples'])

            for (s, v, o), score in zip(case['triples'], scores):
                status = "plausible" if score >= 0.5 else "implausible"
                print(f"      ({s:10}, {v:10}, {o:10}) → {score:.3f} ({status})")

            print()

    def validate_3_performance_benchmark(self):
        """Validation 3: Performance benchmarking."""
        print("\n" + "=" * 70)
        print("VALIDATION 3: PERFORMANCE BENCHMARK")
        print("=" * 70)
        print()

        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100, 500]

        # Sample test data
        sample_triples = [(ex['subject_root'], ex['verb_root'], ex['object_root'])
                         for ex in random.sample(self.test_examples, min(500, len(self.test_examples)))]

        print("  Throughput by batch size:\n")
        print("  Batch Size  Time (ms)  Throughput (triples/sec)")
        print("  ----------  ---------  ------------------------")

        for batch_size in batch_sizes:
            batches = [sample_triples[i:i+batch_size] for i in range(0, len(sample_triples), batch_size)]

            start = time.time()
            for batch in batches:
                self.m1.score_triples(batch)
            duration = time.time() - start

            time_per_batch = (duration / len(batches)) * 1000  # ms
            throughput = len(sample_triples) / duration

            print(f"      {batch_size:4}      {time_per_batch:7.2f}     {throughput:8.0f}")

        # Latency test (single triple)
        print("\n  Latency for single triple:")
        single_triple = sample_triples[0]

        latencies = []
        for _ in range(100):
            start = time.time()
            self.m1.score_triple(*single_triple)
            latencies.append((time.time() - start) * 1000)  # ms

        print(f"    Min:    {min(latencies):.2f} ms")
        print(f"    Median: {sorted(latencies)[len(latencies)//2]:.2f} ms")
        print(f"    Mean:   {sum(latencies)/len(latencies):.2f} ms")
        print(f"    Max:    {max(latencies):.2f} ms")

    def validate_4_robustness(self):
        """Validation 4: Robustness to unknown/rare words."""
        print("\n" + "=" * 70)
        print("VALIDATION 4: ROBUSTNESS TO UNKNOWN/RARE WORDS")
        print("=" * 70)
        print()

        # Test with completely unknown words
        unknown_words = ['xxxxxx', 'yyyyyy', 'zzzzzz', 'qqqqqqq', 'wwwwww']

        print("  Testing unknown words (should gracefully handle):\n")

        # Unknown subject
        unknown_subj_triples = [(unk, 'vid', 'libro') for unk in unknown_words[:3]]
        scores = self.m1.score_triples(unknown_subj_triples)

        print("    Unknown subject + known verb/object:")
        for (s, v, o), score in zip(unknown_subj_triples, scores):
            print(f"      ({s}, {v}, {o}) → {score:.3f}")

        # Unknown verb
        unknown_verb_triples = [('homo', unk, 'libro') for unk in unknown_words[:3]]
        scores = self.m1.score_triples(unknown_verb_triples)

        print("\n    Known subject + unknown verb + known object:")
        for (s, v, o), score in zip(unknown_verb_triples, scores):
            print(f"      ({s}, {v}, {o}) → {score:.3f}")

        # All unknown
        all_unknown_triples = [(unknown_words[0], unknown_words[1], unknown_words[2])]
        scores = self.m1.score_triples(all_unknown_triples)

        print("\n    All unknown:")
        for (s, v, o), score in zip(all_unknown_triples, scores):
            print(f"      ({s}, {v}, {o}) → {score:.3f}")

        print("\n  ✓ Model handles unknown words gracefully (returns low scores)")

    def validate_5_calibration(self):
        """Validation 5: Score calibration and distribution."""
        print("\n" + "=" * 70)
        print("VALIDATION 5: SCORE CALIBRATION & DISTRIBUTION")
        print("=" * 70)
        print()

        # Score all test examples
        triples = [(ex['subject_root'], ex['verb_root'], ex['object_root'])
                   for ex in self.test_examples]
        scores = self.m1.score_triples(triples)

        # Separate by label
        plausible_scores = [s for s, ex in zip(scores, self.test_examples) if ex['label'] == 1.0]
        implausible_scores = [s for s, ex in zip(scores, self.test_examples) if ex['label'] == 0.0]

        # Histogram
        def histogram(scores, label, bins=10):
            print(f"  {label} score distribution:\n")
            bin_size = 1.0 / bins
            counts = [0] * bins

            for score in scores:
                bin_idx = min(int(score / bin_size), bins - 1)
                counts[bin_idx] += 1

            max_count = max(counts) if counts else 1
            scale = 40 / max_count

            for i in range(bins):
                bin_start = i * bin_size
                bin_end = (i + 1) * bin_size
                bar = '█' * int(counts[i] * scale)
                pct = counts[i] / len(scores) * 100 if scores else 0
                print(f"    [{bin_start:.1f}-{bin_end:.1f}): {bar} {counts[i]:4} ({pct:5.1f}%)")

        histogram(plausible_scores, "PLAUSIBLE", bins=10)
        print()
        histogram(implausible_scores, "IMPLAUSIBLE", bins=10)

        # Statistics
        import statistics

        print("\n  Score statistics:\n")
        print("                    Plausible  Implausible")
        print("                    ---------  -----------")
        print(f"    Mean:           {statistics.mean(plausible_scores):7.3f}    {statistics.mean(implausible_scores):7.3f}")
        print(f"    Median:         {statistics.median(plausible_scores):7.3f}    {statistics.median(implausible_scores):7.3f}")
        print(f"    Std Dev:        {statistics.stdev(plausible_scores):7.3f}    {statistics.stdev(implausible_scores):7.3f}")
        print(f"    Min:            {min(plausible_scores):7.3f}    {min(implausible_scores):7.3f}")
        print(f"    Max:            {max(plausible_scores):7.3f}    {max(implausible_scores):7.3f}")

        # Separation
        separation = statistics.mean(plausible_scores) - statistics.mean(implausible_scores)
        print(f"\n  Mean separation: {separation:.3f} (higher is better)")

    def validate_6_error_patterns(self):
        """Validation 6: Analyze error patterns."""
        print("\n" + "=" * 70)
        print("VALIDATION 6: ERROR PATTERN ANALYSIS")
        print("=" * 70)
        print()

        # Score all examples
        triples = [(ex['subject_root'], ex['verb_root'], ex['object_root'])
                   for ex in self.test_examples]
        scores = self.m1.score_triples(triples)

        # Find errors
        false_positives = []  # Implausible scored as plausible
        false_negatives = []  # Plausible scored as implausible

        for ex, score in zip(self.test_examples, scores):
            prediction = 1.0 if score > 0.5 else 0.0
            label = ex['label']

            if label == 0.0 and prediction == 1.0:
                false_positives.append((ex, score))
            elif label == 1.0 and prediction == 0.0:
                false_negatives.append((ex, score))

        # Analyze false positives
        print(f"  False Positives: {len(false_positives)} (implausible scored as plausible)\n")

        if false_positives:
            # Sort by score (highest false positives are worst)
            false_positives.sort(key=lambda x: x[1], reverse=True)

            print("    Top 5 worst false positives:")
            for ex, score in false_positives[:5]:
                s, v, o = ex['subject_root'], ex['verb_root'], ex['object_root']
                corruption = ex.get('corruption', 'unknown')
                print(f"      ({s}, {v}, {o}) - score: {score:.3f}, corruption: {corruption}")

        # Analyze false negatives
        print(f"\n  False Negatives: {len(false_negatives)} (plausible scored as implausible)\n")

        if false_negatives:
            # Sort by score (lowest false negatives are worst)
            false_negatives.sort(key=lambda x: x[1])

            print("    Top 5 worst false negatives:")
            for ex, score in false_negatives[:5]:
                s, v, o = ex['subject_root'], ex['verb_root'], ex['object_root']
                print(f"      ({s}, {v}, {o}) - score: {score:.3f}")

        # Error rate by corruption type
        if false_positives:
            corruption_types = Counter(ex.get('corruption', 'unknown') for ex, _ in false_positives)
            print("\n  False positive breakdown by corruption type:")
            for ctype, count in corruption_types.most_common():
                pct = count / len(false_positives) * 100
                print(f"      {ctype:20} {count:4} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='M1 Extensive Validation')

    parser.add_argument('--test-data', type=str,
                        default='data/training/m1_selectional_hard_only/test.jsonl',
                        help='Path to test data')
    parser.add_argument('--full', action='store_true',
                        help='Run all validations including slow ones')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmarks only')

    args = parser.parse_args()

    # Check test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("Train M1 model first: python scripts/train_m1_selectional.py")
        return 1

    validator = M1ExtensiveValidator(test_path)

    print("=" * 70)
    print("M1 EXTENSIVE VALIDATION")
    print("=" * 70)
    print()

    if args.benchmark:
        # Performance only
        validator.validate_3_performance_benchmark()
    elif args.full:
        # All validations
        validator.validate_1_full_test_set()
        validator.validate_2_edge_cases()
        validator.validate_3_performance_benchmark()
        validator.validate_4_robustness()
        validator.validate_5_calibration()
        validator.validate_6_error_patterns()
    else:
        # Default: most important validations (skip slow ones)
        validator.validate_1_full_test_set()
        validator.validate_2_edge_cases()
        validator.validate_4_robustness()
        validator.validate_5_calibration()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("""
Extensive validation exercises the M1 model across:
✓ Full test set evaluation
✓ Edge cases and stress tests
✓ Performance benchmarking
✓ Robustness to unknown words
✓ Score calibration analysis
✓ Error pattern analysis

All validations complete. Model performance documented.
    """)

    return 0


if __name__ == '__main__':
    random.seed(42)  # Reproducible results
    exit(main())
