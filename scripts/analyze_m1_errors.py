#!/usr/bin/env python3
"""
Analyze M1 Model Errors

Investigates why plausible detection is at 73.3% instead of target 85%+.

Analyzes:
- Which plausible triples are being missed (false negatives)
- Score distribution for plausible vs implausible
- Correlation between individual scores (subj-verb, verb-obj, triple)
- Whether threshold adjustment could help
- Common patterns in misclassified examples

Usage:
    python scripts/analyze_m1_errors.py
    python scripts/analyze_m1_errors.py --verbose  # Show all errors
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.models.m1_selectional import M1SelectionalPreference


class M1ErrorAnalyzer:
    """Analyze M1 model errors and performance."""

    def __init__(self, model_path: Path, stage1_path: Path, test_path: Path):
        """Initialize analyzer."""
        print("Loading models...")

        # Load Stage 1 embeddings
        stage1_checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
        self.root_embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
        self.root_to_idx = stage1_checkpoint['root_to_idx']

        # Load M1 model
        m1_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = M1SelectionalPreference(
            embedding_dim=m1_checkpoint['embedding_dim'],
            hidden_dim=m1_checkpoint['hidden_dim']
        )
        self.model.load_state_dict(m1_checkpoint['model_state_dict'])
        self.model.eval()

        # Load test data
        self.test_examples = []
        with open(test_path) as f:
            for line in f:
                self.test_examples.append(json.loads(line))

        print(f"  Loaded {len(self.test_examples):,} test examples")
        print()

    def get_embedding(self, root: str) -> torch.Tensor:
        """Get embedding for a root."""
        idx = self.root_to_idx.get(root.lower(), 0)
        return self.root_embeddings[idx]

    def score_example(self, example: dict) -> Dict:
        """Score an example and return all metrics."""
        # Get embeddings
        subj_emb = self.get_embedding(example['subject_root']).unsqueeze(0)
        verb_emb = self.get_embedding(example['verb_root']).unsqueeze(0)
        obj_emb = self.get_embedding(example['object_root']).unsqueeze(0)

        # Score with M1
        with torch.no_grad():
            outputs = self.model(subj_emb, verb_emb, obj_emb)

        return {
            'example': example,
            'subj_verb_score': outputs['subj_verb_score'].item(),
            'verb_obj_score': outputs['verb_obj_score'].item(),
            'triple_score': outputs['triple_score'].item(),
            'label': example['label']
        }

    def analyze_score_distribution(self):
        """Analyze score distributions for plausible vs implausible."""
        print("=" * 70)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("=" * 70)

        plausible_scores = []
        implausible_scores = []

        for example in self.test_examples:
            result = self.score_example(example)
            score = result['triple_score']

            if result['label'] == 1.0:
                plausible_scores.append(score)
            else:
                implausible_scores.append(score)

        # Statistics
        import statistics

        print("\n📊 PLAUSIBLE TRIPLES (label=1.0)")
        print(f"  Count: {len(plausible_scores):,}")
        print(f"  Mean:  {statistics.mean(plausible_scores):.3f}")
        print(f"  Median: {statistics.median(plausible_scores):.3f}")
        print(f"  StdDev: {statistics.stdev(plausible_scores):.3f}")
        print(f"  Min:   {min(plausible_scores):.3f}")
        print(f"  Max:   {max(plausible_scores):.3f}")

        print("\n📊 IMPLAUSIBLE TRIPLES (label=0.0)")
        print(f"  Count: {len(implausible_scores):,}")
        print(f"  Mean:  {statistics.mean(implausible_scores):.3f}")
        print(f"  Median: {statistics.median(implausible_scores):.3f}")
        print(f"  StdDev: {statistics.stdev(implausible_scores):.3f}")
        print(f"  Min:   {min(implausible_scores):.3f}")
        print(f"  Max:   {max(implausible_scores):.3f}")

        # Histogram
        print("\n📈 SCORE HISTOGRAM (plausible)")
        self._print_histogram(plausible_scores, "plausible")

        print("\n📈 SCORE HISTOGRAM (implausible)")
        self._print_histogram(implausible_scores, "implausible")

    def _print_histogram(self, scores: List[float], label: str):
        """Print ASCII histogram of scores."""
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts = [0] * (len(bins) - 1)

        for score in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i + 1]:
                    counts[i] += 1
                    break
            else:
                # Handle score == 1.0
                if score == 1.0:
                    counts[-1] += 1

        max_count = max(counts) if counts else 1
        scale = 50 / max_count if max_count > 0 else 1

        for i, count in enumerate(counts):
            bar = '█' * int(count * scale)
            pct = (count / len(scores) * 100) if scores else 0
            bin_label = f"[{bins[i]:.1f}-{bins[i+1]:.1f})"
            print(f"  {bin_label:12} {bar} {count:4} ({pct:5.1f}%)")

    def analyze_threshold_sensitivity(self):
        """Test different thresholds to find optimal."""
        print("\n" + "=" * 70)
        print("THRESHOLD SENSITIVITY ANALYSIS")
        print("=" * 70)

        # Score all examples
        results = [self.score_example(ex) for ex in self.test_examples]

        # Test different thresholds
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

        print("\n  Thresh  Overall  Plaus   Implaus")
        print("  ------  -------  ------  -------")

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in thresholds:
            correct = 0
            plausible_correct = 0
            implausible_correct = 0
            plausible_total = 0
            implausible_total = 0

            for result in results:
                prediction = 1.0 if result['triple_score'] > threshold else 0.0
                label = result['label']

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

            overall_acc = correct / len(results)
            plausible_acc = plausible_correct / plausible_total
            implausible_acc = implausible_correct / implausible_total

            marker = " ←" if threshold == 0.5 else ""
            if overall_acc > best_accuracy:
                best_accuracy = overall_acc
                best_threshold = threshold
                marker = " ← BEST"

            print(f"  {threshold:.2f}    {overall_acc:.1%}   {plausible_acc:.1%}   {implausible_acc:.1%}{marker}")

        print(f"\n✅ Optimal threshold: {best_threshold:.2f} (accuracy: {best_accuracy:.1%})")

    def analyze_false_negatives(self, limit: int = 20):
        """Analyze plausible examples that were misclassified as implausible."""
        print("\n" + "=" * 70)
        print("FALSE NEGATIVE ANALYSIS (Plausible → Predicted Implausible)")
        print("=" * 70)

        false_negatives = []

        for example in self.test_examples:
            result = self.score_example(example)

            # False negative: label=1.0 but triple_score <= 0.5
            if result['label'] == 1.0 and result['triple_score'] <= 0.5:
                false_negatives.append(result)

        # Sort by score (lowest first = worst misses)
        false_negatives.sort(key=lambda x: x['triple_score'])

        print(f"\nFound {len(false_negatives)} false negatives (plausible examples scored ≤0.5)")
        print(f"Showing worst {min(limit, len(false_negatives))} misses:\n")

        for i, result in enumerate(false_negatives[:limit], 1):
            ex = result['example']
            print(f"{i}. ({ex['subject_root']}, {ex['verb_root']}, {ex['object_root']})")
            print(f"   Triple: {result['triple_score']:.3f}  S-V: {result['subj_verb_score']:.3f}  V-O: {result['verb_obj_score']:.3f}")
            if 'original_text' in ex:
                print(f"   Text: {ex['original_text'][:80]}")
            if 'corruption' in ex and ex['corruption']:
                print(f"   Corruption: {ex['corruption']}")
            print()

        # Analyze patterns
        print("🔍 PATTERN ANALYSIS")
        print("-" * 70)

        # Check if corruption type matters
        corruption_types = Counter()
        for result in false_negatives:
            ex = result['example']
            if 'corruption' in ex and ex['corruption']:
                corruption_types[ex['corruption'].get('type', 'unknown')] += 1

        if corruption_types:
            print("\nCorruption types in false negatives:")
            for ctype, count in corruption_types.most_common():
                print(f"  {ctype}: {count}")

        # Check score component patterns
        low_subj_verb = sum(1 for r in false_negatives if r['subj_verb_score'] < 0.3)
        low_verb_obj = sum(1 for r in false_negatives if r['verb_obj_score'] < 0.3)
        low_both = sum(1 for r in false_negatives if r['subj_verb_score'] < 0.3 and r['verb_obj_score'] < 0.3)

        print(f"\nLow component scores (< 0.3):")
        print(f"  Subject-Verb: {low_subj_verb}/{len(false_negatives)} ({low_subj_verb/len(false_negatives)*100:.1f}%)")
        print(f"  Verb-Object:  {low_verb_obj}/{len(false_negatives)} ({low_verb_obj/len(false_negatives)*100:.1f}%)")
        print(f"  Both low:     {low_both}/{len(false_negatives)} ({low_both/len(false_negatives)*100:.1f}%)")

    def analyze_false_positives(self, limit: int = 20):
        """Analyze implausible examples that were misclassified as plausible."""
        print("\n" + "=" * 70)
        print("FALSE POSITIVE ANALYSIS (Implausible → Predicted Plausible)")
        print("=" * 70)

        false_positives = []

        for example in self.test_examples:
            result = self.score_example(example)

            # False positive: label=0.0 but triple_score > 0.5
            if result['label'] == 0.0 and result['triple_score'] > 0.5:
                false_positives.append(result)

        # Sort by score (highest first = worst misses)
        false_positives.sort(key=lambda x: x['triple_score'], reverse=True)

        print(f"\nFound {len(false_positives)} false positives (implausible examples scored >0.5)")
        print(f"Showing worst {min(limit, len(false_positives))} misses:\n")

        for i, result in enumerate(false_positives[:limit], 1):
            ex = result['example']
            print(f"{i}. ({ex['subject_root']}, {ex['verb_root']}, {ex['object_root']})")
            print(f"   Triple: {result['triple_score']:.3f}  S-V: {result['subj_verb_score']:.3f}  V-O: {result['verb_obj_score']:.3f}")
            if 'original_text' in ex:
                print(f"   Text: {ex['original_text'][:80]}")
            if 'corruption' in ex and ex['corruption']:
                print(f"   Corruption: {ex['corruption']}")
            print()

    def run_full_analysis(self, verbose: bool = False):
        """Run complete error analysis."""
        self.analyze_score_distribution()
        self.analyze_threshold_sensitivity()
        self.analyze_false_negatives(limit=20 if verbose else 10)
        self.analyze_false_positives(limit=20 if verbose else 10)

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print("""
Based on this analysis, consider:

1. **Threshold adjustment**: If a different threshold significantly improves
   plausible detection without hurting overall accuracy, use it.

2. **Training data quality**: If false negatives are actually implausible
   (mislabeled), improve hard negative mining or data filtering.

3. **Loss weight rebalancing**: If subject-verb or verb-object scores are
   consistently low for plausible triples, increase their loss weights.

4. **Model capacity**: If patterns are complex, try larger hidden_dim
   (currently 128 → maybe 256).

5. **Training longer**: Early stopping at epoch 7 might be premature if
   validation accuracy is still improving.
        """)


def main():
    parser = argparse.ArgumentParser(description='Analyze M1 model errors')

    parser.add_argument('--model', type=str,
                        default='models/m1_selectional/best_model.pt',
                        help='Path to M1 model')
    parser.add_argument('--stage1', type=str,
                        default='models/root_embeddings/best_model.pt',
                        help='Path to Stage 1 embeddings')
    parser.add_argument('--test', type=str,
                        default='data/training/m1_selectional_hard/test.jsonl',
                        help='Path to test data')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show more examples')

    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model)
    stage1_path = Path(args.stage1)
    test_path = Path(args.test)

    if not model_path.exists():
        print(f"❌ M1 model not found: {model_path}")
        return 1

    if not stage1_path.exists():
        print(f"❌ Stage 1 model not found: {stage1_path}")
        return 1

    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return 1

    # Run analysis
    analyzer = M1ErrorAnalyzer(model_path, stage1_path, test_path)
    analyzer.run_full_analysis(verbose=args.verbose)

    return 0


if __name__ == '__main__':
    exit(main())
