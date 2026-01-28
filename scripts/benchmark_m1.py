#!/usr/bin/env python3
"""
Benchmark M1 Selectional Preferences Model

Tests M1 on its held-out test set and reports:
- Overall accuracy
- Accuracy by corruption type
- Confusion matrix
- Examples of correct/incorrect predictions
- Calibration (confidence vs accuracy)

Usage:
    python scripts/benchmark_m1.py
    python scripts/benchmark_m1.py --test-set data/training/m1_semantic_tier_priority/test.jsonl
    python scripts/benchmark_m1.py --show-examples
    python scripts/benchmark_m1.py --output results.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.m1_inference import M1Inference

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class M1Benchmark:
    """Benchmark M1 model on test set."""

    def __init__(
        self,
        m1: M1Inference,
        threshold: float = 0.5,
    ):
        self.m1 = m1
        self.threshold = threshold
        self.results = []

    def evaluate_test_set(
        self,
        test_set_path: Path,
        max_examples: int = None,
    ) -> Dict:
        """
        Evaluate M1 on test set.

        Args:
            test_set_path: Path to test.jsonl
            max_examples: Limit evaluation (for quick testing)

        Returns:
            Dictionary with metrics and results
        """
        logger.info(f"Loading test set from {test_set_path}...")

        examples = []
        with open(test_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
                if max_examples and len(examples) >= max_examples:
                    break

        logger.info(f"Evaluating {len(examples)} examples...")

        # Metrics tracking
        total = 0
        correct = 0
        by_corruption = defaultdict(lambda: {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        all_predictions = []
        all_labels = []

        # Evaluate each example
        for example in tqdm(examples, desc="Evaluating"):
            subj = example['subject_root']
            verb = example['verb_root']
            obj = example['object_root']
            label = example['label']  # 1.0 = plausible, 0.0 = implausible
            corruption = example.get('corruption', 'positive')

            # Score with M1
            score = self.m1.score_triple(subj, verb, obj)
            prediction = 1 if score >= self.threshold else 0
            label_int = int(label)

            # Update metrics
            total += 1
            is_correct = (prediction == label_int)
            if is_correct:
                correct += 1

            # Update by corruption type
            by_corruption[corruption]['total'] += 1
            if is_correct:
                by_corruption[corruption]['correct'] += 1

            # Confusion matrix
            if prediction == 1 and label_int == 1:
                by_corruption[corruption]['tp'] += 1
            elif prediction == 1 and label_int == 0:
                by_corruption[corruption]['fp'] += 1
            elif prediction == 0 and label_int == 0:
                by_corruption[corruption]['tn'] += 1
            elif prediction == 0 and label_int == 1:
                by_corruption[corruption]['fn'] += 1

            all_predictions.append(prediction)
            all_labels.append(label_int)

            # Store result
            self.results.append({
                'triple': (subj, verb, obj),
                'label': label_int,
                'prediction': prediction,
                'score': score,
                'corruption': corruption,
                'correct': is_correct,
                'original_text': example.get('original_text', ''),
            })

        # Calculate overall metrics
        accuracy = correct / total if total > 0 else 0

        # Calculate confusion matrix for overall
        tp = sum(1 for i in range(len(all_predictions)) if all_predictions[i] == 1 and all_labels[i] == 1)
        fp = sum(1 for i in range(len(all_predictions)) if all_predictions[i] == 1 and all_labels[i] == 0)
        tn = sum(1 for i in range(len(all_predictions)) if all_predictions[i] == 0 and all_labels[i] == 0)
        fn = sum(1 for i in range(len(all_predictions)) if all_predictions[i] == 0 and all_labels[i] == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'overall': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': {
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                }
            },
            'by_corruption': {}
        }

        # Calculate per-corruption metrics
        for corruption, stats in by_corruption.items():
            total_c = stats['total']
            correct_c = stats['correct']
            acc_c = correct_c / total_c if total_c > 0 else 0

            tp_c = stats['tp']
            fp_c = stats['fp']
            tn_c = stats['tn']
            fn_c = stats['fn']

            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
            f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0

            metrics['by_corruption'][corruption] = {
                'total': total_c,
                'correct': correct_c,
                'accuracy': acc_c,
                'precision': prec_c,
                'recall': rec_c,
                'f1': f1_c,
                'confusion_matrix': {
                    'tp': tp_c,
                    'fp': fp_c,
                    'tn': tn_c,
                    'fn': fn_c,
                }
            }

        return metrics

    def get_example_predictions(
        self,
        num_correct: int = 5,
        num_incorrect: int = 5,
    ) -> Dict:
        """Get examples of correct and incorrect predictions."""
        correct = [r for r in self.results if r['correct']]
        incorrect = [r for r in self.results if not r['correct']]

        # Sort by confidence (distance from threshold)
        correct.sort(key=lambda x: abs(x['score'] - self.threshold), reverse=True)
        incorrect.sort(key=lambda x: abs(x['score'] - self.threshold), reverse=True)

        return {
            'correct': correct[:num_correct],
            'incorrect': incorrect[:num_incorrect],
        }


def print_metrics(metrics: Dict):
    """Print benchmark metrics."""
    print("\n" + "=" * 80)
    print("M1 BENCHMARK RESULTS")
    print("=" * 80)

    overall = metrics['overall']
    print(f"\nOverall Performance:")
    print(f"  Total examples: {overall['total']:,}")
    print(f"  Correct: {overall['correct']:,}")
    print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall: {overall['recall']:.4f}")
    print(f"  F1 Score: {overall['f1']:.4f}")

    cm = overall['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {cm['tp']:,}")
    print(f"  False Positives: {cm['fp']:,}")
    print(f"  True Negatives:  {cm['tn']:,}")
    print(f"  False Negatives: {cm['fn']:,}")

    print(f"\nPerformance by Corruption Type:")
    print(f"  {'Type':<20} {'Total':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    # Sort by corruption type (positive first, then alphabetically)
    # Handle None values in corruption type
    sorted_corruptions = sorted(
        metrics['by_corruption'].items(),
        key=lambda x: (x[0] != 'positive', x[0] if x[0] is not None else 'unknown')
    )

    for corruption, stats in sorted_corruptions:
        corruption_str = corruption if corruption is not None else 'unknown'
        print(f"  {corruption_str:<20} {stats['total']:<10,} "
              f"{stats['accuracy']:<12.4f} {stats['precision']:<12.4f} "
              f"{stats['recall']:<12.4f} {stats['f1']:<12.4f}")


def print_examples(examples: Dict):
    """Print example predictions."""
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)

    print("\n✅ CORRECT PREDICTIONS (High Confidence):")
    for i, ex in enumerate(examples['correct'], 1):
        subj, verb, obj = ex['triple']
        label_str = "PLAUSIBLE" if ex['label'] == 1 else "IMPLAUSIBLE"
        print(f"\n  {i}. ({subj}, {verb}, {obj})")
        print(f"     Label: {label_str} | Prediction: {ex['prediction']} | Score: {ex['score']:.4f}")
        print(f"     Corruption: {ex['corruption']}")
        if ex['original_text']:
            print(f"     Text: {ex['original_text'][:80]}...")

    print("\n❌ INCORRECT PREDICTIONS (High Confidence Errors):")
    for i, ex in enumerate(examples['incorrect'], 1):
        subj, verb, obj = ex['triple']
        label_str = "PLAUSIBLE" if ex['label'] == 1 else "IMPLAUSIBLE"
        pred_str = "PLAUSIBLE" if ex['prediction'] == 1 else "IMPLAUSIBLE"
        print(f"\n  {i}. ({subj}, {verb}, {obj})")
        print(f"     Label: {label_str} | Predicted: {pred_str} | Score: {ex['score']:.4f}")
        print(f"     Corruption: {ex['corruption']}")
        if ex['original_text']:
            print(f"     Text: {ex['original_text'][:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark M1 selectional preferences model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--m1-model',
        type=str,
        default='models/m1_semantic_tier_priority/best_model.pt',
        help='Path to M1 model'
    )
    parser.add_argument(
        '--stage1-model',
        type=str,
        default='models/root_embeddings_tier0/best_model.pt',
        help='Path to Stage 1 embeddings'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        default='data/training/m1_semantic_tier_priority/test.jsonl',
        help='Path to test set'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Plausibility threshold (default: 0.5)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        help='Limit evaluation to N examples (for quick testing)'
    )
    parser.add_argument(
        '--show-examples',
        action='store_true',
        help='Show example correct/incorrect predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Check paths exist
    m1_path = Path(args.m1_model)
    stage1_path = Path(args.stage1_model)
    test_path = Path(args.test_set)

    if not m1_path.exists():
        logger.error(f"M1 model not found: {m1_path}")
        sys.exit(1)

    if not stage1_path.exists():
        logger.error(f"Stage 1 model not found: {stage1_path}")
        sys.exit(1)

    if not test_path.exists():
        logger.error(f"Test set not found: {test_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("M1 Benchmark")
    logger.info("=" * 80)
    logger.info(f"M1 model: {m1_path}")
    logger.info(f"Stage 1: {stage1_path}")
    logger.info(f"Test set: {test_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("")

    # Load M1
    logger.info("Loading M1 model...")
    m1 = M1Inference(
        model_path=m1_path,
        stage1_path=stage1_path,
        device='cpu'
    )
    logger.info("  ✓ M1 loaded")

    # Run benchmark
    benchmark = M1Benchmark(m1, threshold=args.threshold)
    metrics = benchmark.evaluate_test_set(
        test_set_path=test_path,
        max_examples=args.max_examples,
    )

    # Print results
    print_metrics(metrics)

    # Show examples
    if args.show_examples:
        examples = benchmark.get_example_predictions(num_correct=5, num_incorrect=5)
        print_examples(examples)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'config': {
                'm1_model': str(m1_path),
                'stage1_model': str(stage1_path),
                'test_set': str(test_path),
                'threshold': args.threshold,
            },
            'metrics': metrics,
        }

        if args.show_examples:
            examples = benchmark.get_example_predictions(num_correct=10, num_incorrect=10)
            results['examples'] = examples

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Results saved to: {output_path}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
