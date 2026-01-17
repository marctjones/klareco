#!/usr/bin/env python3
"""
M1 Selectional Preference Model Demo

Interactive demo showing M1 model capabilities:
- Plausibility scoring for subject-verb-object triples
- Quality metrics and model performance report
- Interactive mode for testing custom triples
- Comparison of plausible vs implausible examples

Usage:
    python scripts/demo_m1_selectional.py              # Full demo + interactive
    python scripts/demo_m1_selectional.py --report     # Quality report only
    python scripts/demo_m1_selectional.py --interactive # Interactive mode only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.models.m1_selectional import M1SelectionalPreference


class M1Demo:
    """M1 Selectional Preference Demo."""

    def __init__(self, model_path: Path, stage1_path: Path):
        """Initialize demo with trained models."""
        print("Loading models...")

        # Load Stage 1 embeddings
        stage1_checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
        self.root_embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
        self.root_to_idx = stage1_checkpoint['root_to_idx']
        self.idx_to_root = stage1_checkpoint['idx_to_root']

        print(f"  Stage 1: {len(self.root_to_idx):,} root embeddings")

        # Load M1 model
        m1_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = M1SelectionalPreference(
            embedding_dim=m1_checkpoint['embedding_dim'],
            hidden_dim=m1_checkpoint['hidden_dim']
        )
        self.model.load_state_dict(m1_checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  M1: {self.model.count_parameters():,} parameters")
        print(f"  Best accuracy: {m1_checkpoint['best_accuracy']:.1%}")
        print()

    def get_embedding(self, root: str) -> torch.Tensor:
        """Get embedding for a root (or zero vector if not found)."""
        idx = self.root_to_idx.get(root.lower(), 0)
        return self.root_embeddings[idx]

    def score_triple(self, subject: str, verb: str, obj: str) -> Dict:
        """Score a subject-verb-object triple."""
        # Get embeddings
        subj_emb = self.get_embedding(subject).unsqueeze(0)
        verb_emb = self.get_embedding(verb).unsqueeze(0)
        obj_emb = self.get_embedding(obj).unsqueeze(0)

        # Score with M1
        with torch.no_grad():
            outputs = self.model(subj_emb, verb_emb, obj_emb)

        return {
            'subject_verb': outputs['subj_verb_score'].item(),
            'verb_object': outputs['verb_obj_score'].item(),
            'triple': outputs['triple_score'].item(),
            'plausible': outputs['triple_score'].item() > 0.5
        }

    def print_score(self, subject: str, verb: str, obj: str):
        """Print formatted score for a triple."""
        scores = self.score_triple(subject, verb, obj)

        # Color coding
        if scores['plausible']:
            status = "\033[92m✓ PLAUSIBLE\033[0m"
        else:
            status = "\033[91m✗ IMPLAUSIBLE\033[0m"

        print(f"\n  ({subject}, {verb}, {obj})")
        print(f"    Subject-Verb:  {scores['subject_verb']:.3f}")
        print(f"    Verb-Object:   {scores['verb_object']:.3f}")
        print(f"    Triple Score:  {scores['triple']:.3f}  {status}")

    def demo_examples(self):
        """Run through curated demo examples."""
        print("=" * 70)
        print("M1 SELECTIONAL PREFERENCE DEMO")
        print("=" * 70)

        print("\n📊 PLAUSIBLE TRIPLES (Expected: High Scores)")
        print("-" * 70)
        print("Real examples from test set labeled as plausible:\n")

        plausible = [
            ("mi", "uz", "ĝi"),           # I use it
            ("jun", "far", "sign"),       # young-one does sign
            ("li", "hav", "barb"),        # he has beard
            ("mi", "rigard", "hund"),     # I look-at dog
            ("mi", "vid", "numer"),       # I see number
        ]

        for subj, verb, obj in plausible:
            self.print_score(subj, verb, obj)

        print("\n\n📊 IMPLAUSIBLE TRIPLES (Expected: Low Scores)")
        print("-" * 70)
        print("Real examples from test set labeled as implausible:\n")

        implausible = [
            ("tie", "ir", "o'connor"),    # there goes O'Connor (corrupted)
            ("pantalon", "kovr", "korp"), # pants covers body (actually plausible?)
            ("ŝtup", "renkont", "humid"), # step meets humidity (implausible)
            ("mi", "hav", "mult"),        # I have much (grammatical issue)
            ("tio", "ch", "lich"),        # that [corrupted verb] [corrupted noun]
        ]

        for subj, verb, obj in implausible:
            self.print_score(subj, verb, obj)

    def quality_report(self):
        """Generate quality report from test data."""
        print("\n" + "=" * 70)
        print("M1 MODEL QUALITY REPORT")
        print("=" * 70)

        # Load test data
        test_path = Path('data/training/m1_selectional_hard/test.jsonl')

        if not test_path.exists():
            print(f"\n⚠️  Test data not found: {test_path}")
            return

        print(f"\nEvaluating on test set: {test_path}")

        # Load examples
        examples = []
        with open(test_path) as f:
            for line in f:
                examples.append(json.loads(line))

        print(f"  Total examples: {len(examples):,}")

        # Evaluate
        correct = 0
        plausible_correct = 0
        implausible_correct = 0
        plausible_total = 0
        implausible_total = 0

        for ex in examples:
            scores = self.score_triple(
                ex['subject_root'],
                ex['verb_root'],
                ex['object_root']
            )

            prediction = 1.0 if scores['plausible'] else 0.0
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

        # Calculate metrics
        accuracy = correct / len(examples)
        plausible_acc = plausible_correct / plausible_total if plausible_total > 0 else 0
        implausible_acc = implausible_correct / implausible_total if implausible_total > 0 else 0

        # Print report
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 70)
        print(f"  Overall Accuracy:      {accuracy:.1%}  ({correct:,}/{len(examples):,})")
        print(f"  Plausible Detection:   {plausible_acc:.1%}  ({plausible_correct:,}/{plausible_total:,})")
        print(f"  Implausible Detection: {implausible_acc:.1%}  ({implausible_correct:,}/{implausible_total:,})")

        # Quality thresholds
        print("\n✅ QUALITY THRESHOLDS")
        print("-" * 70)
        print(f"  Overall Accuracy ≥ 80%:       {'✓' if accuracy >= 0.80 else '✗'}")
        print(f"  Plausible Detection ≥ 85%:    {'✓' if plausible_acc >= 0.85 else '✗'}")
        print(f"  Implausible Detection ≥ 70%:  {'✓' if implausible_acc >= 0.70 else '✗'}")

    def interactive_mode(self):
        """Interactive mode for testing custom triples."""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("\nTest custom subject-verb-object triples!")
        print("Enter triples as: <subject> <verb> <object>")
        print("Type 'quit' to exit\n")

        while True:
            try:
                user_input = input("Enter triple: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                parts = user_input.split()

                if len(parts) != 3:
                    print("  ⚠️  Please enter exactly 3 words (subject verb object)")
                    continue

                subject, verb, obj = parts

                # Check if roots exist
                missing = []
                if subject.lower() not in self.root_to_idx:
                    missing.append(f"subject '{subject}'")
                if verb.lower() not in self.root_to_idx:
                    missing.append(f"verb '{verb}'")
                if obj.lower() not in self.root_to_idx:
                    missing.append(f"object '{obj}'")

                if missing:
                    print(f"  ⚠️  Unknown roots: {', '.join(missing)}")
                    print(f"  Vocabulary size: {len(self.root_to_idx):,} roots")
                    continue

                self.print_score(subject, verb, obj)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  ❌ Error: {e}")

        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='M1 Selectional Preference Demo')

    parser.add_argument('--model', type=str,
                        default='models/m1_selectional/best_model.pt',
                        help='Path to M1 model checkpoint')
    parser.add_argument('--stage1', type=str,
                        default='models/root_embeddings/best_model.pt',
                        help='Path to Stage 1 embeddings')
    parser.add_argument('--report', action='store_true',
                        help='Show quality report only')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode only')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Skip interactive mode')

    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model)
    stage1_path = Path(args.stage1)

    if not model_path.exists():
        print(f"❌ M1 model not found: {model_path}")
        print("Train M1 first: python scripts/train_m1_selectional.py")
        return 1

    if not stage1_path.exists():
        print(f"❌ Stage 1 model not found: {stage1_path}")
        print("Train Stage 1 first: ./scripts/train_roots.sh --vocab tier2-5")
        return 1

    # Initialize demo
    demo = M1Demo(model_path, stage1_path)

    # Run requested modes
    if args.report:
        demo.quality_report()
    elif args.interactive:
        demo.interactive_mode()
    else:
        # Full demo (examples + report + interactive)
        demo.demo_examples()
        demo.quality_report()

        if not args.no_interactive:
            demo.interactive_mode()

    return 0


if __name__ == '__main__':
    exit(main())
