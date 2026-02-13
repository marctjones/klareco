#!/usr/bin/env python3
"""
Generate training data for entity type classifier.

Uses three strategies:
1. Auto-label corpus with deterministic features (~70% coverage)
2. Extract examples from test set
3. Generate synthetic examples from root vocabulary

Usage:
    python scripts/generate_entity_training_data.py [--corpus PATH] [--output DIR]
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.semantic_enrichment.data_generator import TrainingDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate entity type classifier training data')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Path to corpus JSONL (default: data/corpus/unified_corpus.jsonl)'
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        default=Path('data/test_sets/rag_test_set.jsonl'),
        help='Path to test set JSONL (default: data/test_sets/rag_test_set.jsonl)'
    )
    parser.add_argument(
        '--root-vocab',
        type=Path,
        default=Path('data/vocabularies/root_vocab.json'),
        help='Path to root vocabulary JSON (default: data/vocabularies/root_vocab.json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/entity_classifier'),
        help='Output directory (default: data/training/entity_classifier)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.70,
        help='Minimum confidence for corpus auto-labeling (default: 0.70)'
    )
    parser.add_argument(
        '--max-synthetic',
        type=int,
        default=100,
        help='Maximum synthetic examples per affix (default: 100)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation split fraction (default: 0.15)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = TrainingDataGenerator()

    # Paths for intermediate outputs
    enriched_corpus_path = args.output / 'enriched_corpus.jsonl'
    test_examples_path = args.output / 'test_examples.jsonl'
    synthetic_examples_path = args.output / 'synthetic_examples.jsonl'

    print("=" * 60)
    print("TRAINING DATA GENERATION FOR ENTITY TYPE CLASSIFIER")
    print("=" * 60)

    # Step 1: Auto-label corpus
    if args.corpus.exists():
        print(f"\n[1/4] Auto-labeling corpus: {args.corpus}")
        print("-" * 60)
        corpus_stats = generator.auto_label_corpus(
            corpus_path=args.corpus,
            output_path=enriched_corpus_path,
            min_confidence=args.min_confidence
        )
        print(f"✓ Auto-labeled {corpus_stats['enriched']}/{corpus_stats['total_words']} words")
        print(f"  High confidence (>=0.90): {corpus_stats['high_confidence']} "
              f"({corpus_stats['deterministic_coverage']:.1f}%)")
        print(f"  Medium confidence (0.70-0.90): {corpus_stats['medium_confidence']}")
        print(f"  Saved to: {enriched_corpus_path}")
    else:
        print(f"\n[1/4] Skipping corpus auto-labeling (file not found: {args.corpus})")
        enriched_corpus_path = None

    # Step 2: Extract from test set
    if args.test_set.exists():
        print(f"\n[2/4] Extracting examples from test set: {args.test_set}")
        print("-" * 60)
        test_stats = generator.extract_from_test_set(
            test_set_path=args.test_set,
            output_path=test_examples_path
        )
        print(f"✓ Extracted {test_stats['examples_extracted']} examples "
              f"from {test_stats['total_questions']} questions")
        print(f"  Saved to: {test_examples_path}")
    else:
        print(f"\n[2/4] Skipping test set extraction (file not found: {args.test_set})")
        test_examples_path = None

    # Step 3: Generate synthetic examples
    if args.root_vocab.exists():
        print(f"\n[3/4] Generating synthetic examples: {args.root_vocab}")
        print("-" * 60)
        synthetic_stats = generator.generate_synthetic_examples(
            root_vocab_path=args.root_vocab,
            output_path=synthetic_examples_path,
            max_per_affix=args.max_synthetic
        )
        print(f"✓ Generated {synthetic_stats['examples_generated']} synthetic examples")
        print(f"  By affix: {synthetic_stats['by_affix']}")
        print(f"  Saved to: {synthetic_examples_path}")
    else:
        print(f"\n[3/4] Skipping synthetic generation (file not found: {args.root_vocab})")
        synthetic_examples_path = None

    # Step 4: Create training dataset
    print(f"\n[4/4] Creating training dataset")
    print("-" * 60)

    # Use only available sources
    sources = {}
    if enriched_corpus_path and enriched_corpus_path.exists():
        sources['enriched_corpus'] = enriched_corpus_path
    if test_examples_path and test_examples_path.exists():
        sources['test_examples'] = test_examples_path
    if synthetic_examples_path and synthetic_examples_path.exists():
        sources['synthetic_examples'] = synthetic_examples_path

    if not sources:
        print("✗ No data sources available!")
        return 1

    dataset_stats = generator.create_training_dataset(
        enriched_corpus_path=sources.get('enriched_corpus', Path('nonexistent')),
        test_examples_path=sources.get('test_examples', Path('nonexistent')),
        synthetic_examples_path=sources.get('synthetic_examples', Path('nonexistent')),
        output_path=args.output,
        validation_split=args.val_split
    )

    print(f"✓ Created training dataset:")
    print(f"  Total examples: {dataset_stats['total_examples']}")
    print(f"  Training: {dataset_stats['train_examples']} ({dataset_stats['train_examples']/dataset_stats['total_examples']*100:.1f}%)")
    print(f"  Validation: {dataset_stats['val_examples']} ({dataset_stats['val_examples']/dataset_stats['total_examples']*100:.1f}%)")
    print(f"  Source distribution: {dataset_stats['source_distribution']}")
    print(f"  Saved to: {args.output}/train.jsonl and {args.output}/val.jsonl")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training data ready for entity type classifier!")
    print(f"Location: {args.output}/")
    print(f"  - train.jsonl: {dataset_stats['train_examples']} examples")
    print(f"  - val.jsonl: {dataset_stats['val_examples']} examples")
    print(f"\nNext step: Train model using this dataset (Task 1.5)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
