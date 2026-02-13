#!/usr/bin/env python3
"""
Regenerate Training Data with Improvements.

Fixes data quality issues:
1. Filter corpus to only examples with tier3_type labels
2. Generate balanced synthetic examples
3. Ensure coverage of all entity types
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.semantic_enrichment.taxonomy import PersonType, LocationType, TimeType, ThingType


def filter_corpus_examples(input_path: Path, output_path: Path):
    """Filter corpus to only high-quality labeled examples."""
    print("="*60)
    print("STEP 1: Filter Corpus Examples")
    print("="*60)
    print()

    total = 0
    kept = 0
    tier3_counts = Counter()

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue

            total += 1
            data = json.loads(line)

            # Extract tier3 from semantic_annotation (enriched corpus format)
            semantic_annotation = data.get('semantic_annotation', {})
            det_features = semantic_annotation.get('deterministic_features', {})

            tier3 = det_features.get('tier3_type')
            if tier3 is None:
                continue

            # Only keep high confidence (>=0.9)
            conf = det_features.get('confidence', 0)
            if conf < 0.9:
                continue

            # Convert to training format
            training_example = {
                'word_ast': {k: v for k, v in data.items() if k != 'semantic_annotation'},
                'deterministic_priors': {
                    'tier1_category': det_features.get('tier1_category'),
                    'tier2_type': det_features.get('tier2_type'),
                    'tier3_type': tier3,
                    'confidence': conf,
                    'evidence': det_features.get('evidence', {})
                },
                'label': {
                    'tier3_type': tier3,
                    'confidence': conf,
                    'source': 'corpus'
                }
            }

            kept += 1
            tier3_counts[tier3] += 1
            f_out.write(json.dumps(training_example, ensure_ascii=False) + '\n')

            if total % 100000 == 0:
                print(f"  Processed {total:,} examples, kept {kept:,}")

    print()
    print(f"✓ Filtered {total:,} examples → {kept:,} high-quality examples")
    print(f"  Kept: {kept/total*100:.1f}%")
    print()
    print("Tier 3 type distribution:")
    for tier3, count in tier3_counts.most_common():
        print(f"  {tier3:30s}: {count:6,} examples")
    print()

    return kept, dict(tier3_counts)


def generate_synthetic_balanced(root_vocab_path: Path, output_path: Path, samples_per_type: int = 500):
    """Generate balanced synthetic examples covering all entity types."""
    print("="*60)
    print("STEP 2: Generate Balanced Synthetic Examples")
    print("="*60)
    print()

    # Load roots
    with open(root_vocab_path, 'r') as f:
        roots = json.load(f)

    print(f"Loaded {len(roots)} roots from vocabulary")
    print()

    # Affix mappings with tier3 types
    affix_mappings = [
        # PersonType
        ('ist', PersonType.PERSON_PROFESSION.value, 0.95),
        ('ul', PersonType.PERSON_ROLE.value, 0.95),
        ('in', PersonType.PERSON_ROLE.value, 0.90),
        ('an', PersonType.PERSON_ROLE.value, 0.90),
        ('ar', PersonType.PERSON_GROUP.value, 0.90),  # -ar for groups

        # LocationType
        ('ej', 'place_institution', 0.90),

        # ThingType
        ('il', 'thing_tool', 0.90),
        ('aĵ', 'thing_concrete', 0.85),
        ('ar', 'thing_collection', 0.90),  # -ar for collections too
    ]

    examples = []

    for affix, tier3_type, confidence in affix_mappings:
        print(f"Generating {samples_per_type} examples for {tier3_type} (affix: -{affix})")

        # Sample roots
        sampled_roots = random.sample(roots, min(samples_per_type, len(roots)))

        for root in sampled_roots:
            if isinstance(root, dict):
                root = root.get('root', root.get('radiko', ''))

            word = f"{root}{affix}o"

            example = {
                'word_ast': {
                    'tipo': 'vorto',
                    'vortspeco': 'substantivo',
                    'radiko': root,
                    'sufiksoj': [affix],
                    'teksto': word,
                    'kazo': 'nominativo',
                    'nombro': 'singularo',
                    'parse_status': 'success'
                },
                'deterministic_priors': {
                    'tier1_category': 'entity',
                    'tier2_type': 'person' if tier3_type.startswith('person') else
                                  'location' if tier3_type.startswith('place') else
                                  'thing',
                    'tier3_type': tier3_type,
                    'confidence': confidence,
                    'evidence': {
                        'affix': affix,
                        'affix_confidence': confidence
                    }
                },
                'label': {
                    'tier3_type': tier3_type,
                    'confidence': confidence,
                    'source': 'synthetic'
                }
            }

            examples.append(example)

    # Shuffle
    random.shuffle(examples)

    # Save
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print()
    print(f"✓ Generated {len(examples):,} synthetic examples")

    # Count by type
    type_counter = Counter(ex['label']['tier3_type'] for ex in examples)
    print()
    print("Synthetic distribution:")
    for tier3, count in type_counter.most_common():
        print(f"  {tier3:30s}: {count:6,} examples")
    print()

    return len(examples), dict(type_counter)


def create_balanced_dataset(
    filtered_corpus_path: Path,
    synthetic_path: Path,
    output_dir: Path,
    max_per_type: int = 1000,
    val_split: float = 0.15
):
    """Combine and balance the dataset."""
    print("="*60)
    print("STEP 3: Create Balanced Dataset")
    print("="*60)
    print()

    # Collect all examples by type
    examples_by_type = defaultdict(list)

    # Load corpus examples
    print("Loading filtered corpus examples...")
    with open(filtered_corpus_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            tier3 = example['label']['tier3_type']
            examples_by_type[tier3].append(example)

    # Load synthetic examples
    print("Loading synthetic examples...")
    with open(synthetic_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            tier3 = example['label']['tier3_type']
            examples_by_type[tier3].append(example)

    print()
    print(f"Collected examples for {len(examples_by_type)} entity types")

    # Balance by sampling up to max_per_type from each
    balanced_examples = []
    for tier3, examples in examples_by_type.items():
        # Shuffle
        random.shuffle(examples)

        # Take up to max_per_type
        sampled = examples[:max_per_type]
        balanced_examples.extend(sampled)

        print(f"  {tier3:30s}: {len(examples):6,} → {len(sampled):6,} (sampled)")

    # Shuffle all
    random.shuffle(balanced_examples)

    # Split train/val
    val_size = int(len(balanced_examples) * val_split)
    train_examples = balanced_examples[val_size:]
    val_examples = balanced_examples[:val_size]

    print()
    print(f"Total examples: {len(balanced_examples):,}")
    print(f"  Training: {len(train_examples):,} ({len(train_examples)/len(balanced_examples)*100:.1f}%)")
    print(f"  Validation: {len(val_examples):,} ({len(val_examples)/len(balanced_examples)*100:.1f}%)")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'

    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    with open(val_path, 'w') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print()
    print(f"✓ Saved to {output_dir}/")
    print(f"  - train.jsonl: {len(train_examples):,} examples")
    print(f"  - val.jsonl: {len(val_examples):,} examples")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Regenerate improved training data')
    parser.add_argument(
        '--corpus-labeled',
        type=Path,
        default=Path('data/training/entity_classifier/enriched_corpus.jsonl'),
        help='Path to auto-labeled corpus'
    )
    parser.add_argument(
        '--root-vocab',
        type=Path,
        default=Path('data/vocabularies/root_vocab.json'),
        help='Path to root vocabulary'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/entity_classifier_improved'),
        help='Output directory'
    )
    parser.add_argument(
        '--max-per-type',
        type=int,
        default=1000,
        help='Max examples per entity type (for balancing)'
    )
    parser.add_argument(
        '--synthetic-per-type',
        type=int,
        default=500,
        help='Synthetic examples to generate per type'
    )

    args = parser.parse_args()

    print("="*60)
    print("REGENERATE IMPROVED TRAINING DATA")
    print("="*60)
    print()
    print("This will:")
    print("  1. Filter corpus to only labeled examples (tier3_type != None)")
    print("  2. Generate balanced synthetic examples")
    print("  3. Create balanced train/val split")
    print()

    # Check inputs
    if not args.corpus_labeled.exists():
        print(f"ERROR: Labeled corpus not found: {args.corpus_labeled}")
        sys.exit(1)

    if not args.root_vocab.exists():
        print(f"ERROR: Root vocabulary not found: {args.root_vocab}")
        print("Run: python scripts/create_root_vocabulary.py")
        sys.exit(1)

    # Temporary paths
    filtered_corpus = args.output / 'filtered_corpus.jsonl'
    synthetic_examples = args.output / 'synthetic_examples.jsonl'

    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Filter corpus
    corpus_count, corpus_dist = filter_corpus_examples(
        args.corpus_labeled,
        filtered_corpus
    )

    # Step 2: Generate synthetic
    synthetic_count, synthetic_dist = generate_synthetic_balanced(
        args.root_vocab,
        synthetic_examples,
        samples_per_type=args.synthetic_per_type
    )

    # Step 3: Create balanced dataset
    create_balanced_dataset(
        filtered_corpus,
        synthetic_examples,
        args.output,
        max_per_type=args.max_per_type
    )

    print("="*60)
    print("COMPLETE")
    print("="*60)
    print()
    print(f"Improved training data ready at: {args.output}/")
    print()
    print("Next step: Train the model")
    print(f"  ./scripts/train_entity_classifier.sh --data {args.output}")
