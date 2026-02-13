#!/usr/bin/env python3
"""
Regenerate Training Data V3 - With Variations.

Improvements over V2:
- Expanded semantic roots vocabulary (512 roots, 27 categories)
- Multiple variations per root (different cases, numbers)
- Generates 5,000+ semantic examples to hit 50% target

Generates THREE types of examples:
1. High-confidence (30%): Affix-based, teach model to trust deterministic
2. Semantic roots (50%): Root meaning, teach model to fill semantic gaps
3. Ambiguous (20%): Context-dependent, teach model to use context
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.semantic_enrichment.taxonomy import PersonType, LocationType, TimeType, ThingType


def filter_corpus_by_confidence(input_path: Path, output_dir: Path):
    """
    Filter corpus into confidence buckets.
    Returns paths to high_confidence.jsonl and medium_confidence.jsonl
    """
    print("="*60)
    print("STEP 1: Filter Corpus by Confidence")
    print("="*60)
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    high_path = output_dir / 'high_confidence.jsonl'
    medium_path = output_dir / 'medium_confidence.jsonl'

    total = 0
    high_count = 0
    medium_count = 0
    unlabeled_count = 0

    with open(input_path, 'r') as f_in, \
         open(high_path, 'w') as f_high, \
         open(medium_path, 'w') as f_medium:

        for line in f_in:
            if not line.strip():
                continue

            total += 1
            data = json.loads(line)

            semantic = data.get('semantic_annotation', {})
            det = semantic.get('deterministic_features', {})

            tier3 = det.get('tier3_type')
            if tier3 is None:
                unlabeled_count += 1
                continue

            conf = det.get('confidence', 0)

            training_example = {
                'word_ast': {k: v for k, v in data.items() if k != 'semantic_annotation'},
                'deterministic_priors': {
                    'tier1_category': det.get('tier1_category'),
                    'tier2_type': det.get('tier2_type'),
                    'tier3_type': tier3,
                    'confidence': conf,
                    'evidence': det.get('evidence', {})
                },
                'label': {
                    'tier3_type': tier3,
                    'confidence': conf,
                    'source': 'corpus'
                }
            }

            if conf >= 0.9:
                f_high.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                high_count += 1
            else:
                f_medium.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                medium_count += 1

            if total % 100000 == 0:
                print(f"  Processed {total:,} examples...")

    print()
    print(f"✓ Processed {total:,} corpus examples")
    print(f"  High confidence (≥0.9): {high_count:,}")
    print(f"  Medium confidence (0.5-0.9): {medium_count:,}")
    print(f"  Unlabeled (skipped): {unlabeled_count:,}")
    print()

    return high_path, medium_path


def generate_affix_based_examples(root_vocab_path: Path, samples_per_type: int = 300):
    """Generate TYPE 1: High-confidence affix-based examples."""
    print("="*60)
    print("STEP 2: Generate Affix-Based Examples (Type 1)")
    print("="*60)
    print()

    with open(root_vocab_path, 'r') as f:
        roots = json.load(f)

    print(f"Loaded {len(roots)} roots")
    print()

    affix_mappings = [
        ('ist', PersonType.PERSON_PROFESSION.value, 'person', 0.95),
        ('ul', PersonType.PERSON_ROLE.value, 'person', 0.95),
        ('in', PersonType.PERSON_ROLE.value, 'person', 0.90),
        ('ar', PersonType.PERSON_GROUP.value, 'person', 0.90),
        ('ej', 'place_institution', 'location', 0.90),
        ('il', 'thing_tool', 'thing', 0.90),
        ('aĵ', 'thing_concrete', 'thing', 0.85),
        ('ar', 'thing_collection', 'thing', 0.90),
    ]

    examples = []

    for affix, tier3_type, tier2_type, confidence in affix_mappings:
        print(f"  Generating {samples_per_type} examples for {tier3_type} (affix: -{affix})")

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
                    'tier2_type': tier2_type,
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
                    'source': 'synthetic_affix'
                }
            }

            examples.append(example)

    print()
    print(f"✓ Generated {len(examples):,} affix-based examples")
    print()

    return examples


def generate_semantic_root_examples(semantic_roots_path: Path, variations_per_root: int = 4):
    """
    Generate TYPE 2: Semantic root examples with variations.

    Creates multiple grammatical variations per root:
    - nominativo/singularo, nominativo/pluralo
    - akuzativo/singularo, akuzativo/pluralo

    This teaches model that case/number don't change entity type.
    """
    print("="*60)
    print("STEP 3: Generate Semantic Root Examples (Type 2)")
    print("="*60)
    print()

    with open(semantic_roots_path, 'r') as f:
        semantic_roots = json.load(f)

    print(f"Loaded {len(semantic_roots)} semantic categories")
    print(f"Generating {variations_per_root} variations per root")
    print()

    # Grammatical variations to generate
    variations = [
        ('nominativo', 'singularo', 'o'),
        ('nominativo', 'pluralo', 'oj'),
        ('akuzativo', 'singularo', 'on'),
        ('akuzativo', 'pluralo', 'ojn'),
    ][:variations_per_root]  # Take only requested number

    examples = []
    total_roots = sum(len(roots) for roots in semantic_roots.values())

    for tier3_type, roots in semantic_roots.items():
        print(f"  {tier3_type}: {len(roots)} roots × {len(variations)} variations = {len(roots) * len(variations)} examples")

        # Determine tier2 from tier3
        if tier3_type.startswith('person'):
            tier2_type = 'person'
        elif tier3_type.startswith('place'):
            tier2_type = 'location'
        elif tier3_type.startswith('time'):
            tier2_type = 'time_point'
        else:
            tier2_type = 'thing'

        for root in roots:
            # Remove Esperanto characters that might be at the end
            root = root.rstrip('ĝĉĵŝŭ')

            # Handle roots that already have -o ending (like "patro", "urbo")
            if root.endswith('o'):
                base_root = root[:-1]  # Remove -o
            else:
                base_root = root

            # Generate variations
            for kazo, nombro, ending in variations:
                word = f"{base_root}{ending}"

                example = {
                    'word_ast': {
                        'tipo': 'vorto',
                        'vortspeco': 'substantivo',
                        'radiko': base_root,
                        'sufiksoj': [],  # NO informative suffix!
                        'teksto': word,
                        'kazo': kazo,
                        'nombro': nombro,
                        'parse_status': 'success'
                    },
                    'deterministic_priors': {
                        'tier1_category': 'entity',
                        'tier2_type': tier2_type,
                        'tier3_type': None,  # Deterministic CAN'T determine!
                        'confidence': 0.3,   # Low confidence - needs model
                        'evidence': {
                            'reason': 'no_informative_affix'
                        }
                    },
                    'label': {
                        'tier3_type': tier3_type,
                        'confidence': 1.0,
                        'source': 'synthetic_semantic'
                    }
                }

                examples.append(example)

    print()
    print(f"✓ Generated {len(examples):,} semantic root examples")
    print(f"  ({total_roots} roots × {len(variations)} variations)")
    print()

    return examples


def generate_ambiguous_examples(root_vocab_path: Path, samples_per_category: int = 50):
    """
    Generate TYPE 3: Ambiguous context-dependent examples.

    Expanded to generate more ambiguous examples across different patterns.
    """
    print("="*60)
    print("STEP 4: Generate Ambiguous Examples (Type 3)")
    print("="*60)
    print()

    with open(root_vocab_path, 'r') as f:
        roots = json.load(f)

    examples = []

    # Pattern 1: -ar suffix (person_group vs thing_collection)
    person_roots = ['hom', 'vir', 'infan', 'stud', 'labor', 'amik', 'soldat', 'marĝen']
    thing_roots = ['arb', 'libr', 'flor', 'ŝton', 'best', 'insekt', 'fiŝ', 'bird']

    print(f"  Pattern 1: -ar suffix ambiguity")
    print(f"    person_group: {len(person_roots)} roots")
    for root in person_roots:
        word = f"{root}aro"
        example = {
            'word_ast': {
                'tipo': 'vorto',
                'vortspeco': 'substantivo',
                'radiko': root,
                'sufiksoj': ['ar'],
                'teksto': word,
                'kazo': 'nominativo',
                'nombro': 'singularo',
                'parse_status': 'success'
            },
            'deterministic_priors': {
                'tier1_category': 'entity',
                'tier2_type': 'person',
                'tier3_type': PersonType.PERSON_GROUP.value,
                'confidence': 0.7,
                'evidence': {
                    'affix': 'ar',
                    'ambiguity': 'person_group_vs_thing_collection'
                }
            },
            'label': {
                'tier3_type': PersonType.PERSON_GROUP.value,
                'confidence': 1.0,
                'source': 'synthetic_ambiguous'
            }
        }
        examples.append(example)

    print(f"    thing_collection: {len(thing_roots)} roots")
    for root in thing_roots:
        word = f"{root}aro"
        example = {
            'word_ast': {
                'tipo': 'vorto',
                'vortspeco': 'substantivo',
                'radiko': root,
                'sufiksoj': ['ar'],
                'teksto': word,
                'kazo': 'nominativo',
                'nombro': 'singularo',
                'parse_status': 'success'
            },
            'deterministic_priors': {
                'tier1_category': 'entity',
                'tier2_type': 'thing',
                'tier3_type': 'thing_collection',
                'confidence': 0.7,
                'evidence': {
                    'affix': 'ar',
                    'ambiguity': 'person_group_vs_thing_collection'
                }
            },
            'label': {
                'tier3_type': 'thing_collection',
                'confidence': 1.0,
                'source': 'synthetic_ambiguous'
            }
        }
        examples.append(example)

    print()
    print(f"✓ Generated {len(examples):,} ambiguous examples")
    print()

    return examples


def create_balanced_dataset(
    high_confidence_corpus: Path,
    medium_confidence_corpus: Path,
    affix_examples: list,
    semantic_examples: list,
    ambiguous_examples: list,
    output_dir: Path,
    target_size: int = 10000,
    val_split: float = 0.15
):
    """
    Combine all examples into balanced dataset.

    Target distribution:
    - 30% high-confidence (affix-based)
    - 50% semantic roots
    - 20% ambiguous
    """
    print("="*60)
    print("STEP 5: Create Balanced Dataset")
    print("="*60)
    print()

    target_high = int(target_size * 0.30)
    target_semantic = int(target_size * 0.50)
    target_ambiguous = int(target_size * 0.20)

    print(f"Target distribution for {target_size:,} examples:")
    print(f"  High-confidence: {target_high:,} (30%)")
    print(f"  Semantic roots: {target_semantic:,} (50%)")
    print(f"  Ambiguous: {target_ambiguous:,} (20%)")
    print()

    all_examples = []

    # 1. High-confidence
    print("Sampling high-confidence examples...")
    high_corpus = []
    with open(high_confidence_corpus, 'r') as f:
        for line in f:
            if line.strip():
                high_corpus.append(json.loads(line))

    high_pool = high_corpus + affix_examples
    random.shuffle(high_pool)
    high_sampled = high_pool[:target_high]
    all_examples.extend(high_sampled)
    print(f"  Selected {len(high_sampled):,} (target: {target_high:,})")

    # 2. Semantic roots
    print("Sampling semantic root examples...")
    random.shuffle(semantic_examples)
    semantic_sampled = semantic_examples[:target_semantic]
    all_examples.extend(semantic_sampled)
    print(f"  Selected {len(semantic_sampled):,} (target: {target_semantic:,})")

    # 3. Ambiguous
    print("Sampling ambiguous examples...")
    medium_corpus = []
    with open(medium_confidence_corpus, 'r') as f:
        for line in f:
            if line.strip():
                medium_corpus.append(json.loads(line))

    ambiguous_pool = medium_corpus + ambiguous_examples
    random.shuffle(ambiguous_pool)
    ambiguous_sampled = ambiguous_pool[:target_ambiguous]
    all_examples.extend(ambiguous_sampled)
    print(f"  Selected {len(ambiguous_sampled):,} (target: {target_ambiguous:,})")

    # Shuffle all
    random.shuffle(all_examples)

    # Split train/val
    val_size = int(len(all_examples) * val_split)
    train_examples = all_examples[val_size:]
    val_examples = all_examples[:val_size]

    print()
    print(f"Total examples: {len(all_examples):,}")
    print(f"  Training: {len(train_examples):,} ({len(train_examples)/len(all_examples)*100:.1f}%)")
    print(f"  Validation: {len(val_examples):,} ({len(val_examples)/len(all_examples)*100:.1f}%)")

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

    parser = argparse.ArgumentParser(description='Regenerate training data V3')
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
        '--semantic-roots',
        type=Path,
        default=Path('data/vocabularies/semantic_roots_expanded.json'),
        help='Path to semantic roots vocabulary'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/entity_classifier_v3'),
        help='Output directory'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=10000,
        help='Target dataset size'
    )
    parser.add_argument(
        '--variations-per-root',
        type=int,
        default=4,
        help='Grammatical variations per semantic root (1-4)'
    )

    args = parser.parse_args()

    print("="*60)
    print("REGENERATE TRAINING DATA V3")
    print("="*60)
    print()
    print("Strategy:")
    print("  - 30% high-confidence (affix-based, trust deterministic)")
    print("  - 50% semantic roots (fill gaps when no affix)")
    print("  - 20% ambiguous (use context to disambiguate)")
    print()
    print(f"Semantic root variations: {args.variations_per_root}")
    print()

    # Check inputs
    if not args.corpus_labeled.exists():
        print(f"ERROR: Labeled corpus not found: {args.corpus_labeled}")
        sys.exit(1)

    if not args.root_vocab.exists():
        print(f"ERROR: Root vocabulary not found: {args.root_vocab}")
        sys.exit(1)

    if not args.semantic_roots.exists():
        print(f"ERROR: Semantic roots not found: {args.semantic_roots}")
        sys.exit(1)

    # Create temp directory
    temp_dir = args.output / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Filter corpus
    high_corpus, medium_corpus = filter_corpus_by_confidence(
        args.corpus_labeled,
        temp_dir
    )

    # Step 2: Generate affix-based examples
    affix_examples = generate_affix_based_examples(
        args.root_vocab,
        samples_per_type=300
    )

    # Step 3: Generate semantic root examples with variations
    semantic_examples = generate_semantic_root_examples(
        args.semantic_roots,
        variations_per_root=args.variations_per_root
    )

    # Step 4: Generate ambiguous examples
    ambiguous_examples = generate_ambiguous_examples(
        args.root_vocab,
        samples_per_category=50
    )

    # Step 5: Create balanced dataset
    create_balanced_dataset(
        high_corpus,
        medium_corpus,
        affix_examples,
        semantic_examples,
        ambiguous_examples,
        args.output,
        target_size=args.target_size
    )

    print("="*60)
    print("COMPLETE")
    print("="*60)
    print()
    print(f"V3 training data ready at: {args.output}/")
    print()
    print("Next step: Train the model")
    print(f"  python scripts/train_entity_classifier.py \\")
    print(f"      --data {args.output} \\")
    print(f"      --output models/entity_classifier \\")
    print(f"      --epochs 50")
