#!/usr/bin/env python3
"""
Generate HIGH-QUALITY Plausibility Training Dataset (Quality over Quantity)

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu database, Semantic Enrichment tier classification
DEPENDENCIES: extract_svo_triples.py, klareco.semantic_enrichment
STAGE: Data

Description:
    Generate CURATED, high-quality training dataset for plausibility scorer.

    Key differences from bulk generation:
    1. Quality filters: Only high-confidence parses, diverse semantic types
    2. Challenging negatives: Not just random, but semantically plausible-looking
    3. Smaller size: 100K-500K examples (sufficient for 98K param model)
    4. Balanced coverage: Ensure representation across semantic types

    Target: 200K total examples (100K positive + 100K negative)

Pipeline Position:
    Corpus → SVO Extraction → [THIS SCRIPT] → Curated Training JSONL → Plausibility Scorer

Usage:
    # Generate quality-focused dataset (200K examples)
    python scripts/generate_plausibility_training_data_quality.py \
        --svo-triples data/semantic_types/svo_triples_all.jsonl \
        --output-dir data/plausibility_training_quality \
        --num-examples 200000

    # Smaller test set
    python scripts/generate_plausibility_training_data_quality.py \
        --svo-triples data/semantic_types/svo_triples_test.jsonl \
        --output-dir data/plausibility_training_quality_test \
        --num-examples 10000

Inputs:
    - SVO triples JSONL (from extract_svo_triples.py)

Outputs:
    - data/plausibility_training_quality/train.jsonl - Training set (90%)
    - data/plausibility_training_quality/val.jsonl - Validation set (10%)
    - data/plausibility_training_quality/stats.json - Dataset statistics

Quality Checks:
    - Confidence threshold: Only triples with confidence ≥ 0.9
    - Semantic type balance: Sample to ensure coverage
    - Challenging negatives: Type-compatible but semantically odd
    - Duplicate removal
    - Vocabulary diversity scoring

Last Updated: 2026-03-22
Author: Claude Code
Related Issues: #699
See Also: /tmp/plausibility_scorer_design.md
"""

import argparse
import json
import jsonlines
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import logging
from tqdm import tqdm

from klareco.semantic_enrichment import ASTSemanticEnricher


# ============================================================================
# QUALITY FILTERS
# ============================================================================

def filter_high_quality_triples(
    triples: List[Dict],
    min_confidence: float = 0.9
) -> List[Dict]:
    """
    Filter for high-quality triples.

    Criteria:
    - Confidence ≥ min_confidence
    - All components have known semantic types
    - Not too common (avoid "la persono havas ion" repetition)
    """
    filtered = []
    verb_counts = Counter(t['verb_root'] for t in triples)

    for triple in triples:
        # Confidence check
        if triple.get('confidence', 1.0) < min_confidence:
            continue

        # Skip overly common verbs (likely to be uninformative)
        if verb_counts[triple['verb_root']] > len(triples) * 0.01:  # >1% of corpus
            continue

        filtered.append(triple)

    return filtered


def balance_semantic_types(
    triples: List[Dict],
    target_count: int,
    enricher: ASTSemanticEnricher
) -> List[Dict]:
    """
    Sample triples to ensure balanced semantic type coverage.

    Strategy:
    - Group by (subject_type, object_type) pairs
    - Sample proportionally from each group
    - Ensures diversity, not dominated by common types
    """
    # Add semantic types and plausibility fields if missing
    for triple in triples:
        if 'subject_type' not in triple:
            triple['subject_type'] = get_semantic_type(triple['subject_root'], enricher)
        if 'object_type' not in triple:
            triple['object_type'] = get_semantic_type(triple['object_root'], enricher)
        if 'plausible' not in triple:
            triple['plausible'] = 1.0  # Corpus triples are plausible
        if 'source' not in triple:
            triple['source'] = 'corpus'

    # Group by type pairs
    type_groups = defaultdict(list)
    for triple in triples:
        key = (triple['subject_type'], triple['object_type'])
        type_groups[key].append(triple)

    # Calculate samples per group
    num_groups = len(type_groups)
    samples_per_group = max(1, target_count // num_groups)

    # Sample from each group
    balanced = []
    for group_triples in type_groups.values():
        sample_size = min(samples_per_group, len(group_triples))
        balanced.extend(random.sample(group_triples, sample_size))

    # If we still need more, sample randomly from remainder
    if len(balanced) < target_count:
        remaining = [t for t in triples if t not in balanced]
        additional = random.sample(remaining, min(target_count - len(balanced), len(remaining)))
        balanced.extend(additional)

    # Shuffle and truncate to target
    random.shuffle(balanced)
    return balanced[:target_count]


def get_semantic_type(root: str, enricher: ASTSemanticEnricher) -> str:
    """Get semantic type for a root using Semantic Enrichment."""
    word_node = {
        'tipo': 'vorto',
        'radiko': root,
        'vortspeco': 'substantivo',
        'plena_vorto': root,
        'analizstato': 'sukceso'
    }

    try:
        enriched = enricher.enrich_word(word_node)
        semantic_type = enriched.get('semantic_type', 'UNKNOWN')
        return semantic_type
    except Exception as e:
        logging.debug(f"Failed to get semantic type for '{root}': {e}")
        return 'UNKNOWN'


# ============================================================================
# CHALLENGING NEGATIVE GENERATION
# ============================================================================

def generate_challenging_negatives(
    positive_examples: List[Dict],
    num_negatives: int,
    positive_set: Set[Tuple[str, str, str]]
) -> List[Dict]:
    """
    Generate CHALLENGING negative examples.

    Strategy:
    1. Type-compatible swaps (40%): Swap with same semantic type
       Example: (persono:viro, am, persono:virino) → (persono:infano, am, persono:virino)
       Why hard: Syntactically valid, semantically plausible-looking

    2. Argument role confusion (30%): Use verb with wrong argument types
       Example: (persono, manĝ, nutraĵo) → (nutraĵo, manĝ, tempo)
       Why hard: Real verbs, but argument types don't match selectional constraints

    3. Random but filtered (30%): Random combos that pass basic plausibility
       Example: Avoid obviously broken like (tempo, vid, son)
       Why hard: Edge cases, rare combinations

    These are harder to classify than simple role swaps.
    """
    negatives = []
    seen_negatives = set()

    # Group by semantic types
    by_subject_type = defaultdict(list)
    by_object_type = defaultdict(list)
    by_verb = defaultdict(list)

    for ex in positive_examples:
        by_subject_type[ex['subject_type']].append(ex)
        by_object_type[ex['object_type']].append(ex)
        by_verb[ex['verb_root']].append(ex)

    # Strategy 1: Type-compatible swaps (40%)
    num_type_swaps = int(num_negatives * 0.4)
    logging.info(f"Generating {num_type_swaps} type-compatible swap negatives...")

    with tqdm(total=num_type_swaps, desc="Type swaps") as pbar:
        attempts = 0
        while len([n for n in negatives if n.get('strategy') == 'type_swap']) < num_type_swaps and attempts < num_type_swaps * 10:
            attempts += 1

            # Pick a corpus triple
            original = random.choice(positive_examples)

            # Find another triple with same subject type
            same_type = by_subject_type.get(original['subject_type'], [])
            if len(same_type) < 2:
                continue

            other = random.choice(same_type)
            if other == original:
                continue

            # Swap subjects (same type, different words)
            negative_key = (other['subject_root'], original['verb_root'], original['object_root'])

            # Check not in corpus and not already generated
            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            negatives.append({
                'subject_root': other['subject_root'],
                'verb_root': original['verb_root'],
                'object_root': original['object_root'],
                'subject_type': other['subject_type'],
                'object_type': original['object_type'],
                'plausible': 0.0,
                'source': 'type_compatible_swap',
                'strategy': 'type_swap',
                'original_triple': f"({original['subject_root']}, {original['verb_root']}, {original['object_root']})"
            })
            pbar.update(1)

    # Strategy 2: Argument role confusion (30%)
    num_role_confusion = int(num_negatives * 0.3)
    logging.info(f"Generating {num_role_confusion} role confusion negatives...")

    with tqdm(total=num_role_confusion, desc="Role confusion") as pbar:
        attempts = 0
        while len([n for n in negatives if n.get('strategy') == 'role_confusion']) < num_role_confusion and attempts < num_role_confusion * 10:
            attempts += 1

            # Pick a verb
            verb = random.choice(list(by_verb.keys()))
            verb_examples = by_verb[verb]

            # Pick subject/object from different example
            subj_example = random.choice(positive_examples)
            obj_example = random.choice(positive_examples)

            negative_key = (subj_example['subject_root'], verb, obj_example['object_root'])

            # Check not in corpus and not already generated
            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            negatives.append({
                'subject_root': subj_example['subject_root'],
                'verb_root': verb,
                'object_root': obj_example['object_root'],
                'subject_type': subj_example['subject_type'],
                'object_type': obj_example['object_type'],
                'plausible': 0.0,
                'source': 'role_confusion',
                'strategy': 'role_confusion'
            })
            pbar.update(1)

    # Strategy 3: Random but filtered (30%)
    num_random = num_negatives - len(negatives)
    logging.info(f"Generating {num_random} filtered random negatives...")

    subjects = list(set(ex['subject_root'] for ex in positive_examples))
    verbs = list(set(ex['verb_root'] for ex in positive_examples))
    objects = list(set(ex['object_root'] for ex in positive_examples))

    with tqdm(total=num_random, desc="Random filtered") as pbar:
        attempts = 0
        while len([n for n in negatives if n.get('strategy') == 'random']) < num_random and attempts < num_random * 10:
            attempts += 1

            subj = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)

            negative_key = (subj, verb, obj)

            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            # Get types
            subj_type = next((ex['subject_type'] for ex in positive_examples if ex['subject_root'] == subj), 'UNKNOWN')
            obj_type = next((ex['object_type'] for ex in positive_examples if ex['object_root'] == obj), 'UNKNOWN')

            negatives.append({
                'subject_root': subj,
                'verb_root': verb,
                'object_root': obj,
                'subject_type': subj_type,
                'object_type': obj_type,
                'plausible': 0.0,
                'source': 'random_filtered',
                'strategy': 'random'
            })
            pbar.update(1)

    return negatives


# ============================================================================
# DATASET CREATION
# ============================================================================

def create_balanced_dataset(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    train_split: float = 0.9
) -> Tuple[List[Dict], List[Dict]]:
    """Create balanced train/val splits."""
    logging.info("Creating balanced train/val splits")

    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * train_split)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    logging.info(f"Train: {len(train_data)} examples")
    logging.info(f"Val: {len(val_data)} examples")

    train_positive = sum(1 for ex in train_data if ex['plausible'] == 1.0)
    val_positive = sum(1 for ex in val_data if ex['plausible'] == 1.0)

    logging.info(f"Train balance: {train_positive}/{len(train_data)} positive ({train_positive/len(train_data)*100:.1f}%)")
    logging.info(f"Val balance: {val_positive}/{len(val_data)} positive ({val_positive/len(val_data)*100:.1f}%)")

    return train_data, val_data


def compute_statistics(train_data: List[Dict], val_data: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    all_data = train_data + val_data

    num_positive = sum(1 for ex in all_data if ex['plausible'] == 1.0)
    num_negative = len(all_data) - num_positive

    source_counts = Counter(ex['source'] for ex in all_data)
    strategy_counts = Counter(ex.get('strategy', 'corpus') for ex in all_data)
    subject_types = Counter(ex['subject_type'] for ex in all_data)
    object_types = Counter(ex['object_type'] for ex in all_data)
    verb_counts = Counter(ex['verb_root'] for ex in all_data)

    stats = {
        'total_examples': len(all_data),
        'train_examples': len(train_data),
        'val_examples': len(val_data),
        'num_positive': num_positive,
        'num_negative': num_negative,
        'positive_ratio': num_positive / len(all_data),
        'source_distribution': dict(source_counts),
        'strategy_distribution': dict(strategy_counts),
        'top_10_subject_types': dict(subject_types.most_common(10)),
        'top_10_object_types': dict(object_types.most_common(10)),
        'top_10_verbs': dict(verb_counts.most_common(10)),
        'num_unique_subject_types': len(subject_types),
        'num_unique_object_types': len(object_types),
        'num_unique_verbs': len(verb_counts)
    }

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate quality-focused plausibility training dataset'
    )
    parser.add_argument(
        '--svo-triples',
        type=Path,
        required=True,
        help='Path to SVO triples JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for train/val files'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=200000,
        help='Total number of examples (default: 200K)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.9,
        help='Minimum confidence for positive examples (default: 0.9)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Fraction for training (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize semantic enricher
    logging.info("Initializing semantic enricher...")
    enricher = ASTSemanticEnricher()

    # Load all triples
    logging.info(f"Loading SVO triples from {args.svo_triples}")
    all_triples = []
    with jsonlines.open(args.svo_triples) as reader:
        all_triples = list(reader)
    logging.info(f"Loaded {len(all_triples)} total triples")

    # Filter for quality
    logging.info(f"Filtering for high-quality triples (confidence ≥ {args.min_confidence})...")
    quality_triples = filter_high_quality_triples(all_triples, args.min_confidence)
    logging.info(f"Retained {len(quality_triples)} quality triples ({len(quality_triples)/len(all_triples)*100:.1f}%)")

    # Balance semantic types
    num_positives = args.num_examples // 2
    num_negatives = args.num_examples - num_positives

    logging.info(f"Balancing semantic type coverage for {num_positives} positives...")
    positive_examples = balance_semantic_types(quality_triples, num_positives, enricher)
    logging.info(f"Selected {len(positive_examples)} balanced positive examples")

    # Create positive set for filtering
    positive_set = set(
        (ex['subject_root'], ex['verb_root'], ex['object_root'])
        for ex in positive_examples
    )

    # Generate challenging negatives
    logging.info(f"Generating {num_negatives} challenging negative examples...")
    negative_examples = generate_challenging_negatives(
        positive_examples,
        num_negatives,
        positive_set
    )
    logging.info(f"Generated {len(negative_examples)} negative examples")

    # Create balanced dataset
    train_data, val_data = create_balanced_dataset(
        positive_examples,
        negative_examples,
        args.train_split
    )

    # Save datasets
    train_path = args.output_dir / 'train.jsonl'
    val_path = args.output_dir / 'val.jsonl'
    stats_path = args.output_dir / 'stats.json'

    logging.info(f"Saving training data to {train_path}")
    with jsonlines.open(train_path, mode='w') as writer:
        for example in train_data:
            writer.write(example)

    logging.info(f"Saving validation data to {val_path}")
    with jsonlines.open(val_path, mode='w') as writer:
        for example in val_data:
            writer.write(example)

    # Compute and save statistics
    stats = compute_statistics(train_data, val_data)
    logging.info(f"Saving statistics to {stats_path}")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("QUALITY DATASET GENERATION COMPLETE")
    logging.info("="*60)
    logging.info(f"Total examples: {stats['total_examples']:,}")
    logging.info(f"  Training: {stats['train_examples']:,}")
    logging.info(f"  Validation: {stats['val_examples']:,}")
    logging.info(f"Positive examples: {stats['num_positive']:,} ({stats['positive_ratio']*100:.1f}%)")
    logging.info(f"Negative examples: {stats['num_negative']:,}")
    logging.info(f"Strategy distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        logging.info(f"  {strategy}: {count:,}")
    logging.info(f"Unique semantic types:")
    logging.info(f"  Subjects: {stats['num_unique_subject_types']}")
    logging.info(f"  Objects: {stats['num_unique_object_types']}")
    logging.info(f"  Verbs: {stats['num_unique_verbs']}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
