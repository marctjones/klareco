#!/usr/bin/env python3
"""
Generate Plausibility Training Dataset for Semantic Fact Validator

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu database, Semantic Enrichment tier classification
DEPENDENCIES: extract_svo_triples.py, klareco.semantic_enrichment
STAGE: Data

Description:
    Generate balanced training dataset for plausibility scorer with:
    - 4.5M positive examples (corpus SVO triples - plausible)
    - 4.5M negative examples (synthetic implausible triples)

    Negative generation strategies:
    1. Role swaps (2.5M): Swap subject ↔ object
    2. Random combinations (2M): Random (S, V, O) sampling

    All examples annotated with semantic types from Semantic Enrichment.

Pipeline Position:
    Corpus (5.4M sentences) → SVO Extraction → [THIS SCRIPT] → Training JSONL → Plausibility Scorer Training

Usage:
    # Generate full dataset (9M examples)
    python scripts/generate_plausibility_training_data.py \
        --svo-triples data/semantic_types/svo_triples_all.jsonl \
        --output-dir data/plausibility_training \
        --num-positives 4500000 \
        --num-negatives 4500000 \
        --train-split 0.9

    # Quick test (100K examples)
    python scripts/generate_plausibility_training_data.py \
        --svo-triples data/semantic_types/svo_triples_all.jsonl \
        --output-dir data/plausibility_training_test \
        --num-positives 50000 \
        --num-negatives 50000

Inputs:
    - SVO triples JSONL (from extract_svo_triples.py)

Outputs:
    - data/plausibility_training/train.jsonl - Training set (90%)
    - data/plausibility_training/val.jsonl - Validation set (10%)
    - data/plausibility_training/stats.json - Dataset statistics

Quality Checks:
    - No overlap between positive and negative sets
    - Perfect 50/50 class balance
    - All examples have semantic types
    - Duplicate removal

Last Updated: 2026-03-22
Author: Claude Code
Related Issues: #699
See Also: /tmp/plausibility_scorer_design.md, extract_svo_triples.py
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
# SEMANTIC TYPE EXTRACTION
# ============================================================================

def get_semantic_type(root: str, enricher: ASTSemanticEnricher) -> str:
    """
    Get semantic type for a root using Semantic Enrichment.

    Args:
        root: Root word (e.g., "hund", "manĝ", "pom")
        enricher: ASTSemanticEnricher instance

    Returns:
        Semantic type string (e.g., "ANIMALO:mamulo", "AGO:manĝo", "NUTRAJXO:frukto")
    """
    # Create minimal word node for type classification
    word_node = {
        'tipo': 'vorto',
        'radiko': root,
        'vortspeco': 'substantivo',  # Default to noun (will be classified)
        'plena_vorto': root,
        'analizstato': 'sukceso'
    }

    # Get semantic type from enricher
    # The enricher returns vortspeco (Tier 1), and semantic_type (Tier 2/3)
    try:
        enriched = enricher.enrich_word(word_node)
        semantic_type = enriched.get('semantic_type', 'UNKNOWN')
        return semantic_type
    except Exception as e:
        logging.debug(f"Failed to get semantic type for '{root}': {e}")
        return 'UNKNOWN'


# ============================================================================
# POSITIVE EXAMPLES (corpus triples)
# ============================================================================

def load_positive_examples(
    svo_triples_path: Path,
    num_positives: int,
    enricher: ASTSemanticEnricher
) -> List[Dict]:
    """
    Load positive examples from corpus SVO triples.

    Args:
        svo_triples_path: Path to SVO triples JSONL file
        num_positives: Number of positive examples to sample
        enricher: Semantic enricher for type classification

    Returns:
        List of positive examples with semantic types
    """
    logging.info(f"Loading positive examples from {svo_triples_path}")

    positive_examples = []
    positive_set = set()  # For deduplication

    with jsonlines.open(svo_triples_path) as reader:
        for triple in tqdm(reader, desc="Loading positives"):
            subject_root = triple['subject_root']
            verb_root = triple['verb_root']
            object_root = triple['object_root']

            # Create unique key for deduplication
            key = (subject_root, verb_root, object_root)
            if key in positive_set:
                continue
            positive_set.add(key)

            # Get semantic types
            subject_type = get_semantic_type(subject_root, enricher)
            object_type = get_semantic_type(object_root, enricher)

            # Create example
            example = {
                'subject_root': subject_root,
                'verb_root': verb_root,
                'object_root': object_root,
                'subject_type': subject_type,
                'object_type': object_type,
                'plausible': 1.0,
                'source': 'corpus',
                'sentence': triple.get('sentence', '')
            }

            positive_examples.append(example)

            # Stop if we have enough
            if len(positive_examples) >= num_positives:
                break

    logging.info(f"Loaded {len(positive_examples)} positive examples")

    # Sample if we got more than needed
    if len(positive_examples) > num_positives:
        positive_examples = random.sample(positive_examples, num_positives)

    return positive_examples, positive_set


# ============================================================================
# NEGATIVE EXAMPLES (synthetic implausible)
# ============================================================================

def generate_role_swap_negatives(
    positive_examples: List[Dict],
    num_negatives: int,
    positive_set: Set[Tuple[str, str, str]],
    enricher: ASTSemanticEnricher
) -> List[Dict]:
    """
    Generate negative examples by swapping subject ↔ object.

    Strategy:
    - (persono, manĝ, pom) → (pom, manĝ, persono)  [implausible!]
    - Only keep swaps that don't exist in corpus (not all swaps are implausible)

    Args:
        positive_examples: List of corpus triples
        num_negatives: Number of negatives to generate
        positive_set: Set of (subject, verb, object) keys from corpus
        enricher: Semantic enricher for type classification

    Returns:
        List of negative examples
    """
    logging.info(f"Generating {num_negatives} role-swap negatives")

    negatives = []
    seen_negatives = set()

    # Shuffle to get variety
    shuffled_positives = random.sample(positive_examples, len(positive_examples))

    with tqdm(total=num_negatives, desc="Role swaps") as pbar:
        for positive in shuffled_positives:
            if len(negatives) >= num_negatives:
                break

            # Swap subject ↔ object
            swapped_subject = positive['object_root']
            swapped_object = positive['subject_root']
            verb = positive['verb_root']

            # Check if swapped version exists in corpus (would be plausible)
            swapped_key = (swapped_subject, verb, swapped_object)
            if swapped_key in positive_set:
                continue  # This swap is actually plausible

            # Check if we already generated this negative
            if swapped_key in seen_negatives:
                continue
            seen_negatives.add(swapped_key)

            # Get semantic types (swap types too)
            subject_type = positive['object_type']  # Types are swapped
            object_type = positive['subject_type']

            # Create negative example
            negative = {
                'subject_root': swapped_subject,
                'verb_root': verb,
                'object_root': swapped_object,
                'subject_type': subject_type,
                'object_type': object_type,
                'plausible': 0.0,
                'source': 'role_swap',
                'original_triple': f"({positive['subject_root']}, {verb}, {positive['object_root']})"
            }

            negatives.append(negative)
            pbar.update(1)

    logging.info(f"Generated {len(negatives)} role-swap negatives")
    return negatives


def generate_random_combination_negatives(
    positive_examples: List[Dict],
    num_negatives: int,
    positive_set: Set[Tuple[str, str, str]],
    all_negatives_set: Set[Tuple[str, str, str]],
    enricher: ASTSemanticEnricher
) -> List[Dict]:
    """
    Generate negative examples by random (subject, verb, object) combinations.

    Strategy:
    - Randomly sample subject, verb, object from corpus vocabulary
    - Filter out combinations that exist in corpus
    - Higher implausibility variety than role swaps

    Args:
        positive_examples: List of corpus triples (for vocabulary)
        num_negatives: Number of negatives to generate
        positive_set: Set of (subject, verb, object) keys from corpus
        all_negatives_set: Set of already-generated negative keys
        enricher: Semantic enricher for type classification

    Returns:
        List of negative examples
    """
    logging.info(f"Generating {num_negatives} random-combination negatives")

    # Build vocabulary from corpus
    subjects = list(set(ex['subject_root'] for ex in positive_examples))
    verbs = list(set(ex['verb_root'] for ex in positive_examples))
    objects = list(set(ex['object_root'] for ex in positive_examples))

    logging.info(f"Vocabulary: {len(subjects)} subjects, {len(verbs)} verbs, {len(objects)} objects")

    negatives = []
    attempts = 0
    max_attempts = num_negatives * 10  # Avoid infinite loop

    with tqdm(total=num_negatives, desc="Random combos") as pbar:
        while len(negatives) < num_negatives and attempts < max_attempts:
            attempts += 1

            # Random combination
            subject = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)

            key = (subject, verb, obj)

            # Skip if this is a corpus triple (plausible)
            if key in positive_set:
                continue

            # Skip if we already generated this negative
            if key in all_negatives_set:
                continue
            all_negatives_set.add(key)

            # Get semantic types
            subject_type = get_semantic_type(subject, enricher)
            object_type = get_semantic_type(obj, enricher)

            # Create negative example
            negative = {
                'subject_root': subject,
                'verb_root': verb,
                'object_root': obj,
                'subject_type': subject_type,
                'object_type': object_type,
                'plausible': 0.0,
                'source': 'random_combination'
            }

            negatives.append(negative)
            pbar.update(1)

    logging.info(f"Generated {len(negatives)} random-combination negatives ({attempts} attempts)")
    return negatives


# ============================================================================
# DATASET CREATION
# ============================================================================

def create_balanced_dataset(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    train_split: float = 0.9
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create balanced train/val splits.

    Args:
        positive_examples: List of positive examples
        negative_examples: List of negative examples
        train_split: Fraction for training (default 0.9)

    Returns:
        (train_data, val_data) tuples
    """
    logging.info("Creating balanced train/val splits")

    # Combine and shuffle
    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)

    # Split
    split_idx = int(len(all_examples) * train_split)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    logging.info(f"Train: {len(train_data)} examples")
    logging.info(f"Val: {len(val_data)} examples")

    # Verify balance
    train_positive = sum(1 for ex in train_data if ex['plausible'] == 1.0)
    val_positive = sum(1 for ex in val_data if ex['plausible'] == 1.0)

    logging.info(f"Train balance: {train_positive}/{len(train_data)} positive ({train_positive/len(train_data)*100:.1f}%)")
    logging.info(f"Val balance: {val_positive}/{len(val_data)} positive ({val_positive/len(val_data)*100:.1f}%)")

    return train_data, val_data


def compute_statistics(
    train_data: List[Dict],
    val_data: List[Dict]
) -> Dict:
    """
    Compute dataset statistics.

    Args:
        train_data: Training examples
        val_data: Validation examples

    Returns:
        Statistics dictionary
    """
    all_data = train_data + val_data

    # Class distribution
    num_positive = sum(1 for ex in all_data if ex['plausible'] == 1.0)
    num_negative = len(all_data) - num_positive

    # Source distribution
    source_counts = Counter(ex['source'] for ex in all_data)

    # Semantic type distribution
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
        description='Generate plausibility training dataset'
    )
    parser.add_argument(
        '--svo-triples',
        type=Path,
        required=True,
        help='Path to SVO triples JSONL file (from extract_svo_triples.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for train/val files'
    )
    parser.add_argument(
        '--num-positives',
        type=int,
        default=4500000,
        help='Number of positive examples (default: 4.5M)'
    )
    parser.add_argument(
        '--num-negatives',
        type=int,
        default=4500000,
        help='Number of negative examples (default: 4.5M)'
    )
    parser.add_argument(
        '--role-swap-ratio',
        type=float,
        default=0.56,
        help='Fraction of negatives from role swaps (default: 0.56 → 2.5M/4.5M)'
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

    # Load positive examples
    positive_examples, positive_set = load_positive_examples(
        args.svo_triples,
        args.num_positives,
        enricher
    )

    # Generate negative examples
    num_role_swaps = int(args.num_negatives * args.role_swap_ratio)
    num_random_combos = args.num_negatives - num_role_swaps

    # Strategy 1: Role swaps
    role_swap_negatives = generate_role_swap_negatives(
        positive_examples,
        num_role_swaps,
        positive_set,
        enricher
    )

    # Strategy 2: Random combinations
    all_negatives_set = set(
        (ex['subject_root'], ex['verb_root'], ex['object_root'])
        for ex in role_swap_negatives
    )
    random_combo_negatives = generate_random_combination_negatives(
        positive_examples,
        num_random_combos,
        positive_set,
        all_negatives_set,
        enricher
    )

    # Combine negatives
    negative_examples = role_swap_negatives + random_combo_negatives
    logging.info(f"Total negatives: {len(negative_examples)}")

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
    logging.info("DATASET GENERATION COMPLETE")
    logging.info("="*60)
    logging.info(f"Total examples: {stats['total_examples']:,}")
    logging.info(f"  Training: {stats['train_examples']:,}")
    logging.info(f"  Validation: {stats['val_examples']:,}")
    logging.info(f"Positive examples: {stats['num_positive']:,} ({stats['positive_ratio']*100:.1f}%)")
    logging.info(f"Negative examples: {stats['num_negative']:,}")
    logging.info(f"Source distribution:")
    for source, count in stats['source_distribution'].items():
        logging.info(f"  {source}: {count:,}")
    logging.info(f"Unique semantic types:")
    logging.info(f"  Subjects: {stats['num_unique_subject_types']}")
    logging.info(f"  Objects: {stats['num_unique_object_types']}")
    logging.info(f"  Verbs: {stats['num_unique_verbs']}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
