#!/usr/bin/env python3
"""
Generate Word-Level Plausibility Training Dataset (Hybrid v2.0)

VERSION: v2.0
COMPATIBLE WITH: Word-level SVO triples (with decomposition), Hybrid Word Encoder
DEPENDENCIES: extract_svo_triples.py (word-level), hybrid_word.py, affix_semantics.py
STAGE: Data

Description:
    Generate high-quality training dataset for hybrid plausibility scorer (v2.0).

    Key differences from root-level (v1.0):
    1. Works at WORD level, not root level
    2. Uses full AST decomposition (root + affixes)
    3. Negative generation considers affix semantics
    4. More accurate semantic constraints

    Expected improvement: 66% F1 → 85-95% F1

Pipeline Position:
    Corpus → SVO Extraction (word-level) → [THIS SCRIPT] → Word-Level Training JSONL → Hybrid Plausibility Scorer

Usage:
    # Generate word-level dataset (200K examples)
    python scripts/generate_plausibility_training_data_word_level.py \
        --svo-triples data/semantic_types/svo_triples_word_level.jsonl \
        --output-dir data/plausibility_training_word_level \
        --num-examples 200000

Inputs:
    - SVO triples JSONL (word-level with decomposition)
      Format: {'subject': {'text': 'pomisto', 'root': 'pom', 'affixes': ['ist']}, ...}

Outputs:
    - data/plausibility_training_word_level/train.jsonl - Training set (90%)
    - data/plausibility_training_word_level/val.jsonl - Validation set (10%)
    - data/plausibility_training_word_level/stats.json - Dataset statistics

Quality Checks:
    - Confidence threshold: Only triples with confidence ≥ 0.9
    - Word-level semantic balance (considers affixes)
    - Challenging negatives: Affix-aware swaps
    - Function word exclusion: No function words in training

Last Updated: 2026-03-23
Author: Claude Code
Related Issues: #9
See Also: docs/HYBRID_PLAUSIBILITY_V2_PROGRESS.md
"""

import argparse
import json
import jsonlines
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging
from tqdm import tqdm

from klareco.morphology.affix_semantics import get_affix_features, compose_word_semantics
from klareco.morphology.root_lexicon import get_root_features


# ============================================================================
# WORD-LEVEL SEMANTIC FEATURES
# ============================================================================

def get_word_semantics(word_data: Dict) -> Dict[str, str]:
    """
    Get semantic features for a word using composition rules.

    Args:
        word_data: {'text': 'pomisto', 'root': 'pom', 'affixes': ['ist'], ...}

    Returns:
        {'animacy': 'animate', 'type': 'person'}
    """
    root = word_data.get('root', '')
    affixes = word_data.get('affixes', [])

    # Compose semantics (affixes override base)
    return compose_word_semantics(root, affixes)


# ============================================================================
# QUALITY FILTERS
# ============================================================================

def filter_high_quality_triples(
    triples: List[Dict],
    min_confidence: float = 0.9
) -> List[Dict]:
    """
    Filter for high-quality word-level triples.

    Criteria:
    - Confidence ≥ min_confidence
    - All components have valid word decomposition
    - Not function words (kaj, de, la, etc.)
    """
    filtered = []

    # Count verb frequencies to filter overly common
    verb_counts = Counter()
    for t in triples:
        if t.get('verb') and t['verb'].get('root'):
            verb_counts[t['verb']['root']] += 1

    for triple in triples:
        # Confidence check
        if triple.get('confidence', 1.0) < min_confidence:
            continue

        # Must have word decomposition for all components
        if not all(key in triple for key in ['subject', 'verb', 'object']):
            continue

        if not all(triple[key] and triple[key].get('root') for key in ['subject', 'verb', 'object']):
            continue

        # Skip overly common verbs
        verb_root = triple['verb']['root']
        if verb_counts[verb_root] > len(triples) * 0.01:  # >1% of corpus
            continue

        # Skip function words (shouldn't be in SVO extraction, but double-check)
        function_words = {'la', 'de', 'en', 'kaj', 'ke', 'se', 'mi', 'vi', 'li', 'ŝi', 'ni', 'ili'}
        if any(triple[role]['root'] in function_words for role in ['subject', 'verb', 'object']):
            continue

        filtered.append(triple)

    return filtered


def balance_semantic_types(
    triples: List[Dict],
    target_count: int
) -> List[Dict]:
    """
    Sample triples to ensure balanced semantic type coverage (word-level).

    Strategy:
    - Group by (subject_animacy, subject_type, object_animacy, object_type)
    - Sample proportionally from each group
    - Ensures diversity across semantic combinations
    """
    # Add semantic features if missing
    for triple in triples:
        if 'subject_semantics' not in triple:
            triple['subject_semantics'] = get_word_semantics(triple['subject'])
        if 'object_semantics' not in triple:
            triple['object_semantics'] = get_word_semantics(triple['object'])
        if 'plausible' not in triple:
            triple['plausible'] = 1.0  # Corpus triples are plausible
        if 'source' not in triple:
            triple['source'] = 'corpus'

    # Group by semantic features
    type_groups = defaultdict(list)
    for triple in triples:
        key = (
            triple['subject_semantics']['animacy'],
            triple['subject_semantics']['type'],
            triple['object_semantics']['animacy'],
            triple['object_semantics']['type']
        )
        type_groups[key].append(triple)

    # Calculate samples per group
    num_groups = len(type_groups)
    samples_per_group = max(1, target_count // num_groups)

    # Sample from each group
    balanced = []
    for group_triples in type_groups.values():
        sample_size = min(samples_per_group, len(group_triples))
        balanced.extend(random.sample(group_triples, sample_size))

    # If we need more, sample randomly from remainder
    if len(balanced) < target_count:
        remaining = [t for t in triples if t not in balanced]
        additional = random.sample(remaining, min(target_count - len(balanced), len(remaining)))
        balanced.extend(additional)

    # Shuffle and truncate
    random.shuffle(balanced)
    return balanced[:target_count]


# ============================================================================
# CHALLENGING NEGATIVE GENERATION (WORD-LEVEL)
# ============================================================================

def generate_challenging_negatives(
    positive_examples: List[Dict],
    num_negatives: int,
    positive_set: Set[Tuple[str, str, str]]
) -> List[Dict]:
    """
    Generate CHALLENGING negative examples at word level.

    Strategy:
    1. Affix-aware type swaps (40%): Swap words with same animacy/type
       Example: (pomisto:animate_person, manĝ, pomo:inanimate_food)
                → (tablisto:animate_person, manĝ, pomo:inanimate_food)
       Why hard: Both are professions (same affix), but table-seller eating apple is odd

    2. Animacy violations (30%): Break animacy constraints
       Example: (hundo:animate, manĝ, pano:inanimate) → (tablo:inanimate, manĝ, pano:inanimate)
       Why hard: inanimate can't eat

    3. Type mismatches (30%): Break type constraints
       Example: (persono:person, lern:cognition, lingvo:abstract)
                → (persono:person, lern:cognition, tablo:object)
       Why hard: Can't learn a table (abstract vs concrete)
    """
    negatives = []
    seen_negatives = set()

    # Group by semantic features
    by_subject_semantics = defaultdict(list)
    by_object_semantics = defaultdict(list)
    by_verb = defaultdict(list)

    for ex in positive_examples:
        subj_key = (ex['subject_semantics']['animacy'], ex['subject_semantics']['type'])
        obj_key = (ex['object_semantics']['animacy'], ex['object_semantics']['type'])

        by_subject_semantics[subj_key].append(ex)
        by_object_semantics[obj_key].append(ex)
        by_verb[ex['verb']['root']].append(ex)

    # Strategy 1: Affix-aware type swaps (40%)
    num_type_swaps = int(num_negatives * 0.4)
    logging.info(f"Generating {num_type_swaps} affix-aware type swap negatives...")

    with tqdm(total=num_type_swaps, desc="Affix-aware swaps") as pbar:
        attempts = 0
        max_attempts = num_type_swaps * 10

        while len([n for n in negatives if n.get('strategy') == 'type_swap']) < num_type_swaps and attempts < max_attempts:
            attempts += 1

            # Pick a corpus triple
            original = random.choice(positive_examples)

            # Find another triple with same subject semantic features
            subj_key = (original['subject_semantics']['animacy'], original['subject_semantics']['type'])
            same_semantics = by_subject_semantics.get(subj_key, [])

            if len(same_semantics) < 2:
                continue

            other = random.choice(same_semantics)
            if other == original:
                continue

            # Swap subjects (same semantics, different words)
            negative_key = (other['subject']['text'], original['verb']['text'], original['object']['text'])

            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            negatives.append({
                'subject': other['subject'].copy(),
                'verb': original['verb'].copy(),
                'object': original['object'].copy(),
                'subject_semantics': other['subject_semantics'].copy(),
                'object_semantics': original['object_semantics'].copy(),
                'plausible': 0.0,
                'source': 'affix_aware_swap',
                'strategy': 'type_swap',
                'reason': f"Swapped {original['subject']['text']} → {other['subject']['text']} (same type)"
            })
            pbar.update(1)

    # Strategy 2: Animacy violations (30%)
    num_animacy = int(num_negatives * 0.3)
    logging.info(f"Generating {num_animacy} animacy violation negatives...")

    # Find inanimate subjects and animate-requiring verbs
    inanimate_subjects = [ex for ex in positive_examples
                          if ex['subject_semantics']['animacy'] == 'inanimate']

    with tqdm(total=num_animacy, desc="Animacy violations") as pbar:
        attempts = 0
        max_attempts = num_animacy * 10

        while len([n for n in negatives if n.get('strategy') == 'animacy_violation']) < num_animacy and attempts < max_attempts:
            attempts += 1

            # Pick verb that likely requires animate agent
            verb_ex = random.choice(positive_examples)
            if verb_ex['subject_semantics']['animacy'] != 'animate':
                continue  # Only violate verbs that normally have animate subjects

            # Pick inanimate subject
            if not inanimate_subjects:
                continue
            inanimate_ex = random.choice(inanimate_subjects)

            # Pick random object
            obj_ex = random.choice(positive_examples)

            negative_key = (inanimate_ex['subject']['text'], verb_ex['verb']['text'], obj_ex['object']['text'])

            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            negatives.append({
                'subject': inanimate_ex['subject'].copy(),
                'verb': verb_ex['verb'].copy(),
                'object': obj_ex['object'].copy(),
                'subject_semantics': inanimate_ex['subject_semantics'].copy(),
                'object_semantics': obj_ex['object_semantics'].copy(),
                'plausible': 0.0,
                'source': 'animacy_violation',
                'strategy': 'animacy_violation',
                'reason': f"Inanimate {inanimate_ex['subject']['text']} performing {verb_ex['verb']['text']}"
            })
            pbar.update(1)

    # Strategy 3: Type mismatches (30%)
    num_random = num_negatives - len(negatives)
    logging.info(f"Generating {num_random} type mismatch negatives...")

    with tqdm(total=num_random, desc="Type mismatches") as pbar:
        attempts = 0
        max_attempts = num_random * 10

        while len([n for n in negatives if n.get('strategy') == 'type_mismatch']) < num_random and attempts < max_attempts:
            attempts += 1

            subj_ex = random.choice(positive_examples)
            verb_ex = random.choice(positive_examples)
            obj_ex = random.choice(positive_examples)

            negative_key = (subj_ex['subject']['text'], verb_ex['verb']['text'], obj_ex['object']['text'])

            if negative_key in positive_set or negative_key in seen_negatives:
                continue
            seen_negatives.add(negative_key)

            negatives.append({
                'subject': subj_ex['subject'].copy(),
                'verb': verb_ex['verb'].copy(),
                'object': obj_ex['object'].copy(),
                'subject_semantics': subj_ex['subject_semantics'].copy(),
                'object_semantics': obj_ex['object_semantics'].copy(),
                'plausible': 0.0,
                'source': 'type_mismatch',
                'strategy': 'type_mismatch',
                'reason': 'Random combination with type mismatch'
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
    """Compute word-level dataset statistics."""
    all_data = train_data + val_data

    num_positive = sum(1 for ex in all_data if ex['plausible'] == 1.0)
    num_negative = len(all_data) - num_positive

    source_counts = Counter(ex['source'] for ex in all_data)
    strategy_counts = Counter(ex.get('strategy', 'corpus') for ex in all_data)

    # Word-level statistics
    subject_animacy = Counter(ex['subject_semantics']['animacy'] for ex in all_data)
    object_animacy = Counter(ex['object_semantics']['animacy'] for ex in all_data)
    subject_types = Counter(ex['subject_semantics']['type'] for ex in all_data)
    object_types = Counter(ex['object_semantics']['type'] for ex in all_data)

    # Affix statistics
    subject_affixes = Counter()
    object_affixes = Counter()
    for ex in all_data:
        for affix in ex['subject'].get('affixes', []):
            subject_affixes[affix] += 1
        for affix in ex['object'].get('affixes', []):
            object_affixes[affix] += 1

    verb_counts = Counter(ex['verb']['root'] for ex in all_data)

    stats = {
        'total_examples': len(all_data),
        'train_examples': len(train_data),
        'val_examples': len(val_data),
        'num_positive': num_positive,
        'num_negative': num_negative,
        'positive_ratio': num_positive / len(all_data),
        'source_distribution': dict(source_counts),
        'strategy_distribution': dict(strategy_counts),
        'subject_animacy': dict(subject_animacy),
        'object_animacy': dict(object_animacy),
        'top_10_subject_types': dict(subject_types.most_common(10)),
        'top_10_object_types': dict(object_types.most_common(10)),
        'top_10_subject_affixes': dict(subject_affixes.most_common(10)),
        'top_10_object_affixes': dict(object_affixes.most_common(10)),
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
        description='Generate word-level plausibility training dataset (v2.0)'
    )
    parser.add_argument(
        '--svo-triples',
        type=Path,
        required=True,
        help='Path to word-level SVO triples JSONL file'
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

    # Load all triples
    logging.info(f"Loading word-level SVO triples from {args.svo_triples}")
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
    positive_examples = balance_semantic_types(quality_triples, num_positives)
    logging.info(f"Selected {len(positive_examples)} balanced positive examples")

    # Create positive set for filtering (use word text, not roots)
    positive_set = set(
        (ex['subject']['text'], ex['verb']['text'], ex['object']['text'])
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
    logging.info("WORD-LEVEL DATASET GENERATION COMPLETE")
    logging.info("="*60)
    logging.info(f"Total examples: {stats['total_examples']:,}")
    logging.info(f"  Training: {stats['train_examples']:,}")
    logging.info(f"  Validation: {stats['val_examples']:,}")
    logging.info(f"Positive examples: {stats['num_positive']:,} ({stats['positive_ratio']*100:.1f}%)")
    logging.info(f"Negative examples: {stats['num_negative']:,}")
    logging.info(f"Strategy distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        logging.info(f"  {strategy}: {count:,}")
    logging.info(f"Subject animacy:")
    for animacy, count in stats['subject_animacy'].items():
        logging.info(f"  {animacy}: {count:,}")
    logging.info(f"Top 5 subject affixes:")
    for affix, count in list(stats['top_10_subject_affixes'].items())[:5]:
        logging.info(f"  {affix}: {count:,}")
    logging.info(f"Unique semantic types:")
    logging.info(f"  Subjects: {stats['num_unique_subject_types']}")
    logging.info(f"  Objects: {stats['num_unique_object_types']}")
    logging.info(f"  Verbs: {stats['num_unique_verbs']}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
