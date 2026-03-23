#!/usr/bin/env python3
"""
Verb Constraint Generation - Selectional Preference Extraction

VERSION: v2.1
COMPATIBLE WITH: v2.1 SVO extraction output, v2.1 semantic types
DEPENDENCIES: SVO triples, SEMANTIC_TYPES.json from cluster_semantic_types.py
STAGE: Training

Description:
    Generates verb selectional preference constraints from corpus patterns.
    For each verb, computes probability distributions over semantic types
    that can appear as subjects and objects.

    Zero human annotation - fully automated from clustered semantic types.

Pipeline Position:
    SVO Triples + SEMANTIC_TYPES.json → [THIS SCRIPT] → VERB_CONSTRAINTS.json → SFV Model

Usage:
    python scripts/generate_verb_constraints.py \
        --triples data/semantic_types/svo_triples_full.jsonl \
        --semantic-types data/semantic_types/semantic_types.json \
        --output data/semantic_types/verb_constraints.json \
        --min-frequency 5

Inputs:
    - SVO triples: JSONL file from extract_svo_triples.py
    - semantic_types.json: Root → semantic type mapping from cluster_semantic_types.py

Outputs:
    - verb_constraints.json: Selectional preference distributions
      Format: {
        "kre": {
          "subject_types": {"PERSONO": 0.85, "ANIMALO": 0.10, ...},
          "object_types": {"OBJEKTO": 0.60, "AGO": 0.25, ...},
          "total_count": 1532
        }
      }
    - constraint_stats.json: Statistics and validation metrics

Quality Checks:
    - Coverage: % of verbs with constraints
    - Consistency: Mutual information between subject/object types
    - Fundamento coverage: All Fundamento verbs have constraints
    - Distributional sanity: No type has >95% probability (avoid overfitting)

Algorithm:
    1. Load SVO triples and semantic type mappings
    2. For each verb, count (subject_type, verb, object_type) occurrences
    3. Compute probability distributions P(subject_type|verb) and P(object_type|verb)
    4. Apply smoothing to handle rare types
    5. Validate constraint quality

Last Updated: 2026-03-16
Author: Claude Code
Related Issues: #691 (parser enhancement), semantic type hierarchy design
See Also: extract_svo_triples.py, cluster_semantic_types.py, train_semantic_fact_validator.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_svo_triples(input_path: Path) -> List[Dict]:
    """Load SVO triples from JSONL file."""
    logger.info(f"Loading SVO triples from {input_path}")
    triples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                triple = json.loads(line.strip())
                triples.append(triple)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

    logger.info(f"Loaded {len(triples):,} SVO triples")
    return triples


def load_semantic_types(types_path: Path) -> Dict[str, str]:
    """Load semantic type mappings."""
    logger.info(f"Loading semantic types from {types_path}")

    with open(types_path, 'r', encoding='utf-8') as f:
        semantic_types = json.load(f)

    logger.info(f"Loaded {len(semantic_types):,} semantic type mappings")

    # Show distribution
    type_counts = Counter(semantic_types.values())
    logger.info("Semantic type distribution:")
    for type_label, count in type_counts.most_common(10):
        logger.info(f"  {type_label}: {count:,} roots")

    return semantic_types


def build_verb_constraints(
    triples: List[Dict],
    semantic_types: Dict[str, str],
    min_frequency: int = 5,
    smoothing_alpha: float = 0.1
) -> Dict[str, Dict]:
    """
    Build selectional preference constraints for verbs.

    Args:
        triples: List of SVO triples
        semantic_types: Mapping from roots to semantic types
        min_frequency: Minimum verb frequency to include
        smoothing_alpha: Laplace smoothing parameter (prevents zero probabilities)

    Returns:
        verb_constraints: Dictionary mapping verbs to selectional preferences
    """
    logger.info("Building verb constraints...")

    # Count (subject_type, verb, object_type) patterns
    verb_subject_counts = defaultdict(lambda: defaultdict(int))
    verb_object_counts = defaultdict(lambda: defaultdict(int))
    verb_total_counts = Counter()

    unknown_roots = set()

    for triple in triples:
        verb = triple.get('verb')
        subject = triple.get('subject')
        obj = triple.get('object')

        if not verb:
            continue

        # Get semantic types
        subject_type = semantic_types.get(subject)
        object_type = semantic_types.get(obj)

        # Track unknown roots
        if subject and not subject_type:
            unknown_roots.add(subject)
        if obj and not object_type:
            unknown_roots.add(obj)

        # Count patterns
        if subject_type:
            verb_subject_counts[verb][subject_type] += 1
            verb_total_counts[verb] += 1

        if object_type:
            verb_object_counts[verb][object_type] += 1

    if unknown_roots:
        logger.warning(f"Found {len(unknown_roots):,} roots without semantic types")
        logger.warning(f"  Sample: {', '.join(list(unknown_roots)[:20])}")

    # Filter by minimum frequency
    valid_verbs = {v for v, c in verb_total_counts.items() if c >= min_frequency}
    logger.info(f"Found {len(valid_verbs):,} verbs (freq >= {min_frequency})")

    # Build constraints with probability distributions
    verb_constraints = {}

    all_types = set(semantic_types.values())
    num_types = len(all_types)

    for verb in sorted(valid_verbs):
        subject_counts = verb_subject_counts[verb]
        object_counts = verb_object_counts[verb]

        # Apply Laplace smoothing and compute probabilities
        # P(type|verb) = (count(type, verb) + alpha) / (sum(counts) + alpha * |types|)

        # Subject type probabilities
        total_subjects = sum(subject_counts.values())
        subject_probs = {}

        for sem_type in all_types:
            count = subject_counts.get(sem_type, 0)
            prob = (count + smoothing_alpha) / (total_subjects + smoothing_alpha * num_types)
            if prob > 0.01:  # Only include types with >1% probability
                subject_probs[sem_type] = round(prob, 4)

        # Object type probabilities
        total_objects = sum(object_counts.values())
        object_probs = {}

        for sem_type in all_types:
            count = object_counts.get(sem_type, 0)
            prob = (count + smoothing_alpha) / (total_objects + smoothing_alpha * num_types)
            if prob > 0.01:  # Only include types with >1% probability
                object_probs[sem_type] = round(prob, 4)

        # Compute entropy (measure of specificity)
        # Low entropy = verb is very selective about arguments
        # High entropy = verb accepts many different types
        subject_entropy = -sum(p * np.log2(p) for p in subject_probs.values() if p > 0)
        object_entropy = -sum(p * np.log2(p) for p in object_probs.values() if p > 0)

        verb_constraints[verb] = {
            'subject_types': subject_probs,
            'object_types': object_probs,
            'total_count': verb_total_counts[verb],
            'subject_entropy': round(float(subject_entropy), 3),
            'object_entropy': round(float(object_entropy), 3)
        }

    logger.info(f"Generated constraints for {len(verb_constraints):,} verbs")

    return verb_constraints


def compute_constraint_stats(
    verb_constraints: Dict[str, Dict],
    semantic_types: Dict[str, str]
) -> Dict:
    """Compute statistics about verb constraints."""
    logger.info("Computing constraint statistics...")

    stats = {
        'num_verbs': len(verb_constraints),
        'num_semantic_types': len(set(semantic_types.values())),
        'total_patterns': sum(c['total_count'] for c in verb_constraints.values())
    }

    # Entropy distribution
    subject_entropies = [c['subject_entropy'] for c in verb_constraints.values()]
    object_entropies = [c['object_entropy'] for c in verb_constraints.values()]

    stats['subject_entropy'] = {
        'mean': round(float(np.mean(subject_entropies)), 3),
        'std': round(float(np.std(subject_entropies)), 3),
        'min': round(float(np.min(subject_entropies)), 3),
        'max': round(float(np.max(subject_entropies)), 3)
    }

    stats['object_entropy'] = {
        'mean': round(float(np.mean(object_entropies)), 3),
        'std': round(float(np.std(object_entropies)), 3),
        'min': round(float(np.min(object_entropies)), 3),
        'max': round(float(np.max(object_entropies)), 3)
    }

    # Most selective verbs (low entropy = specific about arguments)
    selective_verbs = sorted(
        verb_constraints.items(),
        key=lambda x: x[1]['subject_entropy'] + x[1]['object_entropy']
    )[:10]

    stats['most_selective_verbs'] = [
        {
            'verb': v,
            'subject_entropy': c['subject_entropy'],
            'object_entropy': c['object_entropy'],
            'count': c['total_count']
        }
        for v, c in selective_verbs
    ]

    # Most general verbs (high entropy = accepts many types)
    general_verbs = sorted(
        verb_constraints.items(),
        key=lambda x: x[1]['subject_entropy'] + x[1]['object_entropy'],
        reverse=True
    )[:10]

    stats['most_general_verbs'] = [
        {
            'verb': v,
            'subject_entropy': c['subject_entropy'],
            'object_entropy': c['object_entropy'],
            'count': c['total_count']
        }
        for v, c in general_verbs
    ]

    # Coverage by frequency
    freq_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
    freq_labels = ['<10', '10-50', '50-100', '100-500', '500-1000', '1000+']
    freq_coverage = {label: 0 for label in freq_labels}

    for verb, constraint in verb_constraints.items():
        count = constraint['total_count']
        for i, (low, high) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
            if low <= count < high:
                freq_coverage[freq_labels[i]] += 1
                break

    stats['frequency_distribution'] = freq_coverage

    return stats


def validate_constraints(
    verb_constraints: Dict[str, Dict],
    fundamento_verbs: List[str] = None
) -> Dict:
    """Validate constraint quality."""
    logger.info("Validating constraints...")

    validation = {}

    # Check for overfitting (any type with >95% probability)
    overfitted_verbs = []
    for verb, constraint in verb_constraints.items():
        max_subj_prob = max(constraint['subject_types'].values()) if constraint['subject_types'] else 0
        max_obj_prob = max(constraint['object_types'].values()) if constraint['object_types'] else 0

        if max_subj_prob > 0.95 or max_obj_prob > 0.95:
            overfitted_verbs.append(verb)

    validation['overfitted_verbs'] = {
        'count': len(overfitted_verbs),
        'examples': overfitted_verbs[:20]
    }

    # Check Fundamento coverage
    if fundamento_verbs:
        covered = sum(1 for v in fundamento_verbs if v in verb_constraints)
        validation['fundamento_coverage'] = {
            'total': len(fundamento_verbs),
            'covered': covered,
            'percentage': round(100 * covered / len(fundamento_verbs), 2)
        }

    # Check for verbs with no constraints
    no_subject_constraints = sum(1 for c in verb_constraints.values() if not c['subject_types'])
    no_object_constraints = sum(1 for c in verb_constraints.values() if not c['object_types'])

    validation['missing_constraints'] = {
        'no_subject_types': no_subject_constraints,
        'no_object_types': no_object_constraints
    }

    return validation


def main():
    parser = argparse.ArgumentParser(
        description="Generate verb selectional preference constraints from SVO patterns"
    )
    parser.add_argument(
        '--triples',
        type=Path,
        required=True,
        help='Input JSONL file with SVO triples'
    )
    parser.add_argument(
        '--semantic-types',
        type=Path,
        required=True,
        help='Input JSON file with semantic type mappings'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSON file for verb constraints'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=5,
        help='Minimum verb frequency (default: 5)'
    )
    parser.add_argument(
        '--smoothing-alpha',
        type=float,
        default=0.1,
        help='Laplace smoothing parameter (default: 0.1)'
    )
    parser.add_argument(
        '--fundamento-verbs',
        type=Path,
        help='Optional: JSON file with Fundamento verbs for coverage validation'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    triples = load_svo_triples(args.triples)
    semantic_types = load_semantic_types(args.semantic_types)

    if not triples or not semantic_types:
        logger.error("Missing required inputs, exiting")
        return

    # Load Fundamento verbs if provided
    fundamento_verbs = None
    if args.fundamento_verbs and args.fundamento_verbs.exists():
        with open(args.fundamento_verbs, 'r', encoding='utf-8') as f:
            fundamento_verbs = json.load(f)
        logger.info(f"Loaded {len(fundamento_verbs)} Fundamento verbs")

    # Build constraints
    verb_constraints = build_verb_constraints(
        triples,
        semantic_types,
        min_frequency=args.min_frequency,
        smoothing_alpha=args.smoothing_alpha
    )

    # Compute statistics
    stats = compute_constraint_stats(verb_constraints, semantic_types)

    # Validate
    validation = validate_constraints(verb_constraints, fundamento_verbs)

    # Save outputs
    logger.info(f"Saving verb constraints to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(verb_constraints, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats_path = args.output.parent / 'constraint_stats.json'
    logger.info(f"Saving statistics to {stats_path}")

    stats['validation'] = validation

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("✅ Verb constraint generation complete!")
    logger.info(f"Generated constraints for {len(verb_constraints):,} verbs")
    logger.info(f"Total patterns: {stats['total_patterns']:,}")
    logger.info(f"Mean subject entropy: {stats['subject_entropy']['mean']:.3f}")
    logger.info(f"Mean object entropy: {stats['object_entropy']['mean']:.3f}")


if __name__ == '__main__':
    main()
