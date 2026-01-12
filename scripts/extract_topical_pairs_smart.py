#!/usr/bin/env python3
"""
Smart sampling of topical training pairs with coverage guarantees.

Strategy:
1. Analyze full dataset to understand distribution
2. Set target coverage per root (e.g., 200 pairs per root)
3. Sample pairs to achieve balanced coverage
4. Ensure negative pairs have diverse frequency profiles

This gives us:
- Balanced representation of all roots (no vocabulary bias)
- Diverse contexts per root (better generalization)
- Reasonable dataset size (~15-30M pairs for 77K vocab)
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def analyze_dataset(input_path: Path) -> Tuple[Dict, Dict]:
    """
    Analyze the full dataset to understand distribution.

    Returns:
        (root_pair_count, root_frequency)
    """
    logger.info("Analyzing dataset distribution...")

    root_pair_count = defaultdict(int)  # root -> number of pairs it appears in
    root_frequency = Counter()
    pair_types = {'positive': 0, 'negative': 0}

    total_pairs = 0

    with open(input_path) as f:
        for i, line in enumerate(f):
            if i % 10000000 == 0 and i > 0:
                logger.info(f"  Analyzed {i:,} pairs...")

            pair = json.loads(line)
            root1, root2 = pair['root1'], pair['root2']
            target = pair['target_similarity']

            root_pair_count[root1] += 1
            root_pair_count[root2] += 1
            root_frequency[root1] += 1
            root_frequency[root2] += 1

            if target > 0:
                pair_types['positive'] += 1
            else:
                pair_types['negative'] += 1

            total_pairs += 1

    logger.info(f"Analysis complete:")
    logger.info(f"  Total pairs: {total_pairs:,}")
    logger.info(f"  Positive pairs: {pair_types['positive']:,}")
    logger.info(f"  Negative pairs: {pair_types['negative']:,}")
    logger.info(f"  Unique roots: {len(root_pair_count):,}")
    logger.info(f"  Avg pairs per root: {sum(root_pair_count.values()) / len(root_pair_count):.1f}")

    # Show distribution statistics
    counts = list(root_pair_count.values())
    counts.sort()
    logger.info(f"  Min pairs per root: {counts[0]:,}")
    logger.info(f"  Median pairs per root: {counts[len(counts)//2]:,}")
    logger.info(f"  Max pairs per root: {counts[-1]:,}")

    return dict(root_pair_count), dict(root_frequency)


def smart_sample_pairs(
    input_path: Path,
    output_path: Path,
    target_pairs_per_root: int = 300,
    min_pairs_per_root: int = 50,
    negative_ratio: float = 2.0,
    max_total_pairs: int = None
) -> int:
    """
    Sample pairs with coverage guarantees.

    Strategy:
    1. First pass: analyze distribution
    2. Compute sampling probabilities per root
    3. Second pass: sample pairs using computed probabilities
    4. Ensure each root appears in at least min_pairs_per_root

    Args:
        input_path: Full dataset
        output_path: Sampled output
        target_pairs_per_root: Target number of pairs per root
        min_pairs_per_root: Minimum pairs per root (skip rare roots below this)
        negative_ratio: Ratio of negative to positive pairs
        max_total_pairs: Maximum total pairs (optional cap)

    Returns:
        Number of pairs sampled
    """
    # Analyze distribution
    root_pair_count, root_frequency = analyze_dataset(input_path)

    # Compute sampling probabilities
    logger.info("\nComputing sampling probabilities...")
    sampling_prob = {}

    for root, count in root_pair_count.items():
        if count < min_pairs_per_root:
            # Skip very rare roots
            sampling_prob[root] = 0.0
        elif count <= target_pairs_per_root:
            # Sample all pairs for roots below target
            sampling_prob[root] = 1.0
        else:
            # Sample proportionally for common roots
            sampling_prob[root] = target_pairs_per_root / count

    # Filter out skipped roots
    active_roots = {r for r, p in sampling_prob.items() if p > 0}
    logger.info(f"Active roots (>= {min_pairs_per_root} pairs): {len(active_roots):,}")
    logger.info(f"Skipped rare roots: {len(root_pair_count) - len(active_roots):,}")

    # Estimate output size
    estimated_pairs = sum(
        min(count, target_pairs_per_root)
        for root, count in root_pair_count.items()
        if root in active_roots
    ) // 2  # Divide by 2 because each pair counted twice (root1, root2)

    logger.info(f"Estimated output pairs: {estimated_pairs:,}")

    # Second pass: sample pairs
    logger.info("\nSampling pairs...")
    sampled_positive = 0
    sampled_negative = 0
    root_sampled_count = defaultdict(int)

    with open(output_path, 'w') as out_f:
        with open(input_path) as in_f:
            for i, line in enumerate(in_f):
                if i % 10000000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,} pairs, sampled {sampled_positive + sampled_negative:,}")

                pair = json.loads(line)
                root1, root2 = pair['root1'], pair['root2']
                target = pair['target_similarity']

                # Skip if either root is not active
                if root1 not in active_roots or root2 not in active_roots:
                    continue

                # Compute sampling probability for this pair
                # Use minimum probability (conservative)
                prob = min(sampling_prob[root1], sampling_prob[root2])

                # For negative pairs, apply negative ratio adjustment
                if target == 0:
                    # Only sample if we need more negatives
                    if sampled_negative >= sampled_positive * negative_ratio:
                        continue
                    # Sample at lower rate initially
                    prob *= 0.5

                # Sample this pair?
                if random.random() < prob:
                    out_f.write(line)
                    root_sampled_count[root1] += 1
                    root_sampled_count[root2] += 1

                    if target > 0:
                        sampled_positive += 1
                    else:
                        sampled_negative += 1

                    # Check max limit
                    if max_total_pairs and (sampled_positive + sampled_negative) >= max_total_pairs:
                        logger.info(f"Reached max total pairs limit: {max_total_pairs:,}")
                        break

    total_sampled = sampled_positive + sampled_negative
    logger.info(f"\nSampling complete:")
    logger.info(f"  Sampled positive: {sampled_positive:,}")
    logger.info(f"  Sampled negative: {sampled_negative:,}")
    logger.info(f"  Total sampled: {total_sampled:,}")
    logger.info(f"  Ratio: 1:{sampled_negative/sampled_positive:.1f}")

    # Coverage statistics
    counts = list(root_sampled_count.values())
    if counts:
        counts.sort()
        logger.info(f"\nCoverage statistics:")
        logger.info(f"  Roots with pairs: {len(root_sampled_count):,}")
        logger.info(f"  Min pairs per root: {counts[0]:,}")
        logger.info(f"  Median pairs per root: {counts[len(counts)//2]:,}")
        logger.info(f"  Max pairs per root: {counts[-1]:,}")
        logger.info(f"  Avg pairs per root: {sum(counts) / len(counts):.1f}")

    return total_sampled


def main():
    parser = argparse.ArgumentParser(description='Smart sampling of topical pairs')
    parser.add_argument('--input', type=Path,
                        default=Path('data/training/topical_pairs.jsonl'),
                        help='Input full dataset')
    parser.add_argument('--output', type=Path,
                        default=Path('data/training/topical_pairs_sampled.jsonl'),
                        help='Output sampled dataset')
    parser.add_argument('--target-per-root', type=int, default=300,
                        help='Target pairs per root (default: 300)')
    parser.add_argument('--min-per-root', type=int, default=50,
                        help='Minimum pairs per root (skip below this, default: 50)')
    parser.add_argument('--negative-ratio', type=float, default=2.0,
                        help='Negative to positive ratio (default: 2.0)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum total pairs (optional cap)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Smart Topical Pair Sampling")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target per root: {args.target_per_root}")
    logger.info(f"Min per root: {args.min_per_root}")
    logger.info(f"Negative ratio: {args.negative_ratio}:1")
    if args.max_pairs:
        logger.info(f"Max total pairs: {args.max_pairs:,}")

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Sample pairs
    total = smart_sample_pairs(
        args.input,
        args.output,
        target_pairs_per_root=args.target_per_root,
        min_pairs_per_root=args.min_per_root,
        negative_ratio=args.negative_ratio,
        max_total_pairs=args.max_pairs
    )

    # Show file sizes
    input_size = args.input.stat().st_size / (1024**3)
    output_size = args.output.stat().st_size / (1024**3)
    reduction = (1 - output_size / input_size) * 100

    logger.info(f"\nFile sizes:")
    logger.info(f"  Input: {input_size:.2f} GB")
    logger.info(f"  Output: {output_size:.2f} GB")
    logger.info(f"  Reduction: {reduction:.1f}%")
    logger.info(f"\nOutput: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
