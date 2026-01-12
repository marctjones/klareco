#!/usr/bin/env python3
"""
Generate negative samples only - appends to existing positive pairs file.

Quick script to finish generating negatives with optimized batch sampling.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_existing_pairs(pairs_file: Path) -> tuple:
    """Load existing pairs from file."""
    logger.info(f"Loading existing pairs from {pairs_file}")

    positive_pair_set = set()
    root_pair_count = defaultdict(int)
    positive_count = 0
    negative_count = 0

    with open(pairs_file) as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                logger.info(f"  Loaded {i:,} pairs...")

            pair = json.loads(line)
            idx1, idx2 = pair['idx1'], pair['idx2']
            target = pair['target_similarity']

            pair_key = (min(idx1, idx2), max(idx1, idx2))
            positive_pair_set.add(pair_key)

            # Track per-root counts
            root1, root2 = pair['root1'], pair['root2']
            root_pair_count[root1] += 1
            root_pair_count[root2] += 1

            if target > 0:
                positive_count += 1
            else:
                negative_count += 1

    logger.info(f"Loaded {positive_count:,} positive + {negative_count:,} negative pairs")
    logger.info(f"Unique pairs: {len(positive_pair_set):,}")

    return positive_pair_set, root_pair_count, positive_count, negative_count


def generate_negatives_fast(
    output_path: Path,
    vocab_path: Path,
    positive_pair_set: Set,
    root_pair_count: Dict,
    positive_count: int,
    already_generated: int,
    negative_ratio: float = 2.0,
    target_pairs_per_root: int = 300
):
    """Generate negative samples with optimized batch sampling."""

    # Load vocabulary
    with open(vocab_path) as f:
        vocab = json.load(f)

    root_to_idx = vocab['root_to_idx']
    idx_to_root = {idx: root for root, idx in root_to_idx.items()}

    # Simple uniform weighting (faster than frequency-based)
    roots_list = list(root_to_idx.keys())

    target_negatives = int(positive_count * negative_ratio)
    remaining = target_negatives - already_generated

    logger.info(f"Generating {remaining:,} additional negatives")
    logger.info(f"  Target total: {target_negatives:,}")
    logger.info(f"  Already have: {already_generated:,}")
    logger.info(f"  Sampling from {len(roots_list):,} roots")

    negative_count = 0
    batch_size = 10000

    with open(output_path, 'a') as output_file:
        while negative_count < remaining:
            # Sample batch
            batch = min(batch_size, remaining - negative_count)

            r1_batch = random.choices(roots_list, k=batch)
            r2_batch = random.choices(roots_list, k=batch)

            for r1, r2 in zip(r1_batch, r2_batch):
                if r1 == r2:
                    continue

                idx1, idx2 = root_to_idx[r1], root_to_idx[r2]
                pair_key = (min(idx1, idx2), max(idx1, idx2))

                if pair_key in positive_pair_set:
                    continue

                # Check coverage balance
                count1 = root_pair_count.get(r1, 0)
                count2 = root_pair_count.get(r2, 0)

                if count1 > target_pairs_per_root * 1.5 or count2 > target_pairs_per_root * 1.5:
                    continue

                # Write negative pair
                pair_data = {
                    'idx1': pair_key[0],
                    'idx2': pair_key[1],
                    'target_similarity': 0.0,
                    'weight': 1.0,
                    'root1': idx_to_root[pair_key[0]],
                    'root2': idx_to_root[pair_key[1]]
                }
                output_file.write(json.dumps(pair_data) + '\n')

                positive_pair_set.add(pair_key)
                root_pair_count[r1] += 1
                root_pair_count[r2] += 1
                negative_count += 1

                if negative_count >= remaining:
                    break

            if negative_count % 100000 == 0 and negative_count > 0:
                logger.info(f"  Generated {negative_count:,} negatives ({negative_count/remaining*100:.1f}%)")
                output_file.flush()

    logger.info(f"Complete! Generated {negative_count:,} additional negatives")
    logger.info(f"Total negatives: {already_generated + negative_count:,}")
    logger.info(f"Final ratio: 1:{(already_generated + negative_count)/positive_count:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-file', type=Path,
                        default=Path('data/training/topical_pairs_smart.jsonl'))
    parser.add_argument('--vocab-file', type=Path,
                        default=Path('data/vocabularies/topical_vocab.json'))
    parser.add_argument('--negative-ratio', type=float, default=2.0)
    parser.add_argument('--target-per-root', type=int, default=300)

    args = parser.parse_args()

    # Load existing
    positive_pair_set, root_pair_count, positive_count, negative_count = load_existing_pairs(args.pairs_file)

    # Generate remaining negatives
    generate_negatives_fast(
        args.pairs_file,
        args.vocab_file,
        positive_pair_set,
        root_pair_count,
        positive_count,
        negative_count,
        args.negative_ratio,
        args.target_per_root
    )

    logger.info("Done!")


if __name__ == '__main__':
    sys.exit(main())
