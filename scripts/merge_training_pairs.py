#!/usr/bin/env python3
"""
Merge AST pairs with co-occurrence pairs for enhanced training.

Combines:
- AST-aware pairs (modifier-head, subject-object, antonyms)
- Co-occurrence pairs (distributional semantics)

Usage:
    python scripts/merge_training_pairs.py \
        --ast-pairs data/training/fundamento_ast_pairs/pairs.jsonl \
        --cooccurrence-pairs data/training/fundamento_cooccurrence/pairs.jsonl \
        --output data/training/fundamento_enhanced/pairs.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge AST and co-occurrence pairs")
    parser.add_argument('--ast-pairs', type=Path, required=True,
                        help='Path to AST pairs.jsonl')
    parser.add_argument('--cooccurrence-pairs', type=Path, required=True,
                        help='Path to co-occurrence pairs.jsonl')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for merged pairs.jsonl')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Merging Training Pairs")
    logger.info("=" * 80)

    # Load AST pairs
    logger.info(f"\nLoading AST pairs from {args.ast_pairs}...")
    ast_pairs = []
    with open(args.ast_pairs) as f:
        for line in f:
            pair = json.loads(line)
            ast_pairs.append(pair)

    logger.info(f"Loaded {len(ast_pairs):,} AST pairs")

    # Load co-occurrence pairs
    logger.info(f"\nLoading co-occurrence pairs from {args.cooccurrence_pairs}...")
    cooccurrence_pairs = []
    with open(args.cooccurrence_pairs) as f:
        for line in f:
            pair = json.loads(line)
            cooccurrence_pairs.append(pair)

    logger.info(f"Loaded {len(cooccurrence_pairs):,} co-occurrence pairs")

    # Merge pairs
    all_pairs = ast_pairs + cooccurrence_pairs

    logger.info(f"\nTotal pairs: {len(all_pairs):,}")

    # Get unique roots
    roots = set()
    for pair in all_pairs:
        roots.add(pair['target'])
        roots.add(pair['context'])

    logger.info(f"Unique roots: {len(roots):,}")

    # Count by type
    ast_count = len(ast_pairs)
    cooccurrence_count = len(cooccurrence_pairs)

    logger.info(f"\nComposition:")
    logger.info(f"  AST pairs: {ast_count:,} ({ast_count/len(all_pairs)*100:.1f}%)")
    logger.info(f"  Co-occurrence: {cooccurrence_count:,} ({cooccurrence_count/len(all_pairs)*100:.1f}%)")

    # Save merged pairs
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving merged pairs to {args.output}...")
    with open(args.output, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    # Save vocabulary
    vocab_output = args.output.parent / 'vocab.json'
    logger.info(f"Saving vocabulary to {vocab_output}...")
    with open(vocab_output, 'w') as f:
        json.dump(sorted(roots), f, indent=2)

    # Save statistics
    stats_output = args.output.parent / 'stats.json'
    logger.info(f"Saving statistics to {stats_output}...")

    stats = {
        'total_pairs': len(all_pairs),
        'ast_pairs': ast_count,
        'cooccurrence_pairs': cooccurrence_count,
        'unique_roots': len(roots),
        'ast_percentage': ast_count / len(all_pairs),
        'cooccurrence_percentage': cooccurrence_count / len(all_pairs)
    }

    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Merge Complete")
    logger.info("=" * 80)
    logger.info(f"Output: {args.output}")
    logger.info(f"Total pairs: {len(all_pairs):,}")
    logger.info(f"Unique roots: {len(roots):,}")


if __name__ == '__main__':
    main()
