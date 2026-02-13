#!/usr/bin/env python3
"""
Sample examples for manual annotation.

Samples from semantic gap data (tier3_type=None) for human annotation.
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

def sample_for_annotation(
    input_path: Path,
    output_path: Path,
    sample_size: int = 500,
    stratified: bool = True
):
    """
    Sample examples for annotation.

    Args:
        input_path: Path to training data
        output_path: Path to save sampled examples
        sample_size: Number of examples to sample
        stratified: Whether to stratify by tier2_type for balanced coverage
    """
    print("="*70)
    print("SAMPLE FOR ANNOTATION")
    print("="*70)
    print()

    # Load all examples
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples):,} examples")

    # Check tier2 distribution
    tier2_dist = Counter()
    for ex in examples:
        tier2 = ex['deterministic_priors'].get('tier2_type')
        tier2_dist[tier2] += 1

    print()
    print("Tier 2 distribution:")
    for tier2, count in tier2_dist.most_common():
        pct = count / len(examples) * 100
        tier2_str = str(tier2) if tier2 else 'None'
        print(f"  {tier2_str:20s}: {count:6,} ({pct:5.1f}%)")
    print()

    # Sample
    if stratified and len(tier2_dist) > 1:
        print(f"Stratified sampling: {sample_size} examples balanced across tier2 types")

        # Group by tier2
        by_tier2 = {}
        for ex in examples:
            tier2 = ex['deterministic_priors'].get('tier2_type')
            if tier2 not in by_tier2:
                by_tier2[tier2] = []
            by_tier2[tier2].append(ex)

        # Calculate samples per tier2
        n_types = len(by_tier2)
        per_type = sample_size // n_types

        sampled = []
        for tier2, tier2_examples in by_tier2.items():
            n_sample = min(per_type, len(tier2_examples))
            sampled.extend(random.sample(tier2_examples, n_sample))
            tier2_str = str(tier2) if tier2 else 'None'
            print(f"  {tier2_str:20s}: {n_sample} examples")

        # Fill remaining if needed
        remaining = sample_size - len(sampled)
        if remaining > 0:
            available = [ex for ex in examples if ex not in sampled]
            sampled.extend(random.sample(available, remaining))
            print(f"  (additional):        {remaining} examples")

    else:
        print(f"Random sampling: {sample_size} examples")
        sampled = random.sample(examples, min(sample_size, len(examples)))

    print()
    print(f"✓ Sampled {len(sampled):,} examples")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"✓ Saved to: {output_path}")
    print()

    # Show sample tier2 distribution
    sampled_tier2 = Counter()
    for ex in sampled:
        tier2 = ex['deterministic_priors'].get('tier2_type')
        sampled_tier2[tier2] += 1

    print("Sampled tier2 distribution:")
    for tier2, count in sampled_tier2.most_common():
        pct = count / len(sampled) * 100
        tier2_str = str(tier2) if tier2 else 'None'
        print(f"  {tier2_str:20s}: {count:6,} ({pct:5.1f}%)")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample examples for annotation')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/training/active_learning/iteration_0_train.jsonl'),
        help='Input training data'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/active_learning/iteration_0_to_annotate.jsonl'),
        help='Output path for sampled examples'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Number of examples to sample'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        default=True,
        help='Stratify by tier2_type'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    sample_for_annotation(
        args.input,
        args.output,
        args.sample_size,
        args.stratified
    )
