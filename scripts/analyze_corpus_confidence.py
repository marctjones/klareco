#!/usr/bin/env python3
"""
Analyze confidence distribution in enriched corpus.
Shows what confidence levels we have for training data.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_corpus(corpus_path: Path, max_lines: int = 100000):
    """Analyze confidence distribution and tier3 coverage."""

    confidence_buckets = {
        'none': 0,      # tier3_type is None
        'low': 0,       # < 0.5
        'medium': 0,    # 0.5-0.9
        'high': 0,      # >= 0.9
    }

    tier3_by_confidence = defaultdict(lambda: {'none': 0, 'low': 0, 'medium': 0, 'high': 0})
    total = 0

    print(f"Analyzing {corpus_path}")
    print(f"(Limited to first {max_lines:,} lines for speed)")
    print()

    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i > max_lines:
                break

            if not line.strip():
                continue

            data = json.loads(line)
            total += 1

            # Extract features
            semantic = data.get('semantic_annotation', {})
            det = semantic.get('deterministic_features', {})

            tier3 = det.get('tier3_type')
            conf = det.get('confidence', 0)

            # Categorize
            if tier3 is None:
                bucket = 'none'
            elif conf < 0.5:
                bucket = 'low'
            elif conf < 0.9:
                bucket = 'medium'
            else:
                bucket = 'high'

            confidence_buckets[bucket] += 1

            if tier3:
                tier3_by_confidence[tier3][bucket] += 1

            if i % 10000 == 0:
                print(f"  Processed {i:,} lines...")

    print()
    print("="*70)
    print("CONFIDENCE DISTRIBUTION")
    print("="*70)
    print()

    for bucket, count in confidence_buckets.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {bucket:10s}: {count:7,} ({pct:5.1f}%)")

    print()
    print(f"  TOTAL: {total:,} examples")
    print()

    # Show tier3 types by confidence
    print("="*70)
    print("TIER3 TYPES BY CONFIDENCE")
    print("="*70)
    print()
    print(f"{'Tier3 Type':<30} {'None':>8} {'Low':>8} {'Med':>8} {'High':>8} {'Total':>8}")
    print("-"*70)

    tier3_totals = {}
    for tier3, buckets in tier3_by_confidence.items():
        total_tier3 = sum(buckets.values())
        tier3_totals[tier3] = total_tier3

    # Sort by total count
    for tier3 in sorted(tier3_totals.keys(), key=lambda x: tier3_totals[x], reverse=True):
        buckets = tier3_by_confidence[tier3]
        total_tier3 = tier3_totals[tier3]
        print(f"{tier3:<30} {buckets['none']:>8,} {buckets['low']:>8,} {buckets['medium']:>8,} {buckets['high']:>8,} {total_tier3:>8,}")

    print()

    # Show what we're missing (no examples at all)
    print("="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print()

    labeled_count = total - confidence_buckets['none']
    print(f"✓ Labeled examples (tier3 != None): {labeled_count:,} ({labeled_count/total*100:.1f}%)")
    print(f"✗ Unlabeled examples: {confidence_buckets['none']:,} ({confidence_buckets['none']/total*100:.1f}%)")
    print()

    high_conf = confidence_buckets['high']
    medium_conf = confidence_buckets['medium']
    low_conf = confidence_buckets['low']

    print(f"High confidence (≥0.9): {high_conf:,} ({high_conf/total*100:.1f}%)")
    print(f"  → Model should TRUST deterministic")
    print()
    print(f"Medium confidence (0.5-0.9): {medium_conf:,} ({medium_conf/total*100:.1f}%)")
    print(f"  → Model should use CONTEXT to refine")
    print()
    print(f"Low confidence (<0.5): {low_conf:,} ({low_conf/total*100:.1f}%)")
    print(f"  → Model should FILL SEMANTIC GAP")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze confidence distribution')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/training/entity_classifier/enriched_corpus.jsonl'),
        help='Path to enriched corpus'
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=100000,
        help='Max lines to analyze (for speed)'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: Corpus not found: {args.corpus}")
        sys.exit(1)

    analyze_corpus(args.corpus, args.max_lines)
