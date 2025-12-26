#!/usr/bin/env python3
"""
Merge authoritative and general corpora into unified training corpus.

This script combines:
1. Authoritative corpus (tiers 1-3) from texts/authoritative/
2. General corpus (tiers 5-7) from corpus_with_sources

Output: A single JSONL file with all sources properly weighted and annotated.

Usage:
    python scripts/merge_corpora.py \
        --authoritative data/corpus/authoritative_corpus.jsonl \
        --general data/corpus/tiered_corpus.jsonl \
        --output data/corpus/unified_corpus.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Merge corpora into unified training set')
    parser.add_argument('--authoritative', type=Path, required=True,
                        help='Authoritative corpus (tiers 1-3)')
    parser.add_argument('--general', type=Path,
                        help='General corpus (tiers 5-7), optional')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output unified corpus')
    parser.add_argument('--deduplicate', action='store_true',
                        help='Remove duplicate sentences')
    parser.add_argument('--min-parse-rate', type=float, default=0.5,
                        help='Minimum parse rate to include (default: 0.5)')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Merging Corpora")
    logger.info("=" * 60)

    # Statistics
    tier_counts = defaultdict(int)
    source_counts = defaultdict(int)
    total = 0
    duplicates = 0
    low_quality = 0

    # Track seen sentences for deduplication
    seen_texts = set() if args.deduplicate else None

    with open(args.output, 'w', encoding='utf-8') as outfile:

        # 1. Process authoritative corpus (highest priority)
        if args.authoritative.exists():
            logger.info(f"Processing authoritative corpus: {args.authoritative}")
            with open(args.authoritative, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        text = entry.get('text', '')

                        # Deduplication
                        if seen_texts is not None:
                            if text in seen_texts:
                                duplicates += 1
                                continue
                            seen_texts.add(text)

                        # Quality check
                        stats = entry.get('parse_statistics', {})
                        parse_rate = stats.get('success_rate', 1.0)
                        if parse_rate < args.min_parse_rate:
                            low_quality += 1
                            continue

                        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

                        tier = entry.get('source', {}).get('tier', 0)
                        tier_counts[tier] += 1
                        source_counts[entry.get('source', {}).get('name', 'unknown')] += 1
                        total += 1

                    except json.JSONDecodeError:
                        pass

            logger.info(f"  Added {sum(tier_counts[t] for t in [1,2,3]):,} authoritative entries")

        # 2. Process general corpus
        if args.general and args.general.exists():
            logger.info(f"Processing general corpus: {args.general}")
            with open(args.general, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        text = entry.get('text', '')

                        # Deduplication
                        if seen_texts is not None:
                            if text in seen_texts:
                                duplicates += 1
                                continue
                            seen_texts.add(text)

                        # Quality check
                        stats = entry.get('parse_statistics', {})
                        parse_rate = stats.get('parse_rate', stats.get('success_rate', 1.0))
                        if parse_rate < args.min_parse_rate:
                            low_quality += 1
                            continue

                        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

                        tier = entry.get('source', {}).get('tier', 7)
                        tier_counts[tier] += 1
                        source_counts[entry.get('source', {}).get('name', 'unknown')] += 1
                        total += 1

                        if line_num % 500000 == 0:
                            logger.info(f"  Processed {line_num:,} general entries...")

                    except json.JSONDecodeError:
                        pass

            logger.info(f"  Added {sum(tier_counts[t] for t in [5,6,7]):,} general entries")

    # Report
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total entries: {total:,}")
    if args.deduplicate:
        logger.info(f"Duplicates removed: {duplicates:,}")
    logger.info(f"Low quality filtered: {low_quality:,}")
    logger.info("")

    logger.info("By Tier:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = count / total * 100 if total > 0 else 0
        weight = {1: 10.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.5, 6: 1.0, 7: 0.5}.get(tier, 1.0)
        logger.info(f"  Tier {tier}: {count:,} ({pct:.1f}%) [weight={weight}]")

    # Calculate effective weights
    logger.info("")
    logger.info("Effective Training Weights:")
    total_weighted = sum(
        tier_counts[t] * {1: 10.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.5, 6: 1.0, 7: 0.5}.get(t, 1.0)
        for t in tier_counts
    )
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        weight = {1: 10.0, 2: 5.0, 3: 3.0, 4: 2.0, 5: 1.5, 6: 1.0, 7: 0.5}.get(tier, 1.0)
        weighted = count * weight
        eff_pct = weighted / total_weighted * 100 if total_weighted > 0 else 0
        logger.info(f"  Tier {tier}: {eff_pct:.1f}% of training signal")

    logger.info("")
    logger.info("Top Sources:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        pct = count / total * 100
        logger.info(f"  {source}: {count:,} ({pct:.1f}%)")

    # Write metadata
    metadata_path = args.output.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'created': datetime.now().isoformat(),
            'total_entries': total,
            'duplicates_removed': duplicates,
            'low_quality_filtered': low_quality,
            'min_parse_rate': args.min_parse_rate,
            'tier_counts': dict(tier_counts),
            'source_counts': dict(source_counts),
            'sources': {
                'authoritative': str(args.authoritative),
                'general': str(args.general) if args.general else None
            }
        }, f, indent=2)

    logger.info(f"\nMetadata written to: {metadata_path}")
    logger.info(f"Output written to: {args.output}")


if __name__ == '__main__':
    main()
