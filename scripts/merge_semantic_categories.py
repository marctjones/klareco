#!/usr/bin/env python3
"""
Merge semantic categories from manual curation and external sources.

Priority: manual > ConceptNet > Wikidata
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_categories(
    manual_path: Path,
    conceptnet_path: Path,
    wikidata_path: Path,
    output_path: Path
) -> Dict:
    """Merge categories from multiple sources with priority.

    Args:
        manual_path: Path to manual semantic categories
        conceptnet_path: Path to ConceptNet imports
        wikidata_path: Path to Wikidata imports
        output_path: Path to save merged categories

    Returns:
        Dict with merged categories and metadata
    """
    # Load all sources
    logger.info("Loading category sources...")

    with open(manual_path) as f:
        manual = json.load(f)
    logger.info(f"  Manual: {sum(len(words) for words in manual.values())} words across {len(manual)} categories")

    conceptnet = {}
    if conceptnet_path.exists():
        with open(conceptnet_path) as f:
            conceptnet = json.load(f)
        logger.info(f"  ConceptNet: {sum(len(words) for words in conceptnet.values())} words across {len(conceptnet)} categories")
    else:
        logger.warning(f"  ConceptNet file not found: {conceptnet_path}")

    wikidata = {}
    if wikidata_path.exists():
        with open(wikidata_path) as f:
            wikidata = json.load(f)
        logger.info(f"  Wikidata: {sum(len(words) for words in wikidata.values())} words across {len(wikidata)} categories")
    else:
        logger.warning(f"  Wikidata file not found: {wikidata_path}")

    # Build word -> (category, source) mapping with priority
    word_to_cat = {}
    source_counts = {'manual': 0, 'conceptnet': 0, 'wikidata': 0}

    # Priority 1: Manual (highest confidence)
    for category, words in manual.items():
        for word in words:
            word_to_cat[word] = {'category': category, 'source': 'manual'}
            source_counts['manual'] += 1

    # Priority 2: ConceptNet
    for category, words in conceptnet.items():
        for word in words:
            if word not in word_to_cat:
                word_to_cat[word] = {'category': category, 'source': 'conceptnet'}
                source_counts['conceptnet'] += 1

    # Priority 3: Wikidata
    for category, words in wikidata.items():
        for word in words:
            if word not in word_to_cat:
                word_to_cat[word] = {'category': category, 'source': 'wikidata'}
                source_counts['wikidata'] += 1

    # Reorganize by category
    merged = defaultdict(list)
    for word, info in word_to_cat.items():
        merged[info['category']].append(word)

    # Sort words within each category
    for category in merged:
        merged[category] = sorted(merged[category])

    # Calculate coverage statistics
    # Note: This assumes ~1765 total nouns (from Issue #490)
    TOTAL_NOUNS_ESTIMATE = 1765
    total_categorized = len(word_to_cat)
    coverage_pct = total_categorized / TOTAL_NOUNS_ESTIMATE * 100

    # Generate metadata
    metadata = {
        'total_words': total_categorized,
        'total_categories': len(merged),
        'by_source': source_counts,
        'by_category': {cat: len(words) for cat, words in merged.items()},
        'coverage': {
            'categorized': total_categorized,
            'estimated_total_nouns': TOTAL_NOUNS_ESTIMATE,
            'percentage': coverage_pct
        },
        'target_coverage': 0.80,  # 80% goal from Issue #490
        'meets_target': coverage_pct >= 80.0
    }

    # Save merged categories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dict(merged), f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("MERGED SEMANTIC CATEGORIES")
    logger.info("="*70)
    logger.info(f"\nTotal words categorized: {total_categorized}")
    logger.info(f"Coverage: {total_categorized}/{TOTAL_NOUNS_ESTIMATE} ({coverage_pct:.1f}%)")
    logger.info(f"Target: 80.0%")
    logger.info(f"Meets target: {'✓ YES' if metadata['meets_target'] else '✗ NO'}")

    logger.info(f"\nBy source:")
    logger.info(f"  Manual: {source_counts['manual']} ({source_counts['manual']/total_categorized*100:.1f}%)")
    logger.info(f"  ConceptNet: {source_counts['conceptnet']} ({source_counts['conceptnet']/total_categorized*100:.1f}%)")
    logger.info(f"  Wikidata: {source_counts['wikidata']} ({source_counts['wikidata']/total_categorized*100:.1f}%)")

    logger.info(f"\nBy category:")
    for category in sorted(merged.keys()):
        count = len(merged[category])
        logger.info(f"  {category:15s}: {count:4d} words")

    logger.info(f"\nOutput saved to: {output_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("="*70)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Merge semantic categories from manual and external sources'
    )
    parser.add_argument(
        '--manual',
        type=Path,
        default=Path('data/vocabularies/semantic_categories_expanded.json'),
        help='Path to manual semantic categories'
    )
    parser.add_argument(
        '--conceptnet',
        type=Path,
        default=Path('data/vocabularies/external/conceptnet_categories.json'),
        help='Path to ConceptNet imports'
    )
    parser.add_argument(
        '--wikidata',
        type=Path,
        default=Path('data/vocabularies/external/wikidata_categories.json'),
        help='Path to Wikidata imports'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/semantic_categories_merged.json'),
        help='Output path for merged categories'
    )

    args = parser.parse_args()

    # Check manual categories exist
    if not args.manual.exists():
        logger.error(f"Manual categories file not found: {args.manual}")
        logger.error("This file is required as the base for merging")
        return

    # Merge categories
    metadata = merge_categories(
        manual_path=args.manual,
        conceptnet_path=args.conceptnet,
        wikidata_path=args.wikidata,
        output_path=args.output
    )

    logger.info("\n✓ Merge complete!")

    # Exit code based on coverage target
    if not metadata['meets_target']:
        logger.warning(f"\n⚠ Coverage ({metadata['coverage']['percentage']:.1f}%) below 80% target")
        logger.warning("Consider additional manual categorization or expanding external mappings")


if __name__ == '__main__':
    main()
