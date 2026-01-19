#!/usr/bin/env python3
"""
Import semantic categories from ConceptNet for all uncategorized nouns.

Uses feasibility study cache if available, otherwise queries ConceptNet API.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.external.category_mapper import CategoryMapper
from klareco.external.function_words import filter_function_words, get_function_word_count
from scripts.query_conceptnet import query_conceptnet, get_semantic_relations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def import_from_conceptnet(
    uncategorized_nouns_path: Path,
    cache_dir: Path,
    output_path: Path,
    max_nouns: int = None
) -> Dict:
    """Import categories from ConceptNet for all uncategorized nouns.

    Args:
        uncategorized_nouns_path: Path to uncategorized_nouns.json
        cache_dir: Directory with cached ConceptNet responses
        output_path: Path to save categorized results
        max_nouns: Maximum number of nouns to process (None = all)

    Returns:
        Dict mapping categories to lists of words
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapper = CategoryMapper()

    # Load uncategorized nouns
    with open(uncategorized_nouns_path) as f:
        data = json.load(f)
        uncategorized = data.get('nouns', data)

    # Filter to content nouns only
    content_nouns = [
        item for item in uncategorized
        if isinstance(item, dict) and item.get('frequency', 0) > 0
    ]

    # Count and filter function words
    function_word_count = get_function_word_count(content_nouns)
    content_nouns = filter_function_words(content_nouns)

    logger.info(f"Total uncategorized nouns: {len(uncategorized)}")
    logger.info(f"Function words filtered: {function_word_count}")
    logger.info(f"Content nouns to process: {len(content_nouns)}")

    if max_nouns:
        content_nouns = content_nouns[:max_nouns]
        logger.info(f"Limited to first {max_nouns} nouns for testing")

    logger.info(f"Importing categories for {len(content_nouns)} content nouns")

    categorized = defaultdict(list)
    stats = {
        'total_processed': 0,
        'found_in_conceptnet': 0,
        'successfully_categorized': 0,
        'no_mapping': 0,
        'api_errors': 0
    }

    for i, noun_data in enumerate(content_nouns, 1):
        root = noun_data['root']
        stats['total_processed'] += 1

        # Check cache first
        cache_file = cache_dir / f"{root}.json"
        if cache_file.exists():
            logger.info(f"[{i}/{len(content_nouns)}] Loading from cache: {root}")
            with open(cache_file) as f:
                data = json.load(f)
        else:
            logger.info(f"[{i}/{len(content_nouns)}] Querying ConceptNet: {root}")
            data = query_conceptnet(root)

            if data:
                # Cache response
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                stats['api_errors'] += 1

            # Rate limiting (respectful to ConceptNet API)
            time.sleep(0.5)

        # Extract relations and map to categories
        if data:
            stats['found_in_conceptnet'] += 1
            relations = get_semantic_relations(data)

            if relations:
                category = mapper.map_conceptnet_relations(relations)

                if category:
                    categorized[category].append(root)
                    stats['successfully_categorized'] += 1
                    logger.info(f"  → {category}")
                else:
                    stats['no_mapping'] += 1
                    logger.debug(f"  → No category mapping found")

    # Save categorized results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dict(categorized), f, indent=2, ensure_ascii=False)

    # Save statistics
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("CONCEPTNET IMPORT SUMMARY")
    logger.info("="*70)
    logger.info(f"Total nouns processed: {stats['total_processed']}")
    logger.info(f"Found in ConceptNet: {stats['found_in_conceptnet']} ({stats['found_in_conceptnet']/stats['total_processed']*100:.1f}%)")
    logger.info(f"Successfully categorized: {stats['successfully_categorized']} ({stats['successfully_categorized']/stats['total_processed']*100:.1f}%)")
    logger.info(f"Found but no mapping: {stats['no_mapping']}")
    logger.info(f"API errors: {stats['api_errors']}")
    logger.info(f"\nCategories imported:")
    for category in sorted(categorized.keys()):
        logger.info(f"  {category}: {len(categorized[category])} words")
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Stats saved to: {stats_path}")
    logger.info("="*70)

    return dict(categorized)


def main():
    parser = argparse.ArgumentParser(
        description='Import semantic categories from ConceptNet'
    )
    parser.add_argument(
        '--uncategorized-nouns',
        type=Path,
        default=Path('data/vocabularies/uncategorized_nouns.json'),
        help='Path to uncategorized nouns file'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('data/vocabularies/external/.cache/conceptnet'),
        help='Directory with cached ConceptNet responses'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/external/conceptnet_categories.json'),
        help='Output path for categorized results'
    )
    parser.add_argument(
        '--max-nouns',
        type=int,
        default=None,
        help='Maximum number of nouns to process (for testing)'
    )

    args = parser.parse_args()

    # Import categories
    categorized = import_from_conceptnet(
        uncategorized_nouns_path=args.uncategorized_nouns,
        cache_dir=args.cache_dir,
        output_path=args.output,
        max_nouns=args.max_nouns
    )

    logger.info("\n✓ ConceptNet import complete!")


if __name__ == '__main__':
    main()
