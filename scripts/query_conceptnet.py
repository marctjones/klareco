#!/usr/bin/env python3
"""
Query ConceptNet API for Esperanto semantic relations.

Feasibility study for importing semantic categories from ConceptNet.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.external.function_words import filter_function_words, get_function_word_count

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def query_conceptnet(word: str, lang: str = 'eo') -> Optional[Dict]:
    """Query ConceptNet for Esperanto word.

    Args:
        word: Esperanto word (root form)
        lang: Language code (default: 'eo' for Esperanto)

    Returns:
        ConceptNet API response data, or None if request fails
    """
    url = f"https://api.conceptnet.io/c/{lang}/{word}"
    try:
        response = requests.get(
            url,
            headers={'Accept': 'application/json'},
            timeout=10
        )
        if response.ok:
            return response.json()
        else:
            logger.warning(f"Failed to query '{word}': {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.error(f"Request error for '{word}': {e}")
        return None


def get_semantic_relations(data: Dict) -> List[Dict]:
    """Extract IsA, InstanceOf, UsedFor, CapableOf relations from ConceptNet data.

    Args:
        data: ConceptNet API response

    Returns:
        List of semantic relations with type and target
    """
    if not data:
        return []

    relations = []
    for edge in data.get('edges', []):
        rel_label = edge['rel']['label']

        # Focus on taxonomic and functional relations
        if rel_label in ['IsA', 'InstanceOf', 'UsedFor', 'CapableOf', 'AtLocation', 'PartOf']:
            # Determine which end is the target (not our query word)
            start_label = edge['start']['label']
            end_label = edge['end']['label']

            # Skip if relation doesn't involve Esperanto
            if '/eo/' not in edge['start']['@id'] and '/eo/' not in edge['end']['@id']:
                continue

            relations.append({
                'relation': rel_label,
                'start': start_label,
                'end': end_label,
                'weight': edge.get('weight', 1.0)
            })

    return relations


def query_sample_nouns(
    uncategorized_nouns_path: Path,
    sample_size: int,
    cache_dir: Path,
    output_path: Path
) -> Dict:
    """Query ConceptNet for sample of uncategorized nouns.

    Args:
        uncategorized_nouns_path: Path to uncategorized_nouns.json
        sample_size: Number of nouns to sample
        cache_dir: Directory to cache API responses
        output_path: Path to save feasibility report

    Returns:
        Feasibility report with coverage stats
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load uncategorized nouns
    with open(uncategorized_nouns_path) as f:
        data = json.load(f)
        uncategorized = data.get('nouns', data)  # Handle both formats

    # Filter to content nouns only (frequency > 0)
    content_nouns = [
        item for item in uncategorized
        if isinstance(item, dict) and item.get('frequency', 0) > 0
    ]

    # Count and filter function words
    function_word_count = get_function_word_count(content_nouns)
    content_nouns = filter_function_words(content_nouns)

    logger.info(f"Total uncategorized nouns: {len(uncategorized)}")
    logger.info(f"Function words filtered: {function_word_count}")
    logger.info(f"Content nouns remaining: {len(content_nouns)}")

    # Sample top nouns by frequency
    sample = sorted(content_nouns, key=lambda x: x.get('frequency', 0), reverse=True)[:sample_size]

    logger.info(f"Sampling {len(sample)} nouns for feasibility study")

    # Query ConceptNet for each
    results = []
    found_count = 0
    relation_count = 0

    for i, noun_data in enumerate(sample, 1):
        root = noun_data['root']

        # Check cache
        cache_file = cache_dir / f"{root}.json"
        if cache_file.exists():
            logger.info(f"[{i}/{len(sample)}] Loading from cache: {root}")
            with open(cache_file) as f:
                data = json.load(f)
        else:
            logger.info(f"[{i}/{len(sample)}] Querying ConceptNet: {root}")
            data = query_conceptnet(root)

            # Cache response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Rate limiting (be respectful to ConceptNet API)
            time.sleep(0.5)

        # Extract relations
        relations = get_semantic_relations(data) if data else []

        if relations:
            found_count += 1
            relation_count += len(relations)

        results.append({
            'root': root,
            'frequency': noun_data.get('frequency', 0),
            'found_in_conceptnet': data is not None,
            'num_relations': len(relations),
            'relations': relations
        })

    # Generate report
    report = {
        'sample_size': len(sample),
        'total_content_nouns': len(content_nouns),
        'function_words_filtered': function_word_count,
        'coverage': {
            'found': found_count,
            'not_found': len(sample) - found_count,
            'percentage': found_count / len(sample) * 100
        },
        'relations': {
            'total': relation_count,
            'avg_per_word': relation_count / len(sample)
        },
        'sample_results': results
    }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nFeasibility Report:")
    logger.info(f"  Coverage: {found_count}/{len(sample)} ({report['coverage']['percentage']:.1f}%)")
    logger.info(f"  Avg relations per word: {report['relations']['avg_per_word']:.2f}")
    logger.info(f"  Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Query ConceptNet for Esperanto semantic relations (feasibility study)'
    )
    parser.add_argument(
        '--uncategorized-nouns',
        type=Path,
        default=Path('data/vocabularies/uncategorized_nouns.json'),
        help='Path to uncategorized nouns file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of nouns to sample for feasibility test'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('data/vocabularies/external/.cache/conceptnet'),
        help='Directory to cache ConceptNet API responses'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/external/conceptnet_feasibility_report.json'),
        help='Output path for feasibility report'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run feasibility study
    report = query_sample_nouns(
        uncategorized_nouns_path=args.uncategorized_nouns,
        sample_size=args.sample_size,
        cache_dir=args.cache_dir,
        output_path=args.output
    )

    logger.info("\n✓ Feasibility study complete!")


if __name__ == '__main__':
    main()
