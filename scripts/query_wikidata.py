#!/usr/bin/env python3
"""
Query Wikidata SPARQL endpoint for Esperanto semantic types.

Feasibility study for importing semantic categories from Wikidata.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from klareco.external.function_words import filter_function_words, get_function_word_count

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False
    logging.warning("SPARQLWrapper not installed. Install with: pip install SPARQLWrapper")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def query_wikidata_instance_of(esperanto_label: str) -> List[Dict]:
    """Query Wikidata for instance-of (P31) and subclass-of (P279) relations.

    Args:
        esperanto_label: Esperanto word to search for

    Returns:
        List of Wikidata items with their instance-of relations
    """
    if not SPARQL_AVAILABLE:
        logger.error("SPARQLWrapper not available. Cannot query Wikidata.")
        return []

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(SPARQL_JSON)

    # Query for items with Esperanto label and their instance-of relations
    query = f"""
    SELECT ?item ?itemLabel ?instanceOf ?instanceOfLabel WHERE {{
      ?item rdfs:label "{esperanto_label}"@eo.
      OPTIONAL {{ ?item wdt:P31 ?instanceOf. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "eo,en". }}
    }}
    LIMIT 10
    """

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
        bindings = results.get('results', {}).get('bindings', [])

        items = []
        for binding in bindings:
            item_data = {
                'item_id': binding['item']['value'].split('/')[-1] if 'item' in binding else None,
                'item_label': binding.get('itemLabel', {}).get('value', ''),
                'instance_of_id': binding.get('instanceOf', {}).get('value', '').split('/')[-1] if 'instanceOf' in binding else None,
                'instance_of_label': binding.get('instanceOfLabel', {}).get('value', '')
            }
            items.append(item_data)

        return items

    except Exception as e:
        logger.error(f"SPARQL query failed for '{esperanto_label}': {e}")
        return []


def query_sample_nouns(
    uncategorized_nouns_path: Path,
    sample_size: int,
    cache_dir: Path,
    output_path: Path
) -> Dict:
    """Query Wikidata for sample of uncategorized nouns.

    Args:
        uncategorized_nouns_path: Path to uncategorized_nouns.json
        sample_size: Number of nouns to sample
        cache_dir: Directory to cache SPARQL responses
        output_path: Path to save feasibility report

    Returns:
        Feasibility report with coverage stats
    """
    if not SPARQL_AVAILABLE:
        logger.error("Cannot run feasibility study without SPARQLWrapper")
        return {}

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

    # Query Wikidata for each
    results = []
    found_count = 0
    instance_count = 0

    for i, noun_data in enumerate(sample, 1):
        root = noun_data['root']

        # Check cache
        cache_file = cache_dir / f"{root}.json"
        if cache_file.exists():
            logger.info(f"[{i}/{len(sample)}] Loading from cache: {root}")
            with open(cache_file) as f:
                items = json.load(f)
        else:
            logger.info(f"[{i}/{len(sample)}] Querying Wikidata: {root}")
            items = query_wikidata_instance_of(root)

            # Cache response
            with open(cache_file, 'w') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)

            # Rate limiting (be respectful to Wikidata SPARQL endpoint)
            time.sleep(1.0)

        # Count items with instance-of relations
        items_with_instance = [item for item in items if item.get('instance_of_id')]

        if items:
            found_count += 1
        if items_with_instance:
            instance_count += len(items_with_instance)

        results.append({
            'root': root,
            'frequency': noun_data.get('frequency', 0),
            'found_in_wikidata': len(items) > 0,
            'num_items': len(items),
            'num_with_instance_of': len(items_with_instance),
            'items': items
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
        'instance_relations': {
            'total': instance_count,
            'avg_per_word': instance_count / len(sample)
        },
        'sample_results': results
    }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nFeasibility Report:")
    logger.info(f"  Coverage: {found_count}/{len(sample)} ({report['coverage']['percentage']:.1f}%)")
    logger.info(f"  Avg instance-of relations per word: {report['instance_relations']['avg_per_word']:.2f}")
    logger.info(f"  Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Query Wikidata for Esperanto semantic types (feasibility study)'
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
        default=Path('data/vocabularies/external/.cache/wikidata'),
        help='Directory to cache Wikidata SPARQL responses'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/external/wikidata_feasibility_report.json'),
        help='Output path for feasibility report'
    )

    args = parser.parse_args()

    if not SPARQL_AVAILABLE:
        logger.error("SPARQLWrapper not installed. Install with: pip install SPARQLWrapper")
        return

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
