#!/usr/bin/env python3
"""
Build a pickled index of Esperanto relations from ConceptNet CSV.

This creates a fast-loading index file for instant queries without
re-parsing the full 475 MB CSV each time.
"""

import argparse
import gzip
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_word(uri: str) -> str:
    """Extract word from ConceptNet URI.

    Args:
        uri: ConceptNet URI like /c/eo/hundo/n

    Returns:
        Word like 'hundo'
    """
    parts = uri.split('/')
    if len(parts) >= 4 and parts[2] == 'eo':
        return parts[3]
    return ''


def build_index(csv_path: Path, output_path: Path):
    """Build index from ConceptNet CSV and save as pickle.

    Args:
        csv_path: Path to conceptnet-assertions-5.7.0.csv.gz
        output_path: Path to save index pickle file
    """
    logger.info(f"Building index from {csv_path}")

    index: Dict[str, List[Dict]] = defaultdict(list)
    count = 0
    esperanto_count = 0

    with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 1000000 == 0:
                logger.info(f"Processed {count:,} lines, found {esperanto_count:,} Esperanto relations")

            # Parse TSV line
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            uri, relation, start, end, metadata_json = parts[:5]

            # Only keep Esperanto relations
            if '/c/eo/' not in start and '/c/eo/' not in end:
                continue

            # Parse metadata
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {}

            # Extract word from URI
            esperanto_word = None
            other_concept = None

            if '/c/eo/' in start:
                esperanto_word = extract_word(start)
                other_concept = end
            elif '/c/eo/' in end:
                esperanto_word = extract_word(end)
                other_concept = start

            if not esperanto_word:
                continue

            # Store relation
            index[esperanto_word].append({
                'relation': relation,
                'start': start,
                'end': end,
                'other_concept': other_concept,
                'weight': metadata.get('weight', 1.0),
                'sources': metadata.get('sources', [])
            })

            esperanto_count += 1

    logger.info(f"Loaded {esperanto_count:,} Esperanto relations for {len(index):,} unique words")

    # Save index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving index to {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(dict(index), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata_path = output_path.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'total_relations': esperanto_count,
            'unique_words': len(index),
            'source_csv': str(csv_path),
            'index_file': str(output_path)
        }, f, indent=2)

    logger.info(f"✓ Index saved: {output_path}")
    logger.info(f"✓ Metadata saved: {metadata_path}")

    # Print sample stats
    logger.info(f"\nSample word counts:")
    sample_words = ['hundo', 'pomo', 'tablo', 'manĝi', 'bela']
    for word in sample_words:
        count = len(index.get(word, []))
        logger.info(f"  {word}: {count} relations")


def main():
    parser = argparse.ArgumentParser(
        description='Build fast-loading index from ConceptNet CSV'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz'),
        help='Path to ConceptNet CSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/vocabularies/external/.cache/conceptnet/index.pkl'),
        help='Output path for index pickle'
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        logger.error(f"ConceptNet CSV not found: {args.csv_path}")
        logger.error("Download with: wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        return

    build_index(args.csv_path, args.output)


if __name__ == '__main__':
    main()
