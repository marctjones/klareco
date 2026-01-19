#!/usr/bin/env python3
"""
Query ConceptNet locally from downloaded CSV file.

This script extracts Esperanto semantic relations from the ConceptNet dataset
without requiring API access. It builds an in-memory index for fast querying.
"""

import argparse
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptNetLocalQuery:
    """Query ConceptNet data from local CSV file."""

    def __init__(self, csv_path: Path):
        """Initialize with path to ConceptNet CSV file.

        Args:
            csv_path: Path to conceptnet-assertions-5.7.0.csv.gz
        """
        self.csv_path = csv_path
        self.index: Dict[str, List[Dict]] = defaultdict(list)
        self._load_esperanto_relations()

    def _load_esperanto_relations(self):
        """Load Esperanto relations into memory index."""
        logger.info(f"Loading Esperanto relations from {self.csv_path}")

        count = 0
        esperanto_count = 0

        with gzip.open(self.csv_path, 'rt', encoding='utf-8') as f:
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

                # Extract word from URI (e.g., /c/eo/hundo/n -> hundo)
                esperanto_word = None
                other_concept = None

                if '/c/eo/' in start:
                    esperanto_word = self._extract_word(start)
                    other_concept = end
                elif '/c/eo/' in end:
                    esperanto_word = self._extract_word(end)
                    other_concept = start

                if not esperanto_word:
                    continue

                # Store relation
                self.index[esperanto_word].append({
                    'relation': relation,
                    'start': start,
                    'end': end,
                    'other_concept': other_concept,
                    'weight': metadata.get('weight', 1.0),
                    'sources': metadata.get('sources', [])
                })

                esperanto_count += 1

        logger.info(f"Loaded {esperanto_count:,} Esperanto relations for {len(self.index):,} unique words")

    def _extract_word(self, uri: str) -> str:
        """Extract word from ConceptNet URI.

        Args:
            uri: ConceptNet URI like /c/eo/hundo/n

        Returns:
            Word like 'hundo'
        """
        # Format: /c/eo/word/pos or /c/eo/word
        parts = uri.split('/')
        if len(parts) >= 4 and parts[2] == 'eo':
            return parts[3]
        return ''

    def query(self, word: str, relation_types: Set[str] = None) -> List[Dict]:
        """Query relations for an Esperanto word.

        Args:
            word: Esperanto word (root form)
            relation_types: Optional set of relation types to filter by
                          (e.g., {'/r/IsA', '/r/CapableOf', '/r/UsedFor'})

        Returns:
            List of relations matching the query
        """
        relations = self.index.get(word.lower(), [])

        if relation_types:
            relations = [r for r in relations if r['relation'] in relation_types]

        return relations

    def get_semantic_relations(self, word: str) -> List[Dict]:
        """Get semantic relations relevant for category mapping.

        Focus on IsA, InstanceOf, UsedFor, CapableOf relations.

        Args:
            word: Esperanto word (root form)

        Returns:
            List of semantic relations
        """
        semantic_types = {'/r/IsA', '/r/InstanceOf', '/r/UsedFor', '/r/CapableOf'}
        return self.query(word, semantic_types)


def main():
    parser = argparse.ArgumentParser(
        description='Query ConceptNet locally for Esperanto semantic relations'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz'),
        help='Path to ConceptNet CSV file'
    )
    parser.add_argument(
        '--word',
        type=str,
        help='Query single word (for testing)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test queries on sample words'
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        logger.error(f"ConceptNet CSV not found: {args.csv_path}")
        logger.error("Download with: wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        return

    # Load ConceptNet data
    cn = ConceptNetLocalQuery(args.csv_path)

    if args.word:
        # Query single word
        logger.info(f"\nQuerying: {args.word}")

        # Show ALL relations
        all_relations = cn.query(args.word)
        logger.info(f"Found {len(all_relations)} total relations:")
        for rel in all_relations[:10]:  # Show first 10
            logger.info(f"  {rel['relation']}: {rel['other_concept']} (weight: {rel['weight']})")

        # Show semantic relations specifically
        semantic_relations = cn.get_semantic_relations(args.word)
        logger.info(f"\nFound {len(semantic_relations)} semantic relations (IsA/InstanceOf/UsedFor/CapableOf):")
        for rel in semantic_relations[:10]:
            logger.info(f"  {rel['relation']}: {rel['other_concept']} (weight: {rel['weight']})")

    elif args.test:
        # Test on sample words
        test_words = ['hundo', 'pomo', 'tablo', 'manĝi', 'bela']

        logger.info("\n" + "="*70)
        logger.info("TEST QUERIES")
        logger.info("="*70)

        for word in test_words:
            relations = cn.get_semantic_relations(word)
            logger.info(f"\n{word}: {len(relations)} semantic relations")
            for rel in relations[:3]:  # Show first 3
                logger.info(f"  {rel['relation']}: {rel['other_concept']}")

        logger.info("\n✓ Test complete!")

    else:
        logger.info(f"Loaded {len(cn.index):,} Esperanto words")
        logger.info("Use --word <word> to query a specific word")
        logger.info("Use --test to run sample queries")


if __name__ == '__main__':
    main()
