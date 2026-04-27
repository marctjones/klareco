#!/usr/bin/env python3
"""
Load Semantic Relationships from Annotation Files

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (semantic ontology)
DEPENDENCIES: Annotation JSONL files, Kuzu database with semantic ontology
STAGE: Data

Description:
    Loads semantic classification relationships from annotation JSONL files.
    Creates APARTENAS_AL_VERBA_KLASO and HAVAS_ENTECAN_TIPON relationships.

Usage:
    python scripts/load_semantic_relationships.py \\
        --annotations data/annotations/phase_0_*.jsonl \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --dry-run

Inputs:
    - Annotation JSONL files with semantic classifications
    - Kuzu database with semantic ontology tables

Outputs:
    - APARTENAS_AL_VERBA_KLASO relationships (verbs → verb classes)
    - HAVAS_ENTECAN_TIPON relationships (nouns → entity types)

Quality Checks:
    - Validates roots exist in Radiko table
    - Validates class IDs exist in VerbaKlaso/EntecaTipo tables
    - Reports skipped annotations

Last Updated: 2026-03-31
Author: Claude Sonnet 4.5
Related Issues: #17, #18
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticRelationshipLoader:
    """Load semantic classification relationships from annotations."""

    def __init__(self, kuzu_db_path: Path, dry_run: bool = False):
        self.kuzu_db_path = kuzu_db_path
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        # Load existing data
        self.existing_radikos = self._load_existing_radikos()
        self.existing_verb_classes = self._load_existing_verb_classes()
        self.existing_entity_types = self._load_existing_entity_types()

        # Statistics
        self.stats = {
            'verb_relationships': 0,
            'entity_relationships': 0,
            'skipped_missing_root': 0,
            'skipped_missing_class': 0,
            'skipped_no_classification': 0,
        }

    def _load_existing_radikos(self) -> Set[str]:
        """Load all existing root strings from Radiko table."""
        logger.info("Loading existing radikos...")
        result = self.conn.execute("MATCH (r:Radiko) RETURN r.radiko")
        radikos = set()
        while result.has_next():
            radikos.add(result.get_next()[0])
        logger.info(f"  Found {len(radikos):,} existing radikos")
        return radikos

    def _load_existing_verb_classes(self) -> Set[str]:
        """Load all existing verb class IDs."""
        logger.info("Loading existing verb classes...")
        result = self.conn.execute("MATCH (v:VerbaKlaso) RETURN v.klaso_id")
        classes = set()
        while result.has_next():
            classes.add(result.get_next()[0])
        logger.info(f"  Found {len(classes)} verb classes")
        return classes

    def _load_existing_entity_types(self) -> Set[str]:
        """Load all existing entity type IDs."""
        logger.info("Loading existing entity types...")
        result = self.conn.execute("MATCH (e:EntecaTipo) RETURN e.tipo_id")
        types = set()
        while result.has_next():
            types.add(result.get_next()[0])
        logger.info(f"  Found {len(types)} entity types")
        return types

    def load_annotations(self, annotation_files: List[Path]) -> List[Dict]:
        """Load annotations from JSONL files."""
        annotations = []

        for file_path in annotation_files:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"Reading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        annotation = json.loads(line)
                        if 'radiko' not in annotation:
                            logger.warning(f"  Line {line_num}: Missing 'radiko' field")
                            continue
                        annotations.append(annotation)
                    except json.JSONDecodeError as e:
                        logger.warning(f"  Line {line_num}: Invalid JSON: {e}")

        logger.info(f"Loaded {len(annotations)} total annotations")
        return annotations

    def create_verb_relationship(self, root: str, klaso_id: str) -> bool:
        """Create APARTENAS_AL_VERBA_KLASO relationship."""
        # Validate
        if root not in self.existing_radikos:
            logger.debug(f"  Root '{root}' not in database, skipping")
            self.stats['skipped_missing_root'] += 1
            return False

        if klaso_id not in self.existing_verb_classes:
            logger.warning(f"  Verb class '{klaso_id}' not found, skipping")
            self.stats['skipped_missing_class'] += 1
            return False

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would create: {root} -[:APARTENAS_AL_VERBA_KLASO]-> {klaso_id}")
            return True

        # Create relationship
        try:
            self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
            """)
            self.stats['verb_relationships'] += 1
            return True
        except Exception as e:
            logger.error(f"  Failed to create relationship for '{root}': {e}")
            return False

    def create_entity_relationship(self, root: str, tipo_id: str) -> bool:
        """Create HAVAS_ENTECAN_TIPON relationship."""
        # Validate
        if root not in self.existing_radikos:
            logger.debug(f"  Root '{root}' not in database, skipping")
            self.stats['skipped_missing_root'] += 1
            return False

        if tipo_id not in self.existing_entity_types:
            logger.warning(f"  Entity type '{tipo_id}' not found, skipping")
            self.stats['skipped_missing_class'] += 1
            return False

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would create: {root} -[:HAVAS_ENTECAN_TIPON]-> {tipo_id}")
            return True

        # Create relationship
        try:
            self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
            """)
            self.stats['entity_relationships'] += 1
            return True
        except Exception as e:
            logger.error(f"  Failed to create relationship for '{root}': {e}")
            return False

    def process_annotations(self, annotations: List[Dict]):
        """Process all annotations and create relationships."""
        logger.info(f"\nProcessing {len(annotations)} annotations...")

        for i, annotation in enumerate(annotations, 1):
            radiko = annotation.get('radiko', 'unknown')

            # Verb classification
            if 'verba_klaso' in annotation:
                klaso_id = annotation['verba_klaso']
                logger.info(f"  [{i}/{len(annotations)}] {radiko} → VerbaKlaso:{klaso_id}")
                self.create_verb_relationship(radiko, klaso_id)

            # Entity type classification
            elif 'enteca_tipo' in annotation:
                tipo_id = annotation['enteca_tipo']
                logger.info(f"  [{i}/{len(annotations)}] {radiko} → EntecaTipo:{tipo_id}")
                self.create_entity_relationship(radiko, tipo_id)

            else:
                logger.debug(f"  [{i}/{len(annotations)}] {radiko}: No semantic classification")
                self.stats['skipped_no_classification'] += 1

    def print_stats(self):
        """Print loading statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("LOADING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Verb relationships created:   {self.stats['verb_relationships']:,}")
        logger.info(f"Entity relationships created:  {self.stats['entity_relationships']:,}")
        logger.info(f"Skipped (missing root):        {self.stats['skipped_missing_root']:,}")
        logger.info(f"Skipped (missing class):       {self.stats['skipped_missing_class']:,}")
        logger.info(f"Skipped (no classification):   {self.stats['skipped_no_classification']:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Load semantic relationships from annotation files'
    )
    parser.add_argument(
        '--annotations',
        nargs='+',
        required=True,
        help='Annotation JSONL files (supports glob patterns)'
    )
    parser.add_argument(
        '--db',
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without executing'
    )

    args = parser.parse_args()

    # Expand file paths
    annotation_files = []
    for pattern in args.annotations:
        files = list(Path('.').glob(pattern))
        annotation_files.extend(files)

    if not annotation_files:
        logger.error("No annotation files found")
        return 1

    # Load and process
    loader = SemanticRelationshipLoader(Path(args.db), dry_run=args.dry_run)
    annotations = loader.load_annotations(annotation_files)

    if not annotations:
        logger.error("No annotations to process")
        return 1

    loader.process_annotations(annotations)
    loader.print_stats()

    return 0


if __name__ == '__main__':
    sys.exit(main())
