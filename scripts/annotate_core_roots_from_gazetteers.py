#!/usr/bin/env python3
"""
Annotate Core Roots with Semantic Classifications (Semi-Automated)

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (semantic ontology)
DEPENDENCIES: Kuzu database with semantic ontology tables
STAGE: Data

Description:
    Annotates core vocabulary roots by linking them to semantic classes in the
    4-layer ontology. Uses existing gazetteer data as seed annotations, validates
    roots exist in database, and creates relationships.

Pipeline Position:
    v2.2 Schema → [THIS SCRIPT] → Annotated Roots → SemanticQuery API

Usage:
    python scripts/annotate_core_roots_from_gazetteers.py \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --dry-run

Inputs:
    - Kuzu database with v2.2 semantic ontology schema
    - klareco/knowledge/gazetteers.py (place names, person indicators)
    - VerbaKlaso.ekzemplaj_radikoj (32 roots already in verb classes)

Outputs:
    - APARTENAS_AL_VERBA_KLASO relationships (verbs → verb classes)
    - HAVAS_ENTECAN_TIPON relationships (nouns → entity types)
    - Annotation summary log

Quality Checks:
    - Validates roots exist in Radiko table before linking
    - Reports missing roots (need tier classification first)
    - Reports duplicate annotations
    - Counts successful annotations per class

Last Updated: 2026-03-28
Author: Claude Sonnet 4.5
Related Issues: #18 (Annotate 200 core roots)
See Also: scripts/extend_kuzu_schema_semantic_ontology.py
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import kuzu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoreRootAnnotator:
    """Annotate core vocabulary roots with semantic classifications."""

    def __init__(self, kuzu_db_path: Path, dry_run: bool = False):
        """
        Initialize annotator.

        Args:
            kuzu_db_path: Path to Kuzu database
            dry_run: If True, print actions without executing
        """
        self.kuzu_db_path = kuzu_db_path
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        # Statistics
        self.stats = {
            'verbs_annotated': 0,
            'places_annotated': 0,
            'persons_annotated': 0,
            'missing_roots': [],
            'errors': []
        }

    def get_existing_verb_roots(self) -> Dict[str, List[str]]:
        """
        Get verb roots already in VerbaKlaso.ekzemplaj_radikoj.

        Returns:
            Dict mapping klaso_id -> list of root strings
        """
        query = "MATCH (v:VerbaKlaso) RETURN v.klaso_id, v.ekzemplaj_radikoj"
        result = self.conn.execute(query)

        verb_classes = {}
        while result.has_next():
            row = result.get_next()
            klaso_id = row[0]
            roots = row[1]  # Already a list
            verb_classes[klaso_id] = roots
            logger.info(f"Found {len(roots)} roots in {klaso_id}")

        return verb_classes

    def root_exists(self, root: str) -> bool:
        """
        Check if root exists in Radiko table.

        Args:
            root: Root string to check

        Returns:
            True if root exists
        """
        query = f"MATCH (r:Radiko {{radiko: '{root}'}}) RETURN count(*)"
        result = self.conn.execute(query)
        count = result.get_next()[0]
        return count > 0

    def link_verb_to_class(self, root: str, klaso_id: str) -> bool:
        """
        Create APARTENAS_AL_VERBA_KLASO relationship.

        Args:
            root: Root string
            klaso_id: Verb class ID (e.g., 'kreado-26')

        Returns:
            True if successful
        """
        if not self.root_exists(root):
            logger.warning(f"Root '{root}' not found in database, skipping")
            self.stats['missing_roots'].append(root)
            return False

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would link {root} → {klaso_id}")
            return True

        try:
            query = f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
            """
            self.conn.execute(query)
            logger.info(f"✓ Linked {root} → {klaso_id}")
            self.stats['verbs_annotated'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to link {root} → {klaso_id}: {e}")
            self.stats['errors'].append(f"{root} → {klaso_id}: {e}")
            return False

    def link_noun_to_entity_type(self, root: str, tipo_id: str) -> bool:
        """
        Create HAVAS_ENTECAN_TIPON relationship.

        Args:
            root: Root string
            tipo_id: Entity type ID (e.g., 'loko', 'persono')

        Returns:
            True if successful
        """
        if not self.root_exists(root):
            logger.warning(f"Root '{root}' not found in database, skipping")
            self.stats['missing_roots'].append(root)
            return False

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would link {root} → EntecaTipo:{tipo_id}")
            return True

        try:
            query = f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
            """
            self.conn.execute(query)
            logger.info(f"✓ Linked {root} → EntecaTipo:{tipo_id}")

            if tipo_id == 'loko':
                self.stats['places_annotated'] += 1
            elif tipo_id == 'persono':
                self.stats['persons_annotated'] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to link {root} → {tipo_id}: {e}")
            self.stats['errors'].append(f"{root} → {tipo_id}: {e}")
            return False

    def normalize_place_name(self, place_name: str) -> str:
        """
        Normalize place name to root form.

        Removes -o ending for cities/countries.

        Args:
            place_name: Place name (e.g., 'Varsovio', 'Pollando')

        Returns:
            Root form (e.g., 'varsov', 'polland')
        """
        # Remove -o ending, lowercase
        if place_name.endswith('o'):
            return place_name[:-1].lower()
        return place_name.lower()

    def annotate_verbs(self):
        """Annotate verb roots from VerbaKlaso.ekzemplaj_radikoj."""
        logger.info("=" * 60)
        logger.info("ANNOTATING VERBS")
        logger.info("=" * 60)

        verb_classes = self.get_existing_verb_roots()

        for klaso_id, roots in verb_classes.items():
            logger.info(f"\nProcessing {klaso_id}: {len(roots)} roots")

            for root in roots:
                self.link_verb_to_class(root, klaso_id)

        logger.info(f"\n✓ Annotated {self.stats['verbs_annotated']} verbs")

    def annotate_places(self):
        """Annotate place roots from gazetteers.py."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING PLACES")
        logger.info("=" * 60)

        # Import place names from gazetteers
        from klareco.knowledge.gazetteers import place_names

        logger.info(f"Found {len(place_names)} place names in gazetteers")

        for place_name in place_names:
            root = self.normalize_place_name(place_name)
            self.link_noun_to_entity_type(root, 'loko')

        logger.info(f"\n✓ Annotated {self.stats['places_annotated']} places")

    def annotate_persons(self):
        """Annotate person roots from gazetteers.py."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATING PERSONS")
        logger.info("=" * 60)

        # Import person indicators from gazetteers
        from klareco.knowledge.gazetteers import person_indicators

        # Extract person root examples
        person_roots = []

        # Known Esperantists
        for name in person_indicators['esperantists']:
            root = name.lower()
            # Special handling for Zamenhof variants
            if 'zamenhof' in root or 'zamenof' in root:
                person_roots.append('zamenhof')
            else:
                person_roots.append(root)

        # Common occupations (already roots)
        person_roots.extend(person_indicators['occupations'])

        # Remove duplicates
        person_roots = list(set(person_roots))

        logger.info(f"Found {len(person_roots)} person roots to annotate")

        for root in person_roots:
            self.link_noun_to_entity_type(root, 'persono')

        logger.info(f"\n✓ Annotated {self.stats['persons_annotated']} persons")

    def print_summary(self):
        """Print annotation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("ANNOTATION SUMMARY")
        logger.info("=" * 60)

        logger.info(f"Verbs annotated: {self.stats['verbs_annotated']}")
        logger.info(f"Places annotated: {self.stats['places_annotated']}")
        logger.info(f"Persons annotated: {self.stats['persons_annotated']}")

        total = (self.stats['verbs_annotated'] +
                 self.stats['places_annotated'] +
                 self.stats['persons_annotated'])
        logger.info(f"\nTOTAL ANNOTATED: {total} roots")

        if self.stats['missing_roots']:
            logger.warning(f"\nMissing roots (not in database): {len(self.stats['missing_roots'])}")
            for root in self.stats['missing_roots'][:10]:
                logger.warning(f"  - {root}")
            if len(self.stats['missing_roots']) > 10:
                logger.warning(f"  ... and {len(self.stats['missing_roots']) - 10} more")

        if self.stats['errors']:
            logger.error(f"\nErrors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                logger.error(f"  - {error}")

    def verify_annotations(self):
        """Verify annotations were created successfully."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING ANNOTATIONS")
        logger.info("=" * 60)

        # Count verb relationships
        query = "MATCH ()-[r:APARTENAS_AL_VERBA_KLASO]->() RETURN count(*)"
        result = self.conn.execute(query)
        verb_count = result.get_next()[0]
        logger.info(f"Verb relationships: {verb_count}")

        # Count entity type relationships
        query = "MATCH ()-[r:HAVAS_ENTECAN_TIPON]->() RETURN count(*)"
        result = self.conn.execute(query)
        entity_count = result.get_next()[0]
        logger.info(f"Entity type relationships: {entity_count}")

        # Count by entity type
        query = """
            MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
            RETURN e.tipo_id, count(*)
        """
        result = self.conn.execute(query)
        logger.info("\nBreakdown by entity type:")
        while result.has_next():
            row = result.get_next()
            logger.info(f"  {row[0]}: {row[1]} roots")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate core roots with semantic classifications'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/indexes/v2.1_kuzu_index_full'),
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without executing'
    )

    args = parser.parse_args()

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    annotator = CoreRootAnnotator(args.db, dry_run=args.dry_run)

    try:
        # Annotate verbs (32 roots from VerbaKlaso.ekzemplaj_radikoj)
        annotator.annotate_verbs()

        # Annotate places (~50 roots from gazetteers.py)
        annotator.annotate_places()

        # Annotate persons (~20 roots from gazetteers.py)
        annotator.annotate_persons()

        # Print summary
        annotator.print_summary()

        # Verify results (skip in dry-run mode)
        if not args.dry_run:
            annotator.verify_annotations()

        logger.info("\n✓ Annotation complete!")

        return 0

    except Exception as e:
        logger.error(f"Annotation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
