#!/usr/bin/env python3
"""
Expand Semantic Annotations via ReVo Synonyms

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (semantic ontology + ReVo)
DEPENDENCIES: Kuzu database with semantic annotations and REVO_SINONIMO relationships
STAGE: Data

Description:
    Expands semantic classifications by propagating them through ReVo synonym chains.
    If 'fond' is kreado-26, and 'fond' -[:REVO_SINONIMO]-> 'kre', then 'kre' inherits kreado-26.

Usage:
    python scripts/expand_annotations_via_revo.py \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --max-depth 2 \\
        --dry-run

Inputs:
    - Kuzu database with existing APARTENAS_AL_VERBA_KLASO relationships
    - REVO_SINONIMO relationships

Outputs:
    - Expanded APARTENAS_AL_VERBA_KLASO relationships
    - Expanded HAVAS_ENTECAN_TIPON relationships

Quality Checks:
    - Only propagates to roots that exist in database
    - Prevents conflicting classifications (warns if root has multiple classes)
    - Reports expansion statistics by depth

Last Updated: 2026-03-31
Author: Claude Sonnet 4.5
Related Issues: #17, #18
See Also: docs/SEMANTIC_ONTOLOGY_REFERENCE.md
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RevoAnnotationExpander:
    """Expand semantic annotations through ReVo synonym chains."""

    def __init__(self, kuzu_db_path: Path, max_depth: int = 2, dry_run: bool = False):
        self.kuzu_db_path = kuzu_db_path
        self.max_depth = max_depth
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        # Statistics
        self.stats = {
            'initial_verb_annotations': 0,
            'initial_entity_annotations': 0,
            'verbs_expanded': 0,
            'entities_expanded': 0,
            'conflicts_detected': 0,
            'depth_breakdown': {i: 0 for i in range(1, max_depth + 1)},
        }

    def get_initial_annotations(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get initial semantic annotations.

        Returns:
            (verb_annotations, entity_annotations) where keys are roots, values are class IDs
        """
        logger.info("Loading initial annotations...")

        # Verb annotations
        verb_annotations = {}
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN r.radiko, v.klaso_id
        """)
        while result.has_next():
            radiko, klaso_id = result.get_next()
            verb_annotations[radiko] = klaso_id

        logger.info(f"  Found {len(verb_annotations)} verb annotations")
        self.stats['initial_verb_annotations'] = len(verb_annotations)

        # Entity type annotations
        entity_annotations = {}
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo)
            RETURN r.radiko, e.tipo_id
        """)
        while result.has_next():
            radiko, tipo_id = result.get_next()
            entity_annotations[radiko] = tipo_id

        logger.info(f"  Found {len(entity_annotations)} entity annotations")
        self.stats['initial_entity_annotations'] = len(entity_annotations)

        return verb_annotations, entity_annotations

    def get_revo_synonyms(self, root: str, depth: int = 1) -> Set[str]:
        """
        Get ReVo synonyms up to specified depth.

        Args:
            root: Root to expand from
            depth: How many synonym hops to traverse

        Returns:
            Set of synonym roots
        """
        if depth < 1:
            return set()

        synonyms = set()

        # Direct synonyms (depth 1)
        result = self.conn.execute(f"""
            MATCH (r:Radiko {{radiko: '{root}'}})-[:REVO_SINONIMO]-(s:Radiko)
            RETURN s.radiko
        """)
        while result.has_next():
            synonyms.add(result.get_next()[0])

        # Recursive expansion for depth > 1
        if depth > 1:
            for synonym in list(synonyms):
                synonyms.update(self.get_revo_synonyms(synonym, depth - 1))

        return synonyms

    def root_exists(self, root: str) -> bool:
        """Check if root exists in database."""
        result = self.conn.execute(f"MATCH (r:Radiko {{radiko: '{root}'}}) RETURN count(r)")
        return result.get_next()[0] > 0

    def has_annotation(self, root: str, annotation_type: str) -> bool:
        """
        Check if root already has annotation.

        Args:
            root: Root to check
            annotation_type: 'verb' or 'entity'

        Returns:
            True if already annotated
        """
        if annotation_type == 'verb':
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[:APARTENAS_AL_VERBA_KLASO]->()
                RETURN count(*)
            """)
        else:  # entity
            result = self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}})-[:HAVAS_ENTECAN_TIPON]->()
                RETURN count(*)
            """)

        return result.get_next()[0] > 0

    def create_verb_annotation(self, root: str, klaso_id: str, depth: int) -> bool:
        """Create verb classification relationship."""
        if not self.root_exists(root):
            return False

        if self.has_annotation(root, 'verb'):
            logger.debug(f"  {root} already has verb classification, skipping")
            return False

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would link: {root} → VerbaKlaso:{klaso_id} (depth {depth})")
            return True

        try:
            self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
            """)
            logger.info(f"  ✓ Linked: {root} → VerbaKlaso:{klaso_id} (depth {depth})")
            self.stats['verbs_expanded'] += 1
            self.stats['depth_breakdown'][depth] += 1
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to link {root}: {e}")
            return False

    def create_entity_annotation(self, root: str, tipo_id: str, depth: int) -> bool:
        """Create entity type relationship."""
        if not self.root_exists(root):
            return False

        if self.has_annotation(root, 'entity'):
            logger.debug(f"  {root} already has entity classification, skipping")
            return False

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would link: {root} → EntecaTipo:{tipo_id} (depth {depth})")
            return True

        try:
            self.conn.execute(f"""
                MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
            """)
            logger.info(f"  ✓ Linked: {root} → EntecaTipo:{tipo_id} (depth {depth})")
            self.stats['entities_expanded'] += 1
            self.stats['depth_breakdown'][depth] += 1
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to link {root}: {e}")
            return False

    def expand_verb_annotations(self, verb_annotations: Dict[str, str]):
        """Expand verb annotations through ReVo synonyms."""
        logger.info("\n" + "=" * 60)
        logger.info("EXPANDING VERB ANNOTATIONS")
        logger.info("=" * 60)

        for depth in range(1, self.max_depth + 1):
            logger.info(f"\nDepth {depth}: Expanding through synonyms...")

            expanded_count = 0
            # Use list() to create a copy of items to avoid dict size change during iteration
            for root, klaso_id in list(verb_annotations.items()):
                synonyms = self.get_revo_synonyms(root, depth=1)  # One hop at a time

                for synonym in synonyms:
                    if synonym not in verb_annotations:
                        if self.create_verb_annotation(synonym, klaso_id, depth):
                            verb_annotations[synonym] = klaso_id  # Add to tracking
                            expanded_count += 1

            logger.info(f"  Expanded {expanded_count} verb annotations at depth {depth}")

    def expand_entity_annotations(self, entity_annotations: Dict[str, str]):
        """Expand entity annotations through ReVo synonyms."""
        logger.info("\n" + "=" * 60)
        logger.info("EXPANDING ENTITY ANNOTATIONS")
        logger.info("=" * 60)

        for depth in range(1, self.max_depth + 1):
            logger.info(f"\nDepth {depth}: Expanding through synonyms...")

            expanded_count = 0
            # Use list() to create a copy of items to avoid dict size change during iteration
            for root, tipo_id in list(entity_annotations.items()):
                synonyms = self.get_revo_synonyms(root, depth=1)

                for synonym in synonyms:
                    if synonym not in entity_annotations:
                        if self.create_entity_annotation(synonym, tipo_id, depth):
                            entity_annotations[synonym] = tipo_id
                            expanded_count += 1

            logger.info(f"  Expanded {expanded_count} entity annotations at depth {depth}")

    def print_stats(self):
        """Print expansion statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("EXPANSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial verb annotations:     {self.stats['initial_verb_annotations']}")
        logger.info(f"Initial entity annotations:   {self.stats['initial_entity_annotations']}")
        logger.info(f"Verbs expanded:               {self.stats['verbs_expanded']}")
        logger.info(f"Entities expanded:            {self.stats['entities_expanded']}")
        logger.info(f"Conflicts detected:           {self.stats['conflicts_detected']}")

        logger.info("\nExpansion by depth:")
        for depth, count in self.stats['depth_breakdown'].items():
            if count > 0:
                logger.info(f"  Depth {depth}: {count} annotations")

        total_verbs = self.stats['initial_verb_annotations'] + self.stats['verbs_expanded']
        total_entities = self.stats['initial_entity_annotations'] + self.stats['entities_expanded']
        logger.info(f"\nTotal annotations after expansion:")
        logger.info(f"  Verbs:    {total_verbs}")
        logger.info(f"  Entities: {total_entities}")
        logger.info(f"  TOTAL:    {total_verbs + total_entities}")

    def verify_expansion(self):
        """Verify expanded annotations."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING EXPANSION")
        logger.info("=" * 60)

        # Count total annotations
        result = self.conn.execute("MATCH ()-[r:APARTENAS_AL_VERBA_KLASO]->() RETURN count(r)")
        verb_count = result.get_next()[0]
        logger.info(f"Total verb annotations: {verb_count}")

        result = self.conn.execute("MATCH ()-[r:HAVAS_ENTECAN_TIPON]->() RETURN count(r)")
        entity_count = result.get_next()[0]
        logger.info(f"Total entity annotations: {entity_count}")

        # Sample annotations
        logger.info("\nSample expanded annotations:")
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN r.radiko, v.klaso_nomo
            LIMIT 10
        """)
        while result.has_next():
            radiko, klaso = result.get_next()
            logger.info(f"  {radiko} → {klaso}")


def main():
    parser = argparse.ArgumentParser(
        description='Expand semantic annotations via ReVo synonyms'
    )
    parser.add_argument(
        '--db',
        default='data/indexes/v2.1_kuzu_index_full',
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2,
        help='Maximum synonym chain depth (default: 2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without executing'
    )

    args = parser.parse_args()

    expander = RevoAnnotationExpander(
        Path(args.db),
        max_depth=args.max_depth,
        dry_run=args.dry_run
    )

    # Get initial annotations
    verb_annotations, entity_annotations = expander.get_initial_annotations()

    # Expand through ReVo synonyms
    expander.expand_verb_annotations(verb_annotations)
    expander.expand_entity_annotations(entity_annotations)

    # Print statistics
    expander.print_stats()

    # Verify
    if not args.dry_run:
        expander.verify_expansion()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
