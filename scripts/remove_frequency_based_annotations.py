#!/usr/bin/env python3
"""
Remove Frequency-Based Semantic Annotations

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema + semantic ontology
DEPENDENCIES: Kuzu database with semantic annotations
STAGE: Data Cleanup

Description:
    Removes frequency-based annotations (added by expand_annotations_frequency_based.py)
    that polluted verb classes with morphologically similar but semantically unrelated roots.

    Keeps only high-quality annotations from:
    - Manual curation (23 roots)
    - Gazetteer-based (161 roots)
    - ReVo synonym expansion (143 roots)
    Total: 327 high-quality annotations

Strategy:
    - Top-level verb classes (8) have 200+ members each (polluted)
    - Fine-grained subclasses (31) have 2-15 members each (clean)
    - Delete all annotations to top-level classes (kreado-26, movo-51, etc.)
    - Keep all annotations to fine-grained subclasses (kreado-26.1, vido-30, etc.)

Usage:
    python scripts/remove_frequency_based_annotations.py \\
        --db data/indexes/v2.1_kuzu_index_full \\
        --dry-run

Last Updated: 2026-03-31
Author: Claude Sonnet 4.5
"""

import argparse
import logging
from pathlib import Path
import kuzu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrequencyAnnotationRemover:
    """Remove frequency-based annotations that polluted verb classes."""

    def __init__(self, kuzu_db_path: Path, dry_run: bool = False):
        self.kuzu_db_path = kuzu_db_path
        self.dry_run = dry_run

        logger.info(f"Connecting to Kuzu database: {kuzu_db_path}")
        self.db = kuzu.Database(str(kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        self.stats = {
            'initial_annotations': 0,
            'top_level_annotations': 0,
            'fine_grained_annotations': 0,
            'deleted': 0,
            'kept': 0,
        }

    def get_top_level_classes(self):
        """Get list of top-level verb classes (no parent)."""
        logger.info("Finding top-level verb classes...")

        result = self.conn.execute("""
            MATCH (v:VerbaKlaso)
            WHERE v.superklaso_id IS NULL OR v.superklaso_id = ''
            RETURN v.klaso_id, v.klaso_nomo
        """)

        top_level = []
        while result.has_next():
            klaso_id, klaso_nomo = result.get_next()
            top_level.append((klaso_id, klaso_nomo))

        logger.info(f"  Found {len(top_level)} top-level classes")
        return top_level

    def count_annotations_by_class(self):
        """Count annotations for each verb class."""
        logger.info("\nCounting annotations by class...")

        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN v.klaso_id, v.klaso_nomo, v.superklaso_id, count(r) AS member_count
            ORDER BY member_count DESC
        """)

        logger.info("  Class                     | Members | Type")
        logger.info("  " + "-" * 50)

        while result.has_next():
            klaso_id, klaso_nomo, superklaso_id, count = result.get_next()
            class_type = "TOP-LEVEL" if not superklaso_id else "Fine-grained"
            logger.info(f"  {klaso_nomo:25} | {count:7} | {class_type}")

            if not superklaso_id:
                self.stats['top_level_annotations'] += count
            else:
                self.stats['fine_grained_annotations'] += count

        self.stats['initial_annotations'] = self.stats['top_level_annotations'] + self.stats['fine_grained_annotations']

    def delete_top_level_annotations(self, top_level_classes):
        """Delete annotations to top-level classes."""
        logger.info("\nDeleting annotations to top-level classes...")

        for klaso_id, klaso_nomo in top_level_classes:
            # Count before delete
            result = self.conn.execute(f"""
                MATCH (r:Radiko)-[rel:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                RETURN count(rel)
            """)
            count = result.get_next()[0]

            if count == 0:
                continue

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would delete {count} annotations to {klaso_nomo} ({klaso_id})")
                self.stats['deleted'] += count
            else:
                try:
                    self.conn.execute(f"""
                        MATCH (r:Radiko)-[rel:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {{klaso_id: '{klaso_id}'}})
                        DELETE rel
                    """)
                    logger.info(f"  ✓ Deleted {count} annotations to {klaso_nomo} ({klaso_id})")
                    self.stats['deleted'] += count
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete annotations for {klaso_id}: {e}")

        self.stats['kept'] = self.stats['initial_annotations'] - self.stats['deleted']

    def verify_cleanup(self):
        """Verify annotations after cleanup."""
        logger.info("\nVerifying cleanup...")

        # Count remaining annotations
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN count(r)
        """)
        remaining = result.get_next()[0]

        logger.info(f"  Remaining annotations: {remaining}")

        # Check top-level classes are empty
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            WHERE v.superklaso_id IS NULL OR v.superklaso_id = ''
            RETURN count(r)
        """)
        top_level_remaining = result.get_next()[0]

        if top_level_remaining > 0:
            logger.warning(f"  ⚠ {top_level_remaining} annotations still on top-level classes!")
        else:
            logger.info(f"  ✓ All top-level class annotations removed")

        # Sample remaining annotations
        logger.info("\n  Sample remaining annotations:")
        result = self.conn.execute("""
            MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso)
            RETURN r.radiko, v.klaso_nomo
            LIMIT 10
        """)

        while result.has_next():
            radiko, klaso_nomo = result.get_next()
            logger.info(f"    {radiko} → {klaso_nomo}")

    def print_stats(self):
        """Print cleanup statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial annotations:          {self.stats['initial_annotations']}")
        logger.info(f"  Top-level (polluted):       {self.stats['top_level_annotations']}")
        logger.info(f"  Fine-grained (clean):       {self.stats['fine_grained_annotations']}")
        logger.info(f"Deleted:                      {self.stats['deleted']}")
        logger.info(f"Kept (high-quality):          {self.stats['kept']}")


def main():
    parser = argparse.ArgumentParser(
        description='Remove frequency-based annotations from top-level classes'
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

    remover = FrequencyAnnotationRemover(Path(args.db), dry_run=args.dry_run)

    # Get top-level classes
    top_level_classes = remover.get_top_level_classes()

    # Count current annotations
    remover.count_annotations_by_class()

    # Delete top-level annotations
    remover.delete_top_level_annotations(top_level_classes)

    # Print stats
    remover.print_stats()

    # Verify
    if not args.dry_run:
        remover.verify_cleanup()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
