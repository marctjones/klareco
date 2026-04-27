#!/usr/bin/env python3
"""
Reset Semantic Annotations to High-Quality Only

Deletes ALL semantic annotations and reloads ONLY:
1. Manual annotations (23 roots) - from phase_0_*.jsonl
2. Gazetteer-based (161 roots) - from gazetteers
3. ReVo expansion (depth 1-2) - via synonym chains

Skips frequency-based morphological expansion (corrupted).

Usage:
    python scripts/reset_semantic_annotations.py --dry-run
    python scripts/reset_semantic_annotations.py  # Execute reset
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


def delete_all_annotations(kuzu_db_path: Path, dry_run: bool = False):
    """Delete ALL semantic annotations."""
    logger.info("Connecting to Kuzu database...")
    db = kuzu.Database(str(kuzu_db_path))
    conn = kuzu.Connection(db)

    # Count current annotations
    result = conn.execute("""
        MATCH ()-[rel:APARTENAS_AL_VERBA_KLASO]->()
        RETURN count(rel)
    """)
    verb_count = result.get_next()[0]

    result = conn.execute("""
        MATCH ()-[rel:HAVAS_ENTECAN_TIPON]->()
        RETURN count(rel)
    """)
    entity_count = result.get_next()[0]

    total_count = verb_count + entity_count

    logger.info(f"\nCurrent annotations:")
    logger.info(f"  Verb classifications:  {verb_count}")
    logger.info(f"  Entity classifications: {entity_count}")
    logger.info(f"  TOTAL:                  {total_count}")

    if dry_run:
        logger.info(f"\n[DRY RUN] Would delete all {total_count} annotations")
        return

    # Delete all verb annotations
    logger.info("\nDeleting verb annotations...")
    conn.execute("""
        MATCH ()-[rel:APARTENAS_AL_VERBA_KLASO]->()
        DELETE rel
    """)
    logger.info(f"  ✓ Deleted {verb_count} verb annotations")

    # Delete all entity annotations
    logger.info("Deleting entity annotations...")
    conn.execute("""
        MATCH ()-[rel:HAVAS_ENTECAN_TIPON]->()
        DELETE rel
    """)
    logger.info(f"  ✓ Deleted {entity_count} entity annotations")

    logger.info(f"\n✓ All {total_count} annotations deleted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path,
                       default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kuzu database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without executing')

    args = parser.parse_args()

    delete_all_annotations(args.db, dry_run=args.dry_run)

    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS:")
        logger.info("=" * 60)
        logger.info("1. Reload manual annotations:")
        logger.info("   python scripts/load_semantic_relationships.py --annotations data/annotations/phase_0_*.jsonl")
        logger.info("\n2. Reload gazetteer annotations:")
        logger.info("   python scripts/annotate_core_roots_from_gazetteers.py")
        logger.info("\n3. Expand via ReVo synonyms:")
        logger.info("   python scripts/expand_annotations_via_revo.py --max-depth 2")
        logger.info("\n4. Verify final count (should be ~327):")
        logger.info("   python scripts/verify_semantic_annotations.py")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
