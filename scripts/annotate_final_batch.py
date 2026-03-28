#!/usr/bin/env python3
"""
Annotate Final Batch (Reaching 200+ Roots)

VERSION: v2.2
COMPATIBLE WITH: v2.2 database schema (semantic ontology)
DEPENDENCIES: Kuzu database with semantic ontology tables
STAGE: Data

Description:
    Final batch to push annotations from 174 → 200+ roots.
    Focuses on high-utility roots for QA system.

Usage:
    python scripts/annotate_final_batch.py --db data/indexes/v2.1_kuzu_index_full

Last Updated: 2026-03-28
Author: Claude Sonnet 4.5
Related Issues: #18 (Annotate 200 core roots)
"""

import logging
import argparse
from pathlib import Path
import kuzu

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def annotate_final_batch(db_path: Path, dry_run: bool = False):
    """Annotate final batch of 30+ roots."""

    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    stats = {'total': 0, 'missing': []}

    def link(root: str, tipo_id: str, is_verb: bool = False):
        """Link root to semantic class."""
        # Check exists
        result = conn.execute(f"MATCH (r:Radiko {{radiko: '{root}'}}) RETURN count(*)")
        if result.get_next()[0] == 0:
            logger.warning(f"Missing: {root}")
            stats['missing'].append(root)
            return False

        if dry_run:
            logger.info(f"[DRY-RUN] {root} → {tipo_id}")
            return True

        try:
            if is_verb:
                query = f"""
                    MATCH (r:Radiko {{radiko: '{root}'}}), (v:VerbaKlaso {{klaso_id: '{tipo_id}'}})
                    MERGE (r)-[:APARTENAS_AL_VERBA_KLASO]->(v)
                """
            else:
                query = f"""
                    MATCH (r:Radiko {{radiko: '{root}'}}), (e:EntecaTipo {{tipo_id: '{tipo_id}'}})
                    MERGE (r)-[:HAVAS_ENTECAN_TIPON]->(e)
                """
            conn.execute(query)
            stats['total'] += 1
            return True
        except Exception as e:
            logger.error(f"Error {root}: {e}")
            return False

    logger.info("=== FINAL BATCH: High-Priority Roots ===\n")

    # More creation verbs (kreado-26)
    logger.info("Kreado verbs:")
    for root in ['produc', 'aŭtor', 'kompon', 'fabrik']:
        link(root, 'kreado-26', is_verb=True)

    # More movement verbs (movo-51)
    logger.info("\nMovo verbs:")
    for root in ['voj', 'flu', 'al', 'de', 'en']:
        link(root, 'movo-51', is_verb=True)

    # More cognition verbs (pensado-29)
    logger.info("\nPensado verbs:")
    for root in ['supoz', 'imag', 'ren', 'opini']:
        link(root, 'pensado-29', is_verb=True)

    # More communication verbs (komunikado-37)
    logger.info("\nKomunikado verbs:")
    for root in ['anonc', 'deklar', 'skrib', 'leg']:
        link(root, 'komunikado-37', is_verb=True)

    # More time nouns (tempo)
    logger.info("\nTempo nouns:")
    for root in ['semajn', 'dat', 'epok', 'sekol', 'cent']:
        link(root, 'tempo')

    # More locations (loko)
    logger.info("\nLoko nouns:")
    for root in ['dom', 'palac', 'kastel', 'vilaĝ', 'plac']:
        link(root, 'loko')

    # More persons (persono)
    logger.info("\nPersono nouns:")
    for root in ['homo', 'vir', 'patrino', 'avo', 'nep']:
        link(root, 'persono')

    # More professions (profesio-50 verbs)
    logger.info("\nProfesio verbs:")
    for root in ['instru', 'servi', 'help', 'studi']:
        link(root, 'profesio-50', is_verb=True)

    # Verify total
    logger.info("\n" + "=" * 60)
    result = conn.execute("MATCH ()-[r:APARTENAS_AL_VERBA_KLASO]->() RETURN count(*)")
    verb_count = result.get_next()[0]

    result = conn.execute("MATCH ()-[r:HAVAS_ENTECAN_TIPON]->() RETURN count(*)")
    entity_count = result.get_next()[0]

    total = verb_count + entity_count

    logger.info(f"NEW in this batch: {stats['total']}")
    logger.info(f"TOTAL ANNOTATED: {total} roots")
    logger.info(f"Missing: {len(stats['missing'])} roots")

    if total >= 200:
        logger.info("\n🎉 TARGET REACHED! 200+ roots annotated! 🎉")
    else:
        logger.info(f"\nRemaining: {200 - total} roots")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    return annotate_final_batch(args.db, args.dry_run)


if __name__ == '__main__':
    exit(main())
