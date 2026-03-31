#!/usr/bin/env python3
"""
Fix Proper Noun Annotations in Kuzu Database - Sentence-Based Approach

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu database schema
DEPENDENCIES: Fixed parser with negative detection
STAGE: Utility

Description:
    Re-parses sentences (not isolated words!) to fix proper noun annotations.
    Only updates sentence-initial words where negative detection applies.

Strategy:
    1. Query Frazoteksto nodes where teksto starts with capital letter
    2. Extract first word from sentence text
    3. Re-parse FULL sentence with fixed parser
    4. Find corresponding Vorto node (subject kerno)
    5. Compare old vs new vortspeco
    6. Update if different

Usage:
    python scripts/fix_proper_nouns_in_kuzu_v2.py

    # Dry run:
    python scripts/fix_proper_nouns_in_kuzu_v2.py --dry-run

    # Limit for testing:
    python scripts/fix_proper_nouns_in_kuzu_v2.py --limit 1000

Estimated Time: 1-2 hours for 5.4M sentences

Last Updated: 2026-03-30
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu
from klareco.parser import parse

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"fix_proper_nouns_v2_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def get_sentences_with_capitalized_first_word(conn, limit=None):
    """Get Frazoteksto nodes where first word is capitalized."""
    logger = logging.getLogger(__name__)
    logger.info("Querying sentences with capitalized first word...")

    query = """
        MATCH (ft:Frazoteksto)
        WHERE ft.teksto =~ '^[A-ZĈĜĤĴŜŬ].*'
        RETURN ft.id, ft.teksto
    """

    if limit:
        query += f" LIMIT {limit}"

    result = conn.execute(query)
    sentences = []
    while result.has_next():
        row = result.get_next()
        sentences.append({
            'ft_id': row[0],
            'teksto': row[1]
        })

    logger.info(f"Found {len(sentences):,} sentences")
    return sentences


def get_first_word_vorto_id(conn, ft_id: int, first_word: str):
    """
    Find the Vorto node ID for the first word in a sentence.

    The first word is typically the subject kerno in SVO languages.
    """
    try:
        # Try to get subject Vortgrupo kerno
        result = conn.execute(f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)
            MATCH (ast)-[:AST_HAVAS_FRAZON]->(f:Frazo)
            MATCH (f)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(vg:Vortgrupo)
            MATCH (vg)-[:HAVAS_KERNON]->(v:Vorto)
            WHERE ft.id = {ft_id}
              AND v.plena_vorto = '{first_word}'
            RETURN v.id, v.vortspeco, v.kategorio
        """)
        if result.has_next():
            return result.get_next()

        # Try direct Vorto subject
        result = conn.execute(f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)
            MATCH (ast)-[:AST_HAVAS_FRAZON]->(f:Frazo)
            MATCH (f)-[:HAVAS_SUBJEKTON_VORTO]->(v:Vorto)
            WHERE ft.id = {ft_id}
              AND v.plena_vorto = '{first_word}'
            RETURN v.id, v.vortspeco, v.kategorio
        """)
        if result.has_next():
            return result.get_next()

        # Last resort: find ANY Vorto with matching word
        result = conn.execute(f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(ast:AST)
            MATCH (ast)-[:AST_HAVAS_FRAZON]->(f:Frazo)
            MATCH (f)-[r]-(v:Vorto)
            WHERE ft.id = {ft_id}
              AND v.plena_vorto = '{first_word}'
            RETURN v.id, v.vortspeco, v.kategorio
            LIMIT 1
        """)
        if result.has_next():
            return result.get_next()

    except Exception as e:
        logging.debug(f"Error finding Vorto for '{first_word}': {e}")

    return None


def parse_sentence_get_first_word_vortspeco(teksto: str):
    """Parse sentence and get vortspeco of first word."""
    try:
        ast = parse(teksto)

        # Get first word from subject (typical SVO position)
        if ast.get('subjekto'):
            target = None
            if ast['subjekto'].get('tipo') == 'vorto':
                target = ast['subjekto']
            elif ast['subjekto'].get('tipo') == 'vortgrupo':
                target = ast['subjekto'].get('kerno')

            if target:
                return {
                    'vortspeco': target.get('vortspeco', 'nekonata'),
                    'kategorio': target.get('kategorio', None),
                    'plena_vorto': target.get('plena_vorto', '')
                }

        # Fallback: check verb if sentence starts with verb (VSO)
        if ast.get('verbo') and ast['verbo'].get('tipo') == 'vorto':
            return {
                'vortspeco': ast['verbo'].get('vortspeco', 'nekonata'),
                'kategorio': ast['verbo'].get('kategorio', None),
                'plena_vorto': ast['verbo'].get('plena_vorto', '')
            }

    except Exception as e:
        logging.debug(f"Parse error for '{teksto[:50]}...': {e}")

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Fix proper noun annotations by re-parsing sentences'
    )
    parser.add_argument('--db-path', type=str,
                       default='data/indexes/v2.1_kuzu_index_full',
                       help='Path to Kuzu database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without updating database')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit sentences to process (for testing)')

    args = parser.parse_args()

    log_dir = Path('logs/kuzu_updates')
    logger = setup_logging(log_dir)

    logger.info("=== Fix Proper Nouns - Sentence-Based Approach ===")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Dry run: {args.dry_run}")

    # Connect to database
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info("Connecting to database...")
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    # Get sentences
    sentences = get_sentences_with_capitalized_first_word(conn, args.limit)

    # Process each sentence
    logger.info("Re-parsing sentences with fixed parser...")

    updates = []
    not_found = 0
    no_change = 0

    for i, sent_data in enumerate(sentences):
        ft_id = sent_data['ft_id']
        teksto = sent_data['teksto']

        # Get first word
        first_word = teksto.split()[0] if teksto else ''
        if not first_word:
            continue

        # Re-parse sentence
        parsed = parse_sentence_get_first_word_vortspeco(teksto)
        if not parsed:
            not_found += 1
            continue

        # Get current Vorto node
        vorto_data = get_first_word_vorto_id(conn, ft_id, first_word)
        if not vorto_data:
            not_found += 1
            continue

        vorto_id, old_vortspeco, old_kategorio = vorto_data
        new_vortspeco = parsed['vortspeco']
        new_kategorio = parsed['kategorio']

        # Check if update needed
        if old_vortspeco != new_vortspeco:
            updates.append({
                'vorto_id': vorto_id,
                'word': first_word,
                'old_vortspeco': old_vortspeco,
                'new_vortspeco': new_vortspeco,
                'old_kategorio': old_kategorio,
                'new_kategorio': new_kategorio,
                'sentence': teksto[:60] + '...'
            })
        else:
            no_change += 1

        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i+1:,} / {len(sentences):,} "
                       f"({len(updates):,} updates, {not_found:,} not found)")

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"  Sentences processed: {len(sentences):,}")
    logger.info(f"  Updates needed: {len(updates):,}")
    logger.info(f"  No change needed: {no_change:,}")
    logger.info(f"  Vorto not found: {not_found:,}")

    # Group by change type
    if updates:
        changes = defaultdict(int)
        for update in updates:
            change_key = f"{update['old_vortspeco']} → {update['new_vortspeco']}"
            changes[change_key] += 1

        logger.info("\n=== Changes by Type ===")
        for change, count in sorted(changes.items(), key=lambda x: -x[1]):
            logger.info(f"  {change:40s} {count:,}")

        logger.info("\n=== Sample Updates ===")
        for update in updates[:10]:
            logger.info(f"  {update['word']:20s} {update['old_vortspeco']:15s} → "
                       f"{update['new_vortspeco']:15s} | {update['sentence']}")

        # Apply updates
        if not args.dry_run:
            logger.info(f"\n=== Applying {len(updates):,} Updates ===")
            batch_size = 1000

            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]

                for update in batch:
                    try:
                        set_clause = f"v.vortspeco = '{update['new_vortspeco']}'"
                        if update['new_kategorio']:
                            set_clause += f", v.kategorio = '{update['new_kategorio']}'"

                        query = f"""
                            MATCH (v:Vorto)
                            WHERE v.id = {update['vorto_id']}
                            SET {set_clause}
                        """
                        conn.execute(query)
                    except Exception as e:
                        logger.error(f"Update failed for vorto_id={update['vorto_id']}: {e}")

                if (i + batch_size) % 10000 == 0:
                    logger.info(f"  Updated {i+batch_size:,} / {len(updates):,}")

            logger.info(f"✓ Applied {len(updates):,} updates")
        else:
            logger.info(f"\nDRY RUN: Would update {len(updates):,} words")

    logger.info("\n✓ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
