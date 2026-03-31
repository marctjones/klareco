#!/usr/bin/env python3
"""
Fix Proper Noun Annotations in Kuzu Database (In-Place Update)

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu database schema
DEPENDENCIES: Fixed parser with negative detection
STAGE: Utility

Description:
    Updates vortspeco and kategorio fields for proper nouns directly in
    the existing Kuzu database, avoiding the need for full re-index.

    Uses the fixed parser (with negative detection) to identify which
    capitalized words should be proper nouns vs common nouns.

Strategy:
    1. Query unique capitalized words from database
    2. Re-parse with fixed parser to determine correct vortspeco
    3. Batch update database with correct annotations

Estimated Time: 30 minutes - 2 hours (vs 6-8 hours for full re-index)

Usage:
    python scripts/fix_proper_nouns_in_kuzu.py

    # Dry run (no updates):
    python scripts/fix_proper_nouns_in_kuzu.py --dry-run

    # Limit updates for testing:
    python scripts/fix_proper_nouns_in_kuzu.py --limit 1000

Inputs:
    - Kuzu database: data/indexes/v2.1_kuzu_index_full

Outputs:
    - Updated Vorto nodes with correct vortspeco and kategorio
    - Log file: logs/kuzu_fix_proper_nouns_YYYYMMDD_HHMMSS.log

Last Updated: 2026-03-30
Author: Claude Sonnet 4.5
Related Issues: #TBD
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu
from klareco.parser import parse

# Setup logging
def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"kuzu_fix_proper_nouns_{timestamp}.log"

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


def get_unique_capitalized_words(conn):
    """Get all unique capitalized words that need checking."""
    logger = logging.getLogger(__name__)

    logger.info("Querying unique capitalized words...")

    # Get unique (plena_vorto, radiko, vortspeco, kategorio) combinations
    query = """
        MATCH (v:Vorto)
        WHERE v.plena_vorto =~ '^[A-ZĈĜĤĴŜŬ].*'
        RETURN DISTINCT v.plena_vorto as word,
                       v.radiko as radiko,
                       v.vortspeco as vortspeco,
                       v.kategorio as kategorio
    """

    result = conn.execute(query)
    words = []
    while result.has_next():
        row = result.get_next()
        words.append({
            'word': row[0],
            'radiko': row[1],
            'vortspeco': row[2],
            'kategorio': row[3]
        })

    logger.info(f"Found {len(words):,} unique capitalized words")
    return words


def parse_with_fixed_parser(word: str, position: str = 'mid') -> dict:
    """
    Parse word with fixed parser to determine correct vortspeco.

    Args:
        word: Word to parse
        position: 'start' or 'mid' (sentence position)

    Returns:
        dict with 'vortspeco' and 'kategorio'
    """
    # Create test sentence with word in specified position
    if position == 'start':
        test_sentence = f"{word} estas testo."
    else:
        test_sentence = f"La {word} estas testo."

    try:
        ast = parse(test_sentence)

        # Extract vortspeco from parsed word
        target_word_ast = None
        if ast.get('subjekto'):
            if ast['subjekto'].get('tipo') == 'vorto':
                target_word_ast = ast['subjekto']
            elif ast['subjekto'].get('tipo') == 'vortgrupo':
                # Check priskriboj (modifiers) for our word
                priskriboj = ast['subjekto'].get('priskriboj', [])
                for priskribo in priskriboj:
                    if priskribo.get('tipo') == 'vorto' and priskribo.get('plena_vorto', '').lower() == word.lower():
                        target_word_ast = priskribo
                        break
                # Check kerno
                if not target_word_ast and ast['subjekto'].get('kerno'):
                    kerno = ast['subjekto']['kerno']
                    if kerno.get('plena_vorto', '').lower() == word.lower():
                        target_word_ast = kerno

        if target_word_ast:
            return {
                'vortspeco': target_word_ast.get('vortspeco', 'nekonata'),
                'kategorio': target_word_ast.get('kategorio', None)
            }
    except Exception as e:
        logging.debug(f"Parse error for '{word}': {e}")

    return {'vortspeco': 'nekonata', 'kategorio': None}


def should_update(current: dict, correct: dict) -> bool:
    """Check if word needs updating."""
    # Update if vortspeco changed
    if current['vortspeco'] != correct['vortspeco']:
        return True

    # Update if kategorio changed (and correct is not None)
    if correct['kategorio'] is not None and current['kategorio'] != correct['kategorio']:
        return True

    return False


def batch_update_words(conn, updates: list, dry_run: bool = False):
    """Apply batch updates to database."""
    logger = logging.getLogger(__name__)

    if dry_run:
        logger.info(f"DRY RUN: Would update {len(updates)} words")
        # Show first 10 examples
        for update in updates[:10]:
            logger.info(f"  {update['word']:20s} {update['old_vortspeco']:15s} → {update['new_vortspeco']:15s}")
        return

    logger.info(f"Applying {len(updates):,} updates...")

    # Update in batches
    batch_size = 1000
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]

        for update in batch:
            try:
                # Build SET clause
                set_clause = f"v.vortspeco = '{update['new_vortspeco']}'"
                if update['new_kategorio']:
                    set_clause += f", v.kategorio = '{update['new_kategorio']}'"

                query = f"""
                    MATCH (v:Vorto)
                    WHERE v.plena_vorto = '{update['word']}'
                      AND v.vortspeco = '{update['old_vortspeco']}'
                    SET {set_clause}
                """
                conn.execute(query)
            except Exception as e:
                logger.error(f"Update failed for '{update['word']}': {e}")

        if (i + batch_size) % 10000 == 0:
            logger.info(f"  Updated {i+batch_size:,} / {len(updates):,}")

    logger.info(f"✓ Applied {len(updates):,} updates")


def main():
    parser = argparse.ArgumentParser(description='Fix proper noun annotations in Kuzu database')
    parser.add_argument('--db-path', type=str,
                       default='data/indexes/v2.1_kuzu_index_full',
                       help='Path to Kuzu database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without changing database')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of words to check (for testing)')

    args = parser.parse_args()

    # Setup logging
    log_dir = Path('logs/kuzu_updates')
    logger = setup_logging(log_dir)

    logger.info("=== Fix Proper Noun Annotations in Kuzu ===")
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

    # Get unique capitalized words
    words = get_unique_capitalized_words(conn)

    if args.limit:
        logger.info(f"Limiting to first {args.limit} words")
        words = words[:args.limit]

    # Check each word with fixed parser
    logger.info("Re-parsing words with fixed parser...")
    updates = []

    for i, word_data in enumerate(words):
        word = word_data['word']
        current_vortspeco = word_data['vortspeco']
        current_kategorio = word_data['kategorio']

        # Parse in mid-sentence position (after "la")
        # This triggers proper noun detection for non-initial positions
        correct = parse_with_fixed_parser(word, position='mid')

        # Check if update needed
        if should_update(word_data, correct):
            updates.append({
                'word': word,
                'old_vortspeco': current_vortspeco,
                'new_vortspeco': correct['vortspeco'],
                'old_kategorio': current_kategorio,
                'new_kategorio': correct['kategorio']
            })

        if (i + 1) % 10000 == 0:
            logger.info(f"  Checked {i+1:,} / {len(words):,} words ({len(updates):,} need updates)")

    logger.info(f"\nSummary:")
    logger.info(f"  Total words checked: {len(words):,}")
    logger.info(f"  Words needing update: {len(updates):,}")
    logger.info(f"  Update rate: {100*len(updates)/len(words):.1f}%")

    # Group by change type
    changes = defaultdict(int)
    for update in updates:
        change_key = f"{update['old_vortspeco']} → {update['new_vortspeco']}"
        changes[change_key] += 1

    logger.info("\nChanges by type:")
    for change, count in sorted(changes.items(), key=lambda x: -x[1]):
        logger.info(f"  {change:40s} {count:,}")

    # Apply updates
    if updates:
        batch_update_words(conn, updates, dry_run=args.dry_run)

    logger.info("\n✓ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
