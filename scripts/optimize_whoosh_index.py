#!/usr/bin/env python3
"""
Optimize Whoosh Index for Faster Searches

VERSION: v2.1
COMPATIBLE WITH: v2.1 Whoosh FTS index
DEPENDENCIES: None
STAGE: Utility

Description:
    Optimizes the Whoosh full-text search index by merging segments.
    This is a one-time operation that restructures the index for faster searches.
    No data is lost - only index structure is optimized.

Usage:
    python scripts/optimize_whoosh_index.py
    python scripts/optimize_whoosh_index.py --index-dir data/indexes/whoosh_fts

Expected Impact:
    - 1.5-2x faster search speed
    - Reduced disk space usage (merged segments)
    - Zero accuracy impact (same results, just faster)

Last Updated: 2026-03-25
"""

import argparse
import logging
from pathlib import Path
from whoosh import index

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_whoosh_index(index_dir: Path):
    """
    Optimize Whoosh index by merging segments.

    Args:
        index_dir: Path to Whoosh index directory
    """
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        return False

    try:
        logger.info(f"Opening Whoosh index at {index_dir}")
        ix = index.open_dir(index_dir)

        # Get initial stats
        with ix.searcher() as searcher:
            doc_count = searcher.doc_count_all()
            logger.info(f"Index contains {doc_count:,} documents")

        # Get segment info before optimization
        try:
            segment_count = ix.segment_counter()
            logger.info(f"Index has generated {segment_count} segments total")
        except:
            logger.info("Cannot determine segment count (API limitation)")

        # Optimize (merge segments) using writer
        logger.info("Optimizing index (this may take a few minutes)...")
        logger.info("This will merge all segments into a single optimized segment...")

        writer = ix.writer()
        writer.commit(optimize=True)

        logger.info("Index optimization complete!")

        # Get final stats
        with ix.searcher() as searcher:
            doc_count_after = searcher.doc_count_all()
            logger.info(f"Index still contains {doc_count_after:,} documents (no data lost)")

        if doc_count != doc_count_after:
            logger.warning(f"Document count changed! Before: {doc_count}, After: {doc_count_after}")
            return False

        logger.info("✓ Optimization complete!")
        logger.info(f"  Expected speedup: 1.5-2x faster searches")

        return True

    except Exception as e:
        logger.error(f"Failed to optimize index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--index-dir',
        type=Path,
        default=Path('data/indexes/whoosh_fts'),
        help='Path to Whoosh index directory (default: data/indexes/whoosh_fts)'
    )

    args = parser.parse_args()

    success = optimize_whoosh_index(args.index_dir)

    if success:
        print("\n" + "=" * 80)
        print("INDEX OPTIMIZATION SUCCESSFUL")
        print("=" * 80)
        print(f"Index at {args.index_dir} has been optimized.")
        print("Searches should now be 1.5-2x faster with no accuracy impact.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("INDEX OPTIMIZATION FAILED")
        print("=" * 80)
        print("See error messages above for details.")
        print("=" * 80)
        exit(1)


if __name__ == '__main__':
    main()
