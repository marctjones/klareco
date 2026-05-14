#!/usr/bin/env python3
"""
Build Whoosh Full-Text Search Index from Kuzu Database

Creates a Whoosh index for fast full-text search over sentence corpus.
This solves the retrieval bottleneck in extractive QA by enabling
efficient keyword-based sentence retrieval.

Usage:
    python scripts/index/build_whoosh_index.py
    python scripts/index/build_whoosh_index.py --db data/indexes/v2.1_kuzu_index_full
    python scripts/index/build_whoosh_index.py --output data/indexes/whoosh_fts
"""

import argparse
import logging
import sys
from pathlib import Path

import kuzu
from whoosh import scoring
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(kuzu_db_path: Path, output_dir: Path, overwrite: bool = False):
    """
    Build Whoosh FTS index from Kuzu database.

    Args:
        kuzu_db_path: Path to Kuzu database
        output_dir: Directory to store Whoosh index
        overwrite: If True, rebuild index from scratch
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    if exists_in(str(output_dir)) and not overwrite:
        logger.info(f"Index already exists at {output_dir}")
        logger.info("Use --overwrite to rebuild from scratch")
        return

    # Define schema
    schema = Schema(
        id=ID(stored=True, unique=True),
        text=TEXT(stored=True),  # Full sentence text
        text_lower=TEXT,  # Lowercased for case-insensitive search
    )

    # Create index
    logger.info(f"Creating Whoosh index at {output_dir}")
    ix = create_in(str(output_dir), schema)
    writer = ix.writer(limitmb=512, procs=4, multisegment=True)

    # Connect to Kuzu
    logger.info(f"Loading sentences from {kuzu_db_path}")
    db = kuzu.Database(str(kuzu_db_path))
    conn = kuzu.Connection(db)

    # Query all sentences
    query = """
        MATCH (ft:Frazoteksto)
        WHERE ft.teksto IS NOT NULL
        RETURN ft.id AS id, ft.teksto AS text
    """

    result = conn.execute(query)

    # Index sentences
    count = 0
    batch_size = 10000

    while result.has_next():
        row = result.get_next()
        sentence_id = str(row[0])
        text = row[1]

        if not text:
            continue

        # Add to index
        writer.add_document(
            id=sentence_id,
            text=text,
            text_lower=text.lower()
        )

        count += 1
        if count % batch_size == 0:
            logger.info(f"Indexed {count:,} sentences...")

    # Commit index
    logger.info(f"Committing index ({count:,} sentences total)...")
    writer.commit()

    logger.info(f"✓ Index built successfully: {output_dir}")
    logger.info(f"  Total sentences: {count:,}")
    logger.info(f"  Index size: {sum(f.stat().st_size for f in output_dir.glob('*')) / 1024 / 1024:.1f} MB")


def test_index(index_dir: Path):
    """Test the index with sample queries."""
    if not exists_in(str(index_dir)):
        logger.error(f"Index not found at {index_dir}")
        return

    logger.info(f"\nTesting index at {index_dir}")
    ix = open_dir(str(index_dir))

    # Test queries
    test_queries = [
        "zamenhof",
        "esperanto",
        "zamenhof AND (kre OR fond OR establ)",
    ]

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        for query_str in test_queries:
            logger.info(f"\nQuery: {query_str}")

            from whoosh.qparser import QueryParser
            query = QueryParser("text_lower", ix.schema).parse(query_str)
            results = searcher.search(query, limit=5)

            logger.info(f"  Found {len(results)} results")
            for i, hit in enumerate(results, 1):
                text = hit['text'][:100] + '...' if len(hit['text']) > 100 else hit['text']
                logger.info(f"  {i}. {text}")
                logger.info(f"     Score: {hit.score:.4f}, ID: {hit['id']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--db', type=Path, default=Path('data/indexes/v2.1_kuzu_index_full'),
                       help='Path to Kuzu database')
    parser.add_argument('--output', type=Path, default=Path('data/indexes/whoosh_fts'),
                       help='Output directory for Whoosh index')
    parser.add_argument('--overwrite', action='store_true',
                       help='Rebuild index from scratch')
    parser.add_argument('--test', action='store_true',
                       help='Test index with sample queries')

    args = parser.parse_args()

    if args.test:
        test_index(args.output)
    else:
        build_index(args.db, args.output, args.overwrite)
        test_index(args.output)


if __name__ == '__main__':
    main()
