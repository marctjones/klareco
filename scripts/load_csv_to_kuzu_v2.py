#!/usr/bin/env python3
"""
Load CSV files into Kuzu database using COPY FROM (batch loading).

This is 100-1000x faster than individual CREATE statements.

Usage:
    python scripts/load_csv_to_kuzu_v2.py --csvs data/csv_export \
                                           --output data/indexes/v2_kuzu_index
"""

import argparse
import logging
from pathlib import Path

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    exit(1)

from klareco.schema.kuzu_ast_schema import get_create_statements

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVLoader:
    """Load CSV files into Kuzu using COPY FROM."""

    def __init__(self, db_path: Path, csv_dir: Path):
        """Initialize loader."""
        self.db_path = db_path
        self.csv_dir = csv_dir
        self.nodes_dir = csv_dir / "nodes"
        self.rels_dir = csv_dir / "rels"
        self.db = None
        self.conn = None

    def connect(self):
        """Connect to Kuzu database."""
        logger.info(f"Opening Kuzu database at {self.db_path}")
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

    def create_schema(self):
        """Create schema."""
        logger.info("Creating v2.0 schema...")
        statements = get_create_statements()
        for stmt in statements:
            self.conn.execute(stmt)
        logger.info(f"Schema created: {len(statements)} statements")

    def load_nodes(self):
        """Load all node CSV files."""
        logger.info("Loading node tables...")

        node_tables = [
            'SourceCollection',
            'Document',
            'Sentence',
            'AST',
            'Frazo',
            'Vortgrupo',
            'Vorto',
            'Root'
        ]

        for table in node_tables:
            csv_file = self.nodes_dir / f"{table}.csv"
            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue

            logger.info(f"  Loading {table}...")

            # Use COPY FROM for batch loading
            # Note: PARALLEL=FALSE needed for CSVs with quoted newlines
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false)
            """)

            # Count rows
            result = self.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count} rows")

    def load_relationships(self):
        """Load all relationship CSV files."""
        logger.info("Loading relationship tables...")

        # Relationship files
        rel_tables = [
            ('IN_COLLECTION', 'Document', 'SourceCollection'),
            ('SENTENCE_HAS_AST', 'Sentence', 'AST'),
            ('AST_HAS_FRAZO', 'AST', 'Frazo'),
            ('HAS_SUBJEKTO_VORTGRUPO', 'Frazo', 'Vortgrupo'),
            ('HAS_SUBJEKTO_VORTO', 'Frazo', 'Vorto'),
            ('HAS_VERBO', 'Frazo', 'Vorto'),
            ('HAS_OBJEKTO_VORTGRUPO', 'Frazo', 'Vortgrupo'),
            ('HAS_OBJEKTO_VORTO', 'Frazo', 'Vorto'),
            ('HAS_ALIAJ', 'Frazo', 'Vorto'),
            ('HAS_KERNO', 'Vortgrupo', 'Vorto'),
            ('HAS_PRISKRIBO', 'Vortgrupo', 'Vorto'),
        ]

        for table, from_type, to_type in rel_tables:
            csv_file = self.rels_dir / f"{table}.csv"
            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue

            logger.info(f"  Loading {table}...")

            # Use COPY FROM for batch loading
            # Note: PARALLEL=FALSE needed for CSVs with quoted newlines
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false)
            """)

            # Count relationships
            result = self.conn.execute(f"MATCH ()-[r:{table}]->() RETURN count(r)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count} relationships")

    def create_has_root_relationships(self):
        """Create HAS_ROOT relationships by matching Vorto.radiko to Root.root."""
        logger.info("Creating HAS_ROOT relationships...")

        self.conn.execute("""
            MATCH (v:Vorto), (r:Root)
            WHERE v.radiko = r.root
            CREATE (v)-[:HAS_ROOT {is_primary: true, position: 0}]->(r)
        """)

        result = self.conn.execute("MATCH ()-[r:HAS_ROOT]->() RETURN count(r)")
        count = result.get_next()[0]
        logger.info(f"  Created {count} HAS_ROOT relationships")

    def get_stats(self):
        """Get loading statistics."""
        logger.info("\n=== Loading Statistics ===")

        node_types = [
            'SourceCollection', 'Document', 'Sentence',
            'AST', 'Frazo', 'Vortgrupo', 'Vorto', 'Root'
        ]

        for node_type in node_types:
            result = self.conn.execute(f"MATCH (n:{node_type}) RETURN count(n)")
            count = result.get_next()[0]
            logger.info(f"  {node_type}: {count}")

    def load(self):
        """Load all CSV files."""
        self.connect()
        self.create_schema()
        self.load_nodes()
        self.load_relationships()
        self.create_has_root_relationships()
        self.get_stats()


def main():
    parser = argparse.ArgumentParser(description='Load CSV files into Kuzu')
    parser.add_argument('--csvs', type=Path, required=True, help='Directory with CSV files')
    parser.add_argument('--output', type=Path, required=True, help='Output Kuzu database path')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (delete existing DB)')

    args = parser.parse_args()

    # Check if database exists
    if args.output.exists() and args.fresh:
        logger.info(f"Removing existing database at {args.output}")
        import shutil
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()

    # Load CSV files
    loader = CSVLoader(args.output, args.csvs)
    loader.load()


if __name__ == '__main__':
    main()
