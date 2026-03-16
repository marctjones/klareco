#!/usr/bin/env python3
"""
Memory-efficient batch loader for v2.0 Kuzu database.

Optimizations:
- Batched relationship creation (prevents OOM on large graphs)
- Progress checkpointing (can resume if interrupted)
- Memory-conscious COPY operations
- Incremental validation

Usage:
    python scripts/load_csv_to_kuzu_v2_batched.py --csvs data/csv_export_full \
                                                    --output data/indexes/v2_kuzu_index_full \
                                                    --fresh
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List

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


class BatchedCSVLoader:
    """Memory-efficient CSV loader with batching and checkpointing."""

    def __init__(self, db_path: Path, csv_dir: Path, batch_size: int = 100000):
        """Initialize loader."""
        self.db_path = db_path
        self.csv_dir = csv_dir
        self.nodes_dir = csv_dir / "nodes"
        self.rels_dir = csv_dir / "rels"
        self.batch_size = batch_size
        # Store checkpoint in parent directory
        self.checkpoint_file = db_path.parent / f".{db_path.name}_checkpoint.json"
        self.db = None
        self.conn = None

    def connect(self):
        """Connect to Kuzu database."""
        logger.info(f"Opening Kuzu database at {self.db_path}")
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"completed_steps": []}

    def save_checkpoint(self, step: str):
        """Save checkpoint."""
        checkpoint = self.load_checkpoint()
        if step not in checkpoint["completed_steps"]:
            checkpoint["completed_steps"].append(step)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    def is_step_complete(self, step: str) -> bool:
        """Check if step is complete."""
        checkpoint = self.load_checkpoint()
        return step in checkpoint["completed_steps"]

    def create_schema(self):
        """Create schema."""
        if self.is_step_complete("schema"):
            logger.info("Schema already created (from checkpoint)")
            return

        logger.info("Creating v2.0 schema...")
        statements = get_create_statements()
        for stmt in statements:
            self.conn.execute(stmt)
        logger.info(f"Schema created: {len(statements)} statements")
        self.save_checkpoint("schema")

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
            step = f"node_{table}"
            if self.is_step_complete(step):
                logger.info(f"  {table} already loaded (from checkpoint)")
                continue

            csv_file = self.nodes_dir / f"{table}.csv"
            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue

            logger.info(f"  Loading {table}...")

            # Use COPY FROM with PARALLEL=false for memory efficiency
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false)
            """)

            # Count rows
            result = self.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count:,} rows")

            self.save_checkpoint(step)

    def load_relationships(self):
        """Load all relationship CSV files."""
        logger.info("Loading relationship tables...")

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
            step = f"rel_{table}"
            if self.is_step_complete(step):
                logger.info(f"  {table} already loaded (from checkpoint)")
                continue

            csv_file = self.rels_dir / f"{table}.csv"
            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue

            logger.info(f"  Loading {table}...")

            # Use COPY FROM with PARALLEL=false for memory efficiency
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false)
            """)

            # Count relationships
            result = self.conn.execute(f"MATCH ()-[r:{table}]->() RETURN count(r)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count:,} relationships")

            self.save_checkpoint(step)

    def create_has_root_relationships_batched(self):
        """Create HAS_ROOT relationships via direct MATCH query."""
        if self.is_step_complete("has_root"):
            logger.info("HAS_ROOT relationships already created (from checkpoint)")
            return

        logger.info("Creating HAS_ROOT relationships (direct MATCH)...")

        # Get total Vorto count
        result = self.conn.execute("MATCH (v:Vorto) RETURN count(v)")
        total_vorto = result.get_next()[0]
        logger.info(f"  Total Vorto nodes: {total_vorto:,}")

        logger.info("  Executing MATCH query (this may take 20-30 minutes)...")

        # Create all relationships in one query
        # Kuzu handles this efficiently - matches on indexed properties
        self.conn.execute("""
            MATCH (v:Vorto), (r:Root)
            WHERE v.radiko = r.root
            CREATE (v)-[:HAS_ROOT {is_primary: true, position: 0}]->(r)
        """)

        # Count final relationships
        result = self.conn.execute("MATCH ()-[r:HAS_ROOT]->() RETURN count(r)")
        created = result.get_next()[0]
        logger.info(f"  Created {created:,} HAS_ROOT relationships")

        self.save_checkpoint("has_root")

    def get_stats(self):
        """Get loading statistics."""
        logger.info("\n=== Loading Statistics ===")

        stats = [
            ('SourceCollection', 'SourceCollection'),
            ('Document', 'Document'),
            ('Sentence', 'Sentence'),
            ('AST', 'AST'),
            ('Frazo', 'Frazo'),
            ('Vortgrupo', 'Vortgrupo'),
            ('Vorto', 'Vorto'),
            ('Root', 'Root'),
            ('HAS_ROOT relationships', 'HAS_ROOT')
        ]

        for label, node_type in stats:
            if node_type in ['HAS_ROOT']:
                result = self.conn.execute(f"MATCH ()-[r:{node_type}]->() RETURN count(r)")
            else:
                result = self.conn.execute(f"MATCH (n:{node_type}) RETURN count(n)")
            count = result.get_next()[0]
            logger.info(f"  {label}: {count:,}")

    def load(self):
        """Load all CSV files with batching and checkpointing."""
        self.connect()
        self.create_schema()
        self.load_nodes()
        self.load_relationships()
        self.create_has_root_relationships_batched()
        self.get_stats()

        # Clean up checkpoint file on success
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("\n✓ Loading complete. Checkpoint file removed.")


def main():
    parser = argparse.ArgumentParser(description='Load CSV files into Kuzu (memory-efficient)')
    parser.add_argument('--csvs', type=Path, required=True, help='Directory with CSV files')
    parser.add_argument('--output', type=Path, required=True, help='Output Kuzu database path')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (delete existing DB)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for root processing')

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
    loader = BatchedCSVLoader(args.output, args.csvs, args.batch_size)

    # Delete checkpoint if starting fresh
    if args.fresh and loader.checkpoint_file.exists():
        logger.info(f"Removing checkpoint file (fresh start)")
        loader.checkpoint_file.unlink()

    loader.load()


if __name__ == '__main__':
    main()
