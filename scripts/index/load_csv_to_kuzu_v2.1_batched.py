#!/usr/bin/env python3
"""
Memory-efficient batch loader for v2.1 Kuzu database (Pure Esperanto).

Optimizations:
- Batched relationship creation (prevents OOM on large graphs)
- Progress checkpointing (can resume if interrupted)
- Memory-conscious COPY operations
- Incremental validation

Usage:
    python scripts/index/load_csv_to_kuzu_v2.1_batched.py --csvs data/csv_export_v2.1_full \
                                                     --output data/indexes/v2.1_kuzu_index_full \
                                                     --fresh
"""

import argparse
import logging
import json
import sys
import gc
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import kuzu
except ImportError:
    print("ERROR: kuzu not installed. Run: pip install kuzu")
    exit(1)

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    exit(1)

from klareco.schema.kuzu_ast_schema_v2_1 import get_create_statements

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
        self.process = psutil.Process()

    def get_memory_usage(self):
        """Get current memory usage stats."""
        mem = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        return {
            'rss_mb': mem.rss / 1024 / 1024,  # Resident Set Size
            'percent': virtual_mem.percent,    # System-wide memory %
            'available_gb': virtual_mem.available / 1024 / 1024 / 1024
        }

    def check_memory_pressure(self, batch_num: int, total_batches: int):
        """Check memory usage and warn/GC if high."""
        mem = self.get_memory_usage()

        # Warn if system memory is getting high
        if mem['percent'] > 85:
            logger.warning(f"  ⚠️  High memory usage: {mem['percent']:.1f}% system, {mem['rss_mb']:.0f}MB process")
            logger.warning(f"     Available: {mem['available_gb']:.1f}GB - Running GC...")
            gc.collect()
            return True
        elif mem['percent'] > 75 and batch_num % 100 == 0:
            # Log every 100 batches if memory is elevated
            logger.info(f"  Memory: {mem['percent']:.1f}% system, {mem['rss_mb']:.0f}MB process, {mem['available_gb']:.1f}GB available")

        return False

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

        logger.info("Creating v2.1 schema (Pure Esperanto)...")
        statements = get_create_statements()
        for stmt in statements:
            self.conn.execute(stmt)
        logger.info(f"Schema created: {len(statements)} statements")
        self.save_checkpoint("schema")

    def load_nodes(self):
        """Load all node CSV files (v2.1 Pure Esperanto naming)."""
        logger.info("Loading node tables...")

        node_tables = [
            'Fontaro',
            'Dokumento',
            'Sekcio',
            'Paragrafo',
            'Frazoteksto',
            'AST',
            'Frazo',
            'Vortgrupo',
            'Vorto',
            'Radiko'
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
            # Explicitly set QUOTE and ESCAPE for proper CSV parsing with embedded quotes
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false, QUOTE='"', ESCAPE='"')
            """)

            # Count rows
            result = self.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count:,} rows")

            self.save_checkpoint(step)

    def load_relationships(self):
        """Load all relationship CSV files (v2.1 Pure Esperanto naming)."""
        logger.info("Loading relationship tables...")

        rel_tables = [
            # Hierarchy relationships
            ('EN_FONTARO', 'Dokumento', 'Fontaro'),
            ('EN_DOKUMENTO', 'Sekcio', 'Dokumento'),
            ('EN_SEKCIO', 'Paragrafo', 'Sekcio'),
            ('EN_PARAGRAFO', 'Frazoteksto', 'Paragrafo'),
            ('GEPATRA_SEKCIO', 'Sekcio', 'Sekcio'),
            ('SEKVA_SEKCIO', 'Sekcio', 'Sekcio'),
            ('SEKVA_PARAGRAFO', 'Paragrafo', 'Paragrafo'),
            ('SEKVA_FRAZOTEKSTO', 'Frazoteksto', 'Frazoteksto'),
            # AST relationships
            ('FRAZOTEKSTO_HAVAS_AST', 'Frazoteksto', 'AST'),
            ('AST_HAVAS_FRAZON', 'AST', 'Frazo'),
            ('HAVAS_SUBJEKTON_VORTGRUPO', 'Frazo', 'Vortgrupo'),
            ('HAVAS_SUBJEKTON_VORTO', 'Frazo', 'Vorto'),
            ('HAVAS_VERBON', 'Frazo', 'Vorto'),
            ('HAVAS_OBJEKTON_VORTGRUPO', 'Frazo', 'Vortgrupo'),
            ('HAVAS_OBJEKTON_VORTO', 'Frazo', 'Vorto'),
            ('HAVAS_ALIAJN', 'Frazo', 'Vorto'),
            ('HAVAS_KERNON', 'Vortgrupo', 'Vorto'),
            ('HAVAS_PRISKRIBON', 'Vortgrupo', 'Vorto'),
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
            # Explicitly set QUOTE and ESCAPE for proper CSV parsing with embedded quotes
            self.conn.execute(f"""
                COPY {table} FROM '{csv_file}' (HEADER=true, PARALLEL=false, QUOTE='"', ESCAPE='"')
            """)

            # Count relationships
            result = self.conn.execute(f"MATCH ()-[r:{table}]->() RETURN count(r)")
            count = result.get_next()[0]
            logger.info(f"    Loaded {count:,} relationships")

            self.save_checkpoint(step)

    def create_havas_radikon_relationships(self):
        """Create HAVAS_RADIKON relationships by batching Vorto IDs (memory-efficient)."""
        if self.is_step_complete("havas_radikon"):
            logger.info("HAVAS_RADIKON relationships already created (from checkpoint)")
            return

        logger.info("Creating HAVAS_RADIKON relationships (Vorto ID batching)...")

        # Log initial memory state
        mem = self.get_memory_usage()
        logger.info(f"  Starting memory: {mem['percent']:.1f}% system, {mem['rss_mb']:.0f}MB process, {mem['available_gb']:.1f}GB available")

        # Get Vorto ID range
        logger.info("  Getting Vorto ID range...")
        result = self.conn.execute("MATCH (v:Vorto) RETURN min(v.id), max(v.id), count(v)")
        min_id, max_id, total_vorto = result.get_next()
        logger.info(f"  Vorto ID range: {min_id:,} to {max_id:,} ({total_vorto:,} nodes)")

        # Process in batches of 1M Vorto IDs at a time
        batch_size = 1_000_000  # Process 1M Vorto nodes per batch
        total_batches = (max_id - min_id + batch_size) // batch_size

        # Check for resume from partial completion
        checkpoint = self.load_checkpoint()
        completed_batches = checkpoint.get("havas_radikon_batches", 0)

        if completed_batches > 0:
            logger.info(f"  Resuming from batch {completed_batches + 1}/{total_batches}")
        else:
            logger.info(f"  Processing {total_batches} batches of {batch_size:,} Vorto IDs")
            logger.info(f"  Estimated time: ~60-90 minutes")

        last_logged_percent = -1

        for batch_num in range(completed_batches, total_batches):
            # Check memory pressure before processing batch
            self.check_memory_pressure(batch_num, total_batches)

            # Calculate ID range for this batch
            start_id = min_id + (batch_num * batch_size)
            end_id = min(start_id + batch_size - 1, max_id)

            # Process this batch - scan only Vorto IDs in range
            self.conn.execute(f"""
                MATCH (v:Vorto), (r:Radiko)
                WHERE v.id >= {start_id}
                  AND v.id <= {end_id}
                  AND v.radiko = r.radiko
                CREATE (v)-[:HAVAS_RADIKON {{estas_ĉefa: true, pozicio: 0}}]->(r)
            """)

            # Show progress on same line (non-scrolling)
            percent = 100 * (batch_num + 1) // total_batches
            mem = self.get_memory_usage()
            print(f"\r  Progress: {batch_num + 1}/{total_batches} batches ({percent}%) - Mem: {mem['percent']:.0f}%", end='', flush=True)

            # Log to file every 5% (less verbose)
            if percent >= last_logged_percent + 5 or (batch_num + 1) == total_batches:
                logger.info(f"  Processed batch {batch_num + 1}/{total_batches} ({percent}%) - IDs {start_id:,}-{end_id:,} - Memory: {mem['percent']:.1f}%")
                last_logged_percent = percent

            # Save checkpoint after each batch
            checkpoint = self.load_checkpoint()
            checkpoint["havas_radikon_batches"] = batch_num + 1
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        # Add newline after progress line
        print()

        # Count final relationships
        result = self.conn.execute("MATCH ()-[r:HAVAS_RADIKON]->() RETURN count(r)")
        created = result.get_next()[0]
        logger.info(f"  Created {created:,} HAVAS_RADIKON relationships")

        # Log final memory state
        mem = self.get_memory_usage()
        logger.info(f"  Final memory: {mem['percent']:.1f}% system, {mem['rss_mb']:.0f}MB process, {mem['available_gb']:.1f}GB available")

        self.save_checkpoint("havas_radikon")

    def get_stats(self):
        """Get loading statistics (v2.1 Pure Esperanto)."""
        logger.info("\n=== Loading Statistics ===")

        stats = [
            ('Fontaro', 'Fontaro'),
            ('Dokumento', 'Dokumento'),
            ('Sekcio', 'Sekcio'),
            ('Paragrafo', 'Paragrafo'),
            ('Frazoteksto', 'Frazoteksto'),
            ('AST', 'AST'),
            ('Frazo', 'Frazo'),
            ('Vortgrupo', 'Vortgrupo'),
            ('Vorto', 'Vorto'),
            ('Radiko', 'Radiko'),
            ('HAVAS_RADIKON relationships', 'HAVAS_RADIKON')
        ]

        for label, node_type in stats:
            if node_type in ['HAVAS_RADIKON']:
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
        self.create_havas_radikon_relationships()
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
