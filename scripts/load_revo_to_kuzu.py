#!/usr/bin/env python3
"""
Load ReVo semantic relations into Kuzu graph database.

Extends the Kuzu schema with ReVo-specific semantic relations:
- REVO_SYNONYM (Root → Root)
- REVO_ANTONYM (Root → Root)
- REVO_HYPERNYM (Root → Root) - A is-a B
- REVO_HYPONYM (Root → Root) - A has-subtype B
- REVO_PART_OF (Root → Root) - A is-part-of B

ReVo relations have higher weight (2.0) than ConceptNet (1.0)
because they are Esperanto-specific and authoritative.

Usage:
    python scripts/load_revo_to_kuzu.py
    python scripts/load_revo_to_kuzu.py --fresh  # Rebuild
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("Error: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RevoKuzuLoader:
    """Load ReVo semantic relations into Kuzu graph database."""

    def __init__(
        self,
        kuzu_db_path: Path,
        relations_json_path: Path,
        temp_dir: Path
    ):
        self.kuzu_db_path = Path(kuzu_db_path)
        self.relations_json_path = Path(relations_json_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.db: kuzu.Database = None
        self.conn: kuzu.Connection = None

        # Progress tracking
        self.progress_file = self.kuzu_db_path.parent / "revo_progress.json"
        self.progress: Dict = {}

        self.stats = {
            'revo_synonym': 0,
            'revo_antonym': 0,
            'revo_hypernym': 0,
            'revo_hyponym': 0,
            'revo_part_of': 0,
            'skipped_missing_root': 0,
        }

    def _load_progress(self) -> Dict:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {}

    def _save_progress(self):
        """Save progress to file (atomic)."""
        temp_file = self.progress_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.progress, f)
            temp_file.rename(self.progress_file)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def connect(self):
        """Connect to Kuzu database."""
        logger.info(f"Connecting to Kuzu database: {self.kuzu_db_path}")

        if not self.kuzu_db_path.exists():
            logger.error(f"Kuzu database not found: {self.kuzu_db_path}")
            logger.error("Please run scripts/index_kuzu.py first to create the database")
            sys.exit(1)

        self.db = kuzu.Database(str(self.kuzu_db_path))
        self.conn = kuzu.Connection(self.db)

        self.progress = self._load_progress()

    def extend_schema(self):
        """Extend Kuzu schema with ReVo relation tables."""
        if self.progress.get('schema_extended'):
            logger.info("Schema already extended, skipping...")
            return

        logger.info("Extending Kuzu schema with ReVo relation tables...")

        # REVO_SYNONYM: X is synonym of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS REVO_SYNONYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 2.0,
                source STRING DEFAULT 'revo'
            )
        """)

        # REVO_ANTONYM: X is opposite of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS REVO_ANTONYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 2.0,
                source STRING DEFAULT 'revo'
            )
        """)

        # REVO_HYPERNYM: X is-a Y (X is subtype of Y)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS REVO_HYPERNYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 2.0,
                source STRING DEFAULT 'revo'
            )
        """)

        # REVO_HYPONYM: X has-subtype Y (inverse of hypernym)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS REVO_HYPONYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 2.0,
                source STRING DEFAULT 'revo'
            )
        """)

        # REVO_PART_OF: X is part of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS REVO_PART_OF (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 2.0,
                source STRING DEFAULT 'revo'
            )
        """)

        logger.info("  Schema extended successfully")

        self.progress['schema_extended'] = True
        self._save_progress()

    def extract_to_csvs(self):
        """Extract ReVo relations to CSV files for bulk loading."""
        if self.progress.get('csvs_created'):
            logger.info("CSVs already created, skipping extraction...")
            return

        logger.info("Extracting ReVo relations to CSV files...")
        logger.info(f"  Source: {self.relations_json_path}")

        # Load relations JSON
        with open(self.relations_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        relations = data.get('relations', {})
        metadata = data.get('metadata', {})

        logger.info(f"  Loaded {sum(len(r) for r in relations.values()):,} relations")

        # Get existing roots from Kuzu to validate
        logger.info("  Loading existing roots from Kuzu...")
        existing_roots: Set[str] = set()
        result = self.conn.execute("MATCH (r:Root) RETURN r.root")
        while result.has_next():
            existing_roots.add(result.get_next()[0])
        logger.info(f"  Found {len(existing_roots):,} existing roots")

        # Open CSV writers
        csv_files = {}
        csv_writers = {}

        file_headers = {
            'revo_synonym.csv': ['root1', 'root2', 'weight', 'source'],
            'revo_antonym.csv': ['root1', 'root2', 'weight', 'source'],
            'revo_hypernym.csv': ['root1', 'root2', 'weight', 'source'],
            'revo_hyponym.csv': ['root1', 'root2', 'weight', 'source'],
            'revo_part_of.csv': ['root1', 'root2', 'weight', 'source'],
        }

        for filename, header in file_headers.items():
            filepath = self.temp_dir / filename
            csv_files[filename] = open(filepath, 'w', newline='', encoding='utf-8')
            csv_writers[filename] = csv.writer(csv_files[filename])
            csv_writers[filename].writerow(header)

        # Process each relation type
        for rel_type, rel_list in relations.items():
            csv_key = f'revo_{rel_type}.csv'

            for rel in rel_list:
                source = rel['source']
                target = rel['target']
                weight = rel.get('weight', 2.0)

                # Validate both roots exist in corpus
                if source not in existing_roots or target not in existing_roots:
                    self.stats['skipped_missing_root'] += 1
                    continue

                # Write to CSV
                csv_writers[csv_key].writerow([source, target, weight, 'revo'])
                self.stats[f'revo_{rel_type}'] += 1

        # Close CSV files
        for f in csv_files.values():
            f.close()

        logger.info("  CSV files created:")
        for rel_type in ['synonym', 'antonym', 'hypernym', 'hyponym', 'part_of']:
            count = self.stats[f'revo_{rel_type}']
            if count > 0:
                logger.info(f"    {rel_type}: {count:,}")

        logger.info(f"  Skipped (missing roots): {self.stats['skipped_missing_root']:,}")

        self.progress['csvs_created'] = True
        self._save_progress()

    def bulk_load_csvs(self):
        """Bulk load CSV files into Kuzu using COPY FROM."""
        if self.progress.get('data_loaded'):
            logger.info("Data already loaded, skipping...")
            return

        logger.info("")
        logger.info("Bulk loading CSVs into Kuzu...")

        # Helper function to run COPY with timing
        def copy_csv(table_name: str, csv_file: str):
            csv_path = self.temp_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"  CSV file not found: {csv_path}")
                return 0

            file_size = csv_path.stat().st_size / 1024 / 1024
            logger.info(f"  Loading {table_name} from {csv_file} ({file_size:.1f} MB)...")

            t0 = datetime.now()
            try:
                self.conn.execute(f"COPY {table_name} FROM '{csv_path}' (header=true)")
                elapsed = (datetime.now() - t0).total_seconds()
                logger.info(f"    Done in {elapsed:.1f}s")
                return 1
            except Exception as e:
                logger.error(f"    Error: {e}")
                return 0

        # Load edge tables
        copy_csv("REVO_SYNONYM", "revo_synonym.csv")
        copy_csv("REVO_ANTONYM", "revo_antonym.csv")
        copy_csv("REVO_HYPERNYM", "revo_hypernym.csv")
        copy_csv("REVO_HYPONYM", "revo_hyponym.csv")
        copy_csv("REVO_PART_OF", "revo_part_of.csv")

        logger.info("  Bulk loading complete")

        self.progress['data_loaded'] = True
        self._save_progress()

    def verify_index(self):
        """Verify loaded data."""
        logger.info("")
        logger.info("Verifying loaded data...")

        edge_types = [
            'REVO_SYNONYM',
            'REVO_ANTONYM',
            'REVO_HYPERNYM',
            'REVO_HYPONYM',
            'REVO_PART_OF',
        ]

        logger.info("")
        logger.info("ReVo relation edges:")
        total_edges = 0
        for edge_type in edge_types:
            try:
                result = self.conn.execute(f"MATCH ()-[e:{edge_type}]->() RETURN count(e)")
                count = result.get_next()[0]
                if count > 0:
                    logger.info(f"  {edge_type}: {count:,}")
                    total_edges += count
            except:
                pass

        logger.info(f"  Total ReVo edges: {total_edges:,}")

    def finalize(self):
        """Clean up and print summary."""
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()

        # Clean up temp CSVs
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        logger.info("")
        logger.info("=" * 70)
        logger.info("REVO LOADING COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("ReVo relations loaded into Kuzu!")
        logger.info("")
        logger.info("Query examples:")
        logger.info("")
        logger.info("  # Find synonyms of 'dormi'")
        logger.info("  MATCH (r:Root {root: 'dormi'})-[:REVO_SYNONYM]->(s) RETURN s.root")
        logger.info("")
        logger.info("  # Find what 'hundo' is-a (hypernyms)")
        logger.info("  MATCH (r:Root {root: 'hundo'})-[:REVO_HYPERNYM]->(h) RETURN h.root")
        logger.info("")
        logger.info("  # Find antonyms of 'bona'")
        logger.info("  MATCH (r:Root {root: 'bona'})-[:REVO_ANTONYM]->(a) RETURN a.root")
        logger.info("")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None


def main():
    parser = argparse.ArgumentParser(
        description='Load ReVo semantic relations into Kuzu graph database'
    )
    parser.add_argument(
        '--kuzu-db',
        type=Path,
        default=Path('data/indexes/kuzu_index/kuzu.db'),
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--relations',
        type=Path,
        default=Path('data/raw/eo/dictionaries/revo/revo_semantic_relations.json'),
        help='Path to extracted ReVo relations JSON'
    )
    parser.add_argument(
        '--temp-dir',
        type=Path,
        default=Path('data/indexes/kuzu_index/temp_revo'),
        help='Temporary directory for CSV files'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, rebuild ReVo data'
    )

    args = parser.parse_args()

    if not args.relations.exists():
        logger.error(f"ReVo relations file not found: {args.relations}")
        logger.error("Please run scripts/extract_revo_semantic_relations.py first")
        sys.exit(1)

    loader = RevoKuzuLoader(
        kuzu_db_path=args.kuzu_db,
        relations_json_path=args.relations,
        temp_dir=args.temp_dir
    )

    try:
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("LOADING REVO RELATIONS INTO KUZU")
        logger.info("=" * 70)
        logger.info("")

        loader.connect()

        if args.fresh:
            logger.info("Fresh start requested, clearing progress...")
            loader.progress = {}
            loader._save_progress()

        loader.extend_schema()
        loader.extract_to_csvs()
        loader.bulk_load_csvs()
        loader.verify_index()
        loader.finalize()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total time: {elapsed:.1f}s")

        return 0

    except Exception as e:
        logger.error(f"Loading failed: {e}", exc_info=True)
        return 1
    finally:
        loader.close()


if __name__ == '__main__':
    sys.exit(main())
