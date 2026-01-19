#!/usr/bin/env python3
"""
Load ConceptNet relations into existing Kuzu graph database.

Extends the Kuzu schema with:
- Concept nodes (external/language-agnostic concepts with metadata)
- ConceptNet relation edges (CN_IS_A, CN_SYNONYM, CN_ANTONYM, etc.)

Imports ALL useful semantic relations:
- IsA, Synonym, Antonym, HasContext, MannerOf, PartOf
- Both Esperanto→Esperanto and Esperanto→External
- Properly labels Esperanto roots vs external concepts

Skips noise:
- ExternalURL, FormOf, RelatedTo (inflections), Etymology

Usage:
    python scripts/load_conceptnet_to_kuzu.py
    python scripts/load_conceptnet_to_kuzu.py --fresh  # Rebuild ConceptNet data
"""

import argparse
import csv
import gzip
import json
import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kuzu
except ImportError:
    print("Error: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

# Import parser to extract roots from ConceptNet words
try:
    from klareco.parser import parse_word
except ImportError:
    print("Error: Cannot import klareco.parser")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ConceptNet relation types to import (semantic only, skip noise)
SEMANTIC_RELATIONS = {
    '/r/IsA',         # X is a type of Y (abatejo IsA preĝejo)
    '/r/InstanceOf',  # X is an instance of Y
    '/r/Synonym',     # X is synonym of Y (abdomeno Synonym ventro)
    '/r/Antonym',     # X is opposite of Y (absolute Antonym relative)
    '/r/HasContext',  # X used in context Y (abako HasContext mathematics)
    '/r/MannerOf',    # X is a manner of Y (dormi MannerOf ripozi)
    '/r/PartOf',      # X is part of Y (adreso PartOf letero)
    '/r/SimilarTo',   # X is similar to Y
    '/r/DistinctFrom',# X is distinct from Y
}


class ConceptNetKuzuLoader:
    """Load ConceptNet data into Kuzu graph database."""

    def __init__(
        self,
        kuzu_db_path: Path,
        conceptnet_csv_path: Path,
        temp_dir: Path
    ):
        self.kuzu_db_path = Path(kuzu_db_path)
        self.conceptnet_csv_path = Path(conceptnet_csv_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.db: kuzu.Database = None
        self.conn: kuzu.Connection = None

        # Progress tracking
        self.progress_file = self.kuzu_db_path.parent / "conceptnet_progress.json"
        self.progress: Dict = {}

        self.stats = {
            'cn_is_a': 0,
            'cn_synonym': 0,
            'cn_antonym': 0,
            'cn_has_context': 0,
            'cn_manner_of': 0,
            'cn_part_of': 0,
            'cn_similar_to': 0,
            'cn_distinct_from': 0,
            'external_concepts': 0,
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

    def _extract_word(self, uri: str) -> Tuple[str, str]:
        """Extract word and language from ConceptNet URI.

        Args:
            uri: ConceptNet URI like /c/eo/hundo/n

        Returns:
            (word, language) tuple like ('hundo', 'eo')
        """
        parts = uri.split('/')
        if len(parts) >= 4 and parts[1] == 'c':
            lang = parts[2]
            word = parts[3]
            return (word, lang)
        return ('', '')

    def _extract_root(self, word: str) -> str:
        """Extract Esperanto root from full word using parser.

        Handles multi-word phrases by splitting on underscore and parsing components.

        Args:
            word: Full Esperanto word (e.g., 'hundo', 'tablojn', 'manĝis')
                  or multi-word phrase (e.g., 'natria_klorido', 'artefarita_inteligenteco')

        Returns:
            Root extracted by parser (e.g., 'hund', 'tabl', 'manĝ')
            For multi-word phrases, returns first successfully parsed root
            Returns empty string if parsing fails
        """
        # First try parsing the word as-is
        try:
            ast = parse_word(word)
            root = ast.get('radiko')
            if root:
                return root.lower()
        except Exception:
            pass

        # If word contains underscore (multi-word phrase), try splitting
        if '_' in word:
            components = word.split('_')
            for component in components:
                if not component:  # Skip empty components
                    continue
                try:
                    ast = parse_word(component)
                    root = ast.get('radiko')
                    if root:
                        return root.lower()  # Return first successful parse
                except Exception:
                    continue

        # Fallback: Check for country name patterns (proper nouns)
        # This handles words like "francio", "grekio" where root exists but parser doesn't recognize suffix
        if hasattr(self, 'existing_roots'):
            # Country suffix -io (francio → franc)
            if word.endswith('io') and len(word) > 2:
                stem = word[:-2]
                if stem in self.existing_roots:
                    return stem

            # Country suffix -ujo (francujo → franc)
            if word.endswith('ujo') and len(word) > 3:
                stem = word[:-3]
                if stem in self.existing_roots:
                    return stem

            # Country suffix -lando (skotlando → skot)
            if word.endswith('lando') and len(word) > 5:
                stem = word[:-5]
                if stem in self.existing_roots:
                    return stem

        return ''

    def _classify_concept_type(self, uri: str, relation: str) -> str:
        """Classify external concept type based on context.

        Args:
            uri: ConceptNet URI
            relation: The relation type

        Returns:
            Concept type: 'domain', 'abstract', 'linguistic', 'proper_noun', 'other'
        """
        word, lang = self._extract_word(uri)

        # Domain labels (from HasContext)
        if relation == '/r/HasContext':
            return 'domain'

        # Linguistic/grammatical concepts
        linguistic_patterns = ['modo', 'tempo', 'kazo', 'numero', 'grammar', 'verb', 'noun']
        if any(pattern in word.lower() for pattern in linguistic_patterns):
            return 'linguistic'

        # Abstract concepts (numbers, etc.)
        abstract_patterns = ['numero', 'number', 'concept', 'idea']
        if any(pattern in word.lower() for pattern in abstract_patterns):
            return 'abstract'

        # Proper nouns (capitalized, has /n/wp/ for Wikipedia)
        if '/n/wp/' in uri or (word and word[0].isupper()):
            return 'proper_noun'

        return 'other'

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
        """Extend Kuzu schema with ConceptNet nodes and edges."""
        if self.progress.get('schema_extended'):
            logger.info("Schema already extended, skipping...")
            return

        logger.info("Extending Kuzu schema for ConceptNet data...")

        # Create Concept node table for external/language-agnostic concepts
        # Esperanto words already exist as Root nodes
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Concept (
                uri STRING,
                label STRING,
                language STRING,
                concept_type STRING,
                is_external BOOLEAN DEFAULT true,
                PRIMARY KEY (uri)
            )
        """)

        # Create ConceptNet relation edge tables
        # CN_ prefix distinguishes ConceptNet data from ReVo/curated data

        # All edge types support both Root→Root and Root→Concept
        # We use separate tables for type safety and query efficiency

        # CN_IS_A: X is a type of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_IS_A (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_IS_A_EXT (
                FROM Root TO Concept,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_SYNONYM: X is synonym of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_SYNONYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_SYNONYM_EXT (
                FROM Root TO Concept,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_ANTONYM: X is opposite of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_ANTONYM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_ANTONYM_EXT (
                FROM Root TO Concept,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_HAS_CONTEXT: X used in context/domain Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_HAS_CONTEXT (
                FROM Root TO Concept,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_MANNER_OF: X is a manner of doing Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_MANNER_OF (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_PART_OF: X is part of Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_PART_OF (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_PART_OF_EXT (
                FROM Root TO Concept,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_SIMILAR_TO: X is similar to Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_SIMILAR_TO (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        # CN_DISTINCT_FROM: X is distinct from Y
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CN_DISTINCT_FROM (
                FROM Root TO Root,
                weight DOUBLE DEFAULT 1.0,
                source STRING DEFAULT 'conceptnet'
            )
        """)

        logger.info("  Schema extended successfully")

        self.progress['schema_extended'] = True
        self._save_progress()

    def extract_conceptnet_to_csvs(self):
        """Extract semantic relations from ConceptNet CSV and write to CSVs."""
        if self.progress.get('csvs_created'):
            logger.info("CSVs already created, skipping extraction...")
            return

        logger.info("Extracting semantic relations from ConceptNet CSV...")
        logger.info(f"  Source: {self.conceptnet_csv_path}")

        # Get existing Esperanto roots from Kuzu
        logger.info("  Loading existing roots from Kuzu...")
        self.existing_roots: Set[str] = set()
        result = self.conn.execute("MATCH (r:Root) RETURN r.root")
        while result.has_next():
            self.existing_roots.add(result.get_next()[0])
        logger.info(f"  Found {len(self.existing_roots):,} existing Esperanto roots")

        # Open CSV writers
        csv_files = {}
        csv_writers = {}

        file_headers = {
            'concepts.csv': ['uri', 'label', 'language', 'concept_type', 'is_external'],
            'cn_is_a.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_is_a_ext.csv': ['root', 'concept', 'weight', 'source'],
            'cn_synonym.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_synonym_ext.csv': ['root', 'concept', 'weight', 'source'],
            'cn_antonym.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_antonym_ext.csv': ['root', 'concept', 'weight', 'source'],
            'cn_has_context.csv': ['root', 'concept', 'weight', 'source'],
            'cn_manner_of.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_part_of.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_part_of_ext.csv': ['root', 'concept', 'weight', 'source'],
            'cn_similar_to.csv': ['root1', 'root2', 'weight', 'source'],
            'cn_distinct_from.csv': ['root1', 'root2', 'weight', 'source'],
        }

        for filename, header in file_headers.items():
            filepath = self.temp_dir / filename
            csv_files[filename] = open(filepath, 'w', newline='', encoding='utf-8')
            csv_writers[filename] = csv.writer(csv_files[filename])
            csv_writers[filename].writerow(header)

        # Track concepts seen
        concepts_seen: Dict[str, Tuple[str, str, str]] = {}  # uri → (label, lang, type)

        # Parse ConceptNet CSV
        count = 0
        eo_relations = 0
        semantic_relations = 0

        with gzip.open(self.conceptnet_csv_path, 'rt', encoding='utf-8') as f:
            for line in f:
                count += 1
                if count % 1000000 == 0:
                    logger.info(f"  Processed {count:,} lines, {semantic_relations:,} semantic relations")

                # Parse TSV line
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                uri, relation, start, end, metadata_json = parts[:5]

                # Only keep semantic relations
                if relation not in SEMANTIC_RELATIONS:
                    continue

                # Only keep Esperanto relations (at least one side is Esperanto)
                if '/c/eo/' not in start and '/c/eo/' not in end:
                    continue

                eo_relations += 1

                # Parse metadata
                try:
                    metadata = json.loads(metadata_json)
                    weight = metadata.get('weight', 1.0)
                except json.JSONDecodeError:
                    weight = 1.0

                # Extract words from URIs
                start_word, start_lang = self._extract_word(start)
                end_word, end_lang = self._extract_word(end)

                if not start_word or not end_word:
                    continue

                # For Esperanto words, extract ROOTS using parser
                # This ensures matching with corpus (which has roots, not full words)
                start_root = None
                end_root = None

                if start_lang == 'eo':
                    start_root = self._extract_root(start_word)
                    if not start_root:
                        # Parser failed, skip this relation
                        continue

                if end_lang == 'eo':
                    end_root = self._extract_root(end_word)
                    if not end_root:
                        # Parser failed, skip this relation
                        continue

                # For external words, just normalize to lowercase
                if start_lang != 'eo':
                    start_word = start_word.lower()
                if end_lang != 'eo':
                    end_word = end_word.lower()

                # Map relation to CSV file and stats key
                rel_mapping = {
                    '/r/IsA': 'cn_is_a',
                    '/r/InstanceOf': 'cn_is_a',  # Treat same as IsA
                    '/r/Synonym': 'cn_synonym',
                    '/r/Antonym': 'cn_antonym',
                    '/r/HasContext': 'cn_has_context',
                    '/r/MannerOf': 'cn_manner_of',
                    '/r/PartOf': 'cn_part_of',
                    '/r/SimilarTo': 'cn_similar_to',
                    '/r/DistinctFrom': 'cn_distinct_from',
                }

                rel_key = rel_mapping.get(relation)
                if not rel_key:
                    continue

                # Case 1: Both are Esperanto roots in our index
                if start_lang == 'eo' and end_lang == 'eo':
                    # Exclude self-referential relations (e.g., "pluv IsA pluv")
                    if start_root in self.existing_roots and end_root in self.existing_roots and start_root != end_root:
                        csv_writers[f'{rel_key}.csv'].writerow([start_root, end_root, weight, 'conceptnet'])
                        self.stats[rel_key] += 1
                        semantic_relations += 1

                # Case 2: Esperanto root → external concept
                elif start_lang == 'eo' and start_root in self.existing_roots:
                    # Add external concept if not seen
                    if end not in concepts_seen:
                        concept_type = self._classify_concept_type(end, relation)
                        concepts_seen[end] = (end_word, end_lang, concept_type)
                        csv_writers['concepts.csv'].writerow([end, end_word, end_lang, concept_type, 'true'])

                    # Add edge (some relations only support eo→eo, others support eo→ext)
                    if rel_key in ['cn_is_a', 'cn_synonym', 'cn_antonym', 'cn_part_of']:
                        csv_writers[f'{rel_key}_ext.csv'].writerow([start_root, end, weight, 'conceptnet'])
                        self.stats[rel_key] += 1
                        semantic_relations += 1
                    elif rel_key == 'cn_has_context':
                        csv_writers[f'{rel_key}.csv'].writerow([start_root, end, weight, 'conceptnet'])
                        self.stats[rel_key] += 1
                        semantic_relations += 1

                # Case 3: External concept → Esperanto root (reverse direction)
                # Skip for now (could add if needed for completeness)

        # Close all files
        for f in csv_files.values():
            f.close()

        self.stats['external_concepts'] = len(concepts_seen)

        logger.info(f"  Processed {count:,} total lines")
        logger.info(f"  Found {eo_relations:,} Esperanto relations")
        logger.info(f"  Extracted {semantic_relations:,} semantic relations")
        logger.info(f"  Unique external concepts: {len(concepts_seen):,}")

        self.progress['csvs_created'] = True
        self._save_progress()

    def bulk_load_csvs(self):
        """Bulk load CSVs into Kuzu."""
        if self.progress.get('data_loaded'):
            logger.info("Data already loaded, skipping...")
            return

        logger.info("Bulk loading ConceptNet data into Kuzu...")

        # Load Concept nodes first
        concept_csv = self.temp_dir / 'concepts.csv'
        if concept_csv.exists() and concept_csv.stat().st_size > 100:
            logger.info(f"  Loading Concept nodes from concepts.csv")
            try:
                self.conn.execute(f"COPY Concept FROM '{concept_csv}' (header=true)")
                logger.info("    Done")
            except Exception as e:
                logger.error(f"    Error: {e}")

        # Load edge tables
        edge_files = [
            ('CN_IS_A', 'cn_is_a.csv'),
            ('CN_IS_A_EXT', 'cn_is_a_ext.csv'),
            ('CN_SYNONYM', 'cn_synonym.csv'),
            ('CN_SYNONYM_EXT', 'cn_synonym_ext.csv'),
            ('CN_ANTONYM', 'cn_antonym.csv'),
            ('CN_ANTONYM_EXT', 'cn_antonym_ext.csv'),
            ('CN_HAS_CONTEXT', 'cn_has_context.csv'),
            ('CN_MANNER_OF', 'cn_manner_of.csv'),
            ('CN_PART_OF', 'cn_part_of.csv'),
            ('CN_PART_OF_EXT', 'cn_part_of_ext.csv'),
            ('CN_SIMILAR_TO', 'cn_similar_to.csv'),
            ('CN_DISTINCT_FROM', 'cn_distinct_from.csv'),
        ]

        for table_name, csv_name in edge_files:
            csv_path = self.temp_dir / csv_name
            if csv_path.exists() and csv_path.stat().st_size > 100:
                logger.info(f"  Loading {table_name} from {csv_name}")
                try:
                    self.conn.execute(f"COPY {table_name} FROM '{csv_path}' (header=true)")
                    logger.info("    Done")
                except Exception as e:
                    logger.error(f"    Error: {e}")

        self.progress['data_loaded'] = True
        self._save_progress()

    def verify_counts(self):
        """Verify final counts and show statistics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("CONCEPTNET DATA LOADED")
        logger.info("=" * 70)

        # Count concepts
        try:
            result = self.conn.execute("MATCH (c:Concept) RETURN count(c)")
            concept_count = result.get_next()[0]
            logger.info(f"\nConcept nodes: {concept_count:,}")

            # Count by type
            result = self.conn.execute("""
                MATCH (c:Concept)
                RETURN c.concept_type, count(*)
                ORDER BY count(*) DESC
            """)
            logger.info("  By type:")
            while result.has_next():
                row = result.get_next()
                logger.info(f"    {row[0]}: {row[1]:,}")
        except Exception as e:
            logger.info(f"  Concept nodes: 0 ({e})")

        # Count edges
        logger.info("\nRelation edges:")
        edge_types = [
            'CN_IS_A', 'CN_IS_A_EXT',
            'CN_SYNONYM', 'CN_SYNONYM_EXT',
            'CN_ANTONYM', 'CN_ANTONYM_EXT',
            'CN_HAS_CONTEXT',
            'CN_MANNER_OF',
            'CN_PART_OF', 'CN_PART_OF_EXT',
            'CN_SIMILAR_TO',
            'CN_DISTINCT_FROM',
        ]

        for edge_type in edge_types:
            try:
                result = self.conn.execute(f"MATCH ()-[e:{edge_type}]->() RETURN count(e)")
                count = result.get_next()[0]
                if count > 0:
                    logger.info(f"  {edge_type}: {count:,}")
            except:
                pass

    def finalize(self):
        """Clean up and print summary."""
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()

        # Clean up temp CSVs
        if self.temp_dir.exists():
            logger.info("\nCleaning up temporary CSV files...")
            shutil.rmtree(self.temp_dir)

        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ ConceptNet data successfully loaded into Kuzu!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Query examples:")
        logger.info("  # Find what 'hundo' is")
        logger.info("  MATCH (r:Root {root: 'hundo'})-[:CN_IS_A]->(t) RETURN t.root")
        logger.info("")
        logger.info("  # Find all synonyms of 'dormi'")
        logger.info("  MATCH (r:Root {root: 'dormi'})-[:CN_SYNONYM]->(s) RETURN s.root")
        logger.info("")
        logger.info("  # Find domain context for 'abako'")
        logger.info("  MATCH (r:Root {root: 'abako'})-[:CN_HAS_CONTEXT]->(c) RETURN c.label")
        logger.info("")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None


def main():
    parser = argparse.ArgumentParser(
        description='Load ConceptNet data into Kuzu graph database'
    )
    parser.add_argument(
        '--kuzu-db',
        type=Path,
        default=Path('data/indexes/kuzu_index/kuzu.db'),
        help='Path to Kuzu database'
    )
    parser.add_argument(
        '--conceptnet-csv',
        type=Path,
        default=Path('data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz'),
        help='Path to ConceptNet CSV file'
    )
    parser.add_argument(
        '--temp-dir',
        type=Path,
        default=Path('data/indexes/kuzu_index/temp_conceptnet'),
        help='Temporary directory for CSV files'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh, rebuild ConceptNet data'
    )

    args = parser.parse_args()

    if not args.conceptnet_csv.exists():
        logger.error(f"ConceptNet CSV not found: {args.conceptnet_csv}")
        logger.error("Download with: wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        sys.exit(1)

    loader = ConceptNetKuzuLoader(
        kuzu_db_path=args.kuzu_db,
        conceptnet_csv_path=args.conceptnet_csv,
        temp_dir=args.temp_dir
    )

    try:
        loader.connect()

        if args.fresh:
            logger.info("Fresh start requested, clearing ConceptNet progress...")
            if loader.progress_file.exists():
                loader.progress_file.unlink()
            if loader.temp_dir.exists():
                shutil.rmtree(loader.temp_dir)
            loader.temp_dir.mkdir(parents=True, exist_ok=True)
            loader.progress = {}

        loader.extend_schema()
        loader.extract_conceptnet_to_csvs()
        loader.bulk_load_csvs()
        loader.verify_counts()
        loader.finalize()

    finally:
        loader.close()


if __name__ == '__main__':
    main()
