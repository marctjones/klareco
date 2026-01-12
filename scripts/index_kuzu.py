#!/usr/bin/env python3
"""
Build Kuzu Graph Index from Unified Corpus.

This script builds a unified graph database containing:
- Phase 1: Inverted index (Root nodes, Sentence nodes, HAS_ROOT edges)
- Phase 2: Semantic relations (IS_SYNONYM, IS_HYPERNYM edges)
- Phase 3: Sentence adjacency (NEXT_SENTENCE, IN_DOCUMENT edges)

Performance: Uses CSV bulk loading (COPY FROM) for 100-1000x speedup.
Memory-efficient: Kuzu is disk-based like SQLite.
Restartable: Tracks progress and can resume from interruption.

Usage:
    python scripts/build_kuzu_index.py
    python scripts/build_kuzu_index.py --limit 10000  # Test with subset
    python scripts/build_kuzu_index.py --fresh        # Start over

Input:
    data/corpus/unified_corpus.jsonl (has full ASTs)
    data/raw/eo/dictionaries/revo/revo_semantic_relations.json

Output:
    data/indexes/kuzu_index/
        kuzu.db/             - Kuzu database directory
        documents.jsonl      - Document texts for retrieval
        doc_offsets.npy      - Byte offsets for O(1) access
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

try:
    import kuzu
except ImportError:
    print("Error: kuzu not installed. Run: pip install kuzu")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# Skip these word types (function words handled by AST, not embeddings)
SKIP_VORTSPECO = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}


class KuzuIndexBuilder:
    """
    Builds a Kuzu graph index from unified corpus using CSV bulk loading.

    Schema:
        NODES:
        - Root(root STRING PRIMARY KEY, doc_freq INT64, total_freq INT64)
        - Sentence(id INT64 PRIMARY KEY, text STRING, doc_id INT64, sent_idx INT64)
        - Document(id INT64 PRIMARY KEY, source_name STRING, source_type STRING)

        EDGES:
        - HAS_ROOT(FROM Sentence TO Root, role STRING, grammar STRING)
        - IS_SYNONYM(FROM Root TO Root)
        - IS_HYPERNYM(FROM Root TO Root)  # child → parent (more general)
        - IS_ANTONYM(FROM Root TO Root)
        - NEXT_SENTENCE(FROM Sentence TO Sentence)
        - IN_DOCUMENT(FROM Sentence TO Document)
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.output_path / "kuzu.db"
        self.db: Optional[kuzu.Database] = None
        self.conn: Optional[kuzu.Connection] = None

        # Temp directory for CSV files
        self.temp_dir = self.output_path / "temp_csv"

        # Progress tracking (stored in a JSON file for restartability)
        self.progress_file = self.output_path / "build_progress.json"
        self.progress: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            'sentences': 0,
            'documents': 0,
            'roots': 0,
            'has_root_edges': 0,
            'synonym_edges': 0,
            'hypernym_edges': 0,
            'antonym_edges': 0,
            'next_sentence_edges': 0,
        }

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {}

    def _save_progress(self):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)

    def _init_database(self, fresh: bool = False):
        """Initialize Kuzu database and create schema."""
        if fresh and self.db_path.exists():
            logger.info("Fresh start requested, deleting existing database...")
            # Handle both directory (newer Kuzu) and file (older Kuzu) formats
            if self.db_path.is_dir():
                shutil.rmtree(self.db_path)
            else:
                self.db_path.unlink()
            # Also remove WAL file if it exists
            wal_path = self.db_path.with_suffix('.db.wal')
            if wal_path.exists():
                wal_path.unlink()
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

        self.progress = self._load_progress()

        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create database
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

        # Create schema if not exists
        if not self.progress.get('schema_created'):
            self._create_schema()
            self.progress['schema_created'] = True
            self._save_progress()

    def _create_schema(self):
        """Create Kuzu schema for all phases."""
        logger.info("Creating Kuzu schema...")

        # Node tables
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Root (
                root STRING,
                doc_freq INT64 DEFAULT 0,
                total_freq INT64 DEFAULT 0,
                PRIMARY KEY (root)
            )
        """)

        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Sentence (
                id INT64,
                text STRING,
                doc_id INT64,
                sent_idx INT64,
                PRIMARY KEY (id)
            )
        """)

        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Document (
                id INT64,
                source_name STRING,
                source_type STRING,
                PRIMARY KEY (id)
            )
        """)

        # Edge tables - Phase 1: Inverted Index
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS HAS_ROOT (
                FROM Sentence TO Root,
                role STRING,
                grammar STRING
            )
        """)

        # Edge tables - Phase 2: Semantic Relations
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS IS_SYNONYM (
                FROM Root TO Root
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS IS_HYPERNYM (
                FROM Root TO Root
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS IS_ANTONYM (
                FROM Root TO Root
            )
        """)

        # Edge tables - Phase 3: Sentence Context
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS NEXT_SENTENCE (
                FROM Sentence TO Sentence
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS IN_DOCUMENT (
                FROM Sentence TO Document
            )
        """)

        logger.info("  Schema created successfully")

    def _extract_roots_from_ast(self, ast: Dict) -> List[Tuple[str, str, Dict]]:
        """
        Extract all roots from an AST with their role and grammar.

        Returns list of (root, role, grammar) tuples.
        """
        results = []

        def extract(node, role: str):
            if not node or not isinstance(node, dict):
                return

            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco', '')
                if vortspeco in SKIP_VORTSPECO:
                    return

                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    root = root.lower()

                    # Collect grammar features
                    grammar = {}
                    if node.get('tempo'):
                        grammar['tempo'] = node['tempo']
                    if node.get('modo'):
                        grammar['modo'] = node['modo']
                    if node.get('kazo'):
                        grammar['kazo'] = node['kazo']
                    if node.get('nombro'):
                        grammar['nombro'] = node['nombro']

                    results.append((root, role, grammar))

            elif node.get('tipo') == 'vortgrupo':
                extract(node.get('kerno'), role)
                for p in node.get('priskriboj', []):
                    extract(p, role)

            elif node.get('tipo') == 'frazo':
                extract(node.get('subjekto'), 'subjekto')
                extract(node.get('verbo'), 'verbo')
                extract(node.get('objekto'), 'objekto')
                for a in node.get('aliaj', []):
                    extract(a, 'aliaj')

        extract(ast, 'unknown')
        return results

    def phase1_build_inverted_index(
        self,
        corpus_path: Path,
        limit: Optional[int] = None,
    ):
        """
        Phase 1: Stream corpus, write CSVs, bulk load into Kuzu.

        Creates:
        - Root nodes (unique roots)
        - Sentence nodes
        - Document nodes
        - HAS_ROOT edges (Sentence → Root)
        - IN_DOCUMENT edges (Sentence → Document)
        - NEXT_SENTENCE edges
        """
        if self.progress.get('phase1_complete'):
            logger.info("Phase 1 already complete, skipping...")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 1: Building Inverted Index (CSV Bulk Loading)")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Check if CSVs were already created
        csvs_created = self.progress.get('phase1_csvs_created', False)

        if not csvs_created:
            # Step 1: Stream corpus and write CSVs
            logger.info("Step 1: Streaming corpus and writing CSV files...")
            self._stream_corpus_to_csvs(corpus_path, limit)
            self.progress['phase1_csvs_created'] = True
            self._save_progress()
        else:
            logger.info("CSVs already created, skipping to bulk loading...")

        # Step 2: Bulk load CSVs into Kuzu
        logger.info("")
        logger.info("Step 2: Bulk loading CSVs into Kuzu...")
        self._bulk_load_csvs()

        # Step 3: Compute root statistics
        logger.info("")
        logger.info("Step 3: Computing root statistics...")
        self._compute_root_stats()

        # Mark phase 1 complete
        self.progress['phase1_complete'] = True
        self._save_progress()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Phase 1 complete in {elapsed/60:.1f} minutes")

    def _stream_corpus_to_csvs(self, corpus_path: Path, limit: Optional[int] = None):
        """Stream corpus and write to CSV files for bulk loading."""
        # Count total documents for progress
        logger.info("  Counting documents...")
        with open(corpus_path) as f:
            total_docs = sum(1 for _ in f)
        logger.info(f"  Total documents in corpus: {total_docs:,}")

        if limit:
            total_docs = min(total_docs, limit)
            logger.info(f"  Limited to: {total_docs:,}")

        # Track unique entities
        roots_seen: Set[str] = set()
        docs_seen: Set[int] = set()

        # Open CSV files
        csv_files = {
            'roots': open(self.temp_dir / 'roots.csv', 'w', newline='', encoding='utf-8'),
            'sentences': open(self.temp_dir / 'sentences.csv', 'w', newline='', encoding='utf-8'),
            'documents': open(self.temp_dir / 'documents.csv', 'w', newline='', encoding='utf-8'),
            'has_root': open(self.temp_dir / 'has_root.csv', 'w', newline='', encoding='utf-8'),
            'in_document': open(self.temp_dir / 'in_document.csv', 'w', newline='', encoding='utf-8'),
            'next_sentence': open(self.temp_dir / 'next_sentence.csv', 'w', newline='', encoding='utf-8'),
        }

        csv_writers = {
            'roots': csv.writer(csv_files['roots']),
            'sentences': csv.writer(csv_files['sentences']),
            'documents': csv.writer(csv_files['documents']),
            'has_root': csv.writer(csv_files['has_root']),
            'in_document': csv.writer(csv_files['in_document']),
            'next_sentence': csv.writer(csv_files['next_sentence']),
        }

        # Write headers - must match all columns in table schema
        csv_writers['roots'].writerow(['root', 'doc_freq', 'total_freq'])
        csv_writers['sentences'].writerow(['id', 'text', 'doc_id', 'sent_idx'])
        csv_writers['documents'].writerow(['id', 'source_name', 'source_type'])
        csv_writers['has_root'].writerow(['sent_id', 'root', 'role', 'grammar'])
        csv_writers['in_document'].writerow(['sent_id', 'doc_id'])
        csv_writers['next_sentence'].writerow(['prev_id', 'next_id'])

        # Track for NEXT_SENTENCE edges
        prev_sent_by_source: Dict[str, int] = {}  # source_key → last sent_id

        # For document text output
        docs_file = self.output_path / "documents.jsonl"
        offsets = []

        sent_id = 0
        processed = 0
        skipped_no_ast = 0
        skipped_low_parse = 0
        last_log_time = datetime.now()

        with open(corpus_path) as corpus_f, \
             open(docs_file, 'w') as docs_out:

            for line in corpus_f:
                if limit and processed >= limit:
                    break

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip documents without AST
                ast = doc.get('ast')
                if not ast:
                    skipped_no_ast += 1
                    continue

                # Skip documents with low parse rate
                parse_rate = doc.get('parse_rate', 1.0)
                if parse_rate < 0.5:
                    skipped_low_parse += 1
                    continue

                # Get source info
                source = doc.get('source', {})
                source_name = source.get('name', 'unknown')
                source_type = source.get('type', 'unknown')
                doc_id = processed  # Use processed count as doc_id

                # Add Document node if not seen
                if doc_id not in docs_seen:
                    csv_writers['documents'].writerow([doc_id, source_name, source_type])
                    docs_seen.add(doc_id)

                # Write document text for later retrieval
                offset = docs_out.tell()
                offsets.append(offset)
                docs_out.write(json.dumps({
                    'text': doc.get('text', ''),
                    'source': source,
                    'parse_rate': parse_rate,
                }, ensure_ascii=False) + '\n')

                # Add Sentence node
                text = doc.get('text', '')[:1000]  # Limit text length in graph
                # Escape text for CSV (replace newlines, etc.)
                text = text.replace('\n', ' ').replace('\r', ' ')
                csv_writers['sentences'].writerow([sent_id, text, doc_id, 0])

                # Add IN_DOCUMENT edge
                csv_writers['in_document'].writerow([sent_id, doc_id])

                # Add NEXT_SENTENCE edge (within same source)
                source_key = f"{source_type}:{source_name}"
                if source_key in prev_sent_by_source:
                    prev_sent = prev_sent_by_source[source_key]
                    csv_writers['next_sentence'].writerow([prev_sent, sent_id])
                prev_sent_by_source[source_key] = sent_id

                # Extract roots and create edges
                roots = self._extract_roots_from_ast(ast)
                for root, role, grammar in roots:
                    # Add Root node if not seen
                    if root not in roots_seen:
                        # Write all columns: root, doc_freq (0), total_freq (0)
                        # Stats will be computed later
                        csv_writers['roots'].writerow([root, 0, 0])
                        roots_seen.add(root)

                    # Add HAS_ROOT edge
                    grammar_json = json.dumps(grammar, ensure_ascii=False)
                    csv_writers['has_root'].writerow([sent_id, root, role, grammar_json])

                sent_id += 1
                processed += 1

                # Log progress every 10 seconds
                now = datetime.now()
                if (now - last_log_time).total_seconds() >= 10:
                    elapsed = (now - start_time).total_seconds()
                    rate = processed / elapsed if elapsed > 0 else 0
                    pct = processed / total_docs * 100
                    logger.info(
                        f"  Processed {processed:,}/{total_docs:,} ({pct:.1f}%) - "
                        f"{len(roots_seen):,} roots - {rate:.0f} docs/sec"
                    )
                    last_log_time = now

        # Close CSV files
        for f in csv_files.values():
            f.close()

        # Save document offsets
        np.save(self.output_path / "doc_offsets.npy", np.array(offsets, dtype=np.int64))

        # Update stats
        self.stats['sentences'] = sent_id
        self.stats['documents'] = processed
        self.stats['roots'] = len(roots_seen)

        logger.info(f"  CSV files written:")
        logger.info(f"    Roots: {len(roots_seen):,}")
        logger.info(f"    Sentences: {sent_id:,}")
        logger.info(f"    Documents: {len(docs_seen):,}")
        logger.info(f"    Skipped (no AST): {skipped_no_ast:,}")
        logger.info(f"    Skipped (low parse): {skipped_low_parse:,}")

        # Save progress
        self.progress['phase1_sentences'] = sent_id
        self.progress['phase1_documents'] = processed
        self.progress['phase1_roots'] = len(roots_seen)
        self._save_progress()

    def _bulk_load_csvs(self):
        """Bulk load CSV files into Kuzu using COPY FROM."""
        start_time = datetime.now()

        # Helper function to run COPY with timing
        def copy_csv(table_name: str, csv_file: str, is_node: bool = True):
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

        # Load node tables first
        logger.info("  Loading node tables...")
        copy_csv("Root", "roots.csv")
        copy_csv("Document", "documents.csv")
        copy_csv("Sentence", "sentences.csv")

        # Load edge tables
        logger.info("")
        logger.info("  Loading edge tables...")
        copy_csv("HAS_ROOT", "has_root.csv", is_node=False)
        copy_csv("IN_DOCUMENT", "in_document.csv", is_node=False)
        copy_csv("NEXT_SENTENCE", "next_sentence.csv", is_node=False)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"  Bulk loading complete in {elapsed:.1f}s")

    def _compute_root_stats(self):
        """Compute doc_freq and total_freq for each root."""
        logger.info("  Computing doc_freq and total_freq for each root...")

        t0 = datetime.now()
        # This updates all roots with their document frequency
        # doc_freq = number of distinct sentences containing this root
        self.conn.execute("""
            MATCH (s:Sentence)-[:HAS_ROOT]->(r:Root)
            WITH r, count(DISTINCT s) AS df, count(*) AS tf
            SET r.doc_freq = df, r.total_freq = tf
        """)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"    Done in {elapsed:.1f}s")

    def phase2_load_semantic_relations(
        self,
        revo_path: Path,
        curated_path: Optional[Path] = None,
    ):
        """
        Phase 2: Load semantic relations into Kuzu using CSV bulk loading.

        Creates:
        - IS_SYNONYM edges
        - IS_HYPERNYM edges
        - IS_ANTONYM edges
        """
        if self.progress.get('phase2_complete'):
            logger.info("Phase 2 already complete, skipping...")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 2: Loading Semantic Relations (CSV Bulk Loading)")
        logger.info("=" * 60)

        start_time = datetime.now()

        if not revo_path.exists():
            logger.warning(f"ReVo relations not found at {revo_path}, skipping Phase 2")
            self.progress['phase2_complete'] = True
            self._save_progress()
            return

        # Load ReVo relations
        logger.info(f"Loading from {revo_path}")
        with open(revo_path) as f:
            data = json.load(f)

        relations = data.get('relations', {})

        # Get existing roots in database
        logger.info("Loading existing roots from database...")
        existing_roots: Set[str] = set()
        result = self.conn.execute("MATCH (r:Root) RETURN r.root")
        while result.has_next():
            existing_roots.add(result.get_next()[0])
        logger.info(f"  Found {len(existing_roots):,} roots in database")

        # Write relation CSVs
        synonym_csv = self.temp_dir / 'synonyms.csv'
        hypernym_csv = self.temp_dir / 'hypernyms.csv'
        antonym_csv = self.temp_dir / 'antonyms.csv'

        # Process synonyms
        synonyms = relations.get('synonym', [])
        logger.info(f"Processing {len(synonyms):,} synonym pairs...")
        synonym_count = 0

        with open(synonym_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['root1', 'root2'])

            for root1, root2 in synonyms:
                root1 = root1.lower()
                root2 = root2.lower()

                # Only add if both roots exist in index
                if root1 in existing_roots and root2 in existing_roots:
                    # Bidirectional synonym edges
                    writer.writerow([root1, root2])
                    writer.writerow([root2, root1])
                    synonym_count += 1

        logger.info(f"  Wrote {synonym_count:,} synonym pairs")

        # Process hypernyms
        hypernyms = relations.get('hypernym', [])
        logger.info(f"Processing {len(hypernyms):,} hypernym pairs...")
        hypernym_count = 0

        with open(hypernym_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['specific', 'general'])

            for specific, general in hypernyms:
                specific = specific.lower()
                general = general.lower()

                if specific in existing_roots and general in existing_roots:
                    writer.writerow([specific, general])
                    hypernym_count += 1

        logger.info(f"  Wrote {hypernym_count:,} hypernym pairs")

        # Process antonyms
        antonyms = relations.get('antonym', [])
        logger.info(f"Processing {len(antonyms):,} antonym pairs...")
        antonym_count = 0

        with open(antonym_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['root1', 'root2'])

            for root1, root2 in antonyms:
                root1 = root1.lower()
                root2 = root2.lower()

                if root1 in existing_roots and root2 in existing_roots:
                    # Bidirectional antonym edges
                    writer.writerow([root1, root2])
                    writer.writerow([root2, root1])
                    antonym_count += 1

        logger.info(f"  Wrote {antonym_count:,} antonym pairs")

        # Load curated relations if available
        if curated_path and curated_path.exists():
            logger.info(f"Loading curated relations from {curated_path}")
            with open(curated_path) as f:
                curated = json.load(f)

            # Append to synonym CSV
            curated_count = 0
            with open(synonym_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                for section in ['verb_synonyms', 'noun_synonyms']:
                    for root, info in curated.get(section, {}).items():
                        root = root.lower()
                        if root not in existing_roots:
                            continue

                        for syn in info.get('synonyms', []):
                            syn = syn.lower()
                            if syn in existing_roots:
                                writer.writerow([root, syn])
                                curated_count += 1

            logger.info(f"  Added {curated_count:,} curated synonym edges")

        # Bulk load relation CSVs
        logger.info("")
        logger.info("Bulk loading semantic relations...")

        try:
            self.conn.execute(f"COPY IS_SYNONYM FROM '{synonym_csv}' (header=true)")
            logger.info(f"  Loaded IS_SYNONYM edges")
        except Exception as e:
            logger.warning(f"  Error loading synonyms: {e}")

        try:
            self.conn.execute(f"COPY IS_HYPERNYM FROM '{hypernym_csv}' (header=true)")
            logger.info(f"  Loaded IS_HYPERNYM edges")
        except Exception as e:
            logger.warning(f"  Error loading hypernyms: {e}")

        try:
            self.conn.execute(f"COPY IS_ANTONYM FROM '{antonym_csv}' (header=true)")
            logger.info(f"  Loaded IS_ANTONYM edges")
        except Exception as e:
            logger.warning(f"  Error loading antonyms: {e}")

        self.stats['synonym_edges'] = synonym_count * 2
        self.stats['hypernym_edges'] = hypernym_count
        self.stats['antonym_edges'] = antonym_count * 2

        # Mark phase 2 complete
        self.progress['phase2_complete'] = True
        self._save_progress()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Phase 2 complete in {elapsed:.1f}s")

    def phase3_verify_counts(self):
        """
        Phase 3: Verify final counts and statistics.

        NEXT_SENTENCE edges are now created during Phase 1 for efficiency.
        This phase just verifies and logs statistics.
        """
        if self.progress.get('phase3_complete'):
            logger.info("Phase 3 already complete, skipping...")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 3: Verifying Index")
        logger.info("=" * 60)

        # Count all nodes and edges
        result = self.conn.execute("MATCH (r:Root) RETURN count(r)")
        root_count = result.get_next()[0]
        logger.info(f"  Root nodes: {root_count:,}")

        result = self.conn.execute("MATCH (s:Sentence) RETURN count(s)")
        sent_count = result.get_next()[0]
        logger.info(f"  Sentence nodes: {sent_count:,}")

        result = self.conn.execute("MATCH (d:Document) RETURN count(d)")
        doc_count = result.get_next()[0]
        logger.info(f"  Document nodes: {doc_count:,}")

        result = self.conn.execute("MATCH ()-[e:HAS_ROOT]->() RETURN count(e)")
        has_root_count = result.get_next()[0]
        logger.info(f"  HAS_ROOT edges: {has_root_count:,}")

        result = self.conn.execute("MATCH ()-[e:NEXT_SENTENCE]->() RETURN count(e)")
        next_count = result.get_next()[0]
        logger.info(f"  NEXT_SENTENCE edges: {next_count:,}")
        self.stats['next_sentence_edges'] = next_count

        result = self.conn.execute("MATCH ()-[e:IN_DOCUMENT]->() RETURN count(e)")
        in_doc_count = result.get_next()[0]
        logger.info(f"  IN_DOCUMENT edges: {in_doc_count:,}")

        result = self.conn.execute("MATCH ()-[e:IS_SYNONYM]->() RETURN count(e)")
        synonym_count = result.get_next()[0]
        logger.info(f"  IS_SYNONYM edges: {synonym_count:,}")

        result = self.conn.execute("MATCH ()-[e:IS_HYPERNYM]->() RETURN count(e)")
        hypernym_count = result.get_next()[0]
        logger.info(f"  IS_HYPERNYM edges: {hypernym_count:,}")

        # Mark phase 3 complete
        self.progress['phase3_complete'] = True
        self._save_progress()

        logger.info("Phase 3 complete")

    def finalize(self):
        """Finalize the index and print summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Finalizing Index")
        logger.info("=" * 60)

        # Get final counts
        result = self.conn.execute("MATCH (r:Root) RETURN count(r)")
        root_count = result.get_next()[0]

        result = self.conn.execute("MATCH (s:Sentence) RETURN count(s)")
        sent_count = result.get_next()[0]

        result = self.conn.execute("MATCH (d:Document) RETURN count(d)")
        doc_count = result.get_next()[0]

        result = self.conn.execute("MATCH ()-[e:HAS_ROOT]->() RETURN count(e)")
        has_root_count = result.get_next()[0]

        result = self.conn.execute("MATCH ()-[e:IS_SYNONYM]->() RETURN count(e)")
        synonym_count = result.get_next()[0]

        result = self.conn.execute("MATCH ()-[e:IS_HYPERNYM]->() RETURN count(e)")
        hypernym_count = result.get_next()[0]

        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()

        # Clean up temp CSV files
        if self.temp_dir.exists():
            logger.info("Cleaning up temporary CSV files...")
            shutil.rmtree(self.temp_dir)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Build Complete")
        logger.info("=" * 60)
        logger.info(f"Nodes:")
        logger.info(f"  Root:     {root_count:,}")
        logger.info(f"  Sentence: {sent_count:,}")
        logger.info(f"  Document: {doc_count:,}")
        logger.info(f"")
        logger.info(f"Edges:")
        logger.info(f"  HAS_ROOT:      {has_root_count:,}")
        logger.info(f"  IS_SYNONYM:    {synonym_count:,}")
        logger.info(f"  IS_HYPERNYM:   {hypernym_count:,}")
        logger.info(f"  NEXT_SENTENCE: {self.stats.get('next_sentence_edges', 0):,}")
        logger.info("")
        logger.info(f"Output: {self.output_path}")

        # File sizes
        logger.info("")
        logger.info("Files:")
        if self.db_path.exists():
            # Kuzu database can be a file or directory depending on version
            if self.db_path.is_dir():
                total_size = sum(f.stat().st_size for f in self.db_path.rglob('*') if f.is_file())
                logger.info(f"  kuzu.db/: {total_size / 1024 / 1024:.1f} MB")
            else:
                total_size = self.db_path.stat().st_size
                logger.info(f"  kuzu.db: {total_size / 1024 / 1024:.1f} MB")

        for fname in ['documents.jsonl', 'doc_offsets.npy']:
            fpath = self.output_path / fname
            if fpath.exists():
                logger.info(f"  {fname}: {fpath.stat().st_size / 1024 / 1024:.1f} MB")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None


def build_index(
    corpus_path: Path,
    output_path: Path,
    revo_path: Path,
    curated_path: Optional[Path] = None,
    limit: Optional[int] = None,
    fresh: bool = False,
    phase: Optional[int] = None,
):
    """Build complete Kuzu index with all phases."""
    logger.info("=" * 60)
    logger.info("Building Kuzu Graph Index (CSV Bulk Loading)")
    logger.info("=" * 60)
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"ReVo:   {revo_path}")
    if limit:
        logger.info(f"Limit:  {limit:,} documents")
    if phase:
        logger.info(f"Phase:  {phase}")
    logger.info("")

    builder = KuzuIndexBuilder(output_path)

    try:
        # Initialize database
        builder._init_database(fresh=fresh)

        # Phase 1: Build inverted index
        if phase is None or phase == 1:
            builder.phase1_build_inverted_index(corpus_path, limit=limit)

        # Phase 2: Load semantic relations
        if phase is None or phase == 2:
            builder.phase2_load_semantic_relations(revo_path, curated_path)

        # Phase 3: Verify counts
        if phase is None or phase == 3:
            builder.phase3_verify_counts()

        # Finalize (only if running all phases)
        if phase is None:
            builder.finalize()

    finally:
        builder.close()


# Track start time globally for progress logging
start_time = datetime.now()


def main():
    parser = argparse.ArgumentParser(description="Build Kuzu graph index")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/unified_corpus.jsonl"),
        help="Path to unified corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/indexes/kuzu_index"),
        help="Output directory",
    )
    parser.add_argument(
        "--revo",
        type=Path,
        default=Path("data/raw/eo/dictionaries/revo/revo_semantic_relations.json"),
        help="Path to ReVo semantic relations",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("data/semantic_relations/curated_synonyms.json"),
        help="Path to curated synonyms (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore previous progress",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only specific phase",
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        logger.error(f"Corpus not found: {args.corpus}")
        sys.exit(1)

    global start_time
    start_time = datetime.now()

    build_index(
        corpus_path=args.corpus,
        output_path=args.output,
        revo_path=args.revo,
        curated_path=args.curated if args.curated.exists() else None,
        limit=args.limit,
        fresh=args.fresh,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
