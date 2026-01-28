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

        # Predicate table for fast structural matching
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Predicate (
                id STRING,
                verb STRING,
                subj STRING,
                obj STRING,
                PRIMARY KEY (id)
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS HAS_PREDICATE (
                FROM Sentence TO Predicate
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

    def _extract_predicate_from_ast(self, ast: Dict) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        Extract predicate (subject-verb-object) from AST.

        Returns (verb, subject, object) tuple where verb is required, subject and object are optional.
        For compounds, preserves suffix structure to distinguish (e.g., "esperant" vs "esperant-rond").

        Returns None if no verb found.
        """
        if not ast or not isinstance(ast, dict):
            return None

        def get_root(node):
            """Extract root from a node (handling vortgrupo)."""
            if not node or not isinstance(node, dict):
                return None

            if node.get('tipo') == 'vortgrupo':
                kerno = node.get('kerno', {})
                return kerno.get('radiko', '').lower() if isinstance(kerno, dict) else None
            elif node.get('tipo') == 'vorto':
                return node.get('radiko', '').lower()

            return None

        def get_word_with_compounds(node):
            """
            Extract word preserving compound structure for semantic disambiguation.

            Examples:
            - "Esperanto" → "esperant"
            - "Esperanto-rondo" → "esperant-rond" (compound preserved)
            - "esperantisto" → "esperant-ist" (derivational suffix preserved)

            Issue #550: Preserve compounds to distinguish language from organizations.
            """
            if not node or not isinstance(node, dict):
                return None

            # Handle vortgrupo by extracting kerno
            if node.get('tipo') == 'vortgrupo':
                node = node.get('kerno', {})
                if not isinstance(node, dict):
                    return None

            # Extract root
            root = node.get('radiko', '').lower()
            if not root:
                return None

            # Build compound structure
            parts = []

            # 1. Check for compound words (kunmetajhoj - using h not ĵ)
            if node.get('estas_kunmetita') and 'kunmetajhoj' in node:
                # Extract roots from compound components
                semantic_suffixes = ['ist', 'ul', 'ej', 'ar', 'aĵ', 'ism', 'an', 'estr', 'il', 'uj']
                for component in node.get('kunmetajhoj', []):
                    if isinstance(component, dict):
                        comp_root = component.get('radiko', '').lower()
                        if comp_root:
                            parts.append(comp_root)
                        # Also extract semantic suffixes from components
                        comp_sufixes = component.get('sufiksoj', [])
                        if comp_sufixes:
                            preserved = [suf for suf in comp_sufixes if suf in semantic_suffixes]
                            parts.extend(preserved)

            # 2. Add main root
            parts.append(root)

            # 3. Check for derivational suffixes (semantic type changers)
            sufiksoj = node.get('sufiksoj', [])
            if sufiksoj:
                # Keep: -ist, -ul, -ej, -ar, -aĵ, -ism, -an (semantic type changers)
                # Skip: -in (gender), -et/-eg (size), -ĉj/-nj (affectionate)
                semantic_suffixes = ['ist', 'ul', 'ej', 'ar', 'aĵ', 'ism', 'an', 'estr', 'il', 'uj']
                preserved = [suf for suf in sufiksoj if suf in semantic_suffixes]
                parts.extend(preserved)

            # Join all parts with hyphen
            if len(parts) > 1:
                return '-'.join(parts)

            return root

        # Extract verb (required) - just root, no compounds
        verb = get_root(ast.get('verbo'))
        if not verb:
            return None

        # Extract subject - just root (names/nouns are simple)
        subj = get_root(ast.get('subjekto'))

        # Extract object - preserve compounds for disambiguation
        obj = get_word_with_compounds(ast.get('objekto'))

        return (verb, subj, obj)

    def _extract_virtual_predicates_from_noun_phrases(
        self, ast: Dict
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Extract virtual predicates from participial noun phrases.

        Esperanto commonly expresses facts using participles as nouns/adjectives:
        - "Zamenhof, la kreinto de Esperanto" (Zamenhof, the creator of Esperanto)
        - "La fondinto de Esperanto estis Zamenhof" (The founder of Esperanto was Zamenhof)

        These should be indexed as virtual predicates for retrieval.

        Patterns recognized:
        1. "X-into/anto/onto de Y" → (X, ?, Y)
        2. "A estas B-into/anto/onto de C" → (B, A, C)
        3. Subject with participle + prepositional phrase → virtual predicate

        Issue #549: Most encyclopedia-style facts use participles, not verb predicates.

        Returns:
            List of (verb, subj, obj) virtual predicate tuples
        """
        if not ast or not isinstance(ast, dict):
            return []

        virtual_predicates = []

        # Helper to extract word with participle info
        def get_word_info(node):
            """Get word root, suffixes, and participle metadata."""
            if not node or not isinstance(node, dict):
                return None

            # Handle vortgrupo
            if node.get('tipo') == 'vortgrupo':
                node = node.get('kerno', {})
                if not isinstance(node, dict):
                    return None

            if node.get('tipo') != 'vorto':
                return None

            root = node.get('radiko', '').lower()
            sufiksoj = node.get('sufiksoj', [])
            participo_voĉo = node.get('participo_voĉo')
            participo_tempo = node.get('participo_tempo')

            return {
                'root': root,
                'sufiksoj': sufiksoj,
                'participo_voĉo': participo_voĉo,
                'participo_tempo': participo_tempo,
                'full_node': node
            }

        # Helper to find "de X" prepositional phrases in aliaj
        def find_de_phrase_object(aliaj_list):
            """Find object of 'de' prepositional phrase in aliaj."""
            if not aliaj_list:
                return None

            # Look for pattern: "de" followed by noun
            for i, word in enumerate(aliaj_list):
                if not isinstance(word, dict):
                    continue

                # Check if this is "de" preposition
                if (word.get('tipo') == 'vorto' and
                    word.get('radiko', '').lower() == 'de' and
                    word.get('vortspeco') == 'prepozicio'):

                    # Next word should be the object
                    if i + 1 < len(aliaj_list):
                        next_word = aliaj_list[i + 1]
                        if isinstance(next_word, dict) and next_word.get('tipo') == 'vorto':
                            root = next_word.get('radiko', '').lower()
                            if root:
                                return root

            return None

        # Pattern 1: Subject with participle + "de Y"
        # "La kreinto de Esperanto..." → (kre, ?, esperant)
        subjekto = ast.get('subjekto')
        if subjekto:
            subj_info = get_word_info(subjekto)
            if subj_info and 'int' in subj_info['sufiksoj']:
                # Subject has past participle (-int)
                verb_root = subj_info['root']

                # Look for "de X" in aliaj
                de_object = find_de_phrase_object(ast.get('aliaj', []))
                if de_object:
                    # Virtual predicate: verb from participle, unknown subject, de-object
                    virtual_predicates.append((verb_root, None, de_object))

        # Pattern 2: "A estas B-into de C" → (B, A, C)
        # "Zamenhof estas kreinto de Esperanto" → (kre, zamenhof, esperant)
        verbo = ast.get('verbo')
        if verbo and isinstance(verbo, dict):
            verb_root = verbo.get('radiko', '').lower()

            # Check if verb is "est" (to be)
            if verb_root == 'est':
                # Check if objekto or aliaj contains participle noun
                aliaj = ast.get('aliaj', [])

                for word in aliaj:
                    if not isinstance(word, dict):
                        continue

                    word_info = get_word_info(word)
                    if word_info and 'int' in word_info['sufiksoj']:
                        # Found participle noun in predicate nominative
                        pred_verb = word_info['root']

                        # Subject is the agent
                        subj_info = get_word_info(ast.get('subjekto'))
                        pred_subj = subj_info['root'] if subj_info else None

                        # Object from "de X" phrase
                        de_object = find_de_phrase_object(aliaj)

                        if pred_verb and pred_subj:
                            virtual_predicates.append((pred_verb, pred_subj, de_object))

        # Pattern 3: Participial adjective modifying noun
        # "la fondinto Zamenhof" → (fond, zamenhof, ?)
        # This is less common, but handle if subject has participle descriptor
        if subjekto and isinstance(subjekto, dict) and subjekto.get('tipo') == 'vortgrupo':
            priskriboj = subjekto.get('priskriboj', [])
            kerno = subjekto.get('kerno', {})
            kerno_root = kerno.get('radiko', '').lower() if isinstance(kerno, dict) else None

            for priskribo in priskriboj:
                if not isinstance(priskribo, dict):
                    continue

                pri_info = get_word_info(priskribo)
                if pri_info and 'int' in pri_info['sufiksoj']:
                    # Participle adjective modifying subject
                    verb_root = pri_info['root']

                    if kerno_root:
                        # Virtual predicate: participle verb, modified noun as subject
                        virtual_predicates.append((verb_root, kerno_root, None))

        return virtual_predicates

    def _resolve_pronoun_subject(
        self,
        subj_root: Optional[str],
        document_context: List[str]
    ) -> Optional[str]:
        """
        Resolve pronoun subject to nearest mentioned entity.

        Simple heuristic for Issue #551: If subject is pronoun (li, ŝi, ĝi, ili),
        return the last proper noun mentioned in document context.

        Args:
            subj_root: Subject root (may be pronoun)
            document_context: List of proper nouns mentioned in this document so far

        Returns:
            Resolved subject (entity name) or original if not pronoun/no resolution
        """
        # Check if subject is pronoun
        pronouns = {'li', 'ŝi', 'ĝi', 'ili', 'mi', 'vi', 'ni'}
        if not subj_root or subj_root.lower() not in pronouns:
            return subj_root

        # Resolve to last mentioned entity
        if document_context:
            # Return most recent proper noun
            return document_context[-1]

        # No context, return original
        return subj_root

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

        # Note: Root statistics (doc_freq, total_freq) are now computed during
        # CSV streaming and included in roots.csv, so no separate computation needed.

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
        predicates_seen: Set[str] = set()  # Track unique predicate IDs

        # Track root statistics during streaming (memory efficient)
        root_doc_freq: Dict[str, int] = defaultdict(int)  # root → doc count
        root_total_freq: Dict[str, int] = defaultdict(int)  # root → total occurrences

        # Open CSV files
        csv_files = {
            'roots': open(self.temp_dir / 'roots.csv', 'w', newline='', encoding='utf-8'),
            'sentences': open(self.temp_dir / 'sentences.csv', 'w', newline='', encoding='utf-8'),
            'documents': open(self.temp_dir / 'documents.csv', 'w', newline='', encoding='utf-8'),
            'has_root': open(self.temp_dir / 'has_root.csv', 'w', newline='', encoding='utf-8'),
            'in_document': open(self.temp_dir / 'in_document.csv', 'w', newline='', encoding='utf-8'),
            'next_sentence': open(self.temp_dir / 'next_sentence.csv', 'w', newline='', encoding='utf-8'),
            'predicates': open(self.temp_dir / 'predicates.csv', 'w', newline='', encoding='utf-8'),
            'has_predicate': open(self.temp_dir / 'has_predicate.csv', 'w', newline='', encoding='utf-8'),
        }

        csv_writers = {
            'roots': csv.writer(csv_files['roots']),
            'sentences': csv.writer(csv_files['sentences']),
            'documents': csv.writer(csv_files['documents']),
            'has_root': csv.writer(csv_files['has_root']),
            'in_document': csv.writer(csv_files['in_document']),
            'next_sentence': csv.writer(csv_files['next_sentence']),
            'predicates': csv.writer(csv_files['predicates']),
            'has_predicate': csv.writer(csv_files['has_predicate']),
        }

        # Write headers - must match all columns in table schema
        csv_writers['roots'].writerow(['root', 'doc_freq', 'total_freq'])
        csv_writers['sentences'].writerow(['id', 'text', 'doc_id', 'sent_idx'])
        csv_writers['documents'].writerow(['id', 'source_name', 'source_type'])
        csv_writers['has_root'].writerow(['sent_id', 'root', 'role', 'grammar'])
        csv_writers['in_document'].writerow(['sent_id', 'doc_id'])
        csv_writers['next_sentence'].writerow(['prev_id', 'next_id'])
        csv_writers['predicates'].writerow(['id', 'verb', 'subj', 'obj'])
        csv_writers['has_predicate'].writerow(['sent_id', 'pred_id'])

        # Track for NEXT_SENTENCE edges
        prev_sent_by_source: Dict[str, int] = {}  # source_key → last sent_id

        # Track document context for pronoun resolution (Issue #551)
        doc_context_by_source: Dict[str, List[str]] = {}  # source_key → [proper nouns]

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
                roots_in_this_doc: Set[str] = set()  # For doc_freq counting
                for root, role, grammar in roots:
                    # Track stats
                    root_total_freq[root] += 1
                    roots_in_this_doc.add(root)

                    # Add Root node if not seen (placeholder - will rewrite with stats)
                    if root not in roots_seen:
                        roots_seen.add(root)

                    # Add HAS_ROOT edge
                    grammar_json = json.dumps(grammar, ensure_ascii=False)
                    csv_writers['has_root'].writerow([sent_id, root, role, grammar_json])

                # Update doc_freq for each unique root in this document
                for root in roots_in_this_doc:
                    root_doc_freq[root] += 1

                # Track proper nouns for pronoun resolution (Issue #551)
                # Get document context for this source
                source_key = f"{source_type}:{source_name}"
                if source_key not in doc_context_by_source:
                    doc_context_by_source[source_key] = []

                # Extract proper nouns from subject position (names that can be referenced)
                subjekto = ast.get('subjekto')
                if subjekto and isinstance(subjekto, dict):
                    # Get kerno if vortgrupo
                    node = subjekto if subjekto.get('tipo') == 'vorto' else subjekto.get('kerno', {})
                    if isinstance(node, dict):
                        root = node.get('radiko', '').lower()
                        # Check if proper noun (capitalized in original, or marked as proper name)
                        if root and (
                            node.get('plena_vorto', '')[0].isupper() or
                            root not in {'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili'}
                        ):
                            # Add to context (keep last 10 for recency)
                            doc_context_by_source[source_key].append(root)
                            if len(doc_context_by_source[source_key]) > 10:
                                doc_context_by_source[source_key].pop(0)

                # Extract predicates (both verb predicates and virtual predicates from participles)
                all_predicates = []

                # 1. Extract standard verb predicate
                predicate = self._extract_predicate_from_ast(ast)
                if predicate:
                    all_predicates.append(predicate)

                # 2. Extract virtual predicates from participial noun phrases (Issue #549)
                virtual_predicates = self._extract_virtual_predicates_from_noun_phrases(ast)
                all_predicates.extend(virtual_predicates)

                # Create nodes and edges for all predicates (with pronoun resolution)
                for verb, subj, obj in all_predicates:
                    # Resolve pronoun subject to entity (Issue #551)
                    subj_resolved = self._resolve_pronoun_subject(
                        subj,
                        doc_context_by_source.get(source_key, [])
                    )

                    # Create predicate ID (hash of verb+subj+obj)
                    # Use empty string for None values
                    subj_str = subj_resolved or ''
                    obj_str = obj or ''
                    pred_id = f"{verb}|{subj_str}|{obj_str}"

                    # Add Predicate node if not seen
                    if pred_id not in predicates_seen:
                        csv_writers['predicates'].writerow([pred_id, verb, subj_str, obj_str])
                        predicates_seen.add(pred_id)

                    # Add HAS_PREDICATE edge
                    csv_writers['has_predicate'].writerow([sent_id, pred_id])

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
                        f"{len(roots_seen):,} roots, {len(predicates_seen):,} predicates - {rate:.0f} docs/sec"
                    )
                    last_log_time = now

        # Close CSV files (except roots - we'll rewrite it with stats)
        for name, f in csv_files.items():
            f.close()

        # Rewrite roots.csv with computed statistics
        logger.info("  Writing roots.csv with computed statistics...")
        with open(self.temp_dir / 'roots.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['root', 'doc_freq', 'total_freq'])
            for root in sorted(roots_seen):
                writer.writerow([root, root_doc_freq[root], root_total_freq[root]])

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
        """Bulk load CSV files into Kuzu using COPY FROM or batched INSERT."""
        start_time = datetime.now()

        # Memory threshold: files larger than this use batched loading (MB)
        BATCH_THRESHOLD_MB = 500
        BATCH_SIZE = 200000  # Rows per batch for large files (increased for performance)

        # Helper function to run COPY with timing
        def copy_csv(table_name: str, csv_file: str, is_node: bool = True):
            csv_path = self.temp_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"  CSV file not found: {csv_path}")
                return 0

            file_size_mb = csv_path.stat().st_size / 1024 / 1024
            logger.info(f"  Loading {table_name} from {csv_file} ({file_size_mb:.1f} MB)...")

            # Use batched loading for large files
            if file_size_mb > BATCH_THRESHOLD_MB:
                return batch_load_csv(table_name, csv_path, file_size_mb)

            # Fast path: COPY FROM for small files
            t0 = datetime.now()
            try:
                self.conn.execute(f"COPY {table_name} FROM '{csv_path}' (header=true)")
                elapsed = (datetime.now() - t0).total_seconds()
                logger.info(f"    Done in {elapsed:.1f}s")
                return 1
            except Exception as e:
                logger.error(f"    Error: {e}")
                return 0

        def batch_load_csv(table_name: str, csv_path: Path, file_size_mb: float):
            """Load large CSV in chunks to avoid memory issues."""
            # Check for checkpoint (chunk number only, no byte offset due to csv.reader limitation)
            checkpoint_key = f"last_completed_chunk_{table_name}"
            start_chunk = self.progress.get(checkpoint_key, 0)

            if start_chunk > 0:
                logger.info(f"    Resuming from checkpoint: chunk {start_chunk + 1}")
            else:
                logger.info(f"    Large file detected - splitting into chunks (batches of {BATCH_SIZE:,} rows)")

            t0 = datetime.now()
            total_rows = 0
            chunk_count = 0

            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Read header

                    chunk_num = 0
                    while True:
                        # Read batch_size rows into a chunk file
                        chunk_rows = []
                        for _ in range(BATCH_SIZE):
                            try:
                                row = next(reader)
                                chunk_rows.append(row)
                            except StopIteration:
                                break

                        if not chunk_rows:
                            break  # No more rows

                        # Skip already-completed chunks (resume from checkpoint)
                        if chunk_num < start_chunk:
                            chunk_num += 1
                            continue

                        # Write chunk to temporary CSV
                        chunk_file = self.temp_dir / f"{csv_path.stem}_chunk_{chunk_num}.csv"
                        with open(chunk_file, 'w', newline='', encoding='utf-8') as cf:
                            writer = csv.writer(cf)
                            writer.writerow(header)  # Write header
                            writer.writerows(chunk_rows)

                        # Load chunk using COPY FROM (fast!)
                        try:
                            self.conn.execute(f"COPY {table_name} FROM '{chunk_file}' (header=true)")
                            total_rows += len(chunk_rows)
                            chunk_count += 1

                            # Save checkpoint after each successful chunk (chunk number only)
                            self.progress[checkpoint_key] = chunk_num
                            self._save_progress()

                            # Log progress (show absolute chunk number, not count)
                            elapsed = (datetime.now() - t0).total_seconds()
                            rate = total_rows / elapsed if elapsed > 0 else 0
                            logger.info(f"    Chunk {chunk_num + 1}: {total_rows:,} rows loaded ({rate:.0f} rows/sec)")

                        except Exception as e:
                            logger.error(f"    Chunk {chunk_num + 1} failed: {e}")
                            # Continue with next chunk

                        # Clean up chunk file to save disk space
                        chunk_file.unlink()
                        chunk_num += 1

                elapsed = (datetime.now() - t0).total_seconds()
                logger.info(f"    Done in {elapsed:.1f}s ({total_rows:,} rows, {chunk_count} chunks)")
                return 1

            except Exception as e:
                logger.error(f"    Batch loading error: {e}")
                import traceback
                traceback.print_exc()
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

        # Load predicate tables
        logger.info("")
        logger.info("  Loading predicate tables...")
        copy_csv("Predicate", "predicates.csv", is_node=True)
        copy_csv("HAS_PREDICATE", "has_predicate.csv", is_node=False)

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
        conceptnet_path: Optional[Path] = None,
    ):
        """
        Phase 2: Load semantic relations into Kuzu using CSV bulk loading.

        Creates:
        - IS_SYNONYM edges (from ReVo + curated)
        - IS_HYPERNYM edges (from ReVo)
        - IS_ANTONYM edges (from ReVo)
        - ConceptNet relations (CN_IS_A, CN_SYNONYM, etc.) - optional
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

            for syn in synonyms:
                root1 = syn['source'].lower()
                root2 = syn['target'].lower()

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

            for hyp in hypernyms:
                specific = hyp['source'].lower()
                general = hyp['target'].lower()

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

            for ant in antonyms:
                root1 = ant['source'].lower()
                root2 = ant['target'].lower()

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

        # Load ConceptNet relations if provided
        if conceptnet_path and conceptnet_path.exists():
            logger.info("")
            logger.info("Loading ConceptNet relations...")
            self._load_conceptnet(conceptnet_path)
        elif conceptnet_path:
            logger.warning(f"ConceptNet data not found at {conceptnet_path}, skipping")

        # Mark phase 2 complete
        self.progress['phase2_complete'] = True
        self._save_progress()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Phase 2 complete in {elapsed:.1f}s")

    def _load_conceptnet(self, conceptnet_path: Path):
        """Load ConceptNet relations by calling the specialized loader script."""
        import subprocess

        script_path = Path(__file__).parent / "load_conceptnet_to_kuzu.py"
        if not script_path.exists():
            logger.warning(f"ConceptNet loader script not found: {script_path}")
            return

        # Close database connection to avoid lock conflict with subprocess
        logger.info("  Closing database connection for subprocess...")
        self.close()

        # Call the loader script
        cmd = [
            sys.executable,
            str(script_path),
            "--kuzu-db", str(self.db_path),
            "--conceptnet-csv", str(conceptnet_path),
            "--temp-dir", str(self.temp_dir / "conceptnet"),
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("  ConceptNet loading complete")
            # Log any output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"    {line}")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ConceptNet loading failed: {e}")
            if e.stdout:
                logger.error(f"  stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"  stderr: {e.stderr}")
        finally:
            # Reopen database connection
            logger.info("  Reopening database connection...")
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)

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
        import gc

        # Explicitly close connection and database before releasing references
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        if self.db:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

        # Force garbage collection to release database locks immediately
        gc.collect()

        # Small delay to ensure OS releases file locks
        import time
        time.sleep(0.1)


def build_index(
    corpus_path: Path,
    output_path: Path,
    revo_path: Path,
    curated_path: Optional[Path] = None,
    conceptnet_path: Optional[Path] = None,
    limit: Optional[int] = None,
    fresh: bool = False,
    phase: Optional[int] = None,
):
    """Build complete Kuzu index with all phases."""
    logger.info("=" * 60)
    logger.info("Building Kuzu Graph Index (CSV Bulk Loading)")
    logger.info("=" * 60)
    logger.info(f"Corpus:     {corpus_path}")
    logger.info(f"Output:     {output_path}")
    logger.info(f"ReVo:       {revo_path}")
    if conceptnet_path:
        logger.info(f"ConceptNet: {conceptnet_path}")
    if limit:
        logger.info(f"Limit:      {limit:,} documents")
    if phase:
        logger.info(f"Phase:      {phase}")
    logger.info("")

    builder = KuzuIndexBuilder(output_path)

    try:
        # Initialize database
        builder._init_database(fresh=fresh)

        # Phase 1: Build inverted index
        if phase is None or phase == 1:
            builder.phase1_build_inverted_index(corpus_path, limit=limit)

        # Phase 2: Load semantic relations (ReVo + ConceptNet)
        if phase is None or phase == 2:
            builder.phase2_load_semantic_relations(revo_path, curated_path, conceptnet_path)

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
        "--conceptnet",
        type=Path,
        default=Path("data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz"),
        help="Path to ConceptNet assertions CSV (optional)",
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
        conceptnet_path=args.conceptnet if args.conceptnet.exists() else None,
        limit=args.limit,
        fresh=args.fresh,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
