#!/usr/bin/env python3
"""
Index Esperanto corpus into v2.0 AST-native Kuzu graph database.

This script converts the existing unified_corpus.jsonl into the v2.0 schema:
- Document hierarchy: Collection → Document → Sentence
- AST nodes: AST → Frazo → Vortgrupo/Vorto
- Root index: Esperanto roots with frequency stats

Usage:
    python scripts/index_corpus_v2.py --corpus data/corpus/unified_corpus.jsonl \
                                       --vocab data/vocabularies/root_vocab.json \
                                       --output data/indexes/v2_kuzu_index
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import Counter

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


class CorpusIndexer:
    """Index corpus into v2.0 AST-native Kuzu database."""

    def __init__(self, db_path: Path):
        """Initialize indexer with database path."""
        self.db_path = db_path
        self.db = None
        self.conn = None

        # ID generators (in-memory counters for this indexing run)
        self.collection_ids: Dict[str, int] = {}
        self.document_ids: Dict[str, int] = {}
        self.next_collection_id = 1
        self.next_document_id = 1
        self.next_sentence_id = 1
        self.next_ast_id = 1
        self.next_frazo_id = 1
        self.next_vortgrupo_id = 1
        self.next_vorto_id = 1

        # Root statistics
        self.root_doc_freq: Counter = Counter()  # How many docs contain each root
        self.root_total_freq: Counter = Counter()  # Total occurrences
        self.roots_in_current_doc: Set[str] = set()

    def connect(self):
        """Connect to Kuzu database."""
        logger.info(f"Opening Kuzu database at {self.db_path}")
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

    def create_schema(self):
        """Create v2.0 schema in database."""
        logger.info("Creating v2.0 schema...")
        statements = get_create_statements()
        for i, stmt in enumerate(statements, 1):
            logger.debug(f"Executing statement {i}/{len(statements)}")
            self.conn.execute(stmt)
        logger.info(f"Schema created: {len(statements)} statements executed")

    def get_or_create_collection(self, source: Dict) -> int:
        """Get or create SourceCollection node."""
        name = source.get('name', 'unknown')

        if name in self.collection_ids:
            return self.collection_ids[name]

        collection_id = self.next_collection_id
        self.next_collection_id += 1

        self.conn.execute(f"""
            CREATE (c:SourceCollection {{
                id: {collection_id},
                name: '{name}',
                source_type: '{source.get('source_type', 'unknown')}',
                language: 'eo',
                metadata: '{json.dumps(source)}'
            }})
        """)

        self.collection_ids[name] = collection_id
        logger.debug(f"Created SourceCollection: {name} (id={collection_id})")
        return collection_id

    def create_document(self, source: Dict, collection_id: int) -> int:
        """Create Document node."""
        doc_id = self.next_document_id
        self.next_document_id += 1

        # Extract document metadata
        title = source.get('source_name', source.get('name', 'unknown'))
        author = source.get('author', 'unknown')
        year = source.get('year', 0)
        quality = source.get('quality', 'BRONZE')

        self.conn.execute(f"""
            CREATE (d:Document {{
                id: {doc_id},
                collection_id: {collection_id},
                title: '{title}',
                external_id: '{source.get('name', '')}',
                doc_type: '{source.get('sentence_type', 'text')}',
                author: '{author}',
                year: {year},
                quality: '{quality}',
                metadata: '{json.dumps(source)}'
            }})
        """)

        # Link to collection
        self.conn.execute(f"""
            MATCH (d:Document), (c:SourceCollection)
            WHERE d.id = {doc_id} AND c.id = {collection_id}
            CREATE (d)-[:IN_COLLECTION]->(c)
        """)

        logger.debug(f"Created Document: {title} (id={doc_id})")
        return doc_id

    def create_sentence(self, text: str, doc_id: int) -> int:
        """Create Sentence node."""
        sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        # Escape single quotes in text
        escaped_text = text.replace("'", "\\'")

        self.conn.execute(f"""
            CREATE (s:Sentence {{
                id: {sentence_id},
                paragraph_id: {doc_id},
                text: '{escaped_text}',
                sentence_order: 1,
                global_order: {sentence_id}
            }})
        """)

        return sentence_id

    def create_ast(self, ast_dict: Dict, sentence_id: int) -> int:
        """Create AST node with metadata."""
        ast_id = self.next_ast_id
        self.next_ast_id += 1

        # Extract AST-level metadata
        fraztipo = ast_dict.get('fraztipo', 'deklaro')
        demandotipo = ast_dict.get('demandotipo')
        negita = ast_dict.get('negita', False)

        # Extract parse statistics
        stats = ast_dict.get('parse_statistics', {})
        total_words = stats.get('total_words', 0)
        esperanto_words = stats.get('esperanto_words', 0)
        non_esperanto = stats.get('non_esperanto_words', 0)
        success_rate = stats.get('success_rate', 0.0)
        parse_categories = json.dumps(stats.get('parse_categories', {}))

        # Escape single quotes
        parse_categories = parse_categories.replace("'", "\\'")

        self.conn.execute(f"""
            CREATE (ast:AST {{
                id: {ast_id},
                sentence_id: {sentence_id},
                version: 1,
                created_at: timestamp('{datetime.now().isoformat()}'),
                created_by: 'index_corpus_v2.py',
                is_current: true,
                fraztipo: '{fraztipo}',
                demandotipo: {'NULL' if demandotipo is None else f"'{demandotipo}'"},
                negita: {str(negita).lower()},
                total_words: {total_words},
                esperanto_words: {esperanto_words},
                non_esperanto_words: {non_esperanto},
                success_rate: {success_rate},
                parse_categories: '{parse_categories}'
            }})
        """)

        # Link AST to Sentence
        self.conn.execute(f"""
            MATCH (s:Sentence), (ast:AST)
            WHERE s.id = {sentence_id} AND ast.id = {ast_id}
            CREATE (s)-[:SENTENCE_HAS_AST {{is_current: true}}]->(ast)
        """)

        return ast_id

    def create_frazo(self, ast_dict: Dict, ast_id: int) -> int:
        """Create Frazo node and its structure."""
        frazo_id = self.next_frazo_id
        self.next_frazo_id += 1

        self.conn.execute(f"""
            CREATE (f:Frazo {{
                id: {frazo_id},
                ast_id: {ast_id},
                tipo: 'frazo'
            }})
        """)

        # Link Frazo to AST
        self.conn.execute(f"""
            MATCH (ast:AST), (f:Frazo)
            WHERE ast.id = {ast_id} AND f.id = {frazo_id}
            CREATE (ast)-[:AST_HAS_FRAZO]->(f)
        """)

        # Create subjekto (can be Vortgrupo or Vorto)
        if 'subjekto' in ast_dict and ast_dict['subjekto'] is not None:
            subjekto = ast_dict['subjekto']
            if subjekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(subjekto, ast_id)
                self.conn.execute(f"""
                    MATCH (f:Frazo), (vg:Vortgrupo)
                    WHERE f.id = {frazo_id} AND vg.id = {vg_id}
                    CREATE (f)-[:HAS_SUBJEKTO_VORTGRUPO]->(vg)
                """)
            elif subjekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(subjekto, ast_id)
                self.conn.execute(f"""
                    MATCH (f:Frazo), (v:Vorto)
                    WHERE f.id = {frazo_id} AND v.id = {vorto_id}
                    CREATE (f)-[:HAS_SUBJEKTO_VORTO]->(v)
                """)

        # Create verbo (always Vorto)
        if 'verbo' in ast_dict and ast_dict['verbo'] is not None:
            verbo_id = self.create_vorto(ast_dict['verbo'], ast_id)
            self.conn.execute(f"""
                MATCH (f:Frazo), (v:Vorto)
                WHERE f.id = {frazo_id} AND v.id = {verbo_id}
                CREATE (f)-[:HAS_VERBO]->(v)
            """)

        # Create objekto (can be Vortgrupo or Vorto)
        if 'objekto' in ast_dict and ast_dict['objekto'] is not None:
            objekto = ast_dict['objekto']
            if objekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(objekto, ast_id)
                self.conn.execute(f"""
                    MATCH (f:Frazo), (vg:Vortgrupo)
                    WHERE f.id = {frazo_id} AND vg.id = {vg_id}
                    CREATE (f)-[:HAS_OBJEKTO_VORTGRUPO]->(vg)
                """)
            elif objekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(objekto, ast_id)
                self.conn.execute(f"""
                    MATCH (f:Frazo), (v:Vorto)
                    WHERE f.id = {frazo_id} AND v.id = {vorto_id}
                    CREATE (f)-[:HAS_OBJEKTO_VORTO]->(v)
                """)

        # Create aliaj (modifiers, adverbs)
        if 'aliaj' in ast_dict:
            for position, vorto in enumerate(ast_dict['aliaj']):
                vorto_id = self.create_vorto(vorto, ast_id)
                self.conn.execute(f"""
                    MATCH (f:Frazo), (v:Vorto)
                    WHERE f.id = {frazo_id} AND v.id = {vorto_id}
                    CREATE (f)-[:HAS_ALIAJ {{position: {position}}}]->(v)
                """)

        return frazo_id

    def create_vortgrupo(self, vg_dict: Dict, ast_id: int) -> int:
        """Create Vortgrupo node."""
        vg_id = self.next_vortgrupo_id
        self.next_vortgrupo_id += 1

        self.conn.execute(f"""
            CREATE (vg:Vortgrupo {{
                id: {vg_id},
                ast_id: {ast_id},
                tipo: 'vortgrupo'
            }})
        """)

        # Create kerno (core word)
        if 'kerno' in vg_dict:
            kerno_id = self.create_vorto(vg_dict['kerno'], ast_id)
            self.conn.execute(f"""
                MATCH (vg:Vortgrupo), (v:Vorto)
                WHERE vg.id = {vg_id} AND v.id = {kerno_id}
                CREATE (vg)-[:HAS_KERNO]->(v)
            """)

        # Create priskriboj (modifiers)
        if 'priskriboj' in vg_dict:
            for position, priskribo in enumerate(vg_dict['priskriboj']):
                priskribo_id = self.create_vorto(priskribo, ast_id)
                self.conn.execute(f"""
                    MATCH (vg:Vortgrupo), (v:Vorto)
                    WHERE vg.id = {vg_id} AND v.id = {priskribo_id}
                    CREATE (vg)-[:HAS_PRISKRIBO {{position: {position}}}]->(v)
                """)

        return vg_id

    def create_vorto(self, vorto_dict: Dict, ast_id: int) -> int:
        """Create Vorto node."""
        vorto_id = self.next_vorto_id
        self.next_vorto_id += 1

        # Extract all fields
        plena_vorto = vorto_dict.get('plena_vorto', '').replace("'", "\\'")
        radiko = vorto_dict.get('radiko', '').replace("'", "\\'")
        vortspeco = vorto_dict.get('vortspeco', '').replace("'", "\\'")
        nombro = vorto_dict.get('nombro', '')
        kazo = vorto_dict.get('kazo', '')
        tempo = vorto_dict.get('tempo', '')
        modo = vorto_dict.get('modo', '')
        participo_voco = vorto_dict.get('participo_voco', '')
        participo_tempo = vorto_dict.get('participo_tempo', '')
        prefiksoj = json.dumps(vorto_dict.get('prefiksoj', [])).replace("'", "\\'")
        sufiksoj = json.dumps(vorto_dict.get('sufiksoj', [])).replace("'", "\\'")
        parse_status = vorto_dict.get('parse_status', 'unknown')
        parse_error = vorto_dict.get('parse_error', '').replace("'", "\\'")
        category = vorto_dict.get('category', '').replace("'", "\\'")
        proper_noun_category = vorto_dict.get('proper_noun_category', '')
        proper_noun_frequency = vorto_dict.get('proper_noun_frequency', 0)
        korelativo_prefikso = vorto_dict.get('korelativo_prefikso', '')
        korelativo_sufikso = vorto_dict.get('korelativo_sufikso', '')
        korelativo_signifo = vorto_dict.get('korelativo_signifo', '')
        estas_kunmetita = vorto_dict.get('estas_kunmetita', False)
        kunmetitaj_radikoj = json.dumps(vorto_dict.get('kunmetitaj_radikoj', [])).replace("'", "\\'")

        # Helper to format NULL values
        def fmt(val):
            if val is None or val == '':
                return 'NULL'
            elif isinstance(val, bool):
                return str(val).lower()
            elif isinstance(val, (int, float)):
                return str(val)
            else:
                return f"'{val}'"

        self.conn.execute(f"""
            CREATE (v:Vorto {{
                id: {vorto_id},
                ast_id: {ast_id},
                plena_vorto: '{plena_vorto}',
                radiko: '{radiko}',
                vortspeco: '{vortspeco}',
                nombro: {fmt(nombro)},
                kazo: {fmt(kazo)},
                tempo: {fmt(tempo)},
                modo: {fmt(modo)},
                participo_voco: {fmt(participo_voco)},
                participo_tempo: {fmt(participo_tempo)},
                prefiksoj: '{prefiksoj}',
                sufiksoj: '{sufiksoj}',
                parse_status: '{parse_status}',
                parse_error: {fmt(parse_error)},
                category: {fmt(category)},
                proper_noun_category: {fmt(proper_noun_category)},
                proper_noun_frequency: {proper_noun_frequency},
                korelativo_prefikso: {fmt(korelativo_prefikso)},
                korelativo_sufikso: {fmt(korelativo_sufikso)},
                korelativo_signifo: {fmt(korelativo_signifo)},
                estas_kunmetita: {str(estas_kunmetita).lower()},
                kunmetitaj_radikoj: '{kunmetitaj_radikoj}'
            }})
        """)

        # Track root for statistics
        if radiko:
            self.roots_in_current_doc.add(radiko)
            self.root_total_freq[radiko] += 1

        return vorto_id

    def index_entry(self, entry: Dict):
        """Index a single corpus entry."""
        source = entry['source']
        text = entry['text']
        ast_dict = entry['ast']

        # Reset roots for this document
        self.roots_in_current_doc = set()

        # Create document hierarchy
        collection_id = self.get_or_create_collection(source)
        doc_id = self.create_document(source, collection_id)
        sentence_id = self.create_sentence(text, doc_id)

        # Create AST structure
        ast_id = self.create_ast(ast_dict, sentence_id)
        frazo_id = self.create_frazo(ast_dict, ast_id)

        # Update document frequency for roots
        for root in self.roots_in_current_doc:
            self.root_doc_freq[root] += 1

    def build_root_index(self, vocab_path: Optional[Path] = None):
        """Build Root index from vocabulary and corpus statistics."""
        logger.info("Building Root index...")

        # Collect all roots from vocabulary + corpus
        all_roots = set(self.root_total_freq.keys())

        if vocab_path and vocab_path.exists():
            with open(vocab_path) as f:
                vocab_roots = json.load(f)
                all_roots.update(vocab_roots)

        # Create Root nodes
        for root in all_roots:
            doc_freq = self.root_doc_freq[root]
            total_freq = self.root_total_freq[root]

            escaped_root = root.replace("'", "\\'")
            self.conn.execute(f"""
                CREATE (r:Root {{
                    root: '{escaped_root}',
                    doc_freq: {doc_freq},
                    total_freq: {total_freq}
                }})
            """)

        logger.info(f"Created {len(all_roots)} Root nodes")

        # Link Vorto nodes to Root nodes
        logger.info("Linking Vorto nodes to Roots...")
        self.conn.execute("""
            MATCH (v:Vorto), (r:Root)
            WHERE v.radiko = r.root
            CREATE (v)-[:HAS_ROOT {is_primary: true, position: 0}]->(r)
        """)

        logger.info("Root index built successfully")

    def index_corpus(self, corpus_path: Path, max_entries: Optional[int] = None):
        """Index entire corpus."""
        logger.info(f"Indexing corpus from {corpus_path}")

        with open(corpus_path) as f:
            for i, line in enumerate(f, 1):
                if max_entries and i > max_entries:
                    break

                entry = json.loads(line)

                try:
                    self.index_entry(entry)
                except Exception as e:
                    logger.error(f"Failed to index entry {i}: {e}")
                    logger.debug(f"Entry text: {entry.get('text', 'N/A')[:100]}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue

                if i % 100 == 0:
                    logger.info(f"Indexed {i} entries...")

        logger.info(f"Indexed {i} total entries")

    def get_stats(self) -> Dict:
        """Get indexing statistics."""
        stats = {}

        # Count nodes
        for node_type in ['SourceCollection', 'Document', 'Sentence', 'AST', 'Frazo', 'Vortgrupo', 'Vorto', 'Root']:
            result = self.conn.execute(f"MATCH (n:{node_type}) RETURN count(n)")
            count = result.get_next()[0]
            stats[f"{node_type}_count"] = count

        return stats


def main():
    parser = argparse.ArgumentParser(description='Index corpus into v2.0 Kuzu database')
    parser.add_argument('--corpus', type=Path, required=True, help='Path to corpus JSONL file')
    parser.add_argument('--vocab', type=Path, help='Path to root vocabulary JSON file')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for Kuzu database')
    parser.add_argument('--max-entries', type=int, help='Maximum entries to index (for testing)')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (delete existing database)')

    args = parser.parse_args()

    # Check if database exists
    if args.output.exists() and args.fresh:
        logger.info(f"Removing existing database at {args.output}")
        import shutil
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()

    # Create indexer
    indexer = CorpusIndexer(args.output)
    indexer.connect()

    # Create schema
    indexer.create_schema()

    # Index corpus
    indexer.index_corpus(args.corpus, args.max_entries)

    # Build root index
    indexer.build_root_index(args.vocab)

    # Print statistics
    stats = indexer.get_stats()
    logger.info("=== Indexing Statistics ===")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")


if __name__ == '__main__':
    main()
