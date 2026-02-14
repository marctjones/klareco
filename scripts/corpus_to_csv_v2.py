#!/usr/bin/env python3
"""
Convert corpus JSONL to CSV files for batch loading into Kuzu.

This is MUCH faster than individual CREATE statements:
- Generates CSV files for each node/relationship type
- Kuzu's COPY FROM can load millions of rows in seconds
- Expected speedup: 100-1000x

Output files:
- nodes/SourceCollection.csv
- nodes/Document.csv
- nodes/Sentence.csv
- nodes/AST.csv
- nodes/Frazo.csv
- nodes/Vortgrupo.csv
- nodes/Vorto.csv
- nodes/Root.csv
- rels/IN_COLLECTION.csv
- rels/SENTENCE_HAS_AST.csv
- rels/AST_HAS_FRAZO.csv
- ... (all relationship types)
"""

import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorpusToCSVConverter:
    """Convert corpus to CSV files for batch loading."""

    def __init__(self, output_dir: Path):
        """Initialize converter."""
        self.output_dir = output_dir
        self.nodes_dir = output_dir / "nodes"
        self.rels_dir = output_dir / "rels"

        # Create directories
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.rels_dir.mkdir(parents=True, exist_ok=True)

        # ID counters
        self.collection_ids: Dict[str, int] = {}
        self.next_collection_id = 1
        self.next_document_id = 1
        self.next_sentence_id = 1
        self.next_ast_id = 1
        self.next_frazo_id = 1
        self.next_vortgrupo_id = 1
        self.next_vorto_id = 1

        # Root statistics
        self.root_doc_freq: Counter = Counter()
        self.root_total_freq: Counter = Counter()
        self.roots_in_current_doc: Set[str] = set()

        # CSV writers
        self.writers = {}
        self.files = {}

    def open_csv_writer(self, name: str, is_node: bool, headers: List[str]):
        """Open a CSV writer for a node or relationship type."""
        dir_path = self.nodes_dir if is_node else self.rels_dir
        file_path = dir_path / f"{name}.csv"

        f = open(file_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        writer.writerow(headers)

        self.files[name] = f
        self.writers[name] = writer
        return writer

    def close_all(self):
        """Close all CSV files."""
        for f in self.files.values():
            f.close()

    def escape_json(self, obj) -> str:
        """Escape JSON for CSV."""
        return json.dumps(obj) if obj else ''

    def process_corpus(self, corpus_path: Path, max_entries: int = None):
        """Process corpus and generate CSV files."""
        logger.info(f"Processing corpus from {corpus_path}")

        # Open all CSV writers
        self.writers['SourceCollection'] = self.open_csv_writer(
            'SourceCollection', True,
            ['id', 'name', 'source_type', 'language', 'metadata']
        )

        self.writers['Document'] = self.open_csv_writer(
            'Document', True,
            ['id', 'collection_id', 'title', 'external_id', 'doc_type', 'author', 'year', 'quality', 'metadata']
        )

        self.writers['Sentence'] = self.open_csv_writer(
            'Sentence', True,
            ['id', 'paragraph_id', 'text', 'sentence_order', 'global_order']
        )

        self.writers['AST'] = self.open_csv_writer(
            'AST', True,
            ['id', 'sentence_id', 'version', 'created_at', 'created_by', 'is_current',
             'fraztipo', 'demandotipo', 'negita', 'total_words', 'esperanto_words',
             'non_esperanto_words', 'success_rate', 'parse_categories']
        )

        self.writers['Frazo'] = self.open_csv_writer(
            'Frazo', True,
            ['id', 'ast_id', 'tipo']
        )

        self.writers['Vortgrupo'] = self.open_csv_writer(
            'Vortgrupo', True,
            ['id', 'ast_id', 'tipo']
        )

        self.writers['Vorto'] = self.open_csv_writer(
            'Vorto', True,
            ['id', 'ast_id', 'plena_vorto', 'radiko', 'vortspeco', 'nombro', 'kazo',
             'tempo', 'modo', 'participo_voco', 'participo_tempo', 'prefiksoj', 'sufiksoj',
             'parse_status', 'parse_error', 'category', 'proper_noun_category', 'proper_noun_frequency',
             'korelativo_prefikso', 'korelativo_sufikso', 'korelativo_signifo',
             'estas_kunmetita', 'kunmetitaj_radikoj']
        )

        # Relationship writers
        self.writers['IN_COLLECTION'] = self.open_csv_writer(
            'IN_COLLECTION', False,
            ['from', 'to']
        )

        self.writers['SENTENCE_HAS_AST'] = self.open_csv_writer(
            'SENTENCE_HAS_AST', False,
            ['from', 'to', 'is_current']
        )

        self.writers['AST_HAS_FRAZO'] = self.open_csv_writer(
            'AST_HAS_FRAZO', False,
            ['from', 'to']
        )

        self.writers['HAS_SUBJEKTO_VORTGRUPO'] = self.open_csv_writer(
            'HAS_SUBJEKTO_VORTGRUPO', False,
            ['from', 'to']
        )

        self.writers['HAS_SUBJEKTO_VORTO'] = self.open_csv_writer(
            'HAS_SUBJEKTO_VORTO', False,
            ['from', 'to']
        )

        self.writers['HAS_VERBO'] = self.open_csv_writer(
            'HAS_VERBO', False,
            ['from', 'to']
        )

        self.writers['HAS_OBJEKTO_VORTGRUPO'] = self.open_csv_writer(
            'HAS_OBJEKTO_VORTGRUPO', False,
            ['from', 'to']
        )

        self.writers['HAS_OBJEKTO_VORTO'] = self.open_csv_writer(
            'HAS_OBJEKTO_VORTO', False,
            ['from', 'to']
        )

        self.writers['HAS_ALIAJ'] = self.open_csv_writer(
            'HAS_ALIAJ', False,
            ['from', 'to', 'position']
        )

        self.writers['HAS_KERNO'] = self.open_csv_writer(
            'HAS_KERNO', False,
            ['from', 'to']
        )

        self.writers['HAS_PRISKRIBO'] = self.open_csv_writer(
            'HAS_PRISKRIBO', False,
            ['from', 'to', 'position']
        )

        # Process corpus entries
        with open(corpus_path) as f:
            for i, line in enumerate(f, 1):
                if max_entries and i > max_entries:
                    break

                entry = json.loads(line)

                try:
                    self.process_entry(entry)
                except Exception as e:
                    logger.error(f"Failed to process entry {i}: {e}")
                    continue

                if i % 10000 == 0:
                    logger.info(f"Processed {i} entries...")

        logger.info(f"Processed {i} total entries")

        # Generate Root CSV
        self.generate_root_csv()

        self.close_all()
        logger.info("CSV generation complete")

    def process_entry(self, entry: Dict):
        """Process a single corpus entry."""
        source = entry['source']
        text = entry['text']
        ast_dict = entry['ast']

        # Reset roots for this document
        self.roots_in_current_doc = set()

        # Create collection (if new)
        collection_id = self.get_or_create_collection(source)

        # Create document
        doc_id = self.create_document(source, collection_id)

        # Create sentence
        sentence_id = self.create_sentence(text, doc_id)

        # Create AST
        ast_id = self.create_ast(ast_dict, sentence_id)

        # Create Frazo
        frazo_id = self.create_frazo(ast_dict, ast_id)

        # Update root doc frequency
        for root in self.roots_in_current_doc:
            self.root_doc_freq[root] += 1

    def get_or_create_collection(self, source: Dict) -> int:
        """Get or create collection ID."""
        name = source.get('name', 'unknown')

        if name in self.collection_ids:
            return self.collection_ids[name]

        collection_id = self.next_collection_id
        self.next_collection_id += 1

        self.writers['SourceCollection'].writerow([
            collection_id,
            name,
            source.get('source_type', 'unknown'),
            'eo',
            self.escape_json(source)
        ])

        self.collection_ids[name] = collection_id
        return collection_id

    def create_document(self, source: Dict, collection_id: int) -> int:
        """Create document row."""
        doc_id = self.next_document_id
        self.next_document_id += 1

        title = source.get('source_name', source.get('name', 'unknown'))
        author = source.get('author', 'unknown')
        year = source.get('year', 0)
        quality = source.get('quality', 'BRONZE')

        self.writers['Document'].writerow([
            doc_id,
            collection_id,
            title,
            source.get('name', ''),
            source.get('sentence_type', 'text'),
            author,
            year,
            quality,
            self.escape_json(source)
        ])

        # Add relationship
        self.writers['IN_COLLECTION'].writerow([doc_id, collection_id])

        return doc_id

    def create_sentence(self, text: str, doc_id: int) -> int:
        """Create sentence row."""
        sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        self.writers['Sentence'].writerow([
            sentence_id,
            doc_id,
            text,
            1,
            sentence_id
        ])

        return sentence_id

    def create_ast(self, ast_dict: Dict, sentence_id: int) -> int:
        """Create AST row."""
        ast_id = self.next_ast_id
        self.next_ast_id += 1

        stats = ast_dict.get('parse_statistics', {})

        self.writers['AST'].writerow([
            ast_id,
            sentence_id,
            1,  # version
            '2026-02-13T20:00:00',  # created_at (placeholder)
            'corpus_to_csv_v2.py',
            'true',  # is_current
            ast_dict.get('fraztipo', 'deklaro'),
            ast_dict.get('demandotipo', ''),
            'true' if ast_dict.get('negita') else 'false',
            stats.get('total_words', 0),
            stats.get('esperanto_words', 0),
            stats.get('non_esperanto_words', 0),
            stats.get('success_rate', 0.0),
            self.escape_json(stats.get('parse_categories', {}))
        ])

        # Add relationship
        self.writers['SENTENCE_HAS_AST'].writerow([sentence_id, ast_id, 'true'])

        return ast_id

    def create_frazo(self, ast_dict: Dict, ast_id: int) -> int:
        """Create Frazo row and structure."""
        frazo_id = self.next_frazo_id
        self.next_frazo_id += 1

        self.writers['Frazo'].writerow([frazo_id, ast_id, 'frazo'])

        # Add relationship
        self.writers['AST_HAS_FRAZO'].writerow([ast_id, frazo_id])

        # Process subjekto
        if 'subjekto' in ast_dict and ast_dict['subjekto'] is not None:
            subjekto = ast_dict['subjekto']
            if subjekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(subjekto, ast_id)
                self.writers['HAS_SUBJEKTO_VORTGRUPO'].writerow([frazo_id, vg_id])
            elif subjekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(subjekto, ast_id)
                self.writers['HAS_SUBJEKTO_VORTO'].writerow([frazo_id, vorto_id])

        # Process verbo
        if 'verbo' in ast_dict and ast_dict['verbo'] is not None:
            verbo_id = self.create_vorto(ast_dict['verbo'], ast_id)
            self.writers['HAS_VERBO'].writerow([frazo_id, verbo_id])

        # Process objekto
        if 'objekto' in ast_dict and ast_dict['objekto'] is not None:
            objekto = ast_dict['objekto']
            if objekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(objekto, ast_id)
                self.writers['HAS_OBJEKTO_VORTGRUPO'].writerow([frazo_id, vg_id])
            elif objekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(objekto, ast_id)
                self.writers['HAS_OBJEKTO_VORTO'].writerow([frazo_id, vorto_id])

        # Process aliaj
        if 'aliaj' in ast_dict:
            for position, vorto in enumerate(ast_dict['aliaj']):
                vorto_id = self.create_vorto(vorto, ast_id)
                self.writers['HAS_ALIAJ'].writerow([frazo_id, vorto_id, position])

        return frazo_id

    def create_vortgrupo(self, vg_dict: Dict, ast_id: int) -> int:
        """Create Vortgrupo row."""
        vg_id = self.next_vortgrupo_id
        self.next_vortgrupo_id += 1

        self.writers['Vortgrupo'].writerow([vg_id, ast_id, 'vortgrupo'])

        # Create kerno
        if 'kerno' in vg_dict:
            kerno_id = self.create_vorto(vg_dict['kerno'], ast_id)
            self.writers['HAS_KERNO'].writerow([vg_id, kerno_id])

        # Create priskriboj
        if 'priskriboj' in vg_dict:
            for position, priskribo in enumerate(vg_dict['priskriboj']):
                priskribo_id = self.create_vorto(priskribo, ast_id)
                self.writers['HAS_PRISKRIBO'].writerow([vg_id, priskribo_id, position])

        return vg_id

    def create_vorto(self, vorto_dict: Dict, ast_id: int) -> int:
        """Create Vorto row."""
        vorto_id = self.next_vorto_id
        self.next_vorto_id += 1

        # Track root
        radiko = vorto_dict.get('radiko', '')
        if radiko:
            self.roots_in_current_doc.add(radiko)
            self.root_total_freq[radiko] += 1

        self.writers['Vorto'].writerow([
            vorto_id,
            ast_id,
            vorto_dict.get('plena_vorto', ''),
            radiko,
            vorto_dict.get('vortspeco', ''),
            vorto_dict.get('nombro', ''),
            vorto_dict.get('kazo', ''),
            vorto_dict.get('tempo', ''),
            vorto_dict.get('modo', ''),
            vorto_dict.get('participo_voco', ''),
            vorto_dict.get('participo_tempo', ''),
            self.escape_json(vorto_dict.get('prefiksoj', [])),
            self.escape_json(vorto_dict.get('sufiksoj', [])),
            vorto_dict.get('parse_status', 'unknown'),
            vorto_dict.get('parse_error', ''),
            vorto_dict.get('category', ''),
            vorto_dict.get('proper_noun_category', ''),
            vorto_dict.get('proper_noun_frequency', 0),
            vorto_dict.get('korelativo_prefikso', ''),
            vorto_dict.get('korelativo_sufikso', ''),
            vorto_dict.get('korelativo_signifo', ''),
            'true' if vorto_dict.get('estas_kunmetita') else 'false',
            self.escape_json(vorto_dict.get('kunmetitaj_radikoj', []))
        ])

        return vorto_id

    def generate_root_csv(self):
        """Generate Root node and relationship CSVs."""
        logger.info("Generating Root CSV...")

        # Root nodes
        root_writer = self.open_csv_writer(
            'Root', True,
            ['root', 'doc_freq', 'total_freq']
        )

        # HAS_ROOT relationships
        has_root_writer = self.open_csv_writer(
            'HAS_ROOT', False,
            ['from', 'to', 'is_primary', 'position']
        )

        all_roots = set(self.root_total_freq.keys())

        for root in all_roots:
            root_writer.writerow([
                root,
                self.root_doc_freq[root],
                self.root_total_freq[root]
            ])

        logger.info(f"Generated {len(all_roots)} roots")

        # Note: HAS_ROOT relationships will be created in load script
        # by matching Vorto.radiko to Root.root


def main():
    parser = argparse.ArgumentParser(description='Convert corpus to CSV files')
    parser.add_argument('--corpus', type=Path, required=True, help='Path to corpus JSONL')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for CSVs')
    parser.add_argument('--max-entries', type=int, help='Maximum entries to process')

    args = parser.parse_args()

    # Clean output directory
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)

    converter = CorpusToCSVConverter(args.output)
    converter.process_corpus(args.corpus, args.max_entries)


if __name__ == '__main__':
    main()
