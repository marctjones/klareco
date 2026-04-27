#!/usr/bin/env python3
"""
Convert corpus JSONL to CSV files for batch loading into Kuzu (v2.1 - Pure Esperanto).

NOW WITH STRUCTURE: Extracts document hierarchy from corpus metadata:
- Wikipedia: article_id, section, section_level → Sekcio nodes
- Books: chapter, chapter_number → Sekcio nodes
- Paragraphs: Inferred from sentence sequences → Paragrafo nodes

This is MUCH faster than individual CREATE statements:
- Generates CSV files for each node/relationship type
- Kuzu's COPY FROM can load millions of rows in seconds
- Expected speedup: 100-1000x

Output files (v2.1 Pure Esperanto naming with full hierarchy):
- nodes/Fontaro.csv
- nodes/Dokumento.csv
- nodes/Sekcio.csv (NEW)
- nodes/Paragrafo.csv (NEW)
- nodes/Frazoteksto.csv
- nodes/AST.csv
- nodes/Frazo.csv
- nodes/Vortgrupo.csv
- nodes/Vorto.csv
- nodes/Radiko.csv
- rels/EN_FONTARO.csv
- rels/EN_DOKUMENTO.csv (NEW)
- rels/EN_SEKCIO.csv (NEW)
- rels/EN_PARAGRAFO.csv (NEW)
- rels/GEPATRA_SEKCIO.csv (NEW)
- rels/SEKVA_SEKCIO.csv (NEW)
- rels/SEKVA_PARAGRAFO.csv (NEW)
- rels/SEKVA_FRAZOTEKSTO.csv (NEW)
- rels/FRAZOTEKSTO_HAVAS_AST.csv
- rels/AST_HAVAS_FRAZON.csv
- ... (all AST relationship types)
"""

import argparse
import json
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorpusToCSVConverter:
    """Convert corpus to CSV files for batch loading with full document structure."""

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
        self.next_sekcio_id = 1
        self.next_paragrafo_id = 1
        self.next_sentence_id = 1
        self.next_ast_id = 1
        self.next_frazo_id = 1
        self.next_vortgrupo_id = 1
        self.next_vorto_id = 1

        # Document-level tracking
        self.current_doc_key: Optional[str] = None
        self.current_doc_id: Optional[int] = None
        self.current_section_key: Optional[str] = None
        self.current_sekcio_id: Optional[int] = None
        self.current_paragrafo_id: Optional[int] = None
        self.sentences_in_paragraph: List[Dict] = []  # Buffer for paragraph grouping
        self.paragraph_order_in_section = 0
        self.sentence_order_in_paragraph = 0
        self.sentence_order_in_document = 0

        # Section hierarchy tracking (for GEPATRA_SEKCIO)
        self.section_parent_map: Dict[int, int] = {}  # sekcio_id -> parent_sekcio_id
        self.prev_sekcio_id: Optional[int] = None  # For SEKVA_SEKCIO
        self.prev_paragrafo_id: Optional[int] = None  # For SEKVA_PARAGRAFO
        self.prev_frazoteksto_id: Optional[int] = None  # For SEKVA_FRAZOTEKSTO

        # Root statistics
        self.root_doc_freq: Counter = Counter()
        self.root_total_freq: Counter = Counter()
        self.roots_in_current_doc: Set[str] = set()

        # CSV writers
        self.writers = {}
        self.files = {}

        # Value translation maps for SOURCE metadata (English → Esperanto)
        self.quality_map = {
            'GOLD': 'ORO',
            'SILVER': 'ARĜENTO',
            'BRONZE': 'BRONZO'
        }

        self.doc_type_map = {
            'article': 'artikolo',
            'book': 'libro',
            'qa': 'demandoj-respondoj',
            'grammar_qa': 'gramatiko-demandoj',
            'grammar_reference': 'gramatiko-referenco',
            'text': 'teksto',
            'unknown': 'nekonata'
        }

    def open_csv_writer(self, name: str, is_node: bool, headers: List[str]):
        """Open a CSV writer for a node or relationship type."""
        dir_path = self.nodes_dir if is_node else self.rels_dir
        file_path = dir_path / f"{name}.csv"

        f = open(file_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
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
        """Process corpus and generate CSV files with full document structure."""
        logger.info(f"Processing corpus from {corpus_path}")

        # Open all CSV writers (v2.1 Pure Esperanto naming with hierarchy)
        self.writers['Fontaro'] = self.open_csv_writer(
            'Fontaro', True,
            ['id', 'nomo', 'fontotipo', 'lingvo', 'metadatenoj']
        )

        self.writers['Dokumento'] = self.open_csv_writer(
            'Dokumento', True,
            ['id', 'fontaro_id', 'titolo', 'ekstera_id', 'dokumentipo', 'aŭtoro', 'jaro', 'kvalito', 'metadatenoj']
        )

        self.writers['Sekcio'] = self.open_csv_writer(
            'Sekcio', True,
            ['id', 'dokumento_id', 'sekcio_nomo', 'sekcio_nivelo', 'sekcio_ordo', 'gepatra_sekcio_id']
        )

        self.writers['Paragrafo'] = self.open_csv_writer(
            'Paragrafo', True,
            ['id', 'sekcio_id', 'paragrafo_ordo', 'paragrafo_tipo', 'metadatenoj']
        )

        self.writers['Frazoteksto'] = self.open_csv_writer(
            'Frazoteksto', True,
            ['id', 'paragrafo_id', 'teksto', 'frazo_ordo', 'tutmonda_ordo']
        )

        self.writers['AST'] = self.open_csv_writer(
            'AST', True,
            ['id', 'frazoteksto_id', 'versio', 'kreita_je', 'kreita_de', 'estas_nuna',
             'fraztipo', 'demandotipo', 'negita', 'tutaj_vortoj', 'esperantaj_vortoj',
             'neesperantaj_vortoj', 'sukcesoprocento', 'analizkategorioj']
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
             'analizstato', 'analizeraro', 'kategorio', 'propranoma_kategorio', 'propranoma_ofteco',
             'korelativo_prefikso', 'korelativo_sufikso', 'korelativo_signifo',
             'estas_kunmetita', 'kunmetitaj_radikoj']
        )

        # Hierarchy relationship writers (NEW)
        self.writers['EN_FONTARO'] = self.open_csv_writer(
            'EN_FONTARO', False,
            ['from', 'to']
        )

        self.writers['EN_DOKUMENTO'] = self.open_csv_writer(
            'EN_DOKUMENTO', False,
            ['from', 'to']
        )

        self.writers['EN_SEKCIO'] = self.open_csv_writer(
            'EN_SEKCIO', False,
            ['from', 'to']
        )

        self.writers['EN_PARAGRAFO'] = self.open_csv_writer(
            'EN_PARAGRAFO', False,
            ['from', 'to']
        )

        self.writers['GEPATRA_SEKCIO'] = self.open_csv_writer(
            'GEPATRA_SEKCIO', False,
            ['from', 'to']
        )

        self.writers['SEKVA_SEKCIO'] = self.open_csv_writer(
            'SEKVA_SEKCIO', False,
            ['from', 'to']
        )

        self.writers['SEKVA_PARAGRAFO'] = self.open_csv_writer(
            'SEKVA_PARAGRAFO', False,
            ['from', 'to']
        )

        self.writers['SEKVA_FRAZOTEKSTO'] = self.open_csv_writer(
            'SEKVA_FRAZOTEKSTO', False,
            ['from', 'to']
        )

        # AST relationship writers (existing)
        self.writers['FRAZOTEKSTO_HAVAS_AST'] = self.open_csv_writer(
            'FRAZOTEKSTO_HAVAS_AST', False,
            ['from', 'to', 'estas_nuna']
        )

        self.writers['AST_HAVAS_FRAZON'] = self.open_csv_writer(
            'AST_HAVAS_FRAZON', False,
            ['from', 'to']
        )

        self.writers['HAVAS_SUBJEKTON_VORTGRUPO'] = self.open_csv_writer(
            'HAVAS_SUBJEKTON_VORTGRUPO', False,
            ['from', 'to']
        )

        self.writers['HAVAS_SUBJEKTON_VORTO'] = self.open_csv_writer(
            'HAVAS_SUBJEKTON_VORTO', False,
            ['from', 'to']
        )

        self.writers['HAVAS_VERBON'] = self.open_csv_writer(
            'HAVAS_VERBON', False,
            ['from', 'to']
        )

        self.writers['HAVAS_OBJEKTON_VORTGRUPO'] = self.open_csv_writer(
            'HAVAS_OBJEKTON_VORTGRUPO', False,
            ['from', 'to']
        )

        self.writers['HAVAS_OBJEKTON_VORTO'] = self.open_csv_writer(
            'HAVAS_OBJEKTON_VORTO', False,
            ['from', 'to']
        )

        self.writers['HAVAS_ALIAJN'] = self.open_csv_writer(
            'HAVAS_ALIAJN', False,
            ['from', 'to', 'pozicio']
        )

        self.writers['HAVAS_KERNON'] = self.open_csv_writer(
            'HAVAS_KERNON', False,
            ['from', 'to']
        )

        self.writers['HAVAS_PRISKRIBON'] = self.open_csv_writer(
            'HAVAS_PRISKRIBON', False,
            ['from', 'to', 'pozicio']
        )

        # Process corpus entries in document-aware batches
        with open(corpus_path) as f:
            entry_count = 0
            for i, line in enumerate(f, 1):
                if max_entries and i > max_entries:
                    break

                entry = json.loads(line)

                try:
                    self.process_entry(entry)
                    entry_count += 1
                except Exception as e:
                    logger.error(f"Failed to process entry {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if i % 10000 == 0:
                    logger.info(f"Processed {i} entries...")

        # Flush any remaining paragraph buffer
        self.flush_paragraph()

        logger.info(f"Processed {entry_count} total entries")

        # Generate Root CSV
        self.generate_root_csv()

        self.close_all()
        logger.info("CSV generation complete")

    def process_entry(self, entry: Dict):
        """Process a single corpus entry with document/section/paragraph awareness."""
        # Handle both old format (source is dict) and new format (source is string)
        if isinstance(entry.get('source'), dict):
            source = entry['source']
        else:
            # New format: metadata at top level, source is just a string
            source = entry  # Pass full entry as source for metadata access

        text = entry['text']

        # Parse text to create AST if not present (using FIXED parser)
        if 'ast' in entry:
            ast_dict = entry['ast']
        else:
            try:
                # Parse with fixed parser (comprehensive function word handling)
                ast_dict = parse(text)
                if not ast_dict:
                    logger.warning(f"Failed to parse: {text[:50]}...")
                    return
            except Exception as e:
                logger.warning(f"Parse error: {e} for text: {text[:50]}...")
                return

        # Extract document key (unique identifier for grouping)
        doc_key = self.get_document_key(source)

        # Check if we're switching documents
        if doc_key != self.current_doc_key:
            # Flush previous paragraph if exists
            self.flush_paragraph()

            # Update root doc frequency for previous doc
            if self.current_doc_id is not None:
                for root in self.roots_in_current_doc:
                    self.root_doc_freq[root] += 1
                self.roots_in_current_doc = set()

            # Start new document
            collection_id = self.get_or_create_collection(source)
            self.current_doc_id = self.create_document(source, collection_id)
            self.current_doc_key = doc_key
            self.current_section_key = None
            self.current_sekcio_id = None
            self.sentence_order_in_document = 0
            self.prev_sekcio_id = None

        # Extract section key (for grouping)
        section_key = self.get_section_key(source)

        # Check if we're switching sections
        if section_key != self.current_section_key:
            # Flush previous paragraph (only if we have an existing section context)
            # If current_sekcio_id is None, we just started a new document and
            # already flushed the paragraph above
            if self.current_sekcio_id is not None:
                self.flush_paragraph()

            # Create new section
            try:
                self.current_sekcio_id = self.create_sekcio(source, self.current_doc_id)
                self.current_section_key = section_key
                self.paragraph_order_in_section = 0
                self.prev_paragrafo_id = None
            except Exception as e:
                logger.error(f"Failed to create section: {e}")
                # Skip this entry if we can't create a section
                return

        # Safety check: ensure we have a section before buffering
        if self.current_sekcio_id is None:
            logger.error("Cannot buffer sentence: no section context")
            return

        # Buffer this sentence for paragraph grouping
        self.sentences_in_paragraph.append({
            'text': text,
            'ast': ast_dict,
            'source': source
        })

        # Flush paragraph every 5 sentences (heuristic for paragraph boundary)
        # OR if we have very long text (likely a new paragraph)
        if len(self.sentences_in_paragraph) >= 5:
            self.flush_paragraph()

    def flush_paragraph(self):
        """Flush buffered sentences as a complete paragraph."""
        if not self.sentences_in_paragraph:
            return

        if self.current_sekcio_id is None:
            # No section context - skip this paragraph (shouldn't happen after fix)
            logger.warning(
                f"Attempting to flush paragraph without section context "
                f"({len(self.sentences_in_paragraph)} sentences buffered). "
                f"This is a bug - sentences will be discarded."
            )
            self.sentences_in_paragraph = []
            return

        # Create paragraph
        paragrafo_id = self.create_paragrafo(self.current_sekcio_id)

        # Process all sentences in this paragraph
        self.sentence_order_in_paragraph = 0
        for sentence_data in self.sentences_in_paragraph:
            self.sentence_order_in_paragraph += 1
            self.sentence_order_in_document += 1

            sentence_id = self.create_sentence(
                sentence_data['text'],
                paragrafo_id,
                self.sentence_order_in_paragraph,
                self.sentence_order_in_document
            )

            ast_id = self.create_ast(sentence_data['ast'], sentence_id)
            self.create_frazo(sentence_data['ast'], ast_id)

        # Clear buffer
        self.sentences_in_paragraph = []

    def get_document_key(self, source: Dict) -> str:
        """Get unique document key for grouping."""
        # For Wikipedia: article_id
        if 'article_id' in source:
            return f"wikipedia_{source['article_id']}"

        # For books: source_name + chapter
        if 'chapter' in source:
            source_name = source.get('source_name', source.get('name', 'unknown'))
            return f"book_{source_name}"

        # For grammar/QA: each entry is its own "document"
        source_name = source.get('source_name', source.get('name', 'unknown'))
        return f"grammar_{source_name}_{id(source)}"

    def get_section_key(self, source: Dict) -> str:
        """Get section key for grouping."""
        # Wikipedia: section name + level
        if 'section' in source:
            section = source['section'] or 'intro'
            level = source.get('section_level', 0)
            return f"{section}_{level}"

        # Books: chapter (might be None, default to "main")
        if 'chapter' in source:
            chapter = source['chapter']
            return chapter if chapter else "main"

        # Grammar/QA: single section
        return "main"

    def get_or_create_collection(self, source: Dict) -> int:
        """Get or create collection ID."""
        name = source.get('name', 'unknown')

        if name in self.collection_ids:
            return self.collection_ids[name]

        collection_id = self.next_collection_id
        self.next_collection_id += 1

        self.writers['Fontaro'].writerow([
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

        # Title: article_title or source_name
        title = source.get('article_title') or source.get('source_name') or source.get('name', 'unknown')

        # External ID: article_id or chapter_number
        # Use article_id if present (even if 0), otherwise chapter_number
        if 'article_id' in source:
            external_id = source['article_id']
        elif 'chapter_number' in source:
            external_id = source['chapter_number']
        else:
            external_id = ''

        author = source.get('author', 'unknown')
        year = source.get('year') or 0  # Handle None values
        quality_en = source.get('quality', 'BRONZE')
        quality = self.quality_map.get(quality_en, quality_en)

        doc_type_en = source.get('source_type', 'text')
        doc_type = self.doc_type_map.get(doc_type_en, doc_type_en)

        self.writers['Dokumento'].writerow([
            doc_id,
            collection_id,
            title,
            str(external_id),
            doc_type,
            author,
            year,
            quality,
            self.escape_json(source)
        ])

        # Add relationship
        self.writers['EN_FONTARO'].writerow([doc_id, collection_id])

        return doc_id

    def roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer. Returns None if invalid."""
        if not isinstance(roman, str):
            return None

        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        prev_value = 0

        try:
            for char in reversed(roman.upper()):
                value = roman_map.get(char)
                if value is None:
                    return None
                if value < prev_value:
                    result -= value
                else:
                    result += value
                prev_value = value
            return result
        except:
            return None

    def create_sekcio(self, source: Dict, doc_id: int) -> int:
        """Create section row from metadata."""
        sekcio_id = self.next_sekcio_id
        self.next_sekcio_id += 1

        # Extract section info
        if 'section' in source:
            # Wikipedia article section
            section_name = source['section'] or 'Introduction'
            section_level = source.get('section_level') or 1
            section_order = sekcio_id  # Simple incrementing order
        elif 'chapter' in source:
            # Book chapter
            section_name = source['chapter']
            section_level = 1  # Chapters are level 1

            # chapter_number might be int, Roman numeral string, or None
            chapter_num = source.get('chapter_number')
            if isinstance(chapter_num, int):
                section_order = chapter_num
            elif isinstance(chapter_num, str):
                # Try to convert Roman numeral
                section_order = self.roman_to_int(chapter_num) or sekcio_id
            else:
                section_order = sekcio_id
        else:
            # No explicit sections - create default "main" section
            section_name = 'Main'
            section_level = 1
            section_order = 1

        # Parent section: None for now (could implement hierarchy later)
        parent_sekcio_id = None

        # Safety check: ensure all required fields are integers
        if not isinstance(section_level, int):
            logger.warning(f"section_level not int: {section_level}, using 1")
            section_level = 1
        if not isinstance(section_order, int):
            logger.warning(f"section_order not int: {section_order}, using {sekcio_id}")
            section_order = sekcio_id

        self.writers['Sekcio'].writerow([
            sekcio_id,
            doc_id,
            section_name,
            section_level,
            section_order,
            parent_sekcio_id or ''  # Empty string for NULL
        ])

        # Add relationship
        self.writers['EN_DOKUMENTO'].writerow([sekcio_id, doc_id])

        # Add SEKVA_SEKCIO relationship if there was a previous section
        if self.prev_sekcio_id is not None:
            self.writers['SEKVA_SEKCIO'].writerow([self.prev_sekcio_id, sekcio_id])

        self.prev_sekcio_id = sekcio_id

        return sekcio_id

    def create_paragrafo(self, sekcio_id: int) -> int:
        """Create paragraph row."""
        paragrafo_id = self.next_paragrafo_id
        self.next_paragrafo_id += 1

        self.paragraph_order_in_section += 1

        self.writers['Paragrafo'].writerow([
            paragrafo_id,
            sekcio_id,
            self.paragraph_order_in_section,
            'normal',  # paragraph type
            ''  # metadatenoj (empty for now)
        ])

        # Add relationship
        self.writers['EN_SEKCIO'].writerow([paragrafo_id, sekcio_id])

        # Add SEKVA_PARAGRAFO relationship if there was a previous paragraph
        if self.prev_paragrafo_id is not None:
            self.writers['SEKVA_PARAGRAFO'].writerow([self.prev_paragrafo_id, paragrafo_id])

        self.prev_paragrafo_id = paragrafo_id
        self.current_paragrafo_id = paragrafo_id

        return paragrafo_id

    def create_sentence(self, text: str, paragrafo_id: int, frazo_ordo: int, tutmonda_ordo: int) -> int:
        """Create Frazoteksto row with correct paragraph link and ordering."""
        sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        self.writers['Frazoteksto'].writerow([
            sentence_id,
            paragrafo_id,
            text,
            frazo_ordo,  # Position within paragraph (1, 2, 3, ...)
            tutmonda_ordo  # Position within entire document
        ])

        # Add relationship
        self.writers['EN_PARAGRAFO'].writerow([sentence_id, paragrafo_id])

        # Add SEKVA_FRAZOTEKSTO relationship if there was a previous sentence
        if self.prev_frazoteksto_id is not None:
            self.writers['SEKVA_FRAZOTEKSTO'].writerow([self.prev_frazoteksto_id, sentence_id])

        self.prev_frazoteksto_id = sentence_id

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
            '2026-03-07T12:00:00',  # created_at (placeholder)
            'corpus_to_csv_v2.1.py',
            'true',  # is_current
            ast_dict.get('fraztipo', 'deklaro'),
            ast_dict.get('demandotipo', ''),
            'true' if ast_dict.get('negita') else 'false',
            stats.get('total_words') or stats.get('tutaj_vortoj') or 0,
            stats.get('esperanto_words') or stats.get('esperantaj_vortoj') or 0,
            stats.get('non_esperanto_words') or stats.get('neesperantaj_vortoj') or 0,
            stats.get('success_rate') or stats.get('sukcesoprocento') or 0.0,
            self.escape_json(stats.get('analizkategorioj', {}))
        ])

        # Add relationship
        self.writers['FRAZOTEKSTO_HAVAS_AST'].writerow([sentence_id, ast_id, 'true'])

        return ast_id

    def create_frazo(self, ast_dict: Dict, ast_id: int) -> int:
        """Create Frazo row and structure."""
        frazo_id = self.next_frazo_id
        self.next_frazo_id += 1

        self.writers['Frazo'].writerow([frazo_id, ast_id, 'frazo'])

        # Add relationship
        self.writers['AST_HAVAS_FRAZON'].writerow([ast_id, frazo_id])

        # Process subjekto
        if 'subjekto' in ast_dict and ast_dict['subjekto'] is not None:
            subjekto = ast_dict['subjekto']
            if subjekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(subjekto, ast_id)
                self.writers['HAVAS_SUBJEKTON_VORTGRUPO'].writerow([frazo_id, vg_id])
            elif subjekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(subjekto, ast_id)
                self.writers['HAVAS_SUBJEKTON_VORTO'].writerow([frazo_id, vorto_id])

        # Process verbo
        if 'verbo' in ast_dict and ast_dict['verbo'] is not None:
            verbo_id = self.create_vorto(ast_dict['verbo'], ast_id)
            self.writers['HAVAS_VERBON'].writerow([frazo_id, verbo_id])

        # Process objekto
        if 'objekto' in ast_dict and ast_dict['objekto'] is not None:
            objekto = ast_dict['objekto']
            if objekto.get('tipo') == 'vortgrupo':
                vg_id = self.create_vortgrupo(objekto, ast_id)
                self.writers['HAVAS_OBJEKTON_VORTGRUPO'].writerow([frazo_id, vg_id])
            elif objekto.get('tipo') == 'vorto':
                vorto_id = self.create_vorto(objekto, ast_id)
                self.writers['HAVAS_OBJEKTON_VORTO'].writerow([frazo_id, vorto_id])

        # Process aliaj
        if 'aliaj' in ast_dict:
            for position, vorto in enumerate(ast_dict['aliaj']):
                vorto_id = self.create_vorto(vorto, ast_id)
                self.writers['HAVAS_ALIAJN'].writerow([frazo_id, vorto_id, position])

        return frazo_id

    def create_vortgrupo(self, vg_dict: Dict, ast_id: int) -> int:
        """Create Vortgrupo row."""
        vg_id = self.next_vortgrupo_id
        self.next_vortgrupo_id += 1

        self.writers['Vortgrupo'].writerow([vg_id, ast_id, 'vortgrupo'])

        # Create kerno
        if 'kerno' in vg_dict:
            kerno_id = self.create_vorto(vg_dict['kerno'], ast_id)
            self.writers['HAVAS_KERNON'].writerow([vg_id, kerno_id])

        # Create priskriboj
        if 'priskriboj' in vg_dict:
            for position, priskribo in enumerate(vg_dict['priskriboj']):
                priskribo_id = self.create_vorto(priskribo, ast_id)
                self.writers['HAVAS_PRISKRIBON'].writerow([vg_id, priskribo_id, position])

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

        # Parser now outputs pure Esperanto, no translation needed
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
            vorto_dict.get('analizstato', 'nekonata'),
            vorto_dict.get('analizeraro', ''),
            vorto_dict.get('kategorio', ''),
            vorto_dict.get('propranoma_kategorio', ''),
            vorto_dict.get('propranoma_ofteco') or 0,  # Handle None values
            vorto_dict.get('korelativo_prefikso', ''),
            vorto_dict.get('korelativo_sufikso', ''),
            vorto_dict.get('korelativo_signifo', ''),
            'true' if vorto_dict.get('estas_kunmetita') else 'false',
            self.escape_json(vorto_dict.get('kunmetitaj_radikoj', []))
        ])

        return vorto_id

    def generate_root_csv(self):
        """Generate Radiko node and relationship CSVs (v2.1)."""
        logger.info("Generating Radiko CSV and HAVAS_RADIKON relationships...")

        # Radiko nodes
        root_writer = self.open_csv_writer(
            'Radiko', True,
            ['radiko', 'dokumenta_ofteco', 'tuta_ofteco']
        )

        # Create radiko string -> radiko node ID mapping
        # We'll use the radiko string itself as the primary key
        all_roots = sorted(self.root_total_freq.keys())

        for root in all_roots:
            root_writer.writerow([
                root,
                self.root_doc_freq[root],
                self.root_total_freq[root]
            ])

        logger.info(f"Generated {len(all_roots)} roots")

        # Note: HAVAS_RADIKON relationships CSV is NOT pre-generated
        # because we don't have Vorto IDs during corpus processing.
        # The loader will create these via indexed MATCH query.


def main():
    parser = argparse.ArgumentParser(description='Convert corpus to CSV files with full document structure')
    parser.add_argument('--corpus', type=Path, required=True, help='Path to corpus JSONL')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for CSVs')
    parser.add_argument('--max-entries', type=int, help='Maximum entries to process')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (delete existing output)')

    args = parser.parse_args()

    # Clean output directory if --fresh
    if args.fresh and args.output.exists():
        import shutil
        logger.info(f"Removing existing output directory: {args.output}")
        shutil.rmtree(args.output)

    converter = CorpusToCSVConverter(args.output)
    converter.process_corpus(args.corpus, args.max_entries)


if __name__ == '__main__':
    main()
