"""scripts/eval/build_literary_discourse_pilot.py -- document grouping over
the extracted-but-unindexed data/extracted/eo/free/*.jsonl corpora.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.build_literary_discourse_pilot import load_documents


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def _row(title: str, sentence: str, author: str = 'Some Author') -> dict:
    return {'sentence': sentence, 'source': 'gutenberg', 'source_title': title,
            'author': author, 'kind': 'tradukita', 'licence': 'public domain'}


class TestLoadDocuments:
    def test_consecutive_rows_with_the_same_title_form_one_document(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Book A', 'Unua.'), _row('Book A', 'Dua.'), _row('Book A', 'Tria.'),
        ])
        documents, titles, meta = load_documents(tmp_path, ('gutenberg',))
        assert len(documents) == 1
        [doc] = documents.values()
        assert [text for _sid, text, _sec, _src in doc] == ['Unua.', 'Dua.', 'Tria.']

    def test_a_title_reused_non_contiguously_becomes_two_documents(self, tmp_path):
        """'Tri Noveloj' in the real gutenberg file names two unrelated
        collections -- splitting on any break in contiguity is the
        general fix, not a special case for that one title."""
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Tri Noveloj', 'Unua libro, unua frazo.'),
            _row('Other Book', 'Nerilata frazo.'),
            _row('Tri Noveloj', 'Dua libro, unua frazo.'),
        ])
        documents, titles, meta = load_documents(tmp_path, ('gutenberg',))
        assert len(documents) == 3
        tri_novel_docs = [doc for doc_id, doc in documents.items() if titles[doc_id] == 'Tri Noveloj']
        assert len(tri_novel_docs) == 2
        assert tri_novel_docs[0][0][1] == 'Unua libro, unua frazo.'
        assert tri_novel_docs[1][0][1] == 'Dua libro, unua frazo.'

    def test_an_excluded_title_produces_no_document(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Kabe - Vortaro de Esperanto, 1922.pdf', 'Vorto: difino.'),
            _row('A Real Book', 'Frazo unu.'),
        ])
        documents, titles, meta = load_documents(tmp_path, ('gutenberg',))
        assert list(titles.values()) == ['A Real Book']

    def test_metadata_is_captured_per_document(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Book A', 'Frazo.', author='Poe, Edgar Allan'),
        ])
        documents, titles, meta = load_documents(tmp_path, ('gutenberg',))
        [m] = meta.values()
        assert m['author'] == 'Poe, Edgar Allan'
        assert m['licence'] == 'public domain'

    def test_reads_multiple_source_files(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [_row('Book A', 'X.')])
        _write(tmp_path / 'vikifontaro_sentences.jsonl', [
            {**_row('Book B', 'Y.'), 'source': 'vikifontaro', 'licence': 'PD + CC BY-SA 4.0'},
        ])
        documents, titles, meta = load_documents(tmp_path, ('gutenberg', 'vikifontaro'))
        assert sorted(titles.values()) == ['Book A', 'Book B']
