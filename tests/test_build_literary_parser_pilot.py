"""scripts/eval/build_literary_parser_pilot.py -- flattening the
extracted-but-unindexed data/extracted/eo/free/*.jsonl corpora into
(sid, text, source, title) rows for build_parser_pilot.select().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.build_literary_parser_pilot import load_rows


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def _row(title: str, sentence: str) -> dict:
    return {'sentence': sentence, 'source': 'gutenberg', 'source_title': title,
            'author': 'Some Author', 'kind': 'tradukita', 'licence': 'public domain'}


class TestLoadRows:
    def test_each_sentence_becomes_one_row(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Book A', 'Unua.'), _row('Book A', 'Dua.'),
        ])
        rows = load_rows(tmp_path, ('gutenberg',))
        assert [text for _sid, text, _src, _title in rows] == ['Unua.', 'Dua.']

    def test_row_ids_are_unique(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Book A', 'Unua.'), _row('Book A', 'Dua.'), _row('Book B', 'Tria.'),
        ])
        rows = load_rows(tmp_path, ('gutenberg',))
        ids = [sid for sid, *_ in rows]
        assert len(ids) == len(set(ids))

    def test_an_excluded_title_is_skipped(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Kabe - Vortaro de Esperanto, 1922.pdf', 'Vorto.'),
            _row('A Real Book', 'Frazo.'),
        ])
        rows = load_rows(tmp_path, ('gutenberg',))
        assert [title for _sid, _text, _src, title in rows] == ['A Real Book']

    def test_blank_sentences_are_skipped(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [
            _row('Book A', '   '), _row('Book A', 'Reala frazo.'),
        ])
        rows = load_rows(tmp_path, ('gutenberg',))
        assert len(rows) == 1

    def test_reads_multiple_source_files(self, tmp_path):
        _write(tmp_path / 'gutenberg_sentences.jsonl', [_row('Book A', 'X.')])
        _write(tmp_path / 'libera_folio_sentences.jsonl', [
            {**_row('Article B', 'Y.'), 'source': 'libera_folio', 'licence': 'CC BY 4.0'},
        ])
        rows = load_rows(tmp_path, ('gutenberg', 'libera_folio'))
        assert sorted(src for _sid, _text, src, _title in rows) == ['gutenberg', 'libera_folio']
