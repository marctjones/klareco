"""Exercise the side-store builder through its real CLI and production reader."""

import json
from pathlib import Path
import subprocess
import sys

import duckdb
import pytest

from klareco.parser import parse, expand_ast
from klareco.rag.duckdb_retriever import DuckDBRetriever
from scripts.index.build_duckdb_store import _worker

ROOT = Path(__file__).resolve().parents[2]


def test_rebuild_preserves_identity_and_refreshes_all_views(mini_store, tmp_path):
    source, whoosh = mini_store
    output = tmp_path / "candidate.db"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/index/reparse_store.py"),
            "--source",
            str(source),
            "--output",
            str(output),
            "--limit",
            "0",
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "validated_candidate"
    assert manifest["processed"] == manifest["expected"] == 12
    with duckdb.connect(str(output), read_only=True) as con:
        rows = con.execute(
            "SELECT text,ast_json FROM sentences ORDER BY sid"
        ).fetchall()
        for text, packed in rows:
            assert expand_ast(json.loads(packed)) == parse(text)
        assert con.execute("SELECT count(*) FROM token_edges").fetchone()[0] == sum(
            len(parse(text)["vortoj"]) for text, _ in rows
        )
        assert (
            con.execute(
                "SELECT count(*) FROM clauses WHERE verb_klaso IS NOT NULL"
            ).fetchone()[0]
            > 0
        )
    retriever = DuckDBRetriever(whoosh, output)
    results = retriever.retrieve_with_ast_roles(parse("Kiu kreis Esperanton?"), 3)
    assert results
    assert all(row["ast"] == parse(row["text"]) for row in results)


def test_original_writer_cannot_silently_store_null_asts():
    with pytest.raises(ValueError, match="sentence 42"):
        _worker((42, "", {}))


def test_oversized_sources_block_before_any_candidate_rows_are_written(tmp_path):
    source = tmp_path / 'source.db'
    output = tmp_path / 'blocked.db'
    with duckdb.connect(str(source)) as con:
        con.execute('CREATE TABLE sentences(sid BIGINT, text VARCHAR)')
        con.execute('INSERT INTO sentences VALUES (7, ?)', ['x' * 10001])
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/index/reparse_store.py'),
         '--source', str(source), '--output', str(output), '--limit', '0'],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert manifest['status'] == 'blocked_source_repair'
    assert manifest['processed'] == 0
    assert manifest['oversized_rows'] == [[7, 10001]]
    with duckdb.connect(str(source), read_only=True) as con:
        assert con.execute('SELECT length(text) FROM sentences').fetchone()[0] == 10001
