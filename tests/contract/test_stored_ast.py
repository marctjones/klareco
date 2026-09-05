"""Exercise decoding at the real retrieval boundary, including corrupt blobs."""
import json
import shutil

import duckdb
import pytest

from klareco.parser import parse
from klareco.rag.duckdb_retriever import DuckDBRetriever


@pytest.mark.parametrize('blob', [
    None,
    '{broken json',
    json.dumps({'_ast_format': 999, 'vortoj': [], 'structure': {}}),
    json.dumps({'_ast_format': 2, 'vortoj': [],
                'structure': {'subjekto': {'$token': 123}}}),
])
def test_corrupt_stored_ast_does_not_become_empty_evidence(mini_store, tmp_path, blob):
    original, whoosh = mini_store
    target = tmp_path / 'corrupt.db'
    shutil.copy2(original, target)
    with duckdb.connect(str(target)) as con:
        con.execute('UPDATE sentences SET ast_json = ?', [blob])
    retriever = DuckDBRetriever(whoosh, target)
    try:
        with pytest.raises(ValueError):
            retriever.retrieve_with_ast_roles(parse('Kiu kreis Esperanton?'), top_k=3)
    finally:
        retriever.con.close()
        if retriever._cached_searcher is not None:
            retriever._cached_searcher.close()
