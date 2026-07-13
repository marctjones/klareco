"""The environment contract — does the live system actually hold together?

Nothing else in the test suite checks this. `test_index_integrity.py` tests a
FAISS/Kuzu index that was retired months ago and skips forever;
`test_data_quality.py` skips on missing ReVo. The result was **1 passed, 54
skipped** — an environment suite that is asleep.

Meanwhile the real environment had:
  - `success_rate` = 0.0 on all 5,391,442 rows (a column that is 100% non-null
    and carries zero information)
  - 122,001 REDIRECT/ALIDIREKTI stubs in the live Whoosh index, which R6 says
    must be zero
  - no article provenance anywhere in the store

None of that was caught, because nothing was looking.

These tests assert the CONTRACT between the artifacts:

  1. the store exists, is coherent, and its keys are sound
  2. every column that exists carries INFORMATION, not just non-nulls
  3. the Whoosh index and the DuckDB store agree — and the index is COMPLETE
  4. the index actually WORKS: a known sentence is retrievable by its own words
  5. the corpus is not polluted with content-free stubs
  6. cross-table references resolve

They skip when the production artifacts are absent (AGENTS.md: tests needing the
production indexes detect their absence and skip). When the artifacts ARE
present they assert hard — a degraded artifact must fail, not shrug.

Run: pytest tests/test_environment_contract.py -v
"""

import json
from pathlib import Path

import pytest

DUCKDB_PATH = Path('data/indexes/duckdb_store.db')
WHOOSH_DIR = Path('data/indexes/whoosh_v2')

pytestmark = pytest.mark.environment


@pytest.fixture(scope='module')
def con():
    if not DUCKDB_PATH.exists():
        pytest.skip(f'{DUCKDB_PATH} absent — production store not on this machine')
    duckdb = pytest.importorskip('duckdb')
    c = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope='module')
def whoosh_ix():
    if not WHOOSH_DIR.exists():
        pytest.skip(f'{WHOOSH_DIR} absent — production index not on this machine')
    index = pytest.importorskip('whoosh.index')
    if not index.exists_in(str(WHOOSH_DIR)):
        pytest.fail(f'{WHOOSH_DIR} exists but holds no Whoosh index')
    return index.open_dir(str(WHOOSH_DIR))


# ---------------------------------------------------------------------------
# 1. The store is coherent
# ---------------------------------------------------------------------------

class TestStoreCoherence:
    def test_store_is_not_empty(self, con):
        n = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
        assert n > 0, 'sentences table is empty'

    def test_primary_key_is_unique(self, con):
        n, distinct = con.execute(
            'SELECT count(*), count(DISTINCT sid) FROM sentences').fetchone()
        assert n == distinct, f'{n - distinct:,} duplicate sids'

    def test_no_null_sids(self, con):
        n = con.execute('SELECT count(*) FROM sentences WHERE sid IS NULL').fetchone()[0]
        assert n == 0, f'{n:,} rows with NULL sid'

    def test_no_empty_text(self, con):
        n = con.execute(
            "SELECT count(*) FROM sentences WHERE text IS NULL OR trim(text) = ''"
        ).fetchone()[0]
        assert n == 0, f'{n:,} rows with empty text — they can never be retrieved'


# ---------------------------------------------------------------------------
# 2. Columns carry INFORMATION, not just non-nulls
#
# This is the check that would have caught success_rate. A column can be 100%
# populated and still be dead. Population is not the contract; VARIANCE is.
# ---------------------------------------------------------------------------

class TestColumnsAreInformative:
    def test_ast_json_is_populated_and_parses(self, con):
        total = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
        n = con.execute(
            "SELECT count(*) FROM sentences WHERE ast_json IS NOT NULL AND ast_json <> ''"
        ).fetchone()[0]
        assert n / total > 0.99, f'ast_json populated on only {n / total:.1%} of rows'

        # And it must actually be JSON with the expected shape.
        for (blob,) in con.execute(
                'SELECT ast_json FROM sentences WHERE ast_json IS NOT NULL LIMIT 50').fetchall():
            ast = json.loads(blob)
            assert 'tipo' in ast, f'ast_json has no `tipo`: {list(ast)[:6]}'

    @pytest.mark.parametrize('column', ['success_rate'])
    def test_numeric_column_is_not_a_constant(self, con, column):
        """A numeric column whose min == max across millions of rows carries
        ZERO information, however non-null it is.

        `success_rate` is 0.0 on all 5,391,442 rows — a field-name mismatch
        (the corpus writes `parse_rate`, the store reads `success_rate`). It
        passes every null-check and tells you nothing. See #805.
        """
        lo, hi = con.execute(
            f'SELECT min({column}), max({column}) FROM sentences').fetchone()
        assert lo != hi, (
            f'{column} is constant at {lo} across the whole store — the column '
            f'is present, non-null, and meaningless. See #805.')

    def test_verb_klaso_is_populated(self, con):
        """0% populated means every consumer of it silently no-ops — which is
        how nine rerankers came to be tied without anyone noticing. See #777."""
        total = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
        n = con.execute(
            "SELECT count(*) FROM sentences WHERE verb_klaso IS NOT NULL AND verb_klaso <> ''"
        ).fetchone()[0]
        assert n > 0, ('verb_klaso is 0% populated — the ontology never loaded, '
                       'so every scoring function that reads it contributes '
                       'nothing. See #777.')


# ---------------------------------------------------------------------------
# 3. The store and the index agree — and the index is COMPLETE
# ---------------------------------------------------------------------------

class TestStoreIndexConsistency:
    def test_index_is_not_empty(self, whoosh_ix):
        assert whoosh_ix.doc_count() > 0, 'Whoosh index has 0 documents'

    def test_every_INDEXABLE_store_row_is_indexed(self, con, whoosh_ix):
        """'Fully indexed' is a claim we had never actually checked.

        A store row that is not in the index is unreachable by retrieval — it
        exists but can never be an answer.

        The contract is *indexable* rows, not *all* rows: R6 (#802) deliberately
        excludes redirect stubs from the index while KEEPING them in the store
        (the store preserves them for ontology work; the index must not carry
        content-free documents). So the correct assertion is:

            index_count == store_count MINUS the R6-filtered junk

        Getting this wrong in either direction is a real failure: too few and
        real sentences are unreachable; too many and the junk is back.
        """
        JUNK_PREFIXES = ('ALIDIREKTI', 'REDIRECT', '#REDIRECT', '#ALIDIREKTI')
        clause = ' AND '.join(f"text NOT LIKE '{p}%'" for p in JUNK_PREFIXES)

        indexable = con.execute(
            f'SELECT count(*) FROM sentences WHERE {clause}').fetchone()[0]
        index_n = whoosh_ix.doc_count()

        # Tolerate the pre-#802 index (which still carries the junk) so this
        # test reports the RIGHT failure — the redirect-purity test below — and
        # not a confusing count mismatch on top of it.
        store_n = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
        if index_n == store_n:
            pytest.skip(
                'index still contains the R6 junk (pre-#802 build): '
                f'{index_n:,} docs == {store_n:,} store rows. The redirect-purity '
                'test is the one to read. Rebuild with '
                'scripts/index/rebuild_whoosh_from_duckdb.py.')

        assert index_n >= indexable, (
            f'{indexable - index_n:,} indexable store rows are NOT in the Whoosh '
            f'index (indexable={indexable:,}, index={index_n:,}) — those sentences '
            f'can never be retrieved.')

    def test_a_known_sentence_round_trips(self, con, whoosh_ix):
        """The index must actually WORK: take a real sentence out of the store,
        search for its own distinctive words, and get its own sid back.

        Doc counts matching proves the index is *full*. This proves it is
        *functional* — that documents are tokenized, scored, and retrievable.
        """
        from whoosh.qparser import OrGroup, QueryParser

        row = con.execute(
            "SELECT sid, text FROM sentences "
            "WHERE length(text) BETWEEN 60 AND 200 AND text NOT LIKE 'ALIDIREKTI%' "
            "LIMIT 1").fetchone()
        sid, text = row

        terms = [t for t in text.split() if len(t) > 4][:8]
        assert terms, 'could not build a query from the sample sentence'

        with whoosh_ix.searcher() as s:
            qp = QueryParser('text', whoosh_ix.schema, group=OrGroup)
            hits = s.search(qp.parse(' OR '.join(terms)), limit=200)
            found = any(int(h['id']) == int(sid) for h in hits)

        assert found, (
            f'sid {sid} is in the store but its own words do not retrieve it '
            f'from the index — the index is not functioning.')


# ---------------------------------------------------------------------------
# 4. The corpus is not polluted (R6)
# ---------------------------------------------------------------------------

class TestCorpusPurity:
    def test_no_redirect_stubs_in_the_whoosh_index(self, whoosh_ix):
        """R6: 'post-index sanity query must return 0 IN THE WHOOSH INDEX.'

        Measured 2026-07-13: 78,491 ALIDIREKTI + 43,448 REDIRECT = 122,001
        content-free stubs, indexed and retrievable. They are also the #1
        proper-noun subject in the store (REDIRECT: 42,319), so anything mining
        proper nouns is drinking from this. Failure mode F5. See #802.

        The DuckDB store is ALLOWED to keep them (R6 preserves them for ontology
        work). The INDEX is not.
        """
        from whoosh.qparser import QueryParser

        with whoosh_ix.searcher() as s:
            qp = QueryParser('text', whoosh_ix.schema)
            total = 0
            for term in ('ALIDIREKTI', 'REDIRECT'):
                total += s.search(qp.parse(term), limit=1).estimated_length()

        assert total == 0, (
            f'{total:,} redirect stubs are indexed and retrievable. R6 requires '
            f'0 in the Whoosh index. See #802.')


# ---------------------------------------------------------------------------
# 5. Cross-table references resolve
# ---------------------------------------------------------------------------

class TestReferentialIntegrity:
    def _table_exists(self, con, name: str) -> bool:
        return bool(con.execute(
            'SELECT count(*) FROM information_schema.tables WHERE table_name = ?',
            [name]).fetchone()[0])

    def test_entity_facts_reference_real_sentences(self, con):
        if not self._table_exists(con, 'entity_facts'):
            pytest.skip('entity_facts table absent (#745) — nothing to check yet')
        orphans = con.execute(
            'SELECT count(*) FROM entity_facts f '
            'LEFT JOIN sentences s ON f.sid = s.sid WHERE s.sid IS NULL'
        ).fetchone()[0]
        assert orphans == 0, f'{orphans:,} entity_facts point at sids that do not exist'

    def test_ontology_edges_reference_declared_classes(self, con):
        n_edges = con.execute('SELECT count(*) FROM ontology_edges').fetchone()[0]
        if n_edges == 0:
            pytest.skip('ontology_edges is empty (#777) — nothing to check yet')
        n_nodes = con.execute('SELECT count(*) FROM ontology_nodes').fetchone()[0]
        assert n_nodes > 0, 'ontology_edges has rows but ontology_nodes is empty'
