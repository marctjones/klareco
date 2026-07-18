"""
Contract tests (#884): stage-declared dependencies are preflighted LOUDLY.

The mechanism under test is what turns an #881-class schema drift from a
silent per-question no-op into an itemized construction-time crash.
"""
from __future__ import annotations

import logging

import duckdb
import pytest

from klareco.orchestrator.dependencies import (
    DependencyError, DirDependency, FileDependency, TableDependency,
    preflight_stages,
)


class _FakeStage:
    def __init__(self, name, requires):
        self.name = name
        self.REQUIRES = requires


@pytest.fixture
def triple_schema_db(tmp_path):
    """A store whose entity_facts uses the LIVE triple schema (the #881 drift)."""
    db = tmp_path / 'store.db'
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE entity_facts "
                "(sid BIGINT, entito TEXT, rilato TEXT, valoro TEXT, "
                " klauzo INT, klaso TEXT, fonto TEXT)")
    con.close()
    return db


def test_missing_columns_raise_itemized_with_issue(triple_schema_db):
    stage = _FakeStage('planner', (
        TableDependency('entity_facts',
                        columns=('entity_radiko', 'slot', 'value'),
                        issue='#881'),
    ))
    with pytest.raises(DependencyError) as ei:
        preflight_stages([stage], duckdb_path=triple_schema_db)
    msg = str(ei.value)
    assert '[planner]' in msg
    assert 'entity_facts' in msg
    assert "lacks column 'entity_radiko'" in msg
    assert '#881' in msg
    # the error TEACHES: it names what the table actually has
    assert 'entito' in msg


def test_missing_table_raises(triple_schema_db):
    stage = _FakeStage('x', (TableDependency('no_such_table'),))
    with pytest.raises(DependencyError, match="does not exist"):
        preflight_stages([stage], duckdb_path=triple_schema_db)


def test_satisfied_dependencies_pass(triple_schema_db):
    stage = _FakeStage('x', (
        TableDependency('entity_facts', columns=('entito', 'rilato', 'valoro')),
    ))
    preflight_stages([stage], duckdb_path=triple_schema_db)   # must not raise


def test_file_and_dir_dependencies(tmp_path, triple_schema_db):
    present = tmp_path / 'exists.json'
    present.write_text('{}')
    ok = _FakeStage('x', (FileDependency(str(present)),
                          DirDependency(str(tmp_path))))
    preflight_stages([ok], duckdb_path=triple_schema_db)      # must not raise

    bad = _FakeStage('y', (FileDependency(str(tmp_path / 'gone.json'),
                                          issue='#804'),))
    with pytest.raises(DependencyError, match='#804'):
        preflight_stages([bad], duckdb_path=triple_schema_db)


def test_allow_degraded_downgrades_to_loud_banner(triple_schema_db, caplog):
    stage = _FakeStage('planner', (
        TableDependency('entity_facts', columns=('slot',), issue='#881'),
    ))
    with caplog.at_level(logging.WARNING):
        preflight_stages([stage], duckdb_path=triple_schema_db,
                         allow_degraded=True)                 # no raise
    assert any('#881' in r.message for r in caplog.records)
    assert any('degraded' in r.message.lower() for r in caplog.records)


def test_stages_without_requires_need_no_db(tmp_path):
    """No TableDependency → no store connection → works with no DB at all."""
    stage = _FakeStage('pure', ())
    preflight_stages([stage], duckdb_path=tmp_path / 'nonexistent.db')


def test_planner_and_biography_declare_the_live_triple_schema():
    """Post-#881: both read entity_facts through the SLOTS adapter over the live
    TRIPLE table, so they must DECLARE the triple columns that actually exist —
    not the old slot columns (that was the drift). Preflight then passes and the
    stages degrade gracefully; the remaining fact-quality gap is #745."""
    from klareco.orchestrator.stages.planner import PlannerStage
    from klareco.orchestrator.stages.biography_format import BiographyFormatStage
    for cls in (PlannerStage, BiographyFormatStage):
        deps = {d.table: d for d in cls.REQUIRES
                if isinstance(d, TableDependency)}
        assert 'entity_facts' in deps, f'{cls.__name__} must declare entity_facts'
        cols = deps['entity_facts'].columns
        # the real, existing triple columns — NOT the old slot columns
        assert {'sid', 'entito', 'rilato', 'valoro'} <= set(cols), cols
        assert 'entity_radiko' not in cols, 'slot-schema drift must not reappear'
