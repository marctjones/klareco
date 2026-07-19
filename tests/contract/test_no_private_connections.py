"""#885: stages must NOT open their own DuckDB connection — they receive a
shared StoreView. A private connection is how a stage grows a private, drifting
schema (the #881 disease). Enforced statically here."""
import ast as pyast
from pathlib import Path

import pytest

STAGES = Path(__file__).resolve().parents[2] / 'klareco' / 'orchestrator' / 'stages'


def _stage_files():
    return sorted(p for p in STAGES.glob('*.py') if p.name != '__init__.py')


@pytest.mark.parametrize('path', _stage_files(), ids=lambda p: p.name)
def test_no_duckdb_connect_in_stage(path):
    src = path.read_text()
    assert 'duckdb.connect' not in src, (
        f"{path.name} opens a private DuckDB connection (#885). Stages must take "
        f"a StoreView and use store.connection / store.execute.")


def test_storeview_is_the_only_connector():
    """Sanity: StoreView itself is where the connect lives."""
    sv = (Path(__file__).resolve().parents[2] / 'klareco' / 'orchestrator'
          / 'store_view.py').read_text()
    assert 'duckdb.connect' in sv


def test_connection_backed_stages_take_a_store():
    """planner + ast_aware_rerank accept a StoreView (not a duckdb_path)."""
    from klareco.orchestrator.stages.planner import PlannerStage
    from klareco.orchestrator.stages.ast_aware_rerank import ASTAwareRerankStage
    import inspect
    for cls in (PlannerStage, ASTAwareRerankStage):
        params = inspect.signature(cls.__init__).parameters
        assert 'store' in params, f"{cls.__name__}.__init__ must take `store`"
        assert 'duckdb_path' not in params, f"{cls.__name__} still takes duckdb_path"
