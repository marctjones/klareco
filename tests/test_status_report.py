"""#887: the executable status report generates from the live store without error."""
import importlib.util, json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "status_report", _ROOT / "scripts" / "validate" / "status_report.py")
sr = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(sr)


@pytest.mark.skipif(not (_ROOT / "data/indexes/duckdb_store.db").exists(),
                    reason="needs the live store")
def test_collect_returns_live_facts():
    d = sr.collect()
    assert d["store_present"] and d["sentences"] > 0
    assert d["ontology_edges"] >= 0 and "entity_facts_present" in d
    assert "esperanton_radiko" in d


@pytest.mark.skipif(not (_ROOT / "data/indexes/duckdb_store.db").exists(),
                    reason="needs the live store")
def test_render_has_markers_and_no_crash():
    md = sr.render_md(sr.collect())
    assert sr.MARK_BEGIN in md and sr.MARK_END in md
    assert "Current state (generated" in md
