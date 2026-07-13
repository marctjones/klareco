"""Tests for klareco.preflight — the fail-loudly artifact gate (#779).

These are deterministic and offline: every check runs against temporary paths,
so none of them needs (or touches) the production store.

The behaviour under test is the whole point of the module:

    a missing artifact must never be survivable by accident.
"""

import os
from pathlib import Path

import pytest

from klareco.preflight import (
    Finding,
    PreflightError,
    _check_file,
    _check_whoosh,
    preflight,
)


class TestFileCheck:
    def test_missing_file_is_a_finding(self, tmp_path):
        f = _check_file("thing.json", tmp_path / "nope.json", required=True,
                        consequence="c", remedy="r")
        assert f is not None
        assert "does not exist" in f.detail

    def test_empty_file_is_a_finding(self, tmp_path):
        """A zero-byte file is the classic silent failure: it *exists*, so an
        `if path.exists()` guard sails right past it."""
        p = tmp_path / "empty.json"
        p.write_text("")
        f = _check_file("empty.json", p, required=True, consequence="c", remedy="r")
        assert f is not None
        assert "empty" in f.detail

    def test_present_and_nonempty_is_clean(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text('{"a": 1}')
        assert _check_file("ok.json", p, required=True,
                           consequence="c", remedy="r") is None


class TestWhooshCheck:
    def test_missing_dir_is_required_failure(self, tmp_path):
        findings = _check_whoosh(tmp_path / "absent")
        assert len(findings) == 1
        assert findings[0].required is True

    def test_dir_without_index_is_required_failure(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        findings = _check_whoosh(tmp_path / "empty_dir")
        assert len(findings) == 1
        assert findings[0].required is True


class TestPreflightGate:
    """The gate itself — this is the behaviour that #779 exists to enforce."""

    def test_raises_when_required_artifact_missing(self, tmp_path):
        with pytest.raises(PreflightError):
            preflight(duckdb_path=tmp_path / "no.db",
                      whoosh_index_dir=tmp_path / "no_index",
                      allow_degraded=False)

    def test_required_artifact_cannot_be_waived(self, tmp_path):
        """allow_degraded lets you proceed on a *degraded* system. It must not
        let you proceed on a *broken* one — a missing store means no retrieval
        is possible at all, and pretending otherwise helps nobody."""
        with pytest.raises(PreflightError):
            preflight(duckdb_path=tmp_path / "no.db",
                      whoosh_index_dir=tmp_path / "no_index",
                      allow_degraded=True)

    def test_error_message_names_the_artifact_and_the_consequence(self, tmp_path):
        """A preflight error that doesn't say what broke, what it costs, and how
        to fix it is just a different kind of silence."""
        with pytest.raises(PreflightError) as exc:
            preflight(duckdb_path=tmp_path / "no.db",
                      whoosh_index_dir=tmp_path / "no_index",
                      allow_degraded=False)
        msg = str(exc.value)
        assert "duckdb_store.db" in msg
        assert "consequence:" in msg
        assert "remedy:" in msg

    def test_env_var_is_read_when_allow_degraded_is_none(self, tmp_path, monkeypatch):
        """KLARECO_ALLOW_DEGRADED=1 is the opt-in for scripts. It still must not
        waive a REQUIRED artifact."""
        monkeypatch.setenv("KLARECO_ALLOW_DEGRADED", "1")
        with pytest.raises(PreflightError):
            preflight(duckdb_path=tmp_path / "no.db",
                      whoosh_index_dir=tmp_path / "no_index",
                      allow_degraded=None)


class TestFindingRendering:
    def test_render_includes_all_three_fields(self):
        f = Finding("x", required=False, detail="d", consequence="c", remedy="r")
        out = f.render()
        assert "DEGRADED" in out
        assert "d" in out and "c" in out and "r" in out

    def test_required_findings_are_tagged_differently(self):
        assert "REQUIRED" in Finding("x", True, "d", "c", "r").render()
