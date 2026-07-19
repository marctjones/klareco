"""
StoreView — the single injected handle to the DuckDB store (#885).

Stages must NOT open their own connection. They receive a StoreView, and the
StoreView is the one place that connects and defines how the store is accessed.
This is the structural cure for the #881 disease: three stages had each opened
their own connection and hard-coded a private column schema, which silently
drifted from the live table. When the schema lives in ONE injected object, a
stage cannot hold a private, drifting view of it.

Enforced by tests/contract/test_no_private_connections.py (no `duckdb.connect`
in klareco/orchestrator/stages/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb


class StoreView:
    """A read-only view of the DuckDB store, shared across stages."""

    def __init__(self, duckdb_path: str | Path,
                 *, memory_limit: str = '2GB', threads: int = 4,
                 read_only: bool = True):
        self.path = str(duckdb_path)
        self._con = duckdb.connect(self.path, read_only=read_only)
        self._con.execute(f"SET memory_limit = '{memory_limit}'")
        self._con.execute(f"SET threads = {threads}")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The underlying read-only connection (for code that builds its own SQL)."""
        return self._con

    def execute(self, sql: str, params: Optional[list] = None):
        return self._con.execute(sql, params or [])

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    @classmethod
    def coerce(cls, store_or_path) -> "StoreView":
        """Accept a StoreView or a path — so callers can pass either during the
        migration; a bare path builds one StoreView here (still the one place)."""
        return store_or_path if isinstance(store_or_path, cls) else cls(store_or_path)
