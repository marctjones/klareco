"""
Stage-level dependency declaration + preflight (#884).

Every PipelineStage declares the concrete resources its run() actually
touches — tables WITH the columns it queries, files, directories — in its
`REQUIRES` tuple. `preflight_stages()` validates the declarations against the
live environment at pipeline CONSTRUCTION time and raises loudly, itemized.

This exists because of a measured failure (#881): three stages queried an
`entity_facts` schema the live table did not have, every call raised
BinderException, every stage swallowed it, and the pipeline silently no-opped
for weeks. Declaring the columns a stage queries makes that failure mode a
construction-time crash with the issue number in the message.

Policy mirrors klareco.preflight: KLARECO_ALLOW_DEGRADED=1 (or
allow_degraded=True) downgrades the raise to a loud, itemized banner.
You may run degraded — you may not do so by accident.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """A stage's declared dependency is not satisfied by the environment."""


@dataclass(frozen=True)
class TableDependency:
    """A DuckDB table (and the columns the stage actually queries)."""
    table: str
    columns: tuple = ()
    issue: str = ''          # the tracking issue to cite in the error


@dataclass(frozen=True)
class FileDependency:
    path: str
    issue: str = ''


@dataclass(frozen=True)
class DirDependency:
    path: str
    issue: str = ''


def _check_table(dep: TableDependency, con) -> list[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ?", [dep.table]).fetchall()
    if not rows:
        return [f"table '{dep.table}' does not exist"]
    have = {r[0] for r in rows}
    return [f"table '{dep.table}' lacks column '{c}' "
            f"(has: {', '.join(sorted(have))})"
            for c in dep.columns if c not in have]


def _check_path(dep, want_dir: bool) -> list[str]:
    p = Path(dep.path)
    if not p.exists():
        kind = 'directory' if want_dir else 'file'
        return [f"{kind} '{dep.path}' does not exist"]
    return []


def preflight_stages(stages: list, duckdb_path: Path | str,
                     *, allow_degraded: Optional[bool] = None) -> None:
    """
    Validate every stage's declared REQUIRES against the live environment.

    Raises DependencyError (itemized, with issue references) unless
    allow_degraded / KLARECO_ALLOW_DEGRADED=1, which downgrades to a loud
    banner — the same explicit-and-noisy policy as klareco.preflight.
    """
    if allow_degraded is None:
        allow_degraded = os.environ.get('KLARECO_ALLOW_DEGRADED', '') == '1'

    problems: list[str] = []
    con = None
    try:
        for stage in stages:
            for dep in getattr(stage, 'REQUIRES', ()) or ():
                issue = f'  [{dep.issue}]' if getattr(dep, 'issue', '') else ''
                if isinstance(dep, TableDependency):
                    if con is None:
                        import duckdb
                        con = duckdb.connect(str(duckdb_path), read_only=True)
                    for msg in _check_table(dep, con):
                        problems.append(f'[{stage.name}] {msg}{issue}')
                elif isinstance(dep, FileDependency):
                    for msg in _check_path(dep, want_dir=False):
                        problems.append(f'[{stage.name}] {msg}{issue}')
                elif isinstance(dep, DirDependency):
                    for msg in _check_path(dep, want_dir=True):
                        problems.append(f'[{stage.name}] {msg}{issue}')
                else:
                    problems.append(f'[{stage.name}] unknown dependency '
                                    f'type {type(dep).__name__}')
    finally:
        if con is not None:
            con.close()

    if not problems:
        return

    report = ('STAGE DEPENDENCY PREFLIGHT FAILED — a silently-degrading '
              'dependency is a bug (#884):\n  ' + '\n  '.join(problems))
    if allow_degraded:
        logger.warning('%s\n  (KLARECO_ALLOW_DEGRADED=1 — running degraded '
                       'ANYWAY, on your explicit request)', report)
        return
    raise DependencyError(
        report + '\n  Set KLARECO_ALLOW_DEGRADED=1 to run degraded anyway '
                 '(loud, itemized, deliberate).')
