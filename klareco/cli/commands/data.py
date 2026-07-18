"""
Data lifecycle: acquire -> clean -> extract -> parse -> index -> validate.

Each subcommand is a thin, discoverable wrapper over the canonical script for
that stage. These are LONG-RUNNING; by default the command PRINTS the exact
command to run (so a human runs it deliberately, in its own terminal, per the
long-running-scripts policy). `--run` executes it.

This makes the setup pipeline discoverable from `klareco data --help` without
hiding a multi-hour job behind an innocent-looking subcommand.
"""
from __future__ import annotations

import subprocess

from klareco.cli._base import EXIT_OK, add_common, emit, err

# stage -> canonical command (the source of truth is docs/CLI_ARCHITECTURE.md §1)
STAGES = {
    'acquire':      ['./scripts/acquire/acquire_all_tier0.sh'],
    'clean':        ['./scripts/clean/clean_all.sh'],
    'extract':      ['./scripts/extract/extract_all.sh'],
    'parse':        ['./scripts/parse/parse_corpus.sh'],
    'build-store':  ['python', 'scripts/index/build_duckdb_store.py'],
    'build-search': ['python', 'scripts/index/rebuild_whoosh_from_duckdb.py'],
    'validate':     ['python', 'scripts/index/validate_duckdb_store.py'],
    'rebuild':      ['./scripts/pipeline/rebuild_all.sh'],
}

_NOTES = {
    'parse':   'parse ~15 min for 5.4M sentences; wall-clock is the 20 GB JSONL write',
    'rebuild': 'orchestrates parse -> store -> search -> validate in one pass',
}


def cmd_data(args) -> int:
    cmd = STAGES.get(args.stage)
    if cmd is None:
        return err(f"unknown stage {args.stage!r}; choose from {', '.join(STAGES)}")
    shown = ' '.join(cmd)
    note = _NOTES.get(args.stage, '')
    if not args.run:
        emit(args,
             text=(f"# {args.stage}{f'  — {note}' if note else ''}\n{shown}\n"
                   f"# (long-running: run it yourself, or re-run with --run)"),
             data={'stage': args.stage, 'command': cmd, 'note': note, 'ran': False})
        return EXIT_OK
    emit(args, text=f"$ {shown}", data={'stage': args.stage, 'command': cmd, 'ran': True})
    return subprocess.call(cmd)


def register(sub) -> None:
    d = sub.add_parser('data', help='Data lifecycle: acquire/clean/extract/parse/index/validate')
    d.add_argument('stage', choices=list(STAGES),
                   help='Pipeline stage to run (or preview)')
    d.add_argument('--run', action='store_true',
                   help='Actually execute (default: print the command — these are long-running)')
    add_common(d)
    d.set_defaults(func=cmd_data)
