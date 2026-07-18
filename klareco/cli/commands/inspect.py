"""
Inspect the environment and the store.

  doctor        preflight the runtime (raises loudly if unusable) + summary
  info          versions + capability status
  inspect ast   print a stored sentence's AST (read ast_json — never re-parse)
  inspect store row counts, schema, ontology edges, index freshness
"""
from __future__ import annotations

import sys
from pathlib import Path

from klareco.cli._base import (
    EXIT_DEGRADED, EXIT_OK, add_common, emit, err,
)

DEFAULT_DB = "data/indexes/duckdb_store.db"
DEFAULT_WHOOSH = "data/indexes/whoosh_v2"


def cmd_doctor(args) -> int:
    """Is this machine set up to run anything? Loud by contract."""
    from klareco.preflight import preflight, PreflightError
    try:
        preflight(duckdb_path=DEFAULT_DB, whoosh_index_dir=DEFAULT_WHOOSH,
                  allow_degraded=args.allow_degraded)
    except PreflightError as e:
        emit(args, text=f"✗ NOT READY\n{e}", data={'ready': False, 'detail': str(e)})
        return err("environment preflight failed")
    status = 'degraded' if args.allow_degraded else 'ok'
    emit(args, text=f"✓ environment {status}", data={'ready': True, 'status': status})
    return EXIT_DEGRADED if args.allow_degraded else EXIT_OK


def cmd_info(args) -> int:
    from klareco.parser import KNOWN_PREFIXES, KNOWN_SUFFIXES
    data = {
        'python': sys.version.split()[0],
        'prefixes': len(KNOWN_PREFIXES),
        'suffixes': len(KNOWN_SUFFIXES),
        'store_present': Path(DEFAULT_DB).exists(),
        'whoosh_present': Path(DEFAULT_WHOOSH).exists(),
    }
    emit(args, text=(
        "=== Klareco ===\n"
        f"Python: {data['python']}\n"
        f"Deterministic vocab: {data['prefixes']} prefixes, {data['suffixes']} suffixes\n"
        f"Store:  {'present' if data['store_present'] else 'MISSING'} ({DEFAULT_DB})\n"
        f"Search: {'present' if data['whoosh_present'] else 'MISSING'} ({DEFAULT_WHOOSH})"),
        data=data)
    return EXIT_OK


def _connect():
    import duckdb
    return duckdb.connect(DEFAULT_DB, read_only=True)


def cmd_inspect_store(args) -> int:
    if not Path(DEFAULT_DB).exists():
        return err(f"store not found: {DEFAULT_DB}")
    con = _connect()
    rows = con.execute("SELECT count(*) FROM sentences").fetchone()[0]
    onto = {rel: n for rel, n in con.execute(
        "SELECT rel, count(*) FROM ontology_edges GROUP BY rel").fetchall()}
    ef = con.execute("SELECT count(*) FROM information_schema.tables "
                     "WHERE table_name='entity_facts'").fetchone()[0]
    data = {'sentences': rows, 'ontology_edges': onto, 'entity_facts_table': bool(ef)}
    emit(args, text=(
        f"sentences: {rows:,}\n"
        f"ontology_edges: {sum(onto.values()):,}  {onto}\n"
        f"entity_facts table: {'present' if ef else 'MISSING'}"),
        data=data)
    return EXIT_OK


def cmd_inspect_ast(args) -> int:
    import json as _json
    if not Path(DEFAULT_DB).exists():
        return err(f"store not found: {DEFAULT_DB}")
    row = _connect().execute(
        "SELECT ast_json FROM sentences WHERE sid = ?", [args.sid]).fetchone()
    if not row:
        return err(f"no sentence with sid={args.sid}")
    print(_json.dumps(_json.loads(row[0]), indent=2, ensure_ascii=False))
    return EXIT_OK


def register(sub) -> None:
    doc = sub.add_parser('doctor', help='Preflight the runtime (loud) + summary')
    doc.add_argument('--allow-degraded', action='store_true',
                     help='Report instead of failing on degraded artifacts')
    add_common(doc)
    doc.set_defaults(func=cmd_doctor)

    info = sub.add_parser('info', help='Versions + capability status')
    add_common(info)
    info.set_defaults(func=cmd_info)

    insp = sub.add_parser('inspect', help='Inspect the store / a stored AST')
    isub = insp.add_subparsers(dest='inspect_what')
    st = isub.add_parser('store', help='Row counts, ontology edges, table presence')
    add_common(st)
    st.set_defaults(func=cmd_inspect_store)
    ast = isub.add_parser('ast', help='Print a stored sentence AST by sid')
    ast.add_argument('sid', type=int, help='Sentence id')
    add_common(ast)
    ast.set_defaults(func=cmd_inspect_ast)
