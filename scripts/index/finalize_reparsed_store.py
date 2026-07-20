#!/usr/bin/env python3
"""
Finalize a freshly reparsed side-store: carry ontology, verb_klaso, verify (#807)

VERSION: v1.0
COMPATIBLE WITH: v2.2 store schema (build_duckdb_store.py), post-#872 ontology_edges
DEPENDENCIES: duckdb, klareco.morphology (lexicon)
STAGE: Index

Description:
    build_duckdb_store.py builds `sentences` (+ Whoosh) into a SIDE store, but
    three things it cannot produce must come from elsewhere before the side
    store can be swapped live:

    1. --copy-ontology: `ontology_nodes` / `ontology_edges` copied FROM THE
       LIVE STORE. The ontology snapshot file is gone, and the live tables
       carry the #713 hand-fixed schema plus the #872 ALIASO alias bridge
       (~224k edges, a shipped production asset). Re-deriving them is
       forbidden (CLAUDE.md: load_ontology.py destroys the hand-fix).
       Row counts are verified equal to the live store or we exit(2).

    2. --verb-klaso: populate `clauses.verb_klaso` from the typed root
       lexicon (klareco.morphology.lexicon), replicating ONLY the safe tail
       of load_ontology.py — not its destructive ontology rebuild. Uses
       new-table-swap (CTAS + RENAME), never a bulk in-place UPDATE, per the
       DuckDB dead-page rule. Requires `clauses` (build_clause_table.py first).

    3. --verify: hard gates comparing side store vs live store. Any failure
       exits(2) — a silently-degrading store is a bug:
         - row count within 2% of live (same corpus, same #823 gate)
         - zero REDIRECT/ALIDIREKTI rows (the #823 point)
         - subj_radiko coverage >= live (the #871 point — the reparse must
           RECOVER subjects, never lose them)
         - success_rate has a real distribution (min != max; the #805 bug)
         - ontology_nodes / ontology_edges counts == live
         - clauses / dependency_arcs / entity_facts present and non-empty
         - clauses.verb_klaso non-null fraction >= 1%

Pipeline Position:
    build_duckdb_store.py (side paths) -> [THIS] -> swap -> klareco.preflight

Usage:
    python scripts/index/finalize_reparsed_store.py \
        --new-db data/indexes/duckdb_store_new.db --copy-ontology
    python scripts/index/finalize_reparsed_store.py \
        --new-db data/indexes/duckdb_store_new.db --verb-klaso
    python scripts/index/finalize_reparsed_store.py \
        --new-db data/indexes/duckdb_store_new.db --verify

Inputs:
    --new-db   the freshly built side store (read-write)
    --live-db  the current production store (attached read-only)

Outputs:
    Mutates --new-db (ontology tables, clauses.verb_klaso). Prints every
    check; exit code 0 = pass, 2 = a gate failed.

Quality Checks:
    See --verify above; --copy-ontology re-verifies its own row counts.

Last Updated: 2026-07-19
Author: Claude Fable 5 (with Marc Jones)
Related Issues: #807, #823, #871, #872, #713, #805
See Also: scripts/pipeline/reparse_store_807.sh, scripts/index/build_duckdb_store.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


def _count(con, table: str, db: str = '') -> int:
    prefix = f'{db}.' if db else ''
    return con.execute(f'SELECT count(*) FROM {prefix}{table}').fetchone()[0]


def copy_ontology(con) -> None:
    """Copy ontology_nodes/ontology_edges from the attached live store."""
    for table in ('ontology_nodes', 'ontology_edges'):
        live_n = _count(con, table, 'live')
        if live_n == 0:
            print(f'FATAL: live {table} is EMPTY — refusing to copy nothing '
                  f'over the new store (present-but-empty is the trap).')
            sys.exit(2)
        con.execute(f'DROP TABLE IF EXISTS {table}')
        con.execute(f'CREATE TABLE {table} AS SELECT * FROM live.{table}')
        new_n = _count(con, table)
        status = 'OK' if new_n == live_n else 'MISMATCH'
        print(f'  {table:16s} live={live_n:,} -> new={new_n:,}  [{status}]')
        if new_n != live_n:
            sys.exit(2)
    con.execute('CREATE INDEX IF NOT EXISTS idx_oedges_radiko '
                'ON ontology_edges(radiko)')
    print('  ontology copied (incl. #713 hand-fix + #872 ALIASO bridge)')


def populate_verb_klaso(con) -> None:
    """Populate clauses.verb_klaso from the typed root lexicon.

    Replicates the safe tail of load_ontology.py (#837) — the lexicon join
    only, via new-table-swap instead of a 6.9M-row in-place UPDATE.
    """
    tables = {t[0] for t in con.execute('SHOW TABLES').fetchall()}
    if 'clauses' not in tables:
        print('FATAL: no `clauses` table — run build_clause_table.py first.')
        sys.exit(2)

    from klareco.morphology import lexicon
    lex = lexicon()
    pairs = [(r, p) for r, p in lex.roots.items() if p]
    print(f'  typed lexicon roots: {len(pairs):,}')

    con.execute('DROP TABLE IF EXISTS _root_pos')
    con.execute('CREATE TABLE _root_pos (radiko VARCHAR, pos VARCHAR)')
    con.executemany('INSERT INTO _root_pos VALUES (?,?)', pairs)

    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'clauses' ORDER BY ordinal_position").fetchall()]
    keep = [c for c in cols if c != 'verb_klaso']
    col_list = ', '.join(f'c.{c}' for c in keep)

    con.execute('DROP TABLE IF EXISTS clauses_vk')
    con.execute(f"""
        CREATE TABLE clauses_vk AS
        SELECT {col_list}, r.pos AS verb_klaso
        FROM clauses c LEFT JOIN _root_pos r ON r.radiko = c.verb_radiko
    """)
    old_n, new_n = _count(con, 'clauses'), _count(con, 'clauses_vk')
    if old_n != new_n:
        print(f'FATAL: clause row count changed in swap ({old_n:,} -> {new_n:,})')
        sys.exit(2)
    con.execute('DROP TABLE clauses')
    con.execute('ALTER TABLE clauses_vk RENAME TO clauses')
    con.execute('DROP TABLE _root_pos')

    n = con.execute('SELECT count(*) FROM clauses '
                    'WHERE verb_klaso IS NOT NULL').fetchone()[0]
    print(f'  verb_klaso: {n:,}/{new_n:,} = {n / max(new_n, 1):.1%}')


def verify(con) -> None:
    fails: list[str] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
        if not ok:
            fails.append(name)

    live_rows = _count(con, 'sentences', 'live')
    new_rows = _count(con, 'sentences')
    drift = abs(new_rows - live_rows) / max(live_rows, 1)
    gate('row count', drift <= 0.02,
         f'live={live_rows:,} new={new_rows:,} drift={drift:.2%} (limit 2%)')

    n_junk = con.execute(
        "SELECT count(*) FROM sentences WHERE text LIKE 'REDIRECT%' "
        "OR text LIKE 'ALIDIREKTI%' OR text LIKE '#REDIRECT%' "
        "OR text LIKE '#ALIDIREKTI%'").fetchone()[0]
    gate('#823 junk', n_junk == 0, f'{n_junk} redirect stubs (must be 0)')

    live_subj = con.execute(
        'SELECT count(*) FROM live.sentences WHERE subj_radiko IS NOT NULL'
    ).fetchone()[0]
    new_subj = con.execute(
        'SELECT count(*) FROM sentences WHERE subj_radiko IS NOT NULL'
    ).fetchone()[0]
    gate('#871 subjects', new_subj >= live_subj,
         f'subj_radiko non-null: live={live_subj:,} new={new_subj:,} '
         f'(delta {new_subj - live_subj:+,}; reparse must not LOSE subjects)')

    lo, hi = con.execute(
        'SELECT min(success_rate), max(success_rate) FROM sentences').fetchone()
    gate('#805 success_rate', lo is not None and lo != hi,
         f'min={lo} max={hi} (a constant column carries zero information)')

    for table in ('ontology_nodes', 'ontology_edges'):
        ln, nn = _count(con, table, 'live'), _count(con, table)
        gate(f'{table}', ln == nn and nn > 0, f'live={ln:,} new={nn:,}')

    tables = {t[0] for t in con.execute('SHOW TABLES').fetchall()}
    for table in ('clauses', 'dependency_arcs'):
        n = _count(con, table) if table in tables else 0
        gate(f'{table} present', n > 0, f'{n:,} rows')

    # entity_facts is NOT a gate: the #807 pass deliberately does not rebuild it
    # (see reparse_store_807.sh stage 7 — reviving that answer path is #881 and
    # needs its own number). preflight treats it as required=False. Reported,
    # not gated, so its absence is visible rather than silent.
    n_ef = _count(con, 'entity_facts') if 'entity_facts' in tables else 0
    print(f'  [INFO] entity_facts: {n_ef:,} rows '
          f'(not rebuilt by design — #881 owns this; preflight: required=False)')

    # verb_negated must exist and vary — consumers read it (entity_fact_patterns
    # guards every pattern on it; ast_aware_reranker hard-filters on it).
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'sentences'").fetchall()}
    if 'verb_negated' not in cols:
        gate('verb_negated column', False, 'MISSING from sentences')
    else:
        n_true, n_false = con.execute(
            "SELECT count(*) FILTER (WHERE verb_negated), "
            "       count(*) FILTER (WHERE NOT verb_negated) FROM sentences"
        ).fetchone()
        gate('verb_negated distribution', n_true > 0 and n_false > 0,
             f'{n_true:,} negated / {n_false:,} not — a constant column would '
             f'silently disable every negation guard')

    if 'clauses' in tables:
        tot = _count(con, 'clauses')
        vk = con.execute('SELECT count(*) FROM clauses '
                         'WHERE verb_klaso IS NOT NULL').fetchone()[0]
        frac = vk / max(tot, 1)
        gate('verb_klaso coverage', frac >= 0.01,
             f'{vk:,}/{tot:,} = {frac:.1%} (preflight requires >= 1%)')

    if fails:
        print(f'\n!!! {len(fails)} GATE(S) FAILED: {", ".join(fails)} — '
              f'do NOT swap this store live.')
        sys.exit(2)
    print('\nall gates passed — the side store is swappable')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-db', required=True,
                    help='freshly built side store (read-write)')
    ap.add_argument('--live-db', default='data/indexes/duckdb_store.db',
                    help='current production store (attached read-only)')
    ap.add_argument('--copy-ontology', action='store_true')
    ap.add_argument('--verb-klaso', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    if not (args.copy_ontology or args.verb_klaso or args.verify):
        ap.error('need at least one of --copy-ontology / --verb-klaso / --verify')
    if not Path(args.new_db).exists():
        print(f'FATAL: {args.new_db} does not exist')
        return 2
    if not Path(args.live_db).exists():
        print(f'FATAL: {args.live_db} does not exist')
        return 2

    con = duckdb.connect(args.new_db)
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute(f"ATTACH '{args.live_db}' AS live (READ_ONLY)")

    if args.copy_ontology:
        print('== copy ontology from live store ==')
        copy_ontology(con)
    if args.verb_klaso:
        print('== populate clauses.verb_klaso ==')
        populate_verb_klaso(con)
    if args.verify:
        print('== verify side store vs live ==')
        verify(con)

    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
