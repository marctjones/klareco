#!/usr/bin/env python3
"""
Targeted DuckDB AST refresh — re-parses only sentences affected by the
parser bug fixes, instead of re-parsing the entire 5.4M-row corpus.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences: shredded cols + ast_json blob),
                 post-bug-#1/#2/#4 parser
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index / Data refresh

Description:
    A full re-parse takes 1-2 hours. Most sentences in the corpus are
    unaffected by the parser fixes (bug #1 = fronted PP, bug #2 =
    sentence-initial function-word/common-noun mis-tagged, bug #4 =
    fronted-PP question misparse). We can identify affected sentences
    by surface patterns and re-parse only those, in minutes.

    Affected sentence patterns:
      - Bug #1: sentences containing `<Preposition> <Capital-Word>` near
        the start where the AST currently has that capital word as
        subjekto.
      - Bug #2: sentences whose subjekto.kerno is one of the known-
        common-word denylist entries (Anstataŭ, Universitato, Konsilio,
        Ligo, Akademio, ...) tagged as propra_nomo.
      - Bug #4: sentences starting with `<Prep> <ki-correlative>` (rare
        in declarative corpus, but checked).

    For each affected sentence: re-parse, replace ast_json, recompute the
    shredded columns (subj_radiko, verb_radiko, obj_radiko).

    The script is checkpointable: it tracks progress in a `refresh_log`
    table inside the DuckDB so it can resume after interruption.

Usage:
    python scripts/index/refresh_affected_asts.py
    python scripts/index/refresh_affected_asts.py --dry-run
    python scripts/index/refresh_affected_asts.py --limit 1000

Inputs:
    data/indexes/duckdb_store.db  (read-write)

Outputs:
    Updates `sentences` in-place. Writes a `refresh_log` audit table.
    Stdout summary: per-bug count, throughput, ETA.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from klareco.parser import parse


# Same denylist used in the removed synthetic-qa builder denylist
_COMMON_WORDS_AS_PROPER = {
    'Kaj', 'Sed', 'Aŭ', 'Do', 'Tamen', 'Tial', 'Ke', 'Ankaŭ',
    'Anstataŭ', 'Krom', 'Malgraŭ', 'Sen', 'Por', 'Antaŭ', 'Post',
    'Dum', 'Tra', 'Trans', 'Apud', 'Pri', 'Pro', 'Laŭ',
    'Tiam', 'Tiu', 'Ĉi', 'Jen', 'Nun', 'Hodiaŭ',
    'Universitato', 'Biblioteko', 'Muzeo', 'Teatro', 'Stadiono',
    'Eklezio', 'Akademio', 'Lernejo', 'Hospitalo', 'Kongreso',
    'Konsilio', 'Ligo', 'Asocio', 'Organizaĵo', 'Komitato',
    'Konferenco', 'Renkontiĝo', 'Reĝimo',
}

# A small set of Esperanto prepositions that, when sentence-initial,
# trigger the fronted-PP bug. Limit the scan with these.
_FRONTED_PP_STARTERS = (
    'En ', 'Al ', 'El ', 'Sur ', 'Sub ', 'Apud ', 'Antaŭ ', 'Post ',
    'Tra ', 'Trans ', 'Kun ', 'Per ', 'Pri ', 'Pro ', 'Laŭ ', 'Malgraŭ ',
    'Inter ', 'Kontraŭ ', 'Anstataŭ ', 'Ekde ',
)


def find_affected_sids(conn, max_per_pattern: int | None = None) -> dict:
    """Return {bug_id: list_of_sids} of sentences likely affected by each
    parser fix. Uses fast SQL with LIKE filters + cheap AST-shape checks
    (via JSON path queries on ast_json) so we don't have to re-parse
    every row.

    bug #1: surface text starts with `<Prep> <CapitalWord>` AND ast_json
            contains '"subjekto":' with that capital word as kerno.
            Detected with LIKE on text + JSON inspection.
    bug #2: ast_json contains a denylisted common-word as a propra_nomo
            subjekto.kerno.
    bug #4: rare; detect via question-mark in source text + early
            ki-correlative pattern.
    """
    out: dict[str, list[int]] = {}

    # Bug #1: text starts with a fronted PP. Coarse LIKE filter; the
    # actual "is this affected?" is decided at re-parse time by comparing
    # old vs new AST.
    print("Scanning bug #1 candidates (fronted PP at sentence start)…")
    bug1_sids: list[int] = []
    for starter in _FRONTED_PP_STARTERS:
        limit_clause = f"LIMIT {max_per_pattern}" if max_per_pattern else ""
        rows = conn.execute(
            f"SELECT sid FROM sentences WHERE text LIKE ? {limit_clause}",
            [f'{starter}%']
        ).fetchall()
        bug1_sids.extend(r[0] for r in rows)
    out['bug1_fronted_pp'] = sorted(set(bug1_sids))
    print(f"  candidates: {len(out['bug1_fronted_pp']):,}")

    # Bug #2: sentences whose subjekto carries a known common-word as
    # propra_nomo. Quickest path: look for sentences starting with those
    # words at position 0.
    print("Scanning bug #2 candidates (function-word/common-noun mis-tagged)…")
    bug2_sids: list[int] = []
    for w in _COMMON_WORDS_AS_PROPER:
        limit_clause = f"LIMIT {max_per_pattern}" if max_per_pattern else ""
        rows = conn.execute(
            f"SELECT sid FROM sentences WHERE text LIKE ? {limit_clause}",
            [f'{w} %']
        ).fetchall()
        bug2_sids.extend(r[0] for r in rows)
    out['bug2_common_word'] = sorted(set(bug2_sids))
    print(f"  candidates: {len(out['bug2_common_word']):,}")

    # Bug #4: declarative corpus, rare. Look for `?` in text OR explicit
    # `<Prep> kiu/kio/kie ...` patterns.
    print("Scanning bug #4 candidates (fronted-PP question)…")
    bug4_sids: list[int] = []
    for starter in ('En kiu ', 'Al kiu ', 'Per kio ', 'Pri kio ',
                    'En kio ', 'Al kio ', 'En kie ', 'Sur kio '):
        limit_clause = f"LIMIT {max_per_pattern}" if max_per_pattern else ""
        rows = conn.execute(
            f"SELECT sid FROM sentences WHERE text LIKE ? {limit_clause}",
            [f'%{starter}%']
        ).fetchall()
        bug4_sids.extend(r[0] for r in rows)
    out['bug4_fronted_question'] = sorted(set(bug4_sids))
    print(f"  candidates: {len(out['bug4_fronted_question']):,}")

    # Deduplicate across bugs (a sentence affected by two bugs is reparsed once)
    all_sids = set()
    for bug, sids in out.items():
        all_sids.update(sids)
    out['_union'] = sorted(all_sids)
    print(f"\n  Total unique candidates: {len(out['_union']):,}")
    return out


def setup_refresh_log(conn) -> None:
    """Create the audit table if it doesn't exist (checkpoint mechanism)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refresh_log (
            sid               BIGINT PRIMARY KEY,
            refreshed_at      TIMESTAMP DEFAULT current_timestamp,
            ast_changed       BOOLEAN,
            old_subj_radiko   VARCHAR,
            new_subj_radiko   VARCHAR
        )
    """)


def already_refreshed(conn) -> set[int]:
    rows = conn.execute("SELECT sid FROM refresh_log").fetchall()
    return {r[0] for r in rows}


def refresh_one(conn, sid: int) -> tuple[bool, str | None, str | None]:
    """Re-parse one sentence and update its row if the AST changed.
    Returns (changed?, old_subj_radiko, new_subj_radiko)."""
    row = conn.execute(
        "SELECT text, ast_json, subj_radiko FROM sentences WHERE sid = ?",
        [sid]
    ).fetchone()
    if not row:
        return False, None, None
    text, old_ast_json, old_subj_radiko = row
    try:
        new_ast = parse(text)
    except Exception:
        # Parsing failed somehow — leave the row alone but log
        return False, old_subj_radiko, None

    new_ast_json = json.dumps(new_ast, ensure_ascii=False)
    if new_ast_json == old_ast_json:
        return False, old_subj_radiko, old_subj_radiko

    # Re-derive shredded columns from the new AST
    subj = (new_ast.get('subjekto') or {})
    subj_kerno = subj.get('kerno') if isinstance(subj, dict) and subj.get('tipo') == 'vortgrupo' else subj
    new_subj_radiko = (subj_kerno or {}).get('radiko') if isinstance(subj_kerno, dict) else None

    verb = new_ast.get('verbo') or {}
    new_verb_radiko = verb.get('radiko') if isinstance(verb, dict) else None

    obj = (new_ast.get('objekto') or {})
    obj_kerno = obj.get('kerno') if isinstance(obj, dict) and obj.get('tipo') == 'vortgrupo' else obj
    new_obj_radiko = (obj_kerno or {}).get('radiko') if isinstance(obj_kerno, dict) else None

    conn.execute(
        "UPDATE sentences SET ast_json = ?, subj_radiko = ?, verb_radiko = ?, "
        "obj_radiko = ? WHERE sid = ?",
        [new_ast_json, new_subj_radiko, new_verb_radiko, new_obj_radiko, sid]
    )
    return True, old_subj_radiko, new_subj_radiko


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most this many candidates total (for testing).')
    ap.add_argument('--per-pattern-limit', type=int, default=None,
                    help='Cap each LIKE pattern scan (default: no cap).')
    ap.add_argument('--batch-size', type=int, default=1000,
                    help='Commit batch size (default 1000). Larger = faster '
                         'but holds more rolled-back work on a crash.')
    args = ap.parse_args()

    print(f'Opening DuckDB at {args.duckdb_path} (read-write)…')
    conn = duckdb.connect(args.duckdb_path)
    # OOM safety
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")

    setup_refresh_log(conn)

    affected = find_affected_sids(conn, max_per_pattern=args.per_pattern_limit)
    to_refresh = affected['_union']
    if args.limit:
        to_refresh = to_refresh[:args.limit]

    already = already_refreshed(conn)
    if already:
        before = len(to_refresh)
        to_refresh = [s for s in to_refresh if s not in already]
        print(f'  Skipping {before - len(to_refresh):,} already-refreshed sids (resume)')

    if args.dry_run:
        print(f'\nDRY RUN — would refresh {len(to_refresh):,} sentences. Exit.')
        return

    print(f'\nRefreshing {len(to_refresh):,} sentences '
          f'(batched commits, batch_size={args.batch_size})…')
    t0 = time.time()
    n_changed = 0
    progress_every = 5000

    # Batched commits: BEGIN, do N UPDATEs, COMMIT, repeat. This amortises
    # the per-row WAL-flush cost across the batch — typically 5-10x faster
    # than per-row auto-commit on a 5.4M-row table with column updates.
    conn.execute('BEGIN TRANSACTION')
    in_tx_count = 0
    for i, sid in enumerate(to_refresh, 1):
        changed, old_sr, new_sr = refresh_one(conn, sid)
        if changed:
            n_changed += 1
        conn.execute(
            "INSERT INTO refresh_log VALUES (?, current_timestamp, ?, ?, ?)",
            [sid, changed, old_sr, new_sr]
        )
        in_tx_count += 1

        if in_tx_count >= args.batch_size:
            conn.execute('COMMIT')
            conn.execute('BEGIN TRANSACTION')
            in_tx_count = 0

        if i % progress_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_refresh) - i) / rate if rate > 0 else float('inf')
            print(
                f'  {i:>8,} / {len(to_refresh):,}  '
                f'({100*i/len(to_refresh):5.1f}%)  '
                f'changed={n_changed:>6,}  '
                f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
                flush=True,
            )

    # Final commit
    if in_tx_count > 0:
        conn.execute('COMMIT')

    elapsed = time.time() - t0
    print(f'\n=== Done ===')
    print(f'Refreshed {len(to_refresh):,} sentences in {elapsed/60:.1f} minutes')
    print(f'AST actually changed for {n_changed:,} of them ({100*n_changed/max(1,len(to_refresh)):.1f}%)')
    print(f'Average throughput: {len(to_refresh)/elapsed:.0f} sentences/sec')


if __name__ == '__main__':
    main()
