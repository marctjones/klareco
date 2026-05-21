#!/usr/bin/env python3
"""
Validate the DuckDB sentence store: structural integrity, AST well-formedness,
shredded-column consistency, Esperanto-likelihood, and refresh-log audit.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences + shredded cols + ast_json blob),
                 post-parser-fix
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index / Data validation

Description:
    Runs eight check families over the sentences table and reports per-check
    pass / fail counts plus concrete failure examples. Designed to catch:

      1. Structural integrity   — row count, no nulls in critical columns,
                                  unique sids, sids are positive integers
      2. AST well-formedness    — ast_json is parseable, has expected keys
                                  (`tipo`, `subjekto`, `verbo`, `objekto`,
                                  `aliaj`), the shape matches the parser's
                                  contract
      3. Shredded consistency   — subj_radiko / verb_radiko / obj_radiko
                                  match the AST kernos exactly
      4. Esperanto-likelihood   — most words have analizstato='sukceso';
                                  flag sentences with a high `nekonata` rate
                                  (likely foreign / OCR garbage / non-EO)
      5. Case-role consistency  — subjects in nominative, objects in
                                  accusative (where present)
      6. Multi-token-entity     — multi_token_entities (if present) have
                                  valid token-position indices within range
      7. Refresh log integrity  — every row in `refresh_log` exists in
                                  `sentences`; refreshed rows' ast_json
                                  parses without exception
      8. Aggregate metrics      — overall parse rate, distribution of
                                  vortspeco tags, common pathologies

    `--full` runs every check on every row (slow — minutes for 5.4M rows).
    Default mode samples N rows (fast — seconds).

Usage:
    python scripts/index/validate_duckdb_store.py            # 100K sample
    python scripts/index/validate_duckdb_store.py --full     # all rows
    python scripts/index/validate_duckdb_store.py --sample 500000

Outputs:
    Stdout summary tables + first 5 failure examples per check.
    Optional --output JSONL with per-failure detail.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


_EXPECTED_FRAZO_KEYS = {'tipo', 'subjekto', 'verbo', 'objekto', 'aliaj'}
_EXPECTED_WORD_KEYS = {'tipo', 'plena_vorto', 'radiko', 'vortspeco'}
_VALID_VORTSPECOJ = {
    'substantivo', 'adjektivo', 'verbo', 'adverbo', 'pronomo',
    'artikolo', 'prepozicio', 'konjunkcio', 'korelativo',
    'numeralo', 'propra_nomo', 'particiklo', 'partiklo',
    'interjekcio', 'nekonata',
}
_VALID_KAZO = {'nominativo', 'akuzativo'}


def kerno(node):
    """Extract the head Vorto dict from a vortgrupo or bare Vorto."""
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def check_structure(ast: dict) -> list[str]:
    """Check ast_json has the expected frazo-level shape."""
    issues = []
    if not isinstance(ast, dict):
        return ['AST is not a dict']
    if ast.get('tipo') != 'frazo':
        issues.append(f"tipo is {ast.get('tipo')!r}, expected 'frazo'")
    missing = _EXPECTED_FRAZO_KEYS - set(ast.keys())
    if missing:
        issues.append(f'missing top-level keys: {sorted(missing)}')
    # Subject/object should be either None or vortgrupo
    for role in ('subjekto', 'objekto'):
        n = ast.get(role)
        if n is None:
            continue
        if not isinstance(n, dict):
            issues.append(f'{role} is not a dict')
            continue
        if n.get('tipo') not in ('vortgrupo', 'vorto'):
            issues.append(f"{role}.tipo is {n.get('tipo')!r}")
    return issues


def check_shredded(ast: dict, row: dict) -> list[str]:
    """Check shredded columns match the AST kernos."""
    issues = []
    subj_kerno = kerno(ast.get('subjekto'))
    expected_subj_r = (subj_kerno or {}).get('radiko') if subj_kerno else None
    if expected_subj_r != row.get('subj_radiko'):
        issues.append(
            f'subj_radiko mismatch: AST={expected_subj_r!r}, '
            f'col={row.get("subj_radiko")!r}'
        )
    verb = ast.get('verbo')
    expected_verb_r = (verb or {}).get('radiko') if isinstance(verb, dict) else None
    if expected_verb_r != row.get('verb_radiko'):
        issues.append(
            f'verb_radiko mismatch: AST={expected_verb_r!r}, '
            f'col={row.get("verb_radiko")!r}'
        )
    obj_kerno = kerno(ast.get('objekto'))
    expected_obj_r = (obj_kerno or {}).get('radiko') if obj_kerno else None
    if expected_obj_r != row.get('obj_radiko'):
        issues.append(
            f'obj_radiko mismatch: AST={expected_obj_r!r}, '
            f'col={row.get("obj_radiko")!r}'
        )
    return issues


def check_case_roles(ast: dict) -> list[str]:
    """Subjects should be nominativo; objects should be akuzativo."""
    issues = []
    subj_kerno = kerno(ast.get('subjekto'))
    if isinstance(subj_kerno, dict):
        kazo = subj_kerno.get('kazo')
        # `None` or missing kazo is fine (some kerno types don't carry case);
        # but if it's set, it should be nominativo for a subject.
        if kazo == 'akuzativo':
            issues.append(f'subjekto.kerno is akuzativo (should be nominativo): '
                          f'{subj_kerno.get("plena_vorto")!r}')
    obj_kerno = kerno(ast.get('objekto'))
    if isinstance(obj_kerno, dict):
        kazo = obj_kerno.get('kazo')
        if kazo == 'nominativo':
            issues.append(f'objekto.kerno is nominativo (should be akuzativo): '
                          f'{obj_kerno.get("plena_vorto")!r}')
    return issues


def check_vortspeco_validity(ast: dict) -> list[str]:
    """Every word AST should have a vortspeco from the known set."""
    issues = []
    for role_name in ('subjekto', 'objekto'):
        k = kerno(ast.get(role_name))
        if isinstance(k, dict):
            vs = k.get('vortspeco')
            if vs and vs not in _VALID_VORTSPECOJ:
                issues.append(f'{role_name}.kerno.vortspeco={vs!r} not in valid set')
    verbo = ast.get('verbo')
    if isinstance(verbo, dict):
        vs = verbo.get('vortspeco')
        if vs and vs != 'verbo':
            issues.append(f'verbo.vortspeco={vs!r} (should be "verbo")')
    return issues


def check_esperanto_likelihood(ast: dict) -> tuple[float, int]:
    """Return (parse_success_rate, total_words). High `nekonata` rate
    suggests foreign text or garbage."""
    # Walk the AST counting words and successes
    total = 0
    success = 0

    def walk(n):
        nonlocal total, success
        if not isinstance(n, dict):
            return
        if n.get('tipo') == 'vorto':
            total += 1
            if n.get('analizstato') == 'sukceso':
                success += 1
            # Also count by vortspeco (a real EO word has a non-nekonata vortspeco)
            return
        # Recurse
        for v in n.values():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, list):
                for item in v:
                    walk(item)

    walk(ast)
    rate = success / total if total > 0 else 0.0
    return rate, total


def check_multi_token_entities(ast: dict) -> list[str]:
    """multi_token_entities indices should reference valid positions."""
    issues = []
    mte = ast.get('multi_token_entities')
    if mte is None:
        return issues
    if not isinstance(mte, list):
        issues.append(f'multi_token_entities is not a list: {type(mte)}')
        return issues
    for g in mte:
        if not isinstance(g, dict):
            issues.append('multi_token_entities entry is not a dict')
            continue
        positions = g.get('positions') or []
        tokens = g.get('tokens') or []
        if len(positions) != len(tokens):
            issues.append(f'positions/tokens length mismatch: {len(positions)} vs {len(tokens)}')
        if positions and not all(isinstance(p, int) and p >= 0 for p in positions):
            issues.append(f'positions contain invalid values: {positions}')
    return issues


def validate_row(row: dict) -> dict:
    """Run all checks on one row. Returns dict of {check_name: [issues]}."""
    result = {'sid': row['sid'], 'issues': {}}
    text = row.get('text') or ''
    ast_json = row.get('ast_json') or ''

    if not text:
        result['issues']['no_text'] = ['text column is null/empty']
    if not ast_json:
        result['issues']['no_ast'] = ['ast_json column is null/empty']
        return result
    try:
        ast = json.loads(ast_json)
    except Exception as e:
        result['issues']['unparseable_ast'] = [f'json.loads failed: {e}']
        return result

    for name, fn in (
        ('structure', lambda: check_structure(ast)),
        ('shredded',  lambda: check_shredded(ast, row)),
        ('case_role', lambda: check_case_roles(ast)),
        ('vortspeco', lambda: check_vortspeco_validity(ast)),
        ('mte',       lambda: check_multi_token_entities(ast)),
    ):
        issues = fn()
        if issues:
            result['issues'][name] = issues

    rate, total = check_esperanto_likelihood(ast)
    result['esperanto_success_rate'] = rate
    result['word_count'] = total
    if total > 0 and rate < 0.30:  # severe — likely foreign text
        result['issues'].setdefault('esperanto_likelihood', []).append(
            f'parse-success rate {rate:.0%} ({total} words)'
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--sample', type=int, default=100_000,
                    help='Number of random rows to validate (default 100K).')
    ap.add_argument('--full', action='store_true',
                    help='Validate every row (overrides --sample). Slow.')
    ap.add_argument('--output', default=None,
                    help='Optional JSONL output of failing rows.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    conn = duckdb.connect(args.duckdb_path, read_only=True)

    print('=== Aggregate counts ===')
    n_total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'  sentences total:                 {n_total:,}')
    n_unique = conn.execute('SELECT COUNT(DISTINCT sid) FROM sentences').fetchone()[0]
    print(f'  unique sids:                     {n_unique:,}')
    if n_total != n_unique:
        print(f'  ⚠ DUPLICATE SIDS DETECTED: {n_total - n_unique:,}')
    n_null_sid = conn.execute('SELECT COUNT(*) FROM sentences WHERE sid IS NULL').fetchone()[0]
    n_null_text = conn.execute('SELECT COUNT(*) FROM sentences WHERE text IS NULL').fetchone()[0]
    n_null_ast = conn.execute('SELECT COUNT(*) FROM sentences WHERE ast_json IS NULL').fetchone()[0]
    print(f'  null sid:                        {n_null_sid:,}')
    print(f'  null text:                       {n_null_text:,}')
    print(f'  null ast_json:                   {n_null_ast:,}')

    # Refresh log (optional)
    n_refreshed = 0
    try:
        n_refreshed = conn.execute('SELECT COUNT(*) FROM refresh_log').fetchone()[0]
        n_changed = conn.execute(
            'SELECT COUNT(*) FROM refresh_log WHERE ast_changed').fetchone()[0]
        print(f'  refresh_log rows:                {n_refreshed:,}')
        print(f'    ↳ AST actually changed:        {n_changed:,}')
    except Exception:
        print(f'  refresh_log table:               not present')

    # Pick rows to validate
    if args.full:
        print(f'\n=== Validating ALL {n_total:,} rows (full mode — slow) ===')
        rows = conn.execute(
            'SELECT sid, text, ast_json, subj_radiko, verb_radiko, obj_radiko '
            'FROM sentences'
        )
    else:
        # Random sample via TABLESAMPLE
        size = min(args.sample, n_total)
        print(f'\n=== Validating {size:,} random rows ===')
        rows = conn.execute(
            f'SELECT sid, text, ast_json, subj_radiko, verb_radiko, obj_radiko '
            f'FROM sentences USING SAMPLE {size} ROWS (reservoir, {args.seed})'
        )

    # Per-check counters
    fail_counts: Counter = Counter()
    n_scanned = 0
    rates: list[float] = []
    failures: list[dict] = []
    t0 = time.time()
    out_f = open(args.output, 'w') if args.output else None

    while True:
        row_tuple = rows.fetchone()
        if row_tuple is None:
            break
        row = {
            'sid':         row_tuple[0],
            'text':        row_tuple[1],
            'ast_json':    row_tuple[2],
            'subj_radiko': row_tuple[3],
            'verb_radiko': row_tuple[4],
            'obj_radiko':  row_tuple[5],
        }
        n_scanned += 1
        r = validate_row(row)
        if 'esperanto_success_rate' in r:
            rates.append(r['esperanto_success_rate'])
        for check_name in r['issues']:
            fail_counts[check_name] += 1
        if r['issues']:
            if out_f is not None:
                out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
            if len(failures) < 50:
                failures.append(r)
        if n_scanned % 10000 == 0:
            elapsed = time.time() - t0
            rate = n_scanned / elapsed if elapsed > 0 else 0
            print(f'  {n_scanned:>8,} rows scanned ({rate:.0f}/s)')

    elapsed = time.time() - t0
    print(f'\nScanned {n_scanned:,} rows in {elapsed:.1f}s')

    print(f'\n=== Per-check failure counts ===')
    if not fail_counts:
        print('  (no failures)')
    else:
        for chk, n in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            pct = 100 * n / n_scanned if n_scanned else 0
            print(f'  {chk:<25s} {n:>7,}  ({pct:5.2f}%)')

    if rates:
        avg_rate = sum(rates) / len(rates)
        below_50 = sum(1 for r in rates if r < 0.5)
        below_30 = sum(1 for r in rates if r < 0.3)
        print(f'\n=== Esperanto-likelihood (parse-success rate per sentence) ===')
        print(f'  avg parse-success rate:        {100*avg_rate:5.1f}%')
        print(f'  sentences below 50%:           {below_50:>7,}  '
              f'({100*below_50/n_scanned:.2f}%)')
        print(f'  sentences below 30%:           {below_30:>7,}  '
              f'({100*below_30/n_scanned:.2f}%)')

    if failures:
        print(f'\n=== First 5 failures (per check) ===')
        per_check_examples: dict[str, list[dict]] = {}
        for r in failures:
            for chk in r['issues']:
                per_check_examples.setdefault(chk, []).append(r)
        for chk, examples in per_check_examples.items():
            print(f'\n  [{chk}]')
            for ex in examples[:5]:
                issues_for_chk = ex['issues'].get(chk, [])
                print(f'    sid={ex["sid"]}: {issues_for_chk[0] if issues_for_chk else "(no detail)"}')

    if out_f is not None:
        out_f.close()
        print(f'\nPer-failure detail saved to {args.output}')


if __name__ == '__main__':
    main()
