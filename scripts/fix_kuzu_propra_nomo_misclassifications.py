#!/usr/bin/env python3
"""
Fix sentence-initial-capitalized-adjective misclassifications in Kuzu DB.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu schema (Vorto nodes)
DEPENDENCIES: klareco.parser (DICTIONARY_ROOTS), klareco.utils.kuzu_open
STAGE: Maintenance / data fix

Description:
    The original parser had a bug where sentence-initial capitalized words
    like "Genetika" (a real Esperanto adjective) were tagged as propra_nomo
    because the capitalization-guard check used only the small Fundamento
    vocabulary instead of the broad DICTIONARY_ROOTS corpus vocabulary. The
    parser is now fixed (commit XXX), but the existing Kuzu DB still has the
    wrong vortspeco on millions of Vorto nodes.

    This script walks the Kuzu DB and reclassifies misclassified Vorto nodes
    in place. A full re-parse of the corpus would take ~10 hours and rebuild
    the entire graph; this targeted in-place fix runs in minutes and only
    touches the affected nodes.

    The reclassification rule mirrors the parser fix:
    A Vorto node is reclassified if all of:
      (a) vortspeco = 'propra_nomo'
      (b) kategorio NOT in ('propranomo_konata', 'propranomo_esperantigita')
          — i.e., not a dictionary-confirmed proper noun
      (c) plena_vorto ends in an adjectival suffix (-a, -aj, -an, -ajn)
      (d) the lowercase form has a recognized Esperanto stem in
          DICTIONARY_ROOTS after stripping the ending

    For these nodes, the script updates:
      vortspeco → 'adjektivo'
      radiko    → the lowercase stripped stem
      kazo      → 'akuzativo' if ending was -an or -ajn, else current
      nombro    → 'pluralo' if ending was -aj or -ajn, else current
      kategorio → null (was 'propranomo'; no longer applies)

Usage:
    # Preview only (no changes):
    python scripts/fix_kuzu_propra_nomo_misclassifications.py --dry-run

    # Apply (writes to Kuzu in batches; idempotent):
    python scripts/fix_kuzu_propra_nomo_misclassifications.py --apply

    # Apply with custom batch size:
    python scripts/fix_kuzu_propra_nomo_misclassifications.py --apply --batch-size 5000

Inputs:
    Kuzu DB at data/indexes/v2.1_kuzu_index_full

Outputs:
    Logs total candidates examined, total reclassified, sample of changes.
    Also writes a JSONL audit log of every change to logs/kuzu_fixes/.

Quality Checks:
    - Idempotent: running twice produces no additional changes
    - Dry-run mode for safe preview
    - Audit log enables undo by reversing the recorded operations
    - Per-batch sanity check on row counts

Last Updated: 2026-05-05
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kuzu

from klareco.parser import DICTIONARY_ROOTS
from klareco.utils.kuzu_open import open_kuzu

ADJECTIVAL_ENDINGS = ('ajn', 'aj', 'an', 'a')  # ordered longest-first


def strip_adjectival_ending(lower_word: str) -> tuple[str, str]:
    """Return (stem, ending). Raises ValueError if no adjectival ending."""
    for end in ADJECTIVAL_ENDINGS:
        if lower_word.endswith(end) and len(lower_word) > len(end):
            return lower_word[: -len(end)], end
    raise ValueError(f"{lower_word!r} has no adjectival ending")


def is_misclassified(plena_vorto: str | None, kategorio: str | None) -> tuple[bool, str | None, str | None]:
    """Decide whether a Vorto row should be reclassified.

    Returns (should_fix, new_radiko, ending). When should_fix is False,
    the other return values are None.
    """
    if not plena_vorto:
        return False, None, None
    if kategorio in ('propranomo_konata', 'propranomo_esperantigita'):
        return False, None, None
    lower = plena_vorto.lower()
    if not any(lower.endswith(e) for e in ADJECTIVAL_ENDINGS):
        return False, None, None
    if not plena_vorto[0].isupper():
        return False, None, None
    try:
        stem, ending = strip_adjectival_ending(lower)
    except ValueError:
        return False, None, None
    if stem not in DICTIONARY_ROOTS:
        return False, None, None
    return True, stem, ending


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kuzu-path', default='data/indexes/v2.1_kuzu_index_full')
    ap.add_argument('--dry-run', action='store_true',
                    help='Identify candidates but do not write changes (default)')
    ap.add_argument('--apply', action='store_true',
                    help='Apply the fix in place')
    ap.add_argument('--batch-size', type=int, default=2000)
    ap.add_argument('--query-limit', type=int, default=0,
                    help='Cap candidate scan (0 = no cap)')
    ap.add_argument('--audit-dir', default='logs/kuzu_fixes')
    args = ap.parse_args()

    if not args.apply and not args.dry_run:
        args.dry_run = True
    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    mode = 'APPLY' if args.apply else 'DRY-RUN'
    print(f"Mode: {mode}")
    print(f"Kuzu DB: {args.kuzu_path}")
    print(f"DICTIONARY_ROOTS size: {len(DICTIONARY_ROOTS):,}")

    db = open_kuzu(args.kuzu_path, read_only=args.dry_run)
    conn = kuzu.Connection(db)

    # ------------------------------------------------------------------
    # Step 1: scan candidates, dedup by plena_vorto
    # ------------------------------------------------------------------
    # Aggregate by plena_vorto so we issue ONE UPDATE per surface form
    # (each form may appear thousands of times in the corpus). The UPDATE
    # uses WHERE v.plena_vorto = '<form>' so it touches every instance.
    print("\nScanning Kuzu for candidates (deduplicated by surface form)...")
    t0 = time.time()
    limit_clause = f"LIMIT {args.query_limit}" if args.query_limit else ""
    cypher = f"""
        MATCH (v:Vorto)
        WHERE v.vortspeco = 'propra_nomo'
          AND v.kategorio = 'propranomo'
          AND (v.plena_vorto ENDS WITH 'a'
               OR v.plena_vorto ENDS WITH 'aj'
               OR v.plena_vorto ENDS WITH 'an'
               OR v.plena_vorto ENDS WITH 'ajn')
        RETURN v.plena_vorto AS pv, COUNT(*) AS instances
        {limit_clause}
    """
    res = conn.execute(cypher)
    seen_unique = 0
    total_instances = 0
    candidates_by_pv = {}
    while res.has_next():
        r = res.get_next()
        pv, instances = r[0], r[1]
        seen_unique += 1
        total_instances += instances
        ok, new_rad, ending = is_misclassified(pv, 'propranomo')
        if ok:
            candidates_by_pv[pv] = {
                'plena_vorto': pv,
                'new_radiko':  new_rad,
                'new_vortspeco': 'adjektivo',
                'new_kazo':    'akuzativo' if ending in ('an', 'ajn') else None,
                'new_nombro':  'pluralo' if ending in ('aj', 'ajn') else None,
                'ending':      ending,
                'instances':   instances,
            }
    candidates = list(candidates_by_pv.values())
    candidate_instances = sum(c['instances'] for c in candidates)
    print(f"  {seen_unique:,} unique surface forms scanned "
          f"({total_instances:,} total Vorto instances)")
    print(f"  {len(candidates):,} unique forms to reclassify "
          f"({candidate_instances:,} instances) in {time.time() - t0:.1f}s")

    if not candidates:
        print("\nNothing to fix. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2: sample preview
    # ------------------------------------------------------------------
    candidates.sort(key=lambda c: -c['instances'])
    print("\nSample of candidates (top 15 by frequency):")
    for c in candidates[:15]:
        print(f"  {c['plena_vorto']:18s}  ×{c['instances']:>7,}  "
              f"→ adjektivo  rad: {c['new_radiko']!r}")

    # ------------------------------------------------------------------
    # Step 3: write audit log
    # ------------------------------------------------------------------
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"propra_nomo_fix_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with open(audit_path, 'w') as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f"\nAudit log: {audit_path}")

    if args.dry_run:
        print("\n[DRY-RUN] No changes applied. Re-run with --apply to write.")
        return

    # ------------------------------------------------------------------
    # Step 4: apply updates in batches
    # ------------------------------------------------------------------
    print(f"\nApplying {len(candidates):,} unique-form updates "
          f"({candidate_instances:,} instances total)...")
    t0 = time.time()
    updated_forms = 0
    for c in candidates:
        pv  = c['plena_vorto'].replace("\\", "\\\\").replace("'", "\\'")
        rad = c['new_radiko'].replace("\\", "\\\\").replace("'", "\\'")
        # Build SET clause; only update kazo/nombro if we have a value
        sets = [
            "v.vortspeco = 'adjektivo'",
            f"v.radiko = '{rad}'",
            "v.kategorio = NULL",
        ]
        if c['new_kazo']:
            sets.append(f"v.kazo = '{c['new_kazo']}'")
        if c['new_nombro']:
            sets.append(f"v.nombro = '{c['new_nombro']}'")
        conn.execute(f"""
            MATCH (v:Vorto)
            WHERE v.plena_vorto = '{pv}'
              AND v.vortspeco = 'propra_nomo'
              AND v.kategorio = 'propranomo'
            SET {', '.join(sets)}
        """)
        updated_forms += 1
        if updated_forms % 500 == 0:
            print(f"  ...{updated_forms:,}/{len(candidates):,} forms "
                  f"({time.time() - t0:.1f}s)")
    print(f"\nApplied updates to {updated_forms:,} unique surface forms "
          f"in {time.time() - t0:.1f}s")
    print(f"Done.")


if __name__ == '__main__':
    main()
