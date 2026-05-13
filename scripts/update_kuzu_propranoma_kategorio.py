#!/usr/bin/env python3
"""
In-place update of Vorto.propranoma_kategorio from the merged dict.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu schema (Vorto with propranoma_kategorio field)
DEPENDENCIES: kuzu, klareco.proper_nouns, klareco.utils.kuzu_open
STAGE: Maintenance / data fix

Description:
    The Kuzu graph at data/indexes/v2.1_kuzu_index_full was indexed when
    the proper-noun dict had ~78% category pollution. Many Vorto nodes
    with vortspeco='propra_nomo' have wrong propranoma_kategorio
    (Tablotenisa tagged as place, Gizela tagged as place when she's a
    person, and lots of NULL).

    This script updates propranoma_kategorio in-place using the merged
    dict (data/proper_nouns_dynamic_v3.json), which combines the cleaned
    corpus dict with Esperanto-Wikipedia-derived categories.

    Strategy:
      1. Find all DISTINCT plena_vorto values among propra_nomo Vortos.
      2. For each, look up the v3 dict (using the same stripping logic
         the parser uses) → new category.
      3. Issue ONE Cypher UPDATE per surface form (touches all instances).

    The script is idempotent — running twice writes the same value.
    Backup is mandatory before --apply (the script can also create one).

    A separate audit log records every update for review and reversal.

Pipeline Position:
    data/proper_nouns_dynamic_v3.json
        + data/indexes/v2.1_kuzu_index_full
        → [THIS SCRIPT]
        → in-place updates to propranoma_kategorio on Vorto nodes

Usage:
    # Dry-run (counts only, no writes):
    python scripts/update_kuzu_propranoma_kategorio.py

    # Apply (will refuse without --backup-confirmed):
    python scripts/update_kuzu_propranoma_kategorio.py --apply --backup-confirmed

    # Apply with auto-backup:
    python scripts/update_kuzu_propranoma_kategorio.py --apply --auto-backup

Inputs:
    data/proper_nouns_dynamic_v3.json
    data/indexes/v2.1_kuzu_index_full

Outputs:
    Modifies Vorto.propranoma_kategorio in place
    logs/kuzu_updates/cat_update_<timestamp>.jsonl  (audit log)
    Console summary of changes by old→new category

Quality Checks:
    - Idempotent (rerun produces 0 additional changes)
    - Mandatory backup confirmation before write
    - Audit log enables full reversal
    - Per-surface batch (one query per unique plena_vorto)

Last Updated: 2026-05-07
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kuzu

from klareco.proper_nouns import ProperNounDictionary
from klareco.utils.kuzu_open import open_kuzu


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kuzu-path',
                    default='data/indexes/v2.1_kuzu_index_full')
    ap.add_argument('--dict-path',
                    default='data/proper_nouns_dynamic_v3.json')
    ap.add_argument('--apply', action='store_true',
                    help='Apply changes (default: dry-run)')
    ap.add_argument('--backup-confirmed', action='store_true',
                    help='Confirm that the Kuzu DB has been backed up')
    ap.add_argument('--auto-backup', action='store_true',
                    help='Make a copy of the Kuzu DB before applying changes')
    ap.add_argument('--audit-dir', default='logs/kuzu_updates')
    ap.add_argument('--limit', type=int, default=0,
                    help='Cap on unique-surface scans (0 = no cap; for testing)')
    args = ap.parse_args()

    kuzu_path = Path(args.kuzu_path)
    if not kuzu_path.exists():
        sys.exit(f'ERROR: Kuzu path not found: {kuzu_path}')

    dict_path = Path(args.dict_path)
    if not dict_path.exists():
        sys.exit(f'ERROR: dict not found: {dict_path}')

    if args.apply and not (args.backup_confirmed or args.auto_backup):
        sys.exit('ERROR: --apply requires either --backup-confirmed or --auto-backup')

    print(f'Mode: {"APPLY" if args.apply else "DRY-RUN"}')
    print(f'Kuzu: {kuzu_path}')
    print(f'Dict: {dict_path}')

    # Auto-backup
    if args.apply and args.auto_backup:
        backup_path = kuzu_path.parent / (kuzu_path.name + f'.bak_{datetime.now():%Y%m%d_%H%M%S}')
        print(f'Creating backup: {backup_path}  (this may take a few minutes for large DBs)')
        t0 = time.time()
        shutil.copytree(kuzu_path, backup_path)
        print(f'  backed up in {time.time()-t0:.1f}s')

    # Load the proper-noun dict (we use the lookup directly)
    pn = ProperNounDictionary(dynamic_path=dict_path)
    print(f'Loaded proper-noun dict: {len(pn):,} entries')

    db = open_kuzu(str(kuzu_path), read_only=not args.apply)
    conn = kuzu.Connection(db)

    # Step 1: scan unique plena_vorto values among propra_nomo Vortos.
    # We also pull the current propranoma_kategorio so we know whether
    # an update is even needed.
    print('\nStep 1: scanning Kuzu for distinct propra_nomo plena_vorto values...')
    t0 = time.time()
    limit_clause = f'LIMIT {args.limit}' if args.limit else ''
    res = conn.execute(f"""
        MATCH (v:Vorto)
        WHERE v.vortspeco = 'propra_nomo'
        RETURN v.plena_vorto AS pv,
               v.propranoma_kategorio AS old_cat,
               COUNT(*) AS instances
        {limit_clause}
    """)
    surfaces: dict[str, dict] = {}
    while res.has_next():
        r = res.get_next()
        pv, old_cat, instances = r[0], r[1], r[2]
        if pv is None:
            continue
        # multiple rows possible per plena_vorto if old_cat differs across instances
        key = pv
        if key not in surfaces:
            surfaces[key] = {'pv': pv, 'old_cats': Counter(), 'instances': 0}
        surfaces[key]['old_cats'][old_cat or ''] += instances
        surfaces[key]['instances'] += instances
    print(f'  {len(surfaces):,} unique surface forms; '
          f'{sum(s["instances"] for s in surfaces.values()):,} instances; '
          f'({time.time()-t0:.1f}s)')

    # Step 2: classify each surface via the dict
    plan: list[dict] = []
    not_in_dict = 0
    same_cat = 0
    upgrade_count = 0  # other / NULL → specific
    change_count = 0   # specific → different specific
    null_to_other = 0  # NULL → 'other' (clean-up)

    for pv, info in surfaces.items():
        new_cat = pn.get_category(pv)  # may be None if not in dict
        # Treat dict's None and 'other' equivalently for change detection
        for old_cat_value, n_instances in info['old_cats'].items():
            old = old_cat_value or None
            new = new_cat or None
            if old == new:
                same_cat += n_instances
                continue
            # Categorize the kind of change
            if not pn.is_proper_noun(pv):
                not_in_dict += n_instances
                continue
            if old in (None, '', 'other') and new not in (None, '', 'other'):
                upgrade_count += n_instances
            elif old not in (None, '', 'other') and new not in (None, '', 'other') and old != new:
                change_count += n_instances
            elif old in (None, '') and new == 'other':
                null_to_other += n_instances
            plan.append({
                'pv':           pv,
                'old_cat':      old_cat_value or None,
                'new_cat':      new_cat,
                'instances':    n_instances,
            })

    print(f'\nStep 2: classification summary')
    print(f'  Already-correct (no update needed):    {same_cat:>9,}')
    print(f'  Surface not in v3 dict (skip):         {not_in_dict:>9,}')
    print(f'  NULL/other → specific (upgrade):       {upgrade_count:>9,}')
    print(f'  Specific → different specific:         {change_count:>9,}')
    print(f'  NULL → other (cleanup):                {null_to_other:>9,}')
    print(f'  Total updates planned:                 {len(plan):>9,}')

    # Per-transition matrix
    transitions = Counter()
    for p in plan:
        transitions[(p['old_cat'] or 'NULL', p['new_cat'] or 'NULL')] += p['instances']
    if transitions:
        print(f'\n  Top 12 transitions by instance count:')
        for (old, new), n in transitions.most_common(12):
            print(f'    {old or "NULL":12s} → {new or "NULL":12s}  {n:>9,}')

    # Sample preview
    if plan:
        print(f'\n  Sample updates (first 10):')
        for p in plan[:10]:
            print(f'    {p["pv"]:25s}  {p["old_cat"] or "NULL":10s} → {p["new_cat"] or "NULL":10s}  ×{p["instances"]:,}')

    if not args.apply:
        print('\n[DRY-RUN] No writes. Re-run with --apply --auto-backup to commit.')
        return

    # Step 3: write audit log first
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f'cat_update_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    with open(audit_path, 'w', encoding='utf-8') as f:
        for p in plan:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'\nAudit log: {audit_path}')

    # Step 4: apply updates — one Cypher UPDATE per (pv, new_cat)
    print(f'\nApplying {len(plan):,} update statements...')
    t0 = time.time()
    applied = 0
    for p in plan:
        pv_escaped = p['pv'].replace("\\", "\\\\").replace("'", "\\'")
        new_cat = p['new_cat']
        if new_cat is None:
            # Set to NULL
            conn.execute(f"""
                MATCH (v:Vorto)
                WHERE v.vortspeco = 'propra_nomo'
                  AND v.plena_vorto = '{pv_escaped}'
                SET v.propranoma_kategorio = NULL
            """)
        else:
            new_cat_escaped = new_cat.replace("'", "\\'")
            conn.execute(f"""
                MATCH (v:Vorto)
                WHERE v.vortspeco = 'propra_nomo'
                  AND v.plena_vorto = '{pv_escaped}'
                SET v.propranoma_kategorio = '{new_cat_escaped}'
            """)
        applied += 1
        if applied % 5000 == 0:
            print(f'  ...{applied:,}/{len(plan):,} ({time.time()-t0:.1f}s)')
    print(f'\nApplied {applied:,} updates in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
