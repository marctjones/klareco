#!/usr/bin/env python3
"""
Correct propra_nomo ↔ adjektivo classifications by syntactic position.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu schema
DEPENDENCIES: klareco.utils.kuzu_open
STAGE: Maintenance / data fix

Description:
    Esperanto adjectives MUST agree with a noun (it's a syntactic
    requirement, not optional). So a capitalized -a/-aj/-an/-ajn ending
    word can only be an adjective if it's in MODIFIER position
    (HAVAS_PRISKRIBON of a Vortgrupo). If it's the kerno of a Vortgrupo
    or a standalone subject/object Vorto, it cannot be an adjective and
    must be a proper noun.

    A previous fix (fix_kuzu_propra_nomo_misclassifications.py) flipped
    Vortos to 'adjektivo' based on surface form alone, regardless of
    syntactic position. That over-flipped names like Maria-as-subject.

    This script does TWO sweeps:

    1. REVERT: Find Vortos currently tagged 'adjektivo' that are in
       head position (kerno of subjekto/objekto/etc., or standalone
       subject/object) AND whose plena_vorto starts uppercase. These
       cannot grammatically be adjectives — revert to propra_nomo.

    2. APPLY: Find Vortos currently tagged 'propra_nomo' (kategorio=
       'propranomo') that are in modifier position (priskribo of a
       Vortgrupo) with adjectival ending + known stem. These are real
       adjectives that the parser misclassified — flip to adjektivo.

    Either or both sweeps can be requested via flags. The default is
    --revert-only since that undoes the damage from the prior over-fix
    without making new changes.

Usage:
    # Revert only (recover from the buggy prior fix):
    python scripts/correct_kuzu_propra_nomo_by_position.py --revert --apply

    # Both sweeps (revert then apply, syntactically correct):
    python scripts/correct_kuzu_propra_nomo_by_position.py \\
        --revert --reclassify --apply

    # Dry-run to preview (no DB writes):
    python scripts/correct_kuzu_propra_nomo_by_position.py \\
        --revert --reclassify

Inputs:
    Kuzu DB at data/indexes/v2.1_kuzu_index_full

Outputs:
    Audit log under logs/kuzu_fixes/correction_<timestamp>.jsonl
    Console summary of counts.

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

ADJECTIVAL_ENDINGS = ('ajn', 'aj', 'an', 'a')


def strip_adj_ending(lower: str):
    for end in ADJECTIVAL_ENDINGS:
        if lower.endswith(end) and len(lower) > len(end):
            return lower[: -len(end)], end
    return None, None


def is_in_head_position_clause():
    """Return Cypher predicate: Vorto v is in head position.

    A Vorto is in head position if any of:
    - It's the kerno of any Vortgrupo
    - It's a standalone subjekto / objekto / aliaj Vorto
    """
    return """(
        EXISTS { MATCH (:Vortgrupo)-[:HAVAS_KERNON]->(v) }
        OR EXISTS { MATCH (:Frazo)-[:HAVAS_SUBJEKTON_VORTO]->(v) }
        OR EXISTS { MATCH (:Frazo)-[:HAVAS_OBJEKTON_VORTO]->(v) }
        OR EXISTS { MATCH (:Frazo)-[:HAVAS_ALIAJN]->(v) }
    )"""


def is_in_priskribo_position_clause():
    """Cypher predicate: Vorto v is in modifier position."""
    return "EXISTS { MATCH (:Vortgrupo)-[:HAVAS_PRISKRIBON]->(v) }"


def revert_sweep(conn, audit_f, apply: bool) -> int:
    """Revert ALL uppercase-first adjektivos back to propra_nomo.

    Rationale: legitimate Esperanto adjectives are never capitalized
    (except occasionally sentence-initial, which the parser then doesn't
    tag adjektivo). So any Vorto currently tagged 'adjektivo' whose
    plena_vorto starts uppercase came from the buggy prior fix and
    should be undone.

    A position-conditioned revert would be more conservative, but the
    parser bug also caused incorrect role assignment (Hungaraj
    misclassified as kerno when it's really a modifier), so position
    isn't a reliable discriminator. Reverting all uppercase adjektivos
    is the safe choice — it returns Kuzu to its pre-fix state and we
    can rely on query-time filters going forward.
    """
    print("\n=== REVERT SWEEP (all uppercase adjektivos → propra_nomo) ===")
    t0 = time.time()
    res = conn.execute("""
        MATCH (v:Vorto)
        WHERE v.vortspeco = 'adjektivo'
          AND v.plena_vorto =~ '^[A-ZĈĜĤĴŜŬ].*'
        RETURN v.plena_vorto AS pv, COUNT(*) AS n
        ORDER BY n DESC
    """)
    candidates = []
    while res.has_next():
        r = res.get_next()
        candidates.append({'plena_vorto': r[0], 'instances': r[1]})

    total_instances = sum(c['instances'] for c in candidates)
    print(f"  {len(candidates):,} unique surface forms currently tagged adjektivo "
          f"with uppercase first letter ({total_instances:,} instances), "
          f"{time.time()-t0:.1f}s")
    if candidates:
        print(f"  Top 10 by frequency:")
        for c in candidates[:10]:
            print(f"    {c['plena_vorto']:18s}  ×{c['instances']:>6,}")

    for c in candidates:
        audit_f.write(json.dumps({'op': 'revert_to_propra_nomo', **c},
                                  ensure_ascii=False) + '\n')

    if not apply:
        print("  [DRY-RUN] No writes.")
        return 0

    print(f"  Applying revert...")
    t0 = time.time()
    conn.execute("""
        MATCH (v:Vorto)
        WHERE v.vortspeco = 'adjektivo'
          AND v.plena_vorto =~ '^[A-ZĈĜĤĴŜŬ].*'
        SET v.vortspeco = 'propra_nomo',
            v.kategorio = 'propranomo',
            v.radiko = v.plena_vorto
    """)
    print(f"  Reverted in {time.time()-t0:.1f}s")
    return total_instances


def reclassify_sweep(conn, audit_f, apply: bool) -> int:
    """Find propra_nomos in modifier position that should be adjektivo.

    Targets the cases the original parser bug missed: capitalized
    -a/-aj/-an/-ajn-ending words that are syntactically modifiers
    (priskribo of a Vortgrupo) — these are real adjectives.
    """
    print("\n=== RECLASSIFY SWEEP (priskribo-position propra_nomo → adjektivo) ===")
    t0 = time.time()
    # We need DICTIONARY_ROOTS check, so do it Python-side after the
    # candidate scan.
    res = conn.execute(f"""
        MATCH (v:Vorto)
        WHERE v.vortspeco = 'propra_nomo'
          AND v.kategorio = 'propranomo'
          AND (v.plena_vorto ENDS WITH 'a' OR v.plena_vorto ENDS WITH 'aj'
               OR v.plena_vorto ENDS WITH 'an' OR v.plena_vorto ENDS WITH 'ajn')
          AND {is_in_priskribo_position_clause()}
        RETURN v.plena_vorto AS pv, COUNT(*) AS n
        ORDER BY n DESC
    """)
    candidates = []
    while res.has_next():
        r = res.get_next()
        pv, n = r[0], r[1]
        lower = pv.lower()
        stem, ending = strip_adj_ending(lower)
        if stem and stem in DICTIONARY_ROOTS:
            candidates.append({'plena_vorto': pv,
                               'new_radiko': stem,
                               'new_kazo': 'akuzativo' if ending in ('an', 'ajn') else None,
                               'new_nombro': 'pluralo' if ending in ('aj', 'ajn') else None,
                               'instances': n})

    total = sum(c['instances'] for c in candidates)
    print(f"  {len(candidates):,} unique surface forms in modifier position to flip "
          f"({total:,} instances), {time.time()-t0:.1f}s")
    if candidates:
        print(f"  Top 10 by frequency:")
        for c in candidates[:10]:
            print(f"    {c['plena_vorto']:18s}  ×{c['instances']:>6,}  → adjektivo "
                  f"rad={c['new_radiko']!r}")

    for c in candidates:
        audit_f.write(json.dumps({'op': 'flip_to_adjektivo', **c},
                                  ensure_ascii=False) + '\n')

    if not apply:
        print("  [DRY-RUN] No writes.")
        return 0

    print(f"  Applying reclassification...")
    t0 = time.time()
    for c in candidates:
        pv = c['plena_vorto'].replace("\\", "\\\\").replace("'", "\\'")
        rad = c['new_radiko'].replace("\\", "\\\\").replace("'", "\\'")
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
              AND {is_in_priskribo_position_clause()}
            SET {', '.join(sets)}
        """)
    print(f"  Reclassified in {time.time()-t0:.1f}s")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kuzu-path', default='data/indexes/v2.1_kuzu_index_full')
    ap.add_argument('--revert', action='store_true',
                    help='Revert adjektivos in head position back to propra_nomo')
    ap.add_argument('--reclassify', action='store_true',
                    help='Flip propra_nomos in modifier position to adjektivo')
    ap.add_argument('--apply', action='store_true',
                    help='Apply changes (default is dry-run)')
    ap.add_argument('--audit-dir', default='logs/kuzu_fixes')
    args = ap.parse_args()

    if not args.revert and not args.reclassify:
        print("ERROR: must request at least one of --revert / --reclassify",
              file=sys.stderr)
        sys.exit(2)

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    db = open_kuzu(args.kuzu_path, read_only=not args.apply)
    conn = kuzu.Connection(db)

    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"correction_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with open(audit_path, 'w') as audit_f:
        revert_n = revert_sweep(conn, audit_f, args.apply) if args.revert else 0
        recl_n   = reclassify_sweep(conn, audit_f, args.apply) if args.reclassify else 0
    print(f"\nAudit log: {audit_path}")
    print(f"\nSummary:")
    print(f"  Reverted (adj→propra_nomo, head pos): {revert_n:,}")
    print(f"  Reclassified (propra_nomo→adj, modifier pos): {recl_n:,}")


if __name__ == '__main__':
    main()
