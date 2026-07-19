#!/usr/bin/env python3
"""
Fold Wikipedia-redirect aliases into ontology_edges as an ALIASO relation (#872).

VERSION: v1.0
COMPATIBLE WITH: DuckDB store (ontology_edges: rel, radiko, class_id)
DEPENDENCIES: duckdb; data/indexes/alias_table.json (from build_alias_table.py)
STAGE: Index / Data

Description:
    The alias bridge (#865) proved gate-passing (+0.0226 MRR on the alias_variant
    band, p=0.0014, zero control regression) but was parked: a 224k-entry
    standalone JSON is the wrong HOME. Aliases belong in the store as one source
    of truth, so the retriever reads them like any other ontology relation. This
    loads `alias_table.json` into ontology_edges as
        rel='ALIASO', radiko=<lowercased alias>, class_id=<canonical title>
    Idempotent: clears prior ALIASO rows first. Non-destructive to other rels.

Usage:
    python scripts/index/load_aliaso_edges.py            # dry-run (counts only)
    python scripts/index/load_aliaso_edges.py --apply    # write ALIASO rows

Inputs:  data/indexes/alias_table.json  {alias: canonical_title}
Outputs: ontology_edges rows with rel='ALIASO'
Quality: reports row delta; a bare read-only reversal is
         `DELETE FROM ontology_edges WHERE rel='ALIASO'`.

Last Updated: 2026-07-19
Author: Claude Opus 4.8
Related Issues: #872, #865
See Also: scripts/index/build_alias_table.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'data' / 'indexes' / 'duckdb_store.db'
ALIASES = ROOT / 'data' / 'indexes' / 'alias_table.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    if not ALIASES.exists():
        print(f"ERROR: {ALIASES} missing (run build_alias_table.py)", file=sys.stderr)
        sys.exit(1)
    aliases = json.loads(ALIASES.read_text())
    # (radiko=lowercased alias, class_id=canonical title); dedup on (alias)
    rows = []
    seen = set()
    for alias, canon in aliases.items():
        k = (alias or '').strip().lower()
        if not k or not canon or k in seen:
            continue
        seen.add(k)
        rows.append((k, str(canon)))
    print(f"alias_table.json: {len(aliases):,} entries -> {len(rows):,} ALIASO rows")

    con = duckdb.connect(str(DB), read_only=not a.apply)
    before = con.execute("SELECT count(*) FROM ontology_edges").fetchone()[0]
    existing = con.execute("SELECT count(*) FROM ontology_edges WHERE rel='ALIASO'").fetchone()[0]
    print(f"ontology_edges: {before:,} rows ({existing:,} existing ALIASO)")

    if not a.apply:
        print("DRY-RUN — pass --apply to write. Reversal: "
              "DELETE FROM ontology_edges WHERE rel='ALIASO'")
        return

    con.execute("DELETE FROM ontology_edges WHERE rel='ALIASO'")
    con.executemany(
        "INSERT INTO ontology_edges (rel, radiko, class_id) VALUES ('ALIASO', ?, ?)",
        rows)
    after = con.execute("SELECT count(*) FROM ontology_edges").fetchone()[0]
    aliaso = con.execute("SELECT count(*) FROM ontology_edges WHERE rel='ALIASO'").fetchone()[0]
    con.close()
    print(f"APPLIED: ontology_edges {before:,} -> {after:,}  (ALIASO={aliaso:,})")


if __name__ == '__main__':
    main()
