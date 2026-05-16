#!/usr/bin/env python3
"""
Tier 0: Load hand-curated root annotations into the Kuzu v2.2 ontology

VERSION: v2.2
COMPATIBLE WITH: v2.1 Kuzu DB + v2.2 ontology schema (extend_kuzu_schema_semantic_ontology.py)
DEPENDENCIES: data/annotations/phase_0_roots_40_more.jsonl
STAGE: Index

The schema-extension script seeds the ontology *taxonomy* (VerbaKlaso,
EntecaTipo, AspektaKlaso nodes) but leaves the Radiko -> class EDGES
empty. This loader wires the existing on-disk hand annotations into
those edges so verb-class / entity-type queries return data.

Scope (deliberately minimal and judgment-free):
  - data/annotations/phase_0_roots_40_more.jsonl is loaded. Its schema
    is a direct match for the v2.2 ontology (radiko + verba_klaso +
    substantiva_klaso + aspekta_klaso), so loading is mechanical.
  - data/lexicons/top_500_annotated.jsonl is NOT loaded here: its
    `suggested_category` ("action/verb", ...) does not map 1:1 to a
    VerbaKlaso/EntecaTipo id without a judgment-call mapping table.
  - data/dictionaries/revo_semantic_relations.json is NOT loaded here:
    those are Radiko<->Radiko pairs; the v2.2 schema has no
    synonym/hypernym relation table yet. Both are separate follow-ups.

Edges created (idempotent — existing edges are detected and skipped):
  radiko + verba_klaso      -> (:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(:VerbaKlaso)
  radiko + substantiva_klaso-> (:Radiko)-[:HAVAS_ENTECAN_TIPON]->(:EntecaTipo)   [when value is a known tipo_id]
  radiko + aspekta_klaso    -> (:Radiko)-[:HAVAS_ASPEKTAN_KLASON]->(:AspektaKlaso)

A Radiko / class that does not exist in the graph is skipped and
counted, never invented.

Usage:
    python scripts/index/load_tier0_annotations.py
    python scripts/index/load_tier0_annotations.py --dry-run
    python scripts/index/load_tier0_annotations.py --annotations PATH

Last Updated: 2026-05-16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import kuzu
from klareco.utils.kuzu_open import open_kuzu

DEFAULT_DB = "data/indexes/v2.1_kuzu_index_full"
DEFAULT_ANNOTATIONS = "data/annotations/phase_0_roots_40_more.jsonl"

# (annotation field, relation table, target label, target key property)
MAPPINGS = [
    ("verba_klaso", "APARTENAS_AL_VERBA_KLASO", "VerbaKlaso", "klaso_id"),
    ("substantiva_klaso", "HAVAS_ENTECAN_TIPON", "EntecaTipo", "tipo_id"),
    ("aspekta_klaso", "HAVAS_ASPEKTAN_KLASON", "AspektaKlaso", "klaso_id"),
]


def _ids(conn, label: str, key: str) -> set[str]:
    res = conn.execute(f"MATCH (n:{label}) RETURN n.{key}")
    out = set()
    while res.has_next():
        v = res.get_next()[0]
        if v is not None:
            out.add(v)
    return out


def _radiko_exists(conn, radiko: str) -> bool:
    res = conn.execute(
        "MATCH (r:Radiko {radiko: $r}) RETURN count(r)", {"r": radiko}
    )
    return res.get_next()[0] > 0


def _edge_exists(conn, rel: str, radiko: str, label: str, key: str,
                 target: str) -> bool:
    res = conn.execute(
        f"MATCH (r:Radiko {{radiko: $r}})-[e:{rel}]->"
        f"(k:{label} {{{key}: $k}}) RETURN count(e)",
        {"r": radiko, "k": target},
    )
    return res.get_next()[0] > 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be loaded; write nothing")
    args = p.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        print(f"Annotations file not found: {ann_path}", file=sys.stderr)
        return 1

    db = open_kuzu(args.db, read_only=args.dry_run)
    conn = kuzu.Connection(db)

    # Valid target ids per class, fetched once.
    valid = {
        label: _ids(conn, label, key)
        for _, _, label, key in MAPPINGS
    }
    for label, ids in valid.items():
        print(f"  {label}: {len(ids)} classes in graph")

    created = {rel: 0 for _, rel, _, _ in MAPPINGS}
    skip_radiko_missing = 0
    skip_class_missing: dict[str, int] = {rel: 0 for _, rel, _, _ in MAPPINGS}
    skip_already: dict[str, int] = {rel: 0 for _, rel, _, _ in MAPPINGS}
    rows = 0
    radikoj_seen: set[str] = set()
    radikoj_missing: set[str] = set()

    with ann_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            radiko = rec.get("radiko")
            if not radiko:
                continue
            rows += 1
            radikoj_seen.add(radiko)

            if not _radiko_exists(conn, radiko):
                skip_radiko_missing += 1
                radikoj_missing.add(radiko)
                continue

            for field, rel, label, key in MAPPINGS:
                val = rec.get(field)
                if not val:
                    continue
                if val not in valid[label]:
                    skip_class_missing[rel] += 1
                    continue
                if _edge_exists(conn, rel, radiko, label, key, val):
                    skip_already[rel] += 1
                    continue
                if args.dry_run:
                    created[rel] += 1
                    continue
                conn.execute(
                    f"MATCH (r:Radiko {{radiko: $r}}), "
                    f"(k:{label} {{{key}: $k}}) "
                    f"CREATE (r)-[:{rel} {{fonto: 'tier0_phase_0'}}]->(k)",
                    {"r": radiko, "k": val},
                )
                created[rel] += 1

    print()
    print(f"{'DRY RUN — ' if args.dry_run else ''}Tier 0 load summary")
    print(f"  annotation rows:        {rows}")
    print(f"  distinct radikoj:       {len(radikoj_seen)}")
    print(f"  radikoj not in graph:   {skip_radiko_missing}"
          f"  {sorted(radikoj_missing) if radikoj_missing else ''}")
    for _, rel, _, _ in MAPPINGS:
        print(f"  {rel}:")
        print(f"      edges created:      {created[rel]}")
        print(f"      class id unknown:   {skip_class_missing[rel]}")
        print(f"      already present:    {skip_already[rel]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
