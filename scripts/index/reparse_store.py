#!/usr/bin/env python3
"""Reparse a deterministic sample or a complete store into a separate database.

Preserves source sentence IDs and provenance, so the existing text-only Whoosh
index remains compatible. Never promotes or modifies the input store. Failure
leaves an explicitly incomplete artifact for diagnosis, not a usable release.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def process(row):
    from klareco.parser import parse, compact_ast, expand_ast
    from klareco.morphology import lexicon
    from scripts.index.build_duckdb_store import shred

    sid, text = row
    try:
        ast = parse(text)
        packed = json.dumps(compact_ast(ast), ensure_ascii=False, separators=(",", ":"))
        if expand_ast(json.loads(packed)) != ast:
            raise ValueError("AST serialization changed the parse")
        fields = shred(ast)
        fields.update(sid=sid, ast_json=packed)
        clauses = []
        for index, frame in enumerate(ast["propozicioj"]):
            fields_c = shred(frame)
            clauses.append(
                dict(
                    sid=sid,
                    clause_idx=index,
                    rolo=frame["rolo"],
                    verb_klaso=lexicon().roots.get(fields_c["verb_radiko"]),
                    **{
                        k: fields_c[k]
                        for k in (
                            "subj_radiko",
                            "subj_vortspeco",
                            "subj_kazo",
                            "verb_radiko",
                            "verb_tempo",
                            "verb_negated",
                            "obj_radiko",
                            "obj_kazo",
                        )
                    },
                )
            )
        tokens = {w["id"]: w for w in ast["vortoj"]}
        edges = [
            dict(
                sid=sid,
                token_id=w["id"],
                head_id=w["kapo"],
                rolo=w["rolo"],
                dep_radiko=w.get("radiko"),
                kapo_radiko=tokens.get(w["kapo"], {}).get("radiko"),
            )
            for w in tokens.values()
        ]
        return fields, clauses, edges, None
    except Exception as exc:
        return (
            None,
            None,
            None,
            dict(sid=sid, text=text, error=f"{type(exc).__name__}: {exc}"),
        )


def main():
    import duckdb
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, default=ROOT / "data/indexes/duckdb_store.db"
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--limit", type=int, default=10000, help="0 for the complete corpus"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()
    if (
        args.output.exists()
        or args.output.with_suffix(".manifest.json").exists()
        or args.output.resolve() == args.source.resolve()
    ):
        ap.error("Output must be a new path distinct from the source")
    if args.limit < 0 or args.batch < 1 or args.workers < 1:
        ap.error("Invalid size or worker count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "status": "building",
        "source": str(args.source.resolve()),
        "source_stat": [args.source.stat().st_size, args.source.stat().st_mtime_ns],
        "limit": args.limit,
        "processed": 0,
        "code": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "klareco/parser.py",
                ROOT / "klareco/syntax_graph.py",
                ROOT / "klareco/ast_storage.py",
                ROOT / "klareco/morphology.py",
                ROOT / "scripts/index/build_duckdb_store.py",
                Path(__file__),
                *sorted((ROOT / "data/vocabularies").glob("*.json")),
            ]
        },
        "promotion": "never automatic",
        "entity_facts": "not rebuilt; unavailable in candidate",
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    started = time.monotonic()
    con = duckdb.connect(str(args.output))
    source_sql = str(args.source.resolve()).replace("'", "''")
    con.execute(f"ATTACH '{source_sql}' AS original (READ_ONLY)")
    count = con.execute("SELECT count(*) FROM original.sentences").fetchone()[0]
    manifest["expected"] = min(count, args.limit) if args.limit else count
    # Limit selection is independent of parser behavior and reproducible.
    selection = "SELECT * FROM original.sentences"
    if args.limit:
        selection += f" ORDER BY md5(CAST(sid AS VARCHAR) || 'parser-rebuild-v1') LIMIT {args.limit}"
    from klareco.parser import MAX_SENTENCE_CHARACTERS

    oversized = con.execute(
        "SELECT sid, length(text) FROM (" + selection + ") WHERE length(text) > ?",
        [MAX_SENTENCE_CHARACTERS],
    ).fetchall()
    if oversized:
        manifest.update(
            status="blocked_source_repair",
            oversized_rows=oversized,
            max_sentence_characters=MAX_SENTENCE_CHARACTERS,
        )
        save()
        con.close()
        raise ValueError(
            f"{len(oversized)} oversized source rows; repair segmentation before rebuilding"
        )
    con.execute(
        "CREATE TABLE sentences AS " + selection + " LIMIT 0"
        if not args.limit
        else "CREATE TABLE sentences AS SELECT * FROM (" + selection + ") WHERE false"
    )
    columns = {r[1] for r in con.execute("PRAGMA table_info('sentences')").fetchall()}
    if "verb_negated" not in columns:
        con.execute("ALTER TABLE sentences ADD COLUMN verb_negated BOOLEAN")
    con.execute(
        "CREATE TEMP TABLE selected AS SELECT sid, text FROM (" + selection + ")"
    )
    for table in ("ontology_nodes", "ontology_edges"):
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM original.{table}")
    con.execute(
        "CREATE TABLE clauses(sid BIGINT, clause_idx INTEGER, rolo VARCHAR, subj_radiko VARCHAR, subj_vortspeco VARCHAR, subj_kazo VARCHAR, verb_radiko VARCHAR, verb_tempo VARCHAR, verb_negated BOOLEAN, obj_radiko VARCHAR, obj_kazo VARCHAR, verb_klaso VARCHAR)"
    )
    con.execute(
        "CREATE TABLE token_edges(sid BIGINT, token_id INTEGER, head_id INTEGER, rolo VARCHAR, dep_radiko VARCHAR, kapo_radiko VARCHAR)"
    )
    cursor = -1
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while True:
                rows = con.execute(
                    "SELECT sid, text FROM selected WHERE sid > ? ORDER BY sid LIMIT ?",
                    [cursor, args.batch],
                ).fetchall()
                if not rows:
                    break
                results = list(pool.map(process, rows, chunksize=25))
                failures = [r[3] for r in results if r[3]]
                if failures:
                    manifest["failures"] = failures
                    raise ValueError(
                        f"{len(failures)} parse/validation failures in batch"
                    )
                updates = pd.DataFrame([r[0] for r in results])
                clause_rows = pd.DataFrame([c for r in results for c in r[1]])
                edge_rows = pd.DataFrame([e for r in results for e in r[2]])
                con.register("updates", updates)
                columns = [
                    r[1]
                    for r in con.execute("PRAGMA table_info('sentences')").fetchall()
                ]
                projection = ", ".join(
                    ("u." if c in updates.columns else "s.") + '"' + c + '"'
                    for c in columns
                )
                con.execute("BEGIN")
                con.execute(
                    "INSERT INTO sentences SELECT "
                    + projection
                    + " FROM original.sentences s JOIN updates u USING(sid)"
                )
                if len(clause_rows):
                    con.register("clause_rows", clause_rows)
                    con.execute("INSERT INTO clauses BY NAME SELECT * FROM clause_rows")
                con.register("edge_rows", edge_rows)
                con.execute("INSERT INTO token_edges BY NAME SELECT * FROM edge_rows")
                con.execute("COMMIT")
                cursor = rows[-1][0]
                manifest["processed"] += len(rows)
                manifest["elapsed_seconds"] = time.monotonic() - started
                save()
                print(
                    f"{manifest['processed']}/{manifest['expected']} in {manifest['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        con.execute(
            "CREATE TABLE dependency_arcs AS SELECT sid, kapo_radiko, rolo, dep_radiko FROM token_edges WHERE rolo NOT IN ('punct', 'dep', 'root')"
        )
        con.execute("CREATE UNIQUE INDEX sentence_ids ON sentences(sid)")
        for column in (
            "subj_radiko",
            "verb_radiko",
            "obj_radiko",
            "article_id",
            "source_name",
        ):
            con.execute(f"CREATE INDEX sentence_{column} ON sentences({column})")
        actual = con.execute("SELECT count(*) FROM sentences").fetchone()[0]
        if actual != manifest["expected"]:
            raise ValueError(f"Row count mismatch: {actual}")
        mismatch = con.execute(
            "SELECT count(*) FROM sentences s JOIN original.sentences o USING(sid) WHERE s.text IS DISTINCT FROM o.text OR s.article_title IS DISTINCT FROM o.article_title OR s.source_name IS DISTINCT FROM o.source_name"
        ).fetchone()[0]
        if mismatch:
            raise ValueError(f"{mismatch} source identity changes")
        manifest.update(
            status="validated_candidate",
            identity_mismatches=mismatch,
            elapsed_seconds=time.monotonic() - started,
        )
        con.execute("CHECKPOINT")
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        con.close()
        save()


if __name__ == "__main__":
    main()
