#!/usr/bin/env python3
"""Measure the paragraph-level pronoun-coreference residue on real corpus text.

VISION.md lists cross-sentence coreference as a suspected residue and says
it has never been tested. This is that test: for every li/ŝi/ĝi/ili mention
found in a document-ordered slice of the live store, run
klareco.discourse.coreference's deterministic candidate generator and
report how often it resolves to exactly one candidate, how often it finds
a genuine, honestly-flagged ambiguity, and how often it finds nothing at
all (residue this window size cannot reach).

This is a measurement instrument, not an accuracy evaluator: it has no
coreference gold labels, so it cannot report precision/recall. What it
reports is honest residue-sizing (VISION.md's own bar: "how big is it?"),
plus per-status example sentences a human can spot-check.

Usage:
    python scripts/eval/measure_paragraph_coreference.py \
        --documents 500 --window 5 --output data/perf/parser_research/coref_measurement.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb

from klareco.discourse.coreference import build_entity_type_lookup, resolve_document
from klareco.parser import parse


def _document_groups(con, n_documents: int, seed: int) -> list[list[tuple]]:
    """Sample N documents (grouped by article_id) with >= 3 sentences each,
    returning each as a list of (sid, section, text) IN STORE ORDER (sid is
    monotonic within a document -- verified by inspection, not assumed)."""
    # DuckDB's USING SAMPLE clause does not accept prepared-statement
    # placeholders for its arguments (n_documents/seed are argparse ints,
    # not user-supplied strings, so direct interpolation is safe); it also
    # returns nothing when applied directly to a GROUP BY/HAVING query, so
    # the qualifying set is materialized in a CTE first, then sampled.
    article_ids = [
        r[0] for r in con.execute(
            "WITH qualifying AS ("
            "  SELECT article_id FROM sentences WHERE article_id IS NOT NULL"
            "  GROUP BY article_id HAVING count(*) >= 3"
            ") "
            "SELECT article_id FROM qualifying "
            f"USING SAMPLE reservoir({int(n_documents)} ROWS) REPEATABLE ({int(seed)})"
        ).fetchall()
    ]
    groups = []
    for aid in article_ids:
        rows = con.execute(
            "SELECT sid, section, text FROM sentences "
            "WHERE article_id = ? ORDER BY sid", [aid],
        ).fetchall()
        groups.append(rows)
    return groups


def measure(con, n_documents: int, window: int, seed: int) -> dict:
    entity_types = build_entity_type_lookup(con)
    groups = _document_groups(con, n_documents, seed)

    status_counts = Counter()
    pronoun_status_counts = Counter()
    candidate_count_when_ambiguous = Counter()
    parse_failures = 0
    examples = {"resolved": [], "ambiguous": [], "unresolved": []}
    total_mentions = 0

    for rows in groups:
        # Parse every sentence in the document ONCE, in order.
        parsed: list[tuple] = []
        text_by_sid: dict[int, str] = {}
        for sid, section, text in rows:
            try:
                parsed.append((sid, parse(text)))
                text_by_sid[sid] = text
            except Exception:
                parse_failures += 1
                continue
        for m, res in resolve_document(parsed, entity_types, window=window):
            total_mentions += 1
            status_counts[res["status"]] += 1
            pronoun_status_counts[(m["pronoun"], res["status"])] += 1
            if res["status"] == "ambiguous":
                candidate_count_when_ambiguous[len(res["candidates"])] += 1
            if len(examples[res["status"]]) < 15:
                examples[res["status"]].append({
                    "sentence": text_by_sid.get(m["sid"], ""),
                    "pronoun": m["surface"],
                    "candidates": [c["surface"] for c in res["candidates"]],
                })

    return {
        "documents_sampled": len(groups),
        "parse_failures": parse_failures,
        "total_pronoun_mentions": total_mentions,
        "status_counts": dict(status_counts),
        "status_rate": {
            k: round(v / total_mentions, 4) if total_mentions else 0.0
            for k, v in status_counts.items()
        },
        "by_pronoun_and_status": {
            f"{p}:{s}": c for (p, s), c in pronoun_status_counts.items()
        },
        "ambiguous_candidate_count_histogram": dict(candidate_count_when_ambiguous),
        "examples": examples,
        "window": window,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "data/indexes/duckdb_store.db")
    ap.add_argument("--documents", type=int, default=500)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    con = duckdb.connect(str(args.store), read_only=True)
    result = measure(con, args.documents, args.window, args.seed)

    print(f"documents sampled: {result['documents_sampled']}")
    print(f"parse failures: {result['parse_failures']}")
    print(f"total pronoun mentions: {result['total_pronoun_mentions']}")
    print("status counts:", result["status_counts"])
    print("status rate:", result["status_rate"])
    print("by pronoun+status:", result["by_pronoun_and_status"])
    print("ambiguous candidate-count histogram:", result["ambiguous_candidate_count_histogram"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
